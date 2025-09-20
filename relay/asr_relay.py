#!/usr/bin/env python3
"""
asr_relay.py — NKN DM ⇄ HTTP(S) relay specialized for the ASR service

What it does
------------
- Boots a tiny Node sidecar (nkn-sdk MultiClient) to send/receive NKN direct messages.
- Generic DM: {"event":"relay.http","id":"...","req":{...}} -> mirrors nkn_relay.py.
- ASR helpers (builds HTTP calls for you):

  • {"event":"asr.start","id":"...","opts":{headers?,timeout_ms?,verify?,insecure_tls?,service?}}
      → POST  /recognize/stream/start
      → reply: {"event":"relay.response", ...}

  • {"event":"asr.audio","id":"...","sid":"<session id>","format":"pcm16|wav|mp3|...","sr":16000,
      "body_b64":"...", "opts":{headers?,timeout_ms?,verify?,insecure_tls?,service?}}
      → POST  /recognize/stream/<sid>/audio?format=...&sr=...
      → reply: {"event":"relay.response", ...}

  • {"event":"asr.end","id":"...","sid":"<session id>","opts":{headers?,timeout_ms?,verify?,insecure_tls?,service?}}
      → POST  /recognize/stream/<sid>/end
      → reply: {"event":"relay.response", ...}

  • {"event":"asr.events","id":"...","sid":"<session id>","opts":{headers?,timeout_ms?,verify?,insecure_tls?,service?}}
      → GET   /recognize/stream/<sid>/events (SSE)
      → streamed replies:
          begin : {"event":"relay.response.begin", ...headers/status...}
          chunk : {"event":"relay.response.chunk","id":..., "seq":N, "b64":"..."}
          keep  : {"event":"relay.response.keepalive","id":"...","ts":...}
          end   : {"event":"relay.response.end","id":"...","ok":true,"bytes":...,"last_seq":...}

DM Request schema (generic relay.http — same as nkn_relay.py)
-------------------------------------------------------------
{
  "event": "relay.http",   # alias: "http.request", "relay.fetch"
  "id": "abc123",
  "req": {
    "url": "http://127.0.0.1:8126/recognize",
    # or "service":"asr","path":"/recognize"

    "method": "POST",
    "headers": {"Content-Type":"application/json"},
    "json": { "body_b64":"...", "format":"wav" },   # or:
    # "body_b64": "<base64 bytes>",
    "timeout_ms": 30000,
    "verify": true,
    "insecure_tls": false,
    "stream": "chunks"       # enables chunked stream for response (e.g., SSE)
  }
}

Environment (.env auto-created on first run)
--------------------------------------------
NKN_SEED_HEX=<64 hex chars>
NKN_NUM_SUBCLIENTS=2
NKN_TOPIC_PREFIX=relay
RELAY_WORKERS=4
RELAY_MAX_BODY_B=1048576
RELAY_VERIFY_DEFAULT=1
RELAY_TARGETS={"asr":"http://127.0.0.1:8126"}
NKN_BRIDGE_SEED_WS=
RELAY_CHUNK_RAW_B=12288
RELAY_HEARTBEAT_S=10

Usage
-----
python3 asr_relay.py [--debug]
"""

import os, sys, json, time, base64, threading, queue, subprocess, shutil, argparse, datetime
from pathlib import Path

# ──────────────────────────────────────────────────────────────
# 0) CLI flags
# ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(add_help=True)
parser.add_argument("--debug", action="store_true", help="Run with curses TUI; list all incoming NKN DMs")
ARGS, _UNKNOWN = parser.parse_known_args()
DEBUG = bool(ARGS.debug)

# ──────────────────────────────────────────────────────────────
# 0.1) Tiny venv + deps (requests, python-dotenv)
# ──────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent
VENV = BASE / ".venv"
BIN  = VENV / ("Scripts" if os.name == "nt" else "bin")
PY   = BIN / ("python.exe" if os.name == "nt" else "python")
PIP  = BIN / ("pip.exe" if os.name == "nt" else "pip")

def _in_venv() -> bool:
    try: return Path(sys.executable).resolve() == PY.resolve()
    except Exception: return False

def _ensure_venv():
    if VENV.exists(): return
    import venv; venv.EnvBuilder(with_pip=True).create(VENV)
    subprocess.check_call([str(PY), "-m", "pip", "install", "--upgrade", "pip"])

def _ensure_deps():
    need = []
    try: import requests  # type: ignore
    except Exception: need.append("requests")
    try: import dotenv    # type: ignore
    except Exception: need.append("python-dotenv")
    if DEBUG and os.name == "nt":
        try: import curses  # type: ignore
        except Exception: need.append("windows-curses")
    if need:
        subprocess.check_call([str(PIP), "install", *need])

if not _in_venv():
    _ensure_venv()
    os.execv(str(PY), [str(PY), *sys.argv])
_ensure_deps()

import requests
from dotenv import dotenv_values

# ──────────────────────────────────────────────────────────────
# 0.2) TUI (curses) setup
# ──────────────────────────────────────────────────────────────
try:
    import curses  # type: ignore
except Exception:
    curses = None  # graceful fallback

class _DebugUI:
    """Thread-safe TUI to display incoming NKN DMs and status."""
    def __init__(self, enabled: bool):
        self.enabled = enabled and (curses is not None) and sys.stdout.isatty()
        self.q: "queue.Queue[dict]" = queue.Queue()
        self.rows: list[dict] = []
        self.max_rows = 1000
        self.stats = {"in":0, "out":0, "err":0}
        self._stop = threading.Event()
        self._my_addr_ref = {"nkn": None}
        self._jobs_ref: "queue.Queue|None" = None
        self._cfg = {"workers": 0, "verify": True, "max_body": 0}

    def bind_refs(self, my_addr_ref: dict, jobs_ref: "queue.Queue", workers: int, verify: bool, max_body: int):
        self._my_addr_ref = my_addr_ref
        self._jobs_ref = jobs_ref
        self._cfg = {"workers": workers, "verify": verify, "max_body": max_body}

    def log(self, kind: str, **kw):
        ev = {"ts": time.time(), "kind": str(kind)}
        safe_kw = {}
        for k, v in kw.items():
            try: safe_kw[k] = str(v)
            except Exception: safe_kw[k] = repr(v)
        ev.update(safe_kw)

        if kind == "IN": self.stats["in"] += 1
        elif kind == "OUT": self.stats["out"] += 1
        elif kind == "ERR": self.stats["err"] += 1

        if self.enabled:
            self.q.put(ev)
        else:
            ts = datetime.datetime.fromtimestamp(ev["ts"]).strftime("%H:%M:%S")
            src = ev.get("src",""); to = ev.get("to",""); msg = ev.get("msg","")
            try:
                line = f"[{ts}] {ev['kind']:<3} {(src or to):>40}  {msg}"
            except Exception:
                line = f"[{ts}] {ev['kind']:<3} (render error)"
            print(line, flush=True)

    def run(self):
        if not self.enabled:
            try:
                while not self._stop.is_set(): time.sleep(0.25)
            except KeyboardInterrupt:
                pass
            return
        curses.wrapper(self._main)

    def stop(self):
        self._stop.set()

    def _add_line(self, stdscr, y, x, text, attr=0):
        try:
            stdscr.addnstr(y, x, text, max(0, curses.COLS-1-x), attr)
        except Exception:
            pass

    def _main(self, stdscr):
        curses.curs_set(0)
        stdscr.nodelay(True)
        stdscr.timeout(100)
        while not self._stop.is_set():
            try:
                while True:
                    ev = self.q.get_nowait()
                    self.rows.append(ev)
                    if len(self.rows) > self.max_rows:
                        self.rows = self.rows[-self.max_rows:]
            except queue.Empty:
                pass

            stdscr.erase()
            addr = str(self._my_addr_ref.get("nkn") or "—")
            h1 = f"NKN ASR Relay  |  addr: {addr}  |  workers:{self._cfg['workers']}  verify:{'on' if self._cfg['verify'] else 'off'}  max_body:{self._cfg['max_body']}"
            self._add_line(stdscr, 0, 0, h1, curses.A_BOLD)

            try:
                qsz = self._jobs_ref.qsize() if self._jobs_ref is not None else 0
            except Exception:
                qsz = 0
            h2 = f"IN:{self.stats['in']}  OUT:{self.stats['out']}  ERR:{self.stats['err']}  queue:{qsz}   (q to quit)"
            self._add_line(stdscr, 1, 0, h2)

            self._add_line(stdscr, 3, 0, "   TS     KIND  FROM/TO                                EVENT/STATUS           INFO", curses.A_UNDERLINE)

            max_lines = max(0, curses.LINES - 5)
            view = self.rows[-max_lines:]
            y = 4
            for ev in view:
                try: ts = datetime.datetime.fromtimestamp(float(ev["ts"])).strftime("%H:%M:%S")
                except Exception: ts = "--:--:--"
                kind = f"{str(ev.get('kind','')):<4}"[:4]

                who_str = str(ev.get("src") or ev.get("to") or "")
                who = (who_str[-36:] if len(who_str) > 36 else who_str).rjust(36)

                evs_raw = ev.get("event") or ev.get("status") or ""
                evs = f"{str(evs_raw):<20}"[:20]

                info = str(ev.get("msg") or ev.get("detail") or "")
                line = f" {ts}  {kind}  {who}  {evs}  {info}"
                self._add_line(stdscr, y, 0, line); y += 1

            stdscr.refresh()
            try:
                ch = stdscr.getch()
                if ch in (ord('q'), ord('Q')): self._stop.set()
            except Exception:
                pass
            time.sleep(0.05)

UI = _DebugUI(DEBUG)

# ──────────────────────────────────────────────────────────────
# 1) Config (.env bootstrap)
# ──────────────────────────────────────────────────────────────
ENV = BASE / ".env"
if not ENV.exists():
    import secrets
    ENV.write_text(
        "NKN_SEED_HEX={seed}\n"
        "NKN_NUM_SUBCLIENTS=2\n"
        "NKN_TOPIC_PREFIX=relay\n"
        "RELAY_WORKERS=4\n"
        "RELAY_MAX_BODY_B=1048576\n"
        "RELAY_VERIFY_DEFAULT=1\n"
        "RELAY_TARGETS={targets}\n"
        "NKN_BRIDGE_SEED_WS=\n"
        "RELAY_CHUNK_RAW_B=12288\n"
        "RELAY_HEARTBEAT_S=10\n".format(
            seed=secrets.token_hex(32),
            targets=json.dumps({"asr": "http://127.0.0.1:8126"})
        )
    )
    print("→ wrote .env with defaults")

cfg = {**dotenv_values(str(ENV))}
SEED_HEX   = (cfg.get("NKN_SEED_HEX") or "").lower().replace("0x","")
SUBS       = int(cfg.get("NKN_NUM_SUBCLIENTS") or "2")
TOPIC_NS   = cfg.get("NKN_TOPIC_PREFIX") or "relay"
WORKERS    = int(cfg.get("RELAY_WORKERS") or "4")
MAX_BODY_B = int(cfg.get("RELAY_MAX_BODY_B") or str(1*1024*1024))
VERIFY_DEF = (cfg.get("RELAY_VERIFY_DEFAULT") or "1").strip().lower() in ("1","true","yes","on")
TARGETS    = {}
CHUNK_RAW_B = int((cfg.get("RELAY_CHUNK_RAW_B") or "12288"))
HEARTBEAT_S = int((cfg.get("RELAY_HEARTBEAT_S") or "10"))

DM_OPTS_STREAM = {"noReply": False, "maxHoldingSeconds": 120}
DM_OPTS_SINGLE = {"noReply": True}

try:
    TARGETS = json.loads(cfg.get("RELAY_TARGETS") or "{}")
except Exception:
    TARGETS = {}
SEEDS_WS   = [s.strip() for s in (cfg.get("NKN_BRIDGE_SEED_WS") or "").split(",") if s.strip()]

# ──────────────────────────────────────────────────────────────
# 2) Minimal Node bridge (nkn-sdk)
# ──────────────────────────────────────────────────────────────
NODE = shutil.which("node")
NPM  = shutil.which("npm")
if not NODE or not NPM:
    sys.exit("‼️  Node.js and npm are required (for nkn-sdk).")

BRIDGE_DIR = BASE / "bridge-node"
BRIDGE_JS  = BRIDGE_DIR / "nkn_bridge.js"
PKG_JSON   = BRIDGE_DIR / "package.json"

if not BRIDGE_DIR.exists():
    BRIDGE_DIR.mkdir(parents=True)
if not PKG_JSON.exists():
    subprocess.check_call([NPM, "init", "-y"], cwd=BRIDGE_DIR)
    subprocess.check_call([NPM, "install", "nkn-sdk@^1.3.6"], cwd=BRIDGE_DIR)

BRIDGE_SRC = r"""
'use strict';
const nkn = require('nkn-sdk');
const readline = require('readline');

const SEED_HEX = (process.env.NKN_SEED_HEX || '').toLowerCase().replace(/^0x/,'');
const NUM = parseInt(process.env.NKN_NUM_SUBCLIENTS || '2', 10) || 2;
const SEED_WS = (process.env.NKN_BRIDGE_SEED_WS || '').split(',').map(s=>s.trim()).filter(Boolean);

function out(obj){ try{ process.stdout.write(JSON.stringify(obj)+'\n'); }catch{} }

(async ()=>{
  if(!/^[0-9a-f]{64}$/.test(SEED_HEX)){ out({type:'crit', msg:'bad NKN_SEED_HEX'}); process.exit(1); }
  const client = new nkn.MultiClient({
    seed: SEED_HEX,
    identifier: 'relay',
    numSubClients: NUM,
    seedWsAddr: SEED_WS.length ? SEED_WS : undefined,
    wsConnHeartbeatTimeout: 120000,
  });

  client.on('connect', ()=> out({type:'ready', address: String(client.addr || ''), ts: Date.now()}));

  client.on('message', (a, b)=>{
    try{
      let src, payload;
      if (a && typeof a === 'object' && a.payload !== undefined) { src = String(a.src || ''); payload = a.payload; }
      else { src = String(a || ''); payload = b; }
      const s = Buffer.isBuffer(payload) ? payload.toString('utf8') : (typeof payload === 'string' ? payload : String(payload));
      let parsed = null; try { parsed = JSON.parse(s); } catch {}
      out({type:'nkn-dm', src, msg: parsed || {event:'<non-json>', raw:s}});
    }catch(e){ out({type:'err', msg: String(e && e.message || e)}); }
  });

  const rl = readline.createInterface({input: process.stdin});
  rl.on('line', line=>{
    let cmd; try{ cmd = JSON.parse(line); }catch{ return; }
    if(cmd && cmd.type==='dm' && cmd.to && cmd.data){
      const opts = cmd.opts || { noReply: true };
      client.send(cmd.to, JSON.stringify(cmd.data), opts).catch(()=>{});
    }
  });
})();
"""
if not BRIDGE_JS.exists() or BRIDGE_JS.read_text() != BRIDGE_SRC:
    BRIDGE_JS.write_text(BRIDGE_SRC)

BRIDGE_ENV = os.environ.copy()
BRIDGE_ENV["NKN_SEED_HEX"]       = SEED_HEX
BRIDGE_ENV["NKN_NUM_SUBCLIENTS"] = str(SUBS)
BRIDGE_ENV["NKN_BRIDGE_SEED_WS"] = ",".join(SEEDS_WS)

bridge = subprocess.Popen(
    [NODE, str(BRIDGE_JS)],
    cwd=BRIDGE_DIR,
    env=BRIDGE_ENV,
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    bufsize=1
)

bridge_write_lock = threading.Lock()
def _dm(to: str, data: dict, opts: dict | None = None):
    if not bridge or not bridge.stdin: return
    try:
        payload = {"type": "dm", "to": to, "data": data}
        if opts: payload["opts"] = opts
        with bridge_write_lock:
            bridge.stdin.write(json.dumps(payload) + "\n")
            bridge.stdin.flush()
    except Exception:
        pass

# Pump bridge stderr to our stderr (optional) + UI
def _stderr_pump():
    while True:
        try:
            line = bridge.stderr.readline()
            if not line and bridge.poll() is not None: break
            if line:
                sys.stderr.write(line)
                if DEBUG:
                    UI.log("ERR", msg=line.rstrip())
        except Exception:
            time.sleep(0.1)
threading.Thread(target=_stderr_pump, daemon=True).start()

# ──────────────────────────────────────────────────────────────
# 3) HTTP worker pool (DM → queue → requests → DM)
# ──────────────────────────────────────────────────────────────
Job = dict
jobs: "queue.Queue[Job]" = queue.Queue()
my_addr = {"nkn": None}
UI.bind_refs(my_addr, jobs, WORKERS, VERIFY_DEF, MAX_BODY_B)

def _resolve_url(req: dict) -> str:
    url = (req.get("url") or "").strip()
    if url:
        return url
    service = (req.get("service") or "").strip()
    path    = req.get("path") or "/"
    base    = TARGETS.get(service)
    if not base:
        raise ValueError(f"unknown service '{service}' (configure RELAY_TARGETS)")
    if not path.startswith("/"):
        path = "/" + path
    return (base.rstrip("/") + path)

def _to_b64(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")

def _norm_event(ev: str) -> str:
    s = (ev or "").lower()
    if "relay.http" in s or "http.request" in s or "relay.fetch" in s:
        return "relay.http"
    if s in ("asr.start","asr.audio","asr.end","asr.events"):
        return s
    if s.endswith("relay.ping") or s.endswith(".ping") or s == "ping":
        return "relay.ping"
    if s.endswith("relay.info") or s.endswith(".info") or s == "info":
        return "relay.info"
    return s

def _http_worker():
    s = requests.Session()
    dm_opts_stream = globals().get("DM_OPTS_STREAM", {"noReply": False, "maxHoldingSeconds": 120})
    dm_opts_single = globals().get("DM_OPTS_SINGLE", {"noReply": True})

    while True:
        job = jobs.get()
        if job is None:
            try: jobs.task_done()
            except: pass
            return

        want_stream = False
        rid = ""
        src = ""

        try:
            src = job["src"]
            body = job["body"]
            rid  = body.get("id") or ""
            req  = body.get("req") or {}

            url = _resolve_url(req)
            method  = (req.get("method") or "GET").upper()
            headers = req.get("headers") or {}
            timeout = float(req.get("timeout_ms") or 30000) / 1000.0
            verify  = VERIFY_DEF
            if isinstance(req.get("verify"), bool):
                verify = bool(req["verify"])
            if req.get("insecure_tls") in (True, "1", "true", "on"):
                verify = False

            sv = str(req.get("stream") or "").lower()
            if sv in ("1", "true", "yes", "on", "chunks", "dm"):
                want_stream = True
            if (headers.get("x-relay-stream") or "").lower() in ("1","true","chunks","dm"):
                want_stream = True

            kwargs = {"headers": headers, "timeout": timeout, "verify": verify}
            if "json" in req and req["json"] is not None:
                kwargs["json"] = req["json"]
            elif "body_b64" in req and req["body_b64"] is not None:
                try:
                    kwargs["data"] = base64.b64decode(str(req["body_b64"]), validate=False)
                except Exception:
                    kwargs["data"] = b""

            if want_stream:
                resp = s.request(method, url, stream=True, **kwargs)
                hdrs = {k.lower(): v for k, v in resp.headers.items()}

                # optional filename (for parity with TTS relay)
                filename = None
                cd = resp.headers.get("Content-Disposition") or resp.headers.get("content-disposition") or ""
                try:
                    import re, urllib.parse
                    m = re.search(r"filename\*=UTF-8''([^;]+)|filename=\"?([^\";]+)\"?", cd, re.I)
                    if m: filename = urllib.parse.unquote(m.group(1) or m.group(2))
                except Exception:
                    filename = None

                cl_raw = resp.headers.get("Content-Length") or resp.headers.get("content-length")
                try:
                    cl_num = int(cl_raw) if cl_raw is not None else None
                except Exception:
                    cl_num = None

                _dm(src, {
                    "event": "relay.response.begin",
                    "id": rid, "ok": True,
                    "status": int(resp.status_code),
                    "headers": hdrs,
                    "content_length": cl_num,
                    "filename": filename
                }, dm_opts_stream)
                if DEBUG:
                    UI.log("OUT", to=src, status="begin", msg=f"id={rid}")

                total = 0
                seq = 0
                last_send = time.time()

                for chunk in resp.iter_content(chunk_size=CHUNK_RAW_B):
                    if not chunk:
                        if time.time() - last_send >= HEARTBEAT_S:
                            _dm(src, {"event": "relay.response.keepalive", "id": rid, "ts": int(time.time()*1000)}, dm_opts_stream)
                            last_send = time.time()
                        continue
                    total += len(chunk)
                    seq += 1
                    _dm(src, {"event": "relay.response.chunk", "id": rid, "seq": seq, "b64": _to_b64(chunk)}, dm_opts_stream)
                    last_send = time.time()
                    time.sleep(0.002)

                _dm(src, {
                    "event": "relay.response.end",
                    "id": rid, "ok": True,
                    "bytes": total,
                    "last_seq": seq,
                    "truncated": False,
                    "error": None
                }, dm_opts_stream)
                if DEBUG:
                    UI.log("OUT", to=src, status="end", msg=f"id={rid} bytes={total} last_seq={seq}")
                continue

            # --- single response ---
            resp = s.request(method, url, **kwargs)
            raw  = resp.content or b""
            truncated = False
            if len(raw) > MAX_BODY_B:
                raw = raw[:MAX_BODY_B]
                truncated = True

            payload = {
                "event": "relay.response",
                "id": rid, "ok": True,
                "status": int(resp.status_code),
                "headers": {k.lower(): v for k, v in resp.headers.items()},
                "json": None, "body_b64": None,
                "truncated": truncated, "error": None,
            }
            ctype = (resp.headers.get("Content-Type") or "").lower()
            if "application/json" in ctype or "text/event-stream" in ctype:
                try:
                    payload["json"] = resp.json()
                except Exception:
                    payload["body_b64"] = _to_b64(raw)
            else:
                payload["body_b64"] = _to_b64(raw)

            _dm(src, payload, dm_opts_single)
            if DEBUG:
                UI.log("OUT", to=src, status=str(payload["status"]), msg=f"id={rid} truncated={truncated}")

        except Exception as e:
            if want_stream:
                _dm(src, {"event": "relay.response.end", "id": rid, "ok": False,
                          "bytes": 0, "last_seq": 0, "truncated": False, "error": f"{type(e).__name__}: {e}"}, dm_opts_stream)
            else:
                _dm(src, {"event": "relay.response", "id": rid, "ok": False,
                          "status": 0, "headers": {}, "json": None, "body_b64": None,
                          "truncated": False, "error": f"{type(e).__name__}: {e}"}, dm_opts_single)
            if DEBUG:
                UI.log("ERR", to=src, msg=f"id={rid} {type(e).__name__}: {e}")
        finally:
            try: jobs.task_done()
            except ValueError: pass

for _ in range(max(1, WORKERS)):
    threading.Thread(target=_http_worker, daemon=True).start()

# ──────────────────────────────────────────────────────────────
# 4) Helpers: ASR-specific DM → generic request spec
# ──────────────────────────────────────────────────────────────
def _mk_req_for_asr_start(m: dict) -> dict:
    opts = m.get("opts") or {}
    service = (opts.get("service") or "asr").strip()
    return {
        "service": service, "path": "/recognize/stream/start",
        "method": "POST",
        "headers": opts.get("headers") or {},
        "timeout_ms": opts.get("timeout_ms") or 45000,
        "verify": opts.get("verify") if isinstance(opts.get("verify"), bool) else None,
        "insecure_tls": opts.get("insecure_tls") in (True, "1", "true", "on")
    }

def _mk_req_for_asr_audio(m: dict) -> dict:
    sid = (m.get("sid") or "").strip()
    if not sid: raise ValueError("asr.audio missing 'sid'")
    fmt = (m.get("format") or "pcm16").strip()
    sr  = int(m.get("sr") or 16000)
    body_b64 = m.get("body_b64") or ""
    if not body_b64: raise ValueError("asr.audio missing 'body_b64'")
    opts = m.get("opts") or {}
    service = (opts.get("service") or "asr").strip()
    path = f"/recognize/stream/{sid}/audio?format={fmt}&sr={sr}"
    headers = {"Content-Type": "application/octet-stream"}
    headers.update(opts.get("headers") or {})
    return {
        "service": service, "path": path,
        "method": "POST",
        "headers": headers,
        "body_b64": body_b64,
        "timeout_ms": opts.get("timeout_ms") or 45000,
        "verify": opts.get("verify") if isinstance(opts.get("verify"), bool) else None,
        "insecure_tls": opts.get("insecure_tls") in (True, "1", "true", "on")
    }

def _mk_req_for_asr_end(m: dict) -> dict:
    sid = (m.get("sid") or "").strip()
    if not sid: raise ValueError("asr.end missing 'sid'")
    opts = m.get("opts") or {}
    service = (opts.get("service") or "asr").strip()
    return {
        "service": service, "path": f"/recognize/stream/{sid}/end",
        "method": "POST",
        "headers": opts.get("headers") or {},
        "timeout_ms": opts.get("timeout_ms") or 45000,
        "verify": opts.get("verify") if isinstance(opts.get("verify"), bool) else None,
        "insecure_tls": opts.get("insecure_tls") in (True, "1", "true", "on")
    }

def _mk_req_for_asr_events(m: dict) -> dict:
    sid = (m.get("sid") or "").strip()
    if not sid: raise ValueError("asr.events missing 'sid'")
    opts = m.get("opts") or {}
    service = (opts.get("service") or "asr").strip()
    headers = {"Accept": "text/event-stream", "X-Relay-Stream": "chunks"}
    headers.update(opts.get("headers") or {})
    return {
        "service": service, "path": f"/recognize/stream/{sid}/events",
        "method": "GET",
        "headers": headers,
        "timeout_ms": opts.get("timeout_ms") or 300000,
        "verify": opts.get("verify") if isinstance(opts.get("verify"), bool) else None,
        "insecure_tls": opts.get("insecure_tls") in (True, "1", "true", "on"),
        "stream": "chunks"
    }

# ──────────────────────────────────────────────────────────────
# 5) Bridge stdout reader -> handle minimal events + UI logs
# ──────────────────────────────────────────────────────────────
def _stdout_pump():
    while True:
        try:
            line = bridge.stdout.readline()
            if not line and bridge.poll() is not None: break
            if not line: continue
            try:
                msg = json.loads(line.strip())
            except Exception:
                continue

            t = msg.get("type")
            if t == "ready":
                my_addr["nkn"] = msg.get("address")
                print(f"→ NKN ready: {my_addr['nkn']}", flush=True)
                if DEBUG:
                    UI.log("SYS", msg=f"ready {my_addr['nkn']}")
                continue

            if t == "nkn-dm":
                src = msg.get("src") or ""
                m   = msg.get("msg") or {}
                ev  = _norm_event(m.get("event") or "")
                rid = m.get("id")
                if DEBUG:
                    UI.log("IN", src=src, event=ev or "<unknown>", msg=f"id={rid or '—'}")

                if ev == "relay.ping":
                    _dm(src, {"event":"relay.pong","ts":int(time.time()*1000),"addr": my_addr["nkn"]})
                    continue

                if ev == "relay.info":
                    _dm(src, {
                        "event":"relay.info",
                        "ts": int(time.time()*1000),
                        "addr": my_addr["nkn"],
                        "services": sorted(list(TARGETS.keys())),
                        "verify_default": VERIFY_DEF,
                        "workers": WORKERS,
                        "max_body_b": MAX_BODY_B
                    })
                    continue

                # ASR helpers -> convert to generic relay.http job
                if ev == "asr.start":
                    req = _mk_req_for_asr_start(m)
                    jobs.put({"src": src, "body": {"event": "relay.http", "id": rid, "req": req}})
                    continue
                if ev == "asr.audio":
                    try:
                        req = _mk_req_for_asr_audio(m)
                    except Exception as e:
                        _dm(src, {"event":"relay.response","id": rid,"ok":False,"status":0,"headers":{},"json":None,"body_b64":None,"truncated":False,"error":str(e)}, DM_OPTS_SINGLE)
                        continue
                    jobs.put({"src": src, "body": {"event": "relay.http", "id": rid, "req": req}})
                    continue
                if ev == "asr.end":
                    req = _mk_req_for_asr_end(m)
                    jobs.put({"src": src, "body": {"event": "relay.http", "id": rid, "req": req}})
                    continue
                if ev == "asr.events":
                    req = _mk_req_for_asr_events(m)
                    jobs.put({"src": src, "body": {"event": "relay.http", "id": rid, "req": req}})
                    continue

                # Generic HTTP relay (unchanged)
                if ev == "relay.http":
                    if DEBUG:
                        try:
                            req = m.get("req") or {}
                            url = req.get("url") or (f"{req.get('service','')}:{req.get('path','/')}")
                            UI.log("SYS", msg=f"queue id={rid or '—'} {req.get('method','GET')} {url}")
                        except Exception:
                            pass
                    jobs.put({"src": src, "body": m})
                    continue
                # Unknown → ignore
        except Exception:
            time.sleep(0.05)

threading.Thread(target=_stdout_pump, daemon=True).start()

# ──────────────────────────────────────────────────────────────
# 6) Stay alive (curses UI if --debug, else idle loop)
# ──────────────────────────────────────────────────────────────
try:
    if DEBUG:
        UI.run()
    else:
        while True:
            time.sleep(3600)
except KeyboardInterrupt:
    pass
finally:
    UI.stop()
