#!/usr/bin/env python3
"""
asr_service.py — Whisper ASR microservice (file & live streaming)

What it does
------------
- Bootstraps a tiny venv (first run) and installs dependencies.
- Loads Whisper models once (device-aware) with a process-global cache.
- HTTP API (mirrors the clean REST vibe of your TTS service):

  GET  /health
  GET  /models

  POST /recognize
    • multipart/form-data   file=<audio>
    • application/json      {"body_b64":"...", "format":"wav|mp3|flac|pcm16", "sample_rate":16000}

    Returns:
      {
        "ok": true,
        "text": "...",
        "segments": [{"t0":0.00,"t1":1.23,"text":"..."}, ...],
        "language": "en",
        "duration": 12.34,
        "model": "medium",
        "device": "cuda|cpu"
      }

  # Bi-directional streaming over simple REST + SSE:
  POST /recognize/stream/start
    -> { "ok": true, "id": "sess-..." }

  POST /recognize/stream/<id>/audio?format=pcm16&sr=16000
    (binary body) or (Content-Type: application/octet-stream)
    -> { "ok": true, "received": 12345, "seconds": 0.77, "total_seconds": 3.21 }

  POST /recognize/stream/<id>/end
    -> { "ok": true, "final": { ...same shape as /recognize result... } }

  GET  /recognize/stream/<id>/events        (SSE)
    Emits:
      event: keepalive
      data: {"ts": 1712345678901}

      event: asr.partial
      data: {"id":"...","text":"(running cumulative text)","words_added":"...","ts":...}

      event: asr.final
      data: { "id":"...", "result": { ...final json like /recognize... } }

Environment (.env auto-created on first run)
--------------------------------------------
ASR_HOST=0.0.0.0
ASR_PORT=8126
ASR_DEVICE=auto                     # auto|cpu|cuda
ASR_MODEL_FULL=medium               # full pass
ASR_MODEL_PREVIEW=base              # quick live updates
ASR_PREVIEW_WINDOW_S=6.0            # tail window for live updates
ASR_PREVIEW_STEP_S=0.9              # cadence for live updates
ASR_SESSION_TTL_S=900               # idle GC
CORS_ORIGINS=*                      # comma-separated or *

Added support for passing a text prompt into Whisper:
- File recognition: POST /recognize with multipart field `prompt` or JSON key `prompt`
- Streaming: POST /recognize/stream/start with JSON key `prompt` or query ?prompt=...

The prompt is applied to both live preview partials and the final transcript.
"""

import os, sys, json, time, base64, threading, queue, subprocess, shutil, traceback
from pathlib import Path

# ──────────────────────────────────────────────────────────────
# 0) venv bootstrap (install & re-exec)
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
    try: import flask  # type: ignore
    except Exception: need += ["flask", "flask-cors"]
    try: import dotenv # type: ignore
    except Exception: need += ["python-dotenv"]
    try: import numpy  # type: ignore
    except Exception: need += ["numpy"]
    try: import scipy  # type: ignore
    except Exception: need += ["scipy"]
    try:
        import soundfile  # type: ignore
    except Exception:
        need += ["soundfile"]
    # openai-whisper pulls in torch (CPU build) automatically.
    try: import whisper  # type: ignore
    except Exception: need += ["openai-whisper"]
    if need:
        subprocess.check_call([str(PIP), "install", *need])

if not _in_venv():
    _ensure_venv()
    os.execv(str(PY), [str(PY), *sys.argv])
_ensure_deps()

# ──────────────────────────────────────────────────────────────
# 1) imports (post-venv)
# ──────────────────────────────────────────────────────────────
import math
import numpy as np
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from dotenv import dotenv_values

# Optional helpers
try:
    import soundfile as sf  # decoding for file uploads
except Exception:
    sf = None

try:
    import torch
except Exception:
    class _Dummy:
        @staticmethod
        def cuda_is_available(): return False
    torch = None

import whisper

# ──────────────────────────────────────────────────────────────
# 2) Config (.env bootstrap)
# ──────────────────────────────────────────────────────────────
ENV = BASE / ".env"
if not ENV.exists():
    ENV.write_text(
        "ASR_HOST=0.0.0.0\n"
        "ASR_PORT=8126\n"
        "ASR_DEVICE=auto\n"
        "ASR_MODEL_FULL=medium\n"
        "ASR_MODEL_PREVIEW=base\n"
        "ASR_PREVIEW_WINDOW_S=6.0\n"
        "ASR_PREVIEW_STEP_S=0.9\n"
        "ASR_SESSION_TTL_S=900\n"
        "CORS_ORIGINS=*\n"
    )
    print("→ wrote .env with defaults")

_cfg = {**dotenv_values(str(ENV))}
ASR_HOST = _cfg.get("ASR_HOST") or "0.0.0.0"
ASR_PORT = int(_cfg.get("ASR_PORT") or "8126")
ASR_DEVICE = (_cfg.get("ASR_DEVICE") or "auto").strip().lower()
MODEL_FULL = (_cfg.get("ASR_MODEL_FULL") or "medium").strip()
MODEL_PREV = (_cfg.get("ASR_MODEL_PREVIEW") or "base").strip()
PREVIEW_WIN_S  = float(_cfg.get("ASR_PREVIEW_WINDOW_S") or "6.0")
PREVIEW_STEP_S = float(_cfg.get("ASR_PREVIEW_STEP_S") or "0.9")
SESSION_TTL_S  = int(_cfg.get("ASR_SESSION_TTL_S") or "900")
CORS_ORIGINS   = (_cfg.get("CORS_ORIGINS") or "*").strip()

# ──────────────────────────────────────────────────────────────
# 3) Whisper model pool (refcounted), device normalize
# ──────────────────────────────────────────────────────────────
_WHISPER_LOCK = threading.Lock()
_WHISPER_POOL: dict[tuple[str, str], object] = {}
_WHISPER_REFS: dict[tuple[str, str], int] = {}

def _torch_has_cuda() -> bool:
    try:
        return bool(torch and torch.cuda.is_available())
    except Exception:
        return False

def _device_or_auto(dev: str | None) -> str:
    if not dev or dev == "auto":
        return "cuda" if _torch_has_cuda() else "cpu"
    if dev not in ("cpu","cuda"):
        return "cuda" if _torch_has_cuda() else "cpu"
    if dev == "cuda" and not _torch_has_cuda():
        return "cpu"
    return dev

def _get_whisper_model(name: str, device: str | None = None):
    dev = _device_or_auto(device or ASR_DEVICE)
    key = (name, dev)
    with _WHISPER_LOCK:
        if key in _WHISPER_POOL:
            _WHISPER_REFS[key] += 1
            return _WHISPER_POOL[key]
        m = whisper.load_model(name, device=dev)
        _WHISPER_POOL[key] = m
        _WHISPER_REFS[key] = 1
        return m

def _release_whisper_model(name: str, device: str | None = None):
    dev = _device_or_auto(device or ASR_DEVICE)
    key = (name, dev)
    with _WHISPER_LOCK:
        if key not in _WHISPER_POOL: return
        _WHISPER_REFS[key] -= 1
        if _WHISPER_REFS[key] > 0: return
        try:
            _WHISPER_POOL[key].to("cpu")
        except Exception:
            pass
        try:
            del _WHISPER_POOL[key]; del _WHISPER_REFS[key]
        except Exception:
            pass
    try:
        if _torch_has_cuda():
            torch.cuda.empty_cache()
    except Exception:
        pass

# Preload both models at service start (speeds up first query)
MODEL_PREVIEW = _get_whisper_model(MODEL_PREV, ASR_DEVICE)
MODEL_FULL    = _get_whisper_model(MODEL_FULL, ASR_DEVICE)

# ──────────────────────────────────────────────────────────────
# 4) Flask app
# ──────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app, origins=[o.strip() for o in CORS_ORIGINS.split(",")] if CORS_ORIGINS != "*" else "*")

def _ok(data: dict, code=200): return (jsonify({"ok": True, **data}), code)
def _err(msg: str, code=400):  return (jsonify({"ok": False, "error": msg}), code)

# ──────────────────────────────────────────────────────────────
# 5) Audio decode helpers
# ──────────────────────────────────────────────────────────────
def _pcm16_bytes_to_f32(data: bytes) -> np.ndarray:
    if not data: return np.zeros(0, dtype=np.float32)
    arr = np.frombuffer(data, dtype="<i2")
    return (arr.astype(np.float32) / 32768.0).copy()

def _decode_any_to_f32(data: bytes, declared_fmt: str | None = None, sr_hint: int | None = None) -> tuple[np.ndarray, int]:
    """
    Decode arbitrary audio bytes to mono float32 @ 16000 Hz.
    First tries soundfile; else falls back to ffmpeg if available.
    Returns (audio_f32_16k, sr_out=16000)
    """
    # Fast paths
    fmt = (declared_fmt or "").lower()
    if fmt == "pcm16":
        pcm = _pcm16_bytes_to_f32(data)
        return _resample_mono(pcm, sr_hint or 16000, 16000), 16000

    # soundfile path
    if sf is not None:
        try:
            import io
            x, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
            if x.ndim == 2:
                x = x.mean(axis=1)
            x = _resample_mono(x.astype(np.float32, copy=False), sr, 16000)
            return x, 16000
        except Exception:
            pass

    # ffmpeg fallback
    if shutil.which("ffmpeg"):
        try:
            p = subprocess.Popen(
                ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
                 "-i", "pipe:0", "-f", "f32le", "-acodec", "pcm_f32le",
                 "-ac", "1", "-ar", "16000", "pipe:1"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            out, err = p.communicate(input=data)
            if p.returncode == 0 and out:
                x = np.frombuffer(out, dtype="<f4")
                return x.astype(np.float32, copy=False), 16000
            raise RuntimeError(err.decode("utf-8","ignore") or "ffmpeg failed")
        except Exception as e:
            raise RuntimeError(f"ffmpeg decode failed: {e}")

    raise RuntimeError("No decoder available (install soundfile or ffmpeg)")

def _resample_mono(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out: return x.astype(np.float32, copy=False)
    # polyphase resample (scipy)
    try:
        from scipy.signal import resample_poly
        g = math.gcd(sr_out, sr_in)
        up, down = sr_out//g, sr_in//g
        y = resample_poly(x.astype(np.float32, copy=False), up, down)
        return y.astype(np.float32, copy=False)
    except Exception:
        # fallback: naive linear
        ratio = sr_out / float(sr_in)
        n = int(round(len(x) * ratio))
        if n <= 1: return np.zeros(0, dtype=np.float32)
        xp = np.linspace(0.0, 1.0, num=len(x), endpoint=False)
        fp = x.astype(np.float32, copy=False)
        xq = np.linspace(0.0, 1.0, num=n, endpoint=False)
        y = np.interp(xq, xp, fp).astype(np.float32)
        return y

# ──────────────────────────────────────────────────────────────
# 6) Transcribe helpers (preview & full) — now accept `prompt`
# ──────────────────────────────────────────────────────────────
def _segments_to_json(result: dict) -> list[dict]:
    out = []
    for seg in (result.get("segments") or []):
        out.append({"t0": float(seg.get("start", 0.0)), "t1": float(seg.get("end", 0.0)), "text": (seg.get("text") or "").strip()})
    return out

def _transcribe_preview(x_f32_16k: np.ndarray, prompt: str | None = None) -> dict:
    if x_f32_16k.size == 0:
        return {"text":"", "segments":[], "language": None, "duration":0.0}
    kw = {"language":"en", "fp16":False}
    if prompt:
        kw["prompt"] = str(prompt)
    r = MODEL_PREVIEW.transcribe(x_f32_16k, **kw)
    return {
        "text": (r.get("text") or "").strip(),
        "segments": _segments_to_json(r),
        "language": r.get("language"),
        "duration": float(r.get("duration", 0.0)),
    }

def _transcribe_full(x_f32_16k: np.ndarray, prompt: str | None = None) -> dict:
    if x_f32_16k.size == 0:
        return {"text":"", "segments":[], "language": None, "duration":0.0}
    kw = {"language":"en", "fp16":False}
    if prompt:
        kw["prompt"] = str(prompt)
    r = MODEL_FULL.transcribe(x_f32_16k, **kw)
    return {
        "text": (r.get("text") or "").strip(),
        "segments": _segments_to_json(r),
        "language": r.get("language"),
        "duration": float(r.get("duration", 0.0)),
    }

def _diff_new_suffix(prev: str, curr: str) -> str:
    if not curr: return ""
    prev_t, curr_t = prev.split(), curr.split()
    i, n = 0, min(len(prev_t), len(curr_t))
    while i < n and prev_t[i] == curr_t[i]: i += 1
    return " ".join(curr_t[i:]).strip()

def _extract_prompt_from_request(default: str | None = None) -> str | None:
    """
    Accept 'prompt' (or 'initial_prompt') from:
      - query string (?prompt=...)
      - JSON body ({ "prompt": "...", "initial_prompt": "..." })
      - multipart form-field 'prompt' or 'initial_prompt'
    """
    # query wins if present
    q = (request.args.get("prompt")
         or request.args.get("initial_prompt"))
    if q:
        return q

    # multipart form
    if request.content_type and "multipart/form-data" in request.content_type:
        p = request.form.get("prompt") or request.form.get("initial_prompt")
        if p:
            return p

    # JSON
    try:
        payload = request.get_json(silent=True) or {}
        p = payload.get("prompt") or payload.get("initial_prompt")
        if p:
            return str(p)
    except Exception:
        pass

    return default

# ──────────────────────────────────────────────────────────────
# 7) Streaming session state (stores `prompt`)
# ──────────────────────────────────────────────────────────────
class StreamSession:
    def __init__(self, sid: str, prompt: str | None = None):
        self.id = sid
        self.prompt = (prompt or "").strip() or None
        self.lock = threading.Lock()
        self.buf = np.zeros(0, dtype=np.float32)    # 16k mono
        self.created = time.time()
        self.last_rx = self.created
        self.done = threading.Event()
        self.events: "queue.Queue[dict]" = queue.Queue()
        self._thread = threading.Thread(target=self._preview_loop, daemon=True)
        self._prev_preview = ""
        self._running_text = ""
        self._start_preview_loop()

    def _start_preview_loop(self):
        self._thread.start()

    def append_pcm16(self, data: bytes, sr: int):
        x = _pcm16_bytes_to_f32(data)
        with self.lock:
            if sr != 16000:
                x = _resample_mono(x, sr, 16000)
            self.buf = np.concatenate((self.buf, x))
            self.last_rx = time.time()
        return len(x) / 16000.0

    def append_any(self, data: bytes, fmt: str | None = None, sr_hint: int | None = None):
        x, _ = _decode_any_to_f32(data, fmt, sr_hint)
        with self.lock:
            self.buf = np.concatenate((self.buf, x))
            self.last_rx = time.time()
        return len(x) / 16000.0

    def _preview_loop(self):
        # emits asr.partial on a cadence while not done
        while not self.done.is_set():
            time.sleep(PREVIEW_STEP_S)
            try:
                with self.lock:
                    tail_len = int(PREVIEW_WIN_S * 16000)
                    x = self.buf[-tail_len:].copy()

                if x.size == 0:
                    # heartbeat even when no audio
                    self.events.put({"event":"keepalive", "ts": int(time.time()*1000)})
                    continue

                p = _transcribe_preview(x, prompt=self.prompt)
                preview_text = p.get("text","").strip()
                if not preview_text:
                    self.events.put({"event":"keepalive", "ts": int(time.time()*1000)})
                    continue

                new_suffix = _diff_new_suffix(self._prev_preview, preview_text)
                if new_suffix:
                    # accumulate cumulative running text
                    self._running_text = (self._running_text + " " + new_suffix).strip()
                    self.events.put({
                        "event":"asr.partial",
                        "id": self.id,
                        "text": self._running_text,
                        "words_added": new_suffix,
                        "ts": int(time.time()*1000),
                    })
                else:
                    # still send a keepalive
                    self.events.put({"event":"keepalive", "ts": int(time.time()*1000)})
                self._prev_preview = preview_text
            except Exception:
                self.events.put({"event":"keepalive", "ts": int(time.time()*1000)})

    def finalize(self) -> dict:
        self.done.set()
        with self.lock:
            x = self.buf.copy()
        final = _transcribe_full(x, prompt=self.prompt)
        result = {
            "ok": True,
            "text": final.get("text",""),
            "segments": final.get("segments", []),
            "language": final.get("language"),
            "duration": final.get("duration", 0.0),
            "model": MODEL_FULL._get_name() if hasattr(MODEL_FULL, "_get_name") else os.environ.get("ASR_MODEL_FULL", "medium"),
            "device": _device_or_auto(ASR_DEVICE),
        }
        # push final on SSE
        try:
            self.events.put({"event":"asr.final", "id": self.id, "result": result, "ts": int(time.time()*1000)})
        except Exception:
            pass
        return result

# global sessions
_SESSIONS: dict[str, StreamSession] = {}
_SESS_LOCK = threading.Lock()

def _new_session_id() -> str:
    return f"sess-{int(time.time()*1000)}-{os.getpid()}-{np.random.randint(1,1_000_000)}"

def _get_session_or_404(sid: str) -> StreamSession | None:
    with _SESS_LOCK:
        return _SESSIONS.get(sid)

def _create_session(prompt: str | None = None) -> StreamSession:
    sid = _new_session_id()
    sess = StreamSession(sid, prompt=prompt)
    with _SESS_LOCK:
        _SESSIONS[sid] = sess
    return sess

def _gc_sessions():
    # remove idle/finished sessions
    now = time.time()
    with _SESS_LOCK:
        for sid, sess in list(_SESSIONS.items()):
            idle = now - max(sess.last_rx, sess.created)
            if idle > SESSION_TTL_S:
                try:
                    sess.done.set()
                except Exception:
                    pass
                del _SESSIONS[sid]

# background GC
def _gc_loop():
    while True:
        time.sleep(15)
        _gc_sessions()
threading.Thread(target=_gc_loop, daemon=True).start()

# ──────────────────────────────────────────────────────────────
# 8) Routes
# ──────────────────────────────────────────────────────────────
app.get("/health")(lambda: _ok({
    "status": "ok",
    "whisper": "ready",
    "device": _device_or_auto(ASR_DEVICE),
    "models": {
        "full": MODEL_FULL._get_name() if hasattr(MODEL_FULL, "_get_name") else os.environ.get("ASR_MODEL_FULL",""),
        "preview": MODEL_PREVIEW._get_name() if hasattr(MODEL_PREVIEW, "_get_name") else os.environ.get("ASR_MODEL_PREVIEW",""),
    }
}))

@app.get("/models")
def models():
    available = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]
    current = {
        "full": os.environ.get("ASR_MODEL_FULL", MODEL_FULL.__class__.__name__),
        "preview": os.environ.get("ASR_MODEL_PREVIEW", MODEL_PREVIEW.__class__.__name__),
    }
    return _ok({"models": [{"name": n} for n in available], "current": current})

@app.post("/recognize")
def recognize_file():
    """
    POST multipart/form-data:
        file=<audio>
        prompt=<optional bias text>

    OR application/json:
        {"body_b64":"...", "format":"wav|mp3|flac|pcm16", "sample_rate":16000, "prompt":"..." }
    """
    try:
        prompt = _extract_prompt_from_request()
        if request.content_type and "multipart/form-data" in request.content_type:
            if "file" not in request.files:
                return _err("missing file")
            file = request.files["file"]
            data = file.read()
            x, _ = _decode_any_to_f32(data, None, None)
        else:
            payload = request.get_json(force=True, silent=True) or {}
            b64 = payload.get("body_b64", "")
            fmt = payload.get("format", None)
            sr  = payload.get("sample_rate", None)
            if not b64:
                return _err("missing body_b64 (or send multipart form with file=...)")
            try:
                raw = base64.b64decode(b64, validate=False)
            except Exception:
                return _err("invalid base64")
            x, _ = _decode_any_to_f32(raw, fmt, int(sr) if sr else None)

        r = _transcribe_full(x, prompt=prompt)
        out = {
            "text": r["text"],
            "segments": r["segments"],
            "language": r["language"],
            "duration": r["duration"],
            "model": MODEL_FULL._get_name() if hasattr(MODEL_FULL, "_get_name") else os.environ.get("ASR_MODEL_FULL",""),
            "device": _device_or_auto(ASR_DEVICE),
            "used_prompt": prompt or None
        }
        return _ok(out)
    except Exception as e:
        return _err(f"{type(e).__name__}: {e}", 500)

@app.post("/recognize/stream/start")
def stream_start():
    # accept prompt from JSON or query
    prompt = _extract_prompt_from_request()
    sess = _create_session(prompt=prompt)
    return _ok({"id": sess.id, "ts": int(time.time()*1000), "used_prompt": prompt or None})

@app.post("/recognize/stream/<sid>/audio")
def stream_audio(sid: str):
    sess = _get_session_or_404(sid)
    if not sess:
        return _err("unknown session", 404)
    try:
        fmt = (request.args.get("format") or request.headers.get("X-Audio-Format") or "pcm16").lower()
        sr  = int(request.args.get("sr") or request.headers.get("X-Audio-SampleRate") or "16000")
        data = request.get_data()
        if not data:
            return _err("empty body")
        if fmt == "pcm16":
            secs = sess.append_pcm16(data, sr)
        else:
            secs = sess.append_any(data, fmt, sr)
        total = len(sess.buf) / 16000.0
        return _ok({"received": len(data), "seconds": float(secs), "total_seconds": float(total)})
    except Exception as e:
        return _err(f"{type(e).__name__}: {e}", 500)

@app.post("/recognize/stream/<sid>/end")
def stream_end(sid: str):
    sess = _get_session_or_404(sid)
    if not sess:
        return _err("unknown session", 404)
    try:
        final = sess.finalize()
        return _ok({"final": final})
    except Exception as e:
        return _err(f"{type(e).__name__}: {e}", 500)

@app.get("/recognize/stream/<sid>/events")
def stream_events(sid: str):
    sess = _get_session_or_404(sid)
    if not sess:
        return _err("unknown session", 404)
    @stream_with_context
    def _gen():
        # Initial hello
        yield "event: keepalive\n"
        yield f"data: {json.dumps({'ts': int(time.time()*1000)})}\n\n"
        idle = 0
        while True:
            try:
                ev = sess.events.get(timeout=5.0)
                idle = 0
                yield f"event: {ev.get('event','keepalive')}\n"
                yield f"data: {json.dumps(ev)}\n\n"
                if ev.get("event") == "asr.final":
                    break
            except queue.Empty:
                idle += 5
                yield "event: keepalive\n"
                yield f"data: {json.dumps({'ts': int(time.time()*1000)})}\n\n"
                if idle > SESSION_TTL_S:
                    break
    return Response(_gen(), mimetype="text/event-stream")

# ──────────────────────────────────────────────────────────────
# 9) Main
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    dev = _device_or_auto(ASR_DEVICE)
    print(f"[ASR] device={dev}  models(full,preview)=({os.environ.get('ASR_MODEL_FULL', 'medium')},{os.environ.get('ASR_MODEL_PREVIEW','base')})")
    print(f"[ASR] http://{ASR_HOST}:{ASR_PORT}")
    app.run(host=ASR_HOST, port=ASR_PORT, threaded=True)
