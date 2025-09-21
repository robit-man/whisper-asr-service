#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
asr_service.py — Whisper ASR microservice (file & live streaming)

This version includes:
- Prompt support for preview & final decoding
- Server-side auto-finalize after idle
- Preview debounced to "new audio only" to prevent drift/hallucinations
- Decoding guardrails (temperature, no_speech/logprob/compression thresholds)
- Device-aware fp16/fp32 selection & PyTorch perf hints
- Meta returned with final (avg_logprob, no_speech_prob, compression_ratio)
- Same public API as before

Routes:
  GET  /health
  GET  /models
  POST /recognize                            (file or JSON base64)
  POST /recognize/stream/start               (start session; accepts prompt)
  POST /recognize/stream/<sid>/audio         (PCM16 or any via format hint)
  POST /recognize/stream/<sid>/end           (finalize)
  GET  /recognize/stream/<sid>/events        (SSE: partials, keepalive, final)
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
    try:
        return Path(sys.executable).resolve() == PY.resolve()
    except Exception:
        return False

def _ensure_venv():
    if VENV.exists(): return
    import venv
    venv.EnvBuilder(with_pip=True).create(VENV)
    subprocess.check_call([str(PY), "-m", "pip", "install", "--upgrade", "pip"])

def _ensure_deps():
    need = []
    try: import flask  # type: ignore
    except Exception: need += ["flask", "flask-cors"]
    try: import dotenv  # type: ignore
    except Exception: need += ["python-dotenv"]
    try: import numpy  # type: ignore
    except Exception: need += ["numpy"]
    try: import scipy  # type: ignore
    except Exception: need += ["scipy"]
    try: import soundfile  # type: ignore
    except Exception: need += ["soundfile"]
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

try:
    import soundfile as sf
except Exception:
    sf = None

try:
    import torch
except Exception:
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
        "\n"
        "# NEW toggles\n"
        "ASR_FP16=auto\n"
        "ASR_TEMPERATURE=0.0\n"
        "ASR_CONDITION_PREV=false\n"
        "ASR_NO_SPEECH_THRESHOLD=0.6\n"
        "ASR_LOGPROB_THRESHOLD=-1.0\n"
        "ASR_COMPRESSION_RATIO_THRESHOLD=2.4\n"
        "ASR_PREVIEW_DET=true\n"
        "ASR_PREVIEW_MIN_DELTA_S=0.15\n"
        "ASR_AUTO_FINALIZE_S=1.4\n"
        "ASR_TORCH_THREADS=0\n"
    )
    print("→ wrote .env with defaults")

_cfg = {**dotenv_values(str(ENV))}
ASR_HOST = _cfg.get("ASR_HOST") or "0.0.0.0"
ASR_PORT = int(_cfg.get("ASR_PORT") or "8126")
ASR_DEVICE = (_cfg.get("ASR_DEVICE") or "auto").strip().lower()
MODEL_FULL_NAME = (_cfg.get("ASR_MODEL_FULL") or "medium").strip()
MODEL_PREV_NAME = (_cfg.get("ASR_MODEL_PREVIEW") or "base").strip()
PREVIEW_WIN_S  = float(_cfg.get("ASR_PREVIEW_WINDOW_S") or "6.0")
PREVIEW_STEP_S = float(_cfg.get("ASR_PREVIEW_STEP_S") or "0.9")
SESSION_TTL_S  = int(_cfg.get("ASR_SESSION_TTL_S") or "900")
CORS_ORIGINS   = (_cfg.get("CORS_ORIGINS") or "*").strip()

# New knobs
AUTO_FINALIZE_S = float(_cfg.get("ASR_AUTO_FINALIZE_S") or "1.4")
FP16_MODE = (_cfg.get("ASR_FP16") or "auto").strip().lower()    # auto|true|false
TEMP = float(_cfg.get("ASR_TEMPERATURE") or "0.0")
COND_PREV = (_cfg.get("ASR_CONDITION_PREV") or "false").strip().lower() == "true"
NS_THRESH = float(_cfg.get("ASR_NO_SPEECH_THRESHOLD") or "0.6")
LP_THRESH = float(_cfg.get("ASR_LOGPROB_THRESHOLD") or "-1.0")
CR_THRESH = float(_cfg.get("ASR_COMPRESSION_RATIO_THRESHOLD") or "2.4")
PREVIEW_DET = (_cfg.get("ASR_PREVIEW_DET") or "true").strip().lower() == "true"
PREVIEW_MIN_DELTA_S = float(_cfg.get("ASR_PREVIEW_MIN_DELTA_S") or "0.15")
TORCH_THREADS = int(_cfg.get("ASR_TORCH_THREADS") or "0")

# ──────────────────────────────────────────────────────────────
# 3) Torch/device helpers + model pool
# ──────────────────────────────────────────────────────────────
def _torch_has_cuda() -> bool:
    try:
        return bool(torch and torch.cuda.is_available())
    except Exception:
        return False

def _device_or_auto(dev: str | None) -> str:
    if not dev or dev == "auto":
        return "cuda" if _torch_has_cuda() else "cpu"
    if dev not in ("cpu", "cuda"):
        return "cuda" if _torch_has_cuda() else "cpu"
    if dev == "cuda" and not _torch_has_cuda():
        return "cpu"
    return dev

# Perf hints
if torch is not None:
    try:
        if TORCH_THREADS > 0:
            torch.set_num_threads(TORCH_THREADS)
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    except Exception:
        pass

_WHISPER_LOCK = threading.Lock()
_WHISPER_POOL: dict[tuple[str, str], object] = {}
_WHISPER_REFS: dict[tuple[str, str], int] = {}

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
        if key not in _WHISPER_POOL:
            return
        _WHISPER_REFS[key] -= 1
        if _WHISPER_REFS[key] > 0:
            return
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

# Preload (fast first hit)
MODEL_PREVIEW = _get_whisper_model(MODEL_PREV_NAME, ASR_DEVICE)
MODEL_FULL    = _get_whisper_model(MODEL_FULL_NAME, ASR_DEVICE)

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

def _resample_mono(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x.astype(np.float32, copy=False)
    try:
        from scipy.signal import resample_poly
        g = math.gcd(sr_out, sr_in)
        up, down = sr_out // g, sr_in // g
        y = resample_poly(x.astype(np.float32, copy=False), up, down)
        return y.astype(np.float32, copy=False)
    except Exception:
        ratio = sr_out / float(sr_in)
        n = int(round(len(x) * ratio))
        if n <= 1:
            return np.zeros(0, dtype=np.float32)
        xp = np.linspace(0.0, 1.0, num=len(x), endpoint=False)
        fp = x.astype(np.float32, copy=False)
        xq = np.linspace(0.0, 1.0, num=n, endpoint=False)
        y = np.interp(xq, xp, fp).astype(np.float32)
        return y

def _decode_any_to_f32(data: bytes, declared_fmt: str | None = None, sr_hint: int | None = None) -> tuple[np.ndarray, int]:
    """
    Decode arbitrary audio bytes to mono float32 @ 16000 Hz.
    First tries soundfile; else ffmpeg if available.
    """
    fmt = (declared_fmt or "").lower()
    if fmt == "pcm16":
        pcm = _pcm16_bytes_to_f32(data)
        sr = sr_hint or 16000
        x = _resample_mono(pcm, sr, 16000)
        return x, 16000

    if sf is not None:
        try:
            import io
            x, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
            if getattr(x, "ndim", 1) == 2:
                x = x.mean(axis=1)
            x = _resample_mono(x.astype(np.float32, copy=False), int(sr), 16000)
            return x, 16000
        except Exception:
            pass

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

# ──────────────────────────────────────────────────────────────
# 6) Decoding kwargs & transcription
# ──────────────────────────────────────────────────────────────
def _fp16_for_device(dev: str) -> bool:
    if FP16_MODE == "true": return True
    if FP16_MODE == "false": return False
    return dev == "cuda"

def _decoding_kw(is_preview: bool, prompt: str | None) -> dict:
    dev = _device_or_auto(ASR_DEVICE)
    kw = {
        "language": "en",
        "fp16": _fp16_for_device(dev),
        "temperature": TEMP if (not is_preview or PREVIEW_DET) else 0.2,
        "condition_on_previous_text": COND_PREV if not is_preview else False,
        "no_speech_threshold": NS_THRESH,
        "logprob_threshold": LP_THRESH,
        "compression_ratio_threshold": CR_THRESH,
        "suppress_tokens": [-1],  # keep Whisper defaults, also suppress common junk
    }
    if prompt:
        kw["prompt"] = str(prompt)
    return kw

def _segments_to_json(result: dict) -> list[dict]:
    out = []
    for seg in (result.get("segments") or []):
        out.append({
            "t0": float(seg.get("start", 0.0)),
            "t1": float(seg.get("end", 0.0)),
            "text": (seg.get("text") or "").strip()
        })
    return out

def _transcribe_preview(x_f32_16k: np.ndarray, prompt: str | None = None) -> dict:
    if x_f32_16k.size == 0:
        return {"text":"", "segments":[], "language": None, "duration":0.0}
    kw = _decoding_kw(is_preview=True, prompt=prompt)
    r = MODEL_PREVIEW.transcribe(x_f32_16k, **kw)
    return {
        "text": (r.get("text") or "").strip(),
        "segments": _segments_to_json(r),
        "language": r.get("language"),
        "duration": float(r.get("duration", 0.0)),
        "avg_logprob": float(r.get("avg_logprob", float("nan"))) if "avg_logprob" in r else None,
        "no_speech_prob": float(r.get("no_speech_prob", float("nan"))) if "no_speech_prob" in r else None,
        "compression_ratio": float(r.get("compression_ratio", float("nan"))) if "compression_ratio" in r else None,
    }

def _transcribe_full(x_f32_16k: np.ndarray, prompt: str | None = None) -> dict:
    if x_f32_16k.size == 0:
        return {"text":"", "segments":[], "language": None, "duration":0.0}
    kw = _decoding_kw(is_preview=False, prompt=prompt)
    r = MODEL_FULL.transcribe(x_f32_16k, **kw)
    return {
        "text": (r.get("text") or "").strip(),
        "segments": _segments_to_json(r),
        "language": r.get("language"),
        "duration": float(r.get("duration", 0.0)),
        "avg_logprob": float(r.get("avg_logprob", float("nan"))) if "avg_logprob" in r else None,
        "no_speech_prob": float(r.get("no_speech_prob", float("nan"))) if "no_speech_prob" in r else None,
        "compression_ratio": float(r.get("compression_ratio", float("nan"))) if "compression_ratio" in r else None,
    }

def _diff_new_suffix(prev: str, curr: str) -> str:
    if not curr:
        return ""
    prev_t, curr_t = prev.split(), curr.split()
    i, n = 0, min(len(prev_t), len(curr_t))
    while i < n and prev_t[i] == curr_t[i]:
        i += 1
    return " ".join(curr_t[i:]).strip()

def _extract_prompt_from_request(default: str | None = None) -> str | None:
    q = (request.args.get("prompt") or request.args.get("initial_prompt"))
    if q:
        return q
    if request.content_type and "multipart/form-data" in request.content_type:
        p = request.form.get("prompt") or request.form.get("initial_prompt")
        if p:
            return p
    try:
        payload = request.get_json(silent=True) or {}
        p = payload.get("prompt") or payload.get("initial_prompt")
        if p:
            return str(p)
    except Exception:
        pass
    return default

# ──────────────────────────────────────────────────────────────
# 7) Streaming session (with idle auto-finalize & debounced preview)
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

        # preview bookkeeping
        self._thread = threading.Thread(target=self._preview_loop, daemon=True)
        self._prev_preview = ""
        self._running_text = ""
        self._last_len = 0
        self._last_len_change_ts = time.time()

        self._thread.start()

    def append_pcm16(self, data: bytes, sr: int):
        x = _pcm16_bytes_to_f32(data)
        with self.lock:
            if sr != 16000:
                x = _resample_mono(x, sr, 16000)
            self.buf = np.concatenate((self.buf, x))
            self.last_rx = time.time()
            if len(self.buf) != self._last_len:
                self._last_len = len(self.buf)
                self._last_len_change_ts = self.last_rx
        return len(x) / 16000.0

    def append_any(self, data: bytes, fmt: str | None = None, sr_hint: int | None = None):
        x, _ = _decode_any_to_f32(data, fmt, sr_hint)
        with self.lock:
            self.buf = np.concatenate((self.buf, x))
            self.last_rx = time.time()
            if len(self.buf) != self._last_len:
                self._last_len = len(self.buf)
                self._last_len_change_ts = self.last_rx
        return len(x) / 16000.0

    def _preview_loop(self):
        # emits asr.partial on a cadence while not done
        while not self.done.is_set():
            time.sleep(PREVIEW_STEP_S)
            try:
                now = time.time()
                with self.lock:
                    tail_len = int(PREVIEW_WIN_S * 16000)
                    x = self.buf[-tail_len:].copy()
                    buf_len = len(self.buf)
                    last_change_age = now - self._last_len_change_ts

                # 1) Auto-finalize if no new audio for AUTO_FINALIZE_S
                if AUTO_FINALIZE_S > 0 and last_change_age >= AUTO_FINALIZE_S:
                    if (self._running_text or buf_len > 0) and not self.done.is_set():
                        try:
                            self.finalize()
                        except Exception:
                            pass
                    self.events.put({"event":"keepalive", "ts": int(now*1000)})
                    continue

                # 2) Debounce preview unless we have NEW audio since last tick
                if last_change_age < PREVIEW_MIN_DELTA_S:
                    self.events.put({"event":"keepalive", "ts": int(now*1000)})
                    continue

                if x.size == 0:
                    self.events.put({"event":"keepalive", "ts": int(now*1000)})
                    continue

                p = _transcribe_preview(x, prompt=self.prompt)
                preview_text = p.get("text","").strip()
                if not preview_text:
                    self.events.put({"event":"keepalive", "ts": int(now*1000)})
                    continue

                new_suffix = _diff_new_suffix(self._prev_preview, preview_text)
                if new_suffix:
                    self._running_text = (self._running_text + " " + new_suffix).strip()
                    self.events.put({
                        "event":"asr.partial",
                        "id": self.id,
                        "text": self._running_text,
                        "words_added": new_suffix,
                        "ts": int(now*1000),
                    })
                else:
                    self.events.put({"event":"keepalive", "ts": int(now*1000)})

                self._prev_preview = preview_text
            except Exception:
                self.events.put({"event":"keepalive", "ts": int(time.time()*1000)})

    def finalize(self) -> dict:
        if self.done.is_set():
            # Already finalized; replicate result by reading current buf
            pass
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
            "avg_logprob": final.get("avg_logprob"),
            "no_speech_prob": final.get("no_speech_prob"),
            "compression_ratio": final.get("compression_ratio"),
            "model": getattr(MODEL_FULL, "_get_name", lambda: MODEL_FULL_NAME)(),
            "device": _device_or_auto(ASR_DEVICE),
        }
        try:
            self.events.put({"event":"asr.final", "id": self.id, "result": result, "ts": int(time.time()*1000)})
        except Exception:
            pass
        return result

# Global sessions & GC
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

def _gc_loop():
    while True:
        time.sleep(15)
        _gc_sessions()

threading.Thread(target=_gc_loop, daemon=True).start()

# ──────────────────────────────────────────────────────────────
# 8) Routes
# ──────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    try:
        full_name = getattr(MODEL_FULL, "_get_name", lambda: MODEL_FULL_NAME)()
    except Exception:
        full_name = MODEL_FULL_NAME
    try:
        prev_name = getattr(MODEL_PREVIEW, "_get_name", lambda: MODEL_PREV_NAME)()
    except Exception:
        prev_name = MODEL_PREV_NAME

    return _ok({
        "status": "ok",
        "whisper": "ready",
        "device": _device_or_auto(ASR_DEVICE),
        "models": {"full": full_name, "preview": prev_name}
    })

@app.get("/models")
def models():
    available = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]
    try:
        full_name = getattr(MODEL_FULL, "_get_name", lambda: MODEL_FULL_NAME)()
    except Exception:
        full_name = MODEL_FULL_NAME
    try:
        prev_name = getattr(MODEL_PREVIEW, "_get_name", lambda: MODEL_PREV_NAME)()
    except Exception:
        prev_name = MODEL_PREV_NAME

    current = {"full": full_name, "preview": prev_name}
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
            "model": getattr(MODEL_FULL, "_get_name", lambda: MODEL_FULL_NAME)(),
            "device": _device_or_auto(ASR_DEVICE),
            "used_prompt": prompt or None,
            "avg_logprob": r.get("avg_logprob"),
            "no_speech_prob": r.get("no_speech_prob"),
            "compression_ratio": r.get("compression_ratio"),
        }
        return _ok(out)
    except Exception as e:
        return _err(f"{type(e).__name__}: {e}", 500)

@app.post("/recognize/stream/start")
def stream_start():
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
    try:
        full_name = getattr(MODEL_FULL, "_get_name", lambda: MODEL_FULL_NAME)()
    except Exception:
        full_name = MODEL_FULL_NAME
    try:
        prev_name = getattr(MODEL_PREVIEW, "_get_name", lambda: MODEL_PREV_NAME)()
    except Exception:
        prev_name = MODEL_PREV_NAME

    print(f"[ASR] device={dev}  models(full,preview)=({full_name},{prev_name})")
    print(f"[ASR] http://{ASR_HOST}:{ASR_PORT}")
    app.run(host=ASR_HOST, port=ASR_PORT, threaded=True)
