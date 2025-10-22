#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
asr_service.py — Whisper ASR microservice (file & live streaming)

Upgrades:
- Production-ready WSGI serve by default (waitress). Exported `app` for gunicorn.
- Admission control: bounded sessions + bounded model invocations with 429 back-pressure.
- Session semaphore held for lifetime of stream; released on close/GC.
- Optional per-IP token-bucket rate limits and simple body-size guard.
- Bounded SSE queues with drop strategy to protect memory.
- Fix: release custom preview models (no GPU ref-count leak).
- /metrics endpoint for router-level shedding decisions.

Routes:
  GET  /health
  GET  /metrics
  GET  /models
  POST /recognize
  POST /recognize/stream/start
  POST /recognize/stream/<sid>/audio
  POST /recognize/stream/<sid>/end
  GET  /recognize/stream/<sid>/events
"""

import os, sys, json, time, base64, threading, queue, subprocess, shutil, logging, ssl
import contextlib
from pathlib import Path

# ──────────────────────────────────────────────────────────────
# 0) venv bootstrap (install & re-exec)
# ──────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent
VENV = BASE / ".venv"
BIN  = VENV / ("Scripts" if os.name == "nt" else "bin")
PY   = BIN / ("python.exe" if os.name == "nt" else "python")
PIP  = BIN / ("pip.exe" if os.name == "nt" else "pip")

_TLS_CONFIG = {
    "bundle": None,   # type: str | None
    "disabled": False,
    "explicit": False,
}

def _env_truthy(val: str | None) -> bool:
    if not val:
        return False
    return val.strip().lower() in {"1", "true", "yes", "on"}

def _configure_tls_trust():
    """
    Configure a CA bundle so HTTPS downloads work on non-Debian distros or custom envs.
    """
    global _TLS_CONFIG

    if _env_truthy(os.environ.get("ASR_SSL_NO_VERIFY")):
        try:
            ssl._create_default_https_context = ssl._create_unverified_context
            _TLS_CONFIG["disabled"] = True
            print("[ASR] WARNING: TLS verification disabled via ASR_SSL_NO_VERIFY=1", file=sys.stderr)
        except Exception:
            pass
        return

    ca_hint = os.environ.get("ASR_SSL_CA_BUNDLE")
    if ca_hint:
        hint_path = Path(ca_hint).expanduser()
        if hint_path.exists():
            os.environ.setdefault("SSL_CERT_FILE", str(hint_path))
            os.environ.setdefault("REQUESTS_CA_BUNDLE", str(hint_path))
            os.environ.setdefault("PIP_CERT", str(hint_path))
            _TLS_CONFIG["bundle"] = str(hint_path)
            _TLS_CONFIG["explicit"] = True
            try:
                ssl._create_default_https_context = lambda *args, **kwargs: ssl.create_default_context(cafile=str(hint_path))
            except Exception:
                pass
            return
        else:
            print(f"[ASR] warning: ASR_SSL_CA_BUNDLE={hint_path} does not exist", file=sys.stderr)

    if os.environ.get("SSL_CERT_FILE") or os.environ.get("REQUESTS_CA_BUNDLE") or os.environ.get("PIP_CERT"):
        current = os.environ.get("SSL_CERT_FILE") or os.environ.get("REQUESTS_CA_BUNDLE") or os.environ.get("PIP_CERT")
        _TLS_CONFIG["bundle"] = current
        _TLS_CONFIG["explicit"] = True
        return

    candidates: list[Path] = []
    with contextlib.suppress(Exception):
        import certifi  # type: ignore
        candidates.append(Path(certifi.where()))

    candidates.extend([
        Path("/etc/ssl/certs/ca-certificates.crt"),      # Debian/Ubuntu
        Path("/etc/pki/tls/certs/ca-bundle.crt"),        # RHEL/CentOS/Fedora
        Path("/etc/ssl/ca-bundle.pem"),                  # SLES/openSUSE
        Path("/etc/ssl/cert.pem"),                       # Alpine, macOS python.org builds
        Path("/usr/local/etc/openssl/cert.pem"),         # Homebrew OpenSSL
        Path("/System/Library/OpenSSL/certs/cert.pem"),  # Older macOS
    ])

    for cand in candidates:
        if cand and cand.exists():
            os.environ.setdefault("SSL_CERT_FILE", str(cand))
            os.environ.setdefault("REQUESTS_CA_BUNDLE", str(cand))
            os.environ.setdefault("PIP_CERT", str(cand))
            _TLS_CONFIG["bundle"] = str(cand)
            try:
                ssl._create_default_https_context = lambda *args, **kwargs: ssl.create_default_context(cafile=str(cand))
            except Exception:
                pass
            break

_configure_tls_trust()

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
    try: import waitress  # type: ignore
    except Exception: need += ["waitress"]
    if need:
        subprocess.check_call([str(PIP), "install", *need])

if not _in_venv():
    _ensure_venv()
    os.execv(str(PY), [str(PY), *sys.argv])
_ensure_deps()

# ──────────────────────────────────────────────────────────────
# Logging setup
# ──────────────────────────────────────────────────────────────
LOG_LEVEL = os.environ.get("ASR_LOG_LEVEL", "DEBUG").upper()
if not logging.getLogger().handlers:
    logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO),
                        format='[%(asctime)s] %(levelname)s %(name)s: %(message)s')
LOG = logging.getLogger("whisper_asr.service")
LOG.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

# ──────────────────────────────────────────────────────────────
# 1) imports (post-venv)
# ──────────────────────────────────────────────────────────────
import math
import numpy as np
from flask import Flask, request, jsonify, Response, stream_with_context, g
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
        "CORS_ORIGINS=*\n"
        "\n"
        "# ===== Model names =====\n"
        "ASR_MODEL_FULL=medium\n"
        "ASR_FAST_MODEL_PREVIEW=tiny\n"
        "ASR_ACCURATE_MODEL_PREVIEW=base\n"
        "\n"
        "# ===== Mode defaults =====\n"
        "ASR_MODE_DEFAULT=fast\n"
        "ASR_FAST_PREVIEW_WINDOW_S=3.0\n"
        "ASR_FAST_PREVIEW_STEP_S=0.25\n"
        "ASR_ACCURATE_PREVIEW_WINDOW_S=6.0\n"
        "ASR_ACCURATE_PREVIEW_STEP_S=0.9\n"
        "\n"
        "# ===== Final decode =====\n"
        "ASR_FINAL_BEAM_SIZE=5\n"
        "ASR_FINAL_WORD_TIMESTAMPS=true\n"
        "\n"
        "# ===== Safety/quality knobs =====\n"
        "ASR_TEMPERATURE=0.0\n"
        "ASR_CONDITION_PREV=false\n"
        "ASR_NO_SPEECH_THRESHOLD=0.6\n"
        "ASR_LOGPROB_THRESHOLD=-1.0\n"
        "ASR_COMPRESSION_RATIO_THRESHOLD=2.4\n"
        "\n"
        "# ===== Runtime =====\n"
        "ASR_SESSION_TTL_S=900\n"
        "ASR_AUTO_FINALIZE_S=1.4\n"
        "ASR_FP16=auto\n"
        "ASR_TORCH_THREADS=0\n"
        "\n"
        "# ===== Concurrency / capacity =====\n"
        "ASR_MAX_SESSIONS=16\n"
        "ASR_SESSION_ACQUIRE_TIMEOUT_S=3.0\n"
        "ASR_MAX_TRANSCRIPTS=2\n"
        "ASR_TRANSCRIBE_TIMEOUT_S=30.0\n"
        "ASR_EVENTS_QUEUE_MAX=256\n"
        "ASR_MAX_BODY_BYTES=10485760\n"
        "\n"
        "# ===== Rate limiting =====\n"
        "ASR_RLIMIT_RPS=10\n"
        "ASR_RLIMIT_BURST=20\n"
    )
    print("→ wrote .env with defaults")

_cfg = {**dotenv_values(str(ENV))}

ASR_HOST = _cfg.get("ASR_HOST") or "0.0.0.0"
ASR_PORT = int(_cfg.get("ASR_PORT") or "8126")
ASR_DEVICE = (_cfg.get("ASR_DEVICE") or "auto").strip().lower()
CORS_ORIGINS = (_cfg.get("CORS_ORIGINS") or "*").strip()

# Models
MODEL_FULL_NAME   = (_cfg.get("ASR_MODEL_FULL") or "medium").strip()
FAST_MODEL_PREV   = (_cfg.get("ASR_FAST_MODEL_PREVIEW") or "tiny").strip()
ACC_MODEL_PREV    = (_cfg.get("ASR_ACCURATE_MODEL_PREVIEW") or "base").strip()

# Mode defaults
MODE_DEFAULT      = (_cfg.get("ASR_MODE_DEFAULT") or "fast").strip().lower()
FAST_WIN_S        = float(_cfg.get("ASR_FAST_PREVIEW_WINDOW_S") or "3.0")
FAST_STEP_S       = float(_cfg.get("ASR_FAST_PREVIEW_STEP_S") or "0.25")
ACC_WIN_S         = float(_cfg.get("ASR_ACCURATE_PREVIEW_WINDOW_S") or "6.0")
ACC_STEP_S        = float(_cfg.get("ASR_ACCURATE_PREVIEW_STEP_S") or "0.9")

# Final decode
FINAL_BEAM_SIZE   = int(_cfg.get("ASR_FINAL_BEAM_SIZE") or "5")
FINAL_WORDS       = str(_cfg.get("ASR_FINAL_WORD_TIMESTAMPS") or "true").strip().lower() == "true"

# Safety/quality knobs
TEMP              = float(_cfg.get("ASR_TEMPERATURE") or "0.0")
COND_PREV         = (_cfg.get("ASR_CONDITION_PREV") or "false").strip().lower() == "true"
NS_THRESH         = float(_cfg.get("ASR_NO_SPEECH_THRESHOLD") or "0.6")
LP_THRESH         = float(_cfg.get("ASR_LOGPROB_THRESHOLD") or "-1.0")
CR_THRESH         = float(_cfg.get("ASR_COMPRESSION_RATIO_THRESHOLD") or "2.4")

# Runtime
SESSION_TTL_S     = int(_cfg.get("ASR_SESSION_TTL_S") or "900")
AUTO_FINALIZE_S   = float(_cfg.get("ASR_AUTO_FINALIZE_S") or "1.4")
STREAM_PHRASE_TIMEOUT = float(_cfg.get("ASR_STREAM_PHRASE_TIMEOUT") or "2.2")
FP16_MODE         = (_cfg.get("ASR_FP16") or "auto").strip().lower()    # auto|true|false
TORCH_THREADS     = int(_cfg.get("ASR_TORCH_THREADS") or "0")

# Concurrency / capacity controls
ASR_MAX_SESSIONS                = max(1, int(_cfg.get("ASR_MAX_SESSIONS") or "16"))
ASR_SESSION_ACQUIRE_TIMEOUT_S   = float(_cfg.get("ASR_SESSION_ACQUIRE_TIMEOUT_S") or "3.0")
ASR_MAX_TRANSCRIPTS             = max(1, int(_cfg.get("ASR_MAX_TRANSCRIPTS") or "2"))
ASR_TRANSCRIBE_TIMEOUT_S        = float(_cfg.get("ASR_TRANSCRIBE_TIMEOUT_S") or "30.0")
ASR_EVENTS_QUEUE_MAX            = max(8, int(_cfg.get("ASR_EVENTS_QUEUE_MAX") or "256"))
ASR_MAX_BODY_BYTES              = int(_cfg.get("ASR_MAX_BODY_BYTES") or "10485760")

# Rate limits
ASR_RLIMIT_RPS   = max(1, int(_cfg.get("ASR_RLIMIT_RPS") or "10"))
ASR_RLIMIT_BURST = max(1, int(_cfg.get("ASR_RLIMIT_BURST") or "20"))

# ──────────────────────────────────────────────────────────────
# 3) Torch/device helpers + model pool
# ──────────────────────────────────────────────────────────────

def _detect_cuda_devices():
    devices: list[dict[str, object]] = []
    unusable: list[str] = []
    if torch is None:
        return devices, unusable
    try:
        if not torch.cuda.is_available():
            return devices, unusable
    except Exception as exc:
        unusable.append(f"cuda availability check failed: {exc}")
        return [], unusable

    try:
        arch_list = []
        with contextlib.suppress(Exception):
            arch_list = torch.cuda.get_arch_list()
        compiled_sms = set()
        for arch in arch_list:
            if isinstance(arch, str) and arch.startswith("sm_"):
                with contextlib.suppress(ValueError):
                    compiled_sms.add(int(arch.split("_")[1]))

        for idx in range(torch.cuda.device_count()):
            try:
                name = torch.cuda.get_device_name(idx)
                major, minor = torch.cuda.get_device_capability(idx)
                sm = major * 10 + minor
            except Exception as exc:
                unusable.append(f"cuda:{idx} query failed: {exc}")
                continue

            if compiled_sms and sm not in compiled_sms and sm < min(compiled_sms):
                unusable.append(f"cuda:{idx} {name} sm_{sm} not in supported arch list {sorted(compiled_sms)}")
                continue

            try:
                torch.zeros(1, device=f"cuda:{idx}")
            except Exception as exc:
                unusable.append(f"cuda:{idx} {name} unusable: {exc}")
                continue

            devices.append({"id": idx, "name": name, "sm": sm})
    except Exception as exc:
        unusable.append(f"cuda device scan failed: {exc}")
    return devices, unusable


_CUDA_DEVICES, _CUDA_UNUSABLE = _detect_cuda_devices()
_WARNED_INVALID_DEVICE = False

if _CUDA_DEVICES:
    summary = ", ".join(f"cuda:{d['id']} {d['name']} (sm_{int(d['sm'])})" for d in _CUDA_DEVICES)
    print(f"[ASR] CUDA devices: {summary}")
if _CUDA_UNUSABLE:
    for msg in _CUDA_UNUSABLE:
        print(f"[ASR] cuda warning: {msg}")

def _torch_has_cuda() -> bool:
    return bool(_CUDA_DEVICES)

def _device_or_auto(dev: str | None) -> str:
    global _WARNED_INVALID_DEVICE
    if not dev or dev == "auto":
        return f"cuda:{_CUDA_DEVICES[0]['id']}" if _CUDA_DEVICES else "cpu"

    dev = dev.strip().lower()
    if dev == "cpu":
        return "cpu"

    if dev.startswith("cuda"):
        # Allow cuda or cuda:<idx>
        if dev == "cuda":
            return f"cuda:{_CUDA_DEVICES[0]['id']}" if _CUDA_DEVICES else "cpu"
        parts = dev.split(":", 1)
        if len(parts) == 2:
            try:
                idx = int(parts[1])
                if any(d["id"] == idx for d in _CUDA_DEVICES):
                    return f"cuda:{idx}"
            except ValueError:
                pass
        if _CUDA_DEVICES:
            return f"cuda:{_CUDA_DEVICES[0]['id']}"
        if not _WARNED_INVALID_DEVICE:
            print(f"[ASR] requested device '{dev}' not available, falling back to CPU")
            _WARNED_INVALID_DEVICE = True
        return "cpu"

    return "cpu"

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

_SESSION_SEM = threading.BoundedSemaphore(ASR_MAX_SESSIONS)
_TRANSCRIBE_SEM = threading.BoundedSemaphore(ASR_MAX_TRANSCRIPTS)
_TRANSCRIBE_INFLIGHT = 0
_TRANSCRIBE_INFLIGHT_LOCK = threading.Lock()

def _is_tls_cert_error(exc: Exception | None) -> bool:
    if exc is None:
        return False
    import urllib.error
    if isinstance(exc, ssl.SSLCertVerificationError):
        return True
    if isinstance(exc, ssl.SSLError) and "CERTIFICATE_VERIFY_FAILED" in str(exc):
        return True
    if isinstance(exc, urllib.error.URLError):
        return _is_tls_cert_error(getattr(exc, "reason", None))
    cause = getattr(exc, "__cause__", None)
    if cause and cause is not exc and _is_tls_cert_error(cause):
        return True
    ctx = getattr(exc, "__context__", None)
    if ctx and ctx is not exc and _is_tls_cert_error(ctx):
        return True
    return False

@contextlib.contextmanager
def _transcribe_slot(timeout: float = ASR_TRANSCRIBE_TIMEOUT_S):
    global _TRANSCRIBE_INFLIGHT
    if not _TRANSCRIBE_SEM.acquire(timeout=timeout):
        raise TimeoutError("ASR transcriber at capacity")
    with _TRANSCRIBE_INFLIGHT_LOCK:
        _TRANSCRIBE_INFLIGHT += 1
    try:
        yield
    finally:
        with _TRANSCRIBE_INFLIGHT_LOCK:
            _TRANSCRIBE_INFLIGHT -= 1
        _TRANSCRIBE_SEM.release()

def _get_whisper_model(name: str, device: str | None = None):
    dev = _device_or_auto(device or ASR_DEVICE)
    key = (name, dev)
    with _WHISPER_LOCK:
        if key in _WHISPER_POOL:
            _WHISPER_REFS[key] += 1
            return _WHISPER_POOL[key]
        try:
            m = whisper.load_model(name, device=dev)
        except Exception as exc:
            if _is_tls_cert_error(exc):
                hint_parts = ["failed to download Whisper model due to TLS certificate verification error."]
                if _TLS_CONFIG["disabled"]:
                    hint_parts.append("Certificate checks are disabled (ASR_SSL_NO_VERIFY); the remote certificate may be untrusted.")
                elif _TLS_CONFIG["explicit"]:
                    bundle = _TLS_CONFIG["bundle"] or "custom SSL_CERT_FILE/REQUESTS_CA_BUNDLE"
                    hint_parts.append(f"Verify the bundle at {bundle} trusts the issuer.")
                else:
                    hint_parts.append("Set ASR_SSL_CA_BUNDLE to your CA bundle path or, as a last resort, ASR_SSL_NO_VERIFY=1.")
                raise RuntimeError(" ".join(hint_parts)) from exc
            raise
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

# Preload models
MODEL_FULL          = _get_whisper_model(MODEL_FULL_NAME, ASR_DEVICE)
MODEL_FAST_PREVIEW  = _get_whisper_model(FAST_MODEL_PREV, ASR_DEVICE)
MODEL_ACC_PREVIEW   = _get_whisper_model(ACC_MODEL_PREV, ASR_DEVICE)

def _model_name(m, fallback: str) -> str:
    try:
        return getattr(m, "_get_name", lambda: fallback)()
    except Exception:
        return fallback

# ──────────────────────────────────────────────────────────────
# 4) Flask app + rate limiter
# ──────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app, origins=[o.strip() for o in CORS_ORIGINS.split(",")] if CORS_ORIGINS != "*" else "*")

def _ok(data: dict, code=200, headers: dict | None = None):
    resp = jsonify({"ok": True, **data})
    return (resp, code, headers or {})
def _err(msg: str, code=400, headers: dict | None = None):
    resp = jsonify({"ok": False, "error": msg})
    return (resp, code, headers or {})

# Simple token-bucket per-IP
class _Bucket:
    __slots__ = ("ts","tokens")
    def __init__(self): self.ts=time.time(); self.tokens=float(ASR_RLIMIT_BURST)
_rlock = threading.Lock()
_buckets: dict[str,_Bucket] = {}

@app.before_request
def _before():
    # Body guard
    cl = request.content_length or 0
    if cl and cl > ASR_MAX_BODY_BYTES:
        return _err("payload too large", 413)

    # Rate limit
    ip = request.headers.get("X-Forwarded-For","").split(",")[0].strip() or request.remote_addr or "0.0.0.0"
    now = time.time()
    with _rlock:
        b = _buckets.get(ip)
        if b is None:
            b = _Bucket(); _buckets[ip]=b
        # refill
        dt = max(0.0, now - b.ts); b.ts = now
        b.tokens = min(float(ASR_RLIMIT_BURST), b.tokens + dt*ASR_RLIMIT_RPS)
        if b.tokens < 1.0:
            return _err("rate limit", 429, {"Retry-After":"1"})
        b.tokens -= 1.0
    g.client_ip = ip

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

_COMMON_FILTERS = dict(
    compression_ratio_threshold=CR_THRESH,
    logprob_threshold=LP_THRESH,
    no_speech_threshold=NS_THRESH,
)

FAST_PREVIEW_OPTS = dict(
    language="en",
    fp16=_fp16_for_device(_device_or_auto(ASR_DEVICE)),
    condition_on_previous_text=False,
    temperature=0.0,
    word_timestamps=False,
    **_COMMON_FILTERS,
)

ACCURATE_PREVIEW_OPTS = dict(
    language="en",
    fp16=_fp16_for_device(_device_or_auto(ASR_DEVICE)),
    condition_on_previous_text=False,
    temperature=0.0,
    word_timestamps=False,
    **_COMMON_FILTERS,
)

FINAL_OPTS = dict(
    language="en",
    fp16=_fp16_for_device(_device_or_auto(ASR_DEVICE)),
    temperature=TEMP,
    **_COMMON_FILTERS,
)
if FINAL_BEAM_SIZE and FINAL_BEAM_SIZE > 1:
    FINAL_OPTS["beam_size"] = FINAL_BEAM_SIZE
if FINAL_WORDS:
    FINAL_OPTS["word_timestamps"] = True

def _segments_to_json(result: dict) -> list[dict]:
    out = []
    for seg in (result.get("segments") or []):
        out.append({
            "t0": float(seg.get("start", 0.0)),
            "t1": float(seg.get("end", 0.0)),
            "text": (seg.get("text") or "").strip()
        })
    return out

def _transcribe_preview_with(model, x_f32_16k: np.ndarray, prompt: str | None, opts: dict) -> dict:
    if x_f32_16k.size == 0:
        return {"text":"", "segments":[], "language": None, "duration":0.0}
    kw = dict(opts)
    if prompt:
        kw["prompt"] = str(prompt)
    with _transcribe_slot():
        r = model.transcribe(x_f32_16k, **kw)
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
    kw = dict(FINAL_OPTS)
    if prompt:
        kw["prompt"] = str(prompt)
    with _transcribe_slot():
        r = MODEL_FULL.transcribe(x_f32_16k, **kw)
    out = {
        "text": (r.get("text") or "").strip(),
        "segments": _segments_to_json(r),
        "language": r.get("language"),
        "duration": float(r.get("duration", 0.0)),
        "avg_logprob": float(r.get("avg_logprob", float("nan"))) if "avg_logprob" in r else None,
        "no_speech_prob": float(r.get("no_speech_prob", float("nan"))) if "no_speech_prob" in r else None,
        "compression_ratio": float(r.get("compression_ratio", float("nan"))) if "compression_ratio" in r else None,
    }
    if FINAL_OPTS.get("word_timestamps"):
        words = []
        for seg in (r.get("segments") or []):
            for w in (seg.get("words") or []):
                words.append({
                    "t0": float(w.get("start", 0.0)),
                    "t1": float(w.get("end", 0.0)),
                    "text": (w.get("word") or "").strip()
                })
        out["words"] = words
    return out

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
    if q: return q
    if request.content_type and "multipart/form-data" in request.content_type:
        p = request.form.get("prompt") or request.form.get("initial_prompt")
        if p: return p
    try:
        payload = request.get_json(silent=True) or {}
        p = payload.get("prompt") or payload.get("initial_prompt")
        if p: return str(p)
    except Exception:
        pass
    return default

# ──────────────────────────────────────────────────────────────
# 7) Streaming session
# ──────────────────────────────────────────────────────────────
class StreamSession:
    def __init__(self,
                 sid: str,
                 prompt: str | None = None,
                 mode: str = "fast",
                 preview_model=None,
                 preview_opts: dict | None = None,
                 win_s: float | None = None,
                 step_s: float | None = None,
                 preview_model_name: str | None = None,
                 preview_is_custom: bool = False):
        self.id = sid
        self.prompt = (prompt or "").strip() or None
        self.mode = (mode or "fast").lower()
        self.preview_model = preview_model or (MODEL_FAST_PREVIEW if self.mode == "fast" else MODEL_ACC_PREVIEW)
        self.preview_opts = dict(preview_opts or (FAST_PREVIEW_OPTS if self.mode == "fast" else ACCURATE_PREVIEW_OPTS))
        self.win_s = float(win_s if win_s is not None else (FAST_WIN_S if self.mode == "fast" else ACC_WIN_S))
        self.step_s = float(step_s if step_s is not None else (FAST_STEP_S if self.mode == "fast" else ACC_STEP_S))

        self.lock = threading.Lock()
        self.buf = np.zeros(0, dtype=np.float32)    # 16k mono
        self.created = time.time()
        self.last_rx = self.created
        self.done = threading.Event()
        self.events: "queue.Queue[dict]" = queue.Queue(maxsize=ASR_EVENTS_QUEUE_MAX)
        self.preview_model_name = preview_model_name
        self.preview_is_custom = bool(preview_is_custom and preview_model_name)
        self._closed = False
        self._sem_released = False

        # preview bookkeeping
        self._thread = threading.Thread(target=self._preview_loop, daemon=True, name=f"asr-preview-{sid}")
        self._prev_preview = ""
        self._running_text = ""
        self._last_len = 0
        self._last_len_change_ts = time.time()
        self._last_partial_emit_ts = 0.0
        self._partial_marked_final = False
        self._partial_final_after = max(0.45, min(AUTO_FINALIZE_S * 0.5 if AUTO_FINALIZE_S else 0.8, 1.5))
        self._partial_silence_rms = 0.012
        self._phrase_timeout = STREAM_PHRASE_TIMEOUT
        self._partial_seq = 0

        self._thread.start()

    def _enqueue_event(self, event: dict) -> None:
        try:
            self.events.put_nowait(event)
            return
        except queue.Full:
            pass
        if event.get("event") == "keepalive":
            return
        try:
            _ = self.events.get_nowait()  # drop oldest
        except queue.Empty:
            pass
        try:
            self.events.put_nowait(event)
        except queue.Full:
            pass

    def _release_sem_once(self) -> None:
        if self._sem_released:
            return
        self._sem_released = True
        with contextlib.suppress(ValueError):
            _SESSION_SEM.release()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.done.set()
        if self.preview_is_custom and self.preview_model_name:
            _release_whisper_model(self.preview_model_name, ASR_DEVICE)
        self._release_sem_once()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def _should_emit_partial_final(self, last_change_age: float, tail_rms: float) -> bool:
        if self._partial_marked_final:
            return False
        if not self._running_text:
            return False
        if last_change_age < self._partial_final_after:
            return False
        if tail_rms > self._partial_silence_rms:
            return False
        return True

    def append_pcm16(self, data: bytes, sr: int):
        if self.done.is_set():
            raise RuntimeError("session already finalized")
        x = _pcm16_bytes_to_f32(data)
        with self.lock:
            if sr != 16000:
                x = _resample_mono(x, sr, 16000)
            rms = float(np.sqrt(np.mean(x ** 2))) if x.size else 0.0
            LOG.debug("session %s: chunk %d samples (sr=%d, rms=%.4f)", self.id, len(x), sr, rms)
            self.buf = np.concatenate((self.buf, x))
            self.last_rx = time.time()
            if len(self.buf) != self._last_len:
                self._last_len = len(self.buf)
                self._last_len_change_ts = self.last_rx
            self._partial_marked_final = False
        return len(x) / 16000.0

    def append_any(self, data: bytes, fmt: str | None = None, sr_hint: int | None = None):
        if self.done.is_set():
            raise RuntimeError("session already finalized")
        x, sr = _decode_any_to_f32(data, fmt, sr_hint)
        with self.lock:
            rms = float(np.sqrt(np.mean(x ** 2))) if x.size else 0.0
            LOG.debug("session %s: chunk %d samples (%s, sr=%s, rms=%.4f)", self.id, len(x), fmt or 'raw', sr or 'auto', rms)
            self.buf = np.concatenate((self.buf, x))
            self.last_rx = time.time()
            if len(self.buf) != self._last_len:
                self._last_len = len(self.buf)
                self._last_len_change_ts = self.last_rx
            self._partial_marked_final = False
        return len(x) / 16000.0

    def _preview_loop(self):
        # emits asr.partial/keepalive while not done
        while not self.done.is_set():
            time.sleep(max(0.05, float(self.step_s)))
            try:
                now = time.time()
                with self.lock:
                    tail_len = int(self.win_s * 16000)
                    x = self.buf[-tail_len:].copy()
                    buf_len = len(self.buf)
                    last_change_age = now - self._last_len_change_ts
                    tail_rms = float(np.sqrt(np.mean(x ** 2))) if x.size else 0.0

                phrase_gap = now - (self._last_partial_emit_ts or self.created)

                # 1) Auto-finalize if no new audio for AUTO_FINALIZE_S
                if AUTO_FINALIZE_S > 0 and last_change_age >= AUTO_FINALIZE_S:
                    if (self._running_text or buf_len > 0) and not self.done.is_set():
                        try:
                            self.finalize()
                        except Exception:
                            pass
                    self._enqueue_event({"event":"keepalive", "ts": int(now*1000)})
                    continue

                if x.size == 0:
                    if (self._running_text and not self._partial_marked_final and
                            phrase_gap >= self._phrase_timeout):
                        self._partial_marked_final = True
                        self._partial_seq += 1
                        self._last_partial_emit_ts = now
                        payload = {
                            "event": "asr.partial",
                            "id": self.id,
                            "text": self._running_text,
                            "words_added": "",
                            "mode": self.mode,
                            "final": True,
                            "gap_ms": int(phrase_gap * 1000),
                            "tail_rms": 0.0,
                            "seq": self._partial_seq,
                            "ts": int(now * 1000),
                        }
                        self._enqueue_event(payload)
                        LOG.info("session %s: forced partial finalize text='%s' gap=%.2fs", self.id, self._running_text[-80:], phrase_gap)
                    self._enqueue_event({"event":"keepalive", "ts": int(now*1000)})
                    continue

                r = _transcribe_preview_with(self.preview_model, x, self.prompt, self.preview_opts)
                preview_text = r.get("text","").strip()
                if not preview_text:
                    self._enqueue_event({"event":"keepalive", "ts": int(now*1000)})
                    continue

                new_suffix = _diff_new_suffix(self._prev_preview, preview_text)
                if new_suffix:
                    if phrase_gap >= self._phrase_timeout:
                        self._running_text = new_suffix.strip()
                    else:
                        self._running_text = (self._running_text + " " + new_suffix).strip()
                    self._last_partial_emit_ts = now
                    self._partial_marked_final = False
                    self._partial_seq += 1
                    self._enqueue_event({
                        "event": "asr.partial",
                        "id": self.id,
                        "text": self._running_text,
                        "words_added": new_suffix,
                        "mode": self.mode,
                        "final": False,
                        "gap_ms": int(last_change_age * 1000),
                        "tail_rms": tail_rms,
                        "seq": self._partial_seq,
                        "ts": int(now * 1000),
                    })
                    LOG.debug("session %s: partial update final=%s text='%s'", self.id, False, self._running_text[-80:])
                    self._prev_preview = preview_text
                    continue

                if self._should_emit_partial_final(last_change_age, tail_rms):
                    self._partial_marked_final = True
                    self._partial_seq += 1
                    self._last_partial_emit_ts = now
                    payload = {
                        "event": "asr.partial",
                        "id": self.id,
                        "text": self._running_text,
                        "words_added": "",
                        "mode": self.mode,
                        "final": True,
                        "gap_ms": int(last_change_age * 1000),
                        "tail_rms": tail_rms,
                        "seq": self._partial_seq,
                        "ts": int(now * 1000),
                    }
                    self._enqueue_event(payload)
                    LOG.info("session %s: partial stabilized text='%s' gap=%.2fs rms=%.4f", self.id, self._running_text[-80:], last_change_age, tail_rms)
                    self._prev_preview = preview_text
                    continue

                self._enqueue_event({"event":"keepalive", "ts": int(now*1000)})
                self._prev_preview = preview_text
            except Exception:
                self._enqueue_event({"event":"keepalive", "ts": int(time.time()*1000)})

    def finalize(self) -> dict:
        with self.lock:
            x = self.buf.copy()
        self.done.set()
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
            "model": _model_name(MODEL_FULL, MODEL_FULL_NAME),
            "device": _device_or_auto(ASR_DEVICE),
            "mode": self.mode,
        }
        if "words" in final:
            result["words"] = final["words"]
        final_text = result.get("text") or ""
        snippet = final_text.strip()[:120]
        LOG.info("session %s: finalized transcript (%d chars, duration=%.2fs) text='%s'", self.id, len(final_text), result.get("duration") or 0.0, snippet)
        self._running_text = ""
        self._partial_marked_final = False
        self._last_partial_emit_ts = time.time()
        try:
            self._enqueue_event({"event":"asr.final", "id": self.id, "result": result, "ts": int(time.time()*1000)})
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

def _create_session(*, prompt: str | None, mode: str, preview_model, preview_opts: dict,
                    win_s: float, step_s: float, preview_model_name: str | None, preview_is_custom: bool) -> StreamSession:
    sid = _new_session_id()
    sess = StreamSession(sid,
                         prompt=prompt,
                         mode=mode,
                         preview_model=preview_model,
                         preview_opts=preview_opts,
                         win_s=win_s,
                         step_s=step_s,
                         preview_model_name=preview_model_name,
                         preview_is_custom=preview_is_custom)
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
                    sess.close()
                except Exception:
                    pass
                del _SESSIONS[sid]

def _gc_loop():
    while True:
        time.sleep(15)
        _gc_sessions()
threading.Thread(target=_gc_loop, daemon=True, name="asr-gc").start()

# ──────────────────────────────────────────────────────────────
# 8) Routes
# ──────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return _ok({
        "status": "ok",
        "whisper": "ready",
        "device": _device_or_auto(ASR_DEVICE),
        "models": {
            "full": _model_name(MODEL_FULL, MODEL_FULL_NAME),
            "preview_fast": _model_name(MODEL_FAST_PREVIEW, FAST_MODEL_PREV),
            "preview_accurate": _model_name(MODEL_ACC_PREVIEW, ACC_MODEL_PREV),
        },
        "modes": {
            "default": MODE_DEFAULT,
            "fast": {"window_s": FAST_WIN_S, "step_s": FAST_STEP_S, "model": FAST_MODEL_PREV},
            "accurate": {"window_s": ACC_WIN_S, "step_s": ACC_STEP_S, "model": ACC_MODEL_PREV},
        }
    })

@app.get("/metrics")
def metrics():
    with _SESS_LOCK:
        n_sessions = len(_SESSIONS)
        qsizes = [s.events.qsize() for s in _SESSIONS.values()]
    with _TRANSCRIBE_INFLIGHT_LOCK:
        inflight = _TRANSCRIBE_INFLIGHT
    return _ok({
        "active_sessions": n_sessions,
        "sessions_capacity": ASR_MAX_SESSIONS,
        "events_queue_sizes": qsizes,
        "transcribe_inflight": inflight,
        "transcribe_capacity": ASR_MAX_TRANSCRIPTS
    })

@app.get("/models")
def models():
    available = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]
    current = {
        "full": _model_name(MODEL_FULL, MODEL_FULL_NAME),
        "preview_fast": _model_name(MODEL_FAST_PREVIEW, FAST_MODEL_PREV),
        "preview_accurate": _model_name(MODEL_ACC_PREVIEW, ACC_MODEL_PREV),
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
            "model": _model_name(MODEL_FULL, MODEL_FULL_NAME),
            "device": _device_or_auto(ASR_DEVICE),
            "used_prompt": prompt or None,
            "avg_logprob": r.get("avg_logprob"),
            "no_speech_prob": r.get("no_speech_prob"),
            "compression_ratio": r.get("compression_ratio"),
            "words": r.get("words"),
        }
        return _ok(out)
    except TimeoutError as e:
        return _err(str(e), 429, {"Retry-After":"2"})
    except Exception as e:
        return _err(f"{type(e).__name__}: {e}", 500)

@app.post("/recognize/stream/start")
def stream_start():
    # Capacity admission
    if not _SESSION_SEM.acquire(timeout=ASR_SESSION_ACQUIRE_TIMEOUT_S):
        return _err("asr at capacity", 429, {"Retry-After":"2"})
    payload = request.get_json(silent=True) or {}
    prompt = _extract_prompt_from_request()
    mode = (request.args.get("mode") or payload.get("mode") or MODE_DEFAULT).strip().lower()
    mode = mode if mode in ("fast", "accurate") else MODE_DEFAULT

    # cadence overrides
    win_s  = float(request.args.get("preview_window_s") or payload.get("preview_window_s") or (FAST_WIN_S if mode=="fast" else ACC_WIN_S))
    step_s = float(request.args.get("preview_step_s")  or payload.get("preview_step_s")  or (FAST_STEP_S if mode=="fast" else ACC_STEP_S))

    # preview model override by name (optional)
    model_name = (request.args.get("preview_model") or payload.get("preview_model") or
                  (FAST_MODEL_PREV if mode=="fast" else ACC_MODEL_PREV)).strip()
    is_custom = model_name not in (FAST_MODEL_PREV, ACC_MODEL_PREV)
    if model_name == FAST_MODEL_PREV:
        preview_model = MODEL_FAST_PREVIEW
    elif model_name == ACC_MODEL_PREV:
        preview_model = MODEL_ACC_PREVIEW
    else:
        preview_model = _get_whisper_model(model_name, ASR_DEVICE)

    preview_opts = FAST_PREVIEW_OPTS if mode == "fast" else ACCURATE_PREVIEW_OPTS

    sess = _create_session(prompt=prompt,
                           mode=mode,
                           preview_model=preview_model,
                           preview_opts=preview_opts,
                           win_s=win_s,
                           step_s=step_s,
                           preview_model_name=model_name,
                           preview_is_custom=is_custom)

    return _ok({
        "id": sess.id,
        "ts": int(time.time()*1000),
        "used_prompt": prompt or None,
        "mode": mode,
        "preview_model": model_name,
        "preview_window_s": win_s,
        "preview_step_s": step_s
    })

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
        if len(data) > ASR_MAX_BODY_BYTES:
            return _err("payload too large", 413)
        if fmt == "pcm16":
            secs = sess.append_pcm16(data, sr)
        else:
            secs = sess.append_any(data, fmt, sr)
        total = len(sess.buf) / 16000.0
        return _ok({"received": len(data), "seconds": float(secs), "total_seconds": float(total)})
    except TimeoutError as e:
        return _err(str(e), 429, {"Retry-After":"2"})
    except Exception as e:
        return _err(f"{type(e).__name__}: {e}", 500)

@app.post("/recognize/stream/<sid>/end")
def stream_end(sid: str):
    sess = _get_session_or_404(sid)
    if not sess:
        return _err("unknown session", 404)
    try:
        final = sess.finalize()
        sess.close()
        with _SESS_LOCK:
            _SESSIONS.pop(sid, None)
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
        # close session after streaming final or TTL idle
        try:
            sess.close()
        finally:
            with _SESS_LOCK:
                _SESSIONS.pop(sid, None)

    return Response(_gen(), mimetype="text/event-stream")

# ──────────────────────────────────────────────────────────────
# 9) Main (production WSGI default)
# ──────────────────────────────────────────────────────────────
def create_app():
    return app

if __name__ == "__main__":
    dev = _device_or_auto(ASR_DEVICE)
    print(f"[ASR] device={dev}")
    print(f"[ASR] models: full={_model_name(MODEL_FULL, MODEL_FULL_NAME)}  "
          f"fast_prev={_model_name(MODEL_FAST_PREVIEW, FAST_MODEL_PREV)}  "
          f"acc_prev={_model_name(MODEL_ACC_PREVIEW, ACC_MODEL_PREV)}")
    print(f"[ASR] modes: default={MODE_DEFAULT}  fast(win={FAST_WIN_S}s,step={FAST_STEP_S}s)  "
          f"accurate(win={ACC_WIN_S}s,step={ACC_STEP_S}s)")
    print(f"[ASR] http://{ASR_HOST}:{ASR_PORT}")
    try:
        from waitress import serve
        serve(app, host=ASR_HOST, port=ASR_PORT, threads=max(8, ASR_MAX_TRANSCRIPTS*4))
    except Exception as e:
        print(f"[ASR] waitress failed: {e}; falling back to Flask dev server", file=sys.stderr)
        from werkzeug.serving import run_simple
        run_simple(ASR_HOST, ASR_PORT, app, threaded=True)
