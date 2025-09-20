# 👂 ASR Suite + NKN Relay

End-to-end Automatic Speech Recognition (ASR) stack:

- **`asr/asr_service.py`** — a self-bootstrapping Flask microservice that exposes a clean REST API for file transcription and live streaming with Server-Sent Events (SSE).
- **`relay/asr_relay.py`** — an NKN sidecar that bridges browser DMs to HTTP, returning single responses or streaming bodies as ordered chunk events.
- **`site/index.html`** — a minimal, production-style test UI with dual **HTTP** and **NKN** transports.
- **`site/server.py`** — a tiny static file server for the frontend.

> Designed to mirror the TTS service’s ergonomics: simple endpoints, auth headers if you add them at the edge, robust chunked streaming, and resilient NKN connectivity.

---

## 📁 Folder Layout

```

asr/
asr\_service.py         # Flask ASR microservice (Whisper)
.env                   # created on first run (config)
relay/
asr\_relay.py          # NKN DM relay bridging to ASR HTTP
site/
index.html            # ASR test UI (dual HTTP/NKN)
server.py             # static server for the frontend

````

---

## ⚙️ Prerequisites

- **Python 3.10+** (the ASR service self-creates a `.venv/` and installs deps)
- **FFmpeg** recommended (for non-PCM inputs); `libsndfile` via `soundfile` is attempted first
- (Optional) **CUDA** if you want GPU acceleration (`ASR_DEVICE=cuda` or `auto`)

---

## 🚀 Quick Start

### 1) Start the ASR Service (port **8126** by default)

```bash
cd asr
python3 asr_service.py
````

* First run: creates `.venv/`, installs deps, writes `.env`, preloads models.

### 2) (Optional) Start the NKN Relay

```bash
cd relay
python3 asr_relay.py
```

* Prints your **NKN address** (include the identifier if shown).
* Relay must be configured with the ASR service base URL (see **Relay Config**).

### 3) Serve the Frontend

```bash
cd site
python3 server.py
# visit http://localhost:8080
```

* In the UI set:

  * **Transport:** HTTP (direct to `http://localhost:8126`) or NKN (paste relay’s NKN addr).
  * **Base URL:** how the relay sees the ASR service (e.g., `http://127.0.0.1:8126` on the relay host).

---

## 🔧 Configuration

### ASR Service (`asr/.env`)

Created on first boot; edit and restart:

```
ASR_HOST=0.0.0.0          # bind host
ASR_PORT=8126             # bind port
ASR_DEVICE=auto           # auto|cpu|cuda (auto => cuda if available)
ASR_MODEL_FULL=medium     # model for final transcripts
ASR_MODEL_PREVIEW=base    # model for live partials
ASR_PREVIEW_WINDOW_S=6.0  # tail window for quick live decodes
ASR_PREVIEW_STEP_S=0.9    # cadence for partial updates
ASR_SESSION_TTL_S=900     # idle session GC
CORS_ORIGINS=*            # CORS allowlist or "*"
```

> **Models:** Supported names include `tiny`, `base`, `small`, `medium`, `large`, `large-v2`, `large-v3`.

### NKN Relay (`relay/.env`)

If your relay script creates/uses `.env`, typical knobs look like:

```
# Upstream ASR base as seen from the relay host
UPSTREAM_BASE=http://127.0.0.1:8126

# NKN identity
NKN_IDENTIFIER=asr-relay
# NKN_SEED=...        # optional; if omitted a transient key is used
NKN_SUBCLIENTS=4
RELAY_MAX_HOLDING_SECONDS=120

# Network, logging
LOG_LEVEL=info
```

> The **frontend** sends DMs with `{event:"http.request", id, req:{...}}`; the relay makes the HTTP call to `UPSTREAM_BASE+req.path` and dials back responses as single (`relay.response`) or streaming (`relay.response.begin/chunk/end`) events.

---

## 🛠️ ASR Service API

Base URL: `http://<ASR_HOST>:<ASR_PORT>` (default `http://127.0.0.1:8126`)

### Common headers

The service itself does **not** enforce auth, but accepts pass-through headers; you can put it behind a gateway that injects:

* `Authorization: Bearer <token>` or
* `X-API-Key: <key>`

They are ignored unless you customize the service.

---

### `GET /health`

**Response 200**

```json
{
  "ok": true,
  "status": "ok",
  "whisper": "ready",
  "device": "cpu|cuda",
  "models": {
    "full": "medium",
    "preview": "base"
  }
}
```

---

### `GET /models`

Lists known Whisper presets and currently configured pair.

**Response 200**

```json
{
  "ok": true,
  "models": [{ "name": "tiny" }, { "name": "base" }, ...],
  "current": { "full": "medium", "preview": "base" }
}
```

---

### `POST /recognize` — Transcribe file / bytes

Two input modes:

1. **multipart/form-data** with `file=<audio>` (recommended for browsers)

```bash
curl -X POST http://localhost:8126/recognize \
  -F "file=@/path/to/audio.wav"
```

2. **application/json** with base64 body (use when tunneling via NKN)

```bash
curl -X POST http://localhost:8126/recognize \
  -H "Content-Type: application/json" \
  -d '{
        "body_b64": "<base64-audio>",
        "format": "wav|mp3|flac|pcm16",
        "sample_rate": 16000
      }'
```

> When `format:"pcm16"`, provide **raw little-endian mono** PCM bytes and set `sample_rate`.

**Response 200**

```json
{
  "ok": true,
  "text": "recognized text",
  "segments": [
    { "t0": 0.00, "t1": 1.23, "text": "..." }
  ],
  "language": "en",
  "duration": 12.34,
  "model": "medium",
  "device": "cpu|cuda"
}
```

**Errors**

* `400 {"ok":false,"error":"missing body_b64 (or send multipart form with file=...)"}` — you sent JSON without `body_b64`.
* `500 {"ok":false,"error":"<Type>: <message>"}` — decoder or model error.

---

### Live Streaming Lifecycle (REST + SSE)

#### 1) `POST /recognize/stream/start`

**Response 200**

```json
{ "ok": true, "id": "sess-<...>", "ts": 1712345678901 }
```

#### 2) `POST /recognize/stream/<id>/audio?format=pcm16&sr=16000`

Append **raw PCM16LE mono** bytes to the session.

* **Body:** binary (no JSON); set the **query string**:

  * `format=pcm16` (exact string)
  * `sr=<sample-rate>` (e.g., 16000)

**Response 200**

```json
{
  "ok": true,
  "received": 4096,          // bytes received in this POST
  "seconds": 0.128,          // seconds in this POST after resample
  "total_seconds": 2.560     // cumulative seconds buffered
}
```

> You may POST small chunks (e.g., every 100–200 ms). Compressed formats are accepted too, but then **use `format=<codec>`** and ensure FFmpeg or libsndfile can decode.

#### 3) `GET /recognize/stream/<id>/events` (SSE)

Server-Sent Events with heartbeats and live partials:

* `event: keepalive`

  ```json
  { "ts": 1712345678901 }
  ```

* `event: asr.partial`

  ```json
  {
    "event": "asr.partial",
    "id": "sess-...",
    "text": "(running cumulative text)",
    "words_added": "only the newly appended words",
    "ts": 1712345678
  }
  ```

* `event: asr.final`

  ```json
  {
    "event": "asr.final",
    "id": "sess-...",
    "result": {
      "ok": true,
      "text": "final text",
      "segments": [...],
      "language": "en",
      "duration": 3.21,
      "model": "medium",
      "device": "cpu|cuda"
    },
    "ts": 1712345679
  }
  ```

> Keep the SSE connection open while streaming. You’ll get periodic `keepalive` even when silent.

#### 4) `POST /recognize/stream/<id>/end`

Completes the session and pushes a single `asr.final` on SSE.

**Response 200**

```json
{
  "ok": true,
  "final": {
    "ok": true,
    "text": "final transcript",
    "segments": [...],
    "language": "en",
    "duration": 5.43,
    "model": "medium",
    "device": "cpu|cuda"
  }
}
```

---

## 🔌 NKN Relay Protocol (for `relay/asr_relay.py`)

The relay bridges the browser to the ASR HTTP service via NKN DMs.

### Request (Browser → Relay)

**Single or streaming request envelope**

```json
{
  "event": "http.request",
  "id": "web-<timestamp>-<counter>",
  "req": {
    "url": "http://127.0.0.1:8126/recognize/stream/<id>/events",
    "method": "GET",
    "headers": { "Authorization": "Bearer ...", "X-Relay-Stream": "chunks" }, // optional
    "timeout_ms": 120000,
    "json": { ... }            // OR
    // "body_b64": "<base64-encoded body>"
    // "stream": "chunks"       // when expecting a streaming HTTP body back
  }
}
```

* If `req.stream === "chunks"`, the relay reads the HTTP response body as a stream and emits ordered chunk events below.
* If JSON is provided, the relay sets `Content-Type: application/json` and serializes `req.json`.
* If `body_b64` is provided, the relay sets `Content-Type: application/octet-stream` and sends raw bytes.

### Responses (Relay → Browser)

**Single-response:**

```json
{
  "event": "relay.response",
  "id": "<same id>",
  "ok": true,
  "status": 200,
  "headers": { "content-type":"application/json" },
  "json": { "ok":true, ... }  // when JSON
  // or "body_b64": "<base64 body>" when non-JSON
}
```

**Streaming-response:**

1. Begin:

```json
{
  "event": "relay.response.begin",
  "id": "<same id>",
  "status": 200,
  "headers": { "content-type":"text/event-stream", "transfer-encoding":"chunked" }
}
```

2. Ordered chunks:

```json
{
  "event": "relay.response.chunk",
  "id": "<same id>",
  "seq": 1,
  "b64": "<base64-chunk>"
}
```

3. Keepalive (optional):

```json
{ "event": "relay.response.keepalive", "id": "<same id>" }
```

4. End:

```json
{
  "event": "relay.response.end",
  "id": "<same id>",
  "ok": true,
  "last_seq": 37
}
```

**Ping/Pong (health):**

```json
{ "event": "relay.ping", "ts": 1712345678 }
{ "event": "relay.pong", "ts": 1712345678 }
```

> The frontend reorders `chunk` DMs by `seq`, assembles bodies, and for SSE it parses lines in-place.

---

## 🧪 Frontend (`site/index.html`)

* Single-page test client styled like the TTS UI
* Dual transports:

  * **HTTP:** direct calls with `fetch`
  * **NKN:** DMs using the protocol above
* Features:

  * Health & Model introspection
  * **File transcription:** multipart via HTTP, JSON (base64) via NKN
  * **Live streaming:** microphone capture → PCM16LE mono @ 16 kHz → `/recognize/stream/.../audio`
  * SSE subscription for partials (`asr.partial`) and finals (`asr.final`)
  * Resilient NKN: reconnect + watchdog + ping/pong

### Hosting (`site/server.py`)

```python
#!/usr/bin/env python3
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
import os

# Serve files from the 'site' directory
os.chdir(os.path.dirname(__file__))

addr = ('0.0.0.0', 8080)
print(f"Serving ./site on http://{addr[0]}:{addr[1]}")
ThreadingHTTPServer(addr, SimpleHTTPRequestHandler).serve_forever()
```

Run:

```bash
cd site
python3 server.py
# open http://localhost:8080
```

---

## 🧩 Integration Examples

### A. Transcribe a file (HTTP)

```bash
BASE=http://127.0.0.1:8126
curl -s -X POST "$BASE/recognize" -F "file=@/tmp/audio.wav"
```

### B. Transcribe bytes (NKN → Relay)

Browser DM payload:

```json
{
  "event": "http.request",
  "id": "web-...",
  "req": {
    "url": "http://127.0.0.1:8126/recognize",
    "method": "POST",
    "headers": { "Content-Type": "application/json" },
    "timeout_ms": 600000,
    "json": {
      "body_b64": "<base64-audio>",
      "format": "pcm16",
      "sample_rate": 16000
    }
  }
}
```

Expect a single `relay.response` with `json.text`.

### C. Live stream (HTTP)

```bash
# 1) start
SID=$(curl -s -X POST http://127.0.0.1:8126/recognize/stream/start | jq -r '.id')

# 2) stream PCM16LE mono (simulate with /dev/zero)
#    replace with your chunk generator; keep chunks small (100–200ms)
curl -s -X POST \
  "http://127.0.0.1:8126/recognize/stream/$SID/audio?format=pcm16&sr=16000" \
  --data-binary @chunk1.pcm

# 3) listen to SSE in another terminal
curl -N "http://127.0.0.1:8126/recognize/stream/$SID/events"

# 4) end
curl -s -X POST "http://127.0.0.1:8126/recognize/stream/$SID/end"
```

### D. Live stream (NKN)

* DM the relay with `http.request` for `/recognize/stream/start` → get `id`.
* For each chunk: `http.request` POST to `/recognize/stream/<id>/audio?format=pcm16&sr=16000` with `body_b64` set to the raw bytes (or set `X-Relay-Stream` + `stream:"chunks"` for responses that stream).
* Open SSE: `http.request` GET `/recognize/stream/<id>/events` with `stream:"chunks"`, parse `relay.response.*` events and then parse SSE lines.

---

## 🧭 Troubleshooting

* **400 `missing body_b64` on `/recognize`**
  You sent JSON without `body_b64`. Use **multipart** `file=` (HTTP) or include `body_b64` when using JSON (e.g., via NKN).

* **500s on `/recognize/stream/<id>/audio`**
  Use **`format=pcm16`** (not `pcm16le`) and post **raw** PCM16LE mono bytes. Provide a correct `sr=`.

* **No live partials over SSE**
  Most often you aren’t actually appending audio (see 500s above) or the SSE handler is looking for the wrong event names. The service emits `asr.partial` and `asr.final`.

* **Long compressed files fail**
  Install FFmpeg. The service first tries `soundfile` (libsndfile), then FFmpeg fallback.

* **GPU not used**
  Set `ASR_DEVICE=cuda` (or `auto`) and ensure `torch` CUDA build is available. If CUDA is missing, service silently falls back to CPU.

* **NKN keeps reconnecting**
  Ensure your relay can reach the ASR base URL **from its vantage point** (often `http://127.0.0.1:8126` if co-located). The frontend also sends `relay.ping/pong`; if pong isn’t received, it will recycle the client.

---

## 📐 Data Contracts

### Transcription Object

```json
{
  "ok": true,
  "text": "final or running text",
  "segments": [
    { "t0": 0.00, "t1": 1.23, "text": "..." }
  ],
  "language": "en",
  "duration": 12.34,
  "model": "medium",
  "device": "cpu|cuda"
}
```

### SSE Event Payloads

* **keepalive**: `{ "ts": <ms epoch> }`
* **asr.partial**: `{ "event":"asr.partial", "id":"...", "text":"...", "words_added":"...", "ts":<ms> }`
* **asr.final**: `{ "event":"asr.final", "id":"...", "result": <Transcription Object>, "ts":<ms> }`

---

## 🔒 Security Notes

* The ASR service includes CORS and accepts pass-through auth headers, but doesn’t validate them by default.
* For production, put it behind an API gateway / reverse proxy that:

  * terminates TLS,
  * enforces auth (JWT/API key),
  * optionally rate-limits and sets consistent `Authorization`/`X-API-Key` headers.

---

## 🧱 Implementation Notes

* **Whisper model pool:** Process-global, ref-counted; device normalized (`auto`→`cuda` if available else `cpu`).
* **Decoding path:** `soundfile` → FFmpeg fallback → PCM16 fast path.
* **Streaming partials:** Preview model runs on a rolling tail window; only **new suffix** relative to the last preview is appended to a running string (prevents duplication).
* **Session GC:** Idle sessions cleaned every 15s; TTL controlled by `ASR_SESSION_TTL_S`.

---

## ✅ Compatibility

* Tested with browser capture at **16 kHz PCM16LE mono**.
* Works with compressed uploads (WAV/FLAC/MP3/OGG) if `soundfile` or FFmpeg available.
* NKN protocol matches the TTS frontend (ordered chunks + `last_seq` end-guard).

---

## 🗓️ Changelog (highlights)

* Initial release:

  * File & streaming endpoints
  * SSE partial & final events
  * NKN relay streaming protocol parity with TTS
  * Frontend parity (theme, resilience, dual transport)
