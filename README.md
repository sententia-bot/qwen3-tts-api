# qwen3-tts-api

Production-ready REST wrapper for **Qwen3-TTS** with Docker + Kubernetes manifests.

Designed to mirror the simple Coqui-style flow: **text in → audio bytes out**.

- Default port: `8888` (to avoid Coqui-TTS on `5002`)
- Default model: `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`
- Default response format: `wav`

---

## 1) Qwen3-TTS research summary (1.7B focus)

### Architecture (from official docs)
- Uses **Qwen3-TTS-Tokenizer-12Hz** for speech tokenization.
- End-to-end **discrete multi-codebook LM** architecture (avoids classic cascaded LM+DiT bottlenecks).
- Supports streaming and non-streaming generation.
- Official claims include low first-packet latency in streaming mode.

### Models and packaging
- Python package: `qwen-tts` (PyPI)
- Canonical implementation: `QwenLM/Qwen3-TTS`
- Notable checkpoints:
  - `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`
  - `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`
  - `Qwen/Qwen3-TTS-12Hz-1.7B-Base`

### Inference speed and VRAM expectations (practical)
Official docs emphasize speed + FlashAttention2 support, but do not provide a strict VRAM table for every setup. Practical sizing guidance for the **1.7B** model:

- **Weights (bf16/fp16)** roughly consume a few GB.
- Real runtime usage includes model + KV/cache + activations + framework overhead.
- For stable serving, budget approximately:
  - **~8–12 GB VRAM** for conservative single-request generation settings.
  - More if using large `max_new_tokens`, concurrency, or additional model variants loaded.
- **RTX 3070 Ti (8 GB)** can run the 1.7B model with careful settings (bf16/fp16, low concurrency, moderate generation lengths).

Recommendation for caladan:
- Start with single worker, `bfloat16`, no batching, moderate text length.
- If memory pressure occurs, reduce generation length and/or switch to 0.6B model for headroom.

---

## 2) API design (Coqui-like simplicity)

### Endpoint
`POST /api/tts`

### Request JSON
```json
{
  "text": "Hello from Qwen3-TTS",
  "language": "English",
  "speaker": "Ryan",
  "instruct": "Speak clearly and warmly",
  "audio_format": "wav"
}
```

Fields:
- `text` (required)
- `language` (default `Auto`)
- `speaker` (default from env; used by `CustomVoice` models)
- `instruct` (optional style instruction)
- `model` (optional override model id per request)
- `audio_format` (`wav` or `ogg`, default `wav`)

### Response
- Raw audio bytes with `Content-Type`:
  - `audio/wav` or `audio/ogg`

### Other endpoints
- `GET /healthz` — process health
- `GET /readyz` — model-load readiness
- `GET /speakers` — supported speakers from current model (if available)

---

## 3) Local run (no container)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt

export QWEN_MODEL_ID=Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8888
```

Test:
```bash
curl -s -X POST "http://localhost:8888/api/tts" \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello from caladan","language":"English","speaker":"Ryan","audio_format":"wav"}' \
  --output out.wav
```

---

## 4) Docker build/run

### Build
```bash
docker build -t qwen3-tts-api:latest .
```

### Run (GPU)
```bash
docker run --rm --gpus all -p 8888:8888 \
  -e QWEN_MODEL_ID=Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  qwen3-tts-api:latest
```

Notes:
- Dockerfile defaults to CUDA runtime base image and installs PyTorch CUDA wheels.
- Requires NVIDIA Container Toolkit on host.
- For alternate arch/base image, set `--build-arg BASE_IMAGE=...`.

---

## 5) Kubernetes

Manifests in `k8s/`:
- `configmap.yaml`
- `deployment.yaml`
- `service.yaml`

Apply:
```bash
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

The Deployment requests one GPU (`nvidia.com/gpu: 1`) and exposes container port `8888`.

---

## 6) Environment variables

- `QWEN_MODEL_ID` (default `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`)
- `QWEN_DEVICE` (default `cuda:0`)
- `QWEN_DTYPE` (`bfloat16` or `float16`)
- `QWEN_ATTN_IMPL` (default `flash_attention_2`)
- `QWEN_DEFAULT_LANGUAGE` (default `Auto`)
- `QWEN_DEFAULT_SPEAKER` (default `Ryan`)
- `QWEN_DEFAULT_INSTRUCT` (default empty)

---

## 7) Compatibility and caveats

- FlashAttention2 support depends on hardware + install success.
- If `flash_attention_2` fails, set `QWEN_ATTN_IMPL=eager` and retry.
- `mp3` is intentionally not default in this API implementation to keep dependency and latency overhead lower; recommended output is `wav` (or `ogg` if bandwidth-sensitive).

---

## 8) Source references

- Qwen3-TTS repo: https://github.com/QwenLM/Qwen3-TTS
- HF collection: https://huggingface.co/collections/Qwen/qwen3-tts
- 1.7B custom voice model: https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
