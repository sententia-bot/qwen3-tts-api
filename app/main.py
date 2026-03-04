import base64
import io
import json
import os
import re
import threading
import gc
import time
import queue
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field
from qwen_tts import Qwen3TTSModel
from transformers import LogitsProcessor, LogitsProcessorList

SUPPORTED_LANGUAGES = [
    "Auto", "English", "Chinese", "Japanese", "Korean",
    "French", "German", "Spanish", "Italian", "Portuguese", "Arabic", "Russian",
]
REFERENCE_AUDIO_DIR = Path("/reference-audio")
MODEL_IDS = {
    "clone": {
        "fast": os.getenv("QWEN_CLONE_MODEL_ID_06", "Qwen/Qwen3-TTS-12Hz-0.6B-Base"),
        "quality": os.getenv("QWEN_CLONE_MODEL_ID_17", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"),
    },
    "design": {
        "quality": os.getenv("QWEN_DESIGN_MODEL_ID_17", "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"),
    },
}
model_status = {"state": "idle", "model_id": None}  # idle | loading | ready


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to synthesize")
    language: str = Field(default="Auto", description="Language or Auto")
    audio_format: str = Field(default="wav", description="wav | ogg")
    mode: str = Field(default="clone", description="clone | design")
    model_size: str = Field(default="quality", description="fast | quality")
    reference_audio: Optional[str] = Field(default=None, description="Filename from /reference-audio when mode=clone")
    voice_description: Optional[str] = Field(default=None, description="Plain-text voice description when mode=design")


class SaveDesignRequest(BaseModel):
    filename: str = Field(..., min_length=1)
    description: Optional[str] = Field(default=None)
    audio_b64: str = Field(..., min_length=1)


class ResetRequest(BaseModel):
    restart: bool = Field(default=False, description="Restart process after reset")


class QwenService:
    def __init__(self) -> None:
        self.device = os.getenv("QWEN_DEVICE", "cuda:0")
        self.dtype = os.getenv("QWEN_DTYPE", "bfloat16")
        self.attn_impl = os.getenv("QWEN_ATTN_IMPL", "flash_attention_2")
        self.lock = threading.Lock()
        self.load_lock = threading.Lock()
        self._model = None
        self._model_id: Optional[str] = None

    def _torch_dtype(self):
        return torch.bfloat16 if self.dtype == "bfloat16" else torch.float16

    def _unload_model_locked(self):
        if self._model is not None:
            del self._model
            self._model = None
            self._model_id = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()

    def get_model(self, model_id: str):
        if self._model is not None and self._model_id == model_id:
            return self._model

        if not self.load_lock.acquire(blocking=False):
            raise HTTPException(status_code=409, detail="Model load already in progress")

        try:
            with self.lock:
                if self._model is not None and self._model_id == model_id:
                    return self._model

                model_status.update({"state": "loading", "model_id": model_id})
                self._unload_model_locked()

                self._model = Qwen3TTSModel.from_pretrained(
                    model_id,
                    device_map=self.device,
                    dtype=self._torch_dtype(),
                    attn_implementation=self.attn_impl,
                )
                self._model_id = model_id
                model_status.update({"state": "ready", "model_id": model_id})
                return self._model
        except Exception:
            model_status.update({"state": "idle", "model_id": None})
            raise
        finally:
            self.load_lock.release()

    def hard_reset(self):
        with self.lock:
            self._unload_model_locked()
            model_status.update({"state": "idle", "model_id": None})

    def resolve_reference_audio(self, filename: str) -> Path:
        candidate = (REFERENCE_AUDIO_DIR / filename).resolve()
        if not str(candidate).startswith(str(REFERENCE_AUDIO_DIR.resolve())):
            raise HTTPException(status_code=400, detail="Invalid filename")
        if not candidate.is_file():
            raise HTTPException(status_code=404, detail=f"Reference audio not found: {filename}")
        return candidate

    def synthesize(self, req: TTSRequest, extra_kwargs: Optional[dict] = None):
        extra_kwargs = extra_kwargs or {}
        mode = (req.mode or "clone").strip().lower()
        if mode not in MODEL_IDS:
            raise HTTPException(status_code=400, detail="mode must be clone or design")

        size = (req.model_size or "quality").strip().lower()
        if size not in {"fast", "quality"}:
            size = "quality"

        if mode == "design" and size == "fast":
            print(json.dumps({"event": "tts_warn", "detail": "design mode does not support fast; using quality"}))
            size = "quality"

        if size not in MODEL_IDS[mode]:
            size = "quality"

        model_id = MODEL_IDS[mode][size]

        if mode == "clone":
            if not req.reference_audio:
                raise HTTPException(status_code=400, detail="reference_audio required for clone mode")
            model = self.get_model(model_id)
            ref_path = self.resolve_reference_audio(req.reference_audio)
            wavs, sr = model.generate_voice_clone(
                text=req.text,
                language=req.language or "Auto",
                ref_audio=str(ref_path),
                x_vector_only_mode=True,
                **extra_kwargs,
            )
        else:
            if not req.voice_description:
                raise HTTPException(status_code=400, detail="voice_description required for design mode")
            model = self.get_model(model_id)
            wavs, sr = model.generate_voice_design(
                text=req.text,
                language=req.language or "Auto",
                instruct=req.voice_description,
                **extra_kwargs,
            )

        return wavs[0], sr, model_id, size


app = FastAPI(title="qwen3-tts-api", version="0.5.0")
svc = QwenService()
start_time = time.time()

# ---------------------------------------------------------------------------
# Text chunking
# ---------------------------------------------------------------------------
MAX_CHUNK_CHARS = int(os.getenv("QWEN_MAX_CHUNK_CHARS", "800"))


def chunk_text(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """Split text into contextually-aware chunks.

    Strategy (in order):
      1. If the whole text fits in max_chars, return as-is (no chunking).
      2. Split on paragraph breaks (double newline).
      3. If any paragraph is still too long, split further on sentence boundaries
         (.  !  ?) while keeping sentences grouped up to max_chars.
    """
    if len(text) <= max_chars:
        return [text]

    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    if not paragraphs:
        paragraphs = [text]

    chunks: list[str] = []
    for para in paragraphs:
        if len(para) <= max_chars:
            chunks.append(para)
        else:
            # Split on sentence boundaries
            sentences = re.split(r"(?<=[.!?])\s+", para)
            current = ""
            for sent in sentences:
                if not current:
                    current = sent
                elif len(current) + 1 + len(sent) <= max_chars:
                    current += " " + sent
                else:
                    chunks.append(current)
                    current = sent
            if current:
                chunks.append(current)

    return chunks if chunks else [text]


def meta_path_for(filename: str) -> Path:
    return REFERENCE_AUDIO_DIR / f"{filename}.meta.json"


def write_meta(filename: str, source: str, description: Optional[str] = None):
    payload = {"source": source}
    if description is not None:
        payload["description"] = description
    meta_path_for(filename).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def read_meta(filename: str):
    path = meta_path_for(filename)
    if not path.is_file():
        return {"source": "upload", "description": None}

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"source": "upload", "description": None}

    source = payload.get("source") if isinstance(payload, dict) else "upload"
    description = payload.get("description") if isinstance(payload, dict) else None
    if source not in {"upload", "design"}:
        source = "upload"
    if description is not None and not isinstance(description, str):
        description = str(description)
    return {"source": source, "description": description}


@app.on_event("startup")
def ensure_reference_audio_dir():
    REFERENCE_AUDIO_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/healthz")
def healthz():
    return {"ok": True, "uptime_s": int(time.time() - start_time)}


@app.get("/status")
def status():
    return {
        "model_state": model_status["state"],
        "model_id": model_status["model_id"],
        "uptime_s": int(time.time() - start_time),
    }


@app.get("/readyz")
def readyz():
    if model_status["state"] == "ready":
        return {"ready": True, "model_id": model_status["model_id"]}
    return JSONResponse(
        status_code=503,
        content={"ready": False, "reason": model_status["state"]},
    )


@app.get("/info")
def info():
    return {
        "models": MODEL_IDS,
        "modes": ["clone", "design"],
        "sizes": ["fast", "quality"],
        "current_model": svc._model_id,
        "supported_languages": SUPPORTED_LANGUAGES,
    }


@app.post("/reset")
def reset(req: Optional[ResetRequest] = None):
    svc.hard_reset()

    response = {"ok": True, "state": "idle", "model_id": None}
    if req and req.restart:
        exit_code = int(os.getenv("QWEN_RESET_EXIT_CODE", "1"))

        def _delayed_exit():
            time.sleep(0.2)
            os._exit(exit_code)

        threading.Thread(target=_delayed_exit, daemon=True).start()
        response["restart_scheduled"] = True
        response["exit_code"] = exit_code

    return response


@app.get("/reference-audio")
def list_reference_audio():
    REFERENCE_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    files = []
    for p in sorted(REFERENCE_AUDIO_DIR.iterdir()):
        if not p.is_file() or p.name.startswith("."):
            continue
        if p.name.endswith(".meta.json"):
            continue
        meta = read_meta(p.name)
        files.append(
            {
                "filename": p.name,
                "source": meta.get("source", "upload"),
                "description": meta.get("description"),
            }
        )

    return {"files": files}


@app.post("/reference-audio/upload")
async def upload_reference_audio(file: UploadFile = File(...)):
    REFERENCE_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    filename = Path(file.filename or "").name
    if not filename:
        raise HTTPException(status_code=400, detail="Missing filename")
    target = (REFERENCE_AUDIO_DIR / filename).resolve()
    if not str(target).startswith(str(REFERENCE_AUDIO_DIR.resolve())):
        raise HTTPException(status_code=400, detail="Invalid filename")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    target.write_bytes(data)
    write_meta(filename, source="upload")
    return {"ok": True, "filename": filename, "size": len(data)}


@app.post("/reference-audio/save-design")
def save_design_reference_audio(req: SaveDesignRequest):
    REFERENCE_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    filename = Path(req.filename or "").name
    if not filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    target = (REFERENCE_AUDIO_DIR / filename).resolve()
    if not str(target).startswith(str(REFERENCE_AUDIO_DIR.resolve())):
        raise HTTPException(status_code=400, detail="Invalid filename")

    audio_b64 = req.audio_b64.strip()
    if "," in audio_b64 and audio_b64.lower().startswith("data:"):
        audio_b64 = audio_b64.split(",", 1)[1]

    try:
        data = base64.b64decode(audio_b64, validate=True)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid audio_b64: {e}")

    if not data:
        raise HTTPException(status_code=400, detail="Decoded audio is empty")

    target.write_bytes(data)
    write_meta(filename, source="design", description=(req.description or "").strip() or None)
    return {"ok": True, "filename": filename}


@app.delete("/reference-audio/{filename}")
def delete_reference_audio(filename: str):
    target = svc.resolve_reference_audio(filename)
    target.unlink()
    sidecar = meta_path_for(target.name)
    if sidecar.exists():
        sidecar.unlink()
    return {"ok": True, "deleted": target.name}


@app.post("/api/tts/stream")
def tts_stream(req: TTSRequest):
    audio_format = req.audio_format.lower().strip()
    if audio_format != "wav":
        raise HTTPException(status_code=400, detail="audio_format must be wav for stream endpoint")

    chunks = chunk_text(req.text)
    n_chunks = len(chunks)

    q: queue.Queue = queue.Queue()

    def run_generation():
        try:
            wav_arrays: list[np.ndarray] = []
            sr_final = 24000
            total_tokens = 0

            for chunk_idx, chunk in enumerate(chunks):
                chunk_estimated = max(1, int((len(chunk) / 5) * 2.5 * 12))
                chunk_token_count = [0]

                q.put(("chunk_start", {
                    "chunk": chunk_idx + 1,
                    "of": n_chunks,
                    "chars": len(chunk),
                }))

                class ProgressProcessor(LogitsProcessor):
                    def __call__(self, input_ids, scores):
                        chunk_token_count[0] += 1
                        # Progress within this chunk (0–95%), scaled to its share of total
                        chunk_pct = min(95, int((chunk_token_count[0] / chunk_estimated) * 100))
                        # Overall progress across all chunks
                        overall_pct = int(((chunk_idx + chunk_pct / 100) / n_chunks) * 100)
                        q.put(("progress", {
                            "chunk": chunk_idx + 1,
                            "of": n_chunks,
                            "chunk_pct": chunk_pct,
                            "overall_pct": min(95, overall_pct),
                            "tokens": chunk_token_count[0],
                        }))
                        return scores

                chunk_req = req.model_copy(update={"text": chunk})
                wav, sr, _used_model_id, _used_size = svc.synthesize(
                    chunk_req,
                    extra_kwargs={
                        "logits_processor": LogitsProcessorList([ProgressProcessor()]),
                    },
                )
                wav_arrays.append(np.asarray(wav, dtype=np.float32))
                sr_final = sr
                total_tokens += chunk_token_count[0]

                q.put(("chunk_done", {
                    "chunk": chunk_idx + 1,
                    "of": n_chunks,
                    "tokens": chunk_token_count[0],
                }))

            data = np.concatenate(wav_arrays) if len(wav_arrays) > 1 else wav_arrays[0]
            buf = io.BytesIO()
            sf.write(buf, data, sr_final, format="WAV")
            audio_b64 = base64.b64encode(buf.getvalue()).decode()
            q.put(("done", {
                "audio_b64": audio_b64,
                "chunks": n_chunks,
                "tokens_generated": total_tokens,
            }))
        except Exception as e:
            q.put(("error", {"detail": str(e)}))

    threading.Thread(target=run_generation, daemon=True).start()

    def event_stream():
        while True:
            event_type, payload = q.get()
            yield f"event: {event_type}\ndata: {json.dumps(payload)}\n\n"
            if event_type in {"done", "error"}:
                break

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/api/tts")
def tts(req: TTSRequest):
    t0 = time.time()
    audio_format = req.audio_format.lower().strip()
    if audio_format not in {"wav", "ogg"}:
        raise HTTPException(status_code=400, detail="audio_format must be wav or ogg")

    chunks = chunk_text(req.text)
    n_chunks = len(chunks)

    t1 = time.time()
    wav_arrays: list[np.ndarray] = []
    sr_final: int = 24000
    used_model_id: str = ""
    used_size: str = ""

    for i, chunk in enumerate(chunks):
        chunk_req = req.model_copy(update={"text": chunk})
        wav, sr, used_model_id, used_size = svc.synthesize(chunk_req)
        wav_arrays.append(np.asarray(wav, dtype=np.float32))
        sr_final = sr
        if n_chunks > 1:
            print(json.dumps({"event": "tts_chunk", "chunk": i + 1, "of": n_chunks, "chars": len(chunk)}))

    data = np.concatenate(wav_arrays) if len(wav_arrays) > 1 else wav_arrays[0]
    t2 = time.time()

    buf = io.BytesIO()
    if audio_format == "wav":
        sf.write(buf, data, sr_final, format="WAV")
        mime = "audio/wav"
    else:
        sf.write(buf, data, sr_final, format="OGG", subtype="VORBIS")
        mime = "audio/ogg"
    t3 = time.time()

    total = t3 - t0
    t_synth = t2 - t1
    t_encode = t3 - t2

    try:
        print(
            json.dumps(
                {
                    "event": "tts_timing",
                    "mode": req.mode,
                    "model_size": used_size,
                    "chars": len(req.text or ""),
                    "chunks": n_chunks,
                    "model_id": used_model_id,
                    "t_total_ms": int(total * 1000),
                    "t_synth_ms": int(t_synth * 1000),
                    "t_encode_ms": int(t_encode * 1000),
                }
            )
        )
    except Exception:
        pass

    return Response(content=buf.getvalue(), media_type=mime)
