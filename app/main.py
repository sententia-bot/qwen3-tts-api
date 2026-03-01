import base64
import io
import json
import os
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field
from qwen_tts import Qwen3TTSModel

SUPPORTED_LANGUAGES = [
    "Auto", "English", "Chinese", "Japanese", "Korean",
    "French", "German", "Spanish", "Italian", "Portuguese", "Arabic", "Russian",
]
REFERENCE_AUDIO_DIR = Path("/reference-audio")
CLONE_MODEL_ID = os.getenv("QWEN_CLONE_MODEL_ID", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
DESIGN_MODEL_ID = os.getenv("QWEN_DESIGN_MODEL_ID", "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
model_status = {"state": "idle", "model_id": None}  # idle | loading | ready


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to synthesize")
    language: str = Field(default="Auto", description="Language or Auto")
    audio_format: str = Field(default="wav", description="wav | ogg")
    mode: str = Field(default="clone", description="clone | design")
    reference_audio: Optional[str] = Field(default=None, description="Filename from /reference-audio when mode=clone")
    voice_description: Optional[str] = Field(default=None, description="Plain-text voice description when mode=design")


class SaveDesignRequest(BaseModel):
    filename: str = Field(..., min_length=1)
    description: Optional[str] = Field(default=None)
    audio_b64: str = Field(..., min_length=1)


class QwenService:
    def __init__(self) -> None:
        self.device = os.getenv("QWEN_DEVICE", "cuda:0")
        self.dtype = os.getenv("QWEN_DTYPE", "bfloat16")
        self.attn_impl = os.getenv("QWEN_ATTN_IMPL", "flash_attention_2")
        self.lock = threading.Lock()
        self._model = None
        self._model_id: Optional[str] = None

    def _torch_dtype(self):
        return torch.bfloat16 if self.dtype == "bfloat16" else torch.float16

    def get_model(self, model_id: str):
        global model_status
        with self.lock:
            if self._model is not None and self._model_id == model_id:
                return self._model

            model_status = {"state": "loading", "model_id": model_id}

            if self._model is not None:
                del self._model
                self._model = None
                self._model_id = None
                torch.cuda.empty_cache()

            self._model = Qwen3TTSModel.from_pretrained(
                model_id,
                device_map=self.device,
                dtype=self._torch_dtype(),
                attn_implementation=self.attn_impl,
            )
            self._model_id = model_id
            model_status = {"state": "ready", "model_id": model_id}
            return self._model

    def resolve_reference_audio(self, filename: str) -> Path:
        candidate = (REFERENCE_AUDIO_DIR / filename).resolve()
        if not str(candidate).startswith(str(REFERENCE_AUDIO_DIR.resolve())):
            raise HTTPException(status_code=400, detail="Invalid filename")
        if not candidate.is_file():
            raise HTTPException(status_code=404, detail=f"Reference audio not found: {filename}")
        return candidate

    def synthesize(self, req: TTSRequest):
        mode = (req.mode or "clone").strip().lower()

        if mode == "clone":
            if not req.reference_audio:
                raise HTTPException(status_code=400, detail="reference_audio required for clone mode")
            model = self.get_model(CLONE_MODEL_ID)
            ref_path = self.resolve_reference_audio(req.reference_audio)
            wavs, sr = model.generate_voice_clone(
                text=req.text,
                language=req.language or "Auto",
                ref_audio=str(ref_path),
                x_vector_only_mode=True,
            )
        elif mode == "design":
            if not req.voice_description:
                raise HTTPException(status_code=400, detail="voice_description required for design mode")
            model = self.get_model(DESIGN_MODEL_ID)
            wavs, sr = model.generate_voice_design(
                text=req.text,
                language=req.language or "Auto",
                instruct=req.voice_description,
            )
        else:
            raise HTTPException(status_code=400, detail="mode must be clone or design")

        return wavs[0], sr


app = FastAPI(title="qwen3-tts-api", version="0.5.0")
svc = QwenService()
start_time = time.time()


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
    try:
        svc.get_model(CLONE_MODEL_ID)
        return {"ready": True}
    except Exception as e:
        return JSONResponse(status_code=503, content={"ready": False, "error": str(e)})


@app.get("/info")
def info():
    return {
        "models": {
            "clone": CLONE_MODEL_ID,
            "design": DESIGN_MODEL_ID,
        },
        "modes": ["clone", "design"],
        "current_model": svc._model_id,
        "supported_languages": SUPPORTED_LANGUAGES,
    }


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


@app.post("/api/tts")
def tts(req: TTSRequest):
    audio_format = req.audio_format.lower().strip()
    if audio_format not in {"wav", "ogg"}:
        raise HTTPException(status_code=400, detail="audio_format must be wav or ogg")

    wav, sr = svc.synthesize(req)
    data = np.asarray(wav, dtype=np.float32)
    buf = io.BytesIO()
    if audio_format == "wav":
        sf.write(buf, data, sr, format="WAV")
        mime = "audio/wav"
    else:
        sf.write(buf, data, sr, format="OGG", subtype="VORBIS")
        mime = "audio/ogg"

    return Response(content=buf.getvalue(), media_type=mime)
