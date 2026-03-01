import io
import os
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field
from qwen_tts import Qwen3TTSModel

SUPPORTED_LANGUAGES = [
    "Auto",
    "English",
    "Chinese",
    "Japanese",
    "Korean",
    "French",
    "German",
    "Spanish",
    "Italian",
    "Portuguese",
    "Arabic",
    "Russian",
]
REFERENCE_AUDIO_DIR = Path("/reference-audio")


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to synthesize")
    language: str = Field(default="Auto", description="Language label or Auto")
    speaker: Optional[str] = Field(default=None, description="Speaker name (for CustomVoice)")
    instruct: Optional[str] = Field(default=None, description="Style instruction")
    model: Optional[str] = Field(default=None, description="Override model id for this request")
    audio_format: str = Field(default="wav", description="wav | ogg")
    reference_audio: Optional[str] = Field(
        default=None,
        description="Filename from /reference-audio used as clone reference",
    )


class QwenService:
    def __init__(self) -> None:
        self.default_model_id = os.getenv("QWEN_MODEL_ID", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
        self.device = os.getenv("QWEN_DEVICE", "cuda:0")
        self.dtype = os.getenv("QWEN_DTYPE", "bfloat16")
        self.attn_impl = os.getenv("QWEN_ATTN_IMPL", "flash_attention_2")
        self.default_language = os.getenv("QWEN_DEFAULT_LANGUAGE", "Auto")
        self.default_speaker = os.getenv("QWEN_DEFAULT_SPEAKER", "Ryan")
        self.default_instruct = os.getenv("QWEN_DEFAULT_INSTRUCT", "")
        self.lock = threading.Lock()
        self.models = {}

    def _torch_dtype(self):
        if self.dtype == "float16":
            return torch.float16
        if self.dtype == "bfloat16":
            return torch.bfloat16
        return torch.float16

    def get_or_load_model(self, model_id: Optional[str] = None):
        selected = model_id or self.default_model_id
        with self.lock:
            if selected in self.models:
                return self.models[selected]
            model = Qwen3TTSModel.from_pretrained(
                selected,
                device_map=self.device,
                dtype=self._torch_dtype(),
                attn_implementation=self.attn_impl,
            )
            self.models[selected] = model
            return model

    def resolve_reference_audio(self, filename: str) -> Path:
        candidate = (REFERENCE_AUDIO_DIR / filename).resolve()
        base = REFERENCE_AUDIO_DIR.resolve()
        if not str(candidate).startswith(str(base)):
            raise HTTPException(status_code=400, detail="Invalid reference_audio filename")
        if not candidate.is_file():
            raise HTTPException(status_code=404, detail=f"reference_audio not found: {filename}")
        return candidate

    def synthesize(self, req: TTSRequest):
        language = req.language or self.default_language
        instruct = req.instruct if req.instruct is not None else self.default_instruct

        if req.reference_audio:
            # Voice cloning requires the Base model — auto-select it regardless of default
            clone_model_id = req.model if (req.model and "Base" in req.model) else os.getenv(
                "QWEN_CLONE_MODEL_ID", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
            )
            model = self.get_or_load_model(clone_model_id)
            reference_path = self.resolve_reference_audio(req.reference_audio)

            wavs, sr = model.generate_voice_clone(
                text=req.text,
                language=language,
                ref_audio=str(reference_path),
                x_vector_only_mode=True,
            )
            return wavs[0], sr

        model = self.get_or_load_model(req.model)

        effective_model = req.model or self.default_model_id
        if "CustomVoice" in effective_model:
            speaker = req.speaker or self.default_speaker
            wavs, sr = model.generate_custom_voice(
                text=req.text,
                language=language,
                speaker=speaker,
                instruct=instruct,
            )
        elif "VoiceDesign" in effective_model:
            wavs, sr = model.generate_voice_design(
                text=req.text,
                language=language,
                instruct=instruct or "Neutral natural voice",
            )
        else:
            raise HTTPException(status_code=400, detail="Unsupported model type for this API")

        return wavs[0], sr


app = FastAPI(title="qwen3-tts-api", version="0.2.0")
svc = QwenService()
start_time = time.time()


@app.on_event("startup")
def ensure_reference_audio_dir():
    REFERENCE_AUDIO_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/healthz")
def healthz():
    return {"ok": True, "uptime_s": int(time.time() - start_time)}


@app.get("/readyz")
def readyz():
    try:
        svc.get_or_load_model(None)
        return {"ready": True}
    except Exception as e:  # noqa: BLE001
        return JSONResponse(status_code=503, content={"ready": False, "error": str(e)})


@app.get("/speakers")
def speakers(model: Optional[str] = Query(default=None)):
    m = svc.get_or_load_model(model)
    if hasattr(m, "get_supported_speakers"):
        return {"speakers": m.get_supported_speakers()}
    return {"speakers": []}


@app.get("/info")
def info(model: Optional[str] = Query(default=None)):
    selected = model or svc.default_model_id
    m = svc.get_or_load_model(model)
    if hasattr(m, "get_supported_speakers"):
        supported_speakers = m.get_supported_speakers()
    else:
        supported_speakers = []

    return {
        "default_model": selected,
        "default_speaker": svc.default_speaker,
        "default_language": svc.default_language,
        "supported_languages": SUPPORTED_LANGUAGES,
        "speakers": supported_speakers,
    }


@app.get("/reference-audio")
def list_reference_audio():
    ensure_reference_audio_dir()
    files = [
        p.name
        for p in sorted(REFERENCE_AUDIO_DIR.iterdir())
        if p.is_file() and not p.name.startswith(".")
    ]
    return {"files": files}


@app.post("/reference-audio/upload")
async def upload_reference_audio(file: UploadFile = File(...)):
    ensure_reference_audio_dir()
    filename = Path(file.filename or "").name
    if not filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    target = (REFERENCE_AUDIO_DIR / filename).resolve()
    base = REFERENCE_AUDIO_DIR.resolve()
    if not str(target).startswith(str(base)):
        raise HTTPException(status_code=400, detail="Invalid filename")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    target.write_bytes(data)
    return {"ok": True, "filename": filename, "size": len(data)}


@app.delete("/reference-audio/{filename}")
def delete_reference_audio(filename: str):
    target = svc.resolve_reference_audio(filename)
    target.unlink()
    return {"ok": True, "deleted": target.name}


@app.post("/api/tts")
def tts(req: TTSRequest):
    audio_format = req.audio_format.lower().strip()
    if audio_format not in {"wav", "ogg"}:
        raise HTTPException(status_code=400, detail="audio_format must be one of: wav, ogg")

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
