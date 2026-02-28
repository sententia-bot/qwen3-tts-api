import io
import os
import time
import threading
from typing import Optional

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field
from qwen_tts import Qwen3TTSModel


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to synthesize")
    language: str = Field(default="Auto", description="Language label or Auto")
    speaker: Optional[str] = Field(default=None, description="Speaker name (for CustomVoice)")
    instruct: Optional[str] = Field(default=None, description="Style instruction")
    model: Optional[str] = Field(default=None, description="Override model id for this request")
    audio_format: str = Field(default="wav", description="wav | ogg")


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

    def synthesize(self, req: TTSRequest):
        model = self.get_or_load_model(req.model)
        language = req.language or self.default_language
        instruct = req.instruct if req.instruct is not None else self.default_instruct

        if "CustomVoice" in (req.model or self.default_model_id):
            speaker = req.speaker or self.default_speaker
            wavs, sr = model.generate_custom_voice(
                text=req.text,
                language=language,
                speaker=speaker,
                instruct=instruct,
            )
        elif "VoiceDesign" in (req.model or self.default_model_id):
            wavs, sr = model.generate_voice_design(
                text=req.text,
                language=language,
                instruct=instruct or "Neutral natural voice",
            )
        else:
            raise HTTPException(status_code=400, detail="Unsupported model type for this API")

        return wavs[0], sr


app = FastAPI(title="qwen3-tts-api", version="0.1.0")
svc = QwenService()
start_time = time.time()


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
