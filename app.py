"""
app.py — modular TTS API for OmniVoice, mirroring the Piper project's
FastAPI wrapper (Section 6.1 of the evaluation report): a single
POST /synthesize endpoint decoupling the TTS engine from the rest of the
avatar pipeline, so Group 2 (LLM) and Group 4 (Avatar) can call it over
HTTP without installing OmniVoice, torch, or any model weights locally.

Run:
    uvicorn app:app --host 0.0.0.0 --port 8000

Endpoints:
    GET  /health              -> {"status": "ok", "device": "...", "voices": [...]}
    POST /synthesize          -> body: {"text": "...", "voice": "fasih"} -> audio/wav
    POST /clone-voice         -> form-data: text, ref_text, lang, save_as (optional), audio_file -> audio/wav
"""

import io
import os
import tempfile

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import soundfile as sf

import config
from normalize import normalize_text

app = FastAPI(title="OmniVoice TTS API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_engine = None  # loaded lazily on first request to keep startup fast


def get_engine():
    global _engine
    if _engine is None:
        from omnivoice_engine import OmniVoiceEngine
        _engine = OmniVoiceEngine()
    return _engine


class SynthesizeRequest(BaseModel):
    text: str
    voice: str = "fasih"   # one of config.VOICE_BANK keys


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": config.DEVICE,
        "voices": list(config.VOICE_BANK.keys()),
    }


@app.post("/synthesize")
def synthesize(req: SynthesizeRequest):
    if req.voice not in config.VOICE_BANK:
        raise HTTPException(400, f"Unknown voice '{req.voice}'. Available: {list(config.VOICE_BANK.keys())}")

    voice = config.VOICE_BANK[req.voice]
    language = "ur" if voice["language_id"] == "urd" else "en"
    normalized = normalize_text(req.text, lang=language)

    engine = get_engine()
    from omnivoice import VoiceClonePrompt

    prompt_path = config.voice_bank_prompt_path(req.voice)
    if os.path.exists(prompt_path):
        prompt = VoiceClonePrompt.load(prompt_path)
        audio = engine.clone_with_prompt(text=normalized, voice_clone_prompt=prompt)
    else:
        audio = engine.clone(text=normalized, ref_audio=voice["ref_audio"], ref_text=voice["ref_text"])

    buf = io.BytesIO()
    sf.write(buf, audio[0], config.SAMPLE_RATE, format="WAV")
    buf.seek(0)
    return StreamingResponse(buf, media_type="audio/wav")


@app.post("/clone-voice")
async def clone_voice(
    text: str = Form(...),
    ref_text: str = Form(...),
    lang: str = Form("en"),          # "en" or "ur"
    save_as: str = Form(None),       # optional — if given, permanently saves to voice_bank/
    audio_file: UploadFile = File(...),
):
    """Clone a NEW voice on the fly from an uploaded reference clip —
    no need to pre-register it in config.VOICE_BANK first.
    If save_as is provided, the prompt is also saved to voice_bank/<save_as>.pt
    so it can be reused later without re-uploading the audio."""
    suffix = os.path.splitext(audio_file.filename)[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await audio_file.read())
        tmp_path = tmp.name

    try:
        normalized = normalize_text(text, lang=lang)
        engine = get_engine()
        audio = engine.clone(text=normalized, ref_audio=tmp_path, ref_text=ref_text)

        if save_as:
            prompt = engine.create_prompt(ref_audio=tmp_path, ref_text=ref_text)
            prompt.save(config.voice_bank_prompt_path(save_as))
    finally:
        os.remove(tmp_path)

    buf = io.BytesIO()
    sf.write(buf, audio[0], config.SAMPLE_RATE, format="WAV")
    buf.seek(0)
    return StreamingResponse(buf, media_type="audio/wav")