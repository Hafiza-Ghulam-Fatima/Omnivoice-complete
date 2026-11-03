"""
config.py — single source of truth for paths, device, and voice-bank settings.

Every other script in this project imports from here so you only change
settings in ONE place (mirrors how the Piper project used a single voices
list / API_INTEGRATION.md as the source of truth).
"""

import os
import torch

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
MODEL_ID = "k2-fsa/OmniVoice"          # HuggingFace repo id, auto-downloaded on first run
SAMPLE_RATE = 24000                     # OmniVoice always outputs 24kHz audio

# ---------------------------------------------------------------------------
# Device auto-detection (CPU-only laptop, CUDA box, or Apple Silicon all work)
# ---------------------------------------------------------------------------
def get_device():
    if torch.cuda.is_available():
        return "cuda:0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

DEVICE = os.environ.get("OMNIVOICE_DEVICE", get_device())

# dtype: fp16 only makes sense on GPU/MPS; CPU must stay fp32
DTYPE = torch.float16 if DEVICE.startswith("cuda") else torch.float32

# Diffusion steps: 32 = default/higher quality, 16 = faster (roughly 2x),
# useful fallback if CPU latency in check_latency.py comes back too high.
NUM_STEP_DEFAULT = 32
NUM_STEP_FAST = 16

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
REFERENCE_AUDIO_DIR = os.path.join(BASE_DIR, "reference_audio")
VOICE_BANK_DIR = os.path.join(BASE_DIR, "voice_bank")

for d in (OUTPUT_DIR, REFERENCE_AUDIO_DIR, VOICE_BANK_DIR,
          os.path.join(OUTPUT_DIR, "latency"),
          os.path.join(OUTPUT_DIR, "edge_cases"),
          os.path.join(OUTPUT_DIR, "voice_modes"),
          os.path.join(OUTPUT_DIR, "dummy_llm")):
    os.makedirs(d, exist_ok=True)

# ---------------------------------------------------------------------------
# Voice bank — the OmniVoice equivalent of Piper's "fasih / amy / ryan" voices.
# OmniVoice has no built-in preset voices: every voice is defined by a
# reference clip + its transcript. Reuse the SAME three reference clips your
# Piper project already produced (they already sound clean and are already
# reviewed by ear) so the two engines stay comparable on identical inputs.
#
# ACTION REQUIRED: copy these three files from your Piper repo into
# reference_audio/ before running anything:
#   fasih_ref.wav   <- e.g. from english_test_* style Urdu output
#   amy_ref.wav
#   ryan_ref.wav
# Each entry below needs a short, accurate transcript of that exact clip —
# OmniVoice uses this to align cloning, so an inaccurate transcript degrades
# quality more than skipping ref_text entirely.
# ---------------------------------------------------------------------------
VOICE_BANK = {
    "fasih": {
        "ref_audio": os.path.join(REFERENCE_AUDIO_DIR, "fasih_ref.wav"),
        "ref_text": "آج موسم بہت اچھا ہے میں باہر جانا چاہتا ہوں",
        "language_id": "urd",
        "description": "Urdu — primary avatar voice",
    },
    "amy": {
        "ref_audio": os.path.join(REFERENCE_AUDIO_DIR, "amy_ref.wav"),
        "ref_text": "Hello this is my AI avatar assistant, I hope this voice sounds natural and clear.",
        "language_id": "en",
        "description": "English (US) — clear, neutral",
    },
    "ryan": {
        "ref_audio": os.path.join(REFERENCE_AUDIO_DIR, "ryan_ref.wav"),
        "ref_text": "Hello this is my AI avatar assistant, I hope this voice sounds natural and clear.",
        "language_id": "en",
        "description": "English (US) — alternate tone",
    },
}

# Saved VoiceClonePrompt files (see test_voice_modes.py) live here so later
# runs skip re-encoding the reference audio entirely.
def voice_bank_prompt_path(name: str) -> str:
    return os.path.join(VOICE_BANK_DIR, f"{name}.pt")
