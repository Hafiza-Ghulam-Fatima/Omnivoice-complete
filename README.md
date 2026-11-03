# OmniVoice — Setup & Testing Guide
**AI Avatar Assistant — TTS component (OmniVoice evaluation branch)**

This project mirrors the structure of the Piper TTS setup already used in
the assistant pipeline (fasih / amy / ryan, `normalize.py`, latency checks,
edge-case tests, `dummy_llm_client.py`) so results are directly comparable,
but organized as a real project instead of loose scripts.

> Companion doc: **`SPECIFICATION.md`** — full model spec, comparison
> against Piper, and the shortcomings/mitigations table.

---

## 1. What is OmniVoice

OmniVoice (`k2-fsa/OmniVoice`) is a massively multilingual, zero-shot TTS
model (600+ languages, including Urdu) built on a diffusion-language-model
architecture. Unlike Piper, it has no fixed preset voices — every voice is
produced either by:
- **Voice cloning** — a 3–10s reference audio clip + its transcript, or
- **Voice design** — a text description of the voice ("female, low pitch,
  british accent") — English/Chinese only, see `SPECIFICATION.md`, or
- **Auto voice** — no prompt at all, model picks a voice.

Apache-2.0 licensed. Weights are ~2.45GB (vs. 15–30MB per Piper voice).

---

## 2. System Requirements

| | Minimum | Notes |
|---|---|---|
| Python | 3.10+ | |
| Disk | ~6GB free | model weights + venv + audio outputs |
| RAM | 8GB+ (CPU mode) | more is safer; diffusion models are memory-hungry |
| GPU | Optional | Vendor RTF numbers (0.025, ~40x real-time) are **GPU-benchmarked only** |
| OS | Windows / Linux / macOS | Apple Silicon supported via `mps` |

CPU-only laptops (like the one used for the Piper testing) are supported
but **untested by the vendor** — that's exactly what `check_latency.py`
in this project exists to verify on your actual hardware before you
depend on it.

---

## 3. Installation

```bash
# 1. Create and activate a fresh virtual environment
python -m venv omnivoice-env
# Windows:
omnivoice-env\Scripts\activate
# macOS/Linux:
source omnivoice-env/bin/activate

# 2. Install PyTorch FIRST, matching your hardware (see requirements.txt
#    for exact commands per platform — CPU / CUDA / Apple Silicon)
pip install torch==2.8.0 torchaudio==2.8.0        # CPU-only example

# 3. Install this project's dependencies
pip install -r requirements.txt

# 4. (Optional) enable OmniVoice's built-in number normalizer for EN/ZH
pip install "omnivoice[tn]"
# macOS Apple Silicon note: pynini has no wheel here — install first with:
#   conda install -c conda-forge pynini
```

If HuggingFace is slow or blocked from your network, set a mirror before
running anything that downloads the model:
```bash
export HF_ENDPOINT="https://hf-mirror.com"
```

The model weights download automatically on first use (`from_pretrained`) —
there is no separate manual download step.

---

## 4. Project Structure

```
omnivoice_project/
├── config.py              # single source of truth: device, paths, voice bank
├── normalize.py           # number / phone number / symbol normalization
├── omnivoice_engine.py    # wrapper: load model once, clone/design/auto modes
├── check_latency.py       # RTF benchmark -> outputs/latency/
├── edge_case_test.py      # pronunciation edge cases -> outputs/edge_cases/
├── test_voice_modes.py    # builds the voice bank + tests all 3 modes -> outputs/voice_modes/
├── dummy_llm_client.py    # simulated LLM text -> TTS -> outputs/dummy_llm/
├── app.py                 # FastAPI wrapper (POST /synthesize, GET /health)
├── run_all_tests.py       # runs everything above in the right order
├── requirements.txt
├── reference_audio/       # <- put your 3 reference clips here (see below)
├── voice_bank/            # <- saved VoiceClonePrompt .pt files (auto-generated)
└── outputs/                # <- ALL test outputs land here, organized by test type
    ├── latency/
    ├── edge_cases/
    ├── voice_modes/
    └── dummy_llm/
```

---

## 5. Building the Voice Bank (do this first)

OmniVoice has no preset voices, so before running anything else:

1. Copy three short (3–10s), clean reference clips into `reference_audio/`:
   - `fasih_ref.wav` — reuse the existing Piper Fasih (Urdu) sample if it's
     clean enough, or record a fresh short Urdu clip
   - `amy_ref.wav` — same idea for the English "Amy" voice
   - `ryan_ref.wav` — same idea for the English "Ryan" voice
2. Open `config.py` and fill in `ref_text` for each entry with the **exact**
   transcript of that clip — accuracy here matters more than skipping it.
3. Run:
   ```bash
   python test_voice_modes.py
   ```
   This clones each voice, saves a reusable `voice_bank/<name>.pt` prompt
   file, and re-loads each one to prove it works standalone. Every other
   script in this project automatically prefers these saved prompts once
   they exist, so reference audio only needs to be supplied once.

---

## 6. Running the Test Suite

Run everything in order:
```bash
python run_all_tests.py
```

Or run pieces individually:

```bash
# 1. Latency / real-time factor on THIS machine
python check_latency.py
python check_latency.py --fast          # num_step=16 fallback if CPU is too slow

# 2. Voice bank + all 3 generation modes (cloning / design / auto)
python test_voice_modes.py

# 3. Pronunciation edge cases (numbers, phone numbers, symbols, code-switch...)
python edge_case_test.py --voice fasih --lang ur
python edge_case_test.py --voice amy   --lang en

# 4. Simulated LLM input -> normalized text -> audio out
python dummy_llm_client.py
```

Every script prints its RTF/metrics to the console AND writes a JSON
report next to its `.wav` outputs, so results can be reviewed by ear and
by number.

---

## 7. Serving It to the Rest of the Avatar Pipeline

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

```bash
curl -X POST http://127.0.0.1:8000/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello there", "voice": "amy"}' \
  --output test.wav
```

Group 2 (LLM) and Group 4 (Avatar) can integrate against this endpoint
exactly like they would against the Piper API — same request/response
shape, different engine underneath.

```bash
# Prove the whole pipeline works over HTTP, not just in-process:
python dummy_llm_client.py --via-api
```

---

## 8. Comparing Against Piper

Both engines' test scripts write standard `.wav` + JSON reports, so you
can drop OmniVoice's `outputs/latency/latency_report.json` and
`outputs/edge_cases/*.json` next to the Piper project's equivalents and
compare RTF and pronunciation quality directly, sentence-for-sentence.

See `SPECIFICATION.md` for the full comparison table and the shortcomings
this setup deliberately works around.
