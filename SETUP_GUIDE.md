# OmniVoice Setup & Testing Guide
### Group 3 – TTS component | AI Avatar-Based Assistant Project
### Companion to the Piper TTS setup (this mirrors that project's structure, applied to OmniVoice)

Upstream project: https://github.com/k2-fsa/OmniVoice (k2-fsa, Apache-2.0, arXiv:2604.00688)

---

## 0. Why OmniVoice, and the honest caveat up front

Your evaluation report (Section 3.6 / Section 7) already flagged the trade-off: OmniVoice is the
only alternative besides Piper that supports Urdu **and** it adds zero-shot voice cloning — but its
0.025 RTF ("40x real-time") number is a **GPU** benchmark. This laptop is CPU-only, so that number
cannot be assumed. `check_latency.py` in this project exists specifically to replace that assumption
with a measured number, the same way you did for Piper and NeuTTS Nano.

Treat this project as the "go measure it" step from your report's Section 7 next-steps list, not as
a replacement for Piper yet.

---

## 1. System Requirements

| Item | Piper (existing) | OmniVoice |
|---|---|---|
| Engine size | ~15–30 MB per voice | ~2.45 GB total model weights |
| Runtime | onnxruntime (CPU-only, lightweight) | PyTorch (CPU/CUDA/MPS/XPU) |
| RAM | Low (~200–500 MB) | Several GB (plan for 8 GB+ free RAM on CPU) |
| Recommended hardware | Any CPU | GPU strongly recommended; CPU works but is unbenchmarked for this project until you run Section 4 below |
| Python | 3.9+ | 3.10+ recommended |

If you only have the same CPU-only Windows laptop used for the Piper/NeuTTS tests, expect
noticeably slower generation than the vendor's GPU numbers — that's exactly what you're about
to measure, not guess.

---

## 2. Installation

### Step 1 — Create an isolated environment (same practice as Piper)

```bash
python -m venv omnivoice-env
# Windows
omnivoice-env\Scripts\activate
# macOS / Linux
source omnivoice-env/bin/activate
```

### Step 2 — Install PyTorch (pick ONE, matching your hardware)

```bash
# CPU only (this project's laptop)
pip install torch==2.8.0 torchaudio==2.8.0

# NVIDIA GPU (CUDA 12.8 example — check your CUDA version first)
pip install torch==2.8.0+cu128 torchaudio==2.8.0+cu128 --extra-index-url https://download.pytorch.org/whl/cu128

# Apple Silicon
pip install torch==2.8.0 torchaudio==2.8.0
```

### Step 3 — Install OmniVoice

```bash
pip install omnivoice
# or, latest from source:
pip install git+https://github.com/k2-fsa/OmniVoice.git
```

### Step 4 — Install the rest of this project's dependencies

```bash
pip install -r requirements.txt
```

### Step 5 — (Optional) Number normalization extra

OmniVoice's *built-in* `normalize_text=True` option only fully supports Chinese and English
(via WeTextProcessing); everything else — including Urdu — falls back to `num2words`, which does
not cover Urdu. This is exactly the "digits/phone numbers mispronounced" problem you already solved
once for Piper, and it resurfaces here for the Urdu voice. This project's `normalize.py`
(Section 5 below) is the fix, so this extra is optional/for English only:

```bash
pip install "omnivoice[tn]"
```

### Step 6 — First run / model download

The model weights (~2.45 GB) download automatically from Hugging Face the first time you call
`OmniVoice.from_pretrained("k2-fsa/OmniVoice", ...)`. If your network can't reach Hugging Face
directly, set the mirror first:

```bash
# Windows (PowerShell)
$env:HF_ENDPOINT="https://hf-mirror.com"
# macOS / Linux
export HF_ENDPOINT="https://hf-mirror.com"
```

### Step 7 — Smoke test (no coding, sanity check only)

```bash
omnivoice-demo --ip 0.0.0.0 --port 8001
```

Open `http://localhost:8001` — if the Gradio UI loads and you can generate one sentence, the
install is good and you can move to this project's scripts.

---

## 3. Project Structure

```
omnivoice_project/
├── SETUP_GUIDE.md            # this file
├── requirements.txt
├── normalize.py              # number / symbol / phone-number normalization (EN + UR)
├── dummy_llm_client.py       # simulates Group 2's LLM output -> normalize -> speech
├── check_latency.py          # measures RTF on YOUR CPU, EN + UR sentences
├── edge_case_test.py         # numbers, phone numbers, symbols, code-switching, etc.
├── test_voice_clone.py       # zero-shot voice cloning demo (capability Piper lacks)
├── test_voice_design.py      # attribute-based voice design (male/female/accent/pitch)
├── run_all_tests.py          # runs everything above in one go
├── reference_audio/          # put your 3-10s .wav reference clips here for cloning
└── output/                   # ALL generated .wav files + latency_results.csv land here
```

Every test script writes only into `output/` — nothing is overwritten silently, filenames are
timestamped or category-named the same way your Piper `edge_case_test.py` did.

---

## 4. How the scripts map to your Piper project

| Piper script | OmniVoice equivalent | Notes |
|---|---|---|
| `download_models.py` | (not needed — `from_pretrained` auto-downloads) | See Step 6 above for the HF mirror fallback |
| `normalize.py` | `normalize.py` | Same purpose, different underlying problem (see Section 6) |
| `check_latency.py` | `check_latency.py` | Same RTF methodology, run on the same laptop for a fair comparison |
| `edge_case_test.py` | `edge_case_test.py` | Same edge-case categories, before/after normalization |
| `test_all_english.py` | `test_voice_design.py` | Piper tested fixed voices; OmniVoice tests *designed* voices instead |
| `test_phn.py` | folded into `edge_case_test.py`'s `phone_number` and `symbols` cases | |
| — (Piper has no cloning) | `test_voice_clone.py` | New capability — this is OmniVoice's main advantage over Piper |
| `dummy_llm_client.py` | `dummy_llm_client.py` | Same end-to-end text-in → audio-out simulation |

---

## 5. Running the tests

```bash
# 1. Simulate the LLM -> TTS pipeline with a dummy sentence (EN + UR)
python dummy_llm_client.py

# 2. Measure real CPU latency (fills in the number your report is missing)
python check_latency.py

# 3. Run the edge-case suite
python edge_case_test.py

# 4. Test zero-shot voice cloning (needs a reference clip in reference_audio/)
python test_voice_clone.py

# 5. Test voice design (attribute-based, no reference audio)
python test_voice_design.py

# or just:
python run_all_tests.py
```

All `.wav` output + a `latency_results.csv` land in `output/`.

---

## 6. Shortcomings of OmniVoice (mirrors your Piper "issues → fix" writeup)

**1. GPU-benchmarked speed claim, CPU reality unknown until measured**
The 0.025 RTF (~40x real-time) figure is measured on a single H100 GPU. Nothing in the public
docs benchmarks CPU-only inference. *Fix applied here:* `check_latency.py` measures real RTF on
this laptop instead of trusting the vendor number — same discipline used for Piper and NeuTTS Nano
in your report's Section 4.

**2. Heavy footprint vs. Piper**
~2.45 GB of weights and a full PyTorch stack, versus ~15–30 MB per Piper voice and a lightweight
onnxruntime runtime. This alone makes OmniVoice a harder fit for constrained or offline deployment
than Piper. *Not fully fixable* — it's an architecture trade-off for the broader language coverage
and cloning capability; the only mitigation is running fewer diffusion steps (`num_step=16` instead
of 32) to trade a little quality for speed.

**3. Urdu number/phone-number normalization is not solved out of the box**
`normalize_text=True` only routes Chinese and English through WeTextProcessing; every other
language — Urdu included — falls back to `num2words`, which doesn't support Urdu at all. This is
the same "123 read digit-by-digit instead of as a number" problem you fixed for Piper, and it is
**not yet fixed** for Urdu in OmniVoice by default. *Fix applied here:* `normalize.py` in this
project expands digits, phone numbers, and common symbols to Urdu/English words before the text
ever reaches the model — the same pattern as your Piper `normalize.py`, ported to this engine.

**4. Voice cloning needs a transcript, and auto-transcription adds a dependency**
Cloning requires `ref_audio` + `ref_text`. If you omit `ref_text`, OmniVoice runs Whisper ASR to
auto-transcribe the clip, which downloads an extra model and adds latency on first use. There's no
way around supplying a transcript (manual or via Whisper) — unlike Piper, which needs nothing but
text.

**5. Voice design is English/Chinese-trained — Urdu results may be unstable**
The README states voice design ("male, low pitch, british accent"-style prompts) is trained
primarily on Chinese and English data, and "may produce unstable results for some low-resource
languages" — Urdu is not explicitly validated. *Not yet fixed*: `test_voice_design.py` includes an
Urdu case specifically to observe whether this instability shows up in practice; if it does, the
practical fallback is using OmniVoice's cloning mode (a real Urdu reference clip) instead of design
mode for the Urdu voice, and reserving design mode for English.

**6. First-run cold start**
Model download (~2.45 GB) plus PyTorch/CUDA initialization means the very first generation call is
much slower than subsequent ones. `check_latency.py` includes a warm-up call before timing starts,
the same warm-up pattern your Piper project already used (`warmup.wav`).

---

## 7. Recommended next step (matches your report's Section 7 plan)

1. Run `check_latency.py` on this laptop, record RTF for both Urdu and English sentences.
2. Compare directly against Piper's ~0.22x–0.23x RT (Section 4.2 of your report).
3. If OmniVoice's CPU RTF is under ~1.0x (usable) and voice cloning quality holds up on Urdu, it
   becomes a real upgrade candidate. If not, stay on Piper and fall back to the custom Urdu
   fine-tuning plan already outlined in your report.
