# OmniVoice — Technical Specification & Shortcomings

Companion to `README.md`. This is the OmniVoice equivalent of the Piper
evaluation section in the TTS Engine Evaluation Report — same format,
next model.

---

## 1. Model Overview

| Property | Value |
|---|---|
| Publisher | k2-fsa (Han Zhu et al.) |
| License | Apache 2.0 |
| Architecture | Diffusion language model-style, zero-shot |
| Languages | 600+, including Urdu (`urd`) |
| Voice cloning | Yes — 3–10s reference clip + transcript |
| Voice design | Yes — text-described attributes (English/Chinese trained) |
| Auto voice | Yes — no prompt required |
| Weights size | ~2.45 GB |
| Output sample rate | 24 kHz |
| Vendor RTF | As low as 0.025 (~40x real-time) — **GPU only, H100 benchmark** |
| Fine control | Non-verbal tags (`[laughter]`, `[sigh]`, ...), pinyin/CMU-dict pronunciation overrides |

---

## 2. Generation Modes

| Mode | Needs | Best for |
|---|---|---|
| Voice cloning | `ref_audio` + `ref_text` (or Whisper auto-transcribe) | Matching a specific target voice (Fasih/Amy/Ryan equivalents) |
| Voice design | `instruct` string (e.g. `"female, low pitch, british accent"`) | Quick English/Chinese voices with no reference clip |
| Auto voice | Nothing | Throwaway/testing, not for production consistency |

---

## 3. Comparison vs. Piper (this project's current engine)

| | Piper (current) | OmniVoice |
|---|---|---|
| Voice cloning | No | Yes (zero-shot, 3–10s clip) |
| Urdu support | Yes (fasih) | Yes (urd, ~211h training data) |
| Preset voices | 100+ voices, 30+ languages | None — voice cloning/design only |
| Model size | 15–30MB per voice | ~2.45GB total |
| CPU real-time (measured) | 0.22–0.23x RT | **Not vendor-verified — measure with `check_latency.py`** |
| GPU required for real-time | No | Effectively yes, per vendor benchmarks |
| Symbol/number pronunciation | Fixed via `normalize.py` | Fixed via built-in `normalize_text=True` (EN/ZH) + this project's `normalize.py` (Urdu, phone numbers) |

---

## 4. Shortcomings & Mitigations

Same structure as the Piper section of the evaluation report: each
shortcoming below is paired with what this project does about it.

| # | Shortcoming | Mitigation in this project |
|---|---|---|
| 1 | **No preset voice catalog.** Every voice requires a reference clip + transcript — there's no equivalent of Piper's ready-made fasih/amy/ryan. | `test_voice_modes.py` builds a reusable **voice bank**: clone once from `reference_audio/*.wav`, save as `voice_bank/*.pt`, and every later script/API call reuses the saved prompt instead of re-supplying audio. |
| 2 | **CPU latency is unverified.** The 40x-real-time RTF claim is an H100 GPU benchmark; nothing is published for CPU-only laptops. | `check_latency.py` measures actual RTF on this machine, with a `--fast` (`num_step=16`) fallback and a clear real-time/not-real-time verdict written to `outputs/latency/latency_report.txt`. |
| 3 | **Numbers, phone numbers, and symbols mispronounced** if sent raw — same failure mode that Piper originally had. | `normalize.py`: phone numbers are expanded digit-by-digit, symbols (`%`, `@`, `&`, `#`, `$`, ...) are spelled out, and Urdu digits get a manual word table since OmniVoice's built-in normalizer (WeTextProcessing) does not support Urdu at all. English text also gets OmniVoice's own `normalize_text=True` for natural number reading (e.g. "two thousand three hundred"). |
| 4 | **Voice design is unstable for Urdu.** The vendor README states voice design is trained on English/Chinese only and "may produce unstable results for low-resource languages." | This project restricts voice **design** mode to English test cases only (`test_voice_modes.py`); Urdu always uses voice **cloning** with the fasih reference clip. |
| 5 | **Whisper ASR dependency when `ref_text` is omitted** adds an extra model load and latency. | The voice bank always supplies `ref_text` explicitly (`config.py`), so Whisper auto-transcription is never invoked in this pipeline. |
| 6 | **Zero-shot voice cloning carries misuse risk** (impersonation, unauthorized cloning) — flagged directly in the vendor's own disclaimer. | Only clone from consented, project-owned reference clips (the same fasih/amy/ryan clips already used in Piper testing). Do not add third-party reference audio to `reference_audio/` without consent. |
| 7 | **`pynini`/WeTextProcessing install friction** on Apple Silicon (no prebuilt wheel) blocks the optional `omnivoice[tn]` extra. | Documented workaround in `README.md` (`conda install -c conda-forge pynini` before the pip extra). |
| 8 | **Much heavier than Piper** (~2.45GB vs. tens of MB), so first load and cold-start are slow. | `check_latency.py` runs and discards a warmup generation before timing anything, isolating one-time load cost from steady-state RTF — same approach as the Piper project's `warmup.wav`. |

---

## 5. Open Follow-Up (maps to Section 7 of the evaluation report)

- Run `check_latency.py` on the actual target hardware and record the
  overall avg RTF before deciding whether OmniVoice can run locally or
  needs GPU-backed hosting.
- Run `edge_case_test.py` for both Urdu and English and listen to the
  raw vs. normalized pairs to confirm `normalize.py` is actually fixing
  the pronunciation issues it targets.
- Compare `outputs/latency/latency_report.json` directly against the
  Piper project's Section 4 latency numbers before any decision to adopt,
  drop, or run OmniVoice alongside Piper.
- If CPU RTF comes back too high even with `--fast`, the next step is
  estimating GPU hosting cost (Section 7 of the evaluation report already
  flags this as the fallback decision point).
