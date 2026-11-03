"""
test_voice_modes.py — exercises OmniVoice's three generation modes and
builds the reusable "voice bank" (saved VoiceClonePrompt files), which is
the OmniVoice equivalent of Piper's fixed fasih/amy/ryan voice catalog.

Unlike Piper, OmniVoice has no preset voices — every voice must be defined
by a reference clip. This script:
  1. Clones each of the three voice-bank entries (fasih/amy/ryan) from their
     reference clip + transcript, and SAVES the resulting prompt to
     voice_bank/<name>.pt so future runs don't need to re-encode audio.
  2. Re-loads each saved prompt and generates from it, to prove the saved
     prompts actually work standalone (this is what dummy_llm_client.py and
     any production API would use going forward).
  3. Runs one voice-design example (English only — see SPECIFICATION.md for
     why Urdu is not used here) and one auto-voice example.

Usage:
    python test_voice_modes.py
"""

import os

import config
from omnivoice_engine import OmniVoiceEngine
from normalize import normalize_text

TEST_TEXT_EN = "This is a test of the finalized voice bank for the AI avatar assistant."
TEST_TEXT_UR = normalize_text("السلام علیکم، یہ آواز کے بینک کی جانچ ہے۔", "ur")


def check_voice_bank_files_exist():
    missing = [name for name, v in config.VOICE_BANK.items()
               if not os.path.exists(v["ref_audio"])]
    if missing:
        print("MISSING reference audio for:", missing)
        print(f"Copy the corresponding .wav files into {config.REFERENCE_AUDIO_DIR}/ "
              f"and fill in the exact transcripts in config.py before running this script.")
        raise SystemExit(1)


def main():
    check_voice_bank_files_exist()
    engine = OmniVoiceEngine()
    out_dir = os.path.join(config.OUTPUT_DIR, "voice_modes")

    # 1 + 2: build and verify the voice bank -----------------------------
    for name, voice in config.VOICE_BANK.items():
        text = TEST_TEXT_UR if voice["language_id"] == "urd" else TEST_TEXT_EN

        # Build the prompt once and save it
        prompt = engine.create_prompt(ref_audio=voice["ref_audio"], ref_text=voice["ref_text"])
        prompt.save(config.voice_bank_prompt_path(name))
        print(f"[{name}] saved voice-clone prompt -> {config.voice_bank_prompt_path(name)}")

        # Generate directly from ref_audio (first-time path)
        direct_metrics = engine.synthesize_to_file(
            os.path.join(out_dir, f"{name}_direct.wav"),
            text=text, ref_audio=voice["ref_audio"], ref_text=voice["ref_text"],
        )

        # Reload the saved prompt from disk and generate again (production path)
        from omnivoice import VoiceClonePrompt
        reloaded_prompt = VoiceClonePrompt.load(config.voice_bank_prompt_path(name))
        prompt_metrics = engine.synthesize_to_file(
            os.path.join(out_dir, f"{name}_from_saved_prompt.wav"),
            text=text, voice_clone_prompt=reloaded_prompt,
        )

        print(f"  direct: RTF={direct_metrics['rtf']}   "
              f"from saved prompt: RTF={prompt_metrics['rtf']}")

    # 3: voice design (English only) -------------------------------------
    design_metrics = engine.synthesize_to_file(
        os.path.join(out_dir, "design_female_british.wav"),
        text="This voice was generated without any reference audio.",
        instruct="female, british accent",
    )
    print(f"[voice design] RTF={design_metrics['rtf']}")

    # 4: auto voice ---------------------------------------------------------
    auto_metrics = engine.synthesize_to_file(
        os.path.join(out_dir, "auto_voice.wav"),
        text="This is a sentence synthesized without any voice prompt at all.",
    )
    print(f"[auto voice] RTF={auto_metrics['rtf']}")

    print(f"\nAll voice-mode outputs saved under {out_dir}/")


if __name__ == "__main__":
    main()
