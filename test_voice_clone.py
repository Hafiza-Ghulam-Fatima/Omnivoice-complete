"""
test_voice_clone.py
--------------------
Tests OmniVoice's zero-shot voice cloning — the capability Piper does NOT
have (your report Section 5: "the accepted trade-off for Piper is that it
does not support voice cloning"). This script is the direct test of whether
that gap can be closed.

Before running:
    Put a 3-10 second reference clip (clean speech, ideally in the target
    language) at reference_audio/ref.wav. See reference_audio/README.txt.

Run:
    python test_voice_clone.py
"""

import os
import soundfile as sf
import torch

from normalize import normalize_text

OUTPUT_DIR = "output"
REF_AUDIO_PATH = "reference_audio/ref.wav"
REF_TEXT = None  # set to the exact transcript of ref.wav to skip Whisper auto-ASR

os.makedirs(OUTPUT_DIR, exist_ok=True)

TEST_TEXT = {
    "ur": "یہ آواز کلوننگ کا امتحان ہے، اور یہ نئے جملے پڑھ رہی ہے۔",
    "en": "This is a voice cloning test, reading a brand new sentence.",
}


def load_model():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    from omnivoice import OmniVoice
    return OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map=device, dtype=dtype)


def main():
    if not os.path.exists(REF_AUDIO_PATH):
        print(f"No reference clip found at {REF_AUDIO_PATH}.")
        print("Add a 3-10s .wav clip there before running this test — see reference_audio/README.txt.")
        return

    model = load_model()

    for lang, text in TEST_TEXT.items():
        clean_text = normalize_text(text, lang=lang)

        gen_kwargs = dict(text=clean_text, ref_audio=REF_AUDIO_PATH)
        if REF_TEXT:
            gen_kwargs["ref_text"] = REF_TEXT
        else:
            print(f"[{lang}] No ref_text supplied -> OmniVoice will auto-transcribe "
                  f"reference_audio/ref.wav with Whisper (slower on first run).")

        audio = model.generate(**gen_kwargs)
        out_path = os.path.join(OUTPUT_DIR, f"clone_test_{lang}.wav")
        sf.write(out_path, audio[0], 24000)
        print(f"[{lang}] saved -> {out_path}")

    print("\nCompare clone_test_ur.wav / clone_test_en.wav against your reference clip.")
    print("This is the feature Piper's fasih/amy/ryan voices cannot do at all.")


if __name__ == "__main__":
    main()
