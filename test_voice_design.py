"""
test_voice_design.py
---------------------
Equivalent of your Piper test_all_english.py, adapted to OmniVoice: instead
of testing fixed preset voices (Ryan/Amy/Kristin/Alan), OmniVoice's "voice
design" mode generates a voice from a text description. This script tests
several attribute combinations for English (where the README says design
mode is reliable), plus one Urdu case to directly check the README's
warning that design mode "may produce unstable results for some
low-resource languages."

Run:
    python test_voice_design.py
"""

import os
import soundfile as sf
import torch

from normalize import normalize_text

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TEXT_EN = "Hello, this is a test of the voice design feature."
TEXT_UR = "السلام علیکم، یہ صوتی ڈیزائن کا امتحان ہے۔"

DESIGN_CASES = [
    ("design_en_male", "en", TEXT_EN, "male, medium pitch, american accent"),
    ("design_en_female", "en", TEXT_EN, "female, medium pitch, american accent"),
    ("design_en_male_british", "en", TEXT_EN, "male, low pitch, british accent"),
    ("design_ur_female", "ur", TEXT_UR, "female, medium pitch"),  # checks README's stability caveat for Urdu
]


def load_model():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    from omnivoice import OmniVoice
    return OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map=device, dtype=dtype)


def main():
    model = load_model()

    for name, lang, text, instruct in DESIGN_CASES:
        clean_text = normalize_text(text, lang=lang)
        audio = model.generate(text=clean_text, instruct=instruct)
        out_path = os.path.join(OUTPUT_DIR, f"{name}.wav")
        sf.write(out_path, audio[0], 24000)
        print(f"[{name}] instruct='{instruct}' -> {out_path}")

    print("\nListen especially to design_ur_female.wav — this is the case most likely")
    print("to show the instability the OmniVoice README warns about for low-resource languages.")


if __name__ == "__main__":
    main()
