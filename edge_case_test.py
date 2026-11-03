"""
edge_case_test.py
------------------
Same edge-case categories your Piper project already covers (numbers,
phone numbers, symbols/punctuation, mixed code-switching, long sentences,
short greetings, rare words) — run against OmniVoice, BEFORE and AFTER
normalize.py, so you can hear the difference the same way you validated it
for Piper.

Run:
    python edge_case_test.py
"""

import os
import soundfile as sf
import torch

from normalize import normalize_text

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

EDGE_CASES = {
    "numbers":            ("en", "The package weighs 45.5 kg and costs $1,250."),
    "phone_number":       ("en", "Contact us at 0301-1234567 for support."),
    "symbols":            ("en", "Use A&B Corp, email us at contact@company.com, get 25% off."),
    "punctuation":        ("en", "Wait... really?! That's amazing — I can't believe it."),
    "mixed_code_switch":  ("ur", "آج meeting 3 بجے ہے، please وقت پر آئیں۔"),
    "long_sentence":      ("ur", "یہ ایک لمبا جملہ ہے جس میں کئی الفاظ اور خیالات شامل ہیں تاکہ ماڈل کی کارکردگی کو طویل متن پر پرکھا جا سکے۔"),
    "short_greeting":     ("ur", "السلام علیکم، کیسے ہیں آپ؟"),
    "rare_words":         ("en", "The onomatopoeic susurrus of the cryptobiotic ecosystem was mesmerizing."),
}


def load_model():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    from omnivoice import OmniVoice
    return OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map=device, dtype=dtype)


def main():
    model = load_model()

    for case_name, (lang, text) in EDGE_CASES.items():
        # BEFORE normalization (raw text, as an LLM might actually produce it)
        audio_before = model.generate(text=text)
        before_path = os.path.join(OUTPUT_DIR, f"edge_{case_name}_before.wav")
        sf.write(before_path, audio_before[0], 24000)

        # AFTER normalization
        clean_text = normalize_text(text, lang=lang)
        audio_after = model.generate(text=clean_text)
        after_path = os.path.join(OUTPUT_DIR, f"edge_{case_name}_after.wav")
        sf.write(after_path, audio_after[0], 24000)

        print(f"[{case_name}] lang={lang}")
        print(f"    raw       : {text}")
        print(f"    normalized: {clean_text}")
        print(f"    saved     : {before_path}  |  {after_path}\n")

    print("Done. Listen to *_before.wav vs *_after.wav pairs in output/ to compare.")


if __name__ == "__main__":
    main()
