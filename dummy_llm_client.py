"""
dummy_llm_client.py
--------------------
Mirrors the Piper project's dummy_llm_client.py: since Group 2's real LLM
output isn't available yet, this simulates a fixed LLM response, runs it
through normalization, and synthesizes it with OmniVoice — validating the
full text-in -> audio-out path end-to-end.

Run:
    python dummy_llm_client.py
"""

import os
import time
import soundfile as sf
import torch

from normalize import normalize_text

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Simulated LLM responses (stand-ins for Group 2's real output). Deliberately
# includes numbers/phone numbers/symbols so normalization is exercised.
DUMMY_LLM_RESPONSES = [
    {
        "id": "dummy_ur_01",
        "lang": "ur",
        "text": "السلام علیکم، میں آپ کا AI اسسٹنٹ ہوں۔ آپ سے 20% رعایت پر رابطہ کیلئے 0301-1234567 پر کال کریں۔",
    },
    {
        "id": "dummy_en_01",
        "lang": "en",
        "text": "Hello, I'm your AI assistant. You can reach support at 0301-1234567, or save 20% today.",
    },
]


def load_model():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    print(f"Loading OmniVoice on device={device}, dtype={dtype} (first run downloads ~2.45GB)...")
    model = OmniVoice_from_pretrained(device, dtype)
    return model, device


def OmniVoice_from_pretrained(device, dtype):
    from omnivoice import OmniVoice
    return OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map=device, dtype=dtype)


def main():
    model, device = load_model()

    for item in DUMMY_LLM_RESPONSES:
        raw_text = item["text"]
        clean_text = normalize_text(raw_text, lang=item["lang"])

        print(f"\n[{item['id']}] raw       : {raw_text}")
        print(f"[{item['id']}] normalized: {clean_text}")

        start = time.time()
        # Auto-voice mode: no reference audio / instruct needed, matches the
        # "dummy LLM text -> speech" requirement with zero extra setup.
        audio = model.generate(text=clean_text)
        elapsed = time.time() - start

        out_path = os.path.join(OUTPUT_DIR, f"{item['id']}.wav")
        sf.write(out_path, audio[0], 24000)

        duration = len(audio[0]) / 24000
        rtf = elapsed / duration if duration > 0 else float("nan")
        print(f"[{item['id']}] saved -> {out_path} | gen={elapsed:.2f}s | audio={duration:.2f}s | RTF={rtf:.3f}x")

    print("\nDone. Check the output/ folder for generated .wav files.")


if __name__ == "__main__":
    main()
