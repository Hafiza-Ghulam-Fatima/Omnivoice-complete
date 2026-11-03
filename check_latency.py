"""
check_latency.py
-----------------
Mirrors your Piper check_latency.py methodology exactly, so the numbers are
directly comparable: RTF = generation time / audio length. This is what
turns the vendor's GPU-only "0.025 RTF" claim into a real, measured number
for this laptop, per your report's Section 7 next-step.

Run:
    python check_latency.py
"""

import csv
import os
import time
import soundfile as sf
import torch

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TEST_SENTENCES = [
    ("ur", "یہ ایک ٹیسٹ جملہ ہے تاکہ رفتار کو ناپا جا سکے۔"),
    ("ur", "براہ کرم انتظار کریں، آپ کی درخواست پر کارروائی ہو رہی ہے۔"),
    ("en", "This is a test sentence to measure generation speed."),
    ("en", "Please wait a moment while your request is being processed."),
]


def load_model():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    print(f"Loading OmniVoice on device={device}, dtype={dtype}...")
    from omnivoice import OmniVoice
    model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map=device, dtype=dtype)
    return model, device


def run_latency_test(model, device):
    results = []

    # Warm-up call: excluded from timing, same pattern as your Piper
    # project's warmup.wav — first call pays PyTorch/CUDA init cost.
    print("Warming up...")
    _ = model.generate(text="Warm up.")

    for idx, (lang, text) in enumerate(TEST_SENTENCES, start=1):
        start = time.time()
        audio = model.generate(text=text)
        elapsed = time.time() - start

        duration = len(audio[0]) / 24000
        rtf = elapsed / duration if duration > 0 else float("nan")

        out_path = os.path.join(OUTPUT_DIR, f"latency_test_{idx}_{lang}.wav")
        sf.write(out_path, audio[0], 24000)

        print(f"[{idx}] lang={lang} gen={elapsed:.2f}s audio={duration:.2f}s RTF={rtf:.3f}x -> {out_path}")
        results.append({
            "index": idx,
            "lang": lang,
            "text": text,
            "device": device,
            "gen_time_s": round(elapsed, 3),
            "audio_len_s": round(duration, 3),
            "rtf": round(rtf, 4),
            "faster_than_realtime": rtf < 1.0,
        })

    return results


def save_csv(results):
    csv_path = os.path.join(OUTPUT_DIR, "latency_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved summary -> {csv_path}")


def main():
    model, device = load_model()
    results = run_latency_test(model, device)
    save_csv(results)

    rtfs = [r["rtf"] for r in results]
    print(f"\nAverage RTF: {sum(rtfs)/len(rtfs):.3f}x  (below 1.0x = faster than real-time)")
    print("Compare this directly against Piper's ~0.22x-0.23x RT (report Section 4.2).")


if __name__ == "__main__":
    main()
