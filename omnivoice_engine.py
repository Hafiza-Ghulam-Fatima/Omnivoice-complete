"""
omnivoice_engine.py — thin wrapper around the OmniVoice Python API.

Loads the model ONCE (this is the expensive step — several GB of weights)
and exposes the three generation modes documented at
https://github.com/k2-fsa/OmniVoice: voice cloning, voice design, auto voice.

Every other script (check_latency.py, edge_case_test.py, test_voice_modes.py,
dummy_llm_client.py) imports OmniVoiceEngine from here instead of touching
the raw model directly — same "one wrapper, many callers" shape as the
Piper project's FastAPI app.
"""

import os
import time
import soundfile as sf

import config


class OmniVoiceEngine:
    def __init__(self, device: str = None, dtype=None, verbose: bool = True):
        # Deferred imports: torch/omnivoice are heavy, only pay the cost
        # when the engine is actually instantiated.
        import torch
        from omnivoice import OmniVoice

        self.device = device or config.DEVICE
        self.dtype = dtype or config.DTYPE
        self.verbose = verbose

        if self.verbose:
            print(f"[OmniVoiceEngine] loading {config.MODEL_ID} on {self.device} "
                  f"({self.dtype}) — first run downloads ~2.45GB of weights...")

        t0 = time.time()
        self.model = OmniVoice.from_pretrained(
            config.MODEL_ID,
            device_map=self.device,
            dtype=self.dtype,
        )
        if self.verbose:
            print(f"[OmniVoiceEngine] model ready in {time.time() - t0:.1f}s")

    # -- Voice cloning --------------------------------------------------
    def clone(self, text: str, ref_audio: str, ref_text: str = None,
              normalize_text: bool = False, num_step: int = None, **kw):
        return self.model.generate(
            text=text,
            ref_audio=ref_audio,
            ref_text=ref_text,           # omit to let Whisper auto-transcribe (slower)
            normalize_text=normalize_text,
            num_step=num_step or config.NUM_STEP_DEFAULT,
            **kw,
        )

    def clone_with_prompt(self, text: str, voice_clone_prompt, normalize_text: bool = False,
                           num_step: int = None, **kw):
        """Use a pre-saved VoiceClonePrompt (see test_voice_modes.py) to skip
        re-encoding reference audio on every call — this is the voice-bank
        pattern for reusing fasih/amy/ryan across sessions."""
        return self.model.generate(
            text=text,
            voice_clone_prompt=voice_clone_prompt,
            normalize_text=normalize_text,
            num_step=num_step or config.NUM_STEP_DEFAULT,
            **kw,
        )

    def create_prompt(self, ref_audio: str, ref_text: str):
        return self.model.create_voice_clone_prompt(ref_audio=ref_audio, ref_text=ref_text)

    # -- Voice design (English/Chinese only — see SPECIFICATION.md) -----
    def design(self, text: str, instruct: str, normalize_text: bool = False,
               num_step: int = None, **kw):
        return self.model.generate(
            text=text,
            instruct=instruct,
            normalize_text=normalize_text,
            num_step=num_step or config.NUM_STEP_DEFAULT,
            **kw,
        )

    # -- Auto voice -------------------------------------------------------
    def auto(self, text: str, normalize_text: bool = False, num_step: int = None, **kw):
        return self.model.generate(
            text=text,
            normalize_text=normalize_text,
            num_step=num_step or config.NUM_STEP_DEFAULT,
            **kw,
        )

    # -- Helper: generate + measure + save in one call -------------------
    def synthesize_to_file(self, out_path: str, **generate_kwargs) -> dict:
        """Runs whichever generate() variant fits the given kwargs, times it,
        writes the wav, and returns a small metrics dict used by the test
        scripts (latency, edge cases, dummy LLM)."""
        mode = "clone" if "ref_audio" in generate_kwargs or "voice_clone_prompt" in generate_kwargs \
            else "design" if "instruct" in generate_kwargs else "auto"

        t0 = time.time()
        if mode == "clone" and "voice_clone_prompt" in generate_kwargs:
            audio = self.clone_with_prompt(**generate_kwargs)
        elif mode == "clone":
            audio = self.clone(**generate_kwargs)
        elif mode == "design":
            audio = self.design(**generate_kwargs)
        else:
            audio = self.auto(**generate_kwargs)
        gen_time = time.time() - t0

        wav = audio[0]
        audio_len = len(wav) / config.SAMPLE_RATE
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        sf.write(out_path, wav, config.SAMPLE_RATE)

        return {
            "mode": mode,
            "output_file": out_path,
            "generation_time_sec": round(gen_time, 3),
            "audio_length_sec": round(audio_len, 3),
            "rtf": round(gen_time / audio_len, 4) if audio_len > 0 else None,
        }