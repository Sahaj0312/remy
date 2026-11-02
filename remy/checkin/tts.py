"""Check-in prompts with Kokoro TTS (local, ONNX): audio/checkin_<lang>.wav with a short silent lead-in,
because the camera drops the first second of two-way audio."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from remy.common import AUDIO_DIR, TOOLS

KOKORO_DIR = TOOLS / "kokoro"


def prompt_path(lang: str) -> Path:
    return AUDIO_DIR / f"checkin_{lang}.wav"


def make_prompts(cfg: dict) -> dict[str, Path]:
    """cfg is the "checkin" section: prompts {lang: text with {name}}, voices {lang: kokoro voice}, speed, lead_in_s."""
    import soundfile as sf
    from kokoro_onnx import Kokoro
    model, voices = KOKORO_DIR / "kokoro-v1.0.onnx", KOKORO_DIR / "voices-v1.0.bin"
    if not model.exists() or not voices.exists():
        raise SystemExit(f"Kokoro model files missing in {KOKORO_DIR} (kokoro-v1.0.onnx, voices-v1.0.bin); see docs/hardware-setup.md")
    k = Kokoro(str(model), str(voices))
    out = {}
    for lang, text in cfg["prompts"].items():
        text = text.replace("{name}", cfg["name"])
        samples, sr = k.create(text, voice=cfg["voices"][lang], speed=cfg.get("speed", 1.0), lang="en-us" if lang == "en" else lang)
        lead = np.zeros(int(sr * cfg.get("lead_in_s", 1.0)), dtype=np.float32)
        AUDIO_DIR.mkdir(exist_ok=True)
        sf.write(str(prompt_path(lang)), np.concatenate([lead, samples.astype(np.float32)]), sr)
        out[lang] = prompt_path(lang)
        print(f"{lang}: '{text}' -> {prompt_path(lang).name} ({(len(lead) + len(samples)) / sr:.1f}s)")
    return out
