"""The spoken lines: one WAV per line and language in audio/, each with a short silent lead-in because the
camera drops the first second of two-way audio.

Lines are keyed in config checkin.lines (ask, ask_again, ask_floor, recheck, ack_okay, ack_sent) with {name}
filled in. `remy test-checkin --make-prompts` generates them all with Kokoro (local ONNX TTS); a better
recording of any line (say, from ElevenLabs) replaces it with `remy test-checkin --import file --as key`.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from remy.common import AUDIO_DIR, TOOLS

KOKORO_DIR = TOOLS / "kokoro"
LINE_KEYS = ("ask", "ask_again", "ask_floor", "recheck", "ack_okay", "ack_sent")


def line_text(cfg: dict, key: str = "ask", lang: str | None = None) -> str:
    """The text of a line with the name filled in; falls back to the old `prompts` {lang: text} for `ask`."""
    lang = lang or cfg["language"]
    lines = cfg.get("lines") or {}
    text = (lines.get(key) or {}).get(lang) if isinstance(lines.get(key), dict) else lines.get(key)
    if text is None and key == "ask":
        text = (cfg.get("prompts") or {}).get(lang)
    if text is None:
        raise KeyError(f"no text for line {key!r} in language {lang!r} (config checkin.lines)")
    return text.replace("{name}", cfg["name"])


def prompt_path(key: str = "ask", lang: str = "en") -> Path:
    """audio/<key>_<lang>.wav; the first-round question also accepts the older audio/checkin_<lang>.wav."""
    p = AUDIO_DIR / f"{key}_{lang}.wav"
    if key == "ask" and not p.exists() and (AUDIO_DIR / f"checkin_{lang}.wav").exists():
        return AUDIO_DIR / f"checkin_{lang}.wav"
    return p


def available_keys(cfg: dict) -> list[str]:
    lines = cfg.get("lines") or {}
    keys = [k for k in LINE_KEYS if k in lines] or (["ask"] if cfg.get("prompts") else [])
    return keys


def make_prompts(cfg: dict, keys: list[str] | None = None, force: bool = True) -> dict[str, Path]:
    """Generate the lines with Kokoro. cfg is the "checkin" section: lines, voices {lang: kokoro voice}, speed,
    lead_in_s. With force=False, lines that already have a file (for example an imported one) are kept."""
    import soundfile as sf
    from kokoro_onnx import Kokoro
    model, voices = KOKORO_DIR / "kokoro-v1.0.onnx", KOKORO_DIR / "voices-v1.0.bin"
    if not model.exists() or not voices.exists():
        raise SystemExit(f"Kokoro model files missing in {KOKORO_DIR} (kokoro-v1.0.onnx, voices-v1.0.bin); see docs/hardware-setup.md")
    k = Kokoro(str(model), str(voices))
    lang = cfg["language"]
    out = {}
    AUDIO_DIR.mkdir(exist_ok=True)
    for key in keys or available_keys(cfg):
        dst = AUDIO_DIR / f"{key}_{lang}.wav"
        if dst.exists() and not force:
            out[key] = dst; continue
        text = line_text(cfg, key, lang)
        samples, sr = k.create(text, voice=cfg["voices"][lang], speed=cfg.get("speed", 1.0), lang="en-us" if lang == "en" else lang)
        lead = np.zeros(int(sr * cfg.get("lead_in_s", 1.0)), dtype=np.float32)
        sf.write(str(dst), np.concatenate([lead, samples.astype(np.float32)]), sr)
        out[key] = dst
        print(f"{key}: '{text}' -> {dst.name} ({(len(lead) + len(samples)) / sr:.1f}s, kokoro)")
    return out


def import_prompt(src: Path, key: str, cfg: dict, lang: str | None = None) -> Path:
    """A recording from anywhere (WAV, MP3, M4A...) -> audio/<key>_<lang>.wav: mono, 22.05 kHz, with the lead-in
    silence the camera needs. Replaces the Kokoro line of the same key."""
    if key not in LINE_KEYS:
        raise SystemExit(f"unknown line {key!r}; one of {', '.join(LINE_KEYS)}")
    lang = lang or cfg["language"]
    AUDIO_DIR.mkdir(exist_ok=True)
    dst = AUDIO_DIR / f"{key}_{lang}.wav"
    lead_ms = int(1000 * cfg.get("lead_in_s", 1.0))
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-af", f"adelay={lead_ms}:all=1,loudnorm=I=-16:TP=-1.5",
                    "-ac", "1", "-ar", "22050", "-c:a", "pcm_s16le", str(dst)], check=True)
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(dst)],
                               capture_output=True, text=True).stdout.strip() or 0)
    print(f"{key}: {src.name} -> {dst.name} ({dur:.1f}s incl. {lead_ms / 1000:g}s lead-in)")
    return dst
