"""Regenerate audio/are_you_okay.wav (piper, en_US-lessac-medium) with 1 s of leading silence
(the C200 drops the first second of two-way audio) and an 8 kHz A-law copy."""
import subprocess, sys, wave
from pathlib import Path
from piper import PiperVoice

ROOT = Path(__file__).resolve().parents[1]
text = sys.argv[1] if len(sys.argv) > 1 else "Are you okay?"
voice = PiperVoice.load(str(ROOT / "tools" / "piper" / "en_US-lessac-medium.onnx"))
(ROOT / "audio").mkdir(exist_ok=True)
raw = ROOT / "audio" / "are_you_okay_raw.wav"
with wave.open(str(raw), "wb") as w:
    voice.synthesize_wav(text, w)
subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(raw), "-af", "adelay=1000:all=1", "-ac", "1", "-ar", "22050",
                str(ROOT / "audio" / "are_you_okay.wav")], check=True)
subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(ROOT / "audio" / "are_you_okay.wav"), "-ar", "8000", "-ac", "1",
                "-c:a", "pcm_alaw", str(ROOT / "audio" / "are_you_okay_pcma8k.wav")], check=True)
print("wrote audio/are_you_okay.wav (+1 s lead-in) and audio/are_you_okay_pcma8k.wav")
