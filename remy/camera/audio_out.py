"""Speak a wav on the configured output device: "camera" (Tapo speaker via go2rtc) or "laptop" (afplay)."""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from remy.common import AUDIO_DIR, load_config


def speak(wav: Path, device: str | None = None, wait: bool = True) -> dict:
    cfg = load_config()["audio"]
    device = device or cfg["output"]
    wav = Path(wav).resolve()
    t0 = time.time()
    if device == "camera":
        from remy.camera.go2rtc import ensure_go2rtc, play
        ensure_go2rtc("plain", False)
        ok, msg = play(wav, cfg.get("camera_codec", "pcma"), cfg.get("camera_stream", "tapo"))
        if wait:
            dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(wav)],
                                       capture_output=True, text=True).stdout.strip() or 2.0)
            time.sleep(dur + 0.5)
        return {"device": "camera", "ok": ok, "detail": msg, "took_s": round(time.time() - t0, 2)}
    cmd = cfg.get("laptop_cmd", "afplay")
    p = subprocess.Popen([cmd, str(wav)])
    if wait:
        p.wait()
    return {"device": "laptop", "ok": True, "detail": cmd, "took_s": round(time.time() - t0, 2)}


if __name__ == "__main__":
    print(speak(sys.argv[1] if len(sys.argv) > 1 else AUDIO_DIR / "are_you_okay.wav",
                sys.argv[2] if len(sys.argv) > 2 else None))
