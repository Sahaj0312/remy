"""[experiment] Self-check of the camera speaker: record the camera mic while go2rtc plays the phrase through
the camera, then transcribe the recording. If the transcript contains the phrase, the speaker works.
  python experiments/tapo_speaker_check.py --audio pcma
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling experiment module
from tapo_audio import transcribe  # noqa: E402

from remy.camera.go2rtc import ensure_go2rtc, play, stream_info  # noqa: E402
from remy.common import AUDIO_DIR, ROOT, tapo_rtsp  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--audio", default="pcma")
    p.add_argument("--auth", default="plain", choices=["plain", "md5", "sha256"])
    p.add_argument("--wav", default=str(AUDIO_DIR / "are_you_okay.wav"))
    p.add_argument("--record-s", type=float, default=12)
    p.add_argument("--restart", action="store_true")
    args = p.parse_args()
    ensure_go2rtc(args.auth, args.restart)
    print("go2rtc up; tapo stream info before playback:", stream_info("tapo"), flush=True)
    (ROOT / "tapo").mkdir(exist_ok=True)
    rec = ROOT / "tapo" / f"speaker_check_{args.audio}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
    ff = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-rtsp_transport", "tcp", "-i", tapo_rtsp(1), "-t", str(args.record_s),
                           "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", "-map_metadata", "-1", str(rec)],
                          stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    time.sleep(3.0)
    t0 = time.time()
    ok, msg = play(Path(args.wav).resolve(), args.audio, "tapo")
    print(f"play #audio={args.audio}: {msg} ({time.time() - t0:.1f}s)", flush=True)
    time.sleep(3.0)
    print("tapo stream info during/after playback:", stream_info("tapo"), flush=True)
    ff.wait(timeout=args.record_s + 30)
    err = ff.stderr.read().decode(errors="replace").strip()
    if err:
        print("recorder stderr:", err[:300])
    if rec.exists() and rec.stat().st_size > 1000:
        out = transcribe(rec, "ENDPOINTING")
        heard = "okay" in (out.get("transcript") or "").lower()
        print(f"RESULT: {'phrase heard by the camera mic' if heard else 'phrase NOT found in the mic recording'} -> {rec.name}")
    else:
        print("RESULT: mic recording failed")


if __name__ == "__main__":
    main()
