"""Build 5 fps H.264 mp4 clips in clips/ from the raw downloads.

UR Fall : raw/urfall/extract/<seq>-cam0-rgb/<seq>-cam0-rgb-NNN.png (30 fps, 640x480)
GMDCSA24: raw/gmdcsa/gm_S<k>_<Fall|ADL>_<nn>.mp4
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CLIPS, ROOT, load_config, load_labels, video_info  # noqa: E402

RAW = ROOT / "raw"


def run(cmd):
    subprocess.run(cmd, check=True, capture_output=True)


def main():
    cfg = load_config()
    fps, maxw = cfg["clip_fps"], cfg["clip_max_width"]
    CLIPS.mkdir(exist_ok=True)
    vf = f"fps={fps},scale='min({maxw},iw)':-2"
    for row in load_labels():
        dst = CLIPS / f"{row['clip']}.mp4"
        if dst.exists():
            continue
        if row["source"] == "urfall":
            seq = row["src_file"].replace(".zip", "")           # e.g. fall-01-cam0-rgb
            pattern = RAW / "urfall" / "extract" / seq / f"{seq}-%03d.png"
            # UR Fall RGB frames are 30 fps
            run(["ffmpeg", "-y", "-v", "error", "-framerate", "30", "-start_number", "1",
                 "-i", str(pattern), "-vf", vf, "-c:v", "libx264", "-preset", "fast",
                 "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)])
        else:
            src = RAW / "gmdcsa" / row["src_file"]
            run(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-vf", vf, "-an",
                 "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
                 "-movflags", "+faststart", str(dst)])
        info = video_info(dst)
        print(f"{dst.name}: {info['width']}x{info['height']} {info['n_frames']} frames "
              f"@ {info['fps']:.0f} fps = {info['n_frames']/info['fps']:.1f}s, {dst.stat().st_size/1e3:.0f} KB")


if __name__ == "__main__":
    main()
