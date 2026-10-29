"""Build H.264 mp4 clips in clips/ (default 10 fps, max width 640) from the raw downloads.

UR Fall : raw/urfall/<seq>-cam0-rgb.zip -> PNG frames at 30 fps, 640x480 (extracted to a temp dir, then deleted)
GMDCSA24: raw/gmdcsa/gm_S<k>_<Fall|ADL>_<nn>.mp4 (60 fps, 1280x720)
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CLIPS, ROOT, load_config, load_labels, video_info  # noqa: E402

RAW = ROOT / "raw"


def run(cmd):
    subprocess.run(cmd, check=True, capture_output=True)


def encode(src_args: list[str], dst: Path, vf: str):
    run(["ffmpeg", "-y", "-v", "error", *src_args, "-vf", vf, "-an", "-c:v", "libx264", "-preset", "fast",
         "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fps", type=float, default=None, help="default: config clip_fps")
    p.add_argument("--out", default=None, help="output dir (default clips/)")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    cfg = load_config()
    fps = args.fps or cfg["clip_fps"]
    out_dir = Path(args.out) if args.out else CLIPS
    out_dir.mkdir(exist_ok=True)
    vf = f"fps={fps:g},scale='min({cfg['clip_max_width']},iw)':-2"
    missing = []
    for row in load_labels():
        dst = out_dir / f"{row['clip']}.mp4"
        if dst.exists() and not args.force:
            continue
        if row["source"] == "urfall":
            seq = row["src_file"].replace(".zip", "")
            zpath = RAW / "urfall" / row["src_file"]
            if not zpath.exists():
                missing.append(row["clip"]); continue
            keep = RAW / "urfall" / "extract" / seq              # first-round extraction, if still there
            if keep.exists():
                encode(["-framerate", "30", "-start_number", "1", "-i", str(keep / f"{seq}-%03d.png")], dst, vf)
            else:
                with tempfile.TemporaryDirectory(dir=str(RAW / "urfall")) as tmp:
                    run(["unzip", "-q", "-o", str(zpath), "-d", tmp])
                    encode(["-framerate", "30", "-start_number", "1", "-i", str(Path(tmp) / seq / f"{seq}-%03d.png")], dst, vf)
        else:
            src = RAW / "gmdcsa" / row["src_file"]
            if not src.exists():
                missing.append(row["clip"]); continue
            encode(["-i", str(src)], dst, vf)
        info = video_info(dst)
        print(f"{dst.name}: {info['width']}x{info['height']} {info['n_frames']} frames @ {info['fps']:g} fps = "
              f"{info['n_frames']/info['fps']:.1f}s, {dst.stat().st_size/1e3:.0f} KB", flush=True)
    if missing:
        print(f"MISSING raw files for {len(missing)} clips: {missing}")


if __name__ == "__main__":
    main()
