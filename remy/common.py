"""Shared helpers: project root, config.json, .env, Meta Model API constants, small file utilities.

The project root holds the per-home files (config.json, .env, zones.json, family.json, events/, audio/,
tools/). It is REMY_HOME when set, else the source checkout this package lives in, else the current directory.
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
from pathlib import Path


def _find_root() -> Path:
    env = os.environ.get("REMY_HOME")
    if env:
        return Path(env).expanduser().resolve()
    here = Path(__file__).resolve().parents[1]
    if (here / "pyproject.toml").exists():
        return here
    return Path.cwd()


ROOT = _find_root()
CONFIG_PATH = ROOT / "config.json"
CONFIG_EXAMPLE = ROOT / "config.example.json"
EVENTS = ROOT / "events"            # one folder per live event: clip, snapshot, track, score, log, event.json
AUDIO_DIR = ROOT / "audio"          # generated check-in prompts and recordings
TOOLS = ROOT / "tools"              # go2rtc binary + config, Kokoro model files (not in git)

# evaluation set (eval/): clips, raw downloads, per-clip outputs
EVAL = ROOT / "eval"
CLIPS = EVAL / "clips"
OUT = EVAL / "out"
LABELS_PATH = EVAL / "labels.csv"

META_BASE_URL = "https://api.meta.ai/v1"
SAM_MODEL = "sam-3.1"
SPARK_MODEL = "muse-spark-1.3"

# Pricing from https://dev.meta.ai/docs/pricing-rate-limits (checked 2026-10-29)
SAM_USD_PER_VIDEO_FRAME = 0.20 / 1000
SAM_MIN_BILLED_FRAMES = 50
SPARK_USD_PER_INPUT_TOKEN = 1.25 / 1_000_000
SPARK_USD_PER_OUTPUT_TOKEN = 4.25 / 1_000_000
TRANSCRIBE_USD_PER_HOUR = 0.18


def load_config() -> dict:
    """config.json at the project root; falls back to config.example.json so a fresh checkout runs."""
    path = CONFIG_PATH if CONFIG_PATH.exists() else CONFIG_EXAMPLE
    with open(path) as f:
        return json.load(f)


def load_labels() -> list[dict]:
    with open(LABELS_PATH, newline="") as f:
        return list(csv.DictReader(f))


def env_get(*names: str, required: bool = True) -> str | None:
    """First of `names` found in the environment, else in <root>/.env."""
    for name in names:
        if os.environ.get(name):
            return os.environ[name]
    env_file = ROOT / ".env"
    if env_file.exists():
        vals = {}
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            vals[k.strip()] = v.strip().strip('"').strip("'")
        for name in names:
            if vals.get(name):
                return vals[name]
    if required:
        raise SystemExit(f"{names[0]} is not set. Export it or put {names[0]}=... in {env_file} (see .env.example)")
    return None


def api_key() -> str:
    """META_API_KEY (or MODEL_API_KEY) from the environment or .env."""
    return env_get("META_API_KEY", "MODEL_API_KEY")


def tapo_rtsp(stream: int = 2) -> str:
    """rtsp://user:pass@ip:554/stream<N> from TAPO_IP / TAPO_USER / TAPO_PASS."""
    from urllib.parse import quote
    ip, user, pw = env_get("TAPO_IP"), env_get("TAPO_USER"), env_get("TAPO_PASS")
    return f"rtsp://{quote(user, safe='')}:{quote(pw, safe='')}@{ip}:554/stream{stream}"


def redact(url: str) -> str:
    import re
    return re.sub(r"//[^@/]+@", "//<user>:<pass>@", url)


def video_info(path: Path) -> dict:
    """width, height, fps, n_frames via ffprobe."""
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-count_frames",
            "-show_entries", "stream=width,height,r_frame_rate,nb_read_frames",
            "-of", "json", str(path),
        ],
        capture_output=True, text=True, check=True,
    ).stdout
    s = json.loads(out)["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    return {
        "width": int(s["width"]),
        "height": int(s["height"]),
        "fps": float(num) / float(den),
        "n_frames": int(s["nb_read_frames"]),
    }


def sam_cost_usd(n_frames: int) -> float:
    return max(n_frames, SAM_MIN_BILLED_FRAMES) * SAM_USD_PER_VIDEO_FRAME


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)


def read_json(path: Path):
    with open(path) as f:
        return json.load(f)
