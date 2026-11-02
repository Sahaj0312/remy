"""go2rtc helpers: the Tapo C200 speaker through go2rtc's two-way audio (tapo:// source).

ensure_go2rtc() starts tools/go2rtc/go2rtc with a config generated from infra/go2rtc.template.yaml + .env
if nothing listens on 127.0.0.1:1984; play() then calls
  POST http://127.0.0.1:1984/api/streams?dst=tapo&src=ffmpeg:<wav>#audio=<codec>#input=file
The camera accepts this only after "Third-Party Compatibility" is on in the Tapo app (docs/hardware-setup.md).

  python -m remy.camera.go2rtc                        # plays audio/are_you_okay.wav with #audio=pcma
  python -m remy.camera.go2rtc --wav x.wav --audio pcmu
  python -m remy.camera.go2rtc --try-all              # pcma, then pcmu, then auto, then aac; reports each error
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import time
from pathlib import Path
from urllib.parse import quote

import requests

from remy.common import AUDIO_DIR, ROOT, TOOLS, env_get

GO2RTC = TOOLS / "go2rtc" / "go2rtc"
CFG = TOOLS / "go2rtc" / "go2rtc.yaml"
TEMPLATE = ROOT / "infra" / "go2rtc.template.yaml"
API = "http://127.0.0.1:1984"


def write_config(auth: str) -> None:
    tpl = TEMPLATE.read_text()
    ip, user, pw, cloud = env_get("TAPO_IP"), env_get("TAPO_USER"), env_get("TAPO_PASS"), env_get("TAPO_CLOUD_PASS")
    if auth == "md5":
        cloud_part = "admin:" + hashlib.md5(cloud.encode()).hexdigest().upper()
    elif auth == "sha256":
        cloud_part = "admin:" + hashlib.sha256(cloud.encode()).hexdigest().upper()
    else:
        cloud_part = quote(cloud, safe="")
    cfg = (tpl.replace("${TAPO_CLOUD_PASS}", cloud_part).replace("${TAPO_IP}", ip)
              .replace("${TAPO_USER}", quote(user, safe="")).replace("${TAPO_PASS}", quote(pw, safe="")))
    CFG.write_text(cfg)


def go2rtc_up() -> bool:
    try:
        return requests.get(f"{API}/api", timeout=2).status_code == 200
    except Exception:
        return False


def ensure_go2rtc(auth: str, restart: bool) -> subprocess.Popen | None:
    if go2rtc_up() and not restart:
        return None
    if restart:
        subprocess.run(["pkill", "-f", str(GO2RTC)], capture_output=True); time.sleep(1)
    if not GO2RTC.exists():
        raise SystemExit(f"go2rtc binary not found at {GO2RTC}; see docs/hardware-setup.md")
    write_config(auth)
    log = open(GO2RTC.parent / "go2rtc.log", "a")
    proc = subprocess.Popen([str(GO2RTC), "-config", str(CFG)], stdout=log, stderr=subprocess.STDOUT, cwd=str(CFG.parent))
    for _ in range(40):
        if go2rtc_up():
            return proc
        time.sleep(0.25)
    raise SystemExit("go2rtc did not start; see tools/go2rtc/go2rtc.log")


def stream_info(name: str) -> dict:
    try:
        return requests.get(f"{API}/api/streams", params={"src": name}, timeout=10).json()
    except Exception as e:
        return {"error": repr(e)}


def play(wav: Path, codec: str, dst: str) -> tuple[bool, str]:
    src = f"ffmpeg:{wav}#audio={codec}#input=file"
    r = requests.post(f"{API}/api/streams", params={"dst": dst, "src": src}, timeout=30)
    ok = r.status_code == 200
    return ok, f"HTTP {r.status_code} {r.text.strip()[:300]}"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m remy.camera.go2rtc")
    p.add_argument("--wav", default=str(AUDIO_DIR / "are_you_okay.wav"))
    p.add_argument("--audio", default="pcma", help="pcma | pcmu | auto | aac | opus")
    p.add_argument("--dst", default="tapo")
    p.add_argument("--auth", default="plain", choices=["plain", "md5", "sha256"])
    p.add_argument("--try-all", action="store_true")
    p.add_argument("--restart", action="store_true")
    args = p.parse_args(argv)
    wav = Path(args.wav).resolve()
    ensure_go2rtc(args.auth, args.restart)
    print("go2rtc up at", API, "| stream info:", stream_info(args.dst), flush=True)
    codecs = ["pcma", "pcmu", "auto", "aac"] if args.try_all else [args.audio]
    for codec in codecs:
        t0 = time.time()
        ok, msg = play(wav, codec, args.dst)
        print(f"play #audio={codec}: {msg}  ({time.time() - t0:.1f}s)", flush=True)
        if ok:
            time.sleep(4)          # let the ~2 s file finish
            print("  producers/consumers after playback:", stream_info(args.dst), flush=True)
            if not args.try_all:
                break
            print("  (did you hear it? next codec in 3 s)", flush=True); time.sleep(3)
    print(f"\nTo replay by hand:\n  curl -X POST '{API}/api/streams?dst={args.dst}&src=ffmpeg:{wav}%23audio=pcma%23input=file'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
