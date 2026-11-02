"""[experiment] Record the Tapo C200 microphone and transcribe with Muse Voice Transcribe; prints audio levels too.
The pipeline uses remy.camera.audio_in.record_mic + remy.checkin.transcribe.transcribe_wav instead.

  python experiments/tapo_audio.py --seconds 15 --tag normal     # record -> tapo/audio_normal_<ts>.wav, then transcribe
  python experiments/tapo_audio.py --seconds 15 --tag quiet
  python experiments/tapo_audio.py --transcribe tapo/audio_normal_x.wav   # transcribe an existing wav
Recording: ffmpeg pulls the RTSP audio track (stream1), mono 16-bit PCM at 16 kHz (what the API accepts).
Transcription: POST https://api.meta.ai/v1/asr/transcribe (multipart: request JSON + audio wav), $0.18 per hour.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import requests

from remy.common import META_BASE_URL, ROOT, api_key, redact, tapo_rtsp, write_json

OUT_DIR = ROOT / "tapo"


def record(seconds: float, tag: str, stream: int) -> Path:
    OUT_DIR.mkdir(exist_ok=True)
    url = tapo_rtsp(stream)
    dst = OUT_DIR / f"audio_{tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
    print(f"recording {seconds:g}s from {redact(url)} -> {dst.name}", flush=True)
    t0 = time.time()
    r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-rtsp_transport", "tcp", "-i", url, "-t", str(seconds),
                        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", "-map_metadata", "-1", str(dst)],
                       capture_output=True, text=True, timeout=seconds + 60)
    if r.returncode != 0 or not dst.exists():
        raise SystemExit(f"ffmpeg failed: {r.stderr.strip()[:500]}")
    print(f"  done in {time.time() - t0:.1f}s, {dst.stat().st_size / 1e3:.0f} KB", flush=True)
    return dst


def level_stats(wav: Path) -> dict:
    import wave
    with wave.open(str(wav)) as w:
        n = w.getnframes(); sr = w.getframerate()
        x = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32768
    rms = float(np.sqrt(np.mean(x ** 2))) if len(x) else 0.0
    return {"duration_s": round(n / sr, 2), "sample_rate": sr, "rms_dbfs": round(20 * np.log10(max(rms, 1e-9)), 1),
            "peak_dbfs": round(20 * np.log10(max(float(np.abs(x).max()) if len(x) else 1e-9, 1e-9)), 1)}


def transcribe(wav: Path, mode: str) -> dict:
    settings = {"model": "muse-voice-transcribe-1.0", "audioEncoding": "WAV", "mode": mode}
    t0 = time.perf_counter()
    resp = requests.post(f"{META_BASE_URL}/asr/transcribe", params={"sessionId": f"fall-test-{wav.stem}"},
                         headers={"Authorization": f"Bearer {api_key()}", "Accept": "application/json"},
                         files={"request": (None, json.dumps(settings), "application/json"),
                                "audio": (wav.name, wav.read_bytes(), "audio/wav")}, timeout=300)
    latency = time.perf_counter() - t0
    if resp.status_code != 200:
        raise SystemExit(f"transcribe failed {resp.status_code}: {resp.text[:400]}")
    body = resp.json()
    out = {"wav": str(wav.relative_to(ROOT) if wav.is_relative_to(ROOT) else wav), "mode": mode, "latency_s": round(latency, 2), "levels": level_stats(wav),
           "transcript": body.get("transcript"), "audio_duration_ms": body.get("audioDurationMs"),
           "turns": body.get("turns"), "session_id": body.get("sessionId"),
           "cost_usd": round((body.get("audioDurationMs") or 0) / 3.6e6 * 0.18, 5)}
    write_json(wav.with_suffix(".transcript.json"), out)
    print(f"  transcript ({latency:.1f}s, levels rms {out['levels']['rms_dbfs']} dBFS): {out['transcript']!r}", flush=True)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seconds", type=float, default=15)
    p.add_argument("--tag", default="normal")
    p.add_argument("--stream", type=int, default=1)
    p.add_argument("--transcribe", help="existing wav instead of recording")
    p.add_argument("--mode", default="ENDPOINTING", choices=["PUSH_TO_TALK", "ENDPOINTING", "DIARIZATION"])
    p.add_argument("--no-transcribe", action="store_true")
    args = p.parse_args()
    wav = Path(args.transcribe).resolve() if args.transcribe else record(args.seconds, args.tag, args.stream)
    if not args.no_transcribe:
        transcribe(wav, args.mode)


if __name__ == "__main__":
    main()
