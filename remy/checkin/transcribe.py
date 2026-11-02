"""Speech to text with Muse Voice Transcribe (Meta Model API): POST /v1/asr/transcribe, multipart
(request JSON + WAV), ENDPOINTING mode. $0.18 per audio hour, logged to the cost ledger."""
from __future__ import annotations

import json
import time
from pathlib import Path

from remy import costs
from remy.common import META_BASE_URL, TRANSCRIBE_USD_PER_HOUR, api_key

ASR_MODEL = "muse-voice-transcribe-1.0"


def transcribe_wav(wav: Path, mode: str = "ENDPOINTING", session_prefix: str = "checkin") -> dict:
    import requests
    settings = {"model": ASR_MODEL, "audioEncoding": "WAV", "mode": mode}
    t0 = time.perf_counter()
    resp = requests.post(f"{META_BASE_URL}/asr/transcribe", params={"sessionId": f"{session_prefix}-{wav.stem}"},
                         headers={"Authorization": f"Bearer {api_key()}", "Accept": "application/json"},
                         files={"request": (None, json.dumps(settings), "application/json"),
                                "audio": (wav.name, wav.read_bytes(), "audio/wav")}, timeout=120)
    lat = time.perf_counter() - t0
    if resp.status_code != 200:
        return {"transcript": "", "error": f"{resp.status_code} {resp.text[:200]}", "latency_s": round(lat, 2)}
    body = resp.json()
    usd = (body.get("audioDurationMs") or 0) / 3.6e6 * TRANSCRIBE_USD_PER_HOUR
    costs.add("transcribe", usd, wav.name)
    return {"transcript": (body.get("transcript") or "").strip(), "turns": body.get("turns"), "latency_s": round(lat, 2), "cost_usd": round(usd, 5)}
