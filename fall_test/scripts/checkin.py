"""Spoken check-in: TTS prompt (Kokoro, config language/name), record the camera mic, transcribe
with Muse Voice Transcribe, classify with Spark -> okay | needs_help | unclear | no_answer.

  python scripts/checkin.py --make-prompts            # (re)generate audio/checkin_<lang>.wav for every language in config
  python scripts/checkin.py --classify "I'm fine"      # classification only
  python scripts/checkin.py --run --audio-out laptop   # full round: speak, record 15 s, transcribe, classify
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import costs  # noqa: E402
from common import META_BASE_URL, ROOT, SPARK_USD_PER_INPUT_TOKEN, SPARK_USD_PER_OUTPUT_TOKEN, api_key, load_config, tapo_rtsp  # noqa: E402

PROMPT_DIR = ROOT / "audio"


def prompt_path(lang: str) -> Path:
    return PROMPT_DIR / f"checkin_{lang}.wav"


def make_prompts(cfg: dict) -> dict[str, Path]:
    import soundfile as sf
    from kokoro_onnx import Kokoro
    k = Kokoro(str(ROOT / "tools" / "kokoro" / "kokoro-v1.0.onnx"), str(ROOT / "tools" / "kokoro" / "voices-v1.0.bin"))
    out = {}
    for lang, text in cfg["prompts"].items():
        text = text.replace("{name}", cfg["name"])
        samples, sr = k.create(text, voice=cfg["voices"][lang], speed=cfg.get("speed", 1.0), lang="en-us" if lang == "en" else lang)
        lead = np.zeros(int(sr * cfg.get("lead_in_s", 1.0)), dtype=np.float32)
        PROMPT_DIR.mkdir(exist_ok=True)
        sf.write(str(prompt_path(lang)), np.concatenate([lead, samples.astype(np.float32)]), sr)
        out[lang] = prompt_path(lang)
        print(f"{lang}: '{text}' -> {prompt_path(lang).name} ({(len(lead) + len(samples)) / sr:.1f}s)")
    return out


def record_mic(seconds: float, dst: Path) -> dict:
    t0 = time.time()
    r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-rtsp_transport", "tcp", "-i", tapo_rtsp(1), "-t", str(seconds),
                        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", "-map_metadata", "-1", str(dst)],
                       capture_output=True, text=True, timeout=seconds + 60)
    return {"ok": r.returncode == 0 and dst.exists(), "took_s": round(time.time() - t0, 2), "err": r.stderr.strip()[:200]}


def transcribe_wav(wav: Path) -> dict:
    import requests
    settings = {"model": "muse-voice-transcribe-1.0", "audioEncoding": "WAV", "mode": "ENDPOINTING"}
    t0 = time.perf_counter()
    resp = requests.post(f"{META_BASE_URL}/asr/transcribe", params={"sessionId": f"checkin-{wav.stem}"},
                         headers={"Authorization": f"Bearer {api_key()}", "Accept": "application/json"},
                         files={"request": (None, json.dumps(settings), "application/json"),
                                "audio": (wav.name, wav.read_bytes(), "audio/wav")}, timeout=120)
    lat = time.perf_counter() - t0
    if resp.status_code != 200:
        return {"transcript": "", "error": f"{resp.status_code} {resp.text[:200]}", "latency_s": round(lat, 2)}
    body = resp.json()
    usd = (body.get("audioDurationMs") or 0) / 3.6e6 * 0.18
    costs.add("transcribe", usd, wav.name)
    return {"transcript": (body.get("transcript") or "").strip(), "turns": body.get("turns"), "latency_s": round(lat, 2), "cost_usd": round(usd, 5)}


CLASSIFY_SYSTEM = """You classify the spoken reply of an elderly person who was just asked whether they are okay after a
possible fall at home. The transcript comes from a room microphone and may contain background noise, TV, or other
people. Reply with JSON only.
- "okay": they clearly say they are fine / okay / don't need help (any language, e.g. "I'm okay", "theek hoon", "haan theek hai").
- "needs_help": they ask for help, say they are hurt, cannot get up, or are in pain (e.g. "help", "bachao", "madad", "I can't get up").
- "unclear": words were heard but do not answer the question (TV, unrelated speech, a question back, mumbling).
- "no_answer": nothing meaningful was said.
Return {"status": "...", "quote": "the words that decided it, or empty", "reason": "one short sentence"}."""

SCHEMA = {"type": "object",
          "properties": {"status": {"type": "string", "enum": ["okay", "needs_help", "unclear", "no_answer"]},
                         "quote": {"type": "string"}, "reason": {"type": "string"}},
          "required": ["status", "quote", "reason"], "additionalProperties": False}


def classify(transcript: str, cfg: dict, question: str) -> dict:
    text = (transcript or "").strip()
    if not text:
        return {"status": "no_answer", "quote": "", "reason": "empty transcript", "rule": True, "latency_s": 0.0, "cost_usd": 0.0}
    low = text.lower()
    for w in cfg["help_words"]:
        if w.lower() in low:
            return {"status": "needs_help", "quote": w, "reason": f"help word '{w}' heard", "rule": True, "latency_s": 0.0, "cost_usd": 0.0}
    from openai import OpenAI
    client = OpenAI(base_url=META_BASE_URL, api_key=api_key(), timeout=60)
    t0 = time.perf_counter()
    resp = client.responses.create(
        model="muse-spark-1.3", instructions=CLASSIFY_SYSTEM,
        input=f'Question asked: "{question}"\nTranscript of the next 15 seconds: "{text}"',
        reasoning={"effort": "low"}, max_output_tokens=1500,
        text={"format": {"type": "json_schema", "name": "checkin", "schema": SCHEMA, "strict": True}})
    lat = time.perf_counter() - t0
    usd = resp.usage.input_tokens * SPARK_USD_PER_INPUT_TOKEN + resp.usage.output_tokens * SPARK_USD_PER_OUTPUT_TOKEN
    costs.add("spark-classify", usd, text[:40])
    try:
        out = json.loads(resp.output_text)
    except Exception:
        out = {"status": "unclear", "quote": "", "reason": f"bad JSON: {resp.output_text[:80]}"}
    out.update(rule=False, latency_s=round(lat, 2), cost_usd=round(usd, 5))
    return out


def ask_once(cfg: dict, audio_out: str | None, rec_dir: Path, tag: str, log=print) -> dict:
    """Speak the prompt, record, transcribe, classify. Returns a dict with all timings."""
    from audio_out import speak
    lang = cfg["language"]
    question = cfg["prompts"][lang].replace("{name}", cfg["name"])
    wav = prompt_path(lang)
    if not wav.exists():
        make_prompts(cfg)
    t0 = time.time()
    sp = speak(wav, audio_out, wait=True)
    log(f"check-in: spoke '{question}' on {sp['device']} ({sp['took_s']}s)")
    rec = rec_dir / f"checkin_{tag}_{datetime.now().strftime('%H%M%S')}.wav"
    r = record_mic(cfg["record_s"], rec)
    log(f"check-in: recorded {cfg['record_s']}s mic -> {rec.name} ({r['took_s']}s)")
    tr = transcribe_wav(rec) if r["ok"] else {"transcript": "", "error": r["err"], "latency_s": 0}
    log(f"check-in: transcript ({tr.get('latency_s')}s): {tr.get('transcript')!r}")
    cl = classify(tr.get("transcript", ""), cfg, question)
    log(f"check-in: classified {cl['status']} ({'rule' if cl.get('rule') else 'spark'}, {cl.get('latency_s')}s): {cl.get('reason')}")
    return {"question": question, "speak": sp, "record": r, "wav": str(rec), "transcript": tr, "classification": cl,
            "status": cl["status"], "total_s": round(time.time() - t0, 2)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--make-prompts", action="store_true")
    p.add_argument("--classify")
    p.add_argument("--run", action="store_true")
    p.add_argument("--audio-out", default=None)
    a = p.parse_args()
    cfg = load_config()["checkin"]
    if a.make_prompts:
        make_prompts(cfg)
    if a.classify is not None:
        print(json.dumps(classify(a.classify, cfg, cfg["prompts"][cfg["language"]].replace("{name}", cfg["name"])), ensure_ascii=False))
    if a.run:
        d = ROOT / "events" / "checkin_manual"; d.mkdir(parents=True, exist_ok=True)
        print(json.dumps(ask_once(cfg, a.audio_out, d, "manual"), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
