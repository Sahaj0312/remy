"""Spoken check-in: say "<name>, are you okay?" (Kokoro, on the camera speaker or the laptop), record the
camera microphone for record_s seconds, transcribe with Muse Voice Transcribe, classify the reply.

  okay        a clear statement of being fine, in any wording or language (judged by the triage backend)
  needs_help  a help word (config checkin.help_words) or a statement of being hurt / unable to get up
  unclear     words that do not answer the question (TV, a question back, mumbling)
  no_answer   nothing was heard

  remy test-checkin --make-prompts            # (re)generate audio/checkin_<lang>.wav for every language in config
  remy test-checkin --classify "I'm fine"     # classification only
  remy test-checkin --run --audio-out laptop  # full round: speak, record, transcribe, classify
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from remy.camera.audio_in import record_mic
from remy.checkin.transcribe import transcribe_wav
from remy.checkin.tts import make_prompts, prompt_path
from remy.common import EVENTS, load_config

__all__ = ["ask_once", "classify", "make_prompts", "prompt_path", "record_mic", "transcribe_wav", "question_text"]


def question_text(cfg: dict) -> str:
    return cfg["prompts"][cfg["language"]].replace("{name}", cfg["name"])


def _backend(backend=None):
    if backend is None:
        from remy.triage import get_backend
        backend = get_backend(load_config(), timeout=60)
    return backend


def classify(transcript: str, cfg: dict, question: str, backend=None) -> dict:
    """Rules first (empty -> no_answer, help word -> needs_help), then the triage backend."""
    text = (transcript or "").strip()
    if not text:
        return {"status": "no_answer", "quote": "", "reason": "empty transcript", "rule": True, "latency_s": 0.0, "cost_usd": 0.0}
    low = text.lower()
    for w in cfg["help_words"]:
        if w.lower() in low:
            return {"status": "needs_help", "quote": w, "reason": f"help word '{w}' heard", "rule": True, "latency_s": 0.0, "cost_usd": 0.0}
    return _backend(backend).classify_reply(text, question)


def ask_once(cfg: dict, audio_out: str | None, rec_dir: Path, tag: str, log=print, backend=None) -> dict:
    """Speak the prompt, record, transcribe, classify. Returns a dict with all timings."""
    from remy.camera.audio_out import speak
    lang = cfg["language"]
    question = question_text(cfg)
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
    cl = classify(tr.get("transcript", ""), cfg, question, backend)
    log(f"check-in: classified {cl['status']} ({'rule' if cl.get('rule') else 'spark'}, {cl.get('latency_s')}s): {cl.get('reason')}")
    return {"question": question, "speak": sp, "record": r, "wav": str(rec), "transcript": tr, "classification": cl,
            "status": cl["status"], "total_s": round(time.time() - t0, 2)}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="remy test-checkin", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--make-prompts", action="store_true", help="(re)generate the prompt wavs with Kokoro")
    p.add_argument("--classify", metavar="TEXT", help="classify this transcript and exit")
    p.add_argument("--run", action="store_true", help="one full round: speak, record, transcribe, classify")
    p.add_argument("--audio-out", default=None, help="camera | laptop (default: config audio.output)")
    a = p.parse_args(argv)
    if not (a.make_prompts or a.classify is not None or a.run):
        p.print_help(); return 2
    cfg = load_config()["checkin"]
    if a.make_prompts:
        make_prompts(cfg)
    if a.classify is not None:
        print(json.dumps(classify(a.classify, cfg, question_text(cfg)), ensure_ascii=False))
    if a.run:
        d = EVENTS / "checkin_manual"; d.mkdir(parents=True, exist_ok=True)
        print(json.dumps(ask_once(cfg, a.audio_out, d, "manual"), ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
