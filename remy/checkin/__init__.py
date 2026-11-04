"""Spoken check-in: say "<name>, are you okay?" (Kokoro, on the camera speaker or the laptop), record the
camera microphone for record_s seconds, transcribe with Muse Voice Transcribe, classify the reply.

  okay        a clear statement of being fine, in any wording or language (judged by the triage backend)
  needs_help  a help word (config checkin.help_words) or a statement of being hurt / unable to get up
  unclear     words that do not answer the question (TV, a question back, mumbling), or speech-level sound
              the transcriber could not turn into words (she tried to answer; we could not understand)
  no_answer   nothing was heard

The recording stops on its own once she has answered (checkin.stop_on_silence; remy.camera.silence), so a
quick "yeah" no longer waits out the full window; silence runs to checkin.record_s and gets one more prompt.

  remy test-checkin --make-prompts            # generate every spoken line in config (checkin.lines) with Kokoro
  remy test-checkin --import ask.wav --as ask # replace one line with a better recording (ElevenLabs etc.)
  remy test-checkin --say ask                 # play one line on the camera (or --audio-out laptop)
  remy test-checkin --classify "I'm fine"     # classification only
  remy test-checkin --run --audio-out laptop  # full round: speak, record, transcribe, classify
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from remy.camera.audio_in import MicStream, record_mic, record_until_silence, reply_from_file
from remy.checkin.transcribe import transcribe_wav
from remy.checkin.tts import LINE_KEYS, import_prompt, line_text, make_prompts, prompt_path
from remy.common import EVENTS, load_config

__all__ = ["ask_once", "classify", "make_prompts", "prompt_path", "record_mic", "transcribe_wav", "question_text", "say_line"]


def question_text(cfg: dict, key: str = "ask") -> str:
    return line_text(cfg, key)


def say_line(cfg: dict, key: str, audio_out: str | None, log=print, wait: bool = True) -> dict:
    """Play one spoken line (checkin.lines) on the camera or laptop; generates it with Kokoro if missing."""
    from remy.camera.audio_out import speak
    lang = cfg["language"]
    wav = prompt_path(key, lang)
    if not wav.exists():
        make_prompts(cfg, [key])
        wav = prompt_path(key, lang)
    sp = speak(wav, audio_out, wait=wait)
    log(f"check-in: said {key} ({line_text(cfg, key)!r}) on {sp['device']} ({sp['took_s']}s)")
    return sp


def _backend(backend=None):
    if backend is None:
        from remy.triage import get_backend
        backend = get_backend(load_config(), role="reply", timeout=60)
    return backend


def classify(transcript: str, cfg: dict, question: str, backend=None, heard_speech: bool = False) -> dict:
    """Rules first (empty -> no_answer, or unclear if speech-level sound was heard; help word -> needs_help),
    then the triage backend."""
    text = (transcript or "").strip()
    if not text:
        if heard_speech and cfg.get("escalate_on_unheard_speech", True):
            return {"status": "unclear", "quote": "", "reason": "speech-level sound heard but no words understood", "rule": True,
                    "latency_s": 0.0, "cost_usd": 0.0}
        return {"status": "no_answer", "quote": "", "reason": "empty transcript", "rule": True, "latency_s": 0.0, "cost_usd": 0.0}
    low = text.lower()
    for w in cfg["help_words"]:
        if w.lower() in low:
            return {"status": "needs_help", "quote": w, "reason": f"help word '{w}' heard", "rule": True, "latency_s": 0.0, "cost_usd": 0.0}
    be = _backend(backend)
    out = be.classify_reply(text, question)
    out.setdefault("backend", getattr(be, "label", be.__class__.__name__))
    return out


def ask_once(cfg: dict, audio_out: str | None, rec_dir: Path, tag: str, log=print, backend=None, on_recorded=None,
             line: str = "ask", mic: MicStream | None = None, reply: Path | None = None) -> dict:
    """Speak one of the question lines (ask, ask_again, ask_floor, recheck), record, transcribe, classify.
    Returns a dict with all timings. on_recorded(wav_path, rec_wall_start, record_result) is called right after
    the recording ends, before transcription, so the caller can cut the frames of that moment while they are
    still in its buffer. `mic` is an open MicStream to record from (the monitor's room-audio stream); it is
    left open. Without one the round opens its own. The camera serves three RTSP sessions at a time, so a
    round must not add a connection when one is already running. `reply` is a recorded answer to use instead of
    any microphone (remy run --reply); the camera is not touched."""
    from remy.camera.audio_out import speak
    lang = cfg["language"]
    question = question_text(cfg, line)
    wav = prompt_path(line, lang)
    if not wav.exists():
        make_prompts(cfg, [line])
        wav = prompt_path(line, lang)
    t0 = time.time()
    stop_on_silence = cfg.get("stop_on_silence", True)
    if reply is not None:
        mic = None
    elif mic is not None and not mic.alive:
        log(f"check-in: the room audio stream is down ({mic.err or 'ended'}); opening the mic for this round")
        mic = None
    shared = mic is not None
    if reply is None and mic is None and stop_on_silence:
        try:
            mic = MicStream()                       # connect while the prompt plays
        except Exception as e:
            log(f"check-in: mic stream failed to open ({type(e).__name__}); falling back to a fixed recording")
    sp = speak(wav, audio_out, wait=True)
    log(f"check-in: spoke '{question}' on {sp['device']} ({sp['took_s']}s)" + ("" if sp.get("ok", True) else f", FAILED: {sp.get('detail')}"))
    rec = rec_dir / f"checkin_{tag}_{datetime.now().strftime('%H%M%S')}.wav"
    rec_wall = time.time() + 0.5
    r = None
    if reply is not None:
        r = reply_from_file(reply, rec, cfg["record_s"], cfg.get("silence"))
        speech = "no speech" if r.get("speech_at_s") is None else f"speech at {r['speech_at_s']}s"
        log(f"check-in: reply from {reply.name}, not a microphone -> {rec.name}: {r['duration_s']}s, {speech}" + ("" if r["ok"] else f", FAILED: {r['err']}"))
    elif mic is not None:
        try:
            # the stream lags the room by 0.2 to 0.45 s, so the prompt's tail still arrives after speak() returns
            r = record_until_silence(rec, cfg["record_s"], cfg.get("silence"), mic=mic, since_wall=rec_wall)
        finally:
            if not shared:
                mic.close()
        if r["ok"]:
            speech = "no speech" if r.get("speech_at_s") is None else f"speech at {r['speech_at_s']}s"
            how = f"stopped {r['duration_s']}s in, {speech}" if r.get("stopped_early") else f"ran the full {r.get('duration_s')}s, {speech}"
            log(f"check-in: mic -> {rec.name}: {how} (peak {r.get('peak_db')} dB, {r['took_s']}s)")
        else:
            log(f"check-in: the mic stream gave nothing ({r['err']}); recording {cfg['record_s']}s the plain way")
            r = None
    if r is None:
        r = record_mic(cfg["record_s"], rec)
        log(f"check-in: recorded {cfg['record_s']}s mic -> {rec.name} ({r['took_s']}s{'' if r['ok'] else ', FAILED: ' + r['err']})")
        r.setdefault("duration_s", cfg["record_s"]); r.setdefault("speech_at_s", None)
    r["rec_wall"] = rec_wall
    if on_recorded is not None and r.get("ok"):
        try:
            on_recorded(rec, rec_wall, r)
        except Exception as e:
            log(f"check-in: answer clip failed: {type(e).__name__}: {e}")
    tr = transcribe_wav(rec) if r["ok"] else {"transcript": "", "error": r["err"], "latency_s": 0}
    log(f"check-in: Muse Voice Transcribe ({tr.get('latency_s')}s): {tr.get('transcript')!r}")
    cl = classify(tr.get("transcript", ""), cfg, question, backend, heard_speech=r.get("speech_at_s") is not None)
    log(f"check-in: classified {cl['status']} ({'rule' if cl.get('rule') else cl.get('backend', 'model')}, {cl.get('latency_s')}s): {cl.get('reason')}")
    return {"question": question, "speak": sp, "record": r, "wav": str(rec), "transcript": tr, "classification": cl,
            "status": cl["status"], "total_s": round(time.time() - t0, 2)}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="remy test-checkin", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--make-prompts", action="store_true", help="generate every line in config checkin.lines with Kokoro")
    p.add_argument("--keep-imported", action="store_true", help="with --make-prompts: do not overwrite lines that already have a file")
    p.add_argument("--import", dest="import_file", metavar="FILE", help="a recording to use for one line (any format)")
    p.add_argument("--as", dest="as_key", metavar="KEY", help=f"which line --import replaces: {', '.join(LINE_KEYS)}")
    p.add_argument("--say", metavar="KEY", help="play one line and exit")
    p.add_argument("--classify", metavar="TEXT", help="classify this transcript and exit")
    p.add_argument("--run", action="store_true", help="one full round: speak, record, transcribe, classify")
    p.add_argument("--audio-out", default=None, help="camera | laptop (default: config audio.output)")
    a = p.parse_args(argv)
    if not (a.make_prompts or a.import_file or a.say or a.classify is not None or a.run):
        p.print_help(); return 2
    cfg = load_config()["checkin"]
    if a.make_prompts:
        make_prompts(cfg, force=not a.keep_imported)
    if a.import_file:
        if not a.as_key:
            raise SystemExit("--import needs --as KEY")
        import_prompt(Path(a.import_file), a.as_key, cfg)
    if a.say:
        say_line(cfg, a.say, a.audio_out)
    if a.classify is not None:
        print(json.dumps(classify(a.classify, cfg, question_text(cfg)), ensure_ascii=False))
    if a.run:
        d = EVENTS / "checkin_manual"; d.mkdir(parents=True, exist_ok=True)
        print(json.dumps(ask_once(cfg, a.audio_out, d, "manual"), ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
