"""Where the silence stop would have cut each real check-in recording, and whether the words survive the cut.
Runs over events/*/checkin_*.wav (private, not in git) and writes eval/SILENCE_STOP.md.

  python eval/silence_stop.py              # cut points only (free)
  python eval/silence_stop.py --transcribe # also transcribe the cut recordings that had speech (a few cents)
"""
from __future__ import annotations

import argparse
import glob
import json
import tempfile
import wave
from pathlib import Path

import numpy as np

from remy.camera.silence import cut_point
from remy.common import EVAL, EVENTS, load_config


def read(f):
    with wave.open(f) as w:
        return w.getframerate(), np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--transcribe", action="store_true")
    a = p.parse_args()
    ck = load_config()["checkin"]; params = {**ck.get("silence", {}), "max_s": ck["record_s"]}
    rows = []
    for f in sorted(glob.glob(str(EVENTS / "*" / "checkin_*.wav"))):
        rate, x = read(f)
        evp = Path(f).parent / "event.json"
        if not evp.exists():                                   # manual test-checkin runs have no event.json
            continue
        ev = json.load(open(evp))
        said = next(((r.get("transcript") or {}).get("transcript", "") for r in ev.get("checkin_rounds") or [] if r.get("wav", "").endswith(Path(f).name)), "")
        r = cut_point(x, rate, **params)
        if r["duration_s"] >= len(x) / rate - 0.1:          # the file ended first: that is not a cut
            r["stopped_early"] = False
        row = {"event": Path(f).parent.name, "wav": Path(f).name, "full_s": round(len(x) / rate, 1), "said": said, **r}
        if a.transcribe and r["speech_at_s"] is not None and r["stopped_early"]:
            from remy.checkin.transcribe import transcribe_wav
            with tempfile.TemporaryDirectory() as tmp:
                cut = Path(tmp) / Path(f).name
                with wave.open(str(cut), "wb") as w:
                    w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
                    w.writeframes((x[: int(r["duration_s"] * rate)] * 32767).astype(np.int16).tobytes())
                row["said_after_cut"] = transcribe_wav(cut, session_prefix="cutcheck").get("transcript", "")
        rows.append(row)
    L = ["# Silence stop on the real check-in recordings\n",
         f"{len(rows)} recordings from the camera mic (not in git). Parameters: {json.dumps(params)}. "
         "`cut` is where the recording would have ended; `said` is what Muse Voice Transcribe returned for the full recording"
         + (", `after cut` for the shortened one." if a.transcribe else "."),
         "", "| event | recording | full | speech at | cut | saved | said" + (" | after cut" if a.transcribe else "") + " |",
         "|---|---|---|---|---|---|---|" + ("---|" if a.transcribe else "")]
    for r in rows:
        cut = f"{r['duration_s']} s" if r["stopped_early"] else "no cut"
        saved = f"{r['full_s'] - r['duration_s']:.1f} s" if r["stopped_early"] else "-"
        L.append(f"| {r['event'][:22]} | {r['wav']} | {r['full_s']} s | {r['speech_at_s'] if r['speech_at_s'] is not None else '-'} | {cut} | {saved} | {r['said']!r}"
                 + (f" | {r.get('said_after_cut', '-')!r}" if a.transcribe else "") + " |")
    spoke = [r for r in rows if r["said"]]
    heard_no_words = [r for r in rows if not r["said"] and r["speech_at_s"] is not None]
    L += ["", f"Recordings with words: {len(spoke)}, all cut early; recordings with silence: {len(rows) - len(spoke) - len(heard_no_words)}, none cut; "
          f"speech-level sound but no words: {len(heard_no_words)} (these now count as unclear and escalate at once)."]
    out = EVAL / "SILENCE_STOP.md"; out.write_text("\n".join(L) + "\n")
    print("\n".join(L)); print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
