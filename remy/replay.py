"""Replay a recorded event folder without any API key: re-score the stored SAM 3.1 track with the current
config, then print the timeline, the triage verdict, the check-in rounds and the alert that went out.

  remy replay examples/event_gmdcsa_S1_Fall_12
  remy replay events/20261031_174554_test5_drop

The re-score is deterministic, so a judge can check that the stored verdict follows from the stored track.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from remy.common import load_config, read_json
from remy.vision.score import score_track


def replay(folder: Path, show_log: bool = True) -> int:
    ev = read_json(folder / "event.json")
    print(f"== {folder.name}: kind={ev.get('kind')} event={ev.get('event')} time={ev.get('time')}")
    if show_log and (folder / "log.txt").exists():
        print("\n-- timeline (log.txt)")
        print((folder / "log.txt").read_text().rstrip())
    if (folder / "sam_track.json").exists():
        track = read_json(folder / "sam_track.json")
        res = score_track(track, load_config()["score_v2"], "v2")
        stored = read_json(folder / "score.json") if (folder / "score.json").exists() else None
        b = res.get("best") or {}
        print("\n-- score v2 re-run on the stored SAM 3.1 track (no API)")
        print(f"   flag={res['flag']} hold={res.get('hold')} drop={b.get('drop_rel')} bl at {b.get('drop_speed_rel_max')} bl/s, "
              f"ends ar={b.get('ar_post')}, coverage={res['coverage']}")
        if stored is not None:
            same = bool(stored.get("flag")) == bool(res["flag"]) and (stored.get("best") or {}).get("drop_rel") == b.get("drop_rel")
            print(f"   matches the stored score.json: {'yes' if same else 'NO (config changed since the event?)'}")
    if ev.get("triage"):
        t = ev["triage"]
        print(f"\n-- triage: urgency={t.get('urgency')} lying_still={t.get('is_lying_still')}")
        print(f"   family note: {t.get('summary_for_family')}")
        print(f"   what happened: {t.get('what_happened')}")
    for i, r in enumerate(ev.get("checkin_rounds") or [], 1):
        tr = (r.get("transcript") or {}).get("transcript", "")
        cl = r.get("classification") or {}
        print(f"\n-- check-in round {i}: asked {r.get('question')!r} on {(r.get('speak') or {}).get('device')}")
        print(f"   heard: {tr!r}  ->  {r.get('status')} ({'rule' if cl.get('rule') else cl.get('backend', 'model')}: {cl.get('reason')})")
    if ev.get("outcome"):
        print(f"\n-- outcome: {ev['outcome']}")
    al = ev.get("alert")
    if al:
        print(f"-- alert: {al.get('kind')}{' (' + al['reason'] + ')' if al.get('reason') else ''}"
              f"{' [dry run]' if any((r or {}).get('dry') for r in al.get('responses') or []) else ''}")
        if al.get("text"):
            print("   " + al["text"].replace("\n", "\n   "))
    if ev.get("timings"):
        keys = ("clip_written", "sam_done", "event", "checkin1_done", "checkin2_done", "triage_done", "alert_sent")
        print("\n-- seconds after the drop: " + ", ".join(f"{k} {ev['timings'][k]}" for k in keys if k in ev["timings"]))
    print(f"-- cost: ${ev.get('cost_usd', 0):.4f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="remy replay", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folder", help="an event folder (events/<id> or examples/<id>)")
    p.add_argument("--no-log", action="store_true", help="skip the timeline")
    a = p.parse_args(argv)
    folder = Path(a.folder)
    if not (folder / "event.json").exists():
        raise SystemExit(f"{folder} has no event.json")
    return replay(folder, show_log=not a.no_log)


if __name__ == "__main__":
    raise SystemExit(main())
