"""The `remy` command.

  remy run --source tapo                 watch the camera and handle falls (see `remy run -h` for every flag)
  remy run --source eval/clips/x.mp4 --realtime --no-zones --audio-out laptop --alerts-dry-run
  remy setup-zones --stream 2            build zones.json (floor / rug / sofa / bed) from one camera frame
  remy test-checkin --run                one spoken check-in round; --classify TEXT; --make-prompts
  remy test-alert --test                 send a test WhatsApp message to the family list; --dry-run
  remy costs                             print the API cost ledger
"""
from __future__ import annotations

import importlib
import sys

COMMANDS = {
    "run": ("remy.monitor", "watch a camera or a clip and handle falls end to end"),
    "setup-zones": ("remy.vision.zones_setup", "build zones.json from one camera frame with SAM 3.1"),
    "test-checkin": ("remy.checkin", "speak the prompt, record, transcribe, classify (or --classify TEXT)"),
    "test-alert": ("remy.alerts.whatsapp", "send a test WhatsApp message to the family list"),
    "costs": ("remy.costs", "print the API cost ledger"),
}


def usage() -> str:
    width = max(len(c) for c in COMMANDS)
    lines = ["usage: remy <command> [options]", "", "commands:"]
    lines += [f"  {c.ljust(width)}  {d}" for c, (_, d) in COMMANDS.items()]
    lines += ["", "`remy <command> -h` lists the options of a command."]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(usage()); return 0
    if argv[0] in ("-V", "--version"):
        from remy import __version__
        print(f"remy {__version__}"); return 0
    cmd, rest = argv[0], argv[1:]
    if cmd not in COMMANDS:
        print(f"remy: unknown command {cmd!r}\n\n{usage()}", file=sys.stderr); return 2
    module = importlib.import_module(COMMANDS[cmd][0])
    return int(module.main(rest) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
