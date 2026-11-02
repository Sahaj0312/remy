"""Running API cost ledger: <root>/costs.jsonl, one line per billed call."""
from __future__ import annotations

import json
import time

from remy.common import ROOT

LEDGER = ROOT / "costs.jsonl"


def add(kind: str, usd: float, note: str = "") -> None:
    with open(LEDGER, "a") as f:
        f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "kind": kind, "usd": round(float(usd), 5), "note": note}) + "\n")


def total(kind: str | None = None) -> float:
    if not LEDGER.exists():
        return 0.0
    return sum(json.loads(l)["usd"] for l in LEDGER.read_text().splitlines() if l.strip() and (kind is None or json.loads(l)["kind"] == kind))


def summary() -> dict:
    out: dict[str, float] = {}
    if LEDGER.exists():
        for l in LEDGER.read_text().splitlines():
            if l.strip():
                d = json.loads(l); out[d["kind"]] = out.get(d["kind"], 0.0) + d["usd"]
    out["TOTAL"] = sum(out.values())
    return {k: round(v, 4) for k, v in out.items()}


def main(argv: list[str] | None = None) -> int:
    print(json.dumps(summary(), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
