"""Triage: a language model judges how serious a confirmed fall looks, writes the family note, and
classifies the spoken reply to the check-in. One backend at a time, chosen by config "triage.backend".

Spark (Muse Spark 1.3) is the backend today. See remy.triage.base for the interface a new backend
such as Glimmer has to implement.
"""
from __future__ import annotations

from remy.triage.base import TriageBackend

BACKENDS = {"spark": "remy.triage.spark:SparkTriage"}


def get_backend(cfg: dict, **kwargs) -> TriageBackend:
    """Build the configured backend from the full config dict. kwargs go to its constructor (e.g. timeout)."""
    import importlib
    name = (cfg.get("triage") or {}).get("backend", "spark")
    try:
        mod_name, cls_name = BACKENDS[name].split(":")
    except KeyError:
        raise SystemExit(f"unknown triage backend {name!r}; known: {sorted(BACKENDS)}") from None
    cls = getattr(importlib.import_module(mod_name), cls_name)
    return cls(cfg.get(name) or cfg["spark"], **kwargs)
