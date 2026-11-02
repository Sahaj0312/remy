"""Triage: a language model judges how serious a confirmed fall looks, writes the family note, and
classifies the spoken reply to the check-in. One backend at a time, chosen by config "triage.backend".

Two backends: "spark" (Muse Spark 1.3, Meta Model API) and "spark_open" (Muse Spark Open, open weights,
served by Fireworks). See remy.triage.base for the interface a new backend such as Glimmer has to implement.
"""
from __future__ import annotations

from remy.common import env_get
from remy.triage.base import TriageBackend

BACKENDS = {
    "spark": "remy.triage.spark:SparkTriage",              # Muse Spark 1.3 through the Meta Model API
    "spark_open": "remy.triage.spark_open:SparkOpenTriage",  # Muse Spark Open (open weights) through a partner endpoint
}


_warned: set[str] = set()


def backend_name(cfg: dict, role: str = "triage") -> str:
    """Which backend config asks for: triage.backend judges the fall, triage.reply_backend (default: the same)
    classifies the spoken reply. A partner-hosted backend without its key falls back to spark with a warning."""
    t = cfg.get("triage") or {}
    name = t.get("backend", "spark") if role == "triage" else t.get("reply_backend") or t.get("backend", "spark")
    if name == "spark_open":
        key_env = (cfg.get("spark_open") or {}).get("api_key_env", "FIREWORKS_API_KEY")
        if not env_get(key_env, required=False):
            if key_env not in _warned:
                _warned.add(key_env)
                print(f"[triage] {key_env} is not set: using spark instead of spark_open (see .env.example)", flush=True)
            name = "spark"
    return name


def get_backend(cfg: dict, role: str = "triage", **kwargs) -> TriageBackend:
    """Build the backend for `role` ("triage" or "reply") from the full config dict. kwargs go to its constructor."""
    import importlib
    name = backend_name(cfg, role)
    try:
        mod_name, cls_name = BACKENDS[name].split(":")
    except KeyError:
        raise SystemExit(f"unknown triage backend {name!r}; known: {sorted(BACKENDS)}") from None
    cls = getattr(importlib.import_module(mod_name), cls_name)
    section = cfg.get(name)
    if section is None and name == "spark_open":
        section = {}                      # all defaults live in the backend
    return cls(section if section is not None else cfg["spark"], **kwargs)
