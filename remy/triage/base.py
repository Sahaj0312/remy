"""The triage interface.

A backend does two things with a language model:
  triage_fall(...)    -> how serious does the confirmed fall look, and a one-line note for the family
  classify_reply(...) -> is the spoken reply to "are you okay?" an okay, a call for help, unclear, or nothing

To add a backend (for example Glimmer):
  1. create remy/triage/glimmer.py with `class GlimmerTriage(TriageBackend)` that implements both methods
     and returns the same dict shapes as remy.triage.spark.SparkTriage;
  2. add it to BACKENDS in remy/triage/__init__.py and a "glimmer": {...} section to config.json;
  3. set "triage": {"backend": "glimmer"} in config.json.
Nothing else in the pipeline needs to change.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class TriageBackend(ABC):
    name: str = "base"

    @abstractmethod
    def triage_fall(self, best: dict, fps: float, n_frames: int, src: Path, event_time: str, mode: str = "video") -> dict:
        """Judge a confirmed fall.

        best: the scorer's event dict (start_i, end_i, drop_rel, drop_speed_rel_max, time_to_floor_s, ar_post,
              shape_ok, hold, no_getup_observed_s). src: the clip the event was scored on. event_time: wall-clock
              text for the note ("5:45pm"). mode: "video" (upload a burst), "inline" (base64 burst) or "frames".
        Returns a dict with at least:
          verdict:    {"urgency": "high"|"medium"|"low", "summary_for_family": str, "what_happened": str,
                       "is_lying_still": bool}
          latency_s, cost_usd, usage, model, mode
        """

    @abstractmethod
    def classify_reply(self, transcript: str, question: str) -> dict:
        """Classify what was heard after the check-in question.

        Returns {"status": "okay"|"needs_help"|"unclear"|"no_answer", "quote": str, "reason": str,
                 "rule": False, "latency_s": float, "cost_usd": float}.
        The rule-based part (empty transcript, help words) runs before this in remy.checkin.classify.
        """
