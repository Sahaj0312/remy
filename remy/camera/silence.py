"""Stop the check-in recording when she has answered, instead of always waiting the full window.

A small energy detector on 50 ms chunks of 16 kHz mono PCM, tuned on the real recordings from the Tapo mic:
the camera gates silence to about -72 dBFS, movement and rustle sit at -65 to -50, and speech from the floor
peaks at -30 to -20. So:
  speech starts   : `speech_db` or louder for at least `speech_min_s`, or any single chunk at `loud_db` or louder
                    (a short "yeah" from the floor is 150 ms at -25 dB; rustle never reaches -30)
  recording stops : after speech, `silence_s` with every chunk below `silence_db`, and never before `min_s`
  no speech at all: the recording runs to `max_s` (the caller then asks again or escalates)
Pure Python, no model, so it adds nothing to the critical path. SilenceStop.feed() is incremental and is what
the live recorder calls; cut_point() runs it over a whole file for the offline check in eval/silence_stop.py.
"""
from __future__ import annotations

import math

import numpy as np

DEFAULTS = {"speech_db": -40.0, "loud_db": -30.0, "silence_db": -47.0, "speech_min_s": 0.2, "silence_s": 2.0, "min_s": 4.0, "max_s": 15.0}


def chunk_db(chunk: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2))) if len(chunk) else 0.0
    return 20 * math.log10(max(rms, 1e-6))


class SilenceStop:
    def __init__(self, chunk_s: float = 0.05, **params):
        self.p = {**DEFAULTS, **{k: v for k, v in params.items() if v is not None}}
        self.chunk_s = chunk_s
        self.t = 0.0
        self.speech_run = 0.0      # seconds of consecutive loud chunks
        self.speech_at = None      # when speech was first confirmed
        self.quiet_run = 0.0       # seconds of consecutive quiet chunks since the last speech
        self.peak_db = -120.0

    def feed(self, chunk: np.ndarray) -> bool:
        """Feed one chunk (float32 in [-1, 1] or int16). Returns True when the recording should stop."""
        db = chunk_db(chunk)
        self.t += self.chunk_s
        self.peak_db = max(self.peak_db, db)
        if db >= self.p["speech_db"]:
            self.speech_run += self.chunk_s
            self.quiet_run = 0.0
            if self.speech_at is None and (self.speech_run >= self.p["speech_min_s"] - 1e-9 or db >= self.p["loud_db"]):
                self.speech_at = self.t - self.speech_run
        else:
            self.speech_run = 0.0
            if db < self.p["silence_db"]:
                self.quiet_run += self.chunk_s
            else:
                self.quiet_run = 0.0          # rustle: not speech, but not silence either; keep listening
        if self.t >= self.p["max_s"] - 1e-9:
            return True
        return (self.speech_at is not None and self.quiet_run >= self.p["silence_s"] - 1e-9
                and self.t >= self.p["min_s"] - 1e-9)

    def summary(self) -> dict:
        return {"duration_s": round(self.t, 2), "speech_at_s": None if self.speech_at is None else round(self.speech_at, 2),
                "stopped_early": self.t < self.p["max_s"] - 1e-9, "peak_db": round(self.peak_db, 1)}


def cut_point(samples: np.ndarray, rate: int = 16000, chunk_s: float = 0.05, **params) -> dict:
    """Run the detector over a whole recording: where it would have stopped, and whether speech was seen."""
    det = SilenceStop(chunk_s, **params)
    hop = int(rate * chunk_s)
    for i in range(0, len(samples) - hop + 1, hop):
        if det.feed(samples[i:i + hop]):
            break
    return det.summary()
