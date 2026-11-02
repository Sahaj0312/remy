"""Muse Spark Open (Meta's open-weight reasoning model) as the triage backend, served by a partner
(Fireworks by default) through an OpenAI-compatible chat completions endpoint.

Same prompts and the same JSON shapes as the hosted Spark backend (remy.triage.spark), so the two can be
compared clip for clip. Differences: the burst goes as JPEG frames (the partner endpoint takes images, not
video), reasoning is capped with reasoning_effort, and JSON comes back through response_format json_object.

config.json "spark_open": model, base_url, api_key_env, frames_fps, jpeg_quality, burst_before_s, burst_after_s,
reasoning_effort, max_output_tokens, usd_per_input_token, usd_per_output_token.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from openai import OpenAI

from remy import costs
from remy.common import env_get
from remy.triage.base import TriageBackend
from remy.triage.spark import CLASSIFY_SYSTEM, SYSTEM, burst_window, grab_frames, numbers_text, parse_json

DEFAULTS = {
    "model": "accounts/fireworks/models/muse-spark-open",
    "base_url": "https://api.fireworks.ai/inference/v1",
    "api_key_env": "FIREWORKS_API_KEY",
    "frames_fps": 5,
    "jpeg_quality": 80,
    "burst_before_s": 1.5,
    "burst_after_s": 3.0,
    "reasoning_effort": "low",
    "max_output_tokens": 2000,
    # Fireworks serverless list price for muse-spark-open (checked 2026-11-02)
    "usd_per_input_token": 1.25 / 1_000_000,
    "usd_per_output_token": 4.25 / 1_000_000,
}

URGENCIES = ("high", "medium", "low")
STATUSES = ("okay", "needs_help", "unclear", "no_answer")


class SparkOpenTriage(TriageBackend):
    name = "spark_open"
    label = "Muse Spark Open"

    def __init__(self, cfg: dict, timeout: float = 300, client: OpenAI | None = None):
        self.cfg = {**DEFAULTS, **(cfg or {})}
        self.client = client or OpenAI(base_url=self.cfg["base_url"], api_key=env_get(self.cfg["api_key_env"]), timeout=timeout)

    # ---- one chat completion with JSON out
    def _chat(self, system: str, content: list[dict]) -> tuple[dict, float, object]:
        t0 = time.perf_counter()
        resp = self.client.chat.completions.create(
            model=self.cfg["model"],
            messages=[{"role": "system", "content": system}, {"role": "user", "content": content}],
            response_format={"type": "json_object"},
            reasoning_effort=self.cfg["reasoning_effort"],
            max_tokens=self.cfg["max_output_tokens"])
        lat = time.perf_counter() - t0
        text = resp.choices[0].message.content or ""
        if not text.strip():                                # reasoning ate the cap: one retry with twice the room
            resp = self.client.chat.completions.create(
                model=self.cfg["model"],
                messages=[{"role": "system", "content": system}, {"role": "user", "content": content}],
                response_format={"type": "json_object"},
                reasoning_effort=self.cfg["reasoning_effort"],
                max_tokens=self.cfg["max_output_tokens"] * 2)
            lat = time.perf_counter() - t0
            text = resp.choices[0].message.content or ""
        return parse_json(text), lat, resp

    def _usage(self, resp) -> tuple[dict, float]:
        u = resp.usage
        details = getattr(u, "completion_tokens_details", None)
        usage = {"input_tokens": u.prompt_tokens, "output_tokens": u.completion_tokens,
                 "reasoning_tokens": getattr(details, "reasoning_tokens", None)}
        usd = u.prompt_tokens * self.cfg["usd_per_input_token"] + u.completion_tokens * self.cfg["usd_per_output_token"]
        return usage, usd

    # ---- the interface
    def triage_fall(self, best: dict, fps: float, n_frames: int, src: Path, event_time: str, mode: str = "frames") -> dict:
        cfg = self.cfg
        s, e = burst_window(best, n_frames, fps, cfg)
        step = max(1, int(round(fps / cfg["frames_fps"])))
        idx = list(range(s, e + 1, step))
        frames = grab_frames(src, idx, cfg["jpeg_quality"])
        content = [{"type": "text", "text": f"{len(idx)} frames at {fps / step:g} fps around the fall. " + numbers_text(best, event_time)}]
        for i in idx:
            content.append({"type": "text", "text": f"t={i / fps:.1f}s"})
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frames[i]}"}})
        content.append({"type": "text", "text": 'Return JSON: {"urgency": "high"|"medium"|"low", '
                        '"summary_for_family": "...", "what_happened": "...", "is_lying_still": true|false}'})
        verdict, lat, resp = self._chat(SYSTEM, content)
        if verdict.get("urgency") not in URGENCIES or not isinstance(verdict.get("is_lying_still"), bool):
            raise ValueError(f"bad triage JSON from {self.name}: {json.dumps(verdict)[:200]}")
        usage, usd = self._usage(resp)
        return {"model": cfg["model"], "backend": self.name, "mode": "frames", "file_id": None, "burst_frames": [s, e],
                "burst_s": round((e - s + 1) / fps, 2), "n_images": len(idx), "event_time": event_time,
                "numbers": {k: best[k] for k in ("drop_rel", "drop_speed_rel_max", "time_to_floor_s", "ar_post", "hold", "no_getup_observed_s")},
                "raw_text": resp.choices[0].message.content, "status": "completed", "latency_s": round(lat, 2),
                "usage": usage, "cost_usd": round(usd, 5), "response_id": resp.id, "verdict": verdict}

    def classify_reply(self, transcript: str, question: str) -> dict:
        text = (transcript or "").strip()
        content = [{"type": "text", "text": f'Question asked: "{question}"\nTranscript of the next 15 seconds: "{text}"\n'
                    'Return JSON: {"status": "okay"|"needs_help"|"unclear"|"no_answer", "quote": "...", "reason": "..."}'}]
        try:
            out, lat, resp = self._chat(CLASSIFY_SYSTEM, content)
        except Exception as e:
            return {"status": "unclear", "quote": "", "reason": f"{self.name} error: {type(e).__name__}", "rule": False, "latency_s": 0.0, "cost_usd": 0.0}
        usage, usd = self._usage(resp)
        costs.add("spark-open-classify", usd, text[:40])
        if out.get("status") not in STATUSES:
            out = {"status": "unclear", "quote": "", "reason": f"bad JSON: {json.dumps(out)[:80]}"}
        out.setdefault("quote", ""); out.setdefault("reason", "")
        out.update(rule=False, latency_s=round(lat, 2), cost_usd=round(usd, 5), usage=usage)
        return out
