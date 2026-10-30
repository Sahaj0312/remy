"""Floor / rug / sofa / bed zones from zones.json (polygons in the setup frame's pixel space)."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

FLOOR_LIKE = ("floor", "rug", "carpet")
FURNITURE = ("sofa", "couch", "bed", "mattress")


class Zones:
    def __init__(self, path: Path, frame_w: int, frame_h: int):
        d = json.load(open(path))
        sx, sy = frame_w / d["width"], frame_h / d["height"]
        self.polys: dict[str, list[np.ndarray]] = {}
        for name, items in d["zones"].items():
            self.polys[name] = [np.array([[x * sx, y * sy] for x, y in z["polygon"]], np.float32) for z in items]
        self.w, self.h = frame_w, frame_h

    def inside(self, name: str, x: float, y: float) -> bool:
        return any(cv2.pointPolygonTest(p, (float(x), float(y)), False) >= 0 for p in self.polys.get(name, []))

    def zone_of(self, x: float, y: float) -> str | None:
        """Furniture wins over floor (a person on the sofa also overlaps the floor polygon edge)."""
        for name in self.polys:
            if name in FURNITURE and self.inside(name, x, y):
                return name
        for name in self.polys:
            if name in FLOOR_LIKE and self.inside(name, x, y):
                return name
        return None

    def classify_box(self, box: list[float]) -> dict:
        """Where is the person? Uses the box bottom-center (contact point) and the box center."""
        x1, y1, x2, y2 = box
        foot = ((x1 + x2) / 2, y2)
        mid = ((x1 + x2) / 2, (y1 + y2) / 2)
        zf, zm = self.zone_of(*foot), self.zone_of(*mid)
        on_floor = zf in FLOOR_LIKE and zm not in FURNITURE
        on_furniture = zf in FURNITURE or zm in FURNITURE
        return {"foot_zone": zf, "mid_zone": zm, "on_floor": bool(on_floor), "on_furniture": bool(on_furniture)}

    def draw(self, img):
        colors = {"floor": (80, 200, 80), "rug": (80, 200, 80), "sofa": (40, 160, 255), "bed": (230, 120, 40)}
        for name, polys in self.polys.items():
            for p in polys:
                cv2.polylines(img, [p.astype(np.int32)], True, colors.get(name, (0, 165, 255)), 1)
        return img
