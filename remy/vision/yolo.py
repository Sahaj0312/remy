"""YOLO pose (ultralytics) as the cheap always-on detector.

One detection per person: box, centroid from the confident keypoints (box centre as fallback),
confidence. The live monitor runs this at 5 fps and feeds OnlineTracks; the trigger rule in
remy.vision.score decides when a drop is worth a SAM call.
"""
from __future__ import annotations

from remy.common import ROOT
from remy.vision.tracking import suppress_duplicates


def load_model(ycfg: dict):
    """YOLO weights named in config (yolo.model), looked up at the project root first so the
    run does not depend on the current directory. ultralytics downloads the file if it is missing."""
    from ultralytics import YOLO
    path = ROOT / ycfg["model"]
    return YOLO(str(path) if path.exists() else ycfg["model"])


def detect(model, img, ycfg: dict, tcfg: dict) -> list[dict]:
    res = model.predict(img, verbose=False, conf=ycfg["conf"], imgsz=ycfg["imgsz"], classes=[0])[0]
    dets = []
    if res.boxes is not None and len(res.boxes):
        boxes = res.boxes.xyxy.cpu().numpy(); confs = res.boxes.conf.cpu().numpy()
        for j in range(len(boxes)):
            x1, y1, x2, y2 = boxes[j]; cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            if res.keypoints is not None and len(res.keypoints) > j and res.keypoints.conf is not None:
                kxy = res.keypoints.xy[j].cpu().numpy(); kc = res.keypoints.conf[j].cpu().numpy(); good = kc >= ycfg["kp_conf"]
                if good.sum() >= 3:
                    cx, cy = kxy[good, 0].mean(), kxy[good, 1].mean()
            dets.append({"box": [float(v) for v in (x1, y1, x2, y2)], "cx_px": float(cx), "cy_px": float(cy), "mask_area_px": None,
                         "conf": round(float(confs[j]), 3)})
    return suppress_duplicates(dets, tcfg["dup_max_overlap"])
