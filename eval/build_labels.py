"""Build eval/labels.csv (clip, label, source, src_file, type, lighting, description) for the eval set.

Types
  falls   : fall:standing  fall:chair  fall:bed  fall:floor-sit  fall:slow  fall:onto-bed
  non-fall: adl:lie-bed  adl:lie-floor  adl:sit  adl:bend-pick  adl:exercise-floor  adl:exercise-stand
            adl:get-up  adl:sit-walk
UR Fall has no per-sequence text, so its types are inferred from the published depth features
(urfall-cam0-*.csv): head height H at the start (standing vs chair), lying-label frames and
minimum head height (lying on floor vs on bed/sofa), minimum height/width ratio (bend/crouch).
"""
from __future__ import annotations

import csv
import statistics
from collections import defaultdict

from remy.common import EVAL, LABELS_PATH

RAW = EVAL / "raw"

UR_FALLS = ["01", "02", "04", "05", "07", "08", "10", "11", "13", "14", "16", "17", "19", "20", "22", "23", "25", "28"]
UR_ADLS = ["01", "07", "08", "10", "11", "18", "19", "20", "21", "22", "23", "25", "27", "30", "31", "32", "33", "34",
           "35", "36", "37", "38", "39", "40"]

GM_FALLS = {  # (subject, nn): type
    (1, "05"): "fall:standing", (1, "09"): "fall:standing", (1, "12"): "fall:standing", (1, "15"): "fall:chair",
    (2, "02"): "fall:standing", (2, "04"): "fall:standing", (2, "08"): "fall:floor-sit", (2, "11"): "fall:bed",
    (3, "05"): "fall:slow", (4, "10"): "fall:chair",
    (4, "02"): "fall:bed", (4, "06"): "fall:bed", (4, "12"): "fall:bed", (4, "15"): "fall:bed",
    (3, "12"): "fall:bed", (3, "18"): "fall:bed", (3, "09"): "fall:floor-sit", (3, "13"): "fall:floor-sit",
    (3, "17"): "fall:onto-bed", (3, "16"): "fall:standing", (2, "03"): "fall:onto-bed", (2, "10"): "fall:standing",
}
GM_ADLS = {
    (1, "01"): "adl:lie-bed", (2, "03"): "adl:lie-bed", (2, "08"): "adl:exercise-stand", (2, "12"): "adl:lie-bed",
    (2, "22"): "adl:exercise-floor", (3, "08"): "adl:lie-floor", (3, "09"): "adl:get-up", (4, "05"): "adl:exercise-floor",
    (4, "10"): "adl:lie-bed", (4, "15"): "adl:lie-bed",
    (1, "02"): "adl:lie-bed", (1, "03"): "adl:lie-bed", (1, "04"): "adl:get-up", (1, "15"): "adl:bend-pick",
    (1, "16"): "adl:bend-pick", (2, "02"): "adl:bend-pick", (2, "04"): "adl:exercise-stand", (2, "05"): "adl:get-up",
    (2, "06"): "adl:bend-pick", (2, "09"): "adl:sit", (2, "13"): "adl:sit", (2, "16"): "adl:lie-bed", (2, "20"): "adl:sit",
    (3, "03"): "adl:get-up", (3, "04"): "adl:bend-pick", (3, "05"): "adl:exercise-stand", (3, "06"): "adl:sit",
    (3, "07"): "adl:exercise-stand", (3, "19"): "adl:bend-pick", (4, "06"): "adl:exercise-floor",
    (4, "07"): "adl:exercise-floor", (4, "08"): "adl:get-up", (4, "11"): "adl:sit", (4, "14"): "adl:bend-pick",
    (4, "16"): "adl:lie-bed", (4, "17"): "adl:bend-pick",
}


def gm_meta():
    """(subject, 'Fall'|'ADL', nn) -> (description, lighting) from the dataset CSVs."""
    out = {}
    for s in (1, 2, 3, 4):
        for kind in ("Fall", "ADL"):
            p = RAW / "gmdcsa" / f"gm_S{s}_{kind}.csv"
            with open(p, newline="", encoding="utf-8", errors="replace") as f:
                for row in csv.reader(f):
                    if not row or not row[0].endswith(".mp4"):
                        continue
                    nn = row[0].replace(".mp4", "").strip()
                    when = row[2].strip().lower() if len(row) > 2 else ""
                    lighting = "lights-off" if "off" in when else ("night" if "night" in when else "day")
                    desc = row[4].strip() if len(row) > 4 else ""
                    out[(s, kind, nn)] = (desc, lighting)
    return out


def ur_feats(path):
    per = defaultdict(list)
    with open(path, newline="") as f:
        for row in csv.reader(f):
            per[row[0]].append(row)
    return per


def main():
    rows = []
    falls = ur_feats(RAW / "urfall" / "urfall-cam0-falls.csv")
    adls = ur_feats(RAW / "urfall" / "urfall-cam0-adls.csv")
    for n in UR_FALLS:
        rs = falls[f"fall-{n}"]
        h0 = statistics.median(float(r[8]) for r in rs[:10])
        typ = "fall:standing" if h0 > 1500 else "fall:chair"
        rows.append(dict(clip=f"urfall_fall-{n}", label="fall", source="urfall", src_file=f"fall-{n}-cam0-rgb.zip",
                         type=typ, lighting="day",
                         description=f"fall (inferred {typ.split(':')[1]}: start head height {h0:.0f} mm)"))
    for n in UR_ADLS:
        rs = adls[f"adl-{n}"]
        lying = sum(r[2] == "1" for r in rs)
        min_h = min(float(r[8]) for r in rs)
        min_hw = min(float(r[3]) for r in rs)
        if lying:
            typ = "adl:lie-floor" if min_h < 400 else "adl:lie-bed"
            desc = f"lying down on purpose (inferred: {lying} lying frames, min head height {min_h:.0f} mm)"
        elif min_hw < 1.0:
            typ = "adl:bend-pick"
            desc = f"crouch/bend/pick up (inferred: min height/width ratio {min_hw:.2f})"
        else:
            typ = "adl:sit-walk"
            desc = f"walking/sitting (inferred: min height/width ratio {min_hw:.2f}, no lying frames)"
        rows.append(dict(clip=f"urfall_adl-{n}", label="adl", source="urfall", src_file=f"adl-{n}-cam0-rgb.zip",
                         type=typ, lighting="day", description=desc))
    meta = gm_meta()
    for (s, nn), typ in GM_FALLS.items():
        desc, light = meta[(s, "Fall", nn)]
        rows.append(dict(clip=f"gmdcsa_S{s}_Fall_{nn}", label="fall", source="gmdcsa24", src_file=f"gm_S{s}_Fall_{nn}.mp4",
                         type=typ, lighting=light, description=desc))
    for (s, nn), typ in GM_ADLS.items():
        desc, light = meta[(s, "ADL", nn)]
        rows.append(dict(clip=f"gmdcsa_S{s}_ADL_{nn}", label="adl", source="gmdcsa24", src_file=f"gm_S{s}_ADL_{nn}.mp4",
                         type=typ, lighting=light, description=desc))
    with open(LABELS_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["clip", "label", "source", "src_file", "type", "lighting", "description"])
        w.writeheader(); w.writerows(rows)
    from collections import Counter
    print(len(rows), "clips;", Counter(r["label"] for r in rows))
    for t, c in sorted(Counter(r["type"] for r in rows).items()):
        print(f"  {t:22s} {c}")
    print("lighting:", Counter(r["lighting"] for r in rows))


if __name__ == "__main__":
    main()
