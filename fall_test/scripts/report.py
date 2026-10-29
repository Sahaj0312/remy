"""Assemble REPORT.md: per-clip table, overall metrics before/after Spark, latency, cost."""
from __future__ import annotations

import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, ROOT, load_labels, read_json  # noqa: E402


def comp(best):
    if not best:
        return "---"
    return ("D" if best.get("drop_ok") else ".") + ("S" if best.get("shape_ok") else ".") + ("H" if best.get("hold_ok") else ".")


def med(xs):
    xs = [x for x in xs if x is not None and x >= 0]
    return (st.median(xs), st.mean(xs), max(xs)) if xs else (None, None, None)


def fmt(x, nd=1):
    return "-" if x is None else f"{x:.{nd}f}"


def main():
    labels = load_labels()
    rows, sam_docs, spark_docs, yolo_docs = [], [], [], []
    for r in labels:
        n = r["clip"]
        sam_t = OUT / "sam" / f"{n}.json"
        sam_s = OUT / "scores" / "sam" / f"{n}.json"
        spk = OUT / "spark" / "sam" / f"{n}.json"
        yl_s = OUT / "scores" / "yolo" / f"{n}.json"
        yl_t = OUT / "yolo" / f"{n}.json"
        row = {"clip": n, "source": r["source"], "label": r["label"], "desc": r["description"]}
        if sam_t.exists():
            d = read_json(sam_t); sam_docs.append(d)
            row["sam_cov"] = sum(f["present"] for f in d["frames"]) / max(1, d["n_frames"])
        if sam_s.exists():
            s = read_json(sam_s)
            row["sam_flag"] = s["flag"]; row["sam_comp"] = comp(s.get("best"))
            row["sam_soft"] = (s.get("best") or {}).get("soft_score")
        if spk.exists():
            v = read_json(spk); spark_docs.append(v)
            row["spark_fall"] = v["verdict"].get("fall"); row["spark_conf"] = v["verdict"].get("confidence")
            row["spark_reason"] = v["verdict"].get("reason", "")
        if "sam_flag" in row:
            if row["sam_flag"]:
                row["final"] = row["spark_fall"] if "spark_fall" in row else None
            else:
                row["final"] = False
        if yl_s.exists():
            s = read_json(yl_s); row["yolo_flag"] = s["flag"]; row["yolo_comp"] = comp(s.get("best"))
        if yl_t.exists():
            yolo_docs.append(read_json(yl_t))
        rows.append(row)

    L = []
    L.append("# Fall-detection feasibility test\n")
    L.append(f"{len(rows)} clips: {sum(r['label']=='fall' for r in rows)} falls, {sum(r['label']=='adl' for r in rows)} ADL "
             f"({sum(r['source']=='urfall' for r in rows)} UR Fall, {sum(r['source']=='gmdcsa24' for r in rows)} GMDCSA24). "
             "Clips downsampled to 5 fps, max width 640.\n")

    def tf(v):
        return "-" if v is None else ("YES" if v else "no")

    L.append("## Per-clip results\n")
    L.append("| clip | true | SAM score (D/S/H) | Spark verdict | final | YOLO score | note |")
    L.append("|---|---|---|---|---|---|---|")
    for r in rows:
        sam = f"{tf(r.get('sam_flag'))} {r.get('sam_comp','')}".strip() if "sam_flag" in r else "-"
        if "spark_fall" in r:
            spark = f"{tf(r['spark_fall'])} ({r['spark_conf']:.2f})"
        elif r.get("sam_flag") is False:
            spark = "not sent"
        else:
            spark = "-"
        yolo = f"{tf(r.get('yolo_flag'))} {r.get('yolo_comp','')}".strip() if "yolo_flag" in r else "-"
        L.append(f"| {r['clip']} | {r['label']} | {sam} | {spark} | {tf(r.get('final'))} | {yolo} | {r['desc'][:60]} |")
    L.append("\nD/S/H = drop / tall-to-wide shape / stays-low checks passed on the best candidate event.\n")

    def overall(flag_key, final_key, title):
        falls = [r for r in rows if r["label"] == "fall" and flag_key in r]
        adls = [r for r in rows if r["label"] == "adl" and flag_key in r]
        if not falls and not adls:
            return
        L.append(f"## {title}\n")
        L.append("| stage | falls caught | ADL false alarms |")
        L.append("|---|---|---|")
        L.append(f"| rule score only | {sum(bool(r[flag_key]) for r in falls)}/{len(falls)} | {sum(bool(r[flag_key]) for r in adls)}/{len(adls)} |")
        if final_key:
            fc = sum(bool(r.get(final_key)) for r in falls); fa = sum(bool(r.get(final_key)) for r in adls)
            pending = sum(1 for r in falls + adls if r.get(flag_key) and r.get(final_key) is None)
            L.append(f"| + Spark verification | {fc}/{len(falls)} | {fa}/{len(adls)} |" + (f" ({pending} flagged clips not yet verified)" if pending else ""))
        L.append("")

    overall("sam_flag", "final", "Overall: SAM 3.1 tracking + rule score, before and after Spark")
    overall("yolo_flag", None, "Overall: YOLO pose baseline + same rule score")

    if sam_docs:
        L.append("## SAM 3.1 latency and cost\n")
        up = med([d["timing"]["upload_s"] for d in sam_docs])
        ff = med([d["timing"]["first_frame_s"] for d in sam_docs])
        tot = med([d["timing"]["total_s"] for d in sam_docs])
        fr = [d["n_frames"] for d in sam_docs]
        L.append("| metric | median | mean | max |")
        L.append("|---|---|---|---|")
        L.append(f"| upload (s) | {fmt(up[0])} | {fmt(up[1])} | {fmt(up[2])} |")
        L.append(f"| time to first frame after request (s) | {fmt(ff[0])} | {fmt(ff[1])} | {fmt(ff[2])} |")
        L.append(f"| total per clip incl. upload (s) | {fmt(tot[0])} | {fmt(tot[1])} | {fmt(tot[2])} |")
        L.append(f"| frames per clip | {st.median(fr):.0f} | {st.mean(fr):.0f} | {max(fr)} |")
        L.append(f"\nSAM cost: {len(sam_docs)} clips, {sum(fr)} frames sent, {sum(d['billed_frames'] for d in sam_docs)} billed "
                 f"(50-frame minimum), **${sum(d['cost_usd'] for d in sam_docs):.3f}** at $0.20 per 1,000 video frames. "
                 f"Mean seconds of SAM processing per second of video: "
                 f"{st.mean([d['timing']['stream_s'] / (d['n_frames'] / d['fps']) for d in sam_docs]):.2f}.\n")
    if spark_docs:
        L.append("## Muse Spark latency and cost\n")
        lat = med([d["latency_s"] for d in spark_docs])
        ti = sum(d["usage"]["input_tokens"] for d in spark_docs); to = sum(d["usage"]["output_tokens"] for d in spark_docs)
        L.append(f"{len(spark_docs)} calls, latency median {fmt(lat[0])}s / mean {fmt(lat[1])}s / max {fmt(lat[2])}s. "
                 f"Tokens: {ti} in, {to} out. Cost **${sum(d['cost_usd'] for d in spark_docs):.4f}** "
                 f"(${sum(d['cost_usd'] for d in spark_docs)/len(spark_docs):.4f} per call).\n")
    if sam_docs or spark_docs:
        L.append(f"**Total API cost: ${sum(d['cost_usd'] for d in sam_docs) + sum(d['cost_usd'] for d in spark_docs):.3f}**\n")
    if yolo_docs:
        ms = [d["timing"]["per_frame_ms_mean"] for d in yolo_docs]
        L.append("## YOLO pose baseline speed\n")
        L.append(f"{yolo_docs[0]['tracker']} on this Mac (CPU): mean {st.mean(ms):.0f} ms/frame "
                 f"(median {st.median(ms):.0f}, max {max(ms):.0f}), i.e. ~{1000/st.mean(ms):.0f} fps. No API cost.\n")
    (ROOT / "REPORT.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
