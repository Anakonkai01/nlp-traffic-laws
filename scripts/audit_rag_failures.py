#!/usr/bin/env python
"""Audit RAG failure modes without running model inference.

Compares C and D prediction files and reports copy-noise, metric deltas, and
simple factual-slot coverage (fine/citation/vehicle/action token overlap).
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from rouge_score import rouge_scorer

COPY_NOISE_RE = re.compile(r"\b(Nguồn|Cấu trúc|Tiêu đề|assistant)\b|---")
FINE_RE = re.compile(r"(?:\d{1,3}(?:\.\d{3})+|\d+)\s*(?:đồng|nghìn|triệu)", re.I)
CITATION_RE = re.compile(r"Điều\s+\d+[a-zA-Z]?|khoản\s+\d+|điểm\s+[a-zđ]", re.I)
VEHICLE_TERMS = ["ô tô", "xe máy", "mô tô", "xe gắn máy", "xe đạp", "xe máy chuyên dùng", "người đi bộ"]


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


def token_f1(pred: str, ref: str) -> float:
    pt = re.findall(r"[\wÀ-ỹ]+", norm(pred))
    rt = re.findall(r"[\wÀ-ỹ]+", norm(ref))
    if not pt or not rt:
        return 0.0
    cp, cr = Counter(pt), Counter(rt)
    common = sum((cp & cr).values())
    if not common:
        return 0.0
    p = common / len(pt)
    r = common / len(rt)
    return 2 * p * r / (p + r)


def slots(text: str) -> dict:
    low = norm(text)
    return {
        "fines": sorted(set(FINE_RE.findall(text or ""))),
        "citations": sorted(set(m.group(0) for m in CITATION_RE.finditer(text or ""))),
        "vehicles": [v for v in VEHICLE_TERMS if v in low],
    }


def overlap(a: list[str], b: list[str]) -> float:
    if not a:
        return 1.0 if not b else 0.0
    return len(set(a) & set(b)) / max(1, len(set(a)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c-preds", default="reports/traffic/predictions_c_full_fast.json")
    ap.add_argument("--d-preds", default="reports/traffic/predictions_eval_manual_full_optimized.json")
    ap.add_argument("--output", default="reports/traffic/rag_failure_audit.json")
    args = ap.parse_args()

    c = load_json(Path(args.c_preds))["C"]
    d = load_json(Path(args.d_preds))["D"]
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    rows = []
    for i, (q, ref, cp, dp) in enumerate(zip(c["questions"], c["references"], c["predictions"], d["predictions"])):
        c_rl = scorer.score(ref, cp)["rougeL"].fmeasure
        d_rl = scorer.score(ref, dp)["rougeL"].fmeasure
        ref_slots = slots(ref)
        d_slots = slots(dp)
        row = {
            "index": i,
            "question": q,
            "reference": ref,
            "pred_c": cp,
            "pred_d": dp,
            "rougeL_c": round(c_rl, 4),
            "rougeL_d": round(d_rl, 4),
            "delta_rougeL": round(d_rl - c_rl, 4),
            "f1_c": round(token_f1(cp, ref), 4),
            "f1_d": round(token_f1(dp, ref), 4),
            "copy_noise_d": bool(COPY_NOISE_RE.search(dp or "")),
            "long_answer_d": len(dp or "") > 500,
            "ref_slots": ref_slots,
            "d_slots": d_slots,
            "fine_slot_recall_d": round(overlap(ref_slots["fines"], d_slots["fines"]), 4),
            "vehicle_slot_recall_d": round(overlap(ref_slots["vehicles"], d_slots["vehicles"]), 4),
            "sources_d": (d.get("retrieved_sources") or [[]])[i],
        }
        rows.append(row)

    summary = {
        "n": len(rows),
        "d_worse_rougeL_gt_002": sum(r["delta_rougeL"] < -0.02 for r in rows),
        "d_better_rougeL_gt_002": sum(r["delta_rougeL"] > 0.02 for r in rows),
        "copy_noise_rate_d": round(sum(r["copy_noise_d"] for r in rows) / len(rows), 4),
        "long_answer_rate_d": round(sum(r["long_answer_d"] for r in rows) / len(rows), 4),
        "avg_delta_rougeL": round(sum(r["delta_rougeL"] for r in rows) / len(rows), 4),
        "avg_fine_slot_recall_d": round(sum(r["fine_slot_recall_d"] for r in rows) / len(rows), 4),
        "avg_vehicle_slot_recall_d": round(sum(r["vehicle_slot_recall_d"] for r in rows) / len(rows), 4),
    }
    worst = sorted(rows, key=lambda r: r["delta_rougeL"])[:40]
    out = {"summary": summary, "worst_d_minus_c": worst, "rows": rows}
    Path(args.output).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
