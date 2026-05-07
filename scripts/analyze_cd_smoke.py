#!/usr/bin/env python3
"""Compare no-RAG (C) and RAG (D) predictions for a small smoke run."""
from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRED_PATH = ROOT / "reports/traffic/predictions_cd_smoke20_after_v2hard_ft.json"
OUT_CSV = ROOT / "reports/traffic/cd_smoke20_error_analysis.csv"
OUT_JSON = ROOT / "reports/traffic/cd_smoke20_error_summary.json"

WORD_RE = re.compile(r"\w+", re.UNICODE)


def toks(text: str) -> list[str]:
    return WORD_RE.findall((text or "").lower())


def f1(pred: str, ref: str) -> float:
    pt, rt = toks(pred), toks(ref)
    if not pt or not rt:
        return 0.0
    pc, rc = Counter(pt), Counter(rt)
    overlap = sum((pc & rc).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pt)
    recall = overlap / len(rt)
    return 2 * precision * recall / (precision + recall)


def brief_sources(sources: list[dict], n: int = 3) -> str:
    parts = []
    for src in sources[:n]:
        doc = src.get("doc_id") or src.get("source") or ""
        art = src.get("article_number") or ""
        chunk = src.get("chunk_id") or ""
        parts.append(f"{doc}|Điều {art}|{chunk}")
    return " ; ".join(parts)


def classify(row: dict) -> str:
    delta = row["d_f1"] - row["c_f1"]
    if delta >= 0.08:
        return "D_better"
    if delta <= -0.08:
        return "D_worse"
    return "similar"


def main() -> None:
    data = json.loads(PRED_PATH.read_text(encoding="utf-8"))
    c, d = data["C"], data["D"]
    rows = []
    for i, (q, ref, cp, dp, ds) in enumerate(
        zip(c["questions"], c["references"], c["predictions"], d["predictions"], d["retrieved_sources"]),
        start=1,
    ):
        row = {
            "idx": i,
            "question": q,
            "reference": ref,
            "c_prediction": cp,
            "d_prediction": dp,
            "c_f1": round(f1(cp, ref), 4),
            "d_f1": round(f1(dp, ref), 4),
            "d_top_sources": brief_sources(ds),
        }
        row["delta_d_minus_c"] = round(row["d_f1"] - row["c_f1"], 4)
        row["bucket"] = classify(row)
        rows.append(row)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "n": len(rows),
        "avg_c_f1": round(sum(r["c_f1"] for r in rows) / len(rows), 4),
        "avg_d_f1": round(sum(r["d_f1"] for r in rows) / len(rows), 4),
        "avg_delta_d_minus_c": round(sum(r["delta_d_minus_c"] for r in rows) / len(rows), 4),
        "buckets": dict(Counter(r["bucket"] for r in rows)),
        "top_d_worse": [
            {k: r[k] for k in ["idx", "delta_d_minus_c", "question", "c_f1", "d_f1", "d_top_sources"]}
            for r in sorted(rows, key=lambda x: x["delta_d_minus_c"])[:5]
        ],
        "top_d_better": [
            {k: r[k] for k in ["idx", "delta_d_minus_c", "question", "c_f1", "d_f1", "d_top_sources"]}
            for r in sorted(rows, key=lambda x: x["delta_d_minus_c"], reverse=True)[:5]
        ],
    }
    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"CSV -> {OUT_CSV}")
    print(f"JSON -> {OUT_JSON}")


if __name__ == "__main__":
    main()
