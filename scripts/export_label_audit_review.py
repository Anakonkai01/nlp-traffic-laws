"""Export label-audit JSONL to a spreadsheet-friendly CSV for human review."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "reports" / "traffic"
DEFAULT_IN = REPORT_DIR / "eval_label_audit.jsonl"
DEFAULT_OUT = REPORT_DIR / "eval_label_audit_review.csv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_IN))
    parser.add_argument("--output", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    rows = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            rows.append({
                "index": item.get("index"),
                "reasons": ";".join(item.get("reasons") or []),
                "question": item.get("question"),
                "expected_doc_ids": ";".join(item.get("expected_doc_ids") or []),
                "top1_doc_id": item.get("top1_doc_id"),
                "gold_q_overlap": item.get("gold_q_overlap"),
                "top1_q_overlap": item.get("top1_q_overlap"),
                "top1_gold_rouge": item.get("top1_gold_rouge"),
                "gold_context_preview": (item.get("gold_context_preview") or "").replace("\n", " "),
                "top1_preview": (item.get("top1_preview") or "").replace("\n", " "),
                "human_decision": "",
                "notes": "",
            })

    fields = [
        "index", "reasons", "question", "expected_doc_ids", "top1_doc_id",
        "gold_q_overlap", "top1_q_overlap", "top1_gold_rouge",
        "gold_context_preview", "top1_preview", "human_decision", "notes",
    ]
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[ok] wrote {args.output} rows={len(rows)}")


if __name__ == "__main__":
    main()
