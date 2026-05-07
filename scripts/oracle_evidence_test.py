#!/usr/bin/env python
"""Create an oracle evidence-card dataset from eval rows with gold context.

This does not run model inference. Use the generated JSONL with evaluate.py
--oracle-context-file to test whether the generator can answer from clean cards.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from evidence_cards import evidence_card_from_text  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/eval_manual.jsonl")
    ap.add_argument("--output", default="data/eval_manual_oracle_cards.jsonl")
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--sanction-only", action="store_true", default=True)
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.input).read_text(encoding="utf-8").splitlines() if l.strip()]
    out_rows = []
    for i, row in enumerate(rows):
        q = row.get("question") or ""
        ctx = row.get("context") or row.get("gold_context") or ""
        ans = row.get("answer") or ""
        if args.sanction_only and not any(x in (q + " " + ans).lower() for x in ["phạt", "tiền", "điểm", "tước", "giấy phép"]):
            continue
        if not q or not ctx:
            continue
        md = {k: row.get(k) for k in ["doc_id", "source", "article", "article_number", "clause_number", "point_letter"] if row.get(k)}
        card = evidence_card_from_text(q, ctx, md)
        new = dict(row)
        new["oracle_context"] = card
        new["original_index"] = i
        out_rows.append(new)
        if args.limit and len(out_rows) >= args.limit:
            break
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out_rows), encoding="utf-8")
    print(json.dumps({"written": len(out_rows), "output": str(out)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
