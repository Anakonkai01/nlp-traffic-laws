#!/usr/bin/env python
"""Build clause-reranker training pairs with semi-hard negatives.

This script creates scientific training data for a future learned reranker. It
uses gold context overlap to choose positives and samples negatives from the same
retrieved article/source when possible, avoiding hand-written traffic rules.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rouge_score import rouge_scorer

import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from build_kb import load_vectorstore  # noqa: E402
from legal_units import candidate_units_from_seeds, load_legal_units  # noqa: E402
from retrieval import retrieve_ranked_docs  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/eval_manual.jsonl")
    ap.add_argument("--output", default="data/training/clause_reranker_pairs.jsonl")
    ap.add_argument("--top-seeds", type=int, default=5)
    ap.add_argument("--max-candidates", type=int, default=120)
    ap.add_argument("--negatives", type=int, default=6)
    ap.add_argument("--positive-threshold", type=float, default=0.45)
    ap.add_argument("--false-negative-threshold", type=float, default=0.35)
    args = ap.parse_args()

    data = load_jsonl(Path(args.input))
    units = load_legal_units()
    vs = load_vectorstore()
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = skipped = 0
    with out_path.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(data):
            gold = row.get("context") or row.get("gold_context") or ""
            q = row.get("question") or ""
            if not q or not gold:
                skipped += 1
                continue
            seeds = retrieve_ranked_docs(vs, q, top_k=args.top_seeds, candidate_k=max(30, args.top_seeds * 8))
            candidates = candidate_units_from_seeds(units, seeds, max_units=args.max_candidates)
            if not candidates:
                skipped += 1
                continue
            scored = [(unit, scorer.score(gold, unit.page_content or "")["rougeL"].fmeasure) for unit in candidates]
            positives = [(u, s) for u, s in scored if s >= args.positive_threshold]
            if not positives:
                skipped += 1
                continue
            positives.sort(key=lambda x: -x[1])
            pos, pos_score = positives[0]
            negs = [(u, s) for u, s in scored if s < args.false_negative_threshold]
            # Semi-hard negatives: same candidate pool, high-ish overlap but below false-negative threshold.
            negs.sort(key=lambda x: -x[1])
            examples = [{"text": pos.page_content, "metadata": pos.metadata, "label": 1, "overlap": pos_score}]
            for neg, score in negs[: args.negatives]:
                examples.append({"text": neg.page_content, "metadata": neg.metadata, "label": 0, "overlap": score})
            f.write(json.dumps({"index": idx, "question": q, "positive_overlap": pos_score, "examples": examples}, ensure_ascii=False) + "\n")
            written += 1
    print(json.dumps({"written_queries": written, "skipped": skipped, "output": str(out_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
