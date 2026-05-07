#!/usr/bin/env python
"""Audit structured fact retrieval without running generation.

The script measures whether retrieved fact cards contain answer-bearing slots
before we spend time on LLM evaluation.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from config import DATA_DIR  # noqa: E402
from build_kb import load_vectorstore  # noqa: E402
from retrieval import retrieve_ranked_docs  # noqa: E402
from sanction_facts import load_sanction_facts, candidate_facts_from_seeds, FactBM25  # noqa: E402

FINE_RE = re.compile(r"(?:\d{1,3}(?:\.\d{3})+|\d+)\s*(?:đồng|nghìn|triệu)", re.I)
ARTICLE_RE = re.compile(r"điều\s+(\d+)", re.I)
VEHICLES = ["xe máy chuyên dùng", "ô tô", "xe máy", "mô tô", "xe gắn máy", "xe đạp", "người đi bộ"]


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").lower().strip()


def fine_tokens(s: str) -> set[str]:
    return {re.sub(r"\s+", " ", m.group(0)).lower() for m in FINE_RE.finditer(s or "")}


def vehicles(s: str) -> set[str]:
    low = norm(s)
    return {v for v in VEHICLES if v in low}


def articles(s: str) -> set[str]:
    return set(ARTICLE_RE.findall(s or ""))


def overlap_recall(gold: set[str], pred: set[str]) -> float | None:
    if not gold:
        return None
    return len(gold & pred) / len(gold)


def fact_text(f: dict) -> str:
    return " ".join(str(f.get(k) or "") for k in ["citation", "article_title", "violation_text", "fine_text", "points_deducted", "suspension_text"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-file", default="data/eval_manual.jsonl")
    ap.add_argument("--facts", default="data/legal_sanction_facts.jsonl")
    ap.add_argument("--samples", type=int, default=0)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--seed-k", type=int, default=5)
    ap.add_argument("--output", default="reports/traffic/fact_retrieval_audit.json")
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.test_file).read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.samples:
        rows = rows[: args.samples]

    vs = load_vectorstore()
    facts = load_sanction_facts(Path(args.facts))
    bm25 = FactBM25.build(facts)

    per_case = []
    sums = {"fine": 0.0, "fine_n": 0, "vehicle": 0.0, "vehicle_n": 0, "article": 0.0, "article_n": 0}
    top1_sums = {"fine": 0.0, "fine_n": 0, "vehicle": 0.0, "vehicle_n": 0, "article": 0.0, "article_n": 0}
    type_counts = {}
    no_candidates = 0
    answerish_top1 = 0

    for i, row in enumerate(rows):
        q = row.get("question") or row.get("prompt") or ""
        ref = row.get("answer") or row.get("reference") or row.get("expected_answer") or ""
        gold_context = row.get("context") or ""
        gold_text = f"{ref} {gold_context}"
        seeds = retrieve_ranked_docs(vs, q, top_k=args.seed_k)
        candidates = candidate_facts_from_seeds(facts, seeds)
        if not candidates:
            no_candidates += 1
            candidates = facts
        ranked = bm25.search(q, candidates, k=args.top_k)
        top_facts = [f for f, _ in ranked]
        joined = " ".join(fact_text(f) for f in top_facts)
        top1 = top_facts[0] if top_facts else {}
        top1_joined = fact_text(top1) if top1 else ""
        if top1.get("fine_text") or top1.get("points_deducted") or top1.get("suspension_text"):
            answerish_top1 += 1
        t = top1.get("unit_type", "none")
        type_counts[t] = type_counts.get(t, 0) + 1

        metrics = {}
        for name, getter in [("fine", fine_tokens), ("vehicle", vehicles), ("article", articles)]:
            g = getter(gold_text)
            p = getter(joined)
            p1 = getter(top1_joined)
            r = overlap_recall(g, p)
            r1 = overlap_recall(g, p1)
            metrics[name] = r
            metrics[f"top1_{name}"] = r1
            metrics[f"gold_{name}"] = sorted(g)
            metrics[f"pred_{name}"] = sorted(p)
            metrics[f"top1_pred_{name}"] = sorted(p1)
            if r is not None:
                sums[name] += r
                sums[f"{name}_n"] += 1
            if r1 is not None:
                top1_sums[name] += r1
                top1_sums[f"{name}_n"] += 1

        per_case.append({
            "idx": i,
            "question": q,
            "reference": ref,
            "top_fact_ids": [f.get("fact_id") for f in top_facts],
            "top_citations": [f.get("citation") for f in top_facts],
            "top_fines": [f.get("fine_text") for f in top_facts],
            "top_vehicles": [f.get("vehicle_scope") for f in top_facts],
            "top_types": [f.get("unit_type") for f in top_facts],
            **metrics,
        })

    summary = {
        "n": len(rows),
        "top_k": args.top_k,
        "seed_k": args.seed_k,
        "no_candidates": no_candidates,
        "answerish_top1_rate": answerish_top1 / len(rows) if rows else 0,
        "top1_type_counts": type_counts,
        "avg_fine_recall": sums["fine"] / sums["fine_n"] if sums["fine_n"] else None,
        "top1_fine_recall": top1_sums["fine"] / top1_sums["fine_n"] if top1_sums["fine_n"] else None,
        "fine_eval_n": sums["fine_n"],
        "avg_vehicle_recall": sums["vehicle"] / sums["vehicle_n"] if sums["vehicle_n"] else None,
        "top1_vehicle_recall": top1_sums["vehicle"] / top1_sums["vehicle_n"] if top1_sums["vehicle_n"] else None,
        "vehicle_eval_n": sums["vehicle_n"],
        "avg_article_recall": sums["article"] / sums["article_n"] if sums["article_n"] else None,
        "top1_article_recall": top1_sums["article"] / top1_sums["article_n"] if top1_sums["article_n"] else None,
        "article_eval_n": sums["article_n"],
    }
    out = {"summary": summary, "cases": per_case}
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
