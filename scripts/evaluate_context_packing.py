"""Evaluate post-retrieval context packing against gold contexts.

This does not change retrieval ranking; it measures whether generation context
contains the gold evidence after adding same-article neighbours.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from rouge_score import rouge_scorer  # type: ignore

from build_kb import load_vectorstore  # type: ignore
from context_packing import pack_article_context, vectorstore_docs  # type: ignore
from retrieval_v4 import RetrievalV4Config, build_retriever  # type: ignore

REPORT_DIR = ROOT / "reports" / "traffic"
OUT_CSV = REPORT_DIR / "context_packing_eval.csv"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _hit(scorer, gold: str, docs) -> bool:
    return any(scorer.score(gold, doc.page_content or "")["rougeL"].fmeasure >= 0.5 for doc in docs)


def evaluate(items, retriever, all_docs, seed_k: int, max_chunks: int):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    seed_hits = pack_hits = total = 0
    latencies = []
    avg_chars_seed = avg_chars_pack = 0
    for item in items:
        gold = item.get("context") or ""
        if not gold:
            continue
        total += 1
        t0 = time.perf_counter()
        seeds = retriever(item.get("question") or "", seed_k)
        packed = pack_article_context(seeds, all_docs, top_k=seed_k, max_chunks=max_chunks)
        latencies.append((time.perf_counter() - t0) * 1000)
        seed_hits += int(_hit(scorer, gold, seeds))
        pack_hits += int(_hit(scorer, gold, packed))
        avg_chars_seed += sum(len(d.page_content or "") for d in seeds)
        avg_chars_pack += sum(len(d.page_content or "") for d in packed)
    latencies.sort()
    p95 = latencies[min(len(latencies) - 1, int(round(0.95 * (len(latencies) - 1))))] if latencies else None
    return {
        "n": total,
        "seed_k": seed_k,
        "max_chunks": max_chunks,
        "seed_context_recall": round(seed_hits / total, 4) if total else None,
        "packed_context_recall": round(pack_hits / total, 4) if total else None,
        "delta": round((pack_hits - seed_hits) / total, 4) if total else None,
        "avg_seed_chars": round(avg_chars_seed / total, 1) if total else None,
        "avg_packed_chars": round(avg_chars_pack / total, 1) if total else None,
        "p95_ms": round(p95, 1) if p95 is not None else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", default="data/splits_filtered/qa_test.jsonl")
    parser.add_argument("--seed-k", type=int, default=2)
    parser.add_argument("--max-chunks", nargs="+", type=int, default=[2, 3, 4, 5])
    args = parser.parse_args()

    items = load_jsonl(ROOT / args.eval)
    vs = load_vectorstore()
    all_docs = vectorstore_docs(vs)
    cfg = RetrievalV4Config(
        dense_k=50,
        sparse_k=50,
        fuse_k=40,
        rrf_k=60,
        w_dense=0.6,
        w_sparse=0.4,
        use_article_window=False,
        use_doc_aggregation=False,
    )
    retriever = build_retriever(vs, cfg)
    rows = []
    for max_chunks in args.max_chunks:
        row = evaluate(items, retriever, all_docs, args.seed_k, max_chunks)
        rows.append(row)
        print(row)
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[ok] wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
