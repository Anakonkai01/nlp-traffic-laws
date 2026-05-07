"""Compare v2 heuristic retriever against v4 on a per-question basis."""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from rouge_score import rouge_scorer  # type: ignore

from build_kb import load_vectorstore  # type: ignore

REPORT_DIR = ROOT / "reports" / "traffic"
OUT_CSV = REPORT_DIR / "retrieval_v2_v4_comparison.csv"
OUT_JSON = REPORT_DIR / "retrieval_v2_v4_comparison_summary.json"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _doc_id(doc):
    md = doc.metadata or {}
    return md.get("doc_id") or md.get("source")


def _article(doc):
    return str((doc.metadata or {}).get("article_number") or "")


def hit(scorer, gold: str, docs) -> bool:
    return any(scorer.score(gold, d.page_content or "")["rougeL"].fmeasure >= 0.5 for d in docs)


def load_retrieval_module(version: str):
    if version == "v2":
        os.environ.pop("RETRIEVAL_VERSION", None)
    else:
        os.environ["RETRIEVAL_VERSION"] = version
    sys.modules.pop("retrieval", None)
    return importlib.import_module("retrieval")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", default="data/eval_manual.jsonl")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    items = load_jsonl(ROOT / args.eval)
    if args.limit:
        items = items[: args.limit]
    vs = load_vectorstore()
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    r2 = load_retrieval_module("v2")
    v2_fn = r2.retrieve_ranked_docs
    r4 = load_retrieval_module("v4")
    v4_fn = r4.retrieve_ranked_docs

    rows = []
    counts = {"both_hit": 0, "v2_only": 0, "v4_only": 0, "both_miss": 0, "measured": 0}
    for idx, item in enumerate(items):
        gold = item.get("context") or ""
        if not gold:
            continue
        q = item.get("question") or ""
        v2_docs = v2_fn(vs, q, top_k=args.top_k, candidate_k=max(30, args.top_k * 8))
        v4_docs = v4_fn(vs, q, top_k=args.top_k, candidate_k=max(30, args.top_k * 8))
        h2 = hit(scorer, gold, v2_docs)
        h4 = hit(scorer, gold, v4_docs)
        if h2 and h4:
            bucket = "both_hit"
        elif h2 and not h4:
            bucket = "v2_only"
        elif h4 and not h2:
            bucket = "v4_only"
        else:
            bucket = "both_miss"
        counts[bucket] += 1
        counts["measured"] += 1
        rows.append({
            "index": idx,
            "bucket": bucket,
            "question": q,
            "gold_doc_id": item.get("doc_id") or item.get("source") or "",
            "v2_top_docs": ";".join(filter(None, [_doc_id(d) for d in v2_docs[:args.top_k]])),
            "v2_top_articles": ";".join(_article(d) for d in v2_docs[:args.top_k]),
            "v4_top_docs": ";".join(filter(None, [_doc_id(d) for d in v4_docs[:args.top_k]])),
            "v4_top_articles": ";".join(_article(d) for d in v4_docs[:args.top_k]),
            "v2_top1_preview": (v2_docs[0].page_content if v2_docs else "")[:350].replace("\n", " "),
            "v4_top1_preview": (v4_docs[0].page_content if v4_docs else "")[:350].replace("\n", " "),
            "gold_preview": gold[:350].replace("\n", " "),
        })

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = {**counts, "eval": args.eval, "top_k": args.top_k}
    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[ok] wrote {OUT_CSV}")
    print(f"[ok] wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
