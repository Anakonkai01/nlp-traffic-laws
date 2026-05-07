"""Generate error analysis for the selected Retrieval v4 configuration."""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from rouge_score import rouge_scorer  # type: ignore

from build_kb import load_vectorstore  # type: ignore
from retrieval_v4 import RetrievalV4Config, build_retriever  # type: ignore

REPORT_DIR = ROOT / "reports" / "traffic"
BEST_JSON = REPORT_DIR / "retrieval_v4_best_config.json"
ERROR_JSONL = REPORT_DIR / "retrieval_v4_errors.jsonl"
ERROR_SUMMARY = REPORT_DIR / "retrieval_v4_error_summary.json"

ARTICLE_RE = re.compile(r"[Đđ]i\S*u\s+(\d+)")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _doc_id(doc):
    md = doc.metadata or {}
    return md.get("doc_id") or md.get("source")


def _article(doc):
    md = doc.metadata or {}
    return str(md.get("article_number") or "")


def _gold_article(item: dict) -> str:
    text = " ".join(str(item.get(k) or "") for k in ["article", "context", "question"])
    m = ARTICLE_RE.search(text)
    return m.group(1) if m else ""


def _expected_doc_ids(item: dict) -> set[str]:
    eids = item.get("expected_doc_ids") or []
    if not eids and item.get("doc_id"):
        eids = [item["doc_id"]]
    return {x for x in eids if isinstance(x, str) and x}


def _is_context_hit(scorer, gold_ctx: str, doc) -> bool:
    if not gold_ctx:
        return False
    return scorer.score(gold_ctx, doc.page_content or "")["rougeL"].fmeasure >= 0.5


def classify_error(item: dict, docs, hit: bool) -> str:
    if hit:
        return "hit"
    expected = _expected_doc_ids(item)
    gold_article = _gold_article(item)
    top5 = docs[:5]
    if expected and not any(_doc_id(d) in expected for d in top5):
        return "wrong_document"
    if expected and gold_article and not any(_doc_id(d) in expected and _article(d) == gold_article for d in top5):
        return "correct_document_wrong_article"
    if expected and gold_article and any(_doc_id(d) in expected and _article(d) == gold_article for d in top5):
        return "correct_article_wrong_chunk"
    return "semantic_or_label_mismatch"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", default="data/splits_filtered/qa_test.jsonl")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--out", default=str(ERROR_JSONL))
    args = parser.parse_args()

    best = json.loads(BEST_JSON.read_text())
    cfg = RetrievalV4Config(**best["best_config"])
    items = load_jsonl(ROOT / args.eval)
    vs = load_vectorstore()
    retriever = build_retriever(vs, cfg)
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    counts: dict[str, int] = {}
    rows = []
    for idx, item in enumerate(items):
        docs = retriever(item.get("question") or "", args.top_k)
        gold_ctx = item.get("context") or ""
        hit = any(_is_context_hit(scorer, gold_ctx, doc) for doc in docs[:5])
        err_type = classify_error(item, docs, hit)
        counts[err_type] = counts.get(err_type, 0) + 1
        if hit:
            continue
        top_docs = []
        for rank, doc in enumerate(docs[: args.top_k], start=1):
            md = doc.metadata or {}
            top_docs.append({
                "rank": rank,
                "doc_id": _doc_id(doc),
                "article_number": _article(doc),
                "chunk_id": md.get("chunk_id"),
                "score": md.get("retrieval_v4_score"),
                "parts": md.get("retrieval_v4_parts"),
                "preview": (doc.page_content or "")[:400],
            })
        rows.append({
            "index": idx,
            "error_type": err_type,
            "question": item.get("question"),
            "expected_doc_ids": sorted(_expected_doc_ids(item)),
            "gold_article": _gold_article(item),
            "gold_context_preview": gold_ctx[:700],
            "top_docs": top_docs,
        })

    out_path = Path(args.out)
    with open(out_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "eval": args.eval,
        "n": len(items),
        "counts": counts,
        "n_errors_written": len(rows),
        "best_config": best["best_config"],
    }
    ERROR_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[ok] wrote {out_path}")
    print(f"[ok] wrote {ERROR_SUMMARY}")


if __name__ == "__main__":
    main()
