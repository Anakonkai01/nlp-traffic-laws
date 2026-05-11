"""Lightweight audit for likely noisy retrieval labels/gold contexts."""
from __future__ import annotations

import argparse
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
AUDIT_JSONL = REPORT_DIR / "eval_label_audit.jsonl"
AUDIT_SUMMARY = REPORT_DIR / "eval_label_audit_summary.json"

TOKEN_RE = re.compile(r"[\wÀ-ỹ]+", re.UNICODE)
STOP = {"của", "và", "là", "theo", "được", "các", "cho", "trong", "khi", "nào", "gì", "về", "tại", "này", "đó"}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def tokens(text: str) -> set[str]:
    return {t.lower() for t in TOKEN_RE.findall(text or "") if len(t) >= 3 and t.lower() not in STOP}


def overlap(a: str, b: str) -> float:
    ta, tb = tokens(a), tokens(b)
    return len(ta & tb) / max(1, len(ta))


def _doc_id(doc):
    md = doc.metadata or {}
    return md.get("doc_id") or md.get("source")


def _expected_doc_ids(item: dict) -> set[str]:
    eids = item.get("expected_doc_ids") or []
    if not eids and item.get("doc_id"):
        eids = [item["doc_id"]]
    return {x for x in eids if isinstance(x, str) and x}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", default="data/splits_filtered/qa_test.jsonl")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    best = json.loads(BEST_JSON.read_text()) if BEST_JSON.exists() else {"best_config": {}}
    cfg = RetrievalV4Config(**best.get("best_config", {}))
    items = load_jsonl(ROOT / args.eval)
    vs = load_vectorstore()
    retriever = build_retriever(vs, cfg)
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    flags = []
    counts = {"possible_label_noise": 0, "gold_low_question_overlap": 0, "retrieved_better_than_gold_overlap": 0}
    for idx, item in enumerate(items):
        q = item.get("question") or ""
        gold = item.get("context") or ""
        expected = _expected_doc_ids(item)
        docs = retriever(q, args.top_k)
        top1 = docs[0] if docs else None
        top1_text = top1.page_content if top1 else ""
        gold_q_overlap = overlap(q, gold)
        top1_q_overlap = overlap(q, top1_text)
        top1_gold_rouge = scorer.score(gold, top1_text)["rougeL"].fmeasure if top1 else 0.0
        top1_doc_match = bool(top1 and expected and _doc_id(top1) in expected)

        reasons = []
        if gold_q_overlap < 0.12:
            reasons.append("gold_low_question_overlap")
            counts["gold_low_question_overlap"] += 1
        if top1_q_overlap >= gold_q_overlap + 0.18 and top1_gold_rouge < 0.35:
            reasons.append("retrieved_better_than_gold_overlap")
            counts["retrieved_better_than_gold_overlap"] += 1
        if reasons and not top1_doc_match:
            reasons.append("possible_label_noise")
            counts["possible_label_noise"] += 1
        if reasons:
            flags.append({
                "index": idx,
                "reasons": reasons,
                "question": q,
                "expected_doc_ids": sorted(expected),
                "gold_q_overlap": round(gold_q_overlap, 4),
                "top1_q_overlap": round(top1_q_overlap, 4),
                "top1_gold_rouge": round(top1_gold_rouge, 4),
                "top1_doc_id": _doc_id(top1) if top1 else None,
                "gold_context_preview": gold[:600],
                "top1_preview": top1_text[:600],
            })

    with open(AUDIT_JSONL, "w") as f:
        for row in flags:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {"eval": args.eval, "n": len(items), "n_flagged": len(flags), "counts": counts}
    AUDIT_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[ok] wrote {AUDIT_JSONL}")
    print(f"[ok] wrote {AUDIT_SUMMARY}")


if __name__ == "__main__":
    main()
