"""
Mine hard negatives for cross-encoder fine-tuning.

For each (question, gold_chunk) in qa_train.jsonl:
  1. Retrieve top-K with Dense + BM25 + RRF (no CE) — same recall surface CE will see.
  2. Drop positives:
       - chunks with same (doc_id, article_number) as gold
       - chunks whose page_content overlaps gold context[:300] (substring)
  3. Pick the first ``--negatives`` survivors → hard negatives.

Output JSONL rows:
  {"question": str, "positive": str, "negatives": [str, ...],
   "qa_id": int, "gold_doc_id": str, "gold_article": str}

Run from nlp/:
    python scripts/mine_hard_negatives.py \\
        --train data/splits/qa_train.jsonl --negatives 5 \\
        --out data/training/ce_pairs.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from build_kb import load_vectorstore  # type: ignore
from retrieval_v3 import RetrievalConfig, build_retriever  # type: ignore

ART_RE = re.compile(r"Điều\s+(\d+)")


def gold_article(article_str: str) -> str:
    m = ART_RE.search(article_str or "")
    return m.group(1) if m else ""


def is_positive(doc, gold_doc_id: str, gold_art: str, gold_ctx_head: str) -> bool:
    md = doc.metadata or {}
    if (md.get("doc_id") or md.get("source")) == gold_doc_id and \
            str(md.get("article_number") or "") == gold_art:
        return True
    if gold_ctx_head and gold_ctx_head in (doc.page_content or "")[: len(gold_ctx_head) + 200]:
        return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/splits/qa_train.jsonl")
    ap.add_argument("--negatives", type=int, default=5)
    ap.add_argument("--top-k", type=int, default=30)
    ap.add_argument("--out", default="data/training/ce_pairs.jsonl")
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.train).read_text().splitlines() if l.strip()]
    print(f"[load] {len(rows)} train rows")

    vs = load_vectorstore()
    cfg = RetrievalConfig(use_bm25=True, use_alias=False, use_ce=False,
                          fuse_k=args.top_k)
    retrieve = build_retriever(vs, cfg)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_written = n_missing_pos = n_short_neg = 0
    neg_ranks = Counter()
    same_article_in_neg = 0

    with open(out_path, "w") as fout:
        for qa_id, row in enumerate(rows):
            q = row.get("question") or ""
            gold_ctx = row.get("context") or ""
            gold_doc_id = row.get("doc_id") or ""
            gold_art = gold_article(row.get("article", ""))
            gold_head = gold_ctx[:300]
            if not (q and gold_ctx and gold_doc_id):
                n_missing_pos += 1
                continue

            docs = retrieve(q, args.top_k)
            if not docs:
                n_missing_pos += 1
                continue

            negatives: list[str] = []
            for rank, d in enumerate(docs, start=1):
                if is_positive(d, gold_doc_id, gold_art, gold_head):
                    continue
                neg_ranks[rank] += 1
                md = d.metadata or {}
                if (md.get("doc_id") or md.get("source")) == gold_doc_id and \
                        str(md.get("article_number") or "") == gold_art:
                    same_article_in_neg += 1
                negatives.append(d.page_content or "")
                if len(negatives) >= args.negatives:
                    break
            if len(negatives) < args.negatives:
                n_short_neg += 1

            fout.write(json.dumps({
                "question": q,
                "positive": gold_ctx,
                "negatives": negatives,
                "qa_id": qa_id,
                "gold_doc_id": gold_doc_id,
                "gold_article": gold_art,
            }, ensure_ascii=False) + "\n")
            n_written += 1

    n_total = max(1, n_written)
    mean_rank = sum(r * c for r, c in neg_ranks.items()) / max(1, sum(neg_ranks.values()))
    print(f"[done] wrote {n_written}, missing_positive={n_missing_pos}, short_neg={n_short_neg}")
    print(f"[done] mean negative rank in top-{args.top_k} = {mean_rank:.2f}  "
          f"(must be ≥ 5 for 'hard')")
    print(f"[done] negatives accidentally same-article-as-pos: {same_article_in_neg}  "
          f"(should be 0)")
    print(f"[done] output → {out_path}")


if __name__ == "__main__":
    main()
