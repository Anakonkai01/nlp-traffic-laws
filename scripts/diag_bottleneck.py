"""Diagnostic: where does v3 (A2) lose recall? Run on qa_dev + qa_test."""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from rouge_score import rouge_scorer  # type: ignore

from build_kb import load_vectorstore  # type: ignore
from retrieval_v3 import RetrievalConfig, build_retriever  # type: ignore

ART_RE = re.compile(r"Điều\s+(\d+)")


def gold_article(art: str) -> str:
    m = ART_RE.search(art or "")
    return m.group(1) if m else ""


def main():
    sets = [("qa_dev", ROOT / "data/splits/qa_dev.jsonl"),
            ("qa_test", ROOT / "data/splits/qa_test.jsonl")]
    rows_per_set = {n: [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
                    for n, p in sets}

    vs = load_vectorstore()
    cfg = RetrievalConfig(use_bm25=True, use_alias=False, use_ce=False, fuse_k=50)
    retrieve = build_retriever(vs, cfg)
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    for set_name, rows in rows_per_set.items():
        print(f"\n=== {set_name} (n={len(rows)}) ===")

        recall_chunk = {k: 0 for k in (1, 3, 5, 10, 20, 30, 50)}
        recall_article = {k: 0 for k in (1, 3, 5, 10, 20, 30, 50)}
        recall_doc = {k: 0 for k in (1, 3, 5, 10, 20, 30, 50)}
        ranks_chunk = []  # rank where ROUGE>=0.5 first appears (None if never in top-50)
        ranks_article = []  # rank where same (doc,article) first appears
        per_doc = defaultdict(lambda: {"hit_chunk_at_5": 0, "total": 0})
        n_total = 0

        for r in rows:
            q = r.get("question") or ""
            gold_ctx = r.get("context") or ""
            gold_doc = r.get("doc_id") or ""
            gold_art = gold_article(r.get("article", ""))
            if not (q and gold_ctx and gold_doc):
                continue
            n_total += 1
            cands = retrieve(q, 50)

            # rank of first chunk passing ROUGE>=0.5
            first_chunk_rank = None
            first_article_rank = None
            first_doc_rank = None
            for i, d in enumerate(cands, start=1):
                md = d.metadata or {}
                cdoc = md.get("doc_id") or md.get("source") or ""
                cart = str(md.get("article_number") or "")
                if first_doc_rank is None and cdoc == gold_doc:
                    first_doc_rank = i
                if first_article_rank is None and cdoc == gold_doc and cart == gold_art:
                    first_article_rank = i
                if first_chunk_rank is None and \
                        scorer.score(gold_ctx, d.page_content)["rougeL"].fmeasure >= 0.5:
                    first_chunk_rank = i
                if first_chunk_rank and first_article_rank and first_doc_rank:
                    break

            ranks_chunk.append(first_chunk_rank)
            ranks_article.append(first_article_rank)
            for k in recall_chunk:
                if first_chunk_rank and first_chunk_rank <= k:
                    recall_chunk[k] += 1
                if first_article_rank and first_article_rank <= k:
                    recall_article[k] += 1
                if first_doc_rank and first_doc_rank <= k:
                    recall_doc[k] += 1

            per_doc[gold_doc]["total"] += 1
            if first_chunk_rank and first_chunk_rank <= 5:
                per_doc[gold_doc]["hit_chunk_at_5"] += 1

        print(f"  {'K':<5}{'chunk@K':<10}{'article@K':<12}{'doc@K':<10}")
        for k in (1, 3, 5, 10, 20, 30, 50):
            print(f"  {k:<5}{recall_chunk[k]/n_total:<10.4f}{recall_article[k]/n_total:<12.4f}{recall_doc[k]/n_total:<10.4f}")

        print("  rank-of-gold-chunk distribution (None = not in top-50):")
        c = Counter(ranks_chunk)
        for bucket in [(1,1), (2,3), (4,5), (6,10), (11,20), (21,50)]:
            n = sum(c[i] for i in range(bucket[0], bucket[1]+1))
            print(f"    rank {bucket[0]:>2}-{bucket[1]:>2}: {n:3d} ({n/n_total:.3f})")
        print(f"    rank None : {c[None]:3d} ({c[None]/n_total:.3f})")

        print("  per-doc chunk@5 (sorted by accuracy):")
        rows_d = sorted(per_doc.items(), key=lambda x: x[1]["hit_chunk_at_5"]/max(1,x[1]["total"]))
        for doc, d in rows_d:
            if d["total"] >= 5:
                acc = d["hit_chunk_at_5"]/d["total"]
                print(f"    {doc:<28} {acc:.3f}  ({d['hit_chunk_at_5']:>3d}/{d['total']:>3d})")


if __name__ == "__main__":
    main()
