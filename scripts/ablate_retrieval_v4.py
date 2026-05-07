"""
Ablation harness for Retrieval v4.

Run from nlp/:
    python scripts/ablate_retrieval_v4.py --limit 20
    python scripts/ablate_retrieval_v4.py --configs dense_only bm25_only v4_full
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from rouge_score import rouge_scorer  # type: ignore

from build_kb import load_vectorstore  # type: ignore
from retrieval_v4 import RetrievalV4Config, build_retriever  # type: ignore

REPORT_DIR = ROOT / "reports" / "traffic"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
ABLATION_CSV = REPORT_DIR / "retrieval_v4_ablation.csv"
DIAG_JSON = REPORT_DIR / "retrieval_v4_diagnostics.json"


def _make_v2(vs):
    import os
    os.environ.pop("RETRIEVAL_VERSION", None)
    from retrieval import retrieve_ranked_docs

    def fn(question: str, top_k: int):
        return retrieve_ranked_docs(vs, question, top_k=top_k, candidate_k=max(40, top_k * 8))

    return fn


def _make_v4(vs, cfg: RetrievalV4Config):
    retriever = build_retriever(vs, cfg)

    def fn(question: str, top_k: int):
        return retriever(question, top_k=top_k)

    return fn


def make_retriever(config: str, vs):
    base = dict(dense_k=50, sparse_k=50, fuse_k=40, rrf_k=20)
    if config == "v2_heuristic_reference":
        return _make_v2(vs)
    if config == "dense_only":
        return _make_v4(vs, RetrievalV4Config(**base, use_bm25=False, use_alias=False, use_article=False, use_article_window=False))
    if config == "bm25_only":
        return _make_v4(vs, RetrievalV4Config(**base, w_dense=0.0, w_sparse=1.0, use_bm25=True, use_alias=False, use_article=False, use_article_window=False))
    if config == "dense_bm25_rrf":
        return _make_v4(vs, RetrievalV4Config(**base, use_alias=False, use_article=False, use_article_window=False))
    if config == "plus_alias":
        return _make_v4(vs, RetrievalV4Config(**base, use_alias=True, use_article=False, use_article_window=False))
    if config == "plus_article":
        return _make_v4(vs, RetrievalV4Config(**base, use_alias=True, use_article=True, use_article_window=False))
    if config == "v4_full":
        return _make_v4(vs, RetrievalV4Config(**base, use_alias=True, use_article=True, use_article_window=True))
    if config == "v4_full_ce":
        return _make_v4(vs, RetrievalV4Config(**base, use_alias=True, use_article=True, use_article_window=True, use_ce=True))
    raise ValueError(f"unknown config: {config}")


def _expected_doc_ids(item: dict) -> set[str]:
    eids = item.get("expected_doc_ids") or []
    if not eids and item.get("doc_id"):
        eids = [item["doc_id"]]
    return {e for e in eids if isinstance(e, str) and e}


def _doc_id(doc):
    md = doc.metadata or {}
    return md.get("doc_id") or md.get("source")


def _chunk_id(doc) -> str:
    md = doc.metadata or {}
    return str(md.get("chunk_id") or f"{_doc_id(doc) or 'unknown'}:{hash((doc.page_content or '')[:200]) & 0xffffffff:08x}")


def _is_context_hit(scorer, gold_ctx: str, doc) -> bool:
    if not gold_ctx:
        return False
    return scorer.score(gold_ctx, doc.page_content or "")["rougeL"].fmeasure >= 0.5


def evaluate_set(name: str, items: list[dict], retrieve_fn, k_recall: int = 5, k_mrr: int = 10) -> dict:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    ctx_hits = ctx_total = 0
    src_hits = src_total = 0
    rr_sum = rr_n = 0
    latencies = []
    article_counts = []
    duplicate_rates = []

    for item in items:
        question = item.get("question") or ""
        gold_ctx = item.get("context") or ""
        expected = _expected_doc_ids(item)

        t0 = time.perf_counter()
        docs = retrieve_fn(question, max(k_recall, k_mrr))
        latencies.append((time.perf_counter() - t0) * 1000.0)

        if gold_ctx:
            ctx_total += 1
            if any(_is_context_hit(scorer, gold_ctx, doc) for doc in docs[:k_recall]):
                ctx_hits += 1
            rank_hit = None
            for idx, doc in enumerate(docs[:k_mrr], start=1):
                if _is_context_hit(scorer, gold_ctx, doc):
                    rank_hit = idx
                    break
            rr_n += 1
            rr_sum += (1.0 / rank_hit) if rank_hit else 0.0

        if expected:
            src_total += 1
            top_ids = {_doc_id(doc) for doc in docs[:k_recall] if _doc_id(doc)}
            if expected & top_ids:
                src_hits += 1

        top_docs = docs[:k_recall]
        article_keys = {(_doc_id(doc), str((doc.metadata or {}).get("article_number") or "")) for doc in top_docs}
        article_counts.append(len(article_keys))
        chunk_ids = [_chunk_id(doc) for doc in top_docs]
        duplicate_rates.append(1.0 - (len(set(chunk_ids)) / len(chunk_ids)) if chunk_ids else 0.0)

    latencies.sort()

    def pct(p):
        if not latencies:
            return None
        idx = min(len(latencies) - 1, int(round(p * (len(latencies) - 1))))
        return round(latencies[idx], 1)

    def avg(xs):
        return round(sum(xs) / len(xs), 4) if xs else None

    return {
        "set": name,
        "n": len(items),
        "context_recall_at_5": round(ctx_hits / ctx_total, 4) if ctx_total else None,
        "source_recall_at_5": round(src_hits / src_total, 4) if src_total else None,
        "mrr_at_10": round(rr_sum / rr_n, 4) if rr_n else None,
        "p50_ms": pct(0.50),
        "p95_ms": pct(0.95),
        "avg_articles_at_5": avg(article_counts),
        "avg_duplicate_rate_at_5": avg(duplicate_rates),
    }


CSV_HEADER = [
    "config", "set", "n", "context_recall_at_5", "source_recall_at_5", "mrr_at_10",
    "p50_ms", "p95_ms", "avg_articles_at_5", "avg_duplicate_rate_at_5",
]


def write_csv(rows: list[dict]):
    with open(ABLATION_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", default=["dense_only", "bm25_only", "dense_bm25_rrf", "plus_alias", "plus_article", "v4_full", "v2_heuristic_reference"])
    parser.add_argument("--manual", default="data/eval_manual.jsonl")
    parser.add_argument("--test", default="data/splits_filtered/qa_test.jsonl")
    parser.add_argument("--eval-on", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    vs = load_vectorstore()
    print(f"[ok] vectorstore loaded; chunks={vs.index.ntotal}")

    sets: list[tuple[str, list[dict]]] = []
    if args.eval_on:
        path = Path(args.eval_on)
        sets.append((path.stem, load_jsonl(path)))
    else:
        for name, raw_path in [("eval_manual", args.manual), ("qa_test_filtered", args.test)]:
            path = ROOT / raw_path
            if path.exists():
                sets.append((name, load_jsonl(path)))

    if args.limit:
        sets = [(name, items[: args.limit]) for name, items in sets]

    rows: list[dict] = []
    diagnostics = {"configs": args.configs, "sets": {name: len(items) for name, items in sets}, "rows": []}
    for config in args.configs:
        retrieve_fn = make_retriever(config, vs)
        for set_name, items in sets:
            print(f"\n[run] config={config} set={set_name} n={len(items)}")
            metrics = evaluate_set(set_name, items, retrieve_fn)
            metrics["config"] = config
            rows.append(metrics)
            diagnostics["rows"].append(metrics)
            print(
                f"  ctx@5={metrics['context_recall_at_5']} src@5={metrics['source_recall_at_5']} "
                f"mrr@10={metrics['mrr_at_10']} p50={metrics['p50_ms']}ms p95={metrics['p95_ms']}ms"
            )

    write_csv(rows)
    DIAG_JSON.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2))
    print(f"\n[ok] wrote {ABLATION_CSV}")
    print(f"[ok] wrote {DIAG_JSON}")


if __name__ == "__main__":
    main()
