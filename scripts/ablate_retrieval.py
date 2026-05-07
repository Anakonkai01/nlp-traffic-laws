"""
Ablation harness for retrieval configurations A0..A8.

Each config is a callable retriever ``fn(question, top_k) -> list[Document]``.
For each eval-set we compute Recall@5 (context-rouge for QA-style, doc-id for
source-style), MRR@10 (context-rouge), Source-Hit@3 (when expected_doc_ids is
present) and p95 latency.

Run from nlp/:
    python scripts/ablate_retrieval.py --configs A0 \\
        --dev data/splits/qa_dev.jsonl --manual data/eval_manual.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

# Make ``nlp/src`` importable when running from ``nlp/``.
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from rouge_score import rouge_scorer  # type: ignore

from build_kb import load_vectorstore  # type: ignore

REPORT_DIR = ROOT / "reports" / "traffic"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
ABLATION_CSV = REPORT_DIR / "retrieval_ablation.csv"


# ---------------------------------------------------------------------------
# Retriever factories
# ---------------------------------------------------------------------------
def _make_v2(vs):
    from retrieval import retrieve_ranked_docs

    def fn(question: str, top_k: int):
        return retrieve_ranked_docs(vs, question, top_k=top_k, candidate_k=max(30, top_k * 8))

    return fn


def _make_v3(vs, *, use_bm25=True, use_alias=False, use_ce=False, ce_model=None,
             use_article_window=False, use_article_constraint=False):
    import retrieval_v3 as r3

    cfg = r3.RetrievalConfig(
        use_bm25=use_bm25,
        use_alias=use_alias,
        use_ce=use_ce,
        ce_model=ce_model,
        use_article_window=use_article_window,
        use_article_constraint=use_article_constraint,
    )
    retriever = r3.build_retriever(vs, cfg)

    def fn(question: str, top_k: int):
        return retriever(question, top_k=top_k)

    return fn


def make_retriever(config: str, vs):
    if config == "A0":
        return _make_v2(vs)
    if config == "A1":
        return _make_v3(vs, use_bm25=False)
    if config == "A2":
        return _make_v3(vs)
    if config == "A3":
        return _make_v3(vs, use_alias=True)
    if config == "A4":
        return _make_v3(vs, use_alias=True, use_ce=True)
    if config == "A5":
        ce_path = os.environ.get("RETRIEVAL_CE_MODEL", "models/bge-reranker-v2-m3-traffic-ft")
        return _make_v3(vs, use_alias=True, use_ce=True, ce_model=ce_path)
    # A4/A5 showed CE (pretrained or FT) is net-negative for this corpus. A6/A7/A8 build on A2 (no CE).
    if config == "A6":
        return _make_v3(vs, use_article_window=True)
    if config == "A7":
        return _make_v3(vs, use_article_constraint=True)
    if config == "A8":
        return _make_v3(vs, use_article_window=True, use_article_constraint=True)
    raise ValueError(f"unknown config: {config}")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _expected_doc_ids(item: dict) -> set[str]:
    eids = item.get("expected_doc_ids") or []
    if not eids and item.get("doc_id"):
        eids = [item["doc_id"]]
    return {e for e in eids if isinstance(e, str) and e}


def _doc_id(doc):
    md = doc.metadata or {}
    return md.get("doc_id") or md.get("source")


def _is_context_hit(scorer, gold_ctx: str, doc) -> bool:
    if not gold_ctx:
        return False
    return scorer.score(gold_ctx, doc.page_content)["rougeL"].fmeasure >= 0.5


def evaluate_set(name: str, items: list[dict], retrieve_fn, k_recall: int = 5,
                 k_mrr: int = 10, k_source: int = 3) -> dict:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    ctx_hits = ctx_total = 0
    src_hits = src_total = 0
    src3_hits = src3_total = 0
    rr_sum = rr_n = 0
    latencies = []

    for item in items:
        question = item.get("question") or ""
        gold_ctx = item.get("context") or ""
        expected = _expected_doc_ids(item)

        t0 = time.perf_counter()
        topk = max(k_recall, k_mrr)
        docs = retrieve_fn(question, topk)
        latencies.append((time.perf_counter() - t0) * 1000.0)

        if gold_ctx:
            ctx_total += 1
            top5 = docs[:k_recall]
            if any(_is_context_hit(scorer, gold_ctx, d) for d in top5):
                ctx_hits += 1
            rank_hit = None
            for i, d in enumerate(docs[:k_mrr], start=1):
                if _is_context_hit(scorer, gold_ctx, d):
                    rank_hit = i
                    break
            rr_n += 1
            rr_sum += (1.0 / rank_hit) if rank_hit else 0.0

        if expected:
            src_total += 1
            top5_ids = {_doc_id(d) for d in docs[:k_recall] if _doc_id(d)}
            if expected & top5_ids:
                src_hits += 1
            src3_total += 1
            top3_ids = {_doc_id(d) for d in docs[:k_source] if _doc_id(d)}
            if expected & top3_ids:
                src3_hits += 1

    latencies.sort()

    def pct(p):
        if not latencies:
            return None
        idx = min(len(latencies) - 1, int(round(p * (len(latencies) - 1))))
        return round(latencies[idx], 1)

    return {
        "set": name,
        "n": len(items),
        "context_recall_at_5": round(ctx_hits / ctx_total, 4) if ctx_total else None,
        "context_recall_n": ctx_total,
        "source_recall_at_5": round(src_hits / src_total, 4) if src_total else None,
        "source_recall_n": src_total,
        "source_hit_at_3": round(src3_hits / src3_total, 4) if src3_total else None,
        "mrr_at_10": round(rr_sum / rr_n, 4) if rr_n else None,
        "p50_ms": pct(0.50),
        "p95_ms": pct(0.95),
        "p99_ms": pct(0.99),
    }


# ---------------------------------------------------------------------------
# CSV / runner
# ---------------------------------------------------------------------------
CSV_HEADER = [
    "config", "set", "n",
    "context_recall_at_5", "source_recall_at_5", "source_hit_at_3",
    "mrr_at_10", "p50_ms", "p95_ms", "p99_ms",
]


def append_csv(rows: list[dict]):
    new = not ABLATION_CSV.exists()
    with open(ABLATION_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_HEADER, extrasaction="ignore")
        if new:
            w.writeheader()
        for r in rows:
            w.writerow(r)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=["A0"])
    ap.add_argument("--dev", default="data/splits/qa_dev.jsonl")
    ap.add_argument("--manual", default="data/eval_manual.jsonl")
    ap.add_argument("--eval-on", default=None,
                    help="Override eval set; if given, only this set is scored.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Truncate each eval set for smoke runs.")
    args = ap.parse_args()

    vs = load_vectorstore()
    print(f"[ok] vectorstore loaded; chunks ≈ {vs.index.ntotal}")

    sets: list[tuple[str, list[dict]]] = []
    if args.eval_on:
        sets.append((Path(args.eval_on).stem, load_jsonl(Path(args.eval_on))))
    else:
        if Path(args.dev).exists():
            sets.append(("qa_dev", load_jsonl(Path(args.dev))))
        if Path(args.manual).exists():
            sets.append(("eval_manual", load_jsonl(Path(args.manual))))

    if args.limit:
        sets = [(n, items[: args.limit]) for n, items in sets]

    rows: list[dict] = []
    for cfg in args.configs:
        retrieve_fn = make_retriever(cfg, vs)
        for set_name, items in sets:
            print(f"\n[run] cfg={cfg} set={set_name} n={len(items)}")
            metrics = evaluate_set(set_name, items, retrieve_fn)
            metrics["config"] = cfg
            rows.append(metrics)
            print(f"  ctx@5={metrics['context_recall_at_5']}  "
                  f"src@5={metrics['source_recall_at_5']}  "
                  f"src@3={metrics['source_hit_at_3']}  "
                  f"mrr@10={metrics['mrr_at_10']}  "
                  f"p50={metrics['p50_ms']}ms p95={metrics['p95_ms']}ms")

    append_csv(rows)
    print(f"\n[ok] appended {len(rows)} rows → {ABLATION_CSV}")


if __name__ == "__main__":
    main()
