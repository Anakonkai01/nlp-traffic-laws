"""
Tune Retrieval v4 on a validation set, then report the selected config on test.

This script deliberately separates model selection (dev) from final reporting
(test) to avoid tuning retrieval weights on the held-out set.
"""
from __future__ import annotations

import argparse
import csv
import itertools
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
TUNING_CSV = REPORT_DIR / "retrieval_v4_tuning.csv"
BEST_JSON = REPORT_DIR / "retrieval_v4_best_config.json"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _doc_id(doc):
    md = doc.metadata or {}
    return md.get("doc_id") or md.get("source")


def _expected_doc_ids(item: dict) -> set[str]:
    eids = item.get("expected_doc_ids") or []
    if not eids and item.get("doc_id"):
        eids = [item["doc_id"]]
    return {x for x in eids if isinstance(x, str) and x}


def _is_context_hit(scorer, gold_ctx: str, doc) -> bool:
    if not gold_ctx:
        return False
    return scorer.score(gold_ctx, doc.page_content or "")["rougeL"].fmeasure >= 0.5


def evaluate(items: list[dict], retrieve_fn, k_recall: int = 5, k_mrr: int = 10) -> dict:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    ctx_hits = ctx_total = src_hits = src_total = 0
    rr_sum = rr_n = 0
    latencies = []
    for item in items:
        question = item.get("question") or ""
        gold_ctx = item.get("context") or ""
        expected = _expected_doc_ids(item)
        t0 = time.perf_counter()
        docs = retrieve_fn(question, max(k_recall, k_mrr))
        latencies.append((time.perf_counter() - t0) * 1000.0)
        if gold_ctx:
            ctx_total += 1
            if any(_is_context_hit(scorer, gold_ctx, d) for d in docs[:k_recall]):
                ctx_hits += 1
            rank_hit = None
            for rank, doc in enumerate(docs[:k_mrr], start=1):
                if _is_context_hit(scorer, gold_ctx, doc):
                    rank_hit = rank
                    break
            rr_n += 1
            rr_sum += (1.0 / rank_hit) if rank_hit else 0.0
        if expected:
            src_total += 1
            top_ids = {_doc_id(d) for d in docs[:k_recall] if _doc_id(d)}
            if expected & top_ids:
                src_hits += 1
    latencies.sort()

    def pct(p: float):
        if not latencies:
            return None
        idx = min(len(latencies) - 1, int(round(p * (len(latencies) - 1))))
        return round(latencies[idx], 1)

    return {
        "context_recall_at_5": round(ctx_hits / ctx_total, 4) if ctx_total else None,
        "source_recall_at_5": round(src_hits / src_total, 4) if src_total else None,
        "mrr_at_10": round(rr_sum / rr_n, 4) if rr_n else None,
        "p50_ms": pct(0.50),
        "p95_ms": pct(0.95),
    }


def config_grid(args):
    weights = [
        (0.70, 0.30),
        (0.60, 0.40),
        (0.55, 0.35),
        (0.50, 0.50),
        (0.45, 0.55),
    ]
    if args.small:
        weights = [(0.60, 0.40), (0.55, 0.35), (0.50, 0.50)]
    for (w_dense, w_sparse), rrf_k, fuse_k, window_mode, doc_agg, radius in itertools.product(
        weights,
        [10, 20, 60] if not args.small else [20, 60],
        [30, 40, 60] if not args.small else [40],
        ["off", "selective", "all"],
        [False, True],
        [1, 2] if not args.small else [1],
    ):
        if window_mode == "off" and radius != 1:
            continue
        yield {
            "dense_k": 50,
            "sparse_k": 50,
            "fuse_k": fuse_k,
            "rrf_k": rrf_k,
            "w_dense": w_dense,
            "w_sparse": w_sparse,
            "w_alias": 0.03,
            "w_article": 0.07,
            "use_bm25": True,
            "use_alias": True,
            "use_article": True,
            "use_doc_aggregation": doc_agg,
            "doc_aggregation_top_n": 4,
            "use_article_window": window_mode != "off",
            "article_window_radius": radius,
            "article_window_policy": "all" if window_mode == "all" else "selective",
        }


def selection_score(row: dict) -> float:
    # Primary objective is context recall. MRR breaks ties; large source loss is penalized.
    ctx = row.get("dev_context_recall_at_5") or 0.0
    mrr = row.get("dev_mrr_at_10") or 0.0
    src = row.get("dev_source_recall_at_5") or 0.0
    return ctx + 0.10 * mrr + 0.05 * src


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", default="data/splits_filtered/qa_dev.jsonl")
    parser.add_argument("--test", default="data/splits_filtered/qa_test.jsonl")
    parser.add_argument("--limit-dev", type=int, default=None)
    parser.add_argument("--limit-test", type=int, default=None)
    parser.add_argument("--small", action="store_true", help="Run a smaller grid for quick iteration.")
    args = parser.parse_args()

    dev = load_jsonl(ROOT / args.dev)
    test = load_jsonl(ROOT / args.test)
    if args.limit_dev:
        dev = dev[: args.limit_dev]
    if args.limit_test:
        test = test[: args.limit_test]

    vs = load_vectorstore()
    print(f"[ok] vectorstore loaded; chunks={vs.index.ntotal}; dev={len(dev)} test={len(test)}")

    rows = []
    best = None
    for idx, cfg_dict in enumerate(config_grid(args), start=1):
        cfg = RetrievalV4Config(**cfg_dict)
        retrieve_fn = build_retriever(vs, cfg)
        dev_metrics = evaluate(dev, retrieve_fn)
        row = {"run": idx, **cfg_dict, **{f"dev_{k}": v for k, v in dev_metrics.items()}}
        row["selection_score"] = round(selection_score(row), 6)
        rows.append(row)
        if best is None or selection_score(row) > selection_score(best):
            best = row
        print(
            f"[{idx:03d}] ctx={dev_metrics['context_recall_at_5']} src={dev_metrics['source_recall_at_5']} "
            f"mrr={dev_metrics['mrr_at_10']} score={row['selection_score']} cfg="
            f"wd={cfg.w_dense} ws={cfg.w_sparse} rrf={cfg.rrf_k} fuse={cfg.fuse_k} "
            f"docagg={cfg.use_doc_aggregation} win={cfg.use_article_window}/{cfg.article_window_policy} rad={cfg.article_window_radius}"
        )

    assert best is not None
    best_cfg_keys = [
        "dense_k", "sparse_k", "fuse_k", "rrf_k", "w_dense", "w_sparse", "w_alias", "w_article",
        "use_bm25", "use_alias", "use_article", "use_doc_aggregation", "doc_aggregation_top_n", "use_article_window", "article_window_radius", "article_window_policy",
    ]
    best_cfg = {k: best[k] for k in best_cfg_keys}
    test_fn = build_retriever(vs, RetrievalV4Config(**best_cfg))
    test_metrics = evaluate(test, test_fn)
    result = {"selected_on": args.dev, "tested_on": args.test, "best_config": best_cfg, "dev": best, "test": test_metrics}

    fieldnames = list(rows[0].keys())
    with open(TUNING_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    BEST_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2))

    print("\n[best-dev]")
    print(json.dumps(best, ensure_ascii=False, indent=2))
    print("\n[test-metrics]")
    print(json.dumps(test_metrics, ensure_ascii=False, indent=2))
    print(f"\n[ok] wrote {TUNING_CSV}")
    print(f"[ok] wrote {BEST_JSON}")


if __name__ == "__main__":
    main()
