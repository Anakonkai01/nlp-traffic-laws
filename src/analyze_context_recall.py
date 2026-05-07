"""Analyze context-level recall failures and oracle options."""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from rouge_score import rouge_scorer

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from build_kb import load_vectorstore  # noqa: E402

DIAG_PATH = ROOT / "reports" / "traffic" / "retrieval_diagnostics.json"
OUT_MD = ROOT / "reports" / "traffic" / "context_error_analysis.md"
OUT_JSON = ROOT / "reports" / "traffic" / "context_error_analysis.json"
THRESHOLD = 0.5


def _doc_id(doc) -> str:
    md = doc.metadata or {}
    return md.get("doc_id") or md.get("source") or ""


def _article_key_meta(md: dict) -> tuple[str, str, str]:
    return (md.get("doc_id") or md.get("source") or "", str(md.get("article_number") or ""), md.get("article") or "")


def _chunk_ord(md: dict) -> int | None:
    cid = md.get("chunk_id") or ""
    try:
        return int(cid.rsplit(":", 1)[1])
    except Exception:
        return None


def main() -> None:
    diag = json.loads(DIAG_PATH.read_text(encoding="utf-8"))
    vs = load_vectorstore()
    all_docs = list(getattr(vs.docstore, "_dict", {}).values())
    by_sig = {((d.metadata or {}).get("chunk_id")): d for d in all_docs if (d.metadata or {}).get("chunk_id")}
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    failures = []
    counts = Counter()
    by_doc = defaultdict(Counter)
    oracle_same_doc = 0
    oracle_same_article = 0
    neighbor_fix = 0
    total_ctx_fail = 0

    for sample in diag.get("samples", []):
        if sample.get("context_rouge_hit") is not False:
            continue
        ctx = sample.get("context") or ""
        # diagnostics rows do not include context by default; recover from eval file by index if needed later.
        total_ctx_fail += 1
        expected = set(sample.get("expected_doc_ids") or [])
        retrieved = sample.get("retrieved") or []
        top_scores = []
        for r in retrieved:
            cid = r.get("chunk_id")
            doc = by_sig.get(cid)
            score = scorer.score(ctx, doc.page_content)["rougeL"].fmeasure if ctx and doc else 0.0
            top_scores.append({"rank": r.get("rank"), "chunk_id": cid, "doc_id": r.get("doc_id"), "article": r.get("article"), "rouge": round(score, 4)})

        same_doc_best = {"rouge": 0.0}
        same_article_best = {"rouge": 0.0}
        top_articles = {_article_key_meta(r) for r in retrieved}
        top_chunk_ords = []
        for r in retrieved:
            ordv = _chunk_ord(r)
            if ordv is not None:
                top_chunk_ords.append((_doc_id(type("X", (), {"metadata": r})()), ordv, _article_key_meta(r)))

        if ctx and expected:
            for doc in all_docs:
                md = doc.metadata or {}
                did = _doc_id(doc)
                if did not in expected:
                    continue
                score = scorer.score(ctx, doc.page_content)["rougeL"].fmeasure
                if score > same_doc_best["rouge"]:
                    same_doc_best = {"rouge": score, "chunk_id": md.get("chunk_id"), "article": md.get("article"), "doc_id": did}
                if _article_key_meta(md) in top_articles and score > same_article_best["rouge"]:
                    same_article_best = {"rouge": score, "chunk_id": md.get("chunk_id"), "article": md.get("article"), "doc_id": did}

        if same_doc_best["rouge"] >= THRESHOLD:
            oracle_same_doc += 1
        if same_article_best["rouge"] >= THRESHOLD:
            oracle_same_article += 1

        neighbor_would_fix = False
        best_cid = same_doc_best.get("chunk_id")
        if best_cid:
            best_ord = _chunk_ord({"chunk_id": best_cid})
            best_doc = same_doc_best.get("doc_id")
            for r in retrieved:
                r_ord = _chunk_ord(r)
                if r.get("doc_id") == best_doc and r_ord is not None and best_ord is not None and abs(r_ord - best_ord) <= 1:
                    neighbor_would_fix = True
                    break
        if neighbor_would_fix:
            neighbor_fix += 1

        if not expected:
            ftype = "unlabeled"
        elif same_article_best["rouge"] >= THRESHOLD:
            ftype = "same_article_window_needed"
        elif neighbor_would_fix:
            ftype = "neighbor_window_needed"
        elif same_doc_best["rouge"] >= THRESHOLD:
            ftype = "wrong_article_in_right_doc"
        else:
            ftype = "context_not_in_chunk_store_or_low_overlap"
        counts[ftype] += 1
        for doc_id in expected or ["<none>"]:
            by_doc[doc_id][ftype] += 1
        failures.append({
            "index": sample.get("index"),
            "question": sample.get("question"),
            "expected_doc_ids": sorted(expected),
            "failure_type": ftype,
            "top_scores": top_scores,
            "best_same_doc": {**same_doc_best, "rouge": round(same_doc_best.get("rouge", 0), 4)},
            "best_same_article": {**same_article_best, "rouge": round(same_article_best.get("rouge", 0), 4)},
            "neighbor_would_fix": neighbor_would_fix,
        })

    out = {
        "diagnostics_summary": diag.get("summary", {}),
        "n_context_failures": total_ctx_fail,
        "failure_type_counts": dict(counts),
        "oracle_same_doc_fixable": oracle_same_doc,
        "oracle_same_article_fixable": oracle_same_article,
        "neighbor_fixable": neighbor_fix,
        "by_expected_doc": {k: dict(v) for k, v in by_doc.items()},
        "failures": failures,
    }
    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# Context Recall Error Analysis", "", "## Summary", ""]
    for k, v in out["diagnostics_summary"].items():
        if k != "source_hit_rate_by_doc":
            lines.append(f"- `{k}`: `{v}`")
    lines += [
        f"- `n_context_failures`: `{total_ctx_fail}`",
        f"- `oracle_same_doc_fixable`: `{oracle_same_doc}`",
        f"- `oracle_same_article_fixable`: `{oracle_same_article}`",
        f"- `neighbor_fixable`: `{neighbor_fix}`",
        "", "## Failure Types", "",
    ]
    for k, v in counts.most_common():
        lines.append(f"- `{k}`: {v}")
    lines += ["", "## Top Failures", ""]
    for f in failures[:60]:
        lines += [
            f"### #{f['index']} {f['failure_type']}",
            f"- Q: {f['question']}",
            f"- Expected: {', '.join(f['expected_doc_ids'])}",
            f"- Best same-doc: {f['best_same_doc']}",
            f"- Best same-article: {f['best_same_article']}",
            f"- Neighbor fixable: {f['neighbor_would_fix']}",
            "",
        ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ["n_context_failures", "failure_type_counts", "oracle_same_doc_fixable", "oracle_same_article_fixable", "neighbor_fixable"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
