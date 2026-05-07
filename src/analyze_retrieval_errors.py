"""Analyze retrieval diagnostics and group failure modes."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIAG_PATH = ROOT / "reports" / "traffic" / "retrieval_diagnostics.json"
OUT_MD = ROOT / "reports" / "traffic" / "retrieval_error_analysis.md"
OUT_JSON = ROOT / "reports" / "traffic" / "retrieval_error_analysis.json"


def _doc_id(item: dict) -> str:
    return item.get("doc_id") or item.get("source") or ""


def _article(item: dict) -> str:
    return item.get("article") or ""


def _classify(sample: dict) -> list[str]:
    labels: list[str] = []
    expected = set(sample.get("expected_doc_ids") or [])
    retrieved = sample.get("retrieved") or []
    docs = [_doc_id(x) for x in retrieved]
    articles = [_article(x) for x in retrieved]
    q = (sample.get("question") or "").lower()

    if expected and not sample.get("source_hit"):
        labels.append("wrong_doc")
    if expected and sample.get("source_hit") and not sample.get("context_rouge_hit"):
        labels.append("right_doc_wrong_context")
    if not expected and not sample.get("context_rouge_hit"):
        labels.append("unlabeled_context_miss")
    if "xe máy" in q or "mô tô" in q or "gắn máy" in q:
        if any("ô tô" in (a or "").lower() for a in articles):
            labels.append("bike_car_confusion")
    if "ô tô" in q or "oto" in q or "xe hơi" in q:
        if any(("mô tô" in (a or "").lower() or "gắn máy" in (a or "").lower()) for a in articles):
            labels.append("car_bike_confusion")
    if any(x in q for x in ["hiệu lực", "khi nào có hiệu lực"]):
        if not any("hiệu lực" in (a or "").lower() for a in articles):
            labels.append("effective_date_miss")
    if any(x in q for x in ["nồng độ cồn", "rượu", "bia"]):
        if not any("nồng độ cồn" in json.dumps(r, ensure_ascii=False).lower() for r in retrieved):
            labels.append("alcohol_miss")
    if any(x in q for x in ["đăng kiểm", "kiểm định"]):
        if not any("đăng kiểm" in json.dumps(r, ensure_ascii=False).lower() or "kiểm định" in json.dumps(r, ensure_ascii=False).lower() for r in retrieved):
            labels.append("registration_inspection_miss")
    if any(x in q for x in ["đường thủy", "tàu", "thuyền", "phương tiện thủy"]):
        if not any(d in {"nd_158_2024_nd_cp", "nd_336_2025_nd_cp"} for d in docs):
            labels.append("waterway_miss")
    return labels or ["other"]


def main() -> None:
    data = json.loads(DIAG_PATH.read_text(encoding="utf-8"))
    samples = data.get("samples", [])
    failures = []
    counters = Counter()
    by_expected = defaultdict(Counter)

    for s in samples:
        source_fail = s.get("source_hit") is False
        context_fail = s.get("context_rouge_hit") is False
        if not source_fail and not context_fail:
            continue
        labels = _classify(s)
        failures.append({**s, "error_labels": labels})
        counters.update(labels)
        for doc in s.get("expected_doc_ids") or ["<unlabeled>"]:
            by_expected[doc].update(labels)

    out = {
        "summary": data.get("summary", {}),
        "n_failures": len(failures),
        "error_type_counts": dict(counters),
        "error_by_expected_doc": {k: dict(v) for k, v in by_expected.items()},
        "failures": failures,
    }
    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# Retrieval Error Analysis", "", "## Current Summary", ""]
    for k, v in (data.get("summary") or {}).items():
        if k != "source_hit_rate_by_doc":
            lines.append(f"- `{k}`: `{v}`")
    lines += ["", "## Error Types", ""]
    for k, v in counters.most_common():
        lines.append(f"- `{k}`: {v}")
    lines += ["", "## Errors By Expected Doc", ""]
    for doc, cnt in sorted(by_expected.items()):
        lines.append(f"- `{doc}`: " + ", ".join(f"{k}={v}" for k, v in cnt.most_common()))
    lines += ["", "## Top Failure Samples", ""]
    for s in failures[:80]:
        expected = ", ".join(s.get("expected_doc_ids") or []) or "<none>"
        retrieved = "; ".join(
            f"{r.get('rank')}:{_doc_id(r)}:{_article(r)[:70]}" for r in (s.get("retrieved") or [])[:5]
        )
        lines += [
            f"### #{s.get('index')} {'/'.join(s.get('error_labels', []))}",
            f"- Q: {s.get('question')}",
            f"- Expected: `{expected}` | source_hit={s.get('source_hit')} | context_hit={s.get('context_rouge_hit')}",
            f"- Retrieved: {retrieved}",
            "",
        ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT_MD}")
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
