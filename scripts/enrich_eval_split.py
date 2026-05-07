#!/usr/bin/env python3
"""Add evaluation metadata derived from QA split rows.

The script does not change question/answer/context content. It only adds source
and coarse category fields so retrieval/source/refusal metrics can be computed.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

ARTICLE_RE = re.compile(r"[Đđ]i\S*u\s+(\d+)")

PENALTY_TERMS = [
    "phạt", "mức phạt", "bao nhiêu tiền", "trừ điểm", "tước quyền", "xử phạt",
    "bị xử lý", "tịch thu", "cảnh cáo",
]
DEFINITION_TERMS = ["là gì", "khái niệm", "được hiểu", "định nghĩa", "thế nào là"]
PROCEDURE_TERMS = [
    "thủ tục", "hồ sơ", "làm sao", "như thế nào để", "cấp", "đổi", "cấp lại",
    "đăng ký", "phục hồi", "kiểm tra", "trình tự", "nộp",
]
PROHIBITED_TERMS = ["có được", "được phép", "không được", "bị cấm", "cấm"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def article_number(row: dict[str, Any]) -> str | None:
    value = row.get("article_number")
    if value not in (None, ""):
        return str(value)
    text = str(row.get("article") or row.get("context") or "")
    m = ARTICLE_RE.search(text)
    return m.group(1) if m else None


def infer_category(row: dict[str, Any]) -> tuple[str, str]:
    if row.get("corpus") == "negative":
        return "unsupported", "corpus"
    text = f"{row.get('question') or ''} {row.get('answer') or ''}".lower()
    if any(term in text for term in PENALTY_TERMS):
        return "penalty", "auto_heuristic_v1"
    if any(term in text for term in DEFINITION_TERMS):
        return "definition", "auto_heuristic_v1"
    if any(term in text for term in PROCEDURE_TERMS):
        return "procedure", "auto_heuristic_v1"
    if any(term in text for term in PROHIBITED_TERMS):
        return "prohibited", "auto_heuristic_v1"
    return "general", "auto_heuristic_v1"


def enrich(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    if row.get("corpus") == "negative":
        out["expected_doc_ids"] = []
        out["expected_article_numbers"] = []
    else:
        doc_id = row.get("doc_id") or row.get("source")
        out["expected_doc_ids"] = [str(doc_id)] if doc_id else []
        art = article_number(row)
        out["expected_article_numbers"] = [art] if art else []
    category, source = infer_category(row)
    out["category"] = category
    out["category_source"] = source
    out["expected_source_source"] = "derived_from_gold_context_metadata"
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    rows = read_jsonl(args.input)
    enriched = [enrich(row) for row in rows]
    write_jsonl(args.output, enriched)

    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "n": len(enriched),
        "category_counts": dict(Counter(row.get("category") for row in enriched)),
        "corpus_counts": dict(Counter(row.get("corpus") for row in enriched)),
        "expected_doc_rows": sum(bool(row.get("expected_doc_ids")) for row in enriched),
        "expected_article_rows": sum(bool(row.get("expected_article_numbers")) for row in enriched),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
