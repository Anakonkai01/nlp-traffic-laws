"""
Convert auto-generated QA pairs to eval_manual.jsonl format.

Adds required fields: expected_doc_ids, category.
Skips negative samples.

Usage:
  cd nlp
  # 1. Generate candidates
  python src/generate_qa.py --force --phases general penalties procedures prohibited \
    --target-cap 80 --no-negatives --output data/eval_candidates.jsonl

  # 2. Convert format
  python scripts/convert_to_eval.py \
    --input data/eval_candidates.jsonl \
    --output data/eval_converted.jsonl

  # 3. Review eval_converted.jsonl — delete lines you don't like

  # 4. Append to eval set
  cat data/eval_converted.jsonl >> data/eval_manual.jsonl
"""
import argparse
import json
import re
import sys
from pathlib import Path

# Map source doc_id hints to category
_PENALTY_DOCS = {"nd_168_2024_nd_cp", "nd_336_2025_nd_cp"}
_PROCEDURE_DOCS = {"tt_65_2024_tt_bca", "nd_158_2024_nd_cp"}

_PENALTY_KEYWORDS = re.compile(
    r"phạt tiền|xử phạt|mức phạt|tước|trừ điểm|bị phạt|hình thức phạt", re.I
)
_PROCEDURE_KEYWORDS = re.compile(
    r"thủ tục|hồ sơ|điều kiện|đề nghị|nộp|thời hạn|thẩm quyền|cấp phép|đăng ký", re.I
)
_PROHIBITED_KEYWORDS = re.compile(r"cấm|nghiêm cấm|không được|bị cấm", re.I)
_DEFINITION_KEYWORDS = re.compile(r"là |được hiểu là|được định nghĩa|theo quy định|giải thích từ ngữ", re.I)


def _infer_category(item: dict) -> str:
    doc_id = item.get("doc_id", "") or item.get("source", "")
    question = item.get("question", "")
    answer = item.get("answer", "")
    text = question + " " + answer

    if doc_id in _PENALTY_DOCS or _PENALTY_KEYWORDS.search(text):
        return "penalty"
    if doc_id in _PROCEDURE_DOCS or _PROCEDURE_KEYWORDS.search(text):
        return "procedure"
    if _PROHIBITED_KEYWORDS.search(text):
        return "prohibited"
    if _DEFINITION_KEYWORDS.search(text):
        return "definition"
    return "general"


def convert(input_path: Path, output_path: Path, min_answer_len: int = 15) -> None:
    converted = 0
    skipped = 0

    with input_path.open(encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            item = json.loads(line)

            # Skip negatives
            if item.get("corpus") == "negative":
                skipped += 1
                continue

            # Skip suspiciously short answers
            if len(item.get("answer", "")) < min_answer_len:
                skipped += 1
                continue

            doc_id = item.get("doc_id") or item.get("source") or ""
            eval_item = {
                "question": item["question"],
                "answer": item["answer"],
                "context": item.get("context", ""),
                "expected_doc_ids": [doc_id] if doc_id else [],
                "category": _infer_category(item),
                "source": doc_id,
                "article": item.get("article", ""),
            }
            fout.write(json.dumps(eval_item, ensure_ascii=False) + "\n")
            converted += 1

    print(f"Converted: {converted} | Skipped: {skipped}")
    print(f"Output: {output_path}")
    print()
    print("Next steps:")
    print(f"  1. Review {output_path} — delete any bad pairs")
    print(f"  2. cat {output_path} >> data/eval_manual.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/eval_candidates.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/eval_converted.jsonl"))
    parser.add_argument("--min-answer-len", type=int, default=15)
    args = parser.parse_args()
    convert(args.input, args.output, args.min_answer_len)


if __name__ == "__main__":
    main()
