"""
Import the 600-question Vietnamese driving license (GPLX) exam into MC eval format.

The script handles multiple common input formats:
  - JSON array: [{"id":1, "question":"...", "A":"...", "B":"...", "C":"...", "D":"...", "answer":"A"}, ...]
  - JSON with "data" key: {"data": [...]}
  - CSV with columns: id/no/stt, question/cau_hoi, A/chon_a, B/chon_b, C/chon_c, D/chon_d, answer/dap_an

Excluded question numbers (1-based, reviewed by user):
  5,6,10,11,12,19,20,23,25->28,30,35,50,54,55,56,58,59,62,66,69->72,88,95,102,
  106->109,113,117,123->132,134->139,141,143,147,148,162,164,165,169,170,
  173->175,177,178,180,181,183->185,194,202->205,219,220,233,235,247,251,
  257,258,261->263,265,267,269->271,273,285,294,467,578

Usage:
  cd nlp
  python scripts/import_gplx_exam.py --input PATH_TO_600Q_FILE
  # --input can be .json, .jsonl, or .csv
  # Output appended to data/eval_mc_manual.jsonl by default
"""
import argparse
import csv
import json
import sys
from pathlib import Path

EXCLUDE_STR = (
    "5,6,10,11,12,19,20,23,25->28,30,35,50,54,55,56,58,59,62,66,69->72,88,95,"
    "102,106->109,113,117,123->132,134->139,141,143,147,148,162,164,165,169,170,"
    "173->175,177,178,180,181,183->185,194,202->205,219,220,233,235,247,251,"
    "257,258,261->263,265,267,269->271,273,285,294,467,578"
)

ANSWER_NORM = {
    "1": "A", "2": "B", "3": "C", "4": "D",
    "a": "A", "b": "B", "c": "C", "d": "D",
    "A": "A", "B": "B", "C": "C", "D": "D",
}


def _parse_exclude(s: str) -> set[int]:
    result = set()
    for part in s.replace(" ", "").split(","):
        if "->" in part:
            a, b = part.split("->")
            result.update(range(int(a), int(b) + 1))
        elif part:
            result.add(int(part))
    return result


EXCLUDED = _parse_exclude(EXCLUDE_STR)


def _norm_answer(raw) -> str | None:
    if raw is None:
        return None
    return ANSWER_NORM.get(str(raw).strip(), None)


def _load_json(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("data", "questions", "items", "records"):
            if isinstance(data.get(key), list):
                return data[key]
    raise ValueError(f"Cannot find question list in JSON structure of {path}")


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def _load_csv(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


def _find_field(row: dict, *candidates) -> str | None:
    for k in candidates:
        for key in row:
            if key.strip().lower() == k.lower():
                return row[key]
    return None


def _parse_item(raw: dict, idx_1based: int) -> dict | None:
    """Convert a raw row to MC eval format. Return None if unparseable."""
    # Question
    question = _find_field(raw, "question", "cau_hoi", "câu_hỏi", "noi_dung", "text", "q")
    if not question:
        # Try numbered keys like "question_1"
        for k, v in raw.items():
            if "question" in k.lower() or "cau" in k.lower():
                question = v
                break
    if not question:
        return None
    question = str(question).strip()

    # Choices
    choices = []
    for label in ("A", "B", "C", "D"):
        choice = _find_field(
            raw,
            label,
            f"chon_{label.lower()}",
            f"dap_an_{label.lower()}",
            f"choice_{label.lower()}",
            f"option_{label.lower()}",
        )
        if choice is None:
            return None
        choices.append(str(choice).strip())

    if any(len(c) == 0 for c in choices):
        return None

    # Answer
    answer_raw = _find_field(
        raw,
        "answer", "dap_an", "correct", "correct_answer",
        "dap_an_dung", "correct_choice",
    )
    answer = _norm_answer(answer_raw)
    if answer not in ("A", "B", "C", "D"):
        return None

    return {
        "question": question,
        "choices": choices,
        "answer": answer,
        "source": "gplx_600",
        "original_id": idx_1based,
    }


def load_exam(path: Path) -> list[dict]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        raw_items = _load_csv(path)
    elif suffix == ".jsonl":
        raw_items = _load_jsonl(path)
    else:
        raw_items = _load_json(path)
    return raw_items


def convert_and_filter(
    raw_items: list[dict],
    exclude: set[int],
) -> tuple[list[dict], int, int, int]:
    """
    Returns (converted, n_excluded, n_bad_format, n_ok).
    Uses 1-based indexing for exclusion.
    """
    converted = []
    n_excluded = 0
    n_bad = 0
    for i, raw in enumerate(raw_items, start=1):
        if i in exclude:
            n_excluded += 1
            continue
        item = _parse_item(raw, i)
        if item is None:
            n_bad += 1
            continue
        converted.append(item)
    return converted, n_excluded, n_bad, len(converted)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="Path to 600Q exam file (.json/.jsonl/.csv)")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/eval_mc_manual.jsonl"),
        help="Output file (default: append to data/eval_mc_manual.jsonl)",
    )
    parser.add_argument(
        "--no-append",
        action="store_true",
        help="Overwrite output instead of appending",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and print stats without writing",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading from: {args.input}")
    raw_items = load_exam(args.input)
    print(f"  Loaded {len(raw_items)} raw items")

    converted, n_excl, n_bad, n_ok = convert_and_filter(raw_items, EXCLUDED)
    print(f"  Excluded (user review): {n_excl}")
    print(f"  Bad format / skipped:   {n_bad}")
    print(f"  Converted OK:           {n_ok}")

    if args.dry_run:
        print("\nDry run — first 3 converted items:")
        for item in converted[:3]:
            print(json.dumps(item, ensure_ascii=False))
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if args.no_append else "a"
    with args.output.open(mode, encoding="utf-8") as f:
        for item in converted:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    total = sum(1 for l in args.output.open(encoding="utf-8") if l.strip())
    print(f"\nAppended {n_ok} questions → {args.output}  (total: {total})")


if __name__ == "__main__":
    main()
