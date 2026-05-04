"""
Parse the 600-question GPLX exam PDF and extract Q&A in MC format.

The PDF format:
  - Questions: "Câu X. [text]" (bold)
  - Choices: "1. [text]", "2. [text]", "3. [text]" (sometimes 4 choices)
  - Correct answer: underlined (thin filled rectangle below the choice line)

Output: JSON lines compatible with eval_mc_manual.jsonl format.
  {"question": "...", "choices": ["A","B","C","D"], "answer": "A", ...}

Usage:
  cd nlp
  python scripts/parse_gplx_pdf.py --input data/gplx_600.pdf \
    --output data/gplx_600_parsed.jsonl --dry-run
"""
import argparse
import json
import re
import sys
from pathlib import Path

import fitz

# Underline rects have height < 2pt
UNDERLINE_MAX_HEIGHT = 2.0
# Y-tolerance: underline must be within this many pts of the text baseline
UNDERLINE_Y_TOL = 6.0
# Question header regex
QUESTION_RE = re.compile(r"^Câu\s+(\d+)\.\s*(.*)", re.DOTALL)
# Choice line regex: starts with "1." / "2." / "3." / "4."
CHOICE_RE = re.compile(r"^([1-4])\.\s+(.*)", re.DOTALL)


def _underlines_on_page(page) -> list[float]:
    """Return sorted list of y-midpoints of underline drawings."""
    result = []
    for d in page.get_drawings():
        r = d["rect"]
        h = r.y1 - r.y0
        if h < UNDERLINE_MAX_HEIGHT and r.x1 - r.x0 > 10:
            result.append((r.y0 + r.y1) / 2)
    return sorted(result)


def _extract_page_spans(page) -> list[dict]:
    """
    Return list of {text, y0, y1, bold} for each text span.
    Merge adjacent spans on the same line.
    """
    spans = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            line_text = ""
            y0, y1 = None, None
            bold = False
            for span in line["spans"]:
                t = span["text"]
                if not t.strip():
                    continue
                if y0 is None:
                    y0 = span["bbox"][1]
                    y1 = span["bbox"][3]
                else:
                    y0 = min(y0, span["bbox"][1])
                    y1 = max(y1, span["bbox"][3])
                if span["flags"] & 16:  # bold
                    bold = True
                line_text += t
            if line_text.strip():
                spans.append({"text": line_text.strip(), "y0": y0, "y1": y1, "bold": bold})
    return spans


def _parse_page(page) -> list[dict]:
    """
    Parse one page into a list of partial question dicts:
      {"id": int, "question": str, "raw_choices": [(num, text, underlined)]}
    A question may span multiple pages.
    """
    underline_ys = _underlines_on_page(page)
    spans = _extract_page_spans(page)

    def is_underlined(y0: float, y1: float) -> bool:
        baseline = y1
        for uy in underline_ys:
            if baseline - UNDERLINE_Y_TOL <= uy <= baseline + UNDERLINE_Y_TOL:
                return True
        return False

    questions = []
    current_q = None
    current_choice_num = None
    current_choice_lines = []
    current_choice_underlined = False

    def flush_choice():
        nonlocal current_choice_num, current_choice_lines, current_choice_underlined
        if current_q is not None and current_choice_num is not None and current_choice_lines:
            text = " ".join(current_choice_lines).strip()
            current_q["raw_choices"].append(
                (current_choice_num, text, current_choice_underlined)
            )
        current_choice_num = None
        current_choice_lines = []
        current_choice_underlined = False

    for span in spans:
        text = span["text"]
        y0, y1 = span["y0"], span["y1"]

        # New question header?
        m = QUESTION_RE.match(text)
        if m:
            flush_choice()
            if current_q is not None:
                questions.append(current_q)
            qnum = int(m.group(1))
            qtail = m.group(2).strip()
            current_q = {"id": qnum, "question": qtail, "raw_choices": []}
            current_choice_num = None
            continue

        # Choice line?
        m = CHOICE_RE.match(text)
        if m:
            flush_choice()
            current_choice_num = int(m.group(1))
            current_choice_lines = [m.group(2).strip()]
            current_choice_underlined = is_underlined(y0, y1)
            continue

        # Continuation of question header (multi-line question)?
        if current_q is not None and current_choice_num is None:
            current_q["question"] += " " + text
            continue

        # Continuation of a choice?
        if current_choice_num is not None:
            current_choice_lines.append(text)
            if is_underlined(y0, y1):
                current_choice_underlined = True
            continue

    flush_choice()
    if current_q is not None:
        questions.append(current_q)

    return questions


def _merge_multipage(all_partial: list[dict]) -> list[dict]:
    """Merge partial dicts from multiple pages into complete question dicts."""
    merged = []
    seen_ids = {}
    for pq in all_partial:
        qid = pq["id"]
        if qid not in seen_ids:
            seen_ids[qid] = len(merged)
            merged.append(pq)
        else:
            # Extend existing question with more choices
            existing = merged[seen_ids[qid]]
            if pq["question"] and not existing["question"].endswith(pq["question"]):
                existing["question"] += " " + pq["question"]
            existing["raw_choices"].extend(pq["raw_choices"])
    return merged


LABELS = ["A", "B", "C", "D"]


def _finalize(q: dict) -> dict | None:
    """Convert raw question dict to final MC format."""
    choices_raw = q["raw_choices"]
    if len(choices_raw) < 2:
        return None

    # Sort by choice number
    choices_raw.sort(key=lambda x: x[0])

    choices = [c[1] for c in choices_raw]
    underlined = [c[2] for c in choices_raw]

    # Find correct answer
    correct_indices = [i for i, u in enumerate(underlined) if u]
    if len(correct_indices) != 1:
        # Fallback: try to find via font/format elsewhere
        return None

    correct_idx = correct_indices[0]
    if correct_idx >= len(LABELS):
        return None

    answer = LABELS[correct_idx]
    return {
        "question": q["question"].strip(),
        "choices": choices[:4],  # max 4 choices; evaluate_mc.py handles 2-4
        "answer": answer,
        "source": "gplx_600",
        "original_id": q["id"],
    }


def parse_pdf(path: Path) -> list[dict]:
    doc = fitz.open(str(path))
    all_partial = []
    for page_num in range(doc.page_count):
        page = doc[page_num]
        partial = _parse_page(page)
        all_partial.extend(partial)

    merged = _merge_multipage(all_partial)
    results = []
    for q in merged:
        item = _finalize(q)
        if item:
            results.append(item)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/gplx_600.pdf"))
    parser.add_argument("--output", type=Path, default=Path("data/gplx_600_parsed.jsonl"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: {args.input} not found", file=sys.stderr)
        sys.exit(1)

    print(f"Parsing {args.input} …")
    items = parse_pdf(args.input)
    print(f"Parsed: {len(items)} questions with correct answer")

    if args.dry_run:
        for item in items[:5]:
            print(json.dumps(item, ensure_ascii=False))
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(i, ensure_ascii=False) for i in items) + "\n",
        encoding="utf-8",
    )
    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
