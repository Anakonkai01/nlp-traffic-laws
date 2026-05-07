"""
Generate (question, context) training pairs from structured sanction facts.

Why: the existing QA training data has 0 samples with fine amounts in context,
so the embedder (bge-m3) has never seen (penalty question, penalty clause) pairs.
This script creates ~2000 synthetic pairs from the 975 answer_ready facts,
covering the penalty question types that dominate eval_manual.jsonl (105/145 samples).

Output: data/penalty_training_pairs.jsonl
Fields: question, context, fact_id, doc_id, article_number, clause_number,
        hard_negative_fact_ids (list of same-article, different-clause/vehicle facts)
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
FACTS_PATH = DATA_DIR / "legal_sanction_facts.jsonl"
OUT_PATH = DATA_DIR / "penalty_training_pairs.jsonl"

VEHICLE_LABEL = {
    "car":              "ô tô",
    "motorbike":        "xe máy",
    "special_motorbike": "xe máy chuyên dùng",
    "bicycle":          "xe đạp",
    "pedestrian":       "người đi bộ",
}

VEHICLE_LABEL_ALT = {
    "car":              "xe ô tô",
    "motorbike":        "xe mô tô, xe gắn máy",
    "special_motorbike": "xe máy chuyên dùng",
    "bicycle":          "xe đạp, xe thô sơ",
    "pedestrian":       "người tham gia giao thông bộ hành",
}


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _violation_snippet(violation_text: str) -> str:
    """Extract a concise violation description for use in question templates."""
    t = _clean(violation_text)
    # Strip leading article header (Điều X. ...)
    t = re.sub(r"^Điều\s+\d+[a-z]?\.\s*[^\n]{0,200}", "", t, flags=re.I).strip()
    # Strip leading clause header (Khoản X. ...)
    t = re.sub(r"^Khoản\s+\d+\.\s*", "", t, flags=re.I).strip()
    # Strip leading point letter (a) b) ...)
    t = re.sub(r"^[a-zđ]\)\s*", "", t, flags=re.I).strip()
    # Clip to first sentence / comma group for readability
    for sep in [".", ";", ","]:
        idx = t.find(sep)
        if 20 < idx < 200:
            t = t[:idx].strip()
            break
    return t[:240] if t else ""


def _build_context(fact: dict) -> str:
    """Construct a self-contained legal context string from fact fields."""
    parts = []
    if fact.get("citation"):
        parts.append(fact["citation"])
    vt = _clean(fact.get("violation_text", ""))
    if vt:
        parts.append(vt)
    if fact.get("fine_text"):
        parts.append(f"Mức phạt: {fact['fine_text']}")
    if fact.get("points_deducted"):
        parts.append(f"Trừ điểm giấy phép lái xe: {fact['points_deducted']}")
    if fact.get("suspension_text"):
        parts.append(f"Tước/đình chỉ giấy phép lái xe: {fact['suspension_text']}")
    return "\n".join(parts)


def _generate_questions(fact: dict) -> list[str]:
    """Generate 3-5 question variants for a sanction fact."""
    vscope = fact.get("vehicle_scope") or []
    vh = VEHICLE_LABEL.get(vscope[0], "phương tiện") if vscope else "phương tiện"
    vh_alt = VEHICLE_LABEL_ALT.get(vscope[0], vh) if vscope else vh

    vt = _clean(fact.get("violation_text", ""))
    snippet = _violation_snippet(vt)
    fine = fact.get("fine_text", "")
    deduct = fact.get("points_deducted")
    citation = fact.get("citation", "")

    # Derive a short action phrase for question generation.
    # For article-level facts, fall back to the article title.
    if not snippet and vt:
        snippet = vt[:160]

    questions: list[str] = []

    if snippet and fine:
        questions += [
            f"Điều khiển {vh} {snippet.lower()} bị phạt bao nhiêu tiền?",
            f"Mức phạt tiền cho hành vi {snippet.lower()} khi lái {vh} là bao nhiêu?",
            f"{vh.capitalize()} vi phạm {snippet.lower()} bị xử phạt thế nào?",
        ]

    if snippet and deduct:
        questions.append(
            f"Điều khiển {vh_alt} {snippet.lower()} bị trừ bao nhiêu điểm giấy phép lái xe?"
        )

    if snippet and fact.get("suspension_text"):
        questions.append(
            f"Người điều khiển {vh} {snippet.lower()} có bị tước giấy phép lái xe không?"
        )

    # Citation-based variant (what's the penalty at Article X Clause Y?)
    if citation and fine:
        # Extract article/clause reference from citation string
        art_m = re.search(r"Điều\s+(\d+)", citation, re.I)
        cls_m = re.search(r"khoản\s+(\d+)", citation, re.I)
        if art_m:
            ref = f"Điều {art_m.group(1)}"
            if cls_m:
                ref += f" khoản {cls_m.group(1)}"
            questions.append(f"Mức phạt theo {ref} đối với {vh} là bao nhiêu?")

    # Fallback: if no snippet at all, use the article title area
    if not questions and vt:
        questions = [
            f"Quy định mức phạt đối với {vh} tại {citation} là gì?",
            f"Xử phạt {vh} theo {citation} như thế nào?",
        ]

    return list(dict.fromkeys(questions))  # deduplicate, preserve order


def _hard_negative_ids(fact: dict, all_facts: list[dict]) -> list[str]:
    """Return fact_ids that are hard negatives: same article, different clause or vehicle."""
    fid = fact.get("fact_id", "")
    art = fact.get("article_number")
    doc = fact.get("doc_id")
    cls = fact.get("clause_number")
    vscope = set(fact.get("vehicle_scope") or [])

    negatives: list[str] = []
    for f in all_facts:
        if f.get("fact_id") == fid:
            continue
        if f.get("doc_id") != doc or f.get("article_number") != art:
            continue
        # Different clause → same article, different penalty tier
        if f.get("clause_number") != cls and f.get("fine_text"):
            negatives.append(f["fact_id"])
            continue
        # Same clause, different vehicle → nearly identical text but wrong vehicle/amount
        if f.get("clause_number") == cls and set(f.get("vehicle_scope") or []) != vscope and f.get("fine_text"):
            negatives.append(f["fact_id"])

    return negatives[:8]


def main() -> None:
    random.seed(42)

    facts = [json.loads(l) for l in FACTS_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    ar_facts = [f for f in facts if f.get("answer_ready") and f.get("fine_text")]
    print(f"Answer-ready facts with fine: {len(ar_facts)}")

    # Index for hard negative lookup
    all_facts = [f for f in facts if f.get("fine_text")]

    pairs: list[dict] = []
    skipped = 0

    for fact in ar_facts:
        questions = _generate_questions(fact)
        if not questions:
            skipped += 1
            continue

        context = _build_context(fact)
        hard_neg_ids = _hard_negative_ids(fact, all_facts)

        for q in questions:
            pairs.append({
                "question": q,
                "context": context,
                "fact_id": fact.get("fact_id", ""),
                "doc_id": fact.get("doc_id", ""),
                "article_number": fact.get("article_number", ""),
                "clause_number": fact.get("clause_number", ""),
                "point_letter": fact.get("point_letter", ""),
                "vehicle_scope": fact.get("vehicle_scope") or [],
                "fine_text": fact.get("fine_text", ""),
                "points_deducted": fact.get("points_deducted"),
                "hard_negative_fact_ids": hard_neg_ids,
            })

    random.shuffle(pairs)

    OUT_PATH.write_text(
        "\n".join(json.dumps(p, ensure_ascii=False) for p in pairs),
        encoding="utf-8",
    )

    print(f"Generated {len(pairs)} pairs from {len(ar_facts)} facts ({skipped} skipped)")
    print(f"Saved to {OUT_PATH}")

    # Stats
    from collections import Counter
    veh_counts = Counter(
        str(p["vehicle_scope"]) for p in pairs
    )
    print("\nVehicle distribution:")
    for v, c in veh_counts.most_common():
        print(f"  {v}: {c}")


if __name__ == "__main__":
    main()
