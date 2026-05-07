"""
Build training data for fact reranker v2 with answer-grounded labels.

Why v1 failed: it used token-overlap to create positive/negative labels.
Two facts from the same article can have high token overlap but different fine amounts
(e.g., car 400k-600k vs motorbike 200k-400k) — token overlap gives them similar scores
and cannot distinguish which one answers the question.

This script creates labels from the fact's structured fields:
  positive (1): fact with the same violation + correct vehicle + fine_text
  hard_negative (0): fact with same article, same violation type, but DIFFERENT
                     vehicle_scope OR different clause (different penalty tier)

Training data format: {"query": question, "document": fact_text, "label": 1/0}

Output: data/fact_reranker_v2_train.jsonl
"""
from __future__ import annotations

import json
import random
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
FACTS_PATH = DATA_DIR / "legal_sanction_facts.jsonl"
PENALTY_PAIRS_PATH = DATA_DIR / "penalty_training_pairs.jsonl"
OUT_PATH = DATA_DIR / "fact_reranker_v2_train.jsonl"


def _fact_to_text(fact: dict) -> str:
    parts = []
    if fact.get("citation"):
        parts.append(fact["citation"])
    if fact.get("article_title"):
        parts.append(fact["article_title"])
    if fact.get("vehicle_scope"):
        parts.append(f"Phương tiện: {', '.join(fact['vehicle_scope'])}")
    if fact.get("violation_text"):
        parts.append(fact["violation_text"][:400])
    if fact.get("fine_text"):
        parts.append(f"Mức phạt: {fact['fine_text']}")
    if fact.get("points_deducted"):
        parts.append(f"Trừ điểm: {fact['points_deducted']}")
    if fact.get("suspension_text"):
        parts.append(f"Tước/đình chỉ: {fact['suspension_text']}")
    return "\n".join(parts)


def main() -> None:
    random.seed(42)

    facts = {
        json.loads(l)["fact_id"]: json.loads(l)
        for l in FACTS_PATH.read_text(encoding="utf-8").splitlines()
        if l.strip()
    }

    penalty_pairs = [
        json.loads(l)
        for l in PENALTY_PAIRS_PATH.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]

    records: list[dict] = []
    missing_neg = 0

    for pair in penalty_pairs:
        question = pair["question"]
        pos_id = pair["fact_id"]
        hard_neg_ids = pair.get("hard_negative_fact_ids") or []

        pos_fact = facts.get(pos_id)
        if not pos_fact:
            continue

        pos_text = _fact_to_text(pos_fact)

        # Positive example
        records.append({
            "query": question,
            "document": pos_text,
            "label": 1,
            "fact_id": pos_id,
            "doc_id": pos_fact.get("doc_id", ""),
            "article_number": pos_fact.get("article_number", ""),
        })

        # Hard negative examples (up to 2 per positive to avoid label imbalance)
        neg_added = 0
        for neg_id in hard_neg_ids[:4]:
            if neg_added >= 2:
                break
            neg_fact = facts.get(neg_id)
            if not neg_fact:
                continue
            records.append({
                "query": question,
                "document": _fact_to_text(neg_fact),
                "label": 0,
                "fact_id": neg_id,
                "doc_id": neg_fact.get("doc_id", ""),
                "article_number": neg_fact.get("article_number", ""),
            })
            neg_added += 1

        if neg_added == 0:
            missing_neg += 1

    random.shuffle(records)

    OUT_PATH.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records),
        encoding="utf-8",
    )

    n_pos = sum(1 for r in records if r["label"] == 1)
    n_neg = sum(1 for r in records if r["label"] == 0)
    print(f"Total records: {len(records)}")
    print(f"  Positive: {n_pos}  Negative: {n_neg}  Ratio: {n_neg/max(n_pos,1):.1f}x")
    print(f"  Pairs without hard negatives: {missing_neg}")
    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
