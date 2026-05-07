"""
Fine-tune BAAI/bge-m3 for Vietnamese legal retrieval.

Why: the base bge-m3 has never seen (penalty question, penalty clause) pairs.
     The existing QA training data has 0 samples with fine amounts, so the
     embedder cannot distinguish "xe máy vượt đèn đỏ bị phạt bao nhiêu?" from
     unrelated chunks in the same article.

Approach:
  - Combine qa_train.jsonl (procedure questions) + penalty_training_pairs.jsonl
  - Loss: MultipleNegativesRankingLoss (in-batch negatives — scalable, no manual labeling)
  - Hard negatives: explicitly included via TripletLoss for penalty pairs
  - Output: models/bge-m3-traffic-ft/

Usage:
  cd nlp && python scripts/finetune_embedder.py [--epochs 3] [--batch-size 16] [--max-pairs 6000]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=4,
                   help="Per-device batch size. bge-m3 (570M) needs <=4 on 15GB VRAM with grad checkpointing.")
    p.add_argument("--grad-accum", type=int, default=8,
                   help="Effective batch = batch_size * grad_accum. Default: 4*8=32")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-pairs", type=int, default=8000,
                   help="Max (anchor, positive) pairs to use for MNR stage")
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--output-dir", default=str(PROJECT_ROOT / "models" / "bge-m3-traffic-ft"))
    p.add_argument("--base-model", default="BAAI/bge-m3")
    p.add_argument("--max-seq-length", type=int, default=256,
                   help="Shorter seq saves ~4x activation memory vs 512. Legal chunks are dense.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_procedure_pairs(path: Path) -> list[tuple[str, str]]:
    """Load (question, context) from qa_train.jsonl."""
    pairs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        q = (d.get("question") or "").strip()
        c = (d.get("context") or "").strip()
        if q and c:
            pairs.append((q, c))
    return pairs


def load_penalty_pairs(path: Path) -> list[tuple[str, str, list[str]]]:
    """Load (question, context, hard_negative_fact_ids) from penalty_training_pairs.jsonl."""
    facts_by_id: dict[str, dict] = {}
    facts_path = path.parent / "legal_sanction_facts.jsonl"
    if facts_path.exists():
        for line in facts_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                f = json.loads(line)
                facts_by_id[f.get("fact_id", "")] = f

    pairs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        q = (d.get("question") or "").strip()
        c = (d.get("context") or "").strip()
        hn_ids = d.get("hard_negative_fact_ids") or []
        if q and c:
            # Resolve hard negative texts
            hn_texts = []
            for fid in hn_ids[:3]:
                f = facts_by_id.get(fid)
                if f:
                    parts = []
                    if f.get("citation"):
                        parts.append(f["citation"])
                    if f.get("violation_text"):
                        parts.append(f["violation_text"])
                    if f.get("fine_text"):
                        parts.append(f"Mức phạt: {f['fine_text']}")
                    if f.get("points_deducted"):
                        parts.append(f"Trừ điểm: {f['points_deducted']}")
                    hn_texts.append("\n".join(parts))
            pairs.append((q, c, hn_texts))
    return pairs


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    print("Loading sentence-transformers...")
    from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
    from sentence_transformers.losses import MultipleNegativesRankingLoss
    from sentence_transformers.evaluation import InformationRetrievalEvaluator
    from datasets import Dataset

    # ── 1. Load data ──────────────────────────────────────────────────────────
    proc_path = PROJECT_ROOT / "data" / "splits_filtered" / "qa_train.jsonl"
    pen_path = PROJECT_ROOT / "data" / "penalty_training_pairs.jsonl"

    proc_pairs = load_procedure_pairs(proc_path)
    print(f"Procedure pairs: {len(proc_pairs)}")

    pen_pairs = load_penalty_pairs(pen_path)
    print(f"Penalty pairs: {len(pen_pairs)}")

    # ── 2. Build single MNR dataset ───────────────────────────────────────────
    # MultipleNegativesRankingLoss supports optional `negative` column.
    # When present, the explicit negative is included alongside in-batch negatives.
    anchors, positives, negatives = [], [], []

    # Procedure pairs — no hard negatives available
    for q, c in proc_pairs:
        anchors.append(q)
        positives.append(c)
        negatives.append(None)

    # Penalty pairs — include first hard negative when available
    for q, c, hn_texts in pen_pairs:
        anchors.append(q)
        positives.append(c)
        negatives.append(hn_texts[0] if hn_texts else None)

    # Shuffle and cap
    combined = list(zip(anchors, positives, negatives))
    random.shuffle(combined)
    combined = combined[:args.max_pairs]
    anchors, positives, negatives = zip(*combined)

    has_negatives = any(n is not None for n in negatives)
    dataset_dict: dict = {"anchor": list(anchors), "positive": list(positives)}
    if has_negatives:
        dataset_dict["negative"] = [n or "" for n in negatives]

    train_dataset = Dataset.from_dict(dataset_dict)
    print(f"Training pairs (capped): {len(train_dataset)}")
    if has_negatives:
        n_with_neg = sum(1 for n in negatives if n)
        print(f"  of which {n_with_neg} have explicit hard negatives")

    # ── 3. Load model ─────────────────────────────────────────────────────────
    model = SentenceTransformer(args.base_model, trust_remote_code=True)
    model.max_seq_length = args.max_seq_length
    # Gradient checkpointing: trades recomputation for memory.
    # Saves ~40% activation memory, small speed cost (~20% slower).
    for module in model.modules():
        if hasattr(module, "gradient_checkpointing_enable"):
            module.gradient_checkpointing_enable()
            break

    # ── 4. Loss ───────────────────────────────────────────────────────────────
    mnr_loss = MultipleNegativesRankingLoss(model)

    # ── 5. Evaluator on dev split ─────────────────────────────────────────────
    dev_path = PROJECT_ROOT / "data" / "splits_filtered" / "qa_dev.jsonl"
    dev_pairs = load_procedure_pairs(dev_path)
    if dev_pairs:
        dev_queries = {str(i): q for i, (q, _) in enumerate(dev_pairs[:200])}
        dev_corpus = {str(i): c for i, (_, c) in enumerate(dev_pairs[:200])}
        dev_relevant = {str(i): {str(i)} for i in range(len(dev_queries))}
        evaluator = InformationRetrievalEvaluator(
            queries=dev_queries,
            corpus=dev_corpus,
            relevant_docs=dev_relevant,
            name="qa-dev",
        )
    else:
        evaluator = None

    # ── 6. Training args ──────────────────────────────────────────────────────
    steps_per_epoch = len(train_dataset) // (args.batch_size * args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    training_args = SentenceTransformerTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=warmup_steps,
        fp16=True,
        bf16=False,
        gradient_checkpointing=True,
        dataloader_num_workers=2,
        load_best_model_at_end=evaluator is not None,
        metric_for_best_model="eval_qa-dev_cosine_recall@3" if evaluator else None,
        greater_is_better=True,
        eval_strategy="epoch" if evaluator else "no",
        save_strategy="epoch",
        logging_steps=50,
        seed=args.seed,
        report_to="none",
    )

    # ── 7. Train ──────────────────────────────────────────────────────────────
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss=mnr_loss,
        evaluator=evaluator,
    )

    print(f"\nStarting fine-tune: {args.epochs} epochs, batch={args.batch_size}×{args.grad_accum}, lr={args.lr}")
    trainer.train()

    # ── 8. Save ───────────────────────────────────────────────────────────────
    out = Path(args.output_dir)
    model.save(str(out))
    print(f"\nModel saved to {out}")
    print("Next: rebuild KB with EMBED_MODEL=models/bge-m3-traffic-ft python src/build_kb.py --force")


if __name__ == "__main__":
    main()
