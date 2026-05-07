"""
Fine-tune BAAI/bge-reranker-v2-m3 for fact card selection.

Why v2 vs v1:
  v1 used token-overlap labels → dev acc 0.6931, smoke ROUGE-L 0.3506
  v2 uses answer-grounded labels (same article / different vehicle or clause)
  Eval metric: fine_recall@1, vehicle_recall@1 — not dev accuracy

Output: models/fact-reranker-v2/

Usage:
  cd nlp && python scripts/finetune_fact_reranker_v2.py [--epochs 3] [--batch-size 16]

After training, enable with:
  RAG_FACT_USE_CE=1 RAG_FACT_CE_MODEL=models/fact-reranker-v2 python src/evaluate.py --configs D
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_PATH = DATA_DIR / "fact_reranker_v2_train.jsonl"
FACTS_PATH = DATA_DIR / "legal_sanction_facts.jsonl"
OUT_DIR = PROJECT_ROOT / "models" / "fact-reranker-v2"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=8,
                   help="bge-reranker-v2-m3 ~600M params; 8 is safe on 15GB with checkpointing")
    p.add_argument("--grad-accum", type=int, default=4,
                   help="Effective batch = 8*4=32")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-length", type=int, default=384)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--output-dir", default=str(OUT_DIR))
    p.add_argument("--base-model", default="BAAI/bge-reranker-v2-m3")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_data(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def stratified_split(records: list[dict], val_ratio: float, seed: int):
    """Split preserving pos/neg ratio."""
    rng = random.Random(seed)
    pos = [r for r in records if r["label"] == 1]
    neg = [r for r in records if r["label"] == 0]
    rng.shuffle(pos); rng.shuffle(neg)
    n_val_pos = max(1, int(len(pos) * val_ratio))
    n_val_neg = max(1, int(len(neg) * val_ratio))
    val = pos[:n_val_pos] + neg[:n_val_neg]
    train = pos[n_val_pos:] + neg[n_val_neg:]
    rng.shuffle(train); rng.shuffle(val)
    return train, val


def fine_recall_at_k(model, val_records: list[dict], facts_by_id: dict, k: int = 1) -> float:
    """Recall: for penalty questions, does top-k contain a fact with the correct fine_text?"""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    # Group val records by query
    query_groups: dict[str, list[dict]] = {}
    for r in val_records:
        query_groups.setdefault(r["query"], []).append(r)

    hits = 0
    total = 0
    tok, ce_model, device = model
    for query, group in query_groups.items():
        if not group:
            continue
        texts = [r["document"] for r in group]
        pairs = [(query, t[:600]) for t in texts]
        with torch.inference_mode():
            enc = tok(pairs, padding=True, truncation=True, max_length=384, return_tensors="pt").to(device)
            scores = ce_model(**enc).logits.view(-1).float().cpu().tolist()
        ranked = sorted(zip(scores, group), key=lambda x: -x[0])
        top_k_facts = [r for _, r in ranked[:k]]
        # Hit if any top-k is a positive
        if any(r["label"] == 1 for r in top_k_facts):
            hits += 1
        total += 1

    return hits / max(total, 1)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        DataCollatorWithPadding,
    )

    print("Loading training data...")
    records = load_data(TRAIN_PATH)
    print(f"Total records: {len(records)}  (pos={sum(r['label']==1 for r in records)}, neg={sum(r['label']==0 for r in records)})")

    train_records, val_records = stratified_split(records, args.val_ratio, args.seed)
    print(f"Train: {len(train_records)}  Val: {len(val_records)}")

    # ── Tokenize ──────────────────────────────────────────────────────────────
    tok = AutoTokenizer.from_pretrained(args.base_model)

    def encode(batch):
        enc = tok(
            batch["query"],
            batch["document"],
            padding=False,
            truncation=True,
            max_length=args.max_length,
        )
        enc["labels"] = [float(l) for l in batch["label"]]
        return enc

    train_ds = Dataset.from_dict({
        "query": [r["query"] for r in train_records],
        "document": [r["document"] for r in train_records],
        "label": [r["label"] for r in train_records],
    }).map(encode, batched=True, remove_columns=["query", "document", "label"])

    val_ds = Dataset.from_dict({
        "query": [r["query"] for r in val_records],
        "document": [r["document"] for r in val_records],
        "label": [r["label"] for r in val_records],
    }).map(encode, batched=True, remove_columns=["query", "document", "label"])

    # ── Model ─────────────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ce_model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=1,
    ).to(device)
    ce_model.gradient_checkpointing_enable()

    # ── Training ──────────────────────────────────────────────────────────────
    total_steps = (len(train_ds) // (args.batch_size * args.grad_accum)) * args.epochs
    warmup = int(total_steps * args.warmup_ratio)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=warmup,
        fp16=False,
        bf16=device == "cuda",
        gradient_checkpointing=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=50,
        report_to="none",
        seed=args.seed,
        dataloader_num_workers=2,
    )

    def compute_metrics(eval_pred):
        import numpy as np
        logits, labels = eval_pred
        preds = (logits.squeeze() > 0).astype(int)
        acc = (preds == labels.astype(int)).mean()
        return {"accuracy": float(acc)}

    collator = DataCollatorWithPadding(tok, pad_to_multiple_of=8)

    trainer = Trainer(
        model=ce_model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    print(f"\nStarting training: {args.epochs} epochs, batch={args.batch_size}×{args.grad_accum}, lr={args.lr}")
    trainer.train()

    # ── Save ──────────────────────────────────────────────────────────────────
    out = Path(args.output_dir)
    trainer.save_model(str(out))
    tok.save_pretrained(str(out))
    print(f"\nModel saved to {out}")

    # ── Fine recall evaluation ─────────────────────────────────────────────────
    print("\nComputing fine_recall@1 on val set...")
    ce_model.eval()
    model_tuple = (tok, ce_model, device)
    recall_1 = fine_recall_at_k(model_tuple, val_records, {}, k=1)
    recall_3 = fine_recall_at_k(model_tuple, val_records, {}, k=3)
    print(f"  fine_recall@1 = {recall_1:.4f}")
    print(f"  fine_recall@3 = {recall_3:.4f}")

    results_path = out / "eval_results.json"
    results_path.write_text(json.dumps({
        "fine_recall_at_1": recall_1,
        "fine_recall_at_3": recall_3,
        "val_records": len(val_records),
    }, indent=2), encoding="utf-8")

    print(f"\nTo use: RAG_FACT_USE_CE=1 RAG_FACT_CE_MODEL={out} python src/evaluate.py --configs D")


if __name__ == "__main__":
    main()
