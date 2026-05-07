"""
Fine-tune BAAI/bge-reranker-v2-m3 on (q, positive, negatives) pairs.

Loss: BCE-with-logits (positives → 1, negatives → 0).
Eval: Recall@5 on qa_dev.jsonl with the same Dense+BM25+RRF candidate set
the rerankr will see at inference.

Run from nlp/:
    python scripts/finetune_reranker.py \\
        --pairs data/training/ce_pairs.jsonl \\
        --epochs 2 --bs 16 --lr 2e-5 \\
        --out models/bge-reranker-v2-m3-traffic-ft
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))


def load_pairs(path: Path) -> list[tuple[str, str, int]]:
    out: list[tuple[str, str, int]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        q, pos, negs = row["question"], row["positive"], row["negatives"]
        out.append((q, pos, 1))
        for n in negs:
            out.append((q, n, 0))
    return out


def evaluate_dev(model, dev_path: Path, vs, k: int = 5) -> float:
    """Score dev Recall@5 by running the model on Dense+BM25+RRF top-30 candidates."""
    from retrieval_v3 import RetrievalConfig, build_retriever  # type: ignore
    from rouge_score import rouge_scorer  # type: ignore

    rows = [json.loads(l) for l in dev_path.read_text().splitlines() if l.strip()]
    cfg = RetrievalConfig(use_bm25=True, use_alias=False, use_ce=False, fuse_k=30)
    retrieve = build_retriever(vs, cfg)
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    hits = total = 0
    for r in rows:
        q = r.get("question") or ""
        gold_ctx = r.get("context") or ""
        if not gold_ctx:
            continue
        total += 1
        cands = retrieve(q, 30)
        pairs = [(q, d.page_content or "") for d in cands]
        scores = model.predict(pairs, batch_size=64, show_progress_bar=False)
        ranked = sorted(zip(cands, scores), key=lambda x: -float(x[1]))
        top5 = [d for d, _ in ranked[:k]]
        if any(scorer.score(gold_ctx, d.page_content)["rougeL"].fmeasure >= 0.5 for d in top5):
            hits += 1
    return hits / total if total else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--dev", default="data/splits/qa_dev.jsonl")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--no-amp", action="store_true", help="Disable AMP; useful on GPUs with tight memory/optimizer instability")
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--base", default="BAAI/bge-reranker-v2-m3")
    ap.add_argument("--out", required=True)
    ap.add_argument("--eval-every-frac", type=float, default=0.5,
                    help="Eval at this fraction of total steps (0.5 = mid-epoch + end).")
    args = ap.parse_args()

    from sentence_transformers import CrossEncoder, InputExample
    from torch.utils.data import DataLoader

    pairs = load_pairs(Path(args.pairs))
    print(f"[load] {len(pairs)} (q, doc, label) examples")

    train_examples = [InputExample(texts=[q, d], label=float(y)) for q, d, y in pairs]
    train_loader = DataLoader(train_examples, shuffle=True, batch_size=args.bs)

    model = CrossEncoder(args.base, num_labels=1, max_length=args.max_length)

    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)

    from build_kb import load_vectorstore  # type: ignore

    vs = load_vectorstore()

    best = {"r5": -1.0, "epoch": -1}

    def fit_epoch(epoch: int):
        model.fit(
            train_dataloader=train_loader,
            epochs=1,
            warmup_steps=int(args.warmup_ratio * len(train_loader)),
            optimizer_params={"lr": args.lr},
            show_progress_bar=True,
            use_amp=not args.no_amp,
        )
        r5 = evaluate_dev(model, Path(args.dev), vs, k=5)
        print(f"[epoch {epoch}] dev Recall@5 = {r5:.4f}")
        if r5 > best["r5"]:
            best["r5"] = r5
            best["epoch"] = epoch
            model.save(str(out_path))
            (out_path / "ft_meta.json").write_text(json.dumps({
                "base": args.base,
                "epoch_saved": epoch,
                "dev_recall_at_5": r5,
                "lr": args.lr,
                "bs": args.bs,
                "epochs_total": args.epochs,
                "max_length": args.max_length,
            }, indent=2))
            print(f"[best ↑] saved → {out_path}")

    for epoch in range(1, args.epochs + 1):
        fit_epoch(epoch)

    print(f"[done] best dev Recall@5 = {best['r5']:.4f} at epoch {best['epoch']}")


if __name__ == "__main__":
    main()
