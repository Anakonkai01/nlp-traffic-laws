"""Oracle ceiling: BGE-M3 cosine(question, gold_context) distribution.

Tells us how high recall could plausibly go even with perfect retrieval.
If sim is low for many rows, those rows have noisy QA pairs (LLM paraphrase too far).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))


def main():
    from langchain_huggingface import HuggingFaceEmbeddings

    emb = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
    )

    out_path = ROOT / "reports" / "traffic" / "oracle_sim.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sets = {
        "train": ROOT / "data/splits/qa_train.jsonl",
        "dev": ROOT / "data/splits/qa_dev.jsonl",
        "test": ROOT / "data/splits/qa_test.jsonl",
    }
    rows_by_set: dict[str, list] = {}
    for n, p in sets.items():
        rows_by_set[n] = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]

    summaries = []
    csv_lines = ["set,row_idx,doc_id,article,sim"]

    for split_name, rows in rows_by_set.items():
        questions = [r["question"] for r in rows]
        contexts = [r["context"] for r in rows]
        q_emb = np.array(emb.embed_documents(questions))
        c_emb = np.array(emb.embed_documents(contexts))
        sims = (q_emb * c_emb).sum(axis=1)

        for i, (r, s) in enumerate(zip(rows, sims)):
            csv_lines.append(f"{split_name},{i},{r.get('doc_id','')},\"{r.get('article','')[:80]}\",{s:.4f}")

        bins = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
        hist, _ = np.histogram(sims, bins=bins)
        n = len(sims)
        print(f"\n=== {split_name} (n={n}) ===")
        print(f"  mean={sims.mean():.4f}  median={np.median(sims):.4f}  "
              f"p10={np.percentile(sims, 10):.4f}  p90={np.percentile(sims, 90):.4f}")
        for lo, hi, c in zip(bins[:-1], bins[1:], hist):
            bar = "#" * int(c / max(1, n) * 50)
            print(f"  sim ∈ [{lo:.2f},{hi:.2f}): {c:4d} ({c/n:.3f}) {bar}")
        summaries.append((split_name, n, float(sims.mean()), float(np.median(sims)),
                          float(np.percentile(sims, 10)), float(np.percentile(sims, 25))))

    out_path.write_text("\n".join(csv_lines))
    print(f"\n[done] sim CSV → {out_path}")
    print("\nSummary:")
    print(f"  {'set':<8}{'n':<6}{'mean':<8}{'median':<8}{'p10':<8}{'p25':<8}")
    for s in summaries:
        print(f"  {s[0]:<8}{s[1]:<6}{s[2]:<8.4f}{s[3]:<8.4f}{s[4]:<8.4f}{s[5]:<8.4f}")


if __name__ == "__main__":
    main()
