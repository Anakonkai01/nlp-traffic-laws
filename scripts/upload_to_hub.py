"""
Upload dataset and LoRA adapter to HuggingFace Hub.

Usage:
  cd nlp
  HF_TOKEN=hf_xxx python scripts/upload_to_hub.py \
    --hf-user YOUR_USERNAME \
    --dataset-repo traffic-law-qa-vi \
    --model-repo qwen35-9b-traffic-law-lora
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from config import (
    DATA_DIR,
    EVAL_DATA_PATH,
    EVAL_MC_DATA_PATH,
    MODEL_DIR,
    QA_DATA_PATH,
    REPORTS_DIR,
)

MODEL_CARD_TEMPLATE = """---
language:
- vi
license: apache-2.0
base_model: Qwen/Qwen3.5-9B
tags:
- lora
- qlora
- traffic-law
- vietnamese
- rag
- question-answering
datasets:
- {hf_user}/{dataset_repo}
---

# Qwen3.5-9B Traffic Law LoRA (Vietnamese)

Fine-tuned LoRA adapter for Vietnamese traffic law Q&A, built on top of **Qwen/Qwen3.5-9B**.

## Task

Open-domain Q&A over Vietnamese traffic law regulations (2024-2025 corpus):
- Luật Đường bộ 35/2024/QH15
- Luật Trật tự, an toàn giao thông đường bộ 36/2024/QH15
- Nghị định 168/2024/NĐ-CP (xử phạt vi phạm)
- Nghị định 158/2024/NĐ-CP (vận tải đường bộ)
- Nghị định 165/2024/NĐ-CP (hướng dẫn thi hành)
- Nghị định 336/2025/NĐ-CP
- Thông tư 65/2024/TT-BCA (phục hồi điểm GPLX)

## Training

- Base model: Qwen/Qwen3.5-9B (4-bit NF4 QLoRA)
- LoRA rank: 16, alpha: 32
- Training data: ~500 QA pairs from local legal text (generated + verified)
- Context dropout: 70% keep (trains with RAG context), 30% drop (trains without)
- 5 epochs, LR 2e-4, cosine schedule

## Evaluation (Config D: LoRA + RAG)

See the [dataset card]({hf_user}/{dataset_repo}) for full metrics.

## Usage

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    "{hf_user}/{model_repo}",
    max_seq_length=2048,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)
```
"""


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for l in path.open(encoding="utf-8") if l.strip())


def upload_dataset(api, hf_user: str, repo_name: str) -> None:
    from huggingface_hub import DatasetCardData, HfApi
    import datasets

    repo_id = f"{hf_user}/{repo_name}"
    print(f"\n--- Uploading dataset → {repo_id} ---")

    # Build dataset dict from jsonl files
    splits = {}
    if QA_DATA_PATH.exists():
        train_items = []
        eval_items = []
        lines = [json.loads(l) for l in QA_DATA_PATH.open(encoding="utf-8") if l.strip()]
        total = len(lines)
        val_n = max(1, int(total * 0.1))
        eval_items = lines[-val_n:]
        train_items = lines[:-val_n]
        splits["train"] = datasets.Dataset.from_list(train_items)
        splits["validation"] = datasets.Dataset.from_list(eval_items)
        print(f"  train: {len(train_items)}, validation: {len(eval_items)}")

    if EVAL_DATA_PATH.exists():
        test_items = [json.loads(l) for l in EVAL_DATA_PATH.open(encoding="utf-8") if l.strip()]
        splits["test"] = datasets.Dataset.from_list(test_items)
        print(f"  test (open-ended): {len(test_items)}")

    if EVAL_MC_DATA_PATH.exists():
        mc_items = [json.loads(l) for l in EVAL_MC_DATA_PATH.open(encoding="utf-8") if l.strip()]
        splits["test_mc"] = datasets.Dataset.from_list(mc_items)
        print(f"  test_mc: {len(mc_items)}")

    if not splits:
        print("  No data files found, skipping dataset upload.")
        return

    ds = datasets.DatasetDict(splits)
    ds.push_to_hub(repo_id, token=api.token)
    print(f"  Dataset pushed to {repo_id}")


def upload_model(api, hf_user: str, repo_name: str, dataset_repo: str) -> None:
    repo_id = f"{hf_user}/{repo_name}"
    print(f"\n--- Uploading model adapter → {repo_id} ---")

    if not MODEL_DIR.exists():
        print(f"  Model directory not found: {MODEL_DIR}. Run finetune.py first.")
        return

    adapter_files = list(MODEL_DIR.glob("*.safetensors")) + list(MODEL_DIR.glob("*.bin"))
    if not adapter_files:
        print(f"  No adapter weights found in {MODEL_DIR}. Run finetune.py first.")
        return

    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

    # Upload all files in MODEL_DIR
    for fpath in sorted(MODEL_DIR.iterdir()):
        if fpath.is_file():
            print(f"  uploading {fpath.name} ({fpath.stat().st_size // 1024}KB)...")
            api.upload_file(
                path_or_fileobj=str(fpath),
                path_in_repo=fpath.name,
                repo_id=repo_id,
                repo_type="model",
            )

    # Upload model card
    card_content = MODEL_CARD_TEMPLATE.format(
        hf_user=hf_user, dataset_repo=dataset_repo, model_repo=repo_name
    )
    api.upload_file(
        path_or_fileobj=card_content.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
    )

    # Upload eval results if they exist
    reports_dir = REPORTS_DIR
    if reports_dir.exists():
        for rfile in reports_dir.glob("*.json"):
            print(f"  uploading report {rfile.name}...")
            api.upload_file(
                path_or_fileobj=str(rfile),
                path_in_repo=f"eval/{rfile.name}",
                repo_id=repo_id,
                repo_type="model",
            )

    print(f"  Model pushed to {repo_id}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-user", required=True, help="HuggingFace username")
    parser.add_argument("--dataset-repo", default="traffic-law-qa-vi")
    parser.add_argument("--model-repo", default="qwen35-9b-traffic-law-lora")
    parser.add_argument("--skip-dataset", action="store_true")
    parser.add_argument("--skip-model", action="store_true")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("Error: set HF_TOKEN environment variable first.")
        print("  export HF_TOKEN=hf_xxxxxxxxxxxx")
        sys.exit(1)

    try:
        from huggingface_hub import HfApi
        import datasets
    except ImportError:
        print("Install: pip install huggingface_hub datasets")
        sys.exit(1)

    api = HfApi(token=token)
    print(f"Logged in as: {api.whoami()['name']}")

    if not args.skip_dataset:
        upload_dataset(api, args.hf_user, args.dataset_repo)

    if not args.skip_model:
        upload_model(api, args.hf_user, args.model_repo, args.dataset_repo)

    print("\nAll done. Links:")
    print(f"  Dataset: https://huggingface.co/datasets/{args.hf_user}/{args.dataset_repo}")
    print(f"  Model:   https://huggingface.co/{args.hf_user}/{args.model_repo}")


if __name__ == "__main__":
    main()
