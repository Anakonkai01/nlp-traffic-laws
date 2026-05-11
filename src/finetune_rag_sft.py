"""Fine-tune Qwen3.5-9B on Vietnamese traffic-law QA using RAG-SFT format.

Config E (bonus ablation on top of 4 mandatory configs): the fine-tuned model
learns to answer from evidence cards (structure-parsed compact cards) rather
than raw retrieved legal text. This addresses the train-test distribution shift
between LoRA-v2 (trained on raw context) and the production retrieval path
(which serves cards through evidence_card_from_text).

Pipeline:
  sample -> evidence_card_from_text(question, context, metadata) -> card text
  prompt = system + "Thẻ căn cứ:\n<card>\n\nCâu hỏi: <q>"
  target = answer

Writes LoRA adapter to models/qwen3.5-9b-lora-traffic-rag-sft-v1/ (separate
from the canonical models/qwen3.5-9b-lora-traffic-v2/ used by configs C and D).
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import unsloth  # must be first

import json
import random
from pathlib import Path

def _wandb_available() -> bool:
    try:
        import wandb
        return wandb.api.api_key is not None
    except Exception:
        return False

import torch
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from unsloth import FastLanguageModel, train_on_responses_only

from config import (
    KB_SOURCE_POLICY,
    PROJECT_ROOT,
    MODEL_ID,
    QA_DATA_PATH,
    QA_DEV_PATH,
    TRAFFIC_QA_SYSTEM_PROMPT_NO_CONTEXT,
    TRAFFIC_QA_SYSTEM_PROMPT_WITH_CONTEXT,
)
from evidence_cards import evidence_card_from_text

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUT_DIR = PROJECT_ROOT / "models" / "qwen3.5-9b-lora-traffic-rag-sft-v2"

MAX_SEQ_LEN = 2048
LORA_R      = 32
LORA_ALPHA  = 64
TRAIN_EPOCHS = 1
BATCH_SIZE   = 2
GRAD_ACCUM   = 8
LR           = 3e-5
SEED         = 42
# Mix 3 formats in training so model doesn't over-specialise on cards:
#   50% evidence card,  40% raw context,  10% no-context.
# v1 used 90% card / 10% no-context and over-refused when retrieval missed a field.
CARD_PROB      = 0.50
RAW_PROB       = 0.40
# remainder (0.10) = no-context


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _read_jsonl_checked(path) -> list[dict]:
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            corpus = sample.get("corpus")
            if corpus not in {"local_text", "negative", "hard_context"}:
                raise ValueError(f"Unsupported corpus={corpus!r} at {path}:{line_no}.")
            if sample.get("source_policy") != KB_SOURCE_POLICY:
                raise ValueError(f"Missing/invalid source_policy at {path}:{line_no}.")
            if not sample.get("question") or not sample.get("answer") or not sample.get("context"):
                raise ValueError(f"Invalid QA sample at {path}:{line_no}.")
            samples.append(sample)
    return samples


def load_data() -> tuple[Dataset, Dataset]:
    if not QA_DATA_PATH.exists() or not QA_DEV_PATH.exists():
        raise FileNotFoundError(f"Missing split files: train={QA_DATA_PATH}, dev={QA_DEV_PATH}")
    train_samples = _read_jsonl_checked(QA_DATA_PATH)
    dev_samples = _read_jsonl_checked(QA_DEV_PATH)
    print(f"Training data: train={len(train_samples)} from {QA_DATA_PATH}; dev={len(dev_samples)} from {QA_DEV_PATH}")
    return Dataset.from_list(train_samples), Dataset.from_list(dev_samples)


def _card_from_sample(sample: dict) -> str:
    metadata = {
        "doc_id": sample.get("doc_id") or sample.get("source", ""),
        "source": sample.get("source", ""),
        "article": sample.get("article", ""),
        "title": sample.get("title", ""),
        "source_path": sample.get("source_path", ""),
    }
    return evidence_card_from_text(sample["question"], sample["context"], metadata)


def format_sample(sample: dict, tokenizer) -> dict:
    r = random.random()
    if sample.get("corpus") == "hard_context":
        # hard-context examples: always show a real context (raw) so model learns robustness
        user_content = f"Đoạn văn bản luật:\n{sample['context']}\n\nCâu hỏi: {sample['question']}"
        system_prompt = TRAFFIC_QA_SYSTEM_PROMPT_WITH_CONTEXT
    elif r < CARD_PROB:
        card = _card_from_sample(sample)
        user_content = f"{card}\n\nCâu hỏi: {sample['question']}"
        system_prompt = TRAFFIC_QA_SYSTEM_PROMPT_WITH_CONTEXT
    elif r < CARD_PROB + RAW_PROB:
        user_content = f"Đoạn văn bản luật:\n{sample['context']}\n\nCâu hỏi: {sample['question']}"
        system_prompt = TRAFFIC_QA_SYSTEM_PROMPT_WITH_CONTEXT
    else:
        user_content = f"Câu hỏi: {sample['question']}"
        system_prompt = TRAFFIC_QA_SYSTEM_PROMPT_NO_CONTEXT

    messages = [
        {"role": "system",    "content": system_prompt},
        {"role": "user",      "content": user_content},
        {"role": "assistant", "content": sample["answer"]},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def finetune() -> None:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_ID,
        max_seq_length=MAX_SEQ_LEN,
        dtype=None,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=LORA_ALPHA,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
    )

    train_ds, val_ds = load_data()
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    train_ds = train_ds.map(lambda x: format_sample(x, tokenizer))
    val_ds   = val_ds.map(lambda x: format_sample(x, tokenizer))

    print("\n=== First 2 training samples (check card format) ===")
    for i in range(2):
        print(f"\n--- sample {i} ---\n{train_ds[i]['text'][:800]}\n...")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        max_seq_length=MAX_SEQ_LEN,
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=GRAD_ACCUM,
            num_train_epochs=TRAIN_EPOCHS,
            learning_rate=LR,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            optim="adamw_8bit",
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            output_dir=str(OUTPUT_DIR / "checkpoints"),
            report_to="wandb" if _wandb_available() else "none",
            seed=SEED,
        ),
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    trainer.train()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print(f"\nLoRA adapter saved → {OUTPUT_DIR}")


if __name__ == "__main__":
    finetune()
