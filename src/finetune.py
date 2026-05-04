"""
Fine-tune Qwen3.5-9B-Instruct on Vietnamese traffic-law QA using QLoRA (4-bit).

Project has 4 evaluation configs:
  A: base model,       no RAG  — no training needed, just inference
  B: base model,       + RAG   — no training needed, uses local-text-only KB
  C: fine-tuned model, no RAG  ← uses the adapter saved by this script
  D: fine-tuned model, + RAG   ← uses the same adapter + vector_db_traffic at inference

We always train WITH context in the prompt so the model learns to use
retrieved legal chunks. At inference time:
  - Config C: omit context field
  - Config D: fill context with chunks retrieved from vector_db_traffic
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import unsloth  # must be first — patches torch/transformers before they load

import json
import random

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
    MODEL_DIR_V2 as MODEL_DIR,
    MODEL_ID,
    QA_DATA_PATH,
    TRAFFIC_QA_SYSTEM_PROMPT_NO_CONTEXT,
    TRAFFIC_QA_SYSTEM_PROMPT_WITH_CONTEXT,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_PATH   = QA_DATA_PATH.parent / "qa_pairs_traffic_v3.jsonl"
OUTPUT_DIR  = MODEL_DIR  # where to save LoRA adapter

MAX_SEQ_LEN = 2048
LORA_R      = 32
LORA_ALPHA  = 64
TRAIN_EPOCHS = 2
BATCH_SIZE   = 2
GRAD_ACCUM   = 8
LR           = 5e-5
VAL_RATIO    = 0.1
SEED         = 42
CONTEXT_KEEP_PROB = 0.9

# System prompts differentiated by context presence (matches inference behavior)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_data() -> tuple[Dataset, Dataset]:
    """Read local-text-only qa_pairs_traffic.jsonl and return train/val datasets.

    Each line in the file is one JSON object:
      {"question": "...", "answer": "...", "context": "..."}
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Training data not found: {DATA_PATH}. Run python src/generate_qa.py first."
        )

    samples = []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            corpus = sample.get("corpus")
            if corpus not in {"local_text", "negative", "hard_context"}:
                raise ValueError(
                    f"Unsupported corpus={corpus!r} at {DATA_PATH}:{line_no}. "
                    "Regenerate QA with python src/generate_qa.py --force."
                )
            if sample.get("source_policy") != KB_SOURCE_POLICY:
                raise ValueError(
                    f"Missing/invalid source_policy at {DATA_PATH}:{line_no}. "
                    "Regenerate QA with python src/generate_qa.py --force."
                )
            if not sample.get("question") or not sample.get("answer") or not sample.get("context"):
                raise ValueError(f"Invalid QA sample at {DATA_PATH}:{line_no}.")
            samples.append(sample)

    if len(samples) < 10:
        raise ValueError(f"Not enough QA samples for fine-tuning: {len(samples)}")

    # Shuffle before splitting so validation set is representative
    random.seed(SEED)
    random.shuffle(samples)

    split = int(len(samples) * (1 - VAL_RATIO))
    return Dataset.from_list(samples[:split]), Dataset.from_list(samples[split:])


def format_sample(sample: dict, tokenizer) -> dict:
    """Convert one QA pair into a ChatML-formatted training string.

    Structure:
      system    → role description
      user      → [context if kept] + question
      assistant → answer  ← ONLY these tokens contribute to the loss

    Context dropout (CONTEXT_KEEP_PROB):
      70% of samples include the legal chunk → trains model for RAG mode (Config D).
      30% of samples omit the chunk → trains model for no-context mode (Config C).
      Without dropout, Config C runs out-of-distribution at inference time.

    apply_chat_template() inserts the special tokens the model expects:
      <|im_start|>system\\n...\\n<|im_end|>
      <|im_start|>user\\n...\\n<|im_end|>
      <|im_start|>assistant\\n...\\n<|im_end|>
    """
    keep_context = random.random() < CONTEXT_KEEP_PROB
    # hard_context examples always keep context (teaches model to handle wrong context gracefully)
    if sample.get("corpus") == "hard_context":
        keep_context = True

    if keep_context:
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
    # ── Step 1: Load base model in 4-bit ────────────────────────────────────
    # load_in_4bit=True enables NF4 quantization (QLoRA):
    #   - model weights stored as 4-bit integers → ~4× smaller than fp16
    #   - during forward pass: dequantize block → compute → discard
    #   - gradient flows through LoRA matrices only, not the 4-bit weights
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_ID,
        max_seq_length=MAX_SEQ_LEN,
        dtype=None,        # auto-detect: bf16 on Ampere+, fp16 otherwise
        load_in_4bit=True,
    )

    # ── Step 2: Attach LoRA adapters ────────────────────────────────────────
    # After this call, only the LoRA matrices (A and B) are trainable.
    # The original 4-bit weights are completely frozen.
    #
    # target_modules: which linear layers get LoRA adapters
    #   - q/k/v/o_proj: all 4 attention projections (Query, Key, Value, Output)
    #   - gate/up/down_proj: the 3 FFN projections in SwiGLU architecture
    # Targeting all 7 gives maximum task-specific capacity.
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",    # attention
            "gate_proj", "up_proj", "down_proj",         # FFN
        ],
        lora_alpha=LORA_ALPHA,
        lora_dropout=0,   # Dettmers et al. (QLoRA paper) found dropout=0 optimal
        bias="none",      # don't add bias terms to LoRA layers
        # gradient checkpointing: recompute activations during backward pass
        # instead of storing them — trades compute for VRAM
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
    )

    # ── Step 3: Prepare datasets ─────────────────────────────────────────────
    train_ds, val_ds = load_data()
    print(f"Train: {len(train_ds)} samples | Val: {len(val_ds)} samples")

    train_ds = train_ds.map(lambda x: format_sample(x, tokenizer))
    val_ds   = val_ds.map(lambda x: format_sample(x, tokenizer))

    # ── Step 4: Configure trainer ────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        max_seq_length=MAX_SEQ_LEN,    # moved here: TRL 5.x removed it from SFTConfig
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=1,  # eval pass needs full activations → use batch=1 to avoid OOM
            gradient_accumulation_steps=GRAD_ACCUM,  # effective batch = 16
            num_train_epochs=TRAIN_EPOCHS,
            learning_rate=LR,
            # bf16 on RTX 5070 Ti (Blackwell = Ampere+); fp16 as fallback
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            # adamw_8bit: AdamW optimizer stored in 8-bit → halves optimizer VRAM
            # optimizer normally stores 2 fp32 copies (m and v) per param → expensive
            optim="adamw_8bit",
            lr_scheduler_type="cosine",  # smoothly decay LR from peak to ~0
            warmup_ratio=0.05,           # first 5% of steps: linearly ramp up LR
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,  # restore best checkpoint after training
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            output_dir=str(OUTPUT_DIR / "checkpoints"),
            report_to="wandb" if _wandb_available() else "none",
            seed=SEED,
        ),
    )

    # ── Step 5: Train on responses only ─────────────────────────────────────
    # SFT objective: compute cross-entropy loss ONLY on assistant tokens.
    # Without this, the model also tries to predict the system/user prompts,
    # which wastes gradient capacity and can degrade answer quality.
    #
    # train_on_responses_only masks the loss on all tokens before
    # <|im_start|>assistant\n, so only answer tokens get gradients.
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    # ── Step 6: Train ───────────────────────────────────────────────────────
    trainer.train()

    # ── Step 7: Save LoRA adapter ────────────────────────────────────────────
    # save_pretrained saves ONLY the LoRA adapter weights (~100MB),
    # not the full 9B base model (~18GB).
    # At inference: load base model + merge this adapter.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print(f"\nLoRA adapter saved → {OUTPUT_DIR}")


if __name__ == "__main__":
    finetune()
