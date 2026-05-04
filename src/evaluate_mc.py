"""
Evaluate 4 configs on a local, manifest-controlled traffic-law multiple-choice set.

Input: data/eval_mc_manual.jsonl
Each row:
  {"question": "...", "choices": ["...", "...", "...", "..."], "answer": "A"}
  choices may have 2-4 items (GPLX exam questions have 2-3 choices).
"""

import argparse
import json
import os
import re
import time

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from tqdm import tqdm

from build_kb import load_vectorstore
from config import (
    EVAL_MC_DATA_PATH,
    MODEL_DIR,
    MODEL_ID,
    REPORTS_DIR,
    SOURCE_MANIFEST_PATH,
    TRAFFIC_MC_SYSTEM_PROMPT_NO_CONTEXT,
    TRAFFIC_MC_SYSTEM_PROMPT_WITH_CONTEXT,
)
from corpus import load_source_manifest
from retrieval import retrieve_ranked_docs


LORA_PATH = MODEL_DIR
MC_RESULTS_PATH = REPORTS_DIR / "mc_results.json"
MC_PREDS_PATH = REPORTS_DIR / "mc_predictions.json"

MAX_NEW_TOKENS = 64
RAG_TOP_K = 3
LABELS = ["A", "B", "C", "D"]


def _answer_label(item: dict) -> str:
    answer = item["answer"]
    if isinstance(answer, int):
        return LABELS[answer]
    answer = str(answer).strip().upper()
    if answer not in LABELS:
        raise ValueError(f"Invalid MC answer label: {answer!r}")
    return answer


def load_mc_data(n_samples: int | None = None) -> list[dict]:
    if not EVAL_MC_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Local MC eval file not found: {EVAL_MC_DATA_PATH}. "
            "Create data/eval_mc_manual.jsonl before running evaluate_mc.py."
        )
    data = []
    with EVAL_MC_DATA_PATH.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            n_choices = len(item.get("choices") or [])
            if not (2 <= n_choices <= 4):
                raise ValueError(f"Expected 2-4 choices at {EVAL_MC_DATA_PATH}:{line_no}, got {n_choices}")
            _answer_label(item)
            data.append(item)
    if n_samples:
        data = data[:n_samples]
    print(f"Loaded {len(data)} local MC questions from {EVAL_MC_DATA_PATH}.")
    return data


def load_model(use_lora: bool):
    import unsloth  # must precede FastLanguageModel import so Unsloth patches load correctly
    from unsloth import FastLanguageModel

    path = str(LORA_PATH) if use_lora else MODEL_ID
    label = "fine-tuned (+ LoRA)" if use_lora else "base"
    print(f"\nLoading {label} model from: {path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=path,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def build_retriever():
    print("Loading local-text-only vector store for RAG...")
    return load_vectorstore()


def _expected_doc_ids(item: dict) -> set[str]:
    if item.get("source_measurable") is False:
        return set()
    expected = item.get("expected_doc_ids") or []
    if isinstance(expected, str):
        expected = [expected]
    return {str(x) for x in expected if str(x)}


def _source_record(doc, idx: int) -> dict:
    md = doc.metadata or {}
    return {
        "rank": idx,
        "doc_id": md.get("doc_id") or md.get("source") or "",
        "source": md.get("source") or "",
        "title": md.get("title") or "",
        "article": md.get("article") or "",
        "article_number": md.get("article_number") or "",
        "chunk_id": md.get("chunk_id") or "",
        "source_path": md.get("source_path") or "",
        "corpus": md.get("corpus") or "",
    }


def _source_ids(sources: list[dict]) -> set[str]:
    return {s.get("doc_id") or s.get("source") for s in sources if s.get("doc_id") or s.get("source")}


def _enabled_manifest_doc_ids() -> set[str]:
    try:
        manifest = load_source_manifest(SOURCE_MANIFEST_PATH)
    except Exception:
        return set()
    return {
        str(doc.get("doc_id"))
        for doc in manifest.get("documents", [])
        if doc.get("enabled", True) and doc.get("doc_id")
    }


def validate_mc_data(data: list[dict]) -> dict:
    enabled_doc_ids = _enabled_manifest_doc_ids()
    stats = {
        "n_samples": len(data),
        "source_measurable_rows": 0,
        "source_unmeasurable_rows": 0,
        "expected_doc_ids_not_in_manifest": {},
    }
    invalid_counts: dict[str, int] = {}
    for item in data:
        expected = _expected_doc_ids(item)
        if expected:
            stats["source_measurable_rows"] += 1
        else:
            stats["source_unmeasurable_rows"] += 1
        for doc_id in expected:
            if enabled_doc_ids and doc_id not in enabled_doc_ids:
                invalid_counts[doc_id] = invalid_counts.get(doc_id, 0) + 1
    stats["expected_doc_ids_not_in_manifest"] = invalid_counts
    if invalid_counts:
        print(f"MC eval warning: expected_doc_ids_not_in_manifest={invalid_counts}")
    return stats


def retrieve_context(vs, question: str) -> tuple[str, list[dict]]:
    docs = retrieve_ranked_docs(vs, question, top_k=RAG_TOP_K)
    parts = []
    sources = []
    for idx, doc in enumerate(docs, start=1):
        md = doc.metadata or {}
        sources.append(_source_record(doc, idx))
        header = (
            f"[Nguồn {idx} | {md.get('doc_id') or md.get('source') or 'unknown'}"
            f" | {md.get('article') or ''}]\n"
            f"Tiêu đề: {md.get('title') or ''}\n"
            f"File: {md.get('source_path') or ''}"
        )
        parts.append(f"{header}\n\n{doc.page_content}")
    return "\n\n---\n\n".join(parts), sources


def _item_labels(item: dict) -> list[str]:
    """Return the answer labels actually available for this item (A/B/C/D up to len(choices))."""
    return LABELS[: len(item["choices"])]


def build_mc_prompt(tokenizer, item: dict, context: str | None) -> str:
    labels = _item_labels(item)
    choices_text = "\n".join(f"{label}. {text}" for label, text in zip(labels, item["choices"]))
    if context:
        user_content = f"Đoạn văn bản luật liên quan:\n{context}\n\nCâu hỏi: {item['question']}\n{choices_text}"
        system_prompt = TRAFFIC_MC_SYSTEM_PROMPT_WITH_CONTEXT
    else:
        user_content = f"Câu hỏi: {item['question']}\n{choices_text}"
        system_prompt = TRAFFIC_MC_SYSTEM_PROMPT_NO_CONTEXT

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def parse_answer(raw: str, valid_labels: list[str] | None = None) -> str | None:
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    allowed = "|".join(valid_labels) if valid_labels else "ABCD"
    match = re.search(rf"\b([{allowed}])\b", text)
    return match.group(1) if match else None


def generate_answer(model, tokenizer, item: dict, context: str | None) -> tuple[str, str | None]:
    prompt = build_mc_prompt(tokenizer, item, context)
    inputs = tokenizer(text=prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=1.0,
            use_cache=True,
        )
    new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    raw = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    return raw, parse_answer(raw, _item_labels(item))


def compute_source_hit_rate(data: list[dict], retrieved_sources: list[list[dict]]) -> float | None:
    hits = 0
    total = 0
    for item, sources in zip(data, retrieved_sources):
        expected = _expected_doc_ids(item)
        if not expected:
            continue
        total += 1
        if expected & _source_ids(sources):
            hits += 1
    return (hits / total) if total else None


def evaluate_config(config_name: str, model, tokenizer, data: list[dict], use_rag: bool, retriever=None) -> dict:
    print(f"\n{'=' * 60}")
    print(f"Config {config_name}: {'base' if config_name in ('A', 'B') else 'fine-tuned'}, {'+ RAG' if use_rag else 'no RAG'}")
    print(f"{'=' * 60}")

    predictions = []
    raw_outputs = []
    latencies = []
    retrieved_sources: list[list[dict]] = []
    for item in tqdm(data, desc=f"Config {config_name}"):
        if use_rag:
            context, sources = retrieve_context(retriever, item["question"])
            retrieved_sources.append(sources)
        else:
            context = None
            retrieved_sources.append([])
        t0 = time.time()
        raw, pred_label = generate_answer(model, tokenizer, item, context)
        latencies.append(time.time() - t0)
        predictions.append(pred_label)
        raw_outputs.append(raw)

    truth = [_answer_label(item) for item in data]
    correct = sum(pred == gold for pred, gold in zip(predictions, truth) if pred is not None)
    unparsed = sum(p is None for p in predictions)
    accuracy = correct / len(data)
    avg_latency = sum(latencies) / len(latencies)
    source_hit_rate = compute_source_hit_rate(data, retrieved_sources) if use_rag else None

    print(f"\nResults - Config {config_name}:")
    print(f"  Accuracy:     {accuracy:.4f} ({correct}/{len(data)} correct)")
    print(f"  Unparsed:     {unparsed}/{len(data)}")
    if source_hit_rate is not None:
        print(f"  Source hit:   {source_hit_rate:.4f}")
    print(f"  Avg latency:  {avg_latency:.2f}s/sample")

    metrics = {
        "config": config_name,
        "n_samples": len(data),
        "accuracy": round(accuracy, 4),
        "n_correct": correct,
        "n_unparsed": unparsed,
        "avg_latency_s": round(avg_latency, 2),
    }
    if source_hit_rate is not None:
        metrics["source_hit_rate"] = round(source_hit_rate, 4)

    return {
        "metrics": metrics,
        "predictions": [
            {
                "question": item["question"],
                "choices": item["choices"],
                "ground_truth": gold,
                "predicted": pred,
                "raw_output": raw,
                "correct": pred == gold,
                "retrieved_sources": sources,
            }
            for item, gold, pred, raw, sources in zip(data, truth, predictions, raw_outputs, retrieved_sources)
        ],
    }


def main() -> None:
    import gc

    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", choices=["A", "B", "C", "D"], default=["A", "B", "C", "D"])
    parser.add_argument("--samples", type=int, default=None)
    args = parser.parse_args()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    data = load_mc_data(args.samples)
    mc_validation = validate_mc_data(data)

    lora_ready = (LORA_PATH / "adapter_config.json").exists()
    if any(c in args.configs for c in ("C", "D")) and not lora_ready:
        print("WARNING: LoRA adapter not found. Skipping C and D.")
        args.configs = [c for c in args.configs if c not in ("C", "D")]

    rag_needed = any(c in args.configs for c in ("B", "D"))
    if rag_needed:
        load_vectorstore()

    all_metrics = {}
    all_predictions = {}
    for use_lora, _ in [(False, "base"), (True, "lora")]:
        group = []
        if not use_lora:
            if "A" in args.configs:
                group.append(("A", False))
            if "B" in args.configs:
                group.append(("B", True))
        else:
            if "C" in args.configs:
                group.append(("C", False))
            if "D" in args.configs:
                group.append(("D", True))
        if not group:
            continue

        model, tokenizer = load_model(use_lora)
        retriever = build_retriever() if any(use_rag for _, use_rag in group) else None
        for config_name, use_rag in group:
            result = evaluate_config(config_name, model, tokenizer, data, use_rag, retriever)
            result["metrics"]["mc_validation"] = mc_validation
            all_metrics[config_name] = result["metrics"]
            all_predictions[config_name] = result["predictions"]

        del model, tokenizer
        if retriever:
            del retriever
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("FINAL SUMMARY - Local MC")
    print("=" * 60)
    header = f"{'Config':<8} {'Accuracy':>10} {'Correct':>10} {'Unparsed':>10} {'Latency(s)':>12}"
    print(header)
    print("-" * len(header))
    for cfg in ["A", "B", "C", "D"]:
        if cfg not in all_metrics:
            continue
        m = all_metrics[cfg]
        print(
            f"  {cfg:<6} {m['accuracy']:>10.4f} "
            f"{m['n_correct']:>4}/{m['n_samples']:<5} "
            f"{m['n_unparsed']:>10} "
            f"{m['avg_latency_s']:>12.2f}"
        )

    MC_RESULTS_PATH.write_text(json.dumps(all_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    MC_PREDS_PATH.write_text(json.dumps(all_predictions, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved: {MC_RESULTS_PATH}")
    print(f"Saved: {MC_PREDS_PATH}")


if __name__ == "__main__":
    main()
