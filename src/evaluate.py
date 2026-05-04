"""
Evaluate 4 configs on Vietnamese traffic-law QA.

  A — base model,        no RAG
  B — base model,        + RAG (FAISS vector_db_traffic)
  C — fine-tuned model,  no RAG
  D — fine-tuned model,  + RAG

Metrics:
  ROUGE-1, ROUGE-2, ROUGE-L  — n-gram overlap
  BLEU-4                      — corpus-level precision (sacrebleu, 13a tokenizer, normalized to [0,1])
  METEOR                      — unigram F1 with synonym/stemming alignment (better than BLEU for VN)
  F1-token                    — word-level F1 (SQuAD-style) — partial-credit answer matching
  BERTScore-F1 (PhoBERT)      — semantic similarity
  Exact Match                 — strict string equality
  Recall@5                    — retrieval: original context found in top-5 retrieved chunks
  LLM-Judge (MiniMax)         — 1-5 score from LLM grader, normalized to [0,1] (opt-in via --judge)

Usage:
  python src/evaluate.py                              # use data/eval_manual.jsonl
  python src/evaluate.py --test-file PATH             # custom test JSONL
  python src/evaluate.py --configs A B                # run specific configs
  python src/evaluate.py --configs A B --samples 10   # quick smoke-test
  python src/evaluate.py --judge                      # also run LLM-as-Judge (needs MINIMAX_API_KEY env)

Results saved to: reports/traffic/evaluation_results.json
Predictions saved to: reports/traffic/predictions_all_configs.json
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import json
import random
import time

import torch
from tqdm import tqdm
from rouge_score import rouge_scorer as rs
from sacrebleu.metrics import BLEU as SacreBLEU

from config import (
    EVAL_DATA_PATH,
    FORBIDDEN_LEGACY_TERMS,
    KB_PATH,
    MODEL_DIR_V2 as LORA_PATH_DEFAULT,
    MODEL_ID,
    QA_DATA_PATH,
    REFUSAL_ANSWER,
    REPORTS_DIR,
    SOURCE_MANIFEST_PATH,
    TRAFFIC_QA_SYSTEM_PROMPT_NO_CONTEXT,
    TRAFFIC_QA_SYSTEM_PROMPT_WITH_CONTEXT,
)
from build_kb import load_vectorstore
from corpus import load_source_manifest
from retrieval import classify_intents, expand_traffic_query, retrieve_ranked_docs

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_PATH        = QA_DATA_PATH
TEST_MANUAL_PATH = EVAL_DATA_PATH
LEGACY_TEST_MANUAL_PATH = DATA_PATH.parent / "test_manual.jsonl"
LORA_PATH        = LORA_PATH_DEFAULT
RESULTS_PATH     = REPORTS_DIR / "evaluation_results.json"
PREDS_PATH       = REPORTS_DIR / "predictions_all_configs.json"
RETRIEVAL_DIAGNOSTICS_PATH = REPORTS_DIR / "retrieval_diagnostics.json"

_SYSTEM_NO_CONTEXT   = TRAFFIC_QA_SYSTEM_PROMPT_NO_CONTEXT
_SYSTEM_WITH_CONTEXT = TRAFFIC_QA_SYSTEM_PROMPT_WITH_CONTEXT

MAX_NEW_TOKENS    = 256   # traffic-law answers; 128 was too small when thinking leaked through
EVAL_BATCH_SIZE   = 8     # samples per GPU batch during evaluation
RAG_TOP_K         = 3     # chunks used as context during generation
RAG_TOP_K_RECALL  = 5     # chunks retrieved for Recall@k metric
RAG_MIN_SCORE     = -3.0  # cross-encoder score threshold: if best score < this, skip context (fallback to knowledge)
SEED              = 42
VAL_RATIO         = 0.1   # must match finetune.py
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_JUDGE_MODEL_DEFAULT = "google/gemini-2.0-flash-001"
MINIMAX_BASE_URL  = "https://api.minimax.chat/v1"
MINIMAX_MODEL_DEFAULT = "MiniMax-Text-01"

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_test_data(test_file: str | None = None, n_samples: int | None = None) -> list[dict]:
    """Load test data.

    Priority:
      1. --test-file PATH     explicit file (e.g. data/test_manual.jsonl)
      2. data/eval_manual.jsonl  production local-text eval set
      3. data/test_manual.jsonl  legacy fallback
      4. val split of qa_pairs_traffic.jsonl  fallback (last 10% after shuffle seed=42)
    """
    path = None
    if test_file:
        from pathlib import Path as _P
        path = _P(test_file)
    elif TEST_MANUAL_PATH.exists():
        path = TEST_MANUAL_PATH
    elif LEGACY_TEST_MANUAL_PATH.exists():
        path = LEGACY_TEST_MANUAL_PATH

    if path:
        samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
        if n_samples:
            samples = samples[:n_samples]
        print(f"Test set: {len(samples)} samples (from {path.name})")
        return samples

    # Fallback: val split from generated QA pairs
    samples = []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    random.seed(SEED)
    random.shuffle(samples)
    split = int(len(samples) * (1 - VAL_RATIO))
    test = samples[split:]
    if n_samples:
        test = test[:n_samples]
    print(f"Test set: {len(test)} samples (val split from {DATA_PATH.name})")
    return test


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(use_lora: bool):
    """Load base or fine-tuned model in 4-bit with Unsloth."""
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
    FastLanguageModel.for_inference(model)  # enable optimized inference kernel
    return model, tokenizer


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def build_prompt(tokenizer, question: str, context: str | None) -> str:
    """Build ChatML prompt — same format used during fine-tuning."""
    if context:
        user_content = f"Đoạn văn bản luật:\n{context}\n\nCâu hỏi: {question}"
        system_prompt = _SYSTEM_WITH_CONTEXT
    else:
        user_content = f"Câu hỏi: {question}"
        system_prompt = _SYSTEM_NO_CONTEXT
    messages = [
        {"role": "system",    "content": system_prompt},
        {"role": "user",      "content": user_content},
    ]
    # add_generation_prompt=True appends <|im_start|>assistant\n so the model
    # continues from the assistant turn instead of predicting the prompt.
    # enable_thinking=False suppresses Qwen3's chain-of-thought reasoning block;
    # without it the model outputs "Thinking Process:..." (100-400 tokens) before
    # the actual answer, causing MAX_NEW_TOKENS to be exhausted before any answer.
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        # Fallback for tokenizers that don't support enable_thinking
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def generate_answer(model, tokenizer, question: str, context: str | None) -> str:
    """Run greedy decoding and return the assistant's response text only."""
    import re
    prompt = build_prompt(tokenizer, question, context)
    # Use text= keyword to avoid Unsloth's VL processor treating the string as an image path
    inputs = tokenizer(text=prompt, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,          # greedy — deterministic, faster
            temperature=1.0,          # ignored when do_sample=False
            use_cache=True,
        )

    # Decode only the newly generated tokens (skip the prompt)
    new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    raw = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    # Strip Qwen3 thinking blocks (<think>...</think>) — keep only the final answer
    return re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()


def generate_answers_batch(
    model, tokenizer,
    questions: list[str],
    contexts: list[str | None],
) -> list[str]:
    """Batched greedy decoding — processes EVAL_BATCH_SIZE samples at once.

    Decoder-only models require left-padding so all sequences in a batch end
    at the same position and generation starts immediately after the last token.
    """
    import re

    prompts = [build_prompt(tokenizer, q, ctx) for q, ctx in zip(questions, contexts)]

    tokenizer.padding_side = "left"
    inputs = tokenizer(
        text=prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1800,   # reserve room for MAX_NEW_TOKENS output
    ).to(model.device)

    with torch.inference_mode():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    prompt_len = inputs["input_ids"].shape[1]   # padded length, same for all in batch
    results = []
    for ids in out_ids:
        raw = tokenizer.decode(ids[prompt_len:], skip_special_tokens=True).strip()
        results.append(re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip())
    return results


# ---------------------------------------------------------------------------
# RAG retrieval
# ---------------------------------------------------------------------------

def build_retriever():
    """Load FAISS vector store and return (vectorstore, retriever)."""
    print("Loading vector store for RAG...")
    vs = load_vectorstore()
    return vs, None


def _source_label(doc, idx: int) -> str:
    md = doc.metadata or {}
    parts = [
        f"Nguồn {idx}",
        md.get("doc_id") or md.get("source") or "unknown",
    ]
    if md.get("article"):
        parts.append(md["article"])
    return " | ".join(parts)


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


def _format_context_chunk(doc, idx: int) -> str:
    md = doc.metadata or {}
    header = (
        f"[{_source_label(doc, idx)}]\n"
        f"Tiêu đề: {md.get('title') or ''}\n"
        f"File: {md.get('source_path') or ''}"
    )
    return f"{header}\n\n{doc.page_content}"


def retrieve_context(vs, question: str) -> tuple[str, list[dict]]:
    """Retrieve top-k chunks, returning source-labeled context + source metadata."""
    docs = retrieve_ranked_docs(vs, question, top_k=RAG_TOP_K, min_score=RAG_MIN_SCORE)
    context_parts = []
    sources = []
    for idx, doc in enumerate(docs, start=1):
        sources.append(_source_record(doc, idx))
        context_parts.append(_format_context_chunk(doc, idx))
    return "\n\n---\n\n".join(context_parts), sources


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_rouge(predictions: list[str], references: list[str]) -> tuple[float, float, float]:
    """Compute mean ROUGE-1, ROUGE-2, ROUGE-L F1 over all pairs."""
    scorer = rs.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    r1, r2, rl = [], [], []
    for pred, ref in zip(predictions, references):
        s = scorer.score(ref, pred)
        r1.append(s["rouge1"].fmeasure)
        r2.append(s["rouge2"].fmeasure)
        rl.append(s["rougeL"].fmeasure)
    n = len(predictions)
    return sum(r1) / n, sum(r2) / n, sum(rl) / n


def compute_bleu(predictions: list[str], references: list[str]) -> float:
    """Compute corpus BLEU-4 (sacrebleu, Moses 13a tokenizer), normalized to [0, 1]."""
    metric = SacreBLEU(tokenize="13a")
    result = metric.corpus_score(predictions, [references])
    return result.score / 100


def load_phobert():
    """Load PhoBERT tokenizer and model once — pass to compute_bert_score each call."""
    from transformers import AutoTokenizer, AutoModel
    PHOBERT = "vinai/phobert-base-v2"
    print("Loading PhoBERT for BERTScore...")
    tok = AutoTokenizer.from_pretrained(PHOBERT)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mdl = AutoModel.from_pretrained(PHOBERT).eval().to(device)
    print(f"  PhoBERT on {device}")
    return tok, mdl


def compute_bert_score(
    predictions: list[str],
    references: list[str],
    phobert_tok,
    phobert_mdl,
) -> float:
    """Compute mean BERTScore F1 using PhoBERT via direct transformers call.

    We bypass the bert_score library because it has a known compatibility bug
    with PhoBERT's RoBERTa position offset (max_position_embeddings=258),
    causing an out-of-bounds error on sequences near the max length.

    Algorithm (same as standard BERTScore):
      1. Embed each token of prediction (P) and reference (R) using PhoBERT
      2. Precision: for each token in P, find max cosine-sim with any token in R → average
      3. Recall:    for each token in R, find max cosine-sim with any token in P → average
      4. F1 = harmonic mean of Precision and Recall
    """
    print("  Computing BERTScore with PhoBERT...")

    device = next(phobert_mdl.parameters()).device

    def embed(texts: list[str]) -> list[torch.Tensor]:
        result = []
        for text in texts:
            enc = phobert_tok(
                text, return_tensors="pt",
                truncation=True, max_length=254,  # 254 + 2 special = 256 < 258 limit
                padding=False,
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.inference_mode():
                hidden = phobert_mdl(**enc).last_hidden_state[0]  # (seq, hidden)
            result.append(hidden[1:-1])  # exclude [CLS] and [SEP]
        return result

    pred_embs = embed(predictions)
    ref_embs  = embed(references)

    f1_scores = []
    for p_emb, r_emb in zip(pred_embs, ref_embs):
        p_norm = torch.nn.functional.normalize(p_emb, dim=-1)  # (m, d)
        r_norm = torch.nn.functional.normalize(r_emb, dim=-1)  # (n, d)
        sim = p_norm @ r_norm.T                                 # (m, n)
        precision = sim.max(dim=1).values.mean().item()
        recall    = sim.max(dim=0).values.mean().item()
        denom = precision + recall
        f1_scores.append((2 * precision * recall / denom) if denom > 0 else 0.0)

    return sum(f1_scores) / len(f1_scores)


def compute_meteor(predictions: list[str], references: list[str]) -> float:
    """Mean METEOR via nltk — better than BLEU for Vietnamese open-ended QA.

    Uses word-level unigram matching with recall bias.
    sacrebleu 2.x dropped METEOR support; nltk implementation is equivalent.
    """
    import nltk
    from nltk.translate.meteor_score import meteor_score as _meteor
    from nltk.tokenize import word_tokenize
    nltk.download("wordnet", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    scores = [
        _meteor([word_tokenize(ref.lower())], word_tokenize(pred.lower()))
        for pred, ref in zip(predictions, references)
    ]
    return sum(scores) / len(scores) if scores else 0.0


def compute_f1_token(predictions: list[str], references: list[str]) -> float:
    """Mean token-level F1 (SQuAD-style).

    For each pair: tokenise → compute word-level precision, recall, F1.
    Rewards partial matches — a prediction mentioning half the key terms
    still scores 0.5 rather than 0 (unlike Exact Match).
    """
    import re

    def _tokens(text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())

    scores = []
    for pred, ref in zip(predictions, references):
        p_toks = _tokens(pred)
        r_toks = _tokens(ref)
        if not p_toks and not r_toks:
            scores.append(1.0)
            continue
        if not p_toks or not r_toks:
            scores.append(0.0)
            continue
        common = set(p_toks) & set(r_toks)
        if not common:
            scores.append(0.0)
            continue
        prec = len(common) / len(p_toks)
        rec  = len(common) / len(r_toks)
        scores.append(2 * prec * rec / (prec + rec))
    return sum(scores) / len(scores)


def compute_llm_judge(
    questions: list[str],
    predictions: list[str],
    references: list[str],
    api_key: str,
    model: str = MINIMAX_MODEL_DEFAULT,
    backend: str = "openrouter",
) -> float:
    """LLM-as-Judge: ask an LLM (OpenRouter or MiniMax) to score each prediction 1-5, return mean normalised to [0,1]."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("pip install openai  # needed for LLM-as-Judge")

    if backend == "openrouter":
        client = OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)
    else:
        client = OpenAI(api_key=api_key, base_url=MINIMAX_BASE_URL)

    JUDGE_SYSTEM = (
        "Bạn là chuyên gia đánh giá hệ thống hỏi đáp về Luật Giao thông Đường bộ Việt Nam. "
        "Nhiệm vụ của bạn là chấm điểm câu trả lời của mô hình AI so với đáp án chuẩn. "
        "Chỉ trả lời bằng đúng một chữ số từ 1 đến 5, không giải thích thêm."
    )

    JUDGE_TEMPLATE = """\
Câu hỏi: {question}

Đáp án chuẩn: {reference}

Câu trả lời của mô hình: {prediction}

Thang điểm:
5 — Hoàn toàn chính xác và đầy đủ
4 — Phần lớn chính xác, thiếu vài chi tiết nhỏ
3 — Đúng về đại thể nhưng thiếu thông tin quan trọng
2 — Có phần đúng nhưng sai nhiều điểm
1 — Sai hoàn toàn hoặc không liên quan

Điểm (1-5):"""

    scores = []
    failed = 0
    print(f"  LLM-Judge: scoring {len(predictions)} samples via {backend} ({model})...")

    for q, pred, ref in tqdm(
        zip(questions, predictions, references),
        total=len(predictions),
        desc="  LLM-Judge",
    ):
        prompt = JUDGE_TEMPLATE.format(question=q, reference=ref, prediction=pred)
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user",   "content": prompt},
                ],
                max_tokens=8,
                temperature=0.0,
            )
            text = resp.choices[0].message.content.strip()
            import re as _re
            m = _re.search(r"[1-5]", text)
            if m:
                scores.append(int(m.group()))
            else:
                failed += 1
        except Exception as e:
            print(f"\n    Warning: judge API error — {e}")
            failed += 1

    if not scores:
        print("  LLM-Judge: all requests failed, returning 0.0")
        return 0.0

    if failed:
        print(f"  LLM-Judge: {failed} samples failed/skipped")

    mean_score = sum(scores) / len(scores)
    print(f"  LLM-Judge mean score: {mean_score:.3f}/5  → {mean_score/5:.4f} (normalised)")
    return mean_score / 5.0   # normalise to [0, 1]


def compute_exact_match(predictions: list[str], references: list[str]) -> float:
    """Fraction of predictions that exactly match the reference."""
    return sum(p.strip() == r.strip() for p, r in zip(predictions, references)) / len(predictions)


def _expected_doc_ids(item: dict) -> set[str]:
    expected = item.get("expected_doc_ids") or []
    if isinstance(expected, str):
        expected = [expected]
    return {str(x) for x in expected if str(x)}


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


def validate_eval_data(test_data: list[dict]) -> dict:
    enabled_doc_ids = _enabled_manifest_doc_ids()
    stats = {
        "n_samples": len(test_data),
        "missing_category": 0,
        "supported_missing_expected_doc_ids": 0,
        "unsupported_with_expected_doc_ids": 0,
        "expected_doc_ids_not_in_manifest": {},
        "measurable_source_rows": 0,
    }
    invalid_counts: dict[str, int] = {}
    for item in test_data:
        category = item.get("category")
        expected = _expected_doc_ids(item)
        if not category:
            stats["missing_category"] += 1
        if category == "unsupported":
            if expected:
                stats["unsupported_with_expected_doc_ids"] += 1
        elif expected:
            stats["measurable_source_rows"] += 1
        else:
            stats["supported_missing_expected_doc_ids"] += 1
        for doc_id in expected:
            if enabled_doc_ids and doc_id not in enabled_doc_ids:
                invalid_counts[doc_id] = invalid_counts.get(doc_id, 0) + 1
    stats["expected_doc_ids_not_in_manifest"] = invalid_counts
    if any(v for k, v in stats.items() if k not in {"n_samples", "measurable_source_rows", "expected_doc_ids_not_in_manifest"}):
        print("\nEval data warnings:")
        for key, value in stats.items():
            if key == "expected_doc_ids_not_in_manifest":
                if value:
                    print(f"  {key}: {value}")
            elif key not in {"n_samples", "measurable_source_rows"} and value:
                print(f"  {key}: {value}")
    return stats


def compute_context_recall_at_k(vs, test_data: list[dict], k: int = 5) -> float:
    """ROUGE-L recall against original context chunks; only rows with context are counted."""
    scorer = rs.RougeScorer(["rougeL"], use_stemmer=False)
    hits, total = 0, 0
    for item in tqdm(test_data, desc=f"  Context Recall@{k}"):
        ctx = item.get("context", "")
        if not ctx:
            continue
        total += 1
        for doc in retrieve_ranked_docs(vs, item["question"], top_k=k, candidate_k=max(12, k * 4)):
            if scorer.score(ctx, doc.page_content)["rougeL"].fmeasure >= 0.5:
                hits += 1
                break
    return hits / total if total > 0 else 0.0


def compute_source_recall_at_k(vs, test_data: list[dict], k: int = 5) -> float | None:
    hits, total = 0, 0
    for item in tqdm(test_data, desc=f"  Source Recall@{k}"):
        expected = _expected_doc_ids(item)
        if not expected:
            continue
        total += 1
        docs = retrieve_ranked_docs(vs, item["question"], top_k=k, candidate_k=max(30, k * 8))
        got = {doc.metadata.get("doc_id") or doc.metadata.get("source") for doc in docs if doc.metadata}
        if expected & got:
            hits += 1
    return (hits / total) if total else None


def compute_source_hit_breakdown(test_data: list[dict], retrieved_sources: list[list[dict]]) -> dict[str, float]:
    totals: dict[str, int] = {}
    hits: dict[str, int] = {}
    for item, sources in zip(test_data, retrieved_sources):
        expected = _expected_doc_ids(item)
        if not expected:
            continue
        got = _source_ids(sources)
        for doc_id in expected:
            totals[doc_id] = totals.get(doc_id, 0) + 1
            if doc_id in got:
                hits[doc_id] = hits.get(doc_id, 0) + 1
    return {doc_id: round(hits.get(doc_id, 0) / total, 4) for doc_id, total in sorted(totals.items())}


def compute_recall_at_k(vs, test_data: list[dict], k: int = 5) -> float:
    return compute_context_recall_at_k(vs, test_data, k)


def compute_forbidden_legacy_rate(predictions: list[str]) -> float:
    if not predictions:
        return 0.0
    hits = 0
    lowered_terms = [term.lower() for term in FORBIDDEN_LEGACY_TERMS]
    for pred in predictions:
        p = pred.lower()
        if any(term in p for term in lowered_terms):
            hits += 1
    return hits / len(predictions)


def compute_source_hit_rate(test_data: list[dict], retrieved_sources: list[list[dict]]) -> float | None:
    hits = 0
    total = 0
    for item, sources in zip(test_data, retrieved_sources):
        expected = _expected_doc_ids(item)
        if not expected:
            continue
        total += 1
        if expected & _source_ids(sources):
            hits += 1
    return (hits / total) if total else None


def compute_refusal_rate(test_data: list[dict], predictions: list[str]) -> float | None:
    total = 0
    refused = 0
    refusal_key = "không tìm thấy căn cứ"
    for item, pred in zip(test_data, predictions):
        if item.get("category") != "unsupported":
            continue
        total += 1
        p = pred.lower()
        if REFUSAL_ANSWER.lower() in p or refusal_key in p:
            refused += 1
    return (refused / total) if total else None


def compute_false_refusal_rate(test_data: list[dict], predictions: list[str]) -> float | None:
    """Fraction of SUPPORTED questions that are incorrectly refused."""
    total = 0
    falsely_refused = 0
    refusal_keywords = ["không tìm thấy căn cứ", "không thể trả lời", "không đủ thông tin",
                        "không có thông tin", "không thuộc phạm vi"]
    for item, pred in zip(test_data, predictions):
        if item.get("category") == "unsupported":
            continue
        total += 1
        p = pred.lower()
        if any(kw in p for kw in refusal_keywords):
            falsely_refused += 1
    return (falsely_refused / total) if total else None


def run_retrieval_diagnostics(vs, test_data: list[dict]) -> dict:
    scorer = rs.RougeScorer(["rougeL"], use_stemmer=False)
    rows = []
    source_hits = 0
    source_total = 0
    context_hits = 0
    context_total = 0
    by_doc_totals: dict[str, int] = {}
    by_doc_hits: dict[str, int] = {}

    for idx, item in enumerate(tqdm(test_data, desc="Retrieval diagnostics"), start=1):
        question = item["question"]
        docs = retrieve_ranked_docs(vs, question, top_k=RAG_TOP_K_RECALL, candidate_k=max(30, RAG_TOP_K_RECALL * 8))
        sources = [_source_record(doc, rank) for rank, doc in enumerate(docs, start=1)]
        expected = _expected_doc_ids(item)
        got = _source_ids(sources)
        source_hit = bool(expected & got) if expected else None
        if expected:
            source_total += 1
            if source_hit:
                source_hits += 1
            for doc_id in expected:
                by_doc_totals[doc_id] = by_doc_totals.get(doc_id, 0) + 1
                if doc_id in got:
                    by_doc_hits[doc_id] = by_doc_hits.get(doc_id, 0) + 1

        context_hit = None
        ctx = item.get("context") or ""
        if ctx:
            context_total += 1
            context_hit = any(scorer.score(ctx, doc.page_content)["rougeL"].fmeasure >= 0.5 for doc in docs)
            if context_hit:
                context_hits += 1

        rows.append({
            "index": idx,
            "question": question,
            "category": item.get("category") or "",
            "expected_doc_ids": sorted(expected),
            "expanded_query": expand_traffic_query(question),
            "intents": classify_intents(question),
            "source_hit": source_hit,
            "context_rouge_hit": context_hit,
            "retrieved": sources,
        })

    summary = {
        "n_samples": len(test_data),
        "source_recall_at_5": round(source_hits / source_total, 4) if source_total else None,
        "source_recall_denominator": source_total,
        "context_rouge_recall_at_5": round(context_hits / context_total, 4) if context_total else None,
        "context_rouge_denominator": context_total,
        "source_hit_rate_by_doc": {
            doc_id: round(by_doc_hits.get(doc_id, 0) / total, 4)
            for doc_id, total in sorted(by_doc_totals.items())
        },
    }
    return {"summary": summary, "samples": rows}


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate_config(
    config_name: str,
    model,
    tokenizer,
    test_data: list[dict],
    use_rag: bool,
    phobert_tok,
    phobert_mdl,
    retriever=None,
    llm_judge_fn=None,   # callable(questions, preds, refs) -> float, or None
) -> dict:
    """Run inference on all test samples and return metrics + predictions."""
    print(f"\n{'='*60}")
    print(f"Config {config_name}: {'base' if config_name in ('A','B') else 'fine-tuned'} model, {'+ RAG' if use_rag else 'no RAG'}")
    print(f"{'='*60}")

    predictions = []
    references  = [d["answer"] for d in test_data]
    latencies   = []
    retrieved_sources: list[list[dict]] = []

    for batch_start in tqdm(
        range(0, len(test_data), EVAL_BATCH_SIZE),
        desc=f"Config {config_name}",
        total=(len(test_data) + EVAL_BATCH_SIZE - 1) // EVAL_BATCH_SIZE,
    ):
        batch = test_data[batch_start : batch_start + EVAL_BATCH_SIZE]
        batch_questions = [item["question"] for item in batch]
        batch_contexts = []
        if use_rag:
            for q in batch_questions:
                context, sources = retrieve_context(retriever, q)
                batch_contexts.append(context)
                retrieved_sources.append(sources)
        else:
            batch_contexts = [None for _ in batch_questions]
            retrieved_sources.extend([[] for _ in batch_questions])

        t0      = time.time()
        answers = generate_answers_batch(model, tokenizer, batch_questions, batch_contexts)
        elapsed = time.time() - t0

        predictions.extend(answers)
        latencies.extend([elapsed / len(batch)] * len(batch))

    questions   = [d["question"] for d in test_data]
    rouge_1, rouge_2, rouge_l = compute_rouge(predictions, references)
    bleu        = compute_bleu(predictions, references)
    meteor      = compute_meteor(predictions, references)
    f1_token    = compute_f1_token(predictions, references)
    bert_f1     = compute_bert_score(predictions, references, phobert_tok, phobert_mdl)
    exact_match = compute_exact_match(predictions, references)
    forbidden_legacy = compute_forbidden_legacy_rate(predictions)
    source_hit_rate = compute_source_hit_rate(test_data, retrieved_sources) if use_rag else None
    refusal_rate = compute_refusal_rate(test_data, predictions)
    false_refusal_rate = compute_false_refusal_rate(test_data, predictions)
    avg_latency = sum(latencies) / len(latencies)

    llm_judge = None
    if llm_judge_fn is not None:
        llm_judge = llm_judge_fn(questions, predictions, references)

    metrics = {
        "config":        config_name,
        "n_samples":     len(test_data),
        "rouge_1":       round(rouge_1, 4),
        "rouge_2":       round(rouge_2, 4),
        "rouge_l":       round(rouge_l, 4),
        "bleu":          round(bleu, 4),
        "meteor":        round(meteor, 4),
        "f1_token":      round(f1_token, 4),
        "bert_score_f1": round(bert_f1, 4),
        "exact_match":   round(exact_match, 4),
        "forbidden_legacy_rate": round(forbidden_legacy, 4),
        "avg_latency_s": round(avg_latency, 2),
    }
    if source_hit_rate is not None:
        metrics["source_hit_rate"] = round(source_hit_rate, 4)
        metrics["source_hit_rate_top_3"] = round(source_hit_rate, 4)
        metrics["source_hit_rate_by_doc"] = compute_source_hit_breakdown(test_data, retrieved_sources)
    if refusal_rate is not None:
        metrics["unsupported_refusal_rate"] = round(refusal_rate, 4)
    if false_refusal_rate is not None:
        metrics["false_refusal_rate"] = round(false_refusal_rate, 4)
    if llm_judge is not None:
        metrics["llm_judge"] = round(llm_judge, 4)

    print(f"\nResults — Config {config_name}:")
    print(f"  ROUGE-1:      {rouge_1:.4f}")
    print(f"  ROUGE-2:      {rouge_2:.4f}")
    print(f"  ROUGE-L:      {rouge_l:.4f}")
    print(f"  BLEU-4:       {bleu:.4f}")
    print(f"  METEOR:       {meteor:.4f}")
    print(f"  F1-token:     {f1_token:.4f}")
    print(f"  BERTScore-F1: {bert_f1:.4f}")
    print(f"  Exact Match:  {exact_match:.4f}")
    print(f"  Legacy rate:  {forbidden_legacy:.4f}")
    if source_hit_rate is not None:
        print(f"  Source hit:   {source_hit_rate:.4f}")
    if refusal_rate is not None:
        print(f"  Refusal:      {refusal_rate:.4f}")
    if llm_judge is not None:
        print(f"  LLM-Judge:    {llm_judge:.4f}  (= {llm_judge*5:.2f}/5)")
    print(f"  Avg latency:  {avg_latency:.2f}s/sample")

    return {
        "metrics":     metrics,
        "predictions": predictions,
        "references":  references,
        "questions":   questions,
        "retrieved_sources": retrieved_sources,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs", nargs="+", choices=["A", "B", "C", "D"],
        default=["A", "B", "C", "D"],
        help="Which configs to evaluate (default: all)",
    )
    parser.add_argument(
        "--test-file", type=str, default=None,
        help="Path to JSONL test file (default: data/test_manual.jsonl or val split)",
    )
    parser.add_argument(
        "--samples", type=int, default=None,
        help="Limit test set size",
    )
    parser.add_argument(
        "--judge", action="store_true",
        help="Enable LLM-as-Judge via OpenRouter or MiniMax",
    )
    parser.add_argument(
        "--judge-model", type=str, default=OPENROUTER_JUDGE_MODEL_DEFAULT,
        help=f"Model for judging (default: {OPENROUTER_JUDGE_MODEL_DEFAULT})",
    )
    parser.add_argument(
        "--judge-backend", type=str, choices=["openrouter", "minimax"], default="openrouter",
        help="Judge backend: openrouter (default) or minimax",
    )
    parser.add_argument(
        "--retrieval-only", action="store_true",
        help="Run retrieval diagnostics without loading/generating with LLMs",
    )
    args = parser.parse_args()

    # Set up LLM judge function if requested
    llm_judge_fn = None
    if args.judge:
        if args.judge_backend == "openrouter":
            judge_api_key = os.environ.get("OPENROUTER_API_KEY", "")
            if not judge_api_key:
                print("WARNING: --judge requires OPENROUTER_API_KEY env var. Judge disabled.")
            else:
                def llm_judge_fn(questions, predictions, references):
                    return compute_llm_judge(
                        questions, predictions, references,
                        api_key=judge_api_key,
                        model=args.judge_model,
                        backend="openrouter",
                    )
        else:
            judge_api_key = os.environ.get("MINIMAX_API_KEY", "")
            if not judge_api_key:
                print("WARNING: --judge (minimax) requires MINIMAX_API_KEY env var. Judge disabled.")
            else:
                def llm_judge_fn(questions, predictions, references):
                    return compute_llm_judge(
                        questions, predictions, references,
                        api_key=judge_api_key,
                        model=args.judge_model,
                        backend="minimax",
                    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    test_data = load_test_data(args.test_file, args.samples)
    eval_validation = validate_eval_data(test_data)

    if args.retrieval_only:
        vs, _ = build_retriever()
        diagnostics = run_retrieval_diagnostics(vs, test_data)
        diagnostics["eval_validation"] = eval_validation
        with open(RETRIEVAL_DIAGNOSTICS_PATH, "w", encoding="utf-8") as f:
            json.dump(diagnostics, f, ensure_ascii=False, indent=2)
        print("\nRetrieval diagnostics summary:")
        print(json.dumps(diagnostics["summary"], ensure_ascii=False, indent=2))
        print(f"Saved: {RETRIEVAL_DIAGNOSTICS_PATH}")
        return

    # Check which configs are runnable
    lora_ready = (LORA_PATH / "adapter_config.json").exists()
    if any(c in args.configs for c in ("C", "D")) and not lora_ready:
        print(f"\nWARNING: LoRA adapter not found at {LORA_PATH}")
        print("Configs C and D will be skipped. Run finetune.py first.")
        args.configs = [c for c in args.configs if c not in ("C", "D")]

    rag_needed = any(c in args.configs for c in ("B", "D"))

    # Load PhoBERT once and reuse across all 4 configs
    phobert_tok, phobert_mdl = load_phobert()

    # Load vector store once — used for both RAG generation and Recall@5
    vs = None
    retriever = None
    vs_available = KB_PATH.exists() and any(KB_PATH.iterdir())
    if rag_needed or vs_available:
        try:
            vs, retriever = build_retriever()
        except Exception as e:
            print(f"Warning: Could not load vector store: {e}")
            if rag_needed:
                print("RAG configs (B, D) will be skipped.")
                args.configs = [c for c in args.configs if c not in ("B", "D")]
                rag_needed = False

    # Compute retrieval metrics once — independent of generation configs
    context_recall_at_5 = None
    source_recall_at_5 = None
    if vs is not None:
        print(f"\nComputing Retrieval Recall@{RAG_TOP_K_RECALL}...")
        context_recall_at_5 = compute_context_recall_at_k(vs, test_data, k=RAG_TOP_K_RECALL)
        source_recall_at_5 = compute_source_recall_at_k(vs, test_data, k=RAG_TOP_K_RECALL)
        print(f"  Context Recall@{RAG_TOP_K_RECALL}: {context_recall_at_5:.4f}")
        if source_recall_at_5 is not None:
            print(f"  Source Recall@{RAG_TOP_K_RECALL}:  {source_recall_at_5:.4f}")

    all_metrics     = {}
    all_predictions = {}

    # Group configs by model to avoid loading the same model twice
    # Order: A → B (base, no/rag), then C → D (lora, no/rag)
    for use_lora, group_label in [(False, "base"), (True, "lora")]:
        group = []
        if not use_lora:
            if "A" in args.configs: group.append(("A", False))
            if "B" in args.configs: group.append(("B", True))
        else:
            if "C" in args.configs: group.append(("C", False))
            if "D" in args.configs: group.append(("D", True))

        if not group:
            continue

        model, tokenizer = load_model(use_lora)

        for config_name, use_rag in group:
            result = evaluate_config(
                config_name, model, tokenizer, test_data, use_rag,
                phobert_tok, phobert_mdl,
                vs if use_rag else None,
                llm_judge_fn=llm_judge_fn,
            )
            all_metrics[config_name]     = result["metrics"]
            all_predictions[config_name] = {
                "questions":   result["questions"],
                "references":  result["references"],
                "predictions": result["predictions"],
                "retrieved_sources": result["retrieved_sources"],
            }

        # Free VRAM before loading next model
        del model, tokenizer
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    # Attach retrieval metrics to every config's metrics (shared retrieval metrics)
    for cfg_metrics in all_metrics.values():
        cfg_metrics["eval_validation"] = eval_validation
        if context_recall_at_5 is not None:
            cfg_metrics["context_rouge_recall_at_5"] = round(context_recall_at_5, 4)
            cfg_metrics["recall_at_5"] = round(context_recall_at_5, 4)
        if source_recall_at_5 is not None:
            cfg_metrics["source_recall_at_5"] = round(source_recall_at_5, 4)

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    print("\n" + "="*75)
    print("FINAL SUMMARY")
    print("="*75)
    if context_recall_at_5 is not None:
        print(f"Context Recall@{RAG_TOP_K_RECALL}: {context_recall_at_5:.4f}")
    if source_recall_at_5 is not None:
        print(f"Source Recall@{RAG_TOP_K_RECALL}:  {source_recall_at_5:.4f}")
    if context_recall_at_5 is not None or source_recall_at_5 is not None:
        print()
    has_judge = any("llm_judge" in m for m in all_metrics.values())
    has_false_refusal = any("false_refusal_rate" in m for m in all_metrics.values())
    header = (
        f"{'Config':<8} {'R-1':>7} {'R-2':>7} {'R-L':>7} "
        f"{'BLEU':>7} {'METEOR':>8} {'F1-tok':>8} {'BERTSc':>8} {'EM':>7} {'Legacy':>8}"
        + (f" {'Judge':>7}" if has_judge else "")
        + (f" {'FRef':>7}" if has_false_refusal else "")
        + f" {'Lat':>7}"
    )
    print(header)
    print("-" * len(header))
    for cfg in ["A", "B", "C", "D"]:
        if cfg not in all_metrics:
            continue
        m = all_metrics[cfg]
        judge_col = f" {m['llm_judge']:>7.4f}" if has_judge else ""
        fref_col = f" {m['false_refusal_rate']:>7.4f}" if has_false_refusal and "false_refusal_rate" in m else ""
        print(
            f"  {cfg:<6} {m['rouge_1']:>7.4f} {m['rouge_2']:>7.4f} {m['rouge_l']:>7.4f} "
            f"{m['bleu']:>7.4f} {m['meteor']:>8.4f} {m['f1_token']:>8.4f} "
            f"{m['bert_score_f1']:>8.4f} {m['exact_match']:>7.4f} {m['forbidden_legacy_rate']:>8.4f}"
            + judge_col + fref_col
            + f" {m['avg_latency_s']:>6.1f}s"
        )

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)
    with open(PREDS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_predictions, f, ensure_ascii=False, indent=2)

    print(f"\nSaved: {RESULTS_PATH}")
    print(f"Saved: {PREDS_PATH}")


if __name__ == "__main__":
    main()
