"""Gradio demo: Vietnamese Traffic Law Q&A System.

One question -> four model answers (A/B/C/D) shown side by side for direct comparison.

Configs:
  A — Base Qwen3.5-9B, no RAG
  B — Base Qwen3.5-9B + RAG
  C — LoRA fine-tuned Qwen3.5-9B, no RAG
  D — LoRA fine-tuned Qwen3.5-9B + RAG     ★ recommended

Models are lazy-loaded; one model is in VRAM at a time. The compare flow loads
the base model once for A & B, then swaps to LoRA for C & D. Initial run takes
~2 minutes; subsequent runs that don't change model order are fast.

Run:
  cd nlp && conda activate ai && python src/app.py
  # then open http://localhost:7860
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import re
import gc
import time
import torch
import gradio as gr

import unsloth  # must be first
from unsloth import FastLanguageModel

from config import (
    KB_PATH,
    MODEL_DIR_V2 as MODEL_DIR,
    MODEL_ID,
    TRAFFIC_QA_SYSTEM_PROMPT_NO_CONTEXT,
    TRAFFIC_QA_SYSTEM_PROMPT_WITH_CONTEXT,
)
from build_kb import load_vectorstore
from retrieval import retrieve_ranked_docs

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_NEW_TOKENS = 320
RAG_TOP_K      = 3
WAITING        = "⏳ Đang chờ..."
LOADING_BASE   = "⏳ Đang tải base model..."
LOADING_LORA   = "⏳ Đang tải LoRA model..."
GENERATING     = "⏳ Đang sinh đáp án..."

_SYSTEM_NO_CONTEXT   = TRAFFIC_QA_SYSTEM_PROMPT_NO_CONTEXT
_SYSTEM_WITH_CONTEXT = TRAFFIC_QA_SYSTEM_PROMPT_WITH_CONTEXT

# ---------------------------------------------------------------------------
# Lazy model loader — keeps only ONE model in VRAM at a time (16GB GPU)
# ---------------------------------------------------------------------------

_current_model     = None
_current_tokenizer = None
_current_type      = None   # "base" | "lora"


def _load_model(use_lora: bool):
    global _current_model, _current_tokenizer, _current_type

    model_type = "lora" if use_lora else "base"
    if _current_type == model_type:
        return _current_model, _current_tokenizer

    if _current_model is not None:
        print(f"  Unloading {_current_type} model...")
        del _current_model, _current_tokenizer
        _current_model = _current_tokenizer = None
        gc.collect()
        torch.cuda.empty_cache()

    path  = str(MODEL_DIR) if use_lora else MODEL_ID
    label = "fine-tuned (LoRA)" if use_lora else "base"
    print(f"  Loading {label} model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=path,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    _current_model     = model
    _current_tokenizer = tokenizer
    _current_type      = model_type
    print(f"  {label} model ready.")
    return model, tokenizer


# ---------------------------------------------------------------------------
# RAG vector store — load once at startup
# ---------------------------------------------------------------------------

print("Loading vector store (FAISS + BGE-M3)...")
_vs = load_vectorstore()
print("Vector store ready. Starting Gradio...\n")


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _source_label(doc, idx: int) -> str:
    md = doc.metadata or {}
    parts = [f"Nguồn {idx}", md.get("doc_id") or md.get("source") or "unknown"]
    article = md.get("article")
    if article:
        parts.append(article[:80])
    clause = md.get("clause_number")
    if clause:
        parts.append(f"khoản {clause}")
    point = md.get("point_letter")
    if point:
        parts.append(f"điểm {point}")
    return " · ".join(parts)


def _retrieve(question: str) -> tuple[str, str]:
    """Return (context_for_prompt, markdown_for_display)."""
    docs = retrieve_ranked_docs(_vs, question, top_k=RAG_TOP_K)
    if not docs:
        return "", "_Không tìm thấy đoạn luật phù hợp._"
    context_parts = []
    display_parts = []
    for idx, doc in enumerate(docs, start=1):
        md = doc.metadata or {}
        label = _source_label(doc, idx)
        title = md.get("title") or ""
        context_parts.append(f"[{label}]\n{doc.page_content}")
        display_parts.append(
            f"**{label}**\n\n*{title}*\n\n```\n{doc.page_content[:1200]}\n```"
        )
    return "\n\n---\n\n".join(context_parts), "\n\n---\n\n".join(display_parts)


def _generate(model, tokenizer, question: str, context: str | None) -> str:
    if context:
        user_content = f"Đoạn văn bản luật:\n{context}\n\nCâu hỏi: {question}"
        system_prompt = _SYSTEM_WITH_CONTEXT
    else:
        user_content = f"Câu hỏi: {question}"
        system_prompt = _SYSTEM_NO_CONTEXT
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_content},
    ]
    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    inputs = tokenizer(text=prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
        )
    new_ids = out_ids[0][inputs["input_ids"].shape[1]:]
    raw     = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    return re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()


# ---------------------------------------------------------------------------
# Compare-all handler (streaming)
# ---------------------------------------------------------------------------

def answer_all(question: str):
    """Yield (A, B, C, D, rag_md, status) progressively as each config finishes."""
    if not question or not question.strip():
        yield "", "", "", "", "", "❌ Vui lòng nhập câu hỏi."
        return

    t_start = time.time()

    # Step 1: retrieve once (used by B and D)
    context, rag_md = _retrieve(question)

    # Initial display: all four pending
    yield WAITING, WAITING, WAITING, WAITING, rag_md, LOADING_BASE

    # Step 2: load base model -> A, B
    base_model, base_tok = _load_model(use_lora=False)

    yield GENERATING, WAITING, WAITING, WAITING, rag_md, "🤖 A (base, no RAG)..."
    a_ans = _generate(base_model, base_tok, question, None)

    yield a_ans, GENERATING, WAITING, WAITING, rag_md, "🤖 B (base + RAG)..."
    b_ans = _generate(base_model, base_tok, question, context)

    yield a_ans, b_ans, WAITING, WAITING, rag_md, LOADING_LORA

    # Step 3: swap to LoRA -> C, D
    lora_model, lora_tok = _load_model(use_lora=True)

    yield a_ans, b_ans, GENERATING, WAITING, rag_md, "🤖 C (LoRA, no RAG)..."
    c_ans = _generate(lora_model, lora_tok, question, None)

    yield a_ans, b_ans, c_ans, GENERATING, rag_md, "🤖 D (LoRA + RAG)..."
    d_ans = _generate(lora_model, lora_tok, question, context)

    elapsed = time.time() - t_start
    yield a_ans, b_ans, c_ans, d_ans, rag_md, f"✅ Hoàn tất ({elapsed:.0f}s)"


def answer_d_only(question: str):
    """Fast path: only run config D (best). Yields the same 6 outputs."""
    if not question or not question.strip():
        yield "", "", "", "", "", "❌ Vui lòng nhập câu hỏi."
        return

    t_start = time.time()
    context, rag_md = _retrieve(question)

    yield "—", "—", "—", WAITING, rag_md, LOADING_LORA
    lora_model, lora_tok = _load_model(use_lora=True)

    yield "—", "—", "—", GENERATING, rag_md, "🤖 D (LoRA + RAG)..."
    d_ans = _generate(lora_model, lora_tok, question, context)

    elapsed = time.time() - t_start
    yield "—", "—", "—", d_ans, rag_md, f"✅ D hoàn tất ({elapsed:.0f}s)"


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

EXAMPLE_QUESTIONS = [
    "Người đi xe máy không đội mũ bảo hiểm bị phạt bao nhiêu?",
    "Lái ô tô vượt đèn đỏ bị xử phạt bao nhiêu tiền?",
    "Lái xe máy có nồng độ cồn dưới 0.25 mg/lít khí thở bị phạt như thế nào?",
    "Nghị định 168 có hiệu lực khi nào?",
    "Giấy phép lái xe có bao nhiêu điểm và làm sao để phục hồi?",
    "Tốc độ tối đa của xe ô tô trên đường cao tốc là bao nhiêu?",
    "Đi xe đạp vượt đèn đỏ bị phạt bao nhiêu tiền?",
    "Xe ưu tiên gồm những loại xe nào?",
]

CONFIG_CARDS = {
    "A": ("A · Base, No RAG",  "Qwen3.5-9B gốc trả lời trực tiếp, không tra cứu."),
    "B": ("B · Base + RAG",     "Qwen3.5-9B gốc + truy xuất văn bản luật."),
    "C": ("C · LoRA, No RAG",   "Qwen3.5-9B fine-tune trả lời trực tiếp."),
    "D": ("D · LoRA + RAG ★",   "Qwen3.5-9B fine-tune + truy xuất văn bản — cấu hình tốt nhất."),
}


DEMO_CSS = """
.answer-box textarea { font-size: 14px; }
.config-d { border: 2px solid #4f46e5; border-radius: 8px; padding: 4px; }
"""

with gr.Blocks(title="Hỏi đáp Luật Giao thông VN") as demo:
    gr.Markdown(
        """
        # 🚦 Hỏi đáp Luật Giao thông Đường bộ Việt Nam
        Nhập câu hỏi → so sánh đồng thời câu trả lời của **4 cấu hình A/B/C/D**.
        *Qwen3.5-9B · QLoRA · FAISS+BGE-M3 · BM25 · Cross-Encoder*
        """
    )

    with gr.Row():
        question_box = gr.Textbox(
            label="Câu hỏi",
            placeholder="Ví dụ: Lái xe máy có nồng độ cồn dưới 0.25 mg/lít khí thở bị phạt như thế nào?",
            lines=2,
            scale=4,
        )
        with gr.Column(scale=1, min_width=180):
            submit_btn  = gr.Button("So sánh 4 cấu hình 🔍", variant="primary", size="lg")
            d_only_btn  = gr.Button("Chỉ hỏi D (nhanh) ⚡", variant="secondary")

    status_box = gr.Markdown(value="_Sẵn sàng. Lượt đầu cần ~2 phút để tải model._")

    gr.Examples(
        examples=EXAMPLE_QUESTIONS,
        inputs=question_box,
        label="Câu hỏi mẫu",
    )

    with gr.Row(equal_height=True):
        with gr.Column():
            gr.Markdown(f"### {CONFIG_CARDS['A'][0]}\n*{CONFIG_CARDS['A'][1]}*")
            a_box = gr.Textbox(label="Trả lời", lines=8, interactive=False, elem_classes="answer-box")
        with gr.Column():
            gr.Markdown(f"### {CONFIG_CARDS['B'][0]}\n*{CONFIG_CARDS['B'][1]}*")
            b_box = gr.Textbox(label="Trả lời", lines=8, interactive=False, elem_classes="answer-box")

    with gr.Row(equal_height=True):
        with gr.Column():
            gr.Markdown(f"### {CONFIG_CARDS['C'][0]}\n*{CONFIG_CARDS['C'][1]}*")
            c_box = gr.Textbox(label="Trả lời", lines=8, interactive=False, elem_classes="answer-box")
        with gr.Column(elem_classes="config-d"):
            gr.Markdown(f"### {CONFIG_CARDS['D'][0]}\n*{CONFIG_CARDS['D'][1]}*")
            d_box = gr.Textbox(label="Trả lời", lines=8, interactive=False, elem_classes="answer-box")

    with gr.Accordion("📄 Văn bản luật được truy xuất (RAG context, dùng chung cho B & D)", open=False):
        rag_box = gr.Markdown(value="_Chưa có câu hỏi._")

    outputs = [a_box, b_box, c_box, d_box, rag_box, status_box]

    submit_btn.click(fn=answer_all,   inputs=question_box, outputs=outputs)
    d_only_btn.click(fn=answer_d_only, inputs=question_box, outputs=outputs)
    question_box.submit(fn=answer_all, inputs=question_box, outputs=outputs)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(primary_hue="blue"),
        css=DEMO_CSS,
    )
