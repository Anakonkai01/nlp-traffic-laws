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

# Optional: reuse the precomputed rewrite cache produced by
# scripts/build_rewrite_cache.py. If the user already ran it for their eval set,
# known questions get the Phase-9 legal-style rewrite for free; unknown ones
# fall through to the raw question.
_REWRITE_CACHE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "query_rewrite_cache.json",
)
_REWRITE_CACHE: dict[str, str] = {}
try:
    if os.path.exists(_REWRITE_CACHE_PATH):
        import json as _json
        with open(_REWRITE_CACHE_PATH, "r", encoding="utf-8") as _f:
            _REWRITE_CACHE = _json.load(_f)
        print(f"Loaded {_REWRITE_CACHE_PATH} ({len(_REWRITE_CACHE)} cached rewrites)")
except Exception as _exc:
    print(f"  (rewrite cache not loaded: {_exc})")

_REWRITE_CLIENT = None
_REWRITE_MODEL  = "google/gemini-2.0-flash-001"


def _live_rewrite(question: str) -> str:
    """Live legal-style rewrite via OpenRouter, with persistent cache.

    Returns the rewritten question on success; falls back to the original on
    any error or if OPENROUTER_API_KEY is not set.
    """
    global _REWRITE_CLIENT
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return question
    try:
        if _REWRITE_CLIENT is None:
            from openai import OpenAI
            _REWRITE_CLIENT = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
        system_prompt = (
            "Bạn là chuyên gia luật giao thông đường bộ Việt Nam. "
            "Hãy viết lại câu hỏi sau bằng văn phong pháp lý chuẩn của Nghị định 168/2024/NĐ-CP, "
            "Luật 35/2024/QH15 (Luật Đường bộ), và Luật 36/2024/QH15 (Luật Trật tự ATGT). "
            "Mục đích là để dùng câu viết lại làm truy vấn cho hệ thống tra cứu văn bản pháp luật.\n\n"
            "Quy tắc:\n"
            "1. GIỮ NGUYÊN ý nghĩa, đối tượng (xe gì), tình huống, và mọi con số cụ thể trong câu gốc.\n"
            "2. Thay thuật ngữ khẩu ngữ bằng thuật ngữ pháp lý:\n"
            "   - 'vượt đèn đỏ'/'vượt đèn vàng' -> 'không chấp hành hiệu lệnh của đèn tín hiệu giao thông'\n"
            "   - 'say rượu'/'uống bia'/'uống rượu' -> 'điều khiển xe trong khi trong máu hoặc hơi thở có nồng độ cồn'\n"
            "   - 'xe máy' (đứng một mình) -> 'xe mô tô, xe gắn máy'\n"
            "   - 'ô tô' (đứng một mình) -> 'xe ô tô'\n"
            "   - 'bằng lái' -> 'giấy phép lái xe'\n"
            "3. Nếu câu hỏi đã ở văn phong pháp lý thì giữ NGUYÊN VĂN.\n"
            "4. KHÔNG trả lời câu hỏi. KHÔNG thêm giải thích. Chỉ một câu hỏi viết lại trên một dòng."
        )
        resp = _REWRITE_CLIENT.chat.completions.create(
            model=_REWRITE_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": f"Câu hỏi gốc: {question}\n\nCâu hỏi pháp lý:"},
            ],
            max_tokens=160,
            temperature=0.0,
        )
        text = (resp.choices[0].message.content or "").strip()
        text = re.sub(r"^[\"'`*\-•\s]+", "", text)
        text = re.sub(r"^Câu hỏi pháp lý[:\-]*\s*", "", text)
        text = text.splitlines()[0].strip() if text else ""
        if text and len(text) > 5:
            _REWRITE_CACHE[question] = text
            try:
                import json as _json
                with open(_REWRITE_CACHE_PATH, "w", encoding="utf-8") as _f:
                    _json.dump(_REWRITE_CACHE, _f, ensure_ascii=False, indent=2)
            except Exception:
                pass
            return text
    except Exception as exc:
        print(f"  [rewrite] live call failed: {exc}")
    return question


_SYSTEM_NO_CONTEXT   = TRAFFIC_QA_SYSTEM_PROMPT_NO_CONTEXT
_SYSTEM_WITH_CONTEXT = TRAFFIC_QA_SYSTEM_PROMPT_WITH_CONTEXT

# ---------------------------------------------------------------------------
# Combined model loader — load LoRA-attached model ONCE.
# Toggle the LoRA adapter on/off via PEFT to switch between configs A/B (base
# behaviour) and C/D (fine-tuned) without reloading anything. This avoids the
# swap-OOM problem that hit `_load_model(use_lora=True)` on a 16 GB GPU after
# the base model had been resident.
# ---------------------------------------------------------------------------

_combined_model     = None
_combined_tokenizer = None


def _load_combined():
    global _combined_model, _combined_tokenizer
    if _combined_model is not None:
        return _combined_model, _combined_tokenizer
    print(f"  Loading LoRA-attached model from {MODEL_DIR}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(MODEL_DIR),
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    _combined_model     = model
    _combined_tokenizer = tokenizer
    print("  combined model ready.")
    return model, tokenizer


def _generate_for(use_lora: bool, question: str, context: str | None) -> str:
    """Generate with the LoRA adapter toggled on/off for the same loaded model."""
    model, tokenizer = _load_combined()
    if use_lora:
        return _generate(model, tokenizer, question, context)
    # Configs A and B: temporarily disable the LoRA adapter so we get base behaviour.
    with model.disable_adapter():
        return _generate(model, tokenizer, question, context)


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
    """Return (context_for_prompt, markdown_for_display).

    Uses the Phase-9 query rewrite (colloquial -> legal style) for retrieval
    when available in the cache. Generation still sees the original question.
    """
    rewritten = _REWRITE_CACHE.get(question)
    if rewritten is None:
        rewritten = _live_rewrite(question)
    if rewritten != question:
        print(f"  [rewrite] {question!r} -> {rewritten!r}")
    docs = retrieve_ranked_docs(_vs, rewritten, top_k=RAG_TOP_K)
    if not docs:
        return "", "_Không tìm thấy đoạn luật phù hợp._"
    context_parts = []
    display_parts = []
    if rewritten != question:
        display_parts.append(f"_Truy vấn pháp lý đã dùng_: **{rewritten}**")
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
    yield WAITING, WAITING, WAITING, WAITING, rag_md, "⏳ Đang tải model (lần đầu ~30s)..."

    # Step 2: load once (LoRA-attached); subsequent runs reuse it.
    _load_combined()

    # A: adapter disabled -> base behaviour, no RAG
    yield GENERATING, WAITING, WAITING, WAITING, rag_md, "🤖 A (base, no RAG)..."
    a_ans = _generate_for(use_lora=False, question=question, context=None)

    # B: adapter disabled -> base behaviour, with RAG
    yield a_ans, GENERATING, WAITING, WAITING, rag_md, "🤖 B (base + RAG)..."
    b_ans = _generate_for(use_lora=False, question=question, context=context)

    # C: adapter on -> LoRA, no RAG
    yield a_ans, b_ans, GENERATING, WAITING, rag_md, "🤖 C (LoRA, no RAG)..."
    c_ans = _generate_for(use_lora=True, question=question, context=None)

    # D: adapter on -> LoRA, with RAG
    yield a_ans, b_ans, c_ans, GENERATING, rag_md, "🤖 D (LoRA + RAG)..."
    d_ans = _generate_for(use_lora=True, question=question, context=context)

    elapsed = time.time() - t_start
    yield a_ans, b_ans, c_ans, d_ans, rag_md, f"✅ Hoàn tất ({elapsed:.0f}s)"


def answer_d_only(question: str):
    """Fast path: only run config D (best). Yields the same 6 outputs."""
    if not question or not question.strip():
        yield "", "", "", "", "", "❌ Vui lòng nhập câu hỏi."
        return

    t_start = time.time()
    context, rag_md = _retrieve(question)

    yield "—", "—", "—", WAITING, rag_md, "⏳ Đang tải model (lần đầu ~30s)..."
    _load_combined()

    yield "—", "—", "—", GENERATING, rag_md, "🤖 D (LoRA + RAG)..."
    d_ans = _generate_for(use_lora=True, question=question, context=context)

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
