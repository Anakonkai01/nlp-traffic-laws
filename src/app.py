"""
Gradio demo: Vietnamese Traffic Law Q&A System
4 configs: A (base, no RAG), B (base + RAG), C (fine-tuned, no RAG), D (fine-tuned + RAG)

Models are lazy-loaded and swapped on demand to avoid OOM with 16GB VRAM.

Run:
  cd nlp && conda activate ai && python src/app.py
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import re
import gc
import torch
import gradio as gr

import unsloth  # must be first
from unsloth import FastLanguageModel

from config import (
    KB_PATH,
    MODEL_DIR,
    MODEL_ID,
    TRAFFIC_QA_SYSTEM_PROMPT_NO_CONTEXT,
    TRAFFIC_QA_SYSTEM_PROMPT_WITH_CONTEXT,
)
from build_kb import load_vectorstore
from retrieval import retrieve_ranked_docs

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_NEW_TOKENS = 300
RAG_TOP_K      = 3

_SYSTEM_NO_CONTEXT   = TRAFFIC_QA_SYSTEM_PROMPT_NO_CONTEXT
_SYSTEM_WITH_CONTEXT = TRAFFIC_QA_SYSTEM_PROMPT_WITH_CONTEXT

CONFIG_LABELS = {
    "A": "A — Base model, No RAG",
    "B": "B — Base model + RAG",
    "C": "C — Fine-tuned, No RAG",
    "D": "D — Fine-tuned + RAG  ✓ (recommended)",
}

# ---------------------------------------------------------------------------
# Lazy model loader — keeps only ONE model in VRAM at a time
# ---------------------------------------------------------------------------

_current_model     = None
_current_tokenizer = None
_current_type      = None   # "base" | "lora"


def _load_model(use_lora: bool):
    global _current_model, _current_tokenizer, _current_type

    model_type = "lora" if use_lora else "base"
    if _current_type == model_type:
        return _current_model, _current_tokenizer

    # Unload previous model to free VRAM
    if _current_model is not None:
        print(f"Unloading {_current_type} model...")
        del _current_model, _current_tokenizer
        _current_model = _current_tokenizer = None
        gc.collect()
        torch.cuda.empty_cache()

    path  = str(MODEL_DIR) if use_lora else MODEL_ID
    label = "fine-tuned (LoRA)" if use_lora else "base"
    print(f"Loading {label} model...")
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
# RAG — loaded once at startup (embeddings only, ~1GB)
# ---------------------------------------------------------------------------

print("Loading vector store (RAG)...")
_vs        = load_vectorstore()
print("Vector store ready. Starting Gradio...")


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _source_label(doc, idx: int) -> str:
    md = doc.metadata or {}
    bits = [
        f"Nguồn {idx}",
        md.get("doc_id") or md.get("source") or "unknown",
    ]
    article = md.get("article")
    if article:
        bits.append(article)
    return " | ".join(bits)


def _retrieve(question: str) -> tuple[str, list[tuple[str, str]]]:
    docs = retrieve_ranked_docs(_vs, question, top_k=RAG_TOP_K)
    context_parts = []
    display_parts = []
    for idx, doc in enumerate(docs, start=1):
        md = doc.metadata or {}
        label = _source_label(doc, idx)
        title = md.get("title") or ""
        source_path = md.get("source_path") or ""
        header = f"[{label}]\nTiêu đề: {title}\nFile: {source_path}"
        context_parts.append(f"{header}\n\n{doc.page_content}")
        display_parts.append((header, doc.page_content))
    return "\n\n---\n\n".join(context_parts), display_parts


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
# Main handler
# ---------------------------------------------------------------------------

def answer(question: str, config: str) -> tuple[str, str]:
    if not question.strip():
        return "Vui lòng nhập câu hỏi.", ""

    use_rag  = config in ("B", "D")
    use_lora = config in ("C", "D")

    model, tokenizer = _load_model(use_lora)

    context, chunks = _retrieve(question) if use_rag else ("", [])
    response        = _generate(model, tokenizer, question, context if use_rag else None)

    rag_display = ""
    if use_rag and chunks:
        rag_display = "\n\n---\n\n".join(
            f"**{header}**\n\n{content}" for header, content in chunks
        )

    return response, rag_display


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

EXAMPLE_QUESTIONS = [
    "Người điều khiển xe ô tô có nồng độ cồn vượt 80mg/100ml máu bị phạt bao nhiêu?",
    "Hành vi lạng lách đánh võng trên đường bộ bị xử lý như thế nào?",
    "Tốc độ tối đa của xe con trên đường cao tốc là bao nhiêu?",
    "Điều kiện để được cấp giấy phép lái xe hạng B là gì?",
    "Xe ưu tiên gồm những loại xe nào?",
    "Người đi xe máy không đội mũ bảo hiểm bị phạt bao nhiêu?",
    "Điểm giấy phép lái xe hoạt động như thế nào?",
    "Khi gặp đèn đỏ, người tham gia giao thông phải làm gì?",
]

with gr.Blocks(title="Hỏi đáp Luật Giao thông VN", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🚦 Hệ thống Hỏi đáp Luật Giao thông Đường bộ Việt Nam
        Dựa trên các file text đã bật trong `docs/docs_giaothong/manifest.json`.
        Fine-tuned: **Qwen3.5-9B** + **QLoRA** | RAG: **FAISS** + **BGE-M3** | Source policy: **local_text_only**

        > ⚠️ Đổi giữa config A/B ↔ C/D sẽ cần **~30-60 giây** để swap model.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            config_radio = gr.Radio(
                choices=list(CONFIG_LABELS.keys()),
                value="D",
                label="Config",
                info="A=Base/NoRAG | B=Base+RAG | C=FineTuned/NoRAG | D=FineTuned+RAG",
            )
            gr.Markdown(
                """
                | Config | Model | RAG |
                |--------|-------|-----|
                | A | Base | ✗ |
                | B | Base | ✓ |
                | C | Fine-tuned | ✗ |
                | **D** | **Fine-tuned** | **✓** |
                """
            )

        with gr.Column(scale=3):
            question_box = gr.Textbox(
                label="Câu hỏi",
                placeholder="Ví dụ: Người điều khiển xe máy không đội mũ bảo hiểm bị phạt bao nhiêu?",
                lines=3,
            )
            submit_btn  = gr.Button("Hỏi 🔍", variant="primary")
            answer_box  = gr.Textbox(label="Câu trả lời", lines=6, interactive=False)

    with gr.Accordion("📄 Văn bản luật được truy xuất (RAG)", open=False):
        rag_box = gr.Markdown(value="_Chọn config B hoặc D để xem văn bản tham chiếu._")

    gr.Examples(
        examples=EXAMPLE_QUESTIONS,
        inputs=question_box,
        label="Câu hỏi mẫu",
    )

    submit_btn.click(
        fn=answer,
        inputs=[question_box, config_radio],
        outputs=[answer_box, rag_box],
    )
    question_box.submit(
        fn=answer,
        inputs=[question_box, config_radio],
        outputs=[answer_box, rag_box],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
