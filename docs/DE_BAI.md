# Đề bài — Nhập môn Xử lý ngôn ngữ tự nhiên

Chọn 1 trong 2 đề tài:

## Đề tài 1: Hỏi đáp với RAG + Fine-tune (đang làm)

### 1. Dữ liệu
- Domain tự chọn → đang xài luật giao thông
- Thu thập tài liệu/đoạn văn cho knowledge base
- Tạo ≥300 cặp QA fine-tune, ≥50 cặp test thủ công

### 2. Fine-tuning
- LLM 1B-7B tham số, LoRA/QLoRA
- Chạy được trên Colab Free
- Gợi ý: tự chọn → đang xài Qwen3.5-9B

### 3. Pipeline RAG
- Chunking + embedding + vector store (Chroma/FAISS)
- Retriever top-k + prompt template rõ ràng

### 4. Thực nghiệm — 4 cấu hình

| | No RAG | RAG |
|---|---|---|
| Base LLM | A | B |
| Fine-tuned | C | D |

### 5. Đánh giá
- Định lượng: BLEU, ROUGE-L, BERTScore
- Retrieval: Recall@5
- Human eval: 50 câu

### 6. Demo
Có GUI (khuyến khích)

### Sản phẩm nộp
- GitHub (README)
- Báo cáo 15-20 trang
- Slide + video demo 3-5 phút
- Dataset + checkpoint (HuggingFace Hub + Drive)

---

## Đề tài 2: Tóm tắt văn bản tiếng Việt (đang ko làm)
- Domain giàu số liệu (tài chính, thể thao, y tế...)
- ≥500 cặp train, ≥100 cặp test thủ công
- Giữ nguyên 100% số liệu, độ dài 20-25% bản gốc
- Fine-tune ≥2 mô hình (ViT5, BARTpho, mT5...)
- Đánh giá: ROUGE-1/2/L, BLEU, METEOR, BERTScore, Number Accuracy, Length Compliance
