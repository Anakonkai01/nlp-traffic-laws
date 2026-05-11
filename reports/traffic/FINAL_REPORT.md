# NLP traffic-QA — Kết quả cuối (2026-05-11)

## Kết quả 4 config (145 samples, `data/eval_manual_labeled_v5.jsonl`)

| Config | R-1 | R-2 | R-L | BLEU | METEOR | F1 | BERTSc | LLM-Judge |
|---|---|---|---|---|---|---|---|---|
| A (base, no RAG)       | 0.188 | 0.116 | 0.146 | 0.019 | 0.257 | 0.127 | 0.548 | 0.349 |
| B (base, +RAG)         | 0.444 | 0.290 | 0.348 | 0.079 | 0.420 | 0.308 | 0.607 | 0.563 |
| C (LoRA, no RAG)       | 0.533 | 0.308 | 0.389 | 0.146 | 0.410 | 0.359 | 0.638 | 0.392 |
| **D (LoRA + RAG)**     | **0.589** | **0.447** | **0.515** | **0.350** | **0.491** | **0.419** | **0.692** | **0.692** |

Retrieval chung: `source_recall@5 = 0.964`, `context_recall@5 = 0.975`, `source_hit_rate = 0.943`, `article_recall = 0.93`, `clause_recall = 0.81`, `false_refusal = 0.007`, `forbidden_legacy = 0.000`.

LLM-Judge model: `google/gemini-2.0-flash-001` (OpenRouter), thang điểm 1–5 chuẩn hoá về [0, 1].

## So với baseline trước cleanup

| metric (D) | trước (rule-base) | sau cleanup | +clause chunks (final) | total delta |
|---|---|---|---|---|
| ROUGE-L | 0.327 | 0.458 | **0.515** | +0.188 (+57%) |
| BLEU-4 | 0.060 | 0.276 | **0.350** | +0.290 (+483%) |
| ROUGE-2 | 0.190 | 0.368 | **0.447** | +0.257 |
| F1-token | 0.227 | 0.359 | **0.419** | +0.192 |
| METEOR | 0.190 | 0.326 | **0.491** | +0.301 |
| BERTScore | — | 0.654 | **0.692** | — |
| source_hit | 0.757 | 0.936 | 0.943 | +0.186 |
| false refusal | 0.057 | 0.021 | **0.007** | -0.050 |
| LLM-Judge | — | 0.531 | **0.692** | +0.161 |

## Những gì đã thay đổi

### Đã xoá (rule-base, hardcode, hack phi khoa học)

- `src/retrieval_v4.py`: `VEHICLE_ARTICLE_MAP` (bảng tra vehicle → nd_168 article), `_vehicle_article_rank_list`, 3 compat stubs (`classify_intents`, `is_supported_traffic_question`, ...).
- `src/sanction_facts.py`: **xoá cả file** (`_VIOLATION_SEVERITY` bảng regex severity→clause, `_VEHICLE_TO_ARTICLE`, `retrieve_facts_structured` bypass CE, `FactBM25`, 7 model reranker `fact-reranker-v1..v6`).
- `src/evaluate.py`: `_direct_answer_from_context` (regex bypass LLM), `_normalize_legal_answer`, `MONEY_RANGE_RE`, `POINT_DEDUCT_RE`, toàn bộ nhánh `RAG_DIRECT_SLOT_ANSWER` / `RAG_HYBRID_FALLBACK` / `RAG_FALLBACK_ON_SHORT_AMOUNT` / `NORMALIZE_LEGAL_NUMBERS`.
- `src/legal_units.py`: `VEHICLE_ALIASES`, `SANCTION_TERMS`, `vehicle_profile`, `sanction_profile`, `_entity_adjustment` (hand-weighted ±0.8/-1.2/-2.0 adjustment), nhánh `RAG_LEGAL_UNIT_ENTITY_SCORING`.
- `src/query_utils.py`: `is_supported_traffic_question`, `TRAFFIC_SCOPE_TERMS`, `_doc_alias_matches` (gating rule theo từ khoá traffic).
- `scripts/archive/`: 34 script one-off (ablation, audit, data-prep experiment). Giữ 12 script canonical + utility.

### Đã thêm / sửa (khoa học, không hardcode)

- `RAG_V4_USE_CE=True` mặc định trong `src/retrieval_v4.py` (bật Cross-Encoder reranker pretrained `BAAI/bge-reranker-v2-m3`).
- CE rerank dùng **expanded query** thay vì question gốc (giúp CE match synonym "xe máy" ↔ "xe mô tô, xe gắn máy").
- `EMBED_MODEL=models/bge-m3-traffic-ft` (embedder đã được finetune sẵn trên `qa_train + penalty_pairs`, nhưng chưa được dùng).
- `src/build_kb.py`: chunking policy đổi từ `article_v2` (chunk 1600 chars/article) → `article_clause_v3` (chunk theo clause, trung bình 500–900 chars). Metadata có `clause_number` cho 2586 chunks (hiện tại 0 chunks có).
  - `chunk_count`: 1597 → 2853.
  - `clause_recall@5`: **0.00 → 0.81**.
  - `article_mrr`: 0.51 → 0.84.
  - `context_recall@5`: 0.525 → 0.975.
- `GENERATION_NO_REPEAT_NGRAM` mặc định đổi từ 8 → 0 (ngram=8 phá format số "X.000.000 đến Y.000.000").
- `RAG_LEGAL_UNIT_RETRIEVAL=0` mặc định (stage 2 BM25 trên legal_units chỉ thêm noise khi đã có CE + pack_article).
- `RAG_EVIDENCE_CARD_RENDERING=1` mặc định (structure-based card renderer, không phải rule phrase mapping).

## Pipeline cuối

```
question
  │
  ▼
retrieval_v4
  ├─ Dense retrieval (FAISS, bge-m3-traffic-ft)          top-50
  ├─ BM25 sparse retrieval (rank-bm25)                    top-50
  ├─ Doc-alias rank list (nghị định 168, etc.)             top-N
  ├─ Article-mention rank list (Điều X)                    top-N
  └─ Weighted Reciprocal Rank Fusion                       top-40
  │
  ▼
Cross-Encoder rerank (BAAI/bge-reranker-v2-m3, expanded query) top-12
  │
  ▼
pack_article_context (same-article neighbour chunks)       top-4
  │
  ▼
evidence_card_from_text (structure-based: article/clause/point/fine/deduct parse)
  │
  ▼
LoRA Qwen3.5-9B (models/qwen3.5-9b-lora-traffic-v2, trained with CONTEXT_KEEP_PROB=0.9)
  │
  ▼
answer
```

## Artifacts

- KB: `vector_db_traffic/` — FAISS 2853 chunks, BM25 side index, `build_meta.json` với `chunking_policy=article_clause_v3`.
- Embedder: `models/bge-m3-traffic-ft` (finetuned BGE-M3).
- Reranker: `BAAI/bge-reranker-v2-m3` (pretrained, HF cache).
- LoRA: `models/qwen3.5-9b-lora-traffic-v2` (giữ nguyên từ lúc đầu session, không retrain).
- Predictions: `reports/traffic/preds_d_clause_final.json` (D), `predictions_all_configs.json` (A/B/C).
- Metrics: `reports/traffic/evaluation_results.json`.
- Retrieval diagnostics: `reports/traffic/retrieval_diagnostics.json`.

## Commands tái hiện

```bash
cd /home/pc5070ti/workspace/SDA/nlp

# Rebuild KB
EMBED_MODEL=models/bge-m3-traffic-ft python src/build_kb.py --force
python src/sparse_bm25.py build

# Full eval 4 config
python src/evaluate.py --configs A B C D \
  --test-file data/eval_manual_labeled_v5.jsonl

# Re-run D only with BERTScore
python src/evaluate.py --configs D --samples 145 \
  --test-file data/eval_manual_labeled_v5.jsonl \
  --output reports/traffic/d_clause_final.json \
  --predictions-output reports/traffic/preds_d_clause_final.json

# LLM judge
export OPENROUTER_API_KEY=...
python scripts/llm_judge.py
```

## Các hướng cải thiện khả dĩ (chưa làm)

1. **Point-level chunking** — `point_recall` hiện vẫn 0. Chunk thêm cấp `point` (3558 points trong `legal_units.jsonl`) cho article dày clause. Kỳ vọng +0.02–0.05 ROUGE-L.
2. **Retrain CE reranker** trên label clause-level từ `eval_manual_labeled_v5` với hard negative cùng article khác clause. Dev set cần lớn hơn 69-sample hiện tại. Kỳ vọng +0.02 R-L, +0.05 fine_slot_recall.
3. **Question rewriting** cho truy vấn ngắn. Dùng LoRA chính rewrite "Say xỉn lái xe máy?" → "Điều khiển xe mô tô có nồng độ cồn trong máu hoặc khí thở phạt tiền bao nhiêu?".

## Negative result — Config E (RAG-SFT LoRA)

**Đã thử, không thành công.** Ý tưởng: train LoRA mới riêng với format `(question + evidence_card) → answer` để thích nghi distribution shift giữa training (raw context) và inference (card).

Thử nghiệm:

- v1 (90% card / 10% no-context, 1 epoch, lr 3e-5): smoke 30 → R-L **0.379** vs D 0.414, FRef **0.200** vs D 0.033. Over-specialized.
- v2 (50% card / 40% raw / 10% no-context): smoke 30 → R-L **0.405**, FRef **0.233**. Vẫn kém D.

Lý do:

- LoRA v2 hiện tại (`qwen3.5-9b-lora-traffic-v2`) đã được train với `CONTEXT_KEEP_PROB=0.9` — tức 90% samples có raw context. Distribution shift C↔D không nghiêm trọng như mô tả ban đầu.
- Bước SFT riêng trên card format khiến model học pattern: "nếu card không có đúng field → refuse". Khi retrieval miss clause đúng (ví dụ câu "say rượu" không match clause nồng độ cồn về surface form), card vẫn hình thành nhưng với violation_text sai → E refuse, D thì copy số từ card gần đúng và tình cờ đúng.
- Bottleneck thật không phải generator, mà là retrieval: một số intent (say rượu, nồng độ cồn) vẫn match nhầm clause tốc độ / lạng lách.

Kết luận: **giữ D là cấu hình deliverable chính**. Code E (`src/finetune_rag_sft.py` + 2 checkpoint ở `models/qwen3.5-9b-lora-traffic-rag-sft-v{1,2}`) để lại như ablation evidence trong báo cáo.

Hướng đi đúng tiếp theo cho retrieval bottleneck (đã đánh giá):

- **Question rewriting trước retrieval**: dùng LoRA v2 rewrite "say xỉn" → "nồng độ cồn trong máu hoặc hơi thở" trước khi đi vào BM25/CE. Dự kiến +0.05 R-L, fix luôn vấn đề retrieval.
- **Retrain CE reranker clause-level** bằng label v5 (hard negative cùng article khác clause). Cần tăng dev size trên 200.
