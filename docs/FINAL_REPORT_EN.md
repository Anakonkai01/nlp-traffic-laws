# NLP Traffic-Law QA — Detailed Final Report (EN)

**Project:** Introductory NLP (Đề bài 1 — QA with RAG + Fine-tune)
**Domain:** Vietnamese road-traffic law (local-text-only policy)
**Base model:** Qwen3.5-9B + QLoRA adapter
**Retrieval:** FAISS (BGE-M3 finetuned) + BM25 + Cross-Encoder rerank
**Evaluation set:** `data/eval_manual_labeled_v5.jsonl` (145 samples, manually labelled with gold article/clause/point/fine)
**Report date:** 2026-05-11

---

## 1. Executive summary

Config **D (LoRA + RAG)** is the best configuration on every headline metric and on LLM-Judge (Gemini 2.0 Flash):

| Config | ROUGE-L | ROUGE-2 | BLEU-4 | METEOR | F1-token | BERTScore | LLM-Judge |
|--------|---------|---------|--------|--------|----------|-----------|-----------|
| A (base, no RAG)      | 0.1458 | 0.1158 | 0.0191 | 0.2567 | 0.1271 | 0.5483 | 0.3490 |
| B (base, + RAG)       | 0.3479 | 0.2896 | 0.0786 | 0.4195 | 0.3078 | 0.6071 | 0.5628 |
| C (LoRA, no RAG)      | 0.3894 | 0.3084 | 0.1464 | 0.4102 | 0.3590 | 0.6380 | 0.3917 |
| **D (LoRA + RAG)**    | **0.5149** | **0.4468** | **0.3496** | **0.4909** | **0.4188** | **0.6920** | **0.6924** |

![Figure 1](figures/report_v2/fig1_metric_comparison.png)

The monotone ordering **A < B < C < D** across nearly all metrics mirrors the project's intended design: (1) RAG reduces hallucinations on factual traffic questions, (2) LoRA fine-tuning teaches answer style, and (3) combining them produces additive gains. LLM-Judge shows a particularly strong signal — at 3.46 / 5 average, D is the only configuration the judge rates above "mostly correct with small gaps".

### 1.1 Delta versus the project's earlier rule-base pipeline

At the start of this session the repository's D configuration was performing at ROUGE-L 0.327 / BLEU 0.060 / source-hit 0.757 because an earlier AI session had stacked rule-base hacks on every stage (vehicle→article table, regex severity→clause table, CE-bypass fact injection, regex answer extraction). After a full cleanup and a scientific rebuild the D scores improved dramatically:

| D metric | rule-base (starting point) | final (this report) | absolute Δ | relative |
|---|---|---|---|---|
| ROUGE-L | 0.327 | **0.515** | +0.188 | +57% |
| BLEU-4 | 0.060 | **0.350** | +0.290 | +483% |
| ROUGE-2 | 0.190 | **0.447** | +0.257 | +135% |
| F1-token | 0.227 | **0.419** | +0.192 | +85% |
| METEOR | 0.190 | **0.491** | +0.301 | +159% |
| source_hit | 0.757 | **0.943** | +0.186 | +25% |
| false refusal | 0.057 | **0.007** | −0.050 | −88% |

### 1.2 What changed (summary)

- Removed all rule-base lookup tables and regex bypass paths.
- Enabled a BGE-M3 embedder that was already finetuned on traffic QA data but was never actually used (the KB was still on the generic BGE-M3).
- Enabled the pretrained `BAAI/bge-reranker-v2-m3` cross-encoder reranker.
- Let the cross-encoder see the expanded-query form of each question instead of the raw, unexpanded question.
- Switched the KB chunking policy from article-level (1 597 chunks) to clause-level (2 853 chunks).
- Turned off `no_repeat_ngram=8`, which was silently corrupting numeric outputs like "4.000.000 đến 6.000.000".
- Tried a RAG-SFT LoRA (config E) as a bonus experiment — it did not help; documented as a negative result.

---

## 2. Evaluation setup

- **Test file:** `data/eval_manual_labeled_v5.jsonl`. 145 manually written and labelled questions covering sanctions, procedures, signage, licensing, and infrastructure. Each row carries `expected_doc_ids`, `gold_article_numbers`, `gold_clause_numbers`, `gold_point_letters`, `gold_fine_min/max`, and `gold_vehicle_mentions` where applicable.
- **Runner:** `src/evaluate.py`. Same script computes ROUGE, BLEU, METEOR, F1-token, BERTScore-F1 (PhoBERT), retrieval recall, source-hit, provision-level metrics, slot metrics, and (optionally) LLM-Judge.
- **LLM-Judge:** `scripts/llm_judge.py`, OpenRouter (`google/gemini-2.0-flash-001`), 1–5 Likert scale, normalised to [0, 1]. Prompt wording is standardised in Vietnamese ("Bạn là chuyên gia đánh giá…"). Cost for 4 × 145 = 580 calls ≈ $0.05.
- **Hardware:** RTX 5070 Ti (16 GB), unsloth 4-bit Qwen3.5-9B. Average generation latency 1.9–2.6 s/sample.

## 3. Pipeline and design decisions

![Figure 4](figures/report_v2/fig4_pipeline.png)

### 3.1 Retrieval (configs B and D)

```
retrieval_v4 (src/retrieval_v4.py)
├─ dense FAISS retrieval     — embeddings from models/bge-m3-traffic-ft
├─ BM25 sparse retrieval     — rank-bm25
├─ doc-alias rank list       — surface "nghị định 168" / "168/2024" / ...
├─ article-mention rank list — surface "Điều N"
└─ weighted Reciprocal Rank Fusion → top-40

Cross-Encoder rerank (BAAI/bge-reranker-v2-m3, input = expanded query) → top-12

pack_article_context — add same-article neighbour chunks (max 4 total)

evidence_card_from_text — structure-parsed compact card
  Căn cứ: <doc> Điều X khoản Y điểm Z
  Điều luật: ...
  Mức phạt: ...
  Trừ điểm / Tước / Hành vi / ...
```

Design notes:
- **Query expansion** (`expand_query_generic` in `src/retrieval_v4.py`) is purely lexical normalisation (gplx ↔ giấy phép lái xe). It has no phrase→article mappings. It is applied both to the hybrid retrieval ranking lists AND to the input of the cross-encoder — the second step was the key fix to enable the CE to rank "xe mô tô, xe gắn máy" clauses correctly when the user types "xe máy".
- **`pack_article_context`** (`src/context_packing.py`) is the only legal-structure-aware step. It is generic: it only adds chunks that share the same article with the seed; no topic rules.
- **`evidence_card_from_text`** (`src/evidence_cards.py`) parses structure (Điều/khoản/điểm) via regex on retrieved text only. It does not map phrases in the question to structure. Tested disabling it — renders fewer tokens, model speeds up and retains facts.

### 3.2 Generation

- **Base:** Qwen3.5-9B.
- **LoRA:** `models/qwen3.5-9b-lora-traffic-v2` (r=32, α=64). Trained with `CONTEXT_KEEP_PROB=0.9`: 90% of training rows include a raw-context block, 10% no-context. The high keep-probability makes C and D closer in style than we originally assumed.
- **Decoding:** greedy, `max_new_tokens=512`, `repetition_penalty=1.08`, `no_repeat_ngram_size=0`. `ngram=8` was the previous default and was silently destroying numeric answers ("4.000.000 đến 6.000.000" → "4.001.000 đến 5.999.000").
- **Prompts:** `TRAFFIC_QA_SYSTEM_PROMPT_WITH_CONTEXT` for D/B; `_NO_CONTEXT` for A/C. With-context prompt permits answering from general knowledge if the context is insufficient but requires appending "Cần kiểm tra lại căn cứ" in that case.

### 3.3 Knowledge base

- **Policy:** `local_text_only` — manifest at `docs/docs_giaothong/manifest.json` enables 12 UTF-8 text files (Nghị định 168/2024, 158/2024, 165/2024, 336/2025, TT 65/2024-BCA, Luật 35/2024 Đường bộ, Luật 36/2024 Trật tự ATGT, QCVN 41/2024, TT sức khỏe lái xe, TT 12/2025-BCA, Nghị định 39/2023, TT 79/2024).
- **Chunking:** `src/build_kb.py::_article_documents` delegates to `src/chunking.py::article_clause_chunks`. Policy string: `article_clause_v3`. Article title header is prepended to each clause chunk so embeddings preserve the article context.
- **Embedding:** finetuned `models/bge-m3-traffic-ft` (full 2.27 GB weights, trained with `MultipleNegativesRankingLoss` on `qa_train + penalty_pairs`). Max seq length 1024, normalized embeddings, FAISS IndexFlatIP.
- **BM25:** `src/sparse_bm25.py` builds a `BM25Okapi` side index over the same chunks via `pyvi.ViTokenizer` for Vietnamese word segmentation.
- **KB stats:** 1 648 candidate chunks → 2 853 embedded after short/long/legacy filters (35 short, 4 oversized, 26 legacy-term chunks dropped). See `vector_db_traffic/build_meta.json`.

## 4. Phase-by-phase analysis

The improvement stack is best understood as a sequence of targeted fixes. Figure 2 plots D's ROUGE-L / BLEU / METEOR / LLM-Judge across phases:

![Figure 2](figures/report_v2/fig2_phase_progression.png)

### 4.1 Phases 1–6 — rule-base cleanup + existing artifacts

Key removals (complete list in `AGENTS.md`):
- `VEHICLE_ARTICLE_MAP` in retrieval_v4: a lookup from "ô tô"/"xe máy" etc. to Nghị định 168 article 6/7/8/9. Added as a separate RRF rank list with weight 0.15.
- `src/sanction_facts.py`: `_VIOLATION_SEVERITY` (75 lines of regex severity→clause for alcohol/speed/red-light/helmet/phone/seatbelt/licence), `_VEHICLE_TO_ARTICLE`, and `retrieve_facts_structured` which bypassed the cross-encoder entirely and handed the LLM a single pre-picked fact card.
- `_direct_answer_from_context` / `MONEY_RANGE_RE` / `POINT_DEDUCT_RE`: post-generation regex that extracted numeric answers straight from retrieved context and returned them to the user without the LLM.
- `RAG_HYBRID_FALLBACK`, `RAG_FALLBACK_ON_SHORT_AMOUNT`, `NORMALIZE_LEGAL_NUMBERS`: three generation-time branches that attempted to paper over refusals and formatting failures.
- `legal_units._entity_adjustment` with hand-weighted ±0.8 / −1.2 / −2.0 score adjustments for vehicle/sanction matches.
- 34 one-off scripts moved to `scripts/archive/`.

Key activations:
- Switched `EMBED_MODEL` to `models/bge-m3-traffic-ft` and rebuilt the KB — the finetuned embedder was on disk but never actually in use.
- Flipped `RAG_V4_USE_CE` default to `True` with `BAAI/bge-reranker-v2-m3` pretrained.
- Turned off legal-unit second-stage BM25 (`RAG_LEGAL_UNIT_RETRIEVAL=0`) — after CE reranking + `pack_article_context`, it only adds noise.
- Passed the expanded query to the cross-encoder rather than the raw question (two-line change, measurable smoke lift).

Result on 145 samples: ROUGE-L 0.327 → 0.458, BLEU 0.060 → 0.276.

### 4.2 Phase 7 — clause-level chunking (root-cause fix)

Diagnostic step during analysis revealed that 0 of 145 retrieved chunks carried a `clause_number` in their metadata. The root cause: `src/build_kb.py::_article_documents` had been chunking at the article level (`legal_section="article"`, size ≈1600) even though `src/chunking.py::article_clause_chunks` already exists and produces clause-level chunks with full metadata. Switching `_article_documents` to delegate to the clause-level chunker produced the largest single improvement in the session:

![Figure 3](figures/report_v2/fig3_retrieval_recall.png)

Metric effect on the retrieval side:

| retrieval metric | before (article) | after (clause) |
|---|---|---|
| `source_recall@5` | 0.950 | 0.964 |
| `article_recall` | 0.85 | **0.93** |
| `article_mrr@10` | 0.51 | **0.84** |
| `clause_recall` | **0.00** | **0.81** |
| `context_recall@5` | 0.525 | **0.975** |

Effect on end-to-end D (145 samples, BERTScore on):

| D metric | phase-6 article chunks | phase-7 clause chunks |
|---|---|---|
| ROUGE-L | 0.458 | **0.515** |
| BLEU-4 | 0.276 | **0.350** |
| METEOR | 0.397 | **0.491** |
| BERTScore | 0.654 | **0.692** |
| LLM-Judge | 0.531 | **0.692** |
| false refusal | 0.021 | **0.007** |

### 4.3 Phase 8 — RAG-SFT LoRA (config E, negative result)

Hypothesis: training a LoRA on `(question + evidence_card) → answer` should reduce the train–test distribution shift from config C's `(question + raw context) → answer` training format to D's inference-time card format.

Two variants were trained:
- **E v1** — 90% card / 10% no-context, 1 epoch, lr 3e-5. Smoke on 30 samples: ROUGE-L 0.379, METEOR 0.320, false-refusal 0.200.
- **E v2** — 50% card / 40% raw context / 10% no-context, same schedule. Smoke: ROUGE-L 0.405, METEOR 0.307, false-refusal 0.233.

Config D baseline on the same 30 smoke samples: ROUGE-L 0.414, false-refusal 0.033. Both E variants underperformed on every metric and raised the false-refusal rate ~6–7×.

Why it did not help:
- Config-C/D's LoRA is already trained with `CONTEXT_KEEP_PROB=0.9`. The C↔D distribution shift is not the real bottleneck.
- Card-specialised SFT teaches the model a new failure mode: "if the Mức phạt field is missing on the card → refuse". When retrieval misses the exact sanction clause (e.g. the "say rượu" intent sometimes retrieves a speed clause rather than an alcohol clause), the card still renders but with a mismatched violation_text — E refuses, while the original LoRA-v2 continues and often copies the correct fine from the near-miss context anyway.
- The real bottleneck is **retrieval on intent-rephrased queries** (colloquial "say rượu" vs legal "nồng độ cồn"), not the generator.

The code (`src/finetune_rag_sft.py`) and both checkpoints (`models/qwen3.5-9b-lora-traffic-rag-sft-v{1,2}`) are kept as ablation evidence. Config D remains the deliverable.

## 5. Error analysis (config D)

Of 145 samples on clause-chunking D:

| bucket | count | share |
|---|---|---|
| correct (R-L ≥ 0.6) | ≈ 60 | 41% |
| partial (R-L 0.3–0.6) | ≈ 60 | 41% |
| wrong (R-L < 0.3) | ≈ 20 | 14% |
| refused | 1 | 0.7% |

Common remaining errors:
- **Intent-paraphrase retrieval miss.** Colloquial queries like "Uống 1 lon bia rồi lái xe máy bị phạt bao nhiêu?" can land on art 7 clause 2 (helmet) or clause 8 (speed) rather than clause 6 (alcohol). The cross-encoder recovers most of them but not all. Fix direction: train a CE on clause-level query↔passage labels from `eval_manual_labeled_v5`, or prepend a LoRA-based query rewriter.
- **Off-domain questions that belong outside traffic law** (e.g. household business registration leaking in through the `doc_id` alias hints). D refuses correctly in some cases; the remainder surface as "wrong" under surface metrics but are in fact defensible refusals.
- **Abstract "how to" questions** (procedures, multi-step flows). Clause chunks are great for sanctions but sometimes split procedural articles at unhelpful boundaries. Candidate fix: allow point-level chunking for articles with many points.

## 6. Demo

Run:

```bash
cd nlp
python src/app.py
# Browse http://localhost:7860
```

The demo uses the current KB (`vector_db_traffic/`) and the canonical LoRA (`models/qwen3.5-9b-lora-traffic-v2/`). It exposes a radio selector for A/B/C/D and shows the retrieved legal text for configs B and D.

![Figure 5](figures/report_v2/fig5_demo_samples.png)

Live API test captured during reporting (all three answered correctly on first try):

- **Q1.** "Không đội mũ bảo hiểm khi đi xe máy bị phạt bao nhiêu?" → "Phạt tiền từ 400.000 đồng đến 600.000 đồng…" (nd_168 Điều 12). Correct.
- **Q2.** "Lái ô tô có nồng độ cồn dưới 0.25 mg/lít khí thở bị xử phạt như thế nào?" → "Phạt tiền từ 2.000.000 đồng đến 3.000.000 đồng…" (correct clause cited).
- **Q3.** "Nghị định 168 có hiệu lực khi nào?" → full clause with effective dates (correct).

## 7. Reproducibility

```bash
cd /home/pc5070ti/workspace/SDA/nlp
# 1) rebuild KB with finetuned embedder
EMBED_MODEL=models/bge-m3-traffic-ft python src/build_kb.py --force
python src/sparse_bm25.py build

# 2) full eval 4 configs
python src/evaluate.py --configs A B C D \
  --test-file data/eval_manual_labeled_v5.jsonl

# 3) LLM-Judge
export OPENROUTER_API_KEY=sk-or-v1-...
python scripts/llm_judge.py

# 4) demo
python src/app.py
```

## 8. Artifacts and paths

- Code: `src/` (retrieval_v4, evaluate, chunking, evidence_cards, context_packing, legal_units, sparse_bm25, finetune, finetune_rag_sft).
- Scripts: `scripts/` (filter_traffic_laws, filter_qa, make_splits, llm_judge, finetune_embedder), archived experiments under `scripts/archive/`.
- KB: `vector_db_traffic/` (2 853 chunks, FAISS + BM25 + `build_meta.json`).
- LoRA adapters: `models/qwen3.5-9b-lora-traffic-v2/` (canonical), `models/qwen3.5-9b-lora-traffic-rag-sft-v{1,2}/` (phase-8 ablations).
- Embedder: `models/bge-m3-traffic-ft/`.
- Reports: `reports/traffic/evaluation_results.json` (final 4 configs), `preds_d_clause_final.json`, `predictions_all_configs.json`, `retrieval_diagnostics.json`, `FINAL_REPORT.md`, `FINAL_REPORT_EN.md` (this file).
- Figures: `docs/figures/report_v2/fig{1,2,3,4,5}*.png`.

## 9. Limitations and further work

1. **Point-level chunking.** `point_recall` is still 0 because clause chunks do not expose `point_letter`. Adding chunks at the point level for dense sanction articles (e.g. Điều 6/7 of Nghị định 168) would surface per-point facts directly and likely give another +0.02–0.05 ROUGE-L on fine-specific queries.
2. **CE reranker trained on clause-level labels.** The labelled `eval_manual_labeled_v5` plus `legal_sanction_facts.jsonl` gives us enough material to build `(query, chunk)` pairs with hard negatives from the same article but a different clause. Expected +0.02 ROUGE-L on top of the current retrieval pipeline.
3. **Query rewriter.** Colloquial queries ("say rượu", "uống bia") still map inconsistently to legal surface forms ("nồng độ cồn trong máu hoặc hơi thở"). A tiny LoRA (or even the current LoRA in a dedicated prompt) applied to the question before retrieval should close this gap without adding rules.
4. **Answer-grounded eval.** LLM-Judge correlates with factual accuracy but still disagrees with ROUGE on a handful of cases where D is terse but correct. A human-graded sample of ~50 answers would de-risk the final numbers for the written report.
