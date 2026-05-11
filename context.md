# Scout notes & session history — NLP traffic-QA

This note records the state of `/home/pc5070ti/workspace/SDA/nlp` as of 2026-05-11 after the phase 1–7 cleanup + RAG rebuild.

## Current state (2026-05-11)

Final pipeline and metrics are summarised in `reports/traffic/FINAL_REPORT.md`. Headline: config D reaches ROUGE-L 0.515, BLEU 0.350, BERTScore 0.692, LLM-Judge 0.692 on 145 labelled samples — a +0.188 ROUGE-L lift over the rule-base pipeline this session started with (0.327).

## Session narrative

### Starting point

The previous AI (deepseek) had attempted to "improve RAG" by stacking hand-written lookup tables on every stage:
- retrieval layer: phrase → article map (`VEHICLE_ARTICLE_MAP` in retrieval_v4), vehicle boost rank list.
- reranking: structured fact injection (`_VIOLATION_SEVERITY` regex → `(article, clause)`) that bypassed the cross-encoder entirely and returned a single precomputed fact card.
- generation: regex-extract direct answers from context (`_direct_answer_from_context`), numeric normalize (`_normalize_legal_answer`), hybrid refusal fallback.
- second-stage reranker: hand-weighted vehicle/sanction entity adjustment in `legal_units._entity_adjustment`.

This caused D to score ROUGE-L 0.327 / F1 0.227 / source_hit 0.757 / BLEU 0.060 — worse than the legitimate baseline (ROUGE-L 0.408) the project had earlier. The user asked for a cleanup and a scientific RAG rebuild.

### Phase 1 — cleanup

Removed in `src/`:
- `retrieval_v4.py`: `VEHICLE_ARTICLE_MAP`, `_detect_vehicle_articles`, `_vehicle_article_rank_list`, 3 compat stubs.
- `sanction_facts.py`: entire file deleted (FactBM25, `_VIOLATION_SEVERITY`, `retrieve_facts_structured`, `retrieve_fact_cards`).
- `evaluate.py`: `_direct_answer_from_context`, `_normalize_legal_answer`, `MONEY_RANGE_RE`, `POINT_DEDUCT_RE`, all branches for `RAG_DIRECT_SLOT_ANSWER`, `RAG_HYBRID_FALLBACK`, `RAG_FALLBACK_ON_SHORT_AMOUNT`, `NORMALIZE_LEGAL_NUMBERS`, `RAG_SANCTION_FACT_CARDS`, fact-card wiring in `retrieve_context`.
- `legal_units.py`: `VEHICLE_ALIASES`, `SANCTION_TERMS`, `vehicle_profile`, `sanction_profile`, `_entity_adjustment`, `RAG_LEGAL_UNIT_ENTITY_SCORING` branch.
- `query_utils.py`: `is_supported_traffic_question`, `TRAFFIC_SCOPE_TERMS`, `_doc_alias_matches`.

Moved 34 one-off / rule-scaffolding scripts into `scripts/archive/`.

### Phase 2 — enable CE reranker

Default `RAG_V4_USE_CE=True`, model `BAAI/bge-reranker-v2-m3` pretrained (already in HF cache). The finetuned sibling `models/bge-reranker-v2-m3-traffic-ft` tested worse (source_recall@5 0.87 → 0.57) because its 69-sample dev was fact-style, so it did not generalize. Kept pretrained.

### Phase 3 — rebuild KB with finetuned embedder

Repo already had `models/bge-m3-traffic-ft` (finetuned BGE-M3 on qa_train + penalty pairs) but the live vector DB was still using baseline `BAAI/bge-m3`. Rebuild: `EMBED_MODEL=models/bge-m3-traffic-ft python src/build_kb.py --force`. KB jumped from no-ft to ft without changing chunking.

### Phase 4 — ablation, turn off second-stage legal-unit BM25

Smoke on 30 samples showed `retrieve_legal_units` (BM25 over legal_units.jsonl restricted to same-article seeds) was adding noise once CE reranker + `pack_article_context` were in place. Turning `RAG_LEGAL_UNIT_RETRIEVAL=0` lifted smoke ROUGE-L 0.351 → 0.388 and false-refusal 0.10 → 0.033.

### Phase 5 — first full D eval

With phase 1–4 changes, D on 145 samples reached ROUGE-L 0.458 (+0.131 vs deepseek baseline). All quantitative metrics beat the rule-base version. Called this "D clean".

### Phase 6 — full A/B/C/D deliverable

A (0.146), B (0.348), C (0.389), D (0.458). D wins structural metrics; but C beat D on METEOR and BERTScore because D still refuses too often on alcohol/speed questions.

### Phase 7 — clause chunking (root-cause fix)

Diagnostic: **0 of 145 retrieved chunks had `clause_number`** in metadata. Reason: `src/build_kb.py::_article_documents` chunks at the article level (`legal_section="article"`, size 1600), even though `src/chunking.py::article_clause_chunks` already existed to chunk at clause level.

Fix: `build_kb.py::_article_documents` now delegates to `chunking.article_clause_chunks`, `LEGAL_CHUNKING_POLICY = "article_clause_v3"`. Rebuild: 1597 → 2853 chunks. Smoke retrieval:
- `clause_recall@5`: 0.00 → 0.81
- `article_recall@5`: 0.85 → 0.93
- `article_mrr@10`: 0.51 → 0.84
- `context_recall@5`: 0.525 → 0.975

Full D rerun on 145 samples, with BERTScore and LLM-Judge:

| metric | D phase-6 | D phase-7 (final) |
|---|---|---|
| R-L | 0.458 | **0.515** |
| R-2 | 0.368 | **0.447** |
| BLEU | 0.276 | **0.350** |
| METEOR | 0.397 | **0.491** |
| F1 | 0.359 | **0.419** |
| BERTScore | 0.654 | **0.692** |
| LLM-Judge | 0.531 | **0.692** |
| false_refusal | 0.021 | **0.007** |

D now wins every metric over A/B/C, including LLM-Judge. Order A < B < C < D as the project design intends.

## Final pipeline (current)

```
question
  → retrieval_v4
      ├─ dense FAISS (models/bge-m3-traffic-ft)
      ├─ BM25 sparse
      ├─ doc-alias list (nghị định 168 etc.)
      ├─ article-mention list (Điều N)
      └─ weighted Reciprocal Rank Fusion
  → Cross-Encoder rerank (BAAI/bge-reranker-v2-m3 pretrained, expanded query)
  → pack_article_context (same-article neighbour chunks)
  → evidence_card_from_text (structure-based parse, not rule phrase mapping)
  → LoRA Qwen3.5-9B (models/qwen3.5-9b-lora-traffic-v2)
```

No phrase → article lookup, no regex answer bypass, no CE bypass, no hand-weighted entity rules.

## Key artefacts

- KB: `vector_db_traffic/` (FAISS 2853 chunks + BM25 + `build_meta.json` with `chunking_policy=article_clause_v3`).
- Embedder: `models/bge-m3-traffic-ft`.
- Reranker: `BAAI/bge-reranker-v2-m3` (pretrained, HF cache).
- LoRA: `models/qwen3.5-9b-lora-traffic-v2` (unchanged from session start; already trained with `CONTEXT_KEEP_PROB=0.9`).
- Metrics: `reports/traffic/evaluation_results.json` (4 configs).
- Predictions: `reports/traffic/predictions_all_configs.json` (A/B/C), `reports/traffic/preds_d_clause_final.json` (D).
- Diagnostics: `reports/traffic/retrieval_diagnostics.json`.
- Full report: `reports/traffic/FINAL_REPORT.md`.

## Default knobs (current)

```
RAG_V4_USE_CE=True              # CE pretrained bge-reranker-v2-m3
RAG_LEGAL_UNIT_RETRIEVAL=False  # stage 2 off by default
RAG_EVIDENCE_CARD_RENDERING=True
GENERATION_NO_REPEAT_NGRAM=0    # was 8; 8 broke numeric format
GENERATION_REPETITION_PENALTY=1.08
RAG_TOP_K=2 / RAG_CONTEXT_MAX_CHUNKS=4
```

## What was kept from the original codebase

- `chunking.article_clause_chunks` — already correct, just wasn't wired in.
- `retrieval_v4.build_retriever` core logic — dense/BM25/alias/article + RRF is fine after dropping the vehicle rank list.
- `evidence_cards.evidence_card_from_text` — structure-based parse (article/clause/point/fine/deduct regex on retrieved text, not on the question).
- `context_packing.pack_article_context` — generic same-article neighbour expansion.
- `legal_units.retrieve_legal_units` — kept but default off; still usable as a second stage for clause chunking ablations.

## Potential next steps (not executed)

Ranked by expected lift and effort:

1. **Point-level chunking.** `point_recall` is still 0 because clause chunks don't expose `point_letter`. Adding point-level chunks under dense clauses (e.g. Điều 6/7 in nd_168 which list 12 points per clause) should help sanction-specific questions. Estimated +0.02–0.05 ROUGE-L.
2. **Retrain CE reranker on clause labels.** Use `eval_manual_labeled_v5` + `legal_sanction_facts.jsonl` to build `(query, chunk)` pairs with hard negatives from the same article but wrong clause. The existing `fact_reranker_v6_train/dev.jsonl` can serve as a starting point after converting labels. Estimated +0.02 R-L, +0.05 slot recall.
3. **RAG-SFT LoRA.** Retrain LoRA with `question + evidence_card → answer` format instead of raw context. `data/splits_filtered/qa_train.jsonl` already has context; wrap through evidence_cards renderer at dataloading time, 1 epoch. Estimated +0.05–0.10 LLM-Judge.
4. **Point-level chunking + CE retrain.** Compound gain. Biggest engineering effort.

## Historical / superseded notes

Earlier "research.md" and pre-session drafts describing sanction fact tables, entity-aware rerankers, RAG SFT as first-priority, CE training from token-overlap labels — all superseded. Clause-level chunking + pretrained CE + ft embedder + vanilla LoRA (already trained with context) is the winning combination, reached without any domain rules.
