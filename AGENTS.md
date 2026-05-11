# AGENTS.md

## Scope
- Repo root is `nlp/`; run commands from `/home/pc5070ti/workspace/SDA/nlp` unless command explicitly includes `nlp/` prefix.
- Python scripts import sibling modules from `src/`, so prefer `cd nlp && python src/...` or set `PYTHONPATH=src` for unusual launch paths.

## Data Policy
- Production path is `local_text_only`: KB, QA generation, fine-tune, eval, and demo must read only enabled UTF-8 `.txt` files listed in `docs/docs_giaothong/manifest.json`.
- Do not add VLSP corpus, PDF fallback, web crawling, or legacy traffic-law sources to production flow.
- New legal source: place UTF-8 text under `docs/docs_giaothong/text/`, add manifest entry with `enabled: true` and source policy unchanged.
- `load_vectorstore()` rejects stale KBs missing `vector_db_traffic/build_meta.json` or mismatched `source_policy`; rebuild with `python src/build_kb.py --force`.

## Main Commands
- Source audit: `python scripts/filter_traffic_laws.py` validates manifest policy, enabled text docs, and corpus size.
- KB rebuild: `python src/build_kb.py --force`; rebuild BM25 side index after KB changes with `python src/sparse_bm25.py build`.
- QA generation default from code writes to `data/splits_filtered/qa_train.jsonl` (`QA_DATA_PATH`), not `data/qa_pairs_traffic.jsonl`; use `--output ...` when regenerating monolithic QA files.
- `scripts/filter_qa.py` still reads and overwrites `data/qa_pairs_traffic.jsonl`; do not assume it filters `data/splits_filtered/qa_train.jsonl`.
- Split creation is separate: `python scripts/make_splits.py --qa data/qa_pairs_traffic_filtered.jsonl --outdir data/splits_filtered --seed 42`.
- Fine-tune reads frozen split files `data/splits_filtered/qa_train.jsonl` and `data/splits_filtered/qa_dev.jsonl`; it does not reshuffle monolithic QA data.
- Full verified order from code is source audit -> KB rebuild -> optional QA generation/output -> QA filtering/splitting -> `python src/finetune.py` -> eval.
- Sequential pipeline wrapper uses Conda env `ai` by default: `REBUILD_KB=1 REGENERATE_QA=1 bash auto_pipeline.sh`; override with `CONDA_ENV=...`.
- Demo: `python src/app.py`; it loads RAG at startup and lazy-swaps base/LoRA models to fit 16GB VRAM.
- QA smoke: `python src/generate_qa.py --force --profile smoke`.
- Focused QA generation: `python src/generate_qa.py --force --phases penalties procedures --phase-size 20 --no-negatives --timeout 180 --num-predict 160`.
- Focused eval smoke: `python src/evaluate.py --configs D --samples 10 --fast-metrics` or `python src/evaluate.py --retrieval-only --samples 10`.
- MC eval smoke: `python src/evaluate_mc.py --configs D --samples 10`.

## Runtime Requirements
- Dependencies are only listed in `requirements.txt`; no pytest/ruff/mypy config or CI workflow is present.
- QA generation defaults to Ollama at `http://localhost:11434/api/chat` with model `qwen3.5:9b`; set `OPENROUTER_API_KEY` or `--backend openrouter` for OpenRouter.
- `src/finetune.py`, `src/evaluate.py`, `src/evaluate_mc.py`, and `src/app.py` import `unsloth` before model loading; keep that ordering if editing imports.
- `src/build_kb.py` forces `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` before loading embeddings; required embedding model must already be cached unless `EMBED_MODEL` points local.
- LLM judge in eval is opt-in: `--judge` needs `OPENROUTER_API_KEY` by default or `--judge-backend minimax` with `MINIMAX_API_KEY`.

## Artifacts
- Generated/large artifacts are intentionally ignored: `models/`, `vector_db_traffic/`, `wandb/`, `logs/`, PDFs under `docs/docs_giaothong/`, and large QA JSONL files.
- Embedder in use: `models/bge-m3-traffic-ft` (full weights, finetuned from BAAI/bge-m3 on qa_train + penalty_pairs). Pass as `EMBED_MODEL=models/bge-m3-traffic-ft` to `src/build_kb.py` / `src/corpus.py`.
- Reranker in use: `BAAI/bge-reranker-v2-m3` pretrained (HuggingFace cache). Override via `RETRIEVAL_CE_MODEL`.
- LoRA in use: `models/qwen3.5-9b-lora-traffic-v2` (r=32, α=64, trained with `CONTEXT_KEEP_PROB=0.9` so it already handles RAG context; no retrain needed for phase 1–7).
- Verified data counts: `data/qa_pairs_traffic.jsonl` 2885 rows, `data/qa_pairs_traffic_filtered.jsonl` 2202 rows, `data/splits_filtered/qa_train.jsonl` 1762 rows, dev 220, test 220, `data/eval_manual.jsonl` 145 rows, `data/eval_manual_labeled_v5.jsonl` 145 rows (with gold article/clause/point/fine fields), `data/eval_mc_manual.jsonl` 201 rows, `data/legal_units.jsonl` 6765 units (621 article, 2586 clause, 3558 point).
- Default paths: QA splits under `data/splits_filtered/`, manual eval `data/eval_manual_labeled_v5.jsonl`, MC eval `data/eval_mc_manual.jsonl`, LoRA `models/qwen3.5-9b-lora-traffic-v2/`, reports `reports/traffic/`.
- `QA_SPLIT_DIR` can redirect train/dev/test split paths used by `src/config.py`.

## Removed (do not re-introduce)
- Phrase → article lookup tables (`VEHICLE_ARTICLE_MAP`, `_VEHICLE_TO_ARTICLE`).
- Keyword regex → clause lookup (`_VIOLATION_SEVERITY`).
- `src/sanction_facts.py` and `scripts/build_sanction_facts*.py` — entire sanction fact pipeline was overfit rule-base. Use clause-level chunking + CE reranker instead.
- `retrieve_facts_structured` CE-bypass structured inject path.
- Regex-based direct slot answer (`_direct_answer_from_context`, `MONEY_RANGE_RE`, `POINT_DEDUCT_RE`) and LLM output normalize (`_normalize_legal_answer`).
- Hybrid fallback branches (`RAG_HYBRID_FALLBACK`, `RAG_FALLBACK_ON_SHORT_AMOUNT`, `NORMALIZE_LEGAL_NUMBERS`, `RAG_DIRECT_SLOT_ANSWER`).
- Entity adjustment in legal_units (`vehicle_profile`, `sanction_profile`, `_entity_adjustment`, `RAG_LEGAL_UNIT_ENTITY_SCORING`).
- Traffic scope gating (`TRAFFIC_SCOPE_TERMS`, `is_supported_traffic_question`).
- 34 one-off experiment scripts moved to `scripts/archive/`.

## Retrieval Knobs
- Default retrieval switch is `RETRIEVAL_VERSION=v4` in `src/retrieval.py`; set `RETRIEVAL_VERSION=v3` only for ablations/rollback checks.
- v4 combines FAISS dense (bge-m3-traffic-ft) + BM25 + doc alias + article mention with weighted RRF; main knobs are `RAG_V4_DENSE_K`, `RAG_V4_SPARSE_K`, `RAG_V4_FUSE_K`, `RAG_V4_W_DENSE`, `RAG_V4_W_SPARSE`, `RAG_V4_USE_CE`.
- CE reranker is ON by default (`RAG_V4_USE_CE=True`), model `BAAI/bge-reranker-v2-m3` pretrained, CE rerank uses the expanded query (lexical normalization only, no phrase→article mapping).
- Finetuned reranker at `models/bge-reranker-v2-m3-traffic-ft` underperforms pretrained on held-out eval; keep pretrained unless retrained on clause-level labels.
- BM25 side index is separate from FAISS: rebuild after KB changes with `python src/sparse_bm25.py build`. Inspect with `python src/sparse_bm25.py stats|search`.
- Chunking policy: `article_clause_v3` — clause-level chunks preserving article title header; defined in `src/chunking.py::article_clause_chunks` and invoked via `src/build_kb.py::_article_documents`. KB chunks: 2853. Rebuild with `EMBED_MODEL=models/bge-m3-traffic-ft python src/build_kb.py --force`.
- Second-stage legal-unit BM25 (`RAG_LEGAL_UNIT_RETRIEVAL`) is OFF by default — with clause chunks + CE it only adds noise. Re-enable per ablation only.
- Evidence-card rendering (`RAG_EVIDENCE_CARD_RENDERING=1`) is ON by default — structure-based card renderer in `src/evidence_cards.py`, not rule-based phrase mapping.
- Generation knobs: `GENERATION_REPETITION_PENALTY=1.08`, `GENERATION_NO_REPEAT_NGRAM=0` (ngram=8 breaks numeric formatting "X.000.000 đến Y.000.000").

## Result Baselines
- Current `reports/traffic/evaluation_results.json` (145 samples, `data/eval_manual_labeled_v5.jsonl`, clause-level chunking `article_clause_v3`):
  - A (base, no RAG)    : R-L 0.146, F1 0.127, BERTSc 0.548, Judge 0.349
  - B (base, +RAG)      : R-L 0.348, F1 0.308, BERTSc 0.607, Judge 0.563
  - C (LoRA, no RAG)    : R-L 0.389, F1 0.359, BERTSc 0.638, Judge 0.392
  - D (LoRA + RAG)      : R-L 0.515, F1 0.419, BERTSc 0.692, Judge 0.692 — best
- Retrieval: source_recall@5 0.9643, context_recall@5 0.9750, article_recall 0.93, clause_recall 0.81, false_refusal 0.007, forbidden_legacy 0.0.
- LLM-Judge backend: OpenRouter `google/gemini-2.0-flash-001`, scale 1–5 normalised to [0, 1]. Script: `scripts/llm_judge.py`.
- See `reports/traffic/FINAL_REPORT.md` for the full before/after summary and pipeline diagram.

## Answer/Eval Constraints
- System prompts forbid citing stale sources such as Luật Giao thông đường bộ 2008, NĐ 100/2019, NĐ 46/2016, NĐ 123/2021, or drafts.
- Config meanings are fixed: A base/no RAG, B base+RAG, C LoRA/no RAG, D LoRA+RAG; demo defaults to D.
- No rule-base phrase→article lookup, no regex direct-answer path, no LLM bypass. Retrieval relies on bge-m3-traffic-ft (dense), BM25 (sparse), doc-alias + article-mention lists, RRF fusion, and Cross-Encoder rerank with expanded query.
