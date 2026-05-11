# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Parent repo `/home/pc5070ti/workspace/SDA/CLAUDE.md` covers the two-project split (this is the `nlp/` project). `AGENTS.md` in this directory has the most current operational notes, verified data counts, and result baselines — read it before making decisions about splits, retrieval knobs, or artifact state.

## Project overview

Local-text-only Vietnamese traffic-law QA pipeline:

```text
filter traffic laws → build KB → generate QA → filter/split → LoRA fine-tune → evaluate → demo
```

Four evaluation configs, fixed meanings:
- **A** = base model, no RAG
- **B** = base model + RAG
- **C** = LoRA-fine-tuned model, no RAG
- **D** = LoRA-fine-tuned model + RAG (demo default)

## Non-obvious conventions

### Import / execution
- Scripts in `src/` import sibling modules by bare name (e.g. `from config import ...`). Run via `cd nlp && python src/<script>.py` or set `PYTHONPATH=src`. Do not run them from parent directories without that setup.
- `src/finetune.py`, `src/evaluate.py`, `src/evaluate_mc.py`, `src/app.py` import `unsloth` before any `torch`/`transformers` import. Preserve this ordering when editing imports — unsloth patches torch/transformers at import time.
- `src/build_kb.py` sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` before loading embeddings. The embedding model (default `BAAI/bge-m3`, override with `EMBED_MODEL`) must already be in the HF cache, or build fails.

### Data policy — do not break
- Production path is **local_text_only**: KB, QA gen, fine-tune, eval, demo read only UTF-8 `.txt` files with `enabled: true` in `docs/docs_giaothong/manifest.json`.
- Do not add VLSP corpus, PDF fallback, web crawling, or legacy traffic-law sources to production flow (PDFs live beside the `text/` dir but must not be ingested by production).
- `load_vectorstore()` rejects a KB missing `vector_db_traffic/build_meta.json` or with mismatched `source_policy`. Preserve this guardrail. After changing the manifest or chunking, rebuild with `python src/build_kb.py --force`.
- New legal source: drop UTF-8 text under `docs/docs_giaothong/text/` → add manifest entry with `enabled: true` → rebuild KB.

### QA data files — easy to confuse
- `src/generate_qa.py` default output is `data/splits_filtered/qa_train.jsonl` (from `QA_DATA_PATH` in `src/config.py`) — not the monolithic `data/qa_pairs_traffic.jsonl`. Pass `--output` to regenerate the monolithic file.
- `scripts/filter_qa.py` reads and overwrites `data/qa_pairs_traffic.jsonl` only; it does **not** touch `data/splits_filtered/qa_train.jsonl`.
- Split creation is a separate step: `python scripts/make_splits.py --qa data/qa_pairs_traffic_filtered.jsonl --outdir data/splits_filtered --seed 42`.
- Fine-tune reads the frozen split files (`qa_train.jsonl`, `qa_dev.jsonl`) and does not reshuffle.
- `QA_SPLIT_DIR` env var redirects all three split paths at once.

### Retrieval
- Default is `RETRIEVAL_VERSION=v4` in `src/retrieval.py` (hybrid FAISS dense + BM25 + doc alias + article-mention fusion with weighted RRF). Set `v3` only for ablation/rollback.
- v4 knobs: `RAG_V4_DENSE_K`, `RAG_V4_SPARSE_K`, `RAG_V4_FUSE_K`, `RAG_V4_W_DENSE`, `RAG_V4_W_SPARSE`, `RAG_V4_USE_CE`, `RETRIEVAL_CE_MODEL`. Legacy v2 cross-encoder flag is `RETRIEVAL_USE_CROSS_ENCODER`.
- BM25 side index is separate from FAISS: rebuild after KB changes with `python src/sparse_bm25.py build`. Inspect with `python src/sparse_bm25.py stats|search`.

### Prompts and forbidden citations
- Eval reports `forbidden_legacy_rate` to catch answers citing stale sources. `FORBIDDEN_LEGACY_TERMS` in `src/config.py` lists them (Luật GTĐB 2008, NĐ 100/2019, NĐ 46/2016, NĐ 123/2021, drafts, etc.). System prompts explicitly forbid these.
- No-context vs with-context prompts diverge on refusal behavior: with-context is permitted to answer from general knowledge if the snippet is insufficient (with "Cần kiểm tra lại căn cứ"); no-context always answers from general knowledge. Don't collapse them into one prompt.

### Artifacts (gitignored, must be present/rebuilt to run)
- `vector_db_traffic/` — FAISS + BM25 + `build_meta.json`. Required for configs B/D, retrieval diagnostics, and eval.
- `models/qwen3.5-9b-lora-traffic-v2/` — LoRA adapter. Required for configs C/D.
- Absent in a fresh checkout; any command needing RAG or LoRA will fail until they are rebuilt.

### Environment variables
- QA generation: Ollama at `http://localhost:11434/api/chat` with model `qwen3.5:9b` by default; alt: `--backend openrouter` + `OPENROUTER_API_KEY`, or `GEMINI_API_KEY`.
- Eval LLM judge (opt-in via `--judge`): `OPENROUTER_API_KEY` by default, or `--judge-backend minimax` + `MINIMAX_API_KEY`.
- `auto_pipeline.sh` uses `conda run -n "${CONDA_ENV:-ai}"`; override `CONDA_ENV` if the env name differs.

## Common commands

All from `nlp/` working directory.

Full pipeline (sequential wrapper):
```bash
REBUILD_KB=1 REGENERATE_QA=1 bash auto_pipeline.sh
```

Manual steps:
```bash
python scripts/filter_traffic_laws.py      # audit manifest
python src/build_kb.py --force             # rebuild FAISS KB
python src/sparse_bm25.py build            # rebuild BM25 side index
python src/generate_qa.py --force          # regen QA (writes to data/splits_filtered/qa_train.jsonl)
python scripts/filter_qa.py                # filters data/qa_pairs_traffic.jsonl only
python scripts/make_splits.py --qa data/qa_pairs_traffic_filtered.jsonl --outdir data/splits_filtered --seed 42
python src/finetune.py                     # LoRA → models/qwen3.5-9b-lora-traffic-v2/
python src/evaluate.py --configs A B C D
python src/evaluate_mc.py --configs A B C D
python src/app.py                          # Gradio demo (lazy-swaps base/LoRA for 16GB VRAM)
```

Smoke-size commands (use these while iterating):
```bash
python src/generate_qa.py --force --profile smoke
python src/generate_qa.py --force --phases penalties procedures --phase-size 20 --no-negatives --timeout 180 --num-predict 160
python src/evaluate.py --configs D --samples 10 --fast-metrics
python src/evaluate.py --retrieval-only --samples 10
python src/evaluate_mc.py --configs D --samples 10
```

## Development notes

- No pytest/ruff/mypy config and no CI. Prefer the smoke commands above over inventing test commands.
- Large artifacts (`models/`, `vector_db_traffic/`, `wandb/`, `logs/`, PDFs, large QA JSONL) are intentionally gitignored — avoid `git add .`, avoid committing regenerated checkpoints/datasets unless the user asks.
- Keep changes surgical; do not refactor adjacent code or introduce PDF/VLSP fallbacks "for robustness".
