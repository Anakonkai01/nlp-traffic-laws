#!/usr/bin/env bash
# Deterministic local-text-only pipeline.
#
# Usage:
#   bash nlp/auto_pipeline.sh
#   REBUILD_KB=1 REGENERATE_QA=1 bash nlp/auto_pipeline.sh

set -euo pipefail

WORKDIR="/home/pc5070ti/workspace/SDA/nlp"
CONDA_ENV="${CONDA_ENV:-ai}"
REBUILD_KB="${REBUILD_KB:-0}"
REGENERATE_QA="${REGENERATE_QA:-0}"
RUN_FINETUNE="${RUN_FINETUNE:-1}"
RUN_EVAL="${RUN_EVAL:-1}"

cd "$WORKDIR"

run_py() {
    conda run -n "$CONDA_ENV" python "$@"
}

log() {
    printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*"
}

log "Step 1/5: audit local source manifest"
run_py scripts/filter_traffic_laws.py

log "Step 2/5: build local-text-only KB"
if [[ "$REBUILD_KB" == "1" ]]; then
    run_py src/build_kb.py --force
else
    run_py src/build_kb.py
fi

log "Step 3/5: generate local-text-only QA"
if [[ "$REGENERATE_QA" == "1" ]]; then
    run_py src/generate_qa.py --force
else
    run_py src/generate_qa.py
fi

log "Step 3b/5: filter QA (long answers + templates)"
run_py scripts/filter_qa.py

if [[ "$RUN_FINETUNE" == "1" ]]; then
    log "Step 4/5: fine-tune LoRA adapter"
    run_py src/finetune.py
else
    log "Step 4/5: fine-tune skipped"
fi

if [[ "$RUN_EVAL" == "1" ]]; then
    log "Step 5/5: evaluate local manual sets"
    run_py src/evaluate.py --configs A B C D
    run_py src/evaluate_mc.py --configs A B C D
else
    log "Step 5/5: eval skipped"
fi

log "Pipeline complete."
