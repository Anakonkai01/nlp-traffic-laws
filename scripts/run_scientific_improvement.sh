#!/usr/bin/env bash
# run_scientific_improvement.sh
# Chạy toàn bộ pipeline cải thiện khoa học theo 3 tracks.
# Usage: bash scripts/run_scientific_improvement.sh [--skip-embedder] [--skip-reranker]
#
# Track 1: Fine-tune bge-m3 embedder → rebuild KB → eval D baseline
# Track 2: Fine-tune fact reranker v2 → eval D với CE reranker
# Track 3 (optional): eval D với cả embedder + reranker mới

set -euo pipefail
cd "$(dirname "$0")/.."

SKIP_EMBEDDER=0
SKIP_RERANKER=0
for arg in "$@"; do
  case $arg in
    --skip-embedder) SKIP_EMBEDDER=1 ;;
    --skip-reranker) SKIP_RERANKER=1 ;;
  esac
done

LOG_DIR="reports/traffic"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=== Scientific Improvement Pipeline ($TIMESTAMP) ==="
echo "SKIP_EMBEDDER=$SKIP_EMBEDDER  SKIP_RERANKER=$SKIP_RERANKER"
echo ""

# ── 0. Generate training data (fast, always run) ────────────────────────────
echo "[0/5] Generating penalty training pairs..."
python scripts/generate_penalty_pairs.py
python scripts/build_fact_reranker_v2.py
echo ""

# ── 1. Eval D baseline: skip (run separately to avoid VRAM conflict with training)
echo "[1/5] Baseline eval skipped — run separately after training:"
echo "  see scripts/run_scientific_improvement.sh --skip-embedder --skip-reranker"
echo ""

# ── 2. Track 1: Fine-tune embedder ──────────────────────────────────────────
if [ "$SKIP_EMBEDDER" -eq 0 ]; then
  echo "[2/5] Fine-tuning bge-m3 embedder (Track 1)..."
  python scripts/finetune_embedder.py \
    --epochs 3 \
    --batch-size 4 \
    --grad-accum 8 \
    --max-seq-length 256 \
    --lr 2e-5 \
    --max-pairs 8000 \
    2>&1 | tee "$LOG_DIR/finetune_embedder_${TIMESTAMP}.log"

  echo "Rebuilding KB with fine-tuned embedder..."
  EMBED_MODEL="models/bge-m3-traffic-ft" python src/build_kb.py --force \
    2>&1 | tee "$LOG_DIR/rebuild_kb_ft_embedder_${TIMESTAMP}.log"
  echo "Track 1 done. Eval D separately (see end of this script)."
else
  echo "[2/5] Skipping embedder fine-tune (--skip-embedder)"
fi
echo ""

# ── 3. Track 2: Fine-tune fact reranker v2 ──────────────────────────────────
if [ "$SKIP_RERANKER" -eq 0 ]; then
  echo "[3/5] Fine-tuning fact reranker v2 (Track 2)..."
  python scripts/finetune_fact_reranker_v2.py \
    --epochs 3 \
    --batch-size 8 \
    --grad-accum 4 \
    --lr 2e-5 \
    2>&1 | tee "$LOG_DIR/finetune_reranker_v2_${TIMESTAMP}.log"

  echo "Track 2 done. Eval D separately (see end of this script)."
else
  echo "[3/5] Skipping reranker fine-tune (--skip-reranker)"
fi
echo ""

# ── 4. Summary ───────────────────────────────────────────────────────────────
echo "[4/5] Training complete. Run evals AFTER training (separate process, no VRAM conflict)."
echo ""

# ── 5. Full eval nếu smoke pass ─────────────────────────────────────────────
echo "[5/5] Skipping full eval — run manually if smoke results are good:"
echo ""
echo "  # Full eval: embedder + reranker v2"
echo "  EMBED_MODEL=models/bge-m3-traffic-ft \\"
echo "  RAG_FACT_USE_CE=1 RAG_FACT_CE_MODEL=models/fact-reranker-v2 \\"
echo "  NORMALIZE_LEGAL_NUMBERS=1 RAG_FALLBACK_ON_SHORT_AMOUNT=1 \\"
echo "  RAG_HYBRID_FALLBACK=1 RAG_FACT_MIX_LEGAL_CARDS=1 \\"
echo "  RAG_FACT_MIX_LEGAL_TOP_K=1 RAG_FACT_CARD_FORMAT=narrative \\"
echo "  RAG_FACT_GENERIC_RERANK=1 RAG_SANCTION_FACT_CARDS=1 \\"
echo "  RAG_EVIDENCE_CARD_RENDERING=1 RAG_LEGAL_UNIT_RETRIEVAL=1 \\"
echo "  RAG_LEGAL_UNIT_TOP_K=6 RAG_LEGAL_UNIT_SEED_K=4 \\"
echo "  RAG_LEGAL_UNIT_APPEND_BACKUP=0 MAX_NEW_TOKENS=220 \\"
echo "  python src/evaluate.py --configs D"
echo ""
echo "=== Pipeline complete ==="
