#!/usr/bin/env bash
# Chờ embedder fine-tune xong rồi chạy reranker + rebuild KB
set -euo pipefail
cd "$(dirname "$0")/.."

echo "Waiting for embedder training to finish..."
# Wait until the embedder log shows "Model saved"
LOG=$(ls reports/traffic/finetune_embedder_*.log 2>/dev/null | sort | tail -1)
until grep -q "Model saved to" "$LOG" 2>/dev/null; do sleep 10; done
echo "Embedder done. Starting reranker v2..."

python scripts/finetune_fact_reranker_v2.py \
    --epochs 3 --batch-size 8 --grad-accum 4 --lr 2e-5 \
    2>&1 | tee "reports/traffic/finetune_reranker_v2_$(date +%Y%m%d_%H%M%S).log"

echo "Reranker done."
echo ""
echo "Rebuilding KB with fine-tuned embedder..."
EMBED_MODEL="models/bge-m3-traffic-ft" python src/build_kb.py --force \
    2>&1 | tee "reports/traffic/rebuild_kb_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "=== All training complete ==="
echo "Run eval D with both improvements:"
echo "  EMBED_MODEL=models/bge-m3-traffic-ft \\"
echo "  RAG_FACT_USE_CE=1 RAG_FACT_CE_MODEL=models/fact-reranker-v2 \\"
echo "  NORMALIZE_LEGAL_NUMBERS=1 RAG_FALLBACK_ON_SHORT_AMOUNT=1 \\"
echo "  RAG_HYBRID_FALLBACK=1 RAG_FACT_MIX_LEGAL_CARDS=1 \\"
echo "  RAG_FACT_MIX_LEGAL_TOP_K=1 RAG_FACT_CARD_FORMAT=narrative \\"
echo "  RAG_FACT_GENERIC_RERANK=1 RAG_SANCTION_FACT_CARDS=1 \\"
echo "  RAG_EVIDENCE_CARD_RENDERING=1 RAG_LEGAL_UNIT_RETRIEVAL=1 \\"
echo "  RAG_LEGAL_UNIT_TOP_K=6 RAG_LEGAL_UNIT_SEED_K=4 \\"
echo "  RAG_LEGAL_UNIT_APPEND_BACKUP=0 MAX_NEW_TOKENS=220 \\"
echo "  python src/evaluate.py --configs D --samples 40"
