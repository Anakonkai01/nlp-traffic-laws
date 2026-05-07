#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -d vector_db_traffic_clauses_v1 ]]; then
  echo "[rollback] missing vector_db_traffic_clauses_v1; nothing to restore" >&2
  exit 1
fi

rm -rf vector_db_traffic
mv vector_db_traffic_clauses_v1 vector_db_traffic
python src/sparse_bm25.py build

if [[ -f reports/traffic/retrieval_ablation_clauses_v1.csv ]]; then
  cp reports/traffic/retrieval_ablation_clauses_v1.csv reports/traffic/retrieval_ablation.csv
fi

echo "[rollback] restored clause-level KB from vector_db_traffic_clauses_v1"
