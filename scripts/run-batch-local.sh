#!/bin/bash
# Run batch trajectory collection with local model
#
# Prerequisites:
#   - vLLM server running on localhost:8000
#   - Model loaded
#
# Usage:
#   ./scripts/run-batch-local.sh [limit]

set -e

export PATH="$HOME/.bun/bin:$PATH"

# Configuration
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://localhost:8000/v1}"
export OPENAI_MODEL="${OPENAI_MODEL:-agent-lora}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-not-needed-for-local}"

LIMIT="${1:-0}"
KAGGLE_DIR="${KAGGLE_DIR:-../claude-code-clean/datasets/kaggle-datasets-1001-1100}"
OUTPUT_DIR="${OUTPUT_DIR:-./local-trajectories}"
CHECKPOINT="${CHECKPOINT:-./checkpoint-local.json}"

echo "=============================================="
echo "Batch Local Model Trajectory Collection"
echo "=============================================="
echo "API: ${OPENAI_BASE_URL}"
echo "Model: ${OPENAI_MODEL}"
echo "Dataset dir: ${KAGGLE_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo "Limit: ${LIMIT:-unlimited}"
echo ""

bun run src/entrypoints/batchCollectLocal.tsx \
  --kaggle-dir "${KAGGLE_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --max-tool-iterations 30 \
  --checkpoint "${CHECKPOINT}" \
  --limit "${LIMIT}"
