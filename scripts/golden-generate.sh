#!/usr/bin/env bash
# Generate golden outputs using HuggingFace transformers
# Requires: uv, GPU with CUDA, HuggingFace model access
#
# Usage: golden-generate.sh <model> <version> <corpus-path>
# Example: golden-generate.sh Qwen/Qwen2.5-Coder-1.5B-Instruct v1 ../hf-ground-truth-corpus/oracle

set -euo pipefail

MODEL="${1:?Usage: golden-generate.sh <model> <version> <corpus-path>}"
VERSION="${2:?Usage: golden-generate.sh <model> <version> <corpus-path>}"
HF_CORPUS="${3:?Usage: golden-generate.sh <model> <version> <corpus-path>}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

echo "=== Generating Golden Outputs ==="

MODEL_SHORT=$(echo "${MODEL}" | sed 's|.*/||' | tr '[:upper:]' '[:lower:]')
OUTPUT_DIR="${HF_CORPUS}/${MODEL_SHORT}/${VERSION}"

echo "Model: ${MODEL}"
echo "Output: ${OUTPUT_DIR}"

mkdir -p "${OUTPUT_DIR}"
uv run "${PROJECT_ROOT}/scripts/generate_golden.py" \
    --model "${MODEL}" \
    --output "${OUTPUT_DIR}" \
    --prompts "${PROJECT_ROOT}/prompts/code_bench.txt"
