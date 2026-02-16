#!/bin/bash
# Sequential sub-1B model qualification sweep
# Uses `certify --tier dim-smoke` for metadata-only checks (~2-5s per model)
set -euo pipefail

# Sub-1B model IDs in execution order:
# Phase 1: Class A (kernel proof ref already certified)
# Phase 2: Class B (GPT-NeoX — new kernel class)
# Phase 3: Class D (Phi — new kernel class)
MODELS=(
    # Class A — Qwen/DeepSeek family variants
    "Qwen/Qwen2.5-0.5B-Instruct"
    "Qwen/Qwen2-0.5B-Instruct"
    "Qwen/Qwen3-0.6B"
    "deepseek-ai/deepseek-coder-1.3b-instruct"
    # Class B — GPT-NeoX (Pythia)
    "EleutherAI/pythia-410m-deduped"
    # Class D — Phi
    "microsoft/phi-1_5"
)

PASSED=0
FAILED=0
TOTAL=${#MODELS[@]}

echo "========================================"
echo " Sub-1B Sequential Qualification Sweep"
echo " Models: $TOTAL | Tier: dim-smoke"
echo " Mode: metadata-only (no inference)"
echo "========================================"
echo ""

for model_id in "${MODELS[@]}"; do
    short="${model_id##*/}"
    echo "--- [$((PASSED + FAILED + 1))/$TOTAL] $short ---"

    if cargo run --bin apr-qa -- certify \
        --tier dim-smoke \
        "$model_id" 2>&1; then
        echo "PASS  $short"
        PASSED=$((PASSED + 1))
    else
        echo "FAIL  $short"
        FAILED=$((FAILED + 1))
    fi
    echo ""

    # Brief pause between models to avoid I/O spikes
    sleep 1
done

echo "========================================"
echo " Results: $PASSED passed, $FAILED failed / $TOTAL total"
echo "========================================"

if [ "$FAILED" -gt 0 ]; then
    exit 1
fi
