#!/bin/bash
# Run all pending dim-smoke models on Intel
# Gated models require HF token with accepted licenses
# PyTorch-only models will fail G0-PULL (expected)
set -euo pipefail

cd ~/src/apr-model-qa-playbook
RESULTS="$HOME/data/dim-smoke-pending-$(date +%Y%m%dT%H%M%S).txt"
BINARY="cargo run --bin apr-qa --"

echo "=== Pending Model Dim-Smoke Qualification ==="
echo "Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Results: $RESULTS"
echo ""

PLAYBOOKS=(
  # Meta Llama (gated - manual license)
  "llama-3-2-1b-instruct-dim-smoke.playbook.yaml"
  "llama-3-2-3b-instruct-dim-smoke.playbook.yaml"
  "llama-3.1-8b-dim-smoke.playbook.yaml"
  "llama-3.1-70b-dim-smoke.playbook.yaml"
  "llama-3.3-70b-dim-smoke.playbook.yaml"
  "codellama-7b-dim-smoke.playbook.yaml"
  "codellama-13b-dim-smoke.playbook.yaml"
  "codellama-34b-dim-smoke.playbook.yaml"
  "codellama-70b-dim-smoke.playbook.yaml"
  # Google Gemma (gated - manual license)
  "gemma-2-2b-it-dim-smoke.playbook.yaml"
  "gemma-2-9b-it-dim-smoke.playbook.yaml"
  "gemma-2-27b-dim-smoke.playbook.yaml"
  "codegemma-7b-dim-smoke.playbook.yaml"
  "gemma-3-1b-it-dim-smoke.playbook.yaml"
  "gemma-3-4b-dim-smoke.playbook.yaml"
  "gemma-3-12b-dim-smoke.playbook.yaml"
  "gemma-3-27b-dim-smoke.playbook.yaml"
  # PyTorch-only (expected to fail G0-PULL)
  "vicuna-7b-dim-smoke.playbook.yaml"
  "vicuna-13b-dim-smoke.playbook.yaml"
  "wizardcoder-15b-dim-smoke.playbook.yaml"
  "wizardcoder-33b-dim-smoke.playbook.yaml"
)

> "$RESULTS"
passed=0
failed=0
total=${#PLAYBOOKS[@]}

for pb in "${PLAYBOOKS[@]}"; do
  echo "=== $pb ===" | tee -a "$RESULTS"
  output=$($BINARY run "playbooks/models/$pb" --metadata-only --failure-policy collect-all 2>&1 || true)
  result=$(echo "$output" | grep -E '(Pass rate|Gateway FAILED|G0-PULL)' || echo "  UNKNOWN")
  echo "$result" | tee -a "$RESULTS"
  echo "" >> "$RESULTS"

  if echo "$result" | grep -q "100.0%"; then
    passed=$((passed + 1))
  else
    failed=$((failed + 1))
  fi
done

echo "" | tee -a "$RESULTS"
echo "=== SUMMARY ===" | tee -a "$RESULTS"
echo "Passed: $passed / $total" | tee -a "$RESULTS"
echo "Failed: $failed / $total" | tee -a "$RESULTS"
echo "Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$RESULTS"
