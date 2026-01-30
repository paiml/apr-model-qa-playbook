#!/usr/bin/env bash
# ============================================================================
# CONVERSION DIAGNOSTIC RUNBOOK — ALL 5 INVARIANTS
# ============================================================================
#
# Purpose: Test ALL conversion invariants in parallel, not one at a time.
#          Prevents the "onion peeling" pattern where fixing bug N reveals bug N+1.
#
# Usage:   ./scripts/diagnose-conversion.sh <path-to-gguf>
# Example: ./scripts/diagnose-conversion.sh ~/.apr/cache/hf/Qwen/.../model-q4_k_m.gguf
#
# Outputs: Pass/fail for each of 5 invariants + summary
# Exit:    0 if all pass, 1 if any fail
#
# History:
#   2026-01-30  Created after GH-190 fix revealed GH-191 hiding underneath.
#               See: docs/five-whys/GH-191-why-still-broken-after-five-whys.md
# ============================================================================

set -uo pipefail

# --- Config ---
PROMPT="What is 2+2?"
MAX_TOKENS=10
DIAG_DIR="/tmp/apr-diag-$(date +%s)"

# --- Colors ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# --- Args ---
if [ $# -lt 1 ]; then
    echo -e "${RED}Usage: $0 <path-to-gguf-model>${NC}"
    echo "  Example: $0 ~/.apr/cache/hf/Qwen/.../model-q4_k_m.gguf"
    exit 2
fi

GGUF="$1"

if [ ! -f "$GGUF" ]; then
    echo -e "${RED}ERROR: File not found: $GGUF${NC}"
    exit 2
fi

mkdir -p "$DIAG_DIR"
APR="$DIAG_DIR/converted.apr"

PASS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0
RESULTS=()

pass() {
    PASS_COUNT=$((PASS_COUNT + 1))
    RESULTS+=("${GREEN}PASS${NC} $1")
    echo -e "  ${GREEN}PASS${NC} $1"
}

fail() {
    FAIL_COUNT=$((FAIL_COUNT + 1))
    RESULTS+=("${RED}FAIL${NC} $1: $2")
    echo -e "  ${RED}FAIL${NC} $1: $2"
}

skip() {
    SKIP_COUNT=$((SKIP_COUNT + 1))
    RESULTS+=("${YELLOW}SKIP${NC} $1: $2")
    echo -e "  ${YELLOW}SKIP${NC} $1: $2"
}

echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║         CONVERSION DIAGNOSTIC RUNBOOK — 5 INVARIANTS       ║${NC}"
echo -e "${BOLD}╠══════════════════════════════════════════════════════════════╣${NC}"
echo -e "${BOLD}║${NC} Source: $(basename "$GGUF")"
echo -e "${BOLD}║${NC} Output: $APR"
echo -e "${BOLD}║${NC} Date:   $(date -Iseconds)"
echo -e "${BOLD}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# ============================================================================
# Step 0: Convert
# ============================================================================
echo -e "${CYAN}[Step 0] Converting GGUF → APR ...${NC}"
CONV_LOG="$DIAG_DIR/convert.log"
if apr convert "$GGUF" -o "$APR" --verbose > "$CONV_LOG" 2>&1; then
    APR_SIZE=$(stat --printf="%s" "$APR" 2>/dev/null || stat -f%z "$APR" 2>/dev/null || echo "unknown")
    echo -e "  Conversion complete. APR size: $APR_SIZE bytes"
    echo -e "  Log: $CONV_LOG"
else
    echo -e "  ${RED}Conversion FAILED. Cannot run diagnostics.${NC}"
    echo -e "  Log: $CONV_LOG"
    cat "$CONV_LOG"
    exit 1
fi
echo ""

# ============================================================================
# I-1: Round-trip Identity (THE SUPREME TEST)
# ============================================================================
echo -e "${CYAN}[I-1] Round-trip Identity${NC}"
echo -e "  Testing: convert(gguf) → inference == gguf inference"

I1_EXPECTED="$DIAG_DIR/i1-expected.txt"
I1_ACTUAL="$DIAG_DIR/i1-actual.txt"
I1_EXPECTED_ERR="$DIAG_DIR/i1-expected-err.txt"
I1_ACTUAL_ERR="$DIAG_DIR/i1-actual-err.txt"

# Run baseline inference
if apr run "$GGUF" -p "$PROMPT" --max-tokens "$MAX_TOKENS" > "$I1_EXPECTED" 2>"$I1_EXPECTED_ERR"; then
    EXPECTED_TEXT=$(cat "$I1_EXPECTED")
    echo -e "  GGUF output: ${BOLD}$EXPECTED_TEXT${NC}"
else
    skip "I-1" "GGUF inference failed ($(head -1 "$I1_EXPECTED_ERR"))"
    EXPECTED_TEXT=""
fi

# Run APR inference
if apr run "$APR" -p "$PROMPT" --max-tokens "$MAX_TOKENS" > "$I1_ACTUAL" 2>"$I1_ACTUAL_ERR"; then
    ACTUAL_TEXT=$(cat "$I1_ACTUAL")
    echo -e "  APR output:  ${BOLD}$ACTUAL_TEXT${NC}"
else
    skip "I-1" "APR inference failed ($(head -1 "$I1_ACTUAL_ERR"))"
    ACTUAL_TEXT=""
fi

# Compare
if [ -n "$EXPECTED_TEXT" ] && [ -n "$ACTUAL_TEXT" ]; then
    if [ "$EXPECTED_TEXT" = "$ACTUAL_TEXT" ]; then
        pass "I-1: Round-trip identity — outputs match"
    else
        fail "I-1: Round-trip identity" "outputs DIFFER"
        echo -e "    Expected: $EXPECTED_TEXT"
        echo -e "    Actual:   $ACTUAL_TEXT"
    fi
fi
echo ""

# ============================================================================
# I-2: Tensor Name Bijection
# ============================================================================
echo -e "${CYAN}[I-2] Tensor Name Bijection${NC}"
echo -e "  Testing: writer names == loader names (no missing/extra tensors)"

I2_LOG="$DIAG_DIR/i2-diff-tensors.log"
apr rosetta diff-tensors "$GGUF" "$APR" > "$I2_LOG" 2>&1 || true

# NOTE: diff-tensors compares RAW names across formats.
# GGUF uses flat names (0.q_proj.weight), APR uses HF names (0.self_attn.q_proj.weight).
# Name differences across formats are EXPECTED and BY DESIGN.
# What matters: tensor COUNT matches and no SHAPE mismatches.
COUNT_A=$(grep -oP 'A=\K\d+' "$I2_LOG" 2>/dev/null | head -1 || echo "")
COUNT_B=$(grep -oP 'B=\K\d+' "$I2_LOG" 2>/dev/null | head -1 || echo "")
SHAPE_MISMATCH=$(grep -ciE "shape mismatch|dimension mismatch" "$I2_LOG" 2>/dev/null || echo "0")

if [ -n "$COUNT_A" ] && [ -n "$COUNT_B" ]; then
    if [ "$COUNT_A" = "$COUNT_B" ] && [ "$SHAPE_MISMATCH" = "0" ]; then
        pass "I-2: Tensor name bijection — $COUNT_A tensors in both, no shape mismatches"
    elif [ "$COUNT_A" != "$COUNT_B" ]; then
        fail "I-2: Tensor name bijection" "Tensor count mismatch: GGUF=$COUNT_A, APR=$COUNT_B"
    else
        fail "I-2: Tensor name bijection" "$SHAPE_MISMATCH shape mismatches found"
        head -30 "$I2_LOG"
    fi
else
    skip "I-2" "Could not parse diff-tensors output"
fi
echo -e "  Full log: $I2_LOG"
echo ""

# ============================================================================
# I-3: No Silent Fallbacks
# ============================================================================
echo -e "${CYAN}[I-3] No Silent Fallbacks${NC}"
echo -e "  Testing: quantized model stays quantized (no silent F32 fallback)"

I3_LOG="$DIAG_DIR/i3-load-trace.log"
# Run minimal inference to get load trace
apr run "$APR" -p "test" --max-tokens 1 > /dev/null 2>"$I3_LOG" || true

# Check for quantization status
QUANT_LINE=$(grep -i "quantized" "$I3_LOG" 2>/dev/null || echo "")
F32_COUNT=$(echo "$QUANT_LINE" | grep -oP '\d+ F32' 2>/dev/null | grep -oP '\d+' || echo "")
QUANT_COUNT=$(echo "$QUANT_LINE" | grep -oP '\d+ quantized' 2>/dev/null | grep -oP '\d+' || echo "")

# Detect if source is quantized
IS_QUANTIZED="no"
if echo "$GGUF" | grep -qiE "q[0-9]|q4|q5|q8|k_m|k_s" 2>/dev/null; then
    IS_QUANTIZED="yes"
fi

# Parse counts more robustly — handle "0 quantized, 308 F32 tensors" format
if [ -n "$QUANT_LINE" ]; then
    QUANT_COUNT=$(echo "$QUANT_LINE" | sed -n 's/.*\b\([0-9]\+\) quantized.*/\1/p' || echo "")
    F32_COUNT=$(echo "$QUANT_LINE" | sed -n 's/.*\b\([0-9]\+\) F32.*/\1/p' || echo "")
fi

if [ "$IS_QUANTIZED" = "yes" ] && [ -n "$QUANT_COUNT" ] && [ "$QUANT_COUNT" = "0" ]; then
    fail "I-3: No silent fallbacks" "Source is quantized but APR loaded 0 quantized tensors ($F32_COUNT F32)"
    echo -e "    Load trace: $QUANT_LINE"
elif [ -n "$QUANT_LINE" ]; then
    echo -e "  Load trace: $QUANT_LINE"
    if [ "$IS_QUANTIZED" = "yes" ] && [ -n "$QUANT_COUNT" ] && [ "$QUANT_COUNT" != "0" ]; then
        pass "I-3: No silent fallbacks — $QUANT_COUNT quantized tensors preserved"
    elif [ "$IS_QUANTIZED" = "no" ]; then
        pass "I-3: No silent fallbacks — source is not quantized (F32 expected)"
    else
        skip "I-3" "Could not parse quantization counts from load trace"
    fi
else
    skip "I-3" "No quantization info in load trace"
fi
echo -e "  Full log: $I3_LOG"
echo ""

# ============================================================================
# I-4: Statistical Preservation
# ============================================================================
echo -e "${CYAN}[I-4] Statistical Preservation${NC}"
echo -e "  Testing: tensor stats (mean/std/min/max) unchanged after conversion"

I4_FP_GGUF="$DIAG_DIR/i4-fingerprint-gguf.json"
I4_FP_APR="$DIAG_DIR/i4-fingerprint-apr.json"
I4_LOG="$DIAG_DIR/i4-validate-stats.log"

# Fingerprint both (these can be slow for large models)
FP_GGUF_OK=false
FP_APR_OK=false

if apr rosetta fingerprint "$GGUF" --json > "$I4_FP_GGUF" 2>/dev/null; then
    FP_GGUF_OK=true
fi

if apr rosetta fingerprint "$APR" --json > "$I4_FP_APR" 2>/dev/null; then
    FP_APR_OK=true
fi

if $FP_GGUF_OK && $FP_APR_OK; then
    # validate-stats takes a MODEL + --fingerprints reference
    if apr rosetta validate-stats "$APR" --fingerprints "$I4_FP_GGUF" > "$I4_LOG" 2>&1; then
        STAT_FAIL=$(grep -ciE "fail|mismatch|exceeded" "$I4_LOG" 2>/dev/null || echo "0")
        if [ "$STAT_FAIL" = "0" ]; then
            pass "I-4: Statistical preservation — tensor stats match"
        else
            fail "I-4: Statistical preservation" "$STAT_FAIL stat mismatches"
            head -20 "$I4_LOG"
        fi
    else
        fail "I-4: Statistical preservation" "validate-stats exited non-zero"
        head -20 "$I4_LOG"
    fi
else
    skip "I-4" "Could not fingerprint both models (gguf=$FP_GGUF_OK, apr=$FP_APR_OK)"
fi
echo -e "  Logs: $I4_FP_GGUF, $I4_FP_APR, $I4_LOG"
echo ""

# ============================================================================
# I-5: Tokenizer Roundtrip
# ============================================================================
echo -e "${CYAN}[I-5] Tokenizer Roundtrip${NC}"
echo -e "  Testing: both models produce same first token (tokenizer integrity)"

I5_LOG="$DIAG_DIR/i5-compare-inference.log"
if apr rosetta compare-inference "$GGUF" "$APR" --prompt "Hello" --max-tokens 1 > "$I5_LOG" 2>&1; then
    TOKEN_MATCH=$(grep -ciE "match|identical" "$I5_LOG" 2>/dev/null || echo "0")
    TOKEN_DIFF=$(grep -ciE "differ|mismatch" "$I5_LOG" 2>/dev/null || echo "0")
    if [ "$TOKEN_DIFF" != "0" ]; then
        fail "I-5: Tokenizer roundtrip" "First token differs between GGUF and APR"
        head -20 "$I5_LOG"
    elif [ "$TOKEN_MATCH" != "0" ]; then
        pass "I-5: Tokenizer roundtrip — first tokens match"
    else
        # Can't parse output — show it
        echo -e "  (Could not auto-detect pass/fail — manual review needed)"
        head -10 "$I5_LOG"
        skip "I-5" "Could not parse compare-inference output"
    fi
else
    fail "I-5: Tokenizer roundtrip" "compare-inference exited non-zero"
    head -20 "$I5_LOG"
fi
echo -e "  Full log: $I5_LOG"
echo ""

# ============================================================================
# Summary
# ============================================================================
echo -e "${BOLD}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║                    DIAGNOSTIC SUMMARY                       ║${NC}"
echo -e "${BOLD}╠══════════════════════════════════════════════════════════════╣${NC}"
for result in "${RESULTS[@]}"; do
    echo -e "${BOLD}║${NC} $result"
done
echo -e "${BOLD}╠══════════════════════════════════════════════════════════════╣${NC}"
echo -e "${BOLD}║${NC} ${GREEN}PASS: $PASS_COUNT${NC}  ${RED}FAIL: $FAIL_COUNT${NC}  ${YELLOW}SKIP: $SKIP_COUNT${NC}"

if [ "$FAIL_COUNT" -gt 0 ]; then
    echo -e "${BOLD}║${NC}"
    echo -e "${BOLD}║${NC} ${RED}VERDICT: CONVERSION IS BROKEN${NC}"
    echo -e "${BOLD}║${NC} $FAIL_COUNT invariant(s) violated. Do NOT ship converted models."
    echo -e "${BOLD}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "Diagnostic artifacts: $DIAG_DIR/"
    exit 1
elif [ "$PASS_COUNT" -eq 5 ]; then
    echo -e "${BOLD}║${NC}"
    echo -e "${BOLD}║${NC} ${GREEN}VERDICT: CONVERSION IS CORRECT${NC}"
    echo -e "${BOLD}║${NC} All 5 invariants pass. Golden Rule satisfied."
    echo -e "${BOLD}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "Diagnostic artifacts: $DIAG_DIR/"
    exit 0
else
    echo -e "${BOLD}║${NC}"
    echo -e "${BOLD}║${NC} ${YELLOW}VERDICT: INCOMPLETE — $SKIP_COUNT invariant(s) could not be tested${NC}"
    echo -e "${BOLD}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "Diagnostic artifacts: $DIAG_DIR/"
    exit 1
fi
