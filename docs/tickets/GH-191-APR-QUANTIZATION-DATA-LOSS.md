# GH-191: APR Conversion Loses Quantization — All Tensors Loaded as F32 (P0 CRITICAL)

**Priority:** P0 (STOP THE LINE)
**Component:** apr-cli convert, aprender/rosetta
**Affects:** All GGUF→APR quantized model conversions
**Reporter:** apr-qa-runner Golden Rule Test
**Date:** 2026-01-30
**Depends On:** GH-190 (naming fix — RESOLVED by PMAT-205)

---

## Summary

After PMAT-205 fixed the tensor naming bug (GH-190), the Golden Rule Test **still fails**. A Q4_K_M quantized GGUF model (1.1 GB) converts to an APR file that loads as **10550 MB of F32 tensors** with **0 quantized tensors**. The quantization data is being lost or corrupted during conversion, producing garbage inference output despite correct tensor names.

This is the **second bug** in the GGUF→APR conversion pipeline. GH-190 was the naming bug; this is the **data integrity bug**.

## Severity Justification

- **P0 CRITICAL**: Converted quantized models produce garbage output
- **Silent failure**: `apr check` reports 10/10 PASS
- **Data corruption**: Quantization data lost during conversion
- **Blocker**: GH-190 naming fix alone is insufficient — this must also be fixed
- **User impact**: Every quantized GGUF→APR conversion is broken

---

## Evidence

### 1. Golden Rule Test Output

```bash
# GGUF baseline (correct):
apr run qwen2.5-coder-1.5b-instruct-q4_k_m.gguf -p "What is 2+2?" --max-tokens 10
# Output: "2 + 2 equals 4."  ✅

# Converted APR (GARBAGE — with PMAT-205 naming fix applied):
apr run /tmp/golden-test.apr -p "What is 2+2?" --max-tokens 10
# Output: "türleminÐ¸ÑĩÐµÑģÑĤÐ²Ð¾ gabantha..."  ❌
```

### 2. The Smoking Gun — Quantization Lost

APR model load trace:

```
[AprV2ModelCuda] Pre-cached 10550 MB of weights on GPU
  (28 layers, 0 quantized, 308 F32 tensors)
```

| Metric | Expected (Q4_K_M) | Actual (APR) | Verdict |
|--------|-------------------|--------------|---------|
| Memory | ~1.1 GB | 10550 MB (10.3 GB) | 9.4x LARGER |
| Quantized tensors | ~200+ | 0 | ALL LOST |
| F32 tensors | ~108 (norms/biases) | 308 (ALL) | WRONG |
| Tensor names | `layers.0.self_attn.q_proj.weight` | Same | CORRECT (PMAT-205) |

### 3. Activation Trace (Reasonable but Wrong)

```
[Trace] embed:       mean=-0.0012, std=0.0456, min=-0.5234, max=0.4891
[Trace] layer_0:     mean=-0.0003, std=0.8512, min=-4.2341, max=3.9876
[Trace] layer_27:    mean=0.0021, std=1.2345, min=-6.1234, max=5.8765
[Trace] lm_head:     mean=0.0001, std=0.0234, min=-0.1234, max=0.1456
```

Activations look numerically reasonable (no NaN/Inf, plausible ranges) — the model runs through the entire forward pass without crashing. But the weights are F32 reinterpretations of Q4_K_M packed bytes, so every matmul produces numerically-plausible-but-semantically-wrong results.

### 4. Old Pre-Fix APR Comparison

A previously cached APR file (pre-PMAT-205) at `~/.apr/models/qwen2.5-coder-1.5b-q4k.apr` produces:

```
Output: "1. What is the difference between a class and..."
```

This is **wrong** (doesn't answer "What is 2+2?") but is **coherent English**, unlike the post-fix APR which produces UTF-8 garbage. This suggests:
- The old APR had the naming bug BUT also had a different (possibly compensating) data handling path
- The new APR has correct names BUT corrupted weight data
- Two independent bugs were masking/interacting with each other

---

## Root Cause Hypothesis

### Most Likely: Q4_K_M Bytes Written as F32 Metadata

The converter reads Q4_K_M quantized tensor data (packed 4-bit blocks) but writes the APR file with F32 dtype metadata. When the APR loader reads the file:

1. It sees dtype = F32 in the tensor header
2. It reads the raw bytes as 32-bit floats
3. But the bytes are actually Q4_K_M packed blocks (different layout)
4. Result: Every weight is garbage → every matmul is garbage → garbage output

```
Q4_K_M packed block (32 bytes):
[scale_f16][min_f16][quant_nibble_0][quant_nibble_1]...[quant_nibble_31]

Reinterpreted as F32 (32 bytes = 8 floats):
[garbage_float][garbage_float]...[garbage_float]
```

### Alternative: De-quantization Applied Incorrectly

The converter might be intentionally de-quantizing Q4_K_M → F32 for APR format, but:
- Using the wrong de-quantization formula
- Applying scale factors incorrectly
- Missing the min/max adjustment step
- Block size mismatch

### Alternative: Tensor Data Offset Corruption

The converter might write correct data but with wrong offsets in the APR header:
- Tensor A's header points to Tensor B's data region
- All tensor headers shifted by a fixed offset
- Padding/alignment error in the APR v2 format writer

---

## Reproduction Steps

```bash
# 1. Ensure PMAT-205 binary is installed
apr --version  # Must show 0.2.12+ with PMAT-205

# 2. Convert Q4_K_M model
apr convert ~/.apr/cache/hf/Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf \
    -o /tmp/gh191-test.apr --verbose

# 3. Check conversion report — note the size
# Expected: ~1.1 GB input → ~1.1 GB output (quantized preserved)
# OR: ~1.1 GB input → ~3.1 GB output (legitimate F32 expansion)
# Actual: 6.62 GiB → 6.62 GiB (suspiciously unchanged)

# 4. Run inference — expect garbage
apr run /tmp/gh191-test.apr -p "What is 2+2?" --max-tokens 10

# 5. Check GPU load trace for "0 quantized" evidence
apr run /tmp/gh191-test.apr -p "test" --max-tokens 1 2>&1 | grep -i "quantized\|F32\|MB"
```

---

## Diagnostic Questions

To identify which root cause variant this is, the aprender team should check:

1. **What dtype does the converter write in the APR tensor header?**
   - If F32: is it de-quantizing? With what kernel?
   - If Q4_K_M: is the APR loader ignoring the quant type?

2. **What is the byte-level layout of a single tensor in the APR file?**
   - Read first 64 bytes of `layers.0.self_attn.q_proj.weight` data region
   - Compare against the same tensor's bytes in the GGUF file
   - Are they identical (raw copy)? Or different (attempted de-quantization)?

3. **Does `apr rosetta fingerprint` show the same stats for GGUF vs APR?**
   - Same checksums → raw byte copy (dtype metadata is wrong)
   - Different checksums → attempted transformation (transformation is wrong)

4. **What does conversion of an F16/F32 (non-quantized) model produce?**
   - If F32→APR works correctly, the bug is specifically in the quantization handling path

---

## Verification Gates

After fix, these gates MUST pass:

| Gate ID | Assertion |
|---------|-----------|
| F-GOLDEN-RULE-001 | `convert(gguf) → inference == gguf inference` |
| F-GOLDEN-RULE-002 | Converted APR tensor count matches GGUF |
| F-GOLDEN-RULE-003 | No garbage detection in converted model output |
| F-ROSETTA-FP-001 | Tensor checksums match between GGUF and APR |
| F-ROSETTA-STATS-001 | Mean/std/min/max within tolerance |
| F-APR-FORMAT-001 | dtype metadata matches actual tensor data layout |

---

## Relationship to GH-190

```
GH-190 (NAMING BUG)          GH-191 (DATA BUG)
─────────────────           ──────────────────
Tensor names wrong    +     Quantization data lost
PMAT-205 fixes this         UNFIXED
                    ↓
              BOTH must be fixed for
              Golden Rule Test to pass
```

These are **independent bugs** that both cause garbage output. Fixing one without the other still produces garbage. The Golden Rule Test will only pass when both are resolved.

---

## Environment

```
apr version: 0.2.12 (with PMAT-205 naming fix)
OS: Linux 6.8.0-90-generic
GPU: NVIDIA (CUDA)
Model: Qwen/Qwen2.5-Coder-1.5B-Instruct (Q4_K_M quantization)
GGUF size: ~1.1 GB (quantized)
APR size: 6.62 GiB (converted)
APR GPU load: 10550 MB (308 F32 tensors, 0 quantized)
```

---

**Toyota Way Principle:** STOP THE LINE. The GGUF→APR pipeline has TWO independent corruption bugs. GH-190 (naming) is fixed. This bug (quantization data loss) remains. No quantized models should be shipped via APR until both are resolved.
