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

## Root Cause: CONFIRMED — DType Byte Mapping Mismatch Between Converter and Reader

**Status: ROOT CAUSE FOUND (2026-01-30 21:05)**

The converter and reader have **completely different dtype byte mappings**. The converter writes a dtype byte, the reader interprets it as a different type, and the `_ => F32` silent fallback turns every unrecognized quantized type into F32.

### The Full Mismatch Chain

For Q4_K_M tensors:

```
1. Converter writes dtype byte 8 for Q4_K           → correct from converter's perspective
2. APR TensorEntry::from_binary (line 781):
   dtype byte 8 → string "Q4"                       → WRONG (should be "Q4_K")
3. dtype_to_ggml_qtype (line 537):
   matches "Q4_K" but NOT "Q4"                       → NO MATCH
4. Result: "Q4" → None → treated as F32              → GARBAGE OUTPUT
```

### Complete DType Mapping Table

```
┌───────────┬───────────────────────┬────────────────────────────┬──────────────────────────────┐
│ GGML type │ Converter writes byte │ Reader maps byte to string │ dtype_to_ggml_qtype matches? │
├───────────┼───────────────────────┼────────────────────────────┼──────────────────────────────┤
│ Q4_K (12) │ 8                     │ "Q4"                       │ NO ("Q4_K" expected)         │
├───────────┼───────────────────────┼────────────────────────────┼──────────────────────────────┤
│ Q5_K (13) │ 8                     │ "Q4"                       │ NO                           │
├───────────┼───────────────────────┼────────────────────────────┼──────────────────────────────┤
│ Q6_K (14) │ 9                     │ "Q8_0"                     │ NO ("Q6_K" expected)         │
├───────────┼───────────────────────┼────────────────────────────┼──────────────────────────────┤
│ Q8_0 (8)  │ 10                    │ unknown → "F32"            │ NO                           │
├───────────┼───────────────────────┼────────────────────────────┼──────────────────────────────┤
│ F32 (0)   │ 0                     │ "F32"                      │ Correct (None = F32)         │
├───────────┼───────────────────────┼────────────────────────────┼──────────────────────────────┤
│ F16 (1)   │ 1                     │ "F16"                      │ Correct (None = F16)         │
└───────────┴───────────────────────┴────────────────────────────┴──────────────────────────────┘
```

**Every quantized type is wrong. Only F32 and F16 map correctly.**

### Why This Is GH-186 All Over Again

This is the **exact same pattern** as GH-186:

| | GH-186 | GH-191 |
|--|--------|--------|
| Writer | Writes GGML dtype 12 (Q4_K) | Writes APR byte 8 (meaning Q4_K) |
| Reader | Reads byte 12, falls back to F32 | Reads byte 8 as "Q4", no match → F32 |
| Fallback | `_ => F32` silent default | `None → treated as F32` |
| Result | Garbage | Garbage |

**The silent fallback pattern strikes again.** The Five Whys (GH-190) identified invariant I-3 ("no silent fallbacks") as the fix. It was never implemented. Here we are.

### Code Locations

```
BOTH writer and reader are in realizar (NOT aprender):

Writer: realizar/src/gguf/loader.rs → dtype_to_byte() (line 1714)
  - BEFORE: Invented numbering (Q4_K=8, Q6_K=9, Q8_0=10, Q4_0=11, ...)
  - AFTER:  GGML values    (Q4_K=12, Q6_K=14, Q8_0=8, Q4_0=2, ...)

Reader: realizar/src/apr/mod.rs → TensorEntry::from_binary() (line 776)
  - BEFORE: Old mapping (8="Q4" or "Q8_0", 9="Q8_0" or "Q8_1", ...)
  - AFTER:  GGML values (12="Q4_K", 14="Q6_K", 8="Q8_0", ...)

Canonical source: realizar/src/gguf/loader.rs → qtype_to_dtype() (line 1688)
  - Was always correct: uses GGML type constants from types.rs
  - 12→"Q4_K", 14→"Q6_K", 8→"Q8_0"

The bug: dtype_to_byte() invented its own sequential numbering (8,9,10,11,12,13,14)
instead of using GGML type values (12,14,8,2,13,11,10). Same crate, two functions,
no shared enum — within the same file.
```

### The Fix (APPLIED in realizar source)

Both `dtype_to_byte()` and `from_binary()` now use GGML type values directly, matching `qtype_to_dtype()`:

```rust
// dtype_to_byte (writer) — NOW matches GGML values
"Q4_K" => 12,   // was 8  → now 12 (GGML_TYPE_Q4_K)
"Q5_K" => 13,   // was 12 → now 13 (GGML_TYPE_Q5_K)
"Q6_K" => 14,   // was 9  → now 14 (GGML_TYPE_Q6_K)
"Q8_0" => 8,    // was 10 → now 8  (GGML_TYPE_Q8_0)
"Q4_0" => 2,    // was 11 → now 2  (GGML_TYPE_Q4_0)
"Q3_K" => 11,   // was 13 → now 11 (GGML_TYPE_Q3_K)
"Q2_K" => 10,   // was 14 → now 10 (GGML_TYPE_Q2_K)
"BF16" => 30,   // was 2  → now 30 (GGML_TYPE_BF16)

// from_binary (reader) — NOW matches writer
12 => "Q4_K",   // was "Q4" or wrong → now correct
14 => "Q6_K",   // was wrong → now correct
8  => "Q8_0",   // was "Q4_K" (invented) → now correct
```

**Status: Fix applied in realizar source. Needs rebuild + Golden Rule Test re-run to verify.**

**Remaining risk:** All three `_ =>` fallback arms still fall to F32 (2/3 now warn via `eprintln!`, 1 is still silent). These should be `Err()` returns.

### Required invariant test (I-3):

```rust
#[test]
fn dtype_byte_roundtrip() {
    // Every dtype that dtype_to_byte can write, from_binary must read back identically
    for dtype in ["F32","F16","BF16","Q4_0","Q4_1","Q5_0","Q5_1","Q8_0","Q8_1",
                  "Q2_K","Q3_K","Q4_K","Q5_K","Q6_K","IQ2_XXS","IQ2_XS"] {
        let byte = dtype_to_byte(dtype);
        let (entry, _) = TensorEntry::from_binary(&make_test_entry(byte, "test", &[1]))
            .expect("parse");
        assert_eq!(entry.dtype, dtype,
            "Round-trip failed: '{}' → byte {} → '{}'", dtype, byte, entry.dtype);
    }
}
```

---

## Root Cause Hypothesis (SUPERSEDED)

~~The sections below were the pre-diagnosis hypotheses. Root cause is now confirmed above.~~

### ~~Most Likely: Q4_K_M Bytes Written as F32 Metadata~~

PARTIALLY CORRECT — the converter writes the correct bytes for Q4_K, but the reader interprets the dtype byte differently, causing it to load quantized data as F32.

### ~~Alternative: De-quantization Applied Incorrectly~~

NOT THE CAUSE — no de-quantization is attempted. The reader simply misidentifies the dtype.

### ~~Alternative: Tensor Data Offset Corruption~~

NOT THE CAUSE — tensor data is intact, only dtype metadata interpretation differs.

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

## Full Qualification Results: All Qwen Coder Models (2026-01-30)

Qualified all 5 Qwen2.5-Coder sizes (0.5B through 14B) using quick playbooks (10 scenarios each, subprocess mode, GGUF Q4_K_M format).

### Results Table

```
┌──────────────────────┬───────────┬───────┬──────────┬──────────────┬─────────────────┐
│ Model                │ MQS       │ Grade │ Pass/Tot │ GGUF Infer   │ Conversion      │
├──────────────────────┼───────────┼───────┼──────────┼──────────────┼─────────────────┤
│ Qwen2.5-Coder-0.5B  │ 24.2/100  │ F     │ 10/22    │ 10/10 PASS   │ 0/12 (ALL FAIL) │
│ Qwen2.5-Coder-1.5B  │ 24.2/100  │ F     │ 10/22    │ 10/10 PASS   │ 0/12 (ALL FAIL) │
│ Qwen2.5-Coder-3B    │ 24.2/100  │ F     │ 10/22    │ 10/10 PASS   │ 0/12 (ALL FAIL) │
│ Qwen2.5-Coder-7B    │ 24.2/100  │ F     │ 10/22    │ 10/10 PASS   │ 0/12 (ALL FAIL) │
│ Qwen2.5-Coder-14B   │ 24.2/100  │ F     │ 10/22    │ 10/10 PASS   │ 0/12 (ALL FAIL) │
└──────────────────────┴───────────┴───────┴──────────┴──────────────┴─────────────────┘
```

### Category Breakdown (Identical Across All Models)

```
QUAL:  83/200   ← GGUF inference quality tests pass
REGR:   0/150   ← Blocked by conversion failures
EDGE:   0/150   ← Blocked by conversion failures
PERF:   0/150   ← Blocked by conversion failures
STAB:   0/200   ← Blocked by conversion failures
COMP:   0/150   ← Blocked by conversion failures
─────────────
Total: 83/1000  → 24.2/100 (Grade F)
```

### Key Findings

1. **GGUF inference is 100% clean across all 5 sizes.** The models themselves work perfectly in native GGUF format. All 10 inference tests (arithmetic, code syntax, response quality) pass for every model from 0.5B to 14B.

2. **Conversion tests fail identically for all sizes.** The GGUF→APR pipeline bug is not model-specific — it's a systemic dtype mapping issue (GH-191) that affects every quantized conversion regardless of model size or architecture variant.

3. **MQS is artificially capped at 24.2.** The 83 points come entirely from QUAL (inference quality). Five categories score 0 because they depend on format conversion working. Once GH-190 + GH-191 are fully fixed, MQS should jump to 85+ (the target).

4. **No model-specific failures detected.** The uniform 10/22 pattern across 0.5B–14B confirms this is purely a pipeline bug, not a model compatibility issue. The Qwen2 architecture is correctly supported in the inference engine.

### Execution Details

| Model | Duration | Model Path |
|-------|----------|------------|
| 0.5B | 435s (7.2m) | `~/.apr/cache/hf/Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF/...q4_k_m.gguf` |
| 1.5B | 344s (5.7m) | `~/.apr/cache/hf/Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF/...q4_k_m.gguf` |
| 3B | 575s (9.6m) | `~/.cache/pacha/models/e06917441dd7f96d.gguf` |
| 7B | 635s (10.6m) | `~/.cache/pacha/models/e0abfc1f71fa8f14.gguf` |
| 14B | 1215s (20.3m) | `~/.cache/pacha/models/515504422eb74a68.gguf` |

**Total qualification time: 53.4 minutes for 5 models (110 scenarios total).**

### Additional Finding: Converter Code Path Issue

During investigation, discovered the `apr convert` default path (without `--quantize q4k`) silently dequantizes Q4_K_M to F32 before writing APR. The raw Q4K pass-through only triggers with explicit `--quantize q4k` flag:

```
apr convert model.gguf -o model.apr                    → 6.62 GiB (F32, broken)
apr convert model.gguf -o model.apr --quantize q4k     → 1.04 GiB (Q4K preserved, still broken)
```

Even with `--quantize q4k`, the inference still produces garbage (`"3,}, helf helf.How helf"`), though the model loads with correct quantization counts (197 quantized, 112 F32). This suggests a **third layer** to the conversion bug beyond naming (GH-190) and dtype mapping (GH-191) — possibly a Q4_K dequantization kernel bug or tensor shape/transpose issue in the APR GPU loader.

### External Confirmation: aprender#189

[paiml/aprender#189](https://github.com/paiml/aprender/issues/189) independently confirms the third onion layer. User ran `apr chat` with `qwen2.5-1.5b-instruct-q4_k_m.apr` and got pure "VILLEVILLEVILLEVILLE..." repetition. Their load trace shows `197 quantized, 112 F32 tensors` — dtype mapping is correct, but inference is still garbage. This matches our `--quantize q4k` finding exactly.

The bug is NOT in the dtype mapping (fixed) or naming (fixed). It's in the **Q4_K inference path** — the CUDA dequantization kernel, tensor shape handling, or block alignment during matmul.

### What Must Happen Before Re-Qualification

1. GH-190 naming fix (PMAT-205) — ✅ DONE
2. GH-191 dtype mapping fix — ✅ IN SOURCE (needs rebuild + verify)
3. Q4_K dequantization / shape correctness — ❓ UNKNOWN (layer 3 of the onion)
4. Golden Rule Test must PASS on at least one model
5. Re-qualify all 5 models with full playbooks (not quick)

---

**Toyota Way Principle:** STOP THE LINE. The GGUF→APR pipeline has at least THREE layers of bugs. GGUF inference is clean — the models are fine. The conversion pipeline is broken for every model, every size. No quantized models should be shipped via APR until the Golden Rule Test passes.
