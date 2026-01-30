# GH-190: GGUF→APR Conversion Produces Garbage Output (P0 CRITICAL)

**Priority:** P0 (STOP THE LINE)
**Component:** apr-cli convert, aprender/rosetta
**Affects:** All GGUF→APR conversions
**Reporter:** apr-qa-runner automated testing
**Date:** 2026-01-30

---

## Summary

Converting a valid GGUF model to APR format produces a file that passes all 10 validation stages (`apr check`) but generates complete garbage during inference. This is a **silent data corruption bug** - the most dangerous category.

## Severity Justification

- **P0 CRITICAL**: Model appears valid but produces unusable output
- **Silent failure**: `apr check` reports 10/10 PASS despite corruption
- **Data corruption**: Tensor name mapping breaks inference pipeline
- **User impact**: Any model converted GGUF→APR will be unusable

---

## Reproduction Steps

### 1. Convert GGUF to APR

```bash
apr convert ~/.apr/cache/hf/Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf \
    -o /tmp/test-conv.apr --verbose
```

**Output:**
```
=== APR Convert ===
Input:  /home/noah/.apr/cache/hf/Qwen/...
Output: /tmp/test-conv.apr
Converting...
[PMAT-113] Extracted tokenizer with 151936 vocabulary tokens
[PMAT-113] Mapping 339 GGUF tensor names to HuggingFace format...
[PMAT-171] Embedding 151387 BPE merge rules into APR metadata

=== Conversion Report ===
Original size:  6.62 GiB
Converted size: 6.62 GiB
Tensors:        339
⚠ Conversion completed (output larger than input)
```

### 2. Validate (FALSELY PASSES)

```bash
apr check /tmp/test-conv.apr
```

**Output:**
```
┌─────┬─────────────────────┬──────────────────────────────────────┬──────┐
│  #  │      Component      │               Details                │ Pass │
├─────┼─────────────────────┼──────────────────────────────────────┼──────┤
│ 1   │ Tokenizer           │ tokens=[1, 2]                        │ ✅    │
│ 2   │ Embedding           │ Found embedding tensor               │ ✅    │
│ 3   │ Positional Encoding │ RoPE computed inline                 │ ✅    │
│ 4   │ Q/K/V Projection    │ Q/K/V found                          │ ✅    │
│ 5   │ Attention Scores    │ Attention output found               │ ✅    │
│ 6   │ Feed-Forward (MLP)  │ MLP found                            │ ✅    │
│ 7   │ Layer Norm          │ 28 layers                            │ ✅    │
│ 8   │ LM Head             │ vocab_size=151936                    │ ✅    │
│ 9   │ Logits → Probs      │ logits[151936]                       │ ✅    │
│ 10  │ Sampler/Decode      │ softmax sum = 0.999989               │ ✅    │
└─────┴─────────────────────┴──────────────────────────────────────┴──────┘

✅ 10/10 STAGES PASSED. MODEL PROVEN CORRECT.   <-- FALSE POSITIVE!
```

### 3. Run Inference (PRODUCES GARBAGE)

**Original GGUF (correct):**
```bash
apr run ~/.apr/cache/hf/Qwen/.../qwen2.5-coder-1.5b-instruct-q4_k_m.gguf \
    -p "def fibonacci(n):" --max-tokens 30
```
```
Output:
Certainly! The Fibonacci sequence is a series of numbers where each
number is the sum of the two preceding ones, usually starting with 0 and 1
```

**Converted APR (GARBAGE):**
```bash
apr run /tmp/test-conv.apr -p "def fibonacci(n):" --max-tokens 30
```
```
Output:
ankan billigprov suceorque enorme(ARG kristPropertyParamsylvenate(IOÐ²ÑĢÐµÐ¼ÐµÐ½Ð½Ð¾âĢ¦)
lyrinkdependencetentiber sÃ¡badoâĢľ
iberREMOTE seÃ±orserter.',č
etsy Boxing');?>"
```

---

## Root Cause Analysis

### Tensor Name Mapping Mismatch

`apr rosetta diff-tensors` reveals the issue:

```
╔══════════════════════════════════════════════════════════════════════════════╗
║               TENSOR DIFF REPORT (GH-188: Layout Mismatch Detection)         ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ ✓ Tensor Count: A=339   B=339                                                ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ − 0.down_proj.weight                                                         ║
║   A: [8960, 1536] (missing in B)                                             ║
╠──────────────────────────────────────────────────────────────────────────────╣
║ + 0.mlp.down_proj.weight                                                     ║
║   B: [8960, 1536] (missing in A)                                             ║
╠──────────────────────────────────────────────────────────────────────────────╣
║ − 0.q_proj.weight                                                            ║
║   A: [1536, 1536] (missing in B)                                             ║
╠──────────────────────────────────────────────────────────────────────────────╣
║ + 0.self_attn.q_proj.weight                                                  ║
║   B: [1536, 1536] (missing in A)                                             ║
```

**The bug:**
- GGUF uses **short tensor names**: `0.down_proj.weight`, `0.q_proj.weight`
- APR converter outputs **HuggingFace names**: `0.mlp.down_proj.weight`, `0.self_attn.q_proj.weight`
- APR loader expects **short names** matching GGUF convention
- Result: Tensors not found → random memory → garbage output

### Name Mapping Table (Layer 0)

| GGUF Name (Expected) | APR Name (Actual) | Match |
|---------------------|-------------------|-------|
| `0.down_proj.weight` | `0.mlp.down_proj.weight` | ❌ |
| `0.gate_proj.weight` | `0.mlp.gate_proj.weight` | ❌ |
| `0.up_proj.weight` | `0.mlp.up_proj.weight` | ❌ |
| `0.q_proj.weight` | `0.self_attn.q_proj.weight` | ❌ |
| `0.k_proj.weight` | `0.self_attn.k_proj.weight` | ❌ |
| `0.v_proj.weight` | `0.self_attn.v_proj.weight` | ❌ |
| `0.o_proj.weight` | `0.self_attn.o_proj.weight` | ❌ |
| `0.input_layernorm.weight` | `0.input_layernorm.weight` | ✅ |
| `0.post_attention_layernorm.weight` | `0.post_attention_layernorm.weight` | ✅ |

**Pattern:** Only LayerNorm tensors have matching names.

---

## Rosetta Compare-Inference Output

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                     INFERENCE COMPARISON REPORT (PMAT-114)                   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ Model A: qwen2.5-coder-1.5b-instruct-q4_k_m.gguf                             ║
║ Model B: /tmp/test-conv.apr                                                  ║
║ Prompt: "What is 2+2?"                                                       ║
╠══════════════════════════════════════════════════════════════════════════════╣

=== Generated Text ===
Model A: "2+2 equals 4."
Model B: "×©×Ļ×¨×ķ×ªØ¯Ø§ÙĪÙĦ/software enumDigitsìºĶmemcmp Auckland cÃ³ Filip"

⚠️  TEXT CONTENT DIFFERS:
   Models produced different outputs (may be precision-related)
```

---

## Tensor Fingerprint Analysis

`apr rosetta fingerprint /tmp/test-conv.apr` shows:
- **nan=0, inf=0** for all tensors (no NaN/Inf corruption)
- Statistical properties appear normal
- Checksums are deterministic

**Conclusion:** Tensor *data* is correct; tensor *naming* is wrong.

---

## Why `apr check` Falsely Passes

The validation stages check:
1. Tokenizer exists → ✅ (embedded correctly)
2. Embedding tensor found → ✅ (uses `model.embed_tokens.weight`)
3. Q/K/V found → ✅ (found by pattern matching, not exact name)
4. LM Head found → ✅ (uses `lm_head.weight`)
5. Forward pass → ✅ (runs on whatever tensors it finds)

**Bug:** Validation uses relaxed pattern matching that finds *some* tensors, but inference uses exact name lookup that fails.

---

## Affected Code Paths

### 1. Converter (aprender/src/rosetta/convert.rs)

```rust
// Line ~450: HuggingFace name mapping
fn map_gguf_to_hf_name(gguf_name: &str) -> String {
    // BUG: Maps to full HF names like "model.layers.0.self_attn.q_proj.weight"
    // Should preserve GGUF-style names for APR format compatibility
}
```

### 2. Loader (realizar/src/apr/mod.rs)

```rust
// Line ~320: Tensor lookup
fn get_tensor(&self, name: &str) -> Option<&Tensor> {
    // Expects GGUF-style names: "0.q_proj.weight"
    // APR file contains HF names: "model.layers.0.self_attn.q_proj.weight"
    self.tensors.get(name)  // Returns None → uses uninitialized memory
}
```

---

## Proposed Fix

### Option A: Fix Converter (Recommended)

Preserve GGUF tensor naming convention in APR output:

```rust
fn map_tensor_name_for_apr(hf_name: &str) -> String {
    // "model.layers.0.self_attn.q_proj.weight" → "0.q_proj.weight"
    let re = Regex::new(r"model\.layers\.(\d+)\.self_attn\.(.+)").unwrap();
    if let Some(caps) = re.captures(hf_name) {
        return format!("{}.{}", &caps[1], &caps[2]);
    }
    // Similar for mlp: "model.layers.0.mlp.down_proj.weight" → "0.down_proj.weight"
    let re = Regex::new(r"model\.layers\.(\d+)\.mlp\.(.+)").unwrap();
    if let Some(caps) = re.captures(hf_name) {
        return format!("{}.{}", &caps[1], &caps[2]);
    }
    hf_name.to_string()
}
```

### Option B: Fix Loader (Alternative)

Add name normalization in loader:

```rust
fn normalize_tensor_name(name: &str) -> String {
    // Try both conventions
    if let Some(tensor) = self.tensors.get(name) {
        return tensor;
    }
    // Try HF-style lookup
    let hf_name = convert_to_hf_name(name);
    self.tensors.get(&hf_name)
}
```

### Option C: Add Tensor Name Alias Table

Store name mappings in APR header:

```
[aliases]
0.q_proj.weight = model.layers.0.self_attn.q_proj.weight
0.k_proj.weight = model.layers.0.self_attn.k_proj.weight
...
```

---

## Verification Gates

After fix, these gates MUST pass:

| Gate ID | Assertion |
|---------|-----------|
| F-ROSETTA-DIFF-001 | Tensor shapes match after conversion |
| F-ROSETTA-DIFF-002 | Tensor names accessible by both conventions |
| F-ROSETTA-INF-001 | Token-by-token argmax match |
| F-ROSETTA-INF-002 | Logit diff < 1e-5 |
| F-CONV-G-A | GGUF→APR produces identical inference output |
| F-CONV-RT-001 | Round-trip GGUF→APR→GGUF preserves output |

---

## Test Case

```rust
#[test]
fn test_gguf_to_apr_preserves_tensor_names() {
    let gguf = load_gguf("qwen2.5-coder-1.5b-instruct-q4_k_m.gguf");
    let apr = convert_to_apr(&gguf);

    // All GGUF tensor names must be accessible in APR
    for name in gguf.tensor_names() {
        assert!(apr.has_tensor(name),
            "Tensor '{}' not found in APR output", name);
    }

    // Inference must match
    let gguf_output = gguf.run("What is 2+2?");
    let apr_output = apr.run("What is 2+2?");
    assert_eq!(gguf_output, apr_output);
}
```

---

## Related Issues

- **GH-186**: APR dtype mapping bug (FIXED) - different root cause
- **GH-185**: SafeTensors embedding conversion errors
- **GH-188**: Tensor layout mismatch detection (Rosetta diff-tensors)

---

## Environment

```
apr version: 0.2.12
OS: Linux 6.8.0-90-generic
Model: Qwen/Qwen2.5-Coder-1.5B-Instruct (Q4_K_M)
```

---

## Attachments

1. Full tensor diff output: `/tmp/qwen-quick/evidence.json`
2. Conversion log: verbose output above
3. Fingerprint data: `apr rosetta fingerprint` output

---

## Systemic Analysis: Five Whys (Why This Keeps Happening)

This is the **fourth P0 conversion bug in one sprint** (GH-185, GH-186, GH-189, GH-190). All four are silent failures at format boundaries. A Five Whys analysis reveals they share a single systemic root cause.

### The Pattern: Same Bug, Four Masks

| Bug | Writer Assumes | Reader Assumes | Result |
|-----|---------------|----------------|--------|
| GH-186 | dtype 12 = Q4K | dtype 12 = F32 (fallback) | NaN logits, PAD flood |
| GH-189 | LayerNorm written as-is | LayerNorm must be non-zero | Zeroed weights, garbage |
| GH-190 | HuggingFace tensor names | GGUF tensor names | Names not found, garbage |
| GH-185 | Tokenizer in metadata | Tokenizer in alt path | Missing tokenizer |

### Five Whys

```
WHY 1: Writer and reader disagree on conventions (names, dtypes, layout)
  ↓
WHY 2: No integration test converts AND runs inference on the result
  ↓
WHY 3: Writer (aprender) and loader (realizar) are separate repos with separate CI
  ↓
WHY 4: We treat conversion as a serialization problem, not a behavioral contract
  ↓
WHY 5: We optimized for fast CI (structural checks) over correctness (semantic checks)
```

### Why 220 Gates Didn't Catch It

Our 220 gates are **reactive** (detect known bug symptoms) not **proactive** (prevent unknown bugs). We're playing whack-a-mole:

```
Bug found → Write gate → Add to spec → Next bug found → Write gate → ...
```

`apr check` validates *syntax* (can we parse it?) not *semantics* (does it produce correct output?). This is why it reports 10/10 PASS on a model that outputs garbage.

### The Golden Rule Test (Would Have Caught All 4 Bugs)

One 20-second test prevents the entire class of bugs:

```bash
# THE GOLDEN RULE: convert → inference → diff
apr run original.gguf -p "What is 2+2?" --max-tokens 10 > expected.txt
apr convert original.gguf -o converted.apr
apr run converted.apr -p "What is 2+2?" --max-tokens 10 > actual.txt
diff expected.txt actual.txt   # MUST be identical
```

This test encodes the **only invariant that matters**: converted models must produce the same output as the original.

### Required Systemic Fixes

| # | Fix | Why It Matters |
|---|-----|---------------|
| 1 | **Golden Rule Test in CI** | Catches ALL conversion bugs, not just known ones |
| 2 | **Shared contract crate** between aprender/realizar | Single source of truth for tensor names, dtypes |
| 3 | **Kill silent fallbacks** | `_ => F32` defaults are the #1 cause (GH-186 pattern) |
| 4 | **Behavioral Stage 11 in `apr check`** | Validation must test inference, not just structure |
| 5 | **Canonical tensor naming in APR v2 spec** | No more independent naming assumptions |

### 5 Invariants > 220 Symptom Gates

| Invariant | What It Tests | Catches |
|-----------|--------------|---------|
| **I-1**: Round-trip identity | convert(model) produces same inference | ALL conversion bugs |
| **I-2**: Tensor name bijection | writer names == loader names | GH-190 |
| **I-3**: No silent fallbacks | unknown input → error, never default | GH-186 |
| **I-4**: Statistical preservation | tensor stats unchanged after conversion | GH-189 |
| **I-5**: Tokenizer roundtrip | encode(decode(tokens)) == tokens | GH-185 |

### The Bottom Line

> *"We don't have a testing problem. We have an architecture problem. The conversion pipeline has no single source of truth for tensor naming, dtype mapping, or behavioral correctness. Until writer and reader share a contract, every fix creates the conditions for the next bug."*

Full analysis: [docs/five-whys/GH-190-systemic-conversion-failures.md](../five-whys/GH-190-systemic-conversion-failures.md)

---

**Toyota Way Principle:** STOP THE LINE. Do not ship models converted with current GGUF→APR pipeline until this is fixed.
