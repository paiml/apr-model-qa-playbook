# Gateway Definitions

Gateways are critical checks. Any failure zeros the MQS.

## G0: Integrity

**ID:** `G0-INTEGRITY`
**Severity:** P0

The model's config.json must match actual tensor metadata:
- config.json exists and is valid JSON
- `num_hidden_layers` matches actual tensor count
- `hidden_size` matches tensor dimensions
- Architecture field matches tensor naming conventions

```yaml
# Passes
config.json.hidden_size == tensor_shape[hidden_dim]

# Fails
config.json.hidden_size: 2048
actual_tensor_hidden_dim: 1024
```

## G1: Model Loads

**ID:** `G1-LOAD`
**Severity:** P0

The model file must load into memory without error:
- File exists and is readable
- Header parsing succeeds
- Tensor metadata is valid
- Memory allocation succeeds

```yaml
# Passes
exit_code: 0

# Fails
exit_code: 1
stderr: "Error: File not found"
```

## G2: Basic Inference

**ID:** `G2-INFER`
**Severity:** P0

The model must produce output for a simple prompt:
- Forward pass executes
- Token generation works
- Output is returned (non-empty)

```yaml
# Passes
output: "4"

# Fails
output: ""
```

## G3: No Crashes

**ID:** `G3-STABLE`
**Severity:** P0

The process must complete without crash:
- No Rust panics
- No segmentation faults (SIGSEGV)
- No illegal instructions (SIGILL)
- No aborts (SIGABRT)

```yaml
# Passes
exit_code: 0

# Fails
exit_code: -11  # SIGSEGV
stderr: "thread 'main' panicked"
```

## G4: Output Quality

**ID:** `G4-VALID`
**Severity:** P0

The output must not be garbage:
- Not random bytes
- No excessive repetition (< 30%)
- Minimum 10 unique characters
- No non-printable bytes

```yaml
# Passes
output: "The answer is 4."

# Fails
output: "aaaaaaaaaaaaaaaaaaaaaaaaaaaa"
```

### G4 as Kernel Correctness Oracle

G4 is the primary end-to-end detector for kernel bugs in the Sovereign AI Stack.
When trueno's SIMD/CUDA kernels (quantized matmul, RMSNorm, attention) produce
incorrect results, the effect propagates through inference and manifests as
garbage output that the `GarbageOracle` catches.

**Common kernel-related G4 failures:**

| Garbage Pattern | Likely Kernel Root Cause |
|-----------------|------------------------|
| Non-ASCII gibberish with mixed scripts | LAYOUT-002 violation (column-major data in row-major kernel) |
| Repetitive token loops | Softmax collapse from incorrect attention kernel output |
| NaN/Inf in output | Numerical instability in quantized matmul |
| Replacement character (U+FFFD) | Encoding corruption from dtype mismatch |

This makes the QA playbook complementary to kernel-level testing: trueno and
HuggingFace kernels validate individual kernel operations, while the playbook
validates that the full kernel pipeline produces correct model output.

## Falsification Gates (F-*)

Beyond gateways, falsification gates test specific behaviors:

| Prefix | Category | Severity |
|--------|----------|----------|
| F-QUAL | Quality | P1 |
| F-PERF | Performance | P1 |
| F-STAB | Stability | P1 |
| F-COMP | Compatibility | P1 |
| F-EDGE | Edge Cases | P2 |
| F-HTTP | HTTP/Serve | P1 |
| F-REGR | Regression | P1 |
| F-CONV | Conversion | P0 |
| F-INSPECT | Inspect | P1 |

## Conversion Gates (F-CONV-*)

Format conversion is the most critical subsystem. A single conversion bug
invalidates all downstream inference. These gates implement metamorphic
relations from the Rosetta-Testing spec (PMAT-ROSETTA-002/003).

### Round-Trip Gates

| Gate ID | Spec | Chain | Description |
|---------|------|-------|-------------|
| F-CONV-RT-001 | MR-RT | GGUF→APR→ST→GGUF | Original round-trip chain |
| F-CONV-RT-002 | T-QKV-03 | ST→APR→GGUF→ST | Reverse direction round-trip |
| F-CONV-RT-003 | T-QKV-04 | ST→APR→GGUF→APR→ST | Multi-hop (4 conversions) |
| F-CONV-RT-004 | GH-6/AC-3 | ST→APR→GGUF→APR | Round-trip back to APR (3 conversions) |
| F-CONV-RT-BYTE-001 | GH-6/AC-3 | ST→APR vs ST→APR→GGUF→APR | Byte-level tensor diff after round-trip |

Round-trip gates convert a model through a chain of formats and verify
the final output matches the original. Any divergence indicates data loss
or corruption during conversion.

F-CONV-RT-BYTE-001 is a stricter variant that performs byte-level tensor
comparison between a direct ST→APR conversion and a round-trip ST→APR→GGUF→APR
conversion, catching subtle numerical drift that inference comparison might miss.

### Metamorphic Relation Gates

| Gate ID | Relation | Description |
|---------|----------|-------------|
| F-CONV-CARD-001 | MR-CARD | `tensor_count(out) >= tensor_count(in)` — catches silent tensor fusion (e.g., QKV fusion 338→227) |
| F-CONV-NAME-001 | T-QKV-02 | Tensor name-set preservation — detects unexpected renames (q\_proj+k\_proj+v\_proj → qkv\_proj) |
| F-CONV-IDEM-001 | MR-IDEM | Idempotency — converting A→B twice from same source must produce identical output |
| F-CONV-COM-001 | MR-COM | Commutativity — GGUF→APR must match GGUF→ST→APR (path independence) |

### Inspect Gate

| Gate ID | Spec | Description |
|---------|------|-------------|
| F-INSPECT-META-001 | T-GH192-01 | Model metadata (num\_heads, hidden\_size, etc.) must be present and non-zero |

### Contract Invariant Gates (GH-190/191)

These gates enforce the shared format contract between writer and reader.
Defined in `apr_format_contract.yaml`, they are the corrective actions
prescribed by the GH-190 and GH-191 Five-Whys analyses.

| Gate ID | Invariant | Description |
|---------|-----------|-------------|
| F-CONTRACT-I1-001 | I-1 Round-trip Identity | `inference(convert(model)) == inference(model)` |
| F-CONTRACT-I2-001 | I-2 Tensor Name Bijection | Writer tensor names must exactly match reader tensor names |
| F-CONTRACT-I3-001 | I-3 No Silent Fallbacks | Unknown dtype/tensor must error, never silently default to F32 |
| F-CONTRACT-I4-001 | I-4 Statistical Preservation | Tensor stats (mean, std, min, max) preserved within dtype tolerance |
| F-CONTRACT-I5-001 | I-5 Tokenizer Roundtrip | `encode(decode(tokens)) == tokens` for the embedded tokenizer |

I-1 runs as the Golden Rule Test (F-GOLDEN-RULE-001). I-2 through I-5
run when `contract_tests` is enabled in the playbook.

### Ollama Parity Gates (GH-6/AC-2)

These gates validate cross-runtime consistency between APR and Ollama.

| Gate ID | Description |
|---------|-------------|
| F-OLLAMA-001 | APR and Ollama both produce output for the same prompt |
| F-OLLAMA-002 | APR throughput is at least `min_perf_ratio` of Ollama throughput |
| F-OLLAMA-003 | TTFT (time-to-first-token) ratio APR/Ollama within 3x threshold |
| F-OLLAMA-004 | Ollama API endpoint `/api/tags` is accessible |
| F-OLLAMA-005 | Ollama loads our GGUF via `ollama create` without errors |

Ollama parity testing ensures that APR inference produces comparable results
to Ollama for the same model and quantization. This catches runtime-specific
bugs that format-level tests cannot detect.

### Performance Gates (F-PERF-*)

| Gate ID | Description |
|---------|-------------|
| F-PERF-003 | GPU throughput >= CPU throughput (ratio >= 1.0x) |
| F-PERF-005 | Memory profiling completes and reports peak RSS |

Performance gates run when `profile_ci` is enabled. F-PERF-003 requires both
`cpu` and `gpu` in the profile backends list.

### Legacy Conversion Gates

| Gate ID | Description |
|---------|-------------|
| F-CONV-EMBED-001 | Embedding tensor transposition bug |
| F-CONV-TOK-001 | Embedded tokenizer missing from APR file |
| F-CONV-WEIGHT-001 | Weight tensor corruption (NaN/Inf/zeros) |
| F-CONV-SHAPE-001 | Tensor shape mismatch with model config |
| F-CONV-SEMANTIC-001 | Semantic drift — structurally valid but wrong output |
| F-GOLDEN-RULE-001 | Golden Rule — converted model must produce identical output |
