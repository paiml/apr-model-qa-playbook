# Gateway Definitions

Gateways are critical checks. Any failure zeros the MQS.

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

Round-trip gates convert a model through a chain of formats and verify
the final output matches the original. Any divergence indicates data loss
or corruption during conversion.

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

### Legacy Conversion Gates

| Gate ID | Description |
|---------|-------------|
| F-CONV-EMBED-001 | Embedding tensor transposition bug |
| F-CONV-TOK-001 | Embedded tokenizer missing from APR file |
| F-CONV-WEIGHT-001 | Weight tensor corruption (NaN/Inf/zeros) |
| F-CONV-SHAPE-001 | Tensor shape mismatch with model config |
| F-CONV-SEMANTIC-001 | Semantic drift — structurally valid but wrong output |
| F-GOLDEN-RULE-001 | Golden Rule — converted model must produce identical output |
