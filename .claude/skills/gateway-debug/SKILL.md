---
name: gateway-debug
description: "Diagnoses and fixes G0-G4 gateway failures in model qualification runs. Interprets MQS zero-scores, evidence JSON, and stderr output. Covers LAYOUT-002 violations, config.json mismatches, garbage detection, crash analysis, and all 5 contract invariants (I-1 through I-5). Includes the GarbageOracle detection patterns and conversion diagnostics."
disable-model-invocation: false
user-invocable: true
allowed-tools: "Read, Grep, Glob, Bash"
argument-hint: "gateway: G0, G1, G2, G3, G4, or symptom: garbage output, zero score, crash, timeout, layout violation"
---

# Gateway Debug

This skill diagnoses failures in the G0-G4 gateway system. Any single gateway failure zeroes the entire MQS score to 0.

## Quick Diagnosis

### "My MQS score is 0"

A zero score means at least one gateway failed. Check which:

```bash
# Score the evidence and see gateway results
cargo run --bin apr-qa -- score certifications/my-model/evidence.json

# Or search evidence for failures
grep -i '"Falsified"\|"Crashed"\|"Timeout"' certifications/my-model/evidence.json
```

### Decision Tree

```
MQS = 0?
├── Check G0: Does config.json match tensor metadata?
│   └── Evidence gate_id starts with "G0-"
├── Check G1: Did the model load?
│   └── Evidence gate_id contains "G1"
├── Check G2: Did inference produce output?
│   └── Evidence gate_id contains "G2"
├── Check G3: Any crashes?
│   └── Evidence with outcome "Crashed"
└── Check G4: Is output garbage?
    └── Evidence gate_id contains "G4"
```

## Gateway Reference

| Gate | ID Pattern | Name | Severity | Zero-out Rule |
|------|-----------|------|----------|---------------|
| **G0** | `G0-INTEGRITY-*`, `G0-PULL-*`, `G0-FORMAT-*`, `G0-VALIDATE-*`, `G0-TENSOR-*`, `G0-LAYOUT-*` | Model Integrity | P0 | Any `G0-*` failure |
| **G1** | `G1-*` | Load Success | P0 | Any `G1-*` failure |
| **G2** | `G2-*` | Basic Inference | P0 | Any `G2-*` failure |
| **G3** | (implicit) | Stability | P0 | Any evidence with `outcome == Crashed` |
| **G4** | `G4-GARBAGE-*` | Quality | P0 | >25% of `G4-*` evidence fails |

## G0: Model Integrity

G0 runs before any inference. It validates the model's structural correctness.

### G0 Sub-gates (Execution Order)

| Sub-gate | ID | What It Checks |
|----------|------|----------------|
| G0-PULL | `G0-PULL-001` | Model acquisition from HuggingFace |
| G0-FORMAT | `G0-FORMAT-*` | Format file validity |
| G0-VALIDATE | `G0-VALIDATE-*` | Physics validation (NaN, Inf, all-zeros tensors) |
| G0-TENSOR | `G0-TENSOR-*` | Tensor shape/count matches template |
| G0-INTEGRITY | `G0-INTEGRITY-*` | config.json matches tensor metadata |
| G0-LAYOUT | `G0-LAYOUT-*` | Tensor layout is row-major |

### G0 Failure: "config.json mismatch"

**Symptom:** `G0-INTEGRITY` failure, evidence reason mentions config/tensor mismatch.

**Root Causes:**
1. **Corrupted config.json** - `num_hidden_layers`, `hidden_size` don't match actual tensors
2. **Wrong model version** - config.json from different checkpoint
3. **Incomplete download** - Partial model file

**Diagnosis:**
```bash
# Inspect model metadata
cargo run --bin apr-qa -- validate-contract /path/to/model \
    --format json

# Check config.json fields
apr inspect /path/to/model --json | jq '.tensor_count, .hidden_size'
```

**Fix:**
- Re-download model: `apr pull Org/Model`
- Verify config.json fields match tensor dimensions

### G0 Failure: "Physics validation failed" (G0-VALIDATE)

**Symptom:** Evidence reason mentions NaN, Inf, or all-zeros tensors.

**Root Causes:**
1. **Corrupted weights** - Download interrupted or storage corruption
2. **Bad quantization** - Quantization produced degenerate values
3. **Wrong dtype** - Silent fallback to wrong precision

**Diagnosis:**
```bash
# Strict physics validation
apr rosetta validate-model /path/to/model --strict

# Check tensor statistics
apr rosetta fingerprint-model /path/to/model --json
```

### G0 Failure: "Tensor template mismatch" (G0-TENSOR)

**Symptom:** Expected N tensors, found M. Or tensor names don't match pattern.

**Root Cause:** Model architecture doesn't match the family contract's tensor template.

**Diagnosis:**
```bash
# Inspect tensor names
apr inspect /path/to/model --json | jq '.tensor_names[]'

# Compare against contract
cargo run --bin apr-qa -- validate-contract /path/to/model
```

### G0 Failure: LAYOUT-002 Violation (G0-LAYOUT)

**Symptom:** Layout check failed, tensors not in row-major order.

**This is critical.** The entire Sovereign AI Stack uses row-major layout. GGUF files are column-major and MUST be converted before use.

**Diagnosis:**
```bash
# Run full 5-invariant conversion diagnostic
./scripts/diagnose-conversion.sh /path/to/model.gguf
```

**Fix:** Convert GGUF to APR format:
```bash
apr import model.gguf -o model.apr
```

## G1: Load Success

### G1 Failure: "Model failed to load"

**Symptom:** Evidence with gate_id containing `G1`, outcome `Falsified` or `Crashed`.

**Common Root Causes:**

| Cause | Stderr Pattern | Fix |
|-------|---------------|-----|
| File not found | `No such file or directory` | Check `--model-cache` path |
| OOM | `out of memory`, `CUDA OOM` | Use `--no-gpu` or smaller quant |
| Bad format | `invalid magic`, `unexpected EOF` | Re-download, check file integrity |
| Missing dependency | `libcuda.so not found` | Install CUDA toolkit |
| Wrong quantization | `unsupported quantization` | Use supported quant (q4_k_m, q5_k_m, etc.) |

**Diagnosis:**
```bash
# Check if file exists and has reasonable size
ls -lh /path/to/model.*

# Try loading manually
apr run /path/to/model -p "test" --max-tokens 1
```

## G2: Basic Inference

### G2 Failure: "Inference produced no output"

**Symptom:** Evidence with gate_id containing `G2`, outcome `Falsified` with empty output, or `Timeout`.

**Common Root Causes:**

| Cause | Evidence Pattern | Fix |
|-------|-----------------|-----|
| Timeout | `outcome: Timeout` | Increase `--timeout` |
| Empty output | `output: ""` | Check model compatibility |
| Crash during inference | `outcome: Crashed` | See G3 section |
| GPU memory exhaustion | stderr contains `CUDA` | Use `--no-gpu` |

**Diagnosis:**
```bash
# Run with extended timeout
cargo run --bin apr-qa -- run playbook.yaml --timeout 120000

# Run with fail-fast for enhanced diagnostics
cargo run --bin apr-qa -- run playbook.yaml --fail-fast
```

## G3: Stability (No Crashes)

### G3 Failure: "Process crashed"

**Symptom:** Evidence with `outcome: Crashed`, includes `exit_code` and `stderr`.

**G3 is implicit** - any crashed evidence anywhere triggers G3 failure.

**Common Root Causes:**

| Exit Code | Signal | Likely Cause |
|-----------|--------|-------------|
| 139 | SIGSEGV | Segfault - often LAYOUT-002 violation |
| 134 | SIGABRT | Assertion failure in native code |
| 137 | SIGKILL | OOM killer or timeout |
| 1 | - | General error |

**Diagnosis by exit code:**

**Exit 139 (Segfault):**
```bash
# Almost always a layout violation
# Check if GGUF was used without conversion
./scripts/diagnose-conversion.sh /path/to/model.gguf

# Verify tensor layout
cargo run --bin apr-qa -- validate-contract /path/to/model
```

**Exit 137 (OOM Kill):**
```bash
# Check available memory
free -h

# Check GPU memory
nvidia-smi

# Re-run with CPU only
cargo run --bin apr-qa -- run playbook.yaml --no-gpu
```

**Exit 134 (Abort):**
```bash
# Run with tracing to get detailed stack
cargo run --bin apr-qa -- run playbook.yaml --fail-fast

# Check stderr in evidence
grep -A5 '"Crashed"' evidence.json
```

## G4: Quality (Garbage Detection)

### G4 Failure: "Output is garbage"

**Symptom:** Evidence with gate_id containing `G4`, outcome `Falsified`.

G4 uses the `GarbageOracle` which checks 6 conditions. **Any single check failing = garbage.**

### GarbageOracle Detection Patterns

| Check | What It Detects | Example Garbage |
|-------|----------------|-----------------|
| **Empty output** | No response at all | `""` |
| **Control characters** | Binary data leak (except `\n`, `\t`, `\r`) | `\x00\x01\x02` |
| **NaN/Inf** | Numerical instability | `"NaN"`, `"Inf"`, `"inf"` |
| **Replacement char** | Encoding corruption | `U+FFFD` (&#xFFFD;) |
| **Char-level repetition** | Degenerate looping (period 2-20, >= 3 reps, >= 70% coverage) | `VILLEVILLEVILLEVILLE` |
| **Word-level repetition** | Token-level looping | `foo foo foo foo foo foo` |
| **Whitespace-only** | Empty semantic content | `"   \n\n  "` |

### Interpreting G4 Threshold

G4 does NOT fail on a single garbage output. It fails when **>25% of G4-tagged evidence is garbage**.

This means:
- 1 garbage output in 18 tests: G4 passes (5.6%)
- 5 garbage outputs in 18 tests: G4 fails (27.8%)

### G4 Failure: LAYOUT-002 Root Cause

The most common G4 failure is a **LAYOUT-002 violation** - running GGUF through row-major kernels without conversion.

**Signature garbage pattern:** Non-ASCII gibberish with mixed scripts, random punctuation

**Example:** `olumbia+lsi nunca/localENTS Arbeit\xc3\xa9t`

**Fix:**
```bash
# Convert to APR format first
apr import model.gguf -o model.apr

# Re-run qualification on APR
cargo run --bin apr-qa -- run playbook.yaml --model-path /path/to/model.apr
```

### G4 Failure: Other Root Causes

| Pattern | Root Cause | Fix |
|---------|-----------|-----|
| Empty output | Model not generating | Check G2 first |
| Repetitive tokens | Temperature 0 + weak model | Expected for small models |
| NaN in output | Numerical instability | Check quantization, try different quant |
| Replacement chars | Bad tokenizer or encoding | Re-download model |

## Oracle System

Oracles evaluate output correctness. The oracle is selected based on prompt content.

### Oracle Selection Logic

```
Prompt contains digit AND operator (+, -, *, /)
  → ArithmeticOracle

Prompt starts with "def ", "fn ", "function ", "class ", "async "
  OR contains "```"
  → CodeSyntaxOracle

Everything else
  → GarbageOracle
```

### ArithmeticOracle

**Checks:** Does the output contain the correct arithmetic result?

**Example:** Prompt `"What is 7*8?"` → expects `"56"` in output

**Tolerance:** Exact string match (the number must appear as substring)

**Common false negatives:** Model outputs the answer in words ("fifty-six") instead of digits

### CodeSyntaxOracle

**Checks:** (1) Not garbage (runs GarbageOracle first), (2) Contains code indicators (`fn `, `def `, `{`, `}`, `=>`, `->`, `return `, etc.)

**Lenient:** If output is short (< 20 chars) or contains any code indicator, it passes.

### GarbageOracle

Default fallback. See the detection patterns table above.

## Contract Invariants (I-1 through I-5)

Run all 5 invariants at once to prevent "onion peeling" (fixing bug N reveals bug N+1):

```bash
./scripts/diagnose-conversion.sh /path/to/model.gguf
```

| Invariant | Gate ID | Test | Catches |
|-----------|---------|------|---------|
| **I-1** | `F-CONTRACT-I1-001` | `inference(convert(model)) == inference(model)` | Output divergence after conversion |
| **I-2** | `F-CONTRACT-I2-001` | Writer tensor names == reader tensor names | Missing/extra tensors |
| **I-3** | `F-CONTRACT-I3-001` | Unknown dtype → error, not silent F32 | Silent precision loss |
| **I-4** | `F-CONTRACT-I4-001` | Tensor statistics within tolerance | Statistical drift |
| **I-5** | `F-CONTRACT-I5-001` | First token identical between formats | Tokenizer mismatch |

### Tolerance Table (I-4)

| Dtype | Absolute Tolerance | Relative Tolerance |
|-------|-------------------|-------------------|
| F32 | 0.0 | 0.0 |
| F16 | 0.001 | 0.001 |
| BF16 | 0.005 | 0.005 |
| Q8_0 | 0.01 | 0.01 |
| Q6_K | 0.02 | 0.02 |
| Q5_K | 0.03 | 0.03 |
| Q4_K | 0.05 | 0.05 |
| Q3_K | 0.08 | 0.08 |
| Q2_K | 0.15 | 0.15 |

## Evidence JSON Structure

Each test produces an `Evidence` record:

```json
{
  "id": "uuid-v4",
  "gate_id": "F-QUAL-001",
  "scenario": {
    "model_id": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    "modality": "run",
    "backend": "cpu",
    "format": "safetensors",
    "prompt": "What is 2+2?",
    "seed": 42
  },
  "outcome": "Corroborated",
  "reason": "Test passed",
  "output": "The answer is 4.",
  "stderr": null,
  "exit_code": 0,
  "metrics": {
    "duration_ms": 1500,
    "tokens_per_second": 17.3,
    "time_to_first_token_ms": 200,
    "total_tokens": 26,
    "memory_peak_mb": 1024
  },
  "timestamp": "2026-02-14T10:30:00Z",
  "host": { "hostname": "...", "cpu": "...", "gpu": "..." }
}
```

**Outcome Values:**
- `Corroborated` - Hypothesis survived refutation (pass)
- `Falsified` - Hypothesis refuted, includes `reason` (fail)
- `Timeout` - Exceeded time limit
- `Crashed` - Process crashed, includes `exit_code` and `stderr`
- `Skipped` - Skipped due to stop-on-failure policy

See [evidence-schema.md](references/evidence-schema.md) for the full schema reference.

## Error Types

The runner produces typed errors:

| Error | When | What to Check |
|-------|------|---------------|
| `PlaybookParseError` | Invalid YAML | Validate against `playbook.schema.yaml` |
| `CommandFailed` | Non-zero exit from subprocess | Check `stderr` field |
| `Timeout` | Test exceeded time limit | Increase `--timeout` |
| `GatewayFailed` | Gateway check failed | See gateway sections above |
| `Execution` | General validation error | Read error message |
| `Validation` | Schema/contract violation | Run `validate-contract` |

## Debugging Workflow

### 1. Start with the Score

```bash
cargo run --bin apr-qa -- score evidence.json
```

Look at: gateways (which failed?), category breakdown (where are points lost?), penalties (crashes? timeouts?)

### 2. Filter to Failures

```bash
# Find all failures in evidence
grep -B2 -A10 '"Falsified"\|"Crashed"\|"Timeout"' evidence.json
```

### 3. Identify the Pattern

- All formats fail? → Model-level issue (G1/G2)
- Only GGUF fails? → LAYOUT-002 violation
- Only GPU fails? → CUDA/driver issue
- Only serve fails? → HTTP endpoint issue
- Random failures? → Stability issue (G3)

### 4. Run Targeted Diagnosis

```bash
# For conversion issues
./scripts/diagnose-conversion.sh /path/to/model.gguf

# For contract violations
cargo run --bin apr-qa -- validate-contract /path/to/model

# For parity issues
cargo run --bin apr-qa -- parity --model-family qwen2.5-coder-1.5b --self-check
```

### 5. Re-run with Enhanced Diagnostics

```bash
cargo run --bin apr-qa -- run playbook.yaml --fail-fast
```

## See Also

- [evidence-schema.md](references/evidence-schema.md) - Full evidence JSON schema reference
- `scripts/diagnose-conversion.sh` - 5-invariant conversion tester
- `scripts/validate-schemas.sh` - Evidence schema validation
- `playbooks/spec/` - Gateway specification playbooks
