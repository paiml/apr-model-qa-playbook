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
