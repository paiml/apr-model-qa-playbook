# Oracle Types

Oracles verify output correctness.

## Arithmetic Oracle

Verifies mathematical computations.

```yaml
- type: arithmetic
  config:
    tolerance: 0.01  # Float comparison tolerance
```

**Input:** `"2+2="`
**Expected:** Output contains `4`

Supports: `+`, `-`, `*`, `/`, parentheses

## Garbage Oracle

Detects low-quality output. Also serves as the primary **end-to-end kernel
correctness detector** — when SIMD/CUDA kernels in trueno produce incorrect
results, the effect manifests as garbage output that this oracle catches.

```yaml
- type: garbage
  config:
    max_repetition_ratio: 0.3   # Max % of repeated tokens
    min_unique_chars: 10        # Minimum unique characters
```

**Garbage indicators:**
- Excessive repetition (`"the the the the"`)
- Low unique character count
- Non-printable bytes
- Very low entropy

**Kernel-related garbage patterns:**

| Pattern | Kernel Root Cause |
|---------|-------------------|
| Non-ASCII gibberish | LAYOUT-002: column-major data in row-major kernel |
| Repetitive token loops | Softmax collapse from incorrect attention kernel |
| NaN/Inf strings | Numerical instability in quantized matmul |
| U+FFFD replacement chars | Dtype mismatch in kernel output |

## CodeSyntax Oracle

Validates generated code syntax.

```yaml
- type: code_syntax
  config:
    languages:
      - python
      - rust
      - javascript
      - go
      - java
```

Uses language-specific parsers to check syntax validity.

## Response Oracle

Checks response relevance and coherence.

```yaml
- type: response
  config:
    min_relevance: 0.3    # Minimum relevance score
    check_coherence: true
```

Measures:
- Keyword overlap with prompt
- Response structure
- Length appropriateness

## Composite Oracle

Combines multiple oracles with AND/OR logic.

```yaml
- type: composite
  config:
    mode: all  # all | any
    oracles:
      - type: garbage
        config: { ... }
      - type: response
        config: { ... }
```

## Oracle Results

```rust
pub enum OracleResult {
    Pass,
    Fail { reason: String },
    Skip { reason: String },
}
```
