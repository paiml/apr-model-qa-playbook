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

Detects low-quality output.

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
