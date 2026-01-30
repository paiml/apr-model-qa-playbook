# Playbook Format

Playbooks are YAML files that define model qualification tests. They follow the schema in `playbooks/playbook.schema.yaml`.

## Required Fields

```yaml
name: model-name          # Unique identifier (lowercase, hyphens)
version: "1.0.0"          # Semantic version

model:
  hf_repo: "org/model"    # HuggingFace repository ID

test_matrix:
  modalities:             # At least one required
    - run
  backends:               # At least one required
    - cpu
```

## Model Configuration

```yaml
model:
  hf_repo: "Qwen/Qwen2.5-Coder-1.5B-Instruct"
  formats:
    - gguf              # GGUF quantized format
    - safetensors       # HuggingFace native format
    - apr               # APR native format
  quantizations:
    - q4_k_m            # 4-bit quantization
    - q5_k_m            # 5-bit quantization
    - q8_0              # 8-bit quantization
  size_category: small  # tiny, small, medium, large, xlarge, huge
```

## Test Matrix

```yaml
test_matrix:
  modalities:
    - run               # Single inference
    - chat              # Multi-turn conversation
    - serve             # HTTP API server
  backends:
    - cpu               # CPU inference
    - gpu               # GPU inference
  scenario_count: 100   # Tests per combination
  seed: 42              # Reproducibility
  timeout_ms: 60000     # Per-test timeout
```

## Gateway Checks

```yaml
gates:
  g1_model_loads: true      # Model loads without error
  g2_basic_inference: true  # Produces output
  g3_no_crashes: true       # No panics or SIGSEGV
  g4_output_quality: true   # Output is not garbage
```

## Oracles

```yaml
oracles:
  - type: arithmetic
    config:
      tolerance: 0.01

  - type: garbage
    config:
      max_repetition_ratio: 0.3
      min_unique_chars: 10

  - type: code_syntax
    config:
      languages:
        - python
        - rust
        - javascript

  - type: response
    config:
      min_relevance: 0.3
```

## Failure Policy

```yaml
failure_policy: stop_on_p0   # Options: stop_on_first, stop_on_p0, collect_all
```

## Property Tests

```yaml
property_tests:
  enabled: true
  cases: 100              # Number of proptest cases
  max_shrink_iters: 1000  # Shrinking iterations for failures
```

## Templates

Three templates are provided:

| Template | Use Case | Test Count |
|----------|----------|------------|
| `quick-check.yaml` | Fast smoke test | ~10 tests |
| `ci-pipeline.yaml` | CI/CD integration | ~150 tests |
| `full-qualification.yaml` | Production deployment | ~1,800 tests |
