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
    - safetensors       # Ground truth (HuggingFace source)
    - apr               # APR native optimized format
    - gguf              # Third-party format (NOT ground truth)
  quantizations:
    - f16               # Full precision (16-bit float)
    - q8_0              # 8-bit quantization
    - q6_k              # 6-bit K-quant
    - q5_k_m            # 5-bit K-quant (medium)
    - q4_k_m            # 4-bit K-quant (medium)
  size_category: small  # tiny, small, medium, large, xlarge, huge
```

> **Ground Truth**: SafeTensors is always listed first because it is the source of truth for model weights. APR and GGUF formats are derived from SafeTensors and tested for parity.

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

## Ollama Parity Testing (GH-6/AC-2)

Cross-runtime validation against Ollama:

```yaml
ollama_parity:
  enabled: true
  model_tag: "qwen2.5-coder:7b-instruct-q4_k_m"
  quantizations:
    - q4_k_m
    - q6_k
    - q8_0
  prompts:
    - "What is 2+2?"
    - "def fibonacci(n):"
    - "Explain the difference between a stack and a queue."
  temperature: 0.0
  min_perf_ratio: 0.8
  gates: ["F-OLLAMA-001", "F-OLLAMA-002"]
```

| Field | Default | Description |
|-------|---------|-------------|
| `enabled` | `true` | Enable/disable ollama parity tests |
| `model_tag` | auto | Ollama model tag (default: `{model}:latest`) |
| `quantizations` | `[q4_k_m]` | Quantizations to test |
| `prompts` | `["What is 2+2?"]` | Prompts to compare |
| `temperature` | `0.0` | Sampling temperature (0.0 = deterministic) |
| `min_perf_ratio` | `0.8` | Minimum APR/ollama throughput ratio |
| `gates` | `[]` | Gate IDs for evidence tracking |

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
