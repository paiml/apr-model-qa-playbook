# Playbook Anatomy

Complete field reference for qualification playbook YAML files. Validated against `playbooks/playbook.schema.yaml`.

## Required Fields

```yaml
name: "model-name-tier"          # Pattern: ^[a-z0-9-]+$, 1-64 chars
version: "1.0.0"                 # Semver: ^\d+\.\d+\.\d+$

model:
  hf_repo: "Org/Model-Name"     # HuggingFace repo ID (required)
  formats:                       # At least one required
    - safetensors                # Ground truth (always include)
    - apr                        # Native optimized format
    - gguf                       # Third-party format
  quantizations:
    - q4_k_m                    # Most common default

test_matrix:
  modalities: [run, chat, serve] # At least one (required)
  backends: [cpu, gpu]           # At least one (required)
  scenario_count: 1              # 1-1000, default: 100
  timeout_ms: 90000              # 1000-600000, default: 60000
```

## Optional Fields

### description
```yaml
description: "MVP certification for Model X"  # Max 500 chars
```

### model.size_category
```yaml
model:
  size_category: small  # tiny (<1B) | small (1-5B) | medium (5-10B) | large (10-30B) | xlarge (>30B) | huge
```

### test_matrix.seed
```yaml
test_matrix:
  seed: 42  # Fixed seed for reproducibility
```

### gates
All default to `"true"`. Set to `"false"` to skip a gateway check.
```yaml
gates:
  g1_model_loads: "true"
  g2_basic_inference: "true"
  g3_no_crashes: "true"
  g4_output_quality: "true"
```

### oracles
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
    config: {}
  - type: response
    config:
      min_relevance: 0.2
  - type: composite
    config: {}
```

### failure_policy
```yaml
failure_policy: collect_all  # stop_on_first | stop_on_p0 | collect_all
```

### property_tests
```yaml
property_tests:
  enabled: true
  cases: 100           # 1-10000
  max_shrink_iters: 1000
```

### profile_ci (Performance Profiling)
```yaml
profile_ci:
  enabled: "true"
  warmup: 1
  measure: 2
  formats: [safetensors, apr, gguf]
  backends: [cpu, gpu]
  assertions:
    min_throughput: 5.0           # tokens/sec (any backend)
    min_throughput_cpu: 5.0       # tokens/sec CPU
    min_throughput_gpu: 50.0      # tokens/sec GPU
    max_p99_ms: 5000.0            # p99 latency
    max_p50_ms: 2000.0            # p50 latency
  gates: ["F-PERF-CI-001"]
```

### contract_tests (Tensor Layout Contract)
```yaml
contract_tests:
  invariants: ["I-2", "I-3"]    # Any subset of I-1 through I-5
```

### differential_tests
```yaml
differential_tests:
  format_validation:
    enabled: true
    checks: [tensor_count, tensor_names, tensor_shapes]
    gates: ["F-CONV-001"]
  tensor_diff:
    enabled: true
    filter: "*.weight"
    gates: ["F-CONV-002"]
  inference_compare:
    enabled: true
    prompt: "What is 2+2?"
    max_tokens: 32
    tolerance: 0.01
    gates: ["F-CONV-003"]
  fingerprint:
    enabled: true
    tensors: ["token_embd.weight", "output.weight"]
    stats: [mean, std, min, max]
    gates: ["F-CONV-004"]
  validate_stats:
    enabled: true
    reference: "safetensors"
    tolerance:
      layernorm: 0.01
      embedding: 0.05
      attention: 0.03
    gates: ["F-CONV-005"]
```

### trace_payload (APR-TRACE-001)
```yaml
trace_payload:
  enabled: true
  prompt: "What is 2+2?"
  gates: ["F-TRACE-001"]
```

### ollama_parity (GH-6/AC-2)
```yaml
ollama_parity:
  enabled: true
  model_tag: "qwen2.5-coder:7b-instruct-q4_k_m"
  quantizations: [q4_k_m]
  prompts: ["What is 2+2?", "def fibonacci(n):"]
  temperature: 0.0
  min_perf_ratio: 0.8
  gates: ["F-OLLAMA-001"]
```

### state_machine
```yaml
state_machine:
  enabled: false
  states: [idle, loading, running, error]
  transitions:
    - { from: idle, to: loading, action: load }
    - { from: loading, to: running, action: start }
    - { from: running, to: error, action: crash }
```

### metadata
```yaml
metadata:
  author: "apr-qa-gen"
  tier: "mvp"
  architecture: "qwen2"
  tags: [code, instruct, mvp]
```

## Supported Quantizations

```
f32, f16, q8_0, q6_k, q5_k_m, q5_0, q4_k_m, q4_0, q3_k_m, q2_k
```

## Test Matrix Size Formula

```
total_tests = len(formats) * len(backends) * len(modalities) * scenario_count
```

| Template | Formats | Backends | Modalities | Scenarios | Total |
|----------|---------|----------|------------|-----------|-------|
| MVP | 3 | 2 | 3 | 1 | 18 |
| Quick-check | 1 | 1 | 1 | 10 | 10 |
| Basic-verify | 3 | 1 | 1 | 3 | 9 |
| CI-pipeline | 2 | 1 | 3 | 25 | 150 |
| Full | 3 | 2 | 3 | 100 | 1800 |

## Complete MVP Example

```yaml
name: qwen2.5-coder-1.5b-mvp
version: "1.0.0"
description: "MVP certification for Qwen2.5-Coder-1.5B-Instruct"

model:
  hf_repo: "Qwen/Qwen2.5-Coder-1.5B-Instruct"
  formats: [safetensors, apr, gguf]
  quantizations: [q4_k_m]
  size_category: small

test_matrix:
  modalities: [run, chat, serve]
  backends: [cpu, gpu]
  scenario_count: 1
  seed: 42
  timeout_ms: 90000

gates:
  g1_model_loads: "true"
  g2_basic_inference: "true"
  g3_no_crashes: "true"
  g4_output_quality: "true"

oracles:
  - type: arithmetic
    config: { tolerance: 0.01 }
  - type: garbage
    config: { max_repetition_ratio: 0.3, min_unique_chars: 10 }

failure_policy: collect_all

profile_ci:
  enabled: "true"
  warmup: 1
  measure: 2
  formats: [safetensors, apr, gguf]
  backends: [cpu, gpu]
  assertions:
    min_throughput_cpu: 5.0
    min_throughput_gpu: 50.0

contract_tests:
  invariants: ["I-2", "I-3"]

metadata:
  author: "apr-qa-gen"
  tier: "mvp"
  architecture: "qwen2"
  tags: [code, instruct, mvp]
```
