# Playbook Schema Reference

Full JSON Schema: `playbooks/playbook.schema.yaml`

## Root Object

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Unique identifier `^[a-z0-9-]+$` |
| `version` | string | Yes | Semantic version `^\d+\.\d+\.\d+$` |
| `description` | string | No | Human-readable description |
| `model` | object | Yes | Model specification |
| `test_matrix` | object | Yes | Test configuration |
| `gates` | object | No | Gateway checks |
| `oracles` | array | No | Output verification |
| `failure_policy` | string | No | How to handle failures |
| `property_tests` | object | No | Proptest configuration |
| `state_machine` | object | No | FSM test configuration |
| `metadata` | object | No | Additional metadata |

## model

| Field | Type | Required | Values |
|-------|------|----------|--------|
| `hf_repo` | string | Yes | `org/model-name` |
| `formats` | array | No | `gguf`, `safetensors`, `apr` |
| `quantizations` | array | No | `f32`, `f16`, `q8_0`, `q6_k`, `q5_k_m`, `q5_0`, `q4_k_m`, `q4_0`, `q3_k_m`, `q2_k` |
| `size_category` | string | No | `tiny`, `small`, `medium`, `large`, `xlarge`, `huge` |

## test_matrix

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `modalities` | array | Yes | `run`, `chat`, `serve` |
| `backends` | array | Yes | `cpu`, `gpu` |
| `scenario_count` | integer | No | 1-1000, default 100 |
| `seed` | integer | No | Random seed |
| `timeout_ms` | integer | No | 1000-600000, default 60000 |

## gates

| Field | Type | Default |
|-------|------|---------|
| `g1_model_loads` | boolean | true |
| `g2_basic_inference` | boolean | true |
| `g3_no_crashes` | boolean | true |
| `g4_output_quality` | boolean | true |

## oracles

Array of oracle configurations:

```yaml
oracles:
  - type: arithmetic | garbage | code_syntax | response | composite
    config: { ... }  # Oracle-specific
```

## failure_policy

| Value | Behavior |
|-------|----------|
| `stop_on_first` | Stop on any failure |
| `stop_on_p0` | Stop on gateway failures (default) |
| `collect_all` | Run all tests |

## property_tests

| Field | Type | Default |
|-------|------|---------|
| `enabled` | boolean | true |
| `cases` | integer | 100 |
| `max_shrink_iters` | integer | 1000 |
