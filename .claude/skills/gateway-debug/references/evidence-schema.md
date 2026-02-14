# Evidence Schema Reference

Complete schema for evidence JSON artifacts produced by qualification runs.

## Top-Level Structure

```json
{
  "evidence": [ /* array of Evidence records */ ],
  "metadata": {
    "timestamp": "2026-02-14T10:30:00Z",    // ISO 8601 (required)
    "version": "1.0.0",                       // Schema version (required)
    "model_id": "Qwen/Qwen2.5-Coder-1.5B",  // HF repo ID (required)
    "total_duration_ms": 45000,               // Total run time (optional)
    "passed": 15,                             // Pass count (optional)
    "failed": 3,                              // Fail count (optional)
    "skipped": 0                              // Skip count (optional)
  }
}
```

## Evidence Record

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "gate_id": "F-QUAL-001",
  "scenario": { /* QaScenario */ },
  "outcome": "Corroborated",
  "reason": "Test passed",
  "output": "The answer is 4.",
  "stderr": null,
  "exit_code": 0,
  "metrics": { /* PerformanceMetrics */ },
  "timestamp": "2026-02-14T10:30:00Z",
  "host": { /* HostInfo */ },
  "metadata": { "key": "value" }
}
```

### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | UUID v4 | yes | Unique identifier, generated at runtime |
| `gate_id` | string | yes | Test gate identifier (e.g., `F-QUAL-001`, `G0-PULL-001`) |
| `scenario` | object | yes | Test scenario that was executed |
| `outcome` | enum | yes | `Corroborated`, `Falsified`, `Timeout`, `Crashed`, `Skipped` |
| `reason` | string | yes | Human-readable explanation |
| `output` | string | yes | Raw stdout from subprocess (NOT optional) |
| `stderr` | string? | no | Raw stderr (only populated for `Crashed`) |
| `exit_code` | int? | no | Process exit code (`Some(0)` for corroborated, `None` for falsified) |
| `metrics` | object | yes | Performance measurements |
| `timestamp` | ISO 8601 | yes | When the test executed |
| `host` | object | yes | Machine/environment info for reproducibility |
| `metadata` | map | no | Arbitrary key-value pairs |

## Scenario Object

```json
{
  "model_id": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
  "modality": "run",
  "backend": "cpu",
  "format": "safetensors",
  "prompt": "What is 2+2?",
  "seed": 42
}
```

| Field | Type | Values |
|-------|------|--------|
| `model_id` | string | HuggingFace repo ID |
| `modality` | enum | `run`, `chat`, `serve` |
| `backend` | enum | `cpu`, `gpu` |
| `format` | enum | `gguf`, `safetensors`, `apr` |
| `prompt` | string | Test prompt |
| `seed` | integer | Random seed for reproducibility |

## Outcome Values

| Outcome | `is_pass()` | `is_fail()` | Description |
|---------|------------|------------|-------------|
| `Corroborated` | true | false | Hypothesis survived refutation attempt |
| `Skipped` | true | false | Test skipped (e.g., stop-on-failure policy) |
| `Falsified` | false | true | Hypothesis refuted by evidence |
| `Timeout` | false | true | Exceeded time limit |
| `Crashed` | false | true | Process crashed (non-zero exit) |

## Metrics Object

```json
{
  "duration_ms": 1500,
  "tokens_per_second": 17.3,
  "time_to_first_token_ms": 200,
  "total_tokens": 26,
  "memory_peak_mb": 1024
}
```

| Field | Type | Description |
|-------|------|-------------|
| `duration_ms` | u64 | Total execution time (always present) |
| `tokens_per_second` | f64? | Throughput metric |
| `time_to_first_token_ms` | f64? | Time to first token (TTFT) |
| `total_tokens` | u32? | Number of output tokens |
| `memory_peak_mb` | u64? | Peak memory usage |

## Host Object

```json
{
  "hostname": "build-server-01",
  "os": "Linux 6.8.0",
  "cpu": "AMD EPYC 7763",
  "gpu": "NVIDIA A100 80GB",
  "apr_version": "0.5.0"
}
```

## Gate ID Conventions

| Pattern | Gateway | MQS Category |
|---------|---------|-------------|
| `G0-PULL-*` | G0 | STAB |
| `G0-FORMAT-*` | G0 | STAB |
| `G0-VALIDATE-*` | G0 | STAB |
| `G0-TENSOR-*` | G0 | STAB |
| `G0-INTEGRITY-*` | G0 | STAB |
| `G0-LAYOUT-*` | G0 | STAB |
| `G1-*` | G1 | Gateway (zeroes all) |
| `G2-*` | G2 | Gateway (zeroes all) |
| `G3-*` | G3 (implicit via Crashed) | Gateway (zeroes all) |
| `G4-GARBAGE-*` | G4 | Gateway (zeroes all) |
| `F-QUAL-*` | - | QUAL (200 pts) |
| `F-PERF-*` | - | PERF (150 pts) |
| `F-STAB-*` | - | STAB (200 pts) |
| `F-COMP-*` | - | COMP (150 pts) |
| `F-EDGE-*` | - | EDGE (150 pts) |
| `F-REGR-*` | - | REGR (150 pts) |
| `F-CONV-*` | - | COMP (150 pts) |
| `F-CONV-RT*` | - | REGR (150 pts) |
| `F-CONTRACT-*` | - | COMP (150 pts) |

## Evidence Constructor Mapping

| Constructor | Outcome | reason | exit_code | stderr |
|-------------|---------|--------|-----------|--------|
| `corroborated(gate_id, scenario, output, duration_ms)` | Corroborated | "Test passed" | Some(0) | None |
| `falsified(gate_id, scenario, reason, output, duration_ms)` | Falsified | (provided) | None | None |
| `timeout(gate_id, scenario, timeout_ms)` | Timeout | "Timed out after {N}ms" | None | None |
| `crashed(gate_id, scenario, stderr, exit_code, duration_ms)` | Crashed | "Process crashed with exit code {N}" | Some(N) | Some(stderr) |
| `skipped(gate_id, scenario, reason)` | Skipped | (provided) | None | None |

## Validation

```bash
# Validate evidence against schema
./scripts/validate-schemas.sh

# Schema file location
playbooks/evidence.schema.json
```
