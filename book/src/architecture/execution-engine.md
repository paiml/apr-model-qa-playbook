# Execution Engine

The `apr-qa-runner` crate executes scenarios and collects evidence.

## ParallelExecutor

Uses Rayon for parallel scenario execution:

```rust
let config = ParallelConfig::default()
    .with_max_workers(8)
    .with_timeout(Duration::from_secs(60))
    .with_failure_policy(FailurePolicy::StopOnP0);

let executor = ParallelExecutor::new(config);
let result = executor.execute(&scenarios);
```

## Execution Modes

### Simulate Mode

Fast execution without actual inference. Returns simulated outcomes based on scenario properties.

### Subprocess Mode

Spawns actual inference processes:

```rust
let executor = ParallelExecutor::new(config)
    .with_mode(ExecutionMode::Subprocess);
```

## Evidence Collection

```rust
pub struct Evidence {
    pub gate_id: String,
    pub scenario: QaScenario,
    pub outcome: Outcome,
    pub output: String,
    pub reason: String,
    pub metrics: PerformanceMetrics,
    pub stderr: Option<String>,
}
```

## Jidoka (Stop-on-Failure)

When a P0 gateway fails, execution stops immediately:

```rust
// Atomic flag checked by all workers
let stop_flag = Arc::new(AtomicBool::new(false));

// Workers check flag before each scenario
if stop_flag.load(Ordering::Relaxed) {
    return; // Stop processing
}
```

## Format Conversion Testing (P0 Critical)

The `ConversionExecutor` runs a comprehensive suite of format conversion
tests. Conversion is P0 because a single bit flip invalidates all inference.

### Test Types

```rust
let config = ConversionConfig {
    test_all_pairs: true,      // All 6 format pairs (GGUF↔APR↔ST)
    test_round_trips: true,    // GGUF→APR→ST→GGUF chain (F-CONV-RT-001)
    test_multi_hop: true,      // ST→APR→GGUF→ST, ST→APR→GGUF→APR→ST
    test_cardinality: true,    // tensor_count(out) >= tensor_count(in)
    test_tensor_names: true,   // Tensor name-set preservation
    test_idempotency: true,    // Convert twice, compare outputs
    test_commutativity: true,  // GGUF→APR vs GGUF→ST→APR equivalence
    ..Default::default()
};
let executor = ConversionExecutor::new(config);
let result = executor.execute_all(model_path, &model_id)?;
```

### Metamorphic Relations

These properties are used as oracles — if a metamorphic relation is violated,
a conversion bug has been detected without needing ground truth:

- **MR-RT** (Round-trip): Converting through a chain and back must preserve output
- **MR-CARD** (Cardinality): Conversion must not silently drop tensors
- **MR-IDEM** (Idempotency): Converting A→B twice must produce identical results
- **MR-COM** (Commutativity): Different paths to the same format must agree

### Inspect Metadata Verification

```rust
let tool_exec = ToolExecutor::new(model_path, no_gpu, timeout_ms);
let result = tool_exec.execute_inspect_verified();
// Parses `apr rosetta inspect --json` and validates:
// - tensor_count > 0
// - num_attention_heads > 0 (if present)
// - num_key_value_heads > 0 (if present)
// - hidden_size > 0 (if present)
// Gate: F-INSPECT-META-001
```

## Performance Metrics

```rust
pub struct PerformanceMetrics {
    pub duration_ms: u64,
    pub tokens_per_second: Option<f64>,
    pub total_tokens: Option<u32>,
    pub time_to_first_token_ms: Option<u64>,
}
```
