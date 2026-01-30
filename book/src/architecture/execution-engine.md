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

## Performance Metrics

```rust
pub struct PerformanceMetrics {
    pub duration_ms: u64,
    pub tokens_per_second: Option<f64>,
    pub total_tokens: Option<u32>,
    pub time_to_first_token_ms: Option<u64>,
}
```
