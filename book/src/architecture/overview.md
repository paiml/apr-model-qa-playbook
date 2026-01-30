# Architecture Overview

## Crate Structure

```
┌──────────────────────────────────────────────────────────────────────┐
│                        APR-MODEL-QA-PLAYBOOK                          │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐   │
│  │  apr-qa-gen     │    │  apr-qa-runner  │    │  apr-qa-report  │   │
│  │  (Rust crate)   │    │  (Rust crate)   │    │  (Rust crate)   │   │
│  │                 │    │                 │    │                 │   │
│  │  - proptest     │───▶│  - playbook     │───▶│  - popperian    │   │
│  │  - scenario gen │    │  - parallel     │    │  - junit/html   │   │
│  │  - oracle def   │    │  - evidence     │    │  - mqs scoring  │   │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘   │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

## Data Flow

1. **apr-qa-gen** generates `QaScenario` instances
2. **apr-qa-runner** executes scenarios, produces `Evidence`
3. **apr-qa-report** calculates `MqsScore`, generates reports

## Key Types

### QaScenario

```rust
pub struct QaScenario {
    pub model_id: ModelId,
    pub modality: Modality,
    pub backend: Backend,
    pub format: Format,
    pub prompt: String,
    pub seed: u64,
}
```

### Evidence

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

### MqsScore

```rust
pub struct MqsScore {
    pub model_id: String,
    pub raw_score: u32,           // 0-1000
    pub normalized_score: f64,    // 0-100
    pub grade: String,            // A+, A, A-, B+, ...
    pub gateways_passed: bool,
    pub categories: CategoryScores,
}
```

## Dependency Graph

```
apr-qa-cli
    └── apr-qa-report
            └── apr-qa-runner
                    └── apr-qa-gen
```
