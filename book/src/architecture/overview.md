# Architecture Overview

## Crate Structure

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          APR-MODEL-QA-PLAYBOOK                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │  apr-qa-gen     │    │  apr-qa-runner  │    │  apr-qa-report  │         │
│  │                 │    │                 │    │                 │         │
│  │  - proptest     │───▶│  - playbook     │───▶│  - popperian    │         │
│  │  - scenario gen │    │  - parallel     │    │  - junit/html   │         │
│  │  - oracle def   │    │  - evidence     │    │  - mqs scoring  │         │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘         │
│          │                       │                       │                  │
│          └───────────────────────┼───────────────────────┘                  │
│                                  ▼                                          │
│  ┌─────────────────┐    ┌─────────────────┐                                │
│  │ apr-qa-certify  │    │  apr-qa-cli     │                                │
│  │                 │◀───│                 │                                │
│  │  - tier scoring │    │  - certify cmd  │                                │
│  │  - README sync  │    │  - run/report   │                                │
│  │  - CSV export   │    │  - Jidoka sigs  │                                │
│  └─────────────────┘    └─────────────────┘                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Data Flow

1. **apr-qa-gen** generates `QaScenario` instances via proptest
2. **apr-qa-runner** executes scenarios in parallel, produces `Evidence`
3. **apr-qa-report** calculates `MqsScore`, generates JUnit/HTML/Markdown reports
4. **apr-qa-cli** orchestrates the pipeline with 13 subcommands (certify, run, report, etc.)
5. **apr-qa-certify** handles tier-aware scoring and README certification table sync

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
    ├── apr-qa-certify
    ├── apr-qa-report
    │       ├── apr-qa-runner
    │       │       └── apr-qa-gen
    │       └── apr-qa-gen
    ├── apr-qa-runner
    └── apr-qa-gen
```

## Position in the Sovereign AI Stack

The QA playbook serves as the **end-to-end kernel correctness oracle** for the
Sovereign AI Stack. It validates that the full inference pipeline — from tensor
deserialization through kernel execution to token generation — produces correct
output.

```
┌─────────────────────────────────────────────────────────────┐
│                    Kernel Level (trueno)                     │
│  fused_q4k_parallel_matvec, RMSNorm, Attention, Softmax    │
└──────────────────────────┬──────────────────────────────────┘
                           │ dispatches
┌──────────────────────────▼──────────────────────────────────┐
│                  Runtime Level (realizar)                    │
│  CudaExecutor, SIMD dispatcher, format-aware inference      │
└──────────────────────────┬──────────────────────────────────┘
                           │ converts
┌──────────────────────────▼──────────────────────────────────┐
│                  Format Level (aprender)                     │
│  SafeTensors → APR, GGUF → APR (transposes LAYOUT-002)      │
└──────────────────────────┬──────────────────────────────────┘
                           │ validates
┌──────────────────────────▼──────────────────────────────────┐
│              Qualification Level (this project)              │
│  G0-G4 gateways, GarbageOracle, contract invariants I-1–I-5 │
│  Metamorphic relations (RT, CARD, IDEM, COM)                 │
│  170-point verification matrix, MQS scoring                  │
└─────────────────────────────────────────────────────────────┘
```

A kernel bug at the trueno level (e.g., incorrect quantized matmul) propagates
through realizar and aprender, ultimately manifesting as garbage output or
statistical drift that the playbook's gateways and contract tests detect.
