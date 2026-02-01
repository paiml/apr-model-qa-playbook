# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

APR Model QA Playbook is a property-based model qualification testing framework for HuggingFace models. It implements **Toyota Production System** principles (Jidoka, Poka-Yoke) combined with **Popperian Falsification** methodology—tests are designed to fail, not to pass.

## Build Commands

```bash
make build          # Build all crates
make test           # Run all tests
make lint           # Run clippy (must pass with zero warnings)
make check          # fmt-check + lint + test
make coverage       # Generate coverage report (library code only)
make coverage-check # Verify PMAT compliance (>= 95%)
```

Run a single test:
```bash
cargo test --package apr-qa-gen -- test_name
cargo test --package apr-qa-runner -- test_name --nocapture
```

## Architecture

### Crate Structure

```
crates/
├── apr-qa-gen/     # Scenario generation + oracles (proptest)
├── apr-qa-runner/  # Playbook execution (Rayon parallel)
├── apr-qa-report/  # MQS scoring + JUnit/HTML reports
└── apr-qa-cli/     # CLI binary (not unit tested)
```

### Core Data Flow

1. **apr-qa-gen**: Generates `QaScenario` instances via proptest. Each scenario is a falsifiable hypothesis with prompt, modality (run/chat/serve), backend (cpu/gpu), and format (gguf/safetensors/apr).

2. **apr-qa-runner**: Executes scenarios via `ParallelExecutor` using Rayon. Collects `Evidence` with outcomes: `Corroborated`, `Falsified`, `Timeout`, `Crashed`.

3. **apr-qa-report**: Calculates MQS (Model Qualification Score) 0-1000 with gateway checks G1-G4. Generates JUnit XML and HTML reports.

### Key Types

- `QaScenario` (apr-qa-gen): Test input with model, prompt, modality, backend, format
- `Evidence` (apr-qa-runner): Test result with outcome, metrics, stderr
- `MqsScore` (apr-qa-report): Qualification score with gateways and category breakdowns
- `Oracle` (apr-qa-gen): Verifies output correctness (Arithmetic, Garbage, CodeSyntax, Response)

### Gateway Logic (G1-G4)

Any gateway failure zeros the entire MQS score:
- **G1**: Model loads successfully
- **G2**: Basic inference works
- **G3**: No crashes or panics
- **G4**: Output is not garbage

### Playbook Structure

```
playbooks/
├── playbook.schema.yaml    # JSON Schema for validation
├── models/                 # Per-model qualification playbooks
├── templates/              # Reusable templates (quick-check, ci-pipeline, full)
├── verify/                 # Ticket verification playbooks
└── spec/                   # Executable specifications for gateways
```

## Quality Requirements

- **95% minimum library coverage** (use `cargo llvm-cov --lib`)
- **Zero clippy warnings** with pedantic + nursery lints
- **No unsafe code** (`#![forbid(unsafe_code)]`)
- **No SATD markers** (TODO/FIXME/HACK)
- Never use `cargo tarpaulin` (slow, unreliable)

## Testing Philosophy

Tests follow Popperian falsification:
- `Corroborated`: Hypothesis survived refutation attempt
- `Falsified`: Hypothesis refuted by evidence
- Design tests to fail, not to pass

The parallel executor implements Jidoka—stops on first P0 failure when configured with `stop_on_p0` policy.

## Stack Documentation Search

Query this component's documentation and the entire Sovereign AI Stack using batuta's RAG Oracle:

```bash
# Index all stack documentation (run once, persists to ~/.cache/batuta/rag/)
batuta oracle --rag-index

# Search across the entire stack
batuta oracle --rag "your question here"

# Examples
batuta oracle --rag "property-based testing strategies"
batuta oracle --rag "MQS scoring implementation"
batuta oracle --rag "Jidoka stop-on-error patterns"
```

The oracle indexes:
- All Sovereign AI Stack repos (trueno, aprender, realizar, etc.)
- Ground truth corpora (HuggingFace, JAX, vLLM, TGI patterns)
- This playbook's documentation and source code

Index auto-updates via post-commit hooks and `ora-fresh` on shell login.
To manually check freshness: `ora-fresh`
To force full reindex: `batuta oracle --rag-index --force`
