# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

APR Model QA Playbook is a property-based model qualification testing framework for HuggingFace models. It implements **Toyota Production System** principles (Jidoka, Poka-Yoke) combined with **Popperian Falsification** methodology—tests are designed to fail, not to pass.

## CRITICAL: Ground Truth and Workflow

### Ground Truth Format: SafeTensors

**SafeTensors is the ground truth format.** It is the native HuggingFace format and the source of truth for model weights. All other formats (GGUF, APR) are derived from SafeTensors.

Format hierarchy:
1. **SafeTensors** - Ground truth (HuggingFace source)
2. **APR** - Our native optimized format (converted from SafeTensors)
3. **GGUF** - Third-party format we support (NOT the source of truth)

### NEVER Run Manual Commands for Qualification

**Always use the playbook infrastructure.** Never bypass it with manual `apr` CLI commands.

```bash
# WRONG - bypasses playbook infrastructure
apr qa /path/to/model.gguf
apr run /path/to/model.safetensors
apr pull Qwen/Qwen2.5-Coder-1.5B-Instruct

# CORRECT - uses playbook infrastructure
make certify-mvp
cargo run --bin apr-qa -- certify --family qwen-coder --tier mvp
cargo run --bin apr-qa -- run playbooks/models/qwen2.5-coder-1.5b-mvp.playbook.yaml
```

The playbook infrastructure:
- Ensures consistent test matrix (formats × backends × modalities)
- Tracks evidence in certification directories
- Uses SafeTensors as ground truth
- Generates proper MQS scores and reports

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

## LAYOUT-002: Row-Major Mandate (P0 CRITICAL)

> **This is one of the most important checks in the entire qualification framework.**

### Stack Layout Specification

The Sovereign AI Stack has **ONE tensor layout**: **row-major**. This is a permanent architectural decision.

| Component | Layout | Notes |
|-----------|--------|-------|
| **trueno** | Row-major | Native SIMD kernels, all quant types |
| **aprender** | Row-major | Transposes GGUF at import time |
| **realizar** | Row-major | ONE kernel set, no layout aliases |
| **entrenar** | Row-major | Follows stack convention |

### Kernel API (Final)

```rust
// Q4K - ONE function family
fused_q4k_parallel_matvec(...)
fused_q4k_parallel_matvec_into(...)
fused_q4k_tiled_matvec(...)

// Q5K - ONE function family
fused_q5k_parallel_matvec(...)
fused_q5k_parallel_matvec_into(...)

// Q6K - ONE function family
fused_q6k_parallel_matvec(...)
fused_q6k_parallel_matvec_into(...)
```

**Deleted** (technical debt eliminated):
- ~~`fused_q6k_colmajor_matvec`~~
- ~~`fused_q4k_auto_matvec_into`~~
- All `*_colmajor_*` and `*_auto_*` variants

### Test Results

| Project | Tests | Status |
|---------|-------|--------|
| realizar | 2209 quantize tests | ✅ PASS |
| aprender | 399 converter tests | ✅ PASS |

### Why This Matters for Qualification

**All GGUF models must be converted to APR format before qualification testing.**

GGUF uses column-major layout (GGML convention). Aprender's converter transposes data during import. Testing GGUF directly with realizar will produce garbage output.

**Correct Workflow:**
```bash
# 1. Convert GGUF → APR (aprender transposes layout)
apr import model.gguf -o model.apr

# 2. Run qualification on APR
cargo run --bin apr-qa -- certify --model model.apr
```

**Gateway G4 (Garbage Detection)** specifically catches LAYOUT-002 violations:
- If output contains non-ASCII gibberish like "olumbia+lsi nunca/localENTS"
- The `GarbageOracle` will fail the test with `Falsified` outcome
- This indicates a layout bug, not a model quality issue

**Cross-References:**
- `docs/specifications/apr-playbook-spec.md` → Section 4.1.1
- `docs/tickets/GH-190-GGUF-APR-CONVERSION-GARBAGE-OUTPUT.md`
- `aprender/CLAUDE.md` → LAYOUT-002 section
- `realizar/CLAUDE.md` → LAYOUT-002 section

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
