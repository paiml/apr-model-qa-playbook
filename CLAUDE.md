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
├── apr-qa-report/  # MQS scoring + JUnit/HTML/Markdown reports
├── apr-qa-certify/ # Tier-aware scoring, README sync, CSV export
└── apr-qa-cli/     # CLI binary with 13 subcommands
```

### Core Data Flow

1. **apr-qa-gen**: Generates `QaScenario` instances via proptest. Each scenario is a falsifiable hypothesis with prompt, modality (run/chat/serve), backend (cpu/gpu), and format (gguf/safetensors/apr).

2. **apr-qa-runner**: Executes scenarios via `ParallelExecutor` using Rayon. Collects `Evidence` with outcomes: `Corroborated`, `Falsified`, `Timeout`, `Crashed`.

3. **apr-qa-report**: Calculates MQS (Model Qualification Score) 0-1000 with gateway checks G0-G4. Generates JUnit XML, HTML, and Markdown reports.

4. **apr-qa-certify**: Tier-aware scoring (Smoke/MVP/Quick/Standard/Deep), certification status computation, README table sync from models.csv. Binary: `apr-qa-readme-sync`.

5. **apr-qa-cli**: Orchestrates the full pipeline with 13 subcommands: certify, run, tools, generate, score, report, list, lock-playbooks, tickets, parity, export-csv, export-evidence, validate-contract.

### Key Types

- `QaScenario` (apr-qa-gen): Test input with model, prompt, modality, backend, format
- `Evidence` (apr-qa-runner): Test result with outcome, metrics, stderr
- `MqsScore` (apr-qa-report): Qualification score with gateways and category breakdowns
- `Oracle` (apr-qa-gen): Verifies output correctness (Arithmetic, Garbage, CodeSyntax, Response)

### Gateway Logic (G0-G4)

Any gateway failure zeros the entire MQS score:
- **G0**: config.json matches tensor metadata (Integrity)
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

## Documentation Maintenance (Mandatory)

Documentation drifts at interface boundaries. When changing any of the following, you MUST update the corresponding docs. Run `make docs-check` to verify.

### When adding a workspace crate:
- [ ] `CLAUDE.md` → Crate Structure tree
- [ ] `CLAUDE.md` → Core Data Flow (add numbered item)
- [ ] `README.md` → Architecture diagram
- [ ] `README.md` → Crate Structure table
- [ ] `README.md` → Project Structure tree
- [ ] `book/src/architecture/overview.md` → diagram + dependency graph
- [ ] `book/src/introduction.md` → Project Structure tree

### When adding a CLI subcommand:
- [ ] `book/src/reference/cli.md` → add full section with options and examples

### When adding or removing a gateway (G0-G4):
- [ ] `CLAUDE.md` → Gateway Logic section
- [ ] `README.md` → Features list and MQS Gateway table
- [ ] `book/src/philosophy/mqs.md` → Gateway Logic table
- [ ] `book/src/reference/gateways.md` → add `## GN:` section
- [ ] `book/src/reference/certified-testing.md` → gateway references
- [ ] `book/src/introduction.md` → Key Features
- [ ] `book/src/getting-started.md` → playbook example gates

### When adding certification status/grade variants:
- [ ] `book/src/reference/oracle-integration.md` → models.csv schema
- [ ] `book/src/philosophy/mqs.md` → Grade Mapping table

### After certification runs:
- [ ] Run `make update-certifications` to sync README table from CSV

### Enforcement:
- `make docs-check` runs `scripts/check-docs-consistency.sh`
- `make check` includes `docs-check` in the gate chain
- CI runs documentation consistency checks on every push/PR

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
- **THE SOURCE OF TRUTH:** `aprender/docs/specifications/qwen2.5-coder-showcase-demo.md` → Section E.8 "Tensor Layout Contract"
- `docs/specifications/apr-playbook-spec.md` → Section 4.1.1
- `docs/tickets/GH-190-GGUF-APR-CONVERSION-GARBAGE-OUTPUT.md`
- `aprender/CLAUDE.md` → LAYOUT-002 section
- `realizar/CLAUDE.md` → LAYOUT-002 section

### Per-Tensor Layout Summary (from Tensor Layout Contract)

**DO NOT GREP FOR LAYOUTS.** Read Section E.8 of the Qwen showcase spec.

Quick reference (authoritative source is E.8):

| Tensor | APR Shape | Why |
|--------|-----------|-----|
| `lm_head.weight` | `[vocab, hidden]` | Kernel: `matmul(W, x, vocab, hidden)` |
| `q_proj.weight` | `[heads*head_dim, hidden]` | Kernel: `matmul(W, x, heads*head_dim, hidden)` |
| `embed_tokens.weight` | `[vocab, hidden]` | Lookup table, row = token embedding |

**Key Insight:** The KERNEL defines shape, not comments. When `matmul_q6k_rowmajor(W, x, out_dim, in_dim)` is called with `out_dim=vocab`, W must have `vocab` rows.

## Testing Philosophy

Tests follow Popperian falsification:
- `Corroborated`: Hypothesis survived refutation attempt
- `Falsified`: Hypothesis refuted by evidence
- Design tests to fail, not to pass

The parallel executor implements Jidoka—stops on first P0 failure when configured with `stop_on_p0` policy.

## Oracle Integration (PMAT-275)

The QA framework integrates with aprender's Oracle system for certification tracking and model lookup.

### Data Flow: Certification Pipeline

```
┌─────────────────────┐
│   Run Playbook      │  cargo run --bin apr-qa -- run playbook.yaml
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Collect Evidence   │  certifications/{model}/evidence.json
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Calculate MQS      │  0-1000 score with G0-G4 gateways
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Update models.csv  │  docs/certifications/models.csv
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Oracle Lookup      │  apr oracle <model_id> → certification status
└─────────────────────┘
```

### Key Files

| File | Purpose |
|------|---------|
| `certifications/{model}/evidence.json` | Raw test evidence (JSON array) |
| `docs/certifications/models.csv` | Certification lookup table |
| `playbooks/evidence.schema.json` | Evidence JSON schema |
| `scripts/validate-schemas.sh` | Schema validation script |
| `scripts/validate-aprender-alignment.sh` | Cross-repo consistency check |

### Gateway System (G0-G4)

| Gate | Name | Description |
|------|------|-------------|
| G0 | Integrity | config.json matches tensor metadata |
| G1 | Load | Model loads successfully |
| G2 | Inference | Basic inference works |
| G3 | Stability | No crashes or panics |
| G4 | Quality | Output is not garbage |

**Any gateway failure zeros the MQS score.**

### Family Contract Integration

Playbooks auto-populate from aprender's family YAMLs:
- `size_category` from `certification.size_categories`
- `expected_hidden_dim` from `size_variants.{size}.hidden_dim`
- Tensor templates from `tensor_template`

Cross-repo validation runs in CI to ensure alignment.

### Certification Workflows

**Certify a new model:**
```bash
# 1. Create playbook from template
cp playbooks/templates/mvp-template.yaml playbooks/models/my-model-mvp.playbook.yaml
# Edit playbook with model details

# 2. Run certification
cargo run --bin apr-qa -- run playbooks/models/my-model-mvp.playbook.yaml \
    --output certifications/my-model/evidence.json

# 3. Update models.csv (manual or script)
# 4. Commit evidence and CSV updates
```

**Re-run after fixes:**
```bash
# Clear old evidence (optional - append mode also works)
rm -rf certifications/my-model/

# Re-run same playbook
cargo run --bin apr-qa -- run playbooks/models/my-model-mvp.playbook.yaml \
    --output certifications/my-model/evidence.json
```

### Troubleshooting

**"Gateway G0 failed: config mismatch"**
- Model's config.json doesn't match actual tensor shapes
- Verify config.json exists and is not corrupted
- Check `num_hidden_layers`, `hidden_size` match tensor count/shapes

**"size_category mismatch" in cross-repo validation**
- Playbook's `size_category` doesn't match family YAML
- Update playbook or submit PR to fix family contract

**Evidence JSON validation errors**
- Run `./scripts/validate-schemas.sh` locally
- Check evidence.json is valid JSON array with required fields

## Code Search Policy

**NEVER use grep/glob for code search. ALWAYS prefer `pmat query`.**

`pmat query` returns quality-annotated, semantically ranked results with TDG grades, complexity, fault patterns, and call graphs.

```bash
# Find functions by intent
pmat query "error handling" --limit 10

# Find code with fault patterns (unwrap, panic, unsafe)
pmat query "unwrap" --faults --exclude-tests

# Find coverage gaps ranked by ROI
pmat query --coverage-gaps --rank-by impact --limit 20

# Regex search
pmat query --regex "fn\s+test_\w+" --limit 10
```

When grep IS acceptable: searching non-code files (TOML, YAML, Markdown).

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
