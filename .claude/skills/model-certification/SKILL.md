---
name: model-certification
description: "Guides end-to-end HuggingFace model certification: playbook creation from templates, running qualification at any tier (Smoke/MVP/Quick/Standard/Deep), collecting evidence, computing MQS scores with G0-G4 gateways, updating models.csv, and syncing the README certification table. Covers SafeTensors ground truth, LAYOUT-002 compliance, and Popperian falsification methodology."
disable-model-invocation: false
user-invocable: true
allowed-tools: "Read, Grep, Glob, Bash"
argument-hint: "model or family: qwen-coder, llama, starcoder, mistral, phi, deepseek, or tier: smoke, mvp, quick, standard, deep"
---

# Model Certification

This skill guides the full qualification pipeline for HuggingFace models using Popperian falsification and Toyota Production System principles.

## Quick Start

### Certify a Model (Recommended Path)

```bash
# 1. Pick a tier
cargo run --bin apr-qa -- certify --family qwen-coder --tier mvp

# 2. Or run a specific playbook
cargo run --bin apr-qa -- run playbooks/models/qwen2.5-coder-1.5b-mvp.playbook.yaml \
    --output certifications/qwen2.5-coder-1.5b/evidence.json

# 3. Update certification registry
make update-certifications
```

### Makefile Shortcuts

```bash
make certify-smoke      # Tier 1: ~1-2 min, minimal sanity
make certify-mvp        # Tier 2: ~5-10 min, full surface coverage
make certify-quick      # Tier 3: ~10-30 min, balanced (default)
make certify-standard   # Tier 4: ~1-2 hr, extended matrix
make certify-deep       # Tier 5: ~8-24 hr, production certification
make certify-qwen       # Priority: Qwen Coder family
make ci-smoke           # CI: 1.5B safetensors CPU only
make nightly-7b         # Nightly: 7B MVP qualification
```

## Certification Tiers

| Tier | Time | Test Matrix | Playbook Suffix | Pass Threshold |
|------|------|-------------|-----------------|----------------|
| **Smoke** | ~1-2 min | safetensors/cpu/run only | `-smoke` | MQS >= 700 |
| **MVP** | ~5-10 min | 3 formats x 2 backends x 3 modalities = 18 | `-mvp` | MQS >= 700 |
| **Quick** | ~10-30 min | Balanced coverage, 10+ scenarios | (none) | MQS >= 700 |
| **Standard** | ~1-2 hr | Extended matrix, 170+ data points | (none) | MQS >= 700 |
| **Deep** | ~8-24 hr | Full matrix, 1800+ tests | (none) | MQS >= 900 |

### Tier Selection Guide

- **Smoke**: "Does it load at all?" Quick regression check.
- **MVP**: "Does it work across all format/backend/modality combos?" Pre-release gate.
- **Quick**: "Is it stable enough for development?" CI/CD pipeline default.
- **Standard**: "Is it production-viable?" Extended stress testing.
- **Deep**: "Full production certification." Comprehensive falsification.

## Pipeline Steps

### Step 1: Create a Playbook

Start from a template:

```bash
# Copy the MVP template
cp playbooks/templates/mvp.yaml \
   playbooks/models/my-model-1.5b-mvp.playbook.yaml
```

Edit the playbook with model-specific values. See [playbook-anatomy.md](references/playbook-anatomy.md) for field reference.

**Available Templates:**

| Template | Matrix Size | Use Case |
|----------|-------------|----------|
| `mvp.yaml` | 18 tests (3x2x3) | Full surface coverage |
| `quick-check.yaml` | 10 tests (1x1x1x10) | Fast sanity check |
| `basic-verify.yaml` | 9 tests (3x1x1x3) | Format comparison |
| `ci-pipeline.yaml` | 225 tests (3x1x3x25) | CI/CD gate |
| `full-qualification.yaml` | 1800 tests (3x2x3x100) | Production certification |

### Step 2: Run Certification

```bash
# By family and tier (auto-finds playbook)
cargo run --bin apr-qa -- certify \
    --family qwen-coder \
    --tier mvp \
    --model-cache ~/.cache/apr/models

# By specific playbook
cargo run --bin apr-qa -- run \
    playbooks/models/qwen2.5-coder-1.5b-mvp.playbook.yaml \
    --output certifications/qwen2.5-coder-1.5b/ \
    --failure-policy collect-all

# Dry run first (see what would execute)
cargo run --bin apr-qa -- certify --family qwen-coder --tier mvp --dry-run
```

**Key CLI Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--tier` | `quick` | Certification tier |
| `--failure-policy` | `stop-on-p0` | `stop-on-first`, `stop-on-p0`, `collect-all`, `fail-fast` |
| `--workers` | `4` | Parallel test workers |
| `--timeout` | `60000` | Per-test timeout (ms) |
| `--no-gpu` | false | CPU-only mode |
| `--model-cache` | - | Model file directory |
| `--apr-binary` | `apr` | Path to APR binary |
| `--dry-run` | false | Show plan without executing |
| `--fail-fast` | false | Stop + enhanced diagnostics |

**Failure Policies (Jidoka):**

| Policy | Behavior | When to Use |
|--------|----------|-------------|
| `stop-on-first` | Halt on any failure | Debugging a specific issue |
| `stop-on-p0` | Halt on critical (G0-G4) failure, continue on others | Default for most runs |
| `collect-all` | Run everything, report all failures | MVP/certification runs |
| `fail-fast` | Halt + emit enhanced tracing diagnostics | Deep debugging |

### Step 3: Review Evidence

```bash
# Score the evidence
cargo run --bin apr-qa -- score certifications/my-model/evidence.json

# Generate reports (HTML + JUnit XML + Markdown)
cargo run --bin apr-qa -- report certifications/my-model/evidence.json \
    --output certifications/my-model/ \
    --formats all
```

### Step 4: Export to Registry

```bash
# Export evidence to certification CSV
cargo run --bin apr-qa -- export-csv \
    --evidence-dir docs/certifications/evidence \
    --output docs/certifications/models.csv

# Sync README table from CSV
make update-certifications
```

### Step 5: Validate Contract Compliance

```bash
# Validate tensor layout contract
cargo run --bin apr-qa -- validate-contract /path/to/model.gguf

# Run all 5 conversion invariants
./scripts/diagnose-conversion.sh /path/to/model.gguf
```

## MQS Scoring System (0-1000)

### Score Calculation

Six categories, 1000 raw points total:

| Category | Code | Max Points | What It Measures |
|----------|------|-----------|------------------|
| Quality | QUAL | 200 | Basic quality, loads, responds |
| Performance | PERF | 150 | Throughput, latency metrics |
| Stability | STAB | 200 | Stability under stress |
| Compatibility | COMP | 150 | Format/backend coverage |
| Edge Cases | EDGE | 150 | Edge case handling |
| Regression | REGR | 150 | Regression resistance |

**Penalties:**
- Crash: -20 points each
- Timeout: -10 points each
- Gateway failure: **-1000** (zeroes entire score)

**Normalization:** Logarithmic scaling `f(x) = 100 * log(1 + 9x) / log(10)` maps raw 0-1000 to normalized 0-100.

### Grade Mapping

| Normalized | Grade | Status |
|-----------|-------|--------|
| >= 97 | A+ | CERTIFIED |
| >= 93 | A | CERTIFIED |
| >= 90 | A- | CERTIFIED |
| >= 83 | B | PROVISIONAL |
| >= 70 | C | Qualifies |
| >= 60 | D | Below threshold |
| < 60 | F | BLOCKED |

### Qualification Thresholds

- `qualifies()`: gateways passed AND normalized >= 70
- `is_production_ready()`: gateways passed AND normalized >= 90

## Gateway System (G0-G4)

**Any gateway failure zeroes the entire MQS score to 0.**

| Gate | Name | What It Checks | Common Failure |
|------|------|----------------|----------------|
| **G0** | Integrity | config.json matches tensor metadata | Corrupted config, wrong tensor count |
| **G1** | Load | Model loads without errors | Missing files, bad format, OOM |
| **G2** | Inference | Basic inference produces output | Timeout, crash during forward pass |
| **G3** | Stability | No crashes, panics, or segfaults | LAYOUT-002 violations, null pointers |
| **G4** | Quality | Output is not garbage | Repetitive patterns, NaN/Inf, encoding errors |

See [gateway-diagnostics.md](references/gateway-diagnostics.md) for failure diagnosis.

## Format Hierarchy

```
SafeTensors (Ground Truth)
    |
    +-- APR (Native optimized, converted from SafeTensors)
    |
    +-- GGUF (Third-party, MUST be converted via aprender)
```

**SafeTensors is always the source of truth.** GGUF uses column-major layout (GGML convention); aprender transposes during import. Testing GGUF directly with realizar produces garbage output (LAYOUT-002 violation).

**Correct workflow:**
```bash
# 1. Convert GGUF -> APR (aprender transposes layout)
apr import model.gguf -o model.apr

# 2. Run qualification on APR
cargo run --bin apr-qa -- certify --model model.apr
```

## Contract Invariants (I-1 through I-5)

| Invariant | Name | Gate ID | What It Catches |
|-----------|------|---------|-----------------|
| **I-1** | Round-trip Identity | `F-CONTRACT-I1-001` | Inference divergence after conversion |
| **I-2** | Tensor Name Bijection | `F-CONTRACT-I2-001` | Missing/extra tensors in converted model |
| **I-3** | No Silent Fallbacks | `F-CONTRACT-I3-001` | Unknown dtype defaulting to F32 |
| **I-4** | Statistical Preservation | `F-CONTRACT-I4-001` | Tensor statistics drift beyond tolerance |
| **I-5** | Tokenizer Roundtrip | `F-CONTRACT-I5-001` | First-token mismatch between formats |

## Certification Status State Machine

```
PENDING --[run tests]--> BLOCKED (any gateway fails OR MQS < 700)
PENDING --[run tests]--> PROVISIONAL (MQS >= 700, < 850)
PENDING --[run tests]--> CERTIFIED (MQS >= 850)
```

**CSV Status Values:** `CERTIFIED`, `PROVISIONAL`, `BLOCKED`, `PENDING`, `PARTIAL`, `FAIL`

## models.csv Schema (20 columns)

```
model_id, family, parameters, size_category, status, mqs_score, grade,
certified_tier, last_certified, g1, g2, g3, g4, tps_gguf_cpu, tps_gguf_gpu,
tps_apr_cpu, tps_apr_gpu, tps_st_cpu, tps_st_gpu, provenance_verified
```

| Field | Type | Values |
|-------|------|--------|
| `model_id` | string | HuggingFace repo ID (e.g., `Qwen/Qwen2.5-Coder-1.5B-Instruct`) |
| `size_category` | enum | `tiny`, `small`, `medium`, `large`, `xlarge`, `huge` |
| `status` | enum | `CERTIFIED`, `PROVISIONAL`, `BLOCKED`, `PENDING`, `PARTIAL`, `FAIL` |
| `grade` | enum | `A+`, `A`, `A-`, `B+`, `B`, `B-`, `C+`, `C`, `C-`, `D+`, `D`, `D-`, `F` |
| `certified_tier` | enum | `smoke`, `quick`, `mvp`, `standard`, `deep`, `none` |
| `g1`-`g4` | bool | Gateway pass/fail |
| `tps_*` | float | Tokens/second by format and backend |

## Common Pitfalls

### 1. Using GGUF Directly (LAYOUT-002)

GGUF is column-major. Running it directly through realizar produces garbage. Always convert first.

**Symptom:** G4 failure with garbage like `olumbia+lsi nunca/localENTS`

**Fix:** Convert GGUF -> APR via `apr import model.gguf -o model.apr`

### 2. Missing `--model-cache`

The certify command needs to find model files. Either:
- Set `--model-cache /path/to/models`
- Or ensure models are in the default cache location

### 3. Forgetting to Update README

After certification runs, ALWAYS sync the README table:
```bash
make update-certifications
```

### 4. Wrong Failure Policy for the Task

- Debugging? Use `--fail-fast` for enhanced tracing
- Certification run? Use `--failure-policy collect-all` to get complete picture
- CI gate? Use `--failure-policy stop-on-p0` (default)

## Other CLI Subcommands

| Command | Purpose | Example |
|---------|---------|---------|
| `generate` | Generate test scenarios | `apr-qa generate Qwen/Qwen2.5-Coder-1.5B-Instruct -c 50` |
| `score` | Calculate MQS from evidence | `apr-qa score evidence.json` |
| `report` | Generate HTML/JUnit/Markdown | `apr-qa report evidence.json --formats all` |
| `list` | Query model registry | `apr-qa list --size small` |
| `lock-playbooks` | Generate integrity lock file | `apr-qa lock-playbooks playbooks/models/` |
| `tickets` | Auto-generate failure tickets | `apr-qa tickets evidence.json --repo paiml/aprender` |
| `parity` | HF golden corpus verification | `apr-qa parity --model-family qwen2.5-coder-1.5b` |
| `export-csv` | Export evidence to CSV | `apr-qa export-csv --evidence-dir evidence/` |
| `export-evidence` | Export structured evidence | `apr-qa export-evidence source.json --model Qwen/...` |
| `validate-contract` | Tensor layout contract check | `apr-qa validate-contract model.gguf` |
| `tools` | APR tool coverage tests | `apr-qa tools /path/to/model` |

## See Also

### References
- [playbook-anatomy.md](references/playbook-anatomy.md) - Complete playbook YAML field reference
- [gateway-diagnostics.md](references/gateway-diagnostics.md) - Gateway failure diagnosis guide

### Scripts
- [certify-quickstart.sh](scripts/certify-quickstart.sh) - Interactive certification helper

### Key Files
| File | Purpose |
|------|---------|
| `playbooks/playbook.schema.yaml` | Playbook JSON Schema |
| `playbooks/evidence.schema.json` | Evidence artifact schema |
| `docs/certifications/models.csv` | Certification registry (93+ models) |
| `playbooks/templates/` | 5 reusable templates |
| `playbooks/models/` | 120+ model-specific playbooks |
| `scripts/diagnose-conversion.sh` | 5-invariant conversion test |
| `scripts/validate-schemas.sh` | Schema validation |
| `scripts/validate-aprender-alignment.sh` | Cross-repo consistency |
