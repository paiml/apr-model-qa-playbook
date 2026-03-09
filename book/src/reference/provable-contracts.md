# Provable Contracts

The QA framework's behavioral contracts are expressed as machine-checkable
YAML specifications validated by [`pv lint`](https://github.com/paiml/provable-contracts).
This complements the runtime enforcement in Rust code with a static analysis
layer that verifies the contracts are complete, traceable, and scored above
threshold.

See **Spec §18** for the full design rationale.

## Architecture

```
contracts/                          .pv.toml (config)
├── apr-format-invariants-v1.yaml   ─┐
├── gateway-contract-v1.yaml         ├── pv lint (validate → audit → score)
├── mqs-scoring-v1.yaml              │
├── garbage-oracle-v1.yaml          ─┘
└── binding.yaml                    ── traceability: obligation → crate function
```

Runtime enforcement remains in Rust code — the contracts document **what** the
code enforces, and `pv lint` verifies that documentation is well-specified.

## Contract Format

Each contract YAML has these sections:

| Section | Purpose |
|---------|---------|
| `metadata` | Version, author, description, references, `registry: true` |
| `equations` | Formal equations with domain and invariants |
| `proof_obligations` | Typed obligations (invariant, bound, monotonicity, etc.) |
| `falsification_tests` | FALSIFY-XX-NNN tests with predictions and failure explanations |
| `enforcement` | Rules linking obligations to tests with severity |
| `qa_gate` | Gate ID, checks, pass criteria |

## Contracts

### APR Format Invariants (I-1..I-5)

**File:** `contracts/apr-format-invariants-v1.yaml`

Formalizes the five behavioral invariants from §17 that prevent silent format
conversion corruption:

| ID | Invariant | Formal Property |
|----|-----------|-----------------|
| I-1 | Round-trip identity | `inference(convert(M)) = inference(M)` within dtype tolerance |
| I-2 | Tensor name bijection | `names(write(M)) = names(read(M))` exact set equality |
| I-3 | No silent fallbacks | Unknown dtype must error, never default to F32 |
| I-4 | Statistical preservation | Tensor mean/std/min/max within dtype tolerance |
| I-5 | Tokenizer roundtrip | `encode(decode(tokens)) = tokens` |

Plus 2 equivalence obligations: GGUF-to-APR layout transposition and dtype byte agreement.

### Gateway Pipeline (G0-G4)

**File:** `contracts/gateway-contract-v1.yaml`

Formalizes the gateway preconditions:

- **Gateway zeroing:** Any gateway failure zeros the MQS score
- **Two-phase execution:** G0 sub-gates complete before scenario-based G1-G4
- **G4 garbage threshold:** G4 fails when `garbage_count > evidence_count / 4`
- **Five gateway types:** `check_gateways` produces exactly 5 `GatewayResult` items

### MQS Scoring

**File:** `contracts/mqs-scoring-v1.yaml`

Formalizes the scoring computation:

- **Categories:** QUAL=200, PERF=150, STAB=200, COMP=150, EDGE=150, REGR=150
- **Raw score range:** [0, 1050] (1000 base + 50 proof bonus)
- **Normalized score:** [0.0, 100.0] clamped
- **12-grade scale:** A+ (>=97) through D- (>=60), F (<60) on normalized score
- **Deterministic:** Same evidence always produces same score

### GarbageOracle (G4)

**File:** `contracts/garbage-oracle-v1.yaml`

Formalizes the output quality gate:

- **Soundness:** No false positives on valid model output
- **LAYOUT-002 detection:** Layout violations manifest as garbage (control chars or repetition)
- **Five detectors:** Empty/whitespace, control characters, NaN/Inf, repetitive patterns, U+FFFD
- **Empty output:** Empty or whitespace-only output always classified as garbage

## Binding Registry

**File:** `contracts/binding.yaml`

Maps proof obligations to their implementing functions:

| Contract | Equation | Function | Crate |
|----------|----------|----------|-------|
| format-invariants | round_trip_identity | `run_golden_rule_test` | apr-qa-runner |
| format-invariants | tensor_name_bijection | `run_i2_tensor_bijection` | apr-qa-runner |
| format-invariants | no_silent_fallbacks | `run_i3_no_silent_fallbacks` | apr-qa-runner |
| format-invariants | statistical_preservation | `run_i4_statistical_preservation` | apr-qa-runner |
| format-invariants | tokenizer_roundtrip | `run_i5_tokenizer_roundtrip` | apr-qa-runner |
| gateway | gateway_zeroing | `MqsCalculator::check_gateways` | apr-qa-report |
| gateway | gateway_two_phase | `execute` | apr-qa-runner |
| gateway | gateway_scoring | `MqsCalculator::calculate` | apr-qa-report |
| mqs-scoring | mqs_composite | `MqsCalculator::calculate` | apr-qa-report |
| mqs-scoring | grade_mapping | `MqsCalculator::calculate_grade` | apr-qa-report |
| mqs-scoring | penalty_floor | `MqsCalculator::calculate` (inlined) | apr-qa-report |
| garbage-oracle | garbage_detection | `GarbageOracle::evaluate` | apr-qa-gen |
| garbage-oracle | layout_implication | `GarbageOracle::evaluate` | apr-qa-gen |

## Quality Gate

### Makefile

```bash
make contract-lint          # Runs as part of `make check`
make contract-lint-trend    # With trend snapshot for drift detection
```

### CLI

```bash
# Full lint (validate + audit + score)
pv lint contracts/ --min-score 0.40 --binding contracts/binding.yaml

# Individual contract score
pv score contracts/gateway-contract-v1.yaml --binding contracts/binding.yaml

# Validate schema only
pv validate contracts/gateway-contract-v1.yaml
```

### CI

The `contract-lint` job in `.github/workflows/ci.yml` installs
`provable-contracts-cli` and runs `pv lint` on every push and PR.

## Configuration

Project-level settings in `.pv.toml`:

```toml
[lint]
min_score = 0.60
contracts_dir = "contracts/"
binding = "contracts/binding.yaml"

[lint.rules]
# Kani not yet integrated
PV-PRV-001 = "info"
PV-PRV-002 = "info"
PV-PRV-003 = "info"
PV-PRV-004 = "info"
```

## Scoring Dimensions

`pv score` evaluates contracts across 5 dimensions:

| Dimension | Current | Notes |
|-----------|---------|-------|
| Spec | 0.80-0.85 | Equation and obligation completeness |
| Falsify | 1.00 | All obligations have falsification tests |
| Kani | 0.00 | Deferred — `registry: true` exempts from requirement |
| Lean | 0.00 | No formal proofs yet |
| Bind | 1.00 | All equations mapped to implementations |

## Forjar Provability Playbook

`forjar-provability.yaml` orchestrates a 10-resource provability pipeline:

```
build-pv → build-apr-qa → fmt-check ──┐
                          clippy ──────┤
                                       ├→ unit-tests → invariant-tests → dim-smoke-probe
                          docs-check ──┤                coverage-check
                          contract-lint┘
```

Run with: `forjar plan -f forjar-provability.yaml`
