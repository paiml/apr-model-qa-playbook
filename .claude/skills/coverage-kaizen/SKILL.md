---
name: coverage-kaizen
description: "Systematic coverage gap analysis and test writing for the APR Model QA Playbook. Uses pmat query --coverage-gaps for highest-ROI targets, provides test patterns for each crate (proptest for gen, Evidence assertions for runner, MQS scoring for report), handles CommandRunner trait extensions, clippy pedantic/nursery landmines, and ExecutionConfig construction. Target: 95% library coverage."
disable-model-invocation: false
user-invocable: true
allowed-tools: "Read, Grep, Glob, Bash"
argument-hint: "target: crate name (apr-qa-gen, apr-qa-runner, apr-qa-report, apr-qa-certify), function name, or coverage goal (e.g., 96%)"
---

# Coverage Kaizen

Continuous improvement workflow for maintaining >= 95% library test coverage across the APR Model QA Playbook workspace.

## Quick Start

### Find Coverage Gaps

```bash
# Top coverage gaps ranked by ROI (MANDATORY: always use pmat query)
pmat query --coverage-gaps --rank-by impact --limit 20 --exclude-tests

# Coverage gaps for a specific crate
pmat query --coverage-gaps --limit 30 --exclude-file "tests" | grep "apr-qa-runner"

# Current coverage percentage
cargo llvm-cov --workspace --lib 2>&1 | grep "^TOTAL"
```

### Verify Compliance

```bash
# PMAT compliance check (>= 95%)
make coverage-check

# Or manually
./scripts/coverage-check.sh

# Full HTML coverage report
make coverage
# Opens: target/llvm-cov/html/index.html
```

### Coverage Commands

| Command | What It Does |
|---------|-------------|
| `make coverage` | HTML report (library code only) |
| `make coverage-summary` | Terminal summary |
| `make coverage-check` | Verify >= 95% threshold |
| `cargo llvm-cov --workspace --lib` | Raw coverage data |
| `cargo llvm-cov --workspace --lib --html` | HTML with source annotation |

**Never use `cargo tarpaulin`.** It's slow, unreliable, and causes hangs.

## Kaizen Workflow

### Step 1: Identify Targets

```bash
pmat query --coverage-gaps --rank-by impact --limit 20 --exclude-tests
```

Pick the function with the highest `impact_score` first (best ROI per test written).

### Step 2: Read the Function

```bash
# Use pmat query to read source (NOT cat/Read)
pmat query "function_name" --include-source --limit 1
```

### Step 3: Write Tests

Follow the crate-specific patterns below. Key rules:
- Tests follow Popperian falsification (design to fail, not to pass)
- Use `Evidence::corroborated()` (4 args) and `Evidence::falsified()` (5 args)
- Use `..Default::default()` for `ExecutionConfig` in tests
- Allow `clippy::unwrap_used` and `clippy::expect_used` in test code (already `cfg_attr`-allowed)

### Step 4: Verify Improvement

```bash
# Re-check coverage
pmat query --coverage-gaps --rank-by impact --limit 20 --exclude-tests

# Verify threshold
make coverage-check
```

### Step 5: Run Full Gate

```bash
make check   # fmt-check + lint + test + docs-check
```

## Crate-Specific Test Patterns

### apr-qa-gen (Scenario Generation)

**Key types:** `QaScenario`, `Oracle`, `OracleResult`, `ModelId`, `Modality`, `Backend`, `Format`

**Pattern: Proptest strategies**
```rust
use proptest::prelude::*;
use crate::proptest_impl::*;

proptest! {
    #[test]
    fn scenario_always_has_valid_id(scenario in scenario_strategy()) {
        prop_assert!(!scenario.id.is_empty());
        prop_assert!(scenario.id.contains('_'));
    }
}
```

**Pattern: Oracle evaluation**
```rust
#[test]
fn arithmetic_oracle_correct_addition() {
    let oracle = ArithmeticOracle::new();
    let result = oracle.evaluate("What is 3+4?", "The answer is 7.");
    assert!(matches!(result, OracleResult::Corroborated { .. }));
}

#[test]
fn garbage_oracle_detects_repetition() {
    let oracle = GarbageOracle::new();
    let result = oracle.evaluate("test", "abcabcabcabcabcabcabcabcabc");
    assert!(matches!(result, OracleResult::Falsified { .. }));
}
```

**Pattern: Oracle selection**
```rust
#[test]
fn selects_arithmetic_for_math_prompt() {
    let oracle = select_oracle("What is 5+3?");
    assert_eq!(oracle.name(), "arithmetic");
}
```

**Available proptest strategies:**
- `model_id_strategy()` - Random model IDs from supported families
- `modality_strategy()` - Run/Chat/Serve
- `backend_strategy()` - Cpu/Gpu
- `format_strategy()` - Gguf/SafeTensors/Apr
- `arithmetic_prompt_strategy()` - Verifiable math prompts
- `code_prompt_strategy()` - Code completion prompts
- `edge_case_prompt_strategy()` - Empty, unicode, XSS, SQL injection
- `any_prompt_strategy()` - Weighted combination
- `scenario_strategy()` - Complete random scenarios
- `temperature_strategy()` - 0.0, 0.7, 1.0, or random
- `max_tokens_strategy()` - 1, 32, 128, 512, 2048, or random

### apr-qa-runner (Execution Engine)

**Key types:** `Evidence`, `Outcome`, `EvidenceCollector`, `Executor`, `ExecutionConfig`, `CommandRunner`

**Pattern: Evidence construction**
```rust
use crate::evidence::{Evidence, Outcome};
use apr_qa_gen::scenario::QaScenario;

fn test_scenario() -> QaScenario {
    QaScenario::new(
        ModelId::new("test/model"),
        Modality::Run,
        Backend::Cpu,
        Format::Gguf,
        "What is 2+2?".to_string(),
        42,
    )
}

#[test]
fn corroborated_evidence_is_pass() {
    let e = Evidence::corroborated("F-QUAL-001", test_scenario(), "output", 100);
    assert!(e.outcome.is_pass());
    assert_eq!(e.reason, "Test passed");
    assert_eq!(e.exit_code, Some(0));
}

#[test]
fn falsified_evidence_is_fail() {
    let e = Evidence::falsified("F-QUAL-001", test_scenario(), "bad output", "", 100);
    assert!(e.outcome.is_fail());
}
```

**Pattern: EvidenceCollector**
```rust
#[test]
fn collector_counts_outcomes() {
    let mut collector = EvidenceCollector::new();
    collector.add(Evidence::corroborated("F-001", test_scenario(), "", 0));
    collector.add(Evidence::falsified("F-002", test_scenario(), "fail", "", 0));
    assert_eq!(collector.pass_count(), 1);
    assert_eq!(collector.fail_count(), 1);
    assert_eq!(collector.total(), 2);
}
```

**Pattern: ExecutionConfig construction in tests**
```rust
#[test]
fn executor_respects_timeout() {
    let config = ExecutionConfig {
        default_timeout_ms: 5000,
        dry_run: true,
        ..Default::default()
    };
    let mut executor = Executor::with_config(config);
    // ...
}
```

**Pattern: Custom CommandRunner for testing**

When you need to test executor behavior with controlled subprocess responses:

```rust
struct MyTestRunner;

impl CommandRunner for MyTestRunner {
    fn run_inference(&self, model_path: &Path, prompt: &str,
                     max_tokens: u32, no_gpu: bool, extra_args: &[&str]) -> CommandOutput {
        CommandOutput::success("The answer is 4.\nCompleted in 100ms")
    }

    fn convert_model(&self, _source: &Path, _target: &Path) -> CommandOutput {
        CommandOutput::success("")
    }

    // MUST implement ALL 28 methods - see CommandRunner Trait section below
    // Most can stub with CommandOutput::success("")
    fn inspect_model(&self, _: &Path) -> CommandOutput { CommandOutput::success("") }
    fn validate_model(&self, _: &Path) -> CommandOutput { CommandOutput::success("") }
    // ... (all 28 methods)
}

#[test]
fn test_with_custom_runner() {
    let config = ExecutionConfig::default();
    let runner = Arc::new(MyTestRunner);
    let mut executor = Executor::with_runner(config, runner);
    // ...
}
```

### apr-qa-report (Scoring & Reports)

**Key types:** `MqsScore`, `MqsCalculator`, `GatewayResult`, `CategoryScores`

**Pattern: MQS scoring**
```rust
use crate::mqs::MqsCalculator;
use apr_qa_runner::evidence::{Evidence, EvidenceCollector};

#[test]
fn perfect_score_all_corroborated() {
    let mut collector = EvidenceCollector::new();
    collector.add(Evidence::corroborated("F-QUAL-001", scenario(), "ok", 100));
    collector.add(Evidence::corroborated("F-PERF-001", scenario(), "ok", 100));
    let score = MqsCalculator::calculate("test/model", collector.all());
    assert!(score.gateways_passed);
    assert!(score.raw_score > 0);
}

#[test]
fn gateway_failure_zeroes_score() {
    let mut collector = EvidenceCollector::new();
    collector.add(Evidence::crashed("G1-LOAD-001", scenario(), "segfault", 139, 100));
    let score = MqsCalculator::calculate("test/model", collector.all());
    assert!(!score.gateways_passed);
    assert_eq!(score.raw_score, 0);
}
```

**Pattern: JUnit report generation**
```rust
#[test]
fn junit_report_valid_xml() {
    let collector = build_test_collector();
    let xml = junit::generate_report("test/model", collector.all());
    assert!(xml.starts_with("<?xml"));
    assert!(xml.contains("<testsuites"));
}
```

**Pattern: Grade assertions**

Be careful with float comparison - clippy `float_cmp` is strict. Use ranges:
```rust
// WRONG (clippy::float_cmp)
assert_eq!(score.normalized_score, 95.0);

// CORRECT
assert!(score.normalized_score >= 90.0);
assert!(score.normalized_score <= 100.0);
```

### apr-qa-certify (Certification Tracking)

**Key types:** `ModelCertification`, `CertificationStatus`, `SizeCategory`

**Pattern: CSV parsing**
```rust
#[test]
fn parse_csv_round_trip() {
    let models = vec![ModelCertification { /* ... */ }];
    let csv = write_csv(&models);
    let parsed = parse_csv(&csv).unwrap();
    assert_eq!(parsed.len(), models.len());
    assert_eq!(parsed[0].model_id, models[0].model_id);
}
```

**Pattern: README table generation**
```rust
#[test]
fn generated_table_has_headers() {
    let models = vec![sample_model()];
    let table = generate_table(&models);
    assert!(table.contains("| Model |"));
    assert!(table.contains("| Status |"));
}
```

## CommandRunner Trait (28 Methods)

When implementing a custom `CommandRunner` for tests, you MUST implement all 28 methods. There are currently 4 custom implementations in `executor.rs` tests that serve as reference.

**Complete method list:**

| # | Method | Signature |
|---|--------|-----------|
| 1 | `run_inference` | `(&self, model: &Path, prompt: &str, max_tokens: u32, no_gpu: bool, extra_args: &[&str]) -> CommandOutput` |
| 2 | `convert_model` | `(&self, source: &Path, target: &Path) -> CommandOutput` |
| 3 | `inspect_model` | `(&self, model: &Path) -> CommandOutput` |
| 4 | `validate_model` | `(&self, model: &Path) -> CommandOutput` |
| 5 | `bench_model` | `(&self, model: &Path) -> CommandOutput` |
| 6 | `check_model` | `(&self, model: &Path) -> CommandOutput` |
| 7 | `profile_model` | `(&self, model: &Path, warmup: u32, measure: u32) -> CommandOutput` |
| 8 | `profile_ci` | `(&self, model: &Path, min_throughput: Option<f64>, max_p99: Option<f64>, warmup: u32, measure: u32) -> CommandOutput` |
| 9 | `diff_tensors` | `(&self, model_a: &Path, model_b: &Path, json: bool) -> CommandOutput` |
| 10 | `compare_inference` | `(&self, model_a: &Path, model_b: &Path, prompt: &str, max_tokens: u32, tolerance: f64) -> CommandOutput` |
| 11 | `profile_with_flamegraph` | `(&self, model: &Path, output: &Path, no_gpu: bool) -> CommandOutput` |
| 12 | `profile_with_focus` | `(&self, model: &Path, focus: &str, no_gpu: bool) -> CommandOutput` |
| 13 | `validate_model_strict` | `(&self, model: &Path) -> CommandOutput` |
| 14 | `fingerprint_model` | `(&self, model: &Path, json: bool) -> CommandOutput` |
| 15 | `validate_stats` | `(&self, fp_a: &Path, fp_b: &Path) -> CommandOutput` |
| 16 | `pull_model` | `(&self, hf_repo: &str) -> CommandOutput` |
| 17 | `inspect_model_json` | `(&self, model: &Path) -> CommandOutput` |
| 18 | `run_ollama_inference` | `(&self, model_tag: &str, prompt: &str, temperature: f64) -> CommandOutput` |
| 19 | `pull_ollama_model` | `(&self, model_tag: &str) -> CommandOutput` |
| 20 | `create_ollama_model` | `(&self, model_tag: &str, modelfile: &Path) -> CommandOutput` |
| 21 | `serve_model` | `(&self, model: &Path, port: u16) -> CommandOutput` |
| 22 | `http_get` | `(&self, url: &str) -> CommandOutput` |
| 23 | `profile_memory` | `(&self, model: &Path) -> CommandOutput` |
| 24 | `run_chat` | `(&self, model: &Path, prompt: &str, no_gpu: bool, extra_args: &[&str]) -> CommandOutput` |
| 25 | `http_post` | `(&self, url: &str, body: &str) -> CommandOutput` |
| 26 | `spawn_serve` | `(&self, model: &Path, port: u16, no_gpu: bool) -> CommandOutput` |

**Stub template for most methods:**
```rust
fn method_name(&self, /* args */) -> CommandOutput {
    CommandOutput::success("")
}
```

**When adding a new method to the trait:** You must update ALL 4 custom implementations in `executor.rs` tests plus the `MockCommandRunner` in `command.rs`.

## Clippy Landmine Reference

The workspace uses `clippy::pedantic` + `clippy::nursery` + strict custom rules. These are the lints that most commonly trip up new test code.

### Workspace-Level Denials

| Lint | Level | Impact |
|------|-------|--------|
| `unsafe_code` | **deny** | No unsafe anywhere, `#![forbid(unsafe_code)]` in lib.rs files |
| `unwrap_used` | **deny** | No `.unwrap()` in library code (allowed in tests via `cfg_attr`) |
| `panic` | **deny** | No `panic!()` in library code |
| `expect_used` | warn | Prefer `map_err` / `?` over `.expect()` |

### Common Pedantic/Nursery Traps

| Lint | What Triggers It | Fix |
|------|-----------------|-----|
| `float_cmp` | `assert_eq!(f64, f64)` | Use range: `assert!(x >= 0.9 && x <= 1.0)` |
| `option_if_let_else` | `if let Some(x) = opt { a } else { b }` | Use `opt.map_or(b, \|x\| a)` |
| `manual_let_else` | `let x = match opt { Some(v) => v, None => return }` | Use `let Some(x) = opt else { return };` |
| `doc_link_with_quotes` | `/// See ["quoted"]` in doc comments | Wrap in backticks: `` `"quoted"` `` |
| `or_fun_call` | `.unwrap_or(String::new())` | Use `.unwrap_or_default()` |
| `cast_precision_loss` | `x as f64` when x is u64 | Already `#![allow]`'d in most crates |
| `cast_possible_truncation` | `x as u32` when x is u64 | Already `#![allow]`'d in runner |
| `missing_const_for_fn` | Pure function without `const` | Already `#![allow]`'d in all crates |
| `struct_excessive_bools` | Struct with many bool fields | Already `#![allow]`'d on `ExecutionConfig` |
| `too_many_lines` | Function > 100 lines | Add `#[allow(clippy::too_many_lines)]` |
| `too_many_arguments` | Function with > 7 args | Add `#[allow(clippy::too_many_arguments)]` |
| `needless_pass_by_value` | `fn f(s: String)` when `&str` works | Already `#![allow]`'d in most crates |
| `doc_markdown` | Unlinked type names in docs (`HuggingFace`) | Already `#![allow]`'d in most crates |
| `suboptimal_flops` | `a * b + c` instead of `a.mul_add(b, c)` | Already `#![allow]`'d in report |

### Test-Specific Allowances

These are already allowed in `#[cfg(test)]` blocks via `cfg_attr`:
- `clippy::unwrap_used` - OK to unwrap in tests
- `clippy::expect_used` - OK to expect in tests
- `clippy::redundant_closure_for_method_calls`
- `clippy::redundant_clone`
- `clippy::float_cmp` (only in apr-qa-report)
- `clippy::uninlined_format_args` (only in apr-qa-runner)
- `clippy::cast_sign_loss` (only in apr-qa-runner)

### Per-Crate Allow Lists

Each crate has specific `#![allow(...)]` in its `lib.rs`. Check the relevant `lib.rs` before writing tests to know which lints are pre-allowed.

**Most restrictive:** `apr-qa-certify` (almost no allows)
**Most lenient:** `apr-qa-report` (20+ allows for scoring math)

## Evidence Constructor Cheat Sheet

```
corroborated(gate_id, scenario, output, duration_ms)     → 4 args, reason="Test passed"
falsified(gate_id, scenario, reason, output, duration_ms) → 5 args
timeout(gate_id, scenario, timeout_ms)                    → 3 args
crashed(gate_id, scenario, stderr, exit_code, duration_ms)→ 5 args
skipped(gate_id, scenario, reason)                        → 3 args
```

**Key facts:**
- `Evidence.output` is `String`, NOT `Option<String>`
- `Evidence.stderr` is `Option<String>` (only `Some` for `Crashed`)
- `Evidence.exit_code` is `Option<i32>` (Some(0) for corroborated, None for falsified)
- Constructor uses `impl Into<String>` so both `&str` and `String` work

## Gate ID Conventions

Gate IDs map to MQS categories via prefix:

| Prefix | Category | Max Points |
|--------|----------|-----------|
| `F-QUAL-*` | Quality | 200 |
| `F-PERF-*` | Performance | 150 |
| `F-STAB-*` | Stability | 200 |
| `F-COMP-*` | Compatibility | 150 |
| `F-EDGE-*` | Edge Cases | 150 |
| `F-REGR-*` | Regression | 150 |
| `F-CONV-*` | Compatibility (conversion) | 150 |
| `F-CONV-RT*` | Regression (round-trip) | 150 |
| `F-CONTRACT-*` | Compatibility (contract) | 150 |
| `G0-*` | Stability (integrity) | 200 |
| `G1-*` through `G4-*` | Gateway (zeroes all) | - |

When writing tests, use `F-{CATEGORY}-{NNN}` format for gate IDs to ensure correct MQS category scoring.

## Common Test Count Gotcha

When adding new test phases to the executor's `execute()` method (like contract tests, parity tests):

1. **Existing tests that assert `total_scenarios` counts will break** because the executor now runs more tests
2. **Fix:** Update the expected counts in affected tests
3. **Prevention:** Search for `total_scenarios` assertions before adding phases:
   ```bash
   pmat query --literal "total_scenarios" --exclude-tests --limit 10
   ```

## ExecutionConfig Construction

**In library code (`apr-qa-cli/src/lib.rs`):** Constructed explicitly field-by-field, NO `..Default::default()`. When adding a new field, you must update both construction sites in `lib.rs`.

**In test code:** Use `..Default::default()`:
```rust
let config = ExecutionConfig {
    dry_run: true,
    default_timeout_ms: 5000,
    ..Default::default()
};
```

**Current field count:** 21 fields. Check `executor.rs` line ~81 for the latest.

## See Also

### References
- [test-patterns.md](references/test-patterns.md) - Extended test pattern cookbook
- [clippy-landmines.md](references/clippy-landmines.md) - Full clippy configuration reference

### Commands
```bash
make test           # Run all tests
make lint           # Clippy with zero warnings
make check          # Full gate: fmt + lint + test + docs
make coverage       # HTML coverage report
make coverage-check # Verify >= 95%
```

### Key Files
| File | Purpose |
|------|---------|
| `Cargo.toml` (root) | Workspace lint configuration |
| `crates/*/src/lib.rs` | Per-crate allow lists |
| `crates/apr-qa-runner/src/command.rs` | CommandRunner trait (28 methods) |
| `crates/apr-qa-runner/src/executor.rs` | ExecutionConfig + 4 test runners |
| `crates/apr-qa-runner/src/evidence.rs` | Evidence constructors |
| `scripts/coverage-check.sh` | 95% threshold check |
