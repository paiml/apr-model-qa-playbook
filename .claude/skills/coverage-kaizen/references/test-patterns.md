# Test Pattern Cookbook

Extended test patterns for each crate in the APR Model QA Playbook workspace.

## Common Test Helpers

### Creating Test Scenarios

```rust
use apr_qa_gen::scenario::{QaScenario, Modality, Backend, Format};
use apr_qa_gen::models::ModelId;

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

fn gpu_scenario() -> QaScenario {
    QaScenario::new(
        ModelId::new("test/model"),
        Modality::Run,
        Backend::Gpu,
        Format::SafeTensors,
        "What is 3+4?".to_string(),
        42,
    )
}

fn chat_scenario() -> QaScenario {
    QaScenario::new(
        ModelId::new("test/model"),
        Modality::Chat,
        Backend::Cpu,
        Format::Apr,
        "Hello".to_string(),
        42,
    )
}
```

### Building Evidence Collections

```rust
use apr_qa_runner::evidence::{Evidence, EvidenceCollector};

fn build_mixed_collector() -> EvidenceCollector {
    let mut collector = EvidenceCollector::new();

    // 3 passes
    collector.add(Evidence::corroborated("F-QUAL-001", test_scenario(), "ok", 100));
    collector.add(Evidence::corroborated("F-PERF-001", test_scenario(), "ok", 200));
    collector.add(Evidence::corroborated("F-STAB-001", test_scenario(), "ok", 150));

    // 1 failure
    collector.add(Evidence::falsified(
        "F-EDGE-001", test_scenario(), "Unexpected output", "garbage", 300
    ));

    collector
}

fn build_all_pass_collector(n: usize) -> EvidenceCollector {
    let mut collector = EvidenceCollector::new();
    for i in 0..n {
        collector.add(Evidence::corroborated(
            format!("F-QUAL-{i:03}"),
            test_scenario(),
            "output",
            100,
        ));
    }
    collector
}

fn build_gateway_failure_collector() -> EvidenceCollector {
    let mut collector = EvidenceCollector::new();
    collector.add(Evidence::crashed(
        "G1-LOAD-001",
        test_scenario(),
        "segmentation fault",
        139,
        50,
    ));
    collector
}
```

## apr-qa-gen Patterns

### Testing Oracle Correctness

```rust
use crate::oracle::{ArithmeticOracle, GarbageOracle, CodeSyntaxOracle, Oracle, OracleResult};

#[test]
fn arithmetic_correct_operations() {
    let oracle = ArithmeticOracle::new();

    // Addition
    let r = oracle.evaluate("What is 3+4?", "The answer is 7.");
    assert!(matches!(r, OracleResult::Corroborated { .. }));

    // Multiplication
    let r = oracle.evaluate("Calculate 7*8", "56 is the result");
    assert!(matches!(r, OracleResult::Corroborated { .. }));

    // Wrong answer
    let r = oracle.evaluate("What is 5+3?", "The answer is 9.");
    assert!(matches!(r, OracleResult::Falsified { .. }));
}

#[test]
fn garbage_oracle_all_checks() {
    let oracle = GarbageOracle::new();

    // Empty output
    assert!(matches!(oracle.evaluate("test", ""), OracleResult::Falsified { .. }));

    // Control characters
    assert!(matches!(oracle.evaluate("test", "hello\x00world"), OracleResult::Falsified { .. }));

    // NaN
    assert!(matches!(oracle.evaluate("test", "result is NaN"), OracleResult::Falsified { .. }));

    // Replacement character U+FFFD
    assert!(matches!(oracle.evaluate("test", "hello\u{FFFD}world"), OracleResult::Falsified { .. }));

    // Repetitive pattern
    assert!(matches!(
        oracle.evaluate("test", "abcabcabcabcabcabcabcabcabc"),
        OracleResult::Falsified { .. }
    ));

    // Valid output
    assert!(matches!(
        oracle.evaluate("test", "This is a normal response."),
        OracleResult::Corroborated { .. }
    ));
}
```

### Testing Proptest Strategies

```rust
use proptest::prelude::*;
use crate::proptest_impl::*;

proptest! {
    #[test]
    fn all_scenarios_have_nonempty_id(scenario in scenario_strategy()) {
        prop_assert!(!scenario.id.is_empty());
    }

    #[test]
    fn arithmetic_prompts_contain_operator(prompt in arithmetic_prompt_strategy()) {
        prop_assert!(
            prompt.contains('+') || prompt.contains('-') || prompt.contains('*')
        );
    }

    #[test]
    fn format_display_roundtrip(format in format_strategy()) {
        let s = format.to_string();
        prop_assert!(!s.is_empty());
    }
}
```

### Testing Oracle Selection

```rust
use crate::oracle::select_oracle;

#[test]
fn oracle_selection_by_prompt_type() {
    assert_eq!(select_oracle("What is 2+2?").name(), "arithmetic");
    assert_eq!(select_oracle("def fibonacci(n):").name(), "code_syntax");
    assert_eq!(select_oracle("Hello world").name(), "garbage");
    assert_eq!(select_oracle("fn main() {").name(), "code_syntax");
    assert_eq!(select_oracle("Calculate 7*8").name(), "arithmetic");
}
```

## apr-qa-runner Patterns

### Testing Evidence Methods

```rust
use crate::evidence::{Evidence, Outcome};

#[test]
fn evidence_metadata() {
    let mut e = Evidence::corroborated("F-001", test_scenario(), "ok", 100);
    e.add_metadata("format", "gguf");
    e.add_metadata("backend", "cpu");
    assert_eq!(e.metadata.get("format"), Some(&"gguf".to_string()));
}

#[test]
fn evidence_collector_json_roundtrip() {
    let mut collector = EvidenceCollector::new();
    collector.add(Evidence::corroborated("F-001", test_scenario(), "ok", 100));
    let json = collector.to_json().unwrap();
    assert!(json.contains("F-001"));
    assert!(json.contains("Corroborated"));
}

#[test]
fn collector_failures_filter() {
    let mut collector = EvidenceCollector::new();
    collector.add(Evidence::corroborated("F-001", test_scenario(), "", 0));
    collector.add(Evidence::falsified("F-002", test_scenario(), "bad", "", 0));
    collector.add(Evidence::crashed("F-003", test_scenario(), "segfault", 139, 0));

    let failures = collector.failures();
    assert_eq!(failures.len(), 2);  // falsified + crashed
}
```

### Testing Executor with Mock Runner

```rust
use crate::executor::{Executor, ExecutionConfig, FailurePolicy};
use crate::command::MockCommandRunner;
use std::sync::Arc;

#[test]
fn executor_dry_run_produces_no_evidence() {
    let config = ExecutionConfig {
        dry_run: true,
        ..Default::default()
    };
    let runner = Arc::new(MockCommandRunner::new());
    let mut executor = Executor::with_runner(config, runner);
    let playbook = load_test_playbook();
    let result = executor.execute(&playbook).unwrap();
    assert_eq!(result.collector.total(), 0);
}
```

### Testing Contract Validation

```rust
use crate::contract::{FormatContract, validate_invariant};

#[test]
fn tensor_naming_valid_patterns() {
    let contract = FormatContract::load_default().unwrap();
    assert!(contract.tensor_naming.is_valid("0.q_proj.weight"));
    assert!(contract.tensor_naming.is_valid("token_embd.weight"));
    assert!(!contract.tensor_naming.is_valid("model.layers.0.self_attn.q_proj.weight"));
}
```

## apr-qa-report Patterns

### Testing MQS Scoring

```rust
use crate::mqs::MqsCalculator;

#[test]
fn all_pass_gives_max_score() {
    let collector = build_all_pass_collector(18);
    let score = MqsCalculator::calculate("test/model", collector.all());
    assert!(score.gateways_passed);
    assert_eq!(score.raw_score, 1000);
    assert!(score.normalized_score >= 99.0);
    assert_eq!(score.grade, "A+");
}

#[test]
fn crash_incurs_penalty() {
    let mut collector = EvidenceCollector::new();
    collector.add(Evidence::corroborated("F-QUAL-001", test_scenario(), "", 0));
    collector.add(Evidence::crashed("F-QUAL-002", test_scenario(), "err", 1, 0));
    let score = MqsCalculator::calculate("test/model", collector.all());
    assert!(score.total_penalty > 0);
}

#[test]
fn gateway_failure_zeroes_everything() {
    let collector = build_gateway_failure_collector();
    let score = MqsCalculator::calculate("test/model", collector.all());
    assert!(!score.gateways_passed);
    assert_eq!(score.raw_score, 0);
    assert_eq!(score.grade, "F");
}
```

### Testing Report Generation

```rust
#[test]
fn html_report_contains_model_id() {
    let collector = build_mixed_collector();
    let html = html::generate_report("test/model", collector.all());
    assert!(html.contains("test/model"));
    assert!(html.contains("<html"));
}

#[test]
fn junit_report_valid_structure() {
    let collector = build_mixed_collector();
    let xml = junit::generate_report("test/model", collector.all());
    assert!(xml.starts_with("<?xml"));
    assert!(xml.contains("<testsuite"));
    assert!(xml.contains("<testcase"));
}

#[test]
fn markdown_report_has_summary() {
    let collector = build_mixed_collector();
    let md = markdown::generate_report("test/model", collector.all());
    assert!(md.contains("# "));  // has headers
    assert!(md.contains("test/model"));
}
```

### Float Comparison in Score Tests

```rust
// WRONG - triggers clippy::float_cmp
assert_eq!(score.normalized_score, 95.0);

// CORRECT - use epsilon comparison
assert!((score.normalized_score - 95.0).abs() < f64::EPSILON);

// ALSO CORRECT - use range
assert!(score.normalized_score >= 90.0);
assert!(score.normalized_score <= 100.0);

// BEST - apr-qa-report allows float_cmp in tests via cfg_attr
// But range assertions are clearer for score thresholds
```

## apr-qa-certify Patterns

### Testing CSV Round-trip

```rust
use crate::{ModelCertification, CertificationStatus, SizeCategory, parse_csv, write_csv};

fn sample_model() -> ModelCertification {
    ModelCertification {
        model_id: "test/model".to_string(),
        family: "test".to_string(),
        parameters: "1.5B".to_string(),
        size_category: SizeCategory::Small,
        status: CertificationStatus::Certified,
        mqs_score: 950,
        grade: "A".to_string(),
        certified_tier: "mvp".to_string(),
        last_certified: Some(Utc::now()),
        g1: true, g2: true, g3: true, g4: true,
        tps_gguf_cpu: Some(17.9),
        tps_gguf_gpu: Some(129.8),
        tps_apr_cpu: Some(16.2),
        tps_apr_gpu: Some(0.6),
        tps_st_cpu: Some(2.9),
        tps_st_gpu: Some(23.8),
        provenance_verified: true,
    }
}

#[test]
fn csv_roundtrip_preserves_data() {
    let models = vec![sample_model()];
    let csv = write_csv(&models);
    let parsed = parse_csv(&csv).unwrap();
    assert_eq!(parsed[0].model_id, "test/model");
    assert_eq!(parsed[0].mqs_score, 950);
    assert!(parsed[0].g1);
}
```

### Testing Table Generation

```rust
#[test]
fn table_includes_all_models() {
    let models = vec![sample_model(), another_model()];
    let table = generate_table(&models);
    assert!(table.contains("test/model"));
    assert!(table.contains("other/model"));
    // Verify markdown table structure
    assert!(table.contains("| "));
    assert!(table.contains(" | "));
}

#[test]
fn summary_shows_status_counts() {
    let models = vec![
        certified_model(),
        blocked_model(),
        pending_model(),
    ];
    let summary = generate_summary(&models, "2026-02-14");
    assert!(summary.contains("CERTIFIED"));
    assert!(summary.contains("BLOCKED"));
}
```

## Anti-Patterns to Avoid

### Don't Use unwrap() in Library Code
```rust
// WRONG (clippy::unwrap_used is denied)
let value = some_option.unwrap();

// CORRECT
let value = some_option.ok_or(Error::Execution("missing value".into()))?;

// OK in tests (cfg_attr allows it)
#[test]
fn my_test() {
    let value = some_option.unwrap();  // Fine in #[cfg(test)]
}
```

### Don't Compare Floats Directly
```rust
// WRONG (clippy::float_cmp)
assert_eq!(score, 0.95);

// CORRECT
assert!((score - 0.95).abs() < 1e-10);
```

### Don't Forget All 28 CommandRunner Methods
```rust
// WRONG - compile error
struct MyRunner;
impl CommandRunner for MyRunner {
    fn run_inference(&self, ...) -> CommandOutput { ... }
    // Missing 27 other methods!
}

// CORRECT - implement all 28 (stub the ones you don't need)
```

### Don't Hardcode Total Scenario Counts
```rust
// FRAGILE - breaks when executor adds new test phases
assert_eq!(result.total_scenarios, 18);

// BETTER - assert relative to expected
assert!(result.total_scenarios >= 18);
// Or assert specific counts by outcome
assert_eq!(result.collector.pass_count(), expected_passes);
```
