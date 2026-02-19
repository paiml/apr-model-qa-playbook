
use super::*;
use crate::command::MockCommandRunner;
use apr_qa_gen::{Backend, Format, Modality, ModelId, QaScenario};

fn test_scenario() -> QaScenario {
    QaScenario::new(
        ModelId::new("test", "model"),
        Modality::Run,
        Backend::Cpu,
        Format::Gguf,
        "2+2=".to_string(),
        42,
    )
}

fn test_playbook() -> Playbook {
    let yaml = r#"
name: test-playbook
version: "1.0.0"
model:
  hf_repo: "test/model"
  formats: [gguf]
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 5
"#;
    Playbook::from_yaml(yaml).expect("Failed to parse")
}

/// Create a temp file (file mode) for testing.
/// Returns (tempdir, file_path_string) - keep tempdir alive for test duration.
fn create_test_model_file(format: Format) -> (tempfile::TempDir, String) {
    let tmp = tempfile::tempdir().unwrap();
    let filename = match format {
        Format::Gguf => "model.gguf",
        Format::Apr => "model.apr",
        Format::SafeTensors => "model.safetensors",
    };
    let file_path = tmp.path().join(filename);
    std::fs::write(&file_path, b"fake model data").unwrap();
    let path = file_path.to_string_lossy().to_string();
    (tmp, path)
}

#[test]
fn test_executor_dry_run() {
    let mock_runner = MockCommandRunner::new();
    let config = ExecutionConfig {
        dry_run: true,
        ..Default::default()
    };
    let mut executor = Executor::with_runner(config, Arc::new(mock_runner));
    let playbook = test_playbook();

    let result = executor.execute(&playbook).expect("Execution failed");

    assert_eq!(result.skipped, 5);
    // G0-PULL passes even in dry run (pull still happens)
    assert!(result.passed >= 1);
}

#[test]
fn test_execution_result_pass_rate() {
    let result = ExecutionResult {
        playbook_name: "test".to_string(),
        total_scenarios: 100,
        passed: 95,
        failed: 5,
        skipped: 0,
        duration_ms: 1000,
        gateway_failed: None,
        evidence: EvidenceCollector::new(),
    };

    assert!((result.pass_rate() - 95.0).abs() < f64::EPSILON);
}

#[test]
fn test_failure_policy_stop_on_first() {
    let config = ExecutionConfig {
        failure_policy: FailurePolicy::StopOnFirst,
        ..Default::default()
    };
    let executor = Executor::with_config(config);
    assert_eq!(executor.config.failure_policy, FailurePolicy::StopOnFirst);
}

#[test]
fn test_execution_config_default() {
    let config = ExecutionConfig::default();
    assert_eq!(config.failure_policy, FailurePolicy::StopOnP0);
    assert_eq!(config.default_timeout_ms, 60_000);
    assert_eq!(config.max_workers, 4);
    assert!(!config.dry_run);
}

#[test]
fn test_executor_default() {
    let executor = Executor::default();
    assert_eq!(executor.config.failure_policy, FailurePolicy::StopOnP0);
}

#[test]
fn test_executor_evidence() {
    let executor = Executor::new();
    let evidence = executor.evidence();
    assert_eq!(evidence.all().len(), 0);
}

#[test]
fn test_execution_result_is_success() {
    let success = ExecutionResult {
        playbook_name: "test".to_string(),
        total_scenarios: 10,
        passed: 10,
        failed: 0,
        skipped: 0,
        duration_ms: 100,
        gateway_failed: None,
        evidence: EvidenceCollector::new(),
    };
    assert!(success.is_success());

    let with_failures = ExecutionResult {
        playbook_name: "test".to_string(),
        total_scenarios: 10,
        passed: 8,
        failed: 2,
        skipped: 0,
        duration_ms: 100,
        gateway_failed: None,
        evidence: EvidenceCollector::new(),
    };
    assert!(!with_failures.is_success());

    let with_gateway_failure = ExecutionResult {
        playbook_name: "test".to_string(),
        total_scenarios: 10,
        passed: 0,
        failed: 0,
        skipped: 0,
        duration_ms: 100,
        gateway_failed: Some("G1 failed".to_string()),
        evidence: EvidenceCollector::new(),
    };
    assert!(!with_gateway_failure.is_success());
}

#[test]
fn test_execution_result_pass_rate_zero() {
    let result = ExecutionResult {
        playbook_name: "test".to_string(),
        total_scenarios: 0,
        passed: 0,
        failed: 0,
        skipped: 0,
        duration_ms: 0,
        gateway_failed: None,
        evidence: EvidenceCollector::new(),
    };
    assert!((result.pass_rate() - 0.0).abs() < f64::EPSILON);
}

#[test]
fn test_failure_policy_default() {
    let policy = FailurePolicy::default();
    assert_eq!(policy, FailurePolicy::StopOnP0);
}

#[test]
fn test_failure_policy_debug() {
    let policy = FailurePolicy::CollectAll;
    let debug_str = format!("{policy:?}");
    assert!(debug_str.contains("CollectAll"));
}

#[test]
fn test_executor_with_collect_all_policy() {
    let config = ExecutionConfig {
        failure_policy: FailurePolicy::CollectAll,
        ..Default::default()
    };
    let executor = Executor::with_config(config);
    assert_eq!(executor.config.failure_policy, FailurePolicy::CollectAll);
}

#[test]
fn test_executor_with_stop_on_p0_policy() {
    let config = ExecutionConfig {
        failure_policy: FailurePolicy::StopOnP0,
        ..Default::default()
    };
    let executor = Executor::with_config(config);
    assert_eq!(executor.config.failure_policy, FailurePolicy::StopOnP0);
}

#[test]
fn test_executor_config_clone() {
    let config = ExecutionConfig::default();
    let cloned = config.clone();
    assert_eq!(cloned.failure_policy, config.failure_policy);
    assert_eq!(cloned.max_workers, config.max_workers);
}

#[test]
fn test_execution_result_clone() {
    let result = ExecutionResult {
        playbook_name: "test".to_string(),
        total_scenarios: 10,
        passed: 10,
        failed: 0,
        skipped: 0,
        duration_ms: 100,
        gateway_failed: None,
        evidence: EvidenceCollector::new(),
    };
    let cloned = result.clone();
    assert_eq!(cloned.playbook_name, result.playbook_name);
    assert_eq!(cloned.total_scenarios, result.total_scenarios);
}

#[test]
fn test_check_gateways() {
    let executor = Executor::new();
    let playbook = test_playbook();

    let result = executor.check_gateways(&playbook);
    assert!(result.is_ok());
}

#[test]
fn test_executor_debug() {
    let executor = Executor::new();
    let debug_str = format!("{executor:?}");
    assert!(debug_str.contains("Executor"));
}

#[test]
fn test_execution_config_debug() {
    let config = ExecutionConfig::default();
    let debug_str = format!("{config:?}");
    assert!(debug_str.contains("ExecutionConfig"));
}

#[test]
fn test_execution_result_debug() {
    let result = ExecutionResult {
        playbook_name: "test".to_string(),
        total_scenarios: 10,
        passed: 10,
        failed: 0,
        skipped: 0,
        duration_ms: 100,
        gateway_failed: None,
        evidence: EvidenceCollector::new(),
    };
    let debug_str = format!("{result:?}");
    assert!(debug_str.contains("ExecutionResult"));
}

#[test]
fn test_failure_policy_eq() {
    assert_eq!(FailurePolicy::StopOnFirst, FailurePolicy::StopOnFirst);
    assert_ne!(FailurePolicy::StopOnFirst, FailurePolicy::CollectAll);
}

#[test]
fn test_failure_policy_clone() {
    let policy = FailurePolicy::StopOnP0;
    let cloned = policy;
    assert_eq!(policy, cloned);
}

#[test]
fn test_failure_policy_fail_fast() {
    let policy = FailurePolicy::FailFast;
    assert!(policy.emit_diagnostic());
    assert!(policy.stops_on_any_failure());
}

#[test]
fn test_failure_policy_emit_diagnostic() {
    assert!(FailurePolicy::FailFast.emit_diagnostic());
    assert!(!FailurePolicy::StopOnFirst.emit_diagnostic());
    assert!(!FailurePolicy::StopOnP0.emit_diagnostic());
    assert!(!FailurePolicy::CollectAll.emit_diagnostic());
}

#[test]
fn test_failure_policy_stops_on_any_failure() {
    assert!(FailurePolicy::FailFast.stops_on_any_failure());
    assert!(FailurePolicy::StopOnFirst.stops_on_any_failure());
    assert!(!FailurePolicy::StopOnP0.stops_on_any_failure());
    assert!(!FailurePolicy::CollectAll.stops_on_any_failure());
}

#[test]
fn test_executor_custom_timeout() {
    let config = ExecutionConfig {
        default_timeout_ms: 30_000,
        ..Default::default()
    };
    let executor = Executor::with_config(config);
    assert_eq!(executor.config.default_timeout_ms, 30_000);
}

#[test]
fn test_executor_custom_workers() {
    let config = ExecutionConfig {
        max_workers: 8,
        ..Default::default()
    };
    let executor = Executor::with_config(config);
    assert_eq!(executor.config.max_workers, 8);
}

#[test]
fn test_tool_test_result_to_evidence_passed() {
    let result = ToolTestResult {
        tool: "inspect".to_string(),
        passed: true,
        exit_code: 0,
        stdout: "Model info...".to_string(),
        stderr: String::new(),
        duration_ms: 100,
        gate_id: "F-INSPECT-001".to_string(),
    };

    let model_id = ModelId::new("test", "model");
    let evidence = result.to_evidence(&model_id);

    assert!(evidence.outcome.is_pass());
    assert_eq!(evidence.gate_id, "F-INSPECT-001");
}

#[test]
fn test_tool_test_result_to_evidence_failed() {
    let result = ToolTestResult {
        tool: "validate".to_string(),
        passed: false,
        exit_code: 5,
        stdout: String::new(),
        stderr: "Validation failed".to_string(),
        duration_ms: 50,
        gate_id: "F-VALIDATE-001".to_string(),
    };

    let model_id = ModelId::new("test", "model");
    let evidence = result.to_evidence(&model_id);

    assert!(evidence.outcome.is_fail());
    assert!(!evidence.reason.is_empty());
}

#[test]
fn test_tool_test_result_clone() {
    let result = ToolTestResult {
        tool: "bench".to_string(),
        passed: true,
        exit_code: 0,
        stdout: "Benchmark output".to_string(),
        stderr: String::new(),
        duration_ms: 500,
        gate_id: "F-BENCH-001".to_string(),
    };

    let cloned = result.clone();
    assert_eq!(cloned.tool, result.tool);
    assert_eq!(cloned.passed, result.passed);
    assert_eq!(cloned.exit_code, result.exit_code);
}

#[test]
fn test_tool_test_result_debug() {
    let result = ToolTestResult {
        tool: "profile".to_string(),
        passed: true,
        exit_code: 0,
        stdout: String::new(),
        stderr: String::new(),
        duration_ms: 1000,
        gate_id: "F-PROFILE-001".to_string(),
    };

    let debug_str = format!("{result:?}");
    assert!(debug_str.contains("ToolTestResult"));
    assert!(debug_str.contains("profile"));
}

#[test]
fn test_tool_executor_new() {
    let executor = ToolExecutor::new("/path/to/model.gguf".to_string(), true, 60_000);
    assert!(executor.no_gpu);
}

#[test]
fn test_execution_config_no_gpu() {
    let config = ExecutionConfig {
        no_gpu: true,
        ..Default::default()
    };
    assert!(config.no_gpu);
}

#[test]
fn test_execution_config_conversion_tests() {
    // Default should have conversion tests enabled
    let config = ExecutionConfig::default();
    assert!(config.run_conversion_tests);

    // Can be disabled
    let config_disabled = ExecutionConfig {
        run_conversion_tests: false,
        ..Default::default()
    };
    assert!(!config_disabled.run_conversion_tests);
}

#[test]
fn test_execution_result_with_skipped() {
    let result = ExecutionResult {
        playbook_name: "test".to_string(),
        total_scenarios: 10,
        passed: 5,
        failed: 2,
        skipped: 3,
        duration_ms: 100,
        gateway_failed: None,
        evidence: EvidenceCollector::new(),
    };
    assert_eq!(result.skipped, 3);
    // Pass rate only considers executed (not skipped)
    let executed = result.passed + result.failed;
    assert_eq!(executed, 7);
}

#[test]
fn test_executor_config_method() {
    let executor = Executor::new();
    let config = executor.config();
    assert_eq!(config.failure_policy, FailurePolicy::StopOnP0);
}

#[test]
fn test_execution_config_differential_defaults() {
    let config = ExecutionConfig::default();
    // v1.3.0: Differential testing enabled by default
    assert!(config.run_differential_tests);
    assert!(config.run_trace_payload);
    // Profile CI disabled by default (only for CI pipelines)
    assert!(!config.run_profile_ci);
}

#[test]
fn test_execution_config_differential_custom() {
    let config = ExecutionConfig {
        run_differential_tests: false,
        run_profile_ci: true,
        run_trace_payload: false,
        ..Default::default()
    };
    assert!(!config.run_differential_tests);
    assert!(config.run_profile_ci);
    assert!(!config.run_trace_payload);
}

#[test]
fn test_parse_tps_from_output_valid() {
    let output = "Some text tok/s: 12.34 more text";
    let tps = Executor::parse_tps_from_output(output);
    assert!(tps.is_some());
    assert!((tps.unwrap() - 12.34).abs() < f64::EPSILON);
}

#[test]
fn test_parse_tps_from_output_with_whitespace() {
    let output = "tok/s:   45.67";
    let tps = Executor::parse_tps_from_output(output);
    assert!(tps.is_some());
    assert!((tps.unwrap() - 45.67).abs() < f64::EPSILON);
}

#[test]
fn test_parse_tps_from_output_integer() {
    let output = "tok/s: 100";
    let tps = Executor::parse_tps_from_output(output);
    assert!(tps.is_some());
    assert!((tps.unwrap() - 100.0).abs() < f64::EPSILON);
}

#[test]
fn test_parse_tps_from_output_not_found() {
    let output = "no tokens per second here";
    let tps = Executor::parse_tps_from_output(output);
    assert!(tps.is_none());
}

#[test]
fn test_parse_tps_from_output_empty() {
    let output = "";
    let tps = Executor::parse_tps_from_output(output);
    assert!(tps.is_none());
}

#[test]
fn test_parse_tps_from_output_invalid_number() {
    let output = "tok/s: abc";
    let tps = Executor::parse_tps_from_output(output);
    assert!(tps.is_none());
}

#[test]
fn test_extract_generated_text_simple() {
    let output = "Hello world\nThis is text";
    let result = Executor::extract_generated_text(output);
    assert_eq!(result, "Hello world\nThis is text");
}

#[test]
fn test_extract_generated_text_filters_separator() {
    let output = "Generated text\n=== BENCHMARK ===\nMore stuff";
    let result = Executor::extract_generated_text(output);
    assert!(!result.contains("==="));
    assert!(result.contains("Generated text"));
}

#[test]
fn test_extract_generated_text_filters_tps() {
    let output = "Hello world\ntok/s: 12.34\nAfter tps";
    let result = Executor::extract_generated_text(output);
    assert!(!result.contains("tok/s"));
    assert!(result.contains("Hello world"));
    assert!(result.contains("After tps"));
}

#[test]
fn test_extract_generated_text_empty() {
    let output = "";
    let result = Executor::extract_generated_text(output);
    assert!(result.is_empty());
}

#[test]
fn test_extract_generated_text_only_filtered() {
    let output = "=== START ===\ntok/s: 10\n=== END ===";
    let result = Executor::extract_generated_text(output);
    assert!(result.is_empty());
}

#[test]
fn test_extract_output_text_simple() {
    let output = "Some header\nOutput:\nThe answer is 4\nCompleted in 1.2s";
    let result = Executor::extract_output_text(output);
    assert_eq!(result, "The answer is 4");
}

#[test]
fn test_extract_output_text_multiline() {
    let output = "Header\nOutput:\nLine 1\nLine 2\nLine 3\nCompleted in 1s";
    let result = Executor::extract_output_text(output);
    assert_eq!(result, "Line 1 Line 2 Line 3");
}

#[test]
fn test_extract_output_text_no_output_marker() {
    let output = "Just some text without Output marker";
    let result = Executor::extract_output_text(output);
    assert!(result.is_empty());
}

#[test]
fn test_extract_output_text_empty() {
    let output = "";
    let result = Executor::extract_output_text(output);
    assert!(result.is_empty());
}

#[test]
fn test_extract_output_text_empty_output() {
    let output = "Header\nOutput:\nCompleted in 1s";
    let result = Executor::extract_output_text(output);
    assert!(result.is_empty());
}

#[test]
fn test_extract_output_text_stops_at_empty_line() {
    let output = "Header\nOutput:\nThe answer\n\nMore text after blank";
    let result = Executor::extract_output_text(output);
    assert_eq!(result, "The answer");
}

#[test]
fn test_golden_scenario_creation() {
    let model_id = ModelId::new("test", "model");
    let scenario = Executor::golden_scenario(&model_id);
    assert_eq!(scenario.model.org, "test");
    assert_eq!(scenario.model.name, "model");
    assert_eq!(scenario.modality, Modality::Run);
    assert_eq!(scenario.backend, Backend::Cpu);
    assert_eq!(scenario.format, Format::Apr);
    assert!(scenario.prompt.contains("Golden Rule"));
}

#[test]
fn test_execution_config_golden_rule_default() {
    let config = ExecutionConfig::default();
    assert!(config.run_golden_rule_test);
    assert!(config.golden_reference_path.is_none());
}

#[test]
fn test_execution_config_golden_rule_custom() {
    let config = ExecutionConfig {
        run_golden_rule_test: false,
        golden_reference_path: Some("/path/to/reference.json".to_string()),
        ..Default::default()
    };
    assert!(!config.run_golden_rule_test);
    assert_eq!(
        config.golden_reference_path.as_deref(),
        Some("/path/to/reference.json")
    );
}

#[test]
fn test_tool_executor_fields() {
    let executor = ToolExecutor::new("/path/model.gguf".to_string(), true, 30_000);
    assert_eq!(executor.model_path, "/path/model.gguf");
    assert!(executor.no_gpu);
    assert_eq!(executor.timeout_ms, 30_000);
}

#[test]
fn test_tool_executor_no_gpu_false() {
    let executor = ToolExecutor::new("model.gguf".to_string(), false, 60_000);
    assert!(!executor.no_gpu);
}

#[test]
fn test_tool_test_result_gate_id() {
    let result = ToolTestResult {
        tool: "custom-tool".to_string(),
        passed: true,
        exit_code: 0,
        stdout: String::new(),
        stderr: String::new(),
        duration_ms: 100,
        gate_id: "F-CUSTOM-001".to_string(),
    };
    assert_eq!(result.gate_id, "F-CUSTOM-001");
}
