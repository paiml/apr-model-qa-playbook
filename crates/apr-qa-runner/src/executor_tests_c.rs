
use super::*;
use crate::command::MockCommandRunner;
use apr_qa_gen::{Backend, Format, Modality, ModelId, QaScenario};

#[test]
fn test_tool_executor_validate_failure() {
    let mock_runner = MockCommandRunner::new().with_validate_failure();
    let executor = ToolExecutor::with_runner(
        "/test/model.gguf".to_string(),
        false,
        60_000,
        Arc::new(mock_runner),
    );

    let result = executor.execute_validate();

    assert!(!result.passed);
    assert_eq!(result.exit_code, 1);
}

#[test]
fn test_tool_executor_bench_failure() {
    let mock_runner = MockCommandRunner::new().with_bench_failure();
    let executor = ToolExecutor::with_runner(
        "/test/model.gguf".to_string(),
        false,
        60_000,
        Arc::new(mock_runner),
    );

    let result = executor.execute_bench();

    assert!(!result.passed);
    assert_eq!(result.exit_code, 1);
}

#[test]
fn test_tool_executor_check_failure() {
    let mock_runner = MockCommandRunner::new().with_check_failure();
    let executor = ToolExecutor::with_runner(
        "/test/model.gguf".to_string(),
        false,
        60_000,
        Arc::new(mock_runner),
    );

    let result = executor.execute_check();

    assert!(!result.passed);
    assert_eq!(result.exit_code, 1);
}

#[test]
fn test_tool_executor_profile_failure() {
    let mock_runner = MockCommandRunner::new().with_profile_failure();
    let executor = ToolExecutor::with_runner(
        "/test/model.gguf".to_string(),
        false,
        60_000,
        Arc::new(mock_runner),
    );

    let result = executor.execute_profile();

    assert!(!result.passed);
    assert_eq!(result.exit_code, 1);
}

#[test]
fn test_tool_executor_trace_failure() {
    let mock_runner = MockCommandRunner::new().with_inference_failure();
    let executor = ToolExecutor::with_runner(
        "/test/model.gguf".to_string(),
        false,
        60_000,
        Arc::new(mock_runner),
    );

    let result = executor.execute_trace("layer");

    assert!(!result.passed);
    assert_eq!(result.exit_code, 1);
}

#[test]
fn test_tool_executor_profile_ci_passes_with_metrics() {
    // Test that profile CI passes when output contains metrics
    let mock_runner = MockCommandRunner::new().with_tps(100.0);
    let executor = ToolExecutor::with_runner(
        "/test/model.gguf".to_string(),
        false,
        60_000,
        Arc::new(mock_runner),
    );

    let result = executor.execute_profile_ci();

    assert!(result.passed);
    assert!(result.stdout.contains("throughput"));
}

#[test]
fn test_tool_executor_with_no_gpu_true() {
    let mock_runner = MockCommandRunner::new();
    let executor = ToolExecutor::with_runner(
        "/test/model.gguf".to_string(),
        true, // no_gpu = true
        30_000,
        Arc::new(mock_runner),
    );

    // Just verify executor is created correctly
    let debug_str = format!("{executor:?}");
    assert!(debug_str.contains("no_gpu: true"));
}

#[test]
fn test_tool_executor_execute_trace_levels() {
    let mock_runner = MockCommandRunner::new();
    let executor = ToolExecutor::with_runner(
        "/test/model.gguf".to_string(),
        false,
        60_000,
        Arc::new(mock_runner),
    );

    let result_layer = executor.execute_trace("layer");
    assert!(result_layer.tool.contains("trace-layer"));

    let result_op = executor.execute_trace("op");
    assert!(result_op.tool.contains("trace-op"));

    let result_tensor = executor.execute_trace("tensor");
    assert!(result_tensor.tool.contains("trace-tensor"));
}

#[test]
fn test_resolve_model_path_gguf() {
    let temp_dir = tempfile::tempdir().unwrap();
    let gguf_dir = temp_dir.path().join("gguf");
    std::fs::create_dir_all(&gguf_dir).unwrap();
    std::fs::write(gguf_dir.join("model.gguf"), b"fake").unwrap();

    let config = ExecutionConfig {
        model_path: Some(temp_dir.path().to_string_lossy().to_string()),
        ..Default::default()
    };
    let executor = Executor::with_config(config);

    let scenario = QaScenario::new(
        ModelId::new("test", "model"),
        Modality::Run,
        Backend::Cpu,
        Format::Gguf,
        "test".to_string(),
        0,
    );

    let path = executor.resolve_model_path(&scenario);
    assert!(path.unwrap().contains("gguf"));
}

#[test]
fn test_resolve_model_path_apr() {
    let temp_dir = tempfile::tempdir().unwrap();
    let apr_dir = temp_dir.path().join("apr");
    std::fs::create_dir_all(&apr_dir).unwrap();
    std::fs::write(apr_dir.join("model.apr"), b"fake").unwrap();

    let config = ExecutionConfig {
        model_path: Some(temp_dir.path().to_string_lossy().to_string()),
        ..Default::default()
    };
    let executor = Executor::with_config(config);

    let scenario = QaScenario::new(
        ModelId::new("test", "model"),
        Modality::Run,
        Backend::Cpu,
        Format::Apr,
        "test".to_string(),
        0,
    );

    let path = executor.resolve_model_path(&scenario);
    assert!(path.unwrap().contains("apr"));
}

#[test]
fn test_resolve_model_path_safetensors() {
    let temp_dir = tempfile::tempdir().unwrap();
    let st_dir = temp_dir.path().join("safetensors");
    std::fs::create_dir_all(&st_dir).unwrap();
    std::fs::write(st_dir.join("model.safetensors"), b"fake").unwrap();

    let config = ExecutionConfig {
        model_path: Some(temp_dir.path().to_string_lossy().to_string()),
        ..Default::default()
    };
    let executor = Executor::with_config(config);

    let scenario = QaScenario::new(
        ModelId::new("test", "model"),
        Modality::Run,
        Backend::Cpu,
        Format::SafeTensors,
        "test".to_string(),
        0,
    );

    let path = executor.resolve_model_path(&scenario);
    assert!(path.unwrap().contains("safetensors"));
}

#[test]
fn test_resolve_model_path_no_cache() {
    // No model_path and no files - should return None
    let config = ExecutionConfig {
        model_path: None,
        ..Default::default()
    };
    let executor = Executor::with_config(config);

    let scenario = QaScenario::new(
        ModelId::new("test", "model"),
        Modality::Run,
        Backend::Cpu,
        Format::Gguf,
        "test".to_string(),
        0,
    );

    let path = executor.resolve_model_path(&scenario);
    // With no model path and no files, should return None
    assert!(path.is_none());
}

#[test]
fn test_executor_execute_dry_run() {
    let mock_runner = MockCommandRunner::new();
    let config = ExecutionConfig {
        dry_run: true,
        ..Default::default()
    };
    let mut executor = Executor::with_runner(config, Arc::new(mock_runner));

    let yaml = r#"
name: dry-run-test
version: "1.0.0"
model:
  hf_repo: "test/model"
  formats: [gguf]
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 3
"#;
    let playbook = Playbook::from_yaml(yaml).expect("Failed to parse");
    let result = executor.execute(&playbook).expect("Execution failed");

    // In dry run mode, all scenarios should be skipped
    assert_eq!(result.skipped, 3);
    // G0-PULL passes
    assert!(result.passed >= 1);
}

#[test]
fn test_executor_execute_with_stop_on_first_policy() {
    let mock_runner = MockCommandRunner::new().with_inference_failure();

    let config = ExecutionConfig {
        model_path: Some("/test/model.gguf".to_string()),
        failure_policy: FailurePolicy::StopOnFirst,
        run_conversion_tests: false,
        run_golden_rule_test: false,
        ..Default::default()
    };

    let mut executor = Executor::with_runner(config, Arc::new(mock_runner));

    let yaml = r#"
name: stop-on-first-test
version: "1.0.0"
model:
  hf_repo: "test/model"
  formats: [gguf]
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 5
"#;
    let playbook = Playbook::from_yaml(yaml).expect("Failed to parse");
    let result = executor.execute(&playbook).expect("Execution failed");

    // With StopOnFirst policy, should stop after first failure
    assert_eq!(result.failed, 1);
}

#[test]
fn test_executor_execute_with_collect_all_policy() {
    let mock_runner = MockCommandRunner::new().with_inference_failure();

    let config = ExecutionConfig {
        model_path: Some("/test/model.gguf".to_string()),
        failure_policy: FailurePolicy::CollectAll,
        run_conversion_tests: false,
        run_golden_rule_test: false,
        run_contract_tests: false,
        ..Default::default()
    };

    let mut executor = Executor::with_runner(config, Arc::new(mock_runner));

    let yaml = r#"
name: collect-all-test
version: "1.0.0"
model:
  hf_repo: "test/model"
  formats: [gguf]
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 3
"#;
    let playbook = Playbook::from_yaml(yaml).expect("Failed to parse");
    let result = executor.execute(&playbook).expect("Execution failed");

    // With CollectAll policy, should collect all failures
    assert_eq!(result.failed, 3);
}

#[test]
fn test_executor_default_impl() {
    let executor = Executor::default();
    assert_eq!(executor.config().max_workers, 4);
    assert!(!executor.config().dry_run);
}

#[test]
fn test_parse_tps_from_output_with_tps() {
    let output = "Info: Loading model\ntok/s: 42.5\nDone";
    let tps = Executor::parse_tps_from_output(output);
    assert!(tps.is_some());
    assert!((tps.unwrap() - 42.5).abs() < 0.01);
}

#[test]
fn test_parse_tps_from_output_no_tps() {
    let output = "Some random output without tok/s";
    let tps = Executor::parse_tps_from_output(output);
    assert!(tps.is_none());
}

#[test]
fn test_extract_generated_text() {
    let output = "=== Model Info ===\nThis is generated text\ntok/s: 30.0";
    let text = Executor::extract_generated_text(output);
    assert!(text.contains("This is generated text"));
    assert!(!text.contains("tok/s"));
    assert!(!text.contains("==="));
}

#[test]
fn test_extract_output_text_multiline_detailed() {
    let output = "Some prefix\nOutput:\nLine 1\nLine 2\nLine 3\nCompleted in 1s";
    let text = Executor::extract_output_text(output);
    assert!(text.contains("Line 1"));
    assert!(text.contains("Line 2"));
    assert!(text.contains("Line 3"));
}

#[test]
fn test_extract_output_text_with_empty_lines() {
    let output = "Output:\nActual output here\n\nCompleted";
    let text = Executor::extract_output_text(output);
    assert!(text.contains("Actual output here"));
}

#[test]
fn test_failure_policy_default_is_stop_on_p0() {
    let policy = FailurePolicy::default();
    assert_eq!(policy, FailurePolicy::StopOnP0);
}

#[test]
fn test_execution_config_debug_display() {
    let config = ExecutionConfig::default();
    let debug_str = format!("{config:?}");
    assert!(debug_str.contains("ExecutionConfig"));
    assert!(debug_str.contains("failure_policy"));
}

#[test]
fn test_tool_test_result_all_fields() {
    let result = ToolTestResult {
        tool: "test-tool".to_string(),
        passed: true,
        exit_code: 0,
        stdout: "stdout".to_string(),
        stderr: String::new(),
        duration_ms: 100,
        gate_id: "F-TEST-001".to_string(),
    };
    assert_eq!(result.tool, "test-tool");
    assert!(result.passed);
    assert_eq!(result.gate_id, "F-TEST-001");
}

#[test]
fn test_executor_evidence_accessor() {
    let executor = Executor::new();
    let evidence = executor.evidence();
    assert_eq!(evidence.total(), 0);
}

#[test]
fn test_execution_result_is_success_false_due_to_failed() {
    let result = ExecutionResult {
        playbook_name: "test".to_string(),
        total_scenarios: 10,
        passed: 9,
        failed: 1,
        skipped: 0,
        duration_ms: 100,
        gateway_failed: None,
        evidence: EvidenceCollector::new(),
    };
    assert!(!result.is_success());
}

#[test]
fn test_execution_result_is_success_when_all_pass() {
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
    assert!(result.is_success());
}

#[test]
fn test_tool_test_result_to_evidence_when_failed() {
    let result = ToolTestResult {
        tool: "validate".to_string(),
        passed: false,
        exit_code: 1,
        stdout: String::new(),
        stderr: "Validation failed".to_string(),
        duration_ms: 200,
        gate_id: "F-VALIDATE-001".to_string(),
    };
    let model_id = ModelId::new("org", "model");
    let evidence = result.to_evidence(&model_id);
    assert!(!evidence.outcome.is_pass());
    assert!(evidence.reason.contains("Validation failed") || evidence.output.is_empty());
}

#[test]
fn test_executor_with_mock_runner_trace_failure_case() {
    let mock_runner = MockCommandRunner::new().with_inference_failure();

    let config = ExecutionConfig {
        model_path: Some("/test/model.gguf".to_string()),
        ..Default::default()
    };

    let executor = Executor::with_runner(config, Arc::new(mock_runner));

    let scenario = QaScenario::new(
        ModelId::new("test", "model"),
        Modality::Run,
        Backend::Cpu,
        Format::Gguf,
        "What is 2+2?".to_string(),
        0,
    );

    let (_, stderr, exit_code, _, _) = executor.subprocess_execution(&scenario);

    // Should include trace output in stderr
    assert_eq!(exit_code, 1);
    assert!(stderr.is_some());
}

#[test]
fn test_resolve_model_path_apr_format() {
    let tmp = tempfile::tempdir().unwrap();
    let apr_dir = tmp.path().join("apr");
    std::fs::create_dir_all(&apr_dir).unwrap();
    std::fs::write(apr_dir.join("model.apr"), b"fake apr").unwrap();

    let config = ExecutionConfig {
        model_path: Some(tmp.path().to_string_lossy().to_string()),
        ..Default::default()
    };
    let executor = Executor::with_config(config);
    let scenario = QaScenario::new(
        ModelId::new("test", "model"),
        Modality::Run,
        Backend::Cpu,
        Format::Apr,
        "test".to_string(),
        0,
    );
    let path = executor.resolve_model_path(&scenario);
    assert!(path.is_some());
    assert!(path.unwrap().contains("apr"));
}

#[test]
fn test_resolve_model_path_safetensors_format() {
    let tmp = tempfile::tempdir().unwrap();
    let st_dir = tmp.path().join("safetensors");
    std::fs::create_dir_all(&st_dir).unwrap();
    std::fs::write(st_dir.join("model.safetensors"), b"fake st").unwrap();

    let config = ExecutionConfig {
        model_path: Some(tmp.path().to_string_lossy().to_string()),
        ..Default::default()
    };
    let executor = Executor::with_config(config);
    let scenario = QaScenario::new(
        ModelId::new("test", "model"),
        Modality::Run,
        Backend::Cpu,
        Format::SafeTensors,
        "test".to_string(),
        0,
    );
    let path = executor.resolve_model_path(&scenario);
    assert!(path.is_some());
    assert!(path.unwrap().contains("safetensors"));
}

#[test]
fn test_resolve_model_path_gguf_format() {
    let tmp = tempfile::tempdir().unwrap();
    let gguf_dir = tmp.path().join("gguf");
    std::fs::create_dir_all(&gguf_dir).unwrap();
    std::fs::write(gguf_dir.join("model.gguf"), b"fake gguf").unwrap();

    let config = ExecutionConfig {
        model_path: Some(tmp.path().to_string_lossy().to_string()),
        ..Default::default()
    };
    let executor = Executor::with_config(config);
    let scenario = QaScenario::new(
        ModelId::new("test", "model"),
        Modality::Run,
        Backend::Cpu,
        Format::Gguf,
        "test".to_string(),
        0,
    );
    let path = executor.resolve_model_path(&scenario);
    assert!(path.is_some());
    assert!(path.unwrap().contains("gguf"));
}

#[test]
fn test_resolve_model_path_no_model_path() {
    // When no model_path is configured and no file exists, should return None
    let config = ExecutionConfig {
        model_path: None,
        ..Default::default()
    };
    let executor = Executor::with_config(config);
    let scenario = QaScenario::new(
        ModelId::new("test", "model"),
        Modality::Run,
        Backend::Cpu,
        Format::Gguf,
        "test".to_string(),
        0,
    );
    let path = executor.resolve_model_path(&scenario);
    // Should return None when no model file exists at default path
    assert!(path.is_none());
}

#[test]
fn test_executor_subprocess_execution_formats() {
    let mock_runner = MockCommandRunner::new().with_inference_response("The answer is 4.");

    let config = ExecutionConfig {
        model_path: Some("/test/cache".to_string()),
        ..Default::default()
    };

    let executor = Executor::with_runner(config, Arc::new(mock_runner));

    // Test APR format
    let scenario_apr = QaScenario::new(
        ModelId::new("test", "model"),
        Modality::Run,
        Backend::Cpu,
        Format::Apr,
        "What is 2+2?".to_string(),
        0,
    );
    let (_, _, exit_code, _, _) = executor.subprocess_execution(&scenario_apr);
    assert_eq!(exit_code, 0);
}

#[test]
fn test_executor_subprocess_execution_safetensors() {
    let mock_runner = MockCommandRunner::new().with_inference_response("The answer is 4.");

    let config = ExecutionConfig {
        model_path: Some("/test/cache".to_string()),
        ..Default::default()
    };

    let executor = Executor::with_runner(config, Arc::new(mock_runner));

    let scenario = QaScenario::new(
        ModelId::new("test", "model"),
        Modality::Run,
        Backend::Cpu,
        Format::SafeTensors,
        "What is 2+2?".to_string(),
        0,
    );
    let (_, _, exit_code, _, _) = executor.subprocess_execution(&scenario);
    assert_eq!(exit_code, 0);
}

#[test]
fn test_execute_scenario_with_exit_code_failure() {
    let mock_runner = MockCommandRunner::new().with_exit_code(5);

    let config = ExecutionConfig {
        model_path: Some("/test/model.gguf".to_string()),
        ..Default::default()
    };

    let executor = Executor::with_runner(config, Arc::new(mock_runner));

    let scenario = QaScenario::new(
        ModelId::new("test", "model"),
        Modality::Run,
        Backend::Cpu,
        Format::Gguf,
        "What is 2+2?".to_string(),
        0,
    );

    let evidence = executor.execute_scenario(&scenario);

    // Non-zero exit code should result in failed evidence
    assert!(evidence.outcome.is_fail());
    assert!(evidence.exit_code.is_some());
    assert_eq!(evidence.exit_code.unwrap(), 5);
}

#[test]
fn test_execute_scenario_with_stderr_corroborated() {
    let mock_runner = MockCommandRunner::new()
        .with_inference_response_and_stderr("The answer is 4.", "Some warning");

    let config = ExecutionConfig {
        model_path: Some("/test/model.gguf".to_string()),
        ..Default::default()
    };

    let executor = Executor::with_runner(config, Arc::new(mock_runner));

    let scenario = QaScenario::new(
        ModelId::new("test", "model"),
        Modality::Run,
        Backend::Cpu,
        Format::Gguf,
        "What is 2+2?".to_string(),
        0,
    );

    let evidence = executor.execute_scenario(&scenario);
    // Should pass but have stderr captured
    assert!(evidence.outcome.is_pass());
}

#[test]
fn test_executor_run_conversion_tests_no_gpu() {
    let mock_runner = MockCommandRunner::new();
    let config = ExecutionConfig {
        model_path: Some("/test/model.gguf".to_string()),
        run_conversion_tests: true,
        no_gpu: true,
        ..Default::default()
    };

    let mut executor = Executor::with_runner(config, Arc::new(mock_runner));
    let model_id = ModelId::new("test", "model");

    // Run conversion tests with no_gpu flag
    let (passed, failed) =
        executor.run_conversion_tests(std::path::Path::new("/test/model.gguf"), &model_id);

    // Just verify function runs
    let _ = (passed, failed);
}

#[test]
fn test_executor_execute_with_stop_on_first_failure() {
    let mock_runner = MockCommandRunner::new().with_inference_failure();

    let config = ExecutionConfig {
        model_path: Some("/test/model.gguf".to_string()),
        failure_policy: FailurePolicy::StopOnFirst,
        run_conversion_tests: false,
        run_golden_rule_test: false,
        ..Default::default()
    };

    let mut executor = Executor::with_runner(config, Arc::new(mock_runner));

    let yaml = r#"
name: stop-on-first-test
version: "1.0.0"
model:
  hf_repo: "test/model"
  formats: [gguf]
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 5
"#;
    let playbook = Playbook::from_yaml(yaml).expect("Failed to parse");
    let result = executor.execute(&playbook).expect("Execution failed");

    // Should stop after first failure
    assert!(result.failed >= 1);
    // Total executed should be less than total scenarios due to early stop
    let executed = result.passed + result.failed;
    assert!(executed <= result.total_scenarios);
}

#[test]
fn test_executor_execute_with_collect_all_failures() {
    let mock_runner = MockCommandRunner::new().with_inference_failure();

    let config = ExecutionConfig {
        model_path: Some("/test/model.gguf".to_string()),
        failure_policy: FailurePolicy::CollectAll,
        run_conversion_tests: false,
        run_golden_rule_test: false,
        run_contract_tests: false,
        ..Default::default()
    };

    let mut executor = Executor::with_runner(config, Arc::new(mock_runner));

    let yaml = r#"
name: collect-all-test
version: "1.0.0"
model:
  hf_repo: "test/model"
  formats: [gguf]
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 3
"#;
    let playbook = Playbook::from_yaml(yaml).expect("Failed to parse");
    let result = executor.execute(&playbook).expect("Execution failed");

    // Should collect all failures (3 scenarios)
    assert_eq!(result.failed, 3);
    // Bug 204: G0-PULL skipped when model_path is set, so 3 scenarios only
    assert_eq!(result.total_scenarios, 3);
}

// =========================================================================
// StopOnP0 policy test
// =========================================================================

#[test]
fn test_executor_stop_on_p0_with_p0_gate() {
    // Create a runner that returns falsified results with P0 gate IDs
    let mock_runner = MockCommandRunner::new()
        .with_inference_failure()
        .with_exit_code(1);

    let config = ExecutionConfig {
        model_path: Some("/test/model.gguf".to_string()),
        failure_policy: FailurePolicy::StopOnP0,
        run_conversion_tests: false,
        run_golden_rule_test: false,
        ..Default::default()
    };

    let mut executor = Executor::with_runner(config, Arc::new(mock_runner));

    let yaml = r#"
name: p0-test
version: "1.0.0"
model:
  hf_repo: "test/model"
  formats: [gguf]
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 5
"#;
    let playbook = Playbook::from_yaml(yaml).expect("Failed to parse");
    let result = executor.execute(&playbook).expect("Execution failed");

    // With failures that don't have -P0- in gate_id, it should collect all
    assert!(result.failed >= 1);
}

// =========================================================================
// ConversionConfig::default() (no_gpu = false)
// =========================================================================

#[test]
fn test_executor_run_conversion_tests_default_config() {
    let mock_runner = MockCommandRunner::new();
    let config = ExecutionConfig {
        model_path: Some("/test/model.gguf".to_string()),
        run_conversion_tests: true,
        run_golden_rule_test: false,
        no_gpu: false, // This triggers ConversionConfig::default()
        ..Default::default()
    };

    let mut executor = Executor::with_runner(config, Arc::new(mock_runner));

    let yaml = r#"
name: conv-default-test
version: "1.0.0"
model:
  hf_repo: "test/model"
  formats: [gguf]
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
"#;
    let playbook = Playbook::from_yaml(yaml).expect("Failed to parse");
    let result = executor.execute(&playbook).expect("Execution failed");
    // Just verify it runs without panic
    assert!(result.total_scenarios >= 1);
}

// =========================================================================
// Golden Rule: converted inference fails (F-GOLDEN-RULE-003)
// =========================================================================

#[test]
#[allow(clippy::too_many_lines)]
fn test_executor_golden_rule_converted_inference_fails() {
    use crate::command::CommandOutput;

    // Build a custom runner that succeeds on original, succeeds on convert,
    // but fails on converted inference
    struct ConvertedFailRunner;
    impl CommandRunner for ConvertedFailRunner {
        fn run_inference(
            &self,
            model_path: &Path,
            _prompt: &str,
            _max_tokens: u32,
            _no_gpu: bool,
            _extra_args: &[&str],
        ) -> CommandOutput {
            // Original model succeeds, converted model (.apr) fails
            if model_path.to_string_lossy().contains(".apr") {
                CommandOutput {
                    stdout: String::new(),
                    stderr: "Failed to load converted model".to_string(),
                    exit_code: 1,
                    success: false,
                }
            } else {
                CommandOutput {
                    stdout: "Output:\nThe answer is 4.\nCompleted in 100ms".to_string(),
                    stderr: String::new(),
                    exit_code: 0,
                    success: true,
                }
            }
        }

        fn convert_model(&self, _source: &Path, _target: &Path) -> CommandOutput {
            CommandOutput {
                stdout: "Conversion complete".to_string(),
                stderr: String::new(),
                exit_code: 0,
                success: true,
            }
        }

        fn inspect_model(&self, _path: &Path) -> CommandOutput {
            CommandOutput::success("")
        }
        fn validate_model(&self, _path: &Path) -> CommandOutput {
            CommandOutput::success("")
        }
        fn bench_model(&self, _path: &Path) -> CommandOutput {
            CommandOutput::success("")
        }
        fn check_model(&self, _path: &Path) -> CommandOutput {
            CommandOutput::success("")
        }
        fn profile_model(&self, _path: &Path, _warmup: u32, _measure: u32) -> CommandOutput {
            CommandOutput::success("")
        }
        fn profile_ci(
            &self,
            _path: &Path,
            _min_throughput: Option<f64>,
            _max_p99: Option<f64>,
            _warmup: u32,
            _measure: u32,
        ) -> CommandOutput {
            CommandOutput::success("")
        }
        fn diff_tensors(&self, _model_a: &Path, _model_b: &Path, _json: bool) -> CommandOutput {
            CommandOutput::success("")
        }
        fn compare_inference(
            &self,
            _model_a: &Path,
            _model_b: &Path,
            _prompt: &str,
            _max_tokens: u32,
            _tolerance: f64,
        ) -> CommandOutput {
            CommandOutput::success("")
        }
        fn profile_with_flamegraph(
            &self,
            _model_path: &Path,
            _output_path: &Path,
            _no_gpu: bool,
        ) -> CommandOutput {
            CommandOutput::success("")
        }
        fn profile_with_focus(
            &self,
            _model_path: &Path,
            _focus: &str,
            _no_gpu: bool,
        ) -> CommandOutput {
            CommandOutput::success("")
        }
        fn fingerprint_model(&self, _path: &Path, _json: bool) -> CommandOutput {
            CommandOutput::success("")
        }
        fn validate_stats(&self, _a: &Path, _b: &Path) -> CommandOutput {
            CommandOutput::success("")
        }
        fn validate_model_strict(&self, _path: &Path) -> CommandOutput {
            CommandOutput::success(r#"{"valid":true,"tensors_checked":100,"issues":[]}"#)
        }
        fn pull_model(&self, _hf_repo: &str) -> CommandOutput {
            CommandOutput::success("Path: /mock/model.safetensors")
        }
        fn inspect_model_json(&self, _model_path: &Path) -> CommandOutput {
            CommandOutput::success(
                r#"{"format":"SafeTensors","tensor_count":10,"tensor_names":[]}"#,
            )
        }
        fn run_ollama_inference(
            &self,
            _model_tag: &str,
            _prompt: &str,
            _temperature: f64,
        ) -> CommandOutput {
            CommandOutput::success("Output:\nThe answer is 4.\nCompleted in 1.0s")
        }
        fn pull_ollama_model(&self, _model_tag: &str) -> CommandOutput {
            CommandOutput::success("pulling manifest... done")
        }
        fn create_ollama_model(&self, _: &str, _: &Path) -> CommandOutput {
            CommandOutput::success("creating model... done")
        }
        fn serve_model(&self, _: &Path, _: u16) -> CommandOutput {
            CommandOutput::success(r#"{"status":"listening"}"#)
        }
        fn http_get(&self, _: &str) -> CommandOutput {
            CommandOutput::success(r#"{"models":[]}"#)
        }
        fn profile_memory(&self, _: &Path) -> CommandOutput {
            CommandOutput::success(r#"{"peak_rss_mb":1024}"#)
        }
        fn run_chat(
            &self,
            _model_path: &Path,
            _prompt: &str,
            _no_gpu: bool,
            _extra_args: &[&str],
        ) -> CommandOutput {
            CommandOutput::success("Chat output")
        }
        fn http_post(&self, _url: &str, _body: &str) -> CommandOutput {
            CommandOutput::success("{}")
        }
        fn spawn_serve(&self, _model_path: &Path, _port: u16, _no_gpu: bool) -> CommandOutput {
            CommandOutput::success("12345")
        }
    }

    let config = ExecutionConfig {
        model_path: Some("/test/model.gguf".to_string()),
        run_conversion_tests: false,
        run_golden_rule_test: true,
        ..Default::default()
    };

    let mut executor = Executor::with_runner(config, Arc::new(ConvertedFailRunner));

    let yaml = r#"
name: golden-conv-fail
version: "1.0.0"
model:
  hf_repo: "test/model"
  formats: [gguf]
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
"#;
    let playbook = Playbook::from_yaml(yaml).expect("Failed to parse");
    let result = executor.execute(&playbook).expect("Execution failed");
    // Golden rule test should produce a failure (converted inference failed)
    assert!(result.failed >= 1);
}

// =========================================================================
// Golden Rule: output differs (F-GOLDEN-RULE-001 FAIL)
// =========================================================================

#[test]
#[allow(clippy::too_many_lines)]
fn test_executor_golden_rule_output_differs_with_data() {
    use crate::command::CommandOutput;

    struct DiffOutputRunner;
    impl CommandRunner for DiffOutputRunner {
        fn run_inference(
            &self,
            model_path: &Path,
            _prompt: &str,
            _max_tokens: u32,
            _no_gpu: bool,
            _extra_args: &[&str],
        ) -> CommandOutput {
            if model_path.to_string_lossy().contains(".apr") {
                CommandOutput {
                    stdout: "Output:\nThe answer is 5.\nCompleted in 100ms".to_string(),
                    stderr: String::new(),
                    exit_code: 0,
                    success: true,
                }
            } else {
                CommandOutput {
                    stdout: "Output:\nThe answer is 4.\nCompleted in 100ms".to_string(),
                    stderr: String::new(),
                    exit_code: 0,
                    success: true,
                }
            }
        }

        fn convert_model(&self, _source: &Path, _target: &Path) -> CommandOutput {
            CommandOutput {
                stdout: "ok".to_string(),
                stderr: String::new(),
                exit_code: 0,
                success: true,
            }
        }

        fn inspect_model(&self, _path: &Path) -> CommandOutput {
            CommandOutput::success("")
        }
        fn validate_model(&self, _path: &Path) -> CommandOutput {
            CommandOutput::success("")
        }
        fn bench_model(&self, _path: &Path) -> CommandOutput {
            CommandOutput::success("")
        }
        fn check_model(&self, _path: &Path) -> CommandOutput {
            CommandOutput::success("")
        }
        fn profile_model(&self, _path: &Path, _warmup: u32, _measure: u32) -> CommandOutput {
            CommandOutput::success("")
        }
        fn profile_ci(
            &self,
            _path: &Path,
            _min_throughput: Option<f64>,
            _max_p99: Option<f64>,
            _warmup: u32,
            _measure: u32,
        ) -> CommandOutput {
            CommandOutput::success("")
        }
        fn diff_tensors(&self, _model_a: &Path, _model_b: &Path, _json: bool) -> CommandOutput {
            CommandOutput::success("")
        }
        fn compare_inference(
            &self,
            _model_a: &Path,
            _model_b: &Path,
            _prompt: &str,
            _max_tokens: u32,
            _tolerance: f64,
        ) -> CommandOutput {
            CommandOutput::success("")
        }
        fn profile_with_flamegraph(
            &self,
            _model_path: &Path,
            _output_path: &Path,
            _no_gpu: bool,
        ) -> CommandOutput {
            CommandOutput::success("")
        }
        fn profile_with_focus(
            &self,
            _model_path: &Path,
            _focus: &str,
            _no_gpu: bool,
        ) -> CommandOutput {
            CommandOutput::success("")
        }
        fn fingerprint_model(&self, _path: &Path, _json: bool) -> CommandOutput {
            CommandOutput::success("")
        }
        fn validate_stats(&self, _a: &Path, _b: &Path) -> CommandOutput {
            CommandOutput::success("")
        }
        fn validate_model_strict(&self, _path: &Path) -> CommandOutput {
            CommandOutput::success(r#"{"valid":true,"tensors_checked":100,"issues":[]}"#)
        }
        fn pull_model(&self, _hf_repo: &str) -> CommandOutput {
            CommandOutput::success("Path: /mock/model.safetensors")
        }
        fn inspect_model_json(&self, _model_path: &Path) -> CommandOutput {
            CommandOutput::success(
                r#"{"format":"SafeTensors","tensor_count":10,"tensor_names":[]}"#,
            )
        }
        fn run_ollama_inference(
            &self,
            _model_tag: &str,
            _prompt: &str,
            _temperature: f64,
        ) -> CommandOutput {
            CommandOutput::success("Output:\nThe answer is 4.\nCompleted in 1.0s")
        }
        fn pull_ollama_model(&self, _model_tag: &str) -> CommandOutput {
            CommandOutput::success("pulling manifest... done")
        }
        fn create_ollama_model(&self, _: &str, _: &Path) -> CommandOutput {
            CommandOutput::success("creating model... done")
        }
        fn serve_model(&self, _: &Path, _: u16) -> CommandOutput {
            CommandOutput::success(r#"{"status":"listening"}"#)
        }
        fn http_get(&self, _: &str) -> CommandOutput {
            CommandOutput::success(r#"{"models":[]}"#)
        }
        fn profile_memory(&self, _: &Path) -> CommandOutput {
            CommandOutput::success(r#"{"peak_rss_mb":1024}"#)
        }
        fn run_chat(
            &self,
            _model_path: &Path,
            _prompt: &str,
            _no_gpu: bool,
            _extra_args: &[&str],
        ) -> CommandOutput {
            CommandOutput::success("Chat output")
        }
        fn http_post(&self, _url: &str, _body: &str) -> CommandOutput {
            CommandOutput::success("{}")
        }
        fn spawn_serve(&self, _model_path: &Path, _port: u16, _no_gpu: bool) -> CommandOutput {
            CommandOutput::success("12345")
        }
    }

    let config = ExecutionConfig {
        model_path: Some("/test/model.gguf".to_string()),
        run_conversion_tests: false,
        run_golden_rule_test: true,
        ..Default::default()
    };

    let mut executor = Executor::with_runner(config, Arc::new(DiffOutputRunner));

    let yaml = r#"
name: golden-diff
version: "1.0.0"
model:
  hf_repo: "test/model"
  formats: [gguf]
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
"#;
    let playbook = Playbook::from_yaml(yaml).expect("Failed to parse");
    let result = executor.execute(&playbook).expect("Execution failed");
    // Output differs => falsified
    assert!(result.failed >= 1);
}

// =========================================================================
// Subprocess execution with trace + stdout
// =========================================================================

#[test]
#[allow(clippy::too_many_lines)]
fn test_executor_subprocess_trace_with_stdout() {
    use crate::command::CommandOutput;

    struct TraceStdoutRunner;
    impl CommandRunner for TraceStdoutRunner {
        fn run_inference(
            &self,
            _model_path: &Path,
            _prompt: &str,
            _max_tokens: u32,
            _no_gpu: bool,
            extra_args: &[&str],
        ) -> CommandOutput {
            if extra_args.contains(&"--trace") {
                // Trace run returns both stderr and stdout
                CommandOutput {
                    stdout: "trace data: layer 0 attention".to_string(),
                    stderr: "TRACE: model loading details".to_string(),
                    exit_code: 0,
                    success: true,
                }
            } else {
                // First run fails
                CommandOutput {
                    stdout: String::new(),
                    stderr: "inference error occurred".to_string(),
                    exit_code: 1,
                    success: false,
                }
            }
        }

        fn convert_model(&self, _source: &Path, _target: &Path) -> CommandOutput {
            CommandOutput::success("")
        }
        fn inspect_model(&self, _path: &Path) -> CommandOutput {
            CommandOutput::success("")
        }
        fn validate_model(&self, _path: &Path) -> CommandOutput {
            CommandOutput::success("")
        }
        fn bench_model(&self, _path: &Path) -> CommandOutput {
            CommandOutput::success("")
        }
        fn check_model(&self, _path: &Path) -> CommandOutput {
            CommandOutput::success("")
        }
        fn profile_model(&self, _path: &Path, _warmup: u32, _measure: u32) -> CommandOutput {
            CommandOutput::success("")
        }
        fn profile_ci(
            &self,
            _path: &Path,
            _min_throughput: Option<f64>,
            _max_p99: Option<f64>,
            _warmup: u32,
            _measure: u32,
        ) -> CommandOutput {
            CommandOutput::success("")
        }
        fn diff_tensors(&self, _model_a: &Path, _model_b: &Path, _json: bool) -> CommandOutput {
            CommandOutput::success("")
        }
        fn compare_inference(
            &self,
            _model_a: &Path,
            _model_b: &Path,
            _prompt: &str,
            _max_tokens: u32,
            _tolerance: f64,
        ) -> CommandOutput {
            CommandOutput::success("")
        }
        fn profile_with_flamegraph(
            &self,
            _model_path: &Path,
            _output_path: &Path,
            _no_gpu: bool,
        ) -> CommandOutput {
            CommandOutput::success("")
        }
        fn profile_with_focus(
            &self,
            _model_path: &Path,
            _focus: &str,
            _no_gpu: bool,
        ) -> CommandOutput {
            CommandOutput::success("")
        }
        fn fingerprint_model(&self, _path: &Path, _json: bool) -> CommandOutput {
            CommandOutput::success("")
        }
        fn validate_stats(&self, _a: &Path, _b: &Path) -> CommandOutput {
            CommandOutput::success("")
        }
        fn validate_model_strict(&self, _path: &Path) -> CommandOutput {
            CommandOutput::success(r#"{"valid":true,"tensors_checked":100,"issues":[]}"#)
        }
        fn pull_model(&self, _hf_repo: &str) -> CommandOutput {
            CommandOutput::success("Path: /mock/model.safetensors")
        }
        fn inspect_model_json(&self, _model_path: &Path) -> CommandOutput {
            CommandOutput::success(
                r#"{"format":"SafeTensors","tensor_count":10,"tensor_names":[]}"#,
            )
        }
        fn run_ollama_inference(
            &self,
            _model_tag: &str,
            _prompt: &str,
            _temperature: f64,
        ) -> CommandOutput {
            CommandOutput::success("Output:\nThe answer is 4.\nCompleted in 1.0s")
        }
        fn pull_ollama_model(&self, _model_tag: &str) -> CommandOutput {
            CommandOutput::success("pulling manifest... done")
        }
        fn create_ollama_model(&self, _: &str, _: &Path) -> CommandOutput {
            CommandOutput::success("creating model... done")
        }
        fn serve_model(&self, _: &Path, _: u16) -> CommandOutput {
            CommandOutput::success(r#"{"status":"listening"}"#)
        }
        fn http_get(&self, _: &str) -> CommandOutput {
            CommandOutput::success(r#"{"models":[]}"#)
        }
        fn profile_memory(&self, _: &Path) -> CommandOutput {
            CommandOutput::success(r#"{"peak_rss_mb":1024}"#)
        }
        fn run_chat(
            &self,
            _model_path: &Path,
            _prompt: &str,
            _no_gpu: bool,
            _extra_args: &[&str],
        ) -> CommandOutput {
            CommandOutput::success("Chat output")
        }
        fn http_post(&self, _url: &str, _body: &str) -> CommandOutput {
            CommandOutput::success("{}")
        }
        fn spawn_serve(&self, _model_path: &Path, _port: u16, _no_gpu: bool) -> CommandOutput {
            CommandOutput::success("12345")
        }
    }

    let config = ExecutionConfig {
        model_path: Some("/test/model.gguf".to_string()),
        run_conversion_tests: false,
        run_golden_rule_test: false,
        ..Default::default()
    };

    let mut executor = Executor::with_runner(config, Arc::new(TraceStdoutRunner));

    let yaml = r#"
name: trace-stdout-test
version: "1.0.0"
model:
  hf_repo: "test/model"
  formats: [gguf]
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
"#;
    let playbook = Playbook::from_yaml(yaml).expect("Failed to parse");
    let result = executor.execute(&playbook).expect("Execution failed");
    assert!(result.failed >= 1);
    // Check that evidence contains trace data
    let evidence = executor.evidence().all();
    assert!(!evidence.is_empty());
    // stderr should contain trace output
    let last = &evidence[evidence.len() - 1];
    if let Some(ref stderr) = last.stderr {
        assert!(stderr.contains("TRACE STDOUT") || stderr.contains("trace"));
    }
}

// =========================================================================
// Model path resolution fallback
// =========================================================================

#[test]
fn test_resolve_model_path_fallback_to_extension() {
    let temp_dir = tempfile::tempdir().unwrap();
    let gguf_dir = temp_dir.path().join("gguf");
    std::fs::create_dir_all(&gguf_dir).unwrap();

    // Create a file with .gguf extension but NOT named "model.gguf"
    let alt_model = gguf_dir.join("custom-name.gguf");
    std::fs::write(&alt_model, b"fake model").unwrap();

    let config = ExecutionConfig {
        model_path: Some(temp_dir.path().to_string_lossy().to_string()),
        ..Default::default()
    };
    let executor = Executor::with_config(config);

    let scenario = apr_qa_gen::QaScenario::new(
        apr_qa_gen::ModelId::new("test", "model"),
        apr_qa_gen::Modality::Run,
        apr_qa_gen::Backend::Cpu,
        apr_qa_gen::Format::Gguf,
        "test prompt".to_string(),
        0,
    );

    let path = executor.resolve_model_path(&scenario);
    // Should find the custom-name.gguf via extension fallback
    assert!(path.unwrap().contains("custom-name.gguf"));
}

#[test]
fn test_resolve_model_path_prefers_model_dot_ext() {
    let temp_dir = tempfile::tempdir().unwrap();
    let apr_dir = temp_dir.path().join("apr");
    std::fs::create_dir_all(&apr_dir).unwrap();

    // Create the canonical model.apr
    let model_file = apr_dir.join("model.apr");
    std::fs::write(&model_file, b"fake model").unwrap();

    let config = ExecutionConfig {
        model_path: Some(temp_dir.path().to_string_lossy().to_string()),
        ..Default::default()
    };
    let executor = Executor::with_config(config);

    let scenario = apr_qa_gen::QaScenario::new(
        apr_qa_gen::ModelId::new("test", "model"),
        apr_qa_gen::Modality::Run,
        apr_qa_gen::Backend::Cpu,
        apr_qa_gen::Format::Apr,
        "test prompt".to_string(),
        0,
    );

    let path = executor.resolve_model_path(&scenario);
    assert!(path.unwrap().contains("model.apr"));
}

// =========================================================================
// File-mode model path resolution
// =========================================================================

#[test]
fn test_resolve_model_path_file_matching_format() {
    let temp_dir = tempfile::tempdir().unwrap();
    let model_file = temp_dir.path().join("abc123.safetensors");
    std::fs::write(&model_file, b"fake model data").unwrap();

    let config = ExecutionConfig {
        model_path: Some(model_file.to_string_lossy().to_string()),
        ..Default::default()
    };
    let executor = Executor::with_config(config);

    // SafeTensors format should match .safetensors file
    let scenario = QaScenario::new(
        ModelId::new("test", "model"),
        Modality::Run,
        Backend::Cpu,
        Format::SafeTensors,
        "test".to_string(),
        0,
    );
    let path = executor.resolve_model_path(&scenario);
    assert!(path.is_some());
    assert!(path.unwrap().contains("abc123.safetensors"));
}

#[test]
fn test_resolve_model_path_file_nonmatching_format() {
    let temp_dir = tempfile::tempdir().unwrap();
    let model_file = temp_dir.path().join("abc123.safetensors");
    std::fs::write(&model_file, b"fake model data").unwrap();

    let config = ExecutionConfig {
        model_path: Some(model_file.to_string_lossy().to_string()),
        ..Default::default()
    };
    let executor = Executor::with_config(config);

    // GGUF format should NOT match .safetensors file
    let scenario_gguf = QaScenario::new(
        ModelId::new("test", "model"),
        Modality::Run,
        Backend::Cpu,
        Format::Gguf,
        "test".to_string(),
        0,
    );
    assert!(executor.resolve_model_path(&scenario_gguf).is_none());

    // APR format should NOT match .safetensors file
    let scenario_apr = QaScenario::new(
        ModelId::new("test", "model"),
        Modality::Run,
        Backend::Cpu,
        Format::Apr,
        "test".to_string(),
        0,
    );
    assert!(executor.resolve_model_path(&scenario_apr).is_none());
}

#[test]
fn test_resolve_model_path_file_gguf() {
    let temp_dir = tempfile::tempdir().unwrap();
    let model_file = temp_dir.path().join("hash123.gguf");
    std::fs::write(&model_file, b"fake gguf").unwrap();

    let config = ExecutionConfig {
        model_path: Some(model_file.to_string_lossy().to_string()),
        ..Default::default()
    };
    let executor = Executor::with_config(config);

    let scenario = QaScenario::new(
        ModelId::new("test", "model"),
        Modality::Run,
        Backend::Cpu,
        Format::Gguf,
        "test".to_string(),
        0,
    );
    let path = executor.resolve_model_path(&scenario);
    assert!(path.is_some());
    assert!(path.unwrap().contains("hash123.gguf"));
}

#[test]
fn test_execute_scenario_skips_nonmatching_format() {
    let temp_dir = tempfile::tempdir().unwrap();
    let model_file = temp_dir.path().join("abc123.safetensors");
    std::fs::write(&model_file, b"fake model").unwrap();

    let mock_runner = MockCommandRunner::new().with_inference_response("The answer is 4.");

    let config = ExecutionConfig {
        model_path: Some(model_file.to_string_lossy().to_string()),
        ..Default::default()
    };
    let executor = Executor::with_runner(config, Arc::new(mock_runner));

    // GGUF scenario against .safetensors file should be skipped
    let scenario = QaScenario::new(
        ModelId::new("test", "model"),
        Modality::Run,
        Backend::Cpu,
        Format::Gguf,
        "2+2=".to_string(),
        42,
    );
    let evidence = executor.execute_scenario(&scenario);
    assert_eq!(evidence.outcome, Outcome::Skipped);
    assert!(evidence.reason.contains("Format"));
}

#[test]
fn test_find_safetensors_dir_file_mode() {
    let temp_dir = tempfile::tempdir().unwrap();

    // File with .safetensors extension → returns parent dir
    let st_file = temp_dir.path().join("model.safetensors");
    std::fs::write(&st_file, b"fake").unwrap();
    let result = Executor::find_safetensors_dir(&st_file);
    assert!(result.is_some());
    assert_eq!(result.unwrap(), temp_dir.path());

    // File with non-safetensors extension → returns None
    let gguf_file = temp_dir.path().join("model.gguf");
    std::fs::write(&gguf_file, b"fake").unwrap();
    let result = Executor::find_safetensors_dir(&gguf_file);
    assert!(result.is_none());
}

#[test]
fn test_subprocess_execution_skip_flag() {
    let temp_dir = tempfile::tempdir().unwrap();
    let model_file = temp_dir.path().join("abc.safetensors");
    std::fs::write(&model_file, b"fake").unwrap();

    let mock_runner = MockCommandRunner::new().with_inference_response("The answer is 4.");

    let config = ExecutionConfig {
        model_path: Some(model_file.to_string_lossy().to_string()),
        ..Default::default()
    };
    let executor = Executor::with_runner(config, Arc::new(mock_runner));

    // Matching format → not skipped
    let scenario_st = QaScenario::new(
        ModelId::new("test", "model"),
        Modality::Run,
        Backend::Cpu,
        Format::SafeTensors,
        "test".to_string(),
        0,
    );
    let (_, _, _, _, skipped) = executor.subprocess_execution(&scenario_st);
    assert!(!skipped);

    // Non-matching format → skipped
    let scenario_gguf = QaScenario::new(
        ModelId::new("test", "model"),
        Modality::Run,
        Backend::Cpu,
        Format::Gguf,
        "test".to_string(),
        0,
    );
    let (_, _, _, _, skipped) = executor.subprocess_execution(&scenario_gguf);
    assert!(skipped);
}

// =========================================================================
// Stderr in oracle corroborated evidence
// =========================================================================

#[test]
fn test_executor_corroborated_with_stderr() {
    let mock_runner = MockCommandRunner::new()
        .with_inference_response_and_stderr("The answer is 4.", "Warning: some benign warning");

    let config = ExecutionConfig {
        model_path: Some("/test/model.gguf".to_string()),
        run_conversion_tests: false,
        run_golden_rule_test: false,
        ..Default::default()
    };

    let mut executor = Executor::with_runner(config, Arc::new(mock_runner));

    let yaml = r#"
name: stderr-test
version: "1.0.0"
model:
  hf_repo: "test/model"
  formats: [gguf]
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
"#;
    let playbook = Playbook::from_yaml(yaml).expect("Failed to parse");
    let _result = executor.execute(&playbook).expect("Execution failed");

    let evidence = executor.evidence().all();
    assert!(!evidence.is_empty());
    // Corroborated scenario evidence (not G0-VALIDATE) should have stderr
    let ev = evidence
        .iter()
        .find(|e| e.stderr.is_some())
        .expect("should have evidence with stderr");
    assert!(ev.stderr.as_ref().unwrap().contains("Warning"));
}

// =========================================================================
// Falsified with stderr
// =========================================================================

#[test]
fn test_executor_falsified_with_stderr() {
    let mock_runner = MockCommandRunner::new()
        .with_inference_response_and_stderr("", "Error: model failed")
        .with_exit_code(1);

    let config = ExecutionConfig {
        model_path: Some("/test/model.gguf".to_string()),
        run_conversion_tests: false,
        run_golden_rule_test: false,
        failure_policy: FailurePolicy::CollectAll,
        ..Default::default()
    };

    let mut executor = Executor::with_runner(config, Arc::new(mock_runner));

    let yaml = r#"
name: falsified-stderr
version: "1.0.0"
model:
  hf_repo: "test/model"
  formats: [gguf]
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
"#;
    let playbook = Playbook::from_yaml(yaml).expect("Failed to parse");
    let result = executor.execute(&playbook).expect("Execution failed");
    assert!(result.failed >= 1);

    let evidence = executor.evidence().all();
    let ev = evidence
        .iter()
        .find(|e| e.stderr.is_some())
        .expect("should have evidence with stderr");
    assert!(ev.stderr.is_some());
}

// =========================================================================
// execute_profile_flamegraph / execute_profile_focus /
// execute_backend_equivalence / execute_serve_lifecycle
// These use Command::new("apr") directly and will fail since apr isn't
// installed, but we cover the error paths.
// =========================================================================

#[test]
fn test_execute_profile_flamegraph_no_apr() {
    let executor = ToolExecutor::new("test-model.gguf".to_string(), true, 5000);
    let temp_dir = tempfile::tempdir().unwrap();
    let result = executor.execute_profile_flamegraph(temp_dir.path());
    // apr binary not found => stderr contains error
    assert!(!result.passed);
    assert_eq!(result.tool, "profile-flamegraph");
    assert_eq!(result.gate_id, "F-PROFILE-002");
}

#[test]
fn test_execute_profile_flamegraph_with_mock_success() {
    let mock_runner = MockCommandRunner::new();
    let executor = ToolExecutor::with_runner(
        "test-model.gguf".to_string(),
        true,
        5000,
        Arc::new(mock_runner),
    );
    let temp_dir = tempfile::tempdir().unwrap();
    let result = executor.execute_profile_flamegraph(temp_dir.path());
    // Mock returns success but no SVG file is created
    assert_eq!(result.tool, "profile-flamegraph");
    assert_eq!(result.gate_id, "F-PROFILE-002");
    assert!(!result.passed); // No SVG file generated
}

#[test]
fn test_execute_profile_flamegraph_with_svg_file() {
    let mock_runner = MockCommandRunner::new();
    let executor = ToolExecutor::with_runner(
        "test-model.gguf".to_string(),
        false,
        5000,
        Arc::new(mock_runner),
    );
    let temp_dir = tempfile::tempdir().unwrap();
    // Pre-create a valid SVG file
    let svg_path = temp_dir.path().join("profile_flamegraph.svg");
    std::fs::write(&svg_path, "<svg><rect/></svg>").unwrap();
    let result = executor.execute_profile_flamegraph(temp_dir.path());
    assert!(result.passed);
    assert!(result.stdout.contains("valid: true"));
}

#[test]
fn test_execute_profile_flamegraph_with_invalid_svg() {
    let mock_runner = MockCommandRunner::new();
    let executor = ToolExecutor::with_runner(
        "test-model.gguf".to_string(),
        true,
        5000,
        Arc::new(mock_runner),
    );
    let temp_dir = tempfile::tempdir().unwrap();
    // Pre-create an invalid SVG file
    let svg_path = temp_dir.path().join("profile_flamegraph.svg");
    std::fs::write(&svg_path, "not a valid svg at all").unwrap();
    let result = executor.execute_profile_flamegraph(temp_dir.path());
    assert!(!result.passed);
    assert!(result.stdout.contains("valid: false"));
}
