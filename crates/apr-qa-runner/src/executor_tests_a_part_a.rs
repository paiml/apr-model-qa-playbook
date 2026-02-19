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
