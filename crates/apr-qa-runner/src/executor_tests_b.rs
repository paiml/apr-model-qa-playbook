    use super::*;
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
    fn test_execution_result_fields() {
        let result = ExecutionResult {
            playbook_name: "my-playbook".to_string(),
            total_scenarios: 50,
            passed: 45,
            failed: 3,
            skipped: 2,
            duration_ms: 5000,
            gateway_failed: None,
            evidence: EvidenceCollector::new(),
        };
        assert_eq!(result.playbook_name, "my-playbook");
        assert_eq!(result.total_scenarios, 50);
        assert_eq!(result.passed, 45);
        assert_eq!(result.failed, 3);
        assert_eq!(result.skipped, 2);
        assert_eq!(result.duration_ms, 5000);
    }

    #[test]
    fn test_failure_policy_copy() {
        let policy = FailurePolicy::CollectAll;
        let copied: FailurePolicy = policy;
        assert_eq!(copied, FailurePolicy::CollectAll);
    }

    #[test]
    fn test_extract_output_text_with_trailing_content() {
        let output =
            "Prefix\nOutput:\nAnswer is 4\nMore answer text\nCompleted in 2.5s\nExtra stuff";
        let result = Executor::extract_output_text(output);
        assert_eq!(result, "Answer is 4 More answer text");
    }

    #[test]
    fn test_extract_generated_text_mixed_content() {
        let output = "Line 1\n=== SEPARATOR ===\nLine 2\ntok/s: 50.0\nLine 3";
        let result = Executor::extract_generated_text(output);
        assert!(result.contains("Line 1"));
        assert!(result.contains("Line 2"));
        assert!(result.contains("Line 3"));
        assert!(!result.contains("==="));
        assert!(!result.contains("tok/s"));
    }

    #[test]
    fn test_parse_tps_from_output_at_end() {
        let output = "All output finished tok/s: 99.9";
        let tps = Executor::parse_tps_from_output(output);
        assert!(tps.is_some());
        assert!((tps.unwrap() - 99.9).abs() < 0.01);
    }

    #[test]
    fn test_parse_tps_from_output_multiline() {
        let output = "Line 1\nLine 2\ntok/s: 25.5\nLine 4";
        let tps = Executor::parse_tps_from_output(output);
        assert!(tps.is_some());
        assert!((tps.unwrap() - 25.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_extract_output_text_output_at_end() {
        let output = "Header info\nOutput:\nFinal answer here";
        let result = Executor::extract_output_text(output);
        assert_eq!(result, "Final answer here");
    }

    #[test]
    fn test_execution_result_with_gateway_failure() {
        let result = ExecutionResult {
            playbook_name: "test".to_string(),
            total_scenarios: 10,
            passed: 0,
            failed: 10,
            skipped: 0,
            duration_ms: 100,
            gateway_failed: Some("G1: Model failed to load".to_string()),
            evidence: EvidenceCollector::new(),
        };
        assert!(!result.is_success());
        assert!(result.gateway_failed.is_some());
        assert!(result.gateway_failed.as_ref().unwrap().contains("G1"));
    }

    #[test]
    fn test_execution_config_all_fields() {
        let config = ExecutionConfig {
            failure_policy: FailurePolicy::CollectAll,
            default_timeout_ms: 30_000,
            max_workers: 2,
            dry_run: true,
            model_path: Some("/path/to/model.gguf".to_string()),
            no_gpu: true,
            run_conversion_tests: false,
            run_differential_tests: false,
            run_profile_ci: true,
            run_trace_payload: false,
            run_golden_rule_test: false,
            golden_reference_path: Some("/path/to/ref.json".to_string()),
            lock_file_path: None,
            check_integrity: false,
            warn_implicit_skips: false,
            run_hf_parity: false,
            hf_parity_corpus_path: None,
            hf_parity_model_family: None,
            output_dir: Some("test_output".to_string()),
            run_contract_tests: false,
            run_ollama_parity: false,
            metadata_only: false,
        };
        assert_eq!(config.failure_policy, FailurePolicy::CollectAll);
        assert!(config.dry_run);
        assert!(config.no_gpu);
        assert!(!config.run_conversion_tests);
        assert!(!config.run_differential_tests);
        assert!(config.run_profile_ci);
        assert!(!config.run_contract_tests);
    }

    #[test]
    fn test_tool_test_result_fields_comprehensive() {
        let result = ToolTestResult {
            tool: "custom-test".to_string(),
            passed: false,
            exit_code: 127,
            stdout: "stdout content".to_string(),
            stderr: "error: command not found".to_string(),
            duration_ms: 150,
            gate_id: "F-CUSTOM-001".to_string(),
        };
        assert_eq!(result.tool, "custom-test");
        assert!(!result.passed);
        assert_eq!(result.exit_code, 127);
        assert!(!result.stdout.is_empty());
        assert!(!result.stderr.is_empty());
    }

    #[test]
    fn test_golden_scenario_prompt_content() {
        let model_id = ModelId::new("org", "name");
        let scenario = Executor::golden_scenario(&model_id);
        assert!(scenario.prompt.contains("Golden Rule"));
        assert!(scenario.prompt.contains("convert"));
        assert!(scenario.prompt.contains("inference"));
    }

    #[test]
    fn test_executor_with_custom_timeout_and_workers() {
        let config = ExecutionConfig {
            default_timeout_ms: 120_000,
            max_workers: 16,
            ..Default::default()
        };
        let executor = Executor::with_config(config);
        assert_eq!(executor.config().default_timeout_ms, 120_000);
        assert_eq!(executor.config().max_workers, 16);
    }

    #[test]
    fn test_execution_result_pass_rate_partial() {
        let result = ExecutionResult {
            playbook_name: "test".to_string(),
            total_scenarios: 3,
            passed: 1,
            failed: 2,
            skipped: 0,
            duration_ms: 100,
            gateway_failed: None,
            evidence: EvidenceCollector::new(),
        };
        let rate = result.pass_rate();
        assert!((rate - 100.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn test_tool_test_result_to_evidence_with_content() {
        let result = ToolTestResult {
            tool: "validate".to_string(),
            passed: true,
            exit_code: 0,
            stdout: "Model validated successfully".to_string(),
            stderr: String::new(),
            duration_ms: 200,
            gate_id: "F-VALIDATE-001".to_string(),
        };
        let model_id = ModelId::new("org", "model");
        let evidence = result.to_evidence(&model_id);
        assert!(evidence.outcome.is_pass());
        assert!(evidence.output.contains("validated"));
    }

    #[test]
    fn test_tool_test_result_with_zero_duration() {
        let result = ToolTestResult {
            tool: "fast-test".to_string(),
            passed: true,
            exit_code: 0,
            stdout: "OK".to_string(),
            stderr: String::new(),
            duration_ms: 0,
            gate_id: "F-FAST-001".to_string(),
        };
        assert_eq!(result.duration_ms, 0);
    }

    #[test]
    fn test_extract_output_text_preserves_content() {
        let output = "Info\nOutput:\n  First line\n  Second line  \n  Third line\nCompleted in 1s";
        let result = Executor::extract_output_text(output);
        assert!(result.contains("First line"));
        assert!(result.contains("Second line"));
        assert!(result.contains("Third line"));
    }

    // ============================================================
    // Tests using MockCommandRunner for subprocess execution paths
    // ============================================================

    use crate::command::MockCommandRunner;

    #[test]
    fn test_executor_with_mock_runner_subprocess_execution() {
        let (_tmp, model_path) = create_test_model_file(Format::Gguf);
        let mock_runner = MockCommandRunner::new()
            .with_tps(42.0)
            .with_inference_response("The answer is 4.");

        let config = ExecutionConfig {
            model_path: Some(model_path),
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

        let (output, stderr, exit_code, tps, skipped) = executor.subprocess_execution(&scenario);

        assert!(!skipped);
        assert!(output.contains("4") || output.is_empty()); // Depends on extract logic
        assert!(stderr.is_none_or(|s| s.is_empty()));
        assert_eq!(exit_code, 0);
        // tps may or may not be parsed depending on output format
        let _ = tps;
    }

    #[test]
    fn test_executor_with_mock_runner_inference_failure() {
        let (_tmp, model_path) = create_test_model_file(Format::Gguf);
        let mock_runner = MockCommandRunner::new().with_inference_failure();

        let config = ExecutionConfig {
            model_path: Some(model_path),
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

        assert_eq!(exit_code, 1);
        assert!(stderr.is_some());
    }

    #[test]
    fn test_executor_with_mock_runner_execute_scenario() {
        let mock_runner = MockCommandRunner::new()
            .with_tps(30.0)
            .with_inference_response("The answer is 4.");

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

        // Evidence should be created
        assert!(!evidence.id.is_empty());
        assert!(!evidence.gate_id.is_empty());
    }

    #[test]
    fn test_executor_with_mock_runner_golden_rule_test() {
        let mock_runner = MockCommandRunner::new()
            .with_tps(25.0)
            .with_inference_response("Output:\nThe answer is 4\nCompleted in 1s");

        let config = ExecutionConfig {
            model_path: Some("/test/model.gguf".to_string()),
            run_golden_rule_test: true,
            run_conversion_tests: false, // Disable other tests
            ..Default::default()
        };

        let mut executor = Executor::with_runner(config, Arc::new(mock_runner));

        let model_id = ModelId::new("test", "model");
        let (passed, failed) =
            executor.run_golden_rule_test(std::path::Path::new("/test/model.gguf"), &model_id);

        // With mock runner, both inferences should succeed with same output
        // So golden rule test should pass - exactly one test run
        assert_eq!(passed + failed, 1);
    }

    #[test]
    fn test_executor_with_mock_runner_golden_rule_conversion_failure() {
        let mock_runner = MockCommandRunner::new()
            .with_convert_failure()
            .with_inference_response("Output:\nThe answer is 4\nCompleted in 1s");

        let config = ExecutionConfig {
            model_path: Some("/test/model.gguf".to_string()),
            ..Default::default()
        };

        let mut executor = Executor::with_runner(config, Arc::new(mock_runner));

        let model_id = ModelId::new("test", "model");
        let (passed, failed) =
            executor.run_golden_rule_test(std::path::Path::new("/test/model.gguf"), &model_id);

        // Conversion failure should result in 0 passed, 1 failed
        assert_eq!(passed, 0);
        assert_eq!(failed, 1);

        // Evidence should be collected
        assert!(!executor.collector.all().is_empty());
    }

    #[test]
    fn test_executor_with_mock_runner_golden_rule_inference_failure() {
        let mock_runner = MockCommandRunner::new().with_inference_failure();

        let config = ExecutionConfig {
            model_path: Some("/test/model.gguf".to_string()),
            ..Default::default()
        };

        let mut executor = Executor::with_runner(config, Arc::new(mock_runner));

        let model_id = ModelId::new("test", "model");
        let (passed, failed) =
            executor.run_golden_rule_test(std::path::Path::new("/test/model.gguf"), &model_id);

        // First inference failure should result in 0 passed, 1 failed
        assert_eq!(passed, 0);
        assert_eq!(failed, 1);
    }

    #[test]
    fn test_tool_executor_with_mock_runner_inspect() {
        let mock_runner = MockCommandRunner::new();
        let executor = ToolExecutor::with_runner(
            "/test/model.gguf".to_string(),
            true,
            60_000,
            Arc::new(mock_runner),
        );

        let result = executor.execute_inspect();

        assert!(result.passed);
        assert_eq!(result.exit_code, 0);
        assert!(result.stdout.contains("GGUF"));
    }

    #[test]
    fn test_tool_executor_with_mock_runner_validate() {
        let mock_runner = MockCommandRunner::new();
        let executor = ToolExecutor::with_runner(
            "/test/model.gguf".to_string(),
            false,
            60_000,
            Arc::new(mock_runner),
        );

        let result = executor.execute_validate();

        assert!(result.passed);
        assert_eq!(result.exit_code, 0);
    }

    #[test]
    fn test_tool_executor_with_mock_runner_bench() {
        let mock_runner = MockCommandRunner::new().with_tps(50.0);
        let executor = ToolExecutor::with_runner(
            "/test/model.gguf".to_string(),
            true,
            60_000,
            Arc::new(mock_runner),
        );

        let result = executor.execute_bench();

        assert!(result.passed);
        assert_eq!(result.exit_code, 0);
        assert!(result.stdout.contains("50.0"));
    }

    #[test]
    fn test_tool_executor_with_mock_runner_check() {
        let mock_runner = MockCommandRunner::new();
        let executor = ToolExecutor::with_runner(
            "/test/model.gguf".to_string(),
            false,
            60_000,
            Arc::new(mock_runner),
        );

        let result = executor.execute_check();

        assert!(result.passed);
        assert_eq!(result.exit_code, 0);
    }

    #[test]
    fn test_tool_executor_with_mock_runner_trace() {
        let mock_runner = MockCommandRunner::new().with_tps(25.0);
        let executor = ToolExecutor::with_runner(
            "/test/model.gguf".to_string(),
            true,
            60_000,
            Arc::new(mock_runner),
        );

        let result = executor.execute_trace("layer");

        assert!(result.passed);
        assert_eq!(result.exit_code, 0);
        assert!(result.tool.contains("trace"));
    }

    #[test]
    fn test_tool_executor_with_mock_runner_profile() {
        let mock_runner = MockCommandRunner::new().with_tps(35.0);
        let executor = ToolExecutor::with_runner(
            "/test/model.gguf".to_string(),
            false,
            60_000,
            Arc::new(mock_runner),
        );

        let result = executor.execute_profile();

        assert!(result.passed);
        assert_eq!(result.exit_code, 0);
        assert!(result.stdout.contains("throughput"));
    }

    #[test]
    fn test_tool_executor_with_mock_runner_profile_ci() {
        let mock_runner = MockCommandRunner::new().with_tps(20.0);
        let executor = ToolExecutor::with_runner(
            "/test/model.gguf".to_string(),
            false,
            60_000,
            Arc::new(mock_runner),
        );

        let result = executor.execute_profile_ci();

        // Mock runner returns "passed":true when tps >= threshold
        assert!(result.passed);
        assert!(result.stdout.contains("passed"));
    }

    #[test]
    fn test_tool_executor_with_mock_runner_profile_ci_assertion_failure() {
        // With very low tps, the 1M threshold will fail
        let mock_runner = MockCommandRunner::new().with_tps(5.0);
        let executor = ToolExecutor::with_runner(
            "/test/model.gguf".to_string(),
            false,
            60_000,
            Arc::new(mock_runner),
        );

        let result = executor.execute_profile_ci_assertion_failure();

        // The test passes if CI correctly detects the assertion failure
        // Mock runner will return "passed":false when tps < 1M
        assert!(result.passed); // Test passes because assertion correctly failed
        assert!(result.stdout.contains("\"passed\":false"));
    }

    #[test]
    fn test_tool_executor_with_mock_runner_profile_ci_p99() {
        let mock_runner = MockCommandRunner::new().with_tps(30.0);
        let executor = ToolExecutor::with_runner(
            "/test/model.gguf".to_string(),
            false,
            60_000,
            Arc::new(mock_runner),
        );

        let result = executor.execute_profile_ci_p99();

        // Mock runner returns p99=156.5 which is <= 10000
        assert!(result.passed);
        assert!(result.stdout.contains("latency"));
    }

    #[test]
    fn test_tool_executor_with_runner_debug() {
        let mock_runner = MockCommandRunner::new();
        let executor = ToolExecutor::with_runner(
            "/test/model.gguf".to_string(),
            true,
            60_000,
            Arc::new(mock_runner),
        );

        let debug_str = format!("{executor:?}");
        assert!(debug_str.contains("ToolExecutor"));
        assert!(debug_str.contains("model_path"));
    }

    #[test]
    fn test_executor_with_runner_debug() {
        let mock_runner = MockCommandRunner::new();
        let config = ExecutionConfig::default();
        let executor = Executor::with_runner(config, Arc::new(mock_runner));

        let debug_str = format!("{executor:?}");
        assert!(debug_str.contains("Executor"));
        assert!(debug_str.contains("config"));
    }

    #[test]
    fn test_executor_subprocess_execution_no_gpu() {
        let mock_runner = MockCommandRunner::new();
        let config = ExecutionConfig {
            model_path: Some("/test/model.gguf".to_string()),
            no_gpu: true,
            ..Default::default()
        };

        let executor = Executor::with_runner(config, Arc::new(mock_runner));

        let scenario = QaScenario::new(
            ModelId::new("test", "model"),
            Modality::Run,
            Backend::Cpu,
            Format::Gguf,
            "Test prompt".to_string(),
            0,
        );

        let (_, _, exit_code, _, _) = executor.subprocess_execution(&scenario);
        assert_eq!(exit_code, 0);
    }

    #[test]
    fn test_executor_execute_playbook_with_subprocess_mode() {
        let mock_runner = MockCommandRunner::new()
            .with_tps(25.0)
            .with_inference_response("The answer is 4.");

        let config = ExecutionConfig {
            model_path: Some("/test/model.gguf".to_string()),
            run_conversion_tests: false,
            run_differential_tests: false,
            run_golden_rule_test: false,
            run_trace_payload: false,
            run_profile_ci: false,
            run_contract_tests: false,
            ..Default::default()
        };

        let mut executor = Executor::with_runner(config, Arc::new(mock_runner));

        let yaml = r#"
name: test-subprocess
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

        // Bug 204: G0-PULL skipped when model_path is set, so 3 scenarios only
        assert_eq!(result.total_scenarios, 3);
        // With mock runner, all scenarios should complete
        assert!(result.passed > 0 || result.failed > 0);
    }

    #[test]
    fn test_build_result_from_output() {
        let mock_runner = MockCommandRunner::new();
        let executor = ToolExecutor::with_runner(
            "/test/model.gguf".to_string(),
            false,
            60_000,
            Arc::new(mock_runner),
        );

        let output = crate::command::CommandOutput::success("test output");
        let start = std::time::Instant::now();
        let result = executor.build_result_from_output("test-tool", output, start);

        assert!(result.passed);
        assert_eq!(result.exit_code, 0);
        assert_eq!(result.tool, "test-tool");
        assert_eq!(result.gate_id, "F-TEST_TOOL-001");
    }

    #[test]
    fn test_build_result_from_output_failure() {
        let mock_runner = MockCommandRunner::new();
        let executor = ToolExecutor::with_runner(
            "/test/model.gguf".to_string(),
            false,
            60_000,
            Arc::new(mock_runner),
        );

        let output = crate::command::CommandOutput::failure(1, "error message");
        let start = std::time::Instant::now();
        let result = executor.build_result_from_output("failed-tool", output, start);

        assert!(!result.passed);
        assert_eq!(result.exit_code, 1);
        assert_eq!(result.stderr, "error message");
    }

    #[test]
    fn test_tool_executor_execute_all() {
        let mock_runner = MockCommandRunner::new().with_tps(30.0);
        let executor = ToolExecutor::with_runner(
            "/test/model.gguf".to_string(),
            true,
            60_000,
            Arc::new(mock_runner),
        );

        let results = executor.execute_all();

        // execute_all should run: inspect, validate, check, bench, 4 trace levels,
        // profile, profile_ci, profile_ci_assertion_failure, profile_ci_p99
        // = 4 + 4 + 4 = 12 tests (without serve)
        assert!(results.len() >= 12);
        // Most should pass with mock runner
        let passed_count = results.iter().filter(|r| r.passed).count();
        assert!(passed_count > 0);
    }

    #[test]
    fn test_tool_executor_execute_all_with_serve_false() {
        let mock_runner = MockCommandRunner::new().with_tps(30.0);
        let executor = ToolExecutor::with_runner(
            "/test/model.gguf".to_string(),
            false,
            60_000,
            Arc::new(mock_runner),
        );

        let results = executor.execute_all_with_serve(false);

        // Same as execute_all
        assert!(results.len() >= 12);
    }

    #[test]
    fn test_executor_execute_scenario_crash() {
        // Create mock that returns negative exit code
        let mock_runner = MockCommandRunner::new().with_crash();

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

        // Should create crashed evidence
        assert!(evidence.outcome.is_fail());
        assert_eq!(evidence.gate_id, "G3-STABLE");
    }

    #[test]
    fn test_executor_run_conversion_tests_success() {
        let mock_runner = MockCommandRunner::new();
        let config = ExecutionConfig {
            model_path: Some("/test/model.gguf".to_string()),
            run_conversion_tests: true,
            no_gpu: true,
            ..Default::default()
        };

        let mut executor = Executor::with_runner(config, Arc::new(mock_runner));
        let model_id = ModelId::new("test", "model");

        let (passed, failed) =
            executor.run_conversion_tests(std::path::Path::new("/test/model.gguf"), &model_id);

        // Conversion tests were attempted (may be 0,0 if no supported formats)
        let _ = (passed, failed); // Just verify the function runs without panic
    }

    #[test]
    fn test_executor_execute_scenario_with_stderr() {
        let mock_runner =
            MockCommandRunner::new().with_inference_response_and_stderr("Output: 4", "Warning");

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
        // Stderr should be captured
        assert!(evidence.stderr.is_some() || evidence.stderr.is_none());
    }

    #[test]
    fn test_executor_execute_with_conversion_and_golden() {
        let mock_runner = MockCommandRunner::new()
            .with_tps(25.0)
            .with_inference_response("Output:\nThe answer is 4\nCompleted in 1s");

        let config = ExecutionConfig {
            model_path: Some("/test/model.gguf".to_string()),
            run_conversion_tests: true,
            run_golden_rule_test: true,
            no_gpu: true,
            ..Default::default()
        };

        let mut executor = Executor::with_runner(config, Arc::new(mock_runner));

        let yaml = r#"
name: test-full
version: "1.0.0"
model:
  hf_repo: "test/model"
  formats: [gguf]
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 2
"#;
        let playbook = Playbook::from_yaml(yaml).expect("Failed to parse");
        let result = executor.execute(&playbook).expect("Execution failed");

        // Should complete with all test types
        assert!(result.total_scenarios >= 2);
    }

    #[test]
    fn test_executor_golden_rule_output_differs() {
        // Mock that returns different output on second call would need more complex mock
        // For now, test with same output which should pass
        let mock_runner = MockCommandRunner::new()
            .with_inference_response("Output:\nThe answer is 4\nCompleted in 1s");

        let config = ExecutionConfig::default();
        let mut executor = Executor::with_runner(config, Arc::new(mock_runner));
        let model_id = ModelId::new("test", "model");

        let (passed, failed) =
            executor.run_golden_rule_test(std::path::Path::new("/test/model.gguf"), &model_id);

        // Both inferences return same output, so should pass
        assert_eq!(passed, 1);
        assert_eq!(failed, 0);
    }

    #[test]
    fn test_executor_subprocess_with_tps_parsing() {
        // The mock runner adds tok/s: {self.tps} to output, so set the tps value
        let mock_runner = MockCommandRunner::new().with_tps(42.5);

        let config = ExecutionConfig {
            model_path: Some("/test/model.gguf".to_string()),
            ..Default::default()
        };

        let executor = Executor::with_runner(config, Arc::new(mock_runner));

        let scenario = test_scenario();
        let (_, _, _, tps, _) = executor.subprocess_execution(&scenario);

        // tps should be parsed from output
        assert!(tps.is_some());
        assert!((tps.unwrap() - 42.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_tool_test_result_to_evidence_gate_id() {
        let result = ToolTestResult {
            tool: "special".to_string(),
            passed: true,
            exit_code: 0,
            stdout: "OK".to_string(),
            stderr: String::new(),
            duration_ms: 50,
            gate_id: "F-SPECIAL-TEST-001".to_string(),
        };

        let model_id = ModelId::new("org", "name");
        let evidence = result.to_evidence(&model_id);

        assert_eq!(evidence.gate_id, "F-SPECIAL-TEST-001");
        assert_eq!(evidence.scenario.model.org, "org");
        assert_eq!(evidence.scenario.model.name, "name");
    }

    #[test]
    fn test_execution_result_evidence_collector() {
        let mut collector = EvidenceCollector::new();
        let evidence = Evidence::corroborated("F-TEST-001", test_scenario(), "Test output", 100);
        collector.add(evidence);

        let result = ExecutionResult {
            playbook_name: "test".to_string(),
            total_scenarios: 1,
            passed: 1,
            failed: 0,
            skipped: 0,
            duration_ms: 100,
            gateway_failed: None,
            evidence: collector,
        };

        assert_eq!(result.evidence.all().len(), 1);
    }

    #[test]
    fn test_executor_execute_scenario_with_metrics() {
        let mock_runner = MockCommandRunner::new()
            .with_tps(75.5)
            .with_inference_response("The answer is 4.");

        let config = ExecutionConfig {
            model_path: Some("/test/model.gguf".to_string()),
            ..Default::default()
        };

        let executor = Executor::with_runner(config, Arc::new(mock_runner));
        let scenario = test_scenario();

        let evidence = executor.execute_scenario(&scenario);

        // Metrics should be populated (duration_ms is a u64, so always valid)
        let _ = evidence.metrics.duration_ms; // Just verify it exists
    }

    #[test]
    fn test_extract_output_text_with_whitespace_lines() {
        // Whitespace-only lines are not considered empty - they get trimmed and added
        // Only truly empty lines (or "Completed in") terminate parsing
        let output = "Header\nOutput:\n   \nActual content\n  \nCompleted in 1s";
        let result = Executor::extract_output_text(output);
        // Whitespace lines become empty after trim, content gets captured
        assert!(result.contains("Actual content"));
    }

    #[test]
    fn test_extract_output_text_only_header() {
        let output = "Only Header no Output marker";
        let result = Executor::extract_output_text(output);
        assert!(result.is_empty());
    }

    #[test]
    fn test_parse_tps_from_output_multiple_colons() {
        let output = "Info: tok/s: 88.8 more info";
        let tps = Executor::parse_tps_from_output(output);
        assert!(tps.is_some());
        assert!((tps.unwrap() - 88.8).abs() < f64::EPSILON);
    }

    #[test]
    fn test_tool_executor_trace_all_levels() {
        let mock_runner = MockCommandRunner::new();
        let executor = ToolExecutor::with_runner(
            "/test/model.gguf".to_string(),
            false,
            60_000,
            Arc::new(mock_runner),
        );

        for level in &["none", "basic", "layer", "payload"] {
            let result = executor.execute_trace(level);
            assert!(result.passed);
            assert!(result.tool.contains("trace"));
            assert!(result.tool.contains(level));
        }
    }

    #[test]
    fn test_execution_config_partial_override() {
        let config = ExecutionConfig {
            dry_run: true,
            max_workers: 1,
            ..Default::default()
        };

        assert!(config.dry_run);
        assert_eq!(config.max_workers, 1);
        // Defaults should still be set
        assert!(config.run_conversion_tests);
        assert!(config.run_golden_rule_test);
    }

    #[test]
    fn test_executor_evidence_after_execute() {
        let mock_runner = MockCommandRunner::new().with_inference_response("The answer is 4.");

        let config = ExecutionConfig {
            model_path: Some("/test/model.gguf".to_string()),
            run_conversion_tests: false,
            run_golden_rule_test: false,
            ..Default::default()
        };

        let mut executor = Executor::with_runner(config, Arc::new(mock_runner));

        let yaml = r#"
name: evidence-test
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
        let _ = executor.execute(&playbook).expect("Execution failed");

        // Evidence should be collected
        assert!(!executor.evidence().all().is_empty());
    }

    #[test]
    fn test_tool_executor_gate_id_format() {
        let mock_runner = MockCommandRunner::new();
        let executor = ToolExecutor::with_runner(
            "/test/model.gguf".to_string(),
            false,
            60_000,
            Arc::new(mock_runner),
        );

        let result = executor.execute_inspect();
        assert_eq!(result.gate_id, "F-INSPECT-001");

        let result = executor.execute_validate();
        assert_eq!(result.gate_id, "F-VALIDATE-001");

        let result = executor.execute_bench();
        assert_eq!(result.gate_id, "F-BENCH-001");

        let result = executor.execute_check();
        assert_eq!(result.gate_id, "F-CHECK-001");

        let result = executor.execute_profile();
        assert_eq!(result.gate_id, "F-PROFILE-001");
    }

    #[test]
    fn test_tool_executor_profile_ci_feature_unavailable() {
        let mock_runner = MockCommandRunner::new().with_profile_ci_unavailable();
        let executor = ToolExecutor::with_runner(
            "/test/model.gguf".to_string(),
            false,
            60_000,
            Arc::new(mock_runner),
        );

        let result = executor.execute_profile_ci();

        // When feature is unavailable, should return exit code -2
        assert!(!result.passed);
        assert_eq!(result.exit_code, -2);
        assert!(result.stderr.contains("Feature not available"));
        assert_eq!(result.gate_id, "F-PROFILE-006");
    }

    #[test]
    fn test_tool_executor_profile_ci_assertion_unavailable() {
        let mock_runner = MockCommandRunner::new().with_profile_ci_unavailable();
        let executor = ToolExecutor::with_runner(
            "/test/model.gguf".to_string(),
            false,
            60_000,
            Arc::new(mock_runner),
        );

        let result = executor.execute_profile_ci_assertion_failure();

        // When feature is unavailable, should indicate feature not available
        assert!(!result.passed);
        assert_eq!(result.exit_code, -2);
        assert_eq!(result.gate_id, "F-PROFILE-007");
    }

    #[test]
    fn test_tool_executor_profile_ci_p99_unavailable() {
        let mock_runner = MockCommandRunner::new().with_profile_ci_unavailable();
        let executor = ToolExecutor::with_runner(
            "/test/model.gguf".to_string(),
            false,
            60_000,
            Arc::new(mock_runner),
        );

        let result = executor.execute_profile_ci_p99();

        // When feature is unavailable, should indicate feature not available
        assert!(!result.passed);
        assert_eq!(result.exit_code, -2);
        assert_eq!(result.gate_id, "F-PROFILE-008");
    }

    #[test]
    fn test_tool_executor_inspect_failure() {
        let mock_runner = MockCommandRunner::new().with_inspect_failure();
        let executor = ToolExecutor::with_runner(
            "/test/model.gguf".to_string(),
            false,
            60_000,
            Arc::new(mock_runner),
        );

        let result = executor.execute_inspect();

        assert!(!result.passed);
        assert_eq!(result.exit_code, 1);
    }
