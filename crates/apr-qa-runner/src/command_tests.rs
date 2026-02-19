    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_command_output_success() {
        let output = CommandOutput::success("hello");
        assert!(output.success);
        assert_eq!(output.exit_code, 0);
        assert_eq!(output.stdout, "hello");
        assert!(output.stderr.is_empty());
    }

    #[test]
    fn test_command_output_failure() {
        let output = CommandOutput::failure(1, "error message");
        assert!(!output.success);
        assert_eq!(output.exit_code, 1);
        assert!(output.stdout.is_empty());
        assert_eq!(output.stderr, "error message");
    }

    #[test]
    fn test_command_output_with_output() {
        let output = CommandOutput::with_output("out", "err", 0);
        assert!(output.success);
        assert_eq!(output.stdout, "out");
        assert_eq!(output.stderr, "err");

        let output2 = CommandOutput::with_output("out", "err", 1);
        assert!(!output2.success);
    }

    #[test]
    fn test_mock_runner_default() {
        let runner = MockCommandRunner::new();
        assert!(runner.inference_success);
        assert!(runner.convert_success);
        assert!((runner.tps - 25.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_mock_runner_inference_2plus2() {
        let runner = MockCommandRunner::new();
        let path = PathBuf::from("model.gguf");
        let output = runner.run_inference(&path, "What is 2+2?", 32, false, &[]);
        assert!(output.success);
        assert!(output.stdout.contains("4"));
    }

    #[test]
    fn test_mock_runner_inference_code() {
        let runner = MockCommandRunner::new();
        let path = PathBuf::from("model.gguf");
        let output = runner.run_inference(&path, "def fibonacci(n):", 32, false, &[]);
        assert!(output.success);
        assert!(output.stdout.contains("return"));
    }

    #[test]
    fn test_mock_runner_inference_empty() {
        let runner = MockCommandRunner::new();
        let path = PathBuf::from("model.gguf");
        let output = runner.run_inference(&path, "", 32, false, &[]);
        assert!(output.success);
        // Empty prompt produces empty response content
    }

    #[test]
    fn test_mock_runner_inference_generic() {
        let runner = MockCommandRunner::new().with_inference_response("Custom response");
        let path = PathBuf::from("model.gguf");
        let output = runner.run_inference(&path, "Hello world", 32, false, &[]);
        assert!(output.success);
        assert!(output.stdout.contains("Custom response"));
    }

    #[test]
    fn test_mock_runner_inference_failure() {
        let runner = MockCommandRunner::new().with_inference_failure();
        let path = PathBuf::from("model.gguf");
        let output = runner.run_inference(&path, "test", 32, false, &[]);
        assert!(!output.success);
        assert_eq!(output.exit_code, 1);
    }

    #[test]
    fn test_mock_runner_convert_success() {
        let runner = MockCommandRunner::new();
        let source = PathBuf::from("source.gguf");
        let target = PathBuf::from("target.apr");
        let output = runner.convert_model(&source, &target);
        assert!(output.success);
    }

    #[test]
    fn test_mock_runner_convert_failure() {
        let runner = MockCommandRunner::new().with_convert_failure();
        let source = PathBuf::from("source.gguf");
        let target = PathBuf::from("target.apr");
        let output = runner.convert_model(&source, &target);
        assert!(!output.success);
    }

    #[test]
    fn test_mock_runner_inspect() {
        let runner = MockCommandRunner::new();
        let path = PathBuf::from("model.gguf");
        let output = runner.inspect_model(&path);
        assert!(output.success);
        assert!(output.stdout.contains("GGUF"));
    }

    #[test]
    fn test_mock_runner_validate() {
        let runner = MockCommandRunner::new();
        let path = PathBuf::from("model.gguf");
        let output = runner.validate_model(&path);
        assert!(output.success);
    }

    #[test]
    fn test_mock_runner_bench() {
        let runner = MockCommandRunner::new().with_tps(30.0);
        let path = PathBuf::from("model.gguf");
        let output = runner.bench_model(&path);
        assert!(output.success);
        assert!(output.stdout.contains("30.0"));
    }

    #[test]
    fn test_mock_runner_check() {
        let runner = MockCommandRunner::new();
        let path = PathBuf::from("model.gguf");
        let output = runner.check_model(&path);
        assert!(output.success);
    }

    #[test]
    fn test_mock_runner_profile() {
        let runner = MockCommandRunner::new();
        let path = PathBuf::from("model.gguf");
        let output = runner.profile_model(&path, 1, 2);
        assert!(output.success);
        assert!(output.stdout.contains("throughput_tps"));
    }

    #[test]
    fn test_mock_runner_profile_ci_pass() {
        let runner = MockCommandRunner::new().with_tps(20.0);
        let path = PathBuf::from("model.gguf");
        let output = runner.profile_ci(&path, Some(10.0), Some(200.0), 1, 2);
        assert!(output.success);
        assert!(output.stdout.contains("\"passed\":true"));
    }

    #[test]
    fn test_mock_runner_profile_ci_fail_throughput() {
        let runner = MockCommandRunner::new().with_tps(5.0);
        let path = PathBuf::from("model.gguf");
        let output = runner.profile_ci(&path, Some(10.0), None, 1, 2);
        assert!(!output.success);
        assert!(output.stdout.contains("\"passed\":false"));
    }

    #[test]
    fn test_mock_runner_profile_ci_fail_p99() {
        let runner = MockCommandRunner::new();
        let path = PathBuf::from("model.gguf");
        // p99 is 156.5ms, threshold is 100ms
        let output = runner.profile_ci(&path, None, Some(100.0), 1, 2);
        assert!(!output.success);
    }

    #[test]
    fn test_mock_runner_diff_tensors_json() {
        let runner = MockCommandRunner::new();
        let a = PathBuf::from("a.gguf");
        let b = PathBuf::from("b.apr");
        let output = runner.diff_tensors(&a, &b, true);
        assert!(output.success);
        assert!(output.stdout.contains("\"passed\":true"));
    }

    #[test]
    fn test_mock_runner_diff_tensors_text() {
        let runner = MockCommandRunner::new();
        let a = PathBuf::from("a.gguf");
        let b = PathBuf::from("b.apr");
        let output = runner.diff_tensors(&a, &b, false);
        assert!(output.success);
        assert!(output.stdout.contains("match"));
    }

    #[test]
    fn test_mock_runner_compare_inference() {
        let runner = MockCommandRunner::new();
        let a = PathBuf::from("a.gguf");
        let b = PathBuf::from("b.apr");
        let output = runner.compare_inference(&a, &b, "test prompt", 10, 1e-5);
        assert!(output.success);
        assert!(output.stdout.contains("\"passed\":true"));
    }

    #[test]
    fn test_real_runner_new() {
        let runner = RealCommandRunner::new();
        assert_eq!(runner.apr_binary, "apr");
    }

    #[test]
    fn test_real_runner_with_binary() {
        let runner = RealCommandRunner::with_binary("/custom/apr");
        assert_eq!(runner.apr_binary, "/custom/apr");
    }

    #[test]
    fn test_mock_runner_with_tps() {
        let runner = MockCommandRunner::new().with_tps(100.0);
        assert!((runner.tps - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_mock_runner_chained_config() {
        let runner = MockCommandRunner::new()
            .with_tps(50.0)
            .with_inference_response("Custom")
            .with_convert_failure();

        assert!((runner.tps - 50.0).abs() < f64::EPSILON);
        assert_eq!(runner.inference_response, "Custom");
        assert!(!runner.convert_success);
    }

    #[test]
    fn test_command_output_clone() {
        let output = CommandOutput::success("test");
        let cloned = output.clone();
        assert_eq!(cloned.stdout, output.stdout);
        assert_eq!(cloned.success, output.success);
    }

    #[test]
    fn test_command_output_debug() {
        let output = CommandOutput::success("test");
        let debug_str = format!("{output:?}");
        assert!(debug_str.contains("CommandOutput"));
    }

    #[test]
    fn test_mock_runner_clone() {
        let runner = MockCommandRunner::new().with_tps(42.0);
        let cloned = runner.clone();
        assert!((cloned.tps - 42.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_mock_runner_debug() {
        let runner = MockCommandRunner::new();
        let debug_str = format!("{runner:?}");
        assert!(debug_str.contains("MockCommandRunner"));
    }

    #[test]
    fn test_real_runner_clone() {
        let runner = RealCommandRunner::with_binary("custom");
        let cloned = runner.clone();
        assert_eq!(cloned.apr_binary, "custom");
    }

    #[test]
    fn test_real_runner_debug() {
        let runner = RealCommandRunner::new();
        let debug_str = format!("{runner:?}");
        assert!(debug_str.contains("RealCommandRunner"));
    }

    #[test]
    fn test_real_runner_default() {
        let runner = RealCommandRunner::default();
        assert_eq!(runner.apr_binary, "apr");
    }

    #[test]
    fn test_mock_runner_with_crash() {
        let runner = MockCommandRunner::new().with_crash();
        assert!(runner.crash);
        let path = PathBuf::from("model.gguf");
        let output = runner.run_inference(&path, "test", 32, false, &[]);
        assert!(!output.success);
        assert_eq!(output.exit_code, -11); // SIGSEGV
        assert!(output.stderr.contains("SIGSEGV"));
    }

    #[test]
    fn test_mock_runner_with_inference_response_and_stderr() {
        let runner =
            MockCommandRunner::new().with_inference_response_and_stderr("Response", "Warning");
        assert_eq!(runner.inference_response, "Response");
        assert_eq!(runner.inference_stderr.as_deref(), Some("Warning"));

        let path = PathBuf::from("model.gguf");
        let output = runner.run_inference(&path, "Hello", 32, false, &[]);
        assert!(output.success);
        assert!(output.stdout.contains("Response"));
        assert_eq!(output.stderr, "Warning");
    }

    #[test]
    fn test_mock_runner_inference_fn_code() {
        let runner = MockCommandRunner::new();
        let path = PathBuf::from("model.gguf");
        let output = runner.run_inference(&path, "fn main() {}", 32, false, &[]);
        assert!(output.success);
        assert!(output.stdout.contains("return"));
    }

    #[test]
    fn test_mock_runner_inference_2_plus_2_spaced() {
        let runner = MockCommandRunner::new();
        let path = PathBuf::from("model.gguf");
        let output = runner.run_inference(&path, "What is 2 + 2?", 32, false, &[]);
        assert!(output.success);
        assert!(output.stdout.contains("4"));
    }

    #[test]
    fn test_mock_runner_crash_takes_priority() {
        // Crash should take priority over inference failure
        let runner = MockCommandRunner::new()
            .with_crash()
            .with_inference_failure();
        let path = PathBuf::from("model.gguf");
        let output = runner.run_inference(&path, "test", 32, false, &[]);
        // Crash should be returned, not inference failure
        assert_eq!(output.exit_code, -11);
    }

    #[test]
    fn test_command_output_with_output_success_on_zero() {
        let output = CommandOutput::with_output("stdout", "stderr", 0);
        assert!(output.success);
        assert_eq!(output.exit_code, 0);
    }

    #[test]
    fn test_command_output_with_output_failure_on_nonzero() {
        let output = CommandOutput::with_output("", "error", 42);
        assert!(!output.success);
        assert_eq!(output.exit_code, 42);
    }

    #[test]
    fn test_mock_runner_profile_ci_no_assertions() {
        let runner = MockCommandRunner::new().with_tps(15.0);
        let path = PathBuf::from("model.gguf");
        // No throughput or p99 assertions
        let output = runner.profile_ci(&path, None, None, 1, 2);
        assert!(output.success);
        assert!(output.stdout.contains("\"passed\":true"));
    }

    #[test]
    fn test_mock_runner_fields_after_default() {
        let runner = MockCommandRunner::default();
        assert!(!runner.crash);
        assert!(runner.inference_stderr.is_none());
    }

    #[test]
    fn test_command_output_failure_negative_exit_code() {
        let output = CommandOutput::failure(-9, "killed");
        assert!(!output.success);
        assert_eq!(output.exit_code, -9);
        assert_eq!(output.stderr, "killed");
    }

    #[test]
    fn test_mock_runner_with_all_options() {
        let runner = MockCommandRunner::new()
            .with_tps(100.0)
            .with_inference_response("Custom response")
            .with_crash();

        assert!((runner.tps - 100.0).abs() < f64::EPSILON);
        assert_eq!(runner.inference_response, "Custom response");
        assert!(runner.crash);
    }

    #[test]
    fn test_mock_runner_profile_ci_both_assertions_pass() {
        let runner = MockCommandRunner::new().with_tps(200.0);
        let path = PathBuf::from("model.gguf");
        // Both assertions should pass
        let output = runner.profile_ci(&path, Some(100.0), Some(500.0), 1, 2);
        assert!(output.success);
        assert!(output.stdout.contains("\"passed\":true"));
    }

    #[test]
    fn test_mock_runner_profile_ci_both_assertions_fail() {
        let runner = MockCommandRunner::new().with_tps(5.0);
        let path = PathBuf::from("model.gguf");
        // Throughput too low, p99 too high (156.5 > 100)
        let output = runner.profile_ci(&path, Some(100.0), Some(100.0), 1, 2);
        assert!(!output.success);
        assert!(output.stdout.contains("\"passed\":false"));
    }

    #[test]
    fn test_mock_runner_profile_ci_unavailable() {
        let runner = MockCommandRunner::new().with_profile_ci_unavailable();
        let path = PathBuf::from("model.gguf");
        let output = runner.profile_ci(&path, Some(10.0), None, 1, 2);
        assert!(!output.success);
        assert!(output.stderr.contains("unexpected argument"));
    }

    #[test]
    fn test_mock_runner_profile_ci_custom_stderr() {
        let runner = MockCommandRunner::new()
            .with_profile_ci_unavailable()
            .with_profile_ci_stderr("Custom error: --ci not supported");
        let path = PathBuf::from("model.gguf");
        let output = runner.profile_ci(&path, None, None, 1, 2);
        assert!(!output.success);
        assert!(output.stderr.contains("Custom error"));
    }

    #[test]
    fn test_mock_runner_inspect_failure() {
        let runner = MockCommandRunner::new().with_inspect_failure();
        let path = PathBuf::from("model.gguf");
        let output = runner.inspect_model(&path);
        assert!(!output.success);
        assert!(output.stderr.contains("invalid model format"));
    }

    #[test]
    fn test_mock_runner_validate_failure() {
        let runner = MockCommandRunner::new().with_validate_failure();
        let path = PathBuf::from("model.gguf");
        let output = runner.validate_model(&path);
        assert!(!output.success);
        assert!(output.stderr.contains("corrupted tensors"));
    }

    #[test]
    fn test_mock_runner_bench_failure() {
        let runner = MockCommandRunner::new().with_bench_failure();
        let path = PathBuf::from("model.gguf");
        let output = runner.bench_model(&path);
        assert!(!output.success);
        assert!(output.stderr.contains("model load error"));
    }

    #[test]
    fn test_mock_runner_check_failure() {
        let runner = MockCommandRunner::new().with_check_failure();
        let path = PathBuf::from("model.gguf");
        let output = runner.check_model(&path);
        assert!(!output.success);
        assert!(output.stderr.contains("safety issues"));
    }

    #[test]
    fn test_mock_runner_profile_failure() {
        let runner = MockCommandRunner::new().with_profile_failure();
        let path = PathBuf::from("model.gguf");
        let output = runner.profile_model(&path, 1, 2);
        assert!(!output.success);
        assert!(output.stderr.contains("insufficient memory"));
    }

    #[test]
    fn test_mock_runner_diff_tensors_failure() {
        let runner = MockCommandRunner::new().with_diff_tensors_failure();
        let a = PathBuf::from("a.gguf");
        let b = PathBuf::from("b.apr");
        let output = runner.diff_tensors(&a, &b, true);
        assert!(!output.success);
        assert!(output.stderr.contains("incompatible models"));
    }

    #[test]
    fn test_mock_runner_compare_inference_failure() {
        let runner = MockCommandRunner::new().with_compare_inference_failure();
        let a = PathBuf::from("a.gguf");
        let b = PathBuf::from("b.apr");
        let output = runner.compare_inference(&a, &b, "test", 10, 1e-5);
        assert!(!output.success);
        assert!(output.stderr.contains("output mismatch"));
    }

    #[test]
    fn test_mock_runner_default_new_fields() {
        let runner = MockCommandRunner::default();
        assert!(!runner.profile_ci_unavailable);
        assert!(runner.profile_ci_stderr.is_none());
        assert!(runner.inspect_success);
        assert!(runner.validate_success);
        assert!(runner.bench_success);
        assert!(runner.check_success);
        assert!(runner.profile_success);
        assert!(runner.diff_tensors_success);
        assert!(runner.compare_inference_success);
    }

    #[test]
    fn test_mock_runner_chained_failures() {
        let runner = MockCommandRunner::new()
            .with_inspect_failure()
            .with_validate_failure()
            .with_bench_failure()
            .with_check_failure()
            .with_profile_failure()
            .with_diff_tensors_failure()
            .with_compare_inference_failure();

        assert!(!runner.inspect_success);
        assert!(!runner.validate_success);
        assert!(!runner.bench_success);
        assert!(!runner.check_success);
        assert!(!runner.profile_success);
        assert!(!runner.diff_tensors_success);
        assert!(!runner.compare_inference_success);
    }

    // Tests for RealCommandRunner using nonexistent binary to exercise error paths
    #[test]
    fn test_real_runner_execute_nonexistent_binary() {
        let runner = RealCommandRunner::with_binary("/nonexistent/binary/path");
        let path = PathBuf::from("model.gguf");
        let output = runner.run_inference(&path, "test", 32, false, &[]);
        assert!(!output.success);
        assert_eq!(output.exit_code, -1);
        assert!(output.stderr.contains("Failed to execute"));
    }

    #[test]
    fn test_real_runner_run_inference_with_no_gpu() {
        let runner = RealCommandRunner::with_binary("/nonexistent/binary");
        let path = PathBuf::from("model.gguf");
        let output = runner.run_inference(&path, "test", 32, true, &[]);
        assert!(!output.success);
    }

    #[test]
    fn test_real_runner_run_inference_with_extra_args() {
        let runner = RealCommandRunner::with_binary("/nonexistent/binary");
        let path = PathBuf::from("model.gguf");
        let output = runner.run_inference(&path, "test", 32, false, &["--temp", "0.8"]);
        assert!(!output.success);
    }

    #[test]
    fn test_real_runner_convert_model() {
        let runner = RealCommandRunner::with_binary("/nonexistent/binary");
        let source = PathBuf::from("source.gguf");
        let target = PathBuf::from("target.apr");
        let output = runner.convert_model(&source, &target);
        assert!(!output.success);
    }

    #[test]
    fn test_real_runner_inspect_model() {
        let runner = RealCommandRunner::with_binary("/nonexistent/binary");
        let path = PathBuf::from("model.gguf");
        let output = runner.inspect_model(&path);
        assert!(!output.success);
    }

    #[test]
    fn test_real_runner_validate_model() {
        let runner = RealCommandRunner::with_binary("/nonexistent/binary");
        let path = PathBuf::from("model.gguf");
        let output = runner.validate_model(&path);
        assert!(!output.success);
    }

    #[test]
    fn test_real_runner_bench_model() {
        let runner = RealCommandRunner::with_binary("/nonexistent/binary");
        let path = PathBuf::from("model.gguf");
        let output = runner.bench_model(&path);
        assert!(!output.success);
    }

    #[test]
    fn test_real_runner_check_model() {
        let runner = RealCommandRunner::with_binary("/nonexistent/binary");
        let path = PathBuf::from("model.gguf");
        let output = runner.check_model(&path);
        assert!(!output.success);
    }

    #[test]
    fn test_real_runner_profile_model() {
        let runner = RealCommandRunner::with_binary("/nonexistent/binary");
        let path = PathBuf::from("model.gguf");
        let output = runner.profile_model(&path, 5, 10);
        assert!(!output.success);
    }

    #[test]
    fn test_real_runner_profile_ci_all_options() {
        let runner = RealCommandRunner::with_binary("/nonexistent/binary");
        let path = PathBuf::from("model.gguf");
        let output = runner.profile_ci(&path, Some(10.0), Some(100.0), 5, 10);
        assert!(!output.success);
    }

    #[test]
    fn test_real_runner_profile_ci_throughput_only() {
        let runner = RealCommandRunner::with_binary("/nonexistent/binary");
        let path = PathBuf::from("model.gguf");
        let output = runner.profile_ci(&path, Some(50.0), None, 1, 1);
        assert!(!output.success);
    }

    #[test]
    fn test_real_runner_profile_ci_p99_only() {
        let runner = RealCommandRunner::with_binary("/nonexistent/binary");
        let path = PathBuf::from("model.gguf");
        let output = runner.profile_ci(&path, None, Some(200.0), 1, 1);
        assert!(!output.success);
    }

    #[test]
    fn test_real_runner_profile_ci_no_options() {
        let runner = RealCommandRunner::with_binary("/nonexistent/binary");
        let path = PathBuf::from("model.gguf");
        let output = runner.profile_ci(&path, None, None, 1, 1);
        assert!(!output.success);
    }

    #[test]
    fn test_real_runner_diff_tensors_json() {
        let runner = RealCommandRunner::with_binary("/nonexistent/binary");
        let a = PathBuf::from("a.gguf");
        let b = PathBuf::from("b.apr");
        let output = runner.diff_tensors(&a, &b, true);
        assert!(!output.success);
    }

    #[test]
    fn test_real_runner_diff_tensors_text() {
        let runner = RealCommandRunner::with_binary("/nonexistent/binary");
        let a = PathBuf::from("a.gguf");
        let b = PathBuf::from("b.apr");
        let output = runner.diff_tensors(&a, &b, false);
        assert!(!output.success);
    }

    #[test]
    fn test_real_runner_compare_inference() {
        let runner = RealCommandRunner::with_binary("/nonexistent/binary");
        let a = PathBuf::from("a.gguf");
        let b = PathBuf::from("b.apr");
        let output = runner.compare_inference(&a, &b, "prompt", 10, 1e-5);
        assert!(!output.success);
    }

    #[test]
    fn test_mock_runner_profile_flamegraph_success() {
        let runner = MockCommandRunner::new();
        let model = PathBuf::from("model.gguf");
        let output_path = PathBuf::from("/tmp/profile.svg");
        let output = runner.profile_with_flamegraph(&model, &output_path, false);
        assert!(output.success);
        assert!(output.stdout.contains("flamegraph"));
    }

    #[test]
    fn test_mock_runner_profile_flamegraph_failure() {
        let runner = MockCommandRunner::new().with_profile_flamegraph_failure();
        let model = PathBuf::from("model.gguf");
        let output_path = PathBuf::from("/tmp/profile.svg");
        let output = runner.profile_with_flamegraph(&model, &output_path, false);
        assert!(!output.success);
        assert!(output.stderr.contains("profiler error"));
    }

    #[test]
    fn test_mock_runner_profile_focus_success() {
        let runner = MockCommandRunner::new().with_tps(42.0);
        let model = PathBuf::from("model.gguf");
        let output = runner.profile_with_focus(&model, "attention", false);
        assert!(output.success);
        assert!(output.stdout.contains("42.0"));
    }

    #[test]
    fn test_mock_runner_profile_focus_failure() {
        let runner = MockCommandRunner::new().with_profile_focus_failure();
        let model = PathBuf::from("model.gguf");
        let output = runner.profile_with_focus(&model, "attention", false);
        assert!(!output.success);
        assert!(output.stderr.contains("invalid focus target"));
    }

    #[test]
    fn test_real_runner_profile_flamegraph() {
        let runner = RealCommandRunner::with_binary("/nonexistent/binary");
        let model = PathBuf::from("model.gguf");
        let output_path = PathBuf::from("/tmp/profile.svg");
        let output = runner.profile_with_flamegraph(&model, &output_path, false);
        assert!(!output.success);
    }

    #[test]
    fn test_real_runner_profile_flamegraph_no_gpu() {
        let runner = RealCommandRunner::with_binary("/nonexistent/binary");
        let model = PathBuf::from("model.gguf");
        let output_path = PathBuf::from("/tmp/profile.svg");
        let output = runner.profile_with_flamegraph(&model, &output_path, true);
        assert!(!output.success);
    }

    #[test]
    fn test_real_runner_profile_focus() {
        let runner = RealCommandRunner::with_binary("/nonexistent/binary");
        let model = PathBuf::from("model.gguf");
        let output = runner.profile_with_focus(&model, "attention", false);
        assert!(!output.success);
    }

    #[test]
    fn test_real_runner_profile_focus_no_gpu() {
        let runner = RealCommandRunner::with_binary("/nonexistent/binary");
        let model = PathBuf::from("model.gguf");
        let output = runner.profile_with_focus(&model, "matmul", true);
        assert!(!output.success);
    }

    #[test]
    fn test_mock_runner_default_new_profile_fields() {
        let runner = MockCommandRunner::default();
        assert!(runner.profile_flamegraph_success);
        assert!(runner.profile_focus_success);
    }

    #[test]
    fn test_mock_runner_chained_profile_failures() {
        let runner = MockCommandRunner::new()
            .with_profile_flamegraph_failure()
            .with_profile_focus_failure();
        assert!(!runner.profile_flamegraph_success);
        assert!(!runner.profile_focus_success);
    }

    #[test]
    fn test_mock_runner_validate_strict_success() {
        let runner = MockCommandRunner::new();
        let path = PathBuf::from("model.gguf");
        let output = runner.validate_model_strict(&path);
        assert!(output.success);
        assert!(output.stdout.contains("\"valid\":true"));
    }

    #[test]
    fn test_mock_runner_validate_strict_failure() {
        let runner = MockCommandRunner::new().with_validate_strict_failure();
        let path = PathBuf::from("model.gguf");
        let output = runner.validate_model_strict(&path);
        assert!(!output.success);
        assert!(output.stdout.contains("\"valid\":false"));
        assert!(output.stdout.contains("all-zeros"));
    }

    #[test]
    fn test_mock_runner_validate_strict_default() {
        let runner = MockCommandRunner::default();
        assert!(runner.validate_strict_success);
    }

    #[test]
    fn test_real_runner_validate_strict() {
        let runner = RealCommandRunner::with_binary("/nonexistent/binary");
        let path = PathBuf::from("model.gguf");
        let output = runner.validate_model_strict(&path);
        assert!(!output.success);
    }

    #[test]
    fn test_mock_runner_pull_success() {
        let runner = MockCommandRunner::new();
        let output = runner.pull_model("test/model");
        assert!(output.success);
        assert!(output.stdout.contains("Path: /mock/model.safetensors"));
    }

    #[test]
    fn test_mock_runner_pull_failure() {
        let runner = MockCommandRunner::new().with_pull_failure();
        let output = runner.pull_model("test/model");
        assert!(!output.success);
        assert!(output.stderr.contains("Pull failed"));
    }

    #[test]
    fn test_mock_runner_pull_custom_path() {
        let runner =
            MockCommandRunner::new().with_pull_model_path("/custom/path/model.safetensors");
        let output = runner.pull_model("test/model");
        assert!(output.success);
        assert!(
            output
                .stdout
                .contains("Path: /custom/path/model.safetensors")
        );
    }

    #[test]
    fn test_mock_runner_pull_default() {
        let runner = MockCommandRunner::default();
        assert!(runner.pull_success);
        assert_eq!(runner.pull_model_path, "/mock/model.safetensors");
    }

    #[test]
    fn test_real_runner_pull_model() {
        let runner = RealCommandRunner::with_binary("/nonexistent/binary");
        let output = runner.pull_model("test/model");
        assert!(!output.success);
    }

    // ── Ollama parity tests (GH-6/AC-2) ────────────────────────────────

    #[test]
    fn test_mock_runner_ollama_inference_success() {
        let runner = MockCommandRunner::new();
        let output = runner.run_ollama_inference("qwen2.5-coder:7b-q4_k_m", "What is 2+2?", 0.0);
        assert!(output.success);
        assert!(output.stdout.contains("The answer is 4."));
    }

    #[test]
    fn test_mock_runner_ollama_inference_custom_response() {
        let runner = MockCommandRunner::new().with_ollama_response("Custom ollama response");
        let output = runner.run_ollama_inference("qwen2.5-coder:7b", "Hello", 0.7);
        assert!(output.success);
        assert!(output.stdout.contains("Custom ollama response"));
    }

    #[test]
    fn test_mock_runner_ollama_inference_failure() {
        let runner = MockCommandRunner::new().with_ollama_failure();
        let output = runner.run_ollama_inference("qwen2.5-coder:7b", "test", 0.0);
        assert!(!output.success);
        assert!(output.stderr.contains("Ollama inference failed"));
    }

    #[test]
    fn test_mock_runner_ollama_pull_success() {
        let runner = MockCommandRunner::new();
        let output = runner.pull_ollama_model("qwen2.5-coder:7b-q4_k_m");
        assert!(output.success);
        assert!(output.stdout.contains("pulling manifest"));
    }

    #[test]
    fn test_mock_runner_ollama_pull_failure() {
        let runner = MockCommandRunner::new().with_ollama_pull_failure();
        let output = runner.pull_ollama_model("nonexistent:model");
        assert!(!output.success);
        assert!(output.stderr.contains("Ollama pull failed"));
    }

    #[test]
    fn test_mock_runner_ollama_default_fields() {
        let runner = MockCommandRunner::default();
        assert!(runner.ollama_success);
        assert!(runner.ollama_pull_success);
        assert_eq!(runner.ollama_response, "The answer is 4.");
    }

    // ── New gate methods (F-OLLAMA-003/004/005, F-PERF-003/005) ────────

    #[test]
    fn test_mock_runner_create_ollama_success() {
        let runner = MockCommandRunner::new();
        let path = PathBuf::from("/tmp/Modelfile");
        let output = runner.create_ollama_model("test:latest", &path);
        assert!(output.success);
        assert!(output.stdout.contains("creating model"));
    }

    #[test]
    fn test_mock_runner_create_ollama_failure() {
        let runner = MockCommandRunner::new().with_ollama_create_failure();
        let path = PathBuf::from("/tmp/Modelfile");
        let output = runner.create_ollama_model("test:latest", &path);
        assert!(!output.success);
    }

    #[test]
    fn test_mock_runner_serve_success() {
        let runner = MockCommandRunner::new();
        let path = PathBuf::from("model.gguf");
        let output = runner.serve_model(&path, 8080);
        assert!(output.success);
        assert!(output.stdout.contains("listening"));
    }

    #[test]
    fn test_mock_runner_serve_failure() {
        let runner = MockCommandRunner::new().with_serve_failure();
        let path = PathBuf::from("model.gguf");
        let output = runner.serve_model(&path, 8080);
        assert!(!output.success);
    }

    #[test]
    fn test_mock_runner_http_get_success() {
        let runner = MockCommandRunner::new();
        let output = runner.http_get("http://localhost:8080/v1/models");
        assert!(output.success);
        assert!(output.stdout.contains("models"));
    }

    #[test]
    fn test_mock_runner_http_get_failure() {
        let runner = MockCommandRunner::new().with_http_get_failure();
        let output = runner.http_get("http://localhost:8080/v1/models");
        assert!(!output.success);
    }

    #[test]
    fn test_mock_runner_http_get_custom_response() {
        let runner = MockCommandRunner::new().with_http_get_response(r#"{"status":"ok"}"#);
        let output = runner.http_get("http://localhost:8080/health");
        assert!(output.success);
        assert!(output.stdout.contains("ok"));
    }

    #[test]
    fn test_mock_runner_profile_memory_success() {
        let runner = MockCommandRunner::new();
        let path = PathBuf::from("model.gguf");
        let output = runner.profile_memory(&path);
        assert!(output.success);
        assert!(output.stdout.contains("peak_rss_mb"));
    }

    #[test]
    fn test_mock_runner_profile_memory_failure() {
        let runner = MockCommandRunner::new().with_profile_memory_failure();
        let path = PathBuf::from("model.gguf");
        let output = runner.profile_memory(&path);
        assert!(!output.success);
    }

    #[test]
    fn test_mock_runner_new_default_fields() {
        let runner = MockCommandRunner::default();
        assert!(runner.ollama_create_success);
        assert!(runner.serve_success);
        assert!(runner.http_get_success);
        assert!(runner.profile_memory_success);
    }

    #[test]
    fn test_real_runner_create_ollama_model() {
        // create_ollama_model calls `ollama` binary directly (not apr).
        // With a nonexistent modelfile, it should fail regardless.
        let runner = RealCommandRunner::new();
        let path = PathBuf::from("/nonexistent/path/Modelfile");
        let output = runner.create_ollama_model("apr-test-nonexistent:latest", &path);
        // Either ollama isn't installed (failure) or modelfile is missing (failure)
        // This tests the execution path, not the success case
        assert!(output.exit_code != 0 || !output.success || output.stderr.contains("Error"));
    }

    #[test]
    fn test_real_runner_serve_model() {
        let runner = RealCommandRunner::with_binary("/nonexistent/binary");
        let path = PathBuf::from("model.gguf");
        let output = runner.serve_model(&path, 8080);
        assert!(!output.success);
    }

    #[test]
    fn test_real_runner_profile_memory() {
        let runner = RealCommandRunner::with_binary("/nonexistent/binary");
        let path = PathBuf::from("model.gguf");
        let output = runner.profile_memory(&path);
        assert!(!output.success);
    }

    // ── Bug 200: Chat, HTTP POST, Spawn Serve tests ─────────────────────

    #[test]
    fn test_mock_runner_chat_success_2plus2() {
        let runner = MockCommandRunner::new();
        let path = PathBuf::from("model.gguf");
        let output = runner.run_chat(&path, "What is 2+2?", false, &[]);
        assert!(output.success);
        assert!(output.stdout.contains("4"));
    }

    #[test]
    fn test_mock_runner_chat_success_generic() {
        let runner = MockCommandRunner::new().with_chat_response("Custom chat response");
        let path = PathBuf::from("model.gguf");
        let output = runner.run_chat(&path, "Hello", false, &[]);
        assert!(output.success);
        assert!(output.stdout.contains("Custom chat response"));
    }

    #[test]
    fn test_mock_runner_chat_failure() {
        let runner = MockCommandRunner::new().with_chat_failure();
        let path = PathBuf::from("model.gguf");
        let output = runner.run_chat(&path, "test", false, &[]);
        assert!(!output.success);
        assert!(output.stderr.contains("Chat failed"));
    }

    #[test]
    fn test_mock_runner_http_post_success() {
        let runner = MockCommandRunner::new();
        let output = runner.http_post("http://localhost:8080/v1/completions", "{}");
        assert!(output.success);
        assert!(output.stdout.contains("choices"));
    }

    #[test]
    fn test_mock_runner_http_post_failure() {
        let runner = MockCommandRunner::new().with_http_post_failure();
        let output = runner.http_post("http://localhost:8080/v1/completions", "{}");
        assert!(!output.success);
    }

    #[test]
    fn test_mock_runner_http_post_custom_response() {
        let runner =
            MockCommandRunner::new().with_http_post_response(r#"{"text":"custom output"}"#);
        let output = runner.http_post("http://localhost:8080/v1/completions", "{}");
        assert!(output.success);
        assert!(output.stdout.contains("custom output"));
    }

    #[test]
    fn test_mock_runner_spawn_serve_success() {
        let runner = MockCommandRunner::new();
        let path = PathBuf::from("model.gguf");
        let output = runner.spawn_serve(&path, 8080, false);
        assert!(output.success);
        assert!(output.stdout.contains("12345")); // Mock PID
    }

    #[test]
    fn test_mock_runner_spawn_serve_failure() {
        let runner = MockCommandRunner::new().with_spawn_serve_failure();
        let path = PathBuf::from("model.gguf");
        let output = runner.spawn_serve(&path, 8080, false);
        assert!(!output.success);
    }

    #[test]
    fn test_mock_runner_default_new_chat_fields() {
        let runner = MockCommandRunner::default();
        assert!(runner.chat_success);
        assert!(runner.http_post_success);
        assert!(runner.spawn_serve_success);
    }

    #[test]
    fn test_real_runner_chat_nonexistent() {
        let runner = RealCommandRunner::with_binary("/nonexistent/binary");
        let path = PathBuf::from("model.gguf");
        let output = runner.run_chat(&path, "test", false, &[]);
        assert!(!output.success);
    }

    #[test]
    fn test_real_runner_spawn_serve_nonexistent() {
        let runner = RealCommandRunner::with_binary("/nonexistent/binary");
        let path = PathBuf::from("model.gguf");
        let output = runner.spawn_serve(&path, 8080, false);
        assert!(!output.success);
    }

    // ── Additional coverage tests ──────────────────────────────────────

    #[test]
    fn test_mock_runner_fingerprint_json_success() {
        let runner = MockCommandRunner::new();
        let path = PathBuf::from("model.gguf");
        let output = runner.fingerprint_model(&path, true);
        assert!(output.success);
        assert!(output.stdout.contains("q_proj.weight"));
        assert!(output.stdout.contains("mean"));
    }

    #[test]
    fn test_mock_runner_fingerprint_text_success() {
        let runner = MockCommandRunner::new();
        let path = PathBuf::from("model.gguf");
        let output = runner.fingerprint_model(&path, false);
        assert!(output.success);
        assert!(output.stdout.contains("Fingerprint"));
        assert!(output.stdout.contains("100 tensors"));
    }

    #[test]
    fn test_mock_runner_fingerprint_failure() {
        let runner = MockCommandRunner::new().with_fingerprint_failure();
        let path = PathBuf::from("model.gguf");
        let output = runner.fingerprint_model(&path, true);
        assert!(!output.success);
        assert!(output.stderr.contains("model load error"));
    }

    #[test]
    fn test_mock_runner_validate_stats_success() {
        let runner = MockCommandRunner::new();
        let a = PathBuf::from("fp_a.json");
        let b = PathBuf::from("fp_b.json");
        let output = runner.validate_stats(&a, &b);
        assert!(output.success);
        assert!(output.stdout.contains("\"passed\":true"));
        assert!(output.stdout.contains("\"failed_tensors\":0"));
    }

    #[test]
    fn test_mock_runner_validate_stats_failure() {
        let runner = MockCommandRunner::new().with_validate_stats_failure();
        let a = PathBuf::from("fp_a.json");
        let b = PathBuf::from("fp_b.json");
        let output = runner.validate_stats(&a, &b);
        assert!(!output.success);
        assert!(output.stderr.contains("3 tensors exceed tolerance"));
    }

    #[test]
    fn test_mock_runner_inspect_json_success() {
        let runner = MockCommandRunner::new();
        let path = PathBuf::from("model.safetensors");
        let output = runner.inspect_model_json(&path);
        assert!(output.success);
        assert!(output.stdout.contains("SafeTensors"));
        assert!(output.stdout.contains("tensor_count"));
        assert!(output.stdout.contains("model.embed_tokens.weight"));
        assert!(output.stdout.contains("lm_head.weight"));
    }

    #[test]
    fn test_mock_runner_inspect_json_failure() {
        let runner = MockCommandRunner::new().with_inspect_json_failure();
        let path = PathBuf::from("model.safetensors");
        let output = runner.inspect_model_json(&path);
        assert!(!output.success);
        assert!(output.stderr.contains("invalid model format"));
    }

    #[test]
    fn test_mock_runner_inspect_json_custom_tensor_names() {
        let runner = MockCommandRunner::new().with_tensor_names(vec![
            "layer.0.weight".to_string(),
            "layer.1.weight".to_string(),
        ]);
        let path = PathBuf::from("model.safetensors");
        let output = runner.inspect_model_json(&path);
        assert!(output.success);
        assert!(output.stdout.contains("\"tensor_count\":2"));
        assert!(output.stdout.contains("layer.0.weight"));
        assert!(output.stdout.contains("layer.1.weight"));
    }

    #[test]
    fn test_mock_runner_inspect_json_empty_tensor_names() {
        let runner = MockCommandRunner::new().with_tensor_names(vec![]);
        let path = PathBuf::from("model.safetensors");
        let output = runner.inspect_model_json(&path);
        assert!(output.success);
        assert!(output.stdout.contains("\"tensor_count\":0"));
        assert!(output.stdout.contains("\"tensor_names\":[]"));
    }

    #[test]
    fn test_mock_runner_custom_exit_code() {
        let runner = MockCommandRunner::new().with_exit_code(137);
        let path = PathBuf::from("model.gguf");
        let output = runner.run_inference(&path, "test", 32, false, &[]);
        assert!(!output.success);
        assert_eq!(output.exit_code, 137);
        assert!(output.stderr.contains("Custom exit code error"));
    }

    #[test]
    fn test_mock_runner_custom_exit_code_zero() {
        let runner = MockCommandRunner::new().with_exit_code(0);
        let path = PathBuf::from("model.gguf");
        let output = runner.run_inference(&path, "test", 32, false, &[]);
        assert!(output.success);
        assert_eq!(output.exit_code, 0);
    }

    #[test]
    fn test_mock_runner_custom_exit_code_priority_over_crash() {
        // Custom exit code takes precedence over crash
        let runner = MockCommandRunner::new().with_exit_code(42).with_crash();
        let path = PathBuf::from("model.gguf");
        let output = runner.run_inference(&path, "test", 32, false, &[]);
        assert_eq!(output.exit_code, 42);
    }

    #[test]
    fn test_mock_runner_chat_2_plus_2_spaced() {
        let runner = MockCommandRunner::new();
        let path = PathBuf::from("model.gguf");
        let output = runner.run_chat(&path, "What is 2 + 2?", false, &[]);
        assert!(output.success);
        assert!(output.stdout.contains("4"));
    }

    #[test]
    fn test_real_runner_fingerprint_model() {
        let runner = RealCommandRunner::with_binary("/nonexistent/binary");
        let path = PathBuf::from("model.gguf");
        let output = runner.fingerprint_model(&path, true);
        assert!(!output.success);
        assert!(output.stderr.contains("Failed to execute"));
    }

    #[test]
    fn test_real_runner_fingerprint_model_text() {
        let runner = RealCommandRunner::with_binary("/nonexistent/binary");
        let path = PathBuf::from("model.gguf");
        let output = runner.fingerprint_model(&path, false);
        assert!(!output.success);
    }

    #[test]
    fn test_real_runner_validate_stats() {
        let runner = RealCommandRunner::with_binary("/nonexistent/binary");
        let a = PathBuf::from("fp_a.json");
        let b = PathBuf::from("fp_b.json");
        let output = runner.validate_stats(&a, &b);
        assert!(!output.success);
        assert!(output.stderr.contains("Failed to execute"));
    }

    #[test]
    fn test_real_runner_inspect_model_json() {
        let runner = RealCommandRunner::with_binary("/nonexistent/binary");
        let path = PathBuf::from("model.safetensors");
        let output = runner.inspect_model_json(&path);
        assert!(!output.success);
        assert!(output.stderr.contains("Failed to execute"));
    }

    #[test]
    fn test_real_runner_spawn_serve_no_gpu() {
        let runner = RealCommandRunner::with_binary("/nonexistent/binary");
        let path = PathBuf::from("model.gguf");
        let output = runner.spawn_serve(&path, 9090, true);
        assert!(!output.success);
        assert!(output.stderr.contains("Failed to spawn serve"));
    }

    #[test]
    fn test_real_runner_chat_no_gpu_with_extra_args() {
        let runner = RealCommandRunner::with_binary("/nonexistent/binary");
        let path = PathBuf::from("model.gguf");
        let output = runner.run_chat(&path, "test", true, &["--temp", "0.5"]);
        assert!(!output.success);
        assert!(output.stderr.contains("Failed to execute chat"));
    }

    #[test]
    fn test_mock_runner_fingerprint_default() {
        let runner = MockCommandRunner::default();
        assert!(runner.fingerprint_success);
        assert!(runner.validate_stats_success);
        assert!(runner.inspect_json_success);
        assert_eq!(runner.inspect_tensor_names.len(), 10);
    }

    #[test]
    fn test_mock_runner_chained_new_failures() {
        let runner = MockCommandRunner::new()
            .with_fingerprint_failure()
            .with_validate_stats_failure()
            .with_inspect_json_failure()
            .with_pull_failure()
            .with_ollama_failure()
            .with_ollama_pull_failure()
            .with_ollama_create_failure()
            .with_serve_failure()
            .with_http_get_failure()
            .with_profile_memory_failure()
            .with_chat_failure()
            .with_http_post_failure()
            .with_spawn_serve_failure();

        assert!(!runner.fingerprint_success);
        assert!(!runner.validate_stats_success);
        assert!(!runner.inspect_json_success);
        assert!(!runner.pull_success);
        assert!(!runner.ollama_success);
        assert!(!runner.ollama_pull_success);
        assert!(!runner.ollama_create_success);
        assert!(!runner.serve_success);
        assert!(!runner.http_get_success);
        assert!(!runner.profile_memory_success);
        assert!(!runner.chat_success);
        assert!(!runner.http_post_success);
        assert!(!runner.spawn_serve_success);
    }

    #[test]
    fn test_mock_runner_chat_tps_in_output() {
        let runner = MockCommandRunner::new().with_tps(88.5);
        let path = PathBuf::from("model.gguf");
        let output = runner.run_chat(&path, "Hello world", false, &[]);
        assert!(output.success);
        assert!(output.stdout.contains("88.5"));
        assert!(output.stdout.contains("tok/s"));
    }

    #[test]
    fn test_command_output_failure_with_empty_stderr() {
        let output = CommandOutput::failure(2, "");
        assert!(!output.success);
        assert_eq!(output.exit_code, 2);
        assert!(output.stdout.is_empty());
        assert!(output.stderr.is_empty());
    }

    #[test]
    fn test_command_output_success_with_empty_stdout() {
        let output = CommandOutput::success("");
        assert!(output.success);
        assert_eq!(output.exit_code, 0);
        assert!(output.stdout.is_empty());
    }

    #[test]
    fn test_command_output_with_output_negative_exit_code() {
        let output = CommandOutput::with_output("out", "err", -1);
        assert!(!output.success);
        assert_eq!(output.exit_code, -1);
        assert_eq!(output.stdout, "out");
        assert_eq!(output.stderr, "err");
    }

    #[test]
    fn test_mock_runner_inference_tps_in_output() {
        let runner = MockCommandRunner::new().with_tps(55.3);
        let path = PathBuf::from("model.gguf");
        let output = runner.run_inference(&path, "Hello", 32, false, &[]);
        assert!(output.success);
        assert!(output.stdout.contains("55.3"));
    }
