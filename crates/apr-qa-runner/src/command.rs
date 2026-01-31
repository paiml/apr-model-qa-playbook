//! Command execution abstraction for testability
//!
//! This module provides a trait-based abstraction over subprocess execution,
//! allowing the executor code to be tested with mock implementations.

use std::path::Path;

/// Result of executing a command
#[derive(Debug, Clone)]
pub struct CommandOutput {
    /// Standard output
    pub stdout: String,
    /// Standard error
    pub stderr: String,
    /// Exit code (negative for signals)
    pub exit_code: i32,
    /// Whether the command succeeded
    pub success: bool,
}

impl CommandOutput {
    /// Create a successful command output
    #[must_use]
    pub fn success(stdout: impl Into<String>) -> Self {
        Self {
            stdout: stdout.into(),
            stderr: String::new(),
            exit_code: 0,
            success: true,
        }
    }

    /// Create a failed command output
    #[must_use]
    pub fn failure(exit_code: i32, stderr: impl Into<String>) -> Self {
        Self {
            stdout: String::new(),
            stderr: stderr.into(),
            exit_code,
            success: false,
        }
    }

    /// Create output with both stdout and stderr
    #[must_use]
    pub fn with_output(
        stdout: impl Into<String>,
        stderr: impl Into<String>,
        exit_code: i32,
    ) -> Self {
        Self {
            stdout: stdout.into(),
            stderr: stderr.into(),
            exit_code,
            success: exit_code == 0,
        }
    }
}

/// Trait for executing shell commands
///
/// This abstraction allows for mocking subprocess execution in tests.
pub trait CommandRunner: Send + Sync {
    /// Execute an apr run command
    fn run_inference(
        &self,
        model_path: &Path,
        prompt: &str,
        max_tokens: u32,
        no_gpu: bool,
        extra_args: &[&str],
    ) -> CommandOutput;

    /// Execute an apr convert command
    fn convert_model(&self, source: &Path, target: &Path) -> CommandOutput;

    /// Execute an apr rosetta inspect command
    fn inspect_model(&self, model_path: &Path) -> CommandOutput;

    /// Execute an apr validate command
    fn validate_model(&self, model_path: &Path) -> CommandOutput;

    /// Execute an apr bench command
    fn bench_model(&self, model_path: &Path) -> CommandOutput;

    /// Execute an apr check command
    fn check_model(&self, model_path: &Path) -> CommandOutput;

    /// Execute an apr profile command
    fn profile_model(&self, model_path: &Path, warmup: u32, measure: u32) -> CommandOutput;

    /// Execute apr profile in CI mode
    fn profile_ci(
        &self,
        model_path: &Path,
        min_throughput: Option<f64>,
        max_p99: Option<f64>,
        warmup: u32,
        measure: u32,
    ) -> CommandOutput;

    /// Execute apr rosetta diff-tensors
    fn diff_tensors(&self, model_a: &Path, model_b: &Path, json: bool) -> CommandOutput;

    /// Execute apr rosetta compare-inference
    fn compare_inference(
        &self,
        model_a: &Path,
        model_b: &Path,
        prompt: &str,
        max_tokens: u32,
        tolerance: f64,
    ) -> CommandOutput;
}

/// Real command runner that executes actual subprocess commands
#[derive(Debug, Clone)]
pub struct RealCommandRunner {
    /// Path to apr binary (default: "apr")
    pub apr_binary: String,
}

impl Default for RealCommandRunner {
    fn default() -> Self {
        Self::new()
    }
}

impl RealCommandRunner {
    /// Create a new real command runner
    #[must_use]
    pub fn new() -> Self {
        Self {
            apr_binary: "apr".to_string(),
        }
    }

    /// Create with custom apr binary path
    #[must_use]
    pub fn with_binary(apr_binary: impl Into<String>) -> Self {
        Self {
            apr_binary: apr_binary.into(),
        }
    }

    fn execute(&self, args: &[&str]) -> CommandOutput {
        use std::process::Command;

        match Command::new(&self.apr_binary).args(args).output() {
            Ok(output) => CommandOutput {
                stdout: String::from_utf8_lossy(&output.stdout).to_string(),
                stderr: String::from_utf8_lossy(&output.stderr).to_string(),
                exit_code: output.status.code().unwrap_or(-1),
                success: output.status.success(),
            },
            Err(e) => CommandOutput::failure(-1, format!("Failed to execute command: {e}")),
        }
    }
}

impl CommandRunner for RealCommandRunner {
    fn run_inference(
        &self,
        model_path: &Path,
        prompt: &str,
        max_tokens: u32,
        no_gpu: bool,
        extra_args: &[&str],
    ) -> CommandOutput {
        let model_str = model_path.display().to_string();
        let max_tokens_str = max_tokens.to_string();

        let mut args = vec![
            "run",
            &model_str,
            "-p",
            prompt,
            "--max-tokens",
            &max_tokens_str,
        ];

        if no_gpu {
            args.push("--no-gpu");
        }

        args.extend(extra_args.iter());
        self.execute(&args)
    }

    fn convert_model(&self, source: &Path, target: &Path) -> CommandOutput {
        let source_str = source.display().to_string();
        let target_str = target.display().to_string();
        self.execute(&["rosetta", "convert", &source_str, &target_str])
    }

    fn inspect_model(&self, model_path: &Path) -> CommandOutput {
        let path_str = model_path.display().to_string();
        self.execute(&["rosetta", "inspect", &path_str])
    }

    fn validate_model(&self, model_path: &Path) -> CommandOutput {
        let path_str = model_path.display().to_string();
        self.execute(&["validate", &path_str])
    }

    fn bench_model(&self, model_path: &Path) -> CommandOutput {
        let path_str = model_path.display().to_string();
        self.execute(&["bench", &path_str])
    }

    fn check_model(&self, model_path: &Path) -> CommandOutput {
        let path_str = model_path.display().to_string();
        self.execute(&["check", &path_str])
    }

    fn profile_model(&self, model_path: &Path, warmup: u32, measure: u32) -> CommandOutput {
        let path_str = model_path.display().to_string();
        let warmup_str = warmup.to_string();
        let measure_str = measure.to_string();
        self.execute(&[
            "profile",
            &path_str,
            "--warmup",
            &warmup_str,
            "--measure",
            &measure_str,
        ])
    }

    fn profile_ci(
        &self,
        model_path: &Path,
        min_throughput: Option<f64>,
        max_p99: Option<f64>,
        warmup: u32,
        measure: u32,
    ) -> CommandOutput {
        let path_str = model_path.display().to_string();
        let warmup_str = warmup.to_string();
        let measure_str = measure.to_string();

        let mut args = vec![
            "profile",
            &path_str,
            "--ci",
            "--warmup",
            &warmup_str,
            "--measure",
            &measure_str,
            "--json",
        ];

        let throughput_str;
        if let Some(t) = min_throughput {
            throughput_str = t.to_string();
            args.push("--assert-throughput");
            args.push(&throughput_str);
        }

        let p99_str;
        if let Some(p) = max_p99 {
            p99_str = p.to_string();
            args.push("--assert-p99");
            args.push(&p99_str);
        }

        self.execute(&args)
    }

    fn diff_tensors(&self, model_a: &Path, model_b: &Path, json: bool) -> CommandOutput {
        let a_str = model_a.display().to_string();
        let b_str = model_b.display().to_string();

        let mut args = vec!["rosetta", "diff-tensors", &a_str, &b_str];
        if json {
            args.push("--json");
        }
        self.execute(&args)
    }

    fn compare_inference(
        &self,
        model_a: &Path,
        model_b: &Path,
        prompt: &str,
        max_tokens: u32,
        tolerance: f64,
    ) -> CommandOutput {
        let a_str = model_a.display().to_string();
        let b_str = model_b.display().to_string();
        let max_tokens_str = max_tokens.to_string();
        let tolerance_str = tolerance.to_string();

        self.execute(&[
            "rosetta",
            "compare-inference",
            &a_str,
            &b_str,
            "--prompt",
            prompt,
            "--max-tokens",
            &max_tokens_str,
            "--tolerance",
            &tolerance_str,
            "--json",
        ])
    }
}

/// Mock command runner for testing
///
/// This struct uses many boolean flags intentionally - each flag controls
/// an independent success/failure behavior for testing different scenarios.
#[derive(Debug, Clone)]
#[allow(clippy::struct_excessive_bools)]
pub struct MockCommandRunner {
    /// Default response for inference
    pub inference_response: String,
    /// Whether inference should succeed
    pub inference_success: bool,
    /// Default response for convert
    pub convert_success: bool,
    /// Tokens per second to report
    pub tps: f64,
    /// Simulate a crash (negative exit code)
    pub crash: bool,
    /// Custom stderr message for inference
    pub inference_stderr: Option<String>,
    /// Simulate profile_ci feature not available
    pub profile_ci_unavailable: bool,
    /// Custom stderr for profile_ci
    pub profile_ci_stderr: Option<String>,
    /// Whether inspect should fail
    pub inspect_success: bool,
    /// Whether validate should fail
    pub validate_success: bool,
    /// Whether bench should fail
    pub bench_success: bool,
    /// Whether check should fail
    pub check_success: bool,
    /// Whether profile should fail
    pub profile_success: bool,
    /// Whether diff_tensors should fail
    pub diff_tensors_success: bool,
    /// Whether compare_inference should fail
    pub compare_inference_success: bool,
    /// Custom exit code (if Some, overrides normal exit code logic)
    pub custom_exit_code: Option<i32>,
}

impl Default for MockCommandRunner {
    fn default() -> Self {
        Self {
            inference_response: "The answer is 4.".to_string(),
            inference_success: true,
            convert_success: true,
            tps: 25.0,
            crash: false,
            inference_stderr: None,
            profile_ci_unavailable: false,
            profile_ci_stderr: None,
            inspect_success: true,
            validate_success: true,
            bench_success: true,
            check_success: true,
            profile_success: true,
            diff_tensors_success: true,
            compare_inference_success: true,
            custom_exit_code: None,
        }
    }
}

impl MockCommandRunner {
    /// Create a new mock runner with default responses
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the inference response
    #[must_use]
    pub fn with_inference_response(mut self, response: impl Into<String>) -> Self {
        self.inference_response = response.into();
        self
    }

    /// Set whether inference should fail
    #[must_use]
    pub fn with_inference_failure(mut self) -> Self {
        self.inference_success = false;
        self
    }

    /// Set whether convert should fail
    #[must_use]
    pub fn with_convert_failure(mut self) -> Self {
        self.convert_success = false;
        self
    }

    /// Set the TPS to report
    #[must_use]
    pub fn with_tps(mut self, tps: f64) -> Self {
        self.tps = tps;
        self
    }

    /// Simulate a crash (negative exit code)
    #[must_use]
    pub fn with_crash(mut self) -> Self {
        self.crash = true;
        self
    }

    /// Set the inference response with custom stderr
    #[must_use]
    pub fn with_inference_response_and_stderr(
        mut self,
        response: impl Into<String>,
        stderr: impl Into<String>,
    ) -> Self {
        self.inference_response = response.into();
        self.inference_stderr = Some(stderr.into());
        self
    }

    /// Simulate profile_ci feature not available
    #[must_use]
    pub fn with_profile_ci_unavailable(mut self) -> Self {
        self.profile_ci_unavailable = true;
        self
    }

    /// Set custom stderr for profile_ci
    #[must_use]
    pub fn with_profile_ci_stderr(mut self, stderr: impl Into<String>) -> Self {
        self.profile_ci_stderr = Some(stderr.into());
        self
    }

    /// Set whether inspect should fail
    #[must_use]
    pub fn with_inspect_failure(mut self) -> Self {
        self.inspect_success = false;
        self
    }

    /// Set whether validate should fail
    #[must_use]
    pub fn with_validate_failure(mut self) -> Self {
        self.validate_success = false;
        self
    }

    /// Set whether bench should fail
    #[must_use]
    pub fn with_bench_failure(mut self) -> Self {
        self.bench_success = false;
        self
    }

    /// Set whether check should fail
    #[must_use]
    pub fn with_check_failure(mut self) -> Self {
        self.check_success = false;
        self
    }

    /// Set whether profile should fail
    #[must_use]
    pub fn with_profile_failure(mut self) -> Self {
        self.profile_success = false;
        self
    }

    /// Set whether diff_tensors should fail
    #[must_use]
    pub fn with_diff_tensors_failure(mut self) -> Self {
        self.diff_tensors_success = false;
        self
    }

    /// Set whether compare_inference should fail
    #[must_use]
    pub fn with_compare_inference_failure(mut self) -> Self {
        self.compare_inference_success = false;
        self
    }

    /// Set a custom exit code for inference
    #[must_use]
    pub fn with_exit_code(mut self, code: i32) -> Self {
        self.custom_exit_code = Some(code);
        self
    }
}

impl CommandRunner for MockCommandRunner {
    fn run_inference(
        &self,
        _model_path: &Path,
        prompt: &str,
        _max_tokens: u32,
        _no_gpu: bool,
        _extra_args: &[&str],
    ) -> CommandOutput {
        // Custom exit code takes precedence
        if let Some(exit_code) = self.custom_exit_code {
            return CommandOutput {
                stdout: String::new(),
                stderr: "Custom exit code error".to_string(),
                exit_code,
                success: exit_code == 0,
            };
        }

        // Simulate crash
        if self.crash {
            return CommandOutput {
                stdout: String::new(),
                stderr: "SIGSEGV: Segmentation fault".to_string(),
                exit_code: -11, // SIGSEGV
                success: false,
            };
        }

        if !self.inference_success {
            return CommandOutput::failure(1, "Inference failed");
        }

        // Generate appropriate response based on prompt
        let response = if prompt.contains("2+2") || prompt.contains("2 + 2") {
            "The answer is 4.".to_string()
        } else if prompt.starts_with("def ") || prompt.starts_with("fn ") {
            "    return result".to_string()
        } else if prompt.is_empty() {
            String::new()
        } else {
            self.inference_response.clone()
        };

        let stdout = format!(
            "Output:\n{}\nCompleted in 1.5s\ntok/s: {:.1}",
            response, self.tps
        );

        // Return with stderr if set
        if let Some(ref stderr) = self.inference_stderr {
            CommandOutput::with_output(stdout, stderr.clone(), 0)
        } else {
            CommandOutput::success(stdout)
        }
    }

    fn convert_model(&self, _source: &Path, _target: &Path) -> CommandOutput {
        if self.convert_success {
            CommandOutput::success("Conversion successful")
        } else {
            CommandOutput::failure(1, "Conversion failed")
        }
    }

    fn inspect_model(&self, _model_path: &Path) -> CommandOutput {
        if self.inspect_success {
            CommandOutput::success(r#"{"format":"GGUF","tensors":100,"parameters":"1.5B"}"#)
        } else {
            CommandOutput::failure(1, "Inspect failed: invalid model format")
        }
    }

    fn validate_model(&self, _model_path: &Path) -> CommandOutput {
        if self.validate_success {
            CommandOutput::success("Model validation passed")
        } else {
            CommandOutput::failure(1, "Validation failed: corrupted tensors")
        }
    }

    fn bench_model(&self, _model_path: &Path) -> CommandOutput {
        if self.bench_success {
            let output = format!(
                r#"{{"throughput_tps":{:.1},"latency_p50_ms":78.2,"latency_p99_ms":156.5}}"#,
                self.tps
            );
            CommandOutput::success(output)
        } else {
            CommandOutput::failure(1, "Benchmark failed: model load error")
        }
    }

    fn check_model(&self, _model_path: &Path) -> CommandOutput {
        if self.check_success {
            CommandOutput::success("All checks passed")
        } else {
            CommandOutput::failure(1, "Check failed: safety issues detected")
        }
    }

    fn profile_model(&self, _model_path: &Path, _warmup: u32, _measure: u32) -> CommandOutput {
        if self.profile_success {
            let output = format!(
                r#"{{"throughput_tps":{:.1},"latency_p50_ms":78.2,"latency_p99_ms":156.5}}"#,
                self.tps
            );
            CommandOutput::success(output)
        } else {
            CommandOutput::failure(1, "Profile failed: insufficient memory")
        }
    }

    fn profile_ci(
        &self,
        _model_path: &Path,
        min_throughput: Option<f64>,
        max_p99: Option<f64>,
        _warmup: u32,
        _measure: u32,
    ) -> CommandOutput {
        // Simulate feature not available
        if self.profile_ci_unavailable {
            let stderr = self.profile_ci_stderr.clone().unwrap_or_else(|| {
                "unexpected argument '--ci': apr profile does not support --ci mode".to_string()
            });
            return CommandOutput::with_output("", stderr, 1);
        }

        let throughput_pass = min_throughput.is_none_or(|t| self.tps >= t);
        let p99_pass = max_p99.is_none_or(|p| 156.5 <= p);
        let passed = throughput_pass && p99_pass;

        let output = format!(
            r#"{{"throughput_tps":{:.1},"latency_p50_ms":78.2,"latency_p99_ms":156.5,"passed":{}}}"#,
            self.tps, passed
        );

        if passed {
            CommandOutput::success(output)
        } else {
            CommandOutput::with_output(output, "", 1)
        }
    }

    fn diff_tensors(&self, _model_a: &Path, _model_b: &Path, json: bool) -> CommandOutput {
        if !self.diff_tensors_success {
            return CommandOutput::failure(1, "Diff tensors failed: incompatible models");
        }
        if json {
            CommandOutput::success(
                r#"{"total_tensors":100,"mismatched_tensors":0,"transposed_tensors":0,"mismatches":[],"passed":true}"#,
            )
        } else {
            CommandOutput::success("All tensors match")
        }
    }

    fn compare_inference(
        &self,
        _model_a: &Path,
        _model_b: &Path,
        _prompt: &str,
        _max_tokens: u32,
        _tolerance: f64,
    ) -> CommandOutput {
        if self.compare_inference_success {
            CommandOutput::success(
                r#"{"total_tokens":10,"matching_tokens":10,"max_logit_diff":0.0001,"passed":true,"token_comparisons":[]}"#,
            )
        } else {
            CommandOutput::failure(1, "Compare inference failed: output mismatch")
        }
    }
}

#[cfg(test)]
mod tests {
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
}
