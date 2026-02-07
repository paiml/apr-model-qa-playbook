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

    /// Execute apr run with --profile and --profile-output for flamegraph
    fn profile_with_flamegraph(
        &self,
        model_path: &Path,
        output_path: &Path,
        no_gpu: bool,
    ) -> CommandOutput;

    /// Execute apr run with --profile and --focus
    fn profile_with_focus(&self, model_path: &Path, focus: &str, no_gpu: bool) -> CommandOutput;

    /// Execute an apr validate command with --strict --json flags
    ///
    /// Runs physics-level validation: detects NaN, Inf, and all-zeros tensors
    /// in model weights. Used by the G0-VALIDATE pre-flight gate.
    fn validate_model_strict(&self, model_path: &Path) -> CommandOutput;

    /// Execute apr rosetta fingerprint to capture tensor statistics
    fn fingerprint_model(&self, model_path: &Path, json: bool) -> CommandOutput;

    /// Execute apr rosetta validate-stats to compare tensor statistics
    fn validate_stats(&self, fp_a: &Path, fp_b: &Path) -> CommandOutput;

    /// Execute `apr pull --json <hf_repo>` to acquire model from cache or remote
    fn pull_model(&self, hf_repo: &str) -> CommandOutput;

    /// Execute `apr rosetta inspect --json` to get model metadata including tensor names
    ///
    /// Returns JSON output with tensor_count, tensor_names, and other model metadata.
    /// Used by G0-TENSOR-001 for tensor template validation (PMAT-271).
    fn inspect_model_json(&self, model_path: &Path) -> CommandOutput;

    /// Execute `ollama run <model_tag>` for parity testing (GH-6/AC-2)
    fn run_ollama_inference(
        &self,
        model_tag: &str,
        prompt: &str,
        temperature: f64,
    ) -> CommandOutput;

    /// Execute `ollama pull <model_tag>` to acquire model (GH-6/AC-2)
    fn pull_ollama_model(&self, model_tag: &str) -> CommandOutput;

    /// Execute `ollama create <tag> -f <modelfile>` to register a GGUF with ollama (F-OLLAMA-005)
    fn create_ollama_model(&self, model_tag: &str, modelfile_path: &Path) -> CommandOutput;

    /// Execute `apr serve` and return immediately (F-OLLAMA-004)
    ///
    /// The returned output contains the PID or server info in stdout.
    fn serve_model(&self, model_path: &Path, port: u16) -> CommandOutput;

    /// Execute an HTTP GET request (F-OLLAMA-004)
    fn http_get(&self, url: &str) -> CommandOutput;

    /// Execute `apr profile --memory` for memory usage (F-PERF-005)
    fn profile_memory(&self, model_path: &Path) -> CommandOutput;
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

    fn validate_model_strict(&self, model_path: &Path) -> CommandOutput {
        let path_str = model_path.display().to_string();
        self.execute(&["validate", "--strict", "--json", &path_str])
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

    fn profile_with_flamegraph(
        &self,
        model_path: &Path,
        output_path: &Path,
        no_gpu: bool,
    ) -> CommandOutput {
        let model_str = model_path.display().to_string();
        let output_str = output_path.display().to_string();

        let mut args = vec![
            "run",
            &model_str,
            "-p",
            "Hello",
            "--max-tokens",
            "4",
            "--profile",
            "--profile-output",
            &output_str,
        ];

        if no_gpu {
            args.push("--no-gpu");
        }

        self.execute(&args)
    }

    fn profile_with_focus(&self, model_path: &Path, focus: &str, no_gpu: bool) -> CommandOutput {
        let model_str = model_path.display().to_string();

        let mut args = vec![
            "run",
            &model_str,
            "-p",
            "Hello",
            "--max-tokens",
            "4",
            "--profile",
            "--focus",
            focus,
        ];

        if no_gpu {
            args.push("--no-gpu");
        }

        self.execute(&args)
    }

    fn fingerprint_model(&self, model_path: &Path, json: bool) -> CommandOutput {
        let path_str = model_path.display().to_string();
        let mut args = vec!["rosetta", "fingerprint", &path_str];
        if json {
            args.push("--json");
        }
        self.execute(&args)
    }

    fn validate_stats(&self, fp_a: &Path, fp_b: &Path) -> CommandOutput {
        let a_str = fp_a.display().to_string();
        let b_str = fp_b.display().to_string();
        self.execute(&["rosetta", "validate-stats", &a_str, &b_str])
    }

    fn pull_model(&self, hf_repo: &str) -> CommandOutput {
        self.execute(&["pull", "--json", hf_repo])
    }

    fn inspect_model_json(&self, model_path: &Path) -> CommandOutput {
        let path_str = model_path.display().to_string();
        self.execute(&["rosetta", "inspect", "--json", &path_str])
    }

    fn run_ollama_inference(
        &self,
        model_tag: &str,
        prompt: &str,
        temperature: f64,
    ) -> CommandOutput {
        use std::process::Command;

        let temp_str = temperature.to_string();
        match Command::new("ollama")
            .args(["run", model_tag, "--temp", &temp_str])
            .arg(prompt)
            .output()
        {
            Ok(output) => CommandOutput {
                stdout: String::from_utf8_lossy(&output.stdout).to_string(),
                stderr: String::from_utf8_lossy(&output.stderr).to_string(),
                exit_code: output.status.code().unwrap_or(-1),
                success: output.status.success(),
            },
            Err(e) => CommandOutput::failure(-1, format!("Failed to execute ollama: {e}")),
        }
    }

    fn pull_ollama_model(&self, model_tag: &str) -> CommandOutput {
        use std::process::Command;

        match Command::new("ollama").args(["pull", model_tag]).output() {
            Ok(output) => CommandOutput {
                stdout: String::from_utf8_lossy(&output.stdout).to_string(),
                stderr: String::from_utf8_lossy(&output.stderr).to_string(),
                exit_code: output.status.code().unwrap_or(-1),
                success: output.status.success(),
            },
            Err(e) => CommandOutput::failure(-1, format!("Failed to execute ollama: {e}")),
        }
    }

    fn create_ollama_model(&self, model_tag: &str, modelfile_path: &Path) -> CommandOutput {
        use std::process::Command;

        let path_str = modelfile_path.display().to_string();
        match Command::new("ollama")
            .args(["create", model_tag, "-f", &path_str])
            .output()
        {
            Ok(output) => CommandOutput {
                stdout: String::from_utf8_lossy(&output.stdout).to_string(),
                stderr: String::from_utf8_lossy(&output.stderr).to_string(),
                exit_code: output.status.code().unwrap_or(-1),
                success: output.status.success(),
            },
            Err(e) => CommandOutput::failure(-1, format!("Failed to execute ollama create: {e}")),
        }
    }

    fn serve_model(&self, model_path: &Path, port: u16) -> CommandOutput {
        let model_str = model_path.display().to_string();
        let port_str = port.to_string();
        self.execute(&["serve", &model_str, "--port", &port_str])
    }

    fn http_get(&self, url: &str) -> CommandOutput {
        use std::process::Command;

        match Command::new("curl").args(["-s", "-m", "10", url]).output() {
            Ok(output) => CommandOutput {
                stdout: String::from_utf8_lossy(&output.stdout).to_string(),
                stderr: String::from_utf8_lossy(&output.stderr).to_string(),
                exit_code: output.status.code().unwrap_or(-1),
                success: output.status.success(),
            },
            Err(e) => CommandOutput::failure(-1, format!("Failed to execute curl: {e}")),
        }
    }

    fn profile_memory(&self, model_path: &Path) -> CommandOutput {
        let path_str = model_path.display().to_string();
        self.execute(&["profile", &path_str, "--memory", "--json"])
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
    /// Whether profile_with_flamegraph should fail
    pub profile_flamegraph_success: bool,
    /// Whether profile_with_focus should fail
    pub profile_focus_success: bool,
    /// Whether fingerprint_model should fail
    pub fingerprint_success: bool,
    /// Whether validate_stats should fail
    pub validate_stats_success: bool,
    /// Whether validate_model_strict should fail
    pub validate_strict_success: bool,
    /// Whether pull_model should succeed
    pub pull_success: bool,
    /// Path returned by pull_model on success
    pub pull_model_path: String,
    /// Whether inspect_model_json should succeed
    pub inspect_json_success: bool,
    /// Tensor names returned by inspect_model_json
    pub inspect_tensor_names: Vec<String>,
    /// Whether ollama inference should succeed
    pub ollama_success: bool,
    /// Custom response for ollama inference
    pub ollama_response: String,
    /// Whether ollama pull should succeed
    pub ollama_pull_success: bool,
    /// Whether ollama create should succeed
    pub ollama_create_success: bool,
    /// Whether serve_model should succeed
    pub serve_success: bool,
    /// Whether http_get should succeed
    pub http_get_success: bool,
    /// Custom HTTP response body
    pub http_get_response: String,
    /// Whether profile_memory should succeed
    pub profile_memory_success: bool,
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
            profile_flamegraph_success: true,
            profile_focus_success: true,
            fingerprint_success: true,
            validate_stats_success: true,
            validate_strict_success: true,
            pull_success: true,
            pull_model_path: "/mock/model.safetensors".to_string(),
            inspect_json_success: true,
            inspect_tensor_names: vec![
                "model.embed_tokens.weight".to_string(),
                "model.layers.0.self_attn.q_proj.weight".to_string(),
                "model.layers.0.self_attn.k_proj.weight".to_string(),
                "model.layers.0.self_attn.v_proj.weight".to_string(),
                "model.layers.0.self_attn.o_proj.weight".to_string(),
                "model.layers.0.mlp.gate_proj.weight".to_string(),
                "model.layers.0.mlp.up_proj.weight".to_string(),
                "model.layers.0.mlp.down_proj.weight".to_string(),
                "model.norm.weight".to_string(),
                "lm_head.weight".to_string(),
            ],
            ollama_success: true,
            ollama_response: "The answer is 4.".to_string(),
            ollama_pull_success: true,
            ollama_create_success: true,
            serve_success: true,
            http_get_success: true,
            http_get_response: r#"{"models":[]}"#.to_string(),
            profile_memory_success: true,
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

    /// Set whether profile_with_flamegraph should fail
    #[must_use]
    pub fn with_profile_flamegraph_failure(mut self) -> Self {
        self.profile_flamegraph_success = false;
        self
    }

    /// Set whether profile_with_focus should fail
    #[must_use]
    pub fn with_profile_focus_failure(mut self) -> Self {
        self.profile_focus_success = false;
        self
    }

    /// Set whether fingerprint_model should fail
    #[must_use]
    pub fn with_fingerprint_failure(mut self) -> Self {
        self.fingerprint_success = false;
        self
    }

    /// Set whether validate_stats should fail
    #[must_use]
    pub fn with_validate_stats_failure(mut self) -> Self {
        self.validate_stats_success = false;
        self
    }

    /// Set whether validate_model_strict should fail
    #[must_use]
    pub fn with_validate_strict_failure(mut self) -> Self {
        self.validate_strict_success = false;
        self
    }

    /// Set whether pull_model should fail
    #[must_use]
    pub fn with_pull_failure(mut self) -> Self {
        self.pull_success = false;
        self
    }

    /// Set the model path returned by pull_model
    #[must_use]
    pub fn with_pull_model_path(mut self, path: impl Into<String>) -> Self {
        self.pull_model_path = path.into();
        self
    }

    /// Set whether inspect_model_json should fail
    #[must_use]
    pub fn with_inspect_json_failure(mut self) -> Self {
        self.inspect_json_success = false;
        self
    }

    /// Set custom tensor names for inspect_model_json
    #[must_use]
    pub fn with_tensor_names(mut self, names: Vec<String>) -> Self {
        self.inspect_tensor_names = names;
        self
    }

    /// Set custom ollama inference response
    #[must_use]
    pub fn with_ollama_response(mut self, response: impl Into<String>) -> Self {
        self.ollama_response = response.into();
        self
    }

    /// Set whether ollama inference should fail
    #[must_use]
    pub fn with_ollama_failure(mut self) -> Self {
        self.ollama_success = false;
        self
    }

    /// Set whether ollama pull should fail
    #[must_use]
    pub fn with_ollama_pull_failure(mut self) -> Self {
        self.ollama_pull_success = false;
        self
    }

    /// Set whether ollama create should fail
    #[must_use]
    pub fn with_ollama_create_failure(mut self) -> Self {
        self.ollama_create_success = false;
        self
    }

    /// Set whether serve_model should fail
    #[must_use]
    pub fn with_serve_failure(mut self) -> Self {
        self.serve_success = false;
        self
    }

    /// Set whether http_get should fail
    #[must_use]
    pub fn with_http_get_failure(mut self) -> Self {
        self.http_get_success = false;
        self
    }

    /// Set custom HTTP response body
    #[must_use]
    pub fn with_http_get_response(mut self, response: impl Into<String>) -> Self {
        self.http_get_response = response.into();
        self
    }

    /// Set whether profile_memory should fail
    #[must_use]
    pub fn with_profile_memory_failure(mut self) -> Self {
        self.profile_memory_success = false;
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

    fn validate_model_strict(&self, _model_path: &Path) -> CommandOutput {
        if self.validate_strict_success {
            CommandOutput::success(r#"{"valid":true,"tensors_checked":100,"issues":[]}"#)
        } else {
            CommandOutput::with_output(
                r#"{"valid":false,"tensors_checked":100,"issues":["all-zeros tensor: lm_head.weight (6.7GB F32)","expected BF16 but found F32"]}"#,
                "Validation failed: corrupt model detected",
                1,
            )
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

    fn profile_with_flamegraph(
        &self,
        _model_path: &Path,
        _output_path: &Path,
        _no_gpu: bool,
    ) -> CommandOutput {
        if self.profile_flamegraph_success {
            CommandOutput::success("Profile complete, flamegraph written")
        } else {
            CommandOutput::failure(1, "Profile flamegraph failed: profiler error")
        }
    }

    fn profile_with_focus(&self, _model_path: &Path, _focus: &str, _no_gpu: bool) -> CommandOutput {
        if self.profile_focus_success {
            let output = format!(
                r#"{{"throughput_tps":{:.1},"latency_p50_ms":78.2,"latency_p99_ms":156.5}}"#,
                self.tps
            );
            CommandOutput::success(output)
        } else {
            CommandOutput::failure(1, "Profile focus failed: invalid focus target")
        }
    }

    fn fingerprint_model(&self, _model_path: &Path, json: bool) -> CommandOutput {
        if self.fingerprint_success {
            if json {
                CommandOutput::success(
                    r#"{"tensors":{"0.q_proj.weight":{"mean":0.001,"std":0.05,"min":-0.2,"max":0.2}}}"#,
                )
            } else {
                CommandOutput::success("Fingerprint: 100 tensors captured")
            }
        } else {
            CommandOutput::failure(1, "Fingerprint failed: model load error")
        }
    }

    fn validate_stats(&self, _fp_a: &Path, _fp_b: &Path) -> CommandOutput {
        if self.validate_stats_success {
            CommandOutput::success(
                r#"{"passed":true,"total_tensors":100,"failed_tensors":0,"details":[]}"#,
            )
        } else {
            CommandOutput::failure(1, "Stats validation failed: 3 tensors exceed tolerance")
        }
    }

    fn pull_model(&self, _hf_repo: &str) -> CommandOutput {
        if self.pull_success {
            CommandOutput::success(format!("Path: {}", self.pull_model_path))
        } else {
            CommandOutput::failure(1, "Pull failed: model not found in registry")
        }
    }

    fn inspect_model_json(&self, _model_path: &Path) -> CommandOutput {
        if self.inspect_json_success {
            let tensor_names_json: String = self
                .inspect_tensor_names
                .iter()
                .map(|s| format!("\"{s}\""))
                .collect::<Vec<_>>()
                .join(", ");
            CommandOutput::success(format!(
                r#"{{"format":"SafeTensors","tensor_count":{},"tensor_names":[{}],"parameters":"1.5B"}}"#,
                self.inspect_tensor_names.len(),
                tensor_names_json
            ))
        } else {
            CommandOutput::failure(1, "Inspect failed: invalid model format")
        }
    }

    fn run_ollama_inference(
        &self,
        _model_tag: &str,
        _prompt: &str,
        _temperature: f64,
    ) -> CommandOutput {
        if self.ollama_success {
            CommandOutput::success(format!(
                "Output:\n{}\nCompleted in 1.0s",
                self.ollama_response
            ))
        } else {
            CommandOutput::failure(1, "Ollama inference failed: model not found")
        }
    }

    fn pull_ollama_model(&self, _model_tag: &str) -> CommandOutput {
        if self.ollama_pull_success {
            CommandOutput::success("pulling manifest... done")
        } else {
            CommandOutput::failure(1, "Ollama pull failed: model not found in registry")
        }
    }

    fn create_ollama_model(&self, _model_tag: &str, _modelfile_path: &Path) -> CommandOutput {
        if self.ollama_create_success {
            CommandOutput::success("creating model... done")
        } else {
            CommandOutput::failure(1, "Ollama create failed: invalid modelfile")
        }
    }

    fn serve_model(&self, _model_path: &Path, _port: u16) -> CommandOutput {
        if self.serve_success {
            CommandOutput::success(r#"{"status":"listening","port":8080}"#)
        } else {
            CommandOutput::failure(1, "Serve failed: port in use")
        }
    }

    fn http_get(&self, _url: &str) -> CommandOutput {
        if self.http_get_success {
            CommandOutput::success(&self.http_get_response)
        } else {
            CommandOutput::failure(1, "HTTP request failed: connection refused")
        }
    }

    fn profile_memory(&self, _model_path: &Path) -> CommandOutput {
        if self.profile_memory_success {
            CommandOutput::success(r#"{"peak_rss_mb":1024,"model_size_mb":512,"kv_cache_mb":256}"#)
        } else {
            CommandOutput::failure(1, "Profile memory failed: insufficient memory")
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
}
