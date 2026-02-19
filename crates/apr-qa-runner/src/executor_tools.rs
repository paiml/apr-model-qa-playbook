/// APR tool coverage executor for validating tool integration.
pub struct ToolExecutor {
    model_path: String,
    no_gpu: bool,
    timeout_ms: u64,
    command_runner: Arc<dyn CommandRunner>,
}

impl std::fmt::Debug for ToolExecutor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolExecutor")
            .field("model_path", &self.model_path)
            .field("no_gpu", &self.no_gpu)
            .field("timeout_ms", &self.timeout_ms)
            .field("command_runner", &"<dyn CommandRunner>")
            .finish()
    }
}

impl ToolExecutor {
    /// Create a new tool executor
    #[must_use]
    pub fn new(model_path: String, no_gpu: bool, timeout_ms: u64) -> Self {
        Self {
            model_path,
            no_gpu,
            timeout_ms,
            command_runner: Arc::new(RealCommandRunner::new()),
        }
    }

    /// Create a new tool executor with custom command runner
    #[must_use]
    pub fn with_runner(
        model_path: String,
        no_gpu: bool,
        timeout_ms: u64,
        runner: Arc<dyn CommandRunner>,
    ) -> Self {
        Self {
            model_path,
            no_gpu,
            timeout_ms,
            command_runner: runner,
        }
    }

    /// Execute apr rosetta inspect (works with any format)
    #[must_use]
    pub fn execute_inspect(&self) -> ToolTestResult {
        let start = std::time::Instant::now();
        let output = self
            .command_runner
            .inspect_model(Path::new(&self.model_path));
        self.build_result_from_output("inspect", output, start)
    }

    /// Execute apr rosetta inspect with metadata verification (T-GH192-01)
    ///
    /// Parses `--json` output and validates that critical model metadata
    /// fields are present and non-zero. This catches models with missing
    /// or corrupted config (e.g., num_heads=0, hidden_size=0).
    ///
    /// Gate: `F-INSPECT-META-001`
    #[must_use]
    pub fn execute_inspect_verified(&self) -> ToolTestResult {
        let start = std::time::Instant::now();

        match crate::differential::run_inspect(Path::new(&self.model_path), "apr") {
            Ok(inspect) => {
                let duration_ms = start.elapsed().as_millis() as u64;
                let mut issues = Vec::new();

                // Verify tensor count is non-zero
                if inspect.tensor_count == 0 {
                    issues.push("tensor_count is 0".to_string());
                }

                // Verify critical metadata (if present, must be non-zero)
                if let Some(heads) = inspect.num_attention_heads {
                    if heads == 0 {
                        issues.push("num_attention_heads is 0".to_string());
                    }
                }

                if let Some(kv_heads) = inspect.num_key_value_heads {
                    if kv_heads == 0 {
                        issues.push("num_key_value_heads is 0".to_string());
                    }
                }

                if let Some(hidden) = inspect.hidden_size {
                    if hidden == 0 {
                        issues.push("hidden_size is 0".to_string());
                    }
                }

                let passed = issues.is_empty();
                let stdout = format!(
                    "tensor_count={}, num_attention_heads={:?}, num_key_value_heads={:?}, \
                     hidden_size={:?}, architecture={:?}",
                    inspect.tensor_count,
                    inspect.num_attention_heads,
                    inspect.num_key_value_heads,
                    inspect.hidden_size,
                    inspect.architecture,
                );

                ToolTestResult {
                    tool: "inspect-verified".to_string(),
                    passed,
                    exit_code: i32::from(!passed),
                    stdout,
                    stderr: if passed {
                        String::new()
                    } else {
                        format!("Metadata issues: {}", issues.join(", "))
                    },
                    duration_ms,
                    gate_id: "F-INSPECT-META-001".to_string(),
                }
            }
            Err(e) => {
                let duration_ms = start.elapsed().as_millis() as u64;
                ToolTestResult {
                    tool: "inspect-verified".to_string(),
                    passed: false,
                    exit_code: -1,
                    stdout: String::new(),
                    stderr: format!("Failed to run inspect: {e}"),
                    duration_ms,
                    gate_id: "F-INSPECT-META-001".to_string(),
                }
            }
        }
    }

    /// Execute apr validate
    #[must_use]
    pub fn execute_validate(&self) -> ToolTestResult {
        let start = std::time::Instant::now();
        let output = self
            .command_runner
            .validate_model(Path::new(&self.model_path));
        self.build_result_from_output("validate", output, start)
    }

    /// Execute apr bench
    #[must_use]
    pub fn execute_bench(&self) -> ToolTestResult {
        let start = std::time::Instant::now();
        let output = self.command_runner.bench_model(Path::new(&self.model_path));
        self.build_result_from_output("bench", output, start)
    }

    /// Execute apr check
    #[must_use]
    pub fn execute_check(&self) -> ToolTestResult {
        let start = std::time::Instant::now();
        let output = self.command_runner.check_model(Path::new(&self.model_path));
        self.build_result_from_output("check", output, start)
    }

    /// Execute apr trace with specified level
    #[must_use]
    pub fn execute_trace(&self, level: &str) -> ToolTestResult {
        let start = std::time::Instant::now();
        let output = self.command_runner.run_inference(
            Path::new(&self.model_path),
            "What is 2+2?",
            8,
            self.no_gpu,
            &["--trace", "--trace-level", level],
        );
        self.build_result_from_output(&format!("trace-{level}"), output, start)
    }

    /// Execute apr profile (standalone command)
    #[must_use]
    pub fn execute_profile(&self) -> ToolTestResult {
        let start = std::time::Instant::now();
        let output = self
            .command_runner
            .profile_model(Path::new(&self.model_path), 1, 2);
        self.build_result_from_output("profile", output, start)
    }

    /// Execute apr profile in CI mode with assertions (F-PROFILE-006)
    ///
    /// Tests the CI mode features:
    /// - `--ci` flag for CI mode with assertion checks
    /// - `--assert-throughput` minimum tok/s assertion
    /// - `--warmup` and `--measure` pass counts
    ///
    /// Returns pass if CI mode runs and reports metrics correctly.
    #[must_use]
    pub fn execute_profile_ci(&self) -> ToolTestResult {
        let start = std::time::Instant::now();

        // Run apr profile in CI mode with lenient assertions
        // Use very low throughput threshold (1 tok/s) to ensure it passes
        let output = self.command_runner.profile_ci(
            Path::new(&self.model_path),
            Some(1.0), // Very lenient: 1 tok/s minimum
            None,      // No p99 assertion
            1,         // warmup
            2,         // measure
        );

        let duration_ms = start.elapsed().as_millis() as u64;

        // Check if CI features are available
        if output.stderr.contains("unexpected argument")
            || output.stderr.contains("unrecognized")
            || output.stderr.contains("--ci")
        {
            return ToolTestResult {
                tool: "profile-ci".to_string(),
                passed: false,
                exit_code: -2,
                stdout: output.stdout,
                stderr: "Feature not available: apr profile does not support --ci mode".to_string(),
                duration_ms,
                gate_id: "F-PROFILE-006".to_string(),
            };
        }

        // Verify JSON output contains expected CI fields
        let has_passed_field = output.stdout.contains("\"passed\"");
        let has_metrics = output.stdout.contains("throughput") || output.stdout.contains("tok_s");

        let passed = output.exit_code == 0 && (has_passed_field || has_metrics);

        ToolTestResult {
            tool: "profile-ci".to_string(),
            passed,
            exit_code: output.exit_code,
            stdout: output.stdout,
            stderr: output.stderr,
            duration_ms,
            gate_id: "F-PROFILE-006".to_string(),
        }
    }

    /// Execute apr profile CI with assertion failure test (F-PROFILE-007)
    ///
    /// Tests that CI mode correctly fails when assertions are not met.
    /// Uses an impossibly high throughput assertion to guarantee failure.
    #[must_use]
    pub fn execute_profile_ci_assertion_failure(&self) -> ToolTestResult {
        let start = std::time::Instant::now();

        // Run with impossible throughput assertion (1 million tok/s)
        let output = self.command_runner.profile_ci(
            Path::new(&self.model_path),
            Some(1_000_000.0), // Impossible: 1M tok/s
            None,
            1, // warmup
            1, // measure
        );

        let duration_ms = start.elapsed().as_millis() as u64;

        // Check if CI features are available
        if output.stderr.contains("unexpected argument") || output.stderr.contains("unrecognized") {
            return ToolTestResult {
                tool: "profile-ci-assertion".to_string(),
                passed: false,
                exit_code: -2,
                stdout: output.stdout,
                stderr: "Feature not available: apr profile does not support --ci mode".to_string(),
                duration_ms,
                gate_id: "F-PROFILE-007".to_string(),
            };
        }

        // CI mode should EXIT 1 when assertion fails
        // The test PASSES if apr correctly returns non-zero exit code
        // or reports failure in output (fallback for older versions)
        let assertion_failed_correctly = output.exit_code == 1
            || output.stdout.contains("\"passed\":false")
            || output.stdout.contains("\"passed\": false")
            || output.stdout.contains("ASSERTIONS FAILED");

        ToolTestResult {
            tool: "profile-ci-assertion".to_string(),
            passed: assertion_failed_correctly,
            exit_code: output.exit_code,
            stdout: output.stdout,
            stderr: output.stderr,
            duration_ms,
            gate_id: "F-PROFILE-007".to_string(),
        }
    }

    /// Execute apr profile with p99 latency assertion (F-PROFILE-008)
    #[must_use]
    pub fn execute_profile_ci_p99(&self) -> ToolTestResult {
        let start = std::time::Instant::now();

        // Run with lenient p99 assertion (10 seconds max)
        let output = self.command_runner.profile_ci(
            Path::new(&self.model_path),
            None,           // No throughput assertion
            Some(10_000.0), // 10 seconds max p99
            1,              // warmup
            2,              // measure
        );

        let duration_ms = start.elapsed().as_millis() as u64;

        // Check if p99 assertion feature is available
        if output.stderr.contains("unexpected argument") || output.stderr.contains("--assert-p99") {
            return ToolTestResult {
                tool: "profile-ci-p99".to_string(),
                passed: false,
                exit_code: -2,
                stdout: output.stdout,
                stderr: "Feature not available: apr profile does not support --assert-p99"
                    .to_string(),
                duration_ms,
                gate_id: "F-PROFILE-008".to_string(),
            };
        }

        // Verify p99 metric is in output
        let has_p99 = output.stdout.contains("p99") || output.stdout.contains("latency");
        let passed = output.exit_code == 0 && has_p99;

        ToolTestResult {
            tool: "profile-ci-p99".to_string(),
            passed,
            exit_code: output.exit_code,
            stdout: output.stdout,
            stderr: output.stderr,
            duration_ms,
            gate_id: "F-PROFILE-008".to_string(),
        }
    }

    /// Execute apr profile with flamegraph output (F-PROFILE-002)
    ///
    /// Tests that profile can generate valid SVG flamegraph output.
    /// This feature may not be available in all apr versions.
    #[must_use]
    pub fn execute_profile_flamegraph(&self, output_path: &std::path::Path) -> ToolTestResult {
        let start = std::time::Instant::now();

        let svg_path = output_path.join("profile_flamegraph.svg");
        let output = self.command_runner.profile_with_flamegraph(
            Path::new(&self.model_path),
            &svg_path,
            self.no_gpu,
        );
        let duration_ms = start.elapsed().as_millis() as u64;

        // If apr doesn't support --profile-output, it will error
        if output.stderr.contains("unexpected argument") || output.stderr.contains("unrecognized") {
            return ToolTestResult {
                tool: "profile-flamegraph".to_string(),
                passed: false,
                exit_code: -2,
                stdout: output.stdout,
                stderr: "Feature not available: apr does not support --profile-output".to_string(),
                duration_ms,
                gate_id: "F-PROFILE-002".to_string(),
            };
        }

        // Check if flamegraph was generated
        let flamegraph_exists = svg_path.exists();
        let flamegraph_valid = if flamegraph_exists {
            std::fs::read_to_string(&svg_path)
                .map(|content| content.contains("<svg") && content.contains("</svg>"))
                .unwrap_or(false)
        } else {
            false
        };

        ToolTestResult {
            tool: "profile-flamegraph".to_string(),
            passed: flamegraph_valid,
            exit_code: i32::from(!flamegraph_valid),
            stdout: format!("Flamegraph exists: {flamegraph_exists}, valid: {flamegraph_valid}"),
            stderr: output.stderr,
            duration_ms,
            gate_id: "F-PROFILE-002".to_string(),
        }
    }

    /// Execute apr profile with focus filtering (F-PROFILE-003)
    ///
    /// Tests that profile --focus option works to limit scope.
    /// This feature may not be available in all apr versions.
    #[must_use]
    pub fn execute_profile_focus(&self, focus: &str) -> ToolTestResult {
        let start = std::time::Instant::now();

        let output =
            self.command_runner
                .profile_with_focus(Path::new(&self.model_path), focus, self.no_gpu);
        let duration_ms = start.elapsed().as_millis() as u64;

        // If apr doesn't support --focus, it will error
        if output.stderr.contains("unexpected argument") || output.stderr.contains("unrecognized") {
            return ToolTestResult {
                tool: "profile-focus".to_string(),
                passed: false,
                exit_code: -2,
                stdout: output.stdout,
                stderr: format!("Feature not available: apr does not support --focus {focus}"),
                duration_ms,
                gate_id: "F-PROFILE-003".to_string(),
            };
        }

        let passed = output.success;

        ToolTestResult {
            tool: "profile-focus".to_string(),
            passed,
            exit_code: output.exit_code,
            stdout: output.stdout,
            stderr: output.stderr,
            duration_ms,
            gate_id: "F-PROFILE-003".to_string(),
        }
    }

    /// Execute backend equivalence test (F-CONV-BE-001)
    ///
    /// Compares CPU vs GPU output to verify they produce equivalent results.
    /// Skips if GPU is not available.
    #[must_use]
    pub fn execute_backend_equivalence(&self) -> ToolTestResult {
        use std::process::Command;
        let start = std::time::Instant::now();

        let prompt = "What is 2+2?";

        // Run with CPU (--no-gpu)
        let cpu_output = Command::new("apr")
            .arg("run")
            .arg(&self.model_path)
            .arg("-p")
            .arg(prompt)
            .arg("--max-tokens")
            .arg("8")
            .arg("--no-gpu")
            .output();

        let cpu_result = match cpu_output {
            Ok(out) => {
                if out.status.success() {
                    Some(String::from_utf8_lossy(&out.stdout).to_string())
                } else {
                    None
                }
            }
            Err(_) => None,
        };

        // Run with GPU
        let gpu_output = Command::new("apr")
            .arg("run")
            .arg(&self.model_path)
            .arg("-p")
            .arg(prompt)
            .arg("--max-tokens")
            .arg("8")
            .arg("--gpu")
            .output();

        let gpu_result = match gpu_output {
            Ok(out) => {
                let stderr = String::from_utf8_lossy(&out.stderr);
                // Check if GPU is not available
                if stderr.contains("No GPU") || stderr.contains("CUDA") || !out.status.success() {
                    None // GPU not available
                } else {
                    Some(String::from_utf8_lossy(&out.stdout).to_string())
                }
            }
            Err(_) => None,
        };

        let duration_ms = start.elapsed().as_millis() as u64;

        match (cpu_result, gpu_result) {
            (Some(cpu), Some(gpu)) => {
                // Compare outputs - they should be similar (not necessarily identical due to FP)
                let equivalent = cpu.trim() == gpu.trim();
                ToolTestResult {
                    tool: "backend-equivalence".to_string(),
                    passed: equivalent,
                    exit_code: i32::from(!equivalent),
                    stdout: format!("CPU: {}\nGPU: {}", cpu.trim(), gpu.trim()),
                    stderr: if equivalent {
                        String::new()
                    } else {
                        "CPU and GPU outputs differ".to_string()
                    },
                    duration_ms,
                    gate_id: "F-CONV-BE-001".to_string(),
                }
            }
            (Some(_), None) => ToolTestResult {
                tool: "backend-equivalence".to_string(),
                passed: false,
                exit_code: -2,
                stdout: String::new(),
                stderr: "GPU not available - skipping backend equivalence test".to_string(),
                duration_ms,
                gate_id: "F-CONV-BE-001".to_string(),
            },
            _ => ToolTestResult {
                tool: "backend-equivalence".to_string(),
                passed: false,
                exit_code: -1,
                stdout: String::new(),
                stderr: "Failed to run inference on both backends".to_string(),
                duration_ms,
                gate_id: "F-CONV-BE-001".to_string(),
            },
        }
    }

    /// Execute apr serve lifecycle test (F-INTEG-003)
    ///
    /// Tests the full serve lifecycle:
    /// 1. Start server
    /// 2. Wait for health endpoint
    /// 3. Make inference request
    /// 4. Shutdown cleanly
    #[must_use]
    pub fn execute_serve_lifecycle(&self) -> ToolTestResult {
        use std::io::{BufRead, BufReader};
        use std::process::{Command, Stdio};
        use std::time::Duration;

        let start = std::time::Instant::now();
        let port = 18080; // Use high port to avoid conflicts

        // Start server
        let mut server_cmd = Command::new("apr");
        server_cmd
            .arg("serve")
            .arg(&self.model_path)
            .arg("--port")
            .arg(port.to_string())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        if self.no_gpu {
            server_cmd.arg("--no-gpu");
        }

        let mut server = match server_cmd.spawn() {
            Ok(child) => child,
            Err(e) => {
                return ToolTestResult {
                    tool: "serve-lifecycle".to_string(),
                    passed: false,
                    exit_code: -1,
                    stdout: String::new(),
                    stderr: format!("Failed to start server: {e}"),
                    duration_ms: start.elapsed().as_millis() as u64,
                    gate_id: "F-INTEG-003".to_string(),
                };
            }
        };

        // Wait for server to be ready (check stderr for "Listening on")
        let stderr = server.stderr.take();
        let ready = stderr.map_or_else(
            || {
                // Wait a fixed time if can't read stderr
                std::thread::sleep(Duration::from_secs(3));
                true
            },
            |stderr| {
                let reader = BufReader::new(stderr);
                let mut ready = false;
                for line in reader.lines().take(20).flatten() {
                    if line.contains("Listening") || line.contains("listening") {
                        ready = true;
                        break;
                    }
                }
                ready
            },
        );

        if !ready {
            // Give it more time
            std::thread::sleep(Duration::from_secs(2));
        }

        // Test health endpoint
        let health_result = Command::new("curl")
            .arg("-sf")
            .arg(format!("http://localhost:{port}/health"))
            .arg("--connect-timeout")
            .arg("5")
            .output();

        let health_ok = health_result.map(|o| o.status.success()).unwrap_or(false);

        // Test inference endpoint
        let inference_result = Command::new("curl")
            .arg("-sf")
            .arg("-X")
            .arg("POST")
            .arg(format!("http://localhost:{port}/v1/chat/completions"))
            .arg("-H")
            .arg("Content-Type: application/json")
            .arg("-d")
            .arg(r#"{"messages":[{"role":"user","content":"Hi"}],"max_tokens":5}"#)
            .arg("--connect-timeout")
            .arg("10")
            .output();

        let inference_ok = inference_result
            .map(|o| o.status.success())
            .unwrap_or(false);

        // Shutdown server
        let _ = server.kill();
        let _ = server.wait();

        let duration_ms = start.elapsed().as_millis() as u64;

        let passed = health_ok && inference_ok;
        let stdout = format!(
            "Health check: {}\nInference: {}",
            if health_ok { "OK" } else { "FAILED" },
            if inference_ok { "OK" } else { "FAILED" }
        );
        let stderr = if passed {
            String::new()
        } else {
            format!("Serve lifecycle incomplete: health={health_ok}, inference={inference_ok}")
        };

        ToolTestResult {
            tool: "serve-lifecycle".to_string(),
            passed,
            exit_code: i32::from(!passed),
            stdout,
            stderr,
            duration_ms,
            gate_id: "F-INTEG-003".to_string(),
        }
    }

    /// Execute all tool tests
    #[must_use]
    pub fn execute_all(&self) -> Vec<ToolTestResult> {
        self.execute_all_with_serve(false)
    }

    /// Execute all tool tests, optionally including serve lifecycle
    #[must_use]
    pub fn execute_all_with_serve(&self, include_serve: bool) -> Vec<ToolTestResult> {
        let mut results = vec![
            // Core tool tests
            self.execute_inspect(),
            self.execute_inspect_verified(), // T-GH192-01: metadata verification
            self.execute_validate(),
            self.execute_check(),
            self.execute_bench(),
        ];

        // Trace level tests
        for level in &["none", "basic", "layer", "payload"] {
            results.push(self.execute_trace(level));
        }

        // Profile tests (F-PROFILE-001 basic, F-PROFILE-006/007/008 CI mode)
        results.push(self.execute_profile());
        results.push(self.execute_profile_ci());
        results.push(self.execute_profile_ci_assertion_failure());
        results.push(self.execute_profile_ci_p99());

        // Serve lifecycle test (F-INTEG-003)
        if include_serve {
            results.push(self.execute_serve_lifecycle());
        }

        results
    }

    fn build_result_from_output(
        &self,
        tool: &str,
        output: crate::command::CommandOutput,
        start: std::time::Instant,
    ) -> ToolTestResult {
        let duration_ms = start.elapsed().as_millis() as u64;

        ToolTestResult {
            tool: tool.to_string(),
            passed: output.success,
            exit_code: output.exit_code,
            stdout: output.stdout,
            stderr: output.stderr,
            duration_ms,
            gate_id: format!("F-{}-001", tool.to_uppercase().replace('-', "_")),
        }
    }
}

/// Result of a tool test
#[derive(Debug, Clone)]
pub struct ToolTestResult {
    /// Tool name
    pub tool: String,
    /// Whether test passed
    pub passed: bool,
    /// Exit code
    pub exit_code: i32,
    /// Stdout output
    pub stdout: String,
    /// Stderr output
    pub stderr: String,
    /// Duration in ms
    pub duration_ms: u64,
    /// Gate ID for this test
    pub gate_id: String,
}

impl ToolTestResult {
    /// Convert to Evidence
    #[must_use]
    pub fn to_evidence(&self, model_id: &ModelId) -> Evidence {
        let scenario = QaScenario::new(
            model_id.clone(),
            Modality::Run,
            Backend::Cpu,
            Format::Gguf,
            format!("apr {} test", self.tool),
            0,
        );

        if self.passed {
            Evidence::corroborated(&self.gate_id, scenario, &self.stdout, self.duration_ms)
        } else {
            Evidence::falsified(
                &self.gate_id,
                scenario,
                format!("Exit code: {}, stderr: {}", self.exit_code, self.stderr),
                &self.stdout,
                self.duration_ms,
            )
        }
    }
}

/// Result of playbook execution
#[derive(Debug, Clone)]
pub struct ExecutionResult {
    /// Playbook name
    pub playbook_name: String,
    /// Total scenarios
    pub total_scenarios: usize,
    /// Passed scenarios
    pub passed: usize,
    /// Failed scenarios
    pub failed: usize,
    /// Skipped scenarios
    pub skipped: usize,
    /// Total duration in milliseconds
    pub duration_ms: u64,
    /// Gateway failure (if any)
    pub gateway_failed: Option<String>,
    /// Collected evidence
    pub evidence: EvidenceCollector,
}

impl ExecutionResult {
    /// Check if execution was successful
    #[must_use]
    pub fn is_success(&self) -> bool {
        self.gateway_failed.is_none() && self.failed == 0
    }

    /// Get pass rate as percentage
    #[must_use]
    pub fn pass_rate(&self) -> f64 {
        if self.total_scenarios == 0 {
            return 0.0;
        }
        (self.passed as f64 / self.total_scenarios as f64) * 100.0
    }
}

