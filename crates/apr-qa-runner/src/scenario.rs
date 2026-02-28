
// G0 gateway checks — see executor_gates.rs
include!("executor_gates.rs");

impl Executor {
    /// Execute a single scenario
    fn execute_scenario(&self, scenario: &QaScenario) -> Evidence {
        let start = Instant::now();

        let (output, stderr, exit_code, tps, skipped) = self.subprocess_execution(scenario);

        if skipped {
            let gate_id = format!("F-{}-001", scenario.mqs_category());
            return Evidence::skipped(
                &gate_id,
                scenario.clone(),
                format!("Format {:?} not available for model file", scenario.format),
            );
        }

        let duration = start.elapsed().as_millis() as u64;

        // Check for crash (negative exit code = signal)
        if exit_code < 0 {
            return Evidence::crashed(
                "G3-STABLE",
                scenario.clone(),
                stderr.as_deref().unwrap_or("Process crashed"),
                exit_code,
                duration,
            );
        }

        // Check for command failure (non-zero exit code)
        if exit_code > 0 {
            let error_msg = stderr
                .as_deref()
                .unwrap_or("Command failed with non-zero exit code");
            let mut evidence = Evidence::falsified(
                "G2-BASIC",
                scenario.clone(),
                format!("Command failed (exit {exit_code}): {error_msg}"),
                &output,
                duration,
            );
            evidence.exit_code = Some(exit_code);
            evidence.stderr = stderr;
            return evidence;
        }

        // Evaluate the output
        let oracle_result = scenario.evaluate(&output);

        let gate_id = format!("F-{}-001", scenario.mqs_category());

        match oracle_result {
            apr_qa_gen::OracleResult::Corroborated { evidence: _reason } => {
                let mut evidence =
                    Evidence::corroborated(&gate_id, scenario.clone(), &output, duration);
                evidence.metrics = PerformanceMetrics {
                    duration_ms: duration,
                    tokens_per_second: tps,
                    total_tokens: Some(32),
                    time_to_first_token_ms: None,
                    memory_peak_mb: None,
                };
                if let Some(ref err) = stderr {
                    evidence.stderr = Some(err.clone());
                }
                evidence
            }
            apr_qa_gen::OracleResult::Falsified {
                reason,
                evidence: _,
            } => {
                let mut evidence =
                    Evidence::falsified(&gate_id, scenario.clone(), reason, &output, duration);
                if let Some(ref err) = stderr {
                    evidence.stderr = Some(err.clone());
                }
                evidence
            }
        }
    }

    /// Execute via subprocess (real apr commands)
    /// On failure, re-runs with --trace for full diagnostics
    ///
    /// Returns `(stdout, stderr, exit_code, tps, skipped)`.
    /// When `skipped` is `true` the scenario format is unavailable for the
    /// model file and the caller should emit `Evidence::skipped`.
    fn subprocess_execution(
        &self,
        scenario: &QaScenario,
    ) -> (String, Option<String>, i32, Option<f64>, bool) {
        let Some(model_path) = self.resolve_model_path(scenario) else {
            return (String::new(), None, 0, None, true);
        };

        // Fix 201: Use per-scenario backend, not global no_gpu flag
        let no_gpu = scenario.backend == Backend::Cpu;

        // Fix 200: Dispatch by modality instead of always using `apr run`
        let output = match scenario.modality {
            Modality::Run => self.command_runner.run_inference(
                Path::new(&model_path),
                &scenario.prompt,
                32,
                no_gpu,
                &["--benchmark", "--json"],
            ),
            Modality::Chat => self.command_runner.run_chat(
                Path::new(&model_path),
                &scenario.prompt,
                no_gpu,
                &["--json"],
            ),
            Modality::Serve => {
                return self.run_serve_scenario(&model_path, scenario, no_gpu);
            }
        };

        // Try to parse tok/s from JSON output
        let tps = Self::parse_tps_from_output(&output.stdout);

        // Extract the actual generated text (not the JSON benchmark data)
        let generated_text = Self::extract_generated_text(&output.stdout);

        // On failure, re-run with tracing for full diagnostics
        let (final_stderr, final_exit_code) = if output.success {
            (
                if output.stderr.is_empty() {
                    None
                } else {
                    Some(output.stderr)
                },
                output.exit_code,
            )
        } else {
            // Trace retry uses the same modality as the original command
            let trace_output = match scenario.modality {
                Modality::Run => self.command_runner.run_inference(
                    Path::new(&model_path),
                    &scenario.prompt,
                    32,
                    no_gpu,
                    &["--trace"],
                ),
                Modality::Chat => self.command_runner.run_chat(
                    Path::new(&model_path),
                    &scenario.prompt,
                    no_gpu,
                    &["--trace"],
                ),
                Modality::Serve => {
                    // For serve failures, re-run as `apr run --trace` since
                    // serve lifecycle is complex and trace needs a single shot
                    self.command_runner.run_inference(
                        Path::new(&model_path),
                        &scenario.prompt,
                        32,
                        no_gpu,
                        &["--trace"],
                    )
                }
            };
            let mut full_trace = output.stderr.clone();
            if !trace_output.stderr.is_empty() {
                full_trace.push_str("\n--- TRACE OUTPUT ---\n");
                full_trace.push_str(&trace_output.stderr);
            }
            if !trace_output.stdout.is_empty() {
                full_trace.push_str("\n--- TRACE STDOUT ---\n");
                full_trace.push_str(&trace_output.stdout);
            }
            (Some(full_trace), output.exit_code)
        };

        (generated_text, final_stderr, final_exit_code, tps, false)
    }

    /// Execute a serve scenario: spawn server, send request, parse response, kill server.
    /// Bug 200: Serve modality needs lifecycle management.
    fn run_serve_scenario(
        &self,
        model_path: &str,
        scenario: &QaScenario,
        no_gpu: bool,
    ) -> (String, Option<String>, i32, Option<f64>, bool) {
        // Use a deterministic port based on scenario to avoid collisions
        let port = 18_080 + (scenario.seed % 1000) as u16;

        // Spawn server in background
        let spawn_output = self
            .command_runner
            .spawn_serve(Path::new(model_path), port, no_gpu);
        if !spawn_output.success {
            return (
                String::new(),
                Some(format!("Failed to spawn serve: {}", spawn_output.stderr)),
                spawn_output.exit_code,
                None,
                false,
            );
        }

        let pid_str = spawn_output.stdout.trim().to_string();

        // Wait for server to be ready — poll /health endpoint via GET
        // Large models (14B+) can take 3-5 min to load on CPU.
        // Use configured timeout (from playbook), minimum 120s.
        let serve_timeout_secs = std::cmp::max(self.config.default_timeout_ms / 1000, 120);
        let poll_iterations = serve_timeout_secs / 2;
        let health_url = format!("http://localhost:{port}/health");
        let mut server_ready = false;
        let server_pid: Option<u32> = pid_str.parse().ok();
        for _ in 0..poll_iterations {
            std::thread::sleep(std::time::Duration::from_secs(2));
            // Check if server process is still alive (fail fast if crashed)
            if let Some(pid) = server_pid {
                let alive = std::path::Path::new(&format!("/proc/{pid}")).exists();
                if !alive {
                    break;
                }
            }
            if let Ok(output) = std::process::Command::new("curl")
                .args(["-s", "-m", "2", &health_url])
                .output()
            {
                let body = String::from_utf8_lossy(&output.stdout);
                if output.status.success() && body.contains("healthy") {
                    server_ready = true;
                    break;
                }
            }
        }
        if !server_ready {
            // Kill server and report failure
            if pid_str.parse::<u32>().is_ok() {
                let _ = std::process::Command::new("kill").arg(&pid_str).output();
            }
            return (
                String::new(),
                Some(format!(
                    "Server failed to become ready within {serve_timeout_secs}s"
                )),
                1,
                None,
                false,
            );
        }

        // Send completion request to /generate endpoint
        let body = format!(
            r#"{{"prompt":"{}","max_tokens":32}}"#,
            scenario.prompt.replace('"', "\\\""),
        );
        let url = format!("http://localhost:{port}/generate");
        let output = self.command_runner.http_post(&url, &body);

        // Kill the server process
        if pid_str.parse::<u32>().is_ok() {
            let _ = std::process::Command::new("kill").arg(&pid_str).output();
        }

        let tps = Self::parse_tps_from_output(&output.stdout);
        let generated_text = Self::extract_generated_text(&output.stdout);

        let (final_stderr, final_exit_code) = if output.success {
            (
                if output.stderr.is_empty() {
                    None
                } else {
                    Some(output.stderr)
                },
                output.exit_code,
            )
        } else {
            (Some(output.stderr), output.exit_code)
        };

        (generated_text, final_stderr, final_exit_code, tps, false)
    }
}

// Model resolution + workspace — see executor_resolution.rs
include!("executor_resolution.rs");

impl Default for Executor {
    fn default() -> Self {
        Self::new()
    }
}

// ToolExecutor — see executor_tools.rs
include!("executor_tools.rs");

#[cfg(test)]
#[path = "executor_tests_a.rs"]
mod tests_a;

#[cfg(test)]
#[path = "executor_tests_b.rs"]
mod tests_b;

#[cfg(test)]
#[path = "executor_tests_c.rs"]
mod tests_c;

#[cfg(test)]
#[path = "executor_tests_d.rs"]
mod tests_d;

#[cfg(test)]
#[path = "executor_tests_e.rs"]
mod tests_e;

#[cfg(test)]
#[path = "executor_tests_f.rs"]
mod tests_f;
