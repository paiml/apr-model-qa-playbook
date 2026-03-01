
/// Serve battery: spawn server once, run 8 endpoint checks, kill once.
///
/// Replaces the 1-request-per-server-lifecycle pattern. The primary
/// check (Generate) uses the existing gate ID for backward compatibility.
/// Additional checks emit new Evidence with distinct gate suffixes.
impl Executor {
    /// Run a battery of 8 serve endpoint checks against a single server lifecycle.
    ///
    /// Returns a `Vec<Evidence>` — one per check executed. If the primary
    /// Generate check fails, checks 2-8 are skipped (server is broken).
    #[must_use]
    pub fn run_serve_battery(
        &self,
        model_path: &str,
        scenario: &QaScenario,
        no_gpu: bool,
    ) -> Vec<Evidence> {
        let start = Instant::now();
        let mut results = Vec::with_capacity(8);

        // Use a deterministic port based on scenario to avoid collisions
        let port = 18_080 + (scenario.seed % 1000) as u16;

        // Spawn server in background
        let spawn_output = self
            .command_runner
            .spawn_serve(Path::new(model_path), port, no_gpu);
        if !spawn_output.success {
            let gate_id = format!("F-{}-001", scenario.mqs_category());
            results.push(Evidence::falsified(
                &gate_id,
                scenario.clone(),
                format!("Failed to spawn serve: {}", spawn_output.stderr),
                &spawn_output.stderr,
                start.elapsed().as_millis() as u64,
            ));
            return results;
        }

        let pid_str = spawn_output.stdout.trim().to_string();

        // Wait for server to be ready — poll /health endpoint via http_get.
        // Large models (14B+) can take 3-5 min to load on CPU.
        // Use configured timeout (from playbook), minimum 120s.
        let serve_timeout_secs = std::cmp::max(self.config.default_timeout_ms / 1000, 120);
        let poll_iterations = serve_timeout_secs / 2;
        let health_url = format!("http://localhost:{port}/health");
        let mut server_ready = false;
        let server_pid: Option<u32> = pid_str.parse().ok();
        for _ in 0..poll_iterations {
            std::thread::sleep(std::time::Duration::from_secs(2));
            // Check health first — if server responds, we're good
            let health_output = self.command_runner.http_get(&health_url);
            if health_output.success && health_output.stdout.contains("healthy") {
                server_ready = true;
                break;
            }
            // Then check if server process is still alive (fail fast if crashed)
            if let Some(pid) = server_pid {
                let alive = std::path::Path::new(&format!("/proc/{pid}")).exists();
                if !alive {
                    break;
                }
            }
        }
        if !server_ready {
            if pid_str.parse::<u32>().is_ok() {
                let _ = std::process::Command::new("kill").arg(&pid_str).output();
            }
            let gate_id = format!("F-{}-001", scenario.mqs_category());
            results.push(Evidence::falsified(
                &gate_id,
                scenario.clone(),
                format!("Server failed to become ready within {serve_timeout_secs}s"),
                "",
                start.elapsed().as_millis() as u64,
            ));
            return results;
        }

        // Check 1: Generate (primary — backward compatible gate ID)
        let primary = self.check_serve_generate(port, scenario, &start);
        let primary_passed = primary.outcome.is_pass();
        results.push(primary);

        // If primary failed, skip remaining checks — server is broken
        if primary_passed {
            results.push(self.check_serve_v1_completions(port, scenario, &start));
            results.push(self.check_serve_v1_chat(port, scenario, &start));
            results.push(self.check_serve_streaming(port, scenario, &start));
            results.push(self.check_serve_stop_sequence(port, scenario, &start));
            results.push(self.check_serve_malformed(port, scenario, &start));
            results.push(self.check_serve_info(port, scenario, &start));
            results.push(self.check_serve_metrics(port, scenario, &start));
        }

        // Kill the server process
        if pid_str.parse::<u32>().is_ok() {
            let _ = std::process::Command::new("kill").arg(&pid_str).output();
        }

        results
    }

    /// Check 1: POST /generate — primary serve inference (backward compat)
    fn check_serve_generate(
        &self,
        port: u16,
        scenario: &QaScenario,
        start: &Instant,
    ) -> Evidence {
        let gate_id = format!("F-{}-001", scenario.mqs_category());
        let body = format!(
            r#"{{"prompt":"{}","max_tokens":32}}"#,
            scenario.prompt.replace('"', "\\\""),
        );
        let url = format!("http://localhost:{port}/generate");
        let output = self.command_runner.http_post(&url, &body);
        let duration = start.elapsed().as_millis() as u64;

        if output.success {
            let generated = Self::extract_generated_text(&output.stdout);
            let oracle_result = scenario.evaluate(&generated);
            match oracle_result {
                apr_qa_gen::OracleResult::Corroborated { .. } => {
                    let mut ev = Evidence::corroborated(&gate_id, scenario.clone(), &generated, duration);
                    ev.metrics.tokens_per_second = Self::parse_tps_from_output(&output.stdout);
                    ev
                }
                apr_qa_gen::OracleResult::Falsified { reason, .. } => {
                    Evidence::falsified(&gate_id, scenario.clone(), reason, &generated, duration)
                }
            }
        } else {
            Evidence::falsified(
                &gate_id,
                scenario.clone(),
                format!("HTTP POST /generate failed: {}", output.stderr),
                &output.stdout,
                duration,
            )
        }
    }

    /// Check 2: POST /v1/completions — OpenAI-compatible text completion
    fn check_serve_v1_completions(
        &self,
        port: u16,
        scenario: &QaScenario,
        start: &Instant,
    ) -> Evidence {
        let gate_id = format!("F-{}-COMP-001", scenario.mqs_category());
        let body = format!(
            r#"{{"prompt":"{}","max_tokens":32,"temperature":0.0}}"#,
            scenario.prompt.replace('"', "\\\""),
        );
        let url = format!("http://localhost:{port}/v1/completions");
        let output = self.command_runner.http_post(&url, &body);
        let duration = start.elapsed().as_millis() as u64;

        if output.success {
            Evidence::corroborated(&gate_id, scenario.clone(), &output.stdout, duration)
        } else {
            Evidence::falsified(
                &gate_id,
                scenario.clone(),
                format!("POST /v1/completions failed: {}", output.stderr),
                &output.stdout,
                duration,
            )
        }
    }

    /// Check 3: POST /v1/chat/completions — primary production API (OpenAI format)
    fn check_serve_v1_chat(
        &self,
        port: u16,
        scenario: &QaScenario,
        start: &Instant,
    ) -> Evidence {
        let gate_id = format!("F-{}-CHAT-001", scenario.mqs_category());
        let body = format!(
            r#"{{"model":"apr","messages":[{{"role":"user","content":"{}"}}],"max_tokens":32}}"#,
            scenario.prompt.replace('"', "\\\""),
        );
        let url = format!("http://localhost:{port}/v1/chat/completions");
        let output = self.command_runner.http_post(&url, &body);
        let duration = start.elapsed().as_millis() as u64;

        if output.success {
            Evidence::corroborated(&gate_id, scenario.clone(), &output.stdout, duration)
        } else {
            Evidence::falsified(
                &gate_id,
                scenario.clone(),
                format!("POST /v1/chat/completions failed: {}", output.stderr),
                &output.stdout,
                duration,
            )
        }
    }

    /// Check 4: POST /generate with stream=true — verify SSE format
    fn check_serve_streaming(
        &self,
        port: u16,
        scenario: &QaScenario,
        start: &Instant,
    ) -> Evidence {
        let gate_id = format!("F-{}-STREAM-001", scenario.mqs_category());
        let body = format!(
            r#"{{"prompt":"{}","max_tokens":16,"stream":true}}"#,
            scenario.prompt.replace('"', "\\\""),
        );
        let url = format!("http://localhost:{port}/generate");
        let output = self.command_runner.http_post(&url, &body);
        let duration = start.elapsed().as_millis() as u64;

        if !output.success {
            return Evidence::falsified(
                &gate_id,
                scenario.clone(),
                format!("Streaming request failed: {}", output.stderr),
                &output.stdout,
                duration,
            );
        }

        if Self::verify_sse_response(&output.stdout) {
            Evidence::corroborated(&gate_id, scenario.clone(), &output.stdout, duration)
        } else {
            Evidence::falsified(
                &gate_id,
                scenario.clone(),
                "SSE response format invalid: expected 'data: ' prefixed lines ending with 'data: [DONE]'",
                &output.stdout,
                duration,
            )
        }
    }

    /// Check 5: POST /generate with stop sequence
    fn check_serve_stop_sequence(
        &self,
        port: u16,
        scenario: &QaScenario,
        start: &Instant,
    ) -> Evidence {
        let gate_id = format!("F-{}-STOP-001", scenario.mqs_category());
        let body = r#"{"prompt":"Count: 1, 2, 3, 4, 5","max_tokens":32,"stop":["5"]}"#;
        let url = format!("http://localhost:{port}/generate");
        let output = self.command_runner.http_post(&url, body);
        let duration = start.elapsed().as_millis() as u64;

        if !output.success {
            return Evidence::falsified(
                &gate_id,
                scenario.clone(),
                format!("Stop sequence request failed: {}", output.stderr),
                &output.stdout,
                duration,
            );
        }

        let generated = Self::extract_generated_text(&output.stdout);
        if generated.contains('5') {
            Evidence::falsified(
                &gate_id,
                scenario.clone(),
                "Stop sequence not honored: output contains '5'",
                &generated,
                duration,
            )
        } else {
            Evidence::corroborated(&gate_id, scenario.clone(), &generated, duration)
        }
    }

    /// Check 6: POST /generate with malformed JSON — error resilience
    fn check_serve_malformed(
        &self,
        port: u16,
        scenario: &QaScenario,
        start: &Instant,
    ) -> Evidence {
        let gate_id = format!("F-{}-ERR-001", scenario.mqs_category());
        let bad_body = r#"{"not_a_valid_field": true}"#;
        let url = format!("http://localhost:{port}/generate");
        let output = self.command_runner.http_post(&url, bad_body);
        let duration = start.elapsed().as_millis() as u64;

        // We expect a non-success response (server rejects malformed input)
        // but the server must still be healthy afterward
        let health_url = format!("http://localhost:{port}/health");
        let health = self.command_runner.http_get(&health_url);

        if health.success {
            Evidence::corroborated(
                &gate_id,
                scenario.clone(),
                format!("Server survived malformed request (status={})", output.exit_code),
                duration,
            )
        } else {
            Evidence::falsified(
                &gate_id,
                scenario.clone(),
                "Server became unhealthy after malformed request",
                &health.stderr,
                duration,
            )
        }
    }

    /// Check 7: GET / — server info endpoint
    fn check_serve_info(
        &self,
        port: u16,
        scenario: &QaScenario,
        start: &Instant,
    ) -> Evidence {
        let gate_id = format!("F-{}-INFO-001", scenario.mqs_category());
        let url = format!("http://localhost:{port}/");
        let output = self.command_runner.http_get(&url);
        let duration = start.elapsed().as_millis() as u64;

        if output.success && !output.stdout.is_empty() {
            Evidence::corroborated(&gate_id, scenario.clone(), &output.stdout, duration)
        } else {
            Evidence::falsified(
                &gate_id,
                scenario.clone(),
                format!("GET / failed or empty response: {}", output.stderr),
                &output.stdout,
                duration,
            )
        }
    }

    /// Check 8: GET /metrics — Prometheus metrics endpoint
    fn check_serve_metrics(
        &self,
        port: u16,
        scenario: &QaScenario,
        start: &Instant,
    ) -> Evidence {
        let gate_id = format!("F-{}-METRICS-001", scenario.mqs_category());
        let url = format!("http://localhost:{port}/metrics");
        let output = self.command_runner.http_get(&url);
        let duration = start.elapsed().as_millis() as u64;

        if output.success && !output.stdout.is_empty() {
            Evidence::corroborated(&gate_id, scenario.clone(), &output.stdout, duration)
        } else {
            Evidence::falsified(
                &gate_id,
                scenario.clone(),
                format!("GET /metrics failed or empty: {}", output.stderr),
                &output.stdout,
                duration,
            )
        }
    }

    /// Validate SSE (Server-Sent Events) response format.
    ///
    /// Valid SSE: non-empty lines start with "data: ", ends with "data: [DONE]".
    fn verify_sse_response(response: &str) -> bool {
        let data_lines: Vec<&str> = response
            .lines()
            .filter(|l| !l.is_empty())
            .collect();

        if data_lines.is_empty() {
            return false;
        }

        // All non-empty lines must start with "data: "
        let all_data_prefixed = data_lines.iter().all(|l| l.starts_with("data: "));
        if !all_data_prefixed {
            return false;
        }

        // Last line must be "data: [DONE]"
        data_lines
            .last()
            .is_some_and(|l| *l == "data: [DONE]")
    }
}
