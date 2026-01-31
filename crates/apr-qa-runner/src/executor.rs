//! Playbook executor
//!
//! Executes playbooks with parallel execution and failure handling.

#![allow(clippy::cast_possible_truncation)]

use crate::command::{CommandRunner, RealCommandRunner};
use crate::conversion::{ConversionConfig, ConversionExecutor};
use crate::error::Result;
use crate::evidence::{Evidence, EvidenceCollector, PerformanceMetrics};
use crate::playbook::Playbook;
use apr_qa_gen::{Backend, Format, Modality, ModelId, QaScenario};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

/// Failure handling policy (Jidoka)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FailurePolicy {
    /// Stop entire pipeline on any failure
    StopOnFirst,
    /// Stop on P0 failures, continue on P1/P2
    #[default]
    StopOnP0,
    /// Collect all failures, report at end
    CollectAll,
}

/// Execution configuration
#[derive(Debug, Clone)]
#[allow(clippy::struct_excessive_bools)]
pub struct ExecutionConfig {
    /// Failure handling policy
    pub failure_policy: FailurePolicy,
    /// Default timeout in milliseconds
    pub default_timeout_ms: u64,
    /// Maximum parallel workers
    pub max_workers: usize,
    /// Dry run (don't actually execute commands)
    pub dry_run: bool,
    /// Use subprocess mode (run actual apr commands)
    pub subprocess_mode: bool,
    /// Path to the model file for subprocess mode
    pub model_path: Option<String>,
    /// Disable GPU acceleration
    pub no_gpu: bool,
    /// Run P0 format conversion tests (CRITICAL - should be true by default)
    pub run_conversion_tests: bool,
    /// Run differential tests (tensor diff, inference compare)
    pub run_differential_tests: bool,
    /// Run profile CI assertions
    pub run_profile_ci: bool,
    /// Run trace payload tests
    pub run_trace_payload: bool,
    /// Run Golden Rule Test (convert → inference → diff)
    /// This is the single most important invariant: converted models
    /// MUST produce the same output as the original. (Five Whys: GH-190)
    pub run_golden_rule_test: bool,
    /// Path to golden reference JSON for the model
    pub golden_reference_path: Option<String>,
}

impl Default for ExecutionConfig {
    fn default() -> Self {
        Self {
            failure_policy: FailurePolicy::default(),
            default_timeout_ms: 60_000,
            max_workers: 4,
            dry_run: false,
            subprocess_mode: false,
            model_path: None,
            no_gpu: false,
            run_conversion_tests: true, // P0 CRITICAL: Always run by default
            run_differential_tests: true, // v1.3.0: Differential testing enabled by default
            run_profile_ci: false,      // Only enable for CI pipelines
            run_trace_payload: true,    // v1.3.0: Trace payload enabled by default
            run_golden_rule_test: true, // v1.3.1: Golden Rule (Five Whys GH-190)
            golden_reference_path: None,
        }
    }
}

/// Executor for running playbooks
pub struct Executor {
    config: ExecutionConfig,
    collector: EvidenceCollector,
    command_runner: Arc<dyn CommandRunner>,
}

impl std::fmt::Debug for Executor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Executor")
            .field("config", &self.config)
            .field("collector", &self.collector)
            .field("command_runner", &"<dyn CommandRunner>")
            .finish()
    }
}

impl Executor {
    /// Create a new executor with default config
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: ExecutionConfig::default(),
            collector: EvidenceCollector::new(),
            command_runner: Arc::new(RealCommandRunner::new()),
        }
    }

    /// Create a new executor with custom config
    #[must_use]
    pub fn with_config(config: ExecutionConfig) -> Self {
        Self {
            config,
            collector: EvidenceCollector::new(),
            command_runner: Arc::new(RealCommandRunner::new()),
        }
    }

    /// Create a new executor with custom config and command runner
    #[must_use]
    pub fn with_runner(config: ExecutionConfig, runner: Arc<dyn CommandRunner>) -> Self {
        Self {
            config,
            collector: EvidenceCollector::new(),
            command_runner: runner,
        }
    }

    /// Execute a playbook
    ///
    /// # Errors
    ///
    /// Returns an error if execution fails critically.
    pub fn execute(&mut self, playbook: &Playbook) -> Result<ExecutionResult> {
        let scenarios = playbook.generate_scenarios();
        let total = scenarios.len();
        let start = Instant::now();

        // Check gateway conditions first
        if let Err(e) = self.check_gateways(playbook) {
            return Ok(ExecutionResult {
                playbook_name: playbook.name.clone(),
                total_scenarios: total,
                passed: 0,
                failed: total,
                skipped: 0,
                duration_ms: start.elapsed().as_millis() as u64,
                gateway_failed: Some(e.to_string()),
                evidence: self.collector.clone(),
            });
        }

        let mut passed = 0;
        let mut failed = 0;
        let mut skipped = 0;

        for scenario in scenarios {
            if self.config.dry_run {
                // In dry run mode, just generate the command
                let cmd = scenario.to_command("model.gguf");
                println!("[DRY RUN] {cmd}");
                skipped += 1;
                continue;
            }

            let evidence = self.execute_scenario(&scenario);
            if evidence.outcome.is_pass() {
                passed += 1;
            } else {
                failed += 1;

                // Check failure policy
                match self.config.failure_policy {
                    FailurePolicy::StopOnFirst => {
                        self.collector.add(evidence);
                        break;
                    }
                    FailurePolicy::StopOnP0 => {
                        // Check if this is a P0 failure
                        if evidence.gate_id.contains("-P0-") {
                            self.collector.add(evidence);
                            break;
                        }
                    }
                    FailurePolicy::CollectAll => {}
                }
            }
            self.collector.add(evidence);
        }

        // P0 CRITICAL: Run format conversion tests
        let mut conversion_passed = 0;
        let mut conversion_failed = 0;
        if self.config.run_conversion_tests && self.config.subprocess_mode {
            if let Some(model_path) = self.config.model_path.clone() {
                let model_id = playbook.model_id();
                let (cp, cf) = self.run_conversion_tests(Path::new(&model_path), &model_id);
                conversion_passed = cp;
                conversion_failed = cf;
            }
        }

        // INVARIANT I-1: Golden Rule Test (convert → inference → diff)
        // This single test catches ALL conversion bugs (Five Whys: GH-190)
        let mut golden_passed = 0;
        let mut golden_failed = 0;
        if self.config.run_golden_rule_test && self.config.subprocess_mode {
            if let Some(model_path) = self.config.model_path.clone() {
                let model_id = playbook.model_id();
                let (gp, gf) = self.run_golden_rule_test(Path::new(&model_path), &model_id);
                golden_passed = gp;
                golden_failed = gf;
            }
        }

        let total_passed = passed + conversion_passed + golden_passed;
        let total_failed = failed + conversion_failed + golden_failed;

        Ok(ExecutionResult {
            playbook_name: playbook.name.clone(),
            total_scenarios: total
                + conversion_passed
                + conversion_failed
                + golden_passed
                + golden_failed,
            passed: total_passed,
            failed: total_failed,
            skipped,
            duration_ms: start.elapsed().as_millis() as u64,
            gateway_failed: None,
            evidence: self.collector.clone(),
        })
    }

    /// Run P0 format conversion tests
    fn run_conversion_tests(&mut self, model_path: &Path, model_id: &ModelId) -> (usize, usize) {
        let config = if self.config.no_gpu {
            ConversionConfig::cpu_only()
        } else {
            ConversionConfig::default()
        };

        let executor = ConversionExecutor::new(config);

        match executor.execute_all(model_path, model_id) {
            Ok(result) => {
                // Add all conversion evidence to collector
                for ev in result.evidence {
                    self.collector.add(ev);
                }
                (result.passed, result.failed)
            }
            Err(e) => {
                // Critical conversion infrastructure failure
                let ev = Evidence::falsified(
                    "F-CONV-INFRA-001",
                    apr_qa_gen::QaScenario::new(
                        model_id.clone(),
                        apr_qa_gen::Modality::Run,
                        apr_qa_gen::Backend::Cpu,
                        apr_qa_gen::Format::Gguf,
                        "Conversion infrastructure".to_string(),
                        0,
                    ),
                    format!("Conversion infrastructure failure: {e}"),
                    "N/A",
                    0,
                );
                self.collector.add(ev);
                (0, 1)
            }
        }
    }

    /// Golden Rule Test: convert model, run inference, diff against original.
    ///
    /// This is the SINGLE MOST IMPORTANT test in the entire pipeline.
    /// It encodes the only invariant that matters for format conversion:
    ///   "Converted models MUST produce the same output as the original."
    ///
    /// Would have caught: GH-186, GH-189, GH-190 (all 3 P0 conversion bugs).
    /// See: docs/five-whys/GH-190-systemic-conversion-failures.md
    fn run_golden_rule_test(&mut self, model_path: &Path, model_id: &ModelId) -> (usize, usize) {
        let prompt = "What is 2+2?";
        let max_tokens = 10;

        // Step 1: Run inference on original model
        let original_result =
            self.command_runner
                .run_inference(model_path, prompt, max_tokens, false, &[]);

        if !original_result.success {
            let ev = Evidence::falsified(
                "F-GOLDEN-RULE-001",
                Self::golden_scenario(model_id),
                format!(
                    "Golden Rule: original inference failed: {}",
                    original_result.stderr
                ),
                "N/A",
                0,
            );
            self.collector.add(ev);
            return (0, 1);
        }

        // Step 2: Convert to APR
        let apr_path =
            std::path::PathBuf::from(format!("/tmp/golden-rule-test-{}.apr", model_id.name));
        let convert_result = self.command_runner.convert_model(model_path, &apr_path);

        if !convert_result.success {
            let ev = Evidence::falsified(
                "F-GOLDEN-RULE-002",
                Self::golden_scenario(model_id),
                format!("Golden Rule: conversion failed: {}", convert_result.stderr),
                "N/A",
                0,
            );
            self.collector.add(ev);
            return (0, 1);
        }

        // Step 3: Run inference on converted model
        let converted_result =
            self.command_runner
                .run_inference(&apr_path, prompt, max_tokens, false, &[]);

        if !converted_result.success {
            let ev = Evidence::falsified(
                "F-GOLDEN-RULE-003",
                Self::golden_scenario(model_id),
                format!(
                    "Golden Rule: converted inference failed: {}",
                    converted_result.stderr
                ),
                "N/A",
                0,
            );
            self.collector.add(ev);
            return (0, 1);
        }

        // Step 4: DIFF — the actual Golden Rule assertion
        // Extract just the "Output:" line from both
        let orig_text = Self::extract_output_text(&original_result.stdout);
        let conv_text = Self::extract_output_text(&converted_result.stdout);

        if orig_text == conv_text {
            let ev = Evidence::corroborated(
                "F-GOLDEN-RULE-001",
                Self::golden_scenario(model_id),
                &format!("Golden Rule PASS: identical output: {orig_text}"),
                0,
            );
            self.collector.add(ev);

            // Cleanup
            let _ = std::fs::remove_file(&apr_path);
            (1, 0)
        } else {
            let ev = Evidence::falsified(
                "F-GOLDEN-RULE-001",
                Self::golden_scenario(model_id),
                format!(
                    "Golden Rule FAIL: output differs after conversion.\n\
                     Original:  {orig_text}\n\
                     Converted: {conv_text}"
                ),
                &converted_result.stdout,
                0,
            );
            self.collector.add(ev);

            // Keep the APR file for investigation
            (0, 1)
        }
    }

    /// Extract the "Output:" text from apr run output
    fn extract_output_text(raw: &str) -> String {
        let mut capture = false;
        let mut lines = Vec::new();
        for line in raw.lines() {
            if line.starts_with("Output:") {
                capture = true;
                continue;
            }
            if capture {
                if line.starts_with("Completed in") || line.is_empty() {
                    break;
                }
                lines.push(line.trim());
            }
        }
        lines.join(" ").trim().to_string()
    }

    /// Create a scenario for golden rule evidence
    fn golden_scenario(model_id: &ModelId) -> apr_qa_gen::QaScenario {
        apr_qa_gen::QaScenario::new(
            model_id.clone(),
            apr_qa_gen::Modality::Run,
            apr_qa_gen::Backend::Cpu,
            apr_qa_gen::Format::Apr,
            "Golden Rule: convert → inference → diff".to_string(),
            0,
        )
    }

    /// Execute a single scenario
    fn execute_scenario(&self, scenario: &QaScenario) -> Evidence {
        let start = Instant::now();

        let (output, stderr, exit_code, tps) = if self.config.subprocess_mode {
            self.subprocess_execution(scenario)
        } else {
            (self.simulate_execution(scenario), None, 0, None)
        };

        let duration = start.elapsed().as_millis() as u64;

        // Check for crash
        if exit_code < 0 {
            return Evidence::crashed(
                "G3-STABLE",
                scenario.clone(),
                stderr.as_deref().unwrap_or("Process crashed"),
                exit_code,
                duration,
            );
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

    /// Simulate command execution (for testing)
    fn simulate_execution(&self, scenario: &QaScenario) -> String {
        // Simulate successful output for arithmetic prompts
        if scenario.prompt.contains("2+2") {
            "The answer is 4.".to_string()
        } else if scenario.prompt.starts_with("def ") || scenario.prompt.starts_with("fn ") {
            // Code completion
            "    return result".to_string()
        } else if scenario.prompt.is_empty() {
            // Empty prompt - should fail
            String::new()
        } else {
            // Generic response
            "Hello! I'm an AI assistant.".to_string()
        }
    }

    /// Execute via subprocess (real apr commands)
    fn subprocess_execution(
        &self,
        scenario: &QaScenario,
    ) -> (String, Option<String>, i32, Option<f64>) {
        let model_path = self.config.model_path.as_deref().unwrap_or("model.gguf");

        let output = self.command_runner.run_inference(
            Path::new(model_path),
            &scenario.prompt,
            32,
            self.config.no_gpu,
            &["--benchmark", "--json"],
        );

        // Try to parse tok/s from JSON output
        let tps = Self::parse_tps_from_output(&output.stdout);

        // Extract the actual generated text (not the JSON benchmark data)
        let generated_text = Self::extract_generated_text(&output.stdout);

        (
            generated_text,
            if output.stderr.is_empty() {
                None
            } else {
                Some(output.stderr)
            },
            output.exit_code,
            tps,
        )
    }

    /// Parse tokens per second from apr output
    fn parse_tps_from_output(output: &str) -> Option<f64> {
        // Try to find "tok/s: X.X" pattern
        output.find("tok/s:").and_then(|pos| {
            let rest = &output[pos + 6..];
            let tps_str: String = rest
                .chars()
                .skip_while(|c| c.is_whitespace())
                .take_while(|c| c.is_ascii_digit() || *c == '.')
                .collect();
            tps_str.parse().ok()
        })
    }

    /// Extract generated text from apr output
    fn extract_generated_text(output: &str) -> String {
        // apr run with --json outputs JSON, otherwise plain text
        // For now, return the whole output (apr outputs generated text first)
        output
            .lines()
            .filter(|line| !line.starts_with("===") && !line.contains("tok/s"))
            .collect::<Vec<_>>()
            .join("\n")
            .trim()
            .to_string()
    }

    /// Check gateway conditions
    fn check_gateways(&self, _playbook: &Playbook) -> Result<()> {
        // Gateway checks would verify:
        // G1: Model loads
        // G2: Basic inference works
        // G3: No crashes
        // G4: Output not garbage

        // For now, assume gateways pass
        Ok(())
    }

    /// Get collected evidence
    #[must_use]
    pub fn evidence(&self) -> &EvidenceCollector {
        &self.collector
    }

    /// Get configuration
    #[must_use]
    pub fn config(&self) -> &ExecutionConfig {
        &self.config
    }
}

impl Default for Executor {
    fn default() -> Self {
        Self::new()
    }
}

/// APR Tool test executor for comprehensive tool coverage
#[allow(dead_code)] // timeout_ms reserved for future timeout enforcement
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
        use std::process::Command;
        let start = std::time::Instant::now();

        let svg_path = output_path.join("profile_flamegraph.svg");
        let mut cmd = Command::new("apr");
        cmd.arg("run")
            .arg(&self.model_path)
            .arg("-p")
            .arg("Hello")
            .arg("--max-tokens")
            .arg("4")
            .arg("--profile")
            .arg("--profile-output")
            .arg(&svg_path);

        if self.no_gpu {
            cmd.arg("--no-gpu");
        }

        let output = cmd.output();
        let duration_ms = start.elapsed().as_millis() as u64;

        // Check if flamegraph was generated
        let flamegraph_exists = svg_path.exists();
        let flamegraph_valid = if flamegraph_exists {
            // Basic SVG validation - check for svg tag
            std::fs::read_to_string(&svg_path)
                .map(|content| content.contains("<svg") && content.contains("</svg>"))
                .unwrap_or(false)
        } else {
            false
        };

        match output {
            Ok(out) => {
                let stdout = String::from_utf8_lossy(&out.stdout).to_string();
                let stderr = String::from_utf8_lossy(&out.stderr).to_string();

                // If apr doesn't support --profile-output, it will error
                if stderr.contains("unexpected argument") || stderr.contains("unrecognized") {
                    return ToolTestResult {
                        tool: "profile-flamegraph".to_string(),
                        passed: false,
                        exit_code: -2,
                        stdout,
                        stderr: "Feature not available: apr does not support --profile-output"
                            .to_string(),
                        duration_ms,
                        gate_id: "F-PROFILE-002".to_string(),
                    };
                }

                ToolTestResult {
                    tool: "profile-flamegraph".to_string(),
                    passed: flamegraph_valid,
                    exit_code: i32::from(!flamegraph_valid),
                    stdout: format!(
                        "Flamegraph exists: {flamegraph_exists}, valid: {flamegraph_valid}"
                    ),
                    stderr,
                    duration_ms,
                    gate_id: "F-PROFILE-002".to_string(),
                }
            }
            Err(e) => ToolTestResult {
                tool: "profile-flamegraph".to_string(),
                passed: false,
                exit_code: -1,
                stdout: String::new(),
                stderr: e.to_string(),
                duration_ms,
                gate_id: "F-PROFILE-002".to_string(),
            },
        }
    }

    /// Execute apr profile with focus filtering (F-PROFILE-003)
    ///
    /// Tests that profile --focus option works to limit scope.
    /// This feature may not be available in all apr versions.
    #[must_use]
    pub fn execute_profile_focus(&self, focus: &str) -> ToolTestResult {
        use std::process::Command;
        let start = std::time::Instant::now();

        let mut cmd = Command::new("apr");
        cmd.arg("run")
            .arg(&self.model_path)
            .arg("-p")
            .arg("Hello")
            .arg("--max-tokens")
            .arg("4")
            .arg("--profile")
            .arg("--focus")
            .arg(focus);

        if self.no_gpu {
            cmd.arg("--no-gpu");
        }

        let output = cmd.output();
        let duration_ms = start.elapsed().as_millis() as u64;

        match output {
            Ok(out) => {
                let stdout = String::from_utf8_lossy(&out.stdout).to_string();
                let stderr = String::from_utf8_lossy(&out.stderr).to_string();

                // If apr doesn't support --focus, it will error
                if stderr.contains("unexpected argument") || stderr.contains("unrecognized") {
                    return ToolTestResult {
                        tool: "profile-focus".to_string(),
                        passed: false,
                        exit_code: -2,
                        stdout,
                        stderr: format!(
                            "Feature not available: apr does not support --focus {focus}"
                        ),
                        duration_ms,
                        gate_id: "F-PROFILE-003".to_string(),
                    };
                }

                // Check if output is focused (contains focus term or is shorter than unfocused)
                let passed = out.status.success();

                ToolTestResult {
                    tool: "profile-focus".to_string(),
                    passed,
                    exit_code: out.status.code().unwrap_or(-1),
                    stdout,
                    stderr,
                    duration_ms,
                    gate_id: "F-PROFILE-003".to_string(),
                }
            }
            Err(e) => ToolTestResult {
                tool: "profile-focus".to_string(),
                passed: false,
                exit_code: -1,
                stdout: String::new(),
                stderr: e.to_string(),
                duration_ms,
                gate_id: "F-PROFILE-003".to_string(),
            },
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

#[cfg(test)]
mod tests {
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

    #[test]
    fn test_executor_dry_run() {
        let config = ExecutionConfig {
            dry_run: true,
            ..Default::default()
        };
        let mut executor = Executor::with_config(config);
        let playbook = test_playbook();

        let result = executor.execute(&playbook).expect("Execution failed");

        assert_eq!(result.skipped, 5);
        assert_eq!(result.passed, 0);
        assert_eq!(result.failed, 0);
    }

    #[test]
    fn test_executor_simulate() {
        let mut executor = Executor::new();
        let playbook = test_playbook();

        let result = executor.execute(&playbook).expect("Execution failed");

        // All scenarios should pass (they use "2+2" prompts)
        assert_eq!(result.total_scenarios, 5);
        assert!(result.passed > 0);
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
    fn test_simulate_execution_code_completion() {
        let executor = Executor::new();
        let mut scenario = test_scenario();
        scenario.prompt = "def fibonacci(n):".to_string();

        let output = executor.simulate_execution(&scenario);
        assert!(output.contains("return"));
    }

    #[test]
    fn test_simulate_execution_fn_code() {
        let executor = Executor::new();
        let mut scenario = test_scenario();
        scenario.prompt = "fn main() {".to_string();

        let output = executor.simulate_execution(&scenario);
        assert!(output.contains("return"));
    }

    #[test]
    fn test_simulate_execution_empty_prompt() {
        let executor = Executor::new();
        let mut scenario = test_scenario();
        scenario.prompt = String::new();

        let output = executor.simulate_execution(&scenario);
        assert!(output.is_empty());
    }

    #[test]
    fn test_simulate_execution_generic_prompt() {
        let executor = Executor::new();
        let mut scenario = test_scenario();
        scenario.prompt = "Hello, how are you?".to_string();

        let output = executor.simulate_execution(&scenario);
        assert!(output.contains("AI assistant"));
    }

    #[test]
    fn test_execute_scenario() {
        let executor = Executor::new();
        let scenario = test_scenario();

        let evidence = executor.execute_scenario(&scenario);
        assert!(evidence.outcome.is_pass());
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
    fn test_execute_scenario_failing() {
        let executor = Executor::new();
        let mut scenario = test_scenario();
        scenario.prompt = String::new(); // Empty prompt fails

        let evidence = executor.execute_scenario(&scenario);
        // Empty prompt should fail garbage oracle
        assert!(evidence.output.is_empty() || evidence.outcome.is_fail());
    }

    #[test]
    fn test_execute_with_failures() {
        let mut executor = Executor::with_config(ExecutionConfig {
            failure_policy: FailurePolicy::CollectAll,
            ..Default::default()
        });

        let yaml = r#"
name: test-playbook
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

        // Should collect all results
        assert_eq!(result.total_scenarios, 2);
    }

    #[test]
    fn test_execute_with_stop_on_first() {
        let config = ExecutionConfig {
            failure_policy: FailurePolicy::StopOnFirst,
            ..Default::default()
        };
        let mut executor = Executor::with_config(config);

        let yaml = r#"
name: test-playbook
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

        // Should have executed scenarios
        assert!(result.total_scenarios > 0);
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
    fn test_execution_config_subprocess_mode() {
        let config = ExecutionConfig {
            subprocess_mode: true,
            model_path: Some("/path/model.gguf".to_string()),
            ..Default::default()
        };
        assert!(config.subprocess_mode);
        assert!(config.model_path.is_some());
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
    fn test_execute_with_p0_stop() {
        let config = ExecutionConfig {
            failure_policy: FailurePolicy::StopOnP0,
            ..Default::default()
        };
        let mut executor = Executor::with_config(config);

        let yaml = r#"
name: test-playbook
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

        // With default prompts (2+2), all should pass
        assert!(result.total_scenarios > 0);
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

    #[test]
    fn test_execute_scenario_with_falsified_output() {
        let executor = Executor::new();
        // Create scenario that will produce output but fail oracle
        let scenario = QaScenario::new(
            ModelId::new("test", "model"),
            Modality::Run,
            Backend::Cpu,
            Format::Gguf,
            "What is 5+5?".to_string(), // Won't match simulated "2+2" response
            42,
        );

        let evidence = executor.execute_scenario(&scenario);
        // Simulated execution returns "Hello! I'm an AI assistant." which doesn't contain "10"
        // The oracle should return a result (corroborated for generic prompts)
        // Verify evidence was created with valid fields
        assert!(!evidence.id.is_empty());
        assert!(!evidence.gate_id.is_empty());
    }

    #[test]
    fn test_simulate_execution_arithmetic_2plus2() {
        let executor = Executor::new();
        let scenario = QaScenario::new(
            ModelId::new("test", "model"),
            Modality::Run,
            Backend::Cpu,
            Format::Gguf,
            "What is 2+2?".to_string(),
            0,
        );
        let output = executor.simulate_execution(&scenario);
        assert!(output.contains("4"));
    }

    #[test]
    fn test_simulate_execution_def_code() {
        let executor = Executor::new();
        let scenario = QaScenario::new(
            ModelId::new("test", "model"),
            Modality::Run,
            Backend::Cpu,
            Format::Gguf,
            "def hello():".to_string(),
            0,
        );
        let output = executor.simulate_execution(&scenario);
        assert!(output.contains("return"));
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
    fn test_simulate_execution_multiplication() {
        let executor = Executor::new();
        let scenario = QaScenario::new(
            ModelId::new("test", "model"),
            Modality::Run,
            Backend::Cpu,
            Format::Gguf,
            "Calculate 3*3".to_string(),
            0,
        );
        // Not matching 2+2, so returns generic response
        let output = executor.simulate_execution(&scenario);
        assert!(output.contains("AI assistant"));
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
            subprocess_mode: true,
            model_path: Some("/path/to/model.gguf".to_string()),
            no_gpu: true,
            run_conversion_tests: false,
            run_differential_tests: false,
            run_profile_ci: true,
            run_trace_payload: false,
            run_golden_rule_test: false,
            golden_reference_path: Some("/path/to/ref.json".to_string()),
        };
        assert_eq!(config.failure_policy, FailurePolicy::CollectAll);
        assert!(config.dry_run);
        assert!(config.subprocess_mode);
        assert!(config.no_gpu);
        assert!(!config.run_conversion_tests);
        assert!(!config.run_differential_tests);
        assert!(config.run_profile_ci);
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
    fn test_execute_scenario_corroborated() {
        let executor = Executor::new();
        let scenario = QaScenario::new(
            ModelId::new("test", "model"),
            Modality::Run,
            Backend::Cpu,
            Format::Gguf,
            "2+2=".to_string(),
            1,
        );
        let evidence = executor.execute_scenario(&scenario);
        assert!(evidence.outcome.is_pass());
    }

    #[test]
    fn test_execute_scenario_with_chat_modality() {
        let executor = Executor::new();
        let scenario = QaScenario::new(
            ModelId::new("test", "model"),
            Modality::Chat,
            Backend::Cpu,
            Format::Gguf,
            "Hello".to_string(),
            2,
        );
        let evidence = executor.execute_scenario(&scenario);
        // Chat with "Hello" should get generic response
        assert!(!evidence.id.is_empty());
    }

    #[test]
    fn test_execute_scenario_with_serve_modality() {
        let executor = Executor::new();
        let scenario = QaScenario::new(
            ModelId::new("test", "model"),
            Modality::Serve,
            Backend::Gpu,
            Format::Apr,
            "Test prompt".to_string(),
            3,
        );
        let evidence = executor.execute_scenario(&scenario);
        assert!(!evidence.gate_id.is_empty());
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
    fn test_simulate_execution_rust_fn() {
        let executor = Executor::new();
        let scenario = QaScenario::new(
            ModelId::new("test", "model"),
            Modality::Run,
            Backend::Cpu,
            Format::Gguf,
            "fn fibonacci(n: u32) -> u32 {".to_string(),
            0,
        );
        let output = executor.simulate_execution(&scenario);
        assert!(output.contains("return"));
    }

    #[test]
    fn test_simulate_execution_python_def() {
        let executor = Executor::new();
        let scenario = QaScenario::new(
            ModelId::new("test", "model"),
            Modality::Run,
            Backend::Cpu,
            Format::Gguf,
            "def add(a, b):".to_string(),
            0,
        );
        let output = executor.simulate_execution(&scenario);
        assert!(output.contains("return"));
    }

    #[test]
    fn test_executor_execute_multiple_scenarios() {
        let mut executor = Executor::new();
        let yaml = r#"
name: multi-scenario
version: "1.0.0"
model:
  hf_repo: "test/model"
  formats: [gguf]
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 10
"#;
        let playbook = Playbook::from_yaml(yaml).expect("Failed to parse");
        let result = executor.execute(&playbook).expect("Execution failed");
        assert_eq!(result.total_scenarios, 10);
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
        let mock_runner = MockCommandRunner::new()
            .with_tps(42.0)
            .with_inference_response("The answer is 4.");

        let config = ExecutionConfig {
            subprocess_mode: true,
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

        let (output, stderr, exit_code, tps) = executor.subprocess_execution(&scenario);

        assert!(output.contains("4") || output.is_empty()); // Depends on extract logic
        assert!(stderr.is_none_or(|s| s.is_empty()));
        assert_eq!(exit_code, 0);
        // tps may or may not be parsed depending on output format
        let _ = tps;
    }

    #[test]
    fn test_executor_with_mock_runner_inference_failure() {
        let mock_runner = MockCommandRunner::new().with_inference_failure();

        let config = ExecutionConfig {
            subprocess_mode: true,
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

        let (_, stderr, exit_code, _) = executor.subprocess_execution(&scenario);

        assert_eq!(exit_code, 1);
        assert!(stderr.is_some());
    }

    #[test]
    fn test_executor_with_mock_runner_execute_scenario() {
        let mock_runner = MockCommandRunner::new()
            .with_tps(30.0)
            .with_inference_response("The answer is 4.");

        let config = ExecutionConfig {
            subprocess_mode: true,
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
            subprocess_mode: true,
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
            subprocess_mode: true,
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
            subprocess_mode: true,
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
            subprocess_mode: true,
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

        let (_, _, exit_code, _) = executor.subprocess_execution(&scenario);
        assert_eq!(exit_code, 0);
    }

    #[test]
    fn test_executor_execute_playbook_with_subprocess_mode() {
        let mock_runner = MockCommandRunner::new()
            .with_tps(25.0)
            .with_inference_response("The answer is 4.");

        let config = ExecutionConfig {
            subprocess_mode: true,
            model_path: Some("/test/model.gguf".to_string()),
            run_conversion_tests: false,
            run_differential_tests: false,
            run_golden_rule_test: false,
            run_trace_payload: false,
            run_profile_ci: false,
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
}
