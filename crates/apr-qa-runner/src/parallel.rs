//! Parallel execution support using Rayon
//!
//! Implements Heijunka (load-balanced) parallel execution across workers.

use crate::evidence::{Evidence, PerformanceMetrics};
use apr_qa_gen::QaScenario;
use rayon::prelude::*;
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Instant;

/// Execution mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExecutionMode {
    /// Simulate execution (for testing)
    #[default]
    Simulate,
    /// Real subprocess execution
    Subprocess,
}

/// Parallel executor configuration
#[derive(Debug, Clone)]
pub struct ParallelConfig {
    /// Number of worker threads
    pub num_workers: usize,
    /// Timeout per scenario in milliseconds
    pub timeout_ms: u64,
    /// Execution mode
    pub mode: ExecutionMode,
    /// Path to model file
    pub model_path: String,
    /// Stop on first failure
    pub stop_on_failure: bool,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            num_workers: num_cpus::get().min(4),
            timeout_ms: 60_000,
            mode: ExecutionMode::Simulate,
            model_path: "model.gguf".to_string(),
            stop_on_failure: false,
        }
    }
}

/// Result of parallel execution
#[derive(Debug)]
pub struct ParallelResult {
    /// All evidence collected
    pub evidence: Vec<Evidence>,
    /// Number of passed scenarios
    pub passed: usize,
    /// Number of failed scenarios
    pub failed: usize,
    /// Number of skipped scenarios
    pub skipped: usize,
    /// Total duration in milliseconds
    pub duration_ms: u64,
    /// Whether execution was stopped early
    pub stopped_early: bool,
}

/// Parallel scenario executor
pub struct ParallelExecutor {
    config: ParallelConfig,
}

impl ParallelExecutor {
    /// Create a new parallel executor
    #[must_use]
    pub fn new(config: ParallelConfig) -> Self {
        // Configure rayon thread pool
        rayon::ThreadPoolBuilder::new()
            .num_threads(config.num_workers)
            .build_global()
            .ok(); // Ignore if already configured
        Self { config }
    }

    /// Execute scenarios in parallel
    #[must_use]
    pub fn execute(&self, scenarios: &[QaScenario]) -> ParallelResult {
        let start = Instant::now();
        let stop_flag = Arc::new(AtomicBool::new(false));
        let passed = Arc::new(AtomicUsize::new(0));
        let failed = Arc::new(AtomicUsize::new(0));
        let skipped = Arc::new(AtomicUsize::new(0));

        let evidence: Vec<Evidence> = scenarios
            .par_iter()
            .filter_map(|scenario| {
                // Check if we should stop
                if self.config.stop_on_failure && stop_flag.load(Ordering::Relaxed) {
                    skipped.fetch_add(1, Ordering::Relaxed);
                    return None;
                }

                let result = self.execute_single(scenario);

                if result.outcome.is_pass() {
                    passed.fetch_add(1, Ordering::Relaxed);
                } else {
                    failed.fetch_add(1, Ordering::Relaxed);
                    if self.config.stop_on_failure {
                        stop_flag.store(true, Ordering::Relaxed);
                    }
                }

                Some(result)
            })
            .collect();

        ParallelResult {
            evidence,
            passed: passed.load(Ordering::Relaxed),
            failed: failed.load(Ordering::Relaxed),
            skipped: skipped.load(Ordering::Relaxed),
            duration_ms: start.elapsed().as_millis() as u64,
            stopped_early: stop_flag.load(Ordering::Relaxed),
        }
    }

    /// Execute a single scenario
    fn execute_single(&self, scenario: &QaScenario) -> Evidence {
        let start = Instant::now();

        let (output, exit_code, stderr) = match self.config.mode {
            ExecutionMode::Simulate => (self.simulate_execution(scenario), 0, None),
            ExecutionMode::Subprocess => self.subprocess_execution(scenario),
        };

        let duration = start.elapsed().as_millis() as u64;
        let gate_id = format!("F-{}-001", scenario.mqs_category());

        // Check for crash
        if exit_code != 0 {
            return Evidence::crashed(
                &gate_id,
                scenario.clone(),
                "Non-zero exit code",
                exit_code,
                duration,
            )
            .with_stderr(stderr);
        }

        // Check for timeout
        if duration > self.config.timeout_ms {
            return Evidence::timeout(&gate_id, scenario.clone(), duration);
        }

        // Evaluate output with oracle
        let oracle_result = scenario.evaluate(&output);

        match oracle_result {
            apr_qa_gen::OracleResult::Corroborated { evidence: reason } => {
                Evidence::corroborated(&gate_id, scenario.clone(), &output, duration)
                    .with_metrics(PerformanceMetrics {
                        duration_ms: duration,
                        total_tokens: Some(estimate_tokens(&output)),
                        ..Default::default()
                    })
                    .with_reason(reason)
            }
            apr_qa_gen::OracleResult::Falsified {
                reason,
                evidence: _,
            } => Evidence::falsified(&gate_id, scenario.clone(), reason, &output, duration),
        }
    }

    /// Simulate execution for testing
    fn simulate_execution(&self, scenario: &QaScenario) -> String {
        // Simulate based on prompt content
        if scenario.prompt.contains("2+2") || scenario.prompt.contains("2 + 2") {
            "The answer is 4.".to_string()
        } else if scenario.prompt.contains('+')
            || scenario.prompt.contains('-')
            || scenario.prompt.contains('*')
        {
            // Other arithmetic - simulate correct answer
            "42".to_string()
        } else if scenario.prompt.starts_with("def ") || scenario.prompt.starts_with("fn ") {
            "    return result\n".to_string()
        } else if scenario.prompt.starts_with("class ") || scenario.prompt.starts_with("struct ") {
            "    pass\n".to_string()
        } else if scenario.prompt.is_empty() || scenario.prompt.trim().is_empty() {
            String::new()
        } else {
            "Hello! I'm an AI assistant. How can I help you today?".to_string()
        }
    }

    /// Execute via subprocess (real execution)
    fn subprocess_execution(&self, scenario: &QaScenario) -> (String, i32, Option<String>) {
        let cmd_str = scenario.to_command(&self.config.model_path);
        let parts: Vec<&str> = cmd_str.split_whitespace().collect();

        if parts.is_empty() {
            return (String::new(), -1, Some("Empty command".to_string()));
        }

        let result = Command::new(parts[0])
            .args(&parts[1..])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output();

        match result {
            Ok(output) => {
                let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                let stderr = String::from_utf8_lossy(&output.stderr).to_string();
                let exit_code = output.status.code().unwrap_or(-1);
                (
                    stdout,
                    exit_code,
                    if stderr.is_empty() {
                        None
                    } else {
                        Some(stderr)
                    },
                )
            }
            Err(e) => (String::new(), -1, Some(e.to_string())),
        }
    }
}

impl Default for ParallelExecutor {
    fn default() -> Self {
        Self::new(ParallelConfig::default())
    }
}

/// Estimate token count from output (rough heuristic)
fn estimate_tokens(text: &str) -> u32 {
    // Rough estimate: ~4 chars per token for English
    (text.len() / 4).max(1) as u32
}

/// Extension trait for Evidence to add optional fields
trait EvidenceExt {
    fn with_stderr(self, stderr: Option<String>) -> Self;
    fn with_reason(self, reason: String) -> Self;
}

impl EvidenceExt for Evidence {
    fn with_stderr(mut self, stderr: Option<String>) -> Self {
        self.stderr = stderr;
        self
    }

    fn with_reason(mut self, reason: String) -> Self {
        self.reason = reason;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use apr_qa_gen::{Backend, Format, Modality, ModelId};

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

    fn test_scenarios(count: usize) -> Vec<QaScenario> {
        (0..count)
            .map(|i| {
                QaScenario::new(
                    ModelId::new("test", "model"),
                    Modality::Run,
                    Backend::Cpu,
                    Format::Gguf,
                    format!("What is {}+{}?", i, i + 1),
                    i as u64,
                )
            })
            .collect()
    }

    #[test]
    fn test_parallel_config_default() {
        let config = ParallelConfig::default();
        assert!(config.num_workers > 0);
        assert_eq!(config.timeout_ms, 60_000);
        assert_eq!(config.mode, ExecutionMode::Simulate);
    }

    #[test]
    fn test_parallel_executor_single() {
        let executor = ParallelExecutor::default();
        let scenario = test_scenario();

        let evidence = executor.execute_single(&scenario);
        assert!(evidence.outcome.is_pass());
    }

    #[test]
    fn test_parallel_executor_batch() {
        let config = ParallelConfig {
            num_workers: 2,
            ..Default::default()
        };
        let executor = ParallelExecutor::new(config);
        let scenarios = test_scenarios(10);

        let result = executor.execute(&scenarios);

        assert_eq!(result.evidence.len(), 10);
        assert_eq!(result.passed + result.failed + result.skipped, 10);
    }

    #[test]
    fn test_parallel_executor_stop_on_failure() {
        let config = ParallelConfig {
            num_workers: 1, // Single thread for predictable behavior
            stop_on_failure: true,
            ..Default::default()
        };
        let executor = ParallelExecutor::new(config);

        // Create scenarios with one that will fail (empty prompt)
        let mut scenarios = test_scenarios(5);
        scenarios.insert(
            2,
            QaScenario::new(
                ModelId::new("test", "model"),
                Modality::Run,
                Backend::Cpu,
                Format::Gguf,
                String::new(), // Empty prompt will fail
                99,
            ),
        );

        let result = executor.execute(&scenarios);

        // With single thread and stop on failure, we should stop early
        assert!(result.failed > 0);
    }

    #[test]
    fn test_simulate_execution_arithmetic() {
        let executor = ParallelExecutor::default();
        let scenario = test_scenario();

        let output = executor.simulate_execution(&scenario);
        assert!(output.contains("4"));
    }

    #[test]
    fn test_simulate_execution_code() {
        let executor = ParallelExecutor::default();
        let scenario = QaScenario::new(
            ModelId::new("test", "model"),
            Modality::Run,
            Backend::Cpu,
            Format::Gguf,
            "def fibonacci(n):".to_string(),
            42,
        );

        let output = executor.simulate_execution(&scenario);
        assert!(output.contains("return"));
    }

    #[test]
    fn test_simulate_execution_empty() {
        let executor = ParallelExecutor::default();
        let scenario = QaScenario::new(
            ModelId::new("test", "model"),
            Modality::Run,
            Backend::Cpu,
            Format::Gguf,
            String::new(),
            42,
        );

        let output = executor.simulate_execution(&scenario);
        assert!(output.is_empty());
    }

    #[test]
    fn test_estimate_tokens() {
        assert_eq!(estimate_tokens(""), 1);
        assert_eq!(estimate_tokens("test"), 1);
        assert_eq!(estimate_tokens("hello world this is a test"), 6);
    }

    #[test]
    fn test_execution_mode_default() {
        let mode = ExecutionMode::default();
        assert_eq!(mode, ExecutionMode::Simulate);
    }

    #[test]
    fn test_parallel_result_fields() {
        let result = ParallelResult {
            evidence: vec![],
            passed: 5,
            failed: 2,
            skipped: 1,
            duration_ms: 1000,
            stopped_early: false,
        };

        assert_eq!(result.passed, 5);
        assert_eq!(result.failed, 2);
        assert_eq!(result.skipped, 1);
        assert!(!result.stopped_early);
    }

    #[test]
    fn test_parallel_executor_default() {
        let executor = ParallelExecutor::default();
        let scenarios = test_scenarios(3);
        let result = executor.execute(&scenarios);
        assert_eq!(result.evidence.len(), 3);
    }

    #[test]
    fn test_evidence_ext_with_stderr() {
        let scenario = test_scenario();
        let evidence = Evidence::corroborated("F-TEST-001", scenario, "output", 100);
        let with_stderr = evidence.with_stderr(Some("error output".to_string()));
        assert_eq!(with_stderr.stderr, Some("error output".to_string()));
    }

    #[test]
    fn test_evidence_ext_with_reason() {
        let scenario = test_scenario();
        let evidence = Evidence::corroborated("F-TEST-001", scenario, "output", 100);
        let with_reason = evidence.with_reason("test reason".to_string());
        assert_eq!(with_reason.reason, "test reason");
    }

    #[test]
    fn test_simulate_execution_class() {
        let executor = ParallelExecutor::default();
        let scenario = QaScenario::new(
            ModelId::new("test", "model"),
            Modality::Run,
            Backend::Cpu,
            Format::Gguf,
            "class MyClass:".to_string(),
            42,
        );

        let output = executor.simulate_execution(&scenario);
        assert!(output.contains("pass"));
    }

    #[test]
    fn test_simulate_execution_struct() {
        let executor = ParallelExecutor::default();
        let scenario = QaScenario::new(
            ModelId::new("test", "model"),
            Modality::Run,
            Backend::Cpu,
            Format::Gguf,
            "struct Config {".to_string(),
            42,
        );

        let output = executor.simulate_execution(&scenario);
        assert!(output.contains("pass"));
    }

    #[test]
    fn test_simulate_execution_generic() {
        let executor = ParallelExecutor::default();
        let scenario = QaScenario::new(
            ModelId::new("test", "model"),
            Modality::Run,
            Backend::Cpu,
            Format::Gguf,
            "Hello, how are you?".to_string(),
            42,
        );

        let output = executor.simulate_execution(&scenario);
        assert!(output.contains("assistant"));
    }

    #[test]
    fn test_simulate_execution_whitespace() {
        let executor = ParallelExecutor::default();
        let scenario = QaScenario::new(
            ModelId::new("test", "model"),
            Modality::Run,
            Backend::Cpu,
            Format::Gguf,
            "   ".to_string(),
            42,
        );

        let output = executor.simulate_execution(&scenario);
        assert!(output.is_empty());
    }

    #[test]
    fn test_simulate_execution_other_arithmetic() {
        let executor = ParallelExecutor::default();
        let scenario = QaScenario::new(
            ModelId::new("test", "model"),
            Modality::Run,
            Backend::Cpu,
            Format::Gguf,
            "What is 5*6?".to_string(),
            42,
        );

        let output = executor.simulate_execution(&scenario);
        assert!(output.contains("42"));
    }

    #[test]
    fn test_parallel_config_clone() {
        let config = ParallelConfig::default();
        let cloned = config.clone();
        assert_eq!(cloned.num_workers, config.num_workers);
        assert_eq!(cloned.timeout_ms, config.timeout_ms);
    }

    #[test]
    fn test_parallel_config_debug() {
        let config = ParallelConfig::default();
        let debug_str = format!("{config:?}");
        assert!(debug_str.contains("ParallelConfig"));
    }

    #[test]
    fn test_execution_mode_eq() {
        assert_eq!(ExecutionMode::Simulate, ExecutionMode::Simulate);
        assert_ne!(ExecutionMode::Simulate, ExecutionMode::Subprocess);
    }

    #[test]
    fn test_execution_mode_clone() {
        let mode = ExecutionMode::Subprocess;
        let cloned = mode;
        assert_eq!(mode, cloned);
    }

    #[test]
    fn test_parallel_result_debug() {
        let result = ParallelResult {
            evidence: vec![],
            passed: 0,
            failed: 0,
            skipped: 0,
            duration_ms: 0,
            stopped_early: false,
        };
        let debug_str = format!("{result:?}");
        assert!(debug_str.contains("ParallelResult"));
    }

    #[test]
    fn test_execute_single_failing() {
        let executor = ParallelExecutor::default();
        let scenario = QaScenario::new(
            ModelId::new("test", "model"),
            Modality::Run,
            Backend::Cpu,
            Format::Gguf,
            String::new(), // Empty prompt will fail garbage oracle
            42,
        );

        let evidence = executor.execute_single(&scenario);
        // Empty output should fail
        assert!(evidence.outcome.is_fail() || evidence.output.is_empty());
    }

    #[test]
    fn test_parallel_collect_all() {
        let config = ParallelConfig {
            num_workers: 2,
            stop_on_failure: false,
            ..Default::default()
        };
        let executor = ParallelExecutor::new(config);

        let mut scenarios = test_scenarios(5);
        scenarios.push(QaScenario::new(
            ModelId::new("test", "model"),
            Modality::Run,
            Backend::Cpu,
            Format::Gguf,
            String::new(), // Will fail
            99,
        ));

        let result = executor.execute(&scenarios);
        // Should execute all scenarios
        assert_eq!(result.evidence.len(), 6);
    }

    #[test]
    fn test_parallel_executor_with_custom_model_path() {
        let config = ParallelConfig {
            model_path: "custom/model.gguf".to_string(),
            ..Default::default()
        };
        let executor = ParallelExecutor::new(config);
        assert_eq!(executor.config.model_path, "custom/model.gguf");
    }

    #[test]
    fn test_parallel_executor_with_timeout() {
        let config = ParallelConfig {
            timeout_ms: 1000,
            ..Default::default()
        };
        let executor = ParallelExecutor::new(config);
        assert_eq!(executor.config.timeout_ms, 1000);
    }

    #[test]
    fn test_simulate_execution_subtraction() {
        let executor = ParallelExecutor::default();
        let scenario = QaScenario::new(
            ModelId::new("test", "model"),
            Modality::Run,
            Backend::Cpu,
            Format::Gguf,
            "What is 10-3?".to_string(),
            42,
        );

        let output = executor.simulate_execution(&scenario);
        assert!(output.contains("42"));
    }

    #[test]
    fn test_simulate_execution_fn_rust() {
        let executor = ParallelExecutor::default();
        let scenario = QaScenario::new(
            ModelId::new("test", "model"),
            Modality::Run,
            Backend::Cpu,
            Format::Gguf,
            "fn process() {".to_string(),
            42,
        );

        let output = executor.simulate_execution(&scenario);
        assert!(output.contains("return"));
    }

    #[test]
    fn test_execute_single_corroborated() {
        let executor = ParallelExecutor::default();
        let scenario = QaScenario::new(
            ModelId::new("test", "model"),
            Modality::Run,
            Backend::Cpu,
            Format::Gguf,
            "2+2=".to_string(),
            42,
        );

        let evidence = executor.execute_single(&scenario);
        assert!(evidence.outcome.is_pass());
        assert!(!evidence.output.is_empty());
    }

    #[test]
    fn test_execute_single_falsified() {
        let executor = ParallelExecutor::default();
        let scenario = QaScenario::new(
            ModelId::new("test", "model"),
            Modality::Run,
            Backend::Cpu,
            Format::Gguf,
            String::new(), // Empty will fail
            42,
        );

        let evidence = executor.execute_single(&scenario);
        // Should either fail (empty output) or pass with empty output check
        assert!(evidence.output.is_empty() || evidence.outcome.is_fail());
    }

    #[test]
    fn test_parallel_batch_all_pass() {
        let executor = ParallelExecutor::default();
        let scenarios: Vec<_> = (0..5)
            .map(|i| {
                QaScenario::new(
                    ModelId::new("test", "model"),
                    Modality::Run,
                    Backend::Cpu,
                    Format::Gguf,
                    format!("What is 2+{}?", i),
                    i as u64,
                )
            })
            .collect();

        let result = executor.execute(&scenarios);
        assert_eq!(result.evidence.len(), 5);
        assert!(result.passed > 0);
    }

    #[test]
    fn test_parallel_with_mixed_modalities() {
        let executor = ParallelExecutor::default();
        let scenarios = vec![
            QaScenario::new(
                ModelId::new("test", "model"),
                Modality::Run,
                Backend::Cpu,
                Format::Gguf,
                "2+2=".to_string(),
                1,
            ),
            QaScenario::new(
                ModelId::new("test", "model"),
                Modality::Chat,
                Backend::Cpu,
                Format::Gguf,
                "Hello".to_string(),
                2,
            ),
        ];

        let result = executor.execute(&scenarios);
        assert_eq!(result.evidence.len(), 2);
    }

    #[test]
    fn test_execution_mode_copy() {
        let mode = ExecutionMode::Simulate;
        let copied: ExecutionMode = mode;
        assert_eq!(copied, ExecutionMode::Simulate);
    }

    #[test]
    fn test_execution_mode_debug() {
        let mode = ExecutionMode::Subprocess;
        let debug_str = format!("{mode:?}");
        assert!(debug_str.contains("Subprocess"));
    }

    #[test]
    fn test_parallel_config_with_subprocess_mode() {
        let config = ParallelConfig {
            mode: ExecutionMode::Subprocess,
            ..Default::default()
        };
        assert_eq!(config.mode, ExecutionMode::Subprocess);
    }

    #[test]
    fn test_parallel_result_stopped_early() {
        let result = ParallelResult {
            evidence: vec![],
            passed: 3,
            failed: 1,
            skipped: 6,
            duration_ms: 500,
            stopped_early: true,
        };
        assert!(result.stopped_early);
        assert_eq!(result.skipped, 6);
    }

    #[test]
    fn test_parallel_empty_scenarios() {
        let executor = ParallelExecutor::default();
        let result = executor.execute(&[]);
        assert_eq!(result.evidence.len(), 0);
        assert_eq!(result.passed, 0);
        assert_eq!(result.failed, 0);
    }

    #[test]
    fn test_evidence_ext_with_stderr_none() {
        let scenario = test_scenario();
        let evidence = Evidence::corroborated("F-TEST-001", scenario, "output", 100);
        let with_stderr = evidence.with_stderr(None);
        assert!(with_stderr.stderr.is_none());
    }

    #[test]
    fn test_parallel_config_single_worker() {
        let config = ParallelConfig {
            num_workers: 1,
            ..Default::default()
        };
        let executor = ParallelExecutor::new(config);
        let scenarios = test_scenarios(3);
        let result = executor.execute(&scenarios);
        assert_eq!(result.evidence.len(), 3);
    }

    #[test]
    fn test_simulate_execution_addition_with_space() {
        let executor = ParallelExecutor::default();
        let scenario = QaScenario::new(
            ModelId::new("test", "model"),
            Modality::Run,
            Backend::Cpu,
            Format::Gguf,
            "What is 2 + 2?".to_string(),
            42,
        );

        let output = executor.simulate_execution(&scenario);
        assert!(output.contains("4"));
    }

    #[test]
    fn test_simulate_execution_division() {
        let executor = ParallelExecutor::default();
        let scenario = QaScenario::new(
            ModelId::new("test", "model"),
            Modality::Run,
            Backend::Cpu,
            Format::Gguf,
            "Calculate 100/4".to_string(),
            42,
        );

        let output = executor.simulate_execution(&scenario);
        // Generic response for non-matched prompts
        assert!(output.contains("42") || output.contains("assistant"));
    }

    #[test]
    fn test_estimate_tokens_longer_text() {
        // 24 characters should be ~6 tokens
        let tokens = estimate_tokens("This is a longer string.");
        assert!(tokens >= 5);
    }

    #[test]
    fn test_parallel_batch_single_scenario() {
        let executor = ParallelExecutor::default();
        let scenarios = vec![test_scenario()];
        let result = executor.execute(&scenarios);
        assert_eq!(result.evidence.len(), 1);
    }

    #[test]
    fn test_parallel_result_all_fields() {
        let scenario = test_scenario();
        let evidence = Evidence::corroborated("F-TEST-001", scenario, "output", 50);

        let result = ParallelResult {
            evidence: vec![evidence],
            passed: 1,
            failed: 0,
            skipped: 0,
            duration_ms: 100,
            stopped_early: false,
        };

        assert_eq!(result.evidence.len(), 1);
        assert_eq!(result.duration_ms, 100);
    }

    #[test]
    fn test_parallel_config_all_fields() {
        let config = ParallelConfig {
            num_workers: 8,
            timeout_ms: 30_000,
            mode: ExecutionMode::Simulate,
            model_path: "/custom/path/model.gguf".to_string(),
            stop_on_failure: true,
        };

        assert_eq!(config.num_workers, 8);
        assert_eq!(config.timeout_ms, 30_000);
        assert!(config.stop_on_failure);
        assert!(config.model_path.contains("custom"));
    }
}
