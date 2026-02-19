
// Builder helpers that eliminate repetitive passed/failed branching.

impl PerformanceCheckResult {
    fn check(gate: SpecGate, passed: bool, measured: f64, threshold: f64, pass_desc: String, fail_desc: String) -> Self {
        Self {
            gate_id: gate.id().to_string(),
            passed,
            measured,
            threshold,
            description: if passed { pass_desc } else { fail_desc },
        }
    }
}

impl ParityCheckResult {
    fn check(gate: SpecGate, passed: bool, max_diff: f64, threshold: f64, pass_desc: String, fail_desc: String) -> Self {
        Self {
            gate_id: gate.id().to_string(),
            passed,
            max_diff,
            threshold,
            description: if passed { pass_desc } else { fail_desc },
        }
    }
}

impl IntegrityCheckResult {
    fn check(gate: SpecGate, passed: bool, description: String, evidence: Option<String>) -> Self {
        Self {
            gate_id: gate.id().to_string(),
            passed,
            description,
            evidence,
        }
    }
}

/// Performance validator
pub struct PerformanceValidator;

impl PerformanceValidator {
    /// F-PERF-001: Check minimum TPS
    #[must_use]
    pub fn check_tps(measured_tps: f64, threshold: f64) -> PerformanceCheckResult {
        let p = measured_tps >= threshold;
        PerformanceCheckResult::check(
            SpecGate::PerfMinimumTps, p, measured_tps, threshold,
            format!("TPS {measured_tps:.1} >= {threshold:.1}"),
            format!("TPS {measured_tps:.1} < {threshold:.1} minimum"),
        )
    }

    /// F-PERF-002: Check time to first token
    #[must_use]
    pub fn check_ttft(ttft_ms: u64, max_ttft_ms: u64) -> PerformanceCheckResult {
        let p = ttft_ms <= max_ttft_ms;
        PerformanceCheckResult::check(
            SpecGate::PerfTtft, p, ttft_ms as f64, max_ttft_ms as f64,
            format!("TTFT {ttft_ms}ms <= {max_ttft_ms}ms"),
            format!("TTFT {ttft_ms}ms > {max_ttft_ms}ms maximum"),
        )
    }

    /// F-PERF-003: Check memory leak (RSS growth over N requests)
    #[must_use]
    pub fn check_memory_leak(
        initial_rss_mb: f64,
        final_rss_mb: f64,
        max_growth_percent: f64,
    ) -> PerformanceCheckResult {
        let growth = if initial_rss_mb > 0.0 {
            ((final_rss_mb - initial_rss_mb) / initial_rss_mb) * 100.0
        } else {
            0.0
        };
        let p = growth <= max_growth_percent;
        PerformanceCheckResult::check(
            SpecGate::PerfMemoryLeak, p, growth, max_growth_percent,
            format!("Memory growth {growth:.1}% <= {max_growth_percent}%"),
            format!("Memory leak: {growth:.1}% > {max_growth_percent}% threshold"),
        )
    }

    /// F-PERF-004: Check GPU utilization
    #[must_use]
    pub fn check_gpu_utilization(utilization: f64, min_utilization: f64) -> PerformanceCheckResult {
        let p = utilization >= min_utilization;
        PerformanceCheckResult::check(
            SpecGate::PerfGpuUtilization, p, utilization, min_utilization,
            format!("GPU utilization {utilization:.1}% >= {min_utilization}%"),
            format!("GPU utilization {utilization:.1}% < {min_utilization}% minimum"),
        )
    }
}

// ============================================================================
// CROSS-PLATFORM PARITY (F-PAR-001..003)
// ============================================================================

/// Result of parity check
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParityCheckResult {
    /// Gate ID
    pub gate_id: String,
    /// Whether check passed
    pub passed: bool,
    /// Maximum difference found
    pub max_diff: f64,
    /// Threshold for difference
    pub threshold: f64,
    /// Description
    pub description: String,
}

/// Cross-platform parity checker
pub struct ParityChecker;

impl ParityChecker {
    /// F-PAR-001: Check CPU/GPU equivalence
    #[must_use]
    pub fn check_cpu_gpu_equivalence(
        cpu_output: &[f32],
        gpu_output: &[f32],
        epsilon: f64,
    ) -> ParityCheckResult {
        let max_diff = cpu_output
            .iter()
            .zip(gpu_output.iter())
            .map(|(a, b)| f64::from((a - b).abs()))
            .fold(0.0f64, f64::max);
        let p = max_diff <= epsilon;
        ParityCheckResult::check(
            SpecGate::ParCpuGpuEquivalence, p, max_diff, epsilon,
            format!("CPU/GPU diff {max_diff:.2e} <= {epsilon:.2e}"),
            format!("CPU/GPU mismatch: {max_diff:.2e} > {epsilon:.2e}"),
        )
    }

    /// F-PAR-002: Check format parity (GGUF vs SafeTensors)
    #[must_use]
    pub fn check_format_parity(
        gguf_tokens: &[u32],
        safetensors_tokens: &[u32],
    ) -> ParityCheckResult {
        let diff_count = gguf_tokens
            .iter()
            .zip(safetensors_tokens.iter())
            .filter(|(a, b)| a != b)
            .count();
        let p = gguf_tokens == safetensors_tokens;
        ParityCheckResult::check(
            SpecGate::ParFormatParity, p, diff_count as f64, 0.0,
            "GGUF/SafeTensors output identical".to_string(),
            format!("{diff_count} token differences found"),
        )
    }

    /// F-PAR-003: Check quantization impact on perplexity
    #[must_use]
    pub fn check_quantization_impact(
        f16_perplexity: f64,
        quantized_perplexity: f64,
        max_degradation_percent: f64,
    ) -> ParityCheckResult {
        let degradation = if f16_perplexity > 0.0 {
            ((quantized_perplexity - f16_perplexity) / f16_perplexity) * 100.0
        } else {
            0.0
        };
        let p = degradation <= max_degradation_percent;
        ParityCheckResult::check(
            SpecGate::ParQuantizationImpact, p, degradation, max_degradation_percent,
            format!("Perplexity degradation {degradation:.1}% <= {max_degradation_percent}%"),
            format!("Perplexity degradation {degradation:.1}% > {max_degradation_percent}% max"),
        )
    }
}

// ============================================================================
// FUNDAMENTAL INTEGRITY CHECKS (F-INT-001..005)
// ============================================================================

/// Result of integrity check
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrityCheckResult {
    /// Gate ID
    pub gate_id: String,
    /// Whether check passed
    pub passed: bool,
    /// Description
    pub description: String,
    /// Evidence/details
    pub evidence: Option<String>,
}

/// Fundamental integrity checker
pub struct IntegrityChecker;

/// Signal-based error classification for memory safety checks.
fn classify_signal_error(segfault: bool, bus_error: bool, abort: bool) -> &'static str {
    if segfault { "Segmentation fault detected" }
    else if bus_error { "Bus error detected" }
    else if abort { "Abort signal detected" }
    else { "Memory safety violation in stderr" }
}

impl IntegrityChecker {
    /// F-INT-001: Check for memory safety violations
    #[must_use]
    pub fn check_memory_safety(exit_signal: Option<i32>, stderr: &str) -> IntegrityCheckResult {
        // SIGSEGV = 11, SIGBUS = 7, SIGABRT = 6
        let segfault = exit_signal == Some(11) || exit_signal == Some(139); // 139 = 128 + 11
        let bus_error = exit_signal == Some(7) || exit_signal == Some(135);
        let abort = exit_signal == Some(6) || exit_signal == Some(134);
        let stderr_bad = stderr.contains("SIGSEGV")
            || stderr.contains("Segmentation fault")
            || stderr.contains("buffer overflow")
            || stderr.contains("stack smashing");

        let passed = !segfault && !bus_error && !abort && !stderr_bad;
        let desc = if passed {
            "No memory safety violations".to_string()
        } else {
            classify_signal_error(segfault, bus_error, abort).to_string()
        };
        let evidence = if passed { None } else { Some(format!("Signal: {exit_signal:?}")) };
        IntegrityCheckResult::check(SpecGate::IntMemorySafety, passed, desc, evidence)
    }

    /// F-INT-002: Check process termination
    #[must_use]
    pub fn check_process_termination(
        exit_code: Option<i32>,
        timed_out: bool,
        has_output: bool,
    ) -> IntegrityCheckResult {
        let clean_exit = exit_code == Some(0) && has_output;
        let error_exit = exit_code.is_some() && exit_code != Some(0);
        let passed = !timed_out && (clean_exit || (error_exit && has_output));

        let desc = if timed_out {
            "Process timed out (hang detected)"
        } else if exit_code.is_none() {
            "Zombie process (no exit code)"
        } else if exit_code != Some(0) && !has_output {
            "Unclean exit without error output"
        } else {
            "Clean process termination"
        };
        IntegrityCheckResult::check(
            SpecGate::IntProcessTermination, passed, desc.to_string(),
            exit_code.map(|c| format!("Exit code: {c}")),
        )
    }

    /// F-INT-003: Check tensor validity (delegates to PatternDetector)
    #[must_use]
    pub fn check_tensor_validity(values: &[f32]) -> IntegrityCheckResult {
        let result = PatternDetector::new().check_tensor_validity(values);
        let desc = if result.is_valid {
            "Tensor values valid".to_string()
        } else if result.nan_count > 0 {
            format!("Found {} NaN values", result.nan_count)
        } else if result.inf_count > 0 {
            format!("Found {} Inf values", result.inf_count)
        } else {
            "Tensor validation failed".to_string()
        };
        IntegrityCheckResult::check(
            SpecGate::IntTensorValidity, result.is_valid, desc,
            Some(format!("NaN: {}, Inf: {}, Mean: {:.4}", result.nan_count, result.inf_count, result.mean)),
        )
    }

    /// F-INT-004: Check format fidelity (round-trip)
    #[must_use]
    pub fn check_format_fidelity(original_hash: &str, roundtrip_hash: &str) -> IntegrityCheckResult {
        let p = original_hash == roundtrip_hash;
        let evidence = if p { None } else {
            Some(format!(
                "Original: {}, After: {}",
                &original_hash[..8.min(original_hash.len())],
                &roundtrip_hash[..8.min(roundtrip_hash.len())]
            ))
        };
        IntegrityCheckResult::check(
            SpecGate::IntFormatFidelity, p,
            if p { "Round-trip conversion bitwise identical" } else { "Round-trip conversion altered weights" }.to_string(),
            evidence,
        )
    }

    /// F-INT-005: Check determinism (same seed = same output)
    #[must_use]
    pub fn check_determinism(run1_output: &str, run2_output: &str, seed: u64) -> IntegrityCheckResult {
        let p = run1_output == run2_output;
        let evidence = if p { None } else {
            let diff_pos = run1_output.chars().zip(run2_output.chars())
                .position(|(a, b)| a != b)
                .unwrap_or_else(|| run1_output.len().min(run2_output.len()));
            Some(format!("First difference at position {diff_pos}"))
        };
        IntegrityCheckResult::check(
            SpecGate::IntDeterminism, p,
            if p { format!("Deterministic output with seed {seed}") } else { format!("Non-deterministic output with seed {seed}") },
            evidence,
        )
    }
}

/// Result of tensor validity check
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorValidityResult {
    /// Number of NaN values
    pub nan_count: usize,
    /// Number of Inf values
    pub inf_count: usize,
    /// Number of zero values
    pub zero_count: usize,
    /// Total number of values
    pub total: usize,
    /// Mean value
    pub mean: f64,
    /// Whether tensor is valid
    pub is_valid: bool,
}

/// Result of companion file check
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompanionCheckResult {
    /// Missing companion files
    pub missing: Vec<String>,
    /// Found companion files
    pub found: Vec<String>,
    /// Whether all companions are present
    pub all_present: bool,
}

/// A path safety violation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathViolation {
    /// The dangerous pattern found
    pub pattern: String,
    /// Description of the risk
    pub description: String,
}

/// Result of path safety check
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathSafetyResult {
    /// Whether path is safe
    pub is_safe: bool,
    /// Violations found
    pub violations: Vec<PathViolation>,
}

/// A dangerous prompt pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptPattern {
    /// The pattern found
    pub pattern: String,
    /// Description of the risk
    pub description: String,
}

/// Result of prompt safety check
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptSafetyResult {
    /// Whether prompt is safe
    pub is_safe: bool,
    /// Dangerous patterns found
    pub found_patterns: Vec<PromptPattern>,
}

#[cfg(test)]
#[path = "patterns_tests.rs"]
mod tests;
