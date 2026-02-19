
/// Performance validator
pub struct PerformanceValidator;

impl PerformanceValidator {
    /// F-PERF-001: Check minimum TPS
    #[must_use]
    pub fn check_tps(measured_tps: f64, threshold: f64) -> PerformanceCheckResult {
        let passed = measured_tps >= threshold;
        PerformanceCheckResult {
            gate_id: SpecGate::PerfMinimumTps.id().to_string(),
            passed,
            measured: measured_tps,
            threshold,
            description: if passed {
                format!("TPS {measured_tps:.1} >= {threshold:.1}")
            } else {
                format!("TPS {measured_tps:.1} < {threshold:.1} minimum")
            },
        }
    }

    /// F-PERF-002: Check time to first token
    #[must_use]
    pub fn check_ttft(ttft_ms: u64, max_ttft_ms: u64) -> PerformanceCheckResult {
        let passed = ttft_ms <= max_ttft_ms;
        PerformanceCheckResult {
            gate_id: SpecGate::PerfTtft.id().to_string(),
            passed,
            measured: ttft_ms as f64,
            threshold: max_ttft_ms as f64,
            description: if passed {
                format!("TTFT {ttft_ms}ms <= {max_ttft_ms}ms")
            } else {
                format!("TTFT {ttft_ms}ms > {max_ttft_ms}ms maximum")
            },
        }
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
        let passed = growth <= max_growth_percent;
        PerformanceCheckResult {
            gate_id: SpecGate::PerfMemoryLeak.id().to_string(),
            passed,
            measured: growth,
            threshold: max_growth_percent,
            description: if passed {
                format!("Memory growth {growth:.1}% <= {max_growth_percent}%")
            } else {
                format!("Memory leak: {growth:.1}% > {max_growth_percent}% threshold")
            },
        }
    }

    /// F-PERF-004: Check GPU utilization
    #[must_use]
    pub fn check_gpu_utilization(utilization: f64, min_utilization: f64) -> PerformanceCheckResult {
        let passed = utilization >= min_utilization;
        PerformanceCheckResult {
            gate_id: SpecGate::PerfGpuUtilization.id().to_string(),
            passed,
            measured: utilization,
            threshold: min_utilization,
            description: if passed {
                format!("GPU utilization {utilization:.1}% >= {min_utilization}%")
            } else {
                format!("GPU utilization {utilization:.1}% < {min_utilization}% minimum")
            },
        }
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
        let passed = max_diff <= epsilon;
        ParityCheckResult {
            gate_id: SpecGate::ParCpuGpuEquivalence.id().to_string(),
            passed,
            max_diff,
            threshold: epsilon,
            description: if passed {
                format!("CPU/GPU diff {max_diff:.2e} <= {epsilon:.2e}")
            } else {
                format!("CPU/GPU mismatch: {max_diff:.2e} > {epsilon:.2e}")
            },
        }
    }

    /// F-PAR-002: Check format parity (GGUF vs SafeTensors)
    #[must_use]
    pub fn check_format_parity(
        gguf_tokens: &[u32],
        safetensors_tokens: &[u32],
    ) -> ParityCheckResult {
        let passed = gguf_tokens == safetensors_tokens;
        let diff_count = gguf_tokens
            .iter()
            .zip(safetensors_tokens.iter())
            .filter(|(a, b)| a != b)
            .count();
        ParityCheckResult {
            gate_id: SpecGate::ParFormatParity.id().to_string(),
            passed,
            max_diff: diff_count as f64,
            threshold: 0.0,
            description: if passed {
                "GGUF/SafeTensors output identical".to_string()
            } else {
                format!("{diff_count} token differences found")
            },
        }
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
        let passed = degradation <= max_degradation_percent;
        ParityCheckResult {
            gate_id: SpecGate::ParQuantizationImpact.id().to_string(),
            passed,
            max_diff: degradation,
            threshold: max_degradation_percent,
            description: if passed {
                format!("Perplexity degradation {degradation:.1}% <= {max_degradation_percent}%")
            } else {
                format!("Perplexity degradation {degradation:.1}% > {max_degradation_percent}% max")
            },
        }
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

impl IntegrityChecker {
    /// F-INT-001: Check for memory safety violations
    /// Returns true if no unsafe memory access detected
    #[must_use]
    pub fn check_memory_safety(exit_signal: Option<i32>, stderr: &str) -> IntegrityCheckResult {
        // SIGSEGV = 11, SIGBUS = 7, SIGABRT = 6
        let segfault = exit_signal == Some(11) || exit_signal == Some(139); // 139 = 128 + 11
        let bus_error = exit_signal == Some(7) || exit_signal == Some(135);
        let abort = exit_signal == Some(6) || exit_signal == Some(134);
        let stderr_indicators = stderr.contains("SIGSEGV")
            || stderr.contains("Segmentation fault")
            || stderr.contains("buffer overflow")
            || stderr.contains("stack smashing");

        let passed = !segfault && !bus_error && !abort && !stderr_indicators;
        IntegrityCheckResult {
            gate_id: SpecGate::IntMemorySafety.id().to_string(),
            passed,
            description: if passed {
                "No memory safety violations".to_string()
            } else if segfault {
                "Segmentation fault detected".to_string()
            } else if bus_error {
                "Bus error detected".to_string()
            } else if abort {
                "Abort signal detected".to_string()
            } else {
                "Memory safety violation in stderr".to_string()
            },
            evidence: if passed {
                None
            } else {
                Some(format!("Signal: {exit_signal:?}"))
            },
        }
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
        let passed = clean_exit || (error_exit && has_output);

        IntegrityCheckResult {
            gate_id: SpecGate::IntProcessTermination.id().to_string(),
            passed: !timed_out && passed,
            description: if timed_out {
                "Process timed out (hang detected)".to_string()
            } else if exit_code.is_none() {
                "Zombie process (no exit code)".to_string()
            } else if exit_code != Some(0) && !has_output {
                "Unclean exit without error output".to_string()
            } else {
                "Clean process termination".to_string()
            },
            evidence: exit_code.map(|c| format!("Exit code: {c}")),
        }
    }

    /// F-INT-003: Check tensor validity (delegates to PatternDetector)
    #[must_use]
    pub fn check_tensor_validity(values: &[f32]) -> IntegrityCheckResult {
        let detector = PatternDetector::new();
        let result = detector.check_tensor_validity(values);
        IntegrityCheckResult {
            gate_id: SpecGate::IntTensorValidity.id().to_string(),
            passed: result.is_valid,
            description: if result.is_valid {
                "Tensor values valid".to_string()
            } else if result.nan_count > 0 {
                format!("Found {} NaN values", result.nan_count)
            } else if result.inf_count > 0 {
                format!("Found {} Inf values", result.inf_count)
            } else {
                "Tensor validation failed".to_string()
            },
            evidence: Some(format!(
                "NaN: {}, Inf: {}, Mean: {:.4}",
                result.nan_count, result.inf_count, result.mean
            )),
        }
    }

    /// F-INT-004: Check format fidelity (round-trip)
    #[must_use]
    pub fn check_format_fidelity(
        original_hash: &str,
        roundtrip_hash: &str,
    ) -> IntegrityCheckResult {
        let passed = original_hash == roundtrip_hash;
        IntegrityCheckResult {
            gate_id: SpecGate::IntFormatFidelity.id().to_string(),
            passed,
            description: if passed {
                "Round-trip conversion bitwise identical".to_string()
            } else {
                "Round-trip conversion altered weights".to_string()
            },
            evidence: if passed {
                None
            } else {
                Some(format!(
                    "Original: {}, After: {}",
                    &original_hash[..8.min(original_hash.len())],
                    &roundtrip_hash[..8.min(roundtrip_hash.len())]
                ))
            },
        }
    }

    /// F-INT-005: Check determinism (same seed = same output)
    #[must_use]
    pub fn check_determinism(
        run1_output: &str,
        run2_output: &str,
        seed: u64,
    ) -> IntegrityCheckResult {
        let passed = run1_output == run2_output;
        IntegrityCheckResult {
            gate_id: SpecGate::IntDeterminism.id().to_string(),
            passed,
            description: if passed {
                format!("Deterministic output with seed {seed}")
            } else {
                format!("Non-deterministic output with seed {seed}")
            },
            evidence: if passed {
                None
            } else {
                let diff_pos = run1_output
                    .chars()
                    .zip(run2_output.chars())
                    .position(|(a, b)| a != b)
                    .unwrap_or_else(|| run1_output.len().min(run2_output.len()));
                Some(format!("First difference at position {diff_pos}"))
            },
        }
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
