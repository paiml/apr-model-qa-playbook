//! Differential Testing (GH-188, PMAT-114, PMAT-192)
//!
//! Implements differential testing capabilities:
//! - Tensor diff between models (rosetta diff-tensors)
//! - Inference comparison (rosetta compare-inference)
//! - Performance benchmarking (profile --diff-benchmark)
//! - Trace payload comparison (trace --payload --reference)
//!
//! # Toyota Way Principle
//!
//! "Genchi Genbutsu" (Go and see) - Don't trust that two implementations
//! are equivalent; verify by running both and comparing outputs.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::process::Command;

/// Result of tensor diff operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorDiffResult {
    /// Total tensors compared
    pub total_tensors: usize,
    /// Tensors with shape mismatches
    pub mismatched_tensors: usize,
    /// Tensors with transposed dimensions (GGML vs standard)
    pub transposed_tensors: usize,
    /// Details of each mismatch
    pub mismatches: Vec<TensorMismatch>,
    /// Whether the diff passed (no critical mismatches)
    pub passed: bool,
}

/// A single tensor mismatch
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorMismatch {
    /// Tensor name
    pub name: String,
    /// Shape in model A
    pub shape_a: Vec<usize>,
    /// Shape in model B
    pub shape_b: Vec<usize>,
    /// Type of mismatch
    pub mismatch_type: TensorMismatchType,
}

/// Type of tensor mismatch
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TensorMismatchType {
    /// Dimensions are transposed (e.g., [4096, 32000] vs [32000, 4096])
    Transposed,
    /// Dimensions are completely different
    ShapeMismatch,
    /// Tensor missing in one model
    Missing,
}

impl TensorMismatchType {
    /// Get the gate ID for this mismatch type
    #[must_use]
    #[allow(clippy::match_same_arms)] // ShapeMismatch and Missing share the same gate intentionally
    pub fn gate_id(&self) -> &'static str {
        match self {
            Self::Transposed => "F-ROSETTA-DIFF-001",
            Self::ShapeMismatch => "F-ROSETTA-DIFF-002",
            Self::Missing => "F-ROSETTA-DIFF-002",
        }
    }
}

/// Configuration for differential testing
#[derive(Debug, Clone)]
pub struct DiffConfig {
    /// Path to APR CLI binary
    pub apr_binary: String,
    /// Filter pattern for tensor names
    pub filter: Option<String>,
    /// Only show mismatches
    pub mismatches_only: bool,
    /// Tolerance for numerical comparisons
    pub tolerance: f64,
}

impl Default for DiffConfig {
    fn default() -> Self {
        Self {
            apr_binary: "apr".to_string(),
            filter: None,
            mismatches_only: true,
            tolerance: 1e-5,
        }
    }
}

/// Differential test executor
pub struct DifferentialExecutor {
    config: DiffConfig,
}

impl DifferentialExecutor {
    /// Create a new differential executor
    #[must_use]
    pub fn new(config: DiffConfig) -> Self {
        Self { config }
    }

    /// Run tensor diff between two models
    ///
    /// Uses `apr rosetta diff-tensors` to compare tensor layouts.
    ///
    /// # Errors
    ///
    /// Returns an error if the apr command fails to execute or returns non-zero.
    pub fn diff_tensors(&self, model_a: &Path, model_b: &Path) -> Result<TensorDiffResult> {
        let mut cmd = Command::new(&self.config.apr_binary);
        cmd.arg("rosetta")
            .arg("diff-tensors")
            .arg(model_a)
            .arg(model_b)
            .arg("--json");

        if self.config.mismatches_only {
            cmd.arg("--mismatches-only");
        }

        if let Some(filter) = &self.config.filter {
            cmd.arg("--filter").arg(filter);
        }

        let output = cmd.output().map_err(|e| Error::ExecutionFailed {
            command: format!("{cmd:?}"),
            reason: e.to_string(),
        })?;

        if !output.status.success() {
            // Try to parse error from stderr
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(Error::ExecutionFailed {
                command: "apr rosetta diff-tensors".to_string(),
                reason: stderr.to_string(),
            });
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        self.parse_diff_output(&stdout)
    }

    /// Parse diff-tensors JSON output
    fn parse_diff_output(&self, output: &str) -> Result<TensorDiffResult> {
        // Try to parse as JSON first
        if let Ok(result) = serde_json::from_str::<TensorDiffResult>(output) {
            return Ok(result);
        }

        // Fall back to parsing text output
        let mut mismatches = Vec::new();
        let mut transposed_count = 0;

        for line in output.lines() {
            if line.contains("TRANSPOSED") || line.contains("⚠️") {
                // Parse tensor name and shapes from line
                // Format: "tensor_name: [a, b] vs [b, a] ⚠️ TRANSPOSED"
                if let Some((name, _shapes)) = line.split_once(':') {
                    let name = name.trim().to_string();
                    // Extract shapes (simplified parsing)
                    let mismatch = TensorMismatch {
                        name,
                        shape_a: vec![],
                        shape_b: vec![],
                        mismatch_type: TensorMismatchType::Transposed,
                    };
                    mismatches.push(mismatch);
                    transposed_count += 1;
                }
            }
        }

        Ok(TensorDiffResult {
            total_tensors: 0, // Not available from text output
            mismatched_tensors: mismatches.len(),
            transposed_tensors: transposed_count,
            passed: mismatches.is_empty(),
            mismatches,
        })
    }

    /// Compare inference between two models token-by-token
    ///
    /// Uses `apr rosetta compare-inference` to verify output equivalence.
    ///
    /// # Errors
    ///
    /// Returns an error if the apr command fails to execute.
    pub fn compare_inference(
        &self,
        model_a: &Path,
        model_b: &Path,
        prompt: &str,
        max_tokens: usize,
    ) -> Result<InferenceComparisonResult> {
        let output = Command::new(&self.config.apr_binary)
            .arg("rosetta")
            .arg("compare-inference")
            .arg(model_a)
            .arg(model_b)
            .arg("--prompt")
            .arg(prompt)
            .arg("--max-tokens")
            .arg(max_tokens.to_string())
            .arg("--tolerance")
            .arg(self.config.tolerance.to_string())
            .arg("--json")
            .output()
            .map_err(|e| Error::ExecutionFailed {
                command: "apr rosetta compare-inference".to_string(),
                reason: e.to_string(),
            })?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        self.parse_inference_output(&stdout, output.status.success())
    }

    /// Parse compare-inference output
    fn parse_inference_output(
        &self,
        output: &str,
        success: bool,
    ) -> Result<InferenceComparisonResult> {
        // Try JSON parsing first
        if let Ok(result) = serde_json::from_str::<InferenceComparisonResult>(output) {
            return Ok(result);
        }

        // Fall back to basic result
        Ok(InferenceComparisonResult {
            total_tokens: 0,
            matching_tokens: 0,
            max_logit_diff: 0.0,
            passed: success,
            token_comparisons: vec![],
        })
    }
}

/// Result of inference comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceComparisonResult {
    /// Total tokens compared
    pub total_tokens: usize,
    /// Tokens with matching argmax
    pub matching_tokens: usize,
    /// Maximum logit difference observed
    pub max_logit_diff: f64,
    /// Whether comparison passed
    pub passed: bool,
    /// Per-token comparison details
    pub token_comparisons: Vec<TokenComparison>,
}

/// Comparison of a single token
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenComparison {
    /// Token index
    pub index: usize,
    /// Token ID from model A
    pub token_a: u32,
    /// Token ID from model B
    pub token_b: u32,
    /// Logit difference
    pub logit_diff: f64,
    /// Whether tokens match
    pub matches: bool,
}

/// Result of differential benchmark
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffBenchmarkResult {
    /// Model A metrics
    pub model_a: BenchmarkMetrics,
    /// Model B metrics
    pub model_b: BenchmarkMetrics,
    /// Throughput delta percentage
    pub throughput_delta_pct: f64,
    /// Latency delta percentage (p50)
    pub latency_p50_delta_pct: f64,
    /// Latency delta percentage (p99)
    pub latency_p99_delta_pct: f64,
    /// Whether regression detected
    pub regression_detected: bool,
    /// Regression threshold used
    pub regression_threshold: f64,
}

/// Benchmark metrics for a single model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkMetrics {
    /// Model path
    pub path: String,
    /// Throughput in tokens/second
    pub throughput_tps: f64,
    /// p50 latency in milliseconds
    pub latency_p50_ms: f64,
    /// p99 latency in milliseconds
    pub latency_p99_ms: f64,
}

/// CI profile assertions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CiProfileResult {
    /// Throughput achieved
    pub throughput_tps: f64,
    /// p50 latency
    pub latency_p50_ms: f64,
    /// p99 latency
    pub latency_p99_ms: f64,
    /// Assertion results
    pub assertions: Vec<CiAssertion>,
    /// Overall pass/fail
    pub passed: bool,
}

/// A single CI assertion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CiAssertion {
    /// Assertion name
    pub name: String,
    /// Expected value (threshold)
    pub expected: String,
    /// Actual value
    pub actual: String,
    /// Whether assertion passed
    pub passed: bool,
    /// Gate ID
    pub gate_id: String,
}

/// Execute profile CI mode
///
/// Runs `apr profile --ci` with optional assertion flags.
///
/// # Errors
///
/// Returns an error if the apr command fails to execute.
pub fn run_profile_ci(
    apr_binary: &str,
    model_path: &Path,
    min_throughput: Option<f64>,
    max_p99: Option<f64>,
    max_p50: Option<f64>,
    warmup: usize,
    measure: usize,
) -> Result<CiProfileResult> {
    let mut cmd = Command::new(apr_binary);
    cmd.arg("profile").arg(model_path).arg("--ci");

    if let Some(throughput) = min_throughput {
        cmd.arg("--assert-throughput").arg(throughput.to_string());
    }
    if let Some(p99) = max_p99 {
        cmd.arg("--assert-p99").arg(p99.to_string());
    }
    if let Some(p50) = max_p50 {
        cmd.arg("--assert-p50").arg(p50.to_string());
    }

    cmd.arg("--warmup").arg(warmup.to_string());
    cmd.arg("--measure").arg(measure.to_string());
    cmd.arg("--json");

    let output = cmd.output().map_err(|e| Error::ExecutionFailed {
        command: "apr profile --ci".to_string(),
        reason: e.to_string(),
    })?;

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Try JSON parsing
    if let Ok(result) = serde_json::from_str::<CiProfileResult>(&stdout) {
        return Ok(result);
    }

    // Fall back to basic result based on exit code
    Ok(CiProfileResult {
        throughput_tps: 0.0,
        latency_p50_ms: 0.0,
        latency_p99_ms: 0.0,
        assertions: vec![],
        passed: output.status.success(),
    })
}

/// Execute differential benchmark
///
/// Compares performance between two models to detect regressions.
///
/// # Errors
///
/// Returns an error if the apr command fails or output cannot be parsed.
pub fn run_diff_benchmark(
    apr_binary: &str,
    model_a: &Path,
    model_b: &Path,
    regression_threshold: f64,
) -> Result<DiffBenchmarkResult> {
    let output = Command::new(apr_binary)
        .arg("profile")
        .arg(model_a)
        .arg(model_b)
        .arg("--diff-benchmark")
        .arg("--regression-threshold")
        .arg(regression_threshold.to_string())
        .arg("--json")
        .output()
        .map_err(|e| Error::ExecutionFailed {
            command: "apr profile --diff-benchmark".to_string(),
            reason: e.to_string(),
        })?;

    let stdout = String::from_utf8_lossy(&output.stdout);

    if let Ok(result) = serde_json::from_str::<DiffBenchmarkResult>(&stdout) {
        return Ok(result);
    }

    Err(Error::ExecutionFailed {
        command: "apr profile --diff-benchmark".to_string(),
        reason: "Failed to parse output".to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diff_config_default() {
        let config = DiffConfig::default();
        assert_eq!(config.apr_binary, "apr");
        assert!(config.mismatches_only);
        assert!((config.tolerance - 1e-5).abs() < 1e-10);
    }

    #[test]
    fn test_tensor_mismatch_type_gate_id() {
        assert_eq!(
            TensorMismatchType::Transposed.gate_id(),
            "F-ROSETTA-DIFF-001"
        );
        assert_eq!(
            TensorMismatchType::ShapeMismatch.gate_id(),
            "F-ROSETTA-DIFF-002"
        );
        assert_eq!(TensorMismatchType::Missing.gate_id(), "F-ROSETTA-DIFF-002");
    }

    #[test]
    fn test_tensor_diff_result_passed() {
        let result = TensorDiffResult {
            total_tensors: 100,
            mismatched_tensors: 0,
            transposed_tensors: 0,
            mismatches: vec![],
            passed: true,
        };
        assert!(result.passed);
    }

    #[test]
    fn test_tensor_diff_result_failed() {
        let result = TensorDiffResult {
            total_tensors: 100,
            mismatched_tensors: 2,
            transposed_tensors: 2,
            mismatches: vec![TensorMismatch {
                name: "token_embd.weight".to_string(),
                shape_a: vec![4096, 32000],
                shape_b: vec![32000, 4096],
                mismatch_type: TensorMismatchType::Transposed,
            }],
            passed: false,
        };
        assert!(!result.passed);
        assert_eq!(result.transposed_tensors, 2);
    }

    #[test]
    fn test_inference_comparison_passed() {
        let result = InferenceComparisonResult {
            total_tokens: 10,
            matching_tokens: 10,
            max_logit_diff: 1e-6,
            passed: true,
            token_comparisons: vec![],
        };
        assert!(result.passed);
        assert_eq!(result.matching_tokens, result.total_tokens);
    }

    #[test]
    fn test_ci_profile_assertions() {
        let result = CiProfileResult {
            throughput_tps: 12.8,
            latency_p50_ms: 78.2,
            latency_p99_ms: 156.5,
            assertions: vec![
                CiAssertion {
                    name: "throughput".to_string(),
                    expected: ">= 10.0 tok/s".to_string(),
                    actual: "12.8 tok/s".to_string(),
                    passed: true,
                    gate_id: "F-PROFILE-CI-001".to_string(),
                },
                CiAssertion {
                    name: "p99_latency".to_string(),
                    expected: "<= 200 ms".to_string(),
                    actual: "156.5 ms".to_string(),
                    passed: true,
                    gate_id: "F-PROFILE-CI-002".to_string(),
                },
            ],
            passed: true,
        };
        assert!(result.passed);
        assert!(result.assertions.iter().all(|a| a.passed));
    }

    #[test]
    fn test_diff_benchmark_no_regression() {
        let result = DiffBenchmarkResult {
            model_a: BenchmarkMetrics {
                path: "model_a.gguf".to_string(),
                throughput_tps: 12.3,
                latency_p50_ms: 78.2,
                latency_p99_ms: 156.5,
            },
            model_b: BenchmarkMetrics {
                path: "model_b.gguf".to_string(),
                throughput_tps: 12.5, // Slight improvement
                latency_p50_ms: 76.1,
                latency_p99_ms: 152.3,
            },
            throughput_delta_pct: 1.6,
            latency_p50_delta_pct: -2.7,
            latency_p99_delta_pct: -2.7,
            regression_detected: false,
            regression_threshold: 5.0,
        };
        assert!(!result.regression_detected);
    }

    #[test]
    fn test_diff_benchmark_with_regression() {
        let result = DiffBenchmarkResult {
            model_a: BenchmarkMetrics {
                path: "model_a.gguf".to_string(),
                throughput_tps: 12.3,
                latency_p50_ms: 78.2,
                latency_p99_ms: 156.5,
            },
            model_b: BenchmarkMetrics {
                path: "model_b.gguf".to_string(),
                throughput_tps: 11.0, // 10.6% regression
                latency_p50_ms: 88.0,
                latency_p99_ms: 180.0,
            },
            throughput_delta_pct: -10.6,
            latency_p50_delta_pct: 12.5,
            latency_p99_delta_pct: 15.0,
            regression_detected: true,
            regression_threshold: 5.0,
        };
        assert!(result.regression_detected);
    }

    #[test]
    fn test_differential_executor_new() {
        let config = DiffConfig::default();
        let executor = DifferentialExecutor::new(config);
        assert_eq!(executor.config.apr_binary, "apr");
    }

    #[test]
    fn test_diff_config_with_filter() {
        let config = DiffConfig {
            filter: Some("token_embd".to_string()),
            ..Default::default()
        };
        assert_eq!(config.filter.as_deref(), Some("token_embd"));
    }

    #[test]
    fn test_diff_config_custom_binary() {
        let config = DiffConfig {
            apr_binary: "/custom/path/apr".to_string(),
            ..Default::default()
        };
        assert_eq!(config.apr_binary, "/custom/path/apr");
    }

    #[test]
    fn test_diff_config_custom_tolerance() {
        let config = DiffConfig {
            tolerance: 1e-10,
            ..Default::default()
        };
        assert!((config.tolerance - 1e-10).abs() < 1e-15);
    }

    #[test]
    fn test_diff_config_mismatches_only_false() {
        let config = DiffConfig {
            mismatches_only: false,
            ..Default::default()
        };
        assert!(!config.mismatches_only);
    }

    #[test]
    fn test_tensor_mismatch_clone() {
        let mismatch = TensorMismatch {
            name: "weights.0".to_string(),
            shape_a: vec![100, 200],
            shape_b: vec![200, 100],
            mismatch_type: TensorMismatchType::Transposed,
        };
        let cloned = mismatch.clone();
        assert_eq!(cloned.name, "weights.0");
        assert_eq!(cloned.shape_a, vec![100, 200]);
        assert_eq!(cloned.shape_b, vec![200, 100]);
    }

    #[test]
    fn test_tensor_mismatch_debug() {
        let mismatch = TensorMismatch {
            name: "test".to_string(),
            shape_a: vec![10],
            shape_b: vec![20],
            mismatch_type: TensorMismatchType::ShapeMismatch,
        };
        let debug_str = format!("{mismatch:?}");
        assert!(debug_str.contains("TensorMismatch"));
        assert!(debug_str.contains("test"));
    }

    #[test]
    fn test_tensor_mismatch_type_missing() {
        let mismatch_type = TensorMismatchType::Missing;
        assert_eq!(mismatch_type.gate_id(), "F-ROSETTA-DIFF-002");
    }

    #[test]
    fn test_tensor_mismatch_type_debug() {
        let debug_str = format!("{:?}", TensorMismatchType::Transposed);
        assert!(debug_str.contains("Transposed"));
    }

    #[test]
    fn test_tensor_mismatch_type_eq() {
        assert_eq!(
            TensorMismatchType::Transposed,
            TensorMismatchType::Transposed
        );
        assert_ne!(TensorMismatchType::Transposed, TensorMismatchType::Missing);
    }

    #[test]
    fn test_tensor_mismatch_type_copy() {
        let t = TensorMismatchType::ShapeMismatch;
        let copied: TensorMismatchType = t;
        assert_eq!(copied, TensorMismatchType::ShapeMismatch);
    }

    #[test]
    fn test_tensor_diff_result_clone() {
        let result = TensorDiffResult {
            total_tensors: 10,
            mismatched_tensors: 2,
            transposed_tensors: 1,
            mismatches: vec![],
            passed: false,
        };
        let cloned = result.clone();
        assert_eq!(cloned.total_tensors, 10);
        assert_eq!(cloned.mismatched_tensors, 2);
    }

    #[test]
    fn test_tensor_diff_result_debug() {
        let result = TensorDiffResult {
            total_tensors: 5,
            mismatched_tensors: 0,
            transposed_tensors: 0,
            mismatches: vec![],
            passed: true,
        };
        let debug_str = format!("{result:?}");
        assert!(debug_str.contains("TensorDiffResult"));
    }

    #[test]
    fn test_inference_comparison_result_clone() {
        let result = InferenceComparisonResult {
            total_tokens: 10,
            matching_tokens: 10,
            max_logit_diff: 0.001,
            passed: true,
            token_comparisons: vec![],
        };
        let cloned = result.clone();
        assert_eq!(cloned.total_tokens, 10);
    }

    #[test]
    fn test_inference_comparison_result_debug() {
        let result = InferenceComparisonResult {
            total_tokens: 5,
            matching_tokens: 4,
            max_logit_diff: 0.1,
            passed: false,
            token_comparisons: vec![],
        };
        let debug_str = format!("{result:?}");
        assert!(debug_str.contains("InferenceComparisonResult"));
    }

    #[test]
    fn test_token_comparison_clone() {
        let tc = TokenComparison {
            index: 0,
            token_a: 100,
            token_b: 100,
            logit_diff: 0.0,
            matches: true,
        };
        let cloned = tc.clone();
        assert_eq!(cloned.index, 0);
        assert!(cloned.matches);
    }

    #[test]
    fn test_token_comparison_debug() {
        let tc = TokenComparison {
            index: 5,
            token_a: 42,
            token_b: 43,
            logit_diff: 0.5,
            matches: false,
        };
        let debug_str = format!("{tc:?}");
        assert!(debug_str.contains("TokenComparison"));
    }

    #[test]
    fn test_diff_benchmark_result_clone() {
        let result = DiffBenchmarkResult {
            model_a: BenchmarkMetrics {
                path: "a.gguf".to_string(),
                throughput_tps: 10.0,
                latency_p50_ms: 50.0,
                latency_p99_ms: 100.0,
            },
            model_b: BenchmarkMetrics {
                path: "b.gguf".to_string(),
                throughput_tps: 11.0,
                latency_p50_ms: 48.0,
                latency_p99_ms: 95.0,
            },
            throughput_delta_pct: 10.0,
            latency_p50_delta_pct: -4.0,
            latency_p99_delta_pct: -5.0,
            regression_detected: false,
            regression_threshold: 5.0,
        };
        let cloned = result.clone();
        assert_eq!(cloned.model_a.path, "a.gguf");
    }

    #[test]
    fn test_diff_benchmark_result_debug() {
        let result = DiffBenchmarkResult {
            model_a: BenchmarkMetrics {
                path: "model_a".to_string(),
                throughput_tps: 10.0,
                latency_p50_ms: 50.0,
                latency_p99_ms: 100.0,
            },
            model_b: BenchmarkMetrics {
                path: "model_b".to_string(),
                throughput_tps: 10.0,
                latency_p50_ms: 50.0,
                latency_p99_ms: 100.0,
            },
            throughput_delta_pct: 0.0,
            latency_p50_delta_pct: 0.0,
            latency_p99_delta_pct: 0.0,
            regression_detected: false,
            regression_threshold: 5.0,
        };
        let debug_str = format!("{result:?}");
        assert!(debug_str.contains("DiffBenchmarkResult"));
    }

    #[test]
    fn test_benchmark_metrics_clone() {
        let metrics = BenchmarkMetrics {
            path: "test.gguf".to_string(),
            throughput_tps: 15.5,
            latency_p50_ms: 65.0,
            latency_p99_ms: 130.0,
        };
        let cloned = metrics.clone();
        assert_eq!(cloned.path, "test.gguf");
        assert!((cloned.throughput_tps - 15.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_benchmark_metrics_debug() {
        let metrics = BenchmarkMetrics {
            path: "model.gguf".to_string(),
            throughput_tps: 20.0,
            latency_p50_ms: 40.0,
            latency_p99_ms: 80.0,
        };
        let debug_str = format!("{metrics:?}");
        assert!(debug_str.contains("BenchmarkMetrics"));
    }

    #[test]
    fn test_ci_profile_result_clone() {
        let result = CiProfileResult {
            throughput_tps: 15.0,
            latency_p50_ms: 70.0,
            latency_p99_ms: 140.0,
            assertions: vec![],
            passed: true,
        };
        let cloned = result.clone();
        assert!(cloned.passed);
    }

    #[test]
    fn test_ci_profile_result_debug() {
        let result = CiProfileResult {
            throughput_tps: 10.0,
            latency_p50_ms: 80.0,
            latency_p99_ms: 160.0,
            assertions: vec![],
            passed: false,
        };
        let debug_str = format!("{result:?}");
        assert!(debug_str.contains("CiProfileResult"));
    }

    #[test]
    fn test_ci_assertion_clone() {
        let assertion = CiAssertion {
            name: "throughput".to_string(),
            expected: ">= 10".to_string(),
            actual: "12".to_string(),
            passed: true,
            gate_id: "F-CI-001".to_string(),
        };
        let cloned = assertion.clone();
        assert_eq!(cloned.name, "throughput");
        assert!(cloned.passed);
    }

    #[test]
    fn test_ci_assertion_debug() {
        let assertion = CiAssertion {
            name: "p99".to_string(),
            expected: "<= 200".to_string(),
            actual: "250".to_string(),
            passed: false,
            gate_id: "F-CI-002".to_string(),
        };
        let debug_str = format!("{assertion:?}");
        assert!(debug_str.contains("CiAssertion"));
    }

    #[test]
    fn test_diff_config_clone() {
        let config = DiffConfig::default();
        let cloned = config.clone();
        assert_eq!(cloned.apr_binary, config.apr_binary);
        assert_eq!(cloned.mismatches_only, config.mismatches_only);
    }

    #[test]
    fn test_diff_config_debug() {
        let config = DiffConfig::default();
        let debug_str = format!("{config:?}");
        assert!(debug_str.contains("DiffConfig"));
    }

    #[test]
    fn test_differential_executor_with_custom_config() {
        let config = DiffConfig {
            apr_binary: "custom-apr".to_string(),
            filter: Some("embed".to_string()),
            mismatches_only: false,
            tolerance: 1e-8,
        };
        let executor = DifferentialExecutor::new(config);
        assert_eq!(executor.config.apr_binary, "custom-apr");
        assert_eq!(executor.config.filter.as_deref(), Some("embed"));
        assert!(!executor.config.mismatches_only);
    }

    #[test]
    fn test_parse_diff_output_empty_json() {
        let config = DiffConfig::default();
        let executor = DifferentialExecutor::new(config);
        // Text output with no mismatches
        let output = "All tensors match";
        let result = executor.parse_diff_output(output).unwrap();
        assert!(result.passed);
        assert!(result.mismatches.is_empty());
    }

    #[test]
    fn test_parse_diff_output_with_transposed() {
        let config = DiffConfig::default();
        let executor = DifferentialExecutor::new(config);
        let output = "token_embd.weight: [4096, 32000] vs [32000, 4096] ⚠️ TRANSPOSED\n";
        let result = executor.parse_diff_output(output).unwrap();
        assert!(!result.passed);
        assert_eq!(result.transposed_tensors, 1);
    }

    #[test]
    fn test_parse_diff_output_valid_json() {
        let config = DiffConfig::default();
        let executor = DifferentialExecutor::new(config);
        let json = r#"{"total_tensors":100,"mismatched_tensors":0,"transposed_tensors":0,"mismatches":[],"passed":true}"#;
        let result = executor.parse_diff_output(json).unwrap();
        assert!(result.passed);
        assert_eq!(result.total_tensors, 100);
    }

    #[test]
    fn test_parse_inference_output_success() {
        let config = DiffConfig::default();
        let executor = DifferentialExecutor::new(config);
        let result = executor
            .parse_inference_output("some output", true)
            .unwrap();
        assert!(result.passed);
    }

    #[test]
    fn test_parse_inference_output_failure() {
        let config = DiffConfig::default();
        let executor = DifferentialExecutor::new(config);
        let result = executor.parse_inference_output("error", false).unwrap();
        assert!(!result.passed);
    }

    #[test]
    fn test_parse_inference_output_valid_json() {
        let config = DiffConfig::default();
        let executor = DifferentialExecutor::new(config);
        let json = r#"{"total_tokens":10,"matching_tokens":10,"max_logit_diff":0.0,"passed":true,"token_comparisons":[]}"#;
        let result = executor.parse_inference_output(json, true).unwrap();
        assert!(result.passed);
        assert_eq!(result.total_tokens, 10);
    }

    #[test]
    fn test_tensor_diff_result_with_mismatches() {
        let result = TensorDiffResult {
            total_tensors: 50,
            mismatched_tensors: 3,
            transposed_tensors: 2,
            mismatches: vec![
                TensorMismatch {
                    name: "layer.0.weight".to_string(),
                    shape_a: vec![768, 768],
                    shape_b: vec![768, 768],
                    mismatch_type: TensorMismatchType::ShapeMismatch,
                },
                TensorMismatch {
                    name: "layer.1.weight".to_string(),
                    shape_a: vec![768, 3072],
                    shape_b: vec![3072, 768],
                    mismatch_type: TensorMismatchType::Transposed,
                },
            ],
            passed: false,
        };
        assert!(!result.passed);
        assert_eq!(result.mismatches.len(), 2);
    }

    #[test]
    fn test_inference_comparison_with_token_details() {
        let result = InferenceComparisonResult {
            total_tokens: 5,
            matching_tokens: 4,
            max_logit_diff: 0.05,
            passed: false,
            token_comparisons: vec![
                TokenComparison {
                    index: 0,
                    token_a: 1,
                    token_b: 1,
                    logit_diff: 0.0,
                    matches: true,
                },
                TokenComparison {
                    index: 1,
                    token_a: 5,
                    token_b: 6,
                    logit_diff: 0.05,
                    matches: false,
                },
            ],
        };
        assert!(!result.passed);
        assert_eq!(result.token_comparisons.len(), 2);
        assert!(!result.token_comparisons[1].matches);
    }

    #[test]
    fn test_ci_profile_with_multiple_assertions() {
        let result = CiProfileResult {
            throughput_tps: 12.0,
            latency_p50_ms: 80.0,
            latency_p99_ms: 180.0,
            assertions: vec![
                CiAssertion {
                    name: "throughput".to_string(),
                    expected: ">= 10 tok/s".to_string(),
                    actual: "12.0 tok/s".to_string(),
                    passed: true,
                    gate_id: "F-PROFILE-CI-001".to_string(),
                },
                CiAssertion {
                    name: "p50".to_string(),
                    expected: "<= 100 ms".to_string(),
                    actual: "80.0 ms".to_string(),
                    passed: true,
                    gate_id: "F-PROFILE-CI-002".to_string(),
                },
                CiAssertion {
                    name: "p99".to_string(),
                    expected: "<= 150 ms".to_string(),
                    actual: "180.0 ms".to_string(),
                    passed: false,
                    gate_id: "F-PROFILE-CI-003".to_string(),
                },
            ],
            passed: false,
        };
        assert!(!result.passed);
        assert_eq!(result.assertions.len(), 3);
        assert!(!result.assertions[2].passed);
    }

    // Additional tests for edge cases and coverage

    #[test]
    fn test_parse_diff_output_multiple_transposed() {
        let config = DiffConfig::default();
        let executor = DifferentialExecutor::new(config);
        let output = "tensor1: [100, 200] vs [200, 100] ⚠️ TRANSPOSED\n\
                      tensor2: [50, 100] vs [100, 50] TRANSPOSED\n\
                      tensor3: [32, 64] vs [64, 32] ⚠️";
        let result = executor.parse_diff_output(output).unwrap();
        assert!(!result.passed);
        assert_eq!(result.transposed_tensors, 3);
        assert_eq!(result.mismatched_tensors, 3);
    }

    #[test]
    fn test_parse_diff_output_no_colon() {
        let config = DiffConfig::default();
        let executor = DifferentialExecutor::new(config);
        // Line with TRANSPOSED but no colon should be skipped
        let output = "tensor TRANSPOSED without colon\n\
                      valid_tensor: [10, 20] TRANSPOSED";
        let result = executor.parse_diff_output(output).unwrap();
        assert_eq!(result.transposed_tensors, 1);
    }

    #[test]
    fn test_tensor_mismatch_shape_fields() {
        let mismatch = TensorMismatch {
            name: "layer.0.attn.weight".to_string(),
            shape_a: vec![768, 768, 12],
            shape_b: vec![768, 12, 768],
            mismatch_type: TensorMismatchType::Transposed,
        };
        assert_eq!(mismatch.shape_a.len(), 3);
        assert_eq!(mismatch.shape_b.len(), 3);
    }

    #[test]
    fn test_tensor_diff_result_serialization() {
        let result = TensorDiffResult {
            total_tensors: 50,
            mismatched_tensors: 2,
            transposed_tensors: 1,
            mismatches: vec![TensorMismatch {
                name: "test".to_string(),
                shape_a: vec![10, 20],
                shape_b: vec![20, 10],
                mismatch_type: TensorMismatchType::Transposed,
            }],
            passed: false,
        };
        let json = serde_json::to_string(&result).unwrap();
        let parsed: TensorDiffResult = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.total_tensors, 50);
        assert_eq!(parsed.transposed_tensors, 1);
    }

    #[test]
    fn test_inference_comparison_result_serialization() {
        let result = InferenceComparisonResult {
            total_tokens: 20,
            matching_tokens: 18,
            max_logit_diff: 0.05,
            passed: false,
            token_comparisons: vec![TokenComparison {
                index: 5,
                token_a: 100,
                token_b: 101,
                logit_diff: 0.05,
                matches: false,
            }],
        };
        let json = serde_json::to_string(&result).unwrap();
        let parsed: InferenceComparisonResult = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.total_tokens, 20);
        assert_eq!(parsed.token_comparisons.len(), 1);
    }

    #[test]
    fn test_benchmark_metrics_serialization() {
        let metrics = BenchmarkMetrics {
            path: "/path/to/model.gguf".to_string(),
            throughput_tps: 15.5,
            latency_p50_ms: 65.0,
            latency_p99_ms: 130.0,
        };
        let json = serde_json::to_string(&metrics).unwrap();
        let parsed: BenchmarkMetrics = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.path, "/path/to/model.gguf");
    }

    #[test]
    fn test_ci_assertion_serialization() {
        let assertion = CiAssertion {
            name: "throughput".to_string(),
            expected: ">= 10".to_string(),
            actual: "15".to_string(),
            passed: true,
            gate_id: "F-CI-001".to_string(),
        };
        let json = serde_json::to_string(&assertion).unwrap();
        let parsed: CiAssertion = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.name, "throughput");
        assert!(parsed.passed);
    }

    #[test]
    fn test_ci_profile_result_serialization() {
        let result = CiProfileResult {
            throughput_tps: 20.0,
            latency_p50_ms: 50.0,
            latency_p99_ms: 100.0,
            assertions: vec![],
            passed: true,
        };
        let json = serde_json::to_string(&result).unwrap();
        let parsed: CiProfileResult = serde_json::from_str(&json).unwrap();
        assert!(parsed.passed);
    }

    #[test]
    fn test_diff_benchmark_result_serialization() {
        let result = DiffBenchmarkResult {
            model_a: BenchmarkMetrics {
                path: "a.gguf".to_string(),
                throughput_tps: 10.0,
                latency_p50_ms: 50.0,
                latency_p99_ms: 100.0,
            },
            model_b: BenchmarkMetrics {
                path: "b.gguf".to_string(),
                throughput_tps: 12.0,
                latency_p50_ms: 45.0,
                latency_p99_ms: 90.0,
            },
            throughput_delta_pct: 20.0,
            latency_p50_delta_pct: -10.0,
            latency_p99_delta_pct: -10.0,
            regression_detected: false,
            regression_threshold: 5.0,
        };
        let json = serde_json::to_string(&result).unwrap();
        let parsed: DiffBenchmarkResult = serde_json::from_str(&json).unwrap();
        assert!(!parsed.regression_detected);
    }

    #[test]
    fn test_diff_config_all_fields() {
        let config = DiffConfig {
            apr_binary: "/custom/path/to/apr".to_string(),
            filter: Some("embedding".to_string()),
            mismatches_only: false,
            tolerance: 1e-8,
        };
        assert_eq!(config.apr_binary, "/custom/path/to/apr");
        assert_eq!(config.filter.as_deref(), Some("embedding"));
        assert!(!config.mismatches_only);
    }

    #[test]
    fn test_token_comparison_matching() {
        let tc = TokenComparison {
            index: 10,
            token_a: 42,
            token_b: 42,
            logit_diff: 0.0,
            matches: true,
        };
        assert!(tc.matches);
        assert_eq!(tc.token_a, tc.token_b);
    }

    #[test]
    fn test_tensor_mismatch_type_serialization() {
        let types = [
            TensorMismatchType::Transposed,
            TensorMismatchType::ShapeMismatch,
            TensorMismatchType::Missing,
        ];
        for t in types {
            let json = serde_json::to_string(&t).unwrap();
            let parsed: TensorMismatchType = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed, t);
        }
    }

    #[test]
    fn test_tensor_mismatch_serialization() {
        let mismatch = TensorMismatch {
            name: "layer.weight".to_string(),
            shape_a: vec![1, 2, 3],
            shape_b: vec![3, 2, 1],
            mismatch_type: TensorMismatchType::ShapeMismatch,
        };
        let json = serde_json::to_string(&mismatch).unwrap();
        let parsed: TensorMismatch = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.name, "layer.weight");
        assert_eq!(parsed.mismatch_type, TensorMismatchType::ShapeMismatch);
    }

    #[test]
    fn test_diff_config_default_tolerance() {
        let config = DiffConfig::default();
        assert!((config.tolerance - 1e-5).abs() < 1e-10);
    }

    #[test]
    fn test_parse_inference_output_with_token_data() {
        let config = DiffConfig::default();
        let executor = DifferentialExecutor::new(config);
        let json = r#"{
            "total_tokens": 5,
            "matching_tokens": 4,
            "max_logit_diff": 0.02,
            "passed": false,
            "token_comparisons": [
                {"index": 0, "token_a": 1, "token_b": 1, "logit_diff": 0.0, "matches": true},
                {"index": 1, "token_a": 2, "token_b": 3, "logit_diff": 0.02, "matches": false}
            ]
        }"#;
        let result = executor.parse_inference_output(json, true).unwrap();
        assert_eq!(result.total_tokens, 5);
        assert_eq!(result.token_comparisons.len(), 2);
    }
}
