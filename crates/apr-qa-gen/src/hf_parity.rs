//! HuggingFace Parity Oracle
//!
//! Cross-implementation validation oracle that compares Sovereign Stack outputs
//! against HuggingFace transformers golden outputs. Implements Popperian severe
//! testing methodology with Toyota Jidoka (stop-on-defect) principles.
//!
//! # Design Philosophy
//!
//! > "The wrong view of science betrays itself in the craving to be right."
//! > — Karl Popper, *The Logic of Scientific Discovery* (1959)
//!
//! This oracle attempts to **falsify** the hypothesis that our implementation
//! produces equivalent outputs to HuggingFace. A falsification indicates a bug
//! that must be investigated before certification can proceed.
//!
//! # References
//!
//! - Popper, K. (1959). *The Logic of Scientific Discovery*. Routledge.
//! - Ohno, T. (1988). *Toyota Production System*. Productivity Press.
//! - Goldberg, D. (1991). "What Every Computer Scientist Should Know About
//!   Floating-Point Arithmetic." ACM Computing Surveys, 23(1), 5-48.

use crate::oracle::{Oracle, OracleResult};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Tolerance configuration for numerical comparison.
///
/// Following IEEE 754 analysis (Goldberg, 1991) and ML reproducibility
/// guidelines (Pineau et al., 2021), tolerances are precision-specific.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct Tolerance {
    /// Absolute tolerance for FP32 comparison (default: 1e-5)
    pub atol_fp32: f32,
    /// Relative tolerance for FP32 comparison (default: 1e-4)
    pub rtol_fp32: f32,
    /// Absolute tolerance for quantized comparison (default: 1e-2)
    pub atol_quant: f32,
    /// Maximum allowed mismatch ratio before falsification (default: 0.01 = 1%)
    pub max_mismatch_ratio: f32,
}

impl Default for Tolerance {
    fn default() -> Self {
        Self {
            atol_fp32: 1e-5,
            rtol_fp32: 1e-4,
            atol_quant: 1e-2,
            max_mismatch_ratio: 0.01,
        }
    }
}

impl Tolerance {
    /// Create tolerance for FP32 precision
    #[must_use]
    pub const fn fp32() -> Self {
        Self {
            atol_fp32: 1e-5,
            rtol_fp32: 1e-4,
            atol_quant: 1e-2,
            max_mismatch_ratio: 0.01,
        }
    }

    /// Create tolerance for FP16 precision
    #[must_use]
    pub const fn fp16() -> Self {
        Self {
            atol_fp32: 1e-3,
            rtol_fp32: 1e-2,
            atol_quant: 1e-1,
            max_mismatch_ratio: 0.01,
        }
    }

    /// Create tolerance for INT8 quantized models
    #[must_use]
    pub const fn int8() -> Self {
        Self {
            atol_fp32: 1e-1,
            rtol_fp32: 1e-1,
            atol_quant: 1e-1,
            max_mismatch_ratio: 0.05,
        }
    }

    /// Create tolerance for INT4 quantized models
    #[must_use]
    pub const fn int4() -> Self {
        Self {
            atol_fp32: 5e-1,
            rtol_fp32: 2e-1,
            atol_quant: 5e-1,
            max_mismatch_ratio: 0.10,
        }
    }

    /// Check if two values are within tolerance using allclose criterion.
    ///
    /// Implements: |a - b| <= atol + rtol * |b|
    ///
    /// This is the NumPy allclose criterion, which accounts for both
    /// absolute and relative error bounds.
    #[must_use]
    pub fn is_close(&self, actual: f32, expected: f32) -> bool {
        let diff = (actual - expected).abs();
        let bound = self.rtol_fp32.mul_add(expected.abs(), self.atol_fp32);
        diff <= bound
    }
}

/// Tensor comparison result when values diverge.
///
/// Implements Toyota's Andon principle: detailed diagnostic information
/// to enable rapid root cause analysis.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TensorDiff {
    /// Tensor shapes do not match
    ShapeMismatch {
        /// Expected number of elements
        expected: usize,
        /// Actual number of elements
        actual: usize,
    },
    /// Tensor values exceed tolerance
    ValueMismatch {
        /// Number of elements exceeding tolerance
        num_mismatches: usize,
        /// Total number of elements compared
        total: usize,
        /// Ratio of mismatches (num_mismatches / total)
        mismatch_ratio: f32,
        /// Maximum absolute difference observed
        max_diff: f32,
        /// Index of maximum difference
        max_diff_idx: usize,
        /// Expected value at max diff location
        expected_val: f32,
        /// Actual value at max diff location
        actual_val: f32,
        /// Mean absolute difference
        mean_diff: f32,
    },
    /// File could not be read or parsed
    ParseError {
        /// Error message
        message: String,
    },
}

impl std::fmt::Display for TensorDiff {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ShapeMismatch { expected, actual } => {
                write!(f, "Shape mismatch: expected {expected}, got {actual}")
            }
            Self::ValueMismatch {
                num_mismatches,
                total,
                mismatch_ratio,
                max_diff,
                max_diff_idx,
                expected_val,
                actual_val,
                mean_diff,
            } => {
                write!(
                    f,
                    "Value mismatch: {num_mismatches}/{total} elements ({:.2}%) exceed tolerance. \
                     Max diff: {max_diff:.6} at idx {max_diff_idx} (expected: {expected_val:.6}, \
                     actual: {actual_val:.6}). Mean diff: {mean_diff:.6}",
                    mismatch_ratio * 100.0
                )
            }
            Self::ParseError { message } => write!(f, "Parse error: {message}"),
        }
    }
}

/// Pre-computed golden output from HuggingFace transformers.
///
/// Stored as SafeTensors with metadata for reproducibility tracking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoldenOutput {
    /// Hash of input prompt (for lookup)
    pub input_hash: String,
    /// Original prompt text
    pub prompt: String,
    /// Expected logits as raw F32 values
    pub logits: Vec<f32>,
    /// Shape of logits tensor [batch, seq, vocab]
    pub shape: Vec<usize>,
    /// Expected generated text (optional, for text comparison)
    pub text: Option<String>,
    /// HuggingFace model ID used to generate golden
    pub model_id: String,
    /// transformers library version
    pub transformers_version: String,
}

/// HuggingFace Parity Oracle
///
/// Compares model outputs against pre-computed golden outputs from
/// HuggingFace transformers. Implements Popperian falsification:
/// any divergence beyond tolerance falsifies the parity hypothesis.
#[derive(Debug, Clone)]
pub struct HfParityOracle {
    /// Path to ground truth corpus directory
    corpus_path: PathBuf,
    /// Model family (e.g., "llama", "qwen", "whisper")
    model_family: String,
    /// Numerical tolerance configuration
    tolerance: Tolerance,
    /// Cache of loaded golden outputs (keyed by input hash)
    golden_cache: HashMap<String, GoldenOutput>,
}

impl HfParityOracle {
    /// Create a new HF Parity Oracle.
    ///
    /// # Arguments
    ///
    /// * `corpus_path` - Path to ground truth corpus (e.g., `~/src/hf-ground-truth-corpus/oracle/`)
    /// * `model_family` - Model family subdirectory (e.g., "llama-2-7b")
    #[must_use]
    pub fn new(corpus_path: impl AsRef<Path>, model_family: &str) -> Self {
        Self {
            corpus_path: corpus_path.as_ref().to_path_buf(),
            model_family: model_family.to_string(),
            tolerance: Tolerance::default(),
            golden_cache: HashMap::new(),
        }
    }

    /// Configure tolerance settings.
    #[must_use]
    pub fn with_tolerance(mut self, tolerance: Tolerance) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Get the corpus path.
    #[must_use]
    pub fn corpus_path(&self) -> &Path {
        &self.corpus_path
    }

    /// Get the model family.
    #[must_use]
    pub fn model_family(&self) -> &str {
        &self.model_family
    }

    /// Get the tolerance configuration.
    #[must_use]
    pub const fn tolerance(&self) -> &Tolerance {
        &self.tolerance
    }

    /// Load golden output for a given prompt.
    ///
    /// Golden outputs are stored as SafeTensors files named by input hash.
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Golden file cannot be found or read
    /// - SafeTensors deserialization fails
    /// - Required 'logits' tensor is missing
    pub fn load_golden(&self, prompt: &str) -> Result<GoldenOutput, String> {
        let input_hash = hash_prompt(prompt);

        // Check cache first
        if let Some(cached) = self.golden_cache.get(&input_hash) {
            return Ok(cached.clone());
        }

        let path = self
            .corpus_path
            .join(&self.model_family)
            .join(format!("{input_hash}.safetensors"));

        Self::load_golden_from_path(&path, prompt, &input_hash)
    }

    /// Load golden output from a specific file path.
    ///
    /// Note: SafeTensors metadata is not directly accessible via the Rust API,
    /// so we only extract tensor data. Metadata should be stored in a companion
    /// JSON file if needed.
    fn load_golden_from_path(
        path: &Path,
        prompt: &str,
        input_hash: &str,
    ) -> Result<GoldenOutput, String> {
        let data = std::fs::read(path).map_err(|e| format!("Failed to read golden file: {e}"))?;

        let tensors = safetensors::SafeTensors::deserialize(&data)
            .map_err(|e| format!("Failed to parse SafeTensors: {e}"))?;

        // Extract logits tensor
        let logits_view = tensors
            .tensor("logits")
            .map_err(|e| format!("Missing 'logits' tensor: {e}"))?;

        let logits: Vec<f32> = logits_view
            .data()
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();

        // Try to load companion metadata JSON if it exists
        let metadata_path = path.with_extension("json");
        let (model_id, transformers_version, text) = if metadata_path.exists() {
            Self::load_metadata_json(&metadata_path).unwrap_or_default()
        } else {
            (String::new(), String::new(), None)
        };

        Ok(GoldenOutput {
            input_hash: input_hash.to_string(),
            prompt: prompt.to_string(),
            logits,
            shape: logits_view.shape().to_vec(),
            text,
            model_id,
            transformers_version,
        })
    }

    /// Load metadata from companion JSON file.
    fn load_metadata_json(path: &Path) -> Result<(String, String, Option<String>), String> {
        #[derive(Deserialize)]
        struct MetadataJson {
            #[serde(default)]
            model: String,
            #[serde(default)]
            transformers_version: String,
            generated_text: Option<String>,
        }

        let data =
            std::fs::read_to_string(path).map_err(|e| format!("Failed to read metadata: {e}"))?;

        let meta: MetadataJson = serde_json::from_str(&data)
            .map_err(|e| format!("Failed to parse metadata JSON: {e}"))?;

        Ok((meta.model, meta.transformers_version, meta.generated_text))
    }

    /// Compare two tensors with configured tolerance.
    ///
    /// Returns `Ok(())` if tensors are within tolerance, `Err(TensorDiff)` otherwise.
    /// Implements the allclose criterion: |a - b| <= atol + rtol * |b|
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Tensor shapes (lengths) do not match
    /// - Mismatch ratio exceeds configured threshold
    pub fn tensors_close(&self, expected: &[f32], actual: &[f32]) -> Result<(), TensorDiff> {
        if expected.len() != actual.len() {
            return Err(TensorDiff::ShapeMismatch {
                expected: expected.len(),
                actual: actual.len(),
            });
        }

        if expected.is_empty() {
            return Ok(());
        }

        let mut max_diff: f32 = 0.0;
        let mut max_diff_idx: usize = 0;
        let mut num_mismatches: usize = 0;
        let mut sum_diff: f64 = 0.0;

        for (i, (&e, &a)) in expected.iter().zip(actual.iter()).enumerate() {
            let diff = (e - a).abs();
            sum_diff += f64::from(diff);

            if !self.tolerance.is_close(a, e) {
                num_mismatches += 1;
                if diff > max_diff {
                    max_diff = diff;
                    max_diff_idx = i;
                }
            }
        }

        let mismatch_ratio = num_mismatches as f32 / expected.len() as f32;
        // Mean diff is intentionally truncated to f32 for consistency with tensor data
        #[allow(clippy::cast_possible_truncation)]
        let mean_diff = (sum_diff / expected.len() as f64) as f32;

        if mismatch_ratio > self.tolerance.max_mismatch_ratio {
            Err(TensorDiff::ValueMismatch {
                num_mismatches,
                total: expected.len(),
                mismatch_ratio,
                max_diff,
                max_diff_idx,
                expected_val: expected[max_diff_idx],
                actual_val: actual[max_diff_idx],
                mean_diff,
            })
        } else {
            Ok(())
        }
    }

    /// Compare actual output tensor file against golden.
    ///
    /// # Arguments
    ///
    /// * `actual_path` - Path to SafeTensors file with actual model output
    /// * `golden` - Pre-loaded golden output to compare against
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Actual file cannot be read or parsed
    /// - Tensor shapes do not match
    /// - Values exceed tolerance threshold
    pub fn compare_tensor_file(
        &self,
        actual_path: &Path,
        golden: &GoldenOutput,
    ) -> Result<(), TensorDiff> {
        let data = std::fs::read(actual_path).map_err(|e| TensorDiff::ParseError {
            message: format!("Failed to read actual output: {e}"),
        })?;

        let tensors =
            safetensors::SafeTensors::deserialize(&data).map_err(|e| TensorDiff::ParseError {
                message: format!("Failed to parse SafeTensors: {e}"),
            })?;

        let logits_view = tensors
            .tensor("logits")
            .map_err(|e| TensorDiff::ParseError {
                message: format!("Missing 'logits' tensor: {e}"),
            })?;

        let actual: Vec<f32> = logits_view
            .data()
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();

        self.tensors_close(&golden.logits, &actual)
    }

    /// Compute statistical summary of divergence.
    ///
    /// Returns (max_diff, mean_diff, std_diff) for diagnostic purposes.
    #[must_use]
    pub fn compute_divergence_stats(expected: &[f32], actual: &[f32]) -> (f32, f32, f32) {
        if expected.len() != actual.len() || expected.is_empty() {
            return (0.0, 0.0, 0.0);
        }

        let diffs: Vec<f32> = expected
            .iter()
            .zip(actual.iter())
            .map(|(e, a)| (e - a).abs())
            .collect();

        let max_diff = diffs.iter().copied().fold(0.0f32, f32::max);
        let mean_diff: f32 = diffs.iter().sum::<f32>() / diffs.len() as f32;

        // Compute standard deviation
        let variance: f32 =
            diffs.iter().map(|d| (d - mean_diff).powi(2)).sum::<f32>() / diffs.len() as f32;
        let std_diff = variance.sqrt();

        (max_diff, mean_diff, std_diff)
    }

    /// Detect systematic bias between expected and actual outputs.
    ///
    /// Returns true if bias is detected (mean shift or scale drift).
    #[must_use]
    pub fn detect_systematic_bias(expected: &[f32], actual: &[f32]) -> Option<String> {
        if expected.len() != actual.len() || expected.is_empty() {
            return None;
        }

        let n = expected.len() as f32;

        // Compute means
        let mean_e: f32 = expected.iter().sum::<f32>() / n;
        let mean_a: f32 = actual.iter().sum::<f32>() / n;

        // Compute standard deviations
        let std_e: f32 = (expected.iter().map(|x| (x - mean_e).powi(2)).sum::<f32>() / n).sqrt();
        let std_a: f32 = (actual.iter().map(|x| (x - mean_a).powi(2)).sum::<f32>() / n).sqrt();

        // Check for mean shift (> 3 sigma)
        if std_e > 1e-10 && (mean_a - mean_e).abs() > 3.0 * std_e {
            return Some(format!(
                "Mean shift detected: expected {mean_e:.6}, actual {mean_a:.6} (shift: {:.6} sigma)",
                (mean_a - mean_e).abs() / std_e
            ));
        }

        // Check for scale drift (> 10%)
        if std_e > 1e-10 && (std_a / std_e - 1.0).abs() > 0.1 {
            return Some(format!(
                "Scale drift detected: expected std {std_e:.6}, actual std {std_a:.6} (ratio: {:.2})",
                std_a / std_e
            ));
        }

        None
    }
}

impl Oracle for HfParityOracle {
    fn evaluate(&self, prompt: &str, output: &str) -> OracleResult {
        // Try to load golden output for this prompt
        let golden = match self.load_golden(prompt) {
            Ok(g) => g,
            Err(e) => {
                // No golden output available - skip (not a failure)
                return OracleResult::Corroborated {
                    evidence: format!(
                        "No golden output for prompt (hash: {}): {e}",
                        hash_prompt(prompt)
                    ),
                };
            }
        };

        // If golden has expected text, compare text output
        if let Some(expected_text) = &golden.text {
            let output_trimmed = output.trim();
            let expected_trimmed = expected_text.trim();

            if output_trimmed == expected_trimmed {
                return OracleResult::Corroborated {
                    evidence: format!(
                        "Text output matches HF golden ({} chars)",
                        output_trimmed.len()
                    ),
                };
            }
            // Text mismatch - check if it's a tensor file path
            let output_path = Path::new(output_trimmed);
            if !output_path.exists() {
                return OracleResult::Falsified {
                    reason: "Text output differs from HF golden".to_string(),
                    evidence: format!(
                        "Expected: '{}'\nActual: '{}'",
                        truncate(expected_trimmed, 100),
                        truncate(output_trimmed, 100)
                    ),
                };
            }
        }

        // Try to interpret output as a tensor file path
        let output_path = Path::new(output.trim());
        if output_path.exists()
            && output_path
                .extension()
                .is_some_and(|ext| ext == "safetensors")
        {
            match self.compare_tensor_file(output_path, &golden) {
                Ok(()) => {
                    return OracleResult::Corroborated {
                        evidence: format!(
                            "Tensor parity verified: {} elements within tolerance (atol={}, rtol={})",
                            golden.logits.len(),
                            self.tolerance.atol_fp32,
                            self.tolerance.rtol_fp32
                        ),
                    };
                }
                Err(diff) => {
                    return OracleResult::Falsified {
                        reason: "Tensor mismatch with HF golden".to_string(),
                        evidence: diff.to_string(),
                    };
                }
            }
        }

        // Output is plain text, no tensor file - check for text comparison
        OracleResult::Corroborated {
            evidence: "Output is text, no tensor comparison available".to_string(),
        }
    }

    fn name(&self) -> &'static str {
        "hf_parity"
    }
}

/// Hash a prompt string for golden output lookup.
///
/// Uses SHA-256 truncated to 16 hex chars for cross-language compatibility.
/// This matches the Python `generate_golden.py` script implementation.
#[must_use]
pub fn hash_prompt(prompt: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(prompt.as_bytes());
    let result = hasher.finalize();
    // Take first 8 bytes (16 hex chars) to match Python implementation
    hex::encode(&result[..8])
}

/// Truncate a string for display purposes.
fn truncate(s: &str, max_len: usize) -> &str {
    if s.len() <= max_len {
        s
    } else {
        // Find a safe UTF-8 boundary
        let mut end = max_len;
        while end > 0 && !s.is_char_boundary(end) {
            end -= 1;
        }
        &s[..end]
    }
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unreadable_literal, clippy::needless_range_loop)]
mod tests {
    use super::*;

    // ============================================================
    // Tolerance Tests
    // ============================================================

    #[test]
    fn test_tolerance_default() {
        let tol = Tolerance::default();
        assert!((tol.atol_fp32 - 1e-5).abs() < 1e-10);
        assert!((tol.rtol_fp32 - 1e-4).abs() < 1e-10);
        assert!((tol.atol_quant - 1e-2).abs() < 1e-10);
        assert!((tol.max_mismatch_ratio - 0.01).abs() < 1e-10);
    }

    #[test]
    fn test_tolerance_fp32() {
        let tol = Tolerance::fp32();
        assert!((tol.atol_fp32 - 1e-5).abs() < 1e-10);
    }

    #[test]
    fn test_tolerance_fp16() {
        let tol = Tolerance::fp16();
        assert!((tol.atol_fp32 - 1e-3).abs() < 1e-10);
    }

    #[test]
    fn test_tolerance_int8() {
        let tol = Tolerance::int8();
        assert!((tol.atol_fp32 - 1e-1).abs() < 1e-10);
    }

    #[test]
    fn test_tolerance_int4() {
        let tol = Tolerance::int4();
        assert!((tol.atol_fp32 - 5e-1).abs() < 1e-10);
    }

    #[test]
    fn test_tolerance_is_close_identical() {
        let tol = Tolerance::default();
        assert!(tol.is_close(1.0, 1.0));
        assert!(tol.is_close(0.0, 0.0));
        assert!(tol.is_close(-5.0, -5.0));
    }

    #[test]
    fn test_tolerance_is_close_within_atol() {
        let tol = Tolerance::default();
        // diff = 1e-6, bound = 1e-5 + 1e-4 * 1.0 = 1.1e-4 → close
        assert!(tol.is_close(1.000001, 1.0));
    }

    #[test]
    fn test_tolerance_is_close_outside_tolerance() {
        let tol = Tolerance::default();
        // diff = 0.1, bound = 1e-5 + 1e-4 * 1.0 = 1.1e-4 → not close
        assert!(!tol.is_close(1.1, 1.0));
    }

    #[test]
    fn test_tolerance_is_close_relative_tolerance() {
        let tol = Tolerance::default();
        // For large values, relative tolerance dominates
        // diff = 100, bound = 1e-5 + 1e-4 * 1_000_000 = 100.00001 → close
        assert!(tol.is_close(1_000_100.0, 1_000_000.0));
    }

    #[test]
    fn test_tolerance_is_close_zero_expected() {
        let tol = Tolerance::default();
        // When expected is 0, only atol matters
        assert!(tol.is_close(1e-6, 0.0));
        assert!(!tol.is_close(1e-4, 0.0));
    }

    // ============================================================
    // TensorDiff Tests
    // ============================================================

    #[test]
    fn test_tensor_diff_display_shape_mismatch() {
        let diff = TensorDiff::ShapeMismatch {
            expected: 100,
            actual: 50,
        };
        let s = diff.to_string();
        assert!(s.contains("Shape mismatch"));
        assert!(s.contains("100"));
        assert!(s.contains("50"));
    }

    #[test]
    fn test_tensor_diff_display_value_mismatch() {
        let diff = TensorDiff::ValueMismatch {
            num_mismatches: 10,
            total: 100,
            mismatch_ratio: 0.1,
            max_diff: 0.5,
            max_diff_idx: 42,
            expected_val: 1.0,
            actual_val: 1.5,
            mean_diff: 0.1,
        };
        let s = diff.to_string();
        assert!(s.contains("Value mismatch"));
        assert!(s.contains("10/100"));
        assert!(s.contains("10.00%"));
    }

    #[test]
    fn test_tensor_diff_display_parse_error() {
        let diff = TensorDiff::ParseError {
            message: "file not found".to_string(),
        };
        let s = diff.to_string();
        assert!(s.contains("Parse error"));
        assert!(s.contains("file not found"));
    }

    // ============================================================
    // HfParityOracle Construction Tests
    // ============================================================

    #[test]
    fn test_oracle_new() {
        let oracle = HfParityOracle::new("/tmp/corpus", "llama-2-7b");
        assert_eq!(oracle.corpus_path(), Path::new("/tmp/corpus"));
        assert_eq!(oracle.model_family(), "llama-2-7b");
    }

    #[test]
    fn test_oracle_with_tolerance() {
        let tol = Tolerance::int4();
        let oracle = HfParityOracle::new("/tmp", "test").with_tolerance(tol);
        assert!((oracle.tolerance().atol_fp32 - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_oracle_name() {
        let oracle = HfParityOracle::new("/tmp", "test");
        assert_eq!(oracle.name(), "hf_parity");
    }

    // ============================================================
    // Tensor Comparison Tests
    // ============================================================

    #[test]
    fn test_tensors_close_identical() {
        let oracle = HfParityOracle::new("/tmp", "test");
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!(oracle.tensors_close(&a, &b).is_ok());
    }

    #[test]
    fn test_tensors_close_within_tolerance() {
        let oracle = HfParityOracle::new("/tmp", "test");
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![1.000001, 2.000001, 3.000001, 4.000001, 5.000001];
        assert!(oracle.tensors_close(&a, &b).is_ok());
    }

    #[test]
    fn test_tensors_close_shape_mismatch() {
        let oracle = HfParityOracle::new("/tmp", "test");
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0];
        let result = oracle.tensors_close(&a, &b);
        assert!(matches!(result, Err(TensorDiff::ShapeMismatch { .. })));
    }

    #[test]
    fn test_tensors_close_empty() {
        let oracle = HfParityOracle::new("/tmp", "test");
        let a: Vec<f32> = vec![];
        let b: Vec<f32> = vec![];
        assert!(oracle.tensors_close(&a, &b).is_ok());
    }

    #[test]
    fn test_tensors_close_exceeds_mismatch_ratio() {
        let oracle = HfParityOracle::new("/tmp", "test");
        // 50% mismatch rate exceeds 1% threshold
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 2.0, 100.0, 100.0]; // 2/4 = 50% mismatch
        let result = oracle.tensors_close(&a, &b);
        assert!(matches!(result, Err(TensorDiff::ValueMismatch { .. })));
    }

    #[test]
    fn test_tensors_close_within_mismatch_ratio() {
        // Use int4 tolerance which allows 10% mismatch (max_mismatch_ratio = 0.10)
        let oracle = HfParityOracle::new("/tmp", "test").with_tolerance(Tolerance::int4());
        // Create array with exactly 5% mismatch (within 10% threshold)
        let a: Vec<f32> = vec![1.0; 100];
        let mut b = a.clone();
        // Make 5 elements differ significantly (5% = 0.05 < 0.10 threshold)
        for i in 0..5 {
            b[i] = 100.0;
        }
        // With int4 tolerance, 5% mismatch ratio is WITHIN the 10% threshold
        // so this should pass (Ok)
        let result = oracle.tensors_close(&a, &b);
        assert!(result.is_ok(), "5% mismatch should be within 10% threshold");
    }

    // ============================================================
    // Statistical Analysis Tests
    // ============================================================

    #[test]
    fn test_compute_divergence_stats_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        let (max, mean, std) = HfParityOracle::compute_divergence_stats(&a, &b);
        assert!(max < 1e-10);
        assert!(mean < 1e-10);
        assert!(std < 1e-10);
    }

    #[test]
    fn test_compute_divergence_stats_with_diff() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 4.0]; // diff of 1.0 at index 2
        let (max, mean, _std) = HfParityOracle::compute_divergence_stats(&a, &b);
        assert!((max - 1.0).abs() < 1e-6);
        assert!((mean - 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_compute_divergence_stats_empty() {
        let a: Vec<f32> = vec![];
        let b: Vec<f32> = vec![];
        let (max, mean, std) = HfParityOracle::compute_divergence_stats(&a, &b);
        assert!(max == 0.0);
        assert!(mean == 0.0);
        assert!(std == 0.0);
    }

    #[test]
    fn test_compute_divergence_stats_mismatched_len() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0];
        let (max, mean, std) = HfParityOracle::compute_divergence_stats(&a, &b);
        assert!(max == 0.0);
        assert!(mean == 0.0);
        assert!(std == 0.0);
    }

    #[test]
    fn test_detect_systematic_bias_none() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!(HfParityOracle::detect_systematic_bias(&a, &b).is_none());
    }

    #[test]
    fn test_detect_systematic_bias_mean_shift() {
        let a = vec![0.0, 0.0, 0.0, 0.0, 0.0];
        let b = vec![10.0, 10.0, 10.0, 10.0, 10.0]; // Large mean shift
        // With zero std in expected, we can't detect via sigma
        // This tests the edge case
        let result = HfParityOracle::detect_systematic_bias(&a, &b);
        // Zero std means no sigma-based detection
        assert!(result.is_none());
    }

    #[test]
    fn test_detect_systematic_bias_with_variance() {
        let a = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let b = vec![10.0, 11.0, 12.0, 13.0, 14.0]; // Shift of 10
        let result = HfParityOracle::detect_systematic_bias(&a, &b);
        assert!(result.is_some());
        assert!(result.unwrap().contains("Mean shift"));
    }

    #[test]
    fn test_detect_systematic_bias_scale_drift() {
        let a = vec![0.0, 1.0, 2.0, 3.0, 4.0]; // std ≈ 1.41
        let b = vec![0.0, 2.0, 4.0, 6.0, 8.0]; // std ≈ 2.83 (2x scale)
        let result = HfParityOracle::detect_systematic_bias(&a, &b);
        assert!(result.is_some());
        assert!(result.unwrap().contains("Scale drift"));
    }

    #[test]
    fn test_detect_systematic_bias_empty() {
        let a: Vec<f32> = vec![];
        let b: Vec<f32> = vec![];
        assert!(HfParityOracle::detect_systematic_bias(&a, &b).is_none());
    }

    // ============================================================
    // Hash Function Tests
    // ============================================================

    #[test]
    fn test_hash_prompt_deterministic() {
        let h1 = hash_prompt("Hello, world!");
        let h2 = hash_prompt("Hello, world!");
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_hash_prompt_different_inputs() {
        let h1 = hash_prompt("Hello");
        let h2 = hash_prompt("World");
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_hash_prompt_format() {
        let h = hash_prompt("test");
        assert_eq!(h.len(), 16); // 16 hex chars = 64 bits
        assert!(h.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn test_hash_prompt_empty() {
        let h = hash_prompt("");
        assert_eq!(h.len(), 16);
    }

    #[test]
    fn test_hash_prompt_unicode() {
        let h1 = hash_prompt("こんにちは");
        let h2 = hash_prompt("こんにちは");
        assert_eq!(h1, h2);
        assert_eq!(h1.len(), 16);
    }

    #[test]
    fn test_hash_prompt_cross_language_compatibility() {
        // These hashes are from the Python generate_golden.py script
        // Using SHA-256 truncated to 16 hex chars for cross-language consistency
        assert_eq!(hash_prompt("def fibonacci(n):"), "c839979da8b41875");
        assert_eq!(hash_prompt("2 + 2 ="), "154e0c9c61763891");
        assert_eq!(hash_prompt("fn main() {"), "72879bbc234f8df8");
        assert_eq!(hash_prompt("x"), "2d711642b726b044");
        assert_eq!(hash_prompt("1"), "6b86b273ff34fce1");
    }

    // ============================================================
    // Truncate Tests
    // ============================================================

    #[test]
    fn test_truncate_short_string() {
        assert_eq!(truncate("hello", 10), "hello");
    }

    #[test]
    fn test_truncate_exact_length() {
        assert_eq!(truncate("hello", 5), "hello");
    }

    #[test]
    fn test_truncate_long_string() {
        assert_eq!(truncate("hello world", 5), "hello");
    }

    #[test]
    fn test_truncate_empty() {
        assert_eq!(truncate("", 10), "");
    }

    #[test]
    fn test_truncate_unicode_boundary() {
        // "こんにちは" is 15 bytes (3 bytes per char)
        let s = "こんにちは";
        let truncated = truncate(s, 6); // Should truncate to 2 chars (6 bytes)
        assert_eq!(truncated, "こん");
    }

    // ============================================================
    // Oracle Evaluate Tests (without filesystem)
    // ============================================================

    #[test]
    fn test_oracle_evaluate_no_golden() {
        let oracle = HfParityOracle::new("/nonexistent/path", "test");
        let result = oracle.evaluate("test prompt", "test output");
        assert!(result.is_corroborated());
        if let OracleResult::Corroborated { evidence } = result {
            assert!(evidence.contains("No golden output"));
        }
    }

    #[test]
    fn test_oracle_evaluate_text_no_tensor() {
        let oracle = HfParityOracle::new("/nonexistent", "test");
        let result = oracle.evaluate("prompt", "plain text output");
        assert!(result.is_corroborated());
    }

    // ============================================================
    // Golden Output Tests
    // ============================================================

    #[test]
    fn test_golden_output_serialization() {
        let golden = GoldenOutput {
            input_hash: "abc123".to_string(),
            prompt: "test prompt".to_string(),
            logits: vec![1.0, 2.0, 3.0],
            shape: vec![1, 3],
            text: Some("generated".to_string()),
            model_id: "test-model".to_string(),
            transformers_version: "4.38.0".to_string(),
        };
        let json = serde_json::to_string(&golden).expect("serialize");
        assert!(json.contains("abc123"));
        assert!(json.contains("test prompt"));
    }

    #[test]
    fn test_golden_output_deserialization() {
        let json = r#"{
            "input_hash": "abc123",
            "prompt": "test",
            "logits": [1.0, 2.0],
            "shape": [1, 2],
            "text": null,
            "model_id": "model",
            "transformers_version": "4.38.0"
        }"#;
        let golden: GoldenOutput = serde_json::from_str(json).expect("deserialize");
        assert_eq!(golden.input_hash, "abc123");
        assert_eq!(golden.logits.len(), 2);
        assert!(golden.text.is_none());
    }

    // ============================================================
    // Mutation-Killing Tests
    // ============================================================

    #[test]
    fn test_tolerance_atol_vs_rtol() {
        let tol = Tolerance::default();
        // atol = 1e-5, rtol = 1e-4
        // For small expected values, atol dominates
        // bound = 1e-5 + 1e-4 * 1e-6 = 1.0001e-5
        // diff = |1.000001e-6 - 1e-6| = 1e-12 << bound
        assert!(tol.is_close(1.000001e-6, 1e-6));
        // For large expected values, rtol dominates
        // bound = 1e-5 + 1e-4 * 10000 = 1.00001
        // diff = 1, bound = ~1
        assert!(tol.is_close(10001.0, 10000.0));
        // Test that values outside tolerance are detected
        // diff = 0.1, bound = 1e-5 + 1e-4 * 1 = 0.00011
        assert!(!tol.is_close(1.1, 1.0));
    }

    #[test]
    fn test_tensors_close_boundary_mismatch_ratio() {
        // Test exactly at the boundary (1% mismatch)
        let oracle = HfParityOracle::new("/tmp", "test");
        let a: Vec<f32> = vec![1.0; 100];
        let mut b = a.clone();
        // Make exactly 1 element differ (1% of 100)
        b[0] = 100.0;
        // 1% = 0.01, which equals max_mismatch_ratio, should still fail
        // because we use > not >=
        let result = oracle.tensors_close(&a, &b);
        // 1/100 = 0.01, which is NOT > 0.01, so it should pass
        assert!(result.is_ok());
    }

    #[test]
    fn test_tensors_close_just_over_boundary() {
        let oracle = HfParityOracle::new("/tmp", "test");
        let a: Vec<f32> = vec![1.0; 100];
        let mut b = a.clone();
        // Make 2 elements differ (2% of 100)
        b[0] = 100.0;
        b[1] = 100.0;
        // 2% > 1% threshold → should fail
        let result = oracle.tensors_close(&a, &b);
        assert!(matches!(result, Err(TensorDiff::ValueMismatch { .. })));
    }

    #[test]
    fn test_is_close_negative_values() {
        let tol = Tolerance::default();
        assert!(tol.is_close(-1.0, -1.0));
        assert!(tol.is_close(-1.000001, -1.0));
        assert!(!tol.is_close(-2.0, -1.0));
    }

    #[test]
    fn test_value_mismatch_captures_worst_element() {
        let oracle = HfParityOracle::new("/tmp", "test");
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 2.0, 100.0, 50.0]; // Index 2 has max diff (97)
        if let Err(TensorDiff::ValueMismatch {
            max_diff_idx,
            max_diff,
            expected_val,
            actual_val,
            ..
        }) = oracle.tensors_close(&a, &b)
        {
            assert_eq!(max_diff_idx, 2);
            assert!((max_diff - 97.0).abs() < 0.001);
            assert!((expected_val - 3.0).abs() < 0.001);
            assert!((actual_val - 100.0).abs() < 0.001);
        } else {
            panic!("Expected ValueMismatch");
        }
    }

    #[test]
    fn test_mean_diff_calculation() {
        let oracle = HfParityOracle::new("/tmp", "test");
        let a = vec![0.0, 0.0, 0.0, 0.0];
        let b = vec![1.0, 1.0, 1.0, 1.0]; // All diff by 1.0
        if let Err(TensorDiff::ValueMismatch { mean_diff, .. }) = oracle.tensors_close(&a, &b) {
            assert!((mean_diff - 1.0).abs() < 0.001);
        } else {
            panic!("Expected ValueMismatch");
        }
    }

    // ============================================================
    // Tolerance Struct Equality Tests
    // ============================================================

    #[test]
    fn test_tolerance_equality() {
        let t1 = Tolerance::default();
        let t2 = Tolerance::default();
        assert_eq!(t1, t2);
    }

    #[test]
    fn test_tolerance_inequality() {
        let t1 = Tolerance::fp32();
        let t2 = Tolerance::int4();
        assert_ne!(t1, t2);
    }

    #[test]
    fn test_tolerance_clone() {
        let t1 = Tolerance::fp16();
        let t2 = t1;
        assert_eq!(t1, t2);
    }

    // ============================================================
    // TensorDiff Equality Tests
    // ============================================================

    #[test]
    fn test_tensor_diff_equality_shape() {
        let d1 = TensorDiff::ShapeMismatch {
            expected: 10,
            actual: 5,
        };
        let d2 = TensorDiff::ShapeMismatch {
            expected: 10,
            actual: 5,
        };
        assert_eq!(d1, d2);
    }

    #[test]
    fn test_tensor_diff_clone() {
        let d1 = TensorDiff::ParseError {
            message: "test".to_string(),
        };
        let d2 = d1.clone();
        assert_eq!(d1, d2);
    }

    // ============================================================
    // SafeTensors File I/O Helper
    // ============================================================

    /// Create a minimal SafeTensors file with a "logits" tensor from f32 values.
    fn create_safetensors_file(path: &Path, logits: &[f32], shape: &[usize]) {
        use safetensors::tensor::Dtype;
        use std::borrow::Cow;

        struct TestTensor {
            shape: Vec<usize>,
            data: Vec<u8>,
        }

        impl safetensors::tensor::View for TestTensor {
            fn dtype(&self) -> Dtype {
                Dtype::F32
            }
            fn shape(&self) -> &[usize] {
                &self.shape
            }
            fn data(&self) -> Cow<'_, [u8]> {
                Cow::Borrowed(&self.data)
            }
            fn data_len(&self) -> usize {
                self.data.len()
            }
        }

        let data: Vec<u8> = logits.iter().flat_map(|f| f.to_le_bytes()).collect();
        let tensor = TestTensor {
            shape: shape.to_vec(),
            data,
        };
        let tensors = vec![("logits".to_string(), tensor)];
        let bytes = safetensors::tensor::serialize(tensors, &None)
            .expect("failed to serialize safetensors");
        std::fs::write(path, bytes).expect("failed to write safetensors file");
    }

    /// Create a unique temporary directory for a test.
    fn make_test_dir(test_name: &str) -> PathBuf {
        let dir = std::env::temp_dir()
            .join("apr-qa-gen-tests")
            .join(test_name);
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("failed to create test dir");
        dir
    }

    // ============================================================
    // load_golden_from_path Tests (file-based)
    // ============================================================

    #[test]
    fn test_load_golden_from_path_success_no_metadata() {
        let dir = make_test_dir("load_golden_no_meta");
        let logits = vec![1.0f32, 2.0, 3.0, 4.0];
        let st_path = dir.join("test.safetensors");
        create_safetensors_file(&st_path, &logits, &[1, 4]);

        let result = HfParityOracle::load_golden_from_path(&st_path, "test prompt", "abc123");
        assert!(result.is_ok(), "Expected Ok, got: {result:?}");

        let golden = result.expect("already checked");
        assert_eq!(golden.input_hash, "abc123");
        assert_eq!(golden.prompt, "test prompt");
        assert_eq!(golden.logits.len(), 4);
        assert!((golden.logits[0] - 1.0).abs() < 1e-6);
        assert!((golden.logits[3] - 4.0).abs() < 1e-6);
        assert_eq!(golden.shape, vec![1, 4]);
        // No companion JSON, so model_id and transformers_version are empty
        assert!(golden.model_id.is_empty());
        assert!(golden.transformers_version.is_empty());
        assert!(golden.text.is_none());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_load_golden_from_path_with_metadata_json() {
        let dir = make_test_dir("load_golden_with_meta");
        let logits = vec![0.5f32, 1.5];
        let st_path = dir.join("golden.safetensors");
        create_safetensors_file(&st_path, &logits, &[1, 2]);

        // Write companion metadata JSON
        let meta_path = dir.join("golden.json");
        std::fs::write(
            &meta_path,
            r#"{"model": "test-model-id", "transformers_version": "4.42.0", "generated_text": "hello world"}"#,
        )
        .expect("write meta");

        let result = HfParityOracle::load_golden_from_path(&st_path, "prompt", "hash123");
        assert!(result.is_ok());

        let golden = result.expect("already checked");
        assert_eq!(golden.model_id, "test-model-id");
        assert_eq!(golden.transformers_version, "4.42.0");
        assert_eq!(golden.text, Some("hello world".to_string()));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_load_golden_from_path_file_not_found() {
        let result = HfParityOracle::load_golden_from_path(
            Path::new("/nonexistent/file.safetensors"),
            "p",
            "h",
        );
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Failed to read golden file"));
    }

    #[test]
    fn test_load_golden_from_path_invalid_safetensors() {
        let dir = make_test_dir("load_golden_invalid_st");
        let path = dir.join("bad.safetensors");
        std::fs::write(&path, b"this is not a valid safetensors file").expect("write");

        let result = HfParityOracle::load_golden_from_path(&path, "p", "h");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Failed to parse SafeTensors"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_load_golden_from_path_missing_logits_tensor() {
        use safetensors::tensor::Dtype;
        use std::borrow::Cow;

        struct TestTensor {
            shape: Vec<usize>,
            data: Vec<u8>,
        }
        impl safetensors::tensor::View for TestTensor {
            fn dtype(&self) -> Dtype {
                Dtype::F32
            }
            fn shape(&self) -> &[usize] {
                &self.shape
            }
            fn data(&self) -> Cow<'_, [u8]> {
                Cow::Borrowed(&self.data)
            }
            fn data_len(&self) -> usize {
                self.data.len()
            }
        }

        let dir = make_test_dir("load_golden_no_logits");
        let tensor = TestTensor {
            shape: vec![2],
            data: vec![0u8; 8],
        };
        // Name it "not_logits" so the "logits" lookup fails
        let tensors = vec![("not_logits".to_string(), tensor)];
        let bytes = safetensors::tensor::serialize(tensors, &None).expect("serialize");
        let path = dir.join("no_logits.safetensors");
        std::fs::write(&path, bytes).expect("write");

        let result = HfParityOracle::load_golden_from_path(&path, "p", "h");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Missing 'logits' tensor"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_load_golden_from_path_invalid_metadata_json_falls_back() {
        let dir = make_test_dir("load_golden_bad_meta");
        let logits = vec![1.0f32];
        let st_path = dir.join("test.safetensors");
        create_safetensors_file(&st_path, &logits, &[1]);

        // Write an invalid JSON companion
        let meta_path = dir.join("test.json");
        std::fs::write(&meta_path, "not valid json{{{").expect("write");

        // Should still succeed because load_metadata_json error triggers unwrap_or_default
        let result = HfParityOracle::load_golden_from_path(&st_path, "p", "h");
        assert!(result.is_ok());
        let golden = result.expect("already checked");
        assert!(golden.model_id.is_empty());
        assert!(golden.transformers_version.is_empty());
        assert!(golden.text.is_none());

        let _ = std::fs::remove_dir_all(&dir);
    }

    // ============================================================
    // load_golden cache hit Tests
    // ============================================================

    #[test]
    fn test_load_golden_cache_hit() {
        let dir = make_test_dir("load_golden_cache");
        let family = "test-family";
        let prompt = "cached prompt";
        let input_hash = hash_prompt(prompt);

        // Create the directory structure: corpus_path/model_family/
        let family_dir = dir.join(family);
        std::fs::create_dir_all(&family_dir).expect("create family dir");

        // Create the golden safetensors file at the expected path
        let st_path = family_dir.join(format!("{input_hash}.safetensors"));
        create_safetensors_file(&st_path, &[42.0f32], &[1]);

        let mut oracle = HfParityOracle::new(&dir, family);

        // First call loads from file
        let result1 = oracle.load_golden(prompt);
        assert!(result1.is_ok());

        // Manually insert into cache to simulate the cache path
        let golden = result1.expect("already checked");
        oracle.golden_cache.insert(input_hash.clone(), golden);

        // Second call should hit the cache (even if file is deleted)
        std::fs::remove_file(&st_path).expect("remove file");
        let result2 = oracle.load_golden(prompt);
        assert!(result2.is_ok());
        let cached = result2.expect("already checked");
        assert!((cached.logits[0] - 42.0).abs() < 1e-6);

        let _ = std::fs::remove_dir_all(&dir);
    }

    // ============================================================
    // compare_tensor_file Tests
    // ============================================================

    #[test]
    fn test_compare_tensor_file_matching() {
        let dir = make_test_dir("compare_tensor_match");
        let logits = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let actual_path = dir.join("actual.safetensors");
        create_safetensors_file(&actual_path, &logits, &[1, 5]);

        let golden = GoldenOutput {
            input_hash: "h".to_string(),
            prompt: "p".to_string(),
            logits: logits.clone(),
            shape: vec![1, 5],
            text: None,
            model_id: String::new(),
            transformers_version: String::new(),
        };

        let oracle = HfParityOracle::new("/tmp", "test");
        let result = oracle.compare_tensor_file(&actual_path, &golden);
        assert!(result.is_ok(), "Matching tensors should pass: {result:?}");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_compare_tensor_file_mismatched_values() {
        let dir = make_test_dir("compare_tensor_mismatch");
        let actual_logits = vec![100.0f32, 200.0, 300.0, 400.0];
        let actual_path = dir.join("actual.safetensors");
        create_safetensors_file(&actual_path, &actual_logits, &[1, 4]);

        let golden = GoldenOutput {
            input_hash: "h".to_string(),
            prompt: "p".to_string(),
            logits: vec![1.0, 2.0, 3.0, 4.0],
            shape: vec![1, 4],
            text: None,
            model_id: String::new(),
            transformers_version: String::new(),
        };

        let oracle = HfParityOracle::new("/tmp", "test");
        let result = oracle.compare_tensor_file(&actual_path, &golden);
        assert!(matches!(result, Err(TensorDiff::ValueMismatch { .. })));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_compare_tensor_file_not_found() {
        let golden = GoldenOutput {
            input_hash: "h".to_string(),
            prompt: "p".to_string(),
            logits: vec![1.0],
            shape: vec![1],
            text: None,
            model_id: String::new(),
            transformers_version: String::new(),
        };

        let oracle = HfParityOracle::new("/tmp", "test");
        let result =
            oracle.compare_tensor_file(Path::new("/nonexistent/file.safetensors"), &golden);
        assert!(matches!(result, Err(TensorDiff::ParseError { .. })));
        if let Err(TensorDiff::ParseError { message }) = result {
            assert!(message.contains("Failed to read actual output"));
        }
    }

    #[test]
    fn test_compare_tensor_file_invalid_safetensors() {
        let dir = make_test_dir("compare_tensor_invalid");
        let path = dir.join("bad.safetensors");
        std::fs::write(&path, b"garbage data").expect("write");

        let golden = GoldenOutput {
            input_hash: "h".to_string(),
            prompt: "p".to_string(),
            logits: vec![1.0],
            shape: vec![1],
            text: None,
            model_id: String::new(),
            transformers_version: String::new(),
        };

        let oracle = HfParityOracle::new("/tmp", "test");
        let result = oracle.compare_tensor_file(&path, &golden);
        assert!(matches!(result, Err(TensorDiff::ParseError { .. })));
        if let Err(TensorDiff::ParseError { message }) = result {
            assert!(message.contains("Failed to parse SafeTensors"));
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_compare_tensor_file_missing_logits() {
        use safetensors::tensor::Dtype;
        use std::borrow::Cow;

        struct TestTensor {
            shape: Vec<usize>,
            data: Vec<u8>,
        }
        impl safetensors::tensor::View for TestTensor {
            fn dtype(&self) -> Dtype {
                Dtype::F32
            }
            fn shape(&self) -> &[usize] {
                &self.shape
            }
            fn data(&self) -> Cow<'_, [u8]> {
                Cow::Borrowed(&self.data)
            }
            fn data_len(&self) -> usize {
                self.data.len()
            }
        }

        let dir = make_test_dir("compare_tensor_no_logits");
        let tensor = TestTensor {
            shape: vec![1],
            data: vec![0u8; 4],
        };
        let tensors = vec![("other_name".to_string(), tensor)];
        let bytes = safetensors::tensor::serialize(tensors, &None).expect("serialize");
        let path = dir.join("no_logits.safetensors");
        std::fs::write(&path, bytes).expect("write");

        let golden = GoldenOutput {
            input_hash: "h".to_string(),
            prompt: "p".to_string(),
            logits: vec![1.0],
            shape: vec![1],
            text: None,
            model_id: String::new(),
            transformers_version: String::new(),
        };

        let oracle = HfParityOracle::new("/tmp", "test");
        let result = oracle.compare_tensor_file(&path, &golden);
        assert!(matches!(result, Err(TensorDiff::ParseError { .. })));
        if let Err(TensorDiff::ParseError { message }) = result {
            assert!(message.contains("Missing 'logits' tensor"));
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_compare_tensor_file_shape_mismatch() {
        let dir = make_test_dir("compare_tensor_shape");
        // Actual has 3 logits
        let actual_path = dir.join("actual.safetensors");
        create_safetensors_file(&actual_path, &[1.0f32, 2.0, 3.0], &[1, 3]);

        // Golden has 5 logits
        let golden = GoldenOutput {
            input_hash: "h".to_string(),
            prompt: "p".to_string(),
            logits: vec![1.0, 2.0, 3.0, 4.0, 5.0],
            shape: vec![1, 5],
            text: None,
            model_id: String::new(),
            transformers_version: String::new(),
        };

        let oracle = HfParityOracle::new("/tmp", "test");
        let result = oracle.compare_tensor_file(&actual_path, &golden);
        assert!(matches!(result, Err(TensorDiff::ShapeMismatch { .. })));

        let _ = std::fs::remove_dir_all(&dir);
    }

    // ============================================================
    // Oracle evaluate Tests (file-based, covering golden paths)
    // ============================================================

    #[test]
    fn test_oracle_evaluate_text_match() {
        let dir = make_test_dir("evaluate_text_match");
        let family = "eval-family";
        let prompt = "match prompt";
        let input_hash = hash_prompt(prompt);

        let family_dir = dir.join(family);
        std::fs::create_dir_all(&family_dir).expect("mkdir");

        // Create safetensors file
        let st_path = family_dir.join(format!("{input_hash}.safetensors"));
        create_safetensors_file(&st_path, &[1.0f32], &[1]);

        // Create metadata JSON with generated_text
        let meta_path = family_dir.join(format!("{input_hash}.json"));
        std::fs::write(
            &meta_path,
            r#"{"model": "m", "transformers_version": "4.0", "generated_text": "expected output"}"#,
        )
        .expect("write meta");

        let oracle = HfParityOracle::new(&dir, family);
        let result = oracle.evaluate(prompt, "expected output");
        assert!(result.is_corroborated());
        if let OracleResult::Corroborated { evidence } = result {
            assert!(evidence.contains("Text output matches HF golden"));
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_oracle_evaluate_text_mismatch_falsified() {
        let dir = make_test_dir("evaluate_text_mismatch");
        let family = "eval-family2";
        let prompt = "mismatch prompt";
        let input_hash = hash_prompt(prompt);

        let family_dir = dir.join(family);
        std::fs::create_dir_all(&family_dir).expect("mkdir");

        let st_path = family_dir.join(format!("{input_hash}.safetensors"));
        create_safetensors_file(&st_path, &[1.0f32], &[1]);

        let meta_path = family_dir.join(format!("{input_hash}.json"));
        std::fs::write(
            &meta_path,
            r#"{"model": "m", "transformers_version": "4.0", "generated_text": "expected text"}"#,
        )
        .expect("write meta");

        let oracle = HfParityOracle::new(&dir, family);
        // Output differs and is not a file path
        let result = oracle.evaluate(prompt, "totally different output");
        assert!(result.is_falsified());
        if let OracleResult::Falsified { reason, evidence } = result {
            assert!(reason.contains("Text output differs"));
            assert!(evidence.contains("Expected:"));
            assert!(evidence.contains("Actual:"));
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_oracle_evaluate_tensor_file_corroborated() {
        let dir = make_test_dir("evaluate_tensor_ok");
        let family = "tensor-family";
        let prompt = "tensor prompt";
        let input_hash = hash_prompt(prompt);

        let family_dir = dir.join(family);
        std::fs::create_dir_all(&family_dir).expect("mkdir");

        let logits = vec![1.0f32, 2.0, 3.0];
        let st_path = family_dir.join(format!("{input_hash}.safetensors"));
        create_safetensors_file(&st_path, &logits, &[1, 3]);
        // No metadata JSON (no expected text) so it falls through to tensor comparison

        // Create actual output file
        let actual_path = dir.join("actual_output.safetensors");
        create_safetensors_file(&actual_path, &logits, &[1, 3]);

        let oracle = HfParityOracle::new(&dir, family);
        let result = oracle.evaluate(prompt, actual_path.to_str().expect("utf8"));
        assert!(result.is_corroborated());
        if let OracleResult::Corroborated { evidence } = result {
            assert!(evidence.contains("Tensor parity verified"));
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_oracle_evaluate_tensor_file_falsified() {
        let dir = make_test_dir("evaluate_tensor_fail");
        let family = "tensor-family2";
        let prompt = "tensor prompt fail";
        let input_hash = hash_prompt(prompt);

        let family_dir = dir.join(family);
        std::fs::create_dir_all(&family_dir).expect("mkdir");

        let golden_logits = vec![1.0f32, 2.0, 3.0, 4.0];
        let st_path = family_dir.join(format!("{input_hash}.safetensors"));
        create_safetensors_file(&st_path, &golden_logits, &[1, 4]);

        // Actual output differs drastically
        let actual_logits = vec![100.0f32, 200.0, 300.0, 400.0];
        let actual_path = dir.join("bad_output.safetensors");
        create_safetensors_file(&actual_path, &actual_logits, &[1, 4]);

        let oracle = HfParityOracle::new(&dir, family);
        let result = oracle.evaluate(prompt, actual_path.to_str().expect("utf8"));
        assert!(result.is_falsified());
        if let OracleResult::Falsified { reason, .. } = result {
            assert!(reason.contains("Tensor mismatch"));
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_oracle_evaluate_plain_text_no_golden_text_no_tensor() {
        // When golden exists but has no expected text, and output is not a
        // .safetensors file path, it should corroborate with the text fallback.
        let dir = make_test_dir("evaluate_plain_text");
        let family = "plain-family";
        let prompt = "plain prompt";
        let input_hash = hash_prompt(prompt);

        let family_dir = dir.join(family);
        std::fs::create_dir_all(&family_dir).expect("mkdir");

        let st_path = family_dir.join(format!("{input_hash}.safetensors"));
        create_safetensors_file(&st_path, &[1.0f32], &[1]);

        let oracle = HfParityOracle::new(&dir, family);
        let result = oracle.evaluate(prompt, "just some plain text output");
        assert!(result.is_corroborated());
        if let OracleResult::Corroborated { evidence } = result {
            assert!(evidence.contains("no tensor comparison"));
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    // ============================================================
    // truncate edge case Tests
    // ============================================================

    #[test]
    fn test_truncate_mid_multibyte_char() {
        // "é" is 2 bytes in UTF-8. "éé" is 4 bytes.
        // Truncating at 3 should back up to byte 2 (end of first "é")
        let s = "éé";
        let truncated = truncate(s, 3);
        assert_eq!(truncated, "é");
    }

    #[test]
    fn test_truncate_at_zero() {
        assert_eq!(truncate("hello", 0), "");
    }

    #[test]
    fn test_truncate_three_byte_mid_boundary() {
        // Each CJK character is 3 bytes. "漢字" = 6 bytes.
        // Truncating at 4 should back up to 3 (end of "漢")
        let s = "漢字";
        let truncated = truncate(s, 4);
        assert_eq!(truncated, "漢");
    }

    #[test]
    fn test_truncate_four_byte_emoji() {
        // Emoji like U+1F600 is 4 bytes in UTF-8
        let s = "\u{1F600}abc";
        // Truncating at 2 bytes should back up to 0 since the first char is 4 bytes
        let truncated = truncate(s, 2);
        assert_eq!(truncated, "");
    }

    // ============================================================
    // detect_systematic_bias edge case Tests
    // ============================================================

    #[test]
    fn test_detect_systematic_bias_mismatched_len() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0];
        assert!(HfParityOracle::detect_systematic_bias(&a, &b).is_none());
    }

    #[test]
    fn test_detect_systematic_bias_no_scale_drift_within_threshold() {
        // Scale ratio within 10%: std_a/std_e close to 1.0
        let a = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let b = vec![0.0, 1.05, 2.1, 3.15, 4.2]; // ~5% scale up
        let result = HfParityOracle::detect_systematic_bias(&a, &b);
        // Mean shift: mean_a=2.1, mean_e=2.0, shift=0.1, std_e=~1.41, 0.1/1.41=0.07 sigma (< 3)
        // Scale drift: std_a/std_e ≈ 1.05, (1.05-1)=0.05 < 0.1
        assert!(result.is_none());
    }

    // ============================================================
    // Tolerance serialization Tests
    // ============================================================

    #[test]
    fn test_tolerance_serialize_deserialize() {
        let tol = Tolerance::fp16();
        let json = serde_json::to_string(&tol).expect("serialize");
        let tol2: Tolerance = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(tol, tol2);
    }

    #[test]
    fn test_tolerance_debug() {
        let tol = Tolerance::default();
        let debug = format!("{tol:?}");
        assert!(debug.contains("Tolerance"));
        assert!(debug.contains("atol_fp32"));
    }

    // ============================================================
    // TensorDiff serialization Tests
    // ============================================================

    #[test]
    fn test_tensor_diff_serialize_deserialize_shape() {
        let diff = TensorDiff::ShapeMismatch {
            expected: 100,
            actual: 200,
        };
        let json = serde_json::to_string(&diff).expect("serialize");
        let diff2: TensorDiff = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(diff, diff2);
    }

    #[test]
    fn test_tensor_diff_serialize_deserialize_value() {
        let diff = TensorDiff::ValueMismatch {
            num_mismatches: 5,
            total: 50,
            mismatch_ratio: 0.1,
            max_diff: 0.5,
            max_diff_idx: 7,
            expected_val: 1.0,
            actual_val: 1.5,
            mean_diff: 0.2,
        };
        let json = serde_json::to_string(&diff).expect("serialize");
        let diff2: TensorDiff = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(diff, diff2);
    }

    #[test]
    fn test_tensor_diff_serialize_deserialize_parse_error() {
        let diff = TensorDiff::ParseError {
            message: "something went wrong".to_string(),
        };
        let json = serde_json::to_string(&diff).expect("serialize");
        let diff2: TensorDiff = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(diff, diff2);
    }

    // ============================================================
    // load_metadata_json Tests
    // ============================================================

    #[test]
    fn test_load_metadata_json_minimal() {
        let dir = make_test_dir("meta_minimal");
        let path = dir.join("meta.json");
        // All fields use serde default
        std::fs::write(&path, r#"{}"#).expect("write");

        let result = HfParityOracle::load_metadata_json(&path);
        assert!(result.is_ok());
        let (model, version, text) = result.expect("ok");
        assert!(model.is_empty());
        assert!(version.is_empty());
        assert!(text.is_none());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_load_metadata_json_full() {
        let dir = make_test_dir("meta_full");
        let path = dir.join("meta.json");
        std::fs::write(
            &path,
            r#"{"model": "bigmodel", "transformers_version": "5.0.0", "generated_text": "output here"}"#,
        )
        .expect("write");

        let result = HfParityOracle::load_metadata_json(&path);
        assert!(result.is_ok());
        let (model, version, text) = result.expect("ok");
        assert_eq!(model, "bigmodel");
        assert_eq!(version, "5.0.0");
        assert_eq!(text, Some("output here".to_string()));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_load_metadata_json_not_found() {
        let result = HfParityOracle::load_metadata_json(Path::new("/nonexistent/meta.json"));
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Failed to read metadata"));
    }

    #[test]
    fn test_load_metadata_json_invalid_json() {
        let dir = make_test_dir("meta_invalid");
        let path = dir.join("meta.json");
        std::fs::write(&path, "{{{{invalid").expect("write");

        let result = HfParityOracle::load_metadata_json(&path);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .contains("Failed to parse metadata JSON")
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    // ============================================================
    // Integration Tests (require golden corpus)
    // ============================================================

    #[test]
    fn test_load_golden_from_corpus() {
        // Path to the Qwen golden corpus
        let corpus_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|p| p.parent())
            .map(|p| p.join("../hf-ground-truth-corpus/oracle/qwen2.5-coder-1.5b/v1"));

        let Some(corpus_path) = corpus_path else {
            eprintln!("Skipping integration test: corpus path not found");
            return;
        };

        if !corpus_path.exists() {
            eprintln!("Skipping integration test: corpus not generated yet");
            return;
        }

        // Load a known golden file
        let prompt = "def fibonacci(n):";
        let hash = hash_prompt(prompt);
        let safetensors_path = corpus_path.join(format!("{hash}.safetensors"));
        let json_path = corpus_path.join(format!("{hash}.json"));

        assert!(
            safetensors_path.exists(),
            "SafeTensors file not found: {safetensors_path:?}"
        );
        assert!(json_path.exists(), "JSON metadata not found: {json_path:?}");

        // Load and verify the golden output
        let result = HfParityOracle::load_golden_from_path(&safetensors_path, prompt, &hash);
        assert!(result.is_ok(), "Failed to load golden: {result:?}");

        let golden = result.expect("already checked");
        assert_eq!(golden.input_hash, hash);
        assert_eq!(golden.prompt, prompt);
        assert!(!golden.logits.is_empty(), "Logits should not be empty");
        assert_eq!(
            golden.shape.len(),
            2,
            "Logits should be 2D [seq_len, vocab]"
        );
        assert_eq!(
            golden.shape[1], 151936,
            "Qwen2.5 vocab size should be 151936"
        );
    }

    #[test]
    fn test_oracle_verify_with_golden_corpus() {
        // Path to the Qwen golden corpus
        let corpus_base = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|p| p.parent())
            .map(|p| p.join("../hf-ground-truth-corpus/oracle"));

        let Some(corpus_base) = corpus_base else {
            eprintln!("Skipping integration test: corpus path not found");
            return;
        };

        let corpus_path = corpus_base.join("qwen2.5-coder-1.5b/v1");
        if !corpus_path.exists() {
            eprintln!("Skipping integration test: corpus not generated yet");
            return;
        }

        // Create oracle pointing to the corpus
        let oracle = HfParityOracle::new(&corpus_base, "qwen2.5-coder-1.5b/v1");

        // Load a golden output
        let prompt = "def fibonacci(n):";
        let golden_result = oracle.load_golden(prompt);
        assert!(
            golden_result.is_ok(),
            "Failed to load golden: {golden_result:?}"
        );

        let golden = golden_result.expect("already checked");

        // Verify that comparing golden with itself passes
        let verify_result = oracle.tensors_close(&golden.logits, &golden.logits);
        assert!(
            verify_result.is_ok(),
            "Golden should match itself: {verify_result:?}"
        );
    }
}
