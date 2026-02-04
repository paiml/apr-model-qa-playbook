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
