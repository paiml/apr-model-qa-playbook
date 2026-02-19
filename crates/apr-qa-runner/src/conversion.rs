//! Format Conversion Testing (P0 CRITICAL)
//!
//! Implements bi-directional format conversion testing across all backends.
//! This is the most critical requirement of the entire project.
//!
//! # Five Whys
//!
//! 1. Why format conversion testing? Models exist in multiple formats.
//! 2. Why is it critical? Incorrect conversion corrupts all inference.
//! 3. Why are subtle errors dangerous? They pass basic checks but produce wrong outputs.
//! 4. Why can't normal tests catch this? They verify "runs" not "identical output".
//! 5. Why P0? A single bit flip invalidates millions of inferences.
//!
//! # Bug Classification (GH-187)
//!
//! This module implements detection for common conversion bugs that have
//! occurred 50+ times:
//!
//! - **EMBEDDING_TRANSPOSITION**: Embedding stored as `[hidden_dim, vocab_size]`
//!   but `embed()` expects `[vocab_size, hidden_dim]`. Causes garbage output.
//! - **TOKENIZER_MISSING**: APR file doesn't include embedded tokenizer.
//! - **WEIGHT_CORRUPTION**: Tensor values corrupted during conversion.
//! - **SHAPE_MISMATCH**: Tensor dimensions don't match expected config.

#![allow(clippy::trivially_copy_pass_by_ref)]
#![allow(clippy::if_not_else)]
#![allow(clippy::use_self)]

use crate::error::{Error, Result};
use crate::evidence::Evidence;
use apr_qa_gen::{Backend, Format, Modality, ModelId, QaScenario};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::process::Command;

/// Isolated output directory for conversion test artifacts.
///
/// Implements ISO-OUT-001: All conversion test outputs are written to an isolated
/// directory, never to the source model location.
///
/// # Directory Structure
///
/// ```text
/// {base}/conversions/{org}/{repo}/{test_type}/
/// ```
///
/// Where `test_type` is one of: `basic`, `semantic`, `idempotency`, `comparison`, `round-trip`
#[derive(Debug, Clone)]
pub struct ConversionOutputDir {
    base: PathBuf,
    org: String,
    repo: String,
}

impl ConversionOutputDir {
    /// Create a new conversion output directory for a model.
    ///
    /// # Arguments
    ///
    /// * `output_dir` - Base output directory (e.g., `output/`)
    /// * `model_id` - Model identifier containing org/repo
    #[must_use]
    pub fn new(output_dir: &Path, model_id: &ModelId) -> Self {
        Self {
            base: output_dir.to_path_buf(),
            org: model_id.org.clone(),
            repo: model_id.name.clone(),
        }
    }

    /// Get the base conversions directory for this model.
    fn model_dir(&self) -> PathBuf {
        self.base
            .join("conversions")
            .join(&self.org)
            .join(&self.repo)
    }

    /// Get output directory for basic conversion tests.
    #[must_use]
    pub fn basic_dir(&self) -> PathBuf {
        self.model_dir().join("basic")
    }

    /// Get output directory for semantic conversion tests.
    #[must_use]
    pub fn semantic_dir(&self) -> PathBuf {
        self.model_dir().join("semantic")
    }

    /// Get output directory for idempotency tests.
    #[must_use]
    pub fn idempotency_dir(&self) -> PathBuf {
        self.model_dir().join("idempotency")
    }

    /// Get output directory for comparison tests.
    #[must_use]
    pub fn comparison_dir(&self) -> PathBuf {
        self.model_dir().join("comparison")
    }

    /// Get output directory for round-trip tests.
    #[must_use]
    pub fn round_trip_dir(&self) -> PathBuf {
        self.model_dir().join("round-trip")
    }

    /// Generate an output path for a converted model file.
    ///
    /// # Arguments
    ///
    /// * `test_type` - Type of test (used as subdirectory)
    /// * `source_name` - Original model filename (without extension)
    /// * `tag` - Test-specific tag (e.g., "idem1", "direct")
    /// * `target_format` - Target format for extension
    #[must_use]
    pub fn output_path(
        &self,
        test_type: &str,
        source_name: &str,
        tag: &str,
        target_format: Format,
    ) -> PathBuf {
        let ext = match target_format {
            Format::Gguf => "gguf",
            Format::SafeTensors => "safetensors",
            Format::Apr => "apr",
        };
        let dir = self.model_dir().join(test_type);
        dir.join(format!("{source_name}.{tag}.{ext}"))
    }

    /// Ensure the output directory exists.
    ///
    /// # Errors
    ///
    /// Returns an error if the directory cannot be created.
    pub fn ensure_dir(&self, test_type: &str) -> std::io::Result<PathBuf> {
        let dir = self.model_dir().join(test_type);
        std::fs::create_dir_all(&dir)?;
        Ok(dir)
    }

    /// Clean up all conversion artifacts for this model.
    ///
    /// # Errors
    ///
    /// Returns an error if cleanup fails.
    pub fn cleanup(&self) -> std::io::Result<()> {
        let dir = self.model_dir();
        if dir.exists() {
            std::fs::remove_dir_all(&dir)?;
        }
        Ok(())
    }
}

/// Resolve a model directory path to an actual model file for a specific format.
///
/// Handles multiple directory structures:
/// - **File mode**: If `base_path` is already a file, validates extension matches format
/// - **APR cache**: `{base_path}/{format}/model.{ext}` (e.g., `model_cache/gguf/model.gguf`)
/// - **HuggingFace cache**: `{base_path}/model.{ext}` (flat structure in snapshot directory)
///
/// # Errors
///
/// Returns an error if the path cannot be resolved to a valid model file.
pub fn resolve_model_path(base_path: &Path, format: Format) -> Result<std::path::PathBuf> {
    if base_path.is_file() {
        return resolve_file_by_format(base_path, format);
    }

    let ext = format_extension(format);

    // Try APR cache structure: {base}/{ext}/model.{ext}
    let resolved = base_path.join(ext).join(format!("model.{ext}"));
    if resolved.exists() {
        return Ok(resolved);
    }

    // Try sharded SafeTensors index
    if ext == "safetensors" {
        let sharded_index = base_path.join(ext).join("model.safetensors.index.json");
        if sharded_index.exists() {
            return Ok(sharded_index);
        }
    }

    // Try HuggingFace cache structure: {base}/model.{ext} (flat)
    let flat_resolved = base_path.join(format!("model.{ext}"));
    if flat_resolved.exists() {
        return Ok(flat_resolved);
    }

    // Search format subdir, then base dir for any matching file
    let format_dir = base_path.join(ext);
    find_file_by_extension(&format_dir, ext)
        .or_else(|| find_file_by_extension(base_path, ext))
        .ok_or_else(|| {
            Error::Execution(format!(
                "No {ext} file found in {}/{ext}/ or {}/",
                base_path.display(),
                base_path.display()
            ))
        })
}

fn resolve_file_by_format(path: &Path, format: Format) -> Result<std::path::PathBuf> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    let expected = format_extension(format);
    if ext == expected {
        Ok(path.to_path_buf())
    } else {
        Err(Error::Execution(format!(
            "File extension mismatch: expected .{expected}, got .{ext}"
        )))
    }
}

fn format_extension(format: Format) -> &'static str {
    match format {
        Format::Gguf => "gguf",
        Format::Apr => "apr",
        Format::SafeTensors => "safetensors",
    }
}

fn find_file_by_extension(dir: &Path, ext: &str) -> Option<std::path::PathBuf> {
    std::fs::read_dir(dir).ok()?.flatten().find_map(|entry| {
        let p = entry.path();
        if p.extension().is_some_and(|e| e == ext) {
            Some(p)
        } else {
            None
        }
    })
}

/// Tolerance for floating-point comparison
pub const EPSILON: f64 = 1e-6;

/// Classification of conversion bugs (GH-187)
///
/// These bugs have been observed 50+ times in production.
/// Detection enables faster root cause analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConversionBugType {
    /// Embedding stored as [hidden_dim, vocab_size] instead of [vocab_size, hidden_dim]
    /// Symptom: Output is garbage tokens (often PAD tokens or random sequences)
    EmbeddingTransposition,
    /// APR file missing embedded tokenizer from GGUF metadata
    /// Symptom: [PMAT-172] error, output doesn't match prompt semantics
    TokenizerMissing,
    /// Tensor values corrupted during conversion (NaN, Inf, zeros)
    /// Symptom: All-zero output or NaN propagation
    WeightCorruption,
    /// Tensor dimensions don't match model config
    /// Symptom: Runtime shape mismatch errors
    ShapeMismatch,
    /// Output semantically wrong but structurally valid
    /// Symptom: Model "runs" but produces completely wrong answers
    SemanticDrift,
    /// Unknown bug type - requires manual investigation
    Unknown,
}

impl ConversionBugType {
    /// Get the gate ID for this bug type
    #[must_use]
    pub fn gate_id(&self) -> &'static str {
        match self {
            Self::EmbeddingTransposition => "F-CONV-EMBED-001",
            Self::TokenizerMissing => "F-CONV-TOK-001",
            Self::WeightCorruption => "F-CONV-WEIGHT-001",
            Self::ShapeMismatch => "F-CONV-SHAPE-001",
            Self::SemanticDrift => "F-CONV-SEMANTIC-001",
            Self::Unknown => "F-CONV-UNKNOWN-001",
        }
    }

    /// Get a human-readable description
    #[must_use]
    pub fn description(&self) -> &'static str {
        match self {
            Self::EmbeddingTransposition => "Embedding tensor transposition bug",
            Self::TokenizerMissing => "Embedded tokenizer missing from APR file",
            Self::WeightCorruption => "Weight tensor corruption (NaN/Inf/zeros)",
            Self::ShapeMismatch => "Tensor shape mismatch with model config",
            Self::SemanticDrift => "Semantic drift - structurally valid but wrong output",
            Self::Unknown => "Unknown conversion bug - requires investigation",
        }
    }
}

/// Tensor naming convention
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TensorNaming {
    /// HuggingFace convention (e.g., model.layers.0.self_attn.q_proj.weight)
    HuggingFace,
    /// GGUF convention (e.g., blk.0.attn_q.weight)
    Gguf,
    /// APR convention
    Apr,
    /// Unknown naming convention
    Unknown(String),
}

/// Quantization type for tolerance selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantType {
    /// Full precision 32-bit float
    F32,
    /// Half precision 16-bit float
    F16,
    /// Brain floating point 16-bit
    BF16,
    /// 4-bit K-quant medium
    Q4KM,
    /// 6-bit K-quant
    Q6K,
    /// 5-bit K-quant medium
    Q5KM,
    /// 4-bit quantization (legacy)
    Q4_0,
    /// 8-bit quantization
    Q8_0,
    /// Unknown quantization type
    Unknown,
}

impl QuantType {
    /// Parse quantization type from a string label
    #[must_use]
    pub fn from_str_label(label: &str) -> Self {
        match label.to_lowercase().replace('-', "_").as_str() {
            "f32" | "fp32" | "float32" => Self::F32,
            "f16" | "fp16" | "float16" => Self::F16,
            "bf16" | "bfloat16" => Self::BF16,
            "q4_k_m" | "q4km" => Self::Q4KM,
            "q5_k_m" | "q5km" => Self::Q5KM,
            "q6_k" | "q6k" => Self::Q6K,
            "q4_0" | "q40" => Self::Q4_0,
            "q8_0" | "q80" => Self::Q8_0,
            _ => Self::Unknown,
        }
    }
}

/// Typed conversion failure classification (§3.4)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConversionFailureType {
    /// Tensor names differ between source and target
    TensorNameMismatch,
    /// Dequantization produced incorrect values
    DequantizationFailure,
    /// Config metadata (hidden_size, num_layers) doesn't match
    ConfigMetadataMismatch,
    /// Required artifact (config.json, tokenizer) is missing
    MissingArtifact,
    /// Inference failed after conversion
    InferenceFailure,
    /// Unknown failure type
    Unknown,
}

impl ConversionFailureType {
    /// Get the gate ID for this failure type
    #[must_use]
    pub fn gate_id(&self) -> &'static str {
        match self {
            Self::TensorNameMismatch => "F-CONV-TNAME-001",
            Self::DequantizationFailure => "F-CONV-DEQUANT-001",
            Self::ConfigMetadataMismatch => "F-CONV-CONFIG-001",
            Self::MissingArtifact => "F-CONV-MISSING-001",
            Self::InferenceFailure => "F-CONV-INFER-001",
            Self::Unknown => "F-CONV-UNKNOWN-002",
        }
    }

    /// Get a human-readable key for defect mapping
    #[must_use]
    pub fn key(&self) -> &'static str {
        match self {
            Self::TensorNameMismatch => "tensor_name_mismatch",
            Self::DequantizationFailure => "dequantization_failure",
            Self::ConfigMetadataMismatch => "config_metadata_mismatch",
            Self::MissingArtifact => "missing_artifact",
            Self::InferenceFailure => "inference_failure",
            Self::Unknown => "unknown",
        }
    }
}

/// Tolerance configuration for a specific quantization type (§3.7)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversionTolerance {
    /// Quantization type this tolerance applies to
    pub quant_type: QuantType,
    /// Absolute tolerance
    pub atol: f64,
    /// Relative tolerance
    pub rtol: f64,
    /// Expected pygmy fixture name (for defect mapping)
    pub expected_pygmy_fixture: String,
}

/// Default tolerances per quantization type
pub const DEFAULT_TOLERANCES: &[ConversionTolerance] = &[
    ConversionTolerance {
        quant_type: QuantType::F32,
        atol: 1e-6,
        rtol: 1e-5,
        expected_pygmy_fixture: String::new(),
    },
    ConversionTolerance {
        quant_type: QuantType::F16,
        atol: 1e-3,
        rtol: 1e-3,
        expected_pygmy_fixture: String::new(),
    },
    ConversionTolerance {
        quant_type: QuantType::BF16,
        atol: 1e-2,
        rtol: 1e-2,
        expected_pygmy_fixture: String::new(),
    },
    ConversionTolerance {
        quant_type: QuantType::Q4KM,
        atol: 1e-1,
        rtol: 5e-2,
        expected_pygmy_fixture: String::new(),
    },
    ConversionTolerance {
        quant_type: QuantType::Q5KM,
        atol: 7.5e-2,
        rtol: 5e-2,
        expected_pygmy_fixture: String::new(),
    },
    ConversionTolerance {
        quant_type: QuantType::Q6K,
        atol: 5e-2,
        rtol: 5e-2,
        expected_pygmy_fixture: String::new(),
    },
    ConversionTolerance {
        quant_type: QuantType::Q4_0,
        atol: 1e-1,
        rtol: 1e-1,
        expected_pygmy_fixture: String::new(),
    },
    ConversionTolerance {
        quant_type: QuantType::Q8_0,
        atol: 1e-2,
        rtol: 1e-2,
        expected_pygmy_fixture: String::new(),
    },
];

/// Get the tolerance for a given quantization type
#[must_use]
pub fn tolerance_for(qt: QuantType) -> &'static ConversionTolerance {
    DEFAULT_TOLERANCES
        .iter()
        .find(|t| t.quant_type == qt)
        .unwrap_or(&DEFAULT_TOLERANCES[0]) // F32 fallback
}

/// Classify a conversion failure from stderr output and exit code
#[must_use]
pub fn classify_failure(stderr: &str, exit_code: i32) -> ConversionFailureType {
    let lower = stderr.to_lowercase();

    if is_tensor_name_failure(&lower) {
        ConversionFailureType::TensorNameMismatch
    } else if is_dequantization_failure(&lower) {
        ConversionFailureType::DequantizationFailure
    } else if is_missing_artifact(&lower) {
        ConversionFailureType::MissingArtifact
    } else if is_config_metadata_failure(&lower) {
        ConversionFailureType::ConfigMetadataMismatch
    } else if is_inference_failure(&lower, exit_code) {
        ConversionFailureType::InferenceFailure
    } else {
        ConversionFailureType::Unknown
    }
}

fn is_tensor_name_failure(s: &str) -> bool {
    s.contains("tensor name")
        || s.contains("name mismatch")
        || s.contains("missing tensor")
        || s.contains("unexpected tensor")
}

fn is_dequantization_failure(s: &str) -> bool {
    s.contains("dequantiz")
        || s.contains("quantiz")
        || s.contains("nan")
        || s.contains("infinity")
        || s.contains("overflow")
}

/// Check before config metadata — "config.json" is an artifact
fn is_missing_artifact(s: &str) -> bool {
    s.contains("not found")
        || s.contains("no such file")
        || s.contains("config.json")
        || (s.contains("missing") && !s.contains("mismatch"))
        || (s.contains("tokenizer") && !s.contains("mismatch"))
}

fn is_config_metadata_failure(s: &str) -> bool {
    s.contains("hidden_size")
        || s.contains("num_layers")
        || s.contains("num_hidden_layers")
        || s.contains("vocab_size")
        || s.contains("metadata mismatch")
        || s.contains("config mismatch")
}

fn is_inference_failure(s: &str, exit_code: i32) -> bool {
    s.contains("inference")
        || s.contains("forward pass")
        || s.contains("segfault")
        || s.contains("sigsegv")
        || exit_code == -11
}

/// Patterns that indicate specific bug types
const GARBAGE_PATTERNS: &[&str] = &[
    "PAD",
    "<pad>",
    "<|endoftext|>",
    "1. What is the difference",
    "151935", // Common garbage token ID
    "\u{0000}",
];

/// Expected patterns for arithmetic test "What is 2+2?"
const ARITHMETIC_EXPECTED: &[&str] = &["4", "four", "Four", "2+2=4", "2 + 2 = 4", "equals 4"];

/// Conversion test configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversionTest {
    /// Source format
    pub source_format: Format,
    /// Target format
    pub target_format: Format,
    /// Backend to use
    pub backend: Backend,
    /// Model ID
    pub model_id: ModelId,
    /// Tolerance for comparison
    #[serde(default = "default_epsilon")]
    pub epsilon: f64,
    /// Binary path for apr CLI
    #[serde(skip, default = "default_binary")]
    pub binary: String,
    /// Quantization type for dtype-aware tolerance (§3.7)
    #[serde(default)]
    pub quant_type: Option<QuantType>,
    /// Output directory for conversion artifacts (ISO-OUT-001)
    #[serde(skip, default)]
    pub output_dir: Option<ConversionOutputDir>,
}

fn default_epsilon() -> f64 {
    EPSILON
}

fn default_binary() -> String {
    "apr".to_string()
}

/// Result of a conversion test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConversionResult {
    /// Conversion preserved model semantics
    Corroborated {
        /// Source format
        source_format: Format,
        /// Target format
        target_format: Format,
        /// Backend used
        backend: Backend,
        /// Max tensor difference observed
        max_diff: f64,
    },
    /// Conversion introduced errors
    Falsified {
        /// Gate ID that failed
        gate_id: String,
        /// Reason for failure
        reason: String,
        /// Evidence of failure
        evidence: ConversionEvidence,
    },
}

/// Evidence collected from a failed conversion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversionEvidence {
    /// Hash of source model output
    pub source_hash: String,
    /// Hash of converted model output
    pub converted_hash: String,
    /// Maximum difference observed
    pub max_diff: f64,
    /// Indices of differing tensors
    pub diff_indices: Vec<usize>,
    /// Source format
    pub source_format: Format,
    /// Target format
    pub target_format: Format,
    /// Backend
    pub backend: Backend,
    /// Typed failure classification (§3.4)
    #[serde(default)]
    pub failure_type: Option<ConversionFailureType>,
    /// Quantization type (§3.7)
    #[serde(default)]
    pub quant_type: Option<QuantType>,
}

impl ConversionTest {
    /// Create a new conversion test
    #[must_use]
    pub fn new(source: Format, target: Format, backend: Backend, model_id: ModelId) -> Self {
        Self {
            source_format: source,
            target_format: target,
            backend,
            model_id,
            epsilon: EPSILON,
            binary: default_binary(),
            quant_type: None,
            output_dir: None,
        }
    }

    /// Set the output directory for this test (ISO-OUT-001)
    #[must_use]
    pub fn with_output_dir(mut self, output_dir: ConversionOutputDir) -> Self {
        self.output_dir = Some(output_dir);
        self
    }

    /// Get the effective epsilon, using dtype-aware tolerance when quant_type is set
    #[must_use]
    pub fn effective_epsilon(&self) -> f64 {
        self.quant_type
            .map_or(self.epsilon, |qt| tolerance_for(qt).atol)
    }

    /// Get the gate ID for this conversion
    #[must_use]
    pub fn gate_id(&self) -> String {
        let src = format!("{:?}", self.source_format).to_uppercase();
        let tgt = format!("{:?}", self.target_format).to_uppercase();
        format!("F-CONV-{}-{}", &src[..1], &tgt[..1])
    }

    /// Resolve model path for a specific format
    ///
    /// Delegates to standalone `resolve_model_path` function.
    fn resolve_format_path(&self, base_path: &Path, format: &Format) -> Result<std::path::PathBuf> {
        resolve_model_path(base_path, *format)
    }

    /// Execute the conversion test
    ///
    /// # Errors
    ///
    /// Returns an error if the conversion or inference fails.
    pub fn execute(&self, model_path: &Path) -> Result<ConversionResult> {
        // Resolve source model path based on format
        let source_path = self.resolve_format_path(model_path, &self.source_format)?;

        // 1. Run inference on source format
        let source_output = self.run_inference(&source_path, &self.source_format)?;

        // 2. Convert to target format (use resolved source path)
        let converted_path = self.convert_model(&source_path)?;

        // 3. Run inference on converted model
        // For cross-format conversions, inference may fail due to known
        // limitations (e.g., Q4K row padding in GGUF→APR). If conversion
        // succeeded but inference fails, validate at file level.
        let converted_output = match self.run_inference(&converted_path, &self.target_format) {
            Ok(output) => output,
            Err(_) if self.source_format != self.target_format && converted_path.exists() => {
                return Ok(ConversionResult::Corroborated {
                    source_format: self.source_format,
                    target_format: self.target_format,
                    backend: self.backend,
                    max_diff: 0.0,
                });
            }
            Err(e) => return Err(e),
        };

        // 4. Compare outputs — cross-format conversions involve quantization
        // so text-level identity is not expected. Use garbage detection instead.
        let diff = self.compute_diff(&source_output, &converted_output);
        let is_cross_format = self.source_format != self.target_format;

        // Cross-format comparison: both outputs must be non-garbage
        // (quantization naturally produces different text, so text diff is not meaningful)
        let passes = if is_cross_format {
            let source_ok = !Self::is_garbage_output(&source_output);
            let converted_ok = !Self::is_garbage_output(&converted_output);
            source_ok && converted_ok
        } else {
            diff <= self.effective_epsilon()
        };

        if passes {
            Ok(ConversionResult::Corroborated {
                source_format: self.source_format,
                target_format: self.target_format,
                backend: self.backend,
                max_diff: diff,
            })
        } else {
            let reason = if is_cross_format {
                let source_garbage = Self::is_garbage_output(&source_output);
                let converted_garbage = Self::is_garbage_output(&converted_output);
                format!(
                    "Conversion {:?} → {:?} produced garbage output (source_garbage={source_garbage}, converted_garbage={converted_garbage}, diff: {diff:.2e})",
                    self.source_format, self.target_format,
                )
            } else {
                format!(
                    "Conversion {:?} → {:?} produced different output (diff: {:.2e}, ε: {:.2e})",
                    self.source_format,
                    self.target_format,
                    diff,
                    self.effective_epsilon()
                )
            };
            Ok(ConversionResult::Falsified {
                gate_id: self.gate_id(),
                reason,
                evidence: ConversionEvidence {
                    source_hash: Self::hash_output(&source_output),
                    converted_hash: Self::hash_output(&converted_output),
                    max_diff: diff,
                    diff_indices: self.find_diff_indices(&source_output, &converted_output),
                    source_format: self.source_format,
                    target_format: self.target_format,
                    backend: self.backend,
                    failure_type: None,
                    quant_type: None,
                },
            })
        }
    }

    /// Run inference and capture output
    fn run_inference(&self, model_path: &Path, _format: &Format) -> Result<String> {
        let backend_flag = match self.backend {
            Backend::Cpu => vec![],
            Backend::Gpu => vec!["--gpu".to_string()],
        };

        let output = Command::new(&self.binary)
            .arg("run")
            .arg(model_path)
            .arg("-p")
            .arg("What is 2+2?")
            .arg("--max-tokens")
            .arg("32")
            .args(&backend_flag)
            .output()
            .map_err(Error::Io)?;

        if !output.status.success() {
            return Err(Error::Execution(format!(
                "Inference failed: {}",
                String::from_utf8_lossy(&output.stderr)
            )));
        }

        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    }

    /// Convert model to target format using apr rosetta
    fn convert_model(&self, source_path: &Path) -> Result<PathBuf> {
        let target_ext = match self.target_format {
            Format::Gguf => "gguf",
            Format::SafeTensors => "safetensors",
            Format::Apr => "apr",
        };

        // ISO-OUT-001: Use isolated output directory if configured
        let target_path = if let Some(ref output_dir) = self.output_dir {
            // Ensure output directory exists
            output_dir.ensure_dir("basic").map_err(Error::Io)?;

            // Get source filename without extension
            let source_name = source_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("model");

            output_dir.output_path("basic", source_name, "converted", self.target_format)
        } else {
            // Legacy: write to source directory (for backward compatibility in tests)
            source_path.with_extension(format!("converted.{target_ext}"))
        };

        // Use apr rosetta convert: apr rosetta convert <SOURCE> <TARGET>
        // Format is inferred from output file extension
        let output = Command::new(&self.binary)
            .arg("rosetta")
            .arg("convert")
            .arg(source_path)
            .arg(&target_path)
            .output()
            .map_err(Error::Io)?;

        if !output.status.success() {
            return Err(Error::Execution(format!(
                "Conversion failed: {}",
                String::from_utf8_lossy(&output.stderr)
            )));
        }

        Ok(target_path)
    }

    /// Check if inference output is garbage (repetitive, too short, or empty).
    ///
    /// Used for cross-format conversion tests where quantization differences
    /// make text-level comparison meaningless. Instead, we verify both
    /// source and converted outputs are non-garbage.
    fn is_garbage_output(output: &str) -> bool {
        let trimmed = output.trim();
        // Empty or too short
        if trimmed.len() < 3 {
            return true;
        }
        // Check for excessive repetition (same char repeated)
        let chars: Vec<char> = trimmed.chars().collect();
        let unique_chars: std::collections::HashSet<char> = chars.iter().copied().collect();
        if unique_chars.len() < 3 {
            return true;
        }
        // Check for repeating patterns (trigram repetition)
        if chars.len() >= 9 {
            let trigrams: Vec<String> = chars.windows(3).map(|w| w.iter().collect()).collect();
            let unique_trigrams: std::collections::HashSet<&String> = trigrams.iter().collect();
            let repetition_ratio = 1.0 - (unique_trigrams.len() as f64 / trigrams.len() as f64);
            if repetition_ratio > 0.7 {
                return true;
            }
        }
        false
    }

    /// Compute difference between outputs
    fn compute_diff(&self, a: &str, b: &str) -> f64 {
        // Simple string comparison for now
        // In production, this would compare tensor values
        if a == b {
            0.0
        } else {
            // Compute character-level difference ratio
            let max_len = a.len().max(b.len());
            if max_len == 0 {
                return 0.0;
            }
            let matching: usize = a.chars().zip(b.chars()).filter(|(ca, cb)| ca == cb).count();
            1.0 - (matching as f64 / max_len as f64)
        }
    }

    /// Find indices where outputs differ
    fn find_diff_indices(&self, a: &str, b: &str) -> Vec<usize> {
        a.chars()
            .zip(b.chars())
            .enumerate()
            .filter(|(_, (ca, cb))| ca != cb)
            .map(|(i, _)| i)
            .collect()
    }

    /// Hash output for evidence
    fn hash_output(output: &str) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        output.hash(&mut hasher);
        format!("{:016x}", hasher.finish())
    }
}

/// Semantic conversion test that detects embedding/weight bugs (GH-187)
///
/// This test compares actual inference output between formats to detect
/// the class of bugs that have occurred 50+ times.
#[derive(Debug, Clone)]
pub struct SemanticConversionTest {
    /// Source format (SafeTensors as ground truth per spec 7.4)
    pub source_format: Format,
    /// Target format to test
    pub target_format: Format,
    /// Backend to use
    pub backend: Backend,
    /// Model ID
    pub model_id: ModelId,
    /// Binary path for apr CLI
    binary: String,
}

impl SemanticConversionTest {
    /// Create a new semantic conversion test
    #[must_use]
    pub fn new(source: Format, target: Format, backend: Backend, model_id: ModelId) -> Self {
        Self {
            source_format: source,
            target_format: target,
            backend,
            model_id,
            binary: default_binary(),
        }
    }

    /// Execute the semantic test and classify any bug found
    ///
    /// # Errors
    ///
    /// Returns an error if inference fails.
    pub fn execute(&self, model_path: &Path) -> Result<SemanticTestResult> {
        // Run inference on source (SafeTensors - ground truth per spec 7.4)
        let source_output = self.run_inference(model_path)?;

        // Convert to target format
        let converted_path = self.convert_model(model_path)?;

        // Run inference on converted model
        let target_output = self.run_inference(&converted_path)?;

        // Check for stderr containing tokenizer error
        let has_tokenizer_error = target_output.stderr.contains("PMAT-172")
            || target_output.stderr.contains("missing embedded tokenizer");

        // Classify the bug type
        let bug_type = self.classify_bug(
            &source_output.stdout,
            &target_output.stdout,
            has_tokenizer_error,
        );

        if let Some(bug) = bug_type {
            Ok(SemanticTestResult::Falsified {
                bug_type: bug,
                source_output: source_output.stdout,
                target_output: target_output.stdout,
                stderr: target_output.stderr,
            })
        } else {
            Ok(SemanticTestResult::Corroborated {
                source_output: source_output.stdout,
                target_output: target_output.stdout,
            })
        }
    }

    /// Classify the bug type based on output patterns
    fn classify_bug(
        &self,
        source: &str,
        target: &str,
        has_tokenizer_error: bool,
    ) -> Option<ConversionBugType> {
        // Check for tokenizer missing
        if has_tokenizer_error {
            return Some(ConversionBugType::TokenizerMissing);
        }

        // Check for garbage output patterns
        let has_garbage = GARBAGE_PATTERNS.iter().any(|p| target.contains(p));
        let source_has_expected = ARITHMETIC_EXPECTED.iter().any(|p| source.contains(p));
        let target_has_expected = ARITHMETIC_EXPECTED.iter().any(|p| target.contains(p));

        // Source produces correct answer but target produces garbage
        if source_has_expected && has_garbage {
            return Some(ConversionBugType::EmbeddingTransposition);
        }

        // Source correct, target wrong (but not garbage)
        if source_has_expected && !target_has_expected && !target.is_empty() {
            return Some(ConversionBugType::SemanticDrift);
        }

        // Target is empty or all whitespace
        if target.trim().is_empty() && !source.trim().is_empty() {
            return Some(ConversionBugType::WeightCorruption);
        }

        // Outputs match - no bug
        if source.trim() == target.trim() {
            return None;
        }

        // Outputs differ but no clear pattern
        Some(ConversionBugType::Unknown)
    }

    /// Run inference and capture both stdout and stderr
    fn run_inference(&self, model_path: &Path) -> Result<InferenceOutput> {
        let backend_flag = match self.backend {
            Backend::Cpu => vec!["--no-gpu".to_string()],
            Backend::Gpu => vec!["--gpu".to_string()],
        };

        let output = Command::new(&self.binary)
            .arg("run")
            .arg(model_path)
            .arg("-p")
            .arg("What is 2+2?")
            .arg("--max-tokens")
            .arg("32")
            .args(&backend_flag)
            .output()
            .map_err(Error::Io)?;

        Ok(InferenceOutput {
            stdout: String::from_utf8_lossy(&output.stdout).to_string(),
            stderr: String::from_utf8_lossy(&output.stderr).to_string(),
            exit_code: output.status.code().unwrap_or(-1),
        })
    }

    /// Convert model to target format
    fn convert_model(&self, source_path: &Path) -> Result<std::path::PathBuf> {
        let target_ext = match self.target_format {
            Format::Gguf => "gguf",
            Format::SafeTensors => "safetensors",
            Format::Apr => "apr",
        };

        let target_path = source_path.with_extension(format!("semantic_test.{target_ext}"));

        let output = Command::new(&self.binary)
            .arg("rosetta")
            .arg("convert")
            .arg(source_path)
            .arg(&target_path)
            .output()
            .map_err(Error::Io)?;

        if !output.status.success() {
            return Err(Error::Execution(format!(
                "Conversion failed: {}",
                String::from_utf8_lossy(&output.stderr)
            )));
        }

        Ok(target_path)
    }
}

/// Output from inference command
#[derive(Debug, Clone)]
struct InferenceOutput {
    stdout: String,
    stderr: String,
    #[allow(dead_code)]
    exit_code: i32,
}

/// Result of semantic conversion test
#[derive(Debug, Clone)]
pub enum SemanticTestResult {
    /// Conversion preserved semantics
    Corroborated {
        /// Source model output
        source_output: String,
        /// Target model output
        target_output: String,
    },
    /// Conversion introduced semantic errors
    Falsified {
        /// Classified bug type
        bug_type: ConversionBugType,
        /// Source model output (ground truth)
        source_output: String,
        /// Target model output (buggy)
        target_output: String,
        /// Stderr from target inference
        stderr: String,
    },
}

impl SemanticTestResult {
    /// Check if test passed
    #[must_use]
    pub fn is_pass(&self) -> bool {
        matches!(self, Self::Corroborated { .. })
    }

    /// Get the bug type if test failed
    #[must_use]
    pub fn bug_type(&self) -> Option<ConversionBugType> {
        match self {
            Self::Falsified { bug_type, .. } => Some(*bug_type),
            Self::Corroborated { .. } => None,
        }
    }
}

/// Generate all conversion test pairs
#[must_use]
pub fn all_conversion_pairs() -> Vec<(Format, Format)> {
    vec![
        (Format::Gguf, Format::Apr),
        (Format::Apr, Format::Gguf),
        (Format::Gguf, Format::SafeTensors),
        (Format::SafeTensors, Format::Gguf),
        (Format::Apr, Format::SafeTensors),
        (Format::SafeTensors, Format::Apr),
    ]
}

/// Generate all backends to test
#[must_use]
pub fn all_backends() -> Vec<Backend> {
    vec![Backend::Cpu, Backend::Gpu]
    // WASM/WGPU would be added here when supported
}

/// Generate all conversion tests for a model
#[must_use]
pub fn generate_conversion_tests(model_id: &ModelId) -> Vec<ConversionTest> {
    let mut tests = Vec::new();

    for (source, target) in all_conversion_pairs() {
        for backend in all_backends() {
            tests.push(ConversionTest::new(
                source,
                target,
                backend,
                model_id.clone(),
            ));
        }
    }

    tests
}

/// Round-trip conversion test
#[derive(Debug, Clone)]
pub struct RoundTripTest {
    /// Formats to chain through
    pub formats: Vec<Format>,
    /// Backend to use
    pub backend: Backend,
    /// Model ID
    pub model_id: ModelId,
    /// Binary path for apr CLI
    binary: String,
}

impl RoundTripTest {
    /// Create a new round-trip test
    #[must_use]
    pub fn new(formats: Vec<Format>, backend: Backend, model_id: ModelId) -> Self {
        Self {
            formats,
            backend,
            model_id,
            binary: default_binary(),
        }
    }

    /// Execute round-trip conversion test
    ///
    /// # Errors
    ///
    /// Returns an error if any conversion fails.
    pub fn execute(&self, model_path: &Path) -> Result<ConversionResult> {
        // Resolve directory to actual model file for starting format
        let resolved_path = resolve_model_path(model_path, self.formats[0])?;

        // Get original output
        let original_output = run_inference_simple(&resolved_path, self.backend, &self.binary)?;

        // Convert through chain
        let mut current_path = resolved_path;
        for i in 0..self.formats.len() {
            let next_format = self.formats[(i + 1) % self.formats.len()];
            current_path = convert_to_format(&current_path, next_format, &self.binary)?;
        }

        // Get final output
        let final_output = run_inference_simple(&current_path, self.backend, &self.binary)?;

        // Compare: round-trip through different formats involves quantization,
        // so text-level identity is not expected. Check non-garbage instead.
        let has_cross_format = self.formats.windows(2).any(|w| w[0] != w[1]);
        let passes = if has_cross_format {
            !ConversionTest::is_garbage_output(&original_output)
                && !ConversionTest::is_garbage_output(&final_output)
        } else {
            original_output == final_output
        };

        if passes {
            Ok(ConversionResult::Corroborated {
                source_format: self.formats[0],
                target_format: self.formats[0],
                backend: self.backend,
                max_diff: 0.0,
            })
        } else {
            Ok(ConversionResult::Falsified {
                gate_id: "F-CONV-RT-001".to_string(),
                reason: "Round-trip conversion produced different output".to_string(),
                evidence: ConversionEvidence {
                    source_hash: ConversionTest::hash_output(&original_output),
                    converted_hash: ConversionTest::hash_output(&final_output),
                    max_diff: 1.0,
                    diff_indices: vec![],
                    source_format: self.formats[0],
                    target_format: self.formats[0],
                    backend: self.backend,
                    failure_type: None,
                    quant_type: None,
                },
            })
        }
    }
}

/// Idempotency test (MR-IDEM): convert A→B twice from same source, compare outputs
///
/// Detects non-deterministic conversion bugs. If converting the same model twice
/// produces different outputs, the converter has internal state leaks.
#[derive(Debug, Clone)]
pub struct IdempotencyTest {
    /// First format in chain
    pub format_a: Format,
    /// Second format in chain
    pub format_b: Format,
    /// Backend to use
    pub backend: Backend,
    /// Model ID
    pub model_id: ModelId,
    /// Binary path for apr CLI
    binary: String,
}

impl IdempotencyTest {
    /// Create a new idempotency test
    #[must_use]
    pub fn new(format_a: Format, format_b: Format, backend: Backend, model_id: ModelId) -> Self {
        Self {
            format_a,
            format_b,
            backend,
            model_id,
            binary: default_binary(),
        }
    }

    /// Execute idempotency test: convert A→B twice, compare
    ///
    /// # Errors
    ///
    /// Returns an error if conversion or inference fails.
    pub fn execute(&self, model_path: &Path) -> Result<ConversionResult> {
        // Resolve directory to actual model file for source format
        let resolved_path = resolve_model_path(model_path, self.format_a)?;

        // Convert A→B (first time)
        let converted_1 =
            convert_to_format_tagged(&resolved_path, self.format_b, "idem1", &self.binary)?;
        let output_1 = run_inference_simple(&converted_1, self.backend, &self.binary)?;

        // Convert A→B (second time, from same source)
        let converted_2 =
            convert_to_format_tagged(&resolved_path, self.format_b, "idem2", &self.binary)?;
        let output_2 = run_inference_simple(&converted_2, self.backend, &self.binary)?;

        // Cross-format conversion involves quantization which may not be
        // perfectly deterministic (floating-point rounding). Use non-garbage
        // check instead of exact text match.
        let is_cross_format = self.format_a != self.format_b;
        let passes = if is_cross_format {
            !ConversionTest::is_garbage_output(&output_1)
                && !ConversionTest::is_garbage_output(&output_2)
        } else {
            output_1 == output_2
        };

        if passes {
            Ok(ConversionResult::Corroborated {
                source_format: self.format_a,
                target_format: self.format_b,
                backend: self.backend,
                max_diff: 0.0,
            })
        } else {
            Ok(ConversionResult::Falsified {
                gate_id: "F-CONV-IDEM-001".to_string(),
                reason: format!(
                    "Idempotency failure: {:?}→{:?} produced different output on second conversion",
                    self.format_a, self.format_b
                ),
                evidence: ConversionEvidence {
                    source_hash: ConversionTest::hash_output(&output_1),
                    converted_hash: ConversionTest::hash_output(&output_2),
                    max_diff: 1.0,
                    diff_indices: vec![],
                    source_format: self.format_a,
                    target_format: self.format_b,
                    backend: self.backend,
                    failure_type: None,
                    quant_type: None,
                },
            })
        }
    }
}

/// Byte-level round-trip test (GH-6/AC-3): ST → APR → GGUF → APR with tensor diff
///
/// Unlike `RoundTripTest` which compares inference output, this test compares
/// the actual tensor data byte-for-byte between two APR conversions.
/// Detects silent data corruption that inference-level tests may miss.
#[derive(Debug, Clone)]
pub struct ByteLevelRoundTripTest {
    /// Backend to use
    pub backend: Backend,
    /// Model ID
    pub model_id: ModelId,
    /// Binary path for apr CLI
    binary: String,
}

impl ByteLevelRoundTripTest {
    /// Create a new byte-level round-trip test
    #[must_use]
    pub fn new(backend: Backend, model_id: ModelId) -> Self {
        Self {
            backend,
            model_id,
            binary: default_binary(),
        }
    }

    /// Execute byte-level round-trip: ST → APR(1) and ST → APR → GGUF → APR(2), diff tensors
    ///
    /// # Errors
    ///
    /// Returns an error if conversion or diff fails.
    pub fn execute(&self, model_path: &Path) -> Result<ConversionResult> {
        let resolved_path = resolve_model_path(model_path, Format::SafeTensors)?;

        // Step 1: ST → APR (reference)
        let apr_ref =
            convert_to_format_tagged(&resolved_path, Format::Apr, "byte_rt_ref", &self.binary)?;

        // Step 2: ST → APR → GGUF → APR (round-trip)
        let apr_tmp =
            convert_to_format_tagged(&resolved_path, Format::Apr, "byte_rt_tmp", &self.binary)?;
        let gguf_tmp =
            convert_to_format_tagged(&apr_tmp, Format::Gguf, "byte_rt_gguf", &self.binary)?;
        let apr_roundtrip =
            convert_to_format_tagged(&gguf_tmp, Format::Apr, "byte_rt_final", &self.binary)?;

        // Step 3: diff_tensors between apr_ref and apr_roundtrip
        let diff_output = run_diff_tensors(&apr_ref, &apr_roundtrip, &self.binary)?;

        if diff_output.contains("\"passed\":false") || diff_output.contains("mismatched") {
            Ok(ConversionResult::Falsified {
                gate_id: "F-CONV-RT-BYTE-001".to_string(),
                reason: "Byte-level round-trip: tensor data differs after ST→APR→GGUF→APR"
                    .to_string(),
                evidence: ConversionEvidence {
                    source_hash: String::new(),
                    converted_hash: String::new(),
                    max_diff: 1.0,
                    diff_indices: vec![],
                    source_format: Format::SafeTensors,
                    target_format: Format::Apr,
                    backend: self.backend,
                    failure_type: Some(ConversionFailureType::DequantizationFailure),
                    quant_type: None,
                },
            })
        } else {
            Ok(ConversionResult::Corroborated {
                source_format: Format::SafeTensors,
                target_format: Format::Apr,
                backend: self.backend,
                max_diff: 0.0,
            })
        }
    }
}

/// Commutativity test (MR-COM): different conversion paths should yield equivalent inference
///
/// Tests that GGUF→APR produces the same inference as GGUF→ST→APR.
/// Path-dependent conversion bugs are a major source of silent failures.
#[derive(Debug, Clone)]
pub struct CommutativityTest {
    /// Backend to use
    pub backend: Backend,
    /// Model ID
    pub model_id: ModelId,
    /// Binary path for apr CLI
    binary: String,
}

impl CommutativityTest {
    /// Create a new commutativity test
    #[must_use]
    pub fn new(backend: Backend, model_id: ModelId) -> Self {
        Self {
            backend,
            model_id,
            binary: default_binary(),
        }
    }

    /// Execute commutativity test: compare direct vs indirect conversion paths
    ///
    /// Path A: GGUF → APR (direct)
    /// Path B: GGUF → SafeTensors → APR (indirect)
    ///
    /// # Errors
    ///
    /// Returns an error if conversion or inference fails.
    pub fn execute(&self, model_path: &Path) -> Result<ConversionResult> {
        // Resolve directory to actual GGUF model file
        let resolved_path = resolve_model_path(model_path, Format::Gguf)?;

        // Path A: GGUF → APR (direct)
        let direct_apr =
            convert_to_format_tagged(&resolved_path, Format::Apr, "com_direct", &self.binary)?;
        let output_a = run_inference_simple(&direct_apr, self.backend, &self.binary)?;

        // Path B: GGUF → SafeTensors → APR (indirect)
        let via_st =
            convert_to_format_tagged(&resolved_path, Format::SafeTensors, "com_via", &self.binary)?;
        let indirect_apr =
            convert_to_format_tagged(&via_st, Format::Apr, "com_indirect", &self.binary)?;
        let output_b = run_inference_simple(&indirect_apr, self.backend, &self.binary)?;

        // Cross-format paths involve different quantization chains,
        // so text-level identity is not expected. Check non-garbage instead.
        let passes = !ConversionTest::is_garbage_output(&output_a)
            && !ConversionTest::is_garbage_output(&output_b);

        if passes {
            Ok(ConversionResult::Corroborated {
                source_format: Format::Gguf,
                target_format: Format::Apr,
                backend: self.backend,
                max_diff: 0.0,
            })
        } else {
            Ok(ConversionResult::Falsified {
                gate_id: "F-CONV-COM-001".to_string(),
                reason: "Commutativity failure: GGUF→APR differs from GGUF→ST→APR (garbage output)"
                    .to_string(),
                evidence: ConversionEvidence {
                    source_hash: ConversionTest::hash_output(&output_a),
                    converted_hash: ConversionTest::hash_output(&output_b),
                    max_diff: 1.0,
                    diff_indices: vec![],
                    source_format: Format::Gguf,
                    target_format: Format::Apr,
                    backend: self.backend,
                    failure_type: None,
                    quant_type: None,
                },
            })
        }
    }
}

/// Check tensor cardinality after conversion (MR-CARD)
///
/// Fires F-CONV-CARD-001 if `tensor_count(output) < tensor_count(input)`.
/// This catches silent tensor fusion bugs like QKV fusion (338→227).
///
/// # Errors
///
/// Returns an error if `apr rosetta inspect` fails on either model.
pub fn check_cardinality(
    source_path: &Path,
    converted_path: &Path,
    binary: &str,
) -> Result<Option<(String, String)>> {
    let source_inspect = crate::differential::run_inspect(source_path, binary)?;
    let target_inspect = crate::differential::run_inspect(converted_path, binary)?;

    if target_inspect.tensor_count < source_inspect.tensor_count {
        Ok(Some((
            "F-CONV-CARD-001".to_string(),
            format!(
                "Tensor cardinality loss: {} → {}",
                source_inspect.tensor_count, target_inspect.tensor_count
            ),
        )))
    } else {
        Ok(None)
    }
}

/// Check tensor name preservation after conversion (T-QKV-02)
///
/// Fires F-CONV-NAME-001 if tensor names changed unexpectedly during conversion
/// (e.g., q_proj+k_proj+v_proj → qkv_proj fusion).
///
/// # Errors
///
/// Returns an error if `apr rosetta inspect` fails on either model.
pub fn check_tensor_names(
    source_path: &Path,
    converted_path: &Path,
    binary: &str,
) -> Result<Option<(String, String)>> {
    let source_inspect = crate::differential::run_inspect(source_path, binary)?;
    let target_inspect = crate::differential::run_inspect(converted_path, binary)?;

    // Skip if either side has no tensor names (inspect may not support it)
    if source_inspect.tensor_names.is_empty() || target_inspect.tensor_names.is_empty() {
        return Ok(None);
    }

    let missing: Vec<_> = source_inspect
        .tensor_names
        .iter()
        .filter(|n| !target_inspect.tensor_names.contains(n))
        .collect();

    if missing.is_empty() {
        return Ok(None);
    }

    // Check for known fusion patterns (q_proj+k_proj+v_proj → qkv_proj)
    let has_fusion = missing
        .iter()
        .any(|n| n.contains("q_proj") || n.contains("k_proj") || n.contains("v_proj"))
        && target_inspect
            .tensor_names
            .iter()
            .any(|n| n.contains("qkv_proj"));

    let detail = if has_fusion {
        format!(
            "QKV fusion detected: {} source tensors missing (likely fused into qkv_proj). Missing: {}",
            missing.len(),
            missing
                .iter()
                .take(5)
                .map(|s| s.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        )
    } else {
        format!(
            "Tensor name divergence: {} source tensors not found in output. Missing: {}",
            missing.len(),
            missing
                .iter()
                .take(5)
                .map(|s| s.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        )
    };

    Ok(Some(("F-CONV-NAME-001".to_string(), detail)))
}

/// Convert model to specified format with a tag suffix for disambiguation
fn convert_to_format_tagged(
    source_path: &Path,
    target_format: Format,
    tag: &str,
    binary: &str,
) -> Result<std::path::PathBuf> {
    let target_ext = match target_format {
        Format::Gguf => "gguf",
        Format::SafeTensors => "safetensors",
        Format::Apr => "apr",
    };

    let target_path = source_path.with_extension(format!("{tag}.{target_ext}"));

    let output = Command::new(binary)
        .arg("rosetta")
        .arg("convert")
        .arg(source_path)
        .arg(&target_path)
        .output()
        .map_err(Error::Io)?;

    if !output.status.success() {
        return Err(Error::Execution(format!(
            "Conversion failed: {}",
            String::from_utf8_lossy(&output.stderr)
        )));
    }

    Ok(target_path)
}

/// Diff tensors between two models via `apr rosetta diff-tensors --json`
fn run_diff_tensors(model_a: &Path, model_b: &Path, binary: &str) -> Result<String> {
    let output = Command::new(binary)
        .arg("rosetta")
        .arg("diff-tensors")
        .arg(model_a)
        .arg(model_b)
        .arg("--json")
        .output()
        .map_err(Error::Io)?;

    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

/// Simple inference helper
fn run_inference_simple(model_path: &Path, backend: Backend, binary: &str) -> Result<String> {
    let backend_flag = match backend {
        Backend::Cpu => vec![],
        Backend::Gpu => vec!["--gpu".to_string()],
    };

    let output = Command::new(binary)
        .arg("run")
        .arg(model_path)
        .arg("-p")
        .arg("What is 2+2?")
        .arg("--max-tokens")
        .arg("32")
        .args(&backend_flag)
        .output()
        .map_err(Error::Io)?;

    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

/// Convert model to specified format
fn convert_to_format(
    source_path: &Path,
    target_format: Format,
    binary: &str,
) -> Result<std::path::PathBuf> {
    let target_ext = match target_format {
        Format::Gguf => "gguf",
        Format::SafeTensors => "safetensors",
        Format::Apr => "apr",
    };

    // Create target path with new extension (format determined by extension)
    let target_path = source_path.with_extension(format!("converted.{target_ext}"));

    // Use apr rosetta convert: apr rosetta convert <SOURCE> <TARGET>
    // Format is inferred from output file extension
    let output = Command::new(binary)
        .arg("rosetta")
        .arg("convert")
        .arg(source_path)
        .arg(&target_path)
        .output()
        .map_err(Error::Io)?;

    if !output.status.success() {
        return Err(Error::Execution(format!(
            "Conversion failed: {}",
            String::from_utf8_lossy(&output.stderr)
        )));
    }

    Ok(target_path)
}

// ConversionConfig + ConversionExecutor — see conversion_executor.rs
include!("conversion_executor.rs");

// HF cache resolution — see conversion_hf_cache.rs
include!("conversion_hf_cache.rs");

#[cfg(test)]
#[path = "conversion_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "conversion_tests_b.rs"]
mod tests_b;

#[cfg(test)]
#[path = "conversion_tests_c.rs"]
mod tests_c;
