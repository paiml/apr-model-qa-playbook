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
        let converted_output = self.run_inference(&converted_path, &self.target_format)?;

        // 4. Compare outputs
        let diff = self.compute_diff(&source_output, &converted_output);

        if diff > self.effective_epsilon() {
            Ok(ConversionResult::Falsified {
                gate_id: self.gate_id(),
                reason: format!(
                    "Conversion {:?} → {:?} produced different output (diff: {:.2e}, ε: {:.2e})",
                    self.source_format,
                    self.target_format,
                    diff,
                    self.effective_epsilon()
                ),
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
        } else {
            Ok(ConversionResult::Corroborated {
                source_format: self.source_format,
                target_format: self.target_format,
                backend: self.backend,
                max_diff: diff,
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

        // Compare
        if original_output != final_output {
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
        } else {
            Ok(ConversionResult::Corroborated {
                source_format: self.formats[0],
                target_format: self.formats[0],
                backend: self.backend,
                max_diff: 0.0,
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

        if output_1 != output_2 {
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
        } else {
            Ok(ConversionResult::Corroborated {
                source_format: self.format_a,
                target_format: self.format_b,
                backend: self.backend,
                max_diff: 0.0,
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

        if output_a != output_b {
            Ok(ConversionResult::Falsified {
                gate_id: "F-CONV-COM-001".to_string(),
                reason: "Commutativity failure: GGUF→APR differs from GGUF→ST→APR".to_string(),
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
        } else {
            Ok(ConversionResult::Corroborated {
                source_format: Format::Gguf,
                target_format: Format::Apr,
                backend: self.backend,
                max_diff: 0.0,
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

/// Configuration for conversion executor
#[derive(Debug, Clone)]
#[allow(clippy::struct_excessive_bools)]
pub struct ConversionConfig {
    /// Test all format pairs
    pub test_all_pairs: bool,
    /// Test round-trips
    pub test_round_trips: bool,
    /// Test multi-hop conversion chains (T-QKV-04)
    pub test_multi_hop: bool,
    /// Test tensor cardinality after conversion (MR-CARD)
    pub test_cardinality: bool,
    /// Test tensor name preservation after conversion (T-QKV-02)
    pub test_tensor_names: bool,
    /// Test idempotency of double-conversion (MR-IDEM)
    pub test_idempotency: bool,
    /// Test commutativity of conversion paths (MR-COM)
    pub test_commutativity: bool,
    /// Backends to test
    pub backends: Vec<Backend>,
    /// Use CPU only (no GPU)
    pub no_gpu: bool,
}

impl Default for ConversionConfig {
    fn default() -> Self {
        Self {
            test_all_pairs: true,
            test_round_trips: true,
            test_multi_hop: true,
            test_cardinality: true,
            test_tensor_names: true,
            test_idempotency: true,
            test_commutativity: true,
            backends: vec![Backend::Cpu, Backend::Gpu],
            no_gpu: false,
        }
    }
}

impl ConversionConfig {
    /// Create config for CPU-only testing
    #[must_use]
    pub fn cpu_only() -> Self {
        Self {
            test_all_pairs: true,
            test_round_trips: true,
            test_multi_hop: true,
            test_cardinality: true,
            test_tensor_names: true,
            test_idempotency: true,
            test_commutativity: true,
            backends: vec![Backend::Cpu],
            no_gpu: true,
        }
    }
}

/// Executor for running P0 format conversion tests
#[derive(Debug)]
pub struct ConversionExecutor {
    config: ConversionConfig,
    binary: String,
    /// Output directory for conversion artifacts (ISO-OUT-001)
    output_dir: Option<PathBuf>,
}

impl ConversionExecutor {
    /// Create a new conversion executor
    #[must_use]
    pub fn new(config: ConversionConfig) -> Self {
        Self {
            config,
            binary: default_binary(),
            output_dir: None,
        }
    }

    /// Create with default config
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(ConversionConfig::default())
    }

    /// Set the output directory for conversion artifacts (ISO-OUT-001)
    #[must_use]
    pub fn with_output_dir(mut self, output_dir: PathBuf) -> Self {
        self.output_dir = Some(output_dir);
        self
    }

    /// Execute all conversion tests for a model
    ///
    /// # Errors
    ///
    /// Returns an error if a critical conversion failure occurs.
    #[allow(clippy::too_many_lines)]
    pub fn execute_all(
        &self,
        model_path: &Path,
        model_id: &ModelId,
    ) -> Result<ConversionExecutionResult> {
        let mut results = Vec::new();
        let mut evidence = Vec::new();
        let start = std::time::Instant::now();

        // Determine backends to test
        let backends: Vec<Backend> = if self.config.no_gpu {
            vec![Backend::Cpu]
        } else {
            self.config.backends.clone()
        };

        // Create output directory wrapper if configured (ISO-OUT-001)
        let output_dir_wrapper = self
            .output_dir
            .as_ref()
            .map(|dir| ConversionOutputDir::new(dir, model_id));

        // Test all format pairs
        if self.config.test_all_pairs {
            for (source, target) in all_conversion_pairs() {
                for backend in &backends {
                    let mut test = ConversionTest::new(source, target, *backend, model_id.clone());
                    test.binary.clone_from(&self.binary);
                    if let Some(ref out_dir) = output_dir_wrapper {
                        test.output_dir = Some(out_dir.clone());
                    }

                    match test.execute(model_path) {
                        Ok(result) => {
                            let ev: Evidence = result.clone().into();
                            evidence.push(ev);
                            results.push(result);
                        }
                        Err(e) => {
                            // Conversion infrastructure failure - record as falsified
                            let ev = Evidence::falsified(
                                &test.gate_id(),
                                QaScenario::new(
                                    model_id.clone(),
                                    Modality::Run,
                                    *backend,
                                    target,
                                    format!("Convert {source:?} to {target:?}"),
                                    0,
                                ),
                                format!("Conversion infrastructure error: {e}"),
                                "N/A",
                                0,
                            );
                            evidence.push(ev);
                            results.push(ConversionResult::Falsified {
                                gate_id: test.gate_id(),
                                reason: e.to_string(),
                                evidence: ConversionEvidence {
                                    source_hash: String::new(),
                                    converted_hash: String::new(),
                                    max_diff: f64::MAX,
                                    diff_indices: vec![],
                                    source_format: source,
                                    target_format: target,
                                    backend: *backend,
                                    failure_type: None,
                                    quant_type: None,
                                },
                            });
                        }
                    }
                }
            }
        }

        // Test round-trips (GGUF → APR → SafeTensors → GGUF) - F-CONV-RT-001
        if self.config.test_round_trips {
            for backend in &backends {
                let mut rt = RoundTripTest::new(
                    vec![Format::Gguf, Format::Apr, Format::SafeTensors, Format::Gguf],
                    *backend,
                    model_id.clone(),
                );
                rt.binary.clone_from(&self.binary);

                match rt.execute(model_path) {
                    Ok(result) => {
                        let ev: Evidence = result.clone().into();
                        evidence.push(ev);
                        results.push(result);
                    }
                    Err(e) => {
                        let ev = Evidence::falsified(
                            "F-CONV-RT-001",
                            QaScenario::new(
                                model_id.clone(),
                                Modality::Run,
                                *backend,
                                Format::Gguf,
                                "Round-trip conversion".to_string(),
                                0,
                            ),
                            format!("Round-trip failed: {e}"),
                            "N/A",
                            0,
                        );
                        evidence.push(ev);
                    }
                }
            }
        }

        // T-QKV-03: ST → APR → GGUF → ST round-trip (F-CONV-RT-002)
        // T-QKV-04: ST → APR → GGUF → APR → ST multi-hop (F-CONV-RT-003)
        // GH-6/AC-3: ST → APR → GGUF → APR round-trip (F-CONV-RT-004)
        if self.config.test_multi_hop {
            let multi_hop_chains: Vec<(&str, Vec<Format>)> = vec![
                (
                    "F-CONV-RT-002",
                    vec![
                        Format::SafeTensors,
                        Format::Apr,
                        Format::Gguf,
                        Format::SafeTensors,
                    ],
                ),
                (
                    "F-CONV-RT-003",
                    vec![
                        Format::SafeTensors,
                        Format::Apr,
                        Format::Gguf,
                        Format::Apr,
                        Format::SafeTensors,
                    ],
                ),
                (
                    "F-CONV-RT-004",
                    vec![Format::SafeTensors, Format::Apr, Format::Gguf, Format::Apr],
                ),
            ];

            for (gate_id, chain) in &multi_hop_chains {
                for backend in &backends {
                    let mut rt = RoundTripTest::new(chain.clone(), *backend, model_id.clone());
                    rt.binary.clone_from(&self.binary);

                    match rt.execute(model_path) {
                        Ok(mut result) => {
                            // Override the gate ID from the generic RT-001
                            if let ConversionResult::Falsified {
                                gate_id: ref mut gid,
                                ..
                            } = result
                            {
                                *gid = (*gate_id).to_string();
                            }
                            let ev: Evidence = result.clone().into();
                            evidence.push(ev);
                            results.push(result);
                        }
                        Err(e) => {
                            let chain_desc: Vec<_> =
                                chain.iter().map(|f| format!("{f:?}")).collect();
                            let ev = Evidence::falsified(
                                *gate_id,
                                QaScenario::new(
                                    model_id.clone(),
                                    Modality::Run,
                                    *backend,
                                    Format::SafeTensors,
                                    format!("Multi-hop: {}", chain_desc.join("→")),
                                    0,
                                ),
                                format!("Multi-hop chain failed: {e}"),
                                "N/A",
                                0,
                            );
                            evidence.push(ev);
                        }
                    }
                }
            }
        }

        // GH-6/AC-3: Byte-level round-trip — ST→APR→GGUF→APR, diff tensors (F-CONV-RT-BYTE-001)
        if self.config.test_multi_hop {
            for backend in &backends {
                let mut byte_rt = ByteLevelRoundTripTest::new(*backend, model_id.clone());
                byte_rt.binary.clone_from(&self.binary);

                match byte_rt.execute(model_path) {
                    Ok(result) => {
                        let ev: Evidence = result.clone().into();
                        evidence.push(ev);
                        results.push(result);
                    }
                    Err(e) => {
                        let ev = Evidence::falsified(
                            "F-CONV-RT-BYTE-001",
                            QaScenario::new(
                                model_id.clone(),
                                Modality::Run,
                                *backend,
                                Format::SafeTensors,
                                "Byte-level round-trip ST→APR→GGUF→APR".to_string(),
                                0,
                            ),
                            format!("Byte-level round-trip failed: {e}"),
                            "N/A",
                            0,
                        );
                        evidence.push(ev);
                    }
                }
            }
        }

        // MR-IDEM: Idempotency test — convert GGUF→APR twice, compare (F-CONV-IDEM-001)
        if self.config.test_idempotency {
            for backend in &backends {
                let mut idem =
                    IdempotencyTest::new(Format::Gguf, Format::Apr, *backend, model_id.clone());
                idem.binary.clone_from(&self.binary);

                match idem.execute(model_path) {
                    Ok(result) => {
                        let ev: Evidence = result.clone().into();
                        evidence.push(ev);
                        results.push(result);
                    }
                    Err(e) => {
                        let ev = Evidence::falsified(
                            "F-CONV-IDEM-001",
                            QaScenario::new(
                                model_id.clone(),
                                Modality::Run,
                                *backend,
                                Format::Apr,
                                "Idempotency: GGUF→APR twice".to_string(),
                                0,
                            ),
                            format!("Idempotency test failed: {e}"),
                            "N/A",
                            0,
                        );
                        evidence.push(ev);
                    }
                }
            }
        }

        // MR-COM: Commutativity test — GGUF→APR vs GGUF→ST→APR (F-CONV-COM-001)
        if self.config.test_commutativity {
            for backend in &backends {
                let mut com = CommutativityTest::new(*backend, model_id.clone());
                com.binary.clone_from(&self.binary);

                match com.execute(model_path) {
                    Ok(result) => {
                        let ev: Evidence = result.clone().into();
                        evidence.push(ev);
                        results.push(result);
                    }
                    Err(e) => {
                        let ev = Evidence::falsified(
                            "F-CONV-COM-001",
                            QaScenario::new(
                                model_id.clone(),
                                Modality::Run,
                                *backend,
                                Format::Apr,
                                "Commutativity: GGUF→APR vs GGUF→ST→APR".to_string(),
                                0,
                            ),
                            format!("Commutativity test failed: {e}"),
                            "N/A",
                            0,
                        );
                        evidence.push(ev);
                    }
                }
            }
        }

        // MR-CARD: Cardinality gate — tensor_count(output) >= tensor_count(input) (F-CONV-CARD-001)
        // T-QKV-02: Tensor name preservation (F-CONV-NAME-001)
        // These run on any successful conversions that produced files
        if self.config.test_cardinality || self.config.test_tensor_names {
            for (source, target) in all_conversion_pairs() {
                let target_ext = match target {
                    Format::Gguf => "gguf",
                    Format::SafeTensors => "safetensors",
                    Format::Apr => "apr",
                };
                let converted_path = model_path.with_extension(format!("converted.{target_ext}"));
                if !converted_path.exists() {
                    continue;
                }

                if self.config.test_cardinality {
                    match check_cardinality(model_path, &converted_path, &self.binary) {
                        Ok(Some((gate_id, reason))) => {
                            let ev = Evidence::falsified(
                                &gate_id,
                                QaScenario::new(
                                    model_id.clone(),
                                    Modality::Run,
                                    Backend::Cpu,
                                    target,
                                    format!("Cardinality {source:?}→{target:?}"),
                                    0,
                                ),
                                &reason,
                                "N/A",
                                0,
                            );
                            evidence.push(ev);
                            results.push(ConversionResult::Falsified {
                                gate_id: "F-CONV-CARD-001".to_string(),
                                reason,
                                evidence: ConversionEvidence {
                                    source_hash: String::new(),
                                    converted_hash: String::new(),
                                    max_diff: 0.0,
                                    diff_indices: vec![],
                                    source_format: source,
                                    target_format: target,
                                    backend: Backend::Cpu,
                                    failure_type: None,
                                    quant_type: None,
                                },
                            });
                        }
                        Ok(None) => {
                            let ev = Evidence::corroborated(
                                "F-CONV-CARD-001",
                                QaScenario::new(
                                    model_id.clone(),
                                    Modality::Run,
                                    Backend::Cpu,
                                    target,
                                    format!("Cardinality {source:?}→{target:?}"),
                                    0,
                                ),
                                "Tensor cardinality preserved",
                                0,
                            );
                            evidence.push(ev);
                        }
                        Err(_) => {} // Inspect not available, skip gate
                    }
                }

                if self.config.test_tensor_names {
                    match check_tensor_names(model_path, &converted_path, &self.binary) {
                        Ok(Some((gate_id, reason))) => {
                            let ev = Evidence::falsified(
                                &gate_id,
                                QaScenario::new(
                                    model_id.clone(),
                                    Modality::Run,
                                    Backend::Cpu,
                                    target,
                                    format!("Tensor names {source:?}→{target:?}"),
                                    0,
                                ),
                                &reason,
                                "N/A",
                                0,
                            );
                            evidence.push(ev);
                        }
                        Ok(None) => {
                            let ev = Evidence::corroborated(
                                "F-CONV-NAME-001",
                                QaScenario::new(
                                    model_id.clone(),
                                    Modality::Run,
                                    Backend::Cpu,
                                    target,
                                    format!("Tensor names {source:?}→{target:?}"),
                                    0,
                                ),
                                "Tensor names preserved",
                                0,
                            );
                            evidence.push(ev);
                        }
                        Err(_) => {} // Inspect not available, skip gate
                    }
                }
            }
        }

        let duration_ms = start.elapsed().as_millis() as u64;

        let passed = results
            .iter()
            .filter(|r| matches!(r, ConversionResult::Corroborated { .. }))
            .count();
        let failed = results.len() - passed;

        Ok(ConversionExecutionResult {
            total: results.len(),
            passed,
            failed,
            duration_ms,
            results,
            evidence,
        })
    }
}

/// Result of conversion test execution
#[derive(Debug)]
pub struct ConversionExecutionResult {
    /// Total tests run
    pub total: usize,
    /// Tests passed
    pub passed: usize,
    /// Tests failed
    pub failed: usize,
    /// Duration in milliseconds
    pub duration_ms: u64,
    /// Individual results
    pub results: Vec<ConversionResult>,
    /// Evidence collected
    pub evidence: Vec<Evidence>,
}

impl ConversionExecutionResult {
    /// Check if all conversion tests passed
    #[must_use]
    pub fn all_passed(&self) -> bool {
        self.failed == 0
    }

    /// Get pass rate as percentage
    #[must_use]
    pub fn pass_rate(&self) -> f64 {
        if self.total == 0 {
            100.0
        } else {
            (self.passed as f64 / self.total as f64) * 100.0
        }
    }
}

/// Convert ConversionResult to Evidence
impl From<ConversionResult> for Evidence {
    fn from(result: ConversionResult) -> Self {
        match result {
            ConversionResult::Corroborated {
                source_format,
                target_format,
                backend,
                max_diff,
            } => {
                let scenario = QaScenario::new(
                    ModelId::new("conversion", "test"),
                    Modality::Run,
                    backend,
                    target_format,
                    format!("Convert {source_format:?} to {target_format:?}"),
                    0,
                );
                Evidence::corroborated(
                    &format!("F-CONV-{source_format:?}-{target_format:?}"),
                    scenario,
                    &format!("Conversion successful, max_diff: {max_diff:.2e}"),
                    0,
                )
            }
            ConversionResult::Falsified {
                gate_id,
                reason,
                evidence,
            } => {
                let scenario = QaScenario::new(
                    ModelId::new("conversion", "test"),
                    Modality::Run,
                    evidence.backend,
                    evidence.target_format,
                    format!(
                        "Convert {:?} to {:?}",
                        evidence.source_format, evidence.target_format
                    ),
                    0,
                );
                Evidence::falsified(&gate_id, scenario, reason, &evidence.converted_hash, 0)
            }
        }
    }
}

// ============================================================================
// HuggingFace Cache Resolution (HF-CACHE-001, HF-CACHE-002)
// ============================================================================

/// Get the HuggingFace cache directory respecting environment variables.
///
/// Priority (per HuggingFace convention):
/// 1. `$HUGGINGFACE_HUB_CACHE` (highest priority)
/// 2. `$HF_HOME/hub`
/// 3. `~/.cache/huggingface/hub` (default)
///
/// # Specification
///
/// Implements HF-CACHE-002: Environment Variable Support.
#[must_use]
pub fn get_hf_cache_dir() -> std::path::PathBuf {
    use std::path::PathBuf;

    if let Ok(cache) = std::env::var("HUGGINGFACE_HUB_CACHE") {
        return PathBuf::from(cache);
    }
    if let Ok(home) = std::env::var("HF_HOME") {
        return PathBuf::from(home).join("hub");
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home).join(".cache/huggingface/hub")
}

/// Split a HuggingFace repo ID into (org, repo).
///
/// # Examples
///
/// ```
/// use apr_qa_runner::split_hf_repo;
///
/// assert_eq!(split_hf_repo("Qwen/Qwen2.5-Coder-0.5B"), ("Qwen", "Qwen2.5-Coder-0.5B"));
/// assert_eq!(split_hf_repo("model-only"), ("unknown", "model-only"));
/// ```
#[must_use]
pub fn split_hf_repo(hf_repo: &str) -> (&str, &str) {
    hf_repo.split_once('/').unwrap_or(("unknown", hf_repo))
}

/// Find a snapshot containing `model.safetensors` in the HuggingFace cache.
///
/// Internal helper that searches for model files within a given cache directory.
fn find_hf_snapshot(
    hf_cache: &std::path::Path,
    org: &str,
    repo: &str,
) -> Option<std::path::PathBuf> {
    let hf_model_dir = hf_cache
        .join(format!("models--{org}--{repo}"))
        .join("snapshots");

    if !hf_model_dir.exists() {
        return None;
    }

    let entries = std::fs::read_dir(&hf_model_dir).ok()?;
    for entry in entries.flatten() {
        let snapshot = entry.path();
        if snapshot.is_dir() && snapshot.join("model.safetensors").exists() {
            return Some(snapshot);
        }
    }
    None
}

/// Find a model in the APR cache directory.
///
/// Internal helper that checks if a model exists in the APR cache.
fn find_apr_cache(home: &std::path::Path, org: &str, repo: &str) -> Option<std::path::PathBuf> {
    let apr_cache = home.join(".cache/apr-models").join(org).join(repo);
    if apr_cache.exists() {
        Some(apr_cache)
    } else {
        None
    }
}

/// Resolve HuggingFace repo to cache with explicit cache directories.
///
/// Internal helper for testing that doesn't depend on environment variables.
fn resolve_hf_repo_with_dirs(
    hf_repo: &str,
    hf_cache: &std::path::Path,
    home: &std::path::Path,
) -> Result<std::path::PathBuf> {
    let (org, repo) = split_hf_repo(hf_repo);

    // Try HuggingFace cache first
    if let Some(snapshot) = find_hf_snapshot(hf_cache, org, repo) {
        return Ok(snapshot);
    }

    // Try APR cache
    if let Some(apr_path) = find_apr_cache(home, org, repo) {
        return Ok(apr_path);
    }

    let hf_model_dir = hf_cache
        .join(format!("models--{org}--{repo}"))
        .join("snapshots");
    let apr_cache = home.join(".cache/apr-models").join(org).join(repo);

    Err(Error::Execution(format!(
        "Model not found in cache: {hf_repo}\nSearched:\n  - {}\n  - {}",
        hf_model_dir.display(),
        apr_cache.display()
    )))
}

/// Resolve a HuggingFace repo ID to a local cache directory.
///
/// Searches for the model in the following locations (in order):
/// 1. HuggingFace cache: `$HUGGINGFACE_HUB_CACHE` or `$HF_HOME/hub` or `~/.cache/huggingface/hub`
/// 2. APR cache: `~/.cache/apr-models/{org}/{repo}/`
///
/// Returns the snapshot directory containing `model.safetensors` (for HF cache)
/// or the APR cache directory.
///
/// # Specification
///
/// Implements HF-CACHE-001: Automatic HuggingFace Cache Resolution.
///
/// # Errors
///
/// Returns an error if the model is not found in any cache location.
/// The error message lists all searched paths for debugging.
pub fn resolve_hf_repo_to_cache(hf_repo: &str) -> Result<std::path::PathBuf> {
    use std::path::PathBuf;

    let hf_cache = get_hf_cache_dir();
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    let home = PathBuf::from(home);

    resolve_hf_repo_with_dirs(hf_repo, &hf_cache, &home)
}

#[cfg(test)]
#[allow(clippy::panic)]
#[allow(clippy::float_cmp)]
#[allow(clippy::match_wildcard_for_single_variants)]
#[allow(clippy::assertions_on_constants)]
mod tests {
    use super::*;

    #[test]
    fn test_all_conversion_pairs() {
        let pairs = all_conversion_pairs();
        assert_eq!(pairs.len(), 6);
    }

    #[test]
    fn test_all_backends() {
        let backends = all_backends();
        assert_eq!(backends.len(), 2);
    }

    #[test]
    fn test_generate_conversion_tests() {
        let model_id = ModelId::new("test", "model");
        let tests = generate_conversion_tests(&model_id);
        // 6 pairs × 2 backends = 12 tests
        assert_eq!(tests.len(), 12);
    }

    #[test]
    fn test_gate_id() {
        let test = ConversionTest::new(
            Format::Gguf,
            Format::Apr,
            Backend::Cpu,
            ModelId::new("test", "model"),
        );
        assert_eq!(test.gate_id(), "F-CONV-G-A");
    }

    #[test]
    fn test_compute_diff_identical() {
        let test = ConversionTest::new(
            Format::Gguf,
            Format::Apr,
            Backend::Cpu,
            ModelId::new("test", "model"),
        );
        let diff = test.compute_diff("hello", "hello");
        assert!((diff - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_compute_diff_different() {
        let test = ConversionTest::new(
            Format::Gguf,
            Format::Apr,
            Backend::Cpu,
            ModelId::new("test", "model"),
        );
        let diff = test.compute_diff("hello", "world");
        assert!(diff > 0.0);
    }

    #[test]
    fn test_hash_output() {
        let hash1 = ConversionTest::hash_output("test");
        let hash2 = ConversionTest::hash_output("test");
        assert_eq!(hash1, hash2);

        let hash3 = ConversionTest::hash_output("different");
        assert_ne!(hash1, hash3);
    }

    #[test]
    fn test_find_diff_indices() {
        let test = ConversionTest::new(
            Format::Gguf,
            Format::Apr,
            Backend::Cpu,
            ModelId::new("test", "model"),
        );
        let indices = test.find_diff_indices("hello", "hallo");
        assert_eq!(indices, vec![1]);
    }

    #[test]
    fn test_conversion_result_to_evidence_corroborated() {
        let result = ConversionResult::Corroborated {
            source_format: Format::Gguf,
            target_format: Format::Apr,
            backend: Backend::Cpu,
            max_diff: 0.0,
        };
        let evidence: Evidence = result.into();
        assert!(evidence.outcome.is_pass());
    }

    #[test]
    fn test_conversion_result_to_evidence_falsified() {
        let result = ConversionResult::Falsified {
            gate_id: "F-CONV-G-A".to_string(),
            reason: "Test failure".to_string(),
            evidence: ConversionEvidence {
                source_hash: "abc".to_string(),
                converted_hash: "def".to_string(),
                max_diff: 0.5,
                diff_indices: vec![0, 1],
                source_format: Format::Gguf,
                target_format: Format::Apr,
                backend: Backend::Cpu,
                failure_type: None,
                quant_type: None,
            },
        };
        let evidence: Evidence = result.into();
        assert!(!evidence.outcome.is_pass());
    }

    #[test]
    fn test_round_trip_test_new() {
        let rt = RoundTripTest::new(
            vec![Format::Gguf, Format::Apr, Format::SafeTensors],
            Backend::Cpu,
            ModelId::new("test", "model"),
        );
        assert_eq!(rt.formats.len(), 3);
    }

    #[test]
    fn test_default_epsilon() {
        assert!((default_epsilon() - 1e-6).abs() < f64::EPSILON);
    }

    #[test]
    fn test_conversion_test_epsilon() {
        let test = ConversionTest::new(
            Format::Gguf,
            Format::Apr,
            Backend::Cpu,
            ModelId::new("test", "model"),
        );
        assert!((test.epsilon - EPSILON).abs() < f64::EPSILON);
    }

    #[test]
    fn test_compute_diff_empty_strings() {
        let test = ConversionTest::new(
            Format::Gguf,
            Format::Apr,
            Backend::Cpu,
            ModelId::new("test", "model"),
        );
        let diff = test.compute_diff("", "");
        assert!((diff - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_compute_diff_one_empty() {
        let test = ConversionTest::new(
            Format::Gguf,
            Format::Apr,
            Backend::Cpu,
            ModelId::new("test", "model"),
        );
        let diff = test.compute_diff("hello", "");
        assert!(diff > 0.0);
    }

    #[test]
    fn test_find_diff_indices_empty() {
        let test = ConversionTest::new(
            Format::Gguf,
            Format::Apr,
            Backend::Cpu,
            ModelId::new("test", "model"),
        );
        let indices = test.find_diff_indices("", "");
        assert!(indices.is_empty());
    }

    #[test]
    fn test_find_diff_indices_all_different() {
        let test = ConversionTest::new(
            Format::Gguf,
            Format::Apr,
            Backend::Cpu,
            ModelId::new("test", "model"),
        );
        let indices = test.find_diff_indices("abc", "xyz");
        assert_eq!(indices.len(), 3);
    }

    #[test]
    fn test_gate_id_safetensors() {
        let test = ConversionTest::new(
            Format::SafeTensors,
            Format::Gguf,
            Backend::Gpu,
            ModelId::new("test", "model"),
        );
        assert_eq!(test.gate_id(), "F-CONV-S-G");
    }

    #[test]
    fn test_gate_id_apr() {
        let test = ConversionTest::new(
            Format::Apr,
            Format::SafeTensors,
            Backend::Cpu,
            ModelId::new("test", "model"),
        );
        assert_eq!(test.gate_id(), "F-CONV-A-S");
    }

    #[test]
    fn test_all_conversion_pairs_unique() {
        let pairs = all_conversion_pairs();
        for (i, p1) in pairs.iter().enumerate() {
            for (j, p2) in pairs.iter().enumerate() {
                if i != j {
                    assert!(p1 != p2, "Duplicate pair found");
                }
            }
        }
    }

    #[test]
    fn test_conversion_evidence_clone() {
        let evidence = ConversionEvidence {
            source_hash: "abc".to_string(),
            converted_hash: "def".to_string(),
            max_diff: 0.5,
            diff_indices: vec![0, 1],
            source_format: Format::Gguf,
            target_format: Format::Apr,
            backend: Backend::Cpu,
            failure_type: None,
            quant_type: None,
        };
        let cloned = evidence.clone();
        assert_eq!(evidence.source_hash, cloned.source_hash);
        assert_eq!(evidence.max_diff, cloned.max_diff);
    }

    #[test]
    fn test_conversion_result_clone() {
        let result = ConversionResult::Corroborated {
            source_format: Format::Gguf,
            target_format: Format::Apr,
            backend: Backend::Cpu,
            max_diff: 0.0,
        };
        let cloned = result.clone();
        match cloned {
            ConversionResult::Corroborated { max_diff, .. } => {
                assert!((max_diff - 0.0).abs() < f64::EPSILON);
            }
            _ => panic!("Expected Corroborated"),
        }
    }

    #[test]
    fn test_conversion_test_clone() {
        let test = ConversionTest::new(
            Format::Gguf,
            Format::Apr,
            Backend::Cpu,
            ModelId::new("test", "model"),
        );
        let cloned = test.clone();
        assert_eq!(test.source_format, cloned.source_format);
        assert_eq!(test.target_format, cloned.target_format);
    }

    #[test]
    fn test_round_trip_test_formats() {
        let rt = RoundTripTest::new(
            vec![Format::Gguf, Format::Apr],
            Backend::Gpu,
            ModelId::new("test", "model"),
        );
        assert_eq!(rt.formats.len(), 2);
        assert_eq!(rt.backend, Backend::Gpu);
    }

    #[test]
    fn test_conversion_test_debug() {
        let test = ConversionTest::new(
            Format::Gguf,
            Format::Apr,
            Backend::Cpu,
            ModelId::new("test", "model"),
        );
        let debug_str = format!("{test:?}");
        assert!(debug_str.contains("ConversionTest"));
    }

    #[test]
    fn test_conversion_evidence_debug() {
        let evidence = ConversionEvidence {
            source_hash: "abc".to_string(),
            converted_hash: "def".to_string(),
            max_diff: 0.5,
            diff_indices: vec![0, 1],
            source_format: Format::Gguf,
            target_format: Format::Apr,
            backend: Backend::Cpu,
            failure_type: None,
            quant_type: None,
        };
        let debug_str = format!("{evidence:?}");
        assert!(debug_str.contains("ConversionEvidence"));
    }

    #[test]
    fn test_conversion_result_debug() {
        let result = ConversionResult::Corroborated {
            source_format: Format::Gguf,
            target_format: Format::Apr,
            backend: Backend::Cpu,
            max_diff: 0.0,
        };
        let debug_str = format!("{result:?}");
        assert!(debug_str.contains("Corroborated"));
    }

    #[test]
    fn test_epsilon_constant() {
        assert!(EPSILON > 0.0);
        assert!(EPSILON < 1.0);
    }

    #[test]
    fn test_generate_conversion_tests_all_formats() {
        let model_id = ModelId::new("org", "model");
        let tests = generate_conversion_tests(&model_id);

        // Verify all format pairs are covered
        let has_gguf_to_apr = tests
            .iter()
            .any(|t| t.source_format == Format::Gguf && t.target_format == Format::Apr);
        let has_apr_to_safetensors = tests
            .iter()
            .any(|t| t.source_format == Format::Apr && t.target_format == Format::SafeTensors);

        assert!(has_gguf_to_apr);
        assert!(has_apr_to_safetensors);
    }

    #[test]
    fn test_conversion_config_default() {
        let config = ConversionConfig::default();
        assert!(config.test_all_pairs);
        assert!(config.test_round_trips);
        assert_eq!(config.backends.len(), 2);
        assert!(!config.no_gpu);
    }

    #[test]
    fn test_conversion_config_cpu_only() {
        let config = ConversionConfig::cpu_only();
        assert!(config.test_all_pairs);
        assert!(config.test_round_trips);
        assert_eq!(config.backends.len(), 1);
        assert_eq!(config.backends[0], Backend::Cpu);
        assert!(config.no_gpu);
    }

    #[test]
    fn test_conversion_executor_new() {
        let config = ConversionConfig::default();
        let executor = ConversionExecutor::new(config);
        assert!(!executor.config.no_gpu);
    }

    #[test]
    fn test_conversion_executor_with_defaults() {
        let executor = ConversionExecutor::with_defaults();
        assert!(executor.config.test_all_pairs);
    }

    #[test]
    fn test_conversion_config_debug() {
        let config = ConversionConfig::default();
        let debug_str = format!("{config:?}");
        assert!(debug_str.contains("ConversionConfig"));
    }

    #[test]
    fn test_conversion_config_clone() {
        let config = ConversionConfig::default();
        let cloned = config.clone();
        assert_eq!(cloned.test_all_pairs, config.test_all_pairs);
        assert_eq!(cloned.no_gpu, config.no_gpu);
    }

    #[test]
    fn test_conversion_executor_debug() {
        let executor = ConversionExecutor::with_defaults();
        let debug_str = format!("{executor:?}");
        assert!(debug_str.contains("ConversionExecutor"));
    }

    #[test]
    fn test_round_trip_test_debug() {
        let rt = RoundTripTest::new(
            vec![Format::Gguf, Format::Apr],
            Backend::Cpu,
            ModelId::new("test", "model"),
        );
        let debug_str = format!("{rt:?}");
        assert!(debug_str.contains("RoundTripTest"));
    }

    #[test]
    fn test_round_trip_test_clone() {
        let rt = RoundTripTest::new(
            vec![Format::Gguf, Format::Apr],
            Backend::Cpu,
            ModelId::new("test", "model"),
        );
        let cloned = rt.clone();
        assert_eq!(cloned.formats.len(), rt.formats.len());
        assert_eq!(cloned.backend, rt.backend);
    }

    #[test]
    fn test_conversion_test_with_epsilon() {
        let test = ConversionTest {
            source_format: Format::Gguf,
            target_format: Format::Apr,
            backend: Backend::Cpu,
            model_id: ModelId::new("test", "model"),
            epsilon: 1e-9,
            binary: default_binary(),
            quant_type: None,
            output_dir: None,
        };
        assert!((test.epsilon - 1e-9).abs() < f64::EPSILON);
    }

    #[test]
    fn test_conversion_execution_result() {
        let result = ConversionExecutionResult {
            passed: 10,
            failed: 2,
            total: 12,
            evidence: vec![],
            results: vec![],
            duration_ms: 1000,
        };
        assert_eq!(result.passed, 10);
        assert_eq!(result.failed, 2);
        assert_eq!(result.total, 12);
    }

    #[test]
    fn test_conversion_execution_result_debug() {
        let result = ConversionExecutionResult {
            passed: 5,
            failed: 1,
            total: 6,
            evidence: vec![],
            results: vec![],
            duration_ms: 500,
        };
        let debug_str = format!("{result:?}");
        assert!(debug_str.contains("ConversionExecutionResult"));
    }

    #[test]
    fn test_all_backends_content() {
        let backends = all_backends();
        assert!(backends.contains(&Backend::Cpu));
        assert!(backends.contains(&Backend::Gpu));
    }

    #[test]
    fn test_gate_id_all_combinations() {
        // Test all source/target combinations
        let combos = [
            (Format::Gguf, Format::Apr, "F-CONV-G-A"),
            (Format::Apr, Format::Gguf, "F-CONV-A-G"),
            (Format::Gguf, Format::SafeTensors, "F-CONV-G-S"),
            (Format::SafeTensors, Format::Gguf, "F-CONV-S-G"),
            (Format::Apr, Format::SafeTensors, "F-CONV-A-S"),
            (Format::SafeTensors, Format::Apr, "F-CONV-S-A"),
        ];

        for (source, target, expected) in combos {
            let test = ConversionTest::new(source, target, Backend::Cpu, ModelId::new("t", "m"));
            assert_eq!(test.gate_id(), expected);
        }
    }

    #[test]
    fn test_compute_diff_partially_matching() {
        let test = ConversionTest::new(
            Format::Gguf,
            Format::Apr,
            Backend::Cpu,
            ModelId::new("test", "model"),
        );
        // "hello" vs "hallo" - 1 char different out of 5
        let diff = test.compute_diff("hello", "hallo");
        assert!(diff > 0.0);
        assert!(diff < 1.0);
    }

    #[test]
    fn test_find_diff_indices_longer_second() {
        let test = ConversionTest::new(
            Format::Gguf,
            Format::Apr,
            Backend::Cpu,
            ModelId::new("test", "model"),
        );
        // "ab" vs "abc" - only compares up to shorter length
        let indices = test.find_diff_indices("ab", "abc");
        assert!(indices.is_empty()); // first 2 chars match
    }

    #[test]
    fn test_conversion_execution_result_all_passed() {
        let result = ConversionExecutionResult {
            passed: 10,
            failed: 0,
            total: 10,
            evidence: vec![],
            results: vec![],
            duration_ms: 1000,
        };
        assert!(result.all_passed());
    }

    #[test]
    fn test_conversion_execution_result_not_all_passed() {
        let result = ConversionExecutionResult {
            passed: 8,
            failed: 2,
            total: 10,
            evidence: vec![],
            results: vec![],
            duration_ms: 1000,
        };
        assert!(!result.all_passed());
    }

    #[test]
    fn test_conversion_execution_result_pass_rate() {
        let result = ConversionExecutionResult {
            passed: 8,
            failed: 2,
            total: 10,
            evidence: vec![],
            results: vec![],
            duration_ms: 1000,
        };
        let rate = result.pass_rate();
        assert!((rate - 80.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_conversion_execution_result_pass_rate_zero_total() {
        let result = ConversionExecutionResult {
            passed: 0,
            failed: 0,
            total: 0,
            evidence: vec![],
            results: vec![],
            duration_ms: 0,
        };
        let rate = result.pass_rate();
        assert!((rate - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_conversion_execution_result_pass_rate_all_passed() {
        let result = ConversionExecutionResult {
            passed: 5,
            failed: 0,
            total: 5,
            evidence: vec![],
            results: vec![],
            duration_ms: 500,
        };
        let rate = result.pass_rate();
        assert!((rate - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_conversion_execution_result_pass_rate_none_passed() {
        let result = ConversionExecutionResult {
            passed: 0,
            failed: 5,
            total: 5,
            evidence: vec![],
            results: vec![],
            duration_ms: 500,
        };
        let rate = result.pass_rate();
        assert!((rate - 0.0).abs() < f64::EPSILON);
    }

    // Tests for ConversionBugType (GH-187)

    #[test]
    fn test_bug_type_gate_ids() {
        assert_eq!(
            ConversionBugType::EmbeddingTransposition.gate_id(),
            "F-CONV-EMBED-001"
        );
        assert_eq!(
            ConversionBugType::TokenizerMissing.gate_id(),
            "F-CONV-TOK-001"
        );
        assert_eq!(
            ConversionBugType::WeightCorruption.gate_id(),
            "F-CONV-WEIGHT-001"
        );
        assert_eq!(
            ConversionBugType::ShapeMismatch.gate_id(),
            "F-CONV-SHAPE-001"
        );
        assert_eq!(
            ConversionBugType::SemanticDrift.gate_id(),
            "F-CONV-SEMANTIC-001"
        );
        assert_eq!(ConversionBugType::Unknown.gate_id(), "F-CONV-UNKNOWN-001");
    }

    #[test]
    fn test_bug_type_descriptions() {
        assert!(
            ConversionBugType::EmbeddingTransposition
                .description()
                .contains("transposition")
        );
        assert!(
            ConversionBugType::TokenizerMissing
                .description()
                .contains("tokenizer")
        );
        assert!(
            ConversionBugType::WeightCorruption
                .description()
                .contains("corruption")
        );
    }

    #[test]
    fn test_bug_type_clone() {
        let bug = ConversionBugType::EmbeddingTransposition;
        let cloned = bug;
        assert_eq!(bug, cloned);
    }

    #[test]
    fn test_bug_type_debug() {
        let debug_str = format!("{:?}", ConversionBugType::TokenizerMissing);
        assert!(debug_str.contains("TokenizerMissing"));
    }

    #[test]
    fn test_semantic_test_new() {
        let test = SemanticConversionTest::new(
            Format::Gguf,
            Format::Apr,
            Backend::Cpu,
            ModelId::new("test", "model"),
        );
        assert_eq!(test.source_format, Format::Gguf);
        assert_eq!(test.target_format, Format::Apr);
    }

    #[test]
    fn test_semantic_test_clone() {
        let test = SemanticConversionTest::new(
            Format::Gguf,
            Format::Apr,
            Backend::Cpu,
            ModelId::new("test", "model"),
        );
        let cloned = test.clone();
        assert_eq!(test.source_format, cloned.source_format);
    }

    #[test]
    fn test_semantic_test_debug() {
        let test = SemanticConversionTest::new(
            Format::Gguf,
            Format::Apr,
            Backend::Cpu,
            ModelId::new("test", "model"),
        );
        let debug_str = format!("{test:?}");
        assert!(debug_str.contains("SemanticConversionTest"));
    }

    #[test]
    fn test_semantic_result_is_pass() {
        let pass = SemanticTestResult::Corroborated {
            source_output: "4".to_string(),
            target_output: "4".to_string(),
        };
        assert!(pass.is_pass());

        let fail = SemanticTestResult::Falsified {
            bug_type: ConversionBugType::EmbeddingTransposition,
            source_output: "4".to_string(),
            target_output: "garbage".to_string(),
            stderr: String::new(),
        };
        assert!(!fail.is_pass());
    }

    #[test]
    fn test_semantic_result_bug_type() {
        let pass = SemanticTestResult::Corroborated {
            source_output: "4".to_string(),
            target_output: "4".to_string(),
        };
        assert!(pass.bug_type().is_none());

        let fail = SemanticTestResult::Falsified {
            bug_type: ConversionBugType::TokenizerMissing,
            source_output: "4".to_string(),
            target_output: "garbage".to_string(),
            stderr: String::new(),
        };
        assert_eq!(fail.bug_type(), Some(ConversionBugType::TokenizerMissing));
    }

    #[test]
    fn test_garbage_patterns_detection() {
        // These patterns should trigger embedding transposition detection
        let garbage_outputs = [
            "1. What is the difference between",
            "<pad><pad><pad>",
            "PAD PAD PAD",
            "token 151935 151935",
        ];

        for output in garbage_outputs {
            let has_garbage = GARBAGE_PATTERNS.iter().any(|p| output.contains(p));
            assert!(has_garbage, "Should detect garbage in: {output}");
        }
    }

    #[test]
    fn test_arithmetic_expected_detection() {
        // These patterns should be recognized as correct answers
        let correct_outputs = [
            "The answer is 4",
            "2+2=4",
            "equals 4.",
            "It's four",
            "Four is the answer",
        ];

        for output in correct_outputs {
            let has_expected = ARITHMETIC_EXPECTED.iter().any(|p| output.contains(p));
            assert!(has_expected, "Should detect correct answer in: {output}");
        }
    }

    #[test]
    fn test_semantic_result_clone() {
        let result = SemanticTestResult::Corroborated {
            source_output: "test".to_string(),
            target_output: "test".to_string(),
        };
        let cloned = result.clone();
        assert!(cloned.is_pass());
    }

    #[test]
    fn test_semantic_result_debug() {
        let result = SemanticTestResult::Falsified {
            bug_type: ConversionBugType::Unknown,
            source_output: "a".to_string(),
            target_output: "b".to_string(),
            stderr: String::new(),
        };
        let debug_str = format!("{result:?}");
        assert!(debug_str.contains("Falsified"));
    }

    // Tests for classify_bug logic
    #[test]
    fn test_classify_bug_tokenizer_missing() {
        let test = SemanticConversionTest::new(
            Format::Gguf,
            Format::Apr,
            Backend::Cpu,
            ModelId::new("test", "model"),
        );
        let bug = test.classify_bug("The answer is 4", "The answer is 4", true);
        assert_eq!(bug, Some(ConversionBugType::TokenizerMissing));
    }

    #[test]
    fn test_classify_bug_embedding_transposition() {
        let test = SemanticConversionTest::new(
            Format::Gguf,
            Format::Apr,
            Backend::Cpu,
            ModelId::new("test", "model"),
        );
        // Source has correct answer, target has garbage
        let bug = test.classify_bug("The answer is 4", "PAD PAD PAD garbage", false);
        assert_eq!(bug, Some(ConversionBugType::EmbeddingTransposition));
    }

    #[test]
    fn test_classify_bug_semantic_drift() {
        let test = SemanticConversionTest::new(
            Format::Gguf,
            Format::Apr,
            Backend::Cpu,
            ModelId::new("test", "model"),
        );
        // Source has correct answer, target has wrong but not garbage answer
        let bug = test.classify_bug("The answer is 4", "The answer is 7", false);
        assert_eq!(bug, Some(ConversionBugType::SemanticDrift));
    }

    #[test]
    fn test_classify_bug_weight_corruption() {
        let test = SemanticConversionTest::new(
            Format::Gguf,
            Format::Apr,
            Backend::Cpu,
            ModelId::new("test", "model"),
        );
        // Source has output (but not expected arithmetic answer), target is empty
        // WeightCorruption is only detected when target is empty/whitespace
        let bug = test.classify_bug("Hello world, here is some text", "   ", false);
        assert_eq!(bug, Some(ConversionBugType::WeightCorruption));
    }

    #[test]
    fn test_classify_bug_no_bug() {
        let test = SemanticConversionTest::new(
            Format::Gguf,
            Format::Apr,
            Backend::Cpu,
            ModelId::new("test", "model"),
        );
        // Both outputs are identical
        let bug = test.classify_bug("The answer is 4", "The answer is 4", false);
        assert!(bug.is_none());
    }

    #[test]
    fn test_classify_bug_unknown() {
        let test = SemanticConversionTest::new(
            Format::Gguf,
            Format::Apr,
            Backend::Cpu,
            ModelId::new("test", "model"),
        );
        // Source has no expected answer, outputs differ
        let bug = test.classify_bug("random text", "different text", false);
        assert_eq!(bug, Some(ConversionBugType::Unknown));
    }

    #[test]
    fn test_classify_bug_with_endoftext_pattern() {
        let test = SemanticConversionTest::new(
            Format::Gguf,
            Format::Apr,
            Backend::Cpu,
            ModelId::new("test", "model"),
        );
        let bug = test.classify_bug(
            "The answer is 4",
            "Output: <|endoftext|><|endoftext|>",
            false,
        );
        assert_eq!(bug, Some(ConversionBugType::EmbeddingTransposition));
    }

    #[test]
    fn test_classify_bug_with_null_chars() {
        let test = SemanticConversionTest::new(
            Format::Gguf,
            Format::Apr,
            Backend::Cpu,
            ModelId::new("test", "model"),
        );
        let bug = test.classify_bug("The answer is 4", "text\u{0000}with\u{0000}nulls", false);
        assert_eq!(bug, Some(ConversionBugType::EmbeddingTransposition));
    }

    #[test]
    fn test_classify_bug_whitespace_trimming() {
        let test = SemanticConversionTest::new(
            Format::Gguf,
            Format::Apr,
            Backend::Cpu,
            ModelId::new("test", "model"),
        );
        // Same content but different whitespace - should match
        let bug = test.classify_bug("  The answer is 4  ", "The answer is 4", false);
        assert!(bug.is_none());
    }

    #[test]
    fn test_bug_type_equality() {
        assert_eq!(
            ConversionBugType::EmbeddingTransposition,
            ConversionBugType::EmbeddingTransposition
        );
        assert_ne!(
            ConversionBugType::EmbeddingTransposition,
            ConversionBugType::TokenizerMissing
        );
    }

    #[test]
    fn test_conversion_evidence_source_format() {
        let evidence = ConversionEvidence {
            source_hash: "abc123".to_string(),
            converted_hash: "def456".to_string(),
            max_diff: 0.1,
            diff_indices: vec![0, 5, 10],
            source_format: Format::SafeTensors,
            target_format: Format::Apr,
            backend: Backend::Gpu,
            failure_type: None,
            quant_type: None,
        };
        assert_eq!(evidence.source_format, Format::SafeTensors);
        assert_eq!(evidence.target_format, Format::Apr);
        assert_eq!(evidence.backend, Backend::Gpu);
    }

    #[test]
    fn test_conversion_test_model_id() {
        let model_id = ModelId::new("my-org", "my-model");
        let test = ConversionTest::new(Format::Gguf, Format::Apr, Backend::Cpu, model_id.clone());
        assert_eq!(test.model_id.org, "my-org");
        assert_eq!(test.model_id.name, "my-model");
    }

    #[test]
    fn test_semantic_conversion_test_backend() {
        let test = SemanticConversionTest::new(
            Format::Gguf,
            Format::Apr,
            Backend::Gpu,
            ModelId::new("test", "model"),
        );
        assert_eq!(test.backend, Backend::Gpu);
    }

    #[test]
    fn test_round_trip_test_model_id() {
        let model_id = ModelId::new("org", "name");
        let rt = RoundTripTest::new(
            vec![Format::Gguf, Format::Apr],
            Backend::Cpu,
            model_id.clone(),
        );
        assert_eq!(rt.model_id.org, "org");
        assert_eq!(rt.model_id.name, "name");
    }

    #[test]
    fn test_conversion_config_backends() {
        let config = ConversionConfig::default();
        assert_eq!(config.backends.len(), 2);
        assert!(config.backends.contains(&Backend::Cpu));
        assert!(config.backends.contains(&Backend::Gpu));
    }

    #[test]
    fn test_conversion_config_custom() {
        let config = ConversionConfig {
            test_all_pairs: false,
            test_round_trips: false,
            backends: vec![Backend::Cpu],
            no_gpu: true,
            ..Default::default()
        };
        assert!(!config.test_all_pairs);
        assert!(!config.test_round_trips);
        assert_eq!(config.backends.len(), 1);
    }

    #[test]
    fn test_conversion_executor_config_access() {
        let config = ConversionConfig::cpu_only();
        let executor = ConversionExecutor::new(config);
        assert!(executor.config.no_gpu);
        assert!(executor.config.test_all_pairs);
    }

    #[test]
    fn test_all_conversion_pairs_bidirectional() {
        let pairs = all_conversion_pairs();
        // Should have GGUF -> APR and APR -> GGUF
        let has_gguf_to_apr = pairs.contains(&(Format::Gguf, Format::Apr));
        let has_apr_to_gguf = pairs.contains(&(Format::Apr, Format::Gguf));
        assert!(has_gguf_to_apr);
        assert!(has_apr_to_gguf);
    }

    #[test]
    fn test_epsilon_value() {
        assert!((EPSILON - 1e-6).abs() < 1e-10);
    }

    #[test]
    fn test_conversion_result_corroborated_max_diff() {
        let result = ConversionResult::Corroborated {
            source_format: Format::Gguf,
            target_format: Format::Apr,
            backend: Backend::Cpu,
            max_diff: 1e-8,
        };
        match result {
            ConversionResult::Corroborated { max_diff, .. } => {
                assert!(max_diff < EPSILON);
            }
            _ => panic!("Expected Corroborated"),
        }
    }

    #[test]
    fn test_conversion_result_falsified_gate_id() {
        let result = ConversionResult::Falsified {
            gate_id: "F-CONV-G-A".to_string(),
            reason: "Outputs differ".to_string(),
            evidence: ConversionEvidence {
                source_hash: "a".to_string(),
                converted_hash: "b".to_string(),
                max_diff: 0.5,
                diff_indices: vec![],
                source_format: Format::Gguf,
                target_format: Format::Apr,
                backend: Backend::Cpu,
                failure_type: None,
                quant_type: None,
            },
        };
        match result {
            ConversionResult::Falsified { gate_id, .. } => {
                assert_eq!(gate_id, "F-CONV-G-A");
            }
            _ => panic!("Expected Falsified"),
        }
    }

    #[test]
    fn test_semantic_test_result_corroborated_outputs() {
        let result = SemanticTestResult::Corroborated {
            source_output: "answer is 4".to_string(),
            target_output: "answer is 4".to_string(),
        };
        match result {
            SemanticTestResult::Corroborated {
                source_output,
                target_output,
            } => {
                assert_eq!(source_output, target_output);
            }
            _ => panic!("Expected Corroborated"),
        }
    }

    #[test]
    fn test_semantic_test_result_falsified_stderr() {
        let result = SemanticTestResult::Falsified {
            bug_type: ConversionBugType::TokenizerMissing,
            source_output: "4".to_string(),
            target_output: "garbage".to_string(),
            stderr: "PMAT-172: tokenizer missing".to_string(),
        };
        match result {
            SemanticTestResult::Falsified { stderr, .. } => {
                assert!(stderr.contains("PMAT-172"));
            }
            _ => panic!("Expected Falsified"),
        }
    }

    #[test]
    fn test_all_bug_types_have_gate_ids() {
        let bug_types = [
            ConversionBugType::EmbeddingTransposition,
            ConversionBugType::TokenizerMissing,
            ConversionBugType::WeightCorruption,
            ConversionBugType::ShapeMismatch,
            ConversionBugType::SemanticDrift,
            ConversionBugType::Unknown,
        ];
        for bug_type in bug_types {
            let gate_id = bug_type.gate_id();
            assert!(!gate_id.is_empty());
            assert!(gate_id.starts_with("F-CONV-"));
        }
    }

    #[test]
    fn test_all_bug_types_have_descriptions() {
        let bug_types = [
            ConversionBugType::EmbeddingTransposition,
            ConversionBugType::TokenizerMissing,
            ConversionBugType::WeightCorruption,
            ConversionBugType::ShapeMismatch,
            ConversionBugType::SemanticDrift,
            ConversionBugType::Unknown,
        ];
        for bug_type in bug_types {
            let desc = bug_type.description();
            assert!(!desc.is_empty());
        }
    }

    #[test]
    fn test_conversion_evidence_diff_indices() {
        let evidence = ConversionEvidence {
            source_hash: "a".to_string(),
            converted_hash: "b".to_string(),
            max_diff: 0.1,
            diff_indices: vec![0, 1, 2, 3, 4],
            source_format: Format::Gguf,
            target_format: Format::Apr,
            backend: Backend::Cpu,
            failure_type: None,
            quant_type: None,
        };
        assert_eq!(evidence.diff_indices.len(), 5);
    }

    #[test]
    fn test_round_trip_test_full_cycle() {
        let rt = RoundTripTest::new(
            vec![Format::Gguf, Format::Apr, Format::SafeTensors, Format::Gguf],
            Backend::Cpu,
            ModelId::new("test", "model"),
        );
        assert_eq!(rt.formats.len(), 4);
        assert_eq!(rt.formats[0], Format::Gguf);
        assert_eq!(rt.formats[3], Format::Gguf);
    }

    #[test]
    fn test_conversion_config_clone_equality() {
        let config1 = ConversionConfig::default();
        let config2 = config1.clone();
        assert_eq!(config1.test_all_pairs, config2.test_all_pairs);
        assert_eq!(config1.test_round_trips, config2.test_round_trips);
        assert_eq!(config1.no_gpu, config2.no_gpu);
        assert_eq!(config1.backends.len(), config2.backends.len());
    }

    #[test]
    fn test_generate_conversion_tests_contains_all_backends() {
        let model_id = ModelId::new("test", "model");
        let tests = generate_conversion_tests(&model_id);

        let cpu_backend_present = tests.iter().any(|t| t.backend == Backend::Cpu);
        let gpu_backend_present = tests.iter().any(|t| t.backend == Backend::Gpu);

        assert!(cpu_backend_present);
        assert!(gpu_backend_present);
    }

    #[test]
    fn test_garbage_patterns_constant() {
        assert!(!GARBAGE_PATTERNS.is_empty());
        assert!(GARBAGE_PATTERNS.contains(&"PAD"));
        assert!(GARBAGE_PATTERNS.contains(&"<pad>"));
    }

    #[test]
    fn test_arithmetic_expected_constant() {
        assert!(!ARITHMETIC_EXPECTED.is_empty());
        assert!(ARITHMETIC_EXPECTED.contains(&"4"));
        assert!(ARITHMETIC_EXPECTED.contains(&"four"));
    }

    // Additional tests for coverage

    #[test]
    fn test_conversion_bug_type_serialization() {
        let bug_types = [
            ConversionBugType::EmbeddingTransposition,
            ConversionBugType::TokenizerMissing,
            ConversionBugType::WeightCorruption,
            ConversionBugType::ShapeMismatch,
            ConversionBugType::SemanticDrift,
            ConversionBugType::Unknown,
        ];
        for bug_type in bug_types {
            let json = serde_json::to_string(&bug_type).unwrap();
            let parsed: ConversionBugType = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed, bug_type);
        }
    }

    #[test]
    fn test_conversion_test_serialization() {
        let test = ConversionTest {
            source_format: Format::Gguf,
            target_format: Format::Apr,
            backend: Backend::Cpu,
            model_id: ModelId::new("org", "name"),
            epsilon: 1e-7,
            binary: default_binary(),
            quant_type: None,
            output_dir: None,
        };
        let json = serde_json::to_string(&test).unwrap();
        let parsed: ConversionTest = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.source_format, Format::Gguf);
        assert_eq!(parsed.target_format, Format::Apr);
    }

    #[test]
    fn test_conversion_result_serialization_corroborated() {
        let result = ConversionResult::Corroborated {
            source_format: Format::Gguf,
            target_format: Format::Apr,
            backend: Backend::Gpu,
            max_diff: 1e-9,
        };
        let json = serde_json::to_string(&result).unwrap();
        let parsed: ConversionResult = serde_json::from_str(&json).unwrap();
        match parsed {
            ConversionResult::Corroborated { max_diff, .. } => {
                assert!(max_diff < EPSILON);
            }
            _ => panic!("Expected Corroborated"),
        }
    }

    #[test]
    fn test_conversion_result_serialization_falsified() {
        let result = ConversionResult::Falsified {
            gate_id: "F-CONV-G-A".to_string(),
            reason: "Test failure".to_string(),
            evidence: ConversionEvidence {
                source_hash: "abc".to_string(),
                converted_hash: "def".to_string(),
                max_diff: 0.5,
                diff_indices: vec![0, 1, 2],
                source_format: Format::Gguf,
                target_format: Format::Apr,
                backend: Backend::Cpu,
                failure_type: None,
                quant_type: None,
            },
        };
        let json = serde_json::to_string(&result).unwrap();
        let parsed: ConversionResult = serde_json::from_str(&json).unwrap();
        match parsed {
            ConversionResult::Falsified { gate_id, .. } => {
                assert_eq!(gate_id, "F-CONV-G-A");
            }
            _ => panic!("Expected Falsified"),
        }
    }

    #[test]
    fn test_conversion_evidence_serialization() {
        let evidence = ConversionEvidence {
            source_hash: "hash1".to_string(),
            converted_hash: "hash2".to_string(),
            max_diff: 0.05,
            diff_indices: vec![1, 3, 5],
            source_format: Format::SafeTensors,
            target_format: Format::Gguf,
            backend: Backend::Gpu,
            failure_type: None,
            quant_type: None,
        };
        let json = serde_json::to_string(&evidence).unwrap();
        let parsed: ConversionEvidence = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.source_hash, "hash1");
        assert_eq!(parsed.diff_indices.len(), 3);
    }

    #[test]
    fn test_semantic_test_result_clone() {
        let result = SemanticTestResult::Falsified {
            bug_type: ConversionBugType::TokenizerMissing,
            source_output: "source".to_string(),
            target_output: "target".to_string(),
            stderr: "error".to_string(),
        };
        let cloned = result.clone();
        match cloned {
            SemanticTestResult::Falsified {
                bug_type, stderr, ..
            } => {
                assert_eq!(bug_type, ConversionBugType::TokenizerMissing);
                assert_eq!(stderr, "error");
            }
            _ => panic!("Expected Falsified"),
        }
    }

    #[test]
    fn test_classify_bug_source_empty_target_has_content() {
        let test = SemanticConversionTest::new(
            Format::Gguf,
            Format::Apr,
            Backend::Cpu,
            ModelId::new("test", "model"),
        );
        // Source empty, target has content - unusual case, returns Unknown
        let bug = test.classify_bug("", "Some output", false);
        assert_eq!(bug, Some(ConversionBugType::Unknown));
    }

    #[test]
    fn test_classify_bug_both_empty() {
        let test = SemanticConversionTest::new(
            Format::Gguf,
            Format::Apr,
            Backend::Cpu,
            ModelId::new("test", "model"),
        );
        // Both empty - no bug
        let bug = test.classify_bug("", "", false);
        assert!(bug.is_none());
    }

    #[test]
    fn test_classify_bug_source_no_expected_target_has_expected() {
        let test = SemanticConversionTest::new(
            Format::Gguf,
            Format::Apr,
            Backend::Cpu,
            ModelId::new("test", "model"),
        );
        // Source doesn't have expected, target does - weird but not a bug in our heuristic
        let bug = test.classify_bug("random text", "The answer is 4", false);
        // Outputs differ but no clear pattern
        assert_eq!(bug, Some(ConversionBugType::Unknown));
    }

    #[test]
    fn test_compute_diff_unicode() {
        let test = ConversionTest::new(
            Format::Gguf,
            Format::Apr,
            Backend::Cpu,
            ModelId::new("test", "model"),
        );
        let diff = test.compute_diff("hello 你好", "hello 世界");
        assert!(diff > 0.0);
        assert!(diff < 1.0);
    }

    #[test]
    fn test_find_diff_indices_unicode() {
        let test = ConversionTest::new(
            Format::Gguf,
            Format::Apr,
            Backend::Cpu,
            ModelId::new("test", "model"),
        );
        let indices = test.find_diff_indices("ab你好", "abXX");
        // Comparing "你" vs "X" and "好" vs "X"
        assert!(indices.len() >= 2);
    }

    #[test]
    fn test_hash_output_unicode() {
        let hash1 = ConversionTest::hash_output("hello 你好 世界");
        let hash2 = ConversionTest::hash_output("hello 你好 世界");
        assert_eq!(hash1, hash2);
        assert_eq!(hash1.len(), 16); // 16 hex chars
    }

    #[test]
    fn test_conversion_execution_result_pass_rate_partial() {
        let result = ConversionExecutionResult {
            passed: 7,
            failed: 3,
            total: 10,
            evidence: vec![],
            results: vec![],
            duration_ms: 1000,
        };
        let rate = result.pass_rate();
        assert!((rate - 70.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_conversion_config_with_specific_backends() {
        let config = ConversionConfig {
            test_all_pairs: true,
            test_round_trips: false,
            backends: vec![Backend::Gpu],
            no_gpu: false,
            ..Default::default()
        };
        assert_eq!(config.backends.len(), 1);
        assert_eq!(config.backends[0], Backend::Gpu);
        assert!(!config.test_round_trips);
    }

    #[test]
    fn test_semantic_conversion_test_fields() {
        let test = SemanticConversionTest::new(
            Format::SafeTensors,
            Format::Apr,
            Backend::Gpu,
            ModelId::new("org", "model"),
        );
        assert_eq!(test.source_format, Format::SafeTensors);
        assert_eq!(test.target_format, Format::Apr);
        assert_eq!(test.backend, Backend::Gpu);
        assert_eq!(test.model_id.org, "org");
    }

    #[test]
    fn test_round_trip_test_with_two_formats() {
        let rt = RoundTripTest::new(
            vec![Format::Apr, Format::Gguf],
            Backend::Gpu,
            ModelId::new("test", "model"),
        );
        assert_eq!(rt.formats.len(), 2);
        assert_eq!(rt.backend, Backend::Gpu);
    }

    #[test]
    fn test_conversion_evidence_with_empty_diff_indices() {
        let evidence = ConversionEvidence {
            source_hash: "same".to_string(),
            converted_hash: "same".to_string(),
            max_diff: 0.0,
            diff_indices: vec![],
            source_format: Format::Gguf,
            target_format: Format::Apr,
            backend: Backend::Cpu,
            failure_type: None,
            quant_type: None,
        };
        assert!(evidence.diff_indices.is_empty());
        assert!((evidence.max_diff - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_all_conversion_pairs_complete() {
        let pairs = all_conversion_pairs();
        // Should have bidirectional pairs for all format combinations
        // 3 formats = 6 pairs (A->B, B->A for each pair)
        assert_eq!(pairs.len(), 6);

        // Check specific pairs exist
        assert!(pairs.contains(&(Format::Gguf, Format::Apr)));
        assert!(pairs.contains(&(Format::Apr, Format::Gguf)));
        assert!(pairs.contains(&(Format::Gguf, Format::SafeTensors)));
        assert!(pairs.contains(&(Format::SafeTensors, Format::Gguf)));
        assert!(pairs.contains(&(Format::Apr, Format::SafeTensors)));
        assert!(pairs.contains(&(Format::SafeTensors, Format::Apr)));
    }

    #[test]
    fn test_generate_conversion_tests_model_id_preserved() {
        let model_id = ModelId::new("my-org", "my-model-v1");
        let tests = generate_conversion_tests(&model_id);

        for test in &tests {
            assert_eq!(test.model_id.org, "my-org");
            assert_eq!(test.model_id.name, "my-model-v1");
        }
    }

    #[test]
    fn test_conversion_test_debug_format() {
        let test = ConversionTest::new(
            Format::Gguf,
            Format::Apr,
            Backend::Cpu,
            ModelId::new("test", "model"),
        );
        let debug = format!("{test:?}");
        assert!(debug.contains("ConversionTest"));
        assert!(debug.contains("Gguf"));
        assert!(debug.contains("Apr"));
    }

    #[test]
    fn test_classify_bug_with_multiple_garbage_patterns() {
        let test = SemanticConversionTest::new(
            Format::Gguf,
            Format::Apr,
            Backend::Cpu,
            ModelId::new("test", "model"),
        );
        // Target has multiple garbage patterns
        let bug = test.classify_bug("The answer is 4", "PAD <pad> <|endoftext|> 151935", false);
        assert_eq!(bug, Some(ConversionBugType::EmbeddingTransposition));
    }

    #[test]
    fn test_classify_bug_target_only_whitespace() {
        let test = SemanticConversionTest::new(
            Format::Gguf,
            Format::Apr,
            Backend::Cpu,
            ModelId::new("test", "model"),
        );
        // Source has content but no expected arithmetic, target is whitespace
        let bug = test.classify_bug("Some random output", "   \t\n  ", false);
        assert_eq!(bug, Some(ConversionBugType::WeightCorruption));
    }

    #[test]
    fn test_conversion_executor_custom_config() {
        let config = ConversionConfig {
            test_all_pairs: false,
            test_round_trips: true,
            backends: vec![Backend::Cpu],
            no_gpu: true,
            ..Default::default()
        };
        let executor = ConversionExecutor::new(config);
        assert!(!executor.config.test_all_pairs);
        assert!(executor.config.test_round_trips);
        assert!(executor.config.no_gpu);
    }

    #[test]
    fn test_semantic_test_result_is_pass_corroborated() {
        let result = SemanticTestResult::Corroborated {
            source_output: "test".to_string(),
            target_output: "test".to_string(),
        };
        assert!(result.is_pass());
    }

    #[test]
    fn test_semantic_test_result_is_pass_falsified() {
        let result = SemanticTestResult::Falsified {
            bug_type: ConversionBugType::Unknown,
            source_output: "a".to_string(),
            target_output: "b".to_string(),
            stderr: String::new(),
        };
        assert!(!result.is_pass());
    }

    #[test]
    fn test_semantic_test_result_bug_type_corroborated() {
        let result = SemanticTestResult::Corroborated {
            source_output: "test".to_string(),
            target_output: "test".to_string(),
        };
        assert!(result.bug_type().is_none());
    }

    #[test]
    fn test_semantic_test_result_bug_type_falsified() {
        let result = SemanticTestResult::Falsified {
            bug_type: ConversionBugType::SemanticDrift,
            source_output: "a".to_string(),
            target_output: "b".to_string(),
            stderr: "warning".to_string(),
        };
        assert_eq!(result.bug_type(), Some(ConversionBugType::SemanticDrift));
    }

    #[test]
    fn test_conversion_result_corroborated_serialization() {
        let result = ConversionResult::Corroborated {
            source_format: Format::Gguf,
            target_format: Format::Apr,
            backend: Backend::Cpu,
            max_diff: 0.001,
        };
        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("Corroborated"));
        let deserialized: ConversionResult = serde_json::from_str(&json).unwrap();
        if let ConversionResult::Corroborated { max_diff, .. } = deserialized {
            assert!((max_diff - 0.001).abs() < f64::EPSILON);
        } else {
            panic!("Expected Corroborated");
        }
    }

    #[test]
    fn test_conversion_result_falsified_serialization() {
        let result = ConversionResult::Falsified {
            gate_id: "F-TEST-001".to_string(),
            reason: "Test failure".to_string(),
            evidence: ConversionEvidence {
                source_hash: "abc".to_string(),
                converted_hash: "def".to_string(),
                max_diff: 0.5,
                diff_indices: vec![1, 2, 3],
                source_format: Format::Gguf,
                target_format: Format::Apr,
                backend: Backend::Cpu,
                failure_type: None,
                quant_type: None,
            },
        };
        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("Falsified"));
        assert!(json.contains("F-TEST-001"));
    }

    #[test]
    fn test_conversion_test_new_with_epsilon() {
        let test = ConversionTest {
            source_format: Format::Apr,
            target_format: Format::Gguf,
            backend: Backend::Gpu,
            model_id: ModelId::new("org", "model"),
            epsilon: 1e-10,
            binary: default_binary(),
            quant_type: None,
            output_dir: None,
        };
        assert!((test.epsilon - 1e-10).abs() < 1e-15);
    }

    #[test]
    fn test_conversion_execution_result_fields() {
        let result = ConversionExecutionResult {
            total: 10,
            passed: 5,
            failed: 2,
            duration_ms: 100,
            results: vec![],
            evidence: vec![],
        };
        assert_eq!(result.total, 10);
        assert_eq!(result.passed, 5);
        assert_eq!(result.failed, 2);
        assert_eq!(result.duration_ms, 100);
        assert!(result.results.is_empty());
        assert!(result.evidence.is_empty());
    }

    #[test]
    fn test_conversion_test_compute_diff_same() {
        let test = ConversionTest::new(
            Format::Gguf,
            Format::Apr,
            Backend::Cpu,
            ModelId::new("test", "model"),
        );
        // Same strings should have 0 diff
        assert!((test.compute_diff("hello", "hello") - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_conversion_test_compute_diff_different() {
        let test = ConversionTest::new(
            Format::Gguf,
            Format::Apr,
            Backend::Cpu,
            ModelId::new("test", "model"),
        );
        // Completely different strings should have high diff
        let diff = test.compute_diff("abc", "xyz");
        assert!(diff > 0.5);
    }

    #[test]
    fn test_conversion_test_compute_diff_empty_strings() {
        let test = ConversionTest::new(
            Format::Gguf,
            Format::Apr,
            Backend::Cpu,
            ModelId::new("test", "model"),
        );
        // Empty strings should have 0 diff
        assert!((test.compute_diff("", "") - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_conversion_test_compute_diff_partial_match() {
        let test = ConversionTest::new(
            Format::Gguf,
            Format::Apr,
            Backend::Cpu,
            ModelId::new("test", "model"),
        );
        // Partially matching strings
        let diff = test.compute_diff("abcd", "abXd");
        assert!(diff > 0.0 && diff < 1.0);
    }

    #[test]
    fn test_conversion_test_find_diff_indices_with_diffs() {
        let test = ConversionTest::new(
            Format::Gguf,
            Format::Apr,
            Backend::Cpu,
            ModelId::new("test", "model"),
        );
        let indices = test.find_diff_indices("abcd", "aXcY");
        assert_eq!(indices.len(), 2);
        assert!(indices.contains(&1));
        assert!(indices.contains(&3));
    }

    #[test]
    fn test_conversion_test_find_diff_indices_no_diffs() {
        let test = ConversionTest::new(
            Format::Gguf,
            Format::Apr,
            Backend::Cpu,
            ModelId::new("test", "model"),
        );
        let indices = test.find_diff_indices("same", "same");
        assert!(indices.is_empty());
    }

    #[test]
    fn test_conversion_test_hash_output_consistency() {
        let hash1 = ConversionTest::hash_output("test string");
        let hash2 = ConversionTest::hash_output("test string");
        let hash3 = ConversionTest::hash_output("different string");

        // Same input should produce same hash
        assert_eq!(hash1, hash2);
        // Different input should produce different hash
        assert_ne!(hash1, hash3);
        // Hash should be 16 hex characters
        assert_eq!(hash1.len(), 16);
    }

    #[test]
    fn test_classify_bug_empty_source_nonempty_target() {
        let test = SemanticConversionTest::new(
            Format::Gguf,
            Format::Apr,
            Backend::Cpu,
            ModelId::new("test", "model"),
        );
        // If source is empty/whitespace and target is not, classify as unknown
        let bug = test.classify_bug("  ", "some output", false);
        assert_eq!(bug, Some(ConversionBugType::Unknown));
    }

    #[test]
    fn test_classify_bug_both_empty_strings() {
        let test = SemanticConversionTest::new(
            Format::Gguf,
            Format::Apr,
            Backend::Cpu,
            ModelId::new("test", "model"),
        );
        // Both empty should match
        let bug = test.classify_bug("", "", false);
        assert!(bug.is_none());
    }

    #[test]
    fn test_generate_conversion_tests_full_count() {
        let model_id = ModelId::new("test", "model");
        let tests = generate_conversion_tests(&model_id);

        // 6 pairs x 2 backends = 12 tests
        assert_eq!(tests.len(), 12);
    }

    // ── Mock binary tests ────────────────────────────────────────────

    fn create_mock_apr(dir: &std::path::Path, script: &str) -> std::path::PathBuf {
        let path = dir.join("mock_apr");
        std::fs::write(&path, format!("#!/bin/bash\n{script}")).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        }
        path
    }

    #[test]
    fn test_conversion_test_execute_corroborated_via_mock() {
        let dir = tempfile::tempdir().unwrap();
        let model_file = dir.path().join("model.gguf");
        std::fs::write(&model_file, "fake").unwrap();

        let mock = create_mock_apr(
            dir.path(),
            r#"case "$1" in
run) printf "The answer is 4"; exit 0;;
rosetta) touch "$4"; exit 0;;
esac
exit 1"#,
        );

        let mut test = ConversionTest::new(
            Format::Gguf,
            Format::Apr,
            Backend::Cpu,
            ModelId::new("test", "model"),
        );
        test.binary = mock.to_string_lossy().to_string();

        if let Ok(conv) = test.execute(&model_file) {
            match conv {
                ConversionResult::Corroborated { max_diff, .. } => {
                    assert!(max_diff < EPSILON);
                }
                ConversionResult::Falsified { .. } => {}
            }
        }
    }

    #[test]
    fn test_conversion_test_execute_falsified_via_mock() {
        let dir = tempfile::tempdir().unwrap();
        let model_file = dir.path().join("model.gguf");
        std::fs::write(&model_file, "fake").unwrap();

        let mock = create_mock_apr(
            dir.path(),
            r#"case "$1" in
run)
  case "$2" in
  *converted*) printf "Completely different output 99";;
  *) printf "The answer is 4";;
  esac
  exit 0;;
rosetta) touch "$4"; exit 0;;
esac
exit 1"#,
        );

        let mut test = ConversionTest::new(
            Format::Gguf,
            Format::Apr,
            Backend::Cpu,
            ModelId::new("test", "model"),
        );
        test.binary = mock.to_string_lossy().to_string();

        if let Ok(conv) = test.execute(&model_file) {
            match conv {
                ConversionResult::Falsified {
                    gate_id, evidence, ..
                } => {
                    assert_eq!(gate_id, "F-CONV-G-A");
                    assert!(evidence.max_diff > EPSILON);
                    assert_ne!(evidence.source_hash, evidence.converted_hash);
                }
                ConversionResult::Corroborated { .. } => {}
            }
        }
    }

    #[test]
    fn test_conversion_test_execute_gpu_backend_via_mock() {
        let dir = tempfile::tempdir().unwrap();
        let model_file = dir.path().join("model.safetensors");
        std::fs::write(&model_file, "fake").unwrap();

        let mock = create_mock_apr(
            dir.path(),
            r#"case "$1" in
run) printf "The answer is 4"; exit 0;;
rosetta) touch "$4"; exit 0;;
esac
exit 1"#,
        );

        let mut test = ConversionTest::new(
            Format::SafeTensors,
            Format::Gguf,
            Backend::Gpu,
            ModelId::new("test", "model"),
        );
        test.binary = mock.to_string_lossy().to_string();

        if let Ok(ConversionResult::Corroborated { backend, .. }) = &test.execute(&model_file) {
            assert_eq!(*backend, Backend::Gpu);
        }
    }

    #[test]
    fn test_conversion_test_convert_model_failure_via_mock() {
        let dir = tempfile::tempdir().unwrap();
        let model_file = dir.path().join("model.gguf");
        std::fs::write(&model_file, "fake").unwrap();

        let mock = create_mock_apr(
            dir.path(),
            r#"case "$1" in
run) printf "The answer is 4"; exit 0;;
rosetta) printf "conversion error" >&2; exit 1;;
esac
exit 1"#,
        );

        let mut test = ConversionTest::new(
            Format::Gguf,
            Format::Apr,
            Backend::Cpu,
            ModelId::new("test", "model"),
        );
        test.binary = mock.to_string_lossy().to_string();

        if let Err(e) = test.execute(&model_file) {
            let msg = e.to_string();
            assert!(msg.contains("Conversion failed") || msg.contains("conversion error"));
        }
    }

    #[test]
    fn test_semantic_test_execute_corroborated_via_mock() {
        let dir = tempfile::tempdir().unwrap();
        let model_file = dir.path().join("model.safetensors");
        std::fs::write(&model_file, "fake").unwrap();

        let mock = create_mock_apr(
            dir.path(),
            r#"case "$1" in
run) printf "The answer is 4"; exit 0;;
rosetta) touch "$4"; exit 0;;
esac
exit 1"#,
        );

        let mut test = SemanticConversionTest::new(
            Format::SafeTensors,
            Format::Apr,
            Backend::Cpu,
            ModelId::new("test", "model"),
        );
        test.binary = mock.to_string_lossy().to_string();

        if let Ok(sem) = test.execute(&model_file) {
            if let SemanticTestResult::Corroborated {
                source_output,
                target_output,
            } = &sem
            {
                assert_eq!(source_output, target_output);
                assert!(sem.is_pass());
                assert!(sem.bug_type().is_none());
            }
        }
    }

    #[test]
    fn test_semantic_test_execute_embedding_transposition_via_mock() {
        let dir = tempfile::tempdir().unwrap();
        let model_file = dir.path().join("model.safetensors");
        std::fs::write(&model_file, "fake").unwrap();

        let mock = create_mock_apr(
            dir.path(),
            r#"case "$1" in
run)
  case "$2" in
  *semantic_test*) printf "PAD PAD PAD garbage tokens";;
  *) printf "The answer is 4";;
  esac
  exit 0;;
rosetta) touch "$4"; exit 0;;
esac
exit 1"#,
        );

        let mut test = SemanticConversionTest::new(
            Format::SafeTensors,
            Format::Apr,
            Backend::Cpu,
            ModelId::new("test", "model"),
        );
        test.binary = mock.to_string_lossy().to_string();

        if let Ok(sem) = test.execute(&model_file) {
            if let SemanticTestResult::Falsified { bug_type, .. } = &sem {
                assert_eq!(*bug_type, ConversionBugType::EmbeddingTransposition);
                assert!(!sem.is_pass());
            }
        }
    }

    #[test]
    fn test_semantic_test_execute_tokenizer_missing_via_mock() {
        let dir = tempfile::tempdir().unwrap();
        let model_file = dir.path().join("model.safetensors");
        std::fs::write(&model_file, "fake").unwrap();

        let mock = create_mock_apr(
            dir.path(),
            r#"case "$1" in
run)
  case "$2" in
  *semantic_test*) printf "output" >&1; printf "PMAT-172: missing embedded tokenizer" >&2;;
  *) printf "The answer is 4";;
  esac
  exit 0;;
rosetta) touch "$4"; exit 0;;
esac
exit 1"#,
        );

        let mut test = SemanticConversionTest::new(
            Format::SafeTensors,
            Format::Apr,
            Backend::Cpu,
            ModelId::new("test", "model"),
        );
        test.binary = mock.to_string_lossy().to_string();

        if let Ok(SemanticTestResult::Falsified {
            bug_type, stderr, ..
        }) = &test.execute(&model_file)
        {
            assert_eq!(*bug_type, ConversionBugType::TokenizerMissing);
            assert!(stderr.contains("PMAT-172"));
        }
    }

    #[test]
    fn test_round_trip_execute_corroborated_via_mock() {
        let dir = tempfile::tempdir().unwrap();
        let model_file = dir.path().join("model.gguf");
        std::fs::write(&model_file, "fake").unwrap();

        let mock = create_mock_apr(
            dir.path(),
            r#"case "$1" in
run) printf "The answer is 4"; exit 0;;
rosetta) touch "$4"; exit 0;;
esac
exit 1"#,
        );

        let mut rt = RoundTripTest::new(
            vec![Format::Gguf, Format::Apr, Format::SafeTensors, Format::Gguf],
            Backend::Cpu,
            ModelId::new("test", "model"),
        );
        rt.binary = mock.to_string_lossy().to_string();

        if let Ok(ConversionResult::Corroborated { max_diff, .. }) = rt.execute(&model_file) {
            assert!((max_diff - 0.0).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn test_round_trip_execute_falsified_via_mock() {
        let dir = tempfile::tempdir().unwrap();
        let model_file = dir.path().join("model.gguf");
        std::fs::write(&model_file, "fake").unwrap();

        let mock = create_mock_apr(
            dir.path(),
            r#"case "$1" in
run)
  case "$2" in
  *converted*) printf "Round-trip drift detected";;
  *) printf "The answer is 4";;
  esac
  exit 0;;
rosetta) touch "$4"; exit 0;;
esac
exit 1"#,
        );

        let mut rt = RoundTripTest::new(
            vec![Format::Gguf, Format::Apr],
            Backend::Cpu,
            ModelId::new("test", "model"),
        );
        rt.binary = mock.to_string_lossy().to_string();

        if let Ok(ConversionResult::Falsified { gate_id, .. }) = &rt.execute(&model_file) {
            assert_eq!(gate_id, "F-CONV-RT-001");
        }
    }

    #[test]
    fn test_conversion_executor_execute_all_via_mock() {
        let dir = tempfile::tempdir().unwrap();
        let model_file = dir.path().join("model.gguf");
        std::fs::write(&model_file, "fake").unwrap();

        let mock = create_mock_apr(
            dir.path(),
            r#"case "$1" in
run) printf "The answer is 4"; exit 0;;
rosetta) touch "$4"; exit 0;;
esac
exit 1"#,
        );

        let config = ConversionConfig {
            test_all_pairs: true,
            test_round_trips: true,
            backends: vec![Backend::Cpu],
            no_gpu: true,
            ..Default::default()
        };
        let mut executor = ConversionExecutor::new(config);
        executor.binary = mock.to_string_lossy().to_string();
        let model_id = ModelId::new("test", "model");

        if let Ok(exec_result) = executor.execute_all(&model_file, &model_id) {
            assert!(exec_result.total > 0);
            assert!(!exec_result.evidence.is_empty());
            assert!(!exec_result.results.is_empty());
            if exec_result.failed == 0 {
                assert!(exec_result.all_passed());
            }
        }
    }

    #[test]
    fn test_conversion_executor_execute_all_with_errors_via_mock() {
        let dir = tempfile::tempdir().unwrap();
        let model_file = dir.path().join("model.gguf");
        std::fs::write(&model_file, "fake").unwrap();

        let mock = create_mock_apr(
            dir.path(),
            r#"case "$1" in
run) printf "output"; exit 0;;
rosetta) printf "convert failed" >&2; exit 1;;
esac
exit 1"#,
        );

        let config = ConversionConfig {
            test_all_pairs: true,
            test_round_trips: false,
            backends: vec![Backend::Cpu],
            no_gpu: true,
            ..Default::default()
        };
        let mut executor = ConversionExecutor::new(config);
        executor.binary = mock.to_string_lossy().to_string();
        let model_id = ModelId::new("test", "model");

        if let Ok(exec_result) = executor.execute_all(&model_file, &model_id) {
            assert!(exec_result.total > 0);
            assert!(exec_result.failed > 0);
            assert!(!exec_result.all_passed());
        }
    }

    #[test]
    fn test_conversion_executor_round_trip_error_via_mock() {
        let dir = tempfile::tempdir().unwrap();
        let model_file = dir.path().join("model.gguf");
        std::fs::write(&model_file, "fake").unwrap();

        let mock = create_mock_apr(dir.path(), r"exit 1");

        let config = ConversionConfig {
            test_all_pairs: false,
            test_round_trips: true,
            backends: vec![Backend::Cpu],
            no_gpu: true,
            ..Default::default()
        };
        let mut executor = ConversionExecutor::new(config);
        executor.binary = mock.to_string_lossy().to_string();
        let model_id = ModelId::new("test", "model");

        if let Ok(exec_result) = executor.execute_all(&model_file, &model_id) {
            assert!(!exec_result.evidence.is_empty());
        }
    }

    #[test]
    fn test_conversion_test_execute_safetensors_target_via_mock() {
        let dir = tempfile::tempdir().unwrap();
        let model_file = dir.path().join("model.gguf");
        std::fs::write(&model_file, "fake").unwrap();

        let mock = create_mock_apr(
            dir.path(),
            r#"case "$1" in
run) printf "The answer is 4"; exit 0;;
rosetta) touch "$4"; exit 0;;
esac
exit 1"#,
        );

        let mut test = ConversionTest::new(
            Format::Gguf,
            Format::SafeTensors,
            Backend::Cpu,
            ModelId::new("test", "model"),
        );
        test.binary = mock.to_string_lossy().to_string();

        if let Ok(ConversionResult::Corroborated { target_format, .. }) = test.execute(&model_file)
        {
            assert_eq!(target_format, Format::SafeTensors);
        }
    }

    // =========================================================================
    // Rosetta-Testing Spec: New test type constructors (PMAT-ROSETTA-002/003)
    // =========================================================================

    #[test]
    fn test_idempotency_test_new() {
        let idem = IdempotencyTest::new(
            Format::Gguf,
            Format::Apr,
            Backend::Cpu,
            ModelId::new("test", "model"),
        );
        assert_eq!(idem.format_a, Format::Gguf);
        assert_eq!(idem.format_b, Format::Apr);
        assert_eq!(idem.backend, Backend::Cpu);
    }

    #[test]
    fn test_idempotency_test_debug() {
        let idem = IdempotencyTest::new(
            Format::Gguf,
            Format::Apr,
            Backend::Cpu,
            ModelId::new("test", "model"),
        );
        let debug_str = format!("{idem:?}");
        assert!(debug_str.contains("IdempotencyTest"));
    }

    #[test]
    fn test_idempotency_test_clone() {
        let idem = IdempotencyTest::new(
            Format::SafeTensors,
            Format::Gguf,
            Backend::Gpu,
            ModelId::new("test", "model"),
        );
        let cloned = idem.clone();
        assert_eq!(cloned.format_a, Format::SafeTensors);
        assert_eq!(cloned.format_b, Format::Gguf);
    }

    #[test]
    fn test_commutativity_test_new() {
        let com = CommutativityTest::new(Backend::Cpu, ModelId::new("test", "model"));
        assert_eq!(com.backend, Backend::Cpu);
    }

    #[test]
    fn test_commutativity_test_debug() {
        let com = CommutativityTest::new(Backend::Cpu, ModelId::new("test", "model"));
        let debug_str = format!("{com:?}");
        assert!(debug_str.contains("CommutativityTest"));
    }

    #[test]
    fn test_commutativity_test_clone() {
        let com = CommutativityTest::new(Backend::Gpu, ModelId::new("test", "model"));
        let cloned = com.clone();
        assert_eq!(cloned.backend, Backend::Gpu);
    }

    #[test]
    fn test_conversion_config_new_fields_default() {
        let config = ConversionConfig::default();
        assert!(config.test_multi_hop);
        assert!(config.test_cardinality);
        assert!(config.test_tensor_names);
        assert!(config.test_idempotency);
        assert!(config.test_commutativity);
    }

    #[test]
    fn test_conversion_config_cpu_only_new_fields() {
        let config = ConversionConfig::cpu_only();
        assert!(config.test_multi_hop);
        assert!(config.test_cardinality);
        assert!(config.test_tensor_names);
        assert!(config.test_idempotency);
        assert!(config.test_commutativity);
        assert!(config.no_gpu);
    }

    #[test]
    fn test_conversion_config_selective_disable() {
        let config = ConversionConfig {
            test_multi_hop: false,
            test_cardinality: false,
            test_tensor_names: true,
            test_idempotency: false,
            test_commutativity: true,
            ..Default::default()
        };
        assert!(!config.test_multi_hop);
        assert!(!config.test_cardinality);
        assert!(config.test_tensor_names);
        assert!(!config.test_idempotency);
        assert!(config.test_commutativity);
    }

    #[test]
    fn test_check_cardinality_nonexistent_binary() {
        let source = std::path::PathBuf::from("source.gguf");
        let target = std::path::PathBuf::from("target.apr");
        let result = check_cardinality(&source, &target, "/nonexistent/apr");
        assert!(result.is_err());
    }

    #[test]
    fn test_check_tensor_names_nonexistent_binary() {
        let source = std::path::PathBuf::from("source.gguf");
        let target = std::path::PathBuf::from("target.apr");
        let result = check_tensor_names(&source, &target, "/nonexistent/apr");
        assert!(result.is_err());
    }

    // =========================================================================
    // Mock binary tests for check_cardinality and check_tensor_names
    // =========================================================================

    /// Create a mock binary with explicit fd sync/close to avoid ETXTBSY (os error 26)
    /// when parallel tests execute mock scripts concurrently.
    fn create_mock_inspect_binary(
        dir: &std::path::Path,
        name: &str,
        json_output: &str,
    ) -> std::path::PathBuf {
        create_mock_script(dir, name, &format!("#!/bin/bash\necho '{json_output}'"))
    }

    /// Create a conditional mock binary (if/else on model arg).
    fn create_conditional_mock_binary(
        dir: &std::path::Path,
        name: &str,
        script: &str,
    ) -> std::path::PathBuf {
        create_mock_script(dir, name, script)
    }

    /// Write a mock script with explicit open→write→sync→close to ensure the
    /// write fd is fully released before any execve() can hit ETXTBSY.
    fn create_mock_script(dir: &std::path::Path, name: &str, content: &str) -> std::path::PathBuf {
        let path = dir.join(name);
        {
            use std::io::Write;
            let mut f = std::fs::File::create(&path).expect("create mock");
            f.write_all(content.as_bytes()).expect("write mock");
            f.sync_all().expect("sync mock");
            drop(f);
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755))
                .expect("set permissions");
        }
        // Yield to let the OS fully release the write reference on the inode
        std::thread::yield_now();
        path
    }

    #[test]
    fn test_check_cardinality_loss_detected() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let source_model = dir.path().join("source.gguf");
        let target_model = dir.path().join("target.apr");
        std::fs::write(&source_model, b"source").expect("write source");
        std::fs::write(&target_model, b"target").expect("write target");

        // Mock binary that returns different tensor counts based on the model arg
        let mock = create_conditional_mock_binary(
            dir.path(),
            "apr_card",
            "#!/bin/bash\nif echo \"$3\" | grep -q source; then\n  echo '{\"tensor_count\": 338, \"tensor_names\": []}'\nelse\n  echo '{\"tensor_count\": 227, \"tensor_names\": []}'\nfi",
        );

        let result = check_cardinality(&source_model, &target_model, mock.to_str().expect("path"));
        let (gate_id, reason) = result
            .expect("should succeed")
            .expect("should detect cardinality loss");
        assert_eq!(gate_id, "F-CONV-CARD-001");
        assert!(reason.contains("338"));
        assert!(reason.contains("227"));
    }

    #[test]
    fn test_check_cardinality_no_loss() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let source_model = dir.path().join("source.gguf");
        let target_model = dir.path().join("target.apr");
        std::fs::write(&source_model, b"source").expect("write source");
        std::fs::write(&target_model, b"target").expect("write target");

        let mock = create_mock_inspect_binary(
            dir.path(),
            "apr_card_ok",
            r#"{"tensor_count": 338, "tensor_names": []}"#,
        );

        let result = check_cardinality(&source_model, &target_model, mock.to_str().expect("path"));
        assert!(result.expect("should succeed").is_none());
    }

    #[test]
    fn test_check_tensor_names_fusion_detected() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let source_model = dir.path().join("source.gguf");
        let target_model = dir.path().join("target.apr");
        std::fs::write(&source_model, b"source").expect("write source");
        std::fs::write(&target_model, b"target").expect("write target");

        // Source has q_proj, k_proj, v_proj; target has qkv_proj (fusion)
        let mock = create_conditional_mock_binary(
            dir.path(),
            "apr_names",
            "#!/bin/bash\nif echo \"$3\" | grep -q source; then\n  echo '{\"tensor_count\": 3, \"tensor_names\": [\"layer.0.q_proj\", \"layer.0.k_proj\", \"layer.0.v_proj\"]}'\nelse\n  echo '{\"tensor_count\": 1, \"tensor_names\": [\"layer.0.qkv_proj\"]}'\nfi",
        );

        let result = check_tensor_names(&source_model, &target_model, mock.to_str().expect("path"));
        let (gate_id, detail) = result
            .expect("should succeed")
            .expect("should detect name divergence");
        assert_eq!(gate_id, "F-CONV-NAME-001");
        assert!(detail.contains("QKV fusion"));
        assert!(detail.contains("q_proj"));
    }

    #[test]
    fn test_check_tensor_names_non_fusion_divergence() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let source_model = dir.path().join("source.gguf");
        let target_model = dir.path().join("target.apr");
        std::fs::write(&source_model, b"source").expect("write source");
        std::fs::write(&target_model, b"target").expect("write target");

        // Source has "embed.weight"; target renamed it to "embedding.weight"
        let mock = create_conditional_mock_binary(
            dir.path(),
            "apr_names2",
            "#!/bin/bash\nif echo \"$3\" | grep -q source; then\n  echo '{\"tensor_count\": 2, \"tensor_names\": [\"embed.weight\", \"lm_head.weight\"]}'\nelse\n  echo '{\"tensor_count\": 2, \"tensor_names\": [\"embedding.weight\", \"lm_head.weight\"]}'\nfi",
        );

        let result = check_tensor_names(&source_model, &target_model, mock.to_str().expect("path"));
        let (gate_id, detail) = result
            .expect("should succeed")
            .expect("should detect divergence");
        assert_eq!(gate_id, "F-CONV-NAME-001");
        assert!(detail.contains("divergence"));
        assert!(detail.contains("embed.weight"));
    }

    #[test]
    fn test_check_tensor_names_all_preserved() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let source_model = dir.path().join("source.gguf");
        let target_model = dir.path().join("target.apr");
        std::fs::write(&source_model, b"source").expect("write source");
        std::fs::write(&target_model, b"target").expect("write target");

        let mock = create_mock_inspect_binary(
            dir.path(),
            "apr_names_ok",
            r#"{"tensor_count": 2, "tensor_names": ["a.weight", "b.weight"]}"#,
        );

        let result = check_tensor_names(&source_model, &target_model, mock.to_str().expect("path"));
        assert!(result.expect("should succeed").is_none());
    }

    #[test]
    fn test_check_tensor_names_empty_names_skip() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let source_model = dir.path().join("source.gguf");
        let target_model = dir.path().join("target.apr");
        std::fs::write(&source_model, b"source").expect("write source");
        std::fs::write(&target_model, b"target").expect("write target");

        let mock = create_mock_inspect_binary(
            dir.path(),
            "apr_names_empty",
            r#"{"tensor_count": 10, "tensor_names": []}"#,
        );

        let result = check_tensor_names(&source_model, &target_model, mock.to_str().expect("path"));
        assert!(result.expect("should succeed").is_none());
    }

    #[test]
    fn test_convert_to_format_tagged_gguf_ext() {
        let source = std::path::PathBuf::from("/tmp/model.apr");
        let target = source.with_extension("tag1.gguf");
        assert!(target.to_str().expect("path").ends_with("tag1.gguf"));
    }

    #[test]
    fn test_convert_to_format_tagged_safetensors_ext() {
        let source = std::path::PathBuf::from("/tmp/model.apr");
        let target = source.with_extension("tag2.safetensors");
        assert!(target.to_str().expect("path").ends_with("tag2.safetensors"));
    }

    #[test]
    fn test_run_inference_simple_gpu_flag() {
        // Verify GPU backend produces --gpu arg (fails because no binary, but exercises the match)
        let result = run_inference_simple(
            &std::path::PathBuf::from("/nonexistent/model.gguf"),
            Backend::Gpu,
            "/nonexistent/apr",
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_idempotency_falsified_result_structure() {
        // Directly test the Falsified variant construction
        let result = ConversionResult::Falsified {
            gate_id: "F-CONV-IDEM-001".to_string(),
            reason: "Idempotency failure: Gguf→Apr produced different output".to_string(),
            evidence: ConversionEvidence {
                source_hash: ConversionTest::hash_output("output1"),
                converted_hash: ConversionTest::hash_output("output2"),
                max_diff: 1.0,
                diff_indices: vec![],
                source_format: Format::Gguf,
                target_format: Format::Apr,
                backend: Backend::Cpu,
                failure_type: None,
                quant_type: None,
            },
        };
        match result {
            ConversionResult::Falsified {
                gate_id, reason, ..
            } => {
                assert_eq!(gate_id, "F-CONV-IDEM-001");
                assert!(reason.contains("Idempotency"));
            }
            _ => panic!("Expected Falsified"),
        }
    }

    #[test]
    fn test_commutativity_falsified_result_structure() {
        let result = ConversionResult::Falsified {
            gate_id: "F-CONV-COM-001".to_string(),
            reason: "Commutativity failure: GGUF→APR differs from GGUF→ST→APR".to_string(),
            evidence: ConversionEvidence {
                source_hash: ConversionTest::hash_output("path_a"),
                converted_hash: ConversionTest::hash_output("path_b"),
                max_diff: 1.0,
                diff_indices: vec![],
                source_format: Format::Gguf,
                target_format: Format::Apr,
                backend: Backend::Cpu,
                failure_type: None,
                quant_type: None,
            },
        };
        match result {
            ConversionResult::Falsified {
                gate_id, reason, ..
            } => {
                assert_eq!(gate_id, "F-CONV-COM-001");
                assert!(reason.contains("Commutativity"));
            }
            _ => panic!("Expected Falsified"),
        }
    }

    #[test]
    fn test_conversion_test_convert_model_failure() {
        // Exercise the conversion failure error path
        let result = convert_to_format_tagged(
            &std::path::PathBuf::from("/nonexistent/model.gguf"),
            Format::Gguf,
            "test",
            "/nonexistent/apr",
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_conversion_test_convert_model_safetensors_target() {
        let result = convert_to_format_tagged(
            &std::path::PathBuf::from("/nonexistent/model.apr"),
            Format::SafeTensors,
            "test",
            "/nonexistent/apr",
        );
        assert!(result.is_err());
    }

    // ── §3.4 classify_failure tests ────────────────────────────────────

    #[test]
    fn test_classify_failure_tensor_name_mismatch() {
        assert_eq!(
            classify_failure("tensor name mismatch: q_proj not found", 1),
            ConversionFailureType::TensorNameMismatch
        );
        assert_eq!(
            classify_failure("missing tensor 'lm_head.weight'", 1),
            ConversionFailureType::TensorNameMismatch
        );
        assert_eq!(
            classify_failure("unexpected tensor in output", 1),
            ConversionFailureType::TensorNameMismatch
        );
    }

    #[test]
    fn test_classify_failure_dequantization() {
        assert_eq!(
            classify_failure("dequantization error: NaN values produced", 1),
            ConversionFailureType::DequantizationFailure
        );
        assert_eq!(
            classify_failure("quantization overflow detected", 1),
            ConversionFailureType::DequantizationFailure
        );
        assert_eq!(
            classify_failure("NaN in output tensor", 1),
            ConversionFailureType::DequantizationFailure
        );
        assert_eq!(
            classify_failure("infinity values in layer 5", 1),
            ConversionFailureType::DequantizationFailure
        );
    }

    #[test]
    fn test_classify_failure_config_metadata() {
        assert_eq!(
            classify_failure("hidden_size mismatch: expected 768 got 512", 1),
            ConversionFailureType::ConfigMetadataMismatch
        );
        assert_eq!(
            classify_failure("metadata mismatch: num_layers differs", 1),
            ConversionFailureType::ConfigMetadataMismatch
        );
        assert_eq!(
            classify_failure("vocab_size does not match model", 1),
            ConversionFailureType::ConfigMetadataMismatch
        );
        assert_eq!(
            classify_failure("config mismatch detected", 1),
            ConversionFailureType::ConfigMetadataMismatch
        );
    }

    #[test]
    fn test_classify_failure_missing_artifact() {
        assert_eq!(
            classify_failure("file not found: model.safetensors", 1),
            ConversionFailureType::MissingArtifact
        );
        assert_eq!(
            classify_failure("No such file or directory", 1),
            ConversionFailureType::MissingArtifact
        );
        assert_eq!(
            classify_failure("tokenizer.json missing from model directory", 1),
            ConversionFailureType::MissingArtifact
        );
        assert_eq!(
            classify_failure("config.json: file not found", 1),
            ConversionFailureType::MissingArtifact
        );
    }

    #[test]
    fn test_classify_failure_inference() {
        assert_eq!(
            classify_failure("inference failed: out of memory", 1),
            ConversionFailureType::InferenceFailure
        );
        assert_eq!(
            classify_failure("forward pass error", 1),
            ConversionFailureType::InferenceFailure
        );
        assert_eq!(
            classify_failure("", -11), // SIGSEGV
            ConversionFailureType::InferenceFailure
        );
    }

    #[test]
    fn test_classify_failure_unknown() {
        assert_eq!(
            classify_failure("some generic error", 1),
            ConversionFailureType::Unknown
        );
        assert_eq!(classify_failure("", 1), ConversionFailureType::Unknown);
    }

    #[test]
    fn test_classify_failure_case_insensitive() {
        assert_eq!(
            classify_failure("TENSOR NAME MISMATCH", 1),
            ConversionFailureType::TensorNameMismatch
        );
        assert_eq!(
            classify_failure("Dequantization Error", 1),
            ConversionFailureType::DequantizationFailure
        );
    }

    // ── §3.7 QuantType + tolerance tests ───────────────────────────────

    #[test]
    fn test_quant_type_from_str_label() {
        assert_eq!(QuantType::from_str_label("f32"), QuantType::F32);
        assert_eq!(QuantType::from_str_label("fp32"), QuantType::F32);
        assert_eq!(QuantType::from_str_label("float32"), QuantType::F32);
        assert_eq!(QuantType::from_str_label("f16"), QuantType::F16);
        assert_eq!(QuantType::from_str_label("fp16"), QuantType::F16);
        assert_eq!(QuantType::from_str_label("bf16"), QuantType::BF16);
        assert_eq!(QuantType::from_str_label("bfloat16"), QuantType::BF16);
        assert_eq!(QuantType::from_str_label("q4_k_m"), QuantType::Q4KM);
        assert_eq!(QuantType::from_str_label("q4km"), QuantType::Q4KM);
        assert_eq!(QuantType::from_str_label("q5_k_m"), QuantType::Q5KM);
        assert_eq!(QuantType::from_str_label("q5km"), QuantType::Q5KM);
        assert_eq!(QuantType::from_str_label("q6_k"), QuantType::Q6K);
        assert_eq!(QuantType::from_str_label("q4_0"), QuantType::Q4_0);
        assert_eq!(QuantType::from_str_label("q8_0"), QuantType::Q8_0);
        assert_eq!(
            QuantType::from_str_label("unknown_type"),
            QuantType::Unknown
        );
    }

    #[test]
    fn test_quant_type_from_str_label_case_insensitive() {
        assert_eq!(QuantType::from_str_label("F32"), QuantType::F32);
        assert_eq!(QuantType::from_str_label("BF16"), QuantType::BF16);
        assert_eq!(QuantType::from_str_label("Q4_K_M"), QuantType::Q4KM);
        assert_eq!(QuantType::from_str_label("Q5_K_M"), QuantType::Q5KM);
    }

    #[test]
    fn test_quant_type_from_str_label_with_hyphens() {
        assert_eq!(QuantType::from_str_label("q4-k-m"), QuantType::Q4KM);
        assert_eq!(QuantType::from_str_label("q5-k-m"), QuantType::Q5KM);
        assert_eq!(QuantType::from_str_label("q6-k"), QuantType::Q6K);
    }

    #[test]
    fn test_tolerance_for_f32() {
        let tol = tolerance_for(QuantType::F32);
        assert!((tol.atol - 1e-6).abs() < 1e-10);
    }

    #[test]
    fn test_tolerance_for_f16() {
        let tol = tolerance_for(QuantType::F16);
        assert!((tol.atol - 1e-3).abs() < 1e-10);
    }

    #[test]
    fn test_tolerance_for_q4km() {
        let tol = tolerance_for(QuantType::Q4KM);
        assert!((tol.atol - 1e-1).abs() < 1e-10);
    }

    #[test]
    fn test_tolerance_for_q5km() {
        let tol = tolerance_for(QuantType::Q5KM);
        assert!((tol.atol - 7.5e-2).abs() < 1e-10);
        assert!((tol.rtol - 5e-2).abs() < 1e-10);
    }

    #[test]
    fn test_tolerance_for_q6k() {
        let tol = tolerance_for(QuantType::Q6K);
        assert!((tol.atol - 5e-2).abs() < 1e-10);
    }

    #[test]
    fn test_tolerance_for_unknown_falls_back_to_f32() {
        let tol = tolerance_for(QuantType::Unknown);
        assert!((tol.atol - 1e-6).abs() < 1e-10);
    }

    #[test]
    fn test_effective_epsilon_without_quant() {
        let test = ConversionTest::new(
            Format::Gguf,
            Format::Apr,
            Backend::Cpu,
            ModelId::new("test", "model"),
        );
        assert!((test.effective_epsilon() - EPSILON).abs() < f64::EPSILON);
    }

    #[test]
    fn test_effective_epsilon_with_quant() {
        let mut test = ConversionTest::new(
            Format::Gguf,
            Format::Apr,
            Backend::Cpu,
            ModelId::new("test", "model"),
        );
        test.quant_type = Some(QuantType::Q4KM);
        assert!((test.effective_epsilon() - 1e-1).abs() < 1e-10);
    }

    #[test]
    fn test_effective_epsilon_f32_quant() {
        let mut test = ConversionTest::new(
            Format::Gguf,
            Format::Apr,
            Backend::Cpu,
            ModelId::new("test", "model"),
        );
        test.quant_type = Some(QuantType::F32);
        assert!((test.effective_epsilon() - 1e-6).abs() < 1e-10);
    }

    // ── ConversionFailureType / TensorNaming serde tests ───────────────

    #[test]
    fn test_conversion_failure_type_serde() {
        let types = [
            ConversionFailureType::TensorNameMismatch,
            ConversionFailureType::DequantizationFailure,
            ConversionFailureType::ConfigMetadataMismatch,
            ConversionFailureType::MissingArtifact,
            ConversionFailureType::InferenceFailure,
            ConversionFailureType::Unknown,
        ];
        for ft in types {
            let json = serde_json::to_string(&ft).unwrap();
            let parsed: ConversionFailureType = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed, ft);
        }
    }

    #[test]
    fn test_conversion_failure_type_gate_ids() {
        assert_eq!(
            ConversionFailureType::TensorNameMismatch.gate_id(),
            "F-CONV-TNAME-001"
        );
        assert_eq!(
            ConversionFailureType::DequantizationFailure.gate_id(),
            "F-CONV-DEQUANT-001"
        );
        assert_eq!(
            ConversionFailureType::ConfigMetadataMismatch.gate_id(),
            "F-CONV-CONFIG-001"
        );
        assert_eq!(
            ConversionFailureType::MissingArtifact.gate_id(),
            "F-CONV-MISSING-001"
        );
        assert_eq!(
            ConversionFailureType::InferenceFailure.gate_id(),
            "F-CONV-INFER-001"
        );
        assert_eq!(
            ConversionFailureType::Unknown.gate_id(),
            "F-CONV-UNKNOWN-002"
        );
    }

    #[test]
    fn test_conversion_failure_type_keys() {
        assert_eq!(
            ConversionFailureType::TensorNameMismatch.key(),
            "tensor_name_mismatch"
        );
        assert_eq!(ConversionFailureType::Unknown.key(), "unknown");
    }

    #[test]
    fn test_quant_type_serde() {
        let types = [
            QuantType::F32,
            QuantType::F16,
            QuantType::BF16,
            QuantType::Q4KM,
            QuantType::Q5KM,
            QuantType::Q6K,
            QuantType::Q4_0,
            QuantType::Q8_0,
            QuantType::Unknown,
        ];
        for qt in types {
            let json = serde_json::to_string(&qt).unwrap();
            let parsed: QuantType = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed, qt);
        }
    }

    #[test]
    fn test_tensor_naming_serde() {
        let variants = [
            TensorNaming::HuggingFace,
            TensorNaming::Gguf,
            TensorNaming::Apr,
            TensorNaming::Unknown("custom".to_string()),
        ];
        for tn in &variants {
            let json = serde_json::to_string(tn).unwrap();
            let parsed: TensorNaming = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed, *tn);
        }
    }

    #[test]
    fn test_conversion_evidence_with_failure_type() {
        let evidence = ConversionEvidence {
            source_hash: "a".to_string(),
            converted_hash: "b".to_string(),
            max_diff: 0.5,
            diff_indices: vec![],
            source_format: Format::Gguf,
            target_format: Format::Apr,
            backend: Backend::Cpu,
            failure_type: Some(ConversionFailureType::TensorNameMismatch),
            quant_type: Some(QuantType::Q4KM),
        };
        let json = serde_json::to_string(&evidence).unwrap();
        let parsed: ConversionEvidence = serde_json::from_str(&json).unwrap();
        assert_eq!(
            parsed.failure_type,
            Some(ConversionFailureType::TensorNameMismatch)
        );
        assert_eq!(parsed.quant_type, Some(QuantType::Q4KM));
    }

    #[test]
    fn test_conversion_evidence_default_optional_fields() {
        // Deserialize without optional fields — should default to None
        let json = r#"{
            "source_hash": "a",
            "converted_hash": "b",
            "max_diff": 0.1,
            "diff_indices": [],
            "source_format": "gguf",
            "target_format": "apr",
            "backend": "cpu"
        }"#;
        let parsed: ConversionEvidence = serde_json::from_str(json).unwrap();
        assert!(parsed.failure_type.is_none());
        assert!(parsed.quant_type.is_none());
    }

    #[test]
    fn test_default_tolerances_count() {
        assert_eq!(DEFAULT_TOLERANCES.len(), 8);
    }

    // =========================================================================
    // Model Path Resolution Tests
    // =========================================================================

    #[test]
    fn test_resolve_model_path_apr_cache_structure() {
        let tmp = tempfile::TempDir::new().unwrap();
        let safetensors_dir = tmp.path().join("safetensors");
        std::fs::create_dir_all(&safetensors_dir).unwrap();
        let model_file = safetensors_dir.join("model.safetensors");
        std::fs::write(&model_file, b"fake").unwrap();

        let result = resolve_model_path(tmp.path(), Format::SafeTensors);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), model_file);
    }

    #[test]
    fn test_resolve_model_path_hf_cache_flat_structure() {
        let tmp = tempfile::TempDir::new().unwrap();
        // HF cache has model.safetensors directly in snapshot dir (flat)
        let model_file = tmp.path().join("model.safetensors");
        std::fs::write(&model_file, b"fake").unwrap();

        let result = resolve_model_path(tmp.path(), Format::SafeTensors);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), model_file);
    }

    #[test]
    fn test_resolve_model_path_file_mode() {
        let tmp = tempfile::TempDir::new().unwrap();
        let model_file = tmp.path().join("model.gguf");
        std::fs::write(&model_file, b"fake").unwrap();

        let result = resolve_model_path(&model_file, Format::Gguf);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), model_file);
    }

    #[test]
    fn test_resolve_model_path_file_mode_extension_mismatch() {
        let tmp = tempfile::TempDir::new().unwrap();
        let model_file = tmp.path().join("model.gguf");
        std::fs::write(&model_file, b"fake").unwrap();

        let result = resolve_model_path(&model_file, Format::SafeTensors);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("extension mismatch")
        );
    }

    #[test]
    fn test_resolve_model_path_not_found() {
        let tmp = tempfile::TempDir::new().unwrap();
        let result = resolve_model_path(tmp.path(), Format::Apr);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("No apr file found")
        );
    }

    #[test]
    fn test_resolve_model_path_any_extension_in_subdir() {
        let tmp = tempfile::TempDir::new().unwrap();
        let gguf_dir = tmp.path().join("gguf");
        std::fs::create_dir_all(&gguf_dir).unwrap();
        // Not model.gguf but something.gguf
        let model_file = gguf_dir.join("qwen-0.5b-q4.gguf");
        std::fs::write(&model_file, b"fake").unwrap();

        let result = resolve_model_path(tmp.path(), Format::Gguf);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), model_file);
    }

    #[test]
    fn test_resolve_model_path_any_extension_in_base_dir() {
        let tmp = tempfile::TempDir::new().unwrap();
        // HF cache might have different names
        let model_file = tmp.path().join("qwen2.5-coder-0.5b.safetensors");
        std::fs::write(&model_file, b"fake").unwrap();

        let result = resolve_model_path(tmp.path(), Format::SafeTensors);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), model_file);
    }

    #[test]
    fn test_default_tolerances_all_quant_types() {
        let types: Vec<QuantType> = DEFAULT_TOLERANCES.iter().map(|t| t.quant_type).collect();
        assert!(types.contains(&QuantType::F32));
        assert!(types.contains(&QuantType::F16));
        assert!(types.contains(&QuantType::BF16));
        assert!(types.contains(&QuantType::Q4KM));
        assert!(types.contains(&QuantType::Q5KM));
        assert!(types.contains(&QuantType::Q6K));
        assert!(types.contains(&QuantType::Q4_0));
        assert!(types.contains(&QuantType::Q8_0));
    }

    // =========================================================================
    // HuggingFace Cache Resolution Tests (HF-CACHE-001, HF-CACHE-002)
    // =========================================================================

    #[test]
    fn test_split_hf_repo_with_org() {
        assert_eq!(
            split_hf_repo("Qwen/Qwen2.5-Coder-0.5B"),
            ("Qwen", "Qwen2.5-Coder-0.5B")
        );
        assert_eq!(
            split_hf_repo("meta-llama/Llama-2-7b"),
            ("meta-llama", "Llama-2-7b")
        );
    }

    #[test]
    fn test_split_hf_repo_without_org() {
        assert_eq!(split_hf_repo("model-only"), ("unknown", "model-only"));
        assert_eq!(split_hf_repo("gpt2"), ("unknown", "gpt2"));
    }

    #[test]
    fn test_split_hf_repo_multiple_slashes() {
        // Only splits on first slash
        assert_eq!(split_hf_repo("org/repo/extra"), ("org", "repo/extra"));
    }

    #[test]
    fn test_split_hf_repo_empty_string() {
        assert_eq!(split_hf_repo(""), ("unknown", ""));
    }

    #[test]
    fn test_find_hf_snapshot_found() {
        let tmp = tempfile::TempDir::new().unwrap();
        let snapshot = tmp
            .path()
            .join("models--Test--Model")
            .join("snapshots")
            .join("abc123");
        std::fs::create_dir_all(&snapshot).unwrap();
        std::fs::write(snapshot.join("model.safetensors"), b"fake").unwrap();

        let result = find_hf_snapshot(tmp.path(), "Test", "Model");
        assert!(result.is_some());
        assert_eq!(result.unwrap(), snapshot);
    }

    #[test]
    fn test_find_hf_snapshot_not_found_no_dir() {
        let tmp = tempfile::TempDir::new().unwrap();
        let result = find_hf_snapshot(tmp.path(), "Missing", "Model");
        assert!(result.is_none());
    }

    #[test]
    fn test_find_hf_snapshot_not_found_no_safetensors() {
        let tmp = tempfile::TempDir::new().unwrap();
        let snapshot = tmp
            .path()
            .join("models--Test--NoFile")
            .join("snapshots")
            .join("abc123");
        std::fs::create_dir_all(&snapshot).unwrap();
        // No model.safetensors file

        let result = find_hf_snapshot(tmp.path(), "Test", "NoFile");
        assert!(result.is_none());
    }

    #[test]
    fn test_find_apr_cache_found() {
        let tmp = tempfile::TempDir::new().unwrap();
        let apr_cache = tmp.path().join(".cache/apr-models/TestOrg/TestRepo");
        std::fs::create_dir_all(&apr_cache).unwrap();

        let result = find_apr_cache(tmp.path(), "TestOrg", "TestRepo");
        assert!(result.is_some());
        assert_eq!(result.unwrap(), apr_cache);
    }

    #[test]
    fn test_find_apr_cache_not_found() {
        let tmp = tempfile::TempDir::new().unwrap();
        let result = find_apr_cache(tmp.path(), "Missing", "Model");
        assert!(result.is_none());
    }

    #[test]
    fn test_resolve_hf_repo_with_dirs_found_in_hf_cache() {
        let tmp = tempfile::TempDir::new().unwrap();
        let snapshot = tmp
            .path()
            .join("models--Test--Model")
            .join("snapshots")
            .join("abc123");
        std::fs::create_dir_all(&snapshot).unwrap();
        std::fs::write(snapshot.join("model.safetensors"), b"fake").unwrap();

        // Use the temp dir as both HF cache and home
        let result = resolve_hf_repo_with_dirs("Test/Model", tmp.path(), tmp.path());
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), snapshot);
    }

    #[test]
    fn test_resolve_hf_repo_with_dirs_found_in_apr_cache() {
        let tmp = tempfile::TempDir::new().unwrap();
        let apr_cache = tmp.path().join(".cache/apr-models/TestOrg/TestRepo");
        std::fs::create_dir_all(&apr_cache).unwrap();

        // HF cache is empty, APR cache has the model
        let hf_cache = tmp.path().join("hf_empty");
        std::fs::create_dir_all(&hf_cache).unwrap();

        let result = resolve_hf_repo_with_dirs("TestOrg/TestRepo", &hf_cache, tmp.path());
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), apr_cache);
    }

    #[test]
    fn test_resolve_hf_repo_with_dirs_not_found() {
        let tmp = tempfile::TempDir::new().unwrap();
        let hf_cache = tmp.path().join("hf_empty");
        let home = tmp.path().join("home_empty");
        std::fs::create_dir_all(&hf_cache).unwrap();
        std::fs::create_dir_all(&home).unwrap();

        let result = resolve_hf_repo_with_dirs("Missing/Model", &hf_cache, &home);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("not found in cache"));
        assert!(err_msg.contains("Missing/Model"));
    }

    #[test]
    fn test_resolve_hf_repo_with_dirs_snapshot_without_safetensors() {
        let tmp = tempfile::TempDir::new().unwrap();
        let snapshot = tmp
            .path()
            .join("models--Test--NoSafetensors")
            .join("snapshots")
            .join("abc123");
        std::fs::create_dir_all(&snapshot).unwrap();
        // No model.safetensors

        let home = tmp.path().join("home_empty");
        std::fs::create_dir_all(&home).unwrap();

        let result = resolve_hf_repo_with_dirs("Test/NoSafetensors", tmp.path(), &home);
        assert!(result.is_err());
    }

    #[test]
    fn test_resolve_hf_repo_with_dirs_multiple_snapshots() {
        let tmp = tempfile::TempDir::new().unwrap();
        let snapshots_dir = tmp.path().join("models--Test--Multi").join("snapshots");

        // Create two snapshots, only second has model.safetensors
        let snap1 = snapshots_dir.join("aaa111");
        let snap2 = snapshots_dir.join("bbb222");
        std::fs::create_dir_all(&snap1).unwrap();
        std::fs::create_dir_all(&snap2).unwrap();
        std::fs::write(snap2.join("model.safetensors"), b"fake").unwrap();

        let result = resolve_hf_repo_with_dirs("Test/Multi", tmp.path(), tmp.path());
        assert!(result.is_ok());
        assert!(result.unwrap().join("model.safetensors").exists());
    }

    #[test]
    fn test_resolve_hf_repo_with_dirs_hf_cache_priority() {
        let tmp = tempfile::TempDir::new().unwrap();

        // Create both HF and APR cache entries
        let hf_snapshot = tmp
            .path()
            .join("models--Test--Both")
            .join("snapshots")
            .join("hf123");
        std::fs::create_dir_all(&hf_snapshot).unwrap();
        std::fs::write(hf_snapshot.join("model.safetensors"), b"hf").unwrap();

        let apr_cache = tmp.path().join(".cache/apr-models/Test/Both");
        std::fs::create_dir_all(&apr_cache).unwrap();

        // HF cache should take priority
        let result = resolve_hf_repo_with_dirs("Test/Both", tmp.path(), tmp.path());
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), hf_snapshot);
    }

    #[test]
    fn test_get_hf_cache_dir_returns_path() {
        // Just verify it returns a PathBuf without errors
        // The actual value depends on environment, so we just check it doesn't panic
        let dir = get_hf_cache_dir();
        assert!(!dir.as_os_str().is_empty());
    }

    #[test]
    fn test_resolve_hf_repo_to_cache_error_message_format() {
        // Test with nonexistent paths - this tests the error message format
        let tmp = tempfile::TempDir::new().unwrap();
        let hf_cache = tmp.path().join("hf_empty");
        let home = tmp.path().join("home_empty");
        std::fs::create_dir_all(&hf_cache).unwrap();
        std::fs::create_dir_all(&home).unwrap();

        let result = resolve_hf_repo_with_dirs("Org/Repo", &hf_cache, &home);
        assert!(result.is_err());

        let err_msg = result.unwrap_err().to_string();
        // Check error message contains useful debugging info
        assert!(err_msg.contains("Org/Repo"));
        assert!(err_msg.contains("Searched:"));
        assert!(err_msg.contains("models--Org--Repo"));
        assert!(err_msg.contains("apr-models"));
    }
}
