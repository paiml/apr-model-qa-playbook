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
use std::path::Path;
use std::process::Command;

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
}

fn default_epsilon() -> f64 {
    EPSILON
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
        }
    }

    /// Get the gate ID for this conversion
    #[must_use]
    pub fn gate_id(&self) -> String {
        let src = format!("{:?}", self.source_format).to_uppercase();
        let tgt = format!("{:?}", self.target_format).to_uppercase();
        format!("F-CONV-{}-{}", &src[..1], &tgt[..1])
    }

    /// Execute the conversion test
    ///
    /// # Errors
    ///
    /// Returns an error if the conversion or inference fails.
    pub fn execute(&self, model_path: &Path) -> Result<ConversionResult> {
        // 1. Run inference on source format
        let source_output = self.run_inference(model_path, &self.source_format)?;

        // 2. Convert to target format
        let converted_path = self.convert_model(model_path)?;

        // 3. Run inference on converted model
        let converted_output = self.run_inference(&converted_path, &self.target_format)?;

        // 4. Compare outputs
        let diff = self.compute_diff(&source_output, &converted_output);

        if diff > self.epsilon {
            Ok(ConversionResult::Falsified {
                gate_id: self.gate_id(),
                reason: format!(
                    "Conversion {:?} → {:?} produced different output (diff: {:.2e}, ε: {:.2e})",
                    self.source_format, self.target_format, diff, self.epsilon
                ),
                evidence: ConversionEvidence {
                    source_hash: Self::hash_output(&source_output),
                    converted_hash: Self::hash_output(&converted_output),
                    max_diff: diff,
                    diff_indices: self.find_diff_indices(&source_output, &converted_output),
                    source_format: self.source_format,
                    target_format: self.target_format,
                    backend: self.backend,
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

        let output = Command::new("apr")
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
    fn convert_model(&self, source_path: &Path) -> Result<std::path::PathBuf> {
        let target_ext = match self.target_format {
            Format::Gguf => "gguf",
            Format::SafeTensors => "safetensors",
            Format::Apr => "apr",
        };

        // Create target path with new extension (format determined by extension)
        let target_path = source_path.with_extension(format!("converted.{target_ext}"));

        // Use apr rosetta convert: apr rosetta convert <SOURCE> <TARGET>
        // Format is inferred from output file extension
        let output = Command::new("apr")
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
    /// Source format (typically GGUF as ground truth)
    pub source_format: Format,
    /// Target format to test
    pub target_format: Format,
    /// Backend to use
    pub backend: Backend,
    /// Model ID
    pub model_id: ModelId,
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
        }
    }

    /// Execute the semantic test and classify any bug found
    ///
    /// # Errors
    ///
    /// Returns an error if inference fails.
    pub fn execute(&self, model_path: &Path) -> Result<SemanticTestResult> {
        // Run inference on source (GGUF - ground truth)
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

        let output = Command::new("apr")
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

        let output = Command::new("apr")
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
}

impl RoundTripTest {
    /// Create a new round-trip test
    #[must_use]
    pub fn new(formats: Vec<Format>, backend: Backend, model_id: ModelId) -> Self {
        Self {
            formats,
            backend,
            model_id,
        }
    }

    /// Execute round-trip conversion test
    ///
    /// # Errors
    ///
    /// Returns an error if any conversion fails.
    pub fn execute(&self, model_path: &Path) -> Result<ConversionResult> {
        // Get original output
        let original_output = run_inference_simple(model_path, self.backend)?;

        // Convert through chain
        let mut current_path = model_path.to_path_buf();
        for i in 0..self.formats.len() {
            let next_format = self.formats[(i + 1) % self.formats.len()];
            current_path = convert_to_format(&current_path, next_format)?;
        }

        // Get final output
        let final_output = run_inference_simple(&current_path, self.backend)?;

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

/// Simple inference helper
fn run_inference_simple(model_path: &Path, backend: Backend) -> Result<String> {
    let backend_flag = match backend {
        Backend::Cpu => vec![],
        Backend::Gpu => vec!["--gpu".to_string()],
    };

    let output = Command::new("apr")
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
fn convert_to_format(source_path: &Path, target_format: Format) -> Result<std::path::PathBuf> {
    let target_ext = match target_format {
        Format::Gguf => "gguf",
        Format::SafeTensors => "safetensors",
        Format::Apr => "apr",
    };

    // Create target path with new extension (format determined by extension)
    let target_path = source_path.with_extension(format!("converted.{target_ext}"));

    // Use apr rosetta convert: apr rosetta convert <SOURCE> <TARGET>
    // Format is inferred from output file extension
    let output = Command::new("apr")
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
pub struct ConversionConfig {
    /// Test all format pairs
    pub test_all_pairs: bool,
    /// Test round-trips
    pub test_round_trips: bool,
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
            backends: vec![Backend::Cpu],
            no_gpu: true,
        }
    }
}

/// Executor for running P0 format conversion tests
#[derive(Debug)]
pub struct ConversionExecutor {
    config: ConversionConfig,
}

impl ConversionExecutor {
    /// Create a new conversion executor
    #[must_use]
    pub fn new(config: ConversionConfig) -> Self {
        Self { config }
    }

    /// Create with default config
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(ConversionConfig::default())
    }

    /// Execute all conversion tests for a model
    ///
    /// # Errors
    ///
    /// Returns an error if a critical conversion failure occurs.
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

        // Test all format pairs
        if self.config.test_all_pairs {
            for (source, target) in all_conversion_pairs() {
                for backend in &backends {
                    let test = ConversionTest::new(source, target, *backend, model_id.clone());

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
                                },
                            });
                        }
                    }
                }
            }
        }

        // Test round-trips (GGUF → APR → SafeTensors → GGUF)
        if self.config.test_round_trips {
            for backend in &backends {
                let rt = RoundTripTest::new(
                    vec![Format::Gguf, Format::Apr, Format::SafeTensors, Format::Gguf],
                    *backend,
                    model_id.clone(),
                );

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
}
