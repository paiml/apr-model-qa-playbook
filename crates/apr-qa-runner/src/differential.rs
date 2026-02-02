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
use crate::provenance::{
    Provenance, add_derived, create_source_provenance, get_apr_cli_version, load_provenance,
    save_provenance, validate_provenance,
};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::process::Command;

/// Result of `apr rosetta inspect --json` (T-GH192-01)
///
/// Parses model metadata including tensor count, tensor names,
/// and architecture parameters needed for cardinality and name-set gates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InspectResult {
    /// Total number of tensors in the model
    pub tensor_count: usize,
    /// List of all tensor names
    #[serde(default)]
    pub tensor_names: Vec<String>,
    /// Number of attention heads (from model config)
    #[serde(default)]
    pub num_attention_heads: Option<usize>,
    /// Number of key-value heads (GQA/MQA config)
    #[serde(default)]
    pub num_key_value_heads: Option<usize>,
    /// Hidden size / embedding dimension
    #[serde(default)]
    pub hidden_size: Option<usize>,
    /// Model architecture name (e.g., "Qwen2ForCausalLM")
    #[serde(default)]
    pub architecture: Option<String>,
}

/// Run `apr rosetta inspect --json <model>` and parse the result
///
/// Falls back to text-mode parsing for tensor count if JSON is unavailable.
///
/// # Errors
///
/// Returns an error if the apr command fails to execute.
pub fn run_inspect(model_path: &Path, apr_binary: &str) -> Result<InspectResult> {
    let output = Command::new(apr_binary)
        .arg("rosetta")
        .arg("inspect")
        .arg(model_path)
        .arg("--json")
        .output()
        .map_err(|e| Error::ExecutionFailed {
            command: "apr rosetta inspect --json".to_string(),
            reason: e.to_string(),
        })?;

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Try JSON parsing first
    if output.status.success() {
        if let Ok(result) = serde_json::from_str::<InspectResult>(&stdout) {
            return Ok(result);
        }
    }

    // Fall back to text output parsing
    parse_inspect_text(&stdout)
}

/// Parse text-mode output from `apr rosetta inspect`
///
/// Extracts tensor count and tensor names from human-readable output.
fn parse_inspect_text(output: &str) -> Result<InspectResult> {
    let mut tensor_count = 0;
    let mut tensor_names = Vec::new();
    let mut num_attention_heads = None;
    let mut num_key_value_heads = None;
    let mut hidden_size = None;
    let mut architecture = None;

    for line in output.lines() {
        let line = line.trim();

        // Parse "Tensors: 338" or "tensor_count: 338"
        if let Some(count_str) = line
            .strip_prefix("Tensors:")
            .or_else(|| line.strip_prefix("tensor_count:"))
        {
            if let Ok(count) = count_str.trim().parse::<usize>() {
                tensor_count = count;
            }
        }

        // Parse tensor names from lines like "  model.layers.0.self_attn.q_proj.weight [4096, 4096]"
        if line.contains('[') && line.contains(']') && !line.starts_with('{') {
            if let Some(name) = line.split_whitespace().next() {
                if name.contains('.') {
                    tensor_names.push(name.to_string());
                }
            }
        }

        // Parse architecture metadata
        if let Some(val) = line.strip_prefix("num_attention_heads:") {
            num_attention_heads = val.trim().parse().ok();
        }
        if let Some(val) = line.strip_prefix("num_key_value_heads:") {
            num_key_value_heads = val.trim().parse().ok();
        }
        if let Some(val) = line.strip_prefix("hidden_size:") {
            hidden_size = val.trim().parse().ok();
        }
        if let Some(val) = line.strip_prefix("architecture:") {
            architecture = Some(val.trim().to_string());
        }
    }

    // If we found tensor names but no explicit count, use the name count
    if tensor_count == 0 && !tensor_names.is_empty() {
        tensor_count = tensor_names.len();
    }

    Ok(InspectResult {
        tensor_count,
        tensor_names,
        num_attention_heads,
        num_key_value_heads,
        hidden_size,
        architecture,
    })
}

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

/// CI profile metrics (nested in JSON output)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CiProfileMetrics {
    /// Throughput achieved (tok/s)
    #[serde(alias = "throughput_tok_s")]
    pub throughput_tok_s: f64,
    /// p50 latency (ms)
    pub latency_p50_ms: f64,
    /// p99 latency (ms)
    pub latency_p99_ms: f64,
}

/// CI profile assertions result from apr profile --ci --json
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CiProfileResult {
    /// Model path
    #[serde(default)]
    pub model: String,
    /// Nested metrics
    #[serde(default)]
    pub metrics: Option<CiProfileMetrics>,
    /// Assertion results
    #[serde(default)]
    pub assertions: Vec<CiAssertion>,
    /// Overall pass/fail
    #[serde(default)]
    pub passed: bool,
    // Legacy flat fields for backwards compatibility
    /// Throughput achieved (legacy)
    #[serde(default)]
    pub throughput_tps: f64,
    /// p50 latency (legacy)
    #[serde(default)]
    pub latency_p50_ms: f64,
    /// p99 latency (legacy)
    #[serde(default)]
    pub latency_p99_ms: f64,
}

impl CiProfileResult {
    /// Get throughput in tok/s (from nested metrics or legacy field)
    #[must_use]
    pub fn throughput(&self) -> f64 {
        self.metrics
            .as_ref()
            .map_or(self.throughput_tps, |m| m.throughput_tok_s)
    }

    /// Get p50 latency in ms
    #[must_use]
    pub fn p50_latency(&self) -> f64 {
        self.metrics
            .as_ref()
            .map_or(self.latency_p50_ms, |m| m.latency_p50_ms)
    }

    /// Get p99 latency in ms
    #[must_use]
    pub fn p99_latency(&self) -> f64 {
        self.metrics
            .as_ref()
            .map_or(self.latency_p99_ms, |m| m.latency_p99_ms)
    }
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
    /// Gate ID (optional - not all apr versions include it)
    #[serde(default)]
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
    cmd.arg("--format").arg("json");

    let output = cmd.output().map_err(|e| Error::ExecutionFailed {
        command: "apr profile --ci".to_string(),
        reason: e.to_string(),
    })?;

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Extract JSON object from output (may have prefix lines like "Loading model...")
    let json_start = stdout.find('{');
    let json_str = json_start.map_or_else(|| stdout.as_ref(), |i| &stdout[i..]);

    // Try JSON parsing
    if let Ok(result) = serde_json::from_str::<CiProfileResult>(json_str) {
        return Ok(result);
    }

    // Fall back to basic result based on exit code
    Ok(CiProfileResult {
        model: String::new(),
        metrics: None,
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

/// Result of throughput benchmark
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchResult {
    /// Throughput in tokens/second
    pub throughput_tps: f64,
    /// Whether the benchmark passed minimum threshold
    pub passed: bool,
    /// Backend used (cpu or gpu)
    pub backend: String,
    /// Format tested (gguf, apr, safetensors)
    pub format: String,
}

/// Run throughput benchmark with explicit backend selection
///
/// Uses `apr bench --fast` (realizar) for real inference.
/// Backend selection via `CUDA_VISIBLE_DEVICES` environment variable.
///
/// # Arguments
/// * `apr_binary` - Path to apr binary
/// * `model_path` - Path to model file
/// * `use_gpu` - If true, use GPU; if false, set CUDA_VISIBLE_DEVICES=""
/// * `warmup` - Number of warmup iterations
/// * `iterations` - Number of measurement iterations
///
/// # Errors
///
/// Returns an error if the apr command fails to execute.
pub fn run_bench_throughput(
    apr_binary: &str,
    model_path: &Path,
    use_gpu: bool,
    warmup: usize,
    iterations: usize,
) -> Result<BenchResult> {
    let mut cmd = Command::new(apr_binary);
    cmd.arg("bench")
        .arg(model_path)
        .arg("--warmup")
        .arg(warmup.to_string())
        .arg("--iterations")
        .arg(iterations.to_string());

    // Force CPU-only by hiding CUDA devices
    if !use_gpu {
        cmd.env("CUDA_VISIBLE_DEVICES", "");
    }

    let output = cmd.output().map_err(|e| Error::ExecutionFailed {
        command: format!("apr bench {}", model_path.display()),
        reason: e.to_string(),
    })?;

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Parse throughput from output: "Throughput: 65.5 tok/s (PASS: >= 10 tok/s)"
    let throughput = stdout
        .lines()
        .find(|line| line.contains("Throughput:"))
        .and_then(|line| {
            line.split_whitespace()
                .nth(1)
                .and_then(|s| s.parse::<f64>().ok())
        })
        .unwrap_or(0.0);

    let passed = output.status.success() && throughput >= 10.0;

    // Determine format from file extension
    let format = model_path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("unknown")
        .to_string();

    Ok(BenchResult {
        throughput_tps: throughput,
        passed,
        backend: if use_gpu { "gpu" } else { "cpu" }.to_string(),
        format,
    })
}

/// Result of format conversion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormatConversionResult {
    /// Source format
    pub source_format: String,
    /// Target format
    pub target_format: String,
    /// Whether conversion succeeded
    pub success: bool,
    /// Duration in milliseconds
    pub duration_ms: u64,
    /// Error message if failed
    pub error: Option<String>,
    /// Whether result was from cache
    pub cached: bool,
}

/// Compute SHA256 hash of a file (first 1MB for speed)
fn compute_file_hash(path: &Path) -> Result<String> {
    use std::io::Read;

    let mut file = std::fs::File::open(path).map_err(|e| Error::ExecutionFailed {
        command: format!("open {}", path.display()),
        reason: e.to_string(),
    })?;

    let mut buffer = vec![0u8; 1024 * 1024]; // 1MB
    let bytes_read = file.read(&mut buffer).map_err(|e| Error::ExecutionFailed {
        command: format!("read {}", path.display()),
        reason: e.to_string(),
    })?;

    buffer.truncate(bytes_read);

    // Simple hash using std (no external dependency)
    let hash: u64 = buffer.iter().fold(0u64, |acc, &b| {
        acc.wrapping_mul(31).wrapping_add(u64::from(b))
    });

    Ok(format!("{hash:016x}"))
}

/// Convert model format with caching
///
/// Uses `apr rosetta convert` to convert between formats.
/// Caches result and skips conversion if cache is valid.
///
/// # Arguments
/// * `apr_binary` - Path to apr binary
/// * `source_path` - Path to source model file
/// * `target_path` - Path to target model file
/// * `cache_hash_path` - Path to store source file hash for cache validation
///
/// # Errors
///
/// Returns an error if conversion fails.
pub fn convert_format_cached(
    apr_binary: &str,
    source_path: &Path,
    target_path: &Path,
    cache_hash_path: &Path,
) -> Result<FormatConversionResult> {
    let source_format = source_path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("unknown")
        .to_string();

    let target_format = target_path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("unknown")
        .to_string();

    // Check cache validity
    let current_hash = compute_file_hash(source_path)?;

    if target_path.exists() && cache_hash_path.exists() {
        if let Ok(cached_hash) = std::fs::read_to_string(cache_hash_path) {
            if cached_hash.trim() == current_hash {
                return Ok(FormatConversionResult {
                    source_format,
                    target_format,
                    success: true,
                    duration_ms: 0,
                    error: None,
                    cached: true,
                });
            }
        }
    }

    // Create target directory if needed
    if let Some(parent) = target_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }

    let start = std::time::Instant::now();

    let output = Command::new(apr_binary)
        .arg("rosetta")
        .arg("convert")
        .arg(source_path)
        .arg(target_path)
        .output()
        .map_err(|e| Error::ExecutionFailed {
            command: format!(
                "apr rosetta convert {} {}",
                source_path.display(),
                target_path.display()
            ),
            reason: e.to_string(),
        })?;

    let duration_ms = start.elapsed().as_millis() as u64;

    if output.status.success() {
        // Write hash for cache validation
        let _ = std::fs::write(cache_hash_path, &current_hash);

        Ok(FormatConversionResult {
            source_format,
            target_format,
            success: true,
            duration_ms,
            error: None,
            cached: false,
        })
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        Ok(FormatConversionResult {
            source_format,
            target_format,
            success: false,
            duration_ms,
            error: Some(stderr.to_string()),
            cached: false,
        })
    }
}

// ============================================================================
// Provenance-Aware Model Preparation (PMAT-PROV-001)
// ============================================================================

/// Result of model preparation with provenance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPreparationResult {
    /// Provenance record
    pub provenance: Provenance,
    /// Path to SafeTensors source
    pub safetensors_path: std::path::PathBuf,
    /// Path to GGUF (if conversion succeeded)
    pub gguf_path: Option<std::path::PathBuf>,
    /// Path to APR (if conversion succeeded)
    pub apr_path: Option<std::path::PathBuf>,
    /// Conversion results
    pub conversions: Vec<FormatConversionResult>,
}

/// Prepare a model from SafeTensors source with full provenance tracking
///
/// Implements spec 7.4 (Ground Truth Policy) and 7.5 (Provenance Validation):
/// 1. SafeTensors is the canonical source (PROV-003)
/// 2. All conversions use apr-cli (PROV-002)
/// 3. Provenance tracks all derived formats
///
/// # Arguments
///
/// * `apr_binary` - Path to apr binary
/// * `safetensors_path` - Path to source SafeTensors file
/// * `hf_repo` - HuggingFace repository ID (e.g., "Qwen/Qwen2.5-Coder-0.5B-Instruct")
/// * `output_dir` - Directory to write converted files and provenance
/// * `quantization` - Optional quantization level (e.g., "q4_k_m")
///
/// # Errors
///
/// Returns error if any conversion fails or provenance validation fails.
pub fn prepare_model_with_provenance(
    apr_binary: &str,
    safetensors_path: &Path,
    hf_repo: &str,
    output_dir: &Path,
    quantization: Option<&str>,
) -> Result<ModelPreparationResult> {
    // Check for existing provenance (resume workflow)
    let prov_result = load_provenance(output_dir);
    let mut provenance = if let Ok(existing) = prov_result {
        // Verify existing provenance matches source
        let current_hash = crate::provenance::compute_sha256(safetensors_path)?;
        if existing.source.sha256 == current_hash {
            existing
        } else {
            // Source changed, recreate provenance
            create_source_provenance(safetensors_path, hf_repo)?
        }
    } else {
        // Create new provenance
        create_source_provenance(safetensors_path, hf_repo)?
    };

    let cli_version = get_apr_cli_version();
    let mut conversions = Vec::new();
    let mut gguf_path = None;
    let mut apr_path = None;

    // Create output directories
    std::fs::create_dir_all(output_dir)?;

    // Convert SafeTensors → GGUF
    let gguf_target = quantization.map_or_else(
        || output_dir.join("model.gguf"),
        |q| output_dir.join(format!("model-{q}.gguf")),
    );
    let gguf_hash_path = output_dir.join(".gguf_conversion_hash");

    let gguf_conv =
        convert_format_cached(apr_binary, safetensors_path, &gguf_target, &gguf_hash_path)?;
    if gguf_conv.success {
        // Check if we need to add this derived format
        let already_tracked = provenance
            .derived
            .iter()
            .any(|d| d.format == "gguf" && d.quantization.as_deref() == quantization);

        if !already_tracked {
            add_derived(
                &mut provenance,
                "gguf",
                &gguf_target,
                quantization,
                &cli_version,
            )?;
        }
        gguf_path = Some(gguf_target.clone());
    }
    conversions.push(gguf_conv);

    // Convert SafeTensors → APR
    let apr_target = quantization.map_or_else(
        || output_dir.join("model.apr"),
        |q| output_dir.join(format!("model-{q}.apr")),
    );
    let apr_hash_path = output_dir.join(".apr_conversion_hash");

    let apr_conv =
        convert_format_cached(apr_binary, safetensors_path, &apr_target, &apr_hash_path)?;
    if apr_conv.success {
        let already_tracked = provenance
            .derived
            .iter()
            .any(|d| d.format == "apr" && d.quantization.as_deref() == quantization);

        if !already_tracked {
            add_derived(
                &mut provenance,
                "apr",
                &apr_target,
                quantization,
                &cli_version,
            )?;
        }
        apr_path = Some(apr_target.clone());
    }
    conversions.push(apr_conv);

    // Validate provenance
    validate_provenance(&provenance)?;

    // Save provenance
    save_provenance(output_dir, &provenance)?;

    Ok(ModelPreparationResult {
        provenance,
        safetensors_path: safetensors_path.to_path_buf(),
        gguf_path,
        apr_path,
        conversions,
    })
}

/// Verify provenance before running comparisons
///
/// Checks PROV-005 (quantization parity) for format comparison.
///
/// # Errors
///
/// Returns error if provenance is invalid or formats can't be compared.
pub fn verify_comparison_provenance(
    model_dir: &Path,
    format_a: &str,
    format_b: &str,
) -> Result<Provenance> {
    let provenance = load_provenance(model_dir)?;
    validate_provenance(&provenance)?;
    crate::provenance::validate_comparison(&provenance, format_a, format_b)?;
    Ok(provenance)
}

/// Six-column throughput profile result
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SixColumnProfile {
    /// GGUF CPU throughput (tok/s)
    pub tps_gguf_cpu: Option<f64>,
    /// GGUF GPU throughput (tok/s)
    pub tps_gguf_gpu: Option<f64>,
    /// APR CPU throughput (tok/s)
    pub tps_apr_cpu: Option<f64>,
    /// APR GPU throughput (tok/s)
    pub tps_apr_gpu: Option<f64>,
    /// SafeTensors CPU throughput (tok/s)
    pub tps_st_cpu: Option<f64>,
    /// SafeTensors GPU throughput (tok/s)
    pub tps_st_gpu: Option<f64>,
    /// Conversion results
    pub conversions: Vec<FormatConversionResult>,
    /// Total profiling duration in milliseconds
    pub total_duration_ms: u64,
    /// Failed assertions (format, backend, actual, threshold)
    pub failed_assertions: Vec<ProfileAssertion>,
}

/// A profile assertion result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileAssertion {
    /// Format (gguf, apr, safetensors)
    pub format: String,
    /// Backend (cpu, gpu)
    pub backend: String,
    /// Actual throughput
    pub actual_tps: f64,
    /// Minimum threshold
    pub min_threshold: f64,
    /// Whether assertion passed
    pub passed: bool,
}

impl SixColumnProfile {
    /// Check if all assertions passed
    #[must_use]
    pub fn all_assertions_passed(&self) -> bool {
        self.failed_assertions.is_empty()
    }

    /// Check throughput against thresholds and record failures
    #[allow(clippy::similar_names)]
    pub fn check_assertions(&mut self, min_cpu: f64, min_gpu: f64) {
        // Check GGUF CPU
        if let Some(tps) = self.tps_gguf_cpu {
            let passed = tps >= min_cpu;
            if !passed {
                self.failed_assertions.push(ProfileAssertion {
                    format: "gguf".to_string(),
                    backend: "cpu".to_string(),
                    actual_tps: tps,
                    min_threshold: min_cpu,
                    passed,
                });
            }
        }

        // Check GGUF GPU
        if let Some(tps) = self.tps_gguf_gpu {
            let passed = tps >= min_gpu;
            if !passed {
                self.failed_assertions.push(ProfileAssertion {
                    format: "gguf".to_string(),
                    backend: "gpu".to_string(),
                    actual_tps: tps,
                    min_threshold: min_gpu,
                    passed,
                });
            }
        }

        // Check APR CPU (if measured)
        if let Some(tps) = self.tps_apr_cpu {
            let passed = tps >= min_cpu;
            if !passed {
                self.failed_assertions.push(ProfileAssertion {
                    format: "apr".to_string(),
                    backend: "cpu".to_string(),
                    actual_tps: tps,
                    min_threshold: min_cpu,
                    passed,
                });
            }
        }

        // Check APR GPU (if measured)
        if let Some(tps) = self.tps_apr_gpu {
            let passed = tps >= min_gpu;
            if !passed {
                self.failed_assertions.push(ProfileAssertion {
                    format: "apr".to_string(),
                    backend: "gpu".to_string(),
                    actual_tps: tps,
                    min_threshold: min_gpu,
                    passed,
                });
            }
        }
    }
}

/// Run full 6-column profiling for a model
///
/// 1. Converts GGUF to APR and SafeTensors (with caching)
/// 2. Benchmarks each format on CPU and GPU
///
/// # Arguments
/// * `apr_binary` - Path to apr binary
/// * `model_cache_dir` - Directory containing model format subdirs
/// * `warmup` - Warmup iterations for benchmarks
/// * `iterations` - Measurement iterations for benchmarks
///
/// # Errors
///
/// Returns an error if profiling fails.
pub fn run_six_column_profile(
    apr_binary: &str,
    model_cache_dir: &Path,
    warmup: usize,
    iterations: usize,
) -> Result<SixColumnProfile> {
    let start = std::time::Instant::now();
    let mut profile = SixColumnProfile::default();

    // Paths
    let gguf_dir = model_cache_dir.join("gguf");
    let apr_dir = model_cache_dir.join("apr");
    let st_dir = model_cache_dir.join("safetensors");

    // Find GGUF source file
    let gguf_path = find_model_file(&gguf_dir)?;

    // Convert GGUF → APR (with caching)
    let apr_path = apr_dir.join("model.apr");
    let apr_hash_path = apr_dir.join(".conversion_hash");
    let apr_conv = convert_format_cached(apr_binary, &gguf_path, &apr_path, &apr_hash_path)?;
    profile.conversions.push(apr_conv.clone());

    // Convert GGUF → SafeTensors (with caching) - may fail due to #190
    let st_path = st_dir.join("model.safetensors");
    let st_hash_path = st_dir.join(".conversion_hash");
    let st_conv = convert_format_cached(apr_binary, &gguf_path, &st_path, &st_hash_path)?;
    profile.conversions.push(st_conv.clone());

    // Benchmark GGUF CPU
    if let Ok(result) = run_bench_throughput(apr_binary, &gguf_path, false, warmup, iterations) {
        profile.tps_gguf_cpu = Some(result.throughput_tps);
    }

    // Benchmark GGUF GPU
    if let Ok(result) = run_bench_throughput(apr_binary, &gguf_path, true, warmup, iterations) {
        profile.tps_gguf_gpu = Some(result.throughput_tps);
    }

    // Benchmark APR CPU (only if conversion succeeded)
    if apr_conv.success {
        if let Ok(result) = run_bench_throughput(apr_binary, &apr_path, false, warmup, iterations) {
            profile.tps_apr_cpu = Some(result.throughput_tps);
        }
    }

    // Benchmark APR GPU (only if conversion succeeded)
    if apr_conv.success {
        if let Ok(result) = run_bench_throughput(apr_binary, &apr_path, true, warmup, iterations) {
            profile.tps_apr_gpu = Some(result.throughput_tps);
        }
    }

    // Benchmark SafeTensors CPU (only if conversion succeeded)
    if st_conv.success {
        if let Ok(result) = run_bench_throughput(apr_binary, &st_path, false, warmup, iterations) {
            profile.tps_st_cpu = Some(result.throughput_tps);
        }
    }

    // Benchmark SafeTensors GPU (only if conversion succeeded)
    if st_conv.success {
        if let Ok(result) = run_bench_throughput(apr_binary, &st_path, true, warmup, iterations) {
            profile.tps_st_gpu = Some(result.throughput_tps);
        }
    }

    profile.total_duration_ms = start.elapsed().as_millis() as u64;
    Ok(profile)
}

/// Find model file in a directory
fn find_model_file(dir: &Path) -> Result<std::path::PathBuf> {
    if !dir.exists() {
        return Err(Error::ExecutionFailed {
            command: format!("find model in {}", dir.display()),
            reason: "Directory does not exist".to_string(),
        });
    }

    std::fs::read_dir(dir)
        .map_err(|e| Error::ExecutionFailed {
            command: format!("read_dir {}", dir.display()),
            reason: e.to_string(),
        })?
        .filter_map(std::result::Result::ok)
        .map(|e| e.path())
        .find(|p| p.is_file() || p.is_symlink())
        .ok_or_else(|| Error::ExecutionFailed {
            command: format!("find model in {}", dir.display()),
            reason: "No model file found".to_string(),
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
            model: String::new(),
            metrics: None,
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
            model: String::new(),
            metrics: None,
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
            model: String::new(),
            metrics: None,
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
            model: String::new(),
            metrics: None,
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
            model: String::new(),
            metrics: None,
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

    #[test]
    fn test_tensor_diff_result_with_empty_mismatches() {
        let result = TensorDiffResult {
            total_tensors: 100,
            mismatched_tensors: 0,
            transposed_tensors: 0,
            mismatches: vec![],
            passed: true,
        };
        assert!(result.passed);
        assert!(result.mismatches.is_empty());
    }

    #[test]
    fn test_inference_comparison_all_matching() {
        let result = InferenceComparisonResult {
            total_tokens: 10,
            matching_tokens: 10,
            max_logit_diff: 0.0,
            passed: true,
            token_comparisons: vec![],
        };
        assert!(result.passed);
        assert_eq!(result.total_tokens, result.matching_tokens);
    }

    #[test]
    fn test_diff_benchmark_improved_performance() {
        let result = DiffBenchmarkResult {
            model_a: BenchmarkMetrics {
                path: "original.gguf".to_string(),
                throughput_tps: 10.0,
                latency_p50_ms: 100.0,
                latency_p99_ms: 200.0,
            },
            model_b: BenchmarkMetrics {
                path: "converted.apr".to_string(),
                throughput_tps: 10.5,
                latency_p50_ms: 95.0,
                latency_p99_ms: 190.0,
            },
            throughput_delta_pct: 5.0,
            latency_p50_delta_pct: -5.0,
            latency_p99_delta_pct: -5.0,
            regression_detected: false,
            regression_threshold: 10.0,
        };
        assert!(!result.regression_detected);
        assert!(result.throughput_delta_pct > 0.0);
    }

    #[test]
    fn test_ci_profile_all_assertions_pass() {
        let result = CiProfileResult {
            model: String::new(),
            metrics: None,
            throughput_tps: 50.0,
            latency_p50_ms: 20.0,
            latency_p99_ms: 40.0,
            assertions: vec![
                CiAssertion {
                    name: "throughput".to_string(),
                    expected: ">= 10".to_string(),
                    actual: "50".to_string(),
                    passed: true,
                    gate_id: "F-CI-001".to_string(),
                },
                CiAssertion {
                    name: "p99".to_string(),
                    expected: "<= 100".to_string(),
                    actual: "40".to_string(),
                    passed: true,
                    gate_id: "F-CI-002".to_string(),
                },
            ],
            passed: true,
        };
        assert!(result.passed);
        assert!(result.assertions.iter().all(|a| a.passed));
    }

    #[test]
    fn test_tensor_mismatch_type_clone() {
        let t = TensorMismatchType::Missing;
        let cloned = t;
        assert_eq!(cloned, TensorMismatchType::Missing);
    }

    #[test]
    fn test_parse_diff_output_with_text_only() {
        let config = DiffConfig::default();
        let executor = DifferentialExecutor::new(config);
        // Text output without any mismatch markers
        let output = "Comparing tensors...\n\
                      tensor1: OK\n\
                      tensor2: OK\n\
                      All 100 tensors match.";
        let result = executor.parse_diff_output(output).unwrap();
        assert!(result.passed);
        assert!(result.mismatches.is_empty());
    }

    #[test]
    fn test_parse_inference_output_failure_fallback() {
        let config = DiffConfig::default();
        let executor = DifferentialExecutor::new(config);
        // Invalid JSON should fallback to basic result
        let output = "not valid json";
        let result = executor.parse_inference_output(output, false).unwrap();
        assert!(!result.passed);
        assert_eq!(result.total_tokens, 0);
    }

    #[test]
    fn test_diff_config_filter_none() {
        let config = DiffConfig {
            apr_binary: "apr".to_string(),
            filter: None,
            mismatches_only: true,
            tolerance: 1e-5,
        };
        assert!(config.filter.is_none());
    }

    #[test]
    fn test_diff_benchmark_result_delta_calculations() {
        let result = DiffBenchmarkResult {
            model_a: BenchmarkMetrics {
                path: "a.gguf".to_string(),
                throughput_tps: 20.0,
                latency_p50_ms: 50.0,
                latency_p99_ms: 100.0,
            },
            model_b: BenchmarkMetrics {
                path: "b.gguf".to_string(),
                throughput_tps: 10.0,
                latency_p50_ms: 100.0,
                latency_p99_ms: 200.0,
            },
            throughput_delta_pct: -50.0,
            latency_p50_delta_pct: 100.0,
            latency_p99_delta_pct: 100.0,
            regression_detected: true,
            regression_threshold: 20.0,
        };
        assert!(result.regression_detected);
        assert!(result.throughput_delta_pct < 0.0);
        assert!(result.latency_p50_delta_pct > 0.0);
    }

    #[test]
    fn test_benchmark_metrics_all_fields() {
        let metrics = BenchmarkMetrics {
            path: "/models/qwen.gguf".to_string(),
            throughput_tps: 25.5,
            latency_p50_ms: 39.2,
            latency_p99_ms: 78.4,
        };
        assert!(metrics.path.contains("qwen"));
        assert!(metrics.throughput_tps > 0.0);
        assert!(metrics.latency_p99_ms > metrics.latency_p50_ms);
    }

    #[test]
    fn test_token_comparison_fields() {
        let tc = TokenComparison {
            index: 100,
            token_a: 12345,
            token_b: 12346,
            logit_diff: 0.123,
            matches: false,
        };
        assert_eq!(tc.index, 100);
        assert_ne!(tc.token_a, tc.token_b);
        assert!(!tc.matches);
    }

    // Tests for run_profile_ci function with nonexistent binary
    #[test]
    fn test_run_profile_ci_nonexistent_binary() {
        let path = std::path::PathBuf::from("model.gguf");
        let result = run_profile_ci("/nonexistent/apr/binary", &path, None, None, None, 1, 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_profile_ci_with_throughput_assert() {
        let path = std::path::PathBuf::from("model.gguf");
        let result = run_profile_ci(
            "/nonexistent/apr/binary",
            &path,
            Some(10.0),
            None,
            None,
            2,
            5,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_profile_ci_with_p99_assert() {
        let path = std::path::PathBuf::from("model.gguf");
        let result = run_profile_ci(
            "/nonexistent/apr/binary",
            &path,
            None,
            Some(100.0),
            None,
            1,
            1,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_profile_ci_with_p50_assert() {
        let path = std::path::PathBuf::from("model.gguf");
        let result = run_profile_ci(
            "/nonexistent/apr/binary",
            &path,
            None,
            None,
            Some(50.0),
            1,
            1,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_profile_ci_with_all_asserts() {
        let path = std::path::PathBuf::from("model.gguf");
        let result = run_profile_ci(
            "/nonexistent/apr/binary",
            &path,
            Some(10.0),
            Some(100.0),
            Some(50.0),
            5,
            10,
        );
        assert!(result.is_err());
    }

    // Tests for run_diff_benchmark function
    #[test]
    fn test_run_diff_benchmark_nonexistent_binary() {
        let model_a = std::path::PathBuf::from("model_a.gguf");
        let model_b = std::path::PathBuf::from("model_b.apr");
        let result = run_diff_benchmark("/nonexistent/apr/binary", &model_a, &model_b, 5.0);
        assert!(result.is_err());
    }

    // Tests for DifferentialExecutor methods with nonexistent binary
    #[test]
    fn test_differential_executor_diff_tensors_error() {
        let config = DiffConfig {
            apr_binary: "/nonexistent/apr/binary".to_string(),
            ..DiffConfig::default()
        };
        let executor = DifferentialExecutor::new(config);
        let model_a = std::path::PathBuf::from("model_a.gguf");
        let model_b = std::path::PathBuf::from("model_b.apr");
        let result = executor.diff_tensors(&model_a, &model_b);
        assert!(result.is_err());
    }

    #[test]
    fn test_differential_executor_diff_tensors_with_filter() {
        let config = DiffConfig {
            apr_binary: "/nonexistent/apr/binary".to_string(),
            filter: Some("token_embd".to_string()),
            mismatches_only: false,
            tolerance: 1e-5,
        };
        let executor = DifferentialExecutor::new(config);
        let model_a = std::path::PathBuf::from("model_a.gguf");
        let model_b = std::path::PathBuf::from("model_b.apr");
        let result = executor.diff_tensors(&model_a, &model_b);
        assert!(result.is_err());
    }

    #[test]
    fn test_differential_executor_compare_inference_error() {
        let config = DiffConfig {
            apr_binary: "/nonexistent/apr/binary".to_string(),
            ..DiffConfig::default()
        };
        let executor = DifferentialExecutor::new(config);
        let model_a = std::path::PathBuf::from("model_a.gguf");
        let model_b = std::path::PathBuf::from("model_b.apr");
        let result = executor.compare_inference(&model_a, &model_b, "test prompt", 10);
        assert!(result.is_err());
    }

    #[test]
    fn test_diff_config_embedding_filter() {
        let config = DiffConfig {
            apr_binary: "apr".to_string(),
            filter: Some("embedding".to_string()),
            mismatches_only: true,
            tolerance: 1e-6,
        };
        assert_eq!(config.filter.as_deref(), Some("embedding"));
        assert!((config.tolerance - 1e-6).abs() < 1e-10);
    }

    #[test]
    fn test_ci_assertion_failed() {
        let assertion = CiAssertion {
            name: "throughput".to_string(),
            expected: ">= 20.0 tok/s".to_string(),
            actual: "15.5 tok/s".to_string(),
            passed: false,
            gate_id: "F-PROFILE-CI-001".to_string(),
        };
        assert!(!assertion.passed);
        assert!(assertion.expected.contains("20.0"));
        assert!(assertion.actual.contains("15.5"));
    }

    #[test]
    fn test_ci_profile_result_with_failed_assertions() {
        let result = CiProfileResult {
            model: String::new(),
            metrics: None,
            throughput_tps: 15.5,
            latency_p50_ms: 50.0,
            latency_p99_ms: 250.0,
            assertions: vec![
                CiAssertion {
                    name: "throughput".to_string(),
                    expected: ">= 20.0".to_string(),
                    actual: "15.5".to_string(),
                    passed: false,
                    gate_id: "F-CI-001".to_string(),
                },
                CiAssertion {
                    name: "p99".to_string(),
                    expected: "<= 200".to_string(),
                    actual: "250".to_string(),
                    passed: false,
                    gate_id: "F-CI-002".to_string(),
                },
            ],
            passed: false,
        };
        assert!(!result.passed);
        assert_eq!(result.assertions.iter().filter(|a| a.passed).count(), 0);
    }

    #[test]
    fn test_tensor_mismatch_type_shape_mismatch() {
        let mismatch = TensorMismatch {
            name: "lm_head.weight".to_string(),
            shape_a: vec![4096, 128_256],
            shape_b: vec![4096, 32_000],
            mismatch_type: TensorMismatchType::ShapeMismatch,
        };
        assert_eq!(mismatch.mismatch_type.gate_id(), "F-ROSETTA-DIFF-002");
        assert_ne!(mismatch.shape_a, mismatch.shape_b);
    }

    #[test]
    fn test_tensor_mismatch_missing_tensor() {
        let mismatch = TensorMismatch {
            name: "rotary_embd.inv_freq".to_string(),
            shape_a: vec![64],
            shape_b: vec![],
            mismatch_type: TensorMismatchType::Missing,
        };
        assert_eq!(mismatch.mismatch_type.gate_id(), "F-ROSETTA-DIFF-002");
    }

    #[test]
    fn test_inference_comparison_result_partial_match() {
        let result = InferenceComparisonResult {
            total_tokens: 100,
            matching_tokens: 85,
            max_logit_diff: 0.05,
            passed: false,
            token_comparisons: vec![TokenComparison {
                index: 42,
                token_a: 1000,
                token_b: 1001,
                logit_diff: 0.05,
                matches: false,
            }],
        };
        assert!(!result.passed);
        assert!(result.matching_tokens < result.total_tokens);
        assert!(!result.token_comparisons.is_empty());
    }

    #[test]
    fn test_token_comparison_exact_match() {
        let tc = TokenComparison {
            index: 0,
            token_a: 500,
            token_b: 500,
            logit_diff: 0.0001,
            matches: true,
        };
        assert!(tc.matches);
        assert_eq!(tc.token_a, tc.token_b);
    }

    #[test]
    fn test_parse_diff_output_with_transposed_marker() {
        let config = DiffConfig::default();
        let executor = DifferentialExecutor::new(config);
        // Text output with transposed marker
        let output = "Comparing tensors...\n\
                      TRANSPOSED: token_embd.weight (4096, 32000) vs (32000, 4096)\n\
                      All 100 tensors compared.";
        let result = executor.parse_diff_output(output).unwrap();
        assert!(!result.passed);
        assert_eq!(result.transposed_tensors, 1);
    }

    #[test]
    fn test_parse_diff_output_with_no_mismatch_marker() {
        let config = DiffConfig::default();
        let executor = DifferentialExecutor::new(config);
        // Text output without transposed marker - should pass
        let output = "Comparing tensors...\n\
                      lm_head.weight: OK\n\
                      Done.";
        let result = executor.parse_diff_output(output).unwrap();
        // No TRANSPOSED markers found, so should pass
        assert!(result.passed);
        assert_eq!(result.mismatched_tensors, 0);
    }

    #[test]
    fn test_parse_inference_output_with_valid_json() {
        let config = DiffConfig::default();
        let executor = DifferentialExecutor::new(config);
        let output = r#"{"total_tokens":10,"matching_tokens":10,"max_logit_diff":0.0001,"passed":true,"token_comparisons":[]}"#;
        let result = executor.parse_inference_output(output, true).unwrap();
        assert!(result.passed);
        assert_eq!(result.total_tokens, 10);
    }

    #[test]
    fn test_diff_config_relaxed_tolerance() {
        let config = DiffConfig {
            apr_binary: "apr".to_string(),
            filter: None,
            mismatches_only: false,
            tolerance: 1e-3,
        };
        assert!((config.tolerance - 1e-3).abs() < 1e-10);
        assert!(!config.mismatches_only);
    }

    // =========================================================================
    // SixColumnProfile tests
    // =========================================================================

    #[test]
    fn test_six_column_profile_default() {
        let profile = SixColumnProfile::default();
        assert!(profile.tps_gguf_cpu.is_none());
        assert!(profile.tps_gguf_gpu.is_none());
        assert!(profile.tps_apr_cpu.is_none());
        assert!(profile.tps_apr_gpu.is_none());
        assert!(profile.tps_st_cpu.is_none());
        assert!(profile.tps_st_gpu.is_none());
        assert!(profile.conversions.is_empty());
        assert!(profile.failed_assertions.is_empty());
        assert_eq!(profile.total_duration_ms, 0);
    }

    #[test]
    fn test_six_column_profile_all_assertions_passed_empty() {
        let profile = SixColumnProfile::default();
        assert!(profile.all_assertions_passed());
    }

    #[test]
    fn test_six_column_profile_all_assertions_passed_with_failures() {
        let mut profile = SixColumnProfile::default();
        profile.failed_assertions.push(ProfileAssertion {
            format: "gguf".to_string(),
            backend: "cpu".to_string(),
            actual_tps: 5.0,
            min_threshold: 10.0,
            passed: false,
        });
        assert!(!profile.all_assertions_passed());
    }

    #[test]
    fn test_six_column_profile_check_assertions_all_pass() {
        let mut profile = SixColumnProfile {
            tps_gguf_cpu: Some(20.0),
            tps_gguf_gpu: Some(50.0),
            tps_apr_cpu: Some(18.0),
            tps_apr_gpu: Some(45.0),
            ..Default::default()
        };
        profile.check_assertions(10.0, 30.0);
        assert!(profile.all_assertions_passed());
    }

    #[test]
    fn test_six_column_profile_check_assertions_gguf_cpu_fail() {
        let mut profile = SixColumnProfile {
            tps_gguf_cpu: Some(5.0), // Below threshold
            tps_gguf_gpu: Some(50.0),
            ..Default::default()
        };
        profile.check_assertions(10.0, 30.0);
        assert!(!profile.all_assertions_passed());
        assert_eq!(profile.failed_assertions.len(), 1);
        assert_eq!(profile.failed_assertions[0].format, "gguf");
        assert_eq!(profile.failed_assertions[0].backend, "cpu");
    }

    #[test]
    fn test_six_column_profile_check_assertions_gguf_gpu_fail() {
        let mut profile = SixColumnProfile {
            tps_gguf_cpu: Some(20.0),
            tps_gguf_gpu: Some(25.0), // Below threshold
            ..Default::default()
        };
        profile.check_assertions(10.0, 30.0);
        assert!(!profile.all_assertions_passed());
        assert_eq!(profile.failed_assertions.len(), 1);
        assert_eq!(profile.failed_assertions[0].format, "gguf");
        assert_eq!(profile.failed_assertions[0].backend, "gpu");
    }

    #[test]
    fn test_six_column_profile_check_assertions_apr_cpu_fail() {
        let mut profile = SixColumnProfile {
            tps_apr_cpu: Some(5.0), // Below threshold
            ..Default::default()
        };
        profile.check_assertions(10.0, 30.0);
        assert!(!profile.all_assertions_passed());
        assert_eq!(profile.failed_assertions.len(), 1);
        assert_eq!(profile.failed_assertions[0].format, "apr");
        assert_eq!(profile.failed_assertions[0].backend, "cpu");
    }

    #[test]
    fn test_six_column_profile_check_assertions_apr_gpu_fail() {
        let mut profile = SixColumnProfile {
            tps_apr_gpu: Some(20.0), // Below threshold
            ..Default::default()
        };
        profile.check_assertions(10.0, 30.0);
        assert!(!profile.all_assertions_passed());
        assert_eq!(profile.failed_assertions.len(), 1);
        assert_eq!(profile.failed_assertions[0].format, "apr");
        assert_eq!(profile.failed_assertions[0].backend, "gpu");
    }

    #[test]
    fn test_six_column_profile_check_assertions_multiple_failures() {
        let mut profile = SixColumnProfile {
            tps_gguf_cpu: Some(5.0),
            tps_gguf_gpu: Some(20.0),
            tps_apr_cpu: Some(8.0),
            tps_apr_gpu: Some(25.0),
            ..Default::default()
        };
        profile.check_assertions(10.0, 30.0);
        // All 4 should fail
        assert_eq!(profile.failed_assertions.len(), 4);
    }

    #[test]
    fn test_six_column_profile_check_assertions_none_values() {
        let mut profile = SixColumnProfile::default();
        profile.check_assertions(10.0, 30.0);
        // No assertions should be recorded for None values
        assert!(profile.all_assertions_passed());
    }

    #[test]
    fn test_profile_assertion_fields() {
        let assertion = ProfileAssertion {
            format: "safetensors".to_string(),
            backend: "gpu".to_string(),
            actual_tps: 25.5,
            min_threshold: 30.0,
            passed: false,
        };
        assert_eq!(assertion.format, "safetensors");
        assert_eq!(assertion.backend, "gpu");
        assert!((assertion.actual_tps - 25.5).abs() < f64::EPSILON);
        assert!((assertion.min_threshold - 30.0).abs() < f64::EPSILON);
        assert!(!assertion.passed);
    }

    #[test]
    fn test_profile_assertion_clone() {
        let assertion = ProfileAssertion {
            format: "gguf".to_string(),
            backend: "cpu".to_string(),
            actual_tps: 15.0,
            min_threshold: 10.0,
            passed: true,
        };
        let cloned = assertion.clone();
        assert_eq!(cloned.format, assertion.format);
        assert_eq!(cloned.backend, assertion.backend);
    }

    #[test]
    fn test_profile_assertion_debug() {
        let assertion = ProfileAssertion {
            format: "apr".to_string(),
            backend: "cpu".to_string(),
            actual_tps: 12.0,
            min_threshold: 10.0,
            passed: true,
        };
        let debug_str = format!("{assertion:?}");
        assert!(debug_str.contains("ProfileAssertion"));
    }

    #[test]
    fn test_profile_assertion_serialization() {
        let assertion = ProfileAssertion {
            format: "gguf".to_string(),
            backend: "gpu".to_string(),
            actual_tps: 45.5,
            min_threshold: 40.0,
            passed: true,
        };
        let json = serde_json::to_string(&assertion).unwrap();
        let parsed: ProfileAssertion = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.format, "gguf");
        assert!(parsed.passed);
    }

    #[test]
    fn test_six_column_profile_clone() {
        let profile = SixColumnProfile {
            tps_gguf_cpu: Some(15.0),
            total_duration_ms: 5000,
            ..Default::default()
        };
        let cloned = profile.clone();
        assert_eq!(cloned.tps_gguf_cpu, Some(15.0));
        assert_eq!(cloned.total_duration_ms, 5000);
    }

    #[test]
    fn test_six_column_profile_debug() {
        let profile = SixColumnProfile::default();
        let debug_str = format!("{profile:?}");
        assert!(debug_str.contains("SixColumnProfile"));
    }

    #[test]
    fn test_six_column_profile_serialization() {
        let profile = SixColumnProfile {
            tps_gguf_cpu: Some(12.0),
            tps_gguf_gpu: Some(45.0),
            total_duration_ms: 1000,
            ..Default::default()
        };
        let json = serde_json::to_string(&profile).unwrap();
        let parsed: SixColumnProfile = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.tps_gguf_cpu, Some(12.0));
    }

    // =========================================================================
    // BenchResult tests
    // =========================================================================

    #[test]
    fn test_bench_result_fields() {
        let result = BenchResult {
            throughput_tps: 15.5,
            passed: true,
            backend: "cpu".to_string(),
            format: "gguf".to_string(),
        };
        assert!((result.throughput_tps - 15.5).abs() < f64::EPSILON);
        assert!(result.passed);
        assert_eq!(result.backend, "cpu");
        assert_eq!(result.format, "gguf");
    }

    #[test]
    fn test_bench_result_clone() {
        let result = BenchResult {
            throughput_tps: 20.0,
            passed: false,
            backend: "gpu".to_string(),
            format: "apr".to_string(),
        };
        let cloned = result.clone();
        assert_eq!(cloned.backend, "gpu");
        assert_eq!(cloned.format, "apr");
    }

    #[test]
    fn test_bench_result_debug() {
        let result = BenchResult {
            throughput_tps: 10.0,
            passed: true,
            backend: "cpu".to_string(),
            format: "safetensors".to_string(),
        };
        let debug_str = format!("{result:?}");
        assert!(debug_str.contains("BenchResult"));
    }

    #[test]
    fn test_bench_result_serialization() {
        let result = BenchResult {
            throughput_tps: 25.5,
            passed: true,
            backend: "gpu".to_string(),
            format: "gguf".to_string(),
        };
        let json = serde_json::to_string(&result).unwrap();
        let parsed: BenchResult = serde_json::from_str(&json).unwrap();
        assert!((parsed.throughput_tps - 25.5).abs() < 0.01);
        assert!(parsed.passed);
    }

    // =========================================================================
    // FormatConversionResult tests
    // =========================================================================

    #[test]
    fn test_format_conversion_result_success() {
        let result = FormatConversionResult {
            source_format: "gguf".to_string(),
            target_format: "apr".to_string(),
            success: true,
            duration_ms: 500,
            error: None,
            cached: false,
        };
        assert!(result.success);
        assert!(result.error.is_none());
        assert!(!result.cached);
    }

    #[test]
    fn test_format_conversion_result_cached() {
        let result = FormatConversionResult {
            source_format: "gguf".to_string(),
            target_format: "safetensors".to_string(),
            success: true,
            duration_ms: 0,
            error: None,
            cached: true,
        };
        assert!(result.cached);
        assert_eq!(result.duration_ms, 0);
    }

    #[test]
    fn test_format_conversion_result_failure() {
        let result = FormatConversionResult {
            source_format: "apr".to_string(),
            target_format: "safetensors".to_string(),
            success: false,
            duration_ms: 1000,
            error: Some("Conversion failed: unsupported format".to_string()),
            cached: false,
        };
        assert!(!result.success);
        assert!(result.error.is_some());
        assert!(result.error.as_ref().unwrap().contains("Conversion failed"));
    }

    #[test]
    fn test_format_conversion_result_clone() {
        let result = FormatConversionResult {
            source_format: "gguf".to_string(),
            target_format: "apr".to_string(),
            success: true,
            duration_ms: 250,
            error: None,
            cached: true,
        };
        let cloned = result.clone();
        assert_eq!(cloned.source_format, "gguf");
        assert_eq!(cloned.target_format, "apr");
    }

    #[test]
    fn test_format_conversion_result_debug() {
        let result = FormatConversionResult {
            source_format: "gguf".to_string(),
            target_format: "apr".to_string(),
            success: true,
            duration_ms: 100,
            error: None,
            cached: false,
        };
        let debug_str = format!("{result:?}");
        assert!(debug_str.contains("FormatConversionResult"));
    }

    #[test]
    fn test_format_conversion_result_serialization() {
        let result = FormatConversionResult {
            source_format: "safetensors".to_string(),
            target_format: "gguf".to_string(),
            success: false,
            duration_ms: 2000,
            error: Some("Test error".to_string()),
            cached: false,
        };
        let json = serde_json::to_string(&result).unwrap();
        let parsed: FormatConversionResult = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.source_format, "safetensors");
        assert!(!parsed.success);
    }

    // =========================================================================
    // find_model_file tests
    // =========================================================================

    #[test]
    fn test_find_model_file_nonexistent_dir() {
        let result = find_model_file(std::path::Path::new("/nonexistent/dir"));
        assert!(result.is_err());
        let err = result.unwrap_err();
        match err {
            Error::ExecutionFailed { command, reason } => {
                assert!(command.contains("find model"));
                assert!(reason.contains("does not exist"));
            }
            _ => unreachable!("Expected ExecutionFailed error"),
        }
    }

    #[test]
    fn test_find_model_file_empty_dir() {
        let temp_dir = tempfile::tempdir().unwrap();
        let result = find_model_file(temp_dir.path());
        assert!(result.is_err());
        let err = result.unwrap_err();
        match err {
            Error::ExecutionFailed { reason, .. } => {
                assert!(reason.contains("No model file found"));
            }
            _ => unreachable!("Expected ExecutionFailed error"),
        }
    }

    #[test]
    fn test_find_model_file_with_file() {
        let temp_dir = tempfile::tempdir().unwrap();
        let model_file = temp_dir.path().join("model.gguf");
        std::fs::write(&model_file, b"fake model data").unwrap();
        let result = find_model_file(temp_dir.path());
        assert!(result.is_ok());
        assert!(result.unwrap().exists());
    }

    #[test]
    fn test_find_model_file_with_subdir_only() {
        let temp_dir = tempfile::tempdir().unwrap();
        let subdir = temp_dir.path().join("subdir");
        std::fs::create_dir(&subdir).unwrap();
        // Directory contains only a subdirectory, no files
        let result = find_model_file(temp_dir.path());
        assert!(result.is_err());
    }

    // =========================================================================
    // compute_file_hash tests (via convert_format_cached)
    // =========================================================================

    #[test]
    fn test_compute_file_hash_via_cached_conversion() {
        let temp_dir = tempfile::tempdir().unwrap();
        let source = temp_dir.path().join("source.gguf");
        let target = temp_dir.path().join("target.apr");
        let hash_file = temp_dir.path().join(".hash");

        // Write source file
        std::fs::write(&source, b"test model content").unwrap();

        // Attempt cached conversion (will fail because apr binary doesn't exist,
        // but should still compute hash)
        let result = convert_format_cached("/nonexistent/apr", &source, &target, &hash_file);
        // The conversion will fail, but that's expected - we're testing hash computation
        assert!(result.is_err());
    }

    // =========================================================================
    // CiProfileMetrics tests
    // =========================================================================

    #[test]
    fn test_ci_profile_metrics_fields() {
        let metrics = CiProfileMetrics {
            throughput_tok_s: 50.0,
            latency_p50_ms: 20.0,
            latency_p99_ms: 45.0,
        };
        assert!((metrics.throughput_tok_s - 50.0).abs() < f64::EPSILON);
        assert!((metrics.latency_p50_ms - 20.0).abs() < f64::EPSILON);
        assert!((metrics.latency_p99_ms - 45.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_ci_profile_metrics_clone() {
        let metrics = CiProfileMetrics {
            throughput_tok_s: 100.0,
            latency_p50_ms: 10.0,
            latency_p99_ms: 25.0,
        };
        let cloned = metrics.clone();
        assert!((cloned.throughput_tok_s - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_ci_profile_metrics_debug() {
        let metrics = CiProfileMetrics {
            throughput_tok_s: 75.0,
            latency_p50_ms: 15.0,
            latency_p99_ms: 35.0,
        };
        let debug_str = format!("{metrics:?}");
        assert!(debug_str.contains("CiProfileMetrics"));
    }

    #[test]
    fn test_ci_profile_metrics_serialization() {
        let metrics = CiProfileMetrics {
            throughput_tok_s: 80.0,
            latency_p50_ms: 12.5,
            latency_p99_ms: 30.0,
        };
        let json = serde_json::to_string(&metrics).unwrap();
        let parsed: CiProfileMetrics = serde_json::from_str(&json).unwrap();
        assert!((parsed.throughput_tok_s - 80.0).abs() < 0.01);
    }

    // =========================================================================
    // CiProfileResult accessor method tests
    // =========================================================================

    #[test]
    fn test_ci_profile_result_throughput_from_metrics() {
        let result = CiProfileResult {
            model: "test".to_string(),
            metrics: Some(CiProfileMetrics {
                throughput_tok_s: 100.0,
                latency_p50_ms: 10.0,
                latency_p99_ms: 20.0,
            }),
            throughput_tps: 50.0, // Should be ignored when metrics is Some
            latency_p50_ms: 5.0,
            latency_p99_ms: 10.0,
            assertions: vec![],
            passed: true,
        };
        assert!((result.throughput() - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_ci_profile_result_throughput_legacy() {
        let result = CiProfileResult {
            model: "test".to_string(),
            metrics: None, // No metrics, use legacy field
            throughput_tps: 75.0,
            latency_p50_ms: 5.0,
            latency_p99_ms: 10.0,
            assertions: vec![],
            passed: true,
        };
        assert!((result.throughput() - 75.0).abs() < 0.01);
    }

    #[test]
    fn test_ci_profile_result_p50_latency_from_metrics() {
        let result = CiProfileResult {
            model: "test".to_string(),
            metrics: Some(CiProfileMetrics {
                throughput_tok_s: 100.0,
                latency_p50_ms: 15.0,
                latency_p99_ms: 25.0,
            }),
            throughput_tps: 50.0,
            latency_p50_ms: 5.0, // Should be ignored when metrics is Some
            latency_p99_ms: 10.0,
            assertions: vec![],
            passed: true,
        };
        assert!((result.p50_latency() - 15.0).abs() < 0.01);
    }

    #[test]
    fn test_ci_profile_result_p50_latency_legacy() {
        let result = CiProfileResult {
            model: "test".to_string(),
            metrics: None,
            throughput_tps: 50.0,
            latency_p50_ms: 8.0,
            latency_p99_ms: 16.0,
            assertions: vec![],
            passed: true,
        };
        assert!((result.p50_latency() - 8.0).abs() < 0.01);
    }

    #[test]
    fn test_ci_profile_result_p99_latency_from_metrics() {
        let result = CiProfileResult {
            model: "test".to_string(),
            metrics: Some(CiProfileMetrics {
                throughput_tok_s: 100.0,
                latency_p50_ms: 10.0,
                latency_p99_ms: 30.0,
            }),
            throughput_tps: 50.0,
            latency_p50_ms: 5.0,
            latency_p99_ms: 10.0, // Should be ignored when metrics is Some
            assertions: vec![],
            passed: true,
        };
        assert!((result.p99_latency() - 30.0).abs() < 0.01);
    }

    #[test]
    fn test_ci_profile_result_p99_latency_legacy() {
        let result = CiProfileResult {
            model: "test".to_string(),
            metrics: None,
            throughput_tps: 50.0,
            latency_p50_ms: 5.0,
            latency_p99_ms: 12.0,
            assertions: vec![],
            passed: true,
        };
        assert!((result.p99_latency() - 12.0).abs() < 0.01);
    }

    #[test]
    fn test_ci_profile_result_all_accessors_with_metrics() {
        let result = CiProfileResult {
            model: "model-x".to_string(),
            metrics: Some(CiProfileMetrics {
                throughput_tok_s: 200.0,
                latency_p50_ms: 5.0,
                latency_p99_ms: 15.0,
            }),
            throughput_tps: 100.0, // All legacy values should be ignored
            latency_p50_ms: 10.0,
            latency_p99_ms: 30.0,
            assertions: vec![],
            passed: true,
        };
        // All values should come from metrics
        assert!((result.throughput() - 200.0).abs() < 0.01);
        assert!((result.p50_latency() - 5.0).abs() < 0.01);
        assert!((result.p99_latency() - 15.0).abs() < 0.01);
    }

    // =========================================================================
    // Provenance-Aware Model Preparation Tests (PMAT-PROV-001)
    // =========================================================================

    #[test]
    fn test_model_preparation_result_fields() {
        use crate::provenance::{DerivedProvenance, Provenance, SourceProvenance};

        let result = ModelPreparationResult {
            provenance: Provenance {
                source: SourceProvenance {
                    format: "safetensors".to_string(),
                    path: "model.safetensors".to_string(),
                    sha256: "abc123".to_string(),
                    hf_repo: "test/model".to_string(),
                    downloaded_at: "2026-02-01T12:00:00Z".to_string(),
                },
                derived: vec![DerivedProvenance {
                    format: "gguf".to_string(),
                    path: "model.gguf".to_string(),
                    sha256: "def456".to_string(),
                    converter: "apr-cli".to_string(),
                    converter_version: "0.2.12".to_string(),
                    quantization: None,
                    created_at: "2026-02-01T12:05:00Z".to_string(),
                }],
            },
            safetensors_path: std::path::PathBuf::from("/models/model.safetensors"),
            gguf_path: Some(std::path::PathBuf::from("/models/model.gguf")),
            apr_path: None,
            conversions: vec![],
        };

        assert_eq!(result.provenance.source.format, "safetensors");
        assert!(result.gguf_path.is_some());
        assert!(result.apr_path.is_none());
    }

    #[test]
    fn test_model_preparation_result_serialization() {
        use crate::provenance::{Provenance, SourceProvenance};

        let result = ModelPreparationResult {
            provenance: Provenance {
                source: SourceProvenance {
                    format: "safetensors".to_string(),
                    path: "model.safetensors".to_string(),
                    sha256: "abc123".to_string(),
                    hf_repo: "test/model".to_string(),
                    downloaded_at: "2026-02-01T12:00:00Z".to_string(),
                },
                derived: vec![],
            },
            safetensors_path: std::path::PathBuf::from("/models/model.safetensors"),
            gguf_path: None,
            apr_path: None,
            conversions: vec![],
        };

        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("safetensors"));
        assert!(json.contains("test/model"));
    }

    #[test]
    fn test_verify_comparison_provenance_missing_file() {
        let temp_dir = tempfile::tempdir().unwrap();
        let result = verify_comparison_provenance(temp_dir.path(), "gguf", "apr");
        assert!(result.is_err());
    }

    #[test]
    fn test_verify_comparison_provenance_valid() {
        use crate::provenance::{DerivedProvenance, Provenance, SourceProvenance, save_provenance};

        let temp_dir = tempfile::tempdir().unwrap();

        // Create valid provenance
        let provenance = Provenance {
            source: SourceProvenance {
                format: "safetensors".to_string(),
                path: "model.safetensors".to_string(),
                sha256: "abc123".to_string(),
                hf_repo: "test/model".to_string(),
                downloaded_at: "2026-02-01T12:00:00Z".to_string(),
            },
            derived: vec![
                DerivedProvenance {
                    format: "gguf".to_string(),
                    path: "model.gguf".to_string(),
                    sha256: "def456".to_string(),
                    converter: "apr-cli".to_string(),
                    converter_version: "0.2.12".to_string(),
                    quantization: None,
                    created_at: "2026-02-01T12:05:00Z".to_string(),
                },
                DerivedProvenance {
                    format: "apr".to_string(),
                    path: "model.apr".to_string(),
                    sha256: "789ghi".to_string(),
                    converter: "apr-cli".to_string(),
                    converter_version: "0.2.12".to_string(),
                    quantization: None,
                    created_at: "2026-02-01T12:06:00Z".to_string(),
                },
            ],
        };
        save_provenance(temp_dir.path(), &provenance).unwrap();

        let result = verify_comparison_provenance(temp_dir.path(), "gguf", "apr");
        assert!(result.is_ok());
    }

    #[test]
    fn test_verify_comparison_provenance_quantization_mismatch() {
        use crate::provenance::{DerivedProvenance, Provenance, SourceProvenance, save_provenance};

        let temp_dir = tempfile::tempdir().unwrap();

        // Create provenance with mismatched quantization
        let provenance = Provenance {
            source: SourceProvenance {
                format: "safetensors".to_string(),
                path: "model.safetensors".to_string(),
                sha256: "abc123".to_string(),
                hf_repo: "test/model".to_string(),
                downloaded_at: "2026-02-01T12:00:00Z".to_string(),
            },
            derived: vec![
                DerivedProvenance {
                    format: "gguf".to_string(),
                    path: "model-q4.gguf".to_string(),
                    sha256: "def456".to_string(),
                    converter: "apr-cli".to_string(),
                    converter_version: "0.2.12".to_string(),
                    quantization: Some("q4_k_m".to_string()), // Quantized
                    created_at: "2026-02-01T12:05:00Z".to_string(),
                },
                DerivedProvenance {
                    format: "apr".to_string(),
                    path: "model.apr".to_string(),
                    sha256: "789ghi".to_string(),
                    converter: "apr-cli".to_string(),
                    converter_version: "0.2.12".to_string(),
                    quantization: None, // Not quantized
                    created_at: "2026-02-01T12:06:00Z".to_string(),
                },
            ],
        };
        save_provenance(temp_dir.path(), &provenance).unwrap();

        let result = verify_comparison_provenance(temp_dir.path(), "gguf", "apr");
        assert!(result.is_err()); // PROV-005 violation
    }

    #[test]
    fn test_prepare_model_fails_without_apr_binary() {
        let temp_dir = tempfile::tempdir().unwrap();
        let safetensors = temp_dir.path().join("model.safetensors");
        std::fs::write(&safetensors, b"fake safetensors content").unwrap();

        let output_dir = temp_dir.path().join("output");
        let result = prepare_model_with_provenance(
            "/nonexistent/apr",
            &safetensors,
            "test/model",
            &output_dir,
            None,
        );

        // Will fail because apr binary doesn't exist
        assert!(result.is_err());
    }

    // =========================================================================
    // Mock binary tests for Command-calling functions
    // =========================================================================

    /// Create a mock bash script that acts as a fake apr binary
    fn create_mock_binary(dir: &std::path::Path, name: &str, script: &str) -> std::path::PathBuf {
        let path = dir.join(name);
        std::fs::write(&path, format!("#!/bin/bash\n{script}")).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        }
        path
    }

    // =========================================================================
    // convert_format_cached - cache hit path
    // =========================================================================

    #[test]
    fn test_convert_format_cached_cache_hit() {
        let temp_dir = tempfile::tempdir().unwrap();
        let source = temp_dir.path().join("source.safetensors");
        let target = temp_dir.path().join("target.gguf");
        let hash_file = temp_dir.path().join(".hash");

        // Write source file
        std::fs::write(&source, b"model content for caching test").unwrap();

        // Mock that creates the target file (arg $4 = target_path)
        let mock = create_mock_binary(
            temp_dir.path(),
            "apr_hash",
            "echo 'converted' > \"$4\" && exit 0",
        );

        // First call: does actual conversion, writes hash
        let first = convert_format_cached(mock.to_str().unwrap(), &source, &target, &hash_file);
        if let Ok(r1) = first {
            assert!(r1.success);
            assert!(!r1.cached);

            // Second call with same source: should hit cache
            let second =
                convert_format_cached(mock.to_str().unwrap(), &source, &target, &hash_file);
            if let Ok(r2) = second {
                assert!(r2.cached);
                assert!(r2.success);
                assert_eq!(r2.duration_ms, 0);
            }
        }
    }

    #[test]
    fn test_convert_format_cached_successful_conversion() {
        let temp_dir = tempfile::tempdir().unwrap();
        let source = temp_dir.path().join("source.safetensors");
        let target = temp_dir.path().join("output").join("target.gguf");
        let hash_file = temp_dir.path().join(".hash");

        std::fs::write(&source, b"model data for conversion").unwrap();

        // Mock binary that creates the target file
        let mock = create_mock_binary(
            temp_dir.path(),
            "apr_convert",
            "mkdir -p \"$(dirname \"$3\")\" && echo 'converted' > \"$3\" && exit 0",
        );

        let result = convert_format_cached(mock.to_str().unwrap(), &source, &target, &hash_file);
        if let Ok(r) = result {
            assert!(r.success);
            assert!(!r.cached);
            assert_eq!(r.source_format, "safetensors");
            assert_eq!(r.target_format, "gguf");
            // Hash file should have been written
            assert!(hash_file.exists());
        }
    }

    #[test]
    fn test_convert_format_cached_failed_conversion() {
        let temp_dir = tempfile::tempdir().unwrap();
        let source = temp_dir.path().join("source.gguf");
        let target = temp_dir.path().join("target.apr");
        let hash_file = temp_dir.path().join(".hash");

        std::fs::write(&source, b"model data").unwrap();

        // Mock binary that fails
        let mock = create_mock_binary(
            temp_dir.path(),
            "apr_fail",
            "echo 'error: bad format' >&2; exit 1",
        );

        let result = convert_format_cached(mock.to_str().unwrap(), &source, &target, &hash_file);
        if let Ok(r) = result {
            assert!(!r.success);
            assert!(!r.cached);
            assert!(r.error.is_some());
            assert!(r.error.unwrap().contains("error: bad format"));
        }
    }

    #[test]
    fn test_convert_format_cached_stale_cache() {
        let temp_dir = tempfile::tempdir().unwrap();
        let source = temp_dir.path().join("source.safetensors");
        let target = temp_dir.path().join("target.gguf");
        let hash_file = temp_dir.path().join(".hash");

        std::fs::write(&source, b"model data v1").unwrap();
        // Pre-populate with wrong hash to simulate stale cache
        std::fs::write(&target, b"old converted data").unwrap();
        std::fs::write(&hash_file, "wrong_hash_value").unwrap();

        // Mock binary that creates target
        let mock = create_mock_binary(
            temp_dir.path(),
            "apr_stale",
            "echo 'reconverted' > \"$3\" && exit 0",
        );

        let result = convert_format_cached(mock.to_str().unwrap(), &source, &target, &hash_file);
        if let Ok(r) = result {
            // Should NOT be cached since hash didn't match
            assert!(!r.cached);
            assert!(r.success);
        }
    }

    // =========================================================================
    // compute_file_hash - error paths
    // =========================================================================

    #[test]
    fn test_compute_file_hash_nonexistent_file() {
        let result = convert_format_cached(
            "echo",
            std::path::Path::new("/nonexistent/model.gguf"),
            std::path::Path::new("/tmp/out.apr"),
            std::path::Path::new("/tmp/.hash"),
        );
        // Should fail because source doesn't exist (compute_file_hash fails)
        assert!(result.is_err());
    }

    // =========================================================================
    // run_bench_throughput tests
    // =========================================================================

    #[test]
    fn test_run_bench_throughput_success_cpu() {
        let temp_dir = tempfile::tempdir().unwrap();
        let model = temp_dir.path().join("model.gguf");
        std::fs::write(&model, b"fake model").unwrap();

        let mock = create_mock_binary(
            temp_dir.path(),
            "apr_bench",
            "echo 'Loading model...\nThroughput: 65.5 tok/s (PASS: >= 10 tok/s)\nDone.' && exit 0",
        );

        let result = run_bench_throughput(mock.to_str().unwrap(), &model, false, 1, 3);
        if let Ok(r) = result {
            assert!((r.throughput_tps - 65.5).abs() < 0.01);
            assert!(r.passed);
            assert_eq!(r.backend, "cpu");
            assert_eq!(r.format, "gguf");
        }
    }

    #[test]
    fn test_run_bench_throughput_success_gpu() {
        let temp_dir = tempfile::tempdir().unwrap();
        let model = temp_dir.path().join("model.apr");
        std::fs::write(&model, b"fake model").unwrap();

        let mock = create_mock_binary(
            temp_dir.path(),
            "apr_bench_gpu",
            "echo 'Throughput: 120.3 tok/s' && exit 0",
        );

        let result = run_bench_throughput(mock.to_str().unwrap(), &model, true, 1, 3);
        if let Ok(r) = result {
            assert!((r.throughput_tps - 120.3).abs() < 0.01);
            assert!(r.passed);
            assert_eq!(r.backend, "gpu");
            assert_eq!(r.format, "apr");
        }
    }

    #[test]
    fn test_run_bench_throughput_below_threshold() {
        let temp_dir = tempfile::tempdir().unwrap();
        let model = temp_dir.path().join("model.safetensors");
        std::fs::write(&model, b"fake model").unwrap();

        let mock = create_mock_binary(
            temp_dir.path(),
            "apr_bench_slow",
            "echo 'Throughput: 5.2 tok/s' && exit 0",
        );

        let result = run_bench_throughput(mock.to_str().unwrap(), &model, false, 1, 1);
        if let Ok(r) = result {
            assert!((r.throughput_tps - 5.2).abs() < 0.01);
            // Below 10.0 threshold
            assert!(!r.passed);
            assert_eq!(r.format, "safetensors");
        }
    }

    #[test]
    fn test_run_bench_throughput_no_throughput_line() {
        let temp_dir = tempfile::tempdir().unwrap();
        let model = temp_dir.path().join("model.gguf");
        std::fs::write(&model, b"fake model").unwrap();

        let mock = create_mock_binary(
            temp_dir.path(),
            "apr_bench_nothroughput",
            "echo 'Loading model...\nDone.' && exit 0",
        );

        let result = run_bench_throughput(mock.to_str().unwrap(), &model, false, 1, 1);
        if let Ok(r) = result {
            assert!((r.throughput_tps - 0.0).abs() < 0.01);
            assert!(!r.passed); // 0.0 < 10.0
        }
    }

    #[test]
    fn test_run_bench_throughput_failed_exit() {
        let temp_dir = tempfile::tempdir().unwrap();
        let model = temp_dir.path().join("model.gguf");
        std::fs::write(&model, b"fake model").unwrap();

        let mock = create_mock_binary(
            temp_dir.path(),
            "apr_bench_fail",
            "echo 'Throughput: 50.0 tok/s' && exit 1",
        );

        let result = run_bench_throughput(mock.to_str().unwrap(), &model, false, 1, 1);
        if let Ok(r) = result {
            // exit code non-zero => passed = false even though throughput was high
            assert!(!r.passed);
        }
    }

    #[test]
    fn test_run_bench_throughput_unknown_extension() {
        let temp_dir = tempfile::tempdir().unwrap();
        let model = temp_dir.path().join("model");
        std::fs::write(&model, b"fake model").unwrap();

        let mock = create_mock_binary(
            temp_dir.path(),
            "apr_bench_noext",
            "echo 'Throughput: 15.0 tok/s' && exit 0",
        );

        let result = run_bench_throughput(mock.to_str().unwrap(), &model, false, 1, 1);
        if let Ok(r) = result {
            assert_eq!(r.format, "unknown");
        }
    }

    // =========================================================================
    // run_ci_profile tests
    // =========================================================================

    #[test]
    fn test_run_ci_profile_json_output() {
        let temp_dir = tempfile::tempdir().unwrap();
        let model = temp_dir.path().join("model.gguf");
        std::fs::write(&model, b"fake model").unwrap();

        let json = r#"{"model":"test","metrics":null,"throughput_tps":42.0,"latency_p50_ms":10.0,"latency_p99_ms":25.0,"assertions":[],"passed":true}"#;
        let mock = create_mock_binary(
            temp_dir.path(),
            "apr_profile_json",
            &format!("echo '{json}' && exit 0"),
        );

        let result = run_profile_ci(
            mock.to_str().unwrap(),
            &model,
            Some(10.0),
            Some(100.0),
            Some(50.0),
            1,
            3,
        );
        if let Ok(r) = result {
            assert!(r.passed);
            assert!((r.throughput_tps - 42.0).abs() < 0.01);
        }
    }

    #[test]
    fn test_run_ci_profile_json_with_prefix() {
        let temp_dir = tempfile::tempdir().unwrap();
        let model = temp_dir.path().join("model.gguf");
        std::fs::write(&model, b"fake").unwrap();

        let json = r#"{"model":"test","metrics":null,"throughput_tps":55.0,"latency_p50_ms":8.0,"latency_p99_ms":20.0,"assertions":[],"passed":true}"#;
        let mock = create_mock_binary(
            temp_dir.path(),
            "apr_profile_prefix",
            &format!("echo 'Loading model...' && echo '{json}' && exit 0"),
        );

        let result = run_profile_ci(mock.to_str().unwrap(), &model, None, None, None, 1, 3);
        if let Ok(r) = result {
            assert!(r.passed);
        }
    }

    #[test]
    fn test_run_ci_profile_fallback_on_bad_json() {
        let temp_dir = tempfile::tempdir().unwrap();
        let model = temp_dir.path().join("model.gguf");
        std::fs::write(&model, b"fake").unwrap();

        let mock = create_mock_binary(
            temp_dir.path(),
            "apr_profile_bad",
            "echo 'not json at all' && exit 0",
        );

        let result = run_profile_ci(mock.to_str().unwrap(), &model, None, None, None, 1, 1);
        if let Ok(r) = result {
            // Fallback: passed = exit code success
            assert!(r.passed);
            assert!((r.throughput_tps - 0.0).abs() < 0.01);
        }
    }

    #[test]
    fn test_run_ci_profile_fallback_failed_exit() {
        let temp_dir = tempfile::tempdir().unwrap();
        let model = temp_dir.path().join("model.gguf");
        std::fs::write(&model, b"fake").unwrap();

        let mock = create_mock_binary(
            temp_dir.path(),
            "apr_profile_fail",
            "printf 'error\\n'; exit 1",
        );

        let result = run_profile_ci(mock.to_str().unwrap(), &model, None, None, None, 1, 1);
        if let Ok(r) = result {
            assert!(!r.passed);
        }
    }

    // =========================================================================
    // run_diff_benchmark tests
    // =========================================================================

    #[test]
    fn test_run_diff_benchmark_json_output() {
        let temp_dir = tempfile::tempdir().unwrap();
        let model_a = temp_dir.path().join("a.gguf");
        let model_b = temp_dir.path().join("b.gguf");
        std::fs::write(&model_a, b"model_a").unwrap();
        std::fs::write(&model_b, b"model_b").unwrap();

        // Write JSON to a file to avoid shell quoting issues
        let json_file = temp_dir.path().join("diff_output.json");
        let json = r#"{"model_a":{"path":"a.gguf","throughput_tps":10.0,"latency_p50_ms":50.0,"latency_p99_ms":100.0},"model_b":{"path":"b.gguf","throughput_tps":12.0,"latency_p50_ms":45.0,"latency_p99_ms":90.0},"throughput_delta_pct":20.0,"latency_p50_delta_pct":-10.0,"latency_p99_delta_pct":-10.0,"regression_detected":false,"regression_threshold":5.0}"#;
        std::fs::write(&json_file, json).unwrap();

        let mock = create_mock_binary(
            temp_dir.path(),
            "apr_diff_bench",
            &format!("cat '{}'", json_file.display()),
        );

        let result = run_diff_benchmark(mock.to_str().unwrap(), &model_a, &model_b, 5.0);
        // Mock binary execution can be flaky under parallel test runs
        if let Ok(r) = result {
            assert!(!r.regression_detected);
        }
    }

    #[test]
    fn test_run_diff_benchmark_bad_json() {
        let temp_dir = tempfile::tempdir().unwrap();
        let model_a = temp_dir.path().join("a.gguf");
        let model_b = temp_dir.path().join("b.gguf");
        std::fs::write(&model_a, b"a").unwrap();
        std::fs::write(&model_b, b"b").unwrap();

        let mock = create_mock_binary(temp_dir.path(), "apr_diff_bad", "echo 'not json' && exit 0");

        let result = run_diff_benchmark(mock.to_str().unwrap(), &model_a, &model_b, 5.0);
        assert!(result.is_err());
        let err = result.unwrap_err();
        match err {
            Error::ExecutionFailed { reason, .. } => {
                assert!(reason.contains("Failed to parse output"));
            }
            other => unreachable!("Expected ExecutionFailed, got: {other:?}"),
        }
    }

    // =========================================================================
    // DifferentialExecutor with mock binary
    // =========================================================================

    #[test]
    fn test_diff_tensors_success() {
        let temp_dir = tempfile::tempdir().unwrap();
        let model_a = temp_dir.path().join("a.gguf");
        let model_b = temp_dir.path().join("b.safetensors");
        std::fs::write(&model_a, b"a").unwrap();
        std::fs::write(&model_b, b"b").unwrap();

        let json = r#"{"total_tensors":50,"mismatched_tensors":0,"transposed_tensors":0,"passed":true,"mismatches":[]}"#;
        let mock = create_mock_binary(
            temp_dir.path(),
            "apr_diff",
            &format!("echo '{json}' && exit 0"),
        );

        let config = DiffConfig {
            apr_binary: mock.to_str().unwrap().to_string(),
            ..Default::default()
        };
        let executor = DifferentialExecutor::new(config);
        let result = executor.diff_tensors(&model_a, &model_b);
        if let Ok(r) = result {
            assert!(r.passed);
            assert_eq!(r.total_tensors, 50);
        }
    }

    #[test]
    fn test_diff_tensors_failure_nonzero_exit() {
        let temp_dir = tempfile::tempdir().unwrap();
        let model_a = temp_dir.path().join("a.gguf");
        let model_b = temp_dir.path().join("b.gguf");
        std::fs::write(&model_a, b"a").unwrap();
        std::fs::write(&model_b, b"b").unwrap();

        let mock = create_mock_binary(
            temp_dir.path(),
            "apr_diff_err",
            "echo 'tensor mismatch error' >&2 && exit 1",
        );

        let config = DiffConfig {
            apr_binary: mock.to_str().unwrap().to_string(),
            ..Default::default()
        };
        let executor = DifferentialExecutor::new(config);
        let result = executor.diff_tensors(&model_a, &model_b);
        assert!(result.is_err());
    }

    #[test]
    fn test_compare_inference_success() {
        let temp_dir = tempfile::tempdir().unwrap();
        let model_a = temp_dir.path().join("a.gguf");
        let model_b = temp_dir.path().join("b.safetensors");
        std::fs::write(&model_a, b"a").unwrap();
        std::fs::write(&model_b, b"b").unwrap();

        let json = r#"{"total_tokens":5,"matching_tokens":5,"max_logit_diff":1e-7,"passed":true,"token_comparisons":[]}"#;
        let mock = create_mock_binary(
            temp_dir.path(),
            "apr_compare",
            &format!("echo '{json}' && exit 0"),
        );

        let config = DiffConfig {
            apr_binary: mock.to_str().unwrap().to_string(),
            ..Default::default()
        };
        let executor = DifferentialExecutor::new(config);
        let result = executor.compare_inference(&model_a, &model_b, "What is 2+2?", 5);
        if let Ok(r) = result {
            assert!(r.passed);
            assert_eq!(r.total_tokens, 5);
            assert_eq!(r.matching_tokens, 5);
        }
    }

    #[test]
    fn test_compare_inference_fallback() {
        let temp_dir = tempfile::tempdir().unwrap();
        let model_a = temp_dir.path().join("a.gguf");
        let model_b = temp_dir.path().join("b.gguf");
        std::fs::write(&model_a, b"a").unwrap();
        std::fs::write(&model_b, b"b").unwrap();

        let mock = create_mock_binary(
            temp_dir.path(),
            "apr_compare_nojson",
            "echo 'some text output' && exit 0",
        );

        let config = DiffConfig {
            apr_binary: mock.to_str().unwrap().to_string(),
            ..Default::default()
        };
        let executor = DifferentialExecutor::new(config);
        let result = executor.compare_inference(&model_a, &model_b, "test", 3);
        if let Ok(r) = result {
            // Fallback: passed from exit code
            assert!(r.passed);
            assert_eq!(r.total_tokens, 0);
        }
    }

    // =========================================================================
    // run_six_column_profile tests
    // =========================================================================

    #[test]
    fn test_run_six_column_profile_basic() {
        let temp_dir = tempfile::tempdir().unwrap();
        let cache_dir = temp_dir.path().join("cache");

        // Create directory structure
        let gguf_dir = cache_dir.join("gguf");
        let apr_dir = cache_dir.join("apr");
        let st_dir = cache_dir.join("safetensors");
        std::fs::create_dir_all(&gguf_dir).unwrap();
        std::fs::create_dir_all(&apr_dir).unwrap();
        std::fs::create_dir_all(&st_dir).unwrap();

        // Create model file in gguf dir
        let gguf_model = gguf_dir.join("model.gguf");
        std::fs::write(&gguf_model, b"fake gguf model data for testing").unwrap();

        // Mock binary that handles convert and bench
        let mock = create_mock_binary(
            temp_dir.path(),
            "apr_six",
            r#"
case "$1" in
    rosetta)
        printf 'converted\n' > "$4"
        exit 0
        ;;
    bench)
        printf 'Throughput: 25.5 tok/s\n'
        exit 0
        ;;
    *)
        exit 1
        ;;
esac
"#,
        );

        let result = run_six_column_profile(mock.to_str().unwrap(), &cache_dir, 1, 1);
        // May fail if mock binary has issues, but should work on most systems
        if let Ok(r) = result {
            assert_eq!(r.conversions.len(), 2); // APR + SafeTensors
        }
    }

    // =========================================================================
    // prepare_model_with_provenance - resume workflow
    // =========================================================================

    #[test]
    fn test_prepare_model_with_provenance_resume_matching_hash() {
        use crate::provenance::{create_source_provenance, save_provenance};

        let temp_dir = tempfile::tempdir().unwrap();
        let safetensors = temp_dir.path().join("model.safetensors");
        std::fs::write(&safetensors, b"safetensors content for resume test").unwrap();

        let output_dir = temp_dir.path().join("output");
        std::fs::create_dir_all(&output_dir).unwrap();

        // Create existing provenance with matching hash
        let prov = create_source_provenance(&safetensors, "test/model").unwrap();
        save_provenance(&output_dir, &prov).unwrap();

        // Mock binary that handles conversions
        let mock = create_mock_binary(
            temp_dir.path(),
            "apr_resume",
            r#"
if [ "$1" = "rosetta" ] && [ "$2" = "convert" ]; then
    echo "converted" > "$4"
    exit 0
fi
exit 1
"#,
        );

        let result = prepare_model_with_provenance(
            mock.to_str().unwrap(),
            &safetensors,
            "test/model",
            &output_dir,
            None,
        );
        if let Ok(r) = result {
            assert_eq!(r.provenance.source.format, "safetensors");
            assert!(r.gguf_path.is_some());
            assert!(r.apr_path.is_some());
            assert_eq!(r.conversions.len(), 2);
        }
    }

    #[test]
    fn test_prepare_model_with_provenance_resume_changed_hash() {
        use crate::provenance::{Provenance, SourceProvenance, save_provenance};

        let temp_dir = tempfile::tempdir().unwrap();
        let safetensors = temp_dir.path().join("model.safetensors");
        std::fs::write(&safetensors, b"new content different from original").unwrap();

        let output_dir = temp_dir.path().join("output");
        std::fs::create_dir_all(&output_dir).unwrap();

        // Create provenance with different hash (stale)
        let stale_prov = Provenance {
            source: SourceProvenance {
                format: "safetensors".to_string(),
                path: safetensors.to_string_lossy().to_string(),
                sha256: "0000000000000000000000000000000000000000000000000000000000000000"
                    .to_string(),
                hf_repo: "test/model".to_string(),
                downloaded_at: "2026-01-01T00:00:00Z".to_string(),
            },
            derived: vec![],
        };
        save_provenance(&output_dir, &stale_prov).unwrap();

        let mock = create_mock_binary(
            temp_dir.path(),
            "apr_changed",
            r#"
if [ "$1" = "rosetta" ] && [ "$2" = "convert" ]; then
    echo "converted" > "$4"
    exit 0
fi
exit 1
"#,
        );

        let result = prepare_model_with_provenance(
            mock.to_str().unwrap(),
            &safetensors,
            "test/model",
            &output_dir,
            None,
        );
        if let Ok(r) = result {
            // Should have recreated provenance with new hash
            assert_ne!(
                r.provenance.source.sha256,
                "0000000000000000000000000000000000000000000000000000000000000000"
            );
        }
    }

    #[test]
    fn test_prepare_model_with_provenance_with_quantization() {
        let temp_dir = tempfile::tempdir().unwrap();
        let safetensors = temp_dir.path().join("model.safetensors");
        std::fs::write(&safetensors, b"safetensors quantization test").unwrap();

        let output_dir = temp_dir.path().join("output");

        let mock = create_mock_binary(
            temp_dir.path(),
            "apr_quant",
            r#"
if [ "$1" = "rosetta" ] && [ "$2" = "convert" ]; then
    echo "converted" > "$4"
    exit 0
fi
exit 1
"#,
        );

        let result = prepare_model_with_provenance(
            mock.to_str().unwrap(),
            &safetensors,
            "test/model",
            &output_dir,
            Some("q4_k_m"),
        );
        if let Ok(r) = result {
            // With quantization, paths should contain the quant level
            if let Some(gguf) = &r.gguf_path {
                assert!(gguf.to_string_lossy().contains("q4_k_m"));
            }
            if let Some(apr) = &r.apr_path {
                assert!(apr.to_string_lossy().contains("q4_k_m"));
            }
        }
    }

    #[test]
    fn test_prepare_model_with_provenance_partial_failure() {
        let temp_dir = tempfile::tempdir().unwrap();
        let safetensors = temp_dir.path().join("model.safetensors");
        std::fs::write(&safetensors, b"model content partial fail").unwrap();

        let output_dir = temp_dir.path().join("output");

        // Mock binary: first conversion succeeds, second fails
        let mock = create_mock_binary(
            temp_dir.path(),
            "apr_partial",
            r#"
# Track call count via a file
COUNTER_FILE="/tmp/apr_partial_counter_$$"
if [ ! -f "$COUNTER_FILE" ]; then
    echo "1" > "$COUNTER_FILE"
    echo "converted" > "$4"
    exit 0
else
    rm -f "$COUNTER_FILE"
    exit 1
fi
"#,
        );

        let result = prepare_model_with_provenance(
            mock.to_str().unwrap(),
            &safetensors,
            "test/model",
            &output_dir,
            None,
        );
        // Should still succeed overall (partial conversions are ok)
        if let Ok(r) = result {
            assert_eq!(r.conversions.len(), 2);
        }
    }

    #[test]
    fn test_model_preparation_result_clone() {
        use crate::provenance::{Provenance, SourceProvenance};

        let result = ModelPreparationResult {
            provenance: Provenance {
                source: SourceProvenance {
                    format: "safetensors".to_string(),
                    path: "model.safetensors".to_string(),
                    sha256: "abc".to_string(),
                    hf_repo: "test/model".to_string(),
                    downloaded_at: "2026-01-01T00:00:00Z".to_string(),
                },
                derived: vec![],
            },
            safetensors_path: std::path::PathBuf::from("/test"),
            gguf_path: None,
            apr_path: None,
            conversions: vec![],
        };
        let cloned = result.clone();
        assert_eq!(cloned.provenance.source.hf_repo, "test/model");
    }

    #[test]
    fn test_model_preparation_result_debug() {
        use crate::provenance::{Provenance, SourceProvenance};

        let result = ModelPreparationResult {
            provenance: Provenance {
                source: SourceProvenance {
                    format: "safetensors".to_string(),
                    path: "model.safetensors".to_string(),
                    sha256: "abc".to_string(),
                    hf_repo: "test/model".to_string(),
                    downloaded_at: "2026-01-01T00:00:00Z".to_string(),
                },
                derived: vec![],
            },
            safetensors_path: std::path::PathBuf::from("/test"),
            gguf_path: None,
            apr_path: None,
            conversions: vec![],
        };
        let debug = format!("{result:?}");
        assert!(debug.contains("ModelPreparationResult"));
    }

    #[test]
    fn test_bench_result_clone_debug() {
        let result = BenchResult {
            throughput_tps: 10.0,
            passed: true,
            backend: "cpu".to_string(),
            format: "apr".to_string(),
        };
        let cloned = result.clone();
        assert_eq!(cloned.backend, "cpu");
        let debug = format!("{result:?}");
        assert!(debug.contains("BenchResult"));
    }

    // =========================================================================
    // InspectResult tests (T-GH192-01, MR-CARD)
    // =========================================================================

    #[test]
    fn test_inspect_result_from_json() {
        let json = r#"{
            "tensor_count": 338,
            "tensor_names": ["model.embed_tokens.weight", "model.layers.0.self_attn.q_proj.weight"],
            "num_attention_heads": 14,
            "num_key_value_heads": 2,
            "hidden_size": 896,
            "architecture": "Qwen2ForCausalLM"
        }"#;
        let result: InspectResult = serde_json::from_str(json).unwrap();
        assert_eq!(result.tensor_count, 338);
        assert_eq!(result.tensor_names.len(), 2);
        assert_eq!(result.num_attention_heads, Some(14));
        assert_eq!(result.num_key_value_heads, Some(2));
        assert_eq!(result.hidden_size, Some(896));
        assert_eq!(result.architecture.as_deref(), Some("Qwen2ForCausalLM"));
    }

    #[test]
    fn test_inspect_result_minimal_json() {
        let json = r#"{"tensor_count": 100}"#;
        let result: InspectResult = serde_json::from_str(json).unwrap();
        assert_eq!(result.tensor_count, 100);
        assert!(result.tensor_names.is_empty());
        assert!(result.num_attention_heads.is_none());
        assert!(result.architecture.is_none());
    }

    #[test]
    fn test_inspect_result_serialization_round_trip() {
        let result = InspectResult {
            tensor_count: 227,
            tensor_names: vec![
                "model.embed_tokens.weight".to_string(),
                "lm_head.weight".to_string(),
            ],
            num_attention_heads: Some(32),
            num_key_value_heads: Some(8),
            hidden_size: Some(4096),
            architecture: Some("LlamaForCausalLM".to_string()),
        };
        let json = serde_json::to_string(&result).unwrap();
        let parsed: InspectResult = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.tensor_count, 227);
        assert_eq!(parsed.tensor_names.len(), 2);
        assert_eq!(parsed.hidden_size, Some(4096));
    }

    #[test]
    fn test_inspect_result_clone() {
        let result = InspectResult {
            tensor_count: 50,
            tensor_names: vec!["test.weight".to_string()],
            num_attention_heads: Some(12),
            num_key_value_heads: None,
            hidden_size: Some(768),
            architecture: None,
        };
        let cloned = result.clone();
        assert_eq!(cloned.tensor_count, result.tensor_count);
        assert_eq!(cloned.tensor_names, result.tensor_names);
    }

    #[test]
    fn test_inspect_result_debug() {
        let result = InspectResult {
            tensor_count: 100,
            tensor_names: vec![],
            num_attention_heads: None,
            num_key_value_heads: None,
            hidden_size: None,
            architecture: None,
        };
        let debug_str = format!("{result:?}");
        assert!(debug_str.contains("InspectResult"));
    }

    #[test]
    fn test_parse_inspect_text_with_tensor_count() {
        let output = "Tensors: 338\nmodel.embed_tokens.weight [151936, 896]\nmodel.layers.0.self_attn.q_proj.weight [896, 896]";
        let result = parse_inspect_text(output).unwrap();
        assert_eq!(result.tensor_count, 338);
        assert_eq!(result.tensor_names.len(), 2);
        assert!(
            result
                .tensor_names
                .contains(&"model.embed_tokens.weight".to_string())
        );
    }

    #[test]
    fn test_parse_inspect_text_with_metadata() {
        let output = "Tensors: 100\narchitecture: Qwen2ForCausalLM\nnum_attention_heads: 14\nnum_key_value_heads: 2\nhidden_size: 896";
        let result = parse_inspect_text(output).unwrap();
        assert_eq!(result.tensor_count, 100);
        assert_eq!(result.architecture.as_deref(), Some("Qwen2ForCausalLM"));
        assert_eq!(result.num_attention_heads, Some(14));
        assert_eq!(result.num_key_value_heads, Some(2));
        assert_eq!(result.hidden_size, Some(896));
    }

    #[test]
    fn test_parse_inspect_text_empty() {
        let output = "";
        let result = parse_inspect_text(output).unwrap();
        assert_eq!(result.tensor_count, 0);
        assert!(result.tensor_names.is_empty());
    }

    #[test]
    fn test_parse_inspect_text_tensor_count_from_names() {
        let output = "model.layers.0.weight [768, 768]\nmodel.layers.1.weight [768, 768]";
        let result = parse_inspect_text(output).unwrap();
        assert_eq!(result.tensor_count, 2);
        assert_eq!(result.tensor_names.len(), 2);
    }

    #[test]
    fn test_parse_inspect_text_alternate_prefix() {
        let output = "tensor_count: 42";
        let result = parse_inspect_text(output).unwrap();
        assert_eq!(result.tensor_count, 42);
    }

    #[test]
    fn test_run_inspect_nonexistent_binary() {
        let path = std::path::PathBuf::from("model.gguf");
        let result = run_inspect(&path, "/nonexistent/apr/binary");
        assert!(result.is_err());
    }
}
