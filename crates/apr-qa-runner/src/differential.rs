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
    // Retry on ETXTBSY (os error 26): a transient condition on Linux where
    // fork() inherits write fds from other threads, causing execve() to fail.
    let output = {
        let mut attempts = 0;
        loop {
            match Command::new(apr_binary)
                .arg("rosetta")
                .arg("inspect")
                .arg(model_path)
                .arg("--json")
                .output()
            {
                Ok(output) => break output,
                Err(e) if e.raw_os_error() == Some(26) && attempts < 3 => {
                    attempts += 1;
                    std::thread::sleep(std::time::Duration::from_millis(10));
                }
                Err(e) => {
                    return Err(Error::ExecutionFailed {
                        command: "apr rosetta inspect --json".to_string(),
                        reason: e.to_string(),
                    });
                }
            }
        }
    };

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
    // Retry on ETXTBSY (os error 26) — transient fork/exec race on Linux
    let output = {
        let mut attempts = 0;
        loop {
            match Command::new(apr_binary)
                .arg("profile")
                .arg(model_a)
                .arg(model_b)
                .arg("--diff-benchmark")
                .arg("--regression-threshold")
                .arg(regression_threshold.to_string())
                .arg("--json")
                .output()
            {
                Ok(output) => break output,
                Err(e) if e.raw_os_error() == Some(26) && attempts < 3 => {
                    attempts += 1;
                    std::thread::sleep(std::time::Duration::from_millis(10));
                }
                Err(e) => {
                    return Err(Error::ExecutionFailed {
                        command: "apr profile --diff-benchmark".to_string(),
                        reason: e.to_string(),
                    });
                }
            }
        }
    };

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
#[path = "differential_tests.rs"]
mod tests;
