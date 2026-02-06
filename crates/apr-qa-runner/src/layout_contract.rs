//! Tensor Layout Contract Validation (Issue #4)
//!
//! Implements automated validation against aprender's tensor-layout-v1.yaml contract.
//! This contract is THE SOURCE OF TRUTH for GGUF/SafeTensors→APR tensor conversion.
//!
//! # Validation Rules
//!
//! - F-LAYOUT-CONTRACT-001: All 2D weights are transposed
//! - F-LAYOUT-CONTRACT-002: lm_head shape matches kernel expectation (CRITICAL)
//! - F-LAYOUT-CONTRACT-003: 1D tensors unchanged
//! - F-LAYOUT-CONTRACT-004: Byte size matches kernel expectation
//! - F-LAYOUT-CONTRACT-005: No garbage output from lm_head
//!
//! # References
//!
//! - Contract file: `../aprender/contracts/tensor-layout-v1.yaml`
//! - Spec: Section E.8 of qwen2.5-coder-showcase-demo.md
//! - GH-202: lm_head shape bug that caused garbage output

// Debug format {:?} cannot be inlined
#![allow(clippy::uninlined_format_args)]

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use crate::error::{Error, Result};

/// Default path to the tensor layout contract relative to this repo.
pub const DEFAULT_CONTRACT_PATH: &str = "../aprender/contracts/tensor-layout-v1.yaml";

// ============================================================================
// Contract types (deserialized from YAML)
// ============================================================================

/// Top-level tensor layout contract.
#[derive(Debug, Clone, Deserialize)]
pub struct TensorLayoutContract {
    /// Contract metadata.
    pub metadata: ContractMetadata,

    /// Format conventions (gguf, apr, safetensors).
    pub formats: HashMap<String, FormatConvention>,

    /// Kernel convention defining weight shapes.
    pub kernel: KernelConvention,

    /// Per-tensor specifications.
    pub tensors: HashMap<String, TensorSpec>,

    /// Validation rules for automated testing.
    pub validation_rules: Vec<ValidationRule>,

    /// Semantic validation configuration.
    #[serde(default)]
    pub semantic_validation: Option<SemanticValidation>,
}

/// Contract metadata.
#[derive(Debug, Clone, Deserialize)]
pub struct ContractMetadata {
    /// Contract version.
    pub version: String,
    /// Creation date.
    pub created: String,
    /// Last update date.
    pub updated: String,
    /// Author.
    pub author: String,
    /// Description.
    pub description: String,
}

/// Format convention (layout and shape convention).
#[derive(Debug, Clone, Deserialize)]
pub struct FormatConvention {
    /// Layout: "row-major" or "column-major".
    pub layout: String,
    /// Shape convention description.
    pub shape_convention: String,
    /// Additional notes.
    #[serde(default)]
    pub note: Option<String>,
}

/// Kernel convention - source of truth for shapes.
#[derive(Debug, Clone, Deserialize)]
pub struct KernelConvention {
    /// Kernel function signature.
    pub signature: String,
    /// Weight shape convention.
    pub weight_shape: String,
    /// Computation description.
    pub computation: String,
    /// Byte calculation formula.
    pub byte_calculation: String,
    /// Block sizes for quantized types.
    pub block_sizes: HashMap<String, u32>,
    /// Elements per super-block.
    #[serde(rename = "QK_K")]
    pub qk_k: u32,
}

/// Per-tensor specification.
#[derive(Debug, Clone, Deserialize)]
pub struct TensorSpec {
    /// GGUF tensor name.
    pub gguf_name: String,
    /// APR tensor name.
    pub apr_name: String,
    /// GGUF shape as string (e.g., "[hidden, vocab]").
    pub gguf_shape: String,
    /// APR shape as string (e.g., "[vocab, hidden]").
    pub apr_shape: String,
    /// Whether tensor needs transposition.
    pub transpose: bool,
    /// Kernel that uses this tensor.
    pub kernel: String,
    /// Kernel output dimension expression.
    #[serde(default)]
    pub kernel_out_dim: Option<String>,
    /// Kernel input dimension expression.
    #[serde(default)]
    pub kernel_in_dim: Option<String>,
    /// Validation expression.
    #[serde(default)]
    pub validation: Option<String>,
    /// Whether this is a critical tensor.
    #[serde(default)]
    pub critical: bool,
    /// Additional notes.
    #[serde(default)]
    pub note: Option<String>,
}

/// Validation rule from contract.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ValidationRule {
    /// Rule ID (e.g., "F-LAYOUT-CONTRACT-001").
    pub id: String,
    /// Rule name.
    pub name: String,
    /// Rule description.
    pub description: String,
    /// Severity: P0, P1, P2.
    pub severity: String,
    /// Whether this is critical.
    #[serde(default)]
    pub critical: bool,
    /// Reference ticket.
    #[serde(default)]
    pub reference: Option<String>,
}

/// Semantic validation configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct SemanticValidation {
    /// Density validation config.
    #[serde(default)]
    pub density: Option<DensityConfig>,
    /// Numeric validation config.
    #[serde(default)]
    pub numeric: Option<NumericConfig>,
    /// Distribution validation config.
    #[serde(default)]
    pub distribution: Option<DistributionConfig>,
}

/// Density validation configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct DensityConfig {
    /// Max zero percentage for embeddings.
    pub embedding_max_zero_pct: f64,
    /// Max zero percentage for weights.
    pub weight_max_zero_pct: f64,
}

/// Numeric validation configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct NumericConfig {
    /// Allow NaN values.
    pub allow_nan: bool,
    /// Allow Inf values.
    pub allow_inf: bool,
}

/// Distribution validation configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct DistributionConfig {
    /// Minimum L2 norm.
    pub min_l2_norm: f64,
    /// Require variation in values.
    pub require_variation: bool,
}

// ============================================================================
// Contract loader
// ============================================================================

/// Load the tensor layout contract from the default path.
///
/// # Errors
///
/// Returns an error if the contract file cannot be read or parsed.
pub fn load_contract() -> Result<TensorLayoutContract> {
    load_contract_from(DEFAULT_CONTRACT_PATH)
}

/// Load the tensor layout contract from a specific path.
///
/// # Errors
///
/// Returns an error if the contract file cannot be read or parsed.
pub fn load_contract_from<P: AsRef<Path>>(path: P) -> Result<TensorLayoutContract> {
    let path = path.as_ref();
    let content = std::fs::read_to_string(path).map_err(|e| {
        Error::Execution(format!(
            "Failed to read tensor layout contract from {}: {e}",
            path.display()
        ))
    })?;

    serde_yaml::from_str(&content).map_err(|e| {
        Error::Execution(format!(
            "Failed to parse tensor layout contract from {}: {e}",
            path.display()
        ))
    })
}

// ============================================================================
// Validation result types
// ============================================================================

/// Result of validating a tensor against the contract.
#[derive(Debug, Clone, Serialize)]
pub struct TensorValidationResult {
    /// Tensor name.
    pub tensor_name: String,
    /// Rule ID that was checked.
    pub rule_id: String,
    /// Whether validation passed.
    pub passed: bool,
    /// Details about the validation.
    pub details: String,
    /// Expected value/shape.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expected: Option<String>,
    /// Actual value/shape.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub actual: Option<String>,
}

/// Result of validating an entire model against the contract.
#[derive(Debug, Clone, Serialize)]
pub struct ModelValidationResult {
    /// Model path that was validated.
    pub model_path: PathBuf,
    /// Overall pass/fail status.
    pub passed: bool,
    /// Number of rules checked.
    pub rules_checked: usize,
    /// Number of rules passed.
    pub rules_passed: usize,
    /// Number of rules failed.
    pub rules_failed: usize,
    /// Individual tensor validation results.
    pub tensor_results: Vec<TensorValidationResult>,
    /// Critical failures (P0 violations).
    pub critical_failures: Vec<String>,
}

// ============================================================================
// Validation functions
// ============================================================================

/// Maximum SafeTensors header size (10MB should cover any model)
const MAX_HEADER_SIZE: usize = 10 * 1024 * 1024;

/// Validate a model file against the tensor layout contract.
///
/// # Arguments
///
/// * `model_path` - Path to the APR model file or directory
/// * `contract` - The loaded tensor layout contract
///
/// # Returns
///
/// Validation result with per-tensor details.
///
/// # Errors
///
/// This function does not currently return errors; all validation failures
/// are reported in the `ModelValidationResult`. The `Result` wrapper is
/// reserved for future I/O errors when parsing APR model files.
pub fn validate_model(
    model_path: &Path,
    contract: &TensorLayoutContract,
) -> Result<ModelValidationResult> {
    // Early returns for missing path or no safetensors files
    if let Some(early_result) = check_model_path_preconditions(model_path) {
        return Ok(early_result);
    }

    // Collect all tensor metadata and run validations
    let (results, critical_failures) = run_all_validations(model_path, contract);

    let rules_failed = results.iter().filter(|r| !r.passed).count();
    let rules_passed = results.iter().filter(|r| r.passed).count();

    Ok(ModelValidationResult {
        model_path: model_path.to_path_buf(),
        passed: critical_failures.is_empty() && rules_failed == 0,
        rules_checked: results.len(),
        rules_passed,
        rules_failed,
        tensor_results: results,
        critical_failures,
    })
}

/// Check model path preconditions, returning early result if validation cannot proceed
fn check_model_path_preconditions(model_path: &Path) -> Option<ModelValidationResult> {
    if !model_path.exists() {
        return Some(ModelValidationResult {
            model_path: model_path.to_path_buf(),
            passed: false,
            rules_checked: 0,
            rules_passed: 0,
            rules_failed: 1,
            tensor_results: vec![TensorValidationResult {
                tensor_name: "N/A".to_string(),
                rule_id: "FILE-EXISTS".to_string(),
                passed: false,
                details: format!("Model file not found: {}", model_path.display()),
                expected: Some("File exists".to_string()),
                actual: Some("File not found".to_string()),
            }],
            critical_failures: vec!["Model file not found".to_string()],
        });
    }

    let safetensors_files = find_safetensors_files(model_path);
    if safetensors_files.is_empty() {
        return Some(ModelValidationResult {
            model_path: model_path.to_path_buf(),
            passed: true,
            rules_checked: 0,
            rules_passed: 0,
            rules_failed: 0,
            tensor_results: vec![],
            critical_failures: vec![],
        });
    }

    None
}

/// Run all validation checks and collect results
fn run_all_validations(
    model_path: &Path,
    contract: &TensorLayoutContract,
) -> (Vec<TensorValidationResult>, Vec<String>) {
    let mut results = Vec::new();
    let mut critical_failures = Vec::new();

    let all_tensors = collect_tensor_metadata(model_path, &mut results);
    let config = find_and_load_config(model_path);

    // Validate lm_head (F-LAYOUT-CONTRACT-002 - CRITICAL)
    validate_lm_head(
        &all_tensors,
        &config,
        contract,
        &mut results,
        &mut critical_failures,
    );

    // Validate 2D tensors (F-LAYOUT-CONTRACT-001)
    validate_2d_tensors(contract, &all_tensors, &config, &mut results);

    // Validate 1D tensors (F-LAYOUT-CONTRACT-003)
    validate_1d_tensors(contract, &all_tensors, &config, &mut results);

    (results, critical_failures)
}

/// Collect tensor metadata from all SafeTensors files
fn collect_tensor_metadata(
    model_path: &Path,
    results: &mut Vec<TensorValidationResult>,
) -> HashMap<String, Vec<usize>> {
    let safetensors_files = find_safetensors_files(model_path);
    let mut all_tensors = HashMap::new();

    for file in &safetensors_files {
        match read_safetensors_metadata(file) {
            Ok(tensors) => all_tensors.extend(tensors),
            Err(e) => {
                results.push(TensorValidationResult {
                    tensor_name: file.display().to_string(),
                    rule_id: "PARSE-ERROR".to_string(),
                    passed: false,
                    details: format!("Failed to read SafeTensors metadata: {e}"),
                    expected: None,
                    actual: None,
                });
            }
        }
    }

    all_tensors
}

/// Validate lm_head shape (F-LAYOUT-CONTRACT-002 - GH-202 critical check)
fn validate_lm_head(
    all_tensors: &HashMap<String, Vec<usize>>,
    config: &ModelConfig,
    contract: &TensorLayoutContract,
    results: &mut Vec<TensorValidationResult>,
    critical_failures: &mut Vec<String>,
) {
    if let Some(lm_head_shape) = all_tensors.get("lm_head.weight") {
        let validation = validate_lm_head_shape(lm_head_shape, config, contract);
        if !validation.passed && validation.rule_id == "F-LAYOUT-CONTRACT-002" {
            critical_failures.push(validation.details.clone());
        }
        results.push(validation);
    }
}

/// Validate all 2D tensors (F-LAYOUT-CONTRACT-001)
fn validate_2d_tensors(
    contract: &TensorLayoutContract,
    all_tensors: &HashMap<String, Vec<usize>>,
    config: &ModelConfig,
    results: &mut Vec<TensorValidationResult>,
) {
    for (name, spec) in &contract.tensors {
        if !spec.transpose {
            continue;
        }

        if spec.apr_name.contains("{n}") {
            validate_layer_tensors(&spec.apr_name, all_tensors, config, spec, results);
        } else if let Some(actual_shape) = all_tensors.get(&spec.apr_name) {
            results.push(validate_2d_tensor_shape(name, actual_shape, spec, config));
        }
    }
}

/// Validate all 1D tensors (F-LAYOUT-CONTRACT-003)
fn validate_1d_tensors(
    contract: &TensorLayoutContract,
    all_tensors: &HashMap<String, Vec<usize>>,
    config: &ModelConfig,
    results: &mut Vec<TensorValidationResult>,
) {
    for (name, spec) in &contract.tensors {
        if spec.transpose {
            continue;
        }

        if spec.apr_name.contains("{n}") {
            validate_1d_layer_tensors(&spec.apr_name, all_tensors, config, spec, results);
        } else if let Some(actual_shape) = all_tensors.get(&spec.apr_name) {
            results.push(validate_1d_tensor_shape(name, actual_shape, spec, config));
        }
    }
}

/// Model configuration values for validation
#[derive(Debug, Default)]
struct ModelConfig {
    vocab_size: Option<usize>,
    hidden_size: Option<usize>,
    intermediate_size: Option<usize>,
    num_attention_heads: Option<usize>,
    num_key_value_heads: Option<usize>,
    num_hidden_layers: Option<usize>,
}

/// Find SafeTensors files in a path
fn find_safetensors_files(path: &Path) -> Vec<PathBuf> {
    if path.is_file() {
        if path.extension().is_some_and(|e| e == "safetensors") {
            return vec![path.to_path_buf()];
        }
        return Vec::new();
    }

    // Try safetensors subdirectory first
    let st_dir = path.join("safetensors");
    let search_dir = if st_dir.exists() { &st_dir } else { path };

    let Ok(entries) = search_dir.read_dir() else {
        return Vec::new();
    };

    entries
        .flatten()
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "safetensors"))
        .map(|e| e.path())
        .collect()
}

/// Read SafeTensors header to extract tensor shapes
fn read_safetensors_metadata(
    path: &Path,
) -> std::result::Result<HashMap<String, Vec<usize>>, String> {
    let mut file = File::open(path).map_err(|e| format!("Failed to open: {e}"))?;

    // SafeTensors format: first 8 bytes are header length (little endian u64)
    let mut header_len_bytes = [0u8; 8];
    file.read_exact(&mut header_len_bytes)
        .map_err(|e| format!("Failed to read header length: {e}"))?;
    let header_len = u64::from_le_bytes(header_len_bytes) as usize;

    if header_len > MAX_HEADER_SIZE {
        return Err(format!("Header too large: {header_len}"));
    }

    let mut header_bytes = vec![0u8; header_len];
    file.read_exact(&mut header_bytes)
        .map_err(|e| format!("Failed to read header: {e}"))?;

    let header_str =
        std::str::from_utf8(&header_bytes).map_err(|e| format!("Invalid UTF-8: {e}"))?;

    let header: serde_json::Value =
        serde_json::from_str(header_str).map_err(|e| format!("JSON parse error: {e}"))?;

    let obj = header.as_object().ok_or("Header is not JSON object")?;

    let tensors = obj
        .iter()
        .filter(|(name, _)| *name != "__metadata__")
        .filter_map(|(name, value)| {
            let shape = value.as_object()?.get("shape")?.as_array()?;
            let dims: Vec<usize> = shape
                .iter()
                .filter_map(|v| v.as_u64().map(|n| n as usize))
                .collect();
            Some((name.clone(), dims))
        })
        .collect();

    Ok(tensors)
}

/// Helper to extract usize from JSON
fn get_usize(json: &serde_json::Value, key: &str) -> Option<usize> {
    json.get(key)
        .and_then(serde_json::Value::as_u64)
        .map(|n| n as usize)
}

/// Find and load config.json
fn find_and_load_config(model_path: &Path) -> ModelConfig {
    let config_paths = if model_path.is_file() {
        // For file mode, check parent dir and look for hash-prefixed config
        let parent = model_path.parent().unwrap_or(model_path);
        let stem = model_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("");
        vec![
            parent.join(format!("{stem}.config.json")),
            parent.join("config.json"),
        ]
    } else {
        vec![
            model_path.join("config.json"),
            model_path.join("safetensors/config.json"),
        ]
    };

    for path in config_paths {
        if let Ok(content) = std::fs::read_to_string(&path) {
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(&content) {
                return ModelConfig {
                    vocab_size: get_usize(&json, "vocab_size"),
                    hidden_size: get_usize(&json, "hidden_size"),
                    intermediate_size: get_usize(&json, "intermediate_size"),
                    num_attention_heads: get_usize(&json, "num_attention_heads"),
                    num_key_value_heads: get_usize(&json, "num_key_value_heads"),
                    num_hidden_layers: get_usize(&json, "num_hidden_layers"),
                };
            }
        }
    }

    ModelConfig::default()
}

/// Validate lm_head shape (F-LAYOUT-CONTRACT-002) - CRITICAL
fn validate_lm_head_shape(
    actual_shape: &[usize],
    config: &ModelConfig,
    _contract: &TensorLayoutContract,
) -> TensorValidationResult {
    // lm_head.weight should be [vocab_size, hidden_size] in row-major
    if actual_shape.len() != 2 {
        return TensorValidationResult {
            tensor_name: "lm_head.weight".to_string(),
            rule_id: "F-LAYOUT-CONTRACT-002".to_string(),
            passed: false,
            details: "lm_head.weight must be 2D tensor".to_string(),
            expected: Some("[vocab_size, hidden_size]".to_string()),
            actual: Some(format!("{actual_shape:?}")),
        };
    }

    let (expected_vocab, expected_hidden) = (config.vocab_size, config.hidden_size);

    // Check if shape matches [vocab, hidden]
    let shape_valid = match (expected_vocab, expected_hidden) {
        (Some(vocab), Some(hidden)) => actual_shape[0] == vocab && actual_shape[1] == hidden,
        (Some(vocab), None) => actual_shape[0] == vocab,
        (None, Some(hidden)) => actual_shape[1] == hidden,
        (None, None) => true, // Can't validate without config
    };

    if shape_valid {
        TensorValidationResult {
            tensor_name: "lm_head.weight".to_string(),
            rule_id: "F-LAYOUT-CONTRACT-002".to_string(),
            passed: true,
            details: format!("lm_head.weight shape correct: {:?}", actual_shape),
            expected: Some(format!("[{:?}, {:?}]", expected_vocab, expected_hidden)),
            actual: Some(format!("{actual_shape:?}")),
        }
    } else {
        TensorValidationResult {
            tensor_name: "lm_head.weight".to_string(),
            rule_id: "F-LAYOUT-CONTRACT-002".to_string(),
            passed: false,
            details: format!(
                "lm_head.weight shape MISMATCH (GH-202 bug pattern): expected [{:?}, {:?}], got {:?}",
                expected_vocab, expected_hidden, actual_shape
            ),
            expected: Some(format!("[{:?}, {:?}]", expected_vocab, expected_hidden)),
            actual: Some(format!("{actual_shape:?}")),
        }
    }
}

/// Validate a 2D tensor shape (F-LAYOUT-CONTRACT-001)
fn validate_2d_tensor_shape(
    name: &str,
    actual_shape: &[usize],
    spec: &TensorSpec,
    config: &ModelConfig,
) -> TensorValidationResult {
    if actual_shape.len() != 2 {
        return TensorValidationResult {
            tensor_name: spec.apr_name.clone(),
            rule_id: "F-LAYOUT-CONTRACT-001".to_string(),
            passed: false,
            details: format!("{name} must be 2D, got {}D", actual_shape.len()),
            expected: Some(spec.apr_shape.clone()),
            actual: Some(format!("{actual_shape:?}")),
        };
    }

    // Parse expected shape from contract
    let expected = parse_expected_shape(&spec.apr_shape, config);

    let shape_valid = match expected {
        Some((dim0, dim1)) => actual_shape[0] == dim0 && actual_shape[1] == dim1,
        None => true, // Can't fully validate without all dimensions
    };

    TensorValidationResult {
        tensor_name: spec.apr_name.clone(),
        rule_id: "F-LAYOUT-CONTRACT-001".to_string(),
        passed: shape_valid,
        details: if shape_valid {
            format!("{name} shape correct: {actual_shape:?}")
        } else {
            format!("{name} shape mismatch")
        },
        expected: Some(spec.apr_shape.clone()),
        actual: Some(format!("{actual_shape:?}")),
    }
}

/// Validate layer tensors (for patterns like model.layers.{n}.*)
fn validate_layer_tensors(
    pattern: &str,
    all_tensors: &HashMap<String, Vec<usize>>,
    config: &ModelConfig,
    spec: &TensorSpec,
    results: &mut Vec<TensorValidationResult>,
) {
    let num_layers = config.num_hidden_layers.unwrap_or(0);
    for layer_idx in 0..num_layers {
        let tensor_name = pattern.replace("{n}", &layer_idx.to_string());
        if let Some(actual_shape) = all_tensors.get(&tensor_name) {
            let validation = validate_2d_tensor_shape(&tensor_name, actual_shape, spec, config);
            results.push(validation);
        }
    }
}

/// Validate 1D layer tensors (F-LAYOUT-CONTRACT-003)
fn validate_1d_layer_tensors(
    pattern: &str,
    all_tensors: &HashMap<String, Vec<usize>>,
    config: &ModelConfig,
    spec: &TensorSpec,
    results: &mut Vec<TensorValidationResult>,
) {
    let num_layers = config.num_hidden_layers.unwrap_or(0);
    for layer_idx in 0..num_layers {
        let tensor_name = pattern.replace("{n}", &layer_idx.to_string());
        if let Some(actual_shape) = all_tensors.get(&tensor_name) {
            let validation = validate_1d_tensor_shape(&tensor_name, actual_shape, spec, config);
            results.push(validation);
        }
    }
}

/// Validate a 1D tensor shape (F-LAYOUT-CONTRACT-003)
fn validate_1d_tensor_shape(
    name: &str,
    actual_shape: &[usize],
    spec: &TensorSpec,
    config: &ModelConfig,
) -> TensorValidationResult {
    if actual_shape.len() != 1 {
        return TensorValidationResult {
            tensor_name: name.to_string(),
            rule_id: "F-LAYOUT-CONTRACT-003".to_string(),
            passed: false,
            details: format!("{name} must be 1D, got {}D", actual_shape.len()),
            expected: Some(spec.apr_shape.clone()),
            actual: Some(format!("{actual_shape:?}")),
        };
    }

    // 1D tensors should match hidden_size
    let shape_valid = config.hidden_size.is_none_or(|h| actual_shape[0] == h);

    TensorValidationResult {
        tensor_name: name.to_string(),
        rule_id: "F-LAYOUT-CONTRACT-003".to_string(),
        passed: shape_valid,
        details: if shape_valid {
            format!("{name} shape correct: {actual_shape:?}")
        } else {
            format!(
                "{name} shape mismatch: expected [{}], got {actual_shape:?}",
                config.hidden_size.unwrap_or(0)
            )
        },
        expected: Some(spec.apr_shape.clone()),
        actual: Some(format!("{actual_shape:?}")),
    }
}

/// Parse expected shape from contract string like "[vocab, hidden]"
fn parse_expected_shape(shape_str: &str, config: &ModelConfig) -> Option<(usize, usize)> {
    let shape_parts = parse_shape_dims(shape_str);
    if shape_parts.len() != 2 {
        return None;
    }

    let first_dim = resolve_dimension(&shape_parts[0], config)?;
    let second_dim = resolve_dimension(&shape_parts[1], config)?;
    Some((first_dim, second_dim))
}

/// Resolve a dimension name to its value from config
fn resolve_dimension(dim: &str, config: &ModelConfig) -> Option<usize> {
    match dim {
        "vocab" | "vocab_size" => config.vocab_size,
        "hidden" | "hidden_dim" | "hidden_size" => config.hidden_size,
        "intermediate" | "intermediate_dim" | "intermediate_size" => config.intermediate_size,
        s if s.contains('*') => {
            // Handle expressions like "heads*head_dim" or "kv_heads*head_dim"
            let parts: Vec<&str> = s.split('*').map(str::trim).collect();
            if parts.len() == 2 {
                let left = resolve_dimension(parts[0], config)?;
                let right = resolve_dimension(parts[1], config)?;
                Some(left * right)
            } else {
                None
            }
        }
        "heads" | "num_heads" | "num_attention_heads" => config.num_attention_heads,
        "kv_heads" | "num_kv_heads" | "num_key_value_heads" => config.num_key_value_heads,
        "head_dim" => {
            // head_dim = hidden_size / num_attention_heads
            match (config.hidden_size, config.num_attention_heads) {
                (Some(h), Some(n)) if n > 0 => Some(h / n),
                _ => None,
            }
        }
        _ => dim.parse().ok(),
    }
}

/// Get all validation rules from the contract.
#[must_use]
pub fn get_validation_rules(contract: &TensorLayoutContract) -> &[ValidationRule] {
    &contract.validation_rules
}

/// Get critical tensors from the contract (those marked with critical=true).
#[must_use]
pub fn get_critical_tensors(contract: &TensorLayoutContract) -> Vec<&TensorSpec> {
    contract.tensors.values().filter(|t| t.critical).collect()
}

/// Check if a shape string represents a 2D tensor.
#[must_use]
pub fn is_2d_shape(shape: &str) -> bool {
    // Count commas - 2D has exactly one comma
    shape.matches(',').count() == 1
}

/// Parse shape string to dimensions (e.g., `"[vocab, hidden]"` -> `["vocab", "hidden"]`).
#[must_use]
pub fn parse_shape_dims(shape: &str) -> Vec<String> {
    shape
        .trim_matches(|c| c == '[' || c == ']')
        .split(',')
        .map(|s| s.trim().to_string())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_2d_shape() {
        assert!(is_2d_shape("[vocab, hidden]"));
        assert!(is_2d_shape("[hidden, vocab]"));
        assert!(!is_2d_shape("[hidden]"));
        assert!(!is_2d_shape("[a, b, c]"));
    }

    #[test]
    fn test_parse_shape_dims() {
        let dims = parse_shape_dims("[vocab, hidden]");
        assert_eq!(dims, vec!["vocab", "hidden"]);

        let dims = parse_shape_dims("[hidden]");
        assert_eq!(dims, vec!["hidden"]);
    }

    #[test]
    fn test_load_contract_missing_file() {
        let result = load_contract_from("/nonexistent/path.yaml");
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_model_missing_file() {
        // Create a minimal contract for testing
        let contract = TensorLayoutContract {
            metadata: ContractMetadata {
                version: "1.0".to_string(),
                created: "2026-01-01".to_string(),
                updated: "2026-01-01".to_string(),
                author: "test".to_string(),
                description: "test".to_string(),
            },
            formats: HashMap::new(),
            kernel: KernelConvention {
                signature: "test".to_string(),
                weight_shape: "[out, in]".to_string(),
                computation: "y = Wx".to_string(),
                byte_calculation: "out * in".to_string(),
                block_sizes: HashMap::new(),
                qk_k: 256,
            },
            tensors: HashMap::new(),
            validation_rules: vec![],
            semantic_validation: None,
        };

        let result = validate_model(Path::new("/nonexistent/model.apr"), &contract);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(!result.passed);
        assert!(!result.critical_failures.is_empty());
    }

    #[test]
    fn test_get_critical_tensors() {
        let mut tensors = HashMap::new();
        tensors.insert(
            "lm_head".to_string(),
            TensorSpec {
                gguf_name: "output.weight".to_string(),
                apr_name: "lm_head.weight".to_string(),
                gguf_shape: "[hidden, vocab]".to_string(),
                apr_shape: "[vocab, hidden]".to_string(),
                transpose: true,
                kernel: "matmul".to_string(),
                kernel_out_dim: Some("vocab_size".to_string()),
                kernel_in_dim: Some("hidden_dim".to_string()),
                validation: None,
                critical: true,
                note: Some("GH-202".to_string()),
            },
        );
        tensors.insert(
            "embedding".to_string(),
            TensorSpec {
                gguf_name: "token_embd.weight".to_string(),
                apr_name: "model.embed_tokens.weight".to_string(),
                gguf_shape: "[hidden, vocab]".to_string(),
                apr_shape: "[vocab, hidden]".to_string(),
                transpose: true,
                kernel: "lookup".to_string(),
                kernel_out_dim: None,
                kernel_in_dim: None,
                validation: None,
                critical: false,
                note: None,
            },
        );

        let contract = TensorLayoutContract {
            metadata: ContractMetadata {
                version: "1.0".to_string(),
                created: "2026-01-01".to_string(),
                updated: "2026-01-01".to_string(),
                author: "test".to_string(),
                description: "test".to_string(),
            },
            formats: HashMap::new(),
            kernel: KernelConvention {
                signature: "test".to_string(),
                weight_shape: "[out, in]".to_string(),
                computation: "y = Wx".to_string(),
                byte_calculation: "out * in".to_string(),
                block_sizes: HashMap::new(),
                qk_k: 256,
            },
            tensors,
            validation_rules: vec![],
            semantic_validation: None,
        };

        let critical = get_critical_tensors(&contract);
        assert_eq!(critical.len(), 1);
        assert_eq!(critical[0].apr_name, "lm_head.weight");
    }
}
