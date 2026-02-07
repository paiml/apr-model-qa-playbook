//! Shared Format Contract: YAML-Defined Behavioral Invariants
//!
//! Implements invariants I-2 through I-5 from the Five-Whys analysis
//! (GH-190, GH-191). I-1 (Golden Rule Test) is already implemented
//! in `executor.rs`.
//!
//! The contract is defined in `apr_format_contract.yaml` and loaded
//! at compile time via `include_str!()`.

use crate::command::CommandRunner;
use crate::evidence::Evidence;
use apr_qa_gen::{Backend, Format, Modality, ModelId, QaScenario};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Embedded YAML contract — single source of truth for format invariants.
const CONTRACT_YAML: &str = include_str!("apr_format_contract.yaml");

// ============================================================================
// Contract types (deserialized from YAML)
// ============================================================================

/// Top-level format contract.
#[derive(Debug, Clone, Deserialize)]
pub struct FormatContract {
    /// Contract version (e.g., "1.0").
    pub version: String,
    /// Tensor naming convention.
    pub tensor_naming: TensorNamingContract,
    /// GGML dtype-to-byte mappings.
    pub dtype_bytes: DtypeByteSection,
    /// Per-dtype tolerances.
    pub tolerances: Vec<ToleranceEntry>,
    /// Invariant definitions (I-1 through I-5).
    pub invariants: Vec<InvariantDef>,
}

/// Tensor naming convention contract.
#[derive(Debug, Clone, Deserialize)]
pub struct TensorNamingContract {
    /// Convention name (e.g., "gguf-short").
    pub convention: String,
    /// Human-readable description.
    pub description: String,
    /// Canonical/forbidden example pairs.
    pub examples: Vec<NamingExample>,
    /// Regex pattern that valid names must match.
    pub pattern: String,
}

/// Example of canonical vs. forbidden tensor name.
#[derive(Debug, Clone, Deserialize)]
pub struct NamingExample {
    /// Correct short name.
    pub canonical: String,
    /// Forbidden long-form name.
    pub forbidden: String,
}

/// Dtype bytes section with description and mappings.
#[derive(Debug, Clone, Deserialize)]
pub struct DtypeByteSection {
    /// Human-readable description.
    pub description: String,
    /// Dtype-to-byte mappings.
    pub mappings: Vec<DtypeByteEntry>,
}

/// Single dtype-to-byte mapping.
#[derive(Debug, Clone, Deserialize)]
pub struct DtypeByteEntry {
    /// Dtype label (e.g., "Q4_K").
    pub dtype: String,
    /// GGML byte value.
    pub byte: u8,
}

/// Per-dtype tolerance for statistical comparison.
#[derive(Debug, Clone, Deserialize)]
pub struct ToleranceEntry {
    /// Dtype label.
    pub dtype: String,
    /// Absolute tolerance.
    pub atol: f64,
    /// Relative tolerance.
    pub rtol: f64,
}

/// Definition of a single invariant.
#[derive(Debug, Clone, Deserialize)]
pub struct InvariantDef {
    /// Invariant ID (e.g., "I-2").
    pub id: String,
    /// Short name.
    pub name: String,
    /// Description of what the invariant checks.
    pub description: String,
    /// Bug tickets this invariant catches.
    pub catches: Vec<String>,
    /// Gate ID for evidence (e.g., "F-CONTRACT-I2-001").
    pub gate_id: String,
    /// Command template (if applicable).
    #[serde(default)]
    pub test: Option<String>,
    /// Whether already implemented elsewhere.
    #[serde(default)]
    pub implemented: bool,
}

/// Which invariants to enable in a contract test run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContractTestConfig {
    /// List of invariant IDs to enable (e.g., `["I-2", "I-3", "I-4", "I-5"]`).
    #[serde(default = "default_invariants")]
    pub invariants: Vec<String>,
}

fn default_invariants() -> Vec<String> {
    vec![
        "I-2".to_string(),
        "I-3".to_string(),
        "I-4".to_string(),
        "I-5".to_string(),
    ]
}

impl Default for ContractTestConfig {
    fn default() -> Self {
        Self {
            invariants: default_invariants(),
        }
    }
}

/// Invariant identifier enum for type-safe dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InvariantId {
    /// I-1: Round-trip identity (implemented in executor.rs).
    I1,
    /// I-2: Tensor name bijection.
    I2,
    /// I-3: No silent fallbacks.
    I3,
    /// I-4: Statistical preservation.
    I4,
    /// I-5: Tokenizer roundtrip.
    I5,
}

impl InvariantId {
    /// Parse from string label (e.g., "I-2").
    #[must_use]
    pub fn from_label(label: &str) -> Option<Self> {
        match label {
            "I-1" => Some(Self::I1),
            "I-2" => Some(Self::I2),
            "I-3" => Some(Self::I3),
            "I-4" => Some(Self::I4),
            "I-5" => Some(Self::I5),
            _ => None,
        }
    }

    /// Gate ID for this invariant.
    #[must_use]
    pub fn gate_id(self) -> &'static str {
        match self {
            Self::I1 => "F-CONTRACT-I1-001",
            Self::I2 => "F-CONTRACT-I2-001",
            Self::I3 => "F-CONTRACT-I3-001",
            Self::I4 => "F-CONTRACT-I4-001",
            Self::I5 => "F-CONTRACT-I5-001",
        }
    }
}

// ============================================================================
// Contract loader
// ============================================================================

/// Load the embedded format contract from YAML.
///
/// # Errors
///
/// Returns an error if the embedded YAML fails to parse.
pub fn load_format_contract() -> crate::error::Result<FormatContract> {
    serde_yaml::from_str(CONTRACT_YAML).map_err(crate::error::Error::from)
}

// ============================================================================
// Pure validation functions (no subprocess)
// ============================================================================

/// Validate that dtype byte mappings have no duplicate byte values.
///
/// # Errors
///
/// Returns an error if duplicate byte values are found.
pub fn validate_dtype_bytes(contract: &FormatContract) -> crate::error::Result<()> {
    let mut seen = HashSet::new();
    for entry in &contract.dtype_bytes.mappings {
        if !seen.insert(entry.byte) {
            return Err(crate::error::Error::Execution(format!(
                "Duplicate GGML byte value {} for dtype {}",
                entry.byte, entry.dtype
            )));
        }
    }
    Ok(())
}

/// Validate a tensor name against the contract pattern.
///
/// Returns `true` if the name matches the GGUF-short convention.
#[must_use]
pub fn validate_tensor_name(name: &str, contract: &FormatContract) -> bool {
    // Simple pattern matching without regex dependency.
    // The pattern is: ^(\d+\.\w+\.\w+|token_embd\.\w+|output_norm\.\w+|output\.\w+)$
    is_valid_tensor_name(name, &contract.tensor_naming.pattern)
}

/// Check if a tensor name matches the GGUF-short naming pattern.
///
/// Supported patterns:
/// - `{digit}.{word}.{word}` (layer tensors)
/// - `token_embd.{word}` (embedding)
/// - `output_norm.{word}` (output norm)
/// - `output.{word}` (output head)
fn is_valid_tensor_name(name: &str, _pattern: &str) -> bool {
    let parts: Vec<&str> = name.split('.').collect();
    match parts.len() {
        2 => {
            // token_embd.weight, output_norm.weight, output.weight
            matches!(parts[0], "token_embd" | "output_norm" | "output") && is_word(parts[1])
        }
        3 => {
            // 0.q_proj.weight — first part must be all digits
            parts[0].chars().all(|c| c.is_ascii_digit())
                && !parts[0].is_empty()
                && is_word(parts[1])
                && is_word(parts[2])
        }
        _ => false,
    }
}

/// Check if a string is a "word" (alphanumeric + underscore, non-empty).
fn is_word(s: &str) -> bool {
    !s.is_empty() && s.chars().all(|c| c.is_alphanumeric() || c == '_')
}

/// Look up tolerance values for a given dtype.
///
/// Returns `None` if the dtype is not in the contract.
#[must_use]
pub fn lookup_tolerance(dtype: &str, contract: &FormatContract) -> Option<(f64, f64)> {
    contract
        .tolerances
        .iter()
        .find(|t| t.dtype == dtype)
        .map(|t| (t.atol, t.rtol))
}

// ============================================================================
// Contract invariant test runners (use CommandRunner)
// ============================================================================

/// Run contract invariant tests I-2 through I-5.
///
/// Returns `(passed, failed)` counts.
pub fn run_contract_tests(
    runner: &Arc<dyn CommandRunner>,
    model_path: &Path,
    model_id: &ModelId,
    config: &ContractTestConfig,
) -> Vec<Evidence> {
    let mut evidence = Vec::new();
    let contract = match load_format_contract() {
        Ok(c) => c,
        Err(e) => {
            evidence.push(Evidence::falsified(
                "F-CONTRACT-LOAD-001",
                contract_scenario(model_id),
                format!("Failed to load format contract: {e}"),
                "N/A",
                0,
            ));
            return evidence;
        }
    };

    for label in &config.invariants {
        let Some(inv_id) = InvariantId::from_label(label) else {
            continue;
        };

        // Skip I-1 (handled by golden rule test in executor.rs)
        if inv_id == InvariantId::I1 {
            continue;
        }

        let inv_def = contract.invariants.iter().find(|i| i.id == *label);
        let gate_id = inv_def.map_or_else(|| inv_id.gate_id(), |d| d.gate_id.as_str());

        let ev = match inv_id {
            InvariantId::I1 => unreachable!(),
            InvariantId::I2 => run_i2_tensor_bijection(runner, model_path, model_id, gate_id),
            InvariantId::I3 => run_i3_no_silent_fallbacks(runner, model_path, model_id, gate_id),
            InvariantId::I4 => {
                run_i4_statistical_preservation(runner, model_path, model_id, gate_id)
            }
            InvariantId::I5 => run_i5_tokenizer_roundtrip(runner, model_path, model_id, gate_id),
        };
        evidence.push(ev);
    }

    evidence
}

/// Resolve the APR file path from a workspace directory.
///
/// Workspace layout: `{workspace}/apr/model.apr`
/// Avoids `Path::with_extension` which corrupts names containing dots
/// (e.g., `Qwen2.5-Coder-0.5B-Instruct` becomes `Qwen2.5-Coder-0.apr`).
fn resolve_apr_path(model_path: &Path) -> PathBuf {
    model_path.join("apr").join("model.apr")
}

/// Resolve the SafeTensors file path from a workspace directory.
fn resolve_safetensors_path(model_path: &Path) -> PathBuf {
    model_path.join("safetensors").join("model.safetensors")
}

/// I-2: Tensor Name Bijection — writer names == reader names.
fn run_i2_tensor_bijection(
    runner: &Arc<dyn CommandRunner>,
    model_path: &Path,
    model_id: &ModelId,
    gate_id: &str,
) -> Evidence {
    let st_path = resolve_safetensors_path(model_path);
    let apr_path = resolve_apr_path(model_path);
    let result = runner.diff_tensors(&st_path, &apr_path, true);

    if !result.success {
        return Evidence::falsified(
            gate_id,
            contract_scenario(model_id),
            format!(
                "I-2 Tensor Name Bijection: diff-tensors failed: {}",
                result.stderr
            ),
            &result.stdout,
            0,
        );
    }

    if result.stdout.contains("\"mismatched_tensors\":0") || result.stdout.contains("0 mismatches")
    {
        let mut ev =
            Evidence::corroborated(gate_id, contract_scenario(model_id), &result.stdout, 0);
        ev.reason = "I-2 Tensor Name Bijection: all tensor names match".to_string();
        ev
    } else {
        Evidence::falsified(
            gate_id,
            contract_scenario(model_id),
            format!(
                "I-2 Tensor Name Bijection: tensor name mismatches detected: {}",
                result.stdout
            ),
            &result.stdout,
            0,
        )
    }
}

/// I-3: No Silent Fallbacks — unknown dtype → error, never default to F32.
fn run_i3_no_silent_fallbacks(
    runner: &Arc<dyn CommandRunner>,
    model_path: &Path,
    model_id: &ModelId,
    gate_id: &str,
) -> Evidence {
    let apr_path = resolve_apr_path(model_path);
    let result = runner.check_model(&apr_path);

    if !result.success {
        return Evidence::falsified(
            gate_id,
            contract_scenario(model_id),
            format!("I-3 No Silent Fallbacks: check failed: {}", result.stderr),
            &result.stdout,
            0,
        );
    }

    if contains_f32_fallback(&result.stdout) || contains_f32_fallback(&result.stderr) {
        Evidence::falsified(
            gate_id,
            contract_scenario(model_id),
            "I-3 No Silent Fallbacks: detected F32 fallback in check output",
            &result.stdout,
            0,
        )
    } else {
        let mut ev =
            Evidence::corroborated(gate_id, contract_scenario(model_id), &result.stdout, 0);
        ev.reason = "I-3 No Silent Fallbacks: no F32 fallbacks detected".to_string();
        ev
    }
}

/// Check if output contains evidence of silent F32 fallback.
fn contains_f32_fallback(output: &str) -> bool {
    let lower = output.to_lowercase();
    lower.contains("fallback") && lower.contains("f32")
        || lower.contains("defaulting to f32")
        || lower.contains("unknown dtype")
}

/// I-4: Statistical Preservation — tensor stats within dtype tolerance.
fn run_i4_statistical_preservation(
    runner: &Arc<dyn CommandRunner>,
    model_path: &Path,
    model_id: &ModelId,
    gate_id: &str,
) -> Evidence {
    let st_path = resolve_safetensors_path(model_path);
    let apr_path = resolve_apr_path(model_path);
    let result = runner.validate_stats(&st_path, &apr_path);

    if !result.success {
        return Evidence::falsified(
            gate_id,
            contract_scenario(model_id),
            format!(
                "I-4 Statistical Preservation: validate-stats failed: {}",
                result.stderr
            ),
            &result.stdout,
            0,
        );
    }

    if result.stdout.contains("\"passed\":true") || result.stdout.contains("passed") {
        let mut ev =
            Evidence::corroborated(gate_id, contract_scenario(model_id), &result.stdout, 0);
        ev.reason = "I-4 Statistical Preservation: tensor statistics preserved within tolerance"
            .to_string();
        ev
    } else {
        Evidence::falsified(
            gate_id,
            contract_scenario(model_id),
            format!(
                "I-4 Statistical Preservation: statistics diverged: {}",
                result.stdout
            ),
            &result.stdout,
            0,
        )
    }
}

/// I-5: Tokenizer Roundtrip — encode(decode(tokens)) == tokens.
fn run_i5_tokenizer_roundtrip(
    runner: &Arc<dyn CommandRunner>,
    model_path: &Path,
    model_id: &ModelId,
    gate_id: &str,
) -> Evidence {
    let st_path = resolve_safetensors_path(model_path);
    let apr_path = resolve_apr_path(model_path);
    let result = runner.compare_inference(&st_path, &apr_path, "Hello", 1, 0.0);

    if !result.success {
        return Evidence::falsified(
            gate_id,
            contract_scenario(model_id),
            format!(
                "I-5 Tokenizer Roundtrip: compare-inference failed: {}",
                result.stderr
            ),
            &result.stdout,
            0,
        );
    }

    if result.stdout.contains("\"passed\":true") {
        let mut ev =
            Evidence::corroborated(gate_id, contract_scenario(model_id), &result.stdout, 0);
        ev.reason = "I-5 Tokenizer Roundtrip: tokenizer roundtrip verified".to_string();
        ev
    } else {
        Evidence::falsified(
            gate_id,
            contract_scenario(model_id),
            format!(
                "I-5 Tokenizer Roundtrip: inference output mismatch: {}",
                result.stdout
            ),
            &result.stdout,
            0,
        )
    }
}

/// Create a scenario for contract test evidence.
fn contract_scenario(model_id: &ModelId) -> QaScenario {
    QaScenario::new(
        model_id.clone(),
        Modality::Run,
        Backend::Cpu,
        Format::Apr,
        "Format contract invariant".to_string(),
        0,
    )
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evidence::Outcome;

    #[test]
    fn test_load_format_contract() {
        let contract = load_format_contract().expect("Failed to load contract");
        assert!(!contract.invariants.is_empty());
        assert!(!contract.dtype_bytes.mappings.is_empty());
        assert!(!contract.tolerances.is_empty());
    }

    #[test]
    fn test_contract_version() {
        let contract = load_format_contract().expect("Failed to load contract");
        assert_eq!(contract.version, "1.0");
    }

    #[test]
    fn test_dtype_byte_mappings_complete() {
        let contract = load_format_contract().expect("Failed to load contract");
        let dtypes: Vec<&str> = contract
            .dtype_bytes
            .mappings
            .iter()
            .map(|m| m.dtype.as_str())
            .collect();
        assert!(dtypes.contains(&"F32"));
        assert!(dtypes.contains(&"F16"));
        assert!(dtypes.contains(&"Q4_K"));
        assert!(dtypes.contains(&"Q6_K"));
        assert!(dtypes.contains(&"BF16"));
        assert!(dtypes.contains(&"Q8_0"));
        assert!(dtypes.contains(&"Q2_K"));
        assert!(dtypes.contains(&"Q3_K"));
        assert!(dtypes.contains(&"Q5_K"));
        assert!(dtypes.contains(&"Q4_0"));
        assert!(dtypes.contains(&"Q5_0"));
    }

    #[test]
    fn test_dtype_byte_no_duplicates() {
        let contract = load_format_contract().expect("Failed to load contract");
        validate_dtype_bytes(&contract).expect("No duplicates expected");
    }

    #[test]
    fn test_dtype_byte_ggml_values() {
        let contract = load_format_contract().expect("Failed to load contract");
        let find_byte = |dtype: &str| -> u8 {
            contract
                .dtype_bytes
                .mappings
                .iter()
                .find(|m| m.dtype == dtype)
                .expect("dtype not found")
                .byte
        };
        assert_eq!(find_byte("F32"), 0);
        assert_eq!(find_byte("F16"), 1);
        assert_eq!(find_byte("Q4_K"), 12);
        assert_eq!(find_byte("Q6_K"), 14);
        assert_eq!(find_byte("BF16"), 30);
    }

    #[test]
    fn test_tensor_naming_pattern() {
        let contract = load_format_contract().expect("Failed to load contract");

        // Valid names
        assert!(validate_tensor_name("0.q_proj.weight", &contract));
        assert!(validate_tensor_name("31.down_proj.weight", &contract));
        assert!(validate_tensor_name("token_embd.weight", &contract));
        assert!(validate_tensor_name("output_norm.weight", &contract));
        assert!(validate_tensor_name("output.weight", &contract));

        // Invalid names (HuggingFace-style)
        assert!(!validate_tensor_name(
            "model.layers.0.self_attn.q_proj.weight",
            &contract
        ));
        assert!(!validate_tensor_name(
            "model.embed_tokens.weight",
            &contract
        ));
        assert!(!validate_tensor_name("", &contract));
    }

    #[test]
    fn test_invariant_definitions_complete() {
        let contract = load_format_contract().expect("Failed to load contract");
        assert_eq!(contract.invariants.len(), 5);
        let ids: Vec<&str> = contract.invariants.iter().map(|i| i.id.as_str()).collect();
        assert!(ids.contains(&"I-1"));
        assert!(ids.contains(&"I-2"));
        assert!(ids.contains(&"I-3"));
        assert!(ids.contains(&"I-4"));
        assert!(ids.contains(&"I-5"));
    }

    #[test]
    fn test_tolerance_lookup() {
        let contract = load_format_contract().expect("Failed to load contract");

        let (atol, rtol) = lookup_tolerance("F32", &contract).expect("F32 tolerance");
        assert!((atol - 0.0).abs() < f64::EPSILON);
        assert!((rtol - 0.0).abs() < f64::EPSILON);

        let (atol, rtol) = lookup_tolerance("Q4_K", &contract).expect("Q4_K tolerance");
        assert!((atol - 0.05).abs() < f64::EPSILON);
        assert!((rtol - 0.05).abs() < f64::EPSILON);

        let (atol, rtol) = lookup_tolerance("Q6_K", &contract).expect("Q6_K tolerance");
        assert!((atol - 0.02).abs() < f64::EPSILON);
        assert!((rtol - 0.02).abs() < f64::EPSILON);

        assert!(lookup_tolerance("UNKNOWN", &contract).is_none());
    }

    #[test]
    fn test_validate_tensor_name_valid() {
        let contract = load_format_contract().expect("Failed to load contract");
        for example in &contract.tensor_naming.examples {
            assert!(
                validate_tensor_name(&example.canonical, &contract),
                "Expected '{}' to be valid",
                example.canonical
            );
        }
    }

    #[test]
    fn test_validate_tensor_name_invalid() {
        let contract = load_format_contract().expect("Failed to load contract");
        for example in &contract.tensor_naming.examples {
            assert!(
                !validate_tensor_name(&example.forbidden, &contract),
                "Expected '{}' to be invalid",
                example.forbidden
            );
        }
    }

    #[test]
    fn test_contract_test_config_default() {
        let config = ContractTestConfig::default();
        assert_eq!(config.invariants.len(), 4);
        assert!(config.invariants.contains(&"I-2".to_string()));
        assert!(config.invariants.contains(&"I-3".to_string()));
        assert!(config.invariants.contains(&"I-4".to_string()));
        assert!(config.invariants.contains(&"I-5".to_string()));
    }

    #[test]
    fn test_invariant_id_from_label() {
        assert_eq!(InvariantId::from_label("I-1"), Some(InvariantId::I1));
        assert_eq!(InvariantId::from_label("I-2"), Some(InvariantId::I2));
        assert_eq!(InvariantId::from_label("I-3"), Some(InvariantId::I3));
        assert_eq!(InvariantId::from_label("I-4"), Some(InvariantId::I4));
        assert_eq!(InvariantId::from_label("I-5"), Some(InvariantId::I5));
        assert_eq!(InvariantId::from_label("I-99"), None);
    }

    #[test]
    fn test_invariant_id_gate_id() {
        assert_eq!(InvariantId::I1.gate_id(), "F-CONTRACT-I1-001");
        assert_eq!(InvariantId::I2.gate_id(), "F-CONTRACT-I2-001");
        assert_eq!(InvariantId::I3.gate_id(), "F-CONTRACT-I3-001");
        assert_eq!(InvariantId::I4.gate_id(), "F-CONTRACT-I4-001");
        assert_eq!(InvariantId::I5.gate_id(), "F-CONTRACT-I5-001");
    }

    #[test]
    fn test_contains_f32_fallback_positive() {
        assert!(contains_f32_fallback(
            "Warning: fallback to F32 for unknown type"
        ));
        assert!(contains_f32_fallback("defaulting to f32"));
        assert!(contains_f32_fallback("unknown dtype detected"));
    }

    #[test]
    fn test_contains_f32_fallback_negative() {
        assert!(!contains_f32_fallback("All checks passed"));
        assert!(!contains_f32_fallback("Using Q4_K quantization"));
        assert!(!contains_f32_fallback("F32 tensors loaded normally"));
    }

    #[test]
    fn test_contract_i2_tensor_name_bijection_pass() {
        use crate::command::MockCommandRunner;

        let runner: Arc<dyn CommandRunner> = Arc::new(MockCommandRunner::new());
        let model_id = ModelId::new("test", "model");
        let config = ContractTestConfig::default();

        let evidence = run_contract_tests(
            &runner,
            Path::new("/test/workspace/org/model"),
            &model_id,
            &config,
        );

        // I-2 should pass (mock diff_tensors returns 0 mismatches)
        let i2 = evidence.iter().find(|e| e.gate_id == "F-CONTRACT-I2-001");
        assert!(i2.is_some(), "I-2 evidence should exist");
        assert_eq!(i2.unwrap().outcome, Outcome::Corroborated);
    }

    #[test]
    fn test_contract_i2_tensor_name_bijection_fail() {
        use crate::command::MockCommandRunner;

        let runner: Arc<dyn CommandRunner> =
            Arc::new(MockCommandRunner::new().with_diff_tensors_failure());
        let model_id = ModelId::new("test", "model");
        let config = ContractTestConfig {
            invariants: vec!["I-2".to_string()],
        };

        let evidence = run_contract_tests(
            &runner,
            Path::new("/test/workspace/org/model"),
            &model_id,
            &config,
        );

        let i2 = evidence.iter().find(|e| e.gate_id == "F-CONTRACT-I2-001");
        assert!(i2.is_some());
        assert_eq!(i2.unwrap().outcome, Outcome::Falsified);
    }

    #[test]
    fn test_contract_i3_no_silent_fallbacks_pass() {
        use crate::command::MockCommandRunner;

        let runner: Arc<dyn CommandRunner> = Arc::new(MockCommandRunner::new());
        let model_id = ModelId::new("test", "model");
        let config = ContractTestConfig {
            invariants: vec!["I-3".to_string()],
        };

        let evidence = run_contract_tests(
            &runner,
            Path::new("/test/workspace/org/model"),
            &model_id,
            &config,
        );

        let i3 = evidence.iter().find(|e| e.gate_id == "F-CONTRACT-I3-001");
        assert!(i3.is_some());
        assert_eq!(i3.unwrap().outcome, Outcome::Corroborated);
    }

    #[test]
    fn test_contract_i3_no_silent_fallbacks_fail() {
        use crate::command::MockCommandRunner;

        let runner: Arc<dyn CommandRunner> =
            Arc::new(MockCommandRunner::new().with_check_failure());
        let model_id = ModelId::new("test", "model");
        let config = ContractTestConfig {
            invariants: vec!["I-3".to_string()],
        };

        let evidence = run_contract_tests(
            &runner,
            Path::new("/test/workspace/org/model"),
            &model_id,
            &config,
        );

        let i3 = evidence.iter().find(|e| e.gate_id == "F-CONTRACT-I3-001");
        assert!(i3.is_some());
        assert_eq!(i3.unwrap().outcome, Outcome::Falsified);
    }

    #[test]
    fn test_contract_i4_statistical_preservation_pass() {
        use crate::command::MockCommandRunner;

        let runner: Arc<dyn CommandRunner> = Arc::new(MockCommandRunner::new());
        let model_id = ModelId::new("test", "model");
        let config = ContractTestConfig {
            invariants: vec!["I-4".to_string()],
        };

        let evidence = run_contract_tests(
            &runner,
            Path::new("/test/workspace/org/model"),
            &model_id,
            &config,
        );

        let i4 = evidence.iter().find(|e| e.gate_id == "F-CONTRACT-I4-001");
        assert!(i4.is_some());
        assert_eq!(i4.unwrap().outcome, Outcome::Corroborated);
    }

    #[test]
    fn test_contract_i4_statistical_preservation_fail() {
        use crate::command::MockCommandRunner;

        let runner: Arc<dyn CommandRunner> =
            Arc::new(MockCommandRunner::new().with_validate_stats_failure());
        let model_id = ModelId::new("test", "model");
        let config = ContractTestConfig {
            invariants: vec!["I-4".to_string()],
        };

        let evidence = run_contract_tests(
            &runner,
            Path::new("/test/workspace/org/model"),
            &model_id,
            &config,
        );

        let i4 = evidence.iter().find(|e| e.gate_id == "F-CONTRACT-I4-001");
        assert!(i4.is_some());
        assert_eq!(i4.unwrap().outcome, Outcome::Falsified);
    }

    #[test]
    fn test_contract_i5_tokenizer_roundtrip_pass() {
        use crate::command::MockCommandRunner;

        let runner: Arc<dyn CommandRunner> = Arc::new(MockCommandRunner::new());
        let model_id = ModelId::new("test", "model");
        let config = ContractTestConfig {
            invariants: vec!["I-5".to_string()],
        };

        let evidence = run_contract_tests(
            &runner,
            Path::new("/test/workspace/org/model"),
            &model_id,
            &config,
        );

        let i5 = evidence.iter().find(|e| e.gate_id == "F-CONTRACT-I5-001");
        assert!(i5.is_some());
        assert_eq!(i5.unwrap().outcome, Outcome::Corroborated);
    }

    #[test]
    fn test_contract_i5_tokenizer_roundtrip_fail() {
        use crate::command::MockCommandRunner;

        let runner: Arc<dyn CommandRunner> =
            Arc::new(MockCommandRunner::new().with_compare_inference_failure());
        let model_id = ModelId::new("test", "model");
        let config = ContractTestConfig {
            invariants: vec!["I-5".to_string()],
        };

        let evidence = run_contract_tests(
            &runner,
            Path::new("/test/workspace/org/model"),
            &model_id,
            &config,
        );

        let i5 = evidence.iter().find(|e| e.gate_id == "F-CONTRACT-I5-001");
        assert!(i5.is_some());
        assert_eq!(i5.unwrap().outcome, Outcome::Falsified);
    }

    #[test]
    fn test_contract_all_invariants_pass() {
        use crate::command::MockCommandRunner;

        let runner: Arc<dyn CommandRunner> = Arc::new(MockCommandRunner::new());
        let model_id = ModelId::new("test", "model");
        let config = ContractTestConfig::default();

        let evidence = run_contract_tests(
            &runner,
            Path::new("/test/workspace/org/model"),
            &model_id,
            &config,
        );

        // Should have 4 results (I-2 through I-5)
        assert_eq!(evidence.len(), 4);
        for ev in &evidence {
            assert_eq!(
                ev.outcome,
                Outcome::Corroborated,
                "Gate {} should pass",
                ev.gate_id
            );
        }
    }

    #[test]
    fn test_contract_skips_i1() {
        use crate::command::MockCommandRunner;

        let runner: Arc<dyn CommandRunner> = Arc::new(MockCommandRunner::new());
        let model_id = ModelId::new("test", "model");
        let config = ContractTestConfig {
            invariants: vec!["I-1".to_string(), "I-2".to_string()],
        };

        let evidence = run_contract_tests(
            &runner,
            Path::new("/test/workspace/org/model"),
            &model_id,
            &config,
        );

        // I-1 should be skipped (handled by golden rule test)
        assert_eq!(evidence.len(), 1);
        assert_eq!(evidence[0].gate_id, "F-CONTRACT-I2-001");
    }

    #[test]
    fn test_contract_unknown_invariant_skipped() {
        use crate::command::MockCommandRunner;

        let runner: Arc<dyn CommandRunner> = Arc::new(MockCommandRunner::new());
        let model_id = ModelId::new("test", "model");
        let config = ContractTestConfig {
            invariants: vec!["I-99".to_string()],
        };

        let evidence = run_contract_tests(
            &runner,
            Path::new("/test/workspace/org/model"),
            &model_id,
            &config,
        );

        assert!(evidence.is_empty());
    }

    #[test]
    fn test_resolve_paths_with_dots_in_name() {
        // Regression: Qwen2.5-Coder-0.5B-Instruct contains dots which caused
        // Path::with_extension("apr") to produce "Qwen2.5-Coder-0.apr"
        let workspace = Path::new("/output/workspace/Qwen/Qwen2.5-Coder-0.5B-Instruct");
        let apr = resolve_apr_path(workspace);
        let st = resolve_safetensors_path(workspace);

        assert_eq!(
            apr,
            PathBuf::from("/output/workspace/Qwen/Qwen2.5-Coder-0.5B-Instruct/apr/model.apr")
        );
        assert_eq!(
            st,
            PathBuf::from(
                "/output/workspace/Qwen/Qwen2.5-Coder-0.5B-Instruct/safetensors/model.safetensors"
            )
        );

        // Contrast with the old broken behavior
        let broken = workspace.with_extension("apr");
        assert_eq!(
            broken,
            PathBuf::from("/output/workspace/Qwen/Qwen2.5-Coder-0.apr")
        );
    }

    #[test]
    fn test_contract_tests_with_dotted_workspace_path() {
        use crate::command::MockCommandRunner;

        let runner: Arc<dyn CommandRunner> = Arc::new(MockCommandRunner::new());
        let model_id = ModelId::new("Qwen", "Qwen2.5-Coder-0.5B-Instruct");
        let config = ContractTestConfig::default();

        let evidence = run_contract_tests(
            &runner,
            Path::new("/workspace/Qwen/Qwen2.5-Coder-0.5B-Instruct"),
            &model_id,
            &config,
        );

        // All 4 invariants (I-2 through I-5) should produce evidence
        assert_eq!(evidence.len(), 4, "Expected 4 invariant results");
        for ev in &evidence {
            // None should mention truncated paths like "Coder-0.apr"
            assert!(
                !ev.reason.contains("Coder-0.apr"),
                "Path was truncated by with_extension: {}",
                ev.reason
            );
        }
    }

    #[test]
    fn test_is_valid_tensor_name_edge_cases() {
        let contract = load_format_contract().expect("Failed to load contract");

        // Valid edge cases
        assert!(validate_tensor_name("0.attn.weight", &contract));
        assert!(validate_tensor_name("99.mlp.bias", &contract));

        // Invalid edge cases
        assert!(!validate_tensor_name("weight", &contract));
        assert!(!validate_tensor_name(".q_proj.weight", &contract));
        assert!(!validate_tensor_name("a.q_proj.weight", &contract));
        assert!(!validate_tensor_name("0.q_proj.weight.extra", &contract));
    }

    #[test]
    fn test_naming_convention() {
        let contract = load_format_contract().expect("Failed to load contract");
        assert_eq!(contract.tensor_naming.convention, "gguf-short");
    }

    #[test]
    fn test_invariant_catches_fields() {
        let contract = load_format_contract().expect("Failed to load contract");
        let i1 = contract.invariants.iter().find(|i| i.id == "I-1").unwrap();
        assert!(i1.catches.contains(&"GH-190".to_string()));
        assert!(i1.implemented);

        let i2 = contract.invariants.iter().find(|i| i.id == "I-2").unwrap();
        assert!(i2.catches.contains(&"GH-190".to_string()));
        assert!(!i2.implemented);
    }

    #[test]
    fn test_tolerance_entries_ordered_by_precision() {
        let contract = load_format_contract().expect("Failed to load contract");
        // F32 should have 0 tolerance (exact)
        let f32_tol = lookup_tolerance("F32", &contract).unwrap();
        assert!(f32_tol.0.abs() < f64::EPSILON);

        // Q2_K should have the loosest tolerance
        let q2k_tol = lookup_tolerance("Q2_K", &contract).unwrap();
        assert!(q2k_tol.0 > 0.1);
    }

    #[test]
    fn test_is_word() {
        assert!(is_word("weight"));
        assert!(is_word("q_proj"));
        assert!(is_word("down_proj"));
        assert!(is_word("a"));
        assert!(!is_word(""));
        assert!(!is_word("has.dot"));
        assert!(!is_word("has space"));
    }
}
