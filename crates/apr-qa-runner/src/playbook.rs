//! Playbook definition and parsing
//!
//! Playbooks define test scenarios in YAML format.

use apr_qa_gen::{Backend, Format, Modality, ModelId, QaScenario};
use regex::Regex;
use serde::{Deserialize, Deserializer, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::sync::LazyLock;

use crate::error::{Error, Result};

/// Deserialize a bool that may be quoted as a string in YAML (CB-950 compliance).
/// Accepts both `true`/`false` (YAML boolean) and `"true"`/`"false"` (YAML string).
fn deserialize_bool_or_string<'de, D>(deserializer: D) -> std::result::Result<bool, D::Error>
where
    D: Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum BoolOrString {
        Bool(bool),
        String(String),
    }
    match BoolOrString::deserialize(deserializer)? {
        BoolOrString::Bool(b) => Ok(b),
        BoolOrString::String(s) => match s.to_lowercase().as_str() {
            "true" | "yes" | "on" => Ok(true),
            "false" | "no" | "off" => Ok(false),
            _ => Err(serde::de::Error::custom(format!(
                "expected boolean or truthy string, got '{s}'"
            ))),
        },
    }
}

// ── Playbook Naming Convention (PMAT-266) ────────────────────────────────────
//
// Playbook filenames MUST follow the pattern:
//   {family}-{size}[-{tier}].playbook.yaml
//
// Examples:
//   qwen2.5-coder-0.5b-mvp.playbook.yaml   → family="qwen2.5-coder", size="0.5b", tier="mvp"
//   llama3.2-1b.playbook.yaml              → family="llama3.2", size="1b", tier=None
//   deepseek-coder-v2-16b-full.playbook.yaml → family="deepseek-coder-v2", size="16b", tier="full"
//
// Size patterns: {digits}[.{digits}]b (e.g., 0.5b, 1b, 7b, 70b)
// Tier patterns: dim-smoke, mvp, smoke, quick, ci, full, nightly, release

/// Regex pattern for playbook naming convention
/// Matches: {family}-{size}[-{tier}].playbook.yaml
/// - family: one or more segments separated by `-` (letters, digits, dots)
/// - size: digits optionally with decimal, followed by `b` (e.g., 0.5b, 1b, 7b)
/// - tier (optional): dim-smoke, mvp, smoke, quick, ci, full, nightly, release
static PLAYBOOK_NAME_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    // This regex pattern is verified at compile time, unwrap is safe here
    #[allow(clippy::unwrap_used)]
    Regex::new(
        r"^(?P<family>(?:[a-z0-9]+\.?)+(?:-[a-z0-9]+\.?)*)-(?P<size>\d+(?:\.\d+)?b)(?:-(?P<tier>dim-smoke|mvp|smoke|quick|ci|full|nightly|release))?\.playbook\.yaml$"
    ).unwrap()
});

/// Valid tier values for playbook naming
pub const VALID_TIERS: &[&str] = &[
    "dim-smoke",
    "mvp",
    "smoke",
    "quick",
    "ci",
    "full",
    "nightly",
    "release",
];

/// Parsed components from a playbook filename
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlaybookNameParts {
    /// Model family (e.g., "qwen2.5-coder", "llama3.2")
    pub family: String,
    /// Model size (e.g., "0.5b", "7b", "70b")
    pub size: String,
    /// Optional tier (e.g., "mvp", "full", "nightly")
    pub tier: Option<String>,
}

impl PlaybookNameParts {
    /// Reconstruct the canonical filename from parts
    #[must_use]
    #[allow(clippy::option_if_let_else)]
    pub fn to_filename(&self) -> String {
        match &self.tier {
            Some(tier) => {
                format!("{}-{}-{}.playbook.yaml", self.family, self.size, tier)
            }
            None => format!("{}-{}.playbook.yaml", self.family, self.size),
        }
    }
}

/// Validate a playbook filename against the naming convention (PMAT-266)
///
/// # Arguments
/// * `filename` - The filename to validate (not the full path)
///
/// # Returns
/// * `Ok(PlaybookNameParts)` if valid
/// * `Err` with descriptive message if invalid
///
/// # Errors
///
/// Returns an error if the filename doesn't match the naming convention.
pub fn validate_playbook_name(filename: &str) -> Result<PlaybookNameParts> {
    let captures = PLAYBOOK_NAME_REGEX.captures(filename).ok_or_else(|| {
        Error::Validation(format!(
            "Playbook filename '{filename}' does not match naming convention: \
             {{family}}-{{size}}[-{{tier}}].playbook.yaml\n\
             Examples: qwen2.5-coder-0.5b-mvp.playbook.yaml, llama3.2-7b.playbook.yaml"
        ))
    })?;

    Ok(PlaybookNameParts {
        family: captures["family"].to_string(),
        size: captures["size"].to_string(),
        tier: captures.name("tier").map(|m| m.as_str().to_string()),
    })
}

/// Extract and validate playbook name from a full path
///
/// # Errors
///
/// Returns an error if the path has no filename or doesn't match the naming convention.
pub fn validate_playbook_path(path: impl AsRef<Path>) -> Result<PlaybookNameParts> {
    let path = path.as_ref();
    let filename = path
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| Error::Validation(format!("Invalid playbook path: {}", path.display())))?;

    validate_playbook_name(filename)
}

/// Model size category for resource management (§3.4 Resource-Aware Scheduling)
///
/// These categories enforce worker limits to prevent OOM conditions when testing
/// large models. The executor MUST respect these limits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SizeCategory {
    /// < 1B params: 4 workers, can run in parallel with others
    #[default]
    Tiny,
    /// 1-2B params: 4 workers, can run in parallel with tiny models
    Small,
    /// 2-4B params: 2 workers, should run alone or with tiny/small
    Medium,
    /// 4-10B params: 1 worker, must run alone
    Large,
    /// 10-30B params: 1 worker, must run alone, may need swap
    Xlarge,
    /// > 30B params: 1 worker, requires careful resource management
    Huge,
}

impl SizeCategory {
    /// Maximum workers allowed for this model size
    #[must_use]
    pub const fn max_workers(&self) -> usize {
        match self {
            Self::Tiny | Self::Small => 4,
            Self::Medium => 2,
            Self::Large | Self::Xlarge | Self::Huge => 1,
        }
    }

    /// Estimated memory requirement in GB (rough heuristic)
    #[must_use]
    pub const fn estimated_memory_gb(&self) -> usize {
        match self {
            Self::Tiny => 2,
            Self::Small => 4,
            Self::Medium => 8,
            Self::Large => 16,
            Self::Xlarge => 32,
            Self::Huge => 64,
        }
    }

    /// Can run concurrently with other playbooks
    #[must_use]
    pub const fn can_run_concurrent(&self) -> bool {
        matches!(self, Self::Tiny | Self::Small)
    }

    /// Parse a size category from a string.
    ///
    /// Accepts lowercase category names: tiny, small, medium, large, xlarge, huge.
    ///
    /// # Errors
    ///
    /// Returns an error if the string doesn't match a valid category.
    pub fn from_str_lowercase(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "tiny" => Ok(Self::Tiny),
            "small" => Ok(Self::Small),
            "medium" => Ok(Self::Medium),
            "large" => Ok(Self::Large),
            "xlarge" => Ok(Self::Xlarge),
            "huge" => Ok(Self::Huge),
            _ => Err(Error::Validation(format!(
                "Invalid size category: {s}. Valid: tiny, small, medium, large, xlarge, huge"
            ))),
        }
    }
}

/// A complete playbook for model qualification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Playbook {
    /// Playbook name
    pub name: String,
    /// Version
    pub version: String,
    /// Model configuration
    pub model: ModelConfig,
    /// Test matrix configuration
    pub test_matrix: TestMatrix,
    /// Property test definitions
    #[serde(default)]
    pub property_tests: Vec<PropertyTest>,
    /// Falsification gates
    #[serde(default)]
    pub falsification_gates: Vec<FalsificationGate>,
    /// State machine definition (optional)
    #[serde(default)]
    pub state_machine: Option<StateMachine>,
    /// Differential tests (GH-188, PMAT-114)
    #[serde(default)]
    pub differential_tests: Option<DifferentialTestConfig>,
    /// Profile CI assertions (PMAT-192)
    #[serde(default)]
    pub profile_ci: Option<ProfileCiConfig>,
    /// Trace payload testing (APR-TRACE-001)
    #[serde(default)]
    pub trace_payload: Option<TracePayloadConfig>,
    /// Contract invariant tests (GH-190/191 Five-Whys)
    #[serde(default)]
    pub contract_tests: Option<crate::contract::ContractTestConfig>,
    /// Ollama parity tests (GH-6/AC-2)
    #[serde(default)]
    pub ollama_parity: Option<OllamaParityConfig>,
}

impl Playbook {
    /// Load a playbook from a YAML file
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or parsed.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        Self::from_yaml(&content)
    }

    /// Parse a playbook from YAML string
    ///
    /// # Errors
    ///
    /// Returns an error if the YAML is invalid.
    pub fn from_yaml(yaml: &str) -> Result<Self> {
        serde_yaml::from_str(yaml).map_err(Error::from)
    }

    /// Convert to YAML string
    ///
    /// # Errors
    ///
    /// Returns an error if serialization fails.
    pub fn to_yaml(&self) -> Result<String> {
        serde_yaml::to_string(self).map_err(Error::from)
    }

    /// Generate all scenarios from this playbook
    #[must_use]
    pub fn generate_scenarios(&self) -> Vec<QaScenario> {
        let mut scenarios = Vec::new();
        let mut seed: u64 = 0;

        let model_id = ModelId::new(&self.model.hf_org(), &self.model.hf_name());

        // Use custom prompts from test_matrix if provided, otherwise fall back
        let default_prompt = "What is 2+2?".to_string();
        let prompts: &[String] = self
            .test_matrix
            .prompts
            .as_deref()
            .unwrap_or_else(|| std::slice::from_ref(&default_prompt));

        for modality in &self.test_matrix.modalities {
            for backend in &self.test_matrix.backends {
                for format in &self.model.formats {
                    for i in 0..self.test_matrix.scenario_count {
                        let prompt = prompts[i % prompts.len()].clone();
                        scenarios.push(QaScenario::new(
                            model_id.clone(),
                            *modality,
                            *backend,
                            *format,
                            prompt,
                            seed,
                        ));
                        seed = seed.wrapping_add(1);
                    }
                }
            }
        }

        scenarios
    }

    /// Get total expected test count
    #[must_use]
    pub fn total_tests(&self) -> usize {
        self.test_matrix.modalities.len()
            * self.test_matrix.backends.len()
            * self.model.formats.len()
            * self.test_matrix.scenario_count
    }

    /// Get the model ID for this playbook
    #[must_use]
    pub fn model_id(&self) -> ModelId {
        ModelId::new(&self.model.hf_org(), &self.model.hf_name())
    }

    /// Get the effective maximum workers based on model size (§3.4)
    ///
    /// This ENFORCES resource limits - the executor MUST use this value
    /// and cannot exceed it. Large models get fewer workers to prevent OOM.
    #[must_use]
    pub fn effective_max_workers(&self, requested: usize) -> usize {
        let size_limit = self.model.size_category.max_workers();
        requested.min(size_limit)
    }

    /// Get the model's size category
    #[must_use]
    pub fn size_category(&self) -> SizeCategory {
        self.model.size_category
    }
}

/// Model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// HuggingFace repository
    pub hf_repo: String,
    /// Optional local path
    pub local_path: Option<String>,
    /// Supported formats
    #[serde(default = "default_formats")]
    pub formats: Vec<Format>,
    /// Quantizations to test
    #[serde(default = "default_quantizations")]
    pub quantizations: Vec<String>,
    /// Model size category for resource-aware scheduling (§3.4)
    /// Defaults to `small` which allows 4 workers.
    /// IMPORTANT: Large models (7B+) MUST set this to `large` or higher
    /// to prevent OOM conditions during parallel testing.
    #[serde(default)]
    pub size_category: SizeCategory,

    // ── PMAT-269: Expected architectural parameters from family YAML ────────
    /// Expected hidden dimension (from family YAML size_variants)
    #[serde(default)]
    pub expected_hidden_dim: Option<u32>,
    /// Expected number of layers (from family YAML size_variants)
    #[serde(default)]
    pub expected_num_layers: Option<u32>,
    /// Expected number of attention heads (from family YAML size_variants)
    #[serde(default)]
    pub expected_num_heads: Option<u32>,
    /// Expected number of KV heads for GQA (from family YAML size_variants)
    #[serde(default)]
    pub expected_num_kv_heads: Option<u32>,
    /// Expected vocabulary size (from family YAML size_variants)
    #[serde(default)]
    pub expected_vocab_size: Option<u32>,
    /// Expected intermediate/FFN dimension (from family YAML size_variants)
    #[serde(default)]
    pub expected_intermediate_dim: Option<u32>,
    /// Model family identifier for contract lookup
    #[serde(default)]
    pub family: Option<String>,
    /// Size variant identifier (e.g., "0.5b", "7b")
    #[serde(default)]
    pub size_variant: Option<String>,
}

impl ModelConfig {
    /// Extract org from hf_repo
    #[must_use]
    pub fn hf_org(&self) -> String {
        self.hf_repo
            .split('/')
            .next()
            .unwrap_or("unknown")
            .to_string()
    }

    /// Extract name from hf_repo
    #[must_use]
    pub fn hf_name(&self) -> String {
        self.hf_repo
            .split('/')
            .nth(1)
            .unwrap_or(&self.hf_repo)
            .to_string()
    }

    /// Populate expected architectural parameters from a family contract (PMAT-269).
    ///
    /// This method derives expected values from the family YAML size_variants,
    /// enabling YAML-driven test matrix generation.
    ///
    /// # Arguments
    /// * `contract` - The family contract to derive values from
    /// * `size` - The size variant key (e.g., "0.5b", "7b")
    ///
    /// # Returns
    /// `true` if the size variant was found and values were populated,
    /// `false` if the size variant doesn't exist in the contract.
    pub fn populate_from_family_contract(
        &mut self,
        contract: &crate::family_contract::FamilyContract,
        size: &str,
    ) -> bool {
        let Some(variant) = contract.get_size_variant(size) else {
            return false;
        };

        self.family = Some(contract.family.clone());
        self.size_variant = Some(size.to_string());
        self.expected_hidden_dim = Some(variant.hidden_dim);
        self.expected_num_layers = Some(variant.num_layers);
        self.expected_num_heads = variant.num_heads;
        self.expected_num_kv_heads = variant.num_kv_heads;
        self.expected_vocab_size = variant.vocab_size;
        self.expected_intermediate_dim = variant.intermediate_dim;

        // PMAT-270: Auto-set size_category from family YAML if not explicitly set
        // Only override if the current size_category is the default (Tiny)
        if self.size_category == SizeCategory::default() {
            if let Some(category_str) = contract.get_size_category(size) {
                if let Ok(category) = SizeCategory::from_str_lowercase(category_str) {
                    self.size_category = category;
                }
            }
        }

        true
    }

    /// Check if this config has expected architectural parameters set.
    #[must_use]
    pub fn has_expected_params(&self) -> bool {
        self.expected_hidden_dim.is_some()
            || self.expected_num_layers.is_some()
            || self.expected_num_heads.is_some()
    }

    /// Validate that the model matches expected architectural parameters.
    ///
    /// Returns a list of mismatches if any parameters don't match.
    #[must_use]
    pub fn validate_architecture(
        &self,
        hidden_dim: u32,
        num_layers: u32,
        num_heads: Option<u32>,
        num_kv_heads: Option<u32>,
    ) -> Vec<String> {
        let mut mismatches = Vec::new();

        if let Some(expected) = self.expected_hidden_dim {
            if expected != hidden_dim {
                mismatches.push(format!(
                    "hidden_dim mismatch: expected {expected}, got {hidden_dim}"
                ));
            }
        }

        if let Some(expected) = self.expected_num_layers {
            if expected != num_layers {
                mismatches.push(format!(
                    "num_layers mismatch: expected {expected}, got {num_layers}"
                ));
            }
        }

        if let (Some(expected), Some(actual)) = (self.expected_num_heads, num_heads) {
            if expected != actual {
                mismatches.push(format!(
                    "num_heads mismatch: expected {expected}, got {actual}"
                ));
            }
        }

        if let (Some(expected), Some(actual)) = (self.expected_num_kv_heads, num_kv_heads) {
            if expected != actual {
                mismatches.push(format!(
                    "num_kv_heads mismatch: expected {expected}, got {actual}"
                ));
            }
        }

        mismatches
    }
}

fn default_formats() -> Vec<Format> {
    vec![Format::Gguf, Format::SafeTensors, Format::Apr]
}

fn default_quantizations() -> Vec<String> {
    vec!["q4_k_m".to_string()]
}

/// Test matrix configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestMatrix {
    /// Modalities to test
    #[serde(default = "default_modalities")]
    pub modalities: Vec<Modality>,
    /// Backends to test
    #[serde(default = "default_backends")]
    pub backends: Vec<Backend>,
    /// Number of scenarios per combination
    #[serde(default = "default_scenario_count")]
    pub scenario_count: usize,
    /// Architecture-specific prompts (optional; falls back to default if absent)
    #[serde(default)]
    pub prompts: Option<Vec<String>>,
}

fn default_modalities() -> Vec<Modality> {
    vec![Modality::Run, Modality::Chat, Modality::Serve]
}

fn default_backends() -> Vec<Backend> {
    vec![Backend::Cpu, Backend::Gpu]
}

fn default_scenario_count() -> usize {
    100
}

impl Default for TestMatrix {
    fn default() -> Self {
        Self {
            modalities: default_modalities(),
            backends: default_backends(),
            scenario_count: default_scenario_count(),
            prompts: None,
        }
    }
}

/// Property test definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropertyTest {
    /// Test name
    pub name: String,
    /// Generator expression
    pub generator: String,
    /// Oracle expression
    pub oracle: String,
    /// Number of test cases
    #[serde(default = "default_proptest_count")]
    pub count: usize,
}

fn default_proptest_count() -> usize {
    100
}

/// Falsification gate definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FalsificationGate {
    /// Gate ID (e.g., "F-QUAL-001")
    pub id: String,
    /// Description
    pub description: String,
    /// Condition expression
    pub condition: String,
    /// Severity (P0, P1, P2)
    #[serde(default = "default_severity")]
    pub severity: String,
}

fn default_severity() -> String {
    "P1".to_string()
}

/// State machine for complex workflows
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateMachine {
    /// Initial state
    pub initial: String,
    /// State definitions
    pub states: HashMap<String, State>,
}

/// State in a state machine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct State {
    /// Actions to execute on entering this state
    #[serde(default)]
    pub on_enter: Vec<Action>,
    /// Actions to execute on exiting this state
    #[serde(default)]
    pub on_exit: Vec<Action>,
    /// Transitions from this state
    #[serde(default)]
    pub transitions: Vec<Transition>,
}

/// Action to execute
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Action {
    /// Action name or command
    pub action: String,
}

/// Transition between states
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transition {
    /// Event that triggers this transition
    pub event: String,
    /// Target state
    pub target: String,
    /// Optional action to execute
    pub action: Option<String>,
    /// Guard conditions
    #[serde(default)]
    pub guards: Vec<String>,
}

/// A single step in a playbook
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlaybookStep {
    /// Step name
    pub name: String,
    /// Command to execute
    pub command: String,
    /// Timeout in milliseconds
    #[serde(default = "default_timeout")]
    pub timeout_ms: u64,
    /// Expected exit code
    #[serde(default)]
    pub expected_exit_code: i32,
    /// Expected output patterns
    #[serde(default)]
    pub expected_patterns: Vec<String>,
    /// Forbidden output patterns
    #[serde(default)]
    pub forbidden_patterns: Vec<String>,
}

fn default_timeout() -> u64 {
    60000 // 60 seconds
}

/// Differential test configuration (GH-188, PMAT-114, PMAT-201, PMAT-202)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DifferentialTestConfig {
    /// Format validation configuration (GH-186 prevention)
    #[serde(default)]
    pub format_validation: Option<FormatValidationConfig>,
    /// Tensor diff configuration
    #[serde(default)]
    pub tensor_diff: Option<TensorDiffConfig>,
    /// Inference comparison configuration
    #[serde(default)]
    pub inference_compare: Option<InferenceCompareConfig>,
    /// Fingerprint configuration (PMAT-201)
    #[serde(default)]
    pub fingerprint: Option<FingerprintConfig>,
    /// Validate stats configuration (PMAT-202)
    #[serde(default)]
    pub validate_stats: Option<ValidateStatsConfig>,
}

/// Format validation configuration (GH-186 prevention)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormatValidationConfig {
    /// Enable format validation
    #[serde(default, deserialize_with = "deserialize_bool_or_string")]
    pub enabled: bool,
    /// Checks to run: dtype_mapping, tensor_alignment, header_integrity
    #[serde(default)]
    pub checks: Vec<String>,
    /// Gates to verify
    #[serde(default)]
    pub gates: Vec<String>,
}

/// Tensor diff configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorDiffConfig {
    /// Enable tensor diff
    #[serde(default, deserialize_with = "deserialize_bool_or_string")]
    pub enabled: bool,
    /// Filter pattern for tensor names
    #[serde(default)]
    pub filter: Option<String>,
    /// Gates to verify
    #[serde(default)]
    pub gates: Vec<String>,
}

/// Inference comparison configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceCompareConfig {
    /// Enable inference comparison
    #[serde(default, deserialize_with = "deserialize_bool_or_string")]
    pub enabled: bool,
    /// Prompt to use for comparison
    #[serde(default)]
    pub prompt: Option<String>,
    /// Maximum tokens to generate
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    /// Tolerance for logit comparison
    #[serde(default = "default_tolerance")]
    pub tolerance: f64,
    /// Gates to verify
    #[serde(default)]
    pub gates: Vec<String>,
}

fn default_max_tokens() -> usize {
    10
}

fn default_tolerance() -> f64 {
    1e-5
}

/// Fingerprint configuration (PMAT-201, JAX-STAT-001)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FingerprintConfig {
    /// Enable fingerprint testing
    #[serde(default, deserialize_with = "deserialize_bool_or_string")]
    pub enabled: bool,
    /// Tensors to fingerprint ("all" or comma-separated list)
    #[serde(default = "default_fingerprint_tensors")]
    pub tensors: String,
    /// Statistics to compute
    #[serde(default = "default_fingerprint_stats")]
    pub stats: Vec<String>,
    /// Gates to verify
    #[serde(default)]
    pub gates: Vec<String>,
}

fn default_fingerprint_tensors() -> String {
    "all".to_string()
}

fn default_fingerprint_stats() -> Vec<String> {
    vec![
        "mean".to_string(),
        "std".to_string(),
        "min".to_string(),
        "max".to_string(),
        "checksum".to_string(),
    ]
}

/// Validate stats configuration (PMAT-202)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidateStatsConfig {
    /// Enable stats validation
    #[serde(default, deserialize_with = "deserialize_bool_or_string")]
    pub enabled: bool,
    /// Reference file for comparison
    #[serde(default)]
    pub reference: Option<String>,
    /// Role-specific tolerances
    #[serde(default)]
    pub tolerance: StatsToleranceConfig,
    /// Gates to verify
    #[serde(default)]
    pub gates: Vec<String>,
}

/// Per-role tolerance configuration for validate-stats
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StatsToleranceConfig {
    /// Tolerance for LayerNorm tensors (strict)
    #[serde(default = "default_layernorm_tolerance")]
    pub layernorm: f64,
    /// Tolerance for embedding tensors (loose)
    #[serde(default = "default_embedding_tolerance")]
    pub embedding: f64,
    /// Tolerance for attention tensors (medium)
    #[serde(default = "default_attention_tolerance")]
    pub attention: f64,
}

fn default_layernorm_tolerance() -> f64 {
    0.001
}

fn default_embedding_tolerance() -> f64 {
    0.1
}

fn default_attention_tolerance() -> f64 {
    0.01
}

/// Profile CI configuration (PMAT-192)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileCiConfig {
    /// Enable profile CI
    #[serde(default, deserialize_with = "deserialize_bool_or_string")]
    pub enabled: bool,
    /// Warmup iterations
    #[serde(default = "default_warmup")]
    pub warmup: usize,
    /// Measurement iterations
    #[serde(default = "default_measure")]
    pub measure: usize,
    /// Formats to profile (default: all available)
    #[serde(default = "default_profile_formats")]
    pub formats: Vec<String>,
    /// Backends to profile (default: [cpu, gpu])
    #[serde(default = "default_profile_backends")]
    pub backends: Vec<String>,
    /// Assertions to verify
    #[serde(default)]
    pub assertions: ProfileCiAssertions,
    /// Gates to verify
    #[serde(default)]
    pub gates: Vec<String>,
}

fn default_profile_formats() -> Vec<String> {
    vec![
        "gguf".to_string(),
        "apr".to_string(),
        "safetensors".to_string(),
    ]
}

fn default_profile_backends() -> Vec<String> {
    vec!["cpu".to_string(), "gpu".to_string()]
}

fn default_warmup() -> usize {
    3
}

fn default_measure() -> usize {
    10
}

/// Profile CI assertions
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProfileCiAssertions {
    /// Minimum throughput in tok/s (legacy, applies to all)
    #[serde(default)]
    pub min_throughput: Option<f64>,
    /// Minimum CPU throughput in tok/s
    #[serde(default)]
    pub min_throughput_cpu: Option<f64>,
    /// Minimum GPU throughput in tok/s
    #[serde(default)]
    pub min_throughput_gpu: Option<f64>,
    /// Maximum p99 latency in ms
    #[serde(default)]
    pub max_p99_ms: Option<f64>,
    /// Maximum p50 latency in ms
    #[serde(default)]
    pub max_p50_ms: Option<f64>,
}

impl ProfileCiAssertions {
    /// Get minimum throughput for a given backend
    #[must_use]
    pub fn min_throughput_for(&self, backend: &str) -> Option<f64> {
        match backend {
            "cpu" => self.min_throughput_cpu.or(self.min_throughput),
            "gpu" => self.min_throughput_gpu.or(self.min_throughput),
            _ => self.min_throughput,
        }
    }
}

/// Trace payload configuration (APR-TRACE-001)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TracePayloadConfig {
    /// Enable trace payload
    #[serde(default, deserialize_with = "deserialize_bool_or_string")]
    pub enabled: bool,
    /// Prompt for trace
    #[serde(default)]
    pub prompt: Option<String>,
    /// Gates to verify
    #[serde(default)]
    pub gates: Vec<String>,
}

/// Ollama parity configuration (GH-6/AC-2)
///
/// Tests that APR inference output matches ollama for the same model/quant.
/// Catches format-specific regressions by comparing against an independent runtime.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaParityConfig {
    /// Enable ollama parity testing
    #[serde(default, deserialize_with = "deserialize_bool_or_string")]
    pub enabled: bool,
    /// Ollama model tag (e.g., "qwen2.5-coder:7b-instruct-q4_k_m")
    #[serde(default)]
    pub model_tag: Option<String>,
    /// Quantizations to test (each maps to an ollama tag suffix)
    #[serde(default = "default_ollama_quantizations")]
    pub quantizations: Vec<String>,
    /// Prompts to test parity on
    #[serde(default = "default_ollama_prompts")]
    pub prompts: Vec<String>,
    /// Inference temperature (0.0 for deterministic)
    #[serde(default)]
    pub temperature: f64,
    /// Minimum performance ratio (APR tok/s / ollama tok/s)
    #[serde(default = "default_min_perf_ratio")]
    pub min_perf_ratio: f64,
    /// Gates to verify
    #[serde(default)]
    pub gates: Vec<String>,
}

fn default_ollama_quantizations() -> Vec<String> {
    vec!["q4_k_m".to_string()]
}

fn default_ollama_prompts() -> Vec<String> {
    vec!["What is 2+2?".to_string()]
}

fn default_min_perf_ratio() -> f64 {
    0.8
}

// ── Playbook Integrity Lock (§3.1) ──────────────────────────────────────

/// A single entry in the playbook lock file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlaybookLockEntry {
    /// SHA-256 hash of the playbook file
    pub sha256: String,
    /// Fields that are locked (changing them requires re-approval)
    pub locked_fields: Vec<String>,
}

/// Lock file mapping playbook names to their integrity entries
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PlaybookLockFile {
    /// Map of playbook name → lock entry
    pub entries: HashMap<String, PlaybookLockEntry>,
}

/// Compute SHA-256 hash of a playbook file
///
/// # Errors
///
/// Returns an error if the file cannot be read.
pub fn compute_playbook_hash(path: impl AsRef<Path>) -> Result<String> {
    use sha2::{Digest, Sha256};
    let content = std::fs::read(path)?;
    let mut hasher = Sha256::new();
    hasher.update(&content);
    Ok(format!("{:x}", hasher.finalize()))
}

/// Load a lock file from YAML
///
/// # Errors
///
/// Returns an error if the file cannot be read or parsed.
pub fn load_lock_file(path: impl AsRef<Path>) -> Result<PlaybookLockFile> {
    let content = std::fs::read_to_string(path)?;
    serde_yaml::from_str(&content).map_err(Error::from)
}

/// Save a lock file to YAML
///
/// # Errors
///
/// Returns an error if serialization or writing fails.
pub fn save_lock_file(lock: &PlaybookLockFile, path: impl AsRef<Path>) -> Result<()> {
    let yaml = serde_yaml::to_string(lock).map_err(Error::from)?;
    std::fs::write(path, yaml)?;
    Ok(())
}

/// Verify a playbook's integrity against the lock file
///
/// # Errors
///
/// Returns an error if the hash does not match or if file operations fail.
pub fn verify_playbook_integrity(
    playbook_path: impl AsRef<Path>,
    lock_file: &PlaybookLockFile,
    name: &str,
) -> Result<()> {
    let entry = lock_file
        .entries
        .get(name)
        .ok_or_else(|| Error::Execution(format!("Playbook '{name}' not found in lock file")))?;

    let current_hash = compute_playbook_hash(&playbook_path)?;
    if current_hash != entry.sha256 {
        return Err(Error::Execution(format!(
            "Integrity check failed for '{name}': expected {}, got {current_hash}",
            entry.sha256
        )));
    }

    Ok(())
}

/// Generate a lock entry for a playbook file
///
/// # Errors
///
/// Returns an error if the file cannot be read.
pub fn generate_lock_entry(path: impl AsRef<Path>) -> Result<(String, PlaybookLockEntry)> {
    let path_ref = path.as_ref();
    let name = path_ref
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .to_string();

    // Strip common suffixes like ".playbook"
    let name = name.strip_suffix(".playbook").unwrap_or(&name).to_string();

    let sha256 = compute_playbook_hash(path_ref)?;

    let entry = PlaybookLockEntry {
        sha256,
        locked_fields: vec![
            "model.hf_repo".to_string(),
            "model.formats".to_string(),
            "test_matrix".to_string(),
            "falsification_gates".to_string(),
        ],
    };

    Ok((name, entry))
}

// ── Skip Mechanism (§3.3) ──────────────────────────────────────────────

/// Reason for skipping a test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkipReason {
    /// Format or backend being skipped
    pub format_or_backend: String,
    /// Why it's skipped
    pub reason: String,
    /// Tracking issue (e.g., "GH-123")
    pub tracking_issue: Option<String>,
}

/// Type of skip
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SkipType {
    /// Explicitly declared via .skip file
    Explicit,
    /// Implicitly missing from the format list
    Implicit,
}

/// Find skip files for a given playbook
///
/// Looks for `<playbook_dir>/<name>.skip.yaml` files.
#[must_use]
pub fn find_skip_files(playbook_dir: &Path, name: &str) -> Vec<SkipReason> {
    let skip_path = playbook_dir.join(format!("{name}.skip.yaml"));
    if !skip_path.exists() {
        return Vec::new();
    }

    let Ok(content) = std::fs::read_to_string(&skip_path) else {
        return Vec::new();
    };

    serde_yaml::from_str(&content).unwrap_or_default()
}

/// Detect implicit skips by comparing playbook formats against all known formats
#[must_use]
pub fn detect_implicit_skips(
    playbook: &Playbook,
    all_formats: &[Format],
    skip_files: &[SkipReason],
) -> Vec<String> {
    let mut implicit = Vec::new();
    let explicit_formats: Vec<&str> = skip_files
        .iter()
        .map(|s| s.format_or_backend.as_str())
        .collect();

    for format in all_formats {
        let format_str = format!("{format:?}").to_lowercase();
        if !playbook.model.formats.contains(format)
            && !explicit_formats.contains(&format_str.as_str())
        {
            implicit.push(format_str);
        }
    }

    implicit
}


#[cfg(test)]
#[path = "playbook_tests.rs"]
mod tests;
