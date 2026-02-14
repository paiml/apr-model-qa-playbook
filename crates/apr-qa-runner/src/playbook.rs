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
// Tier patterns: mvp, smoke, quick, ci, full, nightly, release

/// Regex pattern for playbook naming convention
/// Matches: {family}-{size}[-{tier}].playbook.yaml
/// - family: one or more segments separated by `-` (letters, digits, dots)
/// - size: digits optionally with decimal, followed by `b` (e.g., 0.5b, 1b, 7b)
/// - tier (optional): mvp, smoke, quick, ci, full, nightly, release
static PLAYBOOK_NAME_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    // This regex pattern is verified at compile time, unwrap is safe here
    #[allow(clippy::unwrap_used)]
    Regex::new(
        r"^(?P<family>(?:[a-z0-9]+\.?)+(?:-[a-z0-9]+\.?)*)-(?P<size>\d+(?:\.\d+)?b)(?:-(?P<tier>mvp|smoke|quick|ci|full|nightly|release))?\.playbook\.yaml$"
    ).unwrap()
});

/// Valid tier values for playbook naming
pub const VALID_TIERS: &[&str] = &["mvp", "smoke", "quick", "ci", "full", "nightly", "release"];

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
        self.expected_num_heads = Some(variant.num_heads);
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
        num_heads: u32,
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

        if let Some(expected) = self.expected_num_heads {
            if expected != num_heads {
                mismatches.push(format!(
                    "num_heads mismatch: expected {expected}, got {num_heads}"
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
mod tests {
    use super::*;

    #[test]
    fn test_default_formats() {
        let formats = default_formats();
        assert_eq!(formats.len(), 3);
        assert!(formats.contains(&Format::Gguf));
        assert!(formats.contains(&Format::SafeTensors));
        assert!(formats.contains(&Format::Apr));
    }

    #[test]
    fn test_default_quantizations() {
        let quants = default_quantizations();
        assert_eq!(quants, vec!["q4_k_m"]);
    }

    #[test]
    fn test_default_modalities() {
        let modalities = default_modalities();
        assert_eq!(modalities.len(), 3);
        assert!(modalities.contains(&Modality::Run));
        assert!(modalities.contains(&Modality::Chat));
        assert!(modalities.contains(&Modality::Serve));
    }

    #[test]
    fn test_default_backends() {
        let backends = default_backends();
        assert_eq!(backends.len(), 2);
        assert!(backends.contains(&Backend::Cpu));
        assert!(backends.contains(&Backend::Gpu));
    }

    #[test]
    fn test_default_scenario_count() {
        assert_eq!(default_scenario_count(), 100);
    }

    #[test]
    fn test_default_proptest_count() {
        assert_eq!(default_proptest_count(), 100);
    }

    #[test]
    fn test_default_timeout() {
        assert_eq!(default_timeout(), 60000);
    }

    #[test]
    fn test_default_severity() {
        assert_eq!(default_severity(), "P1");
    }

    #[test]
    fn test_test_matrix_default() {
        let matrix = TestMatrix::default();
        assert_eq!(matrix.modalities.len(), 3);
        assert_eq!(matrix.backends.len(), 2);
        assert_eq!(matrix.scenario_count, 100);
    }

    #[test]
    fn test_playbook_to_yaml() {
        let yaml = r#"
name: test-playbook
version: "1.0.0"
model:
  hf_repo: "test/model"
  formats: [gguf]
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 5
"#;
        let playbook = Playbook::from_yaml(yaml).expect("Failed to parse");
        let output = playbook.to_yaml().expect("Failed to serialize");
        assert!(output.contains("test-playbook"));
        assert!(output.contains("test/model"));
    }

    #[test]
    fn test_playbook_with_defaults() {
        // Test playbook that uses default values for model config
        let yaml = r#"
name: minimal
version: "1.0.0"
model:
  hf_repo: "org/model"
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 100
"#;
        let playbook = Playbook::from_yaml(yaml).expect("Failed to parse");
        assert_eq!(playbook.model.formats.len(), 3);
        assert_eq!(playbook.model.quantizations, vec!["q4_k_m"]);
        assert_eq!(playbook.test_matrix.scenario_count, 100);
    }

    #[test]
    fn test_playbook_with_state_machine() {
        let yaml = r#"
name: state-test
version: "1.0.0"
model:
  hf_repo: "test/model"
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
state_machine:
  initial: "ready"
  states:
    ready:
      on_enter:
        - action: "log 'entering ready'"
      transitions:
        - event: "start"
          target: "running"
          action: "initialize"
          guards:
            - "model_loaded"
    running:
      on_exit:
        - action: "cleanup"
"#;
        let playbook = Playbook::from_yaml(yaml).expect("Failed to parse");
        let state_machine = playbook.state_machine.expect("Should have state machine");
        assert_eq!(state_machine.initial, "ready");
        assert_eq!(state_machine.states.len(), 2);

        let ready_state = &state_machine.states["ready"];
        assert_eq!(ready_state.on_enter.len(), 1);
        assert_eq!(ready_state.transitions.len(), 1);

        let transition = &ready_state.transitions[0];
        assert_eq!(transition.event, "start");
        assert_eq!(transition.target, "running");
        assert!(transition.action.is_some());
        assert_eq!(transition.guards.len(), 1);
    }

    #[test]
    fn test_playbook_with_property_tests() {
        let yaml = r#"
name: prop-test
version: "1.0.0"
model:
  hf_repo: "test/model"
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
property_tests:
  - name: "arithmetic"
    generator: "random_arithmetic"
    oracle: "check_arithmetic"
    count: 50
  - name: "code"
    generator: "random_code"
    oracle: "check_code"
"#;
        let playbook = Playbook::from_yaml(yaml).expect("Failed to parse");
        assert_eq!(playbook.property_tests.len(), 2);

        let first = &playbook.property_tests[0];
        assert_eq!(first.name, "arithmetic");
        assert_eq!(first.count, 50);

        let second = &playbook.property_tests[1];
        assert_eq!(second.name, "code");
        assert_eq!(second.count, 100); // default
    }

    #[test]
    fn test_playbook_with_falsification_gates() {
        let yaml = r#"
name: gate-test
version: "1.0.0"
model:
  hf_repo: "test/model"
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
falsification_gates:
  - id: F-QUAL-001
    description: "Output is valid"
    condition: "output.len() > 0"
    severity: P0
  - id: F-QUAL-002
    description: "No errors"
    condition: "!output.contains('error')"
"#;
        let playbook = Playbook::from_yaml(yaml).expect("Failed to parse");
        assert_eq!(playbook.falsification_gates.len(), 2);

        let first = &playbook.falsification_gates[0];
        assert_eq!(first.severity, "P0");

        let second = &playbook.falsification_gates[1];
        assert_eq!(second.severity, "P1"); // default
    }

    #[test]
    fn test_model_config_no_slash() {
        let config = ModelConfig {
            hf_repo: "model-name".to_string(),
            local_path: None,
            formats: vec![Format::Gguf],
            quantizations: vec![],
            size_category: SizeCategory::default(),
            expected_hidden_dim: None,
            expected_num_layers: None,
            expected_num_heads: None,
            expected_num_kv_heads: None,
            expected_vocab_size: None,
            expected_intermediate_dim: None,
            family: None,
            size_variant: None,
        };
        assert_eq!(config.hf_org(), "model-name");
        assert_eq!(config.hf_name(), "model-name");
    }

    #[test]
    fn test_model_config_with_local_path() {
        let config = ModelConfig {
            hf_repo: "org/model".to_string(),
            local_path: Some("/path/to/model".to_string()),
            formats: default_formats(),
            quantizations: default_quantizations(),
            size_category: SizeCategory::default(),
            expected_hidden_dim: None,
            expected_num_layers: None,
            expected_num_heads: None,
            expected_num_kv_heads: None,
            expected_vocab_size: None,
            expected_intermediate_dim: None,
            family: None,
            size_variant: None,
        };
        assert!(config.local_path.is_some());
    }

    #[test]
    fn test_playbook_step() {
        let step = PlaybookStep {
            name: "test-step".to_string(),
            command: "echo test".to_string(),
            timeout_ms: default_timeout(),
            expected_exit_code: 0,
            expected_patterns: vec!["test".to_string()],
            forbidden_patterns: vec!["error".to_string()],
        };
        assert_eq!(step.timeout_ms, 60000);
        assert_eq!(step.expected_exit_code, 0);
    }

    #[test]
    fn test_playbook_parse() {
        let yaml = r#"
name: test-playbook
version: "1.0.0"
model:
  hf_repo: "Qwen/Qwen2.5-Coder-1.5B-Instruct"
  formats: [gguf, safetensors]
test_matrix:
  modalities: [run, chat]
  backends: [cpu]
  scenario_count: 10
falsification_gates:
  - id: F-TEST-001
    description: "Output is non-empty"
    condition: "output.len() > 0"
"#;

        let playbook = Playbook::from_yaml(yaml).expect("Failed to parse playbook");
        assert_eq!(playbook.name, "test-playbook");
        assert_eq!(playbook.model.hf_repo, "Qwen/Qwen2.5-Coder-1.5B-Instruct");
        assert_eq!(playbook.test_matrix.modalities.len(), 2);
        assert_eq!(playbook.falsification_gates.len(), 1);
    }

    #[test]
    fn test_playbook_generate_scenarios() {
        let yaml = r#"
name: test-playbook
version: "1.0.0"
model:
  hf_repo: "test/model"
  formats: [gguf]
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 5
"#;

        let playbook = Playbook::from_yaml(yaml).expect("Failed to parse");
        let scenarios = playbook.generate_scenarios();

        // 1 modality × 1 backend × 1 format × 5 scenarios = 5
        assert_eq!(scenarios.len(), 5);
    }

    #[test]
    fn test_model_config_parse() {
        let config = ModelConfig {
            hf_repo: "Qwen/Qwen2.5-Coder-1.5B-Instruct".to_string(),
            local_path: None,
            formats: vec![Format::Gguf],
            quantizations: vec!["q4_k_m".to_string()],
            size_category: SizeCategory::Small,
            expected_hidden_dim: None,
            expected_num_layers: None,
            expected_num_heads: None,
            expected_num_kv_heads: None,
            expected_vocab_size: None,
            expected_intermediate_dim: None,
            family: None,
            size_variant: None,
        };

        assert_eq!(config.hf_org(), "Qwen");
        assert_eq!(config.hf_name(), "Qwen2.5-Coder-1.5B-Instruct");
    }

    #[test]
    fn test_total_tests() {
        let yaml = r#"
name: test
version: "1.0.0"
model:
  hf_repo: "test/model"
  formats: [gguf, safetensors, apr]
test_matrix:
  modalities: [run, chat, serve]
  backends: [cpu, gpu]
  scenario_count: 100
"#;

        let playbook = Playbook::from_yaml(yaml).expect("Failed to parse");
        // 3 modalities × 2 backends × 3 formats × 100 = 1800
        assert_eq!(playbook.total_tests(), 1800);
    }

    #[test]
    fn test_playbook_with_differential_tests() {
        let yaml = r#"
name: diff-test
version: "1.0.0"
model:
  hf_repo: "test/model"
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
differential_tests:
  tensor_diff:
    enabled: true
    filter: "embed,lm_head"
    gates: ["F-ROSETTA-DIFF-001", "F-ROSETTA-DIFF-002"]
  inference_compare:
    enabled: true
    prompt: "What is 2+2?"
    max_tokens: 10
    tolerance: 0.00001
    gates: ["F-ROSETTA-INF-001"]
"#;
        let playbook = Playbook::from_yaml(yaml).expect("Failed to parse");
        let diff = playbook
            .differential_tests
            .expect("Should have differential tests");

        let tensor = diff.tensor_diff.expect("Should have tensor diff");
        assert!(tensor.enabled);
        assert_eq!(tensor.filter, Some("embed,lm_head".to_string()));
        assert_eq!(tensor.gates.len(), 2);

        let inf = diff
            .inference_compare
            .expect("Should have inference compare");
        assert!(inf.enabled);
        assert_eq!(inf.prompt, Some("What is 2+2?".to_string()));
        assert_eq!(inf.max_tokens, 10);
    }

    #[test]
    fn test_playbook_with_profile_ci() {
        let yaml = r#"
name: profile-test
version: "1.0.0"
model:
  hf_repo: "test/model"
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
profile_ci:
  enabled: true
  warmup: 5
  measure: 20
  assertions:
    min_throughput: 10.0
    max_p99_ms: 500.0
    max_p50_ms: 200.0
  gates: ["F-PROFILE-CI-001", "F-PROFILE-CI-002"]
"#;
        let playbook = Playbook::from_yaml(yaml).expect("Failed to parse");
        let profile = playbook.profile_ci.expect("Should have profile CI");

        assert!(profile.enabled);
        assert_eq!(profile.warmup, 5);
        assert_eq!(profile.measure, 20);
        assert_eq!(profile.assertions.min_throughput, Some(10.0));
        assert_eq!(profile.assertions.max_p99_ms, Some(500.0));
        assert_eq!(profile.assertions.max_p50_ms, Some(200.0));
        assert_eq!(profile.gates.len(), 2);
    }

    #[test]
    fn test_playbook_with_trace_payload() {
        let yaml = r#"
name: trace-test
version: "1.0.0"
model:
  hf_repo: "test/model"
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
trace_payload:
  enabled: true
  prompt: "Test prompt"
  gates: ["F-TRACE-PAYLOAD-001", "F-TRACE-PAYLOAD-002"]
"#;
        let playbook = Playbook::from_yaml(yaml).expect("Failed to parse");
        let trace = playbook.trace_payload.expect("Should have trace payload");

        assert!(trace.enabled);
        assert_eq!(trace.prompt, Some("Test prompt".to_string()));
        assert_eq!(trace.gates.len(), 2);
    }

    #[test]
    fn test_default_max_tokens() {
        assert_eq!(default_max_tokens(), 10);
    }

    #[test]
    fn test_default_tolerance() {
        assert!((default_tolerance() - 1e-5).abs() < 1e-10);
    }

    #[test]
    fn test_default_warmup() {
        assert_eq!(default_warmup(), 3);
    }

    #[test]
    fn test_default_measure() {
        assert_eq!(default_measure(), 10);
    }

    #[test]
    fn test_playbook_with_fingerprint() {
        let yaml = r#"
name: fingerprint-test
version: "1.0.0"
model:
  hf_repo: "test/model"
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
differential_tests:
  fingerprint:
    enabled: true
    tensors: "embed,lm_head"
    stats: ["mean", "std", "checksum"]
    gates: ["F-ROSETTA-FP-001", "F-ROSETTA-FP-002"]
"#;
        let playbook = Playbook::from_yaml(yaml).expect("Failed to parse");
        let diff = playbook
            .differential_tests
            .expect("Should have differential tests");

        let fp = diff.fingerprint.expect("Should have fingerprint");
        assert!(fp.enabled);
        assert_eq!(fp.tensors, "embed,lm_head");
        assert_eq!(fp.stats.len(), 3);
        assert_eq!(fp.gates.len(), 2);
    }

    #[test]
    fn test_playbook_with_validate_stats() {
        let yaml = r#"
name: validate-stats-test
version: "1.0.0"
model:
  hf_repo: "test/model"
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
differential_tests:
  validate_stats:
    enabled: true
    reference: "reference.json"
    tolerance:
      layernorm: 0.001
      embedding: 0.1
      attention: 0.01
    gates: ["F-ROSETTA-STATS-001", "F-ROSETTA-STATS-002"]
"#;
        let playbook = Playbook::from_yaml(yaml).expect("Failed to parse");
        let diff = playbook
            .differential_tests
            .expect("Should have differential tests");

        let stats = diff.validate_stats.expect("Should have validate_stats");
        assert!(stats.enabled);
        assert_eq!(stats.reference, Some("reference.json".to_string()));
        assert!((stats.tolerance.layernorm - 0.001).abs() < 1e-10);
        assert!((stats.tolerance.embedding - 0.1).abs() < 1e-10);
        assert!((stats.tolerance.attention - 0.01).abs() < 1e-10);
        assert_eq!(stats.gates.len(), 2);
    }

    #[test]
    fn test_default_fingerprint_tensors() {
        assert_eq!(default_fingerprint_tensors(), "all");
    }

    #[test]
    fn test_default_fingerprint_stats() {
        let stats = default_fingerprint_stats();
        assert_eq!(stats.len(), 5);
        assert!(stats.contains(&"mean".to_string()));
        assert!(stats.contains(&"checksum".to_string()));
    }

    #[test]
    fn test_default_tolerance_values() {
        assert!((default_layernorm_tolerance() - 0.001).abs() < 1e-10);
        assert!((default_embedding_tolerance() - 0.1).abs() < 1e-10);
        assert!((default_attention_tolerance() - 0.01).abs() < 1e-10);
    }

    #[test]
    fn test_profile_ci_min_throughput_for() {
        // Test with all fields set
        let assertions = ProfileCiAssertions {
            min_throughput: Some(10.0),
            min_throughput_cpu: Some(5.0),
            min_throughput_gpu: Some(50.0),
            max_p99_ms: None,
            max_p50_ms: None,
        };

        assert_eq!(assertions.min_throughput_for("cpu"), Some(5.0));
        assert_eq!(assertions.min_throughput_for("gpu"), Some(50.0));
        assert_eq!(assertions.min_throughput_for("tpu"), Some(10.0));

        // Test with only min_throughput set (fallback)
        let assertions_fallback = ProfileCiAssertions {
            min_throughput: Some(20.0),
            min_throughput_cpu: None,
            min_throughput_gpu: None,
            max_p99_ms: None,
            max_p50_ms: None,
        };

        assert_eq!(assertions_fallback.min_throughput_for("cpu"), Some(20.0));
        assert_eq!(assertions_fallback.min_throughput_for("gpu"), Some(20.0));

        // Test with nothing set
        let assertions_none = ProfileCiAssertions {
            min_throughput: None,
            min_throughput_cpu: None,
            min_throughput_gpu: None,
            max_p99_ms: None,
            max_p50_ms: None,
        };

        assert_eq!(assertions_none.min_throughput_for("cpu"), None);
        assert_eq!(assertions_none.min_throughput_for("gpu"), None);
    }

    // ── §3.1 Playbook integrity lock tests ─────────────────────────────

    #[test]
    fn test_compute_playbook_hash_consistent() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("test.playbook.yaml");
        std::fs::write(&path, "name: test\nversion: 1.0").expect("write");

        let hash1 = compute_playbook_hash(&path).expect("hash1");
        let hash2 = compute_playbook_hash(&path).expect("hash2");
        assert_eq!(hash1, hash2);
        assert_eq!(hash1.len(), 64); // SHA-256 hex
    }

    #[test]
    fn test_compute_playbook_hash_differs() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let path1 = dir.path().join("a.yaml");
        let path2 = dir.path().join("b.yaml");
        std::fs::write(&path1, "content-a").expect("write");
        std::fs::write(&path2, "content-b").expect("write");

        let hash1 = compute_playbook_hash(&path1).expect("hash1");
        let hash2 = compute_playbook_hash(&path2).expect("hash2");
        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_verify_playbook_integrity_pass() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("test.playbook.yaml");
        std::fs::write(&path, "name: test\nversion: 1.0").expect("write");

        let hash = compute_playbook_hash(&path).expect("hash");
        let mut lock = PlaybookLockFile::default();
        lock.entries.insert(
            "test".to_string(),
            PlaybookLockEntry {
                sha256: hash,
                locked_fields: vec!["model.hf_repo".to_string()],
            },
        );

        assert!(verify_playbook_integrity(&path, &lock, "test").is_ok());
    }

    #[test]
    fn test_verify_playbook_integrity_fail_mismatch() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("test.playbook.yaml");
        std::fs::write(&path, "name: test\nversion: 1.0").expect("write");

        let mut lock = PlaybookLockFile::default();
        lock.entries.insert(
            "test".to_string(),
            PlaybookLockEntry {
                sha256: "wrong_hash".to_string(),
                locked_fields: vec![],
            },
        );

        let result = verify_playbook_integrity(&path, &lock, "test");
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Integrity check failed")
        );
    }

    #[test]
    fn test_verify_playbook_integrity_missing_entry() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("test.playbook.yaml");
        std::fs::write(&path, "name: test").expect("write");

        let lock = PlaybookLockFile::default();
        let result = verify_playbook_integrity(&path, &lock, "test");
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("not found in lock file")
        );
    }

    #[test]
    fn test_generate_lock_entry() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("my-model.playbook.yaml");
        std::fs::write(&path, "name: my-model\nversion: 1.0").expect("write");

        let (name, entry) = generate_lock_entry(&path).expect("generate");
        assert_eq!(name, "my-model");
        assert_eq!(entry.sha256.len(), 64);
        assert!(!entry.locked_fields.is_empty());
    }

    #[test]
    fn test_lock_file_save_load_roundtrip() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let lock_path = dir.path().join("playbook.lock.yaml");

        let mut lock = PlaybookLockFile::default();
        lock.entries.insert(
            "model-a".to_string(),
            PlaybookLockEntry {
                sha256: "abc123".to_string(),
                locked_fields: vec!["model.hf_repo".to_string()],
            },
        );

        save_lock_file(&lock, &lock_path).expect("save");
        let loaded = load_lock_file(&lock_path).expect("load");

        assert_eq!(loaded.entries.len(), 1);
        assert_eq!(loaded.entries["model-a"].sha256, "abc123");
    }

    #[test]
    fn test_lock_file_serde_roundtrip() {
        let mut lock = PlaybookLockFile::default();
        lock.entries.insert(
            "test".to_string(),
            PlaybookLockEntry {
                sha256: "deadbeef".to_string(),
                locked_fields: vec!["a".to_string(), "b".to_string()],
            },
        );

        let yaml = serde_yaml::to_string(&lock).expect("serialize");
        let parsed: PlaybookLockFile = serde_yaml::from_str(&yaml).expect("deserialize");
        assert_eq!(parsed.entries["test"].sha256, "deadbeef");
        assert_eq!(parsed.entries["test"].locked_fields.len(), 2);
    }

    // ── §3.3 Skip mechanism tests ──────────────────────────────────────

    #[test]
    fn test_find_skip_files_empty_dir() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let skips = find_skip_files(dir.path(), "test-model");
        assert!(skips.is_empty());
    }

    #[test]
    fn test_find_skip_files_with_skip() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let skip_path = dir.path().join("test-model.skip.yaml");
        std::fs::write(
            &skip_path,
            r#"- format_or_backend: gpu
  reason: "No GPU available"
  tracking_issue: "GH-123"
"#,
        )
        .expect("write");

        let skips = find_skip_files(dir.path(), "test-model");
        assert_eq!(skips.len(), 1);
        assert_eq!(skips[0].format_or_backend, "gpu");
        assert_eq!(skips[0].tracking_issue.as_deref(), Some("GH-123"));
    }

    #[test]
    fn test_detect_implicit_skips() {
        let yaml = r#"
name: test
version: "1.0.0"
model:
  hf_repo: "test/model"
  formats: [gguf]
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
"#;
        let playbook = Playbook::from_yaml(yaml).expect("parse");
        let all = vec![Format::Gguf, Format::SafeTensors, Format::Apr];
        let skips: Vec<SkipReason> = vec![];
        let implicit = detect_implicit_skips(&playbook, &all, &skips);
        // safetensors and apr are missing from playbook formats
        assert_eq!(implicit.len(), 2);
        assert!(implicit.contains(&"safetensors".to_string()));
        assert!(implicit.contains(&"apr".to_string()));
    }

    #[test]
    fn test_detect_implicit_skips_with_explicit() {
        let yaml = r#"
name: test
version: "1.0.0"
model:
  hf_repo: "test/model"
  formats: [gguf]
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
"#;
        let playbook = Playbook::from_yaml(yaml).expect("parse");
        let all = vec![Format::Gguf, Format::SafeTensors, Format::Apr];
        // safetensors is explicitly skipped
        let skips = vec![SkipReason {
            format_or_backend: "safetensors".to_string(),
            reason: "Not supported".to_string(),
            tracking_issue: None,
        }];
        let implicit = detect_implicit_skips(&playbook, &all, &skips);
        // Only apr is implicitly skipped
        assert_eq!(implicit.len(), 1);
        assert_eq!(implicit[0], "apr");
    }

    #[test]
    fn test_detect_implicit_skips_all_covered() {
        let yaml = r#"
name: test
version: "1.0.0"
model:
  hf_repo: "test/model"
  formats: [gguf, safetensors, apr]
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
"#;
        let playbook = Playbook::from_yaml(yaml).expect("parse");
        let all = vec![Format::Gguf, Format::SafeTensors, Format::Apr];
        let skips: Vec<SkipReason> = vec![];
        let implicit = detect_implicit_skips(&playbook, &all, &skips);
        assert!(implicit.is_empty());
    }

    #[test]
    fn test_skip_reason_serde() {
        let reason = SkipReason {
            format_or_backend: "gpu".to_string(),
            reason: "No GPU".to_string(),
            tracking_issue: Some("GH-100".to_string()),
        };
        let json = serde_json::to_string(&reason).expect("serialize");
        let parsed: SkipReason = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.format_or_backend, "gpu");
        assert_eq!(parsed.tracking_issue.as_deref(), Some("GH-100"));
    }

    #[test]
    fn test_skip_type_eq() {
        assert_eq!(SkipType::Explicit, SkipType::Explicit);
        assert_ne!(SkipType::Explicit, SkipType::Implicit);
    }

    // ── §3.4 Resource-aware scheduling tests ────────────────────────────

    #[test]
    fn test_size_category_max_workers() {
        assert_eq!(SizeCategory::Tiny.max_workers(), 4);
        assert_eq!(SizeCategory::Small.max_workers(), 4);
        assert_eq!(SizeCategory::Medium.max_workers(), 2);
        assert_eq!(SizeCategory::Large.max_workers(), 1);
        assert_eq!(SizeCategory::Xlarge.max_workers(), 1);
        assert_eq!(SizeCategory::Huge.max_workers(), 1);
    }

    #[test]
    fn test_size_category_estimated_memory() {
        assert_eq!(SizeCategory::Tiny.estimated_memory_gb(), 2);
        assert_eq!(SizeCategory::Small.estimated_memory_gb(), 4);
        assert_eq!(SizeCategory::Medium.estimated_memory_gb(), 8);
        assert_eq!(SizeCategory::Large.estimated_memory_gb(), 16);
        assert_eq!(SizeCategory::Xlarge.estimated_memory_gb(), 32);
        assert_eq!(SizeCategory::Huge.estimated_memory_gb(), 64);
    }

    #[test]
    fn test_size_category_can_run_concurrent() {
        assert!(SizeCategory::Tiny.can_run_concurrent());
        assert!(SizeCategory::Small.can_run_concurrent());
        assert!(!SizeCategory::Medium.can_run_concurrent());
        assert!(!SizeCategory::Large.can_run_concurrent());
        assert!(!SizeCategory::Xlarge.can_run_concurrent());
        assert!(!SizeCategory::Huge.can_run_concurrent());
    }

    #[test]
    fn test_size_category_default() {
        assert_eq!(SizeCategory::default(), SizeCategory::Tiny);
    }

    #[test]
    fn test_size_category_serde() {
        let yaml = r#"
name: test
version: "1.0.0"
model:
  hf_repo: "test/model"
  size_category: large
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
"#;
        let playbook = Playbook::from_yaml(yaml).expect("parse");
        assert_eq!(playbook.model.size_category, SizeCategory::Large);
    }

    #[test]
    fn test_effective_max_workers_respects_size() {
        let yaml = r#"
name: test
version: "1.0.0"
model:
  hf_repo: "test/model"
  size_category: large
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
"#;
        let playbook = Playbook::from_yaml(yaml).expect("parse");
        // Large model caps at 1 worker regardless of request
        assert_eq!(playbook.effective_max_workers(4), 1);
        assert_eq!(playbook.effective_max_workers(8), 1);
        assert_eq!(playbook.effective_max_workers(1), 1);
    }

    #[test]
    fn test_effective_max_workers_small_model() {
        let yaml = r#"
name: test
version: "1.0.0"
model:
  hf_repo: "test/model"
  size_category: small
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
"#;
        let playbook = Playbook::from_yaml(yaml).expect("parse");
        // Small model allows up to 4 workers
        assert_eq!(playbook.effective_max_workers(4), 4);
        assert_eq!(playbook.effective_max_workers(8), 4); // capped at 4
        assert_eq!(playbook.effective_max_workers(2), 2); // respects lower request
    }

    #[test]
    fn test_effective_max_workers_medium_model() {
        let yaml = r#"
name: test
version: "1.0.0"
model:
  hf_repo: "test/model"
  size_category: medium
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
"#;
        let playbook = Playbook::from_yaml(yaml).expect("parse");
        // Medium model caps at 2 workers
        assert_eq!(playbook.effective_max_workers(4), 2);
        assert_eq!(playbook.effective_max_workers(1), 1);
    }

    // ── PMAT-266 Naming convention tests ─────────────────────────────────

    #[test]
    fn test_validate_playbook_name_basic() {
        let result = validate_playbook_name("qwen2.5-coder-0.5b-mvp.playbook.yaml");
        assert!(result.is_ok());
        let parts = result.unwrap();
        assert_eq!(parts.family, "qwen2.5-coder");
        assert_eq!(parts.size, "0.5b");
        assert_eq!(parts.tier, Some("mvp".to_string()));
    }

    #[test]
    fn test_validate_playbook_name_no_tier() {
        let result = validate_playbook_name("llama3.2-7b.playbook.yaml");
        assert!(result.is_ok());
        let parts = result.unwrap();
        assert_eq!(parts.family, "llama3.2");
        assert_eq!(parts.size, "7b");
        assert_eq!(parts.tier, None);
    }

    #[test]
    fn test_validate_playbook_name_large_model() {
        let result = validate_playbook_name("deepseek-coder-v2-16b-full.playbook.yaml");
        assert!(result.is_ok());
        let parts = result.unwrap();
        assert_eq!(parts.family, "deepseek-coder-v2");
        assert_eq!(parts.size, "16b");
        assert_eq!(parts.tier, Some("full".to_string()));
    }

    #[test]
    fn test_validate_playbook_name_various_tiers() {
        for tier in VALID_TIERS {
            let filename = format!("model-1b-{tier}.playbook.yaml");
            let result = validate_playbook_name(&filename);
            assert!(result.is_ok(), "Failed for tier: {tier}");
            assert_eq!(result.unwrap().tier, Some((*tier).to_string()));
        }
    }

    #[test]
    fn test_validate_playbook_name_various_sizes() {
        let sizes = ["0.5b", "1b", "1.5b", "3b", "7b", "13b", "70b", "405b"];
        for size in sizes {
            let filename = format!("model-{size}.playbook.yaml");
            let result = validate_playbook_name(&filename);
            assert!(result.is_ok(), "Failed for size: {size}");
            assert_eq!(result.unwrap().size, size);
        }
    }

    #[test]
    fn test_validate_playbook_name_invalid_no_size() {
        let result = validate_playbook_name("qwen2.5-coder-mvp.playbook.yaml");
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("does not match naming convention"));
    }

    #[test]
    fn test_validate_playbook_name_invalid_wrong_extension() {
        let result = validate_playbook_name("qwen2.5-coder-0.5b-mvp.yaml");
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_playbook_name_invalid_tier() {
        let result = validate_playbook_name("qwen2.5-coder-0.5b-unknown.playbook.yaml");
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_playbook_name_invalid_format() {
        let invalid_names = [
            "model.playbook.yaml",         // no size
            "model-big.playbook.yaml",     // invalid size format
            "model-7gb.playbook.yaml",     // wrong unit (gb instead of b)
            ".playbook.yaml",              // empty name
            "model-7b-test.playbook.yaml", // invalid tier
        ];
        for name in invalid_names {
            let result = validate_playbook_name(name);
            assert!(result.is_err(), "Expected error for: {name}");
        }
    }

    #[test]
    fn test_validate_playbook_path() {
        let path = std::path::Path::new("/some/path/qwen2.5-coder-1.5b-mvp.playbook.yaml");
        let result = validate_playbook_path(path);
        assert!(result.is_ok());
        let parts = result.unwrap();
        assert_eq!(parts.family, "qwen2.5-coder");
        assert_eq!(parts.size, "1.5b");
        assert_eq!(parts.tier, Some("mvp".to_string()));
    }

    #[test]
    fn test_playbook_name_parts_to_filename() {
        let parts = PlaybookNameParts {
            family: "qwen2.5-coder".to_string(),
            size: "0.5b".to_string(),
            tier: Some("mvp".to_string()),
        };
        assert_eq!(parts.to_filename(), "qwen2.5-coder-0.5b-mvp.playbook.yaml");

        let parts_no_tier = PlaybookNameParts {
            family: "llama3.2".to_string(),
            size: "7b".to_string(),
            tier: None,
        };
        assert_eq!(parts_no_tier.to_filename(), "llama3.2-7b.playbook.yaml");
    }

    #[test]
    fn test_playbook_name_parts_eq() {
        let parts1 = PlaybookNameParts {
            family: "model".to_string(),
            size: "1b".to_string(),
            tier: Some("mvp".to_string()),
        };
        let parts2 = PlaybookNameParts {
            family: "model".to_string(),
            size: "1b".to_string(),
            tier: Some("mvp".to_string()),
        };
        assert_eq!(parts1, parts2);
    }

    #[test]
    fn test_valid_tiers_constant() {
        assert_eq!(VALID_TIERS.len(), 7);
        assert!(VALID_TIERS.contains(&"mvp"));
        assert!(VALID_TIERS.contains(&"smoke"));
        assert!(VALID_TIERS.contains(&"quick"));
        assert!(VALID_TIERS.contains(&"ci"));
        assert!(VALID_TIERS.contains(&"full"));
        assert!(VALID_TIERS.contains(&"nightly"));
        assert!(VALID_TIERS.contains(&"release"));
    }

    // ── PMAT-269 Test matrix generation tests ────────────────────────────

    #[test]
    fn test_populate_from_family_contract() {
        use crate::family_contract::FamilyContract;

        // PMAT-270: Include certification.size_categories for auto-alignment test
        let yaml = r#"
family: qwen2
size_variants:
  0.5b:
    parameters: "0.5B"
    hidden_dim: 896
    num_layers: 24
    num_heads: 14
    num_kv_heads: 2
    vocab_size: 151936
    intermediate_dim: 4864
certification:
  size_categories:
    0.5b: tiny
    1.5b: small
    7b: medium
"#;
        let contract = FamilyContract::from_yaml(yaml).expect("parse");

        let mut config = ModelConfig {
            hf_repo: "Qwen/Qwen2.5-Coder-0.5B-Instruct".to_string(),
            local_path: None,
            formats: vec![Format::Gguf],
            quantizations: vec![],
            size_category: SizeCategory::Tiny, // default
            expected_hidden_dim: None,
            expected_num_layers: None,
            expected_num_heads: None,
            expected_num_kv_heads: None,
            expected_vocab_size: None,
            expected_intermediate_dim: None,
            family: None,
            size_variant: None,
        };

        // Populate from contract
        let result = config.populate_from_family_contract(&contract, "0.5b");
        assert!(result);

        // Verify values populated
        assert_eq!(config.family, Some("qwen2".to_string()));
        assert_eq!(config.size_variant, Some("0.5b".to_string()));
        assert_eq!(config.expected_hidden_dim, Some(896));
        assert_eq!(config.expected_num_layers, Some(24));
        assert_eq!(config.expected_num_heads, Some(14));
        assert_eq!(config.expected_num_kv_heads, Some(2));
        assert_eq!(config.expected_vocab_size, Some(151_936));
        assert_eq!(config.expected_intermediate_dim, Some(4864));
        // PMAT-270: Verify size_category auto-populated
        assert_eq!(config.size_category, SizeCategory::Tiny);
    }

    #[test]
    fn test_populate_from_family_contract_missing_size() {
        use crate::family_contract::FamilyContract;

        let yaml = r#"
family: qwen2
size_variants:
  0.5b:
    parameters: "0.5B"
    hidden_dim: 896
    num_layers: 24
    num_heads: 14
"#;
        let contract = FamilyContract::from_yaml(yaml).expect("parse");

        let mut config = ModelConfig {
            hf_repo: "test".to_string(),
            local_path: None,
            formats: vec![],
            quantizations: vec![],
            size_category: SizeCategory::default(),
            expected_hidden_dim: None,
            expected_num_layers: None,
            expected_num_heads: None,
            expected_num_kv_heads: None,
            expected_vocab_size: None,
            expected_intermediate_dim: None,
            family: None,
            size_variant: None,
        };

        // Try to populate with non-existent size
        let result = config.populate_from_family_contract(&contract, "7b");
        assert!(!result);

        // Values should remain None
        assert!(config.expected_hidden_dim.is_none());
    }

    #[test]
    fn test_has_expected_params() {
        let config_empty = ModelConfig {
            hf_repo: "test".to_string(),
            local_path: None,
            formats: vec![],
            quantizations: vec![],
            size_category: SizeCategory::default(),
            expected_hidden_dim: None,
            expected_num_layers: None,
            expected_num_heads: None,
            expected_num_kv_heads: None,
            expected_vocab_size: None,
            expected_intermediate_dim: None,
            family: None,
            size_variant: None,
        };
        assert!(!config_empty.has_expected_params());

        let config_with_params = ModelConfig {
            hf_repo: "test".to_string(),
            local_path: None,
            formats: vec![],
            quantizations: vec![],
            size_category: SizeCategory::default(),
            expected_hidden_dim: Some(896),
            expected_num_layers: None,
            expected_num_heads: None,
            expected_num_kv_heads: None,
            expected_vocab_size: None,
            expected_intermediate_dim: None,
            family: None,
            size_variant: None,
        };
        assert!(config_with_params.has_expected_params());
    }

    #[test]
    fn test_validate_architecture_match() {
        let config = ModelConfig {
            hf_repo: "test".to_string(),
            local_path: None,
            formats: vec![],
            quantizations: vec![],
            size_category: SizeCategory::default(),
            expected_hidden_dim: Some(896),
            expected_num_layers: Some(24),
            expected_num_heads: Some(14),
            expected_num_kv_heads: Some(2),
            expected_vocab_size: None,
            expected_intermediate_dim: None,
            family: None,
            size_variant: None,
        };

        // All match
        let mismatches = config.validate_architecture(896, 24, 14, Some(2));
        assert!(mismatches.is_empty());
    }

    #[test]
    fn test_validate_architecture_mismatch() {
        let config = ModelConfig {
            hf_repo: "test".to_string(),
            local_path: None,
            formats: vec![],
            quantizations: vec![],
            size_category: SizeCategory::default(),
            expected_hidden_dim: Some(896),
            expected_num_layers: Some(24),
            expected_num_heads: Some(14),
            expected_num_kv_heads: Some(2),
            expected_vocab_size: None,
            expected_intermediate_dim: None,
            family: None,
            size_variant: None,
        };

        // All wrong
        let mismatches = config.validate_architecture(1024, 12, 16, Some(4));
        assert_eq!(mismatches.len(), 4);
        assert!(mismatches[0].contains("hidden_dim"));
        assert!(mismatches[1].contains("num_layers"));
        assert!(mismatches[2].contains("num_heads"));
        assert!(mismatches[3].contains("num_kv_heads"));
    }

    #[test]
    fn test_validate_architecture_partial_expected() {
        let config = ModelConfig {
            hf_repo: "test".to_string(),
            local_path: None,
            formats: vec![],
            quantizations: vec![],
            size_category: SizeCategory::default(),
            expected_hidden_dim: Some(896),
            expected_num_layers: None, // Not set
            expected_num_heads: None,  // Not set
            expected_num_kv_heads: None,
            expected_vocab_size: None,
            expected_intermediate_dim: None,
            family: None,
            size_variant: None,
        };

        // Only hidden_dim is checked
        let mismatches = config.validate_architecture(896, 999, 999, Some(999));
        assert!(mismatches.is_empty()); // hidden_dim matches, others not checked
    }

    // ── PMAT-270: Size category auto-alignment tests ─────────────────────────

    #[test]
    fn test_size_category_auto_alignment_from_family_yaml() {
        use crate::family_contract::FamilyContract;

        // FALSIFY-FAM-001: Size category alignment
        let yaml = r#"
family: qwen2
size_variants:
  7b:
    parameters: "7B"
    hidden_dim: 3584
    num_layers: 28
    num_heads: 28
certification:
  size_categories:
    0.5b: tiny
    1.5b: small
    3b: small
    7b: medium
    14b: large
"#;
        let contract = FamilyContract::from_yaml(yaml).expect("parse");

        // Start with default (Tiny)
        let mut config = ModelConfig {
            hf_repo: "Qwen/Qwen2.5-Coder-7B-Instruct".to_string(),
            local_path: None,
            formats: vec![],
            quantizations: vec![],
            size_category: SizeCategory::Tiny, // default
            expected_hidden_dim: None,
            expected_num_layers: None,
            expected_num_heads: None,
            expected_num_kv_heads: None,
            expected_vocab_size: None,
            expected_intermediate_dim: None,
            family: None,
            size_variant: None,
        };

        // Populate from contract with 7b size
        let result = config.populate_from_family_contract(&contract, "7b");
        assert!(result);

        // PMAT-270: Verify size_category auto-set to Medium (from 7b -> medium mapping)
        assert_eq!(config.size_category, SizeCategory::Medium);
    }

    #[test]
    fn test_size_category_explicit_not_overridden() {
        use crate::family_contract::FamilyContract;

        let yaml = r#"
family: qwen2
size_variants:
  7b:
    parameters: "7B"
    hidden_dim: 3584
    num_layers: 28
    num_heads: 28
certification:
  size_categories:
    7b: medium
"#;
        let contract = FamilyContract::from_yaml(yaml).expect("parse");

        // Explicitly set to Large (user override)
        let mut config = ModelConfig {
            hf_repo: "Qwen/Qwen2.5-Coder-7B-Instruct".to_string(),
            local_path: None,
            formats: vec![],
            quantizations: vec![],
            size_category: SizeCategory::Large, // explicitly set, not default
            expected_hidden_dim: None,
            expected_num_layers: None,
            expected_num_heads: None,
            expected_num_kv_heads: None,
            expected_vocab_size: None,
            expected_intermediate_dim: None,
            family: None,
            size_variant: None,
        };

        // Populate from contract
        config.populate_from_family_contract(&contract, "7b");

        // Should NOT override explicit setting - Large remains Large
        assert_eq!(config.size_category, SizeCategory::Large);
    }

    #[test]
    fn test_size_category_from_str_lowercase() {
        assert_eq!(
            SizeCategory::from_str_lowercase("tiny").unwrap(),
            SizeCategory::Tiny
        );
        assert_eq!(
            SizeCategory::from_str_lowercase("small").unwrap(),
            SizeCategory::Small
        );
        assert_eq!(
            SizeCategory::from_str_lowercase("medium").unwrap(),
            SizeCategory::Medium
        );
        assert_eq!(
            SizeCategory::from_str_lowercase("large").unwrap(),
            SizeCategory::Large
        );
        assert_eq!(
            SizeCategory::from_str_lowercase("xlarge").unwrap(),
            SizeCategory::Xlarge
        );
        assert_eq!(
            SizeCategory::from_str_lowercase("huge").unwrap(),
            SizeCategory::Huge
        );

        // Case insensitive
        assert_eq!(
            SizeCategory::from_str_lowercase("TINY").unwrap(),
            SizeCategory::Tiny
        );
        assert_eq!(
            SizeCategory::from_str_lowercase("Medium").unwrap(),
            SizeCategory::Medium
        );

        // Invalid
        let err = SizeCategory::from_str_lowercase("invalid").unwrap_err();
        assert!(err.to_string().contains("Invalid size category"));
    }

    #[test]
    fn test_size_category_no_certification_config() {
        use crate::family_contract::FamilyContract;

        // No certification section at all
        let yaml = r#"
family: qwen2
size_variants:
  0.5b:
    parameters: "0.5B"
    hidden_dim: 896
    num_layers: 24
    num_heads: 14
"#;
        let contract = FamilyContract::from_yaml(yaml).expect("parse");

        let mut config = ModelConfig {
            hf_repo: "test".to_string(),
            local_path: None,
            formats: vec![],
            quantizations: vec![],
            size_category: SizeCategory::Tiny, // default
            expected_hidden_dim: None,
            expected_num_layers: None,
            expected_num_heads: None,
            expected_num_kv_heads: None,
            expected_vocab_size: None,
            expected_intermediate_dim: None,
            family: None,
            size_variant: None,
        };

        config.populate_from_family_contract(&contract, "0.5b");

        // Should remain default since no certification config
        assert_eq!(config.size_category, SizeCategory::Tiny);
    }

    // ── GH-6/AC-2: Ollama parity config tests ────────────────────────────

    #[test]
    fn test_playbook_with_ollama_parity() {
        let yaml = r#"
name: ollama-test
version: "1.0.0"
model:
  hf_repo: "test/model"
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
ollama_parity:
  enabled: true
  model_tag: "qwen2.5-coder:7b-instruct-q4_k_m"
  quantizations: ["q4_k_m", "q6_k"]
  prompts: ["What is 2+2?", "def hello():"]
  temperature: 0.0
  min_perf_ratio: 0.9
  gates: ["F-OLLAMA-001", "F-OLLAMA-002"]
"#;
        let playbook = Playbook::from_yaml(yaml).expect("Failed to parse");
        let ollama = playbook.ollama_parity.expect("Should have ollama parity");

        assert!(ollama.enabled);
        assert_eq!(
            ollama.model_tag,
            Some("qwen2.5-coder:7b-instruct-q4_k_m".to_string())
        );
        assert_eq!(ollama.quantizations.len(), 2);
        assert_eq!(ollama.prompts.len(), 2);
        assert!((ollama.temperature - 0.0).abs() < f64::EPSILON);
        assert!((ollama.min_perf_ratio - 0.9).abs() < f64::EPSILON);
        assert_eq!(ollama.gates.len(), 2);
    }

    #[test]
    fn test_playbook_without_ollama_parity() {
        let yaml = r#"
name: no-ollama
version: "1.0.0"
model:
  hf_repo: "test/model"
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
"#;
        let playbook = Playbook::from_yaml(yaml).expect("parse");
        assert!(playbook.ollama_parity.is_none());
    }

    #[test]
    fn test_ollama_parity_config_defaults() {
        let yaml = r#"
name: ollama-defaults
version: "1.0.0"
model:
  hf_repo: "test/model"
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
ollama_parity:
  enabled: true
"#;
        let playbook = Playbook::from_yaml(yaml).expect("parse");
        let ollama = playbook.ollama_parity.expect("should exist");

        assert!(ollama.enabled);
        assert!(ollama.model_tag.is_none());
        assert_eq!(ollama.quantizations, vec!["q4_k_m"]);
        assert_eq!(ollama.prompts, vec!["What is 2+2?"]);
        assert!((ollama.temperature - 0.0).abs() < f64::EPSILON);
        assert!((ollama.min_perf_ratio - 0.8).abs() < f64::EPSILON);
        assert!(ollama.gates.is_empty());
    }
}
