//! Playbook definition and parsing
//!
//! Playbooks define test scenarios in YAML format.

use apr_qa_gen::{Backend, Format, Modality, ModelId, QaScenario};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

use crate::error::{Error, Result};

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

        for modality in &self.test_matrix.modalities {
            for backend in &self.test_matrix.backends {
                for format in &self.model.formats {
                    for _ in 0..self.test_matrix.scenario_count {
                        // Use a simple prompt for now
                        let prompt = "What is 2+2?".to_string();
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
    #[serde(default)]
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
    #[serde(default)]
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
    #[serde(default)]
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
    #[serde(default)]
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
    #[serde(default)]
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
    #[serde(default)]
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
    #[serde(default)]
    pub enabled: bool,
    /// Prompt for trace
    #[serde(default)]
    pub prompt: Option<String>,
    /// Gates to verify
    #[serde(default)]
    pub gates: Vec<String>,
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
}
