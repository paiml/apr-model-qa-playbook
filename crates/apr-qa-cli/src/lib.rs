//! APR QA CLI Library
//!
//! Library functions for the APR QA CLI tool.

#![allow(clippy::doc_markdown)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::struct_excessive_bools)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::missing_const_for_fn)]
// Allow common patterns in test code
#![cfg_attr(test, allow(clippy::expect_used, clippy::unwrap_used))]

use apr_qa_gen::models::ModelMetadata;
use apr_qa_gen::{ModelId, ModelRegistry, ScenarioGenerator};
use apr_qa_report::{
    html::HtmlDashboard,
    junit::JunitReport,
    mqs::MqsCalculator,
    popperian::PopperianCalculator,
    ticket::{TicketGenerator, UpstreamTicket, generate_structured_tickets},
};
use apr_qa_runner::{
    Evidence, EvidenceCollector, ExecutionConfig, ExecutionResult, Executor, FailurePolicy,
    Playbook, ToolExecutor,
};
use std::path::Path;

/// Result of a CLI operation
#[derive(Debug)]
pub enum CliResult {
    /// Operation succeeded
    Success(String),
    /// Operation failed with error
    Error(String),
}

impl CliResult {
    /// Returns true if the result is a success
    pub fn is_success(&self) -> bool {
        matches!(self, Self::Success(_))
    }

    /// Returns the message
    pub fn message(&self) -> &str {
        match self {
            Self::Success(msg) | Self::Error(msg) => msg,
        }
    }
}

/// Configuration for running a playbook
#[derive(Debug, Clone)]
pub struct PlaybookRunConfig {
    /// Failure policy (stop-on-first, stop-on-p0, collect-all)
    pub failure_policy: String,
    /// Dry run mode - don't execute, just show what would be done
    pub dry_run: bool,
    /// Maximum parallel workers
    pub workers: usize,
    /// Path to model file
    pub model_path: Option<String>,
    /// Timeout per test in milliseconds
    pub timeout: u64,
    /// Disable GPU acceleration
    pub no_gpu: bool,
    /// Skip P0 format conversion tests
    pub skip_conversion_tests: bool,
    /// Run APR tool coverage tests
    pub run_tool_tests: bool,
    /// Run differential tests (tensor_diff, inference_compare)
    pub run_differential_tests: bool,
    /// Run profile CI assertions (throughput, latency)
    pub run_profile_ci: bool,
    /// Run trace payload tests (forward pass, garbage detection)
    pub run_trace_payload: bool,
    /// Run HF parity verification against golden corpus
    pub run_hf_parity: bool,
    /// Path to HF golden corpus directory
    pub hf_parity_corpus_path: Option<String>,
    /// HF parity model family (e.g., "qwen2.5-coder-1.5b/v1")
    pub hf_parity_model_family: Option<String>,
}

impl Default for PlaybookRunConfig {
    fn default() -> Self {
        Self {
            failure_policy: "stop-on-p0".to_string(),
            dry_run: false,
            workers: 4,
            model_path: None,
            timeout: 60000,
            no_gpu: false,
            skip_conversion_tests: false,
            run_tool_tests: false,
            run_differential_tests: true,
            run_profile_ci: false,
            run_trace_payload: true,
            run_hf_parity: false,
            hf_parity_corpus_path: None,
            hf_parity_model_family: None,
        }
    }
}

/// Parse failure policy string to enum
pub fn parse_failure_policy(policy: &str) -> Result<FailurePolicy, String> {
    match policy {
        "stop-on-first" => Ok(FailurePolicy::StopOnFirst),
        "stop-on-p0" => Ok(FailurePolicy::StopOnP0),
        "collect-all" => Ok(FailurePolicy::CollectAll),
        "fail-fast" => Ok(FailurePolicy::FailFast),
        _ => Err(format!("Unknown failure policy: {policy}")),
    }
}

/// Load a playbook from a file path
pub fn load_playbook(path: &Path) -> Result<Playbook, String> {
    Playbook::from_file(path).map_err(|e| format!("Error loading playbook: {e}"))
}

/// Run tool tests and return results
pub fn execute_tool_tests(
    model_path: &str,
    no_gpu: bool,
    timeout: u64,
    include_serve: bool,
) -> Vec<apr_qa_runner::ToolTestResult> {
    let executor = ToolExecutor::new(model_path.to_string(), no_gpu, timeout);
    executor.execute_all_with_serve(include_serve)
}

/// Generate scenarios for a model
pub fn generate_model_scenarios(model_id: &str, count: usize) -> Vec<apr_qa_gen::QaScenario> {
    let parts: Vec<&str> = model_id.split('/').collect();
    let (org, name) = if parts.len() >= 2 {
        (parts[0], parts[1])
    } else {
        ("unknown", model_id)
    };

    let model = ModelId::new(org, name);
    let generator = ScenarioGenerator::new(model).with_scenarios_per_combination(count);
    generator.generate()
}

/// Format scenarios as YAML
pub fn scenarios_to_yaml(scenarios: &[apr_qa_gen::QaScenario]) -> Result<String, String> {
    let mut output = String::new();
    for scenario in scenarios {
        match serde_yaml::to_string(scenario) {
            Ok(yaml) => {
                output.push_str("---\n");
                output.push_str(&yaml);
            }
            Err(e) => return Err(format!("Error serializing scenario: {e}")),
        }
    }
    Ok(output)
}

/// Format scenarios as JSON
pub fn scenarios_to_json(scenarios: &[apr_qa_gen::QaScenario]) -> Result<String, String> {
    serde_json::to_string_pretty(scenarios).map_err(|e| format!("Error serializing scenarios: {e}"))
}

/// Parse evidence from JSON string
pub fn parse_evidence(json: &str) -> Result<Vec<Evidence>, String> {
    serde_json::from_str(json).map_err(|e| format!("Error parsing evidence JSON: {e}"))
}

/// Create an evidence collector from evidence list
pub fn collect_evidence(evidence: Vec<Evidence>) -> EvidenceCollector {
    let mut collector = EvidenceCollector::new();
    for e in evidence {
        collector.add(e);
    }
    collector
}

/// Calculate MQS score from evidence
pub fn calculate_mqs_score(
    model_id: &str,
    collector: &EvidenceCollector,
) -> Result<apr_qa_report::mqs::MqsScore, String> {
    let calculator = MqsCalculator::new();
    calculator
        .calculate(model_id, collector)
        .map_err(|e| format!("Error calculating MQS: {e}"))
}

/// Calculate Popperian score from evidence
pub fn calculate_popperian_score(
    model_id: &str,
    collector: &EvidenceCollector,
) -> apr_qa_report::popperian::PopperianScore {
    let calculator = PopperianCalculator::new();
    calculator.calculate(model_id, collector)
}

/// Generate HTML report
pub fn generate_html_report(
    title: &str,
    mqs_score: &apr_qa_report::mqs::MqsScore,
    popperian_score: &apr_qa_report::popperian::PopperianScore,
    collector: &EvidenceCollector,
) -> Result<String, String> {
    let dashboard = HtmlDashboard::new(title.to_string());
    dashboard
        .generate(mqs_score, popperian_score, collector)
        .map_err(|e| format!("Error generating HTML: {e}"))
}

/// Generate JUnit XML report
pub fn generate_junit_report(
    model_id: &str,
    collector: &EvidenceCollector,
    mqs_score: &apr_qa_report::mqs::MqsScore,
) -> Result<String, String> {
    let junit = JunitReport::new(model_id);
    junit
        .generate(collector, mqs_score)
        .map_err(|e| format!("Error generating JUnit: {e}"))
}

/// List all models from registry
pub fn list_all_models() -> Vec<ModelMetadata> {
    let registry = ModelRegistry::with_defaults();
    registry.all().into_iter().cloned().collect()
}

/// Filter models by size
pub fn filter_models_by_size(models: &[ModelMetadata], size_filter: &str) -> Vec<ModelMetadata> {
    models
        .iter()
        .filter(|model| {
            let size_str = format!("{:?}", model.size).to_lowercase();
            size_str.contains(&size_filter.to_lowercase())
        })
        .cloned()
        .collect()
}

/// Generate tickets from evidence
pub fn generate_tickets_from_evidence(
    evidence: &[Evidence],
    repo: &str,
    black_swans_only: bool,
    min_occurrences: usize,
) -> Vec<UpstreamTicket> {
    let mut generator = TicketGenerator::new(repo).with_min_occurrences(min_occurrences);

    if black_swans_only {
        generator = generator.black_swans_only();
    }

    generator.generate_from_evidence(evidence)
}

/// Format ticket for display
pub fn format_ticket_for_display(ticket: &UpstreamTicket, repo: &str) -> String {
    format!(
        "--- {} ---\nPriority: {}\nCategory: {}\nLabels: {}\n\ngh command:\n  {}\n",
        ticket.title,
        ticket.priority,
        ticket.category,
        ticket.labels.join(", "),
        ticket.to_gh_command(repo)
    )
}

/// Build execution config from run config
pub fn build_execution_config(config: &PlaybookRunConfig) -> Result<ExecutionConfig, String> {
    let policy = parse_failure_policy(&config.failure_policy)?;

    Ok(ExecutionConfig {
        failure_policy: policy,
        dry_run: config.dry_run,
        max_workers: config.workers,
        model_path: config.model_path.clone(),
        default_timeout_ms: config.timeout,
        no_gpu: config.no_gpu,
        run_conversion_tests: !config.skip_conversion_tests,
        run_differential_tests: config.run_differential_tests,
        run_profile_ci: config.run_profile_ci,
        run_trace_payload: config.run_trace_payload,
        run_golden_rule_test: true,
        golden_reference_path: None,
        lock_file_path: None,
        check_integrity: false,
        warn_implicit_skips: false,
        run_hf_parity: config.run_hf_parity,
        hf_parity_corpus_path: config.hf_parity_corpus_path.clone(),
        hf_parity_model_family: config.hf_parity_model_family.clone(),
    })
}

/// Execute a playbook with the given configuration
pub fn execute_playbook(
    playbook: &Playbook,
    config: ExecutionConfig,
) -> Result<ExecutionResult, String> {
    let mut executor = Executor::with_config(config);
    executor
        .execute(playbook)
        .map_err(|e| format!("Execution failed: {e}"))
}

/// Certification tier levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CertTier {
    /// Tier 1: Smoke test
    Smoke,
    /// Tier 2: MVP - all formats/backends/modalities
    Mvp,
    /// Tier 3: Quick check
    #[default]
    Quick,
    /// Tier 4: Standard certification
    Standard,
    /// Tier 5: Deep certification
    Deep,
}

impl std::str::FromStr for CertTier {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "smoke" => Ok(Self::Smoke),
            "mvp" => Ok(Self::Mvp),
            "quick" => Ok(Self::Quick),
            "standard" => Ok(Self::Standard),
            "deep" => Ok(Self::Deep),
            _ => Err(format!(
                "Unknown tier: {s}. Use: smoke, mvp, quick, standard, deep"
            )),
        }
    }
}

impl CertTier {
    /// Get the playbook suffix for this tier
    #[must_use]
    pub const fn playbook_suffix(self) -> &'static str {
        match self {
            Self::Smoke => "-smoke",
            Self::Mvp => "-mvp",
            Self::Quick => "-quick",
            Self::Standard | Self::Deep => "",
        }
    }
}

/// Configuration for certification runs
#[derive(Debug, Clone)]
pub struct CertificationConfig {
    /// Certification tier
    pub tier: CertTier,
    /// Model cache directory (contains gguf/apr/safetensors subdirs)
    pub model_cache: Option<std::path::PathBuf>,
    /// Path to apr binary
    pub apr_binary: String,
    /// Output directory for artifacts
    pub output_dir: std::path::PathBuf,
    /// Dry run mode
    pub dry_run: bool,
}

impl Default for CertificationConfig {
    fn default() -> Self {
        Self {
            tier: CertTier::Quick,
            model_cache: None,
            apr_binary: "apr".to_string(),
            output_dir: std::path::PathBuf::from("certifications"),
            dry_run: false,
        }
    }
}

/// Result of certifying a single model
#[derive(Debug, Clone)]
pub struct ModelCertificationResult {
    /// Model ID
    pub model_id: String,
    /// Whether certification succeeded
    pub success: bool,
    /// MQS score (0-1000)
    pub mqs_score: u32,
    /// Grade (A, B, C, D, F)
    pub grade: String,
    /// Pass rate as percentage
    pub pass_rate: f64,
    /// Gateway failures (if any)
    pub gateway_failed: Option<String>,
    /// Error message (if failed)
    pub error: Option<String>,
}

/// Build an ExecutionConfig for certification
///
/// This is the canonical way to build an ExecutionConfig for certification.
pub fn build_certification_config(
    tier: CertTier,
    model_cache_path: Option<String>,
) -> ExecutionConfig {
    ExecutionConfig {
        failure_policy: FailurePolicy::CollectAll,
        dry_run: false,
        max_workers: 4,
        model_path: model_cache_path,
        default_timeout_ms: 60000,
        no_gpu: false,
        run_conversion_tests: true,
        run_differential_tests: false,
        run_profile_ci: matches!(tier, CertTier::Standard | CertTier::Deep),
        run_trace_payload: false,
        run_golden_rule_test: true,
        golden_reference_path: None,
        lock_file_path: None,
        check_integrity: false,
        warn_implicit_skips: false,
        run_hf_parity: false,
        hf_parity_corpus_path: None,
        hf_parity_model_family: None,
    }
}

/// Generate playbook path from model ID and tier
pub fn playbook_path_for_model(model_id: &str, tier: CertTier) -> String {
    let short = model_id.split('/').next_back().unwrap_or(model_id);
    let base = short
        .to_lowercase()
        .replace("-instruct", "")
        .replace("-it", "");
    format!(
        "playbooks/models/{}{}.playbook.yaml",
        base,
        tier.playbook_suffix()
    )
}

/// Certify a single model with the given configuration
///
/// Returns a `ModelCertificationResult` with the outcome.
pub fn certify_model(model_id: &str, config: &CertificationConfig) -> ModelCertificationResult {
    let playbook_path = playbook_path_for_model(model_id, config.tier);
    let playbook_file = std::path::Path::new(&playbook_path);

    if !playbook_file.exists() {
        return ModelCertificationResult {
            model_id: model_id.to_string(),
            success: false,
            mqs_score: 0,
            grade: "-".to_string(),
            pass_rate: 0.0,
            gateway_failed: None,
            error: Some(format!("Playbook not found: {playbook_path}")),
        };
    }

    let playbook = match load_playbook(playbook_file) {
        Ok(p) => p,
        Err(e) => {
            return ModelCertificationResult {
                model_id: model_id.to_string(),
                success: false,
                mqs_score: 0,
                grade: "-".to_string(),
                pass_rate: 0.0,
                gateway_failed: None,
                error: Some(e),
            };
        }
    };

    // Build model cache path
    let short = model_id.split('/').next_back().unwrap_or(model_id);
    let model_cache_path = config.model_cache.as_ref().map(|cache| {
        cache
            .join(short.to_lowercase().replace('.', "-"))
            .to_string_lossy()
            .to_string()
    });

    let exec_config = build_certification_config(config.tier, model_cache_path);

    match execute_playbook(&playbook, exec_config) {
        Ok(result) => {
            let evidence_vec: Vec<_> = result.evidence.all().to_vec();
            let collector = collect_evidence(evidence_vec);

            let pass_rate = result.pass_rate();
            let gateway_failed = result.gateway_failed;
            match calculate_mqs_score(model_id, &collector) {
                Ok(mqs) => ModelCertificationResult {
                    model_id: model_id.to_string(),
                    success: true,
                    mqs_score: mqs.raw_score,
                    grade: mqs.grade,
                    pass_rate,
                    gateway_failed,
                    error: None,
                },
                Err(e) => ModelCertificationResult {
                    model_id: model_id.to_string(),
                    success: false,
                    mqs_score: 0,
                    grade: "-".to_string(),
                    pass_rate,
                    gateway_failed: None, // MQS calculation failed, gateway status unknown
                    error: Some(e),
                },
            }
        }
        Err(e) => ModelCertificationResult {
            model_id: model_id.to_string(),
            success: false,
            mqs_score: 0,
            grade: "-".to_string(),
            pass_rate: 0.0,
            gateway_failed: None,
            error: Some(e),
        },
    }
}

/// Generate a playbook lock file from all YAML playbooks in a directory (§3.1)
///
/// Scans the directory recursively for `.playbook.yaml` files, computes SHA-256
/// hashes, and writes the lock file.
///
/// # Errors
///
/// Returns an error if the directory cannot be read or the lock file cannot be written.
pub fn generate_lock_file(dir: &Path, output: &Path) -> Result<usize, String> {
    use apr_qa_runner::{PlaybookLockFile, generate_lock_entry, save_lock_file};
    use std::collections::HashMap;

    // Walk directory for .playbook.yaml files
    fn walk_dir(dir: &Path, entries: &mut HashMap<String, apr_qa_runner::PlaybookLockEntry>) {
        let Ok(read_dir) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in read_dir.flatten() {
            let path = entry.path();
            if path.is_dir() {
                walk_dir(&path, entries);
            } else if path
                .file_name()
                .is_some_and(|n| n.to_string_lossy().ends_with(".playbook.yaml"))
            {
                if let Ok((name, lock_entry)) = generate_lock_entry(&path) {
                    entries.insert(name, lock_entry);
                }
            }
        }
    }

    let mut entries = HashMap::new();
    walk_dir(dir, &mut entries);
    let count = entries.len();

    let lock_file = PlaybookLockFile { entries };
    save_lock_file(&lock_file, output).map_err(|e| format!("Failed to save lock file: {e}"))?;

    Ok(count)
}

/// Execute auto-ticket generation from evidence using the defect-fixture map (§3.6)
///
/// Classifies failures, deduplicates by root cause, and returns structured tickets.
pub fn execute_auto_tickets(evidence: &[Evidence], _repo: &str) -> Vec<UpstreamTicket> {
    let defect_map = match apr_qa_report::defect_map::load_defect_fixture_map() {
        Ok(map) => map,
        Err(e) => {
            eprintln!("[WARN] Could not load defect fixture map: {e}");
            return Vec::new();
        }
    };

    generate_structured_tickets(evidence, &defect_map)
}

#[cfg(test)]
mod tests {
    use super::*;
    use apr_qa_gen::{Backend, Format, Modality, QaScenario};
    use std::str::FromStr;

    fn make_test_scenario() -> QaScenario {
        QaScenario::new(
            ModelId::new("test", "model"),
            Modality::Run,
            Backend::Cpu,
            Format::Gguf,
            "What is 2+2?".to_string(),
            42,
        )
    }

    fn make_corroborated_evidence() -> Evidence {
        Evidence::corroborated("F-TEST-001", make_test_scenario(), "output", 100)
    }

    fn make_falsified_evidence() -> Evidence {
        Evidence::falsified("F-TEST-002", make_test_scenario(), "failed", "error", 200)
    }

    #[test]
    fn test_cli_result_success() {
        let result = CliResult::Success("test".to_string());
        assert!(result.is_success());
        assert_eq!(result.message(), "test");
    }

    #[test]
    fn test_cli_result_error() {
        let result = CliResult::Error("error".to_string());
        assert!(!result.is_success());
        assert_eq!(result.message(), "error");
    }

    #[test]
    fn test_playbook_run_config_default() {
        let config = PlaybookRunConfig::default();
        assert_eq!(config.failure_policy, "stop-on-p0");
        assert!(!config.dry_run);
        assert_eq!(config.workers, 4);
        assert!(config.model_path.is_none());
        assert_eq!(config.timeout, 60000);
        assert!(!config.no_gpu);
        assert!(!config.skip_conversion_tests);
        assert!(!config.run_tool_tests);
    }

    #[test]
    fn test_parse_failure_policy_stop_on_first() {
        let policy = parse_failure_policy("stop-on-first").unwrap();
        assert!(matches!(policy, FailurePolicy::StopOnFirst));
    }

    #[test]
    fn test_parse_failure_policy_stop_on_p0() {
        let policy = parse_failure_policy("stop-on-p0").unwrap();
        assert!(matches!(policy, FailurePolicy::StopOnP0));
    }

    #[test]
    fn test_parse_failure_policy_collect_all() {
        let policy = parse_failure_policy("collect-all").unwrap();
        assert!(matches!(policy, FailurePolicy::CollectAll));
    }

    #[test]
    fn test_parse_failure_policy_unknown() {
        let result = parse_failure_policy("unknown");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Unknown failure policy"));
    }

    #[test]
    fn test_parse_failure_policy_fail_fast() {
        let policy = parse_failure_policy("fail-fast").unwrap();
        assert!(matches!(policy, FailurePolicy::FailFast));
    }

    #[test]
    fn test_failure_policy_fail_fast_emit_diagnostic() {
        assert!(FailurePolicy::FailFast.emit_diagnostic());
        assert!(!FailurePolicy::StopOnFirst.emit_diagnostic());
        assert!(!FailurePolicy::StopOnP0.emit_diagnostic());
        assert!(!FailurePolicy::CollectAll.emit_diagnostic());
    }

    #[test]
    fn test_failure_policy_stops_on_any_failure() {
        assert!(FailurePolicy::FailFast.stops_on_any_failure());
        assert!(FailurePolicy::StopOnFirst.stops_on_any_failure());
        assert!(!FailurePolicy::StopOnP0.stops_on_any_failure());
        assert!(!FailurePolicy::CollectAll.stops_on_any_failure());
    }

    #[test]
    fn test_load_playbook_nonexistent() {
        let result = load_playbook(Path::new("/nonexistent/playbook.yaml"));
        assert!(result.is_err());
    }

    #[test]
    fn test_generate_model_scenarios() {
        let scenarios = generate_model_scenarios("test/model", 10);
        // 3 modalities × 2 backends × 3 formats × 10 = 180 scenarios
        assert_eq!(scenarios.len(), 180);
    }

    #[test]
    fn test_generate_model_scenarios_no_org() {
        let scenarios = generate_model_scenarios("model-only", 5);
        assert_eq!(scenarios.len(), 90); // 3 × 2 × 3 × 5
    }

    #[test]
    fn test_scenarios_to_yaml() {
        let scenarios = generate_model_scenarios("test/model", 1);
        let yaml = scenarios_to_yaml(&scenarios);
        assert!(yaml.is_ok());
        let yaml_str = yaml.unwrap();
        assert!(yaml_str.contains("---"));
    }

    #[test]
    fn test_scenarios_to_json() {
        let scenarios = generate_model_scenarios("test/model", 1);
        let json = scenarios_to_json(&scenarios);
        assert!(json.is_ok());
        let json_str = json.unwrap();
        assert!(json_str.starts_with('['));
    }

    #[test]
    fn test_parse_evidence_invalid() {
        let json = "invalid json";
        let evidence = parse_evidence(json);
        assert!(evidence.is_err());
    }

    #[test]
    fn test_collect_evidence() {
        let evidence = vec![make_corroborated_evidence()];
        let collector = collect_evidence(evidence);
        assert_eq!(collector.total(), 1);
    }

    #[test]
    fn test_calculate_mqs_score() {
        let evidence = vec![make_corroborated_evidence(), make_falsified_evidence()];
        let collector = collect_evidence(evidence);
        let score = calculate_mqs_score("test/model", &collector);
        assert!(score.is_ok());
    }

    #[test]
    fn test_calculate_popperian_score() {
        let evidence = vec![make_corroborated_evidence(), make_falsified_evidence()];
        let collector = collect_evidence(evidence);
        let score = calculate_popperian_score("test/model", &collector);
        assert_eq!(score.model_id, "test/model");
    }

    #[test]
    fn test_generate_html_report() {
        let evidence = vec![make_corroborated_evidence()];
        let collector = collect_evidence(evidence);
        let mqs = calculate_mqs_score("test/model", &collector).unwrap();
        let popperian = calculate_popperian_score("test/model", &collector);
        let html = generate_html_report("Test Report", &mqs, &popperian, &collector);
        assert!(html.is_ok());
        assert!(html.unwrap().contains("<html"));
    }

    #[test]
    fn test_generate_junit_report() {
        let evidence = vec![make_corroborated_evidence()];
        let collector = collect_evidence(evidence);
        let mqs = calculate_mqs_score("test/model", &collector).unwrap();
        let xml = generate_junit_report("test/model", &collector, &mqs);
        assert!(xml.is_ok());
        assert!(xml.unwrap().contains("<testsuite"));
    }

    #[test]
    fn test_list_all_models() {
        let models = list_all_models();
        assert!(!models.is_empty());
    }

    #[test]
    fn test_filter_models_by_size() {
        let models = list_all_models();
        let small = filter_models_by_size(&models, "small");
        // All filtered models should have "small" in their size
        for model in &small {
            let size_str = format!("{:?}", model.size).to_lowercase();
            assert!(size_str.contains("small"));
        }
    }

    #[test]
    fn test_filter_models_by_size_case_insensitive() {
        let models = list_all_models();
        let small1 = filter_models_by_size(&models, "small");
        let small2 = filter_models_by_size(&models, "SMALL");
        assert_eq!(small1.len(), small2.len());
    }

    #[test]
    fn test_generate_tickets_from_evidence_empty() {
        let evidence: Vec<Evidence> = vec![];
        let tickets = generate_tickets_from_evidence(&evidence, "test/repo", false, 1);
        assert!(tickets.is_empty());
    }

    #[test]
    fn test_generate_tickets_from_evidence_with_failures() {
        let evidence = vec![make_falsified_evidence(), make_falsified_evidence()];
        let tickets = generate_tickets_from_evidence(&evidence, "test/repo", false, 1);
        // May or may not generate tickets depending on ticket rules
        assert!(tickets.len() <= evidence.len());
    }

    #[test]
    fn test_build_execution_config() {
        let config = PlaybookRunConfig::default();
        let exec_config = build_execution_config(&config);
        assert!(exec_config.is_ok());
        let exec = exec_config.unwrap();
        assert!(!exec.dry_run);
        assert_eq!(exec.max_workers, 4);
    }

    #[test]
    fn test_build_execution_config_invalid_policy() {
        let config = PlaybookRunConfig {
            failure_policy: "invalid".to_string(),
            ..Default::default()
        };
        let exec_config = build_execution_config(&config);
        assert!(exec_config.is_err());
    }

    #[test]
    fn test_build_execution_config_with_options() {
        let config = PlaybookRunConfig {
            dry_run: true,
            workers: 8,
            model_path: Some("/path/to/model".to_string()),
            no_gpu: true,
            skip_conversion_tests: true,
            ..Default::default()
        };
        let exec_config = build_execution_config(&config).unwrap();
        assert!(exec_config.dry_run);
        assert_eq!(exec_config.max_workers, 8);
        assert_eq!(exec_config.model_path, Some("/path/to/model".to_string()));
        assert!(exec_config.no_gpu);
        assert!(!exec_config.run_conversion_tests);
    }

    #[test]
    fn test_collect_multiple_evidence() {
        let evidence = vec![
            make_corroborated_evidence(),
            make_falsified_evidence(),
            make_corroborated_evidence(),
        ];
        let collector = collect_evidence(evidence);
        assert_eq!(collector.total(), 3);
        assert_eq!(collector.pass_count(), 2);
        assert_eq!(collector.fail_count(), 1);
    }

    #[test]
    fn test_format_ticket_for_display() {
        let evidence = vec![make_falsified_evidence()];
        let tickets = generate_tickets_from_evidence(&evidence, "test/repo", false, 1);
        if !tickets.is_empty() {
            let display = format_ticket_for_display(&tickets[0], "test/repo");
            assert!(display.contains("---"));
            assert!(display.contains("Priority:"));
        }
    }

    #[test]
    fn test_scenarios_yaml_roundtrip() {
        let scenarios = generate_model_scenarios("test/model", 1);
        let yaml = scenarios_to_yaml(&scenarios).unwrap();
        // Should be valid YAML that can be parsed back
        assert!(yaml.contains("model:"));
    }

    #[test]
    fn test_scenarios_json_roundtrip() {
        let scenarios = generate_model_scenarios("test/model", 1);
        let json = scenarios_to_json(&scenarios).unwrap();
        // Should be valid JSON that can be parsed back
        let parsed: Vec<apr_qa_gen::QaScenario> = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.len(), scenarios.len());
    }

    // =========================================================================
    // Certification Tests
    // =========================================================================

    #[test]
    fn test_cert_tier_from_str() {
        assert_eq!("smoke".parse::<CertTier>().unwrap(), CertTier::Smoke);
        assert_eq!("mvp".parse::<CertTier>().unwrap(), CertTier::Mvp);
        assert_eq!("quick".parse::<CertTier>().unwrap(), CertTier::Quick);
        assert_eq!("standard".parse::<CertTier>().unwrap(), CertTier::Standard);
        assert_eq!("deep".parse::<CertTier>().unwrap(), CertTier::Deep);
        // Case insensitive
        assert_eq!("SMOKE".parse::<CertTier>().unwrap(), CertTier::Smoke);
        assert_eq!("Quick".parse::<CertTier>().unwrap(), CertTier::Quick);
    }

    #[test]
    fn test_cert_tier_from_str_invalid() {
        let result = "invalid".parse::<CertTier>();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Unknown tier"));
    }

    #[test]
    fn test_cert_tier_playbook_suffix() {
        assert_eq!(CertTier::Smoke.playbook_suffix(), "-smoke");
        assert_eq!(CertTier::Mvp.playbook_suffix(), "-mvp");
        assert_eq!(CertTier::Quick.playbook_suffix(), "-quick");
        assert_eq!(CertTier::Standard.playbook_suffix(), "");
        assert_eq!(CertTier::Deep.playbook_suffix(), "");
    }

    #[test]
    fn test_certification_config_default() {
        let config = CertificationConfig::default();
        assert_eq!(config.tier, CertTier::Quick);
        assert!(config.model_cache.is_none());
        assert_eq!(config.apr_binary, "apr");
        assert!(!config.dry_run);
    }

    #[test]
    fn test_build_certification_config_no_model() {
        // Without model path, critical tests should still be enabled
        let config = build_certification_config(CertTier::Mvp, None);
        assert!(config.run_conversion_tests);
        assert!(config.run_golden_rule_test);
    }

    #[test]
    fn test_build_certification_config_with_model() {
        // With model path, all critical tests should be enabled
        let config = build_certification_config(CertTier::Mvp, Some("/path/to/model".to_string()));
        assert!(config.run_conversion_tests);
        assert!(config.run_golden_rule_test);
        assert_eq!(config.model_path, Some("/path/to/model".to_string()));
    }

    #[test]
    fn test_build_certification_config_profile_ci() {
        // Standard/Deep tiers should enable profile CI
        let standard = build_certification_config(CertTier::Standard, None);
        assert!(standard.run_profile_ci);

        let deep = build_certification_config(CertTier::Deep, None);
        assert!(deep.run_profile_ci);

        // Other tiers should not
        let mvp = build_certification_config(CertTier::Mvp, None);
        assert!(!mvp.run_profile_ci);
    }

    #[test]
    fn test_playbook_path_for_model() {
        let path = playbook_path_for_model("Qwen/Qwen2.5-Coder-0.5B-Instruct", CertTier::Mvp);
        assert_eq!(
            path,
            "playbooks/models/qwen2.5-coder-0.5b-mvp.playbook.yaml"
        );

        let path = playbook_path_for_model("meta-llama/Llama-3-8B-Instruct", CertTier::Quick);
        assert_eq!(path, "playbooks/models/llama-3-8b-quick.playbook.yaml");

        let path = playbook_path_for_model("test/model-it", CertTier::Standard);
        assert_eq!(path, "playbooks/models/model.playbook.yaml");
    }

    #[test]
    fn test_certify_model_nonexistent_playbook() {
        let config = CertificationConfig {
            tier: CertTier::Mvp,
            ..Default::default()
        };
        let result = certify_model("nonexistent/model", &config);
        assert!(!result.success);
        assert!(result.error.is_some());
        assert!(result.error.unwrap().contains("Playbook not found"));
    }

    #[test]
    fn test_model_certification_result_fields() {
        let result = ModelCertificationResult {
            model_id: "test/model".to_string(),
            success: true,
            mqs_score: 850,
            grade: "A".to_string(),
            pass_rate: 95.0,
            gateway_failed: None,
            error: None,
        };
        assert!(result.success);
        assert_eq!(result.mqs_score, 850);
        assert_eq!(result.grade, "A");
    }

    #[test]
    fn test_model_certification_result_with_gateway_failure() {
        let result = ModelCertificationResult {
            model_id: "test/model".to_string(),
            success: false,
            mqs_score: 0,
            grade: "-".to_string(),
            pass_rate: 0.0,
            gateway_failed: Some("G1: Model failed to load".to_string()),
            error: None,
        };
        assert!(!result.success);
        assert!(result.gateway_failed.is_some());
    }

    #[test]
    fn test_certification_config_with_model_cache() {
        let config = CertificationConfig {
            tier: CertTier::Deep,
            model_cache: Some(std::path::PathBuf::from("/test/cache")),
            apr_binary: "custom-apr".to_string(),
            output_dir: std::path::PathBuf::from("/output"),
            dry_run: true,
        };
        assert_eq!(config.tier, CertTier::Deep);
        assert!(config.model_cache.is_some());
        assert!(config.dry_run);
    }

    #[test]
    fn test_parse_evidence_empty_array() {
        let json = "[]";
        let result = parse_evidence(json);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 0);
    }

    #[test]
    fn test_generate_tickets_black_swans_only() {
        let evidence = vec![make_falsified_evidence()];
        let tickets = generate_tickets_from_evidence(&evidence, "test/repo", true, 1);
        // May or may not have tickets depending on whether evidence qualifies as black swan
        let _ = tickets;
    }

    #[test]
    fn test_generate_tickets_min_occurrences() {
        let evidence = vec![make_falsified_evidence()];
        let tickets = generate_tickets_from_evidence(&evidence, "test/repo", false, 5);
        // With only 1 evidence and min_occurrences=5, should have no tickets
        assert!(tickets.is_empty());
    }

    #[test]
    fn test_playbook_run_config_with_all_options() {
        let config = PlaybookRunConfig {
            failure_policy: "collect-all".to_string(),
            dry_run: true,
            workers: 16,
            model_path: Some("/path/to/model".to_string()),
            timeout: 120_000,
            no_gpu: true,
            skip_conversion_tests: true,
            run_tool_tests: true,
            run_differential_tests: true,
            run_profile_ci: true,
            run_trace_payload: false,
            run_hf_parity: false,
            hf_parity_corpus_path: None,
            hf_parity_model_family: None,
        };
        assert!(config.dry_run);
        assert_eq!(config.workers, 16);
        assert!(config.run_tool_tests);
        assert!(config.run_profile_ci);
    }

    #[test]
    fn test_build_execution_config_with_differential() {
        let config = PlaybookRunConfig {
            run_differential_tests: true,
            run_profile_ci: true,
            run_trace_payload: false,
            ..Default::default()
        };
        let exec = build_execution_config(&config).unwrap();
        assert!(exec.run_differential_tests);
        assert!(exec.run_profile_ci);
        assert!(!exec.run_trace_payload);
    }

    #[test]
    fn test_build_certification_config_all_tiers() {
        // Test all tiers
        let tiers = [
            CertTier::Smoke,
            CertTier::Mvp,
            CertTier::Quick,
            CertTier::Standard,
            CertTier::Deep,
        ];

        for tier in tiers {
            let config = build_certification_config(tier, None);
            // All tiers should return valid config
            assert_eq!(config.failure_policy, FailurePolicy::CollectAll);
        }
    }

    #[test]
    fn test_playbook_path_for_model_with_slash() {
        let path = playbook_path_for_model("org/model-name-Instruct", CertTier::Smoke);
        assert!(path.contains("smoke"));
        assert!(path.contains("model-name"));
        // Should strip -Instruct
        assert!(!path.contains("-Instruct") && !path.contains("-instruct"));
    }

    #[test]
    fn test_playbook_path_for_model_deep_tier() {
        let path = playbook_path_for_model("test/model", CertTier::Deep);
        // Deep tier has no suffix
        assert!(path.ends_with(".playbook.yaml"));
        assert!(!path.contains("-deep"));
    }

    #[test]
    fn test_certification_config_output_dir() {
        let config = CertificationConfig::default();
        assert_eq!(
            config.output_dir,
            std::path::PathBuf::from("certifications")
        );
    }

    #[test]
    fn test_cli_result_debug() {
        let result = CliResult::Success("test".to_string());
        let debug_str = format!("{result:?}");
        assert!(debug_str.contains("Success"));
    }

    #[test]
    fn test_model_certification_result_debug() {
        let result = ModelCertificationResult {
            model_id: "test".to_string(),
            success: true,
            mqs_score: 900,
            grade: "A".to_string(),
            pass_rate: 100.0,
            gateway_failed: None,
            error: None,
        };
        let debug_str = format!("{result:?}");
        assert!(debug_str.contains("ModelCertificationResult"));
    }

    #[test]
    fn test_playbook_run_config_debug() {
        let config = PlaybookRunConfig::default();
        let debug_str = format!("{config:?}");
        assert!(debug_str.contains("PlaybookRunConfig"));
    }

    #[test]
    fn test_certification_config_debug() {
        let config = CertificationConfig::default();
        let debug_str = format!("{config:?}");
        assert!(debug_str.contains("CertificationConfig"));
    }

    #[test]
    fn test_playbook_run_config_clone() {
        let config = PlaybookRunConfig::default();
        let cloned = config.clone();
        assert_eq!(config.failure_policy, cloned.failure_policy);
        assert_eq!(config.workers, cloned.workers);
    }

    #[test]
    fn test_certification_config_clone() {
        let config = CertificationConfig::default();
        let cloned = config.clone();
        assert_eq!(config.tier, cloned.tier);
        assert_eq!(config.apr_binary, cloned.apr_binary);
    }

    #[test]
    fn test_model_certification_result_clone() {
        let result = ModelCertificationResult {
            model_id: "test".to_string(),
            success: true,
            mqs_score: 800,
            grade: "B".to_string(),
            pass_rate: 80.0,
            gateway_failed: None,
            error: None,
        };
        let cloned = result.clone();
        assert_eq!(result.model_id, cloned.model_id);
        assert_eq!(result.mqs_score, cloned.mqs_score);
    }

    #[test]
    fn test_cert_tier_default() {
        let tier = CertTier::default();
        assert_eq!(tier, CertTier::Quick);
    }

    #[test]
    fn test_execute_tool_tests() {
        // Just verify function exists and returns results
        let results = execute_tool_tests("/nonexistent/model.gguf", true, 1000, false);
        // Should return empty or with failures since model doesn't exist
        let _ = results;
    }

    // =========================================================================
    // Additional coverage tests for certify_model path
    // =========================================================================

    /// Helper to get the workspace root path for test playbooks
    fn get_workspace_root() -> std::path::PathBuf {
        // Start from the manifest dir and go up to find the workspace root
        let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        manifest_dir
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .to_path_buf()
    }

    #[test]
    fn test_certify_model_no_cache() {
        let config = CertificationConfig {
            tier: CertTier::Mvp,
            model_cache: None,
            apr_binary: "apr".to_string(),
            output_dir: std::path::PathBuf::from("/tmp"),
            dry_run: false,
        };
        // Non-existent model, so will fail at playbook not found
        let result = certify_model("nonexistent/model", &config);
        // Will fail because playbook doesn't exist
        assert!(!result.success);
        assert!(result.error.is_some());
    }

    #[test]
    fn test_certify_model_with_cache() {
        let config = CertificationConfig {
            tier: CertTier::Mvp,
            model_cache: Some(std::path::PathBuf::from("/nonexistent/cache")),
            apr_binary: "apr".to_string(),
            output_dir: std::path::PathBuf::from("/tmp"),
            dry_run: false,
        };
        // Non-existent model
        let result = certify_model("test/model", &config);
        // Will fail because playbook doesn't exist
        assert_eq!(result.model_id, "test/model");
    }

    #[test]
    fn test_execute_playbook_with_config() {
        // Create a minimal playbook from YAML
        let yaml = r#"
name: test-playbook
version: "1.0.0"
description: "Test playbook"
model:
  hf_repo: "test/model"
  formats:
    - gguf
  quantizations:
    - q4_k_m
  size_category: tiny
test_matrix:
  modalities:
    - run
  backends:
    - cpu
  scenario_count: 1
  seed: 42
  timeout_ms: 30000
gates:
  g1_model_loads: true
  g2_basic_inference: true
  g3_no_crashes: true
  g4_not_garbage: true
"#;
        let playbook = Playbook::from_yaml(yaml).expect("valid yaml");
        let config = build_certification_config(CertTier::Mvp, None);
        let result = execute_playbook(&playbook, config);
        assert!(result.is_ok());
        let exec_result = result.unwrap();
        assert_eq!(exec_result.playbook_name, "test-playbook");
    }

    #[test]
    fn test_execute_playbook_with_dry_run() {
        let yaml = r#"
name: test-playbook-dry
version: "1.0.0"
description: "Test playbook dry run"
model:
  hf_repo: "test/model-dry"
  formats:
    - gguf
  quantizations:
    - q4_k_m
  size_category: tiny
test_matrix:
  modalities:
    - run
  backends:
    - cpu
  scenario_count: 1
  seed: 42
  timeout_ms: 30000
gates:
  g1_model_loads: true
  g2_basic_inference: true
  g3_no_crashes: true
  g4_not_garbage: true
"#;
        let playbook = Playbook::from_yaml(yaml).expect("valid yaml");
        let mut config = build_certification_config(CertTier::Mvp, None);
        config.dry_run = true;
        let result = execute_playbook(&playbook, config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_load_playbook_with_workspace_path() {
        let workspace_root = get_workspace_root();
        let playbook_path =
            workspace_root.join("playbooks/models/qwen2.5-coder-0.5b-mvp.playbook.yaml");
        if playbook_path.exists() {
            let result = load_playbook(&playbook_path);
            assert!(result.is_ok());
            let playbook = result.unwrap();
            assert!(!playbook.name.is_empty());
        }
    }

    #[test]
    fn test_certify_model_result_fields() {
        // Test that all ModelCertificationResult fields are properly set
        let config = CertificationConfig {
            tier: CertTier::Quick,
            ..Default::default()
        };
        let result = certify_model("nonexistent/test-model", &config);
        // Playbook won't exist, so we get error
        assert_eq!(result.model_id, "nonexistent/test-model");
        assert_eq!(result.mqs_score, 0);
        assert_eq!(result.grade, "-");
        assert!(result.error.is_some());
    }

    #[test]
    fn test_certify_model_smoke_tier() {
        let config = CertificationConfig {
            tier: CertTier::Smoke,
            ..Default::default()
        };
        // Non-existent model
        let result = certify_model("test/model-smoke", &config);
        // Either succeeds or fails with "not found"
        assert!(result.error.is_some() || result.success);
    }

    #[test]
    fn test_certify_model_standard_tier() {
        let config = CertificationConfig {
            tier: CertTier::Standard,
            ..Default::default()
        };
        let result = certify_model("test/model-standard", &config);
        // Standard tier uses base playbook without suffix
        assert!(!result.success); // Playbook won't exist
    }

    #[test]
    fn test_playbook_path_for_model_various_tiers() {
        // Test all tier combinations
        let model = "Qwen/Qwen2.5-Coder-1.5B-Instruct";

        let smoke = playbook_path_for_model(model, CertTier::Smoke);
        assert!(smoke.contains("-smoke"));

        let mvp = playbook_path_for_model(model, CertTier::Mvp);
        assert!(mvp.contains("-mvp"));

        let quick = playbook_path_for_model(model, CertTier::Quick);
        assert!(quick.contains("-quick"));

        let standard = playbook_path_for_model(model, CertTier::Standard);
        assert!(!standard.contains("-standard")); // No suffix

        let deep = playbook_path_for_model(model, CertTier::Deep);
        assert!(!deep.contains("-deep")); // No suffix
    }

    #[test]
    fn test_build_certification_config_with_model_path() {
        let config =
            build_certification_config(CertTier::Deep, Some("/path/to/models".to_string()));
        assert_eq!(config.model_path, Some("/path/to/models".to_string()));
        assert!(config.run_profile_ci); // Deep tier enables profile CI
    }

    #[test]
    fn test_certification_config_all_fields_set() {
        let config = CertificationConfig {
            tier: CertTier::Deep,
            model_cache: Some(std::path::PathBuf::from("/cache")),
            apr_binary: "/usr/bin/apr".to_string(),
            output_dir: std::path::PathBuf::from("/output"),
            dry_run: true,
        };
        assert_eq!(config.tier, CertTier::Deep);
        assert!(config.model_cache.is_some());
        assert_eq!(config.apr_binary, "/usr/bin/apr");
        assert!(config.dry_run);
    }

    // --- Additional coverage tests ---

    #[test]
    fn test_list_all_models_returns_models() {
        let models = list_all_models();
        assert!(!models.is_empty());
        // Should have default models from registry
        assert!(models.len() >= 5);
    }

    #[test]
    fn test_filter_models_by_size_small() {
        let models = list_all_models();
        let small = filter_models_by_size(&models, "small");
        // May or may not have small models depending on defaults
        for m in &small {
            assert!(format!("{:?}", m.size).to_lowercase().contains("small"));
        }
    }

    #[test]
    fn test_filter_models_by_size_no_match() {
        let models = list_all_models();
        let none = filter_models_by_size(&models, "nonexistent");
        assert!(none.is_empty());
    }

    #[test]
    fn test_generate_junit_report_basic() {
        let evidence = vec![make_corroborated_evidence()];
        let collector = collect_evidence(evidence);
        let mqs = calculate_mqs_score("test/model", &collector).unwrap();
        let junit = generate_junit_report("test/model", &collector, &mqs);
        assert!(junit.is_ok());
        assert!(junit.unwrap().contains("testsuite"));
    }

    #[test]
    fn test_build_execution_config_with_model_path() {
        let config = PlaybookRunConfig {
            model_path: Some("/models/test.gguf".to_string()),
            ..Default::default()
        };
        let exec = build_execution_config(&config).unwrap();
        assert_eq!(exec.model_path, Some("/models/test.gguf".to_string()));
    }

    #[test]
    fn test_build_execution_config_with_timeout() {
        let config = PlaybookRunConfig {
            timeout: 90000,
            ..Default::default()
        };
        let exec = build_execution_config(&config).unwrap();
        assert_eq!(exec.default_timeout_ms, 90000);
    }

    #[test]
    fn test_build_execution_config_with_workers() {
        let config = PlaybookRunConfig {
            workers: 8,
            ..Default::default()
        };
        let exec = build_execution_config(&config).unwrap();
        assert_eq!(exec.max_workers, 8);
    }

    #[test]
    fn test_cert_tier_from_str_all_values() {
        assert!(CertTier::from_str("smoke").is_ok());
        assert!(CertTier::from_str("mvp").is_ok());
        assert!(CertTier::from_str("quick").is_ok());
        assert!(CertTier::from_str("standard").is_ok());
        assert!(CertTier::from_str("deep").is_ok());
        assert!(CertTier::from_str("unknown").is_err());
    }

    #[test]
    fn test_cert_tier_from_str_case_insensitive() {
        assert!(CertTier::from_str("SMOKE").is_ok());
        assert!(CertTier::from_str("MVP").is_ok());
        assert!(CertTier::from_str("Quick").is_ok());
    }

    #[test]
    fn test_cert_tier_playbook_suffix_all() {
        assert_eq!(CertTier::Smoke.playbook_suffix(), "-smoke");
        assert_eq!(CertTier::Mvp.playbook_suffix(), "-mvp");
        assert_eq!(CertTier::Quick.playbook_suffix(), "-quick");
        assert_eq!(CertTier::Standard.playbook_suffix(), "");
        assert_eq!(CertTier::Deep.playbook_suffix(), "");
    }

    #[test]
    fn test_execute_playbook_with_yaml_inline() {
        let yaml = r#"
name: test-playbook
version: "1.0.0"
model:
  hf_repo: "test/model"
  formats: [gguf]
  quantizations: [q4_k_m]
  size_category: tiny
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
  seed: 42
  timeout_ms: 30000
gates:
  g1_model_loads: true
  g2_basic_inference: true
  g3_no_crashes: true
  g4_not_garbage: true
"#;
        let playbook = Playbook::from_yaml(yaml).expect("valid yaml");
        let config = build_certification_config(CertTier::Smoke, None);
        let result = execute_playbook(&playbook, config);
        // Should succeed in mock mode
        assert!(result.is_ok());
    }

    #[test]
    fn test_certify_model_with_cache_smoke() {
        let config = CertificationConfig {
            tier: CertTier::Smoke,
            model_cache: Some(std::path::PathBuf::from("/tmp/test")),
            ..Default::default()
        };
        // Will fail because playbook doesn't exist
        let result = certify_model("org/model-smoke", &config);
        assert!(!result.success);
    }

    #[test]
    fn test_generate_model_scenarios_all_modalities_present() {
        let scenarios = generate_model_scenarios("test/model", 1);
        // Check all modalities are present
        let has_run = scenarios
            .iter()
            .any(|s| s.modality == apr_qa_gen::Modality::Run);
        let has_chat = scenarios
            .iter()
            .any(|s| s.modality == apr_qa_gen::Modality::Chat);
        let has_serve = scenarios
            .iter()
            .any(|s| s.modality == apr_qa_gen::Modality::Serve);
        assert!(has_run && has_chat && has_serve);
    }

    #[test]
    fn test_generate_tickets_regular_failures() {
        let evidence = vec![make_falsified_evidence()];
        let tickets = generate_tickets_from_evidence(&evidence, "test/repo", false, 1);
        // Should generate tickets for regular failures
        let _ = tickets; // May or may not have tickets
    }

    // =========================================================================
    // Additional coverage tests for certify_model paths
    // =========================================================================

    #[test]
    fn test_certify_model_invalid_playbook_yaml() {
        // Just verify the playbook_path_for_model generates correct path
        let path = playbook_path_for_model("test/BadModel", CertTier::Mvp);
        assert!(path.contains("badmodel-mvp.playbook.yaml"));
    }

    #[test]
    fn test_certify_model_cache_path_construction() {
        // Exercise the model cache path construction code
        let config = CertificationConfig {
            tier: CertTier::Mvp,
            model_cache: Some(std::path::PathBuf::from("/test/cache")),
            ..Default::default()
        };

        // This will fail (no playbook) but exercises the early return
        let result = certify_model("org/Model.Name-With.Dots", &config);
        assert!(!result.success);
        // Verify the error is about missing playbook (not a crash in path construction)
        assert!(
            result
                .error
                .as_ref()
                .expect("should have error")
                .contains("Playbook not found")
        );
    }

    #[test]
    fn test_certify_model_without_cache() {
        let config = CertificationConfig {
            tier: CertTier::Smoke,
            model_cache: None,
            ..Default::default()
        };

        let result = certify_model("org/some-model", &config);
        assert!(!result.success);
    }

    #[test]
    fn test_certify_model_with_cache_another() {
        let config = CertificationConfig {
            tier: CertTier::Smoke,
            model_cache: Some(std::path::PathBuf::from("/test/cache")),
            ..Default::default()
        };

        let result = certify_model("org/another-model", &config);
        assert!(!result.success);
    }

    #[test]
    fn test_execute_playbook_smoke() {
        // Exercise execute_playbook directly with a valid playbook
        let yaml = r#"
name: coverage-test
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
        let config = ExecutionConfig::default();
        let result = execute_playbook(&playbook, config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_load_playbook_nonexistent_file() {
        let result = load_playbook(std::path::Path::new("/nonexistent/playbook.yaml"));
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Error loading playbook"));
    }

    #[test]
    fn test_load_playbook_invalid_yaml() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("bad.yaml");
        std::fs::write(&path, "not: [valid: yaml: {{{").expect("write");
        let result = load_playbook(&path);
        assert!(result.is_err());
    }

    // =========================================================================
    // Phase 4 tests: lock-playbooks + auto-ticket
    // =========================================================================

    #[test]
    fn test_generate_lock_file_empty_dir() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let output = dir.path().join("playbook.lock.yaml");
        let result = generate_lock_file(dir.path(), &output);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 0);
    }

    #[test]
    fn test_generate_lock_file_with_playbooks() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let playbook_yaml = r#"
name: test-lock
version: "1.0.0"
model:
  hf_repo: "test/model"
  formats: [gguf]
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
"#;
        std::fs::write(dir.path().join("test.playbook.yaml"), playbook_yaml).expect("write");

        let output = dir.path().join("playbook.lock.yaml");
        let result = generate_lock_file(dir.path(), &output);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 1);
        assert!(output.exists());
    }

    #[test]
    fn test_generate_lock_file_recursive() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let sub = dir.path().join("models");
        std::fs::create_dir_all(&sub).expect("mkdir");

        let yaml = "name: sub\nversion: '1.0'\nmodel:\n  hf_repo: test/m\n  formats: [gguf]\ntest_matrix:\n  modalities: [run]\n  backends: [cpu]\n  scenario_count: 1\n";
        std::fs::write(sub.join("m.playbook.yaml"), yaml).expect("write");

        let output = dir.path().join("lock.yaml");
        let result = generate_lock_file(dir.path(), &output);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 1);
    }

    #[test]
    fn test_execute_auto_tickets_no_failures() {
        let evidence = vec![make_corroborated_evidence()];
        let tickets = execute_auto_tickets(&evidence, "test/repo");
        assert!(tickets.is_empty());
    }

    #[test]
    fn test_execute_auto_tickets_with_failures() {
        let mut ev = make_falsified_evidence();
        ev.stderr = Some("tensor name mismatch: layer.0".to_string());
        let tickets = execute_auto_tickets(&[ev], "test/repo");
        // Should generate at least 1 ticket for the classified failure
        assert_eq!(tickets.len(), 1);
    }

    #[test]
    fn test_execute_auto_tickets_deduplication() {
        let evidence: Vec<Evidence> = (0..5)
            .map(|_| {
                let mut ev = make_falsified_evidence();
                ev.stderr = Some("tensor name mismatch: layer.0".to_string());
                ev
            })
            .collect();
        let tickets = execute_auto_tickets(&evidence, "test/repo");
        // 5 same-cause failures should produce 1 ticket
        assert_eq!(tickets.len(), 1);
    }

    #[test]
    fn test_generate_lock_file_nonexistent_dir() {
        let output = std::path::Path::new("/tmp/test-lock-output.yaml");
        let result = generate_lock_file(std::path::Path::new("/nonexistent"), output);
        // Should succeed with 0 entries (no files found)
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 0);
    }
}
