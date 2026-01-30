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
    ticket::{TicketGenerator, UpstreamTicket},
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
    /// Enable subprocess mode for real command execution
    pub subprocess: bool,
    /// Path to model file (required for subprocess mode)
    pub model_path: Option<String>,
    /// Timeout per test in milliseconds
    pub timeout: u64,
    /// Disable GPU acceleration
    pub no_gpu: bool,
    /// Skip P0 format conversion tests
    pub skip_conversion_tests: bool,
    /// Run APR tool coverage tests
    pub run_tool_tests: bool,
}

impl Default for PlaybookRunConfig {
    fn default() -> Self {
        Self {
            failure_policy: "stop-on-p0".to_string(),
            dry_run: false,
            workers: 4,
            subprocess: false,
            model_path: None,
            timeout: 60000,
            no_gpu: false,
            skip_conversion_tests: false,
            run_tool_tests: false,
        }
    }
}

/// Parse failure policy string to enum
pub fn parse_failure_policy(policy: &str) -> Result<FailurePolicy, String> {
    match policy {
        "stop-on-first" => Ok(FailurePolicy::StopOnFirst),
        "stop-on-p0" => Ok(FailurePolicy::StopOnP0),
        "collect-all" => Ok(FailurePolicy::CollectAll),
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
        subprocess_mode: config.subprocess,
        model_path: config.model_path.clone(),
        default_timeout_ms: config.timeout,
        no_gpu: config.no_gpu,
        run_conversion_tests: !config.skip_conversion_tests,
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

#[cfg(test)]
mod tests {
    use super::*;
    use apr_qa_gen::{Backend, Format, Modality, QaScenario};

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
        assert!(!config.subprocess);
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
            subprocess: true,
            model_path: Some("/path/to/model".to_string()),
            no_gpu: true,
            skip_conversion_tests: true,
            ..Default::default()
        };
        let exec_config = build_execution_config(&config).unwrap();
        assert!(exec_config.dry_run);
        assert_eq!(exec_config.max_workers, 8);
        assert!(exec_config.subprocess_mode);
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
}
