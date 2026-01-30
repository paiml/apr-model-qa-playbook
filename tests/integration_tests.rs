//! Integration tests for apr-model-qa-playbook
//!
//! Tests the full pipeline from scenario generation through execution and reporting.

use apr_qa_gen::{Backend, Format, ModelId, Modality, QaScenario, ScenarioGenerator};
use apr_qa_report::{
    html::HtmlDashboard, junit::JunitReport, mqs::MqsCalculator, popperian::PopperianCalculator,
    ticket::TicketGenerator,
};
use apr_qa_runner::{Evidence, EvidenceCollector, ExecutionConfig, Executor, FailurePolicy, Playbook};

/// Test that we can generate scenarios for a model
#[test]
fn test_scenario_generation_pipeline() {
    let model = ModelId::new("Qwen", "Qwen2.5-Coder-1.5B-Instruct");
    let generator = ScenarioGenerator::new(model).with_scenarios_per_combination(10);

    let scenarios = generator.generate();

    // 3 modalities × 2 backends × 3 formats × 10 = 180 scenarios
    assert_eq!(scenarios.len(), 180);

    // Verify diversity
    let run_count = scenarios
        .iter()
        .filter(|s| s.modality == Modality::Run)
        .count();
    let cpu_count = scenarios.iter().filter(|s| s.backend == Backend::Cpu).count();
    let gguf_count = scenarios.iter().filter(|s| s.format == Format::Gguf).count();

    assert_eq!(run_count, 60); // 2 backends × 3 formats × 10
    assert_eq!(cpu_count, 90); // 3 modalities × 3 formats × 10
    assert_eq!(gguf_count, 60); // 3 modalities × 2 backends × 10
}

/// Test the MQS calculation pipeline with synthetic evidence
#[test]
fn test_mqs_calculation_pipeline() {
    let mut collector = EvidenceCollector::new();

    // Add passing evidence for various categories
    for i in 0..50 {
        let scenario = QaScenario::new(
            ModelId::new("test", "model"),
            Modality::Run,
            Backend::Cpu,
            Format::Gguf,
            "2+2=".to_string(),
            i,
        );
        collector.add(Evidence::corroborated(
            format!("F-QUAL-{i:03}"),
            scenario,
            "4",
            100,
        ));
    }

    // Add some failing evidence
    for i in 50..55 {
        let scenario = QaScenario::new(
            ModelId::new("test", "model"),
            Modality::Run,
            Backend::Cpu,
            Format::Gguf,
            "test".to_string(),
            i,
        );
        collector.add(Evidence::falsified(
            format!("F-QUAL-{i:03}"),
            scenario,
            "Wrong answer",
            "garbage",
            100,
        ));
    }

    let calculator = MqsCalculator::new();
    let score = calculator
        .calculate("test/model", &collector)
        .expect("Failed to calculate score");

    // Should have some score (not zeroed by gateway)
    assert!(score.raw_score > 0);
    assert!(score.normalized_score > 0.0);
    assert!(score.gateways_passed);
}

/// Test Popperian analysis pipeline
#[test]
fn test_popperian_analysis_pipeline() {
    let mut collector = EvidenceCollector::new();

    // Add corroborated evidence
    for i in 0..90 {
        let scenario = QaScenario::new(
            ModelId::new("test", "model"),
            Modality::Run,
            Backend::Cpu,
            Format::Gguf,
            "test".to_string(),
            i,
        );
        collector.add(Evidence::corroborated(
            format!("F-QUAL-{i:03}"),
            scenario,
            "ok",
            100,
        ));
    }

    // Add falsified evidence
    for i in 90..100 {
        let scenario = QaScenario::new(
            ModelId::new("test", "model"),
            Modality::Run,
            Backend::Cpu,
            Format::Gguf,
            "test".to_string(),
            i,
        );
        collector.add(Evidence::falsified(
            format!("F-QUAL-{i:03}"),
            scenario,
            "failed",
            "output",
            100,
        ));
    }

    let calculator = PopperianCalculator::new();
    let score = calculator.calculate("test/model", &collector);

    assert_eq!(score.corroborated, 90);
    assert_eq!(score.falsified, 10);
    assert!((score.corroboration_ratio - 0.9).abs() < 0.01);
    assert!(!score.is_strongly_corroborated());
}

/// Test report generation pipeline
#[test]
fn test_report_generation_pipeline() {
    let mut collector = EvidenceCollector::new();

    for i in 0..10 {
        let scenario = QaScenario::new(
            ModelId::new("test", "model"),
            Modality::Run,
            Backend::Cpu,
            Format::Gguf,
            "2+2=".to_string(),
            i,
        );
        collector.add(Evidence::corroborated(
            format!("F-QUAL-{i:03}"),
            scenario,
            "4",
            100,
        ));
    }

    let mqs_calc = MqsCalculator::new();
    let mqs_score = mqs_calc
        .calculate("test/model", &collector)
        .expect("Failed to calculate MQS");

    let popperian_calc = PopperianCalculator::new();
    let popperian_score = popperian_calc.calculate("test/model", &collector);

    // Generate JUnit report
    let junit = JunitReport::new("test/model");
    let xml = junit
        .generate(&collector, &mqs_score)
        .expect("Failed to generate JUnit");
    assert!(xml.contains("<?xml"));
    assert!(xml.contains("testsuite"));
    assert!(xml.contains("tests=\"10\""));

    // Generate HTML report
    let dashboard = HtmlDashboard::new("Test Report");
    let html = dashboard
        .generate(&mqs_score, &popperian_score, &collector)
        .expect("Failed to generate HTML");
    assert!(html.contains("<!DOCTYPE html>"));
    assert!(html.contains("test/model"));
}

/// Test ticket generation pipeline
#[test]
fn test_ticket_generation_pipeline() {
    let scenario = QaScenario::new(
        ModelId::new("test", "model"),
        Modality::Run,
        Backend::Cpu,
        Format::Gguf,
        "test".to_string(),
        42,
    );

    let evidence = vec![
        Evidence::crashed("F-STAB-001", scenario.clone(), "SIGSEGV", -11, 0),
        Evidence::crashed("F-STAB-001", scenario.clone(), "SIGSEGV", -11, 0),
    ];

    let generator = TicketGenerator::new("paiml/aprender").with_min_occurrences(2);
    let tickets = generator.generate_from_evidence(&evidence);

    assert_eq!(tickets.len(), 1);
    assert!(tickets[0].is_black_swan);
    assert!(tickets[0].title.contains("F-STAB-001"));
}

/// Test playbook parsing and execution
#[test]
fn test_playbook_execution_pipeline() {
    let yaml = r#"
name: integration-test-playbook
version: "1.0.0"
model:
  hf_repo: "test/model"
  formats: [gguf]
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 5
"#;

    let playbook = Playbook::from_yaml(yaml).expect("Failed to parse playbook");

    assert_eq!(playbook.name, "integration-test-playbook");
    assert_eq!(playbook.total_tests(), 5);

    let config = ExecutionConfig {
        dry_run: true,
        failure_policy: FailurePolicy::CollectAll,
        ..Default::default()
    };

    let mut executor = Executor::with_config(config);
    let result = executor.execute(&playbook).expect("Execution failed");

    assert_eq!(result.total_scenarios, 5);
    assert_eq!(result.skipped, 5); // All skipped in dry run mode
    assert!(result.is_success()); // Dry run is considered success
}

/// Test gateway failure zeroes the score
#[test]
fn test_gateway_failure_zeroes_score() {
    let mut collector = EvidenceCollector::new();

    // Add a crash (triggers G3 gateway failure)
    let scenario = QaScenario::new(
        ModelId::new("test", "model"),
        Modality::Run,
        Backend::Cpu,
        Format::Gguf,
        "test".to_string(),
        42,
    );
    collector.add(Evidence::crashed("F-QUAL-001", scenario, "SIGSEGV", -11, 0));

    let calculator = MqsCalculator::new();
    let score = calculator
        .calculate("test/model", &collector)
        .expect("Failed to calculate");

    assert!(!score.gateways_passed);
    assert_eq!(score.raw_score, 0);
    assert_eq!(score.normalized_score, 0.0);
    assert_eq!(score.grade, "F");
}

/// Test scenario command generation for different modalities
#[test]
fn test_scenario_command_generation() {
    let model = ModelId::new("test", "model");

    let run_scenario = QaScenario::new(
        model.clone(),
        Modality::Run,
        Backend::Cpu,
        Format::Gguf,
        "Hello".to_string(),
        42,
    );
    let cmd = run_scenario.to_command("model.gguf");
    assert!(cmd.contains("apr run"));
    assert!(cmd.contains("model.gguf"));
    assert!(cmd.contains("Hello"));

    let chat_scenario = QaScenario::new(
        model.clone(),
        Modality::Chat,
        Backend::Gpu,
        Format::Gguf,
        "Hello".to_string(),
        42,
    );
    let cmd = chat_scenario.to_command("model.gguf");
    assert!(cmd.contains("apr chat"));
    assert!(cmd.contains("--gpu"));

    let serve_scenario = QaScenario::new(
        model,
        Modality::Serve,
        Backend::Cpu,
        Format::Gguf,
        "Hello".to_string(),
        42,
    );
    let cmd = serve_scenario.to_command("model.gguf");
    assert!(cmd.contains("apr serve"));
    assert!(cmd.contains("/v1/completions"));
}

/// Test evidence serialization round-trip
#[test]
fn test_evidence_serialization() {
    let scenario = QaScenario::new(
        ModelId::new("test", "model"),
        Modality::Run,
        Backend::Cpu,
        Format::Gguf,
        "2+2=".to_string(),
        42,
    );

    let evidence = Evidence::corroborated("F-QUAL-001", scenario, "The answer is 4.", 100);

    // Serialize to JSON
    let json = serde_json::to_string(&evidence).expect("Failed to serialize");
    assert!(json.contains("F-QUAL-001"));
    assert!(json.contains("Corroborated"));

    // Deserialize back
    let parsed: Evidence = serde_json::from_str(&json).expect("Failed to deserialize");
    assert_eq!(parsed.gate_id, evidence.gate_id);
    assert_eq!(parsed.outcome, evidence.outcome);
}
