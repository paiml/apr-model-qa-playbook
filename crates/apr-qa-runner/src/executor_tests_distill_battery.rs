
use super::*;
use crate::command::MockCommandRunner;
use apr_qa_gen::{Backend, Format, Modality, ModelId, QaScenario};

fn distill_scenario() -> QaScenario {
    QaScenario::new(
        ModelId::new("test", "model"),
        Modality::Distill,
        Backend::Cpu,
        Format::Apr,
        "distill".to_string(),
        42,
    )
}

#[test]
fn test_distill_battery_all_pass() {
    let mock_runner = MockCommandRunner::new();
    let config = ExecutionConfig {
        model_path: Some("/test/teacher.safetensors".to_string()),
        failure_policy: FailurePolicy::CollectAll,
        ..Default::default()
    };
    let executor = Executor::with_runner(config, Arc::new(mock_runner));
    let scenario = distill_scenario();
    let results = executor.run_distill_battery(
        "/test/teacher.safetensors",
        &scenario,
        "/test/student.safetensors",
        "/test/data.jsonl",
    );

    // Should produce 5 evidence items
    assert_eq!(results.len(), 5, "Expected 5 battery checks, got {}", results.len());

    // Check gate IDs
    let gate_ids: Vec<&str> = results.iter().map(|e| e.gate_id.as_str()).collect();
    assert!(gate_ids.contains(&"T4-DISTILL-001"), "Missing distill exit gate");
    assert!(gate_ids.contains(&"T4-DISTILL-SIZE-001"), "Missing size gate");
    assert!(gate_ids.contains(&"T4-DISTILL-LOAD-001"), "Missing load gate");
    assert!(gate_ids.contains(&"T4-DISTILL-INFER-001"), "Missing inference gate");
    assert!(gate_ids.contains(&"T4-DISTILL-LOSS-001"), "Missing loss gate");

    // Primary check should pass
    assert!(results[0].outcome.is_pass(), "Primary distill check should pass");
}

#[test]
fn test_distill_battery_fail_stops_early() {
    let mock_runner = MockCommandRunner::new().with_distill_failure();
    let config = ExecutionConfig {
        model_path: Some("/test/teacher.safetensors".to_string()),
        failure_policy: FailurePolicy::CollectAll,
        ..Default::default()
    };
    let executor = Executor::with_runner(config, Arc::new(mock_runner));
    let scenario = distill_scenario();
    let results = executor.run_distill_battery(
        "/test/teacher.safetensors",
        &scenario,
        "/test/student.safetensors",
        "/test/data.jsonl",
    );

    assert_eq!(results.len(), 1, "Should stop after primary failure");
    assert!(results[0].outcome.is_fail(), "Primary check should fail");
    assert_eq!(results[0].gate_id, "T4-DISTILL-001");
}

#[test]
fn test_distill_battery_validation_failure() {
    let mock_runner = MockCommandRunner::new().with_validate_strict_failure();
    let config = ExecutionConfig {
        model_path: Some("/test/teacher.safetensors".to_string()),
        failure_policy: FailurePolicy::CollectAll,
        ..Default::default()
    };
    let executor = Executor::with_runner(config, Arc::new(mock_runner));
    let scenario = distill_scenario();
    let results = executor.run_distill_battery(
        "/test/teacher.safetensors",
        &scenario,
        "/test/student.safetensors",
        "/test/data.jsonl",
    );

    assert_eq!(results.len(), 5);

    let load_result = results.iter().find(|e| e.gate_id == "T4-DISTILL-LOAD-001").unwrap();
    assert!(load_result.outcome.is_fail(), "Load validation should fail");
}

#[test]
fn test_distill_battery_mqs_category() {
    let scenario = distill_scenario();
    assert_eq!(scenario.mqs_category(), "T4");
}

#[test]
fn test_distill_modality_is_transformation() {
    assert!(Modality::Distill.is_transformation());
    assert!(!Modality::Serve.is_transformation());
    assert!(!Modality::Chat.is_transformation());
}
