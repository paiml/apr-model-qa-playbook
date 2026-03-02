
use super::*;
use crate::command::MockCommandRunner;
use apr_qa_gen::{Backend, Format, Modality, ModelId, QaScenario};

fn import_scenario() -> QaScenario {
    QaScenario::new(
        ModelId::new("test", "model"),
        Modality::Import,
        Backend::Cpu,
        Format::Apr,
        "import:gguf".to_string(),
        42,
    )
}

#[test]
fn test_import_battery_all_pass() {
    let mock_runner = MockCommandRunner::new();
    let config = ExecutionConfig {
        model_path: Some("/test/model.gguf".to_string()),
        failure_policy: FailurePolicy::CollectAll,
        ..Default::default()
    };
    let executor = Executor::with_runner(config, Arc::new(mock_runner));
    let scenario = import_scenario();
    let results = executor.run_import_battery("/test/model.gguf", &scenario, "gguf");

    // Should produce 5 evidence items
    assert_eq!(results.len(), 5, "Expected 5 battery checks, got {}", results.len());

    // Check gate IDs
    let gate_ids: Vec<&str> = results.iter().map(|e| e.gate_id.as_str()).collect();
    assert!(gate_ids.contains(&"T2-IMPORT-001"), "Missing import exit gate");
    assert!(gate_ids.contains(&"T2-IMPORT-SIZE-001"), "Missing size gate");
    assert!(gate_ids.contains(&"T2-IMPORT-TENSOR-001"), "Missing tensor count gate");
    assert!(gate_ids.contains(&"T2-IMPORT-LOAD-001"), "Missing load gate");
    assert!(gate_ids.contains(&"T2-IMPORT-INFER-001"), "Missing inference gate");

    // Primary check should pass
    assert!(results[0].outcome.is_pass(), "Primary import check should pass");
}

#[test]
fn test_import_battery_fail_stops_early() {
    let mock_runner = MockCommandRunner::new().with_import_failure();
    let config = ExecutionConfig {
        model_path: Some("/test/model.gguf".to_string()),
        failure_policy: FailurePolicy::CollectAll,
        ..Default::default()
    };
    let executor = Executor::with_runner(config, Arc::new(mock_runner));
    let scenario = import_scenario();
    let results = executor.run_import_battery("/test/model.gguf", &scenario, "gguf");

    assert_eq!(results.len(), 1, "Should stop after primary failure");
    assert!(results[0].outcome.is_fail(), "Primary check should fail");
    assert_eq!(results[0].gate_id, "T2-IMPORT-001");
}

#[test]
fn test_import_battery_validation_failure() {
    let mock_runner = MockCommandRunner::new().with_validate_strict_failure();
    let config = ExecutionConfig {
        model_path: Some("/test/model.gguf".to_string()),
        failure_policy: FailurePolicy::CollectAll,
        ..Default::default()
    };
    let executor = Executor::with_runner(config, Arc::new(mock_runner));
    let scenario = import_scenario();
    let results = executor.run_import_battery("/test/model.gguf", &scenario, "gguf");

    assert_eq!(results.len(), 5);

    let load_result = results.iter().find(|e| e.gate_id == "T2-IMPORT-LOAD-001").unwrap();
    assert!(load_result.outcome.is_fail(), "Load validation should fail");
}

#[test]
fn test_import_battery_mqs_category() {
    let scenario = import_scenario();
    assert_eq!(scenario.mqs_category(), "T2");
}

#[test]
fn test_import_modality_is_transformation() {
    assert!(Modality::Import.is_transformation());
}
