
use super::*;
use crate::command::MockCommandRunner;
use apr_qa_gen::{Backend, Format, Modality, ModelId, QaScenario};

fn prune_scenario() -> QaScenario {
    QaScenario::new(
        ModelId::new("test", "model"),
        Modality::Prune,
        Backend::Cpu,
        Format::Apr,
        "prune:magnitude:0.5".to_string(),
        42,
    )
}

#[test]
fn test_prune_battery_all_pass() {
    let mock_runner = MockCommandRunner::new();
    let config = ExecutionConfig {
        model_path: Some("/test/model.safetensors".to_string()),
        failure_policy: FailurePolicy::CollectAll,
        ..Default::default()
    };
    let executor = Executor::with_runner(config, Arc::new(mock_runner));
    let scenario = prune_scenario();
    let results = executor.run_prune_battery("/test/model.safetensors", &scenario, "magnitude", 0.5);

    // Should produce 6 evidence items
    assert_eq!(results.len(), 6, "Expected 6 battery checks, got {}", results.len());

    // Check gate IDs
    let gate_ids: Vec<&str> = results.iter().map(|e| e.gate_id.as_str()).collect();
    assert!(gate_ids.contains(&"T3-PRUNE-001"), "Missing prune exit gate");
    assert!(gate_ids.contains(&"T3-PRUNE-SIZE-001"), "Missing size gate");
    assert!(gate_ids.contains(&"T3-PRUNE-RATIO-001"), "Missing ratio gate");
    assert!(gate_ids.contains(&"T3-PRUNE-LOAD-001"), "Missing load gate");
    assert!(gate_ids.contains(&"T3-PRUNE-INFER-001"), "Missing inference gate");
    assert!(gate_ids.contains(&"T3-PRUNE-TENSOR-001"), "Missing tensor count gate");

    // Primary check should pass
    assert!(results[0].outcome.is_pass(), "Primary prune check should pass");
}

#[test]
fn test_prune_battery_fail_stops_early() {
    let mock_runner = MockCommandRunner::new().with_prune_failure();
    let config = ExecutionConfig {
        model_path: Some("/test/model.safetensors".to_string()),
        failure_policy: FailurePolicy::CollectAll,
        ..Default::default()
    };
    let executor = Executor::with_runner(config, Arc::new(mock_runner));
    let scenario = prune_scenario();
    let results = executor.run_prune_battery("/test/model.safetensors", &scenario, "magnitude", 0.5);

    assert_eq!(results.len(), 1, "Should stop after primary failure");
    assert!(results[0].outcome.is_fail(), "Primary check should fail");
    assert_eq!(results[0].gate_id, "T3-PRUNE-001");
}

#[test]
fn test_prune_battery_validation_failure() {
    let mock_runner = MockCommandRunner::new().with_validate_strict_failure();
    let config = ExecutionConfig {
        model_path: Some("/test/model.safetensors".to_string()),
        failure_policy: FailurePolicy::CollectAll,
        ..Default::default()
    };
    let executor = Executor::with_runner(config, Arc::new(mock_runner));
    let scenario = prune_scenario();
    let results = executor.run_prune_battery("/test/model.safetensors", &scenario, "magnitude", 0.5);

    assert_eq!(results.len(), 6);

    let load_result = results.iter().find(|e| e.gate_id == "T3-PRUNE-LOAD-001").unwrap();
    assert!(load_result.outcome.is_fail(), "Load validation should fail");
}

#[test]
fn test_prune_battery_inference_failure() {
    let mock_runner = MockCommandRunner::new().with_inference_failure();
    let config = ExecutionConfig {
        model_path: Some("/test/model.safetensors".to_string()),
        failure_policy: FailurePolicy::CollectAll,
        ..Default::default()
    };
    let executor = Executor::with_runner(config, Arc::new(mock_runner));
    let scenario = prune_scenario();
    let results = executor.run_prune_battery("/test/model.safetensors", &scenario, "magnitude", 0.5);

    assert_eq!(results.len(), 6);

    let infer_result = results.iter().find(|e| e.gate_id == "T3-PRUNE-INFER-001").unwrap();
    assert!(infer_result.outcome.is_fail(), "Inference should fail");
}

#[test]
fn test_prune_battery_mqs_category() {
    let scenario = prune_scenario();
    assert_eq!(scenario.mqs_category(), "T3");
}

#[test]
fn test_prune_modality_is_transformation() {
    assert!(Modality::Prune.is_transformation());
}
