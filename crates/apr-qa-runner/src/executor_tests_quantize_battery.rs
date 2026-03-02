
use super::*;
use crate::command::MockCommandRunner;
use apr_qa_gen::{Backend, Format, Modality, ModelId, QaScenario};

fn quantize_scenario() -> QaScenario {
    QaScenario::new(
        ModelId::new("test", "model"),
        Modality::Quantize,
        Backend::Cpu,
        Format::Apr,
        "quantize:q4_k_m".to_string(),
        42,
    )
}

#[test]
fn test_quantize_battery_all_pass() {
    let mock_runner = MockCommandRunner::new();
    let config = ExecutionConfig {
        model_path: Some("/test/model.safetensors".to_string()),
        failure_policy: FailurePolicy::CollectAll,
        ..Default::default()
    };
    let executor = Executor::with_runner(config, Arc::new(mock_runner));
    let scenario = quantize_scenario();
    let results = executor.run_quantize_battery("/test/model.safetensors", &scenario, "q4_k_m");

    // Should produce 6 evidence items
    assert_eq!(results.len(), 6, "Expected 6 battery checks, got {}", results.len());

    // Check gate IDs
    let gate_ids: Vec<&str> = results.iter().map(|e| e.gate_id.as_str()).collect();
    assert!(gate_ids.contains(&"T1-QUANT-001"), "Missing quantize exit gate");
    assert!(gate_ids.contains(&"T1-QUANT-SIZE-001"), "Missing size gate");
    assert!(gate_ids.contains(&"T1-QUANT-TENSOR-001"), "Missing tensor count gate");
    assert!(gate_ids.contains(&"T1-QUANT-LOAD-001"), "Missing load gate");
    assert!(gate_ids.contains(&"T1-QUANT-INFER-001"), "Missing inference gate");
    assert!(gate_ids.contains(&"T1-QUANT-DTYPE-001"), "Missing dtype gate");

    // Primary check should pass
    assert!(results[0].outcome.is_pass(), "Primary quantize check should pass");
}

#[test]
fn test_quantize_battery_fail_stops_early() {
    let mock_runner = MockCommandRunner::new().with_quantize_failure();
    let config = ExecutionConfig {
        model_path: Some("/test/model.safetensors".to_string()),
        failure_policy: FailurePolicy::CollectAll,
        ..Default::default()
    };
    let executor = Executor::with_runner(config, Arc::new(mock_runner));
    let scenario = quantize_scenario();
    let results = executor.run_quantize_battery("/test/model.safetensors", &scenario, "q4_k_m");

    // Should produce only 1 evidence item (primary failure stops battery)
    assert_eq!(results.len(), 1, "Should stop after primary failure");
    assert!(results[0].outcome.is_fail(), "Primary check should fail");
    assert_eq!(results[0].gate_id, "T1-QUANT-001");
}

#[test]
fn test_quantize_battery_validation_failure() {
    let mock_runner = MockCommandRunner::new().with_validate_strict_failure();
    let config = ExecutionConfig {
        model_path: Some("/test/model.safetensors".to_string()),
        failure_policy: FailurePolicy::CollectAll,
        ..Default::default()
    };
    let executor = Executor::with_runner(config, Arc::new(mock_runner));
    let scenario = quantize_scenario();
    let results = executor.run_quantize_battery("/test/model.safetensors", &scenario, "q4_k_m");

    // Should still have 6 checks
    assert_eq!(results.len(), 6);

    // LOAD check should fail
    let load_result = results.iter().find(|e| e.gate_id == "T1-QUANT-LOAD-001").unwrap();
    assert!(load_result.outcome.is_fail(), "Load validation should fail");
}

#[test]
fn test_quantize_battery_inference_failure() {
    let mock_runner = MockCommandRunner::new().with_inference_failure();
    let config = ExecutionConfig {
        model_path: Some("/test/model.safetensors".to_string()),
        failure_policy: FailurePolicy::CollectAll,
        ..Default::default()
    };
    let executor = Executor::with_runner(config, Arc::new(mock_runner));
    let scenario = quantize_scenario();
    let results = executor.run_quantize_battery("/test/model.safetensors", &scenario, "q4_k_m");

    assert_eq!(results.len(), 6);

    let infer_result = results.iter().find(|e| e.gate_id == "T1-QUANT-INFER-001").unwrap();
    assert!(infer_result.outcome.is_fail(), "Inference should fail");
}

#[test]
fn test_quantize_battery_mqs_category() {
    let scenario = quantize_scenario();
    assert_eq!(scenario.mqs_category(), "T1");
}

#[test]
fn test_quantize_modality_is_transformation() {
    assert!(Modality::Quantize.is_transformation());
    assert!(!Modality::Run.is_transformation());
}
