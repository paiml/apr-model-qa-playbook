
use super::*;
use crate::command::MockCommandRunner;
use apr_qa_gen::{Backend, Format, Modality, ModelId, QaScenario};

fn serve_scenario() -> QaScenario {
    QaScenario::new(
        ModelId::new("test", "model"),
        Modality::Serve,
        Backend::Cpu,
        Format::Gguf,
        "2+2=".to_string(),
        42,
    )
}

/// Helper: create a mock runner that passes health checks and returns valid responses
fn mock_with_healthy_server() -> MockCommandRunner {
    MockCommandRunner::new()
        .with_http_get_response(r#"{"status":"healthy"}"#)
        .with_http_post_response(r#"{"choices":[{"text":"The answer is 4."}]}"#)
}

#[test]
fn test_serve_battery_all_endpoints_pass() {
    let mock_runner = mock_with_healthy_server();
    let config = ExecutionConfig {
        model_path: Some("/test/model.gguf".to_string()),
        failure_policy: FailurePolicy::CollectAll,
        ..Default::default()
    };
    let executor = Executor::with_runner(config, Arc::new(mock_runner));
    let scenario = serve_scenario();
    let results = executor.run_serve_battery("/test/model.gguf", &scenario, true);

    // Should produce 8 evidence items (one per check)
    assert_eq!(results.len(), 8, "Expected 8 battery checks, got {}", results.len());

    // Check gate IDs
    let gate_ids: Vec<&str> = results.iter().map(|e| e.gate_id.as_str()).collect();
    assert!(gate_ids.contains(&"F-A5-001"), "Missing primary generate gate");
    assert!(gate_ids.contains(&"F-A5-COMP-001"), "Missing v1/completions gate");
    assert!(gate_ids.contains(&"F-A5-CHAT-001"), "Missing v1/chat gate");
    assert!(gate_ids.contains(&"F-A5-STREAM-001"), "Missing streaming gate");
    assert!(gate_ids.contains(&"F-A5-STOP-001"), "Missing stop sequence gate");
    assert!(gate_ids.contains(&"F-A5-ERR-001"), "Missing error resilience gate");
    assert!(gate_ids.contains(&"F-A5-INFO-001"), "Missing server info gate");
    assert!(gate_ids.contains(&"F-A5-METRICS-001"), "Missing metrics gate");

    // Primary check should pass (mock returns valid response)
    assert!(
        results[0].outcome.is_pass(),
        "Primary generate check should pass"
    );
}

#[test]
fn test_serve_battery_primary_fail_skips_rest() {
    // Health check passes but HTTP POST fails
    let mock_runner = MockCommandRunner::new()
        .with_http_get_response(r#"{"status":"healthy"}"#)
        .with_http_post_failure();
    let config = ExecutionConfig {
        model_path: Some("/test/model.gguf".to_string()),
        failure_policy: FailurePolicy::CollectAll,
        ..Default::default()
    };
    let executor = Executor::with_runner(config, Arc::new(mock_runner));
    let scenario = serve_scenario();
    let results = executor.run_serve_battery("/test/model.gguf", &scenario, true);

    // Primary generate failed → only 1 evidence (rest skipped)
    assert_eq!(results.len(), 1, "Should only have primary evidence when generate fails");
    assert!(results[0].outcome.is_fail(), "Primary check should fail");
    assert_eq!(results[0].gate_id, "F-A5-001");
}

#[test]
fn test_serve_battery_chat_format() {
    // Use the default healthy server mock — all http_post responses return
    // "The answer is 4." which passes the arithmetic oracle for the primary check,
    // allowing the battery to reach the chat endpoint check.
    let mock_runner = mock_with_healthy_server();
    let config = ExecutionConfig {
        model_path: Some("/test/model.gguf".to_string()),
        ..Default::default()
    };
    let executor = Executor::with_runner(config, Arc::new(mock_runner));
    let scenario = serve_scenario();
    let results = executor.run_serve_battery("/test/model.gguf", &scenario, true);

    // Find the chat check
    let chat = results.iter().find(|e| e.gate_id == "F-A5-CHAT-001");
    assert!(chat.is_some(), "Chat evidence should exist");
    assert!(
        chat.unwrap().outcome.is_pass(),
        "Chat check should pass with valid response"
    );
}

#[test]
fn test_serve_battery_sse_valid() {
    let valid_sse = "data: {\"text\":\"hello\"}\n\ndata: {\"text\":\" world\"}\n\ndata: [DONE]\n";
    assert!(Executor::verify_sse_response(valid_sse));
}

#[test]
fn test_serve_battery_sse_invalid_no_done() {
    let invalid = "data: {\"text\":\"hello\"}\n\ndata: {\"text\":\" world\"}\n";
    assert!(!Executor::verify_sse_response(invalid));
}

#[test]
fn test_serve_battery_sse_invalid_no_prefix() {
    let invalid = "{\"text\":\"hello\"}\n{\"text\":\" world\"}\ndata: [DONE]\n";
    assert!(!Executor::verify_sse_response(invalid));
}

#[test]
fn test_serve_battery_sse_empty() {
    assert!(!Executor::verify_sse_response(""));
    assert!(!Executor::verify_sse_response("\n\n\n"));
}

#[test]
fn test_serve_battery_malformed_survives() {
    let mock_runner = mock_with_healthy_server();
    let config = ExecutionConfig {
        model_path: Some("/test/model.gguf".to_string()),
        ..Default::default()
    };
    let executor = Executor::with_runner(config, Arc::new(mock_runner));
    let scenario = serve_scenario();
    let results = executor.run_serve_battery("/test/model.gguf", &scenario, true);

    let malformed = results.iter().find(|e| e.gate_id == "F-A5-ERR-001");
    assert!(malformed.is_some(), "Malformed check evidence should exist");
    // With mock, http_get succeeds with "healthy" → server is healthy → corroborated
    assert!(
        malformed.unwrap().outcome.is_pass(),
        "Server should survive malformed request"
    );
}

#[test]
fn test_serve_battery_spawn_failure() {
    let mock_runner = MockCommandRunner::new()
        .with_spawn_serve_failure();
    let config = ExecutionConfig {
        model_path: Some("/test/model.gguf".to_string()),
        ..Default::default()
    };
    let executor = Executor::with_runner(config, Arc::new(mock_runner));
    let scenario = serve_scenario();
    let results = executor.run_serve_battery("/test/model.gguf", &scenario, true);

    // Spawn failed → 1 failure evidence
    assert_eq!(results.len(), 1);
    assert!(results[0].outcome.is_fail());
    assert!(results[0].reason.contains("Failed to spawn serve"));
}

#[test]
fn test_serve_battery_gpu_backend_gate_ids() {
    let mock_runner = mock_with_healthy_server();
    let config = ExecutionConfig {
        model_path: Some("/test/model.gguf".to_string()),
        failure_policy: FailurePolicy::CollectAll,
        ..Default::default()
    };
    let executor = Executor::with_runner(config, Arc::new(mock_runner));
    let scenario = QaScenario::new(
        ModelId::new("test", "model"),
        Modality::Serve,
        Backend::Gpu,
        Format::Gguf,
        "2+2=".to_string(),
        42,
    );
    let results = executor.run_serve_battery("/test/model.gguf", &scenario, false);

    // GPU serve scenarios use A6 category
    let gate_ids: Vec<&str> = results.iter().map(|e| e.gate_id.as_str()).collect();
    assert!(gate_ids.contains(&"F-A6-001"), "GPU serve should use A6 category");
    assert!(gate_ids.contains(&"F-A6-CHAT-001"), "GPU chat should use A6");
}

// ── Direct check method tests (failure paths) ──────────────────────

#[test]
fn test_check_serve_generate_oracle_falsified() {
    // Response doesn't satisfy the arithmetic oracle for "2+2="
    let mock_runner = MockCommandRunner::new()
        .with_http_post_response(r#"{"choices":[{"text":"I don't know"}]}"#);
    let config = ExecutionConfig {
        model_path: Some("/test/model.gguf".to_string()),
        ..Default::default()
    };
    let executor = Executor::with_runner(config, Arc::new(mock_runner));
    let scenario = serve_scenario();
    let start = Instant::now();
    let ev = executor.check_serve_generate(8080, &scenario, &start);
    assert!(ev.outcome.is_fail(), "Oracle should falsify wrong answer");
    assert_eq!(ev.gate_id, "F-A5-001");
}

#[test]
fn test_check_serve_v1_completions_failure() {
    let mock_runner = MockCommandRunner::new().with_http_post_failure();
    let config = ExecutionConfig::default();
    let executor = Executor::with_runner(config, Arc::new(mock_runner));
    let scenario = serve_scenario();
    let start = Instant::now();
    let ev = executor.check_serve_v1_completions(8080, &scenario, &start);
    assert!(ev.outcome.is_fail());
    assert_eq!(ev.gate_id, "F-A5-COMP-001");
    assert!(ev.reason.contains("v1/completions failed"));
}

#[test]
fn test_check_serve_v1_chat_failure() {
    let mock_runner = MockCommandRunner::new().with_http_post_failure();
    let config = ExecutionConfig::default();
    let executor = Executor::with_runner(config, Arc::new(mock_runner));
    let scenario = serve_scenario();
    let start = Instant::now();
    let ev = executor.check_serve_v1_chat(8080, &scenario, &start);
    assert!(ev.outcome.is_fail());
    assert_eq!(ev.gate_id, "F-A5-CHAT-001");
    assert!(ev.reason.contains("v1/chat/completions failed"));
}

#[test]
fn test_check_serve_streaming_request_failure() {
    let mock_runner = MockCommandRunner::new().with_http_post_failure();
    let config = ExecutionConfig::default();
    let executor = Executor::with_runner(config, Arc::new(mock_runner));
    let scenario = serve_scenario();
    let start = Instant::now();
    let ev = executor.check_serve_streaming(8080, &scenario, &start);
    assert!(ev.outcome.is_fail());
    assert_eq!(ev.gate_id, "F-A5-STREAM-001");
    assert!(ev.reason.contains("Streaming request failed"));
}

#[test]
fn test_check_serve_streaming_valid_sse() {
    // Return valid SSE so the corroborated branch is hit
    let sse_body = "data: {\"text\":\"hello\"}\n\ndata: [DONE]\n";
    let mock_runner = MockCommandRunner::new()
        .with_http_post_response(sse_body);
    let config = ExecutionConfig::default();
    let executor = Executor::with_runner(config, Arc::new(mock_runner));
    let scenario = serve_scenario();
    let start = Instant::now();
    let ev = executor.check_serve_streaming(8080, &scenario, &start);
    assert!(ev.outcome.is_pass(), "Valid SSE should be corroborated");
    assert_eq!(ev.gate_id, "F-A5-STREAM-001");
}

#[test]
fn test_check_serve_stop_sequence_request_failure() {
    let mock_runner = MockCommandRunner::new().with_http_post_failure();
    let config = ExecutionConfig::default();
    let executor = Executor::with_runner(config, Arc::new(mock_runner));
    let scenario = serve_scenario();
    let start = Instant::now();
    let ev = executor.check_serve_stop_sequence(8080, &scenario, &start);
    assert!(ev.outcome.is_fail());
    assert_eq!(ev.gate_id, "F-A5-STOP-001");
    assert!(ev.reason.contains("Stop sequence request failed"));
}

#[test]
fn test_check_serve_stop_sequence_not_honored() {
    // Response contains "5" — stop sequence was not honored
    let mock_runner = MockCommandRunner::new()
        .with_http_post_response(r#"{"choices":[{"text":"Count: 1, 2, 3, 4, 5, 6"}]}"#);
    let config = ExecutionConfig::default();
    let executor = Executor::with_runner(config, Arc::new(mock_runner));
    let scenario = serve_scenario();
    let start = Instant::now();
    let ev = executor.check_serve_stop_sequence(8080, &scenario, &start);
    assert!(ev.outcome.is_fail());
    assert_eq!(ev.gate_id, "F-A5-STOP-001");
    assert!(ev.reason.contains("Stop sequence not honored"));
}

#[test]
fn test_check_serve_malformed_server_unhealthy() {
    // http_get fails → server unhealthy after malformed request
    let mock_runner = MockCommandRunner::new()
        .with_http_get_failure();
    let config = ExecutionConfig::default();
    let executor = Executor::with_runner(config, Arc::new(mock_runner));
    let scenario = serve_scenario();
    let start = Instant::now();
    let ev = executor.check_serve_malformed(8080, &scenario, &start);
    assert!(ev.outcome.is_fail());
    assert_eq!(ev.gate_id, "F-A5-ERR-001");
    assert!(ev.reason.contains("unhealthy after malformed"));
}

#[test]
fn test_check_serve_info_failure() {
    let mock_runner = MockCommandRunner::new().with_http_get_failure();
    let config = ExecutionConfig::default();
    let executor = Executor::with_runner(config, Arc::new(mock_runner));
    let scenario = serve_scenario();
    let start = Instant::now();
    let ev = executor.check_serve_info(8080, &scenario, &start);
    assert!(ev.outcome.is_fail());
    assert_eq!(ev.gate_id, "F-A5-INFO-001");
    assert!(ev.reason.contains("GET / failed"));
}

#[test]
fn test_check_serve_info_empty_response() {
    let mock_runner = MockCommandRunner::new()
        .with_http_get_response("");
    let config = ExecutionConfig::default();
    let executor = Executor::with_runner(config, Arc::new(mock_runner));
    let scenario = serve_scenario();
    let start = Instant::now();
    let ev = executor.check_serve_info(8080, &scenario, &start);
    assert!(ev.outcome.is_fail());
    assert_eq!(ev.gate_id, "F-A5-INFO-001");
}

#[test]
fn test_check_serve_metrics_failure() {
    let mock_runner = MockCommandRunner::new().with_http_get_failure();
    let config = ExecutionConfig::default();
    let executor = Executor::with_runner(config, Arc::new(mock_runner));
    let scenario = serve_scenario();
    let start = Instant::now();
    let ev = executor.check_serve_metrics(8080, &scenario, &start);
    assert!(ev.outcome.is_fail());
    assert_eq!(ev.gate_id, "F-A5-METRICS-001");
    assert!(ev.reason.contains("GET /metrics failed"));
}

#[test]
fn test_check_serve_metrics_empty_response() {
    let mock_runner = MockCommandRunner::new()
        .with_http_get_response("");
    let config = ExecutionConfig::default();
    let executor = Executor::with_runner(config, Arc::new(mock_runner));
    let scenario = serve_scenario();
    let start = Instant::now();
    let ev = executor.check_serve_metrics(8080, &scenario, &start);
    assert!(ev.outcome.is_fail());
    assert_eq!(ev.gate_id, "F-A5-METRICS-001");
}

#[test]
fn test_serve_battery_server_not_ready() {
    // Health check never returns "healthy" — server times out
    // Use minimal timeout to avoid slow test
    let mock_runner = MockCommandRunner::new()
        .with_http_get_response(r#"{"status":"loading"}"#);
    let config = ExecutionConfig {
        model_path: Some("/test/model.gguf".to_string()),
        default_timeout_ms: 2000, // 2s → 1 poll iteration at most
        ..Default::default()
    };
    let executor = Executor::with_runner(config, Arc::new(mock_runner));
    let scenario = serve_scenario();
    let results = executor.run_serve_battery("/test/model.gguf", &scenario, true);

    assert_eq!(results.len(), 1);
    assert!(results[0].outcome.is_fail());
    assert!(results[0].reason.contains("Server failed to become ready"));
}

#[test]
fn test_execute_scenarios_partitions_serve() {
    let mock_runner = mock_with_healthy_server()
        .with_inference_response("The answer is 4.");
    let config = ExecutionConfig {
        model_path: Some("/test/model.gguf".to_string()),
        failure_policy: FailurePolicy::CollectAll,
        ..Default::default()
    };
    let mut executor = Executor::with_runner(config, Arc::new(mock_runner));

    let run_scenario = QaScenario::new(
        ModelId::new("test", "model"),
        Modality::Run,
        Backend::Cpu,
        Format::Gguf,
        "2+2=".to_string(),
        42,
    );
    let serve_scenario_1 = QaScenario::new(
        ModelId::new("test", "model"),
        Modality::Serve,
        Backend::Cpu,
        Format::Gguf,
        "2+2=".to_string(),
        42,
    );

    let scenarios = vec![run_scenario, serve_scenario_1];
    let (passed, failed, _skipped) = executor.execute_scenarios(scenarios, "test");

    // Run scenario: 1 evidence, Serve battery: 8 evidence
    // Total passed should be > 1 (at minimum the run + battery checks)
    assert!(
        passed >= 2,
        "Should have at least run pass + some battery passes, got passed={passed} failed={failed}"
    );
}
