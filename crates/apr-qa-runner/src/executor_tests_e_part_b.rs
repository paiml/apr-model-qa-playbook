#[test]
fn test_g0_tensor_all_tensors_present() {
    // When all expected tensors are present, should pass
    let mock_runner = MockCommandRunner::new().with_tensor_names(vec![
        "embed.weight".to_string(),
        "model.layers.0.self_attn.q_proj.weight".to_string(),
    ]);
    let dir = make_temp_model_dir();

    // Create a temp contracts directory with a minimal family contract
    let contracts_dir = tempfile::TempDir::new().expect("create contracts dir");
    let family_yaml = r#"
family: testfamily
size_variants:
  1b:
    parameters: "1B"
    hidden_dim: 1024
    num_layers: 1
    num_heads: 8
tensor_template:
  embedding: "embed.weight"
"#;
    std::fs::write(contracts_dir.path().join("testfamily.yaml"), family_yaml)
        .expect("write family yaml");

    let config = ExecutionConfig {
        model_path: Some(dir.path().to_string_lossy().to_string()),
        run_conversion_tests: false,
        run_golden_rule_test: false,
        run_contract_tests: false,
        ..Default::default()
    };

    let mut executor = Executor::with_runner(config, Arc::new(mock_runner));
    let model_id = ModelId::new("test", "model");

    let (passed, failed) = executor.run_g0_tensor_template_check(
        dir.path(),
        &model_id,
        "testfamily",
        "1b",
        Some(contracts_dir.path().to_str().expect("path")),
    );

    // Should pass
    assert_eq!(passed, 1);
    assert_eq!(failed, 0);

    let evidence = executor.evidence().all();
    let tensor_ev = evidence
        .iter()
        .find(|e| e.gate_id == "G0-TENSOR-001")
        .expect("should have G0-TENSOR evidence");
    assert!(tensor_ev.output.contains("G0 PASS"));
}

#[test]
fn test_g0_tensor_missing_tensors() {
    // When expected tensors are missing, should fail
    let mock_runner = MockCommandRunner::new().with_tensor_names(vec![
        "some.other.tensor".to_string(), // Not the expected one
    ]);
    let dir = make_temp_model_dir();

    // Create a temp contracts directory with a minimal family contract
    let contracts_dir = tempfile::TempDir::new().expect("create contracts dir");
    let family_yaml = r#"
family: testfamily
size_variants:
  1b:
    parameters: "1B"
    hidden_dim: 1024
    num_layers: 1
    num_heads: 8
tensor_template:
  embedding: "embed.weight"
"#;
    std::fs::write(contracts_dir.path().join("testfamily.yaml"), family_yaml)
        .expect("write family yaml");

    let config = ExecutionConfig {
        model_path: Some(dir.path().to_string_lossy().to_string()),
        run_conversion_tests: false,
        run_golden_rule_test: false,
        run_contract_tests: false,
        ..Default::default()
    };

    let mut executor = Executor::with_runner(config, Arc::new(mock_runner));
    let model_id = ModelId::new("test", "model");

    let (passed, failed) = executor.run_g0_tensor_template_check(
        dir.path(),
        &model_id,
        "testfamily",
        "1b",
        Some(contracts_dir.path().to_str().expect("path")),
    );

    // Should fail
    assert_eq!(passed, 0);
    assert_eq!(failed, 1);

    let evidence = executor.evidence().all();
    let tensor_ev = evidence
        .iter()
        .find(|e| e.gate_id == "G0-TENSOR-001")
        .expect("should have G0-TENSOR evidence");
    assert!(tensor_ev.reason.contains("G0 FAIL"));
    assert!(tensor_ev.reason.contains("Missing"));
    assert!(tensor_ev.reason.contains("embed.weight"));
}

// ── parse_timing_ms tests ──────────────────────────────────────────

#[test]
fn test_parse_timing_ms_standard() {
    let output = "Output:\nHello\nCompleted in 1.5s\ntok/s: 25.0";
    assert!((parse_timing_ms(output).unwrap() - 1500.0).abs() < 0.1);
}

#[test]
fn test_parse_timing_ms_no_timing() {
    let output = "Just some output without timing";
    assert!(parse_timing_ms(output).is_none());
}

#[test]
fn test_parse_timing_ms_zero() {
    let output = "Completed in 0.0s";
    assert!((parse_timing_ms(output).unwrap()).abs() < 0.1);
}

// ── parse_throughput tests ──────────────────────────────────────────

#[test]
fn test_parse_throughput_json() {
    let output = r#"{"throughput_tps":25.0,"latency_p50_ms":78.2}"#;
    assert!((parse_throughput(output).unwrap() - 25.0).abs() < 0.1);
}

#[test]
fn test_parse_throughput_no_match() {
    let output = "no json here";
    assert!(parse_throughput(output).is_none());
}

#[test]
fn test_parse_throughput_integer() {
    let output = r#"{"throughput_tps":100,"other":0}"#;
    assert!((parse_throughput(output).unwrap() - 100.0).abs() < 0.1);
}

// ── F-OLLAMA-003 TTFT comparison test ──────────────────────────────

#[test]
fn test_ollama_parity_ttft_comparison() {
    let runner = MockCommandRunner::new().with_inference_response("Hello world");
    let runner = Arc::new(runner);

    let config = ExecutionConfig {
        run_ollama_parity: true,
        model_path: Some("/mock/model".to_string()),
        ..Default::default()
    };
    let mut executor = Executor::with_runner(config, runner);

    let yaml = r#"
name: test-ollama-ttft
version: "1.0.0"
model:
  hf_repo: "test/model"
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
ollama_parity:
  enabled: true
  model_tag: "test:latest"
  prompts: ["What is 2+2?"]
  temperature: 0.0
"#;
    let playbook: Playbook = serde_yaml::from_str(yaml).unwrap();
    let (passed, failed) = executor.run_ollama_parity_tests(Path::new("/mock/model"), &playbook);
    // F-OLLAMA-001 + F-OLLAMA-003 (TTFT) + F-OLLAMA-005 + F-OLLAMA-004
    assert!(
        passed + failed >= 2,
        "Expected at least 2 evidence items, got passed={passed} failed={failed}"
    );
}

// ── F-OLLAMA-005 GGUF loadability test ─────────────────────────────

#[test]
fn test_ollama_gguf_loadability_success() {
    let runner = Arc::new(MockCommandRunner::new());
    let config = ExecutionConfig {
        run_ollama_parity: true,
        model_path: Some("/mock/model".to_string()),
        ..Default::default()
    };
    let mut executor = Executor::with_runner(config, runner);

    let yaml = r#"
name: test-ollama-gguf
version: "1.0.0"
model:
  hf_repo: "test/model"
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
ollama_parity:
  enabled: true
  prompts: ["test"]
"#;
    let playbook: Playbook = serde_yaml::from_str(yaml).unwrap();
    let (passed, _failed) = executor.run_ollama_parity_tests(Path::new("/mock/model"), &playbook);
    // Should have F-OLLAMA-001, F-OLLAMA-005, F-OLLAMA-004
    assert!(passed >= 3, "Expected at least 3 passes, got {passed}");
    let evidence = executor.evidence().all();
    assert!(evidence.iter().any(|e| e.gate_id == "F-OLLAMA-005"));
}

#[test]
fn test_ollama_gguf_loadability_failure() {
    let runner = Arc::new(MockCommandRunner::new().with_ollama_create_failure());
    let config = ExecutionConfig {
        run_ollama_parity: true,
        model_path: Some("/mock/model".to_string()),
        ..Default::default()
    };
    let mut executor = Executor::with_runner(config, runner);

    let yaml = r#"
name: test-ollama-gguf-fail
version: "1.0.0"
model:
  hf_repo: "test/model"
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
ollama_parity:
  enabled: true
  prompts: ["test"]
"#;
    let playbook: Playbook = serde_yaml::from_str(yaml).unwrap();
    let (_passed, failed) = executor.run_ollama_parity_tests(Path::new("/mock/model"), &playbook);
    assert!(
        failed >= 1,
        "Expected at least 1 failure for create failure"
    );
    let evidence = executor.evidence().all();
    let gguf_ev = evidence
        .iter()
        .find(|e| e.gate_id == "F-OLLAMA-005")
        .unwrap();
    assert!(!gguf_ev.outcome.is_pass());
}

// ── F-OLLAMA-004 API parity test ───────────────────────────────────

#[test]
fn test_ollama_api_parity_success() {
    let runner = Arc::new(MockCommandRunner::new());
    let config = ExecutionConfig {
        run_ollama_parity: true,
        model_path: Some("/mock/model".to_string()),
        ..Default::default()
    };
    let mut executor = Executor::with_runner(config, runner);

    let yaml = r#"
name: test-ollama-api
version: "1.0.0"
model:
  hf_repo: "test/model"
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
ollama_parity:
  enabled: true
  prompts: ["test"]
"#;
    let playbook: Playbook = serde_yaml::from_str(yaml).unwrap();
    let (passed, _failed) = executor.run_ollama_parity_tests(Path::new("/mock/model"), &playbook);
    assert!(passed >= 1);
    let evidence = executor.evidence().all();
    assert!(evidence.iter().any(|e| e.gate_id == "F-OLLAMA-004"));
}

#[test]
fn test_ollama_api_parity_failure() {
    let runner = Arc::new(MockCommandRunner::new().with_http_get_failure());
    let config = ExecutionConfig {
        run_ollama_parity: true,
        model_path: Some("/mock/model".to_string()),
        ..Default::default()
    };
    let mut executor = Executor::with_runner(config, runner);

    let yaml = r#"
name: test-ollama-api-fail
version: "1.0.0"
model:
  hf_repo: "test/model"
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
ollama_parity:
  enabled: true
  prompts: ["test"]
"#;
    let playbook: Playbook = serde_yaml::from_str(yaml).unwrap();
    let (_passed, failed) = executor.run_ollama_parity_tests(Path::new("/mock/model"), &playbook);
    assert!(failed >= 1);
    let evidence = executor.evidence().all();
    let api_ev = evidence
        .iter()
        .find(|e| e.gate_id == "F-OLLAMA-004")
        .unwrap();
    assert!(!api_ev.outcome.is_pass());
}

// ── F-PERF-003 GPU/CPU ratio test ──────────────────────────────────

#[test]
fn test_perf_003_gpu_cpu_ratio() {
    let runner = Arc::new(MockCommandRunner::new().with_tps(50.0));
    let config = ExecutionConfig {
        run_profile_ci: true,
        model_path: Some("/mock/model".to_string()),
        ..Default::default()
    };
    let mut executor = Executor::with_runner(config, runner);

    let yaml = r#"
name: test-perf-003
version: "1.0.0"
model:
  hf_repo: "test/model"
test_matrix:
  modalities: [run]
  backends: [cpu, gpu]
  scenario_count: 1
profile_ci:
  enabled: true
  warmup: 1
  measure: 2
  formats: [safetensors]
  backends: [cpu, gpu]
"#;
    let playbook: Playbook = serde_yaml::from_str(yaml).unwrap();
    let model_id = playbook.model_id();
    let (passed, _failed) = executor.run_perf_gates(Path::new("/mock/model"), &model_id, &playbook);
    // F-PERF-003 (GPU/CPU ratio) + F-PERF-005 (memory)
    assert!(passed >= 2, "Expected at least 2 passes, got {passed}");
    let evidence = executor.evidence().all();
    assert!(evidence.iter().any(|e| e.gate_id == "F-PERF-003"));
    assert!(evidence.iter().any(|e| e.gate_id == "F-PERF-005"));
}

// ── F-PERF-005 memory profiling test ───────────────────────────────

#[test]
fn test_perf_005_memory_profiling_failure() {
    let runner = Arc::new(MockCommandRunner::new().with_profile_memory_failure());
    let config = ExecutionConfig {
        run_profile_ci: true,
        model_path: Some("/mock/model".to_string()),
        ..Default::default()
    };
    let mut executor = Executor::with_runner(config, runner);

    let yaml = r#"
name: test-perf-005-fail
version: "1.0.0"
model:
  hf_repo: "test/model"
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
profile_ci:
  enabled: true
  warmup: 1
  measure: 2
  backends: [cpu]
"#;
    let playbook: Playbook = serde_yaml::from_str(yaml).unwrap();
    let model_id = playbook.model_id();
    let (_passed, failed) = executor.run_perf_gates(Path::new("/mock/model"), &model_id, &playbook);
    assert!(failed >= 1);
    let evidence = executor.evidence().all();
    let mem_ev = evidence.iter().find(|e| e.gate_id == "F-PERF-005").unwrap();
    assert!(!mem_ev.outcome.is_pass());
}

// ── Integration: execute() with ollama parity enabled ─────────────

#[test]
fn test_execute_with_ollama_parity_enabled() {
    let runner =
        MockCommandRunner::new().with_inference_response("Output:\nHello\nCompleted in 0.5s");
    let config = ExecutionConfig {
        run_ollama_parity: true,
        model_path: Some("/mock/model".to_string()),
        no_gpu: true,
        ..Default::default()
    };
    let mut executor = Executor::with_runner(config, Arc::new(runner));

    let yaml = r#"
name: test-ollama-integration
version: "1.0.0"
model:
  hf_repo: "test/model"
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
ollama_parity:
  enabled: true
  prompts: ["What is 2+2?"]
"#;
    let playbook = Playbook::from_yaml(yaml).expect("Failed to parse");
    let result = executor.execute(&playbook).expect("Execution failed");
    assert!(result.total_scenarios >= 1);
    let evidence = executor.evidence().all();
    assert!(evidence.iter().any(|e| e.gate_id == "F-OLLAMA-001"));
}

// ── Integration: execute() with profile_ci (perf gates) enabled ───
