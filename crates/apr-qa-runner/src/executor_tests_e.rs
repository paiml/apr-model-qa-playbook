    use super::*;
    use apr_qa_gen::{Backend, Format, Modality, ModelId, QaScenario};
    use crate::command::MockCommandRunner;


    /// Helper: create a temp model directory with a safetensors file
    fn make_temp_model_dir() -> tempfile::TempDir {
        let dir = tempfile::TempDir::new().expect("create temp dir");
        let st_dir = dir.path().join("safetensors");
        std::fs::create_dir_all(&st_dir).expect("mkdir safetensors");
        std::fs::write(st_dir.join("model.safetensors"), b"fake").expect("write");
        dir
    }


    #[test]
    fn test_workspace_skipped_for_directory() {
        // When model_path is already a directory, workspace creation should be skipped
        let mock_runner = MockCommandRunner::new();
        let config = ExecutionConfig {
            model_path: Some("/some/directory/path".to_string()),
            run_conversion_tests: false,
            run_golden_rule_test: false,
            run_contract_tests: false,
            ..Default::default()
        };

        let mut executor = Executor::with_runner(config, Arc::new(mock_runner));
        let yaml = r#"
name: workspace-skip-test
version: "1.0.0"
model:
  hf_repo: "test/model"
  formats: [safetensors, apr]
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
"#;
        let playbook = Playbook::from_yaml(yaml).expect("parse");
        let result = executor.execute(&playbook).expect("execute");

        // No G0-FORMAT evidence should be present (workspace was skipped)
        let has_format_evidence = result
            .evidence
            .all()
            .iter()
            .any(|e| e.gate_id.starts_with("G0-FORMAT"));
        assert!(
            !has_format_evidence,
            "No G0-FORMAT evidence expected for directory model path"
        );
    }

    #[test]
    fn test_workspace_evidence_emitted() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let output_dir = dir.path().join("output");

        let model_file = dir.path().join("test.safetensors");
        std::fs::write(&model_file, b"fake-model").expect("write model");

        let mock_runner = MockCommandRunner::new();
        let config = ExecutionConfig {
            output_dir: Some(output_dir.to_string_lossy().to_string()),
            run_conversion_tests: false,
            run_golden_rule_test: false,
            run_contract_tests: false,
            ..Default::default()
        };

        let mut executor = Executor::with_runner(config, Arc::new(mock_runner));
        let model_id = ModelId::new("test", "model");
        let formats = vec![Format::SafeTensors, Format::Apr, Format::Gguf];

        let (_workspace, passed, failed) =
            executor.prepare_model_workspace(&model_file, &model_id, &formats);

        // Both APR and GGUF conversions should produce evidence
        assert_eq!(passed + failed, 2, "Should have evidence for APR and GGUF");

        let evidence = executor.evidence().all();
        let format_evidence_count = evidence
            .iter()
            .filter(|e| e.gate_id.starts_with("G0-FORMAT"))
            .count();
        assert_eq!(
            format_evidence_count, 2,
            "Should have 2 G0-FORMAT evidence entries"
        );
    }

    #[test]
    fn test_find_sibling_model_files() {
        let dir = tempfile::tempdir().expect("create temp dir");

        // Create pacha cache structure
        let model_file = dir.path().join("abc123.safetensors");
        std::fs::write(&model_file, b"model").expect("write");
        std::fs::write(dir.path().join("abc123.config.json"), b"config").expect("write");
        std::fs::write(dir.path().join("abc123.tokenizer.json"), b"tokenizer").expect("write");
        // Different model (should be excluded)
        std::fs::write(dir.path().join("def456.safetensors"), b"other").expect("write");
        std::fs::write(dir.path().join("def456.config.json"), b"other-config").expect("write");

        let siblings = Executor::find_sibling_model_files(&model_file);

        // Should find config.json and tokenizer.json for abc123 only
        assert_eq!(siblings.len(), 2, "Should find exactly 2 sibling files");

        let canonical_names: Vec<&str> = siblings.iter().map(|(_, n)| n.as_str()).collect();
        assert!(
            canonical_names.contains(&"config.json"),
            "Should find config.json"
        );
        assert!(
            canonical_names.contains(&"tokenizer.json"),
            "Should find tokenizer.json"
        );
    }

    #[test]
    fn test_find_sibling_model_files_no_siblings() {
        let dir = tempfile::tempdir().expect("create temp dir");

        let model_file = dir.path().join("lonely.safetensors");
        std::fs::write(&model_file, b"model").expect("write");

        let siblings = Executor::find_sibling_model_files(&model_file);
        assert!(siblings.is_empty(), "Should find no siblings");
    }

    #[test]
    fn test_find_sibling_model_files_non_safetensors() {
        let dir = tempfile::tempdir().expect("create temp dir");

        let model_file = dir.path().join("model.gguf");
        std::fs::write(&model_file, b"model").expect("write");

        let siblings = Executor::find_sibling_model_files(&model_file);
        assert!(
            siblings.is_empty(),
            "Should return empty for non-safetensors files"
        );
    }

    #[test]
    fn test_workspace_execute_integration_with_single_file() {
        // Integration test: execute() with a real single .safetensors file
        // should trigger workspace creation and resolve all formats
        let dir = tempfile::tempdir().expect("create temp dir");
        let output_dir = dir.path().join("output");

        let model_file = dir.path().join("test.safetensors");
        std::fs::write(&model_file, b"fake-model").expect("write model");

        let mock_runner =
            MockCommandRunner::new().with_pull_model_path(model_file.to_string_lossy().to_string());
        let config = ExecutionConfig {
            model_path: Some(model_file.to_string_lossy().to_string()),
            output_dir: Some(output_dir.to_string_lossy().to_string()),
            run_conversion_tests: false,
            run_golden_rule_test: false,
            run_contract_tests: false,
            ..Default::default()
        };

        let mut executor = Executor::with_runner(config, Arc::new(mock_runner));
        let yaml = r#"
name: workspace-integration
version: "1.0.0"
model:
  hf_repo: "test/model"
  formats: [safetensors, apr]
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
"#;
        let playbook = Playbook::from_yaml(yaml).expect("parse");
        let result = executor.execute(&playbook).expect("execute");

        // Verify the model_path was changed from file to workspace directory
        let final_model_path = executor.config().model_path.as_deref().unwrap_or("");
        assert!(
            final_model_path.contains("workspace"),
            "model_path should point to workspace: {final_model_path}"
        );
        assert!(
            !final_model_path.ends_with(".safetensors"),
            "model_path should not be a file: {final_model_path}"
        );

        // G0-FORMAT evidence should be present (conversion to APR)
        let has_format_evidence = result
            .evidence
            .all()
            .iter()
            .any(|e| e.gate_id.starts_with("G0-FORMAT"));
        assert!(
            has_format_evidence,
            "Should have G0-FORMAT evidence for APR conversion"
        );
    }

    // ── G0-TENSOR Template Validation Tests (PMAT-271) ─────────────────────────

    #[test]
    fn test_g0_tensor_no_family_configured() {
        // When family/size_variant are not set, G0-TENSOR should be skipped (0, 0)
        let mock_runner = MockCommandRunner::new();
        let dir = make_temp_model_dir();

        let config = ExecutionConfig {
            model_path: Some(dir.path().to_string_lossy().to_string()),
            run_conversion_tests: false,
            run_golden_rule_test: false,
            run_contract_tests: false,
            ..Default::default()
        };

        let mut executor = Executor::with_runner(config, Arc::new(mock_runner));

        // Playbook without family/size_variant
        let yaml = r#"
name: no-family-test
version: "1.0.0"
model:
  hf_repo: "test/model"
  formats: [safetensors]
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
"#;
        let playbook = Playbook::from_yaml(yaml).expect("parse");
        let result = executor.execute(&playbook).expect("execute");

        // No G0-TENSOR evidence when family not configured
        let has_tensor_evidence = result
            .evidence
            .all()
            .iter()
            .any(|e| e.gate_id == "G0-TENSOR-001");
        assert!(
            !has_tensor_evidence,
            "Should NOT have G0-TENSOR evidence when family not configured"
        );
    }

    #[test]
    fn test_g0_tensor_family_contract_not_found() {
        // When family is set but contract doesn't exist, should skip gracefully
        let mock_runner = MockCommandRunner::new();
        let dir = make_temp_model_dir();

        let config = ExecutionConfig {
            model_path: Some(dir.path().to_string_lossy().to_string()),
            run_conversion_tests: false,
            run_golden_rule_test: false,
            run_contract_tests: false,
            ..Default::default()
        };

        let mut executor = Executor::with_runner(config, Arc::new(mock_runner));
        let model_id = ModelId::new("test", "model");

        // Call with a nonexistent family
        let (passed, failed) = executor.run_g0_tensor_template_check(
            dir.path(),
            &model_id,
            "nonexistent-family",
            "1b",
            Some("/nonexistent/path"),
        );

        // Should skip (0, 0) with evidence
        assert_eq!(passed, 0);
        assert_eq!(failed, 0);

        let evidence = executor.evidence().all();
        let tensor_ev = evidence
            .iter()
            .find(|e| e.gate_id == "G0-TENSOR-001")
            .expect("should have G0-TENSOR evidence");
        assert!(tensor_ev.output.contains("G0 SKIP"));
        assert!(tensor_ev.output.contains("Family contract not found"));
    }

    #[test]
    fn test_g0_tensor_no_safetensors_files() {
        // When there are no safetensors files, should skip
        let mock_runner = MockCommandRunner::new();
        let dir = tempfile::TempDir::new().expect("create temp dir");

        let config = ExecutionConfig {
            model_path: Some(dir.path().to_string_lossy().to_string()),
            run_conversion_tests: false,
            run_golden_rule_test: false,
            run_contract_tests: false,
            ..Default::default()
        };

        let mut executor = Executor::with_runner(config, Arc::new(mock_runner));
        let model_id = ModelId::new("test", "model");

        // Call with a valid family name but empty directory
        let (passed, failed) = executor.run_g0_tensor_template_check(
            dir.path(),
            &model_id,
            "qwen2",
            "0.5b",
            Some("/nonexistent/path"), // Will fail to load, but we also don't have safetensors
        );

        // Should skip (0, 0)
        assert_eq!(passed, 0);
        assert_eq!(failed, 0);
    }

    #[test]
    fn test_g0_tensor_inspect_returns_empty_names() {
        // When inspect doesn't return tensor names, should skip
        let mock_runner = MockCommandRunner::new().with_tensor_names(vec![]); // Empty tensor names
        let dir = make_temp_model_dir();

        let config = ExecutionConfig {
            model_path: Some(dir.path().to_string_lossy().to_string()),
            run_conversion_tests: false,
            run_golden_rule_test: false,
            run_contract_tests: false,
            ..Default::default()
        };

        let mut executor = Executor::with_runner(config, Arc::new(mock_runner));
        let model_id = ModelId::new("test", "model");

        // This will fail at registry load since aprender isn't available in tests,
        // but this tests the empty tensor_names path in isolation
        let (passed, failed) = executor.run_g0_tensor_template_check(
            dir.path(),
            &model_id,
            "qwen2",
            "0.5b",
            Some("/nonexistent/path"),
        );

        // Should skip
        assert_eq!(passed, 0);
        assert_eq!(failed, 0);
    }

    #[test]
    fn test_g0_tensor_inspect_failure() {
        // When inspect fails, should report failure
        let mock_runner = MockCommandRunner::new().with_inspect_json_failure();
        let dir = make_temp_model_dir();

        // Create a temp contracts directory with a minimal family contract
        let contracts_dir = tempfile::TempDir::new().expect("create contracts dir");
        let family_yaml = r#"
family: testfamily
size_variants:
  1b:
    parameters: "1B"
    hidden_dim: 1024
    num_layers: 12
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
        assert!(tensor_ev.reason.contains("Could not inspect"));
    }

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
        let (passed, failed) =
            executor.run_ollama_parity_tests(Path::new("/mock/model"), &playbook);
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
        let (passed, _failed) =
            executor.run_ollama_parity_tests(Path::new("/mock/model"), &playbook);
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
        let (_passed, failed) =
            executor.run_ollama_parity_tests(Path::new("/mock/model"), &playbook);
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
        let (passed, _failed) =
            executor.run_ollama_parity_tests(Path::new("/mock/model"), &playbook);
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
        let (_passed, failed) =
            executor.run_ollama_parity_tests(Path::new("/mock/model"), &playbook);
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
        let (passed, _failed) =
            executor.run_perf_gates(Path::new("/mock/model"), &model_id, &playbook);
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
        let (_passed, failed) =
            executor.run_perf_gates(Path::new("/mock/model"), &model_id, &playbook);
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

    #[test]
    fn test_execute_with_profile_ci_perf_gates() {
        let runner = MockCommandRunner::new()
            .with_tps(50.0)
            .with_inference_response("Output:\nHello\nCompleted in 0.5s");
        let config = ExecutionConfig {
            run_profile_ci: true,
            model_path: Some("/mock/model".to_string()),
            no_gpu: true,
            ..Default::default()
        };
        let mut executor = Executor::with_runner(config, Arc::new(runner));

        let yaml = r#"
name: test-perf-integration
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
  formats: [safetensors]
  backends: [cpu, gpu]
"#;
        let playbook = Playbook::from_yaml(yaml).expect("Failed to parse");
        let result = executor.execute(&playbook).expect("Execution failed");
        assert!(result.total_scenarios >= 1);
        let evidence = executor.evidence().all();
        assert!(evidence.iter().any(|e| e.gate_id == "F-PERF-003"));
        assert!(evidence.iter().any(|e| e.gate_id == "F-PERF-005"));
    }

    // ── Bug 202: Sibling-file lookup in file mode ────────────────────────

    #[test]
    fn test_resolve_model_path_file_sibling_gguf() {
        // Given a .safetensors file, resolve_model_path should find sibling .gguf
        let temp_dir = tempfile::tempdir().unwrap();
        let st_file = temp_dir.path().join("model.safetensors");
        let gguf_file = temp_dir.path().join("model.gguf");
        std::fs::write(&st_file, b"fake safetensors").unwrap();
        std::fs::write(&gguf_file, b"fake gguf").unwrap();

        let config = ExecutionConfig {
            model_path: Some(st_file.to_string_lossy().to_string()),
            ..Default::default()
        };
        let executor = Executor::with_config(config);

        let scenario = QaScenario::new(
            ModelId::new("test", "model"),
            Modality::Run,
            Backend::Cpu,
            Format::Gguf,
            "test".to_string(),
            0,
        );
        let path = executor.resolve_model_path(&scenario);
        assert!(path.is_some(), "Should find sibling .gguf file");
        assert!(path.unwrap().contains("model.gguf"));
    }

    #[test]
    fn test_resolve_model_path_file_sibling_apr() {
        // Given a .gguf file, resolve_model_path should find sibling .apr
        let temp_dir = tempfile::tempdir().unwrap();
        let gguf_file = temp_dir.path().join("model.gguf");
        let apr_file = temp_dir.path().join("model.apr");
        std::fs::write(&gguf_file, b"fake gguf").unwrap();
        std::fs::write(&apr_file, b"fake apr").unwrap();

        let config = ExecutionConfig {
            model_path: Some(gguf_file.to_string_lossy().to_string()),
            ..Default::default()
        };
        let executor = Executor::with_config(config);

        let scenario = QaScenario::new(
            ModelId::new("test", "model"),
            Modality::Run,
            Backend::Cpu,
            Format::Apr,
            "test".to_string(),
            0,
        );
        let path = executor.resolve_model_path(&scenario);
        assert!(path.is_some(), "Should find sibling .apr file");
        assert!(path.unwrap().contains("model.apr"));
    }

    #[test]
    fn test_resolve_model_path_file_sibling_not_found() {
        // Given a .safetensors file with no sibling .gguf, should return None
        let temp_dir = tempfile::tempdir().unwrap();
        let st_file = temp_dir.path().join("model.safetensors");
        std::fs::write(&st_file, b"fake safetensors").unwrap();

        let config = ExecutionConfig {
            model_path: Some(st_file.to_string_lossy().to_string()),
            ..Default::default()
        };
        let executor = Executor::with_config(config);

        let scenario = QaScenario::new(
            ModelId::new("test", "model"),
            Modality::Run,
            Backend::Cpu,
            Format::Gguf,
            "test".to_string(),
            0,
        );
        assert!(
            executor.resolve_model_path(&scenario).is_none(),
            "No sibling .gguf exists, should return None"
        );
    }

    #[test]
    fn test_resolve_model_path_file_sibling_fallback_different_stem() {
        // Given a .safetensors file with a DIFFERENT-FAMILY .gguf file in same dir,
        // prefix matching should NOT return it (avoids cross-model confusion).
        let temp_dir = tempfile::tempdir().unwrap();
        let st_file = temp_dir.path().join("abc123.safetensors");
        let gguf_file = temp_dir.path().join("other-name.gguf");
        std::fs::write(&st_file, b"fake safetensors").unwrap();
        std::fs::write(&gguf_file, b"fake gguf").unwrap();

        let config = ExecutionConfig {
            model_path: Some(st_file.to_string_lossy().to_string()),
            ..Default::default()
        };
        let executor = Executor::with_config(config);

        let scenario = QaScenario::new(
            ModelId::new("test", "model"),
            Modality::Run,
            Backend::Cpu,
            Format::Gguf,
            "test".to_string(),
            0,
        );
        let path = executor.resolve_model_path(&scenario);
        assert!(path.is_none(), "Should NOT match unrelated model family");
    }

    #[test]
    fn test_resolve_model_path_file_sibling_prefix_match() {
        // Given a GGUF with quantization suffix, should find APR with same family prefix
        let temp_dir = tempfile::tempdir().unwrap();
        let gguf_file = temp_dir.path().join("qwen2.5-coder-7b-instruct-q4k.gguf");
        let apr_file = temp_dir.path().join("qwen2.5-coder-7b-instruct.apr");
        std::fs::write(&gguf_file, b"fake gguf").unwrap();
        std::fs::write(&apr_file, b"fake apr").unwrap();

        let config = ExecutionConfig {
            model_path: Some(gguf_file.to_string_lossy().to_string()),
            ..Default::default()
        };
        let executor = Executor::with_config(config);

        let scenario = QaScenario::new(
            ModelId::new("test", "model"),
            Modality::Run,
            Backend::Cpu,
            Format::Apr,
            "test".to_string(),
            0,
        );
        let path = executor.resolve_model_path(&scenario);
        assert!(
            path.is_some(),
            "Should find APR via model family prefix match"
        );
        assert!(path.unwrap().contains("qwen2.5-coder-7b-instruct.apr"));
    }

    // ── Bug 200: Modality-aware dispatch ─────────────────────────────────

    #[test]
    fn test_subprocess_execution_chat_modality() {
        let runner = MockCommandRunner::new();
        let config = ExecutionConfig {
            model_path: Some("/mock/model.gguf".to_string()),
            ..Default::default()
        };
        let executor = Executor::with_runner(config, Arc::new(runner));

        let scenario = QaScenario::new(
            ModelId::new("test", "model"),
            Modality::Chat,
            Backend::Cpu,
            Format::Gguf,
            "What is 2+2?".to_string(),
            0,
        );

        let (text, stderr, exit_code, _tps, skipped) = executor.subprocess_execution(&scenario);
        assert!(!skipped, "Chat scenario should not be skipped");
        assert_eq!(exit_code, 0);
        assert!(stderr.is_none() || stderr.as_deref() == Some(""));
        assert!(text.contains("4"), "Chat should return arithmetic answer");
    }

    #[test]
    fn test_subprocess_execution_serve_modality() {
        let runner = MockCommandRunner::new();
        let config = ExecutionConfig {
            model_path: Some("/mock/model.gguf".to_string()),
            ..Default::default()
        };
        let executor = Executor::with_runner(config, Arc::new(runner));

        let scenario = QaScenario::new(
            ModelId::new("test", "model"),
            Modality::Serve,
            Backend::Cpu,
            Format::Gguf,
            "What is 2+2?".to_string(),
            0,
        );

        let (_text, _stderr, _exit_code, _tps, skipped) = executor.subprocess_execution(&scenario);
        // Serve scenario should not be skipped (spawn_serve mock returns success)
        assert!(!skipped, "Serve scenario should not be skipped");
    }

    // ── Bug 201: Per-scenario backend ────────────────────────────────────

    #[test]
    fn test_subprocess_execution_gpu_backend() {
        // GPU scenario should NOT pass --no-gpu
        let runner = MockCommandRunner::new();
        let config = ExecutionConfig {
            model_path: Some("/mock/model.gguf".to_string()),
            no_gpu: true, // Global flag says no GPU — but scenario overrides
            ..Default::default()
        };
        let executor = Executor::with_runner(config, Arc::new(runner));

        let scenario = QaScenario::new(
            ModelId::new("test", "model"),
            Modality::Run,
            Backend::Gpu,
            Format::Gguf,
            "test".to_string(),
            0,
        );

        let (_text, _stderr, exit_code, _tps, skipped) = executor.subprocess_execution(&scenario);
        assert!(!skipped);
        assert_eq!(exit_code, 0);
        // The mock doesn't validate the no_gpu flag directly, but the code path
        // now uses scenario.backend instead of config.no_gpu
    }

    // =========================================================================
    // NEW: Coverage tests for extracted helper methods
    // =========================================================================

    // ── parse_timing_ms ─────────────────────────────────────────────────

    #[test]
    fn test_parse_timing_ms_valid() {
        let output = "Loading model...\nCompleted in 1.5s\nDone";
        let ms = parse_timing_ms(output);
        assert!(ms.is_some());
        assert!((ms.unwrap() - 1500.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_parse_timing_ms_integer_seconds() {
        let output = "Completed in 3s";
        let ms = parse_timing_ms(output);
        assert!(ms.is_some());
        assert!((ms.unwrap() - 3000.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_parse_timing_ms_no_match() {
        let output = "Some random output without timing";
        assert!(parse_timing_ms(output).is_none());
    }

    #[test]
    fn test_parse_timing_ms_empty() {
        assert!(parse_timing_ms("").is_none());
    }

    #[test]
    fn test_parse_timing_ms_case_insensitive() {
        let output = "COMPLETED IN 2.0s";
        let ms = parse_timing_ms(output);
        assert!(ms.is_some());
        assert!((ms.unwrap() - 2000.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_parse_timing_ms_invalid_number() {
        let output = "Completed in abcs";
        assert!(parse_timing_ms(output).is_none());
    }

    // ── parse_throughput ────────────────────────────────────────────────

    #[test]
    fn test_parse_throughput_valid_decimal() {
        let output = r#"{"throughput_tps":25.5,"other":1}"#;
        let tps = parse_throughput(output);
        assert!(tps.is_some());
        assert!((tps.unwrap() - 25.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_parse_throughput_at_end_of_json() {
        let output = r#"{"throughput_tps":100}"#;
        // The parse_throughput function looks for a non-digit/non-dot terminator
        // but at end of string this may not find one
        let tps = parse_throughput(output);
        // "100}" - the "}" terminates it
        assert!(tps.is_some());
        assert!((tps.unwrap() - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_parse_throughput_no_tps_field() {
        let output = r#"{"latency_ms":42}"#;
        assert!(parse_throughput(output).is_none());
    }

    #[test]
    fn test_parse_throughput_empty_string() {
        assert!(parse_throughput("").is_none());
    }

    // ── classify_integrity_gate ─────────────────────────────────────────

    #[test]
    fn test_classify_integrity_gate_layers() {
        let gate = Executor::classify_integrity_gate("LAYERS mismatch: expected 24, got 14");
        assert_eq!(gate, integrity::gate_ids::LAYERS);
    }

    #[test]
    fn test_classify_integrity_gate_hidden() {
        let gate =
            Executor::classify_integrity_gate("HIDDEN size mismatch: expected 896, got 4096");
        assert_eq!(gate, integrity::gate_ids::HIDDEN);
    }

    #[test]
    fn test_classify_integrity_gate_vocab() {
        let gate = Executor::classify_integrity_gate("VOCAB size wrong: 896 vs 151936");
        assert_eq!(gate, integrity::gate_ids::VOCAB);
    }

    #[test]
    fn test_classify_integrity_gate_config_default() {
        let gate = Executor::classify_integrity_gate("Some unknown error");
        assert_eq!(gate, integrity::gate_ids::CONFIG);
    }

    // ── is_conversion_artifact ──────────────────────────────────────────

    #[test]
    fn test_is_conversion_artifact_converted() {
        assert!(Executor::is_conversion_artifact("model-converted.gguf"));
    }

    #[test]
    fn test_is_conversion_artifact_byte_rt() {
        assert!(Executor::is_conversion_artifact("model.byte_rt.apr"));
    }

    #[test]
    fn test_is_conversion_artifact_idem() {
        assert!(Executor::is_conversion_artifact("model.idem.safetensors"));
    }

    #[test]
    fn test_is_conversion_artifact_com() {
        assert!(Executor::is_conversion_artifact("model.com_q4k.gguf"));
    }

    #[test]
    fn test_is_conversion_artifact_clean_file() {
        assert!(!Executor::is_conversion_artifact("model.safetensors"));
        assert!(!Executor::is_conversion_artifact("model.gguf"));
        assert!(!Executor::is_conversion_artifact("model.apr"));
        assert!(!Executor::is_conversion_artifact("config.json"));
    }

    // ── truncate_str ────────────────────────────────────────────────────

    #[test]
    fn test_truncate_str_short() {
        assert_eq!(Executor::truncate_str("hello", 10), "hello");
    }

    #[test]
    fn test_truncate_str_exact() {
        assert_eq!(Executor::truncate_str("hello", 5), "hello");
    }

    #[test]
    fn test_truncate_str_truncates() {
        assert_eq!(Executor::truncate_str("hello world", 5), "hello");
    }

    #[test]
    fn test_truncate_str_empty() {
        assert_eq!(Executor::truncate_str("", 5), "");
    }
