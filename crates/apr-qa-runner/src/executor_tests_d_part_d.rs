#[test]
fn test_integrity_check_disabled_by_default() {
    // With check_integrity=false (default), integrity checks are skipped
    let config = ExecutionConfig {
        run_conversion_tests: false,
        run_golden_rule_test: false,
        ..Default::default()
    };

    assert!(!config.check_integrity);
    assert!(config.lock_file_path.is_none());

    let mock_runner = MockCommandRunner::new();
    let mut executor = Executor::with_runner(config, Arc::new(mock_runner));
    let yaml = r#"
name: no-integrity
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
    let result = executor.execute(&playbook).expect("execute");

    // Should succeed without integrity check
    assert!(result.gateway_failed.is_none());
}

#[test]
fn test_integrity_check_missing_lock_file_warns() {
    // When lock file path is set but file doesn't exist, should warn (not error)
    let mock_runner = MockCommandRunner::new();
    let config = ExecutionConfig {
        check_integrity: true,
        lock_file_path: Some("/nonexistent/playbook.lock.yaml".to_string()),
        run_conversion_tests: false,
        run_golden_rule_test: false,
        ..Default::default()
    };

    let mut executor = Executor::with_runner(config, Arc::new(mock_runner));
    let yaml = r#"
name: missing-lock
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
    let result = executor.execute(&playbook).expect("execute");

    // Should proceed (not fail) when lock file is missing — just warn
    assert!(result.gateway_failed.is_none());
}

#[test]
fn test_warn_implicit_skips_flag() {
    // warn_implicit_skips should not crash even when no skip files exist
    let mock_runner = MockCommandRunner::new();
    let config = ExecutionConfig {
        warn_implicit_skips: true,
        run_conversion_tests: false,
        run_golden_rule_test: false,
        ..Default::default()
    };

    let mut executor = Executor::with_runner(config, Arc::new(mock_runner));
    let yaml = r#"
name: skip-warn-test
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
    let result = executor.execute(&playbook).expect("execute");

    // Should succeed — implicit skip warnings are informational only
    assert!(result.gateway_failed.is_none());
}

#[test]
fn test_backward_compat_new_flags_off() {
    // Ensure old configs (without new fields) still work via Default
    let config = ExecutionConfig::default();
    assert!(!config.check_integrity);
    assert!(!config.warn_implicit_skips);
    assert!(config.lock_file_path.is_none());
}

// ============================================================
// HF Parity Tests
// ============================================================

#[test]
fn test_hf_parity_disabled_by_default() {
    // HF parity should be disabled by default
    let config = ExecutionConfig::default();
    assert!(!config.run_hf_parity);
    assert!(config.hf_parity_corpus_path.is_none());
    assert!(config.hf_parity_model_family.is_none());
}

#[test]
fn test_hf_parity_skipped_when_missing_config() {
    // When HF parity is enabled but config is incomplete, should skip gracefully
    let mock_runner = MockCommandRunner::new();
    let config = ExecutionConfig {
        run_hf_parity: true,
        hf_parity_corpus_path: None,  // Missing!
        hf_parity_model_family: None, // Missing!
        run_conversion_tests: false,
        run_golden_rule_test: false,
        ..Default::default()
    };

    let mut executor = Executor::with_runner(config, Arc::new(mock_runner));
    let yaml = r#"
name: hf-parity-test
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
    let result = executor.execute(&playbook).expect("execute");

    // Should succeed — missing config is handled gracefully
    assert!(result.gateway_failed.is_none());

    // Evidence should contain skip reason
    let has_skip_evidence = result
        .evidence
        .all()
        .iter()
        .any(|e| e.gate_id == "F-HF-PARITY-SKIP");
    assert!(has_skip_evidence, "Expected F-HF-PARITY-SKIP evidence");
}

#[test]
fn test_hf_parity_skipped_when_manifest_missing() {
    // When HF parity config points to non-existent corpus
    let mock_runner = MockCommandRunner::new();
    let config = ExecutionConfig {
        run_hf_parity: true,
        hf_parity_corpus_path: Some("/nonexistent/corpus".to_string()),
        hf_parity_model_family: Some("nonexistent-model/v1".to_string()),
        run_conversion_tests: false,
        run_golden_rule_test: false,
        ..Default::default()
    };

    let mut executor = Executor::with_runner(config, Arc::new(mock_runner));
    let yaml = r#"
name: hf-parity-missing-test
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
    let result = executor.execute(&playbook).expect("execute");

    // The executor should still succeed, but have failures (1 from parity, plus scenario failures)
    assert!(
        result.failed >= 1,
        "Expected at least 1 failed test for missing manifest"
    );

    // Evidence should contain the manifest not found error
    let has_parity_evidence = result
        .evidence
        .all()
        .iter()
        .any(|e| e.gate_id == "F-HF-PARITY-001");
    assert!(
        has_parity_evidence,
        "Expected F-HF-PARITY-001 evidence for missing manifest"
    );
}

// ============================================================
// G0-FORMAT Workspace Tests
// ============================================================

#[test]
fn test_workspace_creates_directory_structure() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let output_dir = dir.path().join("output");

    // Create a fake safetensors file
    let model_file = dir.path().join("abc123.safetensors");
    std::fs::write(&model_file, b"fake-safetensors-content").expect("write model");

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
    let formats = vec![Format::SafeTensors, Format::Apr];

    let (workspace, passed, _failed) =
        executor.prepare_model_workspace(&model_file, &model_id, &formats);

    // Verify workspace directory was created
    let ws_path = Path::new(&workspace);
    assert!(ws_path.exists(), "Workspace directory should exist");

    // Verify safetensors subdir exists with symlinked model
    let st_dir = ws_path.join("safetensors");
    assert!(st_dir.exists(), "safetensors subdir should exist");
    let st_link = st_dir.join("model.safetensors");
    assert!(st_link.exists(), "model.safetensors symlink should exist");

    // Verify APR subdir was created with converted model
    let apr_dir = ws_path.join("apr");
    assert!(apr_dir.exists(), "apr subdir should exist");

    // MockCommandRunner.convert_model returns success, so conversion passed
    assert!(passed >= 1, "At least one format conversion should pass");
}

#[test]
fn test_workspace_symlinks_config_files() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let output_dir = dir.path().join("output");

    // Create model file and sibling config files (pacha cache naming)
    let model_file = dir.path().join("abc123.safetensors");
    std::fs::write(&model_file, b"fake-model").expect("write model");
    std::fs::write(
        dir.path().join("abc123.config.json"),
        r#"{"num_hidden_layers": 24}"#,
    )
    .expect("write config");
    std::fs::write(
        dir.path().join("abc123.tokenizer.json"),
        r#"{"version": "1.0"}"#,
    )
    .expect("write tokenizer");

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
    let formats = vec![Format::SafeTensors];

    let (workspace, _passed, _failed) =
        executor.prepare_model_workspace(&model_file, &model_id, &formats);

    let ws_path = Path::new(&workspace);
    let st_dir = ws_path.join("safetensors");

    // Verify config files were symlinked with canonical names
    assert!(
        st_dir.join("config.json").exists(),
        "config.json should be symlinked"
    );
    assert!(
        st_dir.join("tokenizer.json").exists(),
        "tokenizer.json should be symlinked"
    );
}

#[test]
fn test_workspace_conversion_failure_nonfatal() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let output_dir = dir.path().join("output");

    let model_file = dir.path().join("test.safetensors");
    std::fs::write(&model_file, b"fake-model").expect("write model");

    // Use a mock runner where conversion fails
    let mock_runner = MockCommandRunner::new().with_convert_failure();
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

    let (workspace, passed, failed) =
        executor.prepare_model_workspace(&model_file, &model_id, &formats);

    // Workspace should still be created
    assert!(
        Path::new(&workspace).exists(),
        "Workspace should exist even with conversion failures"
    );
    // SafeTensors subdir should exist
    assert!(
        Path::new(&workspace).join("safetensors").exists(),
        "safetensors dir should exist"
    );

    // Conversions should have failed (APR + GGUF = 2 failures)
    assert_eq!(passed, 0, "No conversions should pass");
    assert_eq!(failed, 2, "Both APR and GGUF conversions should fail");

    // Verify evidence was collected for failures
    let evidence = executor.evidence().all();
    let apr_evidence = evidence.iter().any(|e| e.gate_id == "G0-FORMAT-APR-001");
    let gguf_evidence = evidence.iter().any(|e| e.gate_id == "G0-FORMAT-GGUF-001");
    assert!(apr_evidence, "Should have G0-FORMAT-APR-001 evidence");
    assert!(gguf_evidence, "Should have G0-FORMAT-GGUF-001 evidence");
}
