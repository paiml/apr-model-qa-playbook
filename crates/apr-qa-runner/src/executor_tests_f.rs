
use super::*;
use crate::command::MockCommandRunner;
use apr_qa_gen::{Backend, Format, Modality, ModelId, QaScenario};

fn test_scenario() -> QaScenario {
    QaScenario::new(
        ModelId::new("test", "model"),
        Modality::Run,
        Backend::Cpu,
        Format::Gguf,
        "2+2=".to_string(),
        42,
    )
}

fn test_playbook() -> Playbook {
    let yaml = r#"
name: test-playbook
version: "1.0.0"
model:
  hf_repo: "test/model"
  formats: [gguf]
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 5
"#;
    Playbook::from_yaml(yaml).expect("Failed to parse")
}

#[test]
fn test_truncate_str_utf8_boundary() {
    // Multi-byte UTF-8: each char is 2+ bytes
    let s = "\u{00e9}\u{00e9}\u{00e9}"; // 3 x 'e-acute' (2 bytes each = 6 bytes)
    let result = Executor::truncate_str(s, 3);
    // Should not split in the middle of a char boundary
    assert_eq!(result.len(), 2); // Only the first complete char fits within 3 bytes
}

// ── strip_ansi ──────────────────────────────────────────────────────

#[test]
fn test_strip_ansi_no_escapes() {
    assert_eq!(Executor::strip_ansi("hello"), "hello");
}

#[test]
fn test_strip_ansi_with_color() {
    let colored = "\x1b[32m/path/to/model\x1b[0m";
    assert_eq!(Executor::strip_ansi(colored), "/path/to/model");
}

#[test]
fn test_strip_ansi_empty() {
    assert_eq!(Executor::strip_ansi(""), "");
}

#[test]
fn test_strip_ansi_multiple_sequences() {
    let s = "\x1b[1m\x1b[34mBold Blue\x1b[0m Normal";
    assert_eq!(Executor::strip_ansi(s), "Bold Blue Normal");
}

// ── extract_model_family_prefix ─────────────────────────────────────

#[test]
fn test_extract_model_family_prefix_with_quant() {
    let prefix = Executor::extract_model_family_prefix("qwen2.5-coder-7b-instruct-q4k");
    assert_eq!(prefix, "qwen2.5-coder-7b");
}

#[test]
fn test_extract_model_family_prefix_no_quant() {
    let prefix = Executor::extract_model_family_prefix("qwen2.5-coder-1.5b");
    assert_eq!(prefix, "qwen2.5-coder-1.5b");
}

#[test]
fn test_extract_model_family_prefix_instruct_suffix() {
    let prefix = Executor::extract_model_family_prefix("qwen2.5-coder-7b-instruct");
    assert_eq!(prefix, "qwen2.5-coder-7b");
}

#[test]
fn test_extract_model_family_prefix_q4_k_m() {
    let prefix = Executor::extract_model_family_prefix("TinyLlama-1.1B-Chat-v1.0-Q4_K_M");
    assert_eq!(prefix, "TinyLlama-1.1B-Chat-v1.0");
}

#[test]
fn test_extract_model_family_prefix_f16() {
    let prefix = Executor::extract_model_family_prefix("model-name-f16");
    assert_eq!(prefix, "model-name");
}

// ── should_stop_on_failure ───────────────────────────────────────────

#[test]
fn test_should_stop_on_failure_stop_on_first() {
    let config = ExecutionConfig {
        failure_policy: FailurePolicy::StopOnFirst,
        ..Default::default()
    };
    let executor = Executor::with_config(config);
    let evidence = Evidence::falsified("G2-BASIC", test_scenario(), "test failure", "output", 0);
    assert!(executor.should_stop_on_failure(&evidence, "test"));
}

#[test]
fn test_should_stop_on_failure_collect_all() {
    let config = ExecutionConfig {
        failure_policy: FailurePolicy::CollectAll,
        ..Default::default()
    };
    let executor = Executor::with_config(config);
    let evidence = Evidence::falsified("G2-BASIC", test_scenario(), "test failure", "output", 0);
    assert!(!executor.should_stop_on_failure(&evidence, "test"));
}

#[test]
fn test_should_stop_on_failure_stop_on_p0_with_p0_gate() {
    let config = ExecutionConfig {
        failure_policy: FailurePolicy::StopOnP0,
        ..Default::default()
    };
    let executor = Executor::with_config(config);
    let evidence = Evidence::falsified("F-P0-CRITICAL", test_scenario(), "p0 failure", "output", 0);
    assert!(executor.should_stop_on_failure(&evidence, "test"));
}

#[test]
fn test_should_stop_on_failure_stop_on_p0_without_p0_gate() {
    let config = ExecutionConfig {
        failure_policy: FailurePolicy::StopOnP0,
        ..Default::default()
    };
    let executor = Executor::with_config(config);
    let evidence = Evidence::falsified("G2-BASIC", test_scenario(), "non-p0 failure", "output", 0);
    assert!(!executor.should_stop_on_failure(&evidence, "test"));
}

#[test]
fn test_should_stop_on_failure_fail_fast() {
    let config = ExecutionConfig {
        failure_policy: FailurePolicy::FailFast,
        ..Default::default()
    };
    let executor = Executor::with_config(config);
    let evidence = Evidence::falsified("G2-BASIC", test_scenario(), "test failure", "output", 0);
    assert!(executor.should_stop_on_failure(&evidence, "test"));
}

// ── execute_scenarios ───────────────────────────────────────────────

#[test]
fn test_execute_scenarios_dry_run() {
    let mock_runner = MockCommandRunner::new();
    let config = ExecutionConfig {
        dry_run: true,
        model_path: Some("/test/model.gguf".to_string()),
        ..Default::default()
    };
    let mut executor = Executor::with_runner(config, Arc::new(mock_runner));

    let scenarios = vec![test_scenario(), test_scenario(), test_scenario()];
    let (passed, failed, skipped) = executor.execute_scenarios(scenarios, "test");

    assert_eq!(passed, 0);
    assert_eq!(failed, 0);
    assert_eq!(skipped, 3);
}

#[test]
fn test_execute_scenarios_all_pass() {
    let mock_runner = MockCommandRunner::new().with_inference_response("The answer is 4.");
    let config = ExecutionConfig {
        model_path: Some("/test/model.gguf".to_string()),
        failure_policy: FailurePolicy::CollectAll,
        ..Default::default()
    };
    let mut executor = Executor::with_runner(config, Arc::new(mock_runner));

    let scenarios = vec![test_scenario(), test_scenario()];
    let (passed, failed, skipped) = executor.execute_scenarios(scenarios, "test");

    assert_eq!(passed, 2);
    assert_eq!(failed, 0);
    assert_eq!(skipped, 0);
}

#[test]
fn test_execute_scenarios_stop_on_first_failure() {
    let mock_runner = MockCommandRunner::new().with_inference_failure();
    let config = ExecutionConfig {
        model_path: Some("/test/model.gguf".to_string()),
        failure_policy: FailurePolicy::StopOnFirst,
        ..Default::default()
    };
    let mut executor = Executor::with_runner(config, Arc::new(mock_runner));

    let scenarios = vec![test_scenario(), test_scenario(), test_scenario()];
    let (passed, failed, skipped) = executor.execute_scenarios(scenarios, "test");

    // Should stop after first failure
    assert_eq!(passed, 0);
    assert_eq!(failed, 1);
    assert_eq!(skipped, 0);
}

// ── check_pre_flight ────────────────────────────────────────────────

#[test]
fn test_check_pre_flight_no_integrity_check() {
    let config = ExecutionConfig {
        check_integrity: false,
        warn_implicit_skips: false,
        ..Default::default()
    };
    let executor = Executor::with_config(config);
    let playbook = test_playbook();
    let start = Instant::now();
    let result = executor.check_pre_flight(&playbook, 5, start);
    // No integrity check, no implicit skips, gateways pass => None
    assert!(result.is_none());
}

#[test]
fn test_check_pre_flight_integrity_no_lock_file() {
    let config = ExecutionConfig {
        check_integrity: true,
        lock_file_path: None, // No lock file path
        ..Default::default()
    };
    let executor = Executor::with_config(config);
    let playbook = test_playbook();
    let start = Instant::now();
    let result = executor.check_pre_flight(&playbook, 5, start);
    // check_integrity=true but no lock_file_path => skip integrity, proceed
    assert!(result.is_none());
}

#[test]
fn test_check_pre_flight_integrity_missing_lock_file() {
    let config = ExecutionConfig {
        check_integrity: true,
        lock_file_path: Some("/nonexistent/lock.json".to_string()),
        ..Default::default()
    };
    let executor = Executor::with_config(config);
    let playbook = test_playbook();
    let start = Instant::now();
    let result = executor.check_pre_flight(&playbook, 5, start);
    // Lock file doesn't exist => warning printed but None (continues)
    assert!(result.is_none());
}

// ── run_g0_format_check ─────────────────────────────────────────────

#[test]
fn test_run_g0_format_check_no_model_path() {
    let config = ExecutionConfig {
        model_path: None,
        ..Default::default()
    };
    let mut executor = Executor::with_config(config);
    let playbook = test_playbook();
    let (passed, failed) = executor.run_g0_format_check(&playbook);
    assert_eq!(passed, 0);
    assert_eq!(failed, 0);
}

#[test]
fn test_run_g0_format_check_nonexistent_path() {
    let config = ExecutionConfig {
        model_path: Some("/nonexistent/path/model.gguf".to_string()),
        ..Default::default()
    };
    let mut executor = Executor::with_config(config);
    let playbook = test_playbook();
    let (passed, failed) = executor.run_g0_format_check(&playbook);
    // Path doesn't exist, has gguf extension but doesn't match safetensors checks
    assert_eq!(passed, 0);
    assert_eq!(failed, 0);
}

// ── check_g0_tensor ─────────────────────────────────────────────────

#[test]
fn test_check_g0_tensor_no_model_path() {
    let config = ExecutionConfig {
        model_path: None,
        ..Default::default()
    };
    let mut executor = Executor::with_config(config);
    let playbook = test_playbook();
    let (passed, failed) = executor.check_g0_tensor(&playbook);
    assert_eq!(passed, 0);
    assert_eq!(failed, 0);
}

#[test]
fn test_check_g0_tensor_no_family() {
    let config = ExecutionConfig {
        model_path: Some("/test/model.gguf".to_string()),
        ..Default::default()
    };
    let mut executor = Executor::with_config(config);
    // Default test_playbook has no family set
    let playbook = test_playbook();
    let (passed, failed) = executor.check_g0_tensor(&playbook);
    assert_eq!(passed, 0);
    assert_eq!(failed, 0);
}

#[test]
fn test_check_g0_tensor_no_size_variant() {
    let config = ExecutionConfig {
        model_path: Some("/test/model.gguf".to_string()),
        ..Default::default()
    };
    let mut executor = Executor::with_config(config);
    let yaml = r#"
name: test-with-family
version: "1.0.0"
model:
  hf_repo: "test/model"
  family: "qwen2"
  formats: [gguf]
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
"#;
    let playbook = Playbook::from_yaml(yaml).expect("Failed to parse");
    let (passed, failed) = executor.check_g0_tensor(&playbook);
    // No size_variant => (0, 0)
    assert_eq!(passed, 0);
    assert_eq!(failed, 0);
}

// ── run_extended_tests ──────────────────────────────────────────────

#[test]
fn test_run_extended_tests_all_disabled() {
    let mock_runner = MockCommandRunner::new();
    let config = ExecutionConfig {
        model_path: Some("/test/model.gguf".to_string()),
        run_conversion_tests: false,
        run_golden_rule_test: false,
        run_contract_tests: false,
        run_hf_parity: false,
        run_profile_ci: false,
        run_ollama_parity: false,
        ..Default::default()
    };
    let mut executor = Executor::with_runner(config, Arc::new(mock_runner));
    let playbook = test_playbook();
    let (passed, failed) = executor.run_extended_tests(&playbook);
    assert_eq!(passed, 0);
    assert_eq!(failed, 0);
}

#[test]
fn test_run_extended_tests_no_model_path() {
    let mock_runner = MockCommandRunner::new();
    let config = ExecutionConfig {
        model_path: None,
        run_conversion_tests: true,
        run_golden_rule_test: true,
        run_contract_tests: true,
        run_profile_ci: true,
        run_ollama_parity: true,
        ..Default::default()
    };
    let mut executor = Executor::with_runner(config, Arc::new(mock_runner));
    let playbook = test_playbook();
    let (passed, failed) = executor.run_extended_tests(&playbook);
    // No model path => all model-dependent tests skip silently
    assert_eq!(passed, 0);
    assert_eq!(failed, 0);
}

// ── run_integrity_analysis ──────────────────────────────────────────

#[test]
fn test_run_integrity_analysis_nonexistent_path() {
    let result = Executor::run_integrity_analysis(Path::new("/nonexistent/path"));
    // Non-existent directory should return None (no safetensors dir found)
    assert!(result.is_none());
}

#[test]
fn test_run_integrity_analysis_empty_dir() {
    let tmp = tempfile::tempdir().unwrap();
    let result = Executor::run_integrity_analysis(tmp.path());
    // Empty dir => no safetensors files => None
    assert!(result.is_none());
}

// ── clean_stale_artifacts ───────────────────────────────────────────

#[test]
fn test_clean_stale_artifacts_nonexistent_workspace() {
    // Should not panic
    Executor::clean_stale_artifacts(Path::new("/nonexistent/workspace"));
}

#[test]
fn test_clean_stale_artifacts_empty_workspace() {
    let tmp = tempfile::tempdir().unwrap();
    let workspace = tmp.path();
    // Create empty subdirs
    std::fs::create_dir_all(workspace.join("safetensors")).unwrap();
    std::fs::create_dir_all(workspace.join("apr")).unwrap();
    std::fs::create_dir_all(workspace.join("gguf")).unwrap();
    // Should not panic on empty dirs
    Executor::clean_stale_artifacts(workspace);
}

#[test]
fn test_clean_stale_artifacts_removes_artifacts() {
    let tmp = tempfile::tempdir().unwrap();
    let workspace = tmp.path();
    let apr_dir = workspace.join("apr");
    std::fs::create_dir_all(&apr_dir).unwrap();

    // Create a clean file and an artifact
    std::fs::write(apr_dir.join("model.apr"), b"clean").unwrap();
    std::fs::write(apr_dir.join("model-converted.apr"), b"artifact").unwrap();
    std::fs::write(apr_dir.join("model.idem.apr"), b"artifact").unwrap();
    std::fs::write(apr_dir.join("model.byte_rt.apr"), b"artifact").unwrap();

    Executor::clean_stale_artifacts(workspace);

    // Clean file should remain
    assert!(apr_dir.join("model.apr").exists());
    // Artifacts should be removed
    assert!(!apr_dir.join("model-converted.apr").exists());
    assert!(!apr_dir.join("model.idem.apr").exists());
    assert!(!apr_dir.join("model.byte_rt.apr").exists());
}

// ── setup_source_links (sharded vs single) ─────────────────────────

#[test]
fn test_setup_source_links_single_file() {
    let tmp = tempfile::tempdir().unwrap();
    let source = tmp.path().join("model.safetensors");
    std::fs::write(&source, b"fake model data").unwrap();
    let st_dir = tmp.path().join("workspace_st");
    std::fs::create_dir_all(&st_dir).unwrap();

    let err = Executor::setup_source_links(&source, &st_dir, false);
    assert!(err.is_none(), "Expected no error, got: {err:?}");
    // The symlink should exist
    assert!(st_dir.join("model.safetensors").exists());
}

#[test]
fn test_setup_source_links_sharded() {
    let tmp = tempfile::tempdir().unwrap();
    let source_dir = tmp.path().join("source");
    std::fs::create_dir_all(&source_dir).unwrap();
    let index_file = source_dir.join("model.safetensors.index.json");
    std::fs::write(&index_file, b"{}").unwrap();
    std::fs::write(
        source_dir.join("model-00001-of-00002.safetensors"),
        b"shard1",
    )
    .unwrap();
    std::fs::write(
        source_dir.join("model-00002-of-00002.safetensors"),
        b"shard2",
    )
    .unwrap();

    let st_dir = tmp.path().join("workspace_st");
    std::fs::create_dir_all(&st_dir).unwrap();

    let err = Executor::setup_source_links(&index_file, &st_dir, true);
    assert!(err.is_none(), "Expected no error, got: {err:?}");
    // All files from source should be symlinked
    assert!(st_dir.join("model.safetensors.index.json").exists());
}

// ── resolve_sharded_index ───────────────────────────────────────────

#[test]
fn test_resolve_sharded_index_safetensors_format() {
    let scenario = QaScenario::new(
        ModelId::new("test", "model"),
        Modality::Run,
        Backend::Cpu,
        Format::SafeTensors,
        "test".to_string(),
        0,
    );
    let path = Path::new("/cache/model.safetensors.index.json");
    let result =
        Executor::resolve_sharded_index(path, "/cache/model.safetensors.index.json", &scenario);
    assert_eq!(
        result,
        Some("/cache/model.safetensors.index.json".to_string())
    );
}

#[test]
fn test_resolve_sharded_index_gguf_format() {
    let tmp = tempfile::tempdir().unwrap();
    let index = tmp.path().join("model.safetensors.index.json");
    std::fs::write(&index, b"{}").unwrap();
    // Put a gguf file in the same directory
    std::fs::write(tmp.path().join("model.gguf"), b"fake gguf").unwrap();

    let scenario = QaScenario::new(
        ModelId::new("test", "model"),
        Modality::Run,
        Backend::Cpu,
        Format::Gguf,
        "test".to_string(),
        0,
    );
    let result = Executor::resolve_sharded_index(&index, &index.to_string_lossy(), &scenario);
    // Should find sibling .gguf file
    assert!(result.is_some());
    assert!(result.unwrap().contains("gguf"));
}

// ── resolve_file_model ──────────────────────────────────────────────

#[test]
fn test_resolve_file_model_matching_extension() {
    let scenario = QaScenario::new(
        ModelId::new("test", "model"),
        Modality::Run,
        Backend::Cpu,
        Format::Gguf,
        "test".to_string(),
        0,
    );
    let result = Executor::resolve_file_model(
        Path::new("/path/model.gguf"),
        "/path/model.gguf",
        "gguf",
        &scenario,
    );
    assert_eq!(result, Some("/path/model.gguf".to_string()));
}

#[test]
fn test_resolve_file_model_safetensors_match() {
    let scenario = QaScenario::new(
        ModelId::new("test", "model"),
        Modality::Run,
        Backend::Cpu,
        Format::SafeTensors,
        "test".to_string(),
        0,
    );
    let result = Executor::resolve_file_model(
        Path::new("/path/model.safetensors"),
        "/path/model.safetensors",
        "safetensors",
        &scenario,
    );
    assert_eq!(result, Some("/path/model.safetensors".to_string()));
}

#[test]
fn test_resolve_file_model_apr_match() {
    let scenario = QaScenario::new(
        ModelId::new("test", "model"),
        Modality::Run,
        Backend::Cpu,
        Format::Apr,
        "test".to_string(),
        0,
    );
    let result = Executor::resolve_file_model(
        Path::new("/path/model.apr"),
        "/path/model.apr",
        "apr",
        &scenario,
    );
    assert_eq!(result, Some("/path/model.apr".to_string()));
}

#[test]
fn test_resolve_file_model_mismatch_with_sibling() {
    let tmp = tempfile::tempdir().unwrap();
    let gguf_file = tmp.path().join("model.gguf");
    let apr_file = tmp.path().join("model.apr");
    std::fs::write(&gguf_file, b"fake gguf").unwrap();
    std::fs::write(&apr_file, b"fake apr").unwrap();

    let scenario = QaScenario::new(
        ModelId::new("test", "model"),
        Modality::Run,
        Backend::Cpu,
        Format::Apr,
        "test".to_string(),
        0,
    );
    let result =
        Executor::resolve_file_model(&gguf_file, &gguf_file.to_string_lossy(), "gguf", &scenario);
    // Should find sibling .apr file
    assert!(result.is_some());
    assert!(result.unwrap().contains("apr"));
}

// ── resolve_directory_model ─────────────────────────────────────────

#[test]
fn test_resolve_directory_model_apr_cache_structure() {
    let tmp = tempfile::tempdir().unwrap();
    let gguf_dir = tmp.path().join("gguf");
    std::fs::create_dir_all(&gguf_dir).unwrap();
    std::fs::write(gguf_dir.join("model.gguf"), b"fake").unwrap();

    let scenario = QaScenario::new(
        ModelId::new("test", "model"),
        Modality::Run,
        Backend::Cpu,
        Format::Gguf,
        "test".to_string(),
        0,
    );
    let result = Executor::resolve_directory_model(tmp.path(), &scenario);
    assert!(result.is_some());
    assert!(result.unwrap().contains("gguf"));
}

#[test]
fn test_resolve_directory_model_flat_hf_structure() {
    let tmp = tempfile::tempdir().unwrap();
    std::fs::write(tmp.path().join("model.safetensors"), b"fake").unwrap();

    let scenario = QaScenario::new(
        ModelId::new("test", "model"),
        Modality::Run,
        Backend::Cpu,
        Format::SafeTensors,
        "test".to_string(),
        0,
    );
    let result = Executor::resolve_directory_model(tmp.path(), &scenario);
    assert!(result.is_some());
    assert!(result.unwrap().contains("model.safetensors"));
}

#[test]
fn test_resolve_directory_model_sharded_safetensors() {
    let tmp = tempfile::tempdir().unwrap();
    let st_dir = tmp.path().join("safetensors");
    std::fs::create_dir_all(&st_dir).unwrap();
    std::fs::write(st_dir.join("model.safetensors.index.json"), b"{}").unwrap();

    let scenario = QaScenario::new(
        ModelId::new("test", "model"),
        Modality::Run,
        Backend::Cpu,
        Format::SafeTensors,
        "test".to_string(),
        0,
    );
    let result = Executor::resolve_directory_model(tmp.path(), &scenario);
    assert!(result.is_some());
    assert!(result.unwrap().contains("model.safetensors.index.json"));
}

#[test]
fn test_resolve_directory_model_nothing_found() {
    let tmp = tempfile::tempdir().unwrap();
    // Empty directory - nothing to find
    let scenario = QaScenario::new(
        ModelId::new("test", "model"),
        Modality::Run,
        Backend::Cpu,
        Format::Gguf,
        "test".to_string(),
        0,
    );
    let result = Executor::resolve_directory_model(tmp.path(), &scenario);
    assert!(result.is_none());
}

// ── find_clean_model_file ───────────────────────────────────────────

#[test]
fn test_find_clean_model_file_skips_artifacts() {
    let tmp = tempfile::tempdir().unwrap();
    std::fs::write(tmp.path().join("model-converted.gguf"), b"artifact").unwrap();
    std::fs::write(tmp.path().join("model.idem.gguf"), b"artifact").unwrap();
    std::fs::write(tmp.path().join("model.com_q4k.gguf"), b"artifact").unwrap();
    std::fs::write(tmp.path().join("model.rt_q6k.gguf"), b"artifact").unwrap();
    std::fs::write(tmp.path().join("model.gguf"), b"clean").unwrap();

    let result = Executor::find_clean_model_file(tmp.path(), "gguf");
    assert!(result.is_some());
    let found = result.unwrap();
    assert!(found.contains("model.gguf"));
    assert!(!found.contains("converted"));
    assert!(!found.contains("idem"));
    assert!(!found.contains("com_"));
    assert!(!found.contains("rt_"));
}

#[test]
fn test_find_clean_model_file_no_files() {
    let tmp = tempfile::tempdir().unwrap();
    let result = Executor::find_clean_model_file(tmp.path(), "gguf");
    assert!(result.is_none());
}

#[test]
fn test_find_clean_model_file_nonexistent_dir() {
    let result = Executor::find_clean_model_file(Path::new("/nonexistent"), "gguf");
    assert!(result.is_none());
}

// ── find_sibling_model_files ────────────────────────────────────────

#[test]
fn test_find_sibling_model_files_pacha_cache() {
    let tmp = tempfile::tempdir().unwrap();
    let model_file = tmp.path().join("abc123.safetensors");
    std::fs::write(&model_file, b"model data").unwrap();
    std::fs::write(tmp.path().join("abc123.config.json"), b"config").unwrap();
    std::fs::write(tmp.path().join("abc123.tokenizer.json"), b"tokenizer").unwrap();
    std::fs::write(tmp.path().join("other_file.txt"), b"unrelated").unwrap();

    let siblings = Executor::find_sibling_model_files(&model_file);
    assert_eq!(siblings.len(), 2);
    let canonical_names: Vec<&str> = siblings.iter().map(|(_, n)| n.as_str()).collect();
    assert!(canonical_names.contains(&"config.json"));
    assert!(canonical_names.contains(&"tokenizer.json"));
}

#[test]
fn test_find_sibling_model_files_flat_hf_dir() {
    let tmp = tempfile::tempdir().unwrap();
    let model_file = tmp.path().join("model.safetensors");
    std::fs::write(&model_file, b"model data").unwrap();
    std::fs::write(tmp.path().join("config.json"), b"config").unwrap();
    std::fs::write(tmp.path().join("tokenizer.json"), b"tokenizer").unwrap();
    std::fs::write(tmp.path().join("random.txt"), b"unrelated").unwrap();

    let siblings = Executor::find_sibling_model_files(&model_file);
    assert!(siblings.len() >= 2);
    let canonical_names: Vec<&str> = siblings.iter().map(|(_, n)| n.as_str()).collect();
    assert!(canonical_names.contains(&"config.json"));
    assert!(canonical_names.contains(&"tokenizer.json"));
}

// ── print_fail_fast_diagnostics ─────────────────────────────────────

#[test]
fn test_print_fail_fast_diagnostics_no_model_path() {
    let config = ExecutionConfig {
        model_path: None,
        ..Default::default()
    };
    let executor = Executor::with_config(config);
    let evidence = Evidence::falsified("G2-BASIC", test_scenario(), "test failure", "output", 0);
    // Should not panic, just prints to stderr
    executor.print_fail_fast_diagnostics(&evidence, "test-playbook");
}

#[test]
fn test_print_fail_fast_diagnostics_with_stderr_and_exit_code() {
    let config = ExecutionConfig {
        model_path: None,
        ..Default::default()
    };
    let executor = Executor::with_config(config);
    let mut evidence =
        Evidence::falsified("G2-BASIC", test_scenario(), "test failure", "output", 0);
    evidence.stderr = Some("Error message in stderr".to_string());
    evidence.exit_code = Some(42);
    // Should print stderr and exit code
    executor.print_fail_fast_diagnostics(&evidence, "test-playbook");
}

// ── format_tensor_failure ───────────────────────────────────────────

#[test]
fn test_format_tensor_failure_without_expected_actual() {
    let result = crate::layout_contract::TensorValidationResult {
        tensor_name: "lm_head.weight".to_string(),
        rule_id: "R001".to_string(),
        passed: false,
        details: "tensor not found".to_string(),
        expected: None,
        actual: None,
    };
    let formatted = Executor::format_tensor_failure(&result);
    assert!(formatted.contains("R001"));
    assert!(formatted.contains("tensor not found"));
    assert!(!formatted.contains("Expected:"));
}

// ── scenario creation helpers (unique tests) ────────────────────────

#[test]
fn test_format_scenario_creation_apr() {
    let model_id = ModelId::new("org", "name");
    let scenario = Executor::format_scenario(&model_id, Format::Apr);
    assert_eq!(scenario.format, Format::Apr);
    assert!(scenario.prompt.contains("Format"));
}

#[test]
fn test_format_scenario_creation_gguf() {
    let model_id = ModelId::new("org", "name");
    let scenario = Executor::format_scenario(&model_id, Format::Gguf);
    assert_eq!(scenario.format, Format::Gguf);
}

#[test]
fn test_hf_parity_scenario_truncates_long_prompt() {
    let model_id = ModelId::new("org", "name");
    let long_prompt = "A".repeat(100);
    let scenario = Executor::hf_parity_scenario(&model_id, &long_prompt);
    assert_eq!(scenario.format, Format::Apr);
    // Prompt should be truncated to 40 chars
    assert!(scenario.prompt.len() < 100);
}

// ── has_safetensors_files (unique) ──────────────────────────────────

#[test]
fn test_has_safetensors_files_with_st_among_others() {
    let tmp = tempfile::tempdir().unwrap();
    std::fs::write(tmp.path().join("model.safetensors"), b"data").unwrap();
    std::fs::write(tmp.path().join("config.json"), b"{}").unwrap();
    std::fs::write(tmp.path().join("model.gguf"), b"gguf").unwrap();
    assert!(Executor::has_safetensors_files(tmp.path()));
}

// ── find_safetensors_dir (unique) ───────────────────────────────────

#[test]
fn test_find_safetensors_dir_prefers_subdir() {
    let tmp = tempfile::tempdir().unwrap();
    // Create both a subdir with safetensors and direct safetensors
    let st_dir = tmp.path().join("safetensors");
    std::fs::create_dir_all(&st_dir).unwrap();
    std::fs::write(st_dir.join("model.safetensors"), b"subdir").unwrap();
    std::fs::write(tmp.path().join("model.safetensors"), b"direct").unwrap();
    let result = Executor::find_safetensors_dir(tmp.path());
    assert!(result.is_some());
    // Should prefer the subdir
    assert!(result.unwrap().to_string_lossy().contains("safetensors"));
}

// ── find_model_by_prefix (unique) ───────────────────────────────────

#[test]
fn test_find_model_by_prefix_case_insensitive() {
    let tmp = tempfile::tempdir().unwrap();
    std::fs::write(tmp.path().join("Qwen2.5-Coder-7b-q4k.gguf"), b"data").unwrap();
    let result = Executor::find_model_by_prefix(tmp.path(), "qwen2.5-coder-7b", "gguf");
    assert!(result.is_some());
}

#[test]
fn test_find_model_by_prefix_nonexistent_dir() {
    let result = Executor::find_model_by_prefix(Path::new("/nonexistent/dir"), "model", "gguf");
    assert!(result.is_none());
}

// ── find_clean_model_file (unique) ──────────────────────────────────

#[test]
fn test_find_clean_model_file_wrong_extension() {
    let tmp = tempfile::tempdir().unwrap();
    std::fs::write(tmp.path().join("model.safetensors"), b"data").unwrap();
    let result = Executor::find_clean_model_file(tmp.path(), "gguf");
    assert!(result.is_none());
}

// ── metadata_only mode ─────────────────────────────────────────────

#[test]
fn test_metadata_only_skips_inference() {
    let tmp = tempfile::tempdir().unwrap();
    // Write config.json with matching dimensions
    let config = serde_json::json!({
        "hidden_size": 896,
        "num_hidden_layers": 24,
        "num_attention_heads": 14,
        "num_key_value_heads": 2,
        "vocab_size": 151_936
    });
    let config_path = tmp.path().join("config.json");
    std::fs::write(&config_path, serde_json::to_string(&config).unwrap()).unwrap();

    // Write minimal safetensors with correct shapes
    {
        use std::collections::HashMap;
        use std::io::Write;
        let mut header_map: HashMap<&str, serde_json::Value> = HashMap::new();
        header_map.insert(
            "model.embed_tokens.weight",
            serde_json::json!({"dtype": "F32", "shape": [151_936, 896], "data_offsets": [0, 8]}),
        );
        header_map.insert(
            "lm_head.weight",
            serde_json::json!({"dtype": "F32", "shape": [151_936, 896], "data_offsets": [8, 16]}),
        );
        let header_json = serde_json::to_string(&header_map).unwrap();
        let header_bytes = header_json.as_bytes();
        let header_len = header_bytes.len() as u64;

        let st_path = tmp.path().join("model.safetensors");
        let mut f = std::fs::File::create(st_path).unwrap();
        f.write_all(&header_len.to_le_bytes()).unwrap();
        f.write_all(header_bytes).unwrap();
        f.write_all(&[0u8; 16]).unwrap();
    }

    let playbook_yaml = r#"
name: test-dim-smoke
version: "1.0"
model:
  hf_repo: "test/model"
  expected_hidden_dim: 896
  expected_num_layers: 24
  expected_num_heads: 14
  expected_num_kv_heads: 2
  expected_vocab_size: 151936
test_matrix:
  modalities: [run]
  backends: [cpu]
  formats: [safetensors]
  prompts:
    - "hello"
"#;
    let playbook = crate::playbook::Playbook::from_yaml(playbook_yaml).expect("valid playbook");

    let exec_config = ExecutionConfig {
        metadata_only: true,
        model_path: Some(tmp.path().to_string_lossy().to_string()),
        ..Default::default()
    };
    let mut executor = Executor::with_config(exec_config);
    let result = executor.execute(&playbook).expect("should succeed");

    // All dimensional checks should pass
    assert_eq!(
        result.failed, 0,
        "no checks should fail: gateway={:?}",
        result.gateway_failed
    );
    assert!(result.passed > 0, "should have passing checks");
    assert!(result.gateway_failed.is_none());
}

#[test]
fn test_metadata_only_detects_mismatch() {
    let tmp = tempfile::tempdir().unwrap();
    // Write config.json with WRONG hidden_size
    let model_config = serde_json::json!({
        "hidden_size": 512,
        "num_hidden_layers": 24
    });
    std::fs::write(
        tmp.path().join("config.json"),
        serde_json::to_string(&model_config).unwrap(),
    )
    .unwrap();

    let playbook_yaml = r#"
name: test-dim-smoke-fail
version: "1.0"
model:
  hf_repo: "test/model"
  expected_hidden_dim: 896
  expected_num_layers: 24
test_matrix:
  modalities: [run]
  backends: [cpu]
  formats: [safetensors]
  prompts:
    - "hello"
"#;
    let playbook = crate::playbook::Playbook::from_yaml(playbook_yaml).expect("valid playbook");

    let exec_config = ExecutionConfig {
        metadata_only: true,
        model_path: Some(tmp.path().to_string_lossy().to_string()),
        ..Default::default()
    };
    let mut executor = Executor::with_config(exec_config);
    let result = executor.execute(&playbook).expect("should succeed");

    // hidden_size mismatch + no safetensors => failures
    assert!(result.failed > 0, "should have failures");
    assert!(result.gateway_failed.is_some());
}
