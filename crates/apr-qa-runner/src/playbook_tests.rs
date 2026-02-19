
use super::*;

#[test]
fn test_default_formats() {
    let formats = default_formats();
    assert_eq!(formats.len(), 3);
    assert!(formats.contains(&Format::Gguf));
    assert!(formats.contains(&Format::SafeTensors));
    assert!(formats.contains(&Format::Apr));
}

#[test]
fn test_default_quantizations() {
    let quants = default_quantizations();
    assert_eq!(quants, vec!["q4_k_m"]);
}

#[test]
fn test_default_modalities() {
    let modalities = default_modalities();
    assert_eq!(modalities.len(), 3);
    assert!(modalities.contains(&Modality::Run));
    assert!(modalities.contains(&Modality::Chat));
    assert!(modalities.contains(&Modality::Serve));
}

#[test]
fn test_default_backends() {
    let backends = default_backends();
    assert_eq!(backends.len(), 2);
    assert!(backends.contains(&Backend::Cpu));
    assert!(backends.contains(&Backend::Gpu));
}

#[test]
fn test_default_scenario_count() {
    assert_eq!(default_scenario_count(), 100);
}

#[test]
fn test_default_proptest_count() {
    assert_eq!(default_proptest_count(), 100);
}

#[test]
fn test_default_timeout() {
    assert_eq!(default_timeout(), 60000);
}

#[test]
fn test_default_severity() {
    assert_eq!(default_severity(), "P1");
}

#[test]
fn test_test_matrix_default() {
    let matrix = TestMatrix::default();
    assert_eq!(matrix.modalities.len(), 3);
    assert_eq!(matrix.backends.len(), 2);
    assert_eq!(matrix.scenario_count, 100);
}

#[test]
fn test_playbook_to_yaml() {
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
    let playbook = Playbook::from_yaml(yaml).expect("Failed to parse");
    let output = playbook.to_yaml().expect("Failed to serialize");
    assert!(output.contains("test-playbook"));
    assert!(output.contains("test/model"));
}

#[test]
fn test_playbook_with_defaults() {
    // Test playbook that uses default values for model config
    let yaml = r#"
name: minimal
version: "1.0.0"
model:
  hf_repo: "org/model"
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 100
"#;
    let playbook = Playbook::from_yaml(yaml).expect("Failed to parse");
    assert_eq!(playbook.model.formats.len(), 3);
    assert_eq!(playbook.model.quantizations, vec!["q4_k_m"]);
    assert_eq!(playbook.test_matrix.scenario_count, 100);
}

#[test]
fn test_playbook_with_state_machine() {
    let yaml = r#"
name: state-test
version: "1.0.0"
model:
  hf_repo: "test/model"
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
state_machine:
  initial: "ready"
  states:
    ready:
      on_enter:
        - action: "log 'entering ready'"
      transitions:
        - event: "start"
          target: "running"
          action: "initialize"
          guards:
            - "model_loaded"
    running:
      on_exit:
        - action: "cleanup"
"#;
    let playbook = Playbook::from_yaml(yaml).expect("Failed to parse");
    let state_machine = playbook.state_machine.expect("Should have state machine");
    assert_eq!(state_machine.initial, "ready");
    assert_eq!(state_machine.states.len(), 2);

    let ready_state = &state_machine.states["ready"];
    assert_eq!(ready_state.on_enter.len(), 1);
    assert_eq!(ready_state.transitions.len(), 1);

    let transition = &ready_state.transitions[0];
    assert_eq!(transition.event, "start");
    assert_eq!(transition.target, "running");
    assert!(transition.action.is_some());
    assert_eq!(transition.guards.len(), 1);
}

#[test]
fn test_playbook_with_property_tests() {
    let yaml = r#"
name: prop-test
version: "1.0.0"
model:
  hf_repo: "test/model"
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
property_tests:
  - name: "arithmetic"
    generator: "random_arithmetic"
    oracle: "check_arithmetic"
    count: 50
  - name: "code"
    generator: "random_code"
    oracle: "check_code"
"#;
    let playbook = Playbook::from_yaml(yaml).expect("Failed to parse");
    assert_eq!(playbook.property_tests.len(), 2);

    let first = &playbook.property_tests[0];
    assert_eq!(first.name, "arithmetic");
    assert_eq!(first.count, 50);

    let second = &playbook.property_tests[1];
    assert_eq!(second.name, "code");
    assert_eq!(second.count, 100); // default
}

#[test]
fn test_playbook_with_falsification_gates() {
    let yaml = r#"
name: gate-test
version: "1.0.0"
model:
  hf_repo: "test/model"
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
falsification_gates:
  - id: F-QUAL-001
    description: "Output is valid"
    condition: "output.len() > 0"
    severity: P0
  - id: F-QUAL-002
    description: "No errors"
    condition: "!output.contains('error')"
"#;
    let playbook = Playbook::from_yaml(yaml).expect("Failed to parse");
    assert_eq!(playbook.falsification_gates.len(), 2);

    let first = &playbook.falsification_gates[0];
    assert_eq!(first.severity, "P0");

    let second = &playbook.falsification_gates[1];
    assert_eq!(second.severity, "P1"); // default
}

#[test]
fn test_model_config_no_slash() {
    let config = ModelConfig {
        hf_repo: "model-name".to_string(),
        local_path: None,
        formats: vec![Format::Gguf],
        quantizations: vec![],
        size_category: SizeCategory::default(),
        expected_hidden_dim: None,
        expected_num_layers: None,
        expected_num_heads: None,
        expected_num_kv_heads: None,
        expected_vocab_size: None,
        expected_intermediate_dim: None,
        family: None,
        size_variant: None,
    };
    assert_eq!(config.hf_org(), "model-name");
    assert_eq!(config.hf_name(), "model-name");
}

#[test]
fn test_model_config_with_local_path() {
    let config = ModelConfig {
        hf_repo: "org/model".to_string(),
        local_path: Some("/path/to/model".to_string()),
        formats: default_formats(),
        quantizations: default_quantizations(),
        size_category: SizeCategory::default(),
        expected_hidden_dim: None,
        expected_num_layers: None,
        expected_num_heads: None,
        expected_num_kv_heads: None,
        expected_vocab_size: None,
        expected_intermediate_dim: None,
        family: None,
        size_variant: None,
    };
    assert!(config.local_path.is_some());
}

#[test]
fn test_playbook_step() {
    let step = PlaybookStep {
        name: "test-step".to_string(),
        command: "echo test".to_string(),
        timeout_ms: default_timeout(),
        expected_exit_code: 0,
        expected_patterns: vec!["test".to_string()],
        forbidden_patterns: vec!["error".to_string()],
    };
    assert_eq!(step.timeout_ms, 60000);
    assert_eq!(step.expected_exit_code, 0);
}

#[test]
fn test_playbook_parse() {
    let yaml = r#"
name: test-playbook
version: "1.0.0"
model:
  hf_repo: "Qwen/Qwen2.5-Coder-1.5B-Instruct"
  formats: [gguf, safetensors]
test_matrix:
  modalities: [run, chat]
  backends: [cpu]
  scenario_count: 10
falsification_gates:
  - id: F-TEST-001
    description: "Output is non-empty"
    condition: "output.len() > 0"
"#;

    let playbook = Playbook::from_yaml(yaml).expect("Failed to parse playbook");
    assert_eq!(playbook.name, "test-playbook");
    assert_eq!(playbook.model.hf_repo, "Qwen/Qwen2.5-Coder-1.5B-Instruct");
    assert_eq!(playbook.test_matrix.modalities.len(), 2);
    assert_eq!(playbook.falsification_gates.len(), 1);
}

#[test]
fn test_playbook_generate_scenarios() {
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

    let playbook = Playbook::from_yaml(yaml).expect("Failed to parse");
    let scenarios = playbook.generate_scenarios();

    // 1 modality × 1 backend × 1 format × 5 scenarios = 5
    assert_eq!(scenarios.len(), 5);
}

#[test]
fn test_model_config_parse() {
    let config = ModelConfig {
        hf_repo: "Qwen/Qwen2.5-Coder-1.5B-Instruct".to_string(),
        local_path: None,
        formats: vec![Format::Gguf],
        quantizations: vec!["q4_k_m".to_string()],
        size_category: SizeCategory::Small,
        expected_hidden_dim: None,
        expected_num_layers: None,
        expected_num_heads: None,
        expected_num_kv_heads: None,
        expected_vocab_size: None,
        expected_intermediate_dim: None,
        family: None,
        size_variant: None,
    };

    assert_eq!(config.hf_org(), "Qwen");
    assert_eq!(config.hf_name(), "Qwen2.5-Coder-1.5B-Instruct");
}

#[test]
fn test_total_tests() {
    let yaml = r#"
name: test
version: "1.0.0"
model:
  hf_repo: "test/model"
  formats: [gguf, safetensors, apr]
test_matrix:
  modalities: [run, chat, serve]
  backends: [cpu, gpu]
  scenario_count: 100
"#;

    let playbook = Playbook::from_yaml(yaml).expect("Failed to parse");
    // 3 modalities × 2 backends × 3 formats × 100 = 1800
    assert_eq!(playbook.total_tests(), 1800);
}

#[test]
fn test_playbook_with_differential_tests() {
    let yaml = r#"
name: diff-test
version: "1.0.0"
model:
  hf_repo: "test/model"
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
differential_tests:
  tensor_diff:
    enabled: true
    filter: "embed,lm_head"
    gates: ["F-ROSETTA-DIFF-001", "F-ROSETTA-DIFF-002"]
  inference_compare:
    enabled: true
    prompt: "What is 2+2?"
    max_tokens: 10
    tolerance: 0.00001
    gates: ["F-ROSETTA-INF-001"]
"#;
    let playbook = Playbook::from_yaml(yaml).expect("Failed to parse");
    let diff = playbook
        .differential_tests
        .expect("Should have differential tests");

    let tensor = diff.tensor_diff.expect("Should have tensor diff");
    assert!(tensor.enabled);
    assert_eq!(tensor.filter, Some("embed,lm_head".to_string()));
    assert_eq!(tensor.gates.len(), 2);

    let inf = diff
        .inference_compare
        .expect("Should have inference compare");
    assert!(inf.enabled);
    assert_eq!(inf.prompt, Some("What is 2+2?".to_string()));
    assert_eq!(inf.max_tokens, 10);
}

#[test]
fn test_playbook_with_profile_ci() {
    let yaml = r#"
name: profile-test
version: "1.0.0"
model:
  hf_repo: "test/model"
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
profile_ci:
  enabled: true
  warmup: 5
  measure: 20
  assertions:
    min_throughput: 10.0
    max_p99_ms: 500.0
    max_p50_ms: 200.0
  gates: ["F-PROFILE-CI-001", "F-PROFILE-CI-002"]
"#;
    let playbook = Playbook::from_yaml(yaml).expect("Failed to parse");
    let profile = playbook.profile_ci.expect("Should have profile CI");

    assert!(profile.enabled);
    assert_eq!(profile.warmup, 5);
    assert_eq!(profile.measure, 20);
    assert_eq!(profile.assertions.min_throughput, Some(10.0));
    assert_eq!(profile.assertions.max_p99_ms, Some(500.0));
    assert_eq!(profile.assertions.max_p50_ms, Some(200.0));
    assert_eq!(profile.gates.len(), 2);
}

#[test]
fn test_playbook_with_trace_payload() {
    let yaml = r#"
name: trace-test
version: "1.0.0"
model:
  hf_repo: "test/model"
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
trace_payload:
  enabled: true
  prompt: "Test prompt"
  gates: ["F-TRACE-PAYLOAD-001", "F-TRACE-PAYLOAD-002"]
"#;
    let playbook = Playbook::from_yaml(yaml).expect("Failed to parse");
    let trace = playbook.trace_payload.expect("Should have trace payload");

    assert!(trace.enabled);
    assert_eq!(trace.prompt, Some("Test prompt".to_string()));
    assert_eq!(trace.gates.len(), 2);
}

#[test]
fn test_default_max_tokens() {
    assert_eq!(default_max_tokens(), 10);
}

#[test]
fn test_default_tolerance() {
    assert!((default_tolerance() - 1e-5).abs() < 1e-10);
}

#[test]
fn test_default_warmup() {
    assert_eq!(default_warmup(), 3);
}

#[test]
fn test_default_measure() {
    assert_eq!(default_measure(), 10);
}

#[test]
fn test_playbook_with_fingerprint() {
    let yaml = r#"
name: fingerprint-test
version: "1.0.0"
model:
  hf_repo: "test/model"
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
differential_tests:
  fingerprint:
    enabled: true
    tensors: "embed,lm_head"
    stats: ["mean", "std", "checksum"]
    gates: ["F-ROSETTA-FP-001", "F-ROSETTA-FP-002"]
"#;
    let playbook = Playbook::from_yaml(yaml).expect("Failed to parse");
    let diff = playbook
        .differential_tests
        .expect("Should have differential tests");

    let fp = diff.fingerprint.expect("Should have fingerprint");
    assert!(fp.enabled);
    assert_eq!(fp.tensors, "embed,lm_head");
    assert_eq!(fp.stats.len(), 3);
    assert_eq!(fp.gates.len(), 2);
}

#[test]
fn test_playbook_with_validate_stats() {
    let yaml = r#"
name: validate-stats-test
version: "1.0.0"
model:
  hf_repo: "test/model"
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
differential_tests:
  validate_stats:
    enabled: true
    reference: "reference.json"
    tolerance:
      layernorm: 0.001
      embedding: 0.1
      attention: 0.01
    gates: ["F-ROSETTA-STATS-001", "F-ROSETTA-STATS-002"]
"#;
    let playbook = Playbook::from_yaml(yaml).expect("Failed to parse");
    let diff = playbook
        .differential_tests
        .expect("Should have differential tests");

    let stats = diff.validate_stats.expect("Should have validate_stats");
    assert!(stats.enabled);
    assert_eq!(stats.reference, Some("reference.json".to_string()));
    assert!((stats.tolerance.layernorm - 0.001).abs() < 1e-10);
    assert!((stats.tolerance.embedding - 0.1).abs() < 1e-10);
    assert!((stats.tolerance.attention - 0.01).abs() < 1e-10);
    assert_eq!(stats.gates.len(), 2);
}

#[test]
fn test_default_fingerprint_tensors() {
    assert_eq!(default_fingerprint_tensors(), "all");
}

#[test]
fn test_default_fingerprint_stats() {
    let stats = default_fingerprint_stats();
    assert_eq!(stats.len(), 5);
    assert!(stats.contains(&"mean".to_string()));
    assert!(stats.contains(&"checksum".to_string()));
}

#[test]
fn test_default_tolerance_values() {
    assert!((default_layernorm_tolerance() - 0.001).abs() < 1e-10);
    assert!((default_embedding_tolerance() - 0.1).abs() < 1e-10);
    assert!((default_attention_tolerance() - 0.01).abs() < 1e-10);
}

#[test]
fn test_profile_ci_min_throughput_for() {
    // Test with all fields set
    let assertions = ProfileCiAssertions {
        min_throughput: Some(10.0),
        min_throughput_cpu: Some(5.0),
        min_throughput_gpu: Some(50.0),
        max_p99_ms: None,
        max_p50_ms: None,
    };

    assert_eq!(assertions.min_throughput_for("cpu"), Some(5.0));
    assert_eq!(assertions.min_throughput_for("gpu"), Some(50.0));
    assert_eq!(assertions.min_throughput_for("tpu"), Some(10.0));

    // Test with only min_throughput set (fallback)
    let assertions_fallback = ProfileCiAssertions {
        min_throughput: Some(20.0),
        min_throughput_cpu: None,
        min_throughput_gpu: None,
        max_p99_ms: None,
        max_p50_ms: None,
    };

    assert_eq!(assertions_fallback.min_throughput_for("cpu"), Some(20.0));
    assert_eq!(assertions_fallback.min_throughput_for("gpu"), Some(20.0));

    // Test with nothing set
    let assertions_none = ProfileCiAssertions {
        min_throughput: None,
        min_throughput_cpu: None,
        min_throughput_gpu: None,
        max_p99_ms: None,
        max_p50_ms: None,
    };

    assert_eq!(assertions_none.min_throughput_for("cpu"), None);
    assert_eq!(assertions_none.min_throughput_for("gpu"), None);
}

// ── §3.1 Playbook integrity lock tests ─────────────────────────────

#[test]
fn test_compute_playbook_hash_consistent() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let path = dir.path().join("test.playbook.yaml");
    std::fs::write(&path, "name: test\nversion: 1.0").expect("write");

    let hash1 = compute_playbook_hash(&path).expect("hash1");
    let hash2 = compute_playbook_hash(&path).expect("hash2");
    assert_eq!(hash1, hash2);
    assert_eq!(hash1.len(), 64); // SHA-256 hex
}

#[test]
fn test_compute_playbook_hash_differs() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let path1 = dir.path().join("a.yaml");
    let path2 = dir.path().join("b.yaml");
    std::fs::write(&path1, "content-a").expect("write");
    std::fs::write(&path2, "content-b").expect("write");

    let hash1 = compute_playbook_hash(&path1).expect("hash1");
    let hash2 = compute_playbook_hash(&path2).expect("hash2");
    assert_ne!(hash1, hash2);
}

#[test]
fn test_verify_playbook_integrity_pass() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let path = dir.path().join("test.playbook.yaml");
    std::fs::write(&path, "name: test\nversion: 1.0").expect("write");

    let hash = compute_playbook_hash(&path).expect("hash");
    let mut lock = PlaybookLockFile::default();
    lock.entries.insert(
        "test".to_string(),
        PlaybookLockEntry {
            sha256: hash,
            locked_fields: vec!["model.hf_repo".to_string()],
        },
    );

    assert!(verify_playbook_integrity(&path, &lock, "test").is_ok());
}

#[test]
fn test_verify_playbook_integrity_fail_mismatch() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let path = dir.path().join("test.playbook.yaml");
    std::fs::write(&path, "name: test\nversion: 1.0").expect("write");

    let mut lock = PlaybookLockFile::default();
    lock.entries.insert(
        "test".to_string(),
        PlaybookLockEntry {
            sha256: "wrong_hash".to_string(),
            locked_fields: vec![],
        },
    );

    let result = verify_playbook_integrity(&path, &lock, "test");
    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("Integrity check failed")
    );
}

#[test]
fn test_verify_playbook_integrity_missing_entry() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let path = dir.path().join("test.playbook.yaml");
    std::fs::write(&path, "name: test").expect("write");

    let lock = PlaybookLockFile::default();
    let result = verify_playbook_integrity(&path, &lock, "test");
    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("not found in lock file")
    );
}

#[test]
fn test_generate_lock_entry() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let path = dir.path().join("my-model.playbook.yaml");
    std::fs::write(&path, "name: my-model\nversion: 1.0").expect("write");

    let (name, entry) = generate_lock_entry(&path).expect("generate");
    assert_eq!(name, "my-model");
    assert_eq!(entry.sha256.len(), 64);
    assert!(!entry.locked_fields.is_empty());
}

#[test]
fn test_lock_file_save_load_roundtrip() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let lock_path = dir.path().join("playbook.lock.yaml");

    let mut lock = PlaybookLockFile::default();
    lock.entries.insert(
        "model-a".to_string(),
        PlaybookLockEntry {
            sha256: "abc123".to_string(),
            locked_fields: vec!["model.hf_repo".to_string()],
        },
    );

    save_lock_file(&lock, &lock_path).expect("save");
    let loaded = load_lock_file(&lock_path).expect("load");

    assert_eq!(loaded.entries.len(), 1);
    assert_eq!(loaded.entries["model-a"].sha256, "abc123");
}

#[test]
fn test_lock_file_serde_roundtrip() {
    let mut lock = PlaybookLockFile::default();
    lock.entries.insert(
        "test".to_string(),
        PlaybookLockEntry {
            sha256: "deadbeef".to_string(),
            locked_fields: vec!["a".to_string(), "b".to_string()],
        },
    );

    let yaml = serde_yaml::to_string(&lock).expect("serialize");
    let parsed: PlaybookLockFile = serde_yaml::from_str(&yaml).expect("deserialize");
    assert_eq!(parsed.entries["test"].sha256, "deadbeef");
    assert_eq!(parsed.entries["test"].locked_fields.len(), 2);
}

// ── §3.3 Skip mechanism tests ──────────────────────────────────────

#[test]
fn test_find_skip_files_empty_dir() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let skips = find_skip_files(dir.path(), "test-model");
    assert!(skips.is_empty());
}

#[test]
fn test_find_skip_files_with_skip() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let skip_path = dir.path().join("test-model.skip.yaml");
    std::fs::write(
        &skip_path,
        r#"- format_or_backend: gpu
  reason: "No GPU available"
  tracking_issue: "GH-123"
"#,
    )
    .expect("write");

    let skips = find_skip_files(dir.path(), "test-model");
    assert_eq!(skips.len(), 1);
    assert_eq!(skips[0].format_or_backend, "gpu");
    assert_eq!(skips[0].tracking_issue.as_deref(), Some("GH-123"));
}

#[test]
fn test_detect_implicit_skips() {
    let yaml = r#"
name: test
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
    let all = vec![Format::Gguf, Format::SafeTensors, Format::Apr];
    let skips: Vec<SkipReason> = vec![];
    let implicit = detect_implicit_skips(&playbook, &all, &skips);
    // safetensors and apr are missing from playbook formats
    assert_eq!(implicit.len(), 2);
    assert!(implicit.contains(&"safetensors".to_string()));
    assert!(implicit.contains(&"apr".to_string()));
}

#[test]
fn test_detect_implicit_skips_with_explicit() {
    let yaml = r#"
name: test
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
    let all = vec![Format::Gguf, Format::SafeTensors, Format::Apr];
    // safetensors is explicitly skipped
    let skips = vec![SkipReason {
        format_or_backend: "safetensors".to_string(),
        reason: "Not supported".to_string(),
        tracking_issue: None,
    }];
    let implicit = detect_implicit_skips(&playbook, &all, &skips);
    // Only apr is implicitly skipped
    assert_eq!(implicit.len(), 1);
    assert_eq!(implicit[0], "apr");
}

#[test]
fn test_detect_implicit_skips_all_covered() {
    let yaml = r#"
name: test
version: "1.0.0"
model:
  hf_repo: "test/model"
  formats: [gguf, safetensors, apr]
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
"#;
    let playbook = Playbook::from_yaml(yaml).expect("parse");
    let all = vec![Format::Gguf, Format::SafeTensors, Format::Apr];
    let skips: Vec<SkipReason> = vec![];
    let implicit = detect_implicit_skips(&playbook, &all, &skips);
    assert!(implicit.is_empty());
}

#[test]
fn test_skip_reason_serde() {
    let reason = SkipReason {
        format_or_backend: "gpu".to_string(),
        reason: "No GPU".to_string(),
        tracking_issue: Some("GH-100".to_string()),
    };
    let json = serde_json::to_string(&reason).expect("serialize");
    let parsed: SkipReason = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.format_or_backend, "gpu");
    assert_eq!(parsed.tracking_issue.as_deref(), Some("GH-100"));
}

#[test]
fn test_skip_type_eq() {
    assert_eq!(SkipType::Explicit, SkipType::Explicit);
    assert_ne!(SkipType::Explicit, SkipType::Implicit);
}

// ── §3.4 Resource-aware scheduling tests ────────────────────────────

#[test]
fn test_size_category_max_workers() {
    assert_eq!(SizeCategory::Tiny.max_workers(), 4);
    assert_eq!(SizeCategory::Small.max_workers(), 4);
    assert_eq!(SizeCategory::Medium.max_workers(), 2);
    assert_eq!(SizeCategory::Large.max_workers(), 1);
    assert_eq!(SizeCategory::Xlarge.max_workers(), 1);
    assert_eq!(SizeCategory::Huge.max_workers(), 1);
}

#[test]
fn test_size_category_estimated_memory() {
    assert_eq!(SizeCategory::Tiny.estimated_memory_gb(), 2);
    assert_eq!(SizeCategory::Small.estimated_memory_gb(), 4);
    assert_eq!(SizeCategory::Medium.estimated_memory_gb(), 8);
    assert_eq!(SizeCategory::Large.estimated_memory_gb(), 16);
    assert_eq!(SizeCategory::Xlarge.estimated_memory_gb(), 32);
    assert_eq!(SizeCategory::Huge.estimated_memory_gb(), 64);
}

#[test]
fn test_size_category_can_run_concurrent() {
    assert!(SizeCategory::Tiny.can_run_concurrent());
    assert!(SizeCategory::Small.can_run_concurrent());
    assert!(!SizeCategory::Medium.can_run_concurrent());
    assert!(!SizeCategory::Large.can_run_concurrent());
    assert!(!SizeCategory::Xlarge.can_run_concurrent());
    assert!(!SizeCategory::Huge.can_run_concurrent());
}

#[test]
fn test_size_category_default() {
    assert_eq!(SizeCategory::default(), SizeCategory::Tiny);
}

#[test]
fn test_size_category_serde() {
    let yaml = r#"
name: test
version: "1.0.0"
model:
  hf_repo: "test/model"
  size_category: large
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
"#;
    let playbook = Playbook::from_yaml(yaml).expect("parse");
    assert_eq!(playbook.model.size_category, SizeCategory::Large);
}

#[test]
fn test_effective_max_workers_respects_size() {
    let yaml = r#"
name: test
version: "1.0.0"
model:
  hf_repo: "test/model"
  size_category: large
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
"#;
    let playbook = Playbook::from_yaml(yaml).expect("parse");
    // Large model caps at 1 worker regardless of request
    assert_eq!(playbook.effective_max_workers(4), 1);
    assert_eq!(playbook.effective_max_workers(8), 1);
    assert_eq!(playbook.effective_max_workers(1), 1);
}

#[test]
fn test_effective_max_workers_small_model() {
    let yaml = r#"
name: test
version: "1.0.0"
model:
  hf_repo: "test/model"
  size_category: small
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
"#;
    let playbook = Playbook::from_yaml(yaml).expect("parse");
    // Small model allows up to 4 workers
    assert_eq!(playbook.effective_max_workers(4), 4);
    assert_eq!(playbook.effective_max_workers(8), 4); // capped at 4
    assert_eq!(playbook.effective_max_workers(2), 2); // respects lower request
}

#[test]
fn test_effective_max_workers_medium_model() {
    let yaml = r#"
name: test
version: "1.0.0"
model:
  hf_repo: "test/model"
  size_category: medium
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
"#;
    let playbook = Playbook::from_yaml(yaml).expect("parse");
    // Medium model caps at 2 workers
    assert_eq!(playbook.effective_max_workers(4), 2);
    assert_eq!(playbook.effective_max_workers(1), 1);
}

// ── PMAT-266 Naming convention tests ─────────────────────────────────

#[test]
fn test_validate_playbook_name_basic() {
    let result = validate_playbook_name("qwen2.5-coder-0.5b-mvp.playbook.yaml");
    assert!(result.is_ok());
    let parts = result.unwrap();
    assert_eq!(parts.family, "qwen2.5-coder");
    assert_eq!(parts.size, "0.5b");
    assert_eq!(parts.tier, Some("mvp".to_string()));
}

#[test]
fn test_validate_playbook_name_no_tier() {
    let result = validate_playbook_name("llama3.2-7b.playbook.yaml");
    assert!(result.is_ok());
    let parts = result.unwrap();
    assert_eq!(parts.family, "llama3.2");
    assert_eq!(parts.size, "7b");
    assert_eq!(parts.tier, None);
}

#[test]
fn test_validate_playbook_name_large_model() {
    let result = validate_playbook_name("deepseek-coder-v2-16b-full.playbook.yaml");
    assert!(result.is_ok());
    let parts = result.unwrap();
    assert_eq!(parts.family, "deepseek-coder-v2");
    assert_eq!(parts.size, "16b");
    assert_eq!(parts.tier, Some("full".to_string()));
}

#[test]
fn test_validate_playbook_name_various_tiers() {
    for tier in VALID_TIERS {
        let filename = format!("model-1b-{tier}.playbook.yaml");
        let result = validate_playbook_name(&filename);
        assert!(result.is_ok(), "Failed for tier: {tier}");
        assert_eq!(result.unwrap().tier, Some((*tier).to_string()));
    }
}

#[test]
fn test_validate_playbook_name_various_sizes() {
    let sizes = ["0.5b", "1b", "1.5b", "3b", "7b", "13b", "70b", "405b"];
    for size in sizes {
        let filename = format!("model-{size}.playbook.yaml");
        let result = validate_playbook_name(&filename);
        assert!(result.is_ok(), "Failed for size: {size}");
        assert_eq!(result.unwrap().size, size);
    }
}

#[test]
fn test_validate_playbook_name_invalid_no_size() {
    let result = validate_playbook_name("qwen2.5-coder-mvp.playbook.yaml");
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("does not match naming convention"));
}

#[test]
fn test_validate_playbook_name_invalid_wrong_extension() {
    let result = validate_playbook_name("qwen2.5-coder-0.5b-mvp.yaml");
    assert!(result.is_err());
}

#[test]
fn test_validate_playbook_name_invalid_tier() {
    let result = validate_playbook_name("qwen2.5-coder-0.5b-unknown.playbook.yaml");
    assert!(result.is_err());
}

#[test]
fn test_validate_playbook_name_invalid_format() {
    let invalid_names = [
        "model.playbook.yaml",         // no size
        "model-big.playbook.yaml",     // invalid size format
        "model-7gb.playbook.yaml",     // wrong unit (gb instead of b)
        ".playbook.yaml",              // empty name
        "model-7b-test.playbook.yaml", // invalid tier
    ];
    for name in invalid_names {
        let result = validate_playbook_name(name);
        assert!(result.is_err(), "Expected error for: {name}");
    }
}

#[test]
fn test_validate_playbook_path() {
    let path = std::path::Path::new("/some/path/qwen2.5-coder-1.5b-mvp.playbook.yaml");
    let result = validate_playbook_path(path);
    assert!(result.is_ok());
    let parts = result.unwrap();
    assert_eq!(parts.family, "qwen2.5-coder");
    assert_eq!(parts.size, "1.5b");
    assert_eq!(parts.tier, Some("mvp".to_string()));
}

#[test]
fn test_playbook_name_parts_to_filename() {
    let parts = PlaybookNameParts {
        family: "qwen2.5-coder".to_string(),
        size: "0.5b".to_string(),
        tier: Some("mvp".to_string()),
    };
    assert_eq!(parts.to_filename(), "qwen2.5-coder-0.5b-mvp.playbook.yaml");

    let parts_no_tier = PlaybookNameParts {
        family: "llama3.2".to_string(),
        size: "7b".to_string(),
        tier: None,
    };
    assert_eq!(parts_no_tier.to_filename(), "llama3.2-7b.playbook.yaml");
}

#[test]
fn test_playbook_name_parts_eq() {
    let parts1 = PlaybookNameParts {
        family: "model".to_string(),
        size: "1b".to_string(),
        tier: Some("mvp".to_string()),
    };
    let parts2 = PlaybookNameParts {
        family: "model".to_string(),
        size: "1b".to_string(),
        tier: Some("mvp".to_string()),
    };
    assert_eq!(parts1, parts2);
}

#[test]
fn test_valid_tiers_constant() {
    assert_eq!(VALID_TIERS.len(), 8);
    assert!(VALID_TIERS.contains(&"dim-smoke"));
    assert!(VALID_TIERS.contains(&"mvp"));
    assert!(VALID_TIERS.contains(&"smoke"));
    assert!(VALID_TIERS.contains(&"quick"));
    assert!(VALID_TIERS.contains(&"ci"));
    assert!(VALID_TIERS.contains(&"full"));
    assert!(VALID_TIERS.contains(&"nightly"));
    assert!(VALID_TIERS.contains(&"release"));
}

#[test]
fn test_validate_playbook_name_dim_smoke() {
    let result = validate_playbook_name("qwen2.5-coder-0.5b-dim-smoke.playbook.yaml");
    assert!(result.is_ok());
    let parts = result.unwrap();
    assert_eq!(parts.family, "qwen2.5-coder");
    assert_eq!(parts.size, "0.5b");
    assert_eq!(parts.tier, Some("dim-smoke".to_string()));
}

// ── PMAT-269 Test matrix generation tests ────────────────────────────

#[test]
fn test_populate_from_family_contract() {
    use crate::family_contract::FamilyContract;

    // PMAT-270: Include certification.size_categories for auto-alignment test
    let yaml = r#"
family: qwen2
size_variants:
  0.5b:
    parameters: "0.5B"
    hidden_dim: 896
    num_layers: 24
    num_heads: 14
    num_kv_heads: 2
    vocab_size: 151936
    intermediate_dim: 4864
certification:
  size_categories:
    0.5b: tiny
    1.5b: small
    7b: medium
"#;
    let contract = FamilyContract::from_yaml(yaml).expect("parse");

    let mut config = ModelConfig {
        hf_repo: "Qwen/Qwen2.5-Coder-0.5B-Instruct".to_string(),
        local_path: None,
        formats: vec![Format::Gguf],
        quantizations: vec![],
        size_category: SizeCategory::Tiny, // default
        expected_hidden_dim: None,
        expected_num_layers: None,
        expected_num_heads: None,
        expected_num_kv_heads: None,
        expected_vocab_size: None,
        expected_intermediate_dim: None,
        family: None,
        size_variant: None,
    };

    // Populate from contract
    let result = config.populate_from_family_contract(&contract, "0.5b");
    assert!(result);

    // Verify values populated
    assert_eq!(config.family, Some("qwen2".to_string()));
    assert_eq!(config.size_variant, Some("0.5b".to_string()));
    assert_eq!(config.expected_hidden_dim, Some(896));
    assert_eq!(config.expected_num_layers, Some(24));
    assert_eq!(config.expected_num_heads, Some(14));
    assert_eq!(config.expected_num_kv_heads, Some(2));
    assert_eq!(config.expected_vocab_size, Some(151_936));
    assert_eq!(config.expected_intermediate_dim, Some(4864));
    // PMAT-270: Verify size_category auto-populated
    assert_eq!(config.size_category, SizeCategory::Tiny);
}

#[test]
fn test_populate_from_family_contract_missing_size() {
    use crate::family_contract::FamilyContract;

    let yaml = r#"
family: qwen2
size_variants:
  0.5b:
    parameters: "0.5B"
    hidden_dim: 896
    num_layers: 24
    num_heads: 14
"#;
    let contract = FamilyContract::from_yaml(yaml).expect("parse");

    let mut config = ModelConfig {
        hf_repo: "test".to_string(),
        local_path: None,
        formats: vec![],
        quantizations: vec![],
        size_category: SizeCategory::default(),
        expected_hidden_dim: None,
        expected_num_layers: None,
        expected_num_heads: None,
        expected_num_kv_heads: None,
        expected_vocab_size: None,
        expected_intermediate_dim: None,
        family: None,
        size_variant: None,
    };

    // Try to populate with non-existent size
    let result = config.populate_from_family_contract(&contract, "7b");
    assert!(!result);

    // Values should remain None
    assert!(config.expected_hidden_dim.is_none());
}

#[test]
fn test_has_expected_params() {
    let config_empty = ModelConfig {
        hf_repo: "test".to_string(),
        local_path: None,
        formats: vec![],
        quantizations: vec![],
        size_category: SizeCategory::default(),
        expected_hidden_dim: None,
        expected_num_layers: None,
        expected_num_heads: None,
        expected_num_kv_heads: None,
        expected_vocab_size: None,
        expected_intermediate_dim: None,
        family: None,
        size_variant: None,
    };
    assert!(!config_empty.has_expected_params());

    let config_with_params = ModelConfig {
        hf_repo: "test".to_string(),
        local_path: None,
        formats: vec![],
        quantizations: vec![],
        size_category: SizeCategory::default(),
        expected_hidden_dim: Some(896),
        expected_num_layers: None,
        expected_num_heads: None,
        expected_num_kv_heads: None,
        expected_vocab_size: None,
        expected_intermediate_dim: None,
        family: None,
        size_variant: None,
    };
    assert!(config_with_params.has_expected_params());
}

#[test]
fn test_validate_architecture_match() {
    let config = ModelConfig {
        hf_repo: "test".to_string(),
        local_path: None,
        formats: vec![],
        quantizations: vec![],
        size_category: SizeCategory::default(),
        expected_hidden_dim: Some(896),
        expected_num_layers: Some(24),
        expected_num_heads: Some(14),
        expected_num_kv_heads: Some(2),
        expected_vocab_size: None,
        expected_intermediate_dim: None,
        family: None,
        size_variant: None,
    };

    // All match
    let mismatches = config.validate_architecture(896, 24, Some(14), Some(2));
    assert!(mismatches.is_empty());
}

#[test]
fn test_validate_architecture_mismatch() {
    let config = ModelConfig {
        hf_repo: "test".to_string(),
        local_path: None,
        formats: vec![],
        quantizations: vec![],
        size_category: SizeCategory::default(),
        expected_hidden_dim: Some(896),
        expected_num_layers: Some(24),
        expected_num_heads: Some(14),
        expected_num_kv_heads: Some(2),
        expected_vocab_size: None,
        expected_intermediate_dim: None,
        family: None,
        size_variant: None,
    };

    // All wrong
    let mismatches = config.validate_architecture(1024, 12, Some(16), Some(4));
    assert_eq!(mismatches.len(), 4);
    assert!(mismatches[0].contains("hidden_dim"));
    assert!(mismatches[1].contains("num_layers"));
    assert!(mismatches[2].contains("num_heads"));
    assert!(mismatches[3].contains("num_kv_heads"));
}

#[test]
fn test_validate_architecture_partial_expected() {
    let config = ModelConfig {
        hf_repo: "test".to_string(),
        local_path: None,
        formats: vec![],
        quantizations: vec![],
        size_category: SizeCategory::default(),
        expected_hidden_dim: Some(896),
        expected_num_layers: None, // Not set
        expected_num_heads: None,  // Not set
        expected_num_kv_heads: None,
        expected_vocab_size: None,
        expected_intermediate_dim: None,
        family: None,
        size_variant: None,
    };

    // Only hidden_dim is checked
    let mismatches = config.validate_architecture(896, 999, Some(999), Some(999));
    assert!(mismatches.is_empty()); // hidden_dim matches, others not checked
}

// ── PMAT-270: Size category auto-alignment tests ─────────────────────────

#[test]
fn test_size_category_auto_alignment_from_family_yaml() {
    use crate::family_contract::FamilyContract;

    // FALSIFY-FAM-001: Size category alignment
    let yaml = r#"
family: qwen2
size_variants:
  7b:
    parameters: "7B"
    hidden_dim: 3584
    num_layers: 28
    num_heads: 28
certification:
  size_categories:
    0.5b: tiny
    1.5b: small
    3b: small
    7b: medium
    14b: large
"#;
    let contract = FamilyContract::from_yaml(yaml).expect("parse");

    // Start with default (Tiny)
    let mut config = ModelConfig {
        hf_repo: "Qwen/Qwen2.5-Coder-7B-Instruct".to_string(),
        local_path: None,
        formats: vec![],
        quantizations: vec![],
        size_category: SizeCategory::Tiny, // default
        expected_hidden_dim: None,
        expected_num_layers: None,
        expected_num_heads: None,
        expected_num_kv_heads: None,
        expected_vocab_size: None,
        expected_intermediate_dim: None,
        family: None,
        size_variant: None,
    };

    // Populate from contract with 7b size
    let result = config.populate_from_family_contract(&contract, "7b");
    assert!(result);

    // PMAT-270: Verify size_category auto-set to Medium (from 7b -> medium mapping)
    assert_eq!(config.size_category, SizeCategory::Medium);
}

#[test]
fn test_size_category_explicit_not_overridden() {
    use crate::family_contract::FamilyContract;

    let yaml = r#"
family: qwen2
size_variants:
  7b:
    parameters: "7B"
    hidden_dim: 3584
    num_layers: 28
    num_heads: 28
certification:
  size_categories:
    7b: medium
"#;
    let contract = FamilyContract::from_yaml(yaml).expect("parse");

    // Explicitly set to Large (user override)
    let mut config = ModelConfig {
        hf_repo: "Qwen/Qwen2.5-Coder-7B-Instruct".to_string(),
        local_path: None,
        formats: vec![],
        quantizations: vec![],
        size_category: SizeCategory::Large, // explicitly set, not default
        expected_hidden_dim: None,
        expected_num_layers: None,
        expected_num_heads: None,
        expected_num_kv_heads: None,
        expected_vocab_size: None,
        expected_intermediate_dim: None,
        family: None,
        size_variant: None,
    };

    // Populate from contract
    config.populate_from_family_contract(&contract, "7b");

    // Should NOT override explicit setting - Large remains Large
    assert_eq!(config.size_category, SizeCategory::Large);
}

#[test]
fn test_size_category_from_str_lowercase() {
    assert_eq!(
        SizeCategory::from_str_lowercase("tiny").unwrap(),
        SizeCategory::Tiny
    );
    assert_eq!(
        SizeCategory::from_str_lowercase("small").unwrap(),
        SizeCategory::Small
    );
    assert_eq!(
        SizeCategory::from_str_lowercase("medium").unwrap(),
        SizeCategory::Medium
    );
    assert_eq!(
        SizeCategory::from_str_lowercase("large").unwrap(),
        SizeCategory::Large
    );
    assert_eq!(
        SizeCategory::from_str_lowercase("xlarge").unwrap(),
        SizeCategory::Xlarge
    );
    assert_eq!(
        SizeCategory::from_str_lowercase("huge").unwrap(),
        SizeCategory::Huge
    );

    // Case insensitive
    assert_eq!(
        SizeCategory::from_str_lowercase("TINY").unwrap(),
        SizeCategory::Tiny
    );
    assert_eq!(
        SizeCategory::from_str_lowercase("Medium").unwrap(),
        SizeCategory::Medium
    );

    // Invalid
    let err = SizeCategory::from_str_lowercase("invalid").unwrap_err();
    assert!(err.to_string().contains("Invalid size category"));
}

#[test]
fn test_size_category_no_certification_config() {
    use crate::family_contract::FamilyContract;

    // No certification section at all
    let yaml = r#"
family: qwen2
size_variants:
  0.5b:
    parameters: "0.5B"
    hidden_dim: 896
    num_layers: 24
    num_heads: 14
"#;
    let contract = FamilyContract::from_yaml(yaml).expect("parse");

    let mut config = ModelConfig {
        hf_repo: "test".to_string(),
        local_path: None,
        formats: vec![],
        quantizations: vec![],
        size_category: SizeCategory::Tiny, // default
        expected_hidden_dim: None,
        expected_num_layers: None,
        expected_num_heads: None,
        expected_num_kv_heads: None,
        expected_vocab_size: None,
        expected_intermediate_dim: None,
        family: None,
        size_variant: None,
    };

    config.populate_from_family_contract(&contract, "0.5b");

    // Should remain default since no certification config
    assert_eq!(config.size_category, SizeCategory::Tiny);
}

// ── GH-6/AC-2: Ollama parity config tests ────────────────────────────

#[test]
fn test_playbook_with_ollama_parity() {
    let yaml = r#"
name: ollama-test
version: "1.0.0"
model:
  hf_repo: "test/model"
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
ollama_parity:
  enabled: true
  model_tag: "qwen2.5-coder:7b-instruct-q4_k_m"
  quantizations: ["q4_k_m", "q6_k"]
  prompts: ["What is 2+2?", "def hello():"]
  temperature: 0.0
  min_perf_ratio: 0.9
  gates: ["F-OLLAMA-001", "F-OLLAMA-002"]
"#;
    let playbook = Playbook::from_yaml(yaml).expect("Failed to parse");
    let ollama = playbook.ollama_parity.expect("Should have ollama parity");

    assert!(ollama.enabled);
    assert_eq!(
        ollama.model_tag,
        Some("qwen2.5-coder:7b-instruct-q4_k_m".to_string())
    );
    assert_eq!(ollama.quantizations.len(), 2);
    assert_eq!(ollama.prompts.len(), 2);
    assert!((ollama.temperature - 0.0).abs() < f64::EPSILON);
    assert!((ollama.min_perf_ratio - 0.9).abs() < f64::EPSILON);
    assert_eq!(ollama.gates.len(), 2);
}

#[test]
fn test_playbook_without_ollama_parity() {
    let yaml = r#"
name: no-ollama
version: "1.0.0"
model:
  hf_repo: "test/model"
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
"#;
    let playbook = Playbook::from_yaml(yaml).expect("parse");
    assert!(playbook.ollama_parity.is_none());
}

#[test]
fn test_ollama_parity_config_defaults() {
    let yaml = r#"
name: ollama-defaults
version: "1.0.0"
model:
  hf_repo: "test/model"
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
ollama_parity:
  enabled: true
"#;
    let playbook = Playbook::from_yaml(yaml).expect("parse");
    let ollama = playbook.ollama_parity.expect("should exist");

    assert!(ollama.enabled);
    assert!(ollama.model_tag.is_none());
    assert_eq!(ollama.quantizations, vec!["q4_k_m"]);
    assert_eq!(ollama.prompts, vec!["What is 2+2?"]);
    assert!((ollama.temperature - 0.0).abs() < f64::EPSILON);
    assert!((ollama.min_perf_ratio - 0.8).abs() < f64::EPSILON);
    assert!(ollama.gates.is_empty());
}
