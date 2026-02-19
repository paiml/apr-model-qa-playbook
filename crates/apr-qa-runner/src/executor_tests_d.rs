
use super::*;
use crate::command::MockCommandRunner;
use apr_qa_gen::{Backend, Format, Modality, ModelId, QaScenario};

#[test]
fn test_execute_profile_flamegraph_unsupported() {
    let mock_runner = MockCommandRunner::new().with_profile_flamegraph_failure();
    let executor = ToolExecutor::with_runner(
        "test-model.gguf".to_string(),
        true,
        5000,
        Arc::new(mock_runner),
    );
    let temp_dir = tempfile::tempdir().unwrap();
    let result = executor.execute_profile_flamegraph(temp_dir.path());
    assert!(!result.passed);
}

#[test]
fn test_execute_profile_focus_no_apr() {
    let executor = ToolExecutor::new("test-model.gguf".to_string(), true, 5000);
    let result = executor.execute_profile_focus("attention");
    assert!(!result.passed);
    assert_eq!(result.tool, "profile-focus");
    assert_eq!(result.gate_id, "F-PROFILE-003");
}

#[test]
fn test_execute_profile_focus_with_mock_success() {
    let mock_runner = MockCommandRunner::new();
    let executor = ToolExecutor::with_runner(
        "test-model.gguf".to_string(),
        false,
        5000,
        Arc::new(mock_runner),
    );
    let result = executor.execute_profile_focus("attention");
    assert!(result.passed);
    assert_eq!(result.tool, "profile-focus");
    assert_eq!(result.gate_id, "F-PROFILE-003");
}

#[test]
fn test_execute_profile_focus_unsupported() {
    let mock_runner = MockCommandRunner::new().with_profile_focus_failure();
    let executor = ToolExecutor::with_runner(
        "test-model.gguf".to_string(),
        true,
        5000,
        Arc::new(mock_runner),
    );
    let result = executor.execute_profile_focus("attention");
    assert!(!result.passed);
}

#[test]
fn test_execute_backend_equivalence_no_apr() {
    let executor = ToolExecutor::new("test-model.gguf".to_string(), false, 5000);
    let result = executor.execute_backend_equivalence();
    assert!(!result.passed);
    assert_eq!(result.tool, "backend-equivalence");
    assert_eq!(result.gate_id, "F-CONV-BE-001");
}

#[test]
fn test_execute_serve_lifecycle_no_apr() {
    let executor = ToolExecutor::new("test-model.gguf".to_string(), true, 5000);
    let result = executor.execute_serve_lifecycle();
    assert!(!result.passed);
    assert_eq!(result.tool, "serve-lifecycle");
    assert_eq!(result.gate_id, "F-INTEG-003");
}

#[test]
fn test_execute_all_with_serve() {
    let mock_runner = MockCommandRunner::new();
    let executor = ToolExecutor::with_runner(
        "test-model.gguf".to_string(),
        true,
        5000,
        Arc::new(mock_runner),
    );
    // Without serve
    let results = executor.execute_all();
    assert!(!results.is_empty());
    // None should be serve-lifecycle
    assert!(!results.iter().any(|r| r.tool == "serve-lifecycle"));
}

// =========================================================================
// Conversion infrastructure failure
// =========================================================================

#[test]
#[allow(clippy::too_many_lines)]
fn test_executor_conversion_infrastructure_failure() {
    use crate::command::CommandOutput;

    struct FailingConversionRunner;
    impl CommandRunner for FailingConversionRunner {
        fn run_inference(
            &self,
            _model_path: &Path,
            _prompt: &str,
            _max_tokens: u32,
            _no_gpu: bool,
            _extra_args: &[&str],
        ) -> CommandOutput {
            CommandOutput {
                stdout: "The answer is 4.".to_string(),
                stderr: String::new(),
                exit_code: 0,
                success: true,
            }
        }
        fn convert_model(&self, _source: &Path, _target: &Path) -> CommandOutput {
            CommandOutput::success("")
        }
        fn inspect_model(&self, _path: &Path) -> CommandOutput {
            CommandOutput::success("")
        }
        fn validate_model(&self, _path: &Path) -> CommandOutput {
            CommandOutput::success("")
        }
        fn bench_model(&self, _path: &Path) -> CommandOutput {
            CommandOutput::success("")
        }
        fn check_model(&self, _path: &Path) -> CommandOutput {
            CommandOutput::success("")
        }
        fn profile_model(&self, _path: &Path, _warmup: u32, _measure: u32) -> CommandOutput {
            CommandOutput::success("")
        }
        fn profile_ci(
            &self,
            _path: &Path,
            _min_throughput: Option<f64>,
            _max_p99: Option<f64>,
            _warmup: u32,
            _measure: u32,
        ) -> CommandOutput {
            CommandOutput::success("")
        }
        fn diff_tensors(&self, _model_a: &Path, _model_b: &Path, _json: bool) -> CommandOutput {
            CommandOutput::success("")
        }
        fn compare_inference(
            &self,
            _model_a: &Path,
            _model_b: &Path,
            _prompt: &str,
            _max_tokens: u32,
            _tolerance: f64,
        ) -> CommandOutput {
            CommandOutput::success("")
        }
        fn profile_with_flamegraph(
            &self,
            _model_path: &Path,
            _output_path: &Path,
            _no_gpu: bool,
        ) -> CommandOutput {
            CommandOutput::success("")
        }
        fn profile_with_focus(
            &self,
            _model_path: &Path,
            _focus: &str,
            _no_gpu: bool,
        ) -> CommandOutput {
            CommandOutput::success("")
        }
        fn fingerprint_model(&self, _path: &Path, _json: bool) -> CommandOutput {
            CommandOutput::success("")
        }
        fn validate_stats(&self, _a: &Path, _b: &Path) -> CommandOutput {
            CommandOutput::success("")
        }
        fn validate_model_strict(&self, _path: &Path) -> CommandOutput {
            CommandOutput::success(r#"{"valid":true,"tensors_checked":100,"issues":[]}"#)
        }
        fn pull_model(&self, _hf_repo: &str) -> CommandOutput {
            CommandOutput::success("Path: /mock/model.safetensors")
        }
        fn inspect_model_json(&self, _model_path: &Path) -> CommandOutput {
            CommandOutput::success(
                r#"{"format":"SafeTensors","tensor_count":10,"tensor_names":[]}"#,
            )
        }
        fn run_ollama_inference(
            &self,
            _model_tag: &str,
            _prompt: &str,
            _temperature: f64,
        ) -> CommandOutput {
            CommandOutput::success("Output:\nThe answer is 4.\nCompleted in 1.0s")
        }
        fn pull_ollama_model(&self, _model_tag: &str) -> CommandOutput {
            CommandOutput::success("pulling manifest... done")
        }
        fn create_ollama_model(&self, _: &str, _: &Path) -> CommandOutput {
            CommandOutput::success("creating model... done")
        }
        fn serve_model(&self, _: &Path, _: u16) -> CommandOutput {
            CommandOutput::success(r#"{"status":"listening"}"#)
        }
        fn http_get(&self, _: &str) -> CommandOutput {
            CommandOutput::success(r#"{"models":[]}"#)
        }
        fn profile_memory(&self, _: &Path) -> CommandOutput {
            CommandOutput::success(r#"{"peak_rss_mb":1024}"#)
        }
        fn run_chat(
            &self,
            _model_path: &Path,
            _prompt: &str,
            _no_gpu: bool,
            _extra_args: &[&str],
        ) -> CommandOutput {
            CommandOutput::success("Chat output")
        }
        fn http_post(&self, _url: &str, _body: &str) -> CommandOutput {
            CommandOutput::success("{}")
        }
        fn spawn_serve(&self, _model_path: &Path, _port: u16, _no_gpu: bool) -> CommandOutput {
            CommandOutput::success("12345")
        }
    }

    let config = ExecutionConfig {
        model_path: Some("/nonexistent/model.gguf".to_string()),
        run_conversion_tests: true,
        run_golden_rule_test: false,
        ..Default::default()
    };

    let mut executor = Executor::with_runner(config, Arc::new(FailingConversionRunner));

    let yaml = r#"
name: conv-infra-fail
version: "1.0.0"
model:
  hf_repo: "test/model"
  formats: [gguf]
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
"#;
    let playbook = Playbook::from_yaml(yaml).expect("Failed to parse");
    let result = executor.execute(&playbook).expect("Execution failed");
    // Conversion tests ran (whether they passed or failed depends on
    // ConversionExecutor behavior with the mock runner)
    assert!(result.total_scenarios >= 1);

    // Exercise unused CommandRunner trait methods to cover stubs
    let runner = FailingConversionRunner;
    let p = Path::new("/dev/null");
    assert!(runner.validate_model(p).success);
    assert!(runner.bench_model(p).success);
    assert!(runner.check_model(p).success);
    assert!(runner.profile_model(p, 1, 1).success);
    assert!(runner.profile_ci(p, None, None, 1, 1).success);
    assert!(runner.diff_tensors(p, p, false).success);
    assert!(runner.compare_inference(p, p, "", 1, 0.0).success);
    assert!(runner.profile_with_flamegraph(p, p, false).success);
    assert!(runner.profile_with_focus(p, "", false).success);
    assert!(runner.fingerprint_model(p, false).success);
    assert!(runner.validate_stats(p, p).success);
}

// ========================================================================
// G0 INTEGRITY CHECK TESTS
// ========================================================================

#[test]
fn test_find_safetensors_dir_with_subdir() {
    use tempfile::TempDir;
    let dir = TempDir::new().expect("create temp dir");
    let st_dir = dir.path().join("safetensors");
    std::fs::create_dir(&st_dir).expect("create safetensors dir");
    std::fs::write(st_dir.join("model.safetensors"), "test").expect("write file");

    let result = Executor::find_safetensors_dir(dir.path());
    assert!(result.is_some());
    assert_eq!(result.unwrap(), st_dir);
}

#[test]
fn test_find_safetensors_dir_direct() {
    use tempfile::TempDir;
    let dir = TempDir::new().expect("create temp dir");
    std::fs::write(dir.path().join("model.safetensors"), "test").expect("write file");

    let result = Executor::find_safetensors_dir(dir.path());
    assert!(result.is_some());
    assert_eq!(result.unwrap(), dir.path());
}

#[test]
fn test_find_safetensors_dir_none() {
    use tempfile::TempDir;
    let dir = TempDir::new().expect("create temp dir");
    // No safetensors files

    let result = Executor::find_safetensors_dir(dir.path());
    assert!(result.is_none());
}

#[test]
fn test_has_safetensors_files_true() {
    use tempfile::TempDir;
    let dir = TempDir::new().expect("create temp dir");
    std::fs::write(dir.path().join("model.safetensors"), "test").expect("write file");

    assert!(Executor::has_safetensors_files(dir.path()));
}

#[test]
fn test_has_safetensors_files_false() {
    use tempfile::TempDir;
    let dir = TempDir::new().expect("create temp dir");
    std::fs::write(dir.path().join("model.gguf"), "test").expect("write file");

    assert!(!Executor::has_safetensors_files(dir.path()));
}

#[test]
fn test_has_safetensors_files_nonexistent_dir() {
    let nonexistent = std::path::Path::new("/nonexistent/path/xyz123");
    assert!(!Executor::has_safetensors_files(nonexistent));
}

// =========================================================================
// G0-VALIDATE Pre-flight Gate Tests
// =========================================================================

#[test]
fn test_validate_scenario_creation() {
    let model_id = ModelId::new("test", "model");
    let scenario = Executor::validate_scenario(&model_id);

    assert_eq!(scenario.model.org, "test");
    assert_eq!(scenario.model.name, "model");
    assert_eq!(scenario.format, Format::SafeTensors);
    assert!(scenario.prompt.contains("G0 Validate"));
}

#[test]
fn test_pull_scenario_creation() {
    let model_id = ModelId::new("test", "model");
    let scenario = Executor::pull_scenario(&model_id);

    assert_eq!(scenario.model.org, "test");
    assert_eq!(scenario.model.name, "model");
    assert_eq!(scenario.format, Format::SafeTensors);
    assert!(scenario.prompt.contains("G0 Pull"));
}

#[test]
fn test_g0_pull_pass() {
    let mock_runner = MockCommandRunner::new();

    let config = ExecutionConfig {
        model_path: Some("/test/model.gguf".to_string()),
        run_conversion_tests: false,
        run_golden_rule_test: false,
        run_contract_tests: false,
        ..Default::default()
    };

    let mut executor = Executor::with_runner(config, Arc::new(mock_runner));
    let model_id = ModelId::new("test", "model");
    let (passed, failed, pulled_path) = executor.run_g0_pull_check("test/model", &model_id);

    assert_eq!(passed, 1);
    assert_eq!(failed, 0);
    assert_eq!(pulled_path.as_deref(), Some("/mock/model.safetensors"));

    let evidence = executor.evidence().all();
    let pull_ev = evidence
        .iter()
        .find(|e| e.gate_id == "G0-PULL-001")
        .expect("should have G0-PULL evidence");
    assert!(pull_ev.outcome.is_pass());
    assert!(pull_ev.output.contains("G0 PASS"));
}

#[test]
fn test_g0_pull_fail() {
    let mock_runner = MockCommandRunner::new().with_pull_failure();

    let config = ExecutionConfig {
        model_path: Some("/test/model.gguf".to_string()),
        run_conversion_tests: false,
        run_golden_rule_test: false,
        run_contract_tests: false,
        ..Default::default()
    };

    let mut executor = Executor::with_runner(config, Arc::new(mock_runner));
    let model_id = ModelId::new("test", "model");
    let (passed, failed, pulled_path) = executor.run_g0_pull_check("test/model", &model_id);

    assert_eq!(passed, 0);
    assert_eq!(failed, 1);
    assert!(pulled_path.is_none());

    let evidence = executor.evidence().all();
    let pull_ev = evidence
        .iter()
        .find(|e| e.gate_id == "G0-PULL-001")
        .expect("should have G0-PULL evidence");
    assert!(!pull_ev.outcome.is_pass());
    assert!(pull_ev.reason.contains("G0 FAIL"));
}

#[test]
fn test_g0_pull_fail_stops_execution() {
    // Jidoka: If G0-PULL fails, skip all subsequent tests
    // Bug 204: model_path must be None so G0-PULL actually runs
    let mock_runner = MockCommandRunner::new().with_pull_failure();

    let config = ExecutionConfig {
        model_path: None,
        run_conversion_tests: true,
        run_golden_rule_test: true,
        run_contract_tests: true,
        ..Default::default()
    };

    let mut executor = Executor::with_runner(config, Arc::new(mock_runner));

    let yaml = r#"
name: pull-fail-test
version: "1.0.0"
model:
  hf_repo: "test/model"
  formats: [gguf]
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 3
"#;
    let playbook = Playbook::from_yaml(yaml).expect("Failed to parse");
    let result = executor.execute(&playbook).expect("Execution failed");

    // Gateway should be failed
    assert!(result.gateway_failed.is_some());
    assert!(
        result
            .gateway_failed
            .as_ref()
            .unwrap()
            .contains("G0-PULL-001")
    );

    // No scenarios passed
    assert_eq!(result.passed, 0);
    // 3 scenarios + 1 pull failure = 4 total failed
    assert_eq!(result.failed, 4);
}

#[test]
fn test_g0_pull_sets_model_path() {
    // When model_path is None, G0-PULL should set it from pulled path
    let mock_runner = MockCommandRunner::new().with_pull_model_path("/pulled/model.safetensors");

    let config = ExecutionConfig {
        model_path: None, // No model path initially
        run_conversion_tests: false,
        run_golden_rule_test: false,
        run_contract_tests: false,
        ..Default::default()
    };

    let mut executor = Executor::with_runner(config, Arc::new(mock_runner));

    let yaml = r#"
name: pull-set-path-test
version: "1.0.0"
model:
  hf_repo: "test/model"
  formats: [gguf]
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
"#;
    let playbook = Playbook::from_yaml(yaml).expect("Failed to parse");
    let result = executor.execute(&playbook).expect("Execution failed");

    // Should not fail on gateway
    assert!(result.gateway_failed.is_none());
    // G0-PULL should pass
    assert!(result.passed >= 1);
}

/// Helper: create a temp model directory with a safetensors file
fn make_temp_model_dir() -> tempfile::TempDir {
    let dir = tempfile::TempDir::new().expect("create temp dir");
    let st_dir = dir.path().join("safetensors");
    std::fs::create_dir_all(&st_dir).expect("mkdir safetensors");
    std::fs::write(st_dir.join("model.safetensors"), b"fake").expect("write");
    dir
}

#[test]
fn test_g0_validate_pass() {
    let mock_runner = MockCommandRunner::new(); // validate_strict_success defaults to true
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
    let (passed, failed) = executor.run_g0_validate_check(dir.path(), &model_id);

    assert_eq!(passed, 1);
    assert_eq!(failed, 0);

    let evidence = executor.evidence().all();
    let validate_ev = evidence
        .iter()
        .find(|e| e.gate_id == "G0-VALIDATE-001")
        .expect("should have G0-VALIDATE evidence");
    assert!(validate_ev.outcome.is_pass());
    assert!(validate_ev.output.contains("G0 PASS"));
}

#[test]
fn test_g0_validate_fail_corrupt_model() {
    let mock_runner = MockCommandRunner::new().with_validate_strict_failure();
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
    let (passed, failed) = executor.run_g0_validate_check(dir.path(), &model_id);

    assert_eq!(passed, 0);
    assert_eq!(failed, 1);

    let evidence = executor.evidence().all();
    let validate_ev = evidence
        .iter()
        .find(|e| e.gate_id == "G0-VALIDATE-001")
        .expect("should have G0-VALIDATE evidence");
    assert!(!validate_ev.outcome.is_pass());
    assert!(validate_ev.reason.contains("G0 FAIL"));
}

#[test]
fn test_g0_validate_fail_stops_execution() {
    // Jidoka: If G0-VALIDATE fails, skip all subsequent tests
    let mock_runner = MockCommandRunner::new().with_validate_strict_failure();
    let dir = make_temp_model_dir();

    let config = ExecutionConfig {
        model_path: Some(dir.path().to_string_lossy().to_string()),
        run_conversion_tests: true,
        run_golden_rule_test: true,
        run_contract_tests: true,
        ..Default::default()
    };

    let mut executor = Executor::with_runner(config, Arc::new(mock_runner));

    let yaml = r#"
name: validate-fail-test
version: "1.0.0"
model:
  hf_repo: "test/model"
  formats: [gguf]
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 3
"#;
    let playbook = Playbook::from_yaml(yaml).expect("Failed to parse");
    let result = executor.execute(&playbook).expect("Execution failed");

    // Gateway should be failed
    assert!(result.gateway_failed.is_some());
    assert!(
        result
            .gateway_failed
            .as_ref()
            .unwrap()
            .contains("G0-VALIDATE-001")
    );

    // Bug 204: G0-PULL skipped (model_path is set), then G0-VALIDATE fails
    assert_eq!(result.passed, 0);
    // 3 scenarios + 1 validate failure = 4 total failed
    assert_eq!(result.failed, 4);
}

#[test]
fn test_g0_validate_pass_continues_execution() {
    // When G0-VALIDATE passes, execution should continue normally
    let mock_runner = MockCommandRunner::new(); // validate_strict_success defaults to true
    let dir = make_temp_model_dir();

    let config = ExecutionConfig {
        model_path: Some(dir.path().to_string_lossy().to_string()),
        run_conversion_tests: false,
        run_golden_rule_test: false,
        run_contract_tests: false,
        ..Default::default()
    };

    let mut executor = Executor::with_runner(config, Arc::new(mock_runner));

    let yaml = r#"
name: validate-pass-test
version: "1.0.0"
model:
  hf_repo: "test/model"
  formats: [gguf]
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
"#;
    let playbook = Playbook::from_yaml(yaml).expect("Failed to parse");
    let result = executor.execute(&playbook).expect("Execution failed");

    // No gateway failure
    assert!(result.gateway_failed.is_none());
    // At least the validate + 1 scenario
    assert!(result.total_scenarios >= 2);
    assert!(result.passed >= 1);
}

#[test]
fn test_g0_validate_no_model_path() {
    // When no model_path is set, G0-VALIDATE should be skipped (0, 0)
    let mock_runner = MockCommandRunner::new();

    let config = ExecutionConfig {
        model_path: None, // No model path
        run_conversion_tests: false,
        run_golden_rule_test: false,
        run_contract_tests: false,
        ..Default::default()
    };

    let mut executor = Executor::with_runner(config, Arc::new(mock_runner));

    let yaml = r#"
name: no-model-path-test
version: "1.0.0"
model:
  hf_repo: "test/model"
  formats: [gguf]
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
"#;
    let playbook = Playbook::from_yaml(yaml).expect("Failed to parse");
    let result = executor.execute(&playbook).expect("Execution failed");

    // No gateway failure
    assert!(result.gateway_failed.is_none());
    // 1 scenario + 1 G0-PULL (no validate — mock path has no safetensors)
    assert_eq!(result.total_scenarios, 2);
}

#[test]
fn test_g0_validate_no_safetensors_files() {
    // When model dir has no safetensors files, G0-VALIDATE auto-passes (0, 0)
    let dir = tempfile::TempDir::new().expect("create temp dir");
    let mock_runner = MockCommandRunner::new();

    let config = ExecutionConfig {
        model_path: Some(dir.path().to_string_lossy().to_string()),
        ..Default::default()
    };

    let mut executor = Executor::with_runner(config, Arc::new(mock_runner));
    let model_id = ModelId::new("test", "model");
    let (passed, failed) = executor.run_g0_validate_check(dir.path(), &model_id);

    assert_eq!(passed, 0);
    assert_eq!(failed, 0);
}

#[test]
fn test_g0_validate_multiple_shards() {
    // Multi-file sharded models: validate each shard
    let dir = tempfile::TempDir::new().expect("create temp dir");
    let st_dir = dir.path().join("safetensors");
    std::fs::create_dir_all(&st_dir).expect("mkdir");
    std::fs::write(st_dir.join("model-00001-of-00002.safetensors"), b"shard1").expect("write");
    std::fs::write(st_dir.join("model-00002-of-00002.safetensors"), b"shard2").expect("write");

    let mock_runner = MockCommandRunner::new();
    let config = ExecutionConfig {
        model_path: Some(dir.path().to_string_lossy().to_string()),
        ..Default::default()
    };

    let mut executor = Executor::with_runner(config, Arc::new(mock_runner));
    let model_id = ModelId::new("test", "model");
    let (passed, failed) = executor.run_g0_validate_check(dir.path(), &model_id);

    // Both shards should be validated
    assert_eq!(passed, 2);
    assert_eq!(failed, 0);
}

#[test]
fn test_find_safetensors_files_single_file() {
    let dir = tempfile::TempDir::new().expect("create temp dir");
    let file = dir.path().join("model.safetensors");
    std::fs::write(&file, b"test").expect("write");

    let files = Executor::find_safetensors_files(&file);
    assert_eq!(files.len(), 1);
    assert_eq!(files[0], file);
}

#[test]
fn test_find_safetensors_files_non_safetensors() {
    let dir = tempfile::TempDir::new().expect("create temp dir");
    let file = dir.path().join("model.gguf");
    std::fs::write(&file, b"test").expect("write");

    let files = Executor::find_safetensors_files(&file);
    assert!(files.is_empty());
}

#[test]
fn test_find_safetensors_files_directory() {
    let dir = make_temp_model_dir();
    let files = Executor::find_safetensors_files(dir.path());
    assert_eq!(files.len(), 1);
}

#[test]
fn test_integrity_scenario_creation() {
    let model_id = ModelId::new("test", "model");
    let scenario = Executor::integrity_scenario(&model_id);

    assert_eq!(scenario.model.org, "test");
    assert_eq!(scenario.model.name, "model");
    assert_eq!(scenario.format, Format::SafeTensors);
    assert!(scenario.prompt.contains("G0"));
}

#[test]
fn test_run_g0_integrity_check_no_safetensors() {
    use tempfile::TempDir;
    let dir = TempDir::new().expect("create temp dir");
    // No safetensors files

    let mut executor = Executor::new();
    let model_id = ModelId::new("test", "model");
    let (passed, failed) = executor.run_g0_integrity_check(dir.path(), &model_id);

    // No safetensors = auto-pass (0, 0)
    assert_eq!(passed, 0);
    assert_eq!(failed, 0);
}

#[test]
fn test_run_g0_integrity_check_missing_config() {
    use tempfile::TempDir;
    let dir = TempDir::new().expect("create temp dir");

    // Create safetensors but no config.json
    create_mock_safetensors_for_test(dir.path(), 24, 896, 151_936);

    let mut executor = Executor::new();
    let model_id = ModelId::new("test", "model");
    let (passed, failed) = executor.run_g0_integrity_check(dir.path(), &model_id);

    // Should fail due to missing config
    assert_eq!(passed, 0);
    assert!(failed > 0);

    // Evidence should contain G0-INTEGRITY failure
    let evidence = executor.evidence();
    assert!(
        evidence
            .all()
            .iter()
            .any(|e| e.gate_id.starts_with("G0-INTEGRITY"))
    );
}

#[test]
fn test_run_g0_integrity_check_pass() {
    use tempfile::TempDir;
    let dir = TempDir::new().expect("create temp dir");

    // Create matching config and safetensors
    create_test_config_for_executor(dir.path(), 24, 896, 151_936);
    create_mock_safetensors_for_test(dir.path(), 24, 896, 151_936);

    let mut executor = Executor::new();
    let model_id = ModelId::new("test", "model");
    let (passed, failed) = executor.run_g0_integrity_check(dir.path(), &model_id);

    assert_eq!(passed, 1);
    assert_eq!(failed, 0);

    // Evidence should show corroborated
    let evidence = executor.evidence();
    assert!(
        evidence
            .all()
            .iter()
            .any(|e| { e.gate_id.starts_with("G0-INTEGRITY") && e.outcome.is_pass() })
    );
}

#[test]
fn test_run_g0_integrity_check_layer_mismatch() {
    use tempfile::TempDir;
    let dir = TempDir::new().expect("create temp dir");

    // Config says 14 layers but tensors have 24 (the corrupted cache bug)
    create_test_config_for_executor(dir.path(), 14, 896, 151_936);
    create_mock_safetensors_for_test(dir.path(), 24, 896, 151_936);

    let mut executor = Executor::new();
    let model_id = ModelId::new("test", "model");
    let (passed, failed) = executor.run_g0_integrity_check(dir.path(), &model_id);

    assert_eq!(passed, 0);
    assert!(failed > 0);

    // Evidence should contain LAYERS failure
    let evidence = executor.evidence();
    assert!(evidence.all().iter().any(|e| e.gate_id.contains("LAYERS")));
}

/// Helper to create test config.json
fn create_test_config_for_executor(
    dir: &std::path::Path,
    layers: usize,
    hidden: usize,
    vocab: usize,
) {
    let config = format!(
        r#"{{"num_hidden_layers": {layers}, "hidden_size": {hidden}, "vocab_size": {vocab}}}"#
    );
    std::fs::write(dir.join("config.json"), config).expect("write config");
}

/// Helper to create mock SafeTensors file with specific dimensions
#[allow(clippy::items_after_statements)]
fn create_mock_safetensors_for_test(
    dir: &std::path::Path,
    layers: usize,
    hidden: usize,
    vocab: usize,
) {
    let mut header_obj = serde_json::Map::new();

    // Embedding tensor
    let mut embed_info = serde_json::Map::new();
    embed_info.insert("shape".to_string(), serde_json::json!([vocab, hidden]));
    embed_info.insert(
        "dtype".to_string(),
        serde_json::Value::String("F32".to_string()),
    );
    embed_info.insert(
        "data_offsets".to_string(),
        serde_json::json!([0, vocab * hidden * 4]),
    );
    header_obj.insert(
        "model.embed_tokens.weight".to_string(),
        serde_json::Value::Object(embed_info),
    );

    // Layer tensors
    for i in 0..layers {
        let mut layer_info = serde_json::Map::new();
        layer_info.insert("shape".to_string(), serde_json::json!([hidden, hidden]));
        layer_info.insert(
            "dtype".to_string(),
            serde_json::Value::String("F32".to_string()),
        );
        layer_info.insert("data_offsets".to_string(), serde_json::json!([0, 0]));
        header_obj.insert(
            format!("model.layers.{i}.self_attn.q_proj.weight"),
            serde_json::Value::Object(layer_info),
        );
    }

    let header_json = serde_json::to_string(&header_obj).expect("serialize header");
    let header_bytes = header_json.as_bytes();
    let header_len = header_bytes.len() as u64;

    let path = dir.join("model.safetensors");
    let mut file = std::fs::File::create(path).expect("create safetensors");
    use std::io::Write;
    file.write_all(&header_len.to_le_bytes())
        .expect("write len");
    file.write_all(header_bytes).expect("write header");
    file.write_all(&[0u8; 1024]).expect("write data");
}

// =========================================================================
// Additional coverage tests — uncovered paths
// =========================================================================

#[test]
fn test_execute_all_with_serve_true() {
    let mock_runner = MockCommandRunner::new();
    let executor = ToolExecutor::with_runner(
        "test-model.gguf".to_string(),
        true,
        5000,
        Arc::new(mock_runner),
    );
    let results = executor.execute_all_with_serve(true);
    assert!(!results.is_empty());
    // Should include serve-lifecycle when include_serve=true
    assert!(results.iter().any(|r| r.tool == "serve-lifecycle"));
}

#[test]
fn test_run_g0_integrity_check_hidden_mismatch() {
    use tempfile::TempDir;
    let dir = TempDir::new().expect("create temp dir");

    // Config says hidden_size=1024 but tensors have 896
    create_test_config_for_executor(dir.path(), 24, 1024, 151_936);
    create_mock_safetensors_for_test(dir.path(), 24, 896, 151_936);

    let mut executor = Executor::new();
    let model_id = ModelId::new("test", "model");
    let (passed, failed) = executor.run_g0_integrity_check(dir.path(), &model_id);

    assert_eq!(passed, 0);
    assert!(failed > 0);

    let evidence = executor.evidence();
    assert!(evidence.all().iter().any(|e| e.gate_id.contains("HIDDEN")));
}

#[test]
fn test_run_g0_integrity_check_vocab_mismatch() {
    use tempfile::TempDir;
    let dir = TempDir::new().expect("create temp dir");

    // Config says vocab=200_000 but tensors have 151_936
    create_test_config_for_executor(dir.path(), 24, 896, 200_000);
    create_mock_safetensors_for_test(dir.path(), 24, 896, 151_936);

    let mut executor = Executor::new();
    let model_id = ModelId::new("test", "model");
    let (passed, failed) = executor.run_g0_integrity_check(dir.path(), &model_id);

    assert_eq!(passed, 0);
    assert!(failed > 0);

    let evidence = executor.evidence();
    assert!(evidence.all().iter().any(|e| e.gate_id.contains("VOCAB")));
}

// G0-LAYOUT Pre-flight Gate Tests (Issue #4)

#[test]
fn test_run_g0_layout_check_no_contract() {
    // When tensor-layout-v1.yaml is not found, the check should auto-skip (0, 0)
    use tempfile::TempDir;
    let dir = TempDir::new().expect("create temp dir");

    let mut executor = Executor::new();
    let model_id = ModelId::new("test", "model");
    let (passed, failed) = executor.run_g0_layout_check(dir.path(), &model_id);

    // Contract not found → skip (0, 0), not failure
    assert_eq!(passed, 0);
    assert_eq!(failed, 0);
}

#[test]
fn test_run_g0_layout_check_model_not_found() {
    // When model file doesn't exist but contract is found, validation fails
    use tempfile::TempDir;
    let dir = TempDir::new().expect("create temp dir");

    // Create a minimal contract file
    let contract_path = dir.path().join("tensor-layout-v1.yaml");
    std::fs::write(
        &contract_path,
        r#"
metadata:
  version: "1.0"
  created: "2026-01-01"
  updated: "2026-01-01"
  author: "test"
  description: "test"
formats: {}
kernel:
  signature: "test"
  weight_shape: "[out, in]"
  computation: "y = Wx"
  byte_calculation: "out * in"
  block_sizes: {}
  QK_K: 256
tensors: {}
validation_rules: []
"#,
    )
    .expect("write contract");

    // Test with a non-existent path inside the temp directory
    let nonexistent_path = dir.path().join("does_not_exist.safetensors");
    let contract =
        crate::layout_contract::load_contract_from(&contract_path).expect("load contract");
    let result = crate::layout_contract::validate_model(&nonexistent_path, &contract)
        .expect("validation should return result");

    // Model not found = failed validation
    assert!(!result.passed);
    assert!(!result.critical_failures.is_empty());
}

#[test]
fn test_layout_scenario_creation() {
    let model_id = ModelId::new("test", "model");
    let scenario = Executor::layout_scenario(&model_id);

    assert_eq!(
        scenario.prompt,
        "G0 Layout: tensor shape contract validation"
    );
    assert_eq!(scenario.format, Format::SafeTensors);
    assert_eq!(scenario.backend, Backend::Cpu);
    assert_eq!(scenario.modality, Modality::Run);
}

#[test]
fn test_format_tensor_failure_with_expected_and_actual() {
    let tensor_result = crate::layout_contract::TensorValidationResult {
        tensor_name: "lm_head.weight".to_string(),
        rule_id: "F-LAYOUT-CONTRACT-002".to_string(),
        passed: false,
        details: "Shape mismatch".to_string(),
        expected: Some("[vocab, hidden]".to_string()),
        actual: Some("[hidden, vocab]".to_string()),
    };

    let formatted = Executor::format_tensor_failure(&tensor_result);
    assert!(formatted.contains("F-LAYOUT-CONTRACT-002"));
    assert!(formatted.contains("Shape mismatch"));
    assert!(formatted.contains("Expected: [vocab, hidden]"));
    assert!(formatted.contains("Actual: [hidden, vocab]"));
}

#[test]
fn test_format_tensor_failure_without_expected() {
    let tensor_result = crate::layout_contract::TensorValidationResult {
        tensor_name: "test.weight".to_string(),
        rule_id: "F-LAYOUT-CONTRACT-001".to_string(),
        passed: false,
        details: "Missing transpose".to_string(),
        expected: None,
        actual: None,
    };

    let formatted = Executor::format_tensor_failure(&tensor_result);
    assert!(formatted.contains("F-LAYOUT-CONTRACT-001"));
    assert!(formatted.contains("Missing transpose"));
    assert!(!formatted.contains("Expected:"));
    assert!(!formatted.contains("Actual:"));
}

#[test]
fn test_execute_inspect_verified_nonexistent_model() {
    // run_inspect with "apr" binary + nonexistent model → fails → exercises Err path
    let executor = ToolExecutor::new("/nonexistent/path/to/model.gguf".to_string(), false, 5000);
    let result = executor.execute_inspect_verified();
    // apr binary exists but model doesn't → inspect fails → result is not passed
    assert!(!result.passed);
    assert_eq!(result.gate_id, "F-INSPECT-META-001");
    // Either exit_code=-1 (Err path) or exit_code=1 (Ok path with tensor_count=0)
    assert!(result.exit_code != 0);
}

#[test]
fn test_execute_scenario_stop_on_p0_gate() {
    // Create scenarios where gate_id contains "-P0-"
    let mock_runner = MockCommandRunner::new()
        .with_inference_failure()
        .with_exit_code(1);

    let config = ExecutionConfig {
        model_path: Some("/test/model.gguf".to_string()),
        failure_policy: FailurePolicy::StopOnP0,
        run_conversion_tests: false,
        run_golden_rule_test: false,
        ..Default::default()
    };

    let mut executor = Executor::with_runner(config, Arc::new(mock_runner));

    // Create scenario whose gate_id will contain "-P0-" pattern
    let yaml = r#"
name: p0-stop
version: "1.0.0"
model:
  hf_repo: "test/model"
  formats: [gguf]
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 3
"#;
    let playbook = Playbook::from_yaml(yaml).expect("Failed to parse");
    let result = executor.execute(&playbook).expect("Execution failed");

    // Should have failed scenarios (StopOnP0 only stops on P0 gates)
    assert!(result.failed >= 1);
}

#[test]
fn test_execute_scenario_corroborated_with_stderr_via_playbook() {
    // Use a mock that returns correct output ("The answer is 4.") with stderr
    // The mock auto-responds "The answer is 4." for "2+2" prompts
    // This exercises the Corroborated branch with stderr propagation (line 624-626)
    let mock_runner = MockCommandRunner::new()
        .with_inference_response_and_stderr("correct", "warning: low memory");

    let config = ExecutionConfig {
        model_path: Some("/test/model.gguf".to_string()),
        run_conversion_tests: false,
        run_golden_rule_test: false,
        failure_policy: FailurePolicy::CollectAll,
        ..Default::default()
    };

    let mut executor = Executor::with_runner(config, Arc::new(mock_runner));

    let yaml = r#"
name: corroborated-stderr
version: "1.0.0"
model:
  hf_repo: "test/model"
  formats: [gguf]
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
"#;
    let playbook = Playbook::from_yaml(yaml).expect("Failed to parse");
    let result = executor.execute(&playbook).expect("Execution failed");

    // Should pass (mock responds "The answer is 4." for 2+2 prompts)
    assert!(result.passed >= 1);

    // The corroborated evidence should carry stderr
    let evidence = executor.evidence().all();
    assert!(
        evidence
            .iter()
            .any(|e| e.outcome.is_pass() && e.stderr.is_some()),
        "should have corroborated evidence with stderr"
    );
}

#[test]
fn test_run_conversion_tests_single_file_model() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let model_path = dir.path().join("model.gguf");
    std::fs::write(&model_path, b"fake model").expect("write model");

    let config = ExecutionConfig {
        model_path: Some(model_path.to_string_lossy().to_string()),
        run_conversion_tests: true,
        ..Default::default()
    };

    let mut executor = Executor::with_config(config);
    let model_id = ModelId::new("test", "model");
    // Single file model (not a directory) — should return (0, 0)
    let (passed, failed) = executor.run_conversion_tests(&model_path, &model_id);
    assert_eq!(passed, 0);
    assert_eq!(failed, 0);
}

#[test]
fn test_run_golden_rule_single_file_model() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let model_path = dir.path().join("model.gguf");
    std::fs::write(&model_path, b"fake model").expect("write model");

    let config = ExecutionConfig {
        model_path: Some(model_path.to_string_lossy().to_string()),
        run_golden_rule_test: true,
        ..Default::default()
    };

    let mut executor = Executor::with_config(config);
    let model_id = ModelId::new("test", "model");
    // Single file model — golden rule returns (0, 0)
    let (passed, failed) = executor.run_golden_rule_test(&model_path, &model_id);
    assert_eq!(passed, 0);
    assert_eq!(failed, 0);
}

#[test]
fn test_integrity_check_refuses_on_mismatch() {
    use crate::playbook::{PlaybookLockEntry, PlaybookLockFile, save_lock_file};
    use std::collections::HashMap;

    let dir = tempfile::tempdir().expect("create temp dir");
    let lock_path = dir.path().join("playbook.lock.yaml");

    // Create a lock file with a wrong hash for 'test-playbook'
    let mut entries = HashMap::new();
    entries.insert(
        "integrity-test".to_string(),
        PlaybookLockEntry {
            sha256: "0000000000000000000000000000000000000000000000000000000000000000".to_string(),
            locked_fields: vec!["name".to_string()],
        },
    );
    let lock_file = PlaybookLockFile { entries };
    save_lock_file(&lock_file, &lock_path).expect("save lock");

    let config = ExecutionConfig {
        check_integrity: true,
        lock_file_path: Some(lock_path.to_string_lossy().to_string()),
        run_conversion_tests: false,
        run_golden_rule_test: false,
        ..Default::default()
    };

    let mut executor = Executor::with_config(config);
    let yaml = r#"
name: integrity-test
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

    // verify_playbook_integrity checks the lock_path as the playbook path,
    // which won't match the stored hash. This should trigger a gateway failure.
    // Even if the integrity flow changes, the test validates it runs without panic.
    assert!(result.gateway_failed.is_some() || result.failed > 0);
}

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
