#[test]
fn test_resolve_hf_repo_with_dirs_multiple_snapshots() {
    let tmp = tempfile::TempDir::new().unwrap();
    let snapshots_dir = tmp.path().join("models--Test--Multi").join("snapshots");

    // Create two snapshots, only second has model.safetensors
    let snap1 = snapshots_dir.join("aaa111");
    let snap2 = snapshots_dir.join("bbb222");
    std::fs::create_dir_all(&snap1).unwrap();
    std::fs::create_dir_all(&snap2).unwrap();
    std::fs::write(snap2.join("model.safetensors"), b"fake").unwrap();

    let result = resolve_hf_repo_with_dirs("Test/Multi", tmp.path(), tmp.path());
    assert!(result.is_ok());
    assert!(result.unwrap().join("model.safetensors").exists());
}

#[test]
fn test_resolve_hf_repo_with_dirs_hf_cache_priority() {
    let tmp = tempfile::TempDir::new().unwrap();

    // Create both HF and APR cache entries
    let hf_snapshot = tmp
        .path()
        .join("models--Test--Both")
        .join("snapshots")
        .join("hf123");
    std::fs::create_dir_all(&hf_snapshot).unwrap();
    std::fs::write(hf_snapshot.join("model.safetensors"), b"hf").unwrap();

    let apr_cache = tmp.path().join(".cache/apr-models/Test/Both");
    std::fs::create_dir_all(&apr_cache).unwrap();

    // HF cache should take priority
    let result = resolve_hf_repo_with_dirs("Test/Both", tmp.path(), tmp.path());
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), hf_snapshot);
}

#[test]
fn test_get_hf_cache_dir_returns_path() {
    // Just verify it returns a PathBuf without errors
    // The actual value depends on environment, so we just check it doesn't panic
    let dir = get_hf_cache_dir();
    assert!(!dir.as_os_str().is_empty());
}

#[test]
fn test_resolve_hf_repo_to_cache_error_message_format() {
    // Test with nonexistent paths - this tests the error message format
    let tmp = tempfile::TempDir::new().unwrap();
    let hf_cache = tmp.path().join("hf_empty");
    let home = tmp.path().join("home_empty");
    std::fs::create_dir_all(&hf_cache).unwrap();
    std::fs::create_dir_all(&home).unwrap();

    let result = resolve_hf_repo_with_dirs("Org/Repo", &hf_cache, &home);
    assert!(result.is_err());

    let err_msg = result.unwrap_err().to_string();
    // Check error message contains useful debugging info
    assert!(err_msg.contains("Org/Repo"));
    assert!(err_msg.contains("Searched:"));
    assert!(err_msg.contains("models--Org--Repo"));
    assert!(err_msg.contains("apr-models"));
}

// =========================================================================
// Coverage improvement tests: ConversionOutputDir, is_garbage_output,
// format_extension, classify_failure edge cases, tolerance_for, etc.
// =========================================================================

#[test]
fn test_conversion_output_dir_all_paths() {
    let model_id = ModelId::new("my-org", "my-repo");
    let out_dir = ConversionOutputDir::new(std::path::Path::new("/tmp/output"), &model_id);

    let base = std::path::PathBuf::from("/tmp/output/conversions/my-org/my-repo");
    assert_eq!(out_dir.basic_dir(), base.join("basic"));
    assert_eq!(out_dir.semantic_dir(), base.join("semantic"));
    assert_eq!(out_dir.idempotency_dir(), base.join("idempotency"));
    assert_eq!(out_dir.comparison_dir(), base.join("comparison"));
    assert_eq!(out_dir.round_trip_dir(), base.join("round-trip"));
}

#[test]
fn test_conversion_output_dir_output_path_all_formats() {
    let model_id = ModelId::new("org", "repo");
    let out_dir = ConversionOutputDir::new(std::path::Path::new("/out"), &model_id);

    let gguf_path = out_dir.output_path("basic", "model", "direct", Format::Gguf);
    assert!(gguf_path.to_str().unwrap().ends_with("model.direct.gguf"));

    let st_path = out_dir.output_path("semantic", "model", "ref", Format::SafeTensors);
    assert!(st_path.to_str().unwrap().ends_with("model.ref.safetensors"));

    let apr_path = out_dir.output_path("round-trip", "model", "rt1", Format::Apr);
    assert!(apr_path.to_str().unwrap().ends_with("model.rt1.apr"));
}

#[test]
fn test_conversion_output_dir_ensure_dir_and_cleanup() {
    let tmp = tempfile::TempDir::new().unwrap();
    let model_id = ModelId::new("test-org", "test-repo");
    let out_dir = ConversionOutputDir::new(tmp.path(), &model_id);

    // ensure_dir creates the directory
    let created = out_dir.ensure_dir("basic").unwrap();
    assert!(created.exists());
    assert!(created.is_dir());

    // cleanup removes the model directory
    out_dir.cleanup().unwrap();
    assert!(!out_dir.basic_dir().exists());
}

#[test]
fn test_conversion_output_dir_cleanup_nonexistent() {
    let tmp = tempfile::TempDir::new().unwrap();
    let model_id = ModelId::new("no-such", "model");
    let out_dir = ConversionOutputDir::new(tmp.path(), &model_id);

    // cleanup on nonexistent dir should not error
    assert!(out_dir.cleanup().is_ok());
}

#[test]
fn test_conversion_output_dir_clone() {
    let model_id = ModelId::new("org", "repo");
    let out_dir = ConversionOutputDir::new(std::path::Path::new("/tmp"), &model_id);
    let cloned = out_dir.clone();
    assert_eq!(out_dir.basic_dir(), cloned.basic_dir());
}

#[test]
fn test_conversion_output_dir_debug() {
    let model_id = ModelId::new("org", "repo");
    let out_dir = ConversionOutputDir::new(std::path::Path::new("/tmp"), &model_id);
    let debug_str = format!("{out_dir:?}");
    assert!(debug_str.contains("ConversionOutputDir"));
}

#[test]
fn test_conversion_test_with_output_dir() {
    let model_id = ModelId::new("org", "repo");
    let out = ConversionOutputDir::new(std::path::Path::new("/tmp"), &model_id);
    let test = ConversionTest::new(Format::Gguf, Format::Apr, Backend::Cpu, model_id)
        .with_output_dir(out.clone());
    assert!(test.output_dir.is_some());
    assert_eq!(test.output_dir.unwrap().basic_dir(), out.basic_dir());
}

#[test]
fn test_conversion_executor_with_output_dir() {
    let executor =
        ConversionExecutor::with_defaults().with_output_dir(std::path::PathBuf::from("/tmp/out"));
    assert!(executor.output_dir.is_some());
    assert_eq!(
        executor.output_dir.unwrap(),
        std::path::PathBuf::from("/tmp/out")
    );
}

// ── is_garbage_output edge cases ──────────────────────────────────

#[test]
fn test_is_garbage_output_empty() {
    assert!(ConversionTest::is_garbage_output(""));
    assert!(ConversionTest::is_garbage_output("  "));
}

#[test]
fn test_is_garbage_output_too_short() {
    assert!(ConversionTest::is_garbage_output("ab"));
    assert!(ConversionTest::is_garbage_output("a"));
}

#[test]
fn test_is_garbage_output_few_unique_chars() {
    // Less than 3 unique characters
    assert!(ConversionTest::is_garbage_output("aaabbb"));
    assert!(ConversionTest::is_garbage_output("xxxxxx"));
}

#[test]
fn test_is_garbage_output_repetitive_trigrams() {
    // Highly repetitive pattern with >= 9 chars
    assert!(ConversionTest::is_garbage_output("abcabcabcabcabc"));
}

#[test]
fn test_is_garbage_output_valid_text() {
    assert!(!ConversionTest::is_garbage_output("The answer is 4."));
    assert!(!ConversionTest::is_garbage_output(
        "Hello world, this is a test."
    ));
}

#[test]
fn test_is_garbage_output_short_but_diverse() {
    // Exactly 3 chars with 3 unique - not garbage
    assert!(!ConversionTest::is_garbage_output("abc"));
}

#[test]
fn test_is_garbage_output_nine_chars_nonrepetitive() {
    // 9 chars, diverse trigrams - should not be garbage
    assert!(!ConversionTest::is_garbage_output("abcdefghi"));
}

// ── format_extension coverage ─────────────────────────────────────

#[test]
fn test_format_extension_all_formats() {
    assert_eq!(format_extension(Format::Gguf), "gguf");
    assert_eq!(format_extension(Format::Apr), "apr");
    assert_eq!(format_extension(Format::SafeTensors), "safetensors");
}

// ── resolve_file_by_format edge cases ─────────────────────────────

#[test]
fn test_resolve_file_by_format_no_extension() {
    let tmp = tempfile::TempDir::new().unwrap();
    let file = tmp.path().join("model");
    std::fs::write(&file, b"data").unwrap();

    // File with no extension should fail for any format
    let result = resolve_file_by_format(&file, Format::Gguf);
    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("extension mismatch")
    );
}

#[test]
fn test_resolve_file_by_format_correct_extension() {
    let tmp = tempfile::TempDir::new().unwrap();
    let file = tmp.path().join("model.apr");
    std::fs::write(&file, b"data").unwrap();

    let result = resolve_file_by_format(&file, Format::Apr);
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), file);
}

// ── classify_failure additional edge cases ────────────────────────

#[test]
fn test_classify_failure_segfault() {
    assert_eq!(
        classify_failure("segfault in model execution", 0),
        ConversionFailureType::InferenceFailure
    );
}

#[test]
fn test_classify_failure_sigsegv_in_stderr() {
    assert_eq!(
        classify_failure("SIGSEGV received", 0),
        ConversionFailureType::InferenceFailure
    );
}

#[test]
fn test_classify_failure_num_hidden_layers() {
    assert_eq!(
        classify_failure("num_hidden_layers mismatch: 24 vs 32", 1),
        ConversionFailureType::ConfigMetadataMismatch
    );
}

#[test]
fn test_classify_failure_name_mismatch_keyword() {
    assert_eq!(
        classify_failure("name mismatch between source and target", 1),
        ConversionFailureType::TensorNameMismatch
    );
}

#[test]
fn test_classify_failure_missing_without_mismatch() {
    // "missing" without "mismatch" should be MissingArtifact
    assert_eq!(
        classify_failure("missing file in model directory", 1),
        ConversionFailureType::MissingArtifact
    );
}

#[test]
fn test_classify_failure_tokenizer_without_mismatch() {
    // "tokenizer" without "mismatch" should be MissingArtifact
    assert_eq!(
        classify_failure("tokenizer not available", 1),
        ConversionFailureType::MissingArtifact
    );
}

#[test]
fn test_classify_failure_overflow_keyword() {
    assert_eq!(
        classify_failure("overflow detected in tensor values", 1),
        ConversionFailureType::DequantizationFailure
    );
}

#[test]
fn test_classify_failure_forward_pass() {
    assert_eq!(
        classify_failure("forward pass failed: shape error", 1),
        ConversionFailureType::InferenceFailure
    );
}

#[test]
fn test_classify_failure_exit_code_minus_11_empty_stderr() {
    assert_eq!(
        classify_failure("", -11),
        ConversionFailureType::InferenceFailure
    );
}

// ── tolerance_for remaining types ──────────────────────────────────

#[test]
fn test_tolerance_for_bf16() {
    let tol = tolerance_for(QuantType::BF16);
    assert!((tol.atol - 1e-2).abs() < 1e-10);
    assert!((tol.rtol - 1e-2).abs() < 1e-10);
}

#[test]
fn test_tolerance_for_q4_0() {
    let tol = tolerance_for(QuantType::Q4_0);
    assert!((tol.atol - 1e-1).abs() < 1e-10);
    assert!((tol.rtol - 1e-1).abs() < 1e-10);
}

#[test]
fn test_tolerance_for_q8_0() {
    let tol = tolerance_for(QuantType::Q8_0);
    assert!((tol.atol - 1e-2).abs() < 1e-10);
    assert!((tol.rtol - 1e-2).abs() < 1e-10);
}

// ── ConversionFailureType::key() all variants ─────────────────────

#[test]
fn test_conversion_failure_type_key_all_variants() {
    assert_eq!(
        ConversionFailureType::DequantizationFailure.key(),
        "dequantization_failure"
    );
    assert_eq!(
        ConversionFailureType::ConfigMetadataMismatch.key(),
        "config_metadata_mismatch"
    );
    assert_eq!(
        ConversionFailureType::MissingArtifact.key(),
        "missing_artifact"
    );
    assert_eq!(
        ConversionFailureType::InferenceFailure.key(),
        "inference_failure"
    );
}

// ── ConversionBugType::description() remaining variants ───────────

#[test]
fn test_bug_type_description_shape_mismatch() {
    let desc = ConversionBugType::ShapeMismatch.description();
    assert!(desc.contains("shape") || desc.contains("Shape"));
}

#[test]
fn test_bug_type_description_unknown() {
    let desc = ConversionBugType::Unknown.description();
    assert!(desc.contains("Unknown") || desc.contains("investigation"));
}

#[test]
fn test_bug_type_description_semantic_drift() {
    let desc = ConversionBugType::SemanticDrift.description();
    assert!(desc.contains("Semantic") || desc.contains("semantic"));
}

// ── Sharded SafeTensors resolution ────────────────────────────────

#[test]
fn test_resolve_model_path_sharded_safetensors_index() {
    let tmp = tempfile::TempDir::new().unwrap();
    let st_dir = tmp.path().join("safetensors");
    std::fs::create_dir_all(&st_dir).unwrap();
    let index_file = st_dir.join("model.safetensors.index.json");
    std::fs::write(&index_file, b"{}").unwrap();

    let result = resolve_model_path(tmp.path(), Format::SafeTensors);
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), index_file);
}

#[test]
fn test_resolve_model_path_sharded_not_for_gguf() {
    // Sharded index only applies to safetensors, not gguf
    let tmp = tempfile::TempDir::new().unwrap();
    let gguf_dir = tmp.path().join("gguf");
    std::fs::create_dir_all(&gguf_dir).unwrap();
    // Even if there's a .index.json, it shouldn't matter for gguf
    let result = resolve_model_path(tmp.path(), Format::Gguf);
    assert!(result.is_err());
}

// ── ByteLevelRoundTripTest constructor ─────────────────────────────

#[test]
fn test_byte_level_round_trip_test_new() {
    let bt = ByteLevelRoundTripTest::new(Backend::Cpu, ModelId::new("org", "model"));
    assert_eq!(bt.backend, Backend::Cpu);
    assert_eq!(bt.model_id.org, "org");
}

#[test]
fn test_byte_level_round_trip_test_clone() {
    let bt = ByteLevelRoundTripTest::new(Backend::Gpu, ModelId::new("t", "m"));
    let cloned = bt.clone();
    assert_eq!(cloned.backend, Backend::Gpu);
}

#[test]
fn test_byte_level_round_trip_test_debug() {
    let bt = ByteLevelRoundTripTest::new(Backend::Cpu, ModelId::new("t", "m"));
    let debug_str = format!("{bt:?}");
    assert!(debug_str.contains("ByteLevelRoundTripTest"));
}

// ── ConversionConfig fields ───────────────────────────────────────
