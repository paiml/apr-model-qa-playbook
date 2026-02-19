    use super::*;

    fn create_mock_apr(dir: &std::path::Path, script: &str) -> std::path::PathBuf {
        let path = dir.join("mock_apr");
        std::fs::write(&path, format!("#!/bin/bash\n{script}")).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        }
        path
    }

    #[test]
    fn test_tolerance_for_unknown_falls_back_to_f32() {
        let tol = tolerance_for(QuantType::Unknown);
        assert!((tol.atol - 1e-6).abs() < 1e-10);
    }

    #[test]
    fn test_effective_epsilon_without_quant() {
        let test = ConversionTest::new(
            Format::Gguf,
            Format::Apr,
            Backend::Cpu,
            ModelId::new("test", "model"),
        );
        assert!((test.effective_epsilon() - EPSILON).abs() < f64::EPSILON);
    }

    #[test]
    fn test_effective_epsilon_with_quant() {
        let mut test = ConversionTest::new(
            Format::Gguf,
            Format::Apr,
            Backend::Cpu,
            ModelId::new("test", "model"),
        );
        test.quant_type = Some(QuantType::Q4KM);
        assert!((test.effective_epsilon() - 1e-1).abs() < 1e-10);
    }

    #[test]
    fn test_effective_epsilon_f32_quant() {
        let mut test = ConversionTest::new(
            Format::Gguf,
            Format::Apr,
            Backend::Cpu,
            ModelId::new("test", "model"),
        );
        test.quant_type = Some(QuantType::F32);
        assert!((test.effective_epsilon() - 1e-6).abs() < 1e-10);
    }

    // ── ConversionFailureType / TensorNaming serde tests ───────────────

    #[test]
    fn test_conversion_failure_type_serde() {
        let types = [
            ConversionFailureType::TensorNameMismatch,
            ConversionFailureType::DequantizationFailure,
            ConversionFailureType::ConfigMetadataMismatch,
            ConversionFailureType::MissingArtifact,
            ConversionFailureType::InferenceFailure,
            ConversionFailureType::Unknown,
        ];
        for ft in types {
            let json = serde_json::to_string(&ft).unwrap();
            let parsed: ConversionFailureType = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed, ft);
        }
    }

    #[test]
    fn test_conversion_failure_type_gate_ids() {
        assert_eq!(
            ConversionFailureType::TensorNameMismatch.gate_id(),
            "F-CONV-TNAME-001"
        );
        assert_eq!(
            ConversionFailureType::DequantizationFailure.gate_id(),
            "F-CONV-DEQUANT-001"
        );
        assert_eq!(
            ConversionFailureType::ConfigMetadataMismatch.gate_id(),
            "F-CONV-CONFIG-001"
        );
        assert_eq!(
            ConversionFailureType::MissingArtifact.gate_id(),
            "F-CONV-MISSING-001"
        );
        assert_eq!(
            ConversionFailureType::InferenceFailure.gate_id(),
            "F-CONV-INFER-001"
        );
        assert_eq!(
            ConversionFailureType::Unknown.gate_id(),
            "F-CONV-UNKNOWN-002"
        );
    }

    #[test]
    fn test_conversion_failure_type_keys() {
        assert_eq!(
            ConversionFailureType::TensorNameMismatch.key(),
            "tensor_name_mismatch"
        );
        assert_eq!(ConversionFailureType::Unknown.key(), "unknown");
    }

    #[test]
    fn test_quant_type_serde() {
        let types = [
            QuantType::F32,
            QuantType::F16,
            QuantType::BF16,
            QuantType::Q4KM,
            QuantType::Q5KM,
            QuantType::Q6K,
            QuantType::Q4_0,
            QuantType::Q8_0,
            QuantType::Unknown,
        ];
        for qt in types {
            let json = serde_json::to_string(&qt).unwrap();
            let parsed: QuantType = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed, qt);
        }
    }

    #[test]
    fn test_tensor_naming_serde() {
        let variants = [
            TensorNaming::HuggingFace,
            TensorNaming::Gguf,
            TensorNaming::Apr,
            TensorNaming::Unknown("custom".to_string()),
        ];
        for tn in &variants {
            let json = serde_json::to_string(tn).unwrap();
            let parsed: TensorNaming = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed, *tn);
        }
    }

    #[test]
    fn test_conversion_evidence_with_failure_type() {
        let evidence = ConversionEvidence {
            source_hash: "a".to_string(),
            converted_hash: "b".to_string(),
            max_diff: 0.5,
            diff_indices: vec![],
            source_format: Format::Gguf,
            target_format: Format::Apr,
            backend: Backend::Cpu,
            failure_type: Some(ConversionFailureType::TensorNameMismatch),
            quant_type: Some(QuantType::Q4KM),
        };
        let json = serde_json::to_string(&evidence).unwrap();
        let parsed: ConversionEvidence = serde_json::from_str(&json).unwrap();
        assert_eq!(
            parsed.failure_type,
            Some(ConversionFailureType::TensorNameMismatch)
        );
        assert_eq!(parsed.quant_type, Some(QuantType::Q4KM));
    }

    #[test]
    fn test_conversion_evidence_default_optional_fields() {
        // Deserialize without optional fields — should default to None
        let json = r#"{
            "source_hash": "a",
            "converted_hash": "b",
            "max_diff": 0.1,
            "diff_indices": [],
            "source_format": "gguf",
            "target_format": "apr",
            "backend": "cpu"
        }"#;
        let parsed: ConversionEvidence = serde_json::from_str(json).unwrap();
        assert!(parsed.failure_type.is_none());
        assert!(parsed.quant_type.is_none());
    }

    #[test]
    fn test_default_tolerances_count() {
        assert_eq!(DEFAULT_TOLERANCES.len(), 8);
    }

    // =========================================================================
    // Model Path Resolution Tests
    // =========================================================================

    #[test]
    fn test_resolve_model_path_apr_cache_structure() {
        let tmp = tempfile::TempDir::new().unwrap();
        let safetensors_dir = tmp.path().join("safetensors");
        std::fs::create_dir_all(&safetensors_dir).unwrap();
        let model_file = safetensors_dir.join("model.safetensors");
        std::fs::write(&model_file, b"fake").unwrap();

        let result = resolve_model_path(tmp.path(), Format::SafeTensors);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), model_file);
    }

    #[test]
    fn test_resolve_model_path_hf_cache_flat_structure() {
        let tmp = tempfile::TempDir::new().unwrap();
        // HF cache has model.safetensors directly in snapshot dir (flat)
        let model_file = tmp.path().join("model.safetensors");
        std::fs::write(&model_file, b"fake").unwrap();

        let result = resolve_model_path(tmp.path(), Format::SafeTensors);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), model_file);
    }

    #[test]
    fn test_resolve_model_path_file_mode() {
        let tmp = tempfile::TempDir::new().unwrap();
        let model_file = tmp.path().join("model.gguf");
        std::fs::write(&model_file, b"fake").unwrap();

        let result = resolve_model_path(&model_file, Format::Gguf);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), model_file);
    }

    #[test]
    fn test_resolve_model_path_file_mode_extension_mismatch() {
        let tmp = tempfile::TempDir::new().unwrap();
        let model_file = tmp.path().join("model.gguf");
        std::fs::write(&model_file, b"fake").unwrap();

        let result = resolve_model_path(&model_file, Format::SafeTensors);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("extension mismatch")
        );
    }

    #[test]
    fn test_resolve_model_path_not_found() {
        let tmp = tempfile::TempDir::new().unwrap();
        let result = resolve_model_path(tmp.path(), Format::Apr);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("No apr file found")
        );
    }

    #[test]
    fn test_resolve_model_path_any_extension_in_subdir() {
        let tmp = tempfile::TempDir::new().unwrap();
        let gguf_dir = tmp.path().join("gguf");
        std::fs::create_dir_all(&gguf_dir).unwrap();
        // Not model.gguf but something.gguf
        let model_file = gguf_dir.join("qwen-0.5b-q4.gguf");
        std::fs::write(&model_file, b"fake").unwrap();

        let result = resolve_model_path(tmp.path(), Format::Gguf);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), model_file);
    }

    #[test]
    fn test_resolve_model_path_any_extension_in_base_dir() {
        let tmp = tempfile::TempDir::new().unwrap();
        // HF cache might have different names
        let model_file = tmp.path().join("qwen2.5-coder-0.5b.safetensors");
        std::fs::write(&model_file, b"fake").unwrap();

        let result = resolve_model_path(tmp.path(), Format::SafeTensors);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), model_file);
    }

    #[test]
    fn test_default_tolerances_all_quant_types() {
        let types: Vec<QuantType> = DEFAULT_TOLERANCES.iter().map(|t| t.quant_type).collect();
        assert!(types.contains(&QuantType::F32));
        assert!(types.contains(&QuantType::F16));
        assert!(types.contains(&QuantType::BF16));
        assert!(types.contains(&QuantType::Q4KM));
        assert!(types.contains(&QuantType::Q5KM));
        assert!(types.contains(&QuantType::Q6K));
        assert!(types.contains(&QuantType::Q4_0));
        assert!(types.contains(&QuantType::Q8_0));
    }

    // =========================================================================
    // HuggingFace Cache Resolution Tests (HF-CACHE-001, HF-CACHE-002)
    // =========================================================================

    #[test]
    fn test_split_hf_repo_with_org() {
        assert_eq!(
            split_hf_repo("Qwen/Qwen2.5-Coder-0.5B"),
            ("Qwen", "Qwen2.5-Coder-0.5B")
        );
        assert_eq!(
            split_hf_repo("meta-llama/Llama-2-7b"),
            ("meta-llama", "Llama-2-7b")
        );
    }

    #[test]
    fn test_split_hf_repo_without_org() {
        assert_eq!(split_hf_repo("model-only"), ("unknown", "model-only"));
        assert_eq!(split_hf_repo("gpt2"), ("unknown", "gpt2"));
    }

    #[test]
    fn test_split_hf_repo_multiple_slashes() {
        // Only splits on first slash
        assert_eq!(split_hf_repo("org/repo/extra"), ("org", "repo/extra"));
    }

    #[test]
    fn test_split_hf_repo_empty_string() {
        assert_eq!(split_hf_repo(""), ("unknown", ""));
    }

    #[test]
    fn test_find_hf_snapshot_found() {
        let tmp = tempfile::TempDir::new().unwrap();
        let snapshot = tmp
            .path()
            .join("models--Test--Model")
            .join("snapshots")
            .join("abc123");
        std::fs::create_dir_all(&snapshot).unwrap();
        std::fs::write(snapshot.join("model.safetensors"), b"fake").unwrap();

        let result = find_hf_snapshot(tmp.path(), "Test", "Model");
        assert!(result.is_some());
        assert_eq!(result.unwrap(), snapshot);
    }

    #[test]
    fn test_find_hf_snapshot_not_found_no_dir() {
        let tmp = tempfile::TempDir::new().unwrap();
        let result = find_hf_snapshot(tmp.path(), "Missing", "Model");
        assert!(result.is_none());
    }

    #[test]
    fn test_find_hf_snapshot_not_found_no_safetensors() {
        let tmp = tempfile::TempDir::new().unwrap();
        let snapshot = tmp
            .path()
            .join("models--Test--NoFile")
            .join("snapshots")
            .join("abc123");
        std::fs::create_dir_all(&snapshot).unwrap();
        // No model.safetensors file

        let result = find_hf_snapshot(tmp.path(), "Test", "NoFile");
        assert!(result.is_none());
    }

    #[test]
    fn test_find_apr_cache_found() {
        let tmp = tempfile::TempDir::new().unwrap();
        let apr_cache = tmp.path().join(".cache/apr-models/TestOrg/TestRepo");
        std::fs::create_dir_all(&apr_cache).unwrap();

        let result = find_apr_cache(tmp.path(), "TestOrg", "TestRepo");
        assert!(result.is_some());
        assert_eq!(result.unwrap(), apr_cache);
    }

    #[test]
    fn test_find_apr_cache_not_found() {
        let tmp = tempfile::TempDir::new().unwrap();
        let result = find_apr_cache(tmp.path(), "Missing", "Model");
        assert!(result.is_none());
    }

    #[test]
    fn test_resolve_hf_repo_with_dirs_found_in_hf_cache() {
        let tmp = tempfile::TempDir::new().unwrap();
        let snapshot = tmp
            .path()
            .join("models--Test--Model")
            .join("snapshots")
            .join("abc123");
        std::fs::create_dir_all(&snapshot).unwrap();
        std::fs::write(snapshot.join("model.safetensors"), b"fake").unwrap();

        // Use the temp dir as both HF cache and home
        let result = resolve_hf_repo_with_dirs("Test/Model", tmp.path(), tmp.path());
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), snapshot);
    }

    #[test]
    fn test_resolve_hf_repo_with_dirs_found_in_apr_cache() {
        let tmp = tempfile::TempDir::new().unwrap();
        let apr_cache = tmp.path().join(".cache/apr-models/TestOrg/TestRepo");
        std::fs::create_dir_all(&apr_cache).unwrap();

        // HF cache is empty, APR cache has the model
        let hf_cache = tmp.path().join("hf_empty");
        std::fs::create_dir_all(&hf_cache).unwrap();

        let result = resolve_hf_repo_with_dirs("TestOrg/TestRepo", &hf_cache, tmp.path());
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), apr_cache);
    }

    #[test]
    fn test_resolve_hf_repo_with_dirs_not_found() {
        let tmp = tempfile::TempDir::new().unwrap();
        let hf_cache = tmp.path().join("hf_empty");
        let home = tmp.path().join("home_empty");
        std::fs::create_dir_all(&hf_cache).unwrap();
        std::fs::create_dir_all(&home).unwrap();

        let result = resolve_hf_repo_with_dirs("Missing/Model", &hf_cache, &home);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("not found in cache"));
        assert!(err_msg.contains("Missing/Model"));
    }

    #[test]
    fn test_resolve_hf_repo_with_dirs_snapshot_without_safetensors() {
        let tmp = tempfile::TempDir::new().unwrap();
        let snapshot = tmp
            .path()
            .join("models--Test--NoSafetensors")
            .join("snapshots")
            .join("abc123");
        std::fs::create_dir_all(&snapshot).unwrap();
        // No model.safetensors

        let home = tmp.path().join("home_empty");
        std::fs::create_dir_all(&home).unwrap();

        let result = resolve_hf_repo_with_dirs("Test/NoSafetensors", tmp.path(), &home);
        assert!(result.is_err());
    }

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
        let executor = ConversionExecutor::with_defaults()
            .with_output_dir(std::path::PathBuf::from("/tmp/out"));
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

    #[test]
    fn test_conversion_config_all_disabled() {
        let config = ConversionConfig {
            test_all_pairs: false,
            test_round_trips: false,
            test_multi_hop: false,
            test_cardinality: false,
            test_tensor_names: false,
            test_idempotency: false,
            test_commutativity: false,
            backends: vec![Backend::Cpu],
            no_gpu: true,
        };
        let executor = ConversionExecutor::new(config);
        let model_id = ModelId::new("test", "model");
        let tmp = tempfile::TempDir::new().unwrap();
        let model_file = tmp.path().join("model.gguf");
        std::fs::write(&model_file, b"fake").unwrap();

        // With everything disabled, should still return Ok but with no results
        let result = executor.execute_all(&model_file, &model_id);
        assert!(result.is_ok());
        let exec_result = result.unwrap();
        assert_eq!(exec_result.total, 0);
        assert_eq!(exec_result.passed, 0);
        assert_eq!(exec_result.failed, 0);
        assert!(exec_result.all_passed());
        assert!((exec_result.pass_rate() - 100.0).abs() < f64::EPSILON);
    }

    // ── ConversionExecutor with only structural checks ────────────────

    #[test]
    fn test_conversion_executor_only_cardinality_no_converted_files() {
        let tmp = tempfile::TempDir::new().unwrap();
        let model_file = tmp.path().join("model.gguf");
        std::fs::write(&model_file, b"fake").unwrap();

        let config = ConversionConfig {
            test_all_pairs: false,
            test_round_trips: false,
            test_multi_hop: false,
            test_cardinality: true,
            test_tensor_names: true,
            test_idempotency: false,
            test_commutativity: false,
            backends: vec![Backend::Cpu],
            no_gpu: true,
        };
        let executor = ConversionExecutor::new(config);
        let model_id = ModelId::new("test", "model");

        // Structural checks skip when no converted files exist
        let result = executor.execute_all(&model_file, &model_id);
        assert!(result.is_ok());
        let exec_result = result.unwrap();
        // No converted files means structural checks are skipped
        assert_eq!(exec_result.total, 0);
    }

    // ── QuantType debug/clone/copy ────────────────────────────────────

    #[test]
    fn test_quant_type_debug() {
        let debug_str = format!("{:?}", QuantType::Q6K);
        assert!(debug_str.contains("Q6K"));
    }

    #[test]
    fn test_quant_type_copy() {
        let qt = QuantType::BF16;
        let copied = qt;
        assert_eq!(qt, copied);
    }

    // ── ConversionFailureType debug/copy ──────────────────────────────

    #[test]
    fn test_conversion_failure_type_debug() {
        let debug_str = format!("{:?}", ConversionFailureType::DequantizationFailure);
        assert!(debug_str.contains("DequantizationFailure"));
    }

    #[test]
    fn test_conversion_failure_type_copy() {
        let ft = ConversionFailureType::MissingArtifact;
        let copied = ft;
        assert_eq!(ft, copied);
    }

    // ── TensorNaming additional variants ──────────────────────────────

    #[test]
    fn test_tensor_naming_debug() {
        let debug_str = format!("{:?}", TensorNaming::Apr);
        assert!(debug_str.contains("Apr"));
    }

    #[test]
    fn test_tensor_naming_unknown_variant() {
        let naming = TensorNaming::Unknown("custom_conv".to_string());
        let debug_str = format!("{naming:?}");
        assert!(debug_str.contains("custom_conv"));
    }

    #[test]
    fn test_tensor_naming_equality() {
        assert_eq!(TensorNaming::HuggingFace, TensorNaming::HuggingFace);
        assert_ne!(TensorNaming::HuggingFace, TensorNaming::Gguf);
        assert_ne!(
            TensorNaming::Unknown("a".to_string()),
            TensorNaming::Unknown("b".to_string())
        );
    }

    // ── ConversionTest::convert_model with output_dir (ISO-OUT-001) ───

    #[test]
    fn test_conversion_test_execute_with_output_dir_via_mock() {
        let dir = tempfile::tempdir().unwrap();
        let model_file = dir.path().join("model.gguf");
        std::fs::write(&model_file, "fake").unwrap();

        let mock = create_mock_apr(
            dir.path(),
            r#"case "$1" in
run) printf "The answer is 4"; exit 0;;
rosetta) touch "$4"; exit 0;;
esac
exit 1"#,
        );

        let model_id = ModelId::new("test", "model");
        let out = ConversionOutputDir::new(dir.path(), &model_id);
        let mut test = ConversionTest::new(Format::Gguf, Format::Apr, Backend::Cpu, model_id)
            .with_output_dir(out);
        test.binary = mock.to_string_lossy().to_string();

        // This exercises the ISO-OUT-001 output_dir branch in convert_model
        if let Ok(conv) = test.execute(&model_file) {
            match conv {
                ConversionResult::Corroborated { .. } | ConversionResult::Falsified { .. } => {}
            }
        }
    }

    // ── Idempotency and Commutativity executor error paths ────────────

    #[test]
    fn test_conversion_executor_idempotency_error_via_mock() {
        let dir = tempfile::tempdir().unwrap();
        let model_file = dir.path().join("model.gguf");
        std::fs::write(&model_file, "fake").unwrap();

        let mock = create_mock_apr(dir.path(), r"exit 1");

        let config = ConversionConfig {
            test_all_pairs: false,
            test_round_trips: false,
            test_multi_hop: false,
            test_cardinality: false,
            test_tensor_names: false,
            test_idempotency: true,
            test_commutativity: false,
            backends: vec![Backend::Cpu],
            no_gpu: true,
        };
        let mut executor = ConversionExecutor::new(config);
        executor.binary = mock.to_string_lossy().to_string();
        let model_id = ModelId::new("test", "model");

        if let Ok(exec_result) = executor.execute_all(&model_file, &model_id) {
            // Idempotency error gets recorded as evidence
            assert!(!exec_result.evidence.is_empty());
        }
    }

    #[test]
    fn test_conversion_executor_commutativity_error_via_mock() {
        let dir = tempfile::tempdir().unwrap();
        let model_file = dir.path().join("model.gguf");
        std::fs::write(&model_file, "fake").unwrap();

        let mock = create_mock_apr(dir.path(), r"exit 1");

        let config = ConversionConfig {
            test_all_pairs: false,
            test_round_trips: false,
            test_multi_hop: false,
            test_cardinality: false,
            test_tensor_names: false,
            test_idempotency: false,
            test_commutativity: true,
            backends: vec![Backend::Cpu],
            no_gpu: true,
        };
        let mut executor = ConversionExecutor::new(config);
        executor.binary = mock.to_string_lossy().to_string();
        let model_id = ModelId::new("test", "model");

        if let Ok(exec_result) = executor.execute_all(&model_file, &model_id) {
            assert!(!exec_result.evidence.is_empty());
        }
    }

    #[test]
    fn test_conversion_executor_multi_hop_error_via_mock() {
        let dir = tempfile::tempdir().unwrap();
        let model_file = dir.path().join("model.gguf");
        std::fs::write(&model_file, "fake").unwrap();

        let mock = create_mock_apr(dir.path(), r"exit 1");

        let config = ConversionConfig {
            test_all_pairs: false,
            test_round_trips: false,
            test_multi_hop: true,
            test_cardinality: false,
            test_tensor_names: false,
            test_idempotency: false,
            test_commutativity: false,
            backends: vec![Backend::Cpu],
            no_gpu: true,
        };
        let mut executor = ConversionExecutor::new(config);
        executor.binary = mock.to_string_lossy().to_string();
        let model_id = ModelId::new("test", "model");

        if let Ok(exec_result) = executor.execute_all(&model_file, &model_id) {
            // Multi-hop and byte-level errors recorded as evidence
            assert!(!exec_result.evidence.is_empty());
        }
    }

    // ── ConversionEvidence with all optional fields populated ─────────

    #[test]
    fn test_conversion_evidence_full_serde_round_trip() {
        let evidence = ConversionEvidence {
            source_hash: "src_hash".to_string(),
            converted_hash: "conv_hash".to_string(),
            max_diff: 0.42,
            diff_indices: vec![0, 3, 7],
            source_format: Format::SafeTensors,
            target_format: Format::Gguf,
            backend: Backend::Gpu,
            failure_type: Some(ConversionFailureType::DequantizationFailure),
            quant_type: Some(QuantType::Q6K),
        };
        let json = serde_json::to_string(&evidence).unwrap();
        let parsed: ConversionEvidence = serde_json::from_str(&json).unwrap();
        assert_eq!(
            parsed.failure_type,
            Some(ConversionFailureType::DequantizationFailure)
        );
        assert_eq!(parsed.quant_type, Some(QuantType::Q6K));
        assert_eq!(parsed.diff_indices, vec![0, 3, 7]);
        assert_eq!(parsed.backend, Backend::Gpu);
    }

    // ── ConversionTolerance fields ────────────────────────────────────

    #[test]
    fn test_default_tolerances_all_have_positive_atol_rtol() {
        for tol in DEFAULT_TOLERANCES {
            assert!(tol.atol > 0.0, "{:?} atol should be > 0", tol.quant_type);
            assert!(tol.rtol > 0.0, "{:?} rtol should be > 0", tol.quant_type);
        }
    }

    #[test]
    fn test_default_tolerances_stricter_for_higher_precision() {
        let f32_tol = tolerance_for(QuantType::F32);
        let f16_tol = tolerance_for(QuantType::F16);
        let q4km_tol = tolerance_for(QuantType::Q4KM);

        // Higher precision should have stricter (smaller) tolerances
        assert!(f32_tol.atol < f16_tol.atol);
        assert!(f16_tol.atol < q4km_tol.atol);
    }

    // ── convert_to_format (non-tagged) error path ─────────────────────

    #[test]
    fn test_convert_to_format_error_nonexistent_binary() {
        let result = convert_to_format(
            &std::path::PathBuf::from("/nonexistent/model.gguf"),
            Format::Apr,
            "/nonexistent/apr",
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_convert_to_format_safetensors_target() {
        let result = convert_to_format(
            &std::path::PathBuf::from("/nonexistent/model.apr"),
            Format::SafeTensors,
            "/nonexistent/apr",
        );
        assert!(result.is_err());
    }

    // ── run_diff_tensors error path ───────────────────────────────────

    #[test]
    fn test_run_diff_tensors_nonexistent_binary() {
        let result = run_diff_tensors(
            &std::path::PathBuf::from("/a.apr"),
            &std::path::PathBuf::from("/b.apr"),
            "/nonexistent/apr",
        );
        assert!(result.is_err());
    }

    // ── run_inference_simple CPU flag ──────────────────────────────────

    #[test]
    fn test_run_inference_simple_cpu_flag() {
        let result = run_inference_simple(
            &std::path::PathBuf::from("/nonexistent/model.gguf"),
            Backend::Cpu,
            "/nonexistent/apr",
        );
        assert!(result.is_err());
    }
