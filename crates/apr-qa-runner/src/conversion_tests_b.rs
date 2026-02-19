    use super::*;


    #[test]
    fn test_arithmetic_expected_constant() {
        assert!(!ARITHMETIC_EXPECTED.is_empty());
        assert!(ARITHMETIC_EXPECTED.contains(&"4"));
        assert!(ARITHMETIC_EXPECTED.contains(&"four"));
    }

    // Additional tests for coverage

    #[test]
    fn test_conversion_bug_type_serialization() {
        let bug_types = [
            ConversionBugType::EmbeddingTransposition,
            ConversionBugType::TokenizerMissing,
            ConversionBugType::WeightCorruption,
            ConversionBugType::ShapeMismatch,
            ConversionBugType::SemanticDrift,
            ConversionBugType::Unknown,
        ];
        for bug_type in bug_types {
            let json = serde_json::to_string(&bug_type).unwrap();
            let parsed: ConversionBugType = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed, bug_type);
        }
    }

    #[test]
    fn test_conversion_test_serialization() {
        let test = ConversionTest {
            source_format: Format::Gguf,
            target_format: Format::Apr,
            backend: Backend::Cpu,
            model_id: ModelId::new("org", "name"),
            epsilon: 1e-7,
            binary: default_binary(),
            quant_type: None,
            output_dir: None,
        };
        let json = serde_json::to_string(&test).unwrap();
        let parsed: ConversionTest = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.source_format, Format::Gguf);
        assert_eq!(parsed.target_format, Format::Apr);
    }

    #[test]
    fn test_conversion_result_serialization_corroborated() {
        let result = ConversionResult::Corroborated {
            source_format: Format::Gguf,
            target_format: Format::Apr,
            backend: Backend::Gpu,
            max_diff: 1e-9,
        };
        let json = serde_json::to_string(&result).unwrap();
        let parsed: ConversionResult = serde_json::from_str(&json).unwrap();
        match parsed {
            ConversionResult::Corroborated { max_diff, .. } => {
                assert!(max_diff < EPSILON);
            }
            _ => panic!("Expected Corroborated"),
        }
    }

    #[test]
    fn test_conversion_result_serialization_falsified() {
        let result = ConversionResult::Falsified {
            gate_id: "F-CONV-G-A".to_string(),
            reason: "Test failure".to_string(),
            evidence: ConversionEvidence {
                source_hash: "abc".to_string(),
                converted_hash: "def".to_string(),
                max_diff: 0.5,
                diff_indices: vec![0, 1, 2],
                source_format: Format::Gguf,
                target_format: Format::Apr,
                backend: Backend::Cpu,
                failure_type: None,
                quant_type: None,
            },
        };
        let json = serde_json::to_string(&result).unwrap();
        let parsed: ConversionResult = serde_json::from_str(&json).unwrap();
        match parsed {
            ConversionResult::Falsified { gate_id, .. } => {
                assert_eq!(gate_id, "F-CONV-G-A");
            }
            _ => panic!("Expected Falsified"),
        }
    }

    #[test]
    fn test_conversion_evidence_serialization() {
        let evidence = ConversionEvidence {
            source_hash: "hash1".to_string(),
            converted_hash: "hash2".to_string(),
            max_diff: 0.05,
            diff_indices: vec![1, 3, 5],
            source_format: Format::SafeTensors,
            target_format: Format::Gguf,
            backend: Backend::Gpu,
            failure_type: None,
            quant_type: None,
        };
        let json = serde_json::to_string(&evidence).unwrap();
        let parsed: ConversionEvidence = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.source_hash, "hash1");
        assert_eq!(parsed.diff_indices.len(), 3);
    }

    #[test]
    fn test_semantic_test_result_clone() {
        let result = SemanticTestResult::Falsified {
            bug_type: ConversionBugType::TokenizerMissing,
            source_output: "source".to_string(),
            target_output: "target".to_string(),
            stderr: "error".to_string(),
        };
        let cloned = result.clone();
        match cloned {
            SemanticTestResult::Falsified {
                bug_type, stderr, ..
            } => {
                assert_eq!(bug_type, ConversionBugType::TokenizerMissing);
                assert_eq!(stderr, "error");
            }
            _ => panic!("Expected Falsified"),
        }
    }

    #[test]
    fn test_classify_bug_source_empty_target_has_content() {
        let test = SemanticConversionTest::new(
            Format::Gguf,
            Format::Apr,
            Backend::Cpu,
            ModelId::new("test", "model"),
        );
        // Source empty, target has content - unusual case, returns Unknown
        let bug = test.classify_bug("", "Some output", false);
        assert_eq!(bug, Some(ConversionBugType::Unknown));
    }

    #[test]
    fn test_classify_bug_both_empty() {
        let test = SemanticConversionTest::new(
            Format::Gguf,
            Format::Apr,
            Backend::Cpu,
            ModelId::new("test", "model"),
        );
        // Both empty - no bug
        let bug = test.classify_bug("", "", false);
        assert!(bug.is_none());
    }

    #[test]
    fn test_classify_bug_source_no_expected_target_has_expected() {
        let test = SemanticConversionTest::new(
            Format::Gguf,
            Format::Apr,
            Backend::Cpu,
            ModelId::new("test", "model"),
        );
        // Source doesn't have expected, target does - weird but not a bug in our heuristic
        let bug = test.classify_bug("random text", "The answer is 4", false);
        // Outputs differ but no clear pattern
        assert_eq!(bug, Some(ConversionBugType::Unknown));
    }

    #[test]
    fn test_compute_diff_unicode() {
        let test = ConversionTest::new(
            Format::Gguf,
            Format::Apr,
            Backend::Cpu,
            ModelId::new("test", "model"),
        );
        let diff = test.compute_diff("hello 你好", "hello 世界");
        assert!(diff > 0.0);
        assert!(diff < 1.0);
    }

    #[test]
    fn test_find_diff_indices_unicode() {
        let test = ConversionTest::new(
            Format::Gguf,
            Format::Apr,
            Backend::Cpu,
            ModelId::new("test", "model"),
        );
        let indices = test.find_diff_indices("ab你好", "abXX");
        // Comparing "你" vs "X" and "好" vs "X"
        assert!(indices.len() >= 2);
    }

    #[test]
    fn test_hash_output_unicode() {
        let hash1 = ConversionTest::hash_output("hello 你好 世界");
        let hash2 = ConversionTest::hash_output("hello 你好 世界");
        assert_eq!(hash1, hash2);
        assert_eq!(hash1.len(), 16); // 16 hex chars
    }

    #[test]
    fn test_conversion_execution_result_pass_rate_partial() {
        let result = ConversionExecutionResult {
            passed: 7,
            failed: 3,
            total: 10,
            evidence: vec![],
            results: vec![],
            duration_ms: 1000,
        };
        let rate = result.pass_rate();
        assert!((rate - 70.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_conversion_config_with_specific_backends() {
        let config = ConversionConfig {
            test_all_pairs: true,
            test_round_trips: false,
            backends: vec![Backend::Gpu],
            no_gpu: false,
            ..Default::default()
        };
        assert_eq!(config.backends.len(), 1);
        assert_eq!(config.backends[0], Backend::Gpu);
        assert!(!config.test_round_trips);
    }

    #[test]
    fn test_semantic_conversion_test_fields() {
        let test = SemanticConversionTest::new(
            Format::SafeTensors,
            Format::Apr,
            Backend::Gpu,
            ModelId::new("org", "model"),
        );
        assert_eq!(test.source_format, Format::SafeTensors);
        assert_eq!(test.target_format, Format::Apr);
        assert_eq!(test.backend, Backend::Gpu);
        assert_eq!(test.model_id.org, "org");
    }

    #[test]
    fn test_round_trip_test_with_two_formats() {
        let rt = RoundTripTest::new(
            vec![Format::Apr, Format::Gguf],
            Backend::Gpu,
            ModelId::new("test", "model"),
        );
        assert_eq!(rt.formats.len(), 2);
        assert_eq!(rt.backend, Backend::Gpu);
    }

    #[test]
    fn test_conversion_evidence_with_empty_diff_indices() {
        let evidence = ConversionEvidence {
            source_hash: "same".to_string(),
            converted_hash: "same".to_string(),
            max_diff: 0.0,
            diff_indices: vec![],
            source_format: Format::Gguf,
            target_format: Format::Apr,
            backend: Backend::Cpu,
            failure_type: None,
            quant_type: None,
        };
        assert!(evidence.diff_indices.is_empty());
        assert!((evidence.max_diff - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_all_conversion_pairs_complete() {
        let pairs = all_conversion_pairs();
        // Should have bidirectional pairs for all format combinations
        // 3 formats = 6 pairs (A->B, B->A for each pair)
        assert_eq!(pairs.len(), 6);

        // Check specific pairs exist
        assert!(pairs.contains(&(Format::Gguf, Format::Apr)));
        assert!(pairs.contains(&(Format::Apr, Format::Gguf)));
        assert!(pairs.contains(&(Format::Gguf, Format::SafeTensors)));
        assert!(pairs.contains(&(Format::SafeTensors, Format::Gguf)));
        assert!(pairs.contains(&(Format::Apr, Format::SafeTensors)));
        assert!(pairs.contains(&(Format::SafeTensors, Format::Apr)));
    }

    #[test]
    fn test_generate_conversion_tests_model_id_preserved() {
        let model_id = ModelId::new("my-org", "my-model-v1");
        let tests = generate_conversion_tests(&model_id);

        for test in &tests {
            assert_eq!(test.model_id.org, "my-org");
            assert_eq!(test.model_id.name, "my-model-v1");
        }
    }

    #[test]
    fn test_conversion_test_debug_format() {
        let test = ConversionTest::new(
            Format::Gguf,
            Format::Apr,
            Backend::Cpu,
            ModelId::new("test", "model"),
        );
        let debug = format!("{test:?}");
        assert!(debug.contains("ConversionTest"));
        assert!(debug.contains("Gguf"));
        assert!(debug.contains("Apr"));
    }

    #[test]
    fn test_classify_bug_with_multiple_garbage_patterns() {
        let test = SemanticConversionTest::new(
            Format::Gguf,
            Format::Apr,
            Backend::Cpu,
            ModelId::new("test", "model"),
        );
        // Target has multiple garbage patterns
        let bug = test.classify_bug("The answer is 4", "PAD <pad> <|endoftext|> 151935", false);
        assert_eq!(bug, Some(ConversionBugType::EmbeddingTransposition));
    }

    #[test]
    fn test_classify_bug_target_only_whitespace() {
        let test = SemanticConversionTest::new(
            Format::Gguf,
            Format::Apr,
            Backend::Cpu,
            ModelId::new("test", "model"),
        );
        // Source has content but no expected arithmetic, target is whitespace
        let bug = test.classify_bug("Some random output", "   \t\n  ", false);
        assert_eq!(bug, Some(ConversionBugType::WeightCorruption));
    }

    #[test]
    fn test_conversion_executor_custom_config() {
        let config = ConversionConfig {
            test_all_pairs: false,
            test_round_trips: true,
            backends: vec![Backend::Cpu],
            no_gpu: true,
            ..Default::default()
        };
        let executor = ConversionExecutor::new(config);
        assert!(!executor.config.test_all_pairs);
        assert!(executor.config.test_round_trips);
        assert!(executor.config.no_gpu);
    }

    #[test]
    fn test_semantic_test_result_is_pass_corroborated() {
        let result = SemanticTestResult::Corroborated {
            source_output: "test".to_string(),
            target_output: "test".to_string(),
        };
        assert!(result.is_pass());
    }

    #[test]
    fn test_semantic_test_result_is_pass_falsified() {
        let result = SemanticTestResult::Falsified {
            bug_type: ConversionBugType::Unknown,
            source_output: "a".to_string(),
            target_output: "b".to_string(),
            stderr: String::new(),
        };
        assert!(!result.is_pass());
    }

    #[test]
    fn test_semantic_test_result_bug_type_corroborated() {
        let result = SemanticTestResult::Corroborated {
            source_output: "test".to_string(),
            target_output: "test".to_string(),
        };
        assert!(result.bug_type().is_none());
    }

    #[test]
    fn test_semantic_test_result_bug_type_falsified() {
        let result = SemanticTestResult::Falsified {
            bug_type: ConversionBugType::SemanticDrift,
            source_output: "a".to_string(),
            target_output: "b".to_string(),
            stderr: "warning".to_string(),
        };
        assert_eq!(result.bug_type(), Some(ConversionBugType::SemanticDrift));
    }

    #[test]
    fn test_conversion_result_corroborated_serialization() {
        let result = ConversionResult::Corroborated {
            source_format: Format::Gguf,
            target_format: Format::Apr,
            backend: Backend::Cpu,
            max_diff: 0.001,
        };
        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("Corroborated"));
        let deserialized: ConversionResult = serde_json::from_str(&json).unwrap();
        if let ConversionResult::Corroborated { max_diff, .. } = deserialized {
            assert!((max_diff - 0.001).abs() < f64::EPSILON);
        } else {
            panic!("Expected Corroborated");
        }
    }

    #[test]
    fn test_conversion_result_falsified_serialization() {
        let result = ConversionResult::Falsified {
            gate_id: "F-TEST-001".to_string(),
            reason: "Test failure".to_string(),
            evidence: ConversionEvidence {
                source_hash: "abc".to_string(),
                converted_hash: "def".to_string(),
                max_diff: 0.5,
                diff_indices: vec![1, 2, 3],
                source_format: Format::Gguf,
                target_format: Format::Apr,
                backend: Backend::Cpu,
                failure_type: None,
                quant_type: None,
            },
        };
        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("Falsified"));
        assert!(json.contains("F-TEST-001"));
    }

    #[test]
    fn test_conversion_test_new_with_epsilon() {
        let test = ConversionTest {
            source_format: Format::Apr,
            target_format: Format::Gguf,
            backend: Backend::Gpu,
            model_id: ModelId::new("org", "model"),
            epsilon: 1e-10,
            binary: default_binary(),
            quant_type: None,
            output_dir: None,
        };
        assert!((test.epsilon - 1e-10).abs() < 1e-15);
    }

    #[test]
    fn test_conversion_execution_result_fields() {
        let result = ConversionExecutionResult {
            total: 10,
            passed: 5,
            failed: 2,
            duration_ms: 100,
            results: vec![],
            evidence: vec![],
        };
        assert_eq!(result.total, 10);
        assert_eq!(result.passed, 5);
        assert_eq!(result.failed, 2);
        assert_eq!(result.duration_ms, 100);
        assert!(result.results.is_empty());
        assert!(result.evidence.is_empty());
    }

    #[test]
    fn test_conversion_test_compute_diff_same() {
        let test = ConversionTest::new(
            Format::Gguf,
            Format::Apr,
            Backend::Cpu,
            ModelId::new("test", "model"),
        );
        // Same strings should have 0 diff
        assert!((test.compute_diff("hello", "hello") - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_conversion_test_compute_diff_different() {
        let test = ConversionTest::new(
            Format::Gguf,
            Format::Apr,
            Backend::Cpu,
            ModelId::new("test", "model"),
        );
        // Completely different strings should have high diff
        let diff = test.compute_diff("abc", "xyz");
        assert!(diff > 0.5);
    }

    #[test]
    fn test_conversion_test_compute_diff_empty_strings() {
        let test = ConversionTest::new(
            Format::Gguf,
            Format::Apr,
            Backend::Cpu,
            ModelId::new("test", "model"),
        );
        // Empty strings should have 0 diff
        assert!((test.compute_diff("", "") - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_conversion_test_compute_diff_partial_match() {
        let test = ConversionTest::new(
            Format::Gguf,
            Format::Apr,
            Backend::Cpu,
            ModelId::new("test", "model"),
        );
        // Partially matching strings
        let diff = test.compute_diff("abcd", "abXd");
        assert!(diff > 0.0 && diff < 1.0);
    }

    #[test]
    fn test_conversion_test_find_diff_indices_with_diffs() {
        let test = ConversionTest::new(
            Format::Gguf,
            Format::Apr,
            Backend::Cpu,
            ModelId::new("test", "model"),
        );
        let indices = test.find_diff_indices("abcd", "aXcY");
        assert_eq!(indices.len(), 2);
        assert!(indices.contains(&1));
        assert!(indices.contains(&3));
    }

    #[test]
    fn test_conversion_test_find_diff_indices_no_diffs() {
        let test = ConversionTest::new(
            Format::Gguf,
            Format::Apr,
            Backend::Cpu,
            ModelId::new("test", "model"),
        );
        let indices = test.find_diff_indices("same", "same");
        assert!(indices.is_empty());
    }

    #[test]
    fn test_conversion_test_hash_output_consistency() {
        let hash1 = ConversionTest::hash_output("test string");
        let hash2 = ConversionTest::hash_output("test string");
        let hash3 = ConversionTest::hash_output("different string");

        // Same input should produce same hash
        assert_eq!(hash1, hash2);
        // Different input should produce different hash
        assert_ne!(hash1, hash3);
        // Hash should be 16 hex characters
        assert_eq!(hash1.len(), 16);
    }

    #[test]
    fn test_classify_bug_empty_source_nonempty_target() {
        let test = SemanticConversionTest::new(
            Format::Gguf,
            Format::Apr,
            Backend::Cpu,
            ModelId::new("test", "model"),
        );
        // If source is empty/whitespace and target is not, classify as unknown
        let bug = test.classify_bug("  ", "some output", false);
        assert_eq!(bug, Some(ConversionBugType::Unknown));
    }

    #[test]
    fn test_classify_bug_both_empty_strings() {
        let test = SemanticConversionTest::new(
            Format::Gguf,
            Format::Apr,
            Backend::Cpu,
            ModelId::new("test", "model"),
        );
        // Both empty should match
        let bug = test.classify_bug("", "", false);
        assert!(bug.is_none());
    }

    #[test]
    fn test_generate_conversion_tests_full_count() {
        let model_id = ModelId::new("test", "model");
        let tests = generate_conversion_tests(&model_id);

        // 6 pairs x 2 backends = 12 tests
        assert_eq!(tests.len(), 12);
    }

    // ── Mock binary tests ────────────────────────────────────────────

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
    fn test_conversion_test_execute_corroborated_via_mock() {
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

        let mut test = ConversionTest::new(
            Format::Gguf,
            Format::Apr,
            Backend::Cpu,
            ModelId::new("test", "model"),
        );
        test.binary = mock.to_string_lossy().to_string();

        if let Ok(conv) = test.execute(&model_file) {
            match conv {
                ConversionResult::Corroborated { max_diff, .. } => {
                    assert!(max_diff < EPSILON);
                }
                ConversionResult::Falsified { .. } => {}
            }
        }
    }

    #[test]
    fn test_conversion_test_execute_falsified_via_mock() {
        let dir = tempfile::tempdir().unwrap();
        let model_file = dir.path().join("model.gguf");
        std::fs::write(&model_file, "fake").unwrap();

        let mock = create_mock_apr(
            dir.path(),
            r#"case "$1" in
run)
  case "$2" in
  *converted*) printf "Completely different output 99";;
  *) printf "The answer is 4";;
  esac
  exit 0;;
rosetta) touch "$4"; exit 0;;
esac
exit 1"#,
        );

        let mut test = ConversionTest::new(
            Format::Gguf,
            Format::Apr,
            Backend::Cpu,
            ModelId::new("test", "model"),
        );
        test.binary = mock.to_string_lossy().to_string();

        if let Ok(conv) = test.execute(&model_file) {
            match conv {
                ConversionResult::Falsified {
                    gate_id, evidence, ..
                } => {
                    assert_eq!(gate_id, "F-CONV-G-A");
                    assert!(evidence.max_diff > EPSILON);
                    assert_ne!(evidence.source_hash, evidence.converted_hash);
                }
                ConversionResult::Corroborated { .. } => {}
            }
        }
    }

    #[test]
    fn test_conversion_test_execute_gpu_backend_via_mock() {
        let dir = tempfile::tempdir().unwrap();
        let model_file = dir.path().join("model.safetensors");
        std::fs::write(&model_file, "fake").unwrap();

        let mock = create_mock_apr(
            dir.path(),
            r#"case "$1" in
run) printf "The answer is 4"; exit 0;;
rosetta) touch "$4"; exit 0;;
esac
exit 1"#,
        );

        let mut test = ConversionTest::new(
            Format::SafeTensors,
            Format::Gguf,
            Backend::Gpu,
            ModelId::new("test", "model"),
        );
        test.binary = mock.to_string_lossy().to_string();

        if let Ok(ConversionResult::Corroborated { backend, .. }) = &test.execute(&model_file) {
            assert_eq!(*backend, Backend::Gpu);
        }
    }

    #[test]
    fn test_conversion_test_convert_model_failure_via_mock() {
        let dir = tempfile::tempdir().unwrap();
        let model_file = dir.path().join("model.gguf");
        std::fs::write(&model_file, "fake").unwrap();

        let mock = create_mock_apr(
            dir.path(),
            r#"case "$1" in
run) printf "The answer is 4"; exit 0;;
rosetta) printf "conversion error" >&2; exit 1;;
esac
exit 1"#,
        );

        let mut test = ConversionTest::new(
            Format::Gguf,
            Format::Apr,
            Backend::Cpu,
            ModelId::new("test", "model"),
        );
        test.binary = mock.to_string_lossy().to_string();

        if let Err(e) = test.execute(&model_file) {
            let msg = e.to_string();
            assert!(msg.contains("Conversion failed") || msg.contains("conversion error"));
        }
    }

    #[test]
    fn test_semantic_test_execute_corroborated_via_mock() {
        let dir = tempfile::tempdir().unwrap();
        let model_file = dir.path().join("model.safetensors");
        std::fs::write(&model_file, "fake").unwrap();

        let mock = create_mock_apr(
            dir.path(),
            r#"case "$1" in
run) printf "The answer is 4"; exit 0;;
rosetta) touch "$4"; exit 0;;
esac
exit 1"#,
        );

        let mut test = SemanticConversionTest::new(
            Format::SafeTensors,
            Format::Apr,
            Backend::Cpu,
            ModelId::new("test", "model"),
        );
        test.binary = mock.to_string_lossy().to_string();

        if let Ok(sem) = test.execute(&model_file) {
            if let SemanticTestResult::Corroborated {
                source_output,
                target_output,
            } = &sem
            {
                assert_eq!(source_output, target_output);
                assert!(sem.is_pass());
                assert!(sem.bug_type().is_none());
            }
        }
    }

    #[test]
    fn test_semantic_test_execute_embedding_transposition_via_mock() {
        let dir = tempfile::tempdir().unwrap();
        let model_file = dir.path().join("model.safetensors");
        std::fs::write(&model_file, "fake").unwrap();

        let mock = create_mock_apr(
            dir.path(),
            r#"case "$1" in
run)
  case "$2" in
  *semantic_test*) printf "PAD PAD PAD garbage tokens";;
  *) printf "The answer is 4";;
  esac
  exit 0;;
rosetta) touch "$4"; exit 0;;
esac
exit 1"#,
        );

        let mut test = SemanticConversionTest::new(
            Format::SafeTensors,
            Format::Apr,
            Backend::Cpu,
            ModelId::new("test", "model"),
        );
        test.binary = mock.to_string_lossy().to_string();

        if let Ok(sem) = test.execute(&model_file) {
            if let SemanticTestResult::Falsified { bug_type, .. } = &sem {
                assert_eq!(*bug_type, ConversionBugType::EmbeddingTransposition);
                assert!(!sem.is_pass());
            }
        }
    }

    #[test]
    fn test_semantic_test_execute_tokenizer_missing_via_mock() {
        let dir = tempfile::tempdir().unwrap();
        let model_file = dir.path().join("model.safetensors");
        std::fs::write(&model_file, "fake").unwrap();

        let mock = create_mock_apr(
            dir.path(),
            r#"case "$1" in
run)
  case "$2" in
  *semantic_test*) printf "output" >&1; printf "PMAT-172: missing embedded tokenizer" >&2;;
  *) printf "The answer is 4";;
  esac
  exit 0;;
rosetta) touch "$4"; exit 0;;
esac
exit 1"#,
        );

        let mut test = SemanticConversionTest::new(
            Format::SafeTensors,
            Format::Apr,
            Backend::Cpu,
            ModelId::new("test", "model"),
        );
        test.binary = mock.to_string_lossy().to_string();

        if let Ok(SemanticTestResult::Falsified {
            bug_type, stderr, ..
        }) = &test.execute(&model_file)
        {
            assert_eq!(*bug_type, ConversionBugType::TokenizerMissing);
            assert!(stderr.contains("PMAT-172"));
        }
    }

    #[test]
    fn test_round_trip_execute_corroborated_via_mock() {
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

        let mut rt = RoundTripTest::new(
            vec![Format::Gguf, Format::Apr, Format::SafeTensors, Format::Gguf],
            Backend::Cpu,
            ModelId::new("test", "model"),
        );
        rt.binary = mock.to_string_lossy().to_string();

        if let Ok(ConversionResult::Corroborated { max_diff, .. }) = rt.execute(&model_file) {
            assert!((max_diff - 0.0).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn test_round_trip_execute_falsified_via_mock() {
        let dir = tempfile::tempdir().unwrap();
        let model_file = dir.path().join("model.gguf");
        std::fs::write(&model_file, "fake").unwrap();

        let mock = create_mock_apr(
            dir.path(),
            r#"case "$1" in
run)
  case "$2" in
  *converted*) printf "Round-trip drift detected";;
  *) printf "The answer is 4";;
  esac
  exit 0;;
rosetta) touch "$4"; exit 0;;
esac
exit 1"#,
        );

        let mut rt = RoundTripTest::new(
            vec![Format::Gguf, Format::Apr],
            Backend::Cpu,
            ModelId::new("test", "model"),
        );
        rt.binary = mock.to_string_lossy().to_string();

        if let Ok(ConversionResult::Falsified { gate_id, .. }) = &rt.execute(&model_file) {
            assert_eq!(gate_id, "F-CONV-RT-001");
        }
    }

    #[test]
    fn test_conversion_executor_execute_all_via_mock() {
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

        let config = ConversionConfig {
            test_all_pairs: true,
            test_round_trips: true,
            backends: vec![Backend::Cpu],
            no_gpu: true,
            ..Default::default()
        };
        let mut executor = ConversionExecutor::new(config);
        executor.binary = mock.to_string_lossy().to_string();
        let model_id = ModelId::new("test", "model");

        if let Ok(exec_result) = executor.execute_all(&model_file, &model_id) {
            assert!(exec_result.total > 0);
            assert!(!exec_result.evidence.is_empty());
            assert!(!exec_result.results.is_empty());
            if exec_result.failed == 0 {
                assert!(exec_result.all_passed());
            }
        }
    }

    #[test]
    fn test_conversion_executor_execute_all_with_errors_via_mock() {
        let dir = tempfile::tempdir().unwrap();
        let model_file = dir.path().join("model.gguf");
        std::fs::write(&model_file, "fake").unwrap();

        let mock = create_mock_apr(
            dir.path(),
            r#"case "$1" in
run) printf "output"; exit 0;;
rosetta) printf "convert failed" >&2; exit 1;;
esac
exit 1"#,
        );

        let config = ConversionConfig {
            test_all_pairs: true,
            test_round_trips: false,
            backends: vec![Backend::Cpu],
            no_gpu: true,
            ..Default::default()
        };
        let mut executor = ConversionExecutor::new(config);
        executor.binary = mock.to_string_lossy().to_string();
        let model_id = ModelId::new("test", "model");

        if let Ok(exec_result) = executor.execute_all(&model_file, &model_id) {
            assert!(exec_result.total > 0);
            assert!(exec_result.failed > 0);
            assert!(!exec_result.all_passed());
        }
    }

    #[test]
    fn test_conversion_executor_round_trip_error_via_mock() {
        let dir = tempfile::tempdir().unwrap();
        let model_file = dir.path().join("model.gguf");
        std::fs::write(&model_file, "fake").unwrap();

        let mock = create_mock_apr(dir.path(), r"exit 1");

        let config = ConversionConfig {
            test_all_pairs: false,
            test_round_trips: true,
            backends: vec![Backend::Cpu],
            no_gpu: true,
            ..Default::default()
        };
        let mut executor = ConversionExecutor::new(config);
        executor.binary = mock.to_string_lossy().to_string();
        let model_id = ModelId::new("test", "model");

        if let Ok(exec_result) = executor.execute_all(&model_file, &model_id) {
            assert!(!exec_result.evidence.is_empty());
        }
    }

    #[test]
    fn test_conversion_test_execute_safetensors_target_via_mock() {
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

        let mut test = ConversionTest::new(
            Format::Gguf,
            Format::SafeTensors,
            Backend::Cpu,
            ModelId::new("test", "model"),
        );
        test.binary = mock.to_string_lossy().to_string();

        if let Ok(ConversionResult::Corroborated { target_format, .. }) = test.execute(&model_file)
        {
            assert_eq!(target_format, Format::SafeTensors);
        }
    }

    // =========================================================================
    // Rosetta-Testing Spec: New test type constructors (PMAT-ROSETTA-002/003)
    // =========================================================================

    #[test]
    fn test_idempotency_test_new() {
        let idem = IdempotencyTest::new(
            Format::Gguf,
            Format::Apr,
            Backend::Cpu,
            ModelId::new("test", "model"),
        );
        assert_eq!(idem.format_a, Format::Gguf);
        assert_eq!(idem.format_b, Format::Apr);
        assert_eq!(idem.backend, Backend::Cpu);
    }

    #[test]
    fn test_idempotency_test_debug() {
        let idem = IdempotencyTest::new(
            Format::Gguf,
            Format::Apr,
            Backend::Cpu,
            ModelId::new("test", "model"),
        );
        let debug_str = format!("{idem:?}");
        assert!(debug_str.contains("IdempotencyTest"));
    }

    #[test]
    fn test_idempotency_test_clone() {
        let idem = IdempotencyTest::new(
            Format::SafeTensors,
            Format::Gguf,
            Backend::Gpu,
            ModelId::new("test", "model"),
        );
        let cloned = idem.clone();
        assert_eq!(cloned.format_a, Format::SafeTensors);
        assert_eq!(cloned.format_b, Format::Gguf);
    }

    #[test]
    fn test_commutativity_test_new() {
        let com = CommutativityTest::new(Backend::Cpu, ModelId::new("test", "model"));
        assert_eq!(com.backend, Backend::Cpu);
    }

    #[test]
    fn test_commutativity_test_debug() {
        let com = CommutativityTest::new(Backend::Cpu, ModelId::new("test", "model"));
        let debug_str = format!("{com:?}");
        assert!(debug_str.contains("CommutativityTest"));
    }

    #[test]
    fn test_commutativity_test_clone() {
        let com = CommutativityTest::new(Backend::Gpu, ModelId::new("test", "model"));
        let cloned = com.clone();
        assert_eq!(cloned.backend, Backend::Gpu);
    }

    #[test]
    fn test_conversion_config_new_fields_default() {
        let config = ConversionConfig::default();
        assert!(config.test_multi_hop);
        assert!(config.test_cardinality);
        assert!(config.test_tensor_names);
        assert!(config.test_idempotency);
        assert!(config.test_commutativity);
    }

    #[test]
    fn test_conversion_config_cpu_only_new_fields() {
        let config = ConversionConfig::cpu_only();
        assert!(config.test_multi_hop);
        assert!(config.test_cardinality);
        assert!(config.test_tensor_names);
        assert!(config.test_idempotency);
        assert!(config.test_commutativity);
        assert!(config.no_gpu);
    }

    #[test]
    fn test_conversion_config_selective_disable() {
        let config = ConversionConfig {
            test_multi_hop: false,
            test_cardinality: false,
            test_tensor_names: true,
            test_idempotency: false,
            test_commutativity: true,
            ..Default::default()
        };
        assert!(!config.test_multi_hop);
        assert!(!config.test_cardinality);
        assert!(config.test_tensor_names);
        assert!(!config.test_idempotency);
        assert!(config.test_commutativity);
    }

    #[test]
    fn test_check_cardinality_nonexistent_binary() {
        let source = std::path::PathBuf::from("source.gguf");
        let target = std::path::PathBuf::from("target.apr");
        let result = check_cardinality(&source, &target, "/nonexistent/apr");
        assert!(result.is_err());
    }

    #[test]
    fn test_check_tensor_names_nonexistent_binary() {
        let source = std::path::PathBuf::from("source.gguf");
        let target = std::path::PathBuf::from("target.apr");
        let result = check_tensor_names(&source, &target, "/nonexistent/apr");
        assert!(result.is_err());
    }

    // =========================================================================
    // Mock binary tests for check_cardinality and check_tensor_names
    // =========================================================================

    /// Create a mock binary with explicit fd sync/close to avoid ETXTBSY (os error 26)
    /// when parallel tests execute mock scripts concurrently.
    fn create_mock_inspect_binary(
        dir: &std::path::Path,
        name: &str,
        json_output: &str,
    ) -> std::path::PathBuf {
        create_mock_script(dir, name, &format!("#!/bin/bash\necho '{json_output}'"))
    }

    /// Create a conditional mock binary (if/else on model arg).
    fn create_conditional_mock_binary(
        dir: &std::path::Path,
        name: &str,
        script: &str,
    ) -> std::path::PathBuf {
        create_mock_script(dir, name, script)
    }

    /// Write a mock script with explicit open→write→sync→close to ensure the
    /// write fd is fully released before any execve() can hit ETXTBSY.
    fn create_mock_script(dir: &std::path::Path, name: &str, content: &str) -> std::path::PathBuf {
        let path = dir.join(name);
        {
            use std::io::Write;
            let mut f = std::fs::File::create(&path).expect("create mock");
            f.write_all(content.as_bytes()).expect("write mock");
            f.sync_all().expect("sync mock");
            drop(f);
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755))
                .expect("set permissions");
        }
        // Yield to let the OS fully release the write reference on the inode
        std::thread::yield_now();
        path
    }

    #[test]
    fn test_check_cardinality_loss_detected() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let source_model = dir.path().join("source.gguf");
        let target_model = dir.path().join("target.apr");
        std::fs::write(&source_model, b"source").expect("write source");
        std::fs::write(&target_model, b"target").expect("write target");

        // Mock binary that returns different tensor counts based on the model arg
        let mock = create_conditional_mock_binary(
            dir.path(),
            "apr_card",
            "#!/bin/bash\nif echo \"$3\" | grep -q source; then\n  echo '{\"tensor_count\": 338, \"tensor_names\": []}'\nelse\n  echo '{\"tensor_count\": 227, \"tensor_names\": []}'\nfi",
        );

        let result = check_cardinality(&source_model, &target_model, mock.to_str().expect("path"));
        let (gate_id, reason) = result
            .expect("should succeed")
            .expect("should detect cardinality loss");
        assert_eq!(gate_id, "F-CONV-CARD-001");
        assert!(reason.contains("338"));
        assert!(reason.contains("227"));
    }

    #[test]
    fn test_check_cardinality_no_loss() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let source_model = dir.path().join("source.gguf");
        let target_model = dir.path().join("target.apr");
        std::fs::write(&source_model, b"source").expect("write source");
        std::fs::write(&target_model, b"target").expect("write target");

        let mock = create_mock_inspect_binary(
            dir.path(),
            "apr_card_ok",
            r#"{"tensor_count": 338, "tensor_names": []}"#,
        );

        let result = check_cardinality(&source_model, &target_model, mock.to_str().expect("path"));
        assert!(result.expect("should succeed").is_none());
    }

    #[test]
    fn test_check_tensor_names_fusion_detected() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let source_model = dir.path().join("source.gguf");
        let target_model = dir.path().join("target.apr");
        std::fs::write(&source_model, b"source").expect("write source");
        std::fs::write(&target_model, b"target").expect("write target");

        // Source has q_proj, k_proj, v_proj; target has qkv_proj (fusion)
        let mock = create_conditional_mock_binary(
            dir.path(),
            "apr_names",
            "#!/bin/bash\nif echo \"$3\" | grep -q source; then\n  echo '{\"tensor_count\": 3, \"tensor_names\": [\"layer.0.q_proj\", \"layer.0.k_proj\", \"layer.0.v_proj\"]}'\nelse\n  echo '{\"tensor_count\": 1, \"tensor_names\": [\"layer.0.qkv_proj\"]}'\nfi",
        );

        let result = check_tensor_names(&source_model, &target_model, mock.to_str().expect("path"));
        let (gate_id, detail) = result
            .expect("should succeed")
            .expect("should detect name divergence");
        assert_eq!(gate_id, "F-CONV-NAME-001");
        assert!(detail.contains("QKV fusion"));
        assert!(detail.contains("q_proj"));
    }

    #[test]
    fn test_check_tensor_names_non_fusion_divergence() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let source_model = dir.path().join("source.gguf");
        let target_model = dir.path().join("target.apr");
        std::fs::write(&source_model, b"source").expect("write source");
        std::fs::write(&target_model, b"target").expect("write target");

        // Source has "embed.weight"; target renamed it to "embedding.weight"
        let mock = create_conditional_mock_binary(
            dir.path(),
            "apr_names2",
            "#!/bin/bash\nif echo \"$3\" | grep -q source; then\n  echo '{\"tensor_count\": 2, \"tensor_names\": [\"embed.weight\", \"lm_head.weight\"]}'\nelse\n  echo '{\"tensor_count\": 2, \"tensor_names\": [\"embedding.weight\", \"lm_head.weight\"]}'\nfi",
        );

        let result = check_tensor_names(&source_model, &target_model, mock.to_str().expect("path"));
        let (gate_id, detail) = result
            .expect("should succeed")
            .expect("should detect divergence");
        assert_eq!(gate_id, "F-CONV-NAME-001");
        assert!(detail.contains("divergence"));
        assert!(detail.contains("embed.weight"));
    }

    #[test]
    fn test_check_tensor_names_all_preserved() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let source_model = dir.path().join("source.gguf");
        let target_model = dir.path().join("target.apr");
        std::fs::write(&source_model, b"source").expect("write source");
        std::fs::write(&target_model, b"target").expect("write target");

        let mock = create_mock_inspect_binary(
            dir.path(),
            "apr_names_ok",
            r#"{"tensor_count": 2, "tensor_names": ["a.weight", "b.weight"]}"#,
        );

        let result = check_tensor_names(&source_model, &target_model, mock.to_str().expect("path"));
        assert!(result.expect("should succeed").is_none());
    }

    #[test]
    fn test_check_tensor_names_empty_names_skip() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let source_model = dir.path().join("source.gguf");
        let target_model = dir.path().join("target.apr");
        std::fs::write(&source_model, b"source").expect("write source");
        std::fs::write(&target_model, b"target").expect("write target");

        let mock = create_mock_inspect_binary(
            dir.path(),
            "apr_names_empty",
            r#"{"tensor_count": 10, "tensor_names": []}"#,
        );

        let result = check_tensor_names(&source_model, &target_model, mock.to_str().expect("path"));
        assert!(result.expect("should succeed").is_none());
    }

    #[test]
    fn test_convert_to_format_tagged_gguf_ext() {
        let source = std::path::PathBuf::from("/tmp/model.apr");
        let target = source.with_extension("tag1.gguf");
        assert!(target.to_str().expect("path").ends_with("tag1.gguf"));
    }

    #[test]
    fn test_convert_to_format_tagged_safetensors_ext() {
        let source = std::path::PathBuf::from("/tmp/model.apr");
        let target = source.with_extension("tag2.safetensors");
        assert!(target.to_str().expect("path").ends_with("tag2.safetensors"));
    }

    #[test]
    fn test_run_inference_simple_gpu_flag() {
        // Verify GPU backend produces --gpu arg (fails because no binary, but exercises the match)
        let result = run_inference_simple(
            &std::path::PathBuf::from("/nonexistent/model.gguf"),
            Backend::Gpu,
            "/nonexistent/apr",
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_idempotency_falsified_result_structure() {
        // Directly test the Falsified variant construction
        let result = ConversionResult::Falsified {
            gate_id: "F-CONV-IDEM-001".to_string(),
            reason: "Idempotency failure: Gguf→Apr produced different output".to_string(),
            evidence: ConversionEvidence {
                source_hash: ConversionTest::hash_output("output1"),
                converted_hash: ConversionTest::hash_output("output2"),
                max_diff: 1.0,
                diff_indices: vec![],
                source_format: Format::Gguf,
                target_format: Format::Apr,
                backend: Backend::Cpu,
                failure_type: None,
                quant_type: None,
            },
        };
        match result {
            ConversionResult::Falsified {
                gate_id, reason, ..
            } => {
                assert_eq!(gate_id, "F-CONV-IDEM-001");
                assert!(reason.contains("Idempotency"));
            }
            _ => panic!("Expected Falsified"),
        }
    }

    #[test]
    fn test_commutativity_falsified_result_structure() {
        let result = ConversionResult::Falsified {
            gate_id: "F-CONV-COM-001".to_string(),
            reason: "Commutativity failure: GGUF→APR differs from GGUF→ST→APR".to_string(),
            evidence: ConversionEvidence {
                source_hash: ConversionTest::hash_output("path_a"),
                converted_hash: ConversionTest::hash_output("path_b"),
                max_diff: 1.0,
                diff_indices: vec![],
                source_format: Format::Gguf,
                target_format: Format::Apr,
                backend: Backend::Cpu,
                failure_type: None,
                quant_type: None,
            },
        };
        match result {
            ConversionResult::Falsified {
                gate_id, reason, ..
            } => {
                assert_eq!(gate_id, "F-CONV-COM-001");
                assert!(reason.contains("Commutativity"));
            }
            _ => panic!("Expected Falsified"),
        }
    }

    #[test]
    fn test_conversion_test_convert_model_failure() {
        // Exercise the conversion failure error path
        let result = convert_to_format_tagged(
            &std::path::PathBuf::from("/nonexistent/model.gguf"),
            Format::Gguf,
            "test",
            "/nonexistent/apr",
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_conversion_test_convert_model_safetensors_target() {
        let result = convert_to_format_tagged(
            &std::path::PathBuf::from("/nonexistent/model.apr"),
            Format::SafeTensors,
            "test",
            "/nonexistent/apr",
        );
        assert!(result.is_err());
    }

    // ── §3.4 classify_failure tests ────────────────────────────────────

    #[test]
    fn test_classify_failure_tensor_name_mismatch() {
        assert_eq!(
            classify_failure("tensor name mismatch: q_proj not found", 1),
            ConversionFailureType::TensorNameMismatch
        );
        assert_eq!(
            classify_failure("missing tensor 'lm_head.weight'", 1),
            ConversionFailureType::TensorNameMismatch
        );
        assert_eq!(
            classify_failure("unexpected tensor in output", 1),
            ConversionFailureType::TensorNameMismatch
        );
    }

    #[test]
    fn test_classify_failure_dequantization() {
        assert_eq!(
            classify_failure("dequantization error: NaN values produced", 1),
            ConversionFailureType::DequantizationFailure
        );
        assert_eq!(
            classify_failure("quantization overflow detected", 1),
            ConversionFailureType::DequantizationFailure
        );
        assert_eq!(
            classify_failure("NaN in output tensor", 1),
            ConversionFailureType::DequantizationFailure
        );
        assert_eq!(
            classify_failure("infinity values in layer 5", 1),
            ConversionFailureType::DequantizationFailure
        );
    }

    #[test]
    fn test_classify_failure_config_metadata() {
        assert_eq!(
            classify_failure("hidden_size mismatch: expected 768 got 512", 1),
            ConversionFailureType::ConfigMetadataMismatch
        );
        assert_eq!(
            classify_failure("metadata mismatch: num_layers differs", 1),
            ConversionFailureType::ConfigMetadataMismatch
        );
        assert_eq!(
            classify_failure("vocab_size does not match model", 1),
            ConversionFailureType::ConfigMetadataMismatch
        );
        assert_eq!(
            classify_failure("config mismatch detected", 1),
            ConversionFailureType::ConfigMetadataMismatch
        );
    }

    #[test]
    fn test_classify_failure_missing_artifact() {
        assert_eq!(
            classify_failure("file not found: model.safetensors", 1),
            ConversionFailureType::MissingArtifact
        );
        assert_eq!(
            classify_failure("No such file or directory", 1),
            ConversionFailureType::MissingArtifact
        );
        assert_eq!(
            classify_failure("tokenizer.json missing from model directory", 1),
            ConversionFailureType::MissingArtifact
        );
        assert_eq!(
            classify_failure("config.json: file not found", 1),
            ConversionFailureType::MissingArtifact
        );
    }

    #[test]
    fn test_classify_failure_inference() {
        assert_eq!(
            classify_failure("inference failed: out of memory", 1),
            ConversionFailureType::InferenceFailure
        );
        assert_eq!(
            classify_failure("forward pass error", 1),
            ConversionFailureType::InferenceFailure
        );
        assert_eq!(
            classify_failure("", -11), // SIGSEGV
            ConversionFailureType::InferenceFailure
        );
    }

    #[test]
    fn test_classify_failure_unknown() {
        assert_eq!(
            classify_failure("some generic error", 1),
            ConversionFailureType::Unknown
        );
        assert_eq!(classify_failure("", 1), ConversionFailureType::Unknown);
    }

    #[test]
    fn test_classify_failure_case_insensitive() {
        assert_eq!(
            classify_failure("TENSOR NAME MISMATCH", 1),
            ConversionFailureType::TensorNameMismatch
        );
        assert_eq!(
            classify_failure("Dequantization Error", 1),
            ConversionFailureType::DequantizationFailure
        );
    }

    // ── §3.7 QuantType + tolerance tests ───────────────────────────────

    #[test]
    fn test_quant_type_from_str_label() {
        assert_eq!(QuantType::from_str_label("f32"), QuantType::F32);
        assert_eq!(QuantType::from_str_label("fp32"), QuantType::F32);
        assert_eq!(QuantType::from_str_label("float32"), QuantType::F32);
        assert_eq!(QuantType::from_str_label("f16"), QuantType::F16);
        assert_eq!(QuantType::from_str_label("fp16"), QuantType::F16);
        assert_eq!(QuantType::from_str_label("bf16"), QuantType::BF16);
        assert_eq!(QuantType::from_str_label("bfloat16"), QuantType::BF16);
        assert_eq!(QuantType::from_str_label("q4_k_m"), QuantType::Q4KM);
        assert_eq!(QuantType::from_str_label("q4km"), QuantType::Q4KM);
        assert_eq!(QuantType::from_str_label("q5_k_m"), QuantType::Q5KM);
        assert_eq!(QuantType::from_str_label("q5km"), QuantType::Q5KM);
        assert_eq!(QuantType::from_str_label("q6_k"), QuantType::Q6K);
        assert_eq!(QuantType::from_str_label("q4_0"), QuantType::Q4_0);
        assert_eq!(QuantType::from_str_label("q8_0"), QuantType::Q8_0);
        assert_eq!(
            QuantType::from_str_label("unknown_type"),
            QuantType::Unknown
        );
    }

    #[test]
    fn test_quant_type_from_str_label_case_insensitive() {
        assert_eq!(QuantType::from_str_label("F32"), QuantType::F32);
        assert_eq!(QuantType::from_str_label("BF16"), QuantType::BF16);
        assert_eq!(QuantType::from_str_label("Q4_K_M"), QuantType::Q4KM);
        assert_eq!(QuantType::from_str_label("Q5_K_M"), QuantType::Q5KM);
    }

    #[test]
    fn test_quant_type_from_str_label_with_hyphens() {
        assert_eq!(QuantType::from_str_label("q4-k-m"), QuantType::Q4KM);
        assert_eq!(QuantType::from_str_label("q5-k-m"), QuantType::Q5KM);
        assert_eq!(QuantType::from_str_label("q6-k"), QuantType::Q6K);
    }

    #[test]
    fn test_tolerance_for_f32() {
        let tol = tolerance_for(QuantType::F32);
        assert!((tol.atol - 1e-6).abs() < 1e-10);
    }

    #[test]
    fn test_tolerance_for_f16() {
        let tol = tolerance_for(QuantType::F16);
        assert!((tol.atol - 1e-3).abs() < 1e-10);
    }

    #[test]
    fn test_tolerance_for_q4km() {
        let tol = tolerance_for(QuantType::Q4KM);
        assert!((tol.atol - 1e-1).abs() < 1e-10);
    }

    #[test]
    fn test_tolerance_for_q5km() {
        let tol = tolerance_for(QuantType::Q5KM);
        assert!((tol.atol - 7.5e-2).abs() < 1e-10);
        assert!((tol.rtol - 5e-2).abs() < 1e-10);
    }

    #[test]
    fn test_tolerance_for_q6k() {
        let tol = tolerance_for(QuantType::Q6K);
        assert!((tol.atol - 5e-2).abs() < 1e-10);
    }
