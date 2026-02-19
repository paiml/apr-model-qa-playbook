
use super::*;

#[test]
fn test_all_conversion_pairs() {
    let pairs = all_conversion_pairs();
    assert_eq!(pairs.len(), 6);
}

#[test]
fn test_all_backends() {
    let backends = all_backends();
    assert_eq!(backends.len(), 2);
}

#[test]
fn test_generate_conversion_tests() {
    let model_id = ModelId::new("test", "model");
    let tests = generate_conversion_tests(&model_id);
    // 6 pairs × 2 backends = 12 tests
    assert_eq!(tests.len(), 12);
}

#[test]
fn test_gate_id() {
    let test = ConversionTest::new(
        Format::Gguf,
        Format::Apr,
        Backend::Cpu,
        ModelId::new("test", "model"),
    );
    assert_eq!(test.gate_id(), "F-CONV-G-A");
}

#[test]
fn test_compute_diff_identical() {
    let test = ConversionTest::new(
        Format::Gguf,
        Format::Apr,
        Backend::Cpu,
        ModelId::new("test", "model"),
    );
    let diff = test.compute_diff("hello", "hello");
    assert!((diff - 0.0).abs() < f64::EPSILON);
}

#[test]
fn test_compute_diff_different() {
    let test = ConversionTest::new(
        Format::Gguf,
        Format::Apr,
        Backend::Cpu,
        ModelId::new("test", "model"),
    );
    let diff = test.compute_diff("hello", "world");
    assert!(diff > 0.0);
}

#[test]
fn test_hash_output() {
    let hash1 = ConversionTest::hash_output("test");
    let hash2 = ConversionTest::hash_output("test");
    assert_eq!(hash1, hash2);

    let hash3 = ConversionTest::hash_output("different");
    assert_ne!(hash1, hash3);
}

#[test]
fn test_find_diff_indices() {
    let test = ConversionTest::new(
        Format::Gguf,
        Format::Apr,
        Backend::Cpu,
        ModelId::new("test", "model"),
    );
    let indices = test.find_diff_indices("hello", "hallo");
    assert_eq!(indices, vec![1]);
}

#[test]
fn test_conversion_result_to_evidence_corroborated() {
    let result = ConversionResult::Corroborated {
        source_format: Format::Gguf,
        target_format: Format::Apr,
        backend: Backend::Cpu,
        max_diff: 0.0,
    };
    let evidence: Evidence = result.into();
    assert!(evidence.outcome.is_pass());
}

#[test]
fn test_conversion_result_to_evidence_falsified() {
    let result = ConversionResult::Falsified {
        gate_id: "F-CONV-G-A".to_string(),
        reason: "Test failure".to_string(),
        evidence: ConversionEvidence {
            source_hash: "abc".to_string(),
            converted_hash: "def".to_string(),
            max_diff: 0.5,
            diff_indices: vec![0, 1],
            source_format: Format::Gguf,
            target_format: Format::Apr,
            backend: Backend::Cpu,
            failure_type: None,
            quant_type: None,
        },
    };
    let evidence: Evidence = result.into();
    assert!(!evidence.outcome.is_pass());
}

#[test]
fn test_round_trip_test_new() {
    let rt = RoundTripTest::new(
        vec![Format::Gguf, Format::Apr, Format::SafeTensors],
        Backend::Cpu,
        ModelId::new("test", "model"),
    );
    assert_eq!(rt.formats.len(), 3);
}

#[test]
fn test_default_epsilon() {
    assert!((default_epsilon() - 1e-6).abs() < f64::EPSILON);
}

#[test]
fn test_conversion_test_epsilon() {
    let test = ConversionTest::new(
        Format::Gguf,
        Format::Apr,
        Backend::Cpu,
        ModelId::new("test", "model"),
    );
    assert!((test.epsilon - EPSILON).abs() < f64::EPSILON);
}

#[test]
fn test_compute_diff_empty_strings() {
    let test = ConversionTest::new(
        Format::Gguf,
        Format::Apr,
        Backend::Cpu,
        ModelId::new("test", "model"),
    );
    let diff = test.compute_diff("", "");
    assert!((diff - 0.0).abs() < f64::EPSILON);
}

#[test]
fn test_compute_diff_one_empty() {
    let test = ConversionTest::new(
        Format::Gguf,
        Format::Apr,
        Backend::Cpu,
        ModelId::new("test", "model"),
    );
    let diff = test.compute_diff("hello", "");
    assert!(diff > 0.0);
}

#[test]
fn test_find_diff_indices_empty() {
    let test = ConversionTest::new(
        Format::Gguf,
        Format::Apr,
        Backend::Cpu,
        ModelId::new("test", "model"),
    );
    let indices = test.find_diff_indices("", "");
    assert!(indices.is_empty());
}

#[test]
fn test_find_diff_indices_all_different() {
    let test = ConversionTest::new(
        Format::Gguf,
        Format::Apr,
        Backend::Cpu,
        ModelId::new("test", "model"),
    );
    let indices = test.find_diff_indices("abc", "xyz");
    assert_eq!(indices.len(), 3);
}

#[test]
fn test_gate_id_safetensors() {
    let test = ConversionTest::new(
        Format::SafeTensors,
        Format::Gguf,
        Backend::Gpu,
        ModelId::new("test", "model"),
    );
    assert_eq!(test.gate_id(), "F-CONV-S-G");
}

#[test]
fn test_gate_id_apr() {
    let test = ConversionTest::new(
        Format::Apr,
        Format::SafeTensors,
        Backend::Cpu,
        ModelId::new("test", "model"),
    );
    assert_eq!(test.gate_id(), "F-CONV-A-S");
}

#[test]
fn test_all_conversion_pairs_unique() {
    let pairs = all_conversion_pairs();
    for (i, p1) in pairs.iter().enumerate() {
        for (j, p2) in pairs.iter().enumerate() {
            if i != j {
                assert!(p1 != p2, "Duplicate pair found");
            }
        }
    }
}

#[test]
fn test_conversion_evidence_clone() {
    let evidence = ConversionEvidence {
        source_hash: "abc".to_string(),
        converted_hash: "def".to_string(),
        max_diff: 0.5,
        diff_indices: vec![0, 1],
        source_format: Format::Gguf,
        target_format: Format::Apr,
        backend: Backend::Cpu,
        failure_type: None,
        quant_type: None,
    };
    let cloned = evidence.clone();
    assert_eq!(evidence.source_hash, cloned.source_hash);
    assert_eq!(evidence.max_diff, cloned.max_diff);
}

#[test]
fn test_conversion_result_clone() {
    let result = ConversionResult::Corroborated {
        source_format: Format::Gguf,
        target_format: Format::Apr,
        backend: Backend::Cpu,
        max_diff: 0.0,
    };
    let cloned = result.clone();
    match cloned {
        ConversionResult::Corroborated { max_diff, .. } => {
            assert!((max_diff - 0.0).abs() < f64::EPSILON);
        }
        _ => panic!("Expected Corroborated"),
    }
}

#[test]
fn test_conversion_test_clone() {
    let test = ConversionTest::new(
        Format::Gguf,
        Format::Apr,
        Backend::Cpu,
        ModelId::new("test", "model"),
    );
    let cloned = test.clone();
    assert_eq!(test.source_format, cloned.source_format);
    assert_eq!(test.target_format, cloned.target_format);
}

#[test]
fn test_round_trip_test_formats() {
    let rt = RoundTripTest::new(
        vec![Format::Gguf, Format::Apr],
        Backend::Gpu,
        ModelId::new("test", "model"),
    );
    assert_eq!(rt.formats.len(), 2);
    assert_eq!(rt.backend, Backend::Gpu);
}

#[test]
fn test_conversion_test_debug() {
    let test = ConversionTest::new(
        Format::Gguf,
        Format::Apr,
        Backend::Cpu,
        ModelId::new("test", "model"),
    );
    let debug_str = format!("{test:?}");
    assert!(debug_str.contains("ConversionTest"));
}

#[test]
fn test_conversion_evidence_debug() {
    let evidence = ConversionEvidence {
        source_hash: "abc".to_string(),
        converted_hash: "def".to_string(),
        max_diff: 0.5,
        diff_indices: vec![0, 1],
        source_format: Format::Gguf,
        target_format: Format::Apr,
        backend: Backend::Cpu,
        failure_type: None,
        quant_type: None,
    };
    let debug_str = format!("{evidence:?}");
    assert!(debug_str.contains("ConversionEvidence"));
}

#[test]
fn test_conversion_result_debug() {
    let result = ConversionResult::Corroborated {
        source_format: Format::Gguf,
        target_format: Format::Apr,
        backend: Backend::Cpu,
        max_diff: 0.0,
    };
    let debug_str = format!("{result:?}");
    assert!(debug_str.contains("Corroborated"));
}

#[test]
fn test_epsilon_constant() {
    assert!(EPSILON > 0.0);
    assert!(EPSILON < 1.0);
}

#[test]
fn test_generate_conversion_tests_all_formats() {
    let model_id = ModelId::new("org", "model");
    let tests = generate_conversion_tests(&model_id);

    // Verify all format pairs are covered
    let has_gguf_to_apr = tests
        .iter()
        .any(|t| t.source_format == Format::Gguf && t.target_format == Format::Apr);
    let has_apr_to_safetensors = tests
        .iter()
        .any(|t| t.source_format == Format::Apr && t.target_format == Format::SafeTensors);

    assert!(has_gguf_to_apr);
    assert!(has_apr_to_safetensors);
}

#[test]
fn test_conversion_config_default() {
    let config = ConversionConfig::default();
    assert!(config.test_all_pairs);
    assert!(config.test_round_trips);
    assert_eq!(config.backends.len(), 2);
    assert!(!config.no_gpu);
}

#[test]
fn test_conversion_config_cpu_only() {
    let config = ConversionConfig::cpu_only();
    assert!(config.test_all_pairs);
    assert!(config.test_round_trips);
    assert_eq!(config.backends.len(), 1);
    assert_eq!(config.backends[0], Backend::Cpu);
    assert!(config.no_gpu);
}

#[test]
fn test_conversion_executor_new() {
    let config = ConversionConfig::default();
    let executor = ConversionExecutor::new(config);
    assert!(!executor.config.no_gpu);
}

#[test]
fn test_conversion_executor_with_defaults() {
    let executor = ConversionExecutor::with_defaults();
    assert!(executor.config.test_all_pairs);
}

#[test]
fn test_conversion_config_debug() {
    let config = ConversionConfig::default();
    let debug_str = format!("{config:?}");
    assert!(debug_str.contains("ConversionConfig"));
}

#[test]
fn test_conversion_config_clone() {
    let config = ConversionConfig::default();
    let cloned = config.clone();
    assert_eq!(cloned.test_all_pairs, config.test_all_pairs);
    assert_eq!(cloned.no_gpu, config.no_gpu);
}

#[test]
fn test_conversion_executor_debug() {
    let executor = ConversionExecutor::with_defaults();
    let debug_str = format!("{executor:?}");
    assert!(debug_str.contains("ConversionExecutor"));
}

#[test]
fn test_round_trip_test_debug() {
    let rt = RoundTripTest::new(
        vec![Format::Gguf, Format::Apr],
        Backend::Cpu,
        ModelId::new("test", "model"),
    );
    let debug_str = format!("{rt:?}");
    assert!(debug_str.contains("RoundTripTest"));
}

#[test]
fn test_round_trip_test_clone() {
    let rt = RoundTripTest::new(
        vec![Format::Gguf, Format::Apr],
        Backend::Cpu,
        ModelId::new("test", "model"),
    );
    let cloned = rt.clone();
    assert_eq!(cloned.formats.len(), rt.formats.len());
    assert_eq!(cloned.backend, rt.backend);
}

#[test]
fn test_conversion_test_with_epsilon() {
    let test = ConversionTest {
        source_format: Format::Gguf,
        target_format: Format::Apr,
        backend: Backend::Cpu,
        model_id: ModelId::new("test", "model"),
        epsilon: 1e-9,
        binary: default_binary(),
        quant_type: None,
        output_dir: None,
    };
    assert!((test.epsilon - 1e-9).abs() < f64::EPSILON);
}

#[test]
fn test_conversion_execution_result() {
    let result = ConversionExecutionResult {
        passed: 10,
        failed: 2,
        total: 12,
        evidence: vec![],
        results: vec![],
        duration_ms: 1000,
    };
    assert_eq!(result.passed, 10);
    assert_eq!(result.failed, 2);
    assert_eq!(result.total, 12);
}

#[test]
fn test_conversion_execution_result_debug() {
    let result = ConversionExecutionResult {
        passed: 5,
        failed: 1,
        total: 6,
        evidence: vec![],
        results: vec![],
        duration_ms: 500,
    };
    let debug_str = format!("{result:?}");
    assert!(debug_str.contains("ConversionExecutionResult"));
}

#[test]
fn test_all_backends_content() {
    let backends = all_backends();
    assert!(backends.contains(&Backend::Cpu));
    assert!(backends.contains(&Backend::Gpu));
}

#[test]
fn test_gate_id_all_combinations() {
    // Test all source/target combinations
    let combos = [
        (Format::Gguf, Format::Apr, "F-CONV-G-A"),
        (Format::Apr, Format::Gguf, "F-CONV-A-G"),
        (Format::Gguf, Format::SafeTensors, "F-CONV-G-S"),
        (Format::SafeTensors, Format::Gguf, "F-CONV-S-G"),
        (Format::Apr, Format::SafeTensors, "F-CONV-A-S"),
        (Format::SafeTensors, Format::Apr, "F-CONV-S-A"),
    ];

    for (source, target, expected) in combos {
        let test = ConversionTest::new(source, target, Backend::Cpu, ModelId::new("t", "m"));
        assert_eq!(test.gate_id(), expected);
    }
}

#[test]
fn test_compute_diff_partially_matching() {
    let test = ConversionTest::new(
        Format::Gguf,
        Format::Apr,
        Backend::Cpu,
        ModelId::new("test", "model"),
    );
    // "hello" vs "hallo" - 1 char different out of 5
    let diff = test.compute_diff("hello", "hallo");
    assert!(diff > 0.0);
    assert!(diff < 1.0);
}

#[test]
fn test_find_diff_indices_longer_second() {
    let test = ConversionTest::new(
        Format::Gguf,
        Format::Apr,
        Backend::Cpu,
        ModelId::new("test", "model"),
    );
    // "ab" vs "abc" - only compares up to shorter length
    let indices = test.find_diff_indices("ab", "abc");
    assert!(indices.is_empty()); // first 2 chars match
}

#[test]
fn test_conversion_execution_result_all_passed() {
    let result = ConversionExecutionResult {
        passed: 10,
        failed: 0,
        total: 10,
        evidence: vec![],
        results: vec![],
        duration_ms: 1000,
    };
    assert!(result.all_passed());
}

#[test]
fn test_conversion_execution_result_not_all_passed() {
    let result = ConversionExecutionResult {
        passed: 8,
        failed: 2,
        total: 10,
        evidence: vec![],
        results: vec![],
        duration_ms: 1000,
    };
    assert!(!result.all_passed());
}

#[test]
fn test_conversion_execution_result_pass_rate() {
    let result = ConversionExecutionResult {
        passed: 8,
        failed: 2,
        total: 10,
        evidence: vec![],
        results: vec![],
        duration_ms: 1000,
    };
    let rate = result.pass_rate();
    assert!((rate - 80.0).abs() < f64::EPSILON);
}

#[test]
fn test_conversion_execution_result_pass_rate_zero_total() {
    let result = ConversionExecutionResult {
        passed: 0,
        failed: 0,
        total: 0,
        evidence: vec![],
        results: vec![],
        duration_ms: 0,
    };
    let rate = result.pass_rate();
    assert!((rate - 100.0).abs() < f64::EPSILON);
}

#[test]
fn test_conversion_execution_result_pass_rate_all_passed() {
    let result = ConversionExecutionResult {
        passed: 5,
        failed: 0,
        total: 5,
        evidence: vec![],
        results: vec![],
        duration_ms: 500,
    };
    let rate = result.pass_rate();
    assert!((rate - 100.0).abs() < f64::EPSILON);
}

#[test]
fn test_conversion_execution_result_pass_rate_none_passed() {
    let result = ConversionExecutionResult {
        passed: 0,
        failed: 5,
        total: 5,
        evidence: vec![],
        results: vec![],
        duration_ms: 500,
    };
    let rate = result.pass_rate();
    assert!((rate - 0.0).abs() < f64::EPSILON);
}

// Tests for ConversionBugType (GH-187)

#[test]
fn test_bug_type_gate_ids() {
    assert_eq!(
        ConversionBugType::EmbeddingTransposition.gate_id(),
        "F-CONV-EMBED-001"
    );
    assert_eq!(
        ConversionBugType::TokenizerMissing.gate_id(),
        "F-CONV-TOK-001"
    );
    assert_eq!(
        ConversionBugType::WeightCorruption.gate_id(),
        "F-CONV-WEIGHT-001"
    );
    assert_eq!(
        ConversionBugType::ShapeMismatch.gate_id(),
        "F-CONV-SHAPE-001"
    );
    assert_eq!(
        ConversionBugType::SemanticDrift.gate_id(),
        "F-CONV-SEMANTIC-001"
    );
    assert_eq!(ConversionBugType::Unknown.gate_id(), "F-CONV-UNKNOWN-001");
}

#[test]
fn test_bug_type_descriptions() {
    assert!(
        ConversionBugType::EmbeddingTransposition
            .description()
            .contains("transposition")
    );
    assert!(
        ConversionBugType::TokenizerMissing
            .description()
            .contains("tokenizer")
    );
    assert!(
        ConversionBugType::WeightCorruption
            .description()
            .contains("corruption")
    );
}

#[test]
fn test_bug_type_clone() {
    let bug = ConversionBugType::EmbeddingTransposition;
    let cloned = bug;
    assert_eq!(bug, cloned);
}

#[test]
fn test_bug_type_debug() {
    let debug_str = format!("{:?}", ConversionBugType::TokenizerMissing);
    assert!(debug_str.contains("TokenizerMissing"));
}

#[test]
fn test_semantic_test_new() {
    let test = SemanticConversionTest::new(
        Format::Gguf,
        Format::Apr,
        Backend::Cpu,
        ModelId::new("test", "model"),
    );
    assert_eq!(test.source_format, Format::Gguf);
    assert_eq!(test.target_format, Format::Apr);
}

#[test]
fn test_semantic_test_clone() {
    let test = SemanticConversionTest::new(
        Format::Gguf,
        Format::Apr,
        Backend::Cpu,
        ModelId::new("test", "model"),
    );
    let cloned = test.clone();
    assert_eq!(test.source_format, cloned.source_format);
}

#[test]
fn test_semantic_test_debug() {
    let test = SemanticConversionTest::new(
        Format::Gguf,
        Format::Apr,
        Backend::Cpu,
        ModelId::new("test", "model"),
    );
    let debug_str = format!("{test:?}");
    assert!(debug_str.contains("SemanticConversionTest"));
}

#[test]
fn test_semantic_result_is_pass() {
    let pass = SemanticTestResult::Corroborated {
        source_output: "4".to_string(),
        target_output: "4".to_string(),
    };
    assert!(pass.is_pass());

    let fail = SemanticTestResult::Falsified {
        bug_type: ConversionBugType::EmbeddingTransposition,
        source_output: "4".to_string(),
        target_output: "garbage".to_string(),
        stderr: String::new(),
    };
    assert!(!fail.is_pass());
}

#[test]
fn test_semantic_result_bug_type() {
    let pass = SemanticTestResult::Corroborated {
        source_output: "4".to_string(),
        target_output: "4".to_string(),
    };
    assert!(pass.bug_type().is_none());

    let fail = SemanticTestResult::Falsified {
        bug_type: ConversionBugType::TokenizerMissing,
        source_output: "4".to_string(),
        target_output: "garbage".to_string(),
        stderr: String::new(),
    };
    assert_eq!(fail.bug_type(), Some(ConversionBugType::TokenizerMissing));
}

#[test]
fn test_garbage_patterns_detection() {
    // These patterns should trigger embedding transposition detection
    let garbage_outputs = [
        "1. What is the difference between",
        "<pad><pad><pad>",
        "PAD PAD PAD",
        "token 151935 151935",
    ];

    for output in garbage_outputs {
        let has_garbage = GARBAGE_PATTERNS.iter().any(|p| output.contains(p));
        assert!(has_garbage, "Should detect garbage in: {output}");
    }
}

#[test]
fn test_arithmetic_expected_detection() {
    // These patterns should be recognized as correct answers
    let correct_outputs = [
        "The answer is 4",
        "2+2=4",
        "equals 4.",
        "It's four",
        "Four is the answer",
    ];

    for output in correct_outputs {
        let has_expected = ARITHMETIC_EXPECTED.iter().any(|p| output.contains(p));
        assert!(has_expected, "Should detect correct answer in: {output}");
    }
}

#[test]
fn test_semantic_result_clone() {
    let result = SemanticTestResult::Corroborated {
        source_output: "test".to_string(),
        target_output: "test".to_string(),
    };
    let cloned = result.clone();
    assert!(cloned.is_pass());
}

#[test]
fn test_semantic_result_debug() {
    let result = SemanticTestResult::Falsified {
        bug_type: ConversionBugType::Unknown,
        source_output: "a".to_string(),
        target_output: "b".to_string(),
        stderr: String::new(),
    };
    let debug_str = format!("{result:?}");
    assert!(debug_str.contains("Falsified"));
}

// Tests for classify_bug logic
#[test]
fn test_classify_bug_tokenizer_missing() {
    let test = SemanticConversionTest::new(
        Format::Gguf,
        Format::Apr,
        Backend::Cpu,
        ModelId::new("test", "model"),
    );
    let bug = test.classify_bug("The answer is 4", "The answer is 4", true);
    assert_eq!(bug, Some(ConversionBugType::TokenizerMissing));
}

#[test]
fn test_classify_bug_embedding_transposition() {
    let test = SemanticConversionTest::new(
        Format::Gguf,
        Format::Apr,
        Backend::Cpu,
        ModelId::new("test", "model"),
    );
    // Source has correct answer, target has garbage
    let bug = test.classify_bug("The answer is 4", "PAD PAD PAD garbage", false);
    assert_eq!(bug, Some(ConversionBugType::EmbeddingTransposition));
}

#[test]
fn test_classify_bug_semantic_drift() {
    let test = SemanticConversionTest::new(
        Format::Gguf,
        Format::Apr,
        Backend::Cpu,
        ModelId::new("test", "model"),
    );
    // Source has correct answer, target has wrong but not garbage answer
    let bug = test.classify_bug("The answer is 4", "The answer is 7", false);
    assert_eq!(bug, Some(ConversionBugType::SemanticDrift));
}

#[test]
fn test_classify_bug_weight_corruption() {
    let test = SemanticConversionTest::new(
        Format::Gguf,
        Format::Apr,
        Backend::Cpu,
        ModelId::new("test", "model"),
    );
    // Source has output (but not expected arithmetic answer), target is empty
    // WeightCorruption is only detected when target is empty/whitespace
    let bug = test.classify_bug("Hello world, here is some text", "   ", false);
    assert_eq!(bug, Some(ConversionBugType::WeightCorruption));
}

#[test]
fn test_classify_bug_no_bug() {
    let test = SemanticConversionTest::new(
        Format::Gguf,
        Format::Apr,
        Backend::Cpu,
        ModelId::new("test", "model"),
    );
    // Both outputs are identical
    let bug = test.classify_bug("The answer is 4", "The answer is 4", false);
    assert!(bug.is_none());
}

#[test]
fn test_classify_bug_unknown() {
    let test = SemanticConversionTest::new(
        Format::Gguf,
        Format::Apr,
        Backend::Cpu,
        ModelId::new("test", "model"),
    );
    // Source has no expected answer, outputs differ
    let bug = test.classify_bug("random text", "different text", false);
    assert_eq!(bug, Some(ConversionBugType::Unknown));
}

#[test]
fn test_classify_bug_with_endoftext_pattern() {
    let test = SemanticConversionTest::new(
        Format::Gguf,
        Format::Apr,
        Backend::Cpu,
        ModelId::new("test", "model"),
    );
    let bug = test.classify_bug(
        "The answer is 4",
        "Output: <|endoftext|><|endoftext|>",
        false,
    );
    assert_eq!(bug, Some(ConversionBugType::EmbeddingTransposition));
}

#[test]
fn test_classify_bug_with_null_chars() {
    let test = SemanticConversionTest::new(
        Format::Gguf,
        Format::Apr,
        Backend::Cpu,
        ModelId::new("test", "model"),
    );
    let bug = test.classify_bug("The answer is 4", "text\u{0000}with\u{0000}nulls", false);
    assert_eq!(bug, Some(ConversionBugType::EmbeddingTransposition));
}

#[test]
fn test_classify_bug_whitespace_trimming() {
    let test = SemanticConversionTest::new(
        Format::Gguf,
        Format::Apr,
        Backend::Cpu,
        ModelId::new("test", "model"),
    );
    // Same content but different whitespace - should match
    let bug = test.classify_bug("  The answer is 4  ", "The answer is 4", false);
    assert!(bug.is_none());
}

#[test]
fn test_bug_type_equality() {
    assert_eq!(
        ConversionBugType::EmbeddingTransposition,
        ConversionBugType::EmbeddingTransposition
    );
    assert_ne!(
        ConversionBugType::EmbeddingTransposition,
        ConversionBugType::TokenizerMissing
    );
}

#[test]
fn test_conversion_evidence_source_format() {
    let evidence = ConversionEvidence {
        source_hash: "abc123".to_string(),
        converted_hash: "def456".to_string(),
        max_diff: 0.1,
        diff_indices: vec![0, 5, 10],
        source_format: Format::SafeTensors,
        target_format: Format::Apr,
        backend: Backend::Gpu,
        failure_type: None,
        quant_type: None,
    };
    assert_eq!(evidence.source_format, Format::SafeTensors);
    assert_eq!(evidence.target_format, Format::Apr);
    assert_eq!(evidence.backend, Backend::Gpu);
}

#[test]
fn test_conversion_test_model_id() {
    let model_id = ModelId::new("my-org", "my-model");
    let test = ConversionTest::new(Format::Gguf, Format::Apr, Backend::Cpu, model_id.clone());
    assert_eq!(test.model_id.org, "my-org");
    assert_eq!(test.model_id.name, "my-model");
}

#[test]
fn test_semantic_conversion_test_backend() {
    let test = SemanticConversionTest::new(
        Format::Gguf,
        Format::Apr,
        Backend::Gpu,
        ModelId::new("test", "model"),
    );
    assert_eq!(test.backend, Backend::Gpu);
}

#[test]
fn test_round_trip_test_model_id() {
    let model_id = ModelId::new("org", "name");
    let rt = RoundTripTest::new(
        vec![Format::Gguf, Format::Apr],
        Backend::Cpu,
        model_id.clone(),
    );
    assert_eq!(rt.model_id.org, "org");
    assert_eq!(rt.model_id.name, "name");
}

#[test]
fn test_conversion_config_backends() {
    let config = ConversionConfig::default();
    assert_eq!(config.backends.len(), 2);
    assert!(config.backends.contains(&Backend::Cpu));
    assert!(config.backends.contains(&Backend::Gpu));
}

#[test]
fn test_conversion_config_custom() {
    let config = ConversionConfig {
        test_all_pairs: false,
        test_round_trips: false,
        backends: vec![Backend::Cpu],
        no_gpu: true,
        ..Default::default()
    };
    assert!(!config.test_all_pairs);
    assert!(!config.test_round_trips);
    assert_eq!(config.backends.len(), 1);
}

#[test]
fn test_conversion_executor_config_access() {
    let config = ConversionConfig::cpu_only();
    let executor = ConversionExecutor::new(config);
    assert!(executor.config.no_gpu);
    assert!(executor.config.test_all_pairs);
}

#[test]
fn test_all_conversion_pairs_bidirectional() {
    let pairs = all_conversion_pairs();
    // Should have GGUF -> APR and APR -> GGUF
    let has_gguf_to_apr = pairs.contains(&(Format::Gguf, Format::Apr));
    let has_apr_to_gguf = pairs.contains(&(Format::Apr, Format::Gguf));
    assert!(has_gguf_to_apr);
    assert!(has_apr_to_gguf);
}

#[test]
fn test_epsilon_value() {
    assert!((EPSILON - 1e-6).abs() < 1e-10);
}

#[test]
fn test_conversion_result_corroborated_max_diff() {
    let result = ConversionResult::Corroborated {
        source_format: Format::Gguf,
        target_format: Format::Apr,
        backend: Backend::Cpu,
        max_diff: 1e-8,
    };
    match result {
        ConversionResult::Corroborated { max_diff, .. } => {
            assert!(max_diff < EPSILON);
        }
        _ => panic!("Expected Corroborated"),
    }
}

#[test]
fn test_conversion_result_falsified_gate_id() {
    let result = ConversionResult::Falsified {
        gate_id: "F-CONV-G-A".to_string(),
        reason: "Outputs differ".to_string(),
        evidence: ConversionEvidence {
            source_hash: "a".to_string(),
            converted_hash: "b".to_string(),
            max_diff: 0.5,
            diff_indices: vec![],
            source_format: Format::Gguf,
            target_format: Format::Apr,
            backend: Backend::Cpu,
            failure_type: None,
            quant_type: None,
        },
    };
    match result {
        ConversionResult::Falsified { gate_id, .. } => {
            assert_eq!(gate_id, "F-CONV-G-A");
        }
        _ => panic!("Expected Falsified"),
    }
}

#[test]
fn test_semantic_test_result_corroborated_outputs() {
    let result = SemanticTestResult::Corroborated {
        source_output: "answer is 4".to_string(),
        target_output: "answer is 4".to_string(),
    };
    match result {
        SemanticTestResult::Corroborated {
            source_output,
            target_output,
        } => {
            assert_eq!(source_output, target_output);
        }
        _ => panic!("Expected Corroborated"),
    }
}

#[test]
fn test_semantic_test_result_falsified_stderr() {
    let result = SemanticTestResult::Falsified {
        bug_type: ConversionBugType::TokenizerMissing,
        source_output: "4".to_string(),
        target_output: "garbage".to_string(),
        stderr: "PMAT-172: tokenizer missing".to_string(),
    };
    match result {
        SemanticTestResult::Falsified { stderr, .. } => {
            assert!(stderr.contains("PMAT-172"));
        }
        _ => panic!("Expected Falsified"),
    }
}

#[test]
fn test_all_bug_types_have_gate_ids() {
    let bug_types = [
        ConversionBugType::EmbeddingTransposition,
        ConversionBugType::TokenizerMissing,
        ConversionBugType::WeightCorruption,
        ConversionBugType::ShapeMismatch,
        ConversionBugType::SemanticDrift,
        ConversionBugType::Unknown,
    ];
    for bug_type in bug_types {
        let gate_id = bug_type.gate_id();
        assert!(!gate_id.is_empty());
        assert!(gate_id.starts_with("F-CONV-"));
    }
}

#[test]
fn test_all_bug_types_have_descriptions() {
    let bug_types = [
        ConversionBugType::EmbeddingTransposition,
        ConversionBugType::TokenizerMissing,
        ConversionBugType::WeightCorruption,
        ConversionBugType::ShapeMismatch,
        ConversionBugType::SemanticDrift,
        ConversionBugType::Unknown,
    ];
    for bug_type in bug_types {
        let desc = bug_type.description();
        assert!(!desc.is_empty());
    }
}

#[test]
fn test_conversion_evidence_diff_indices() {
    let evidence = ConversionEvidence {
        source_hash: "a".to_string(),
        converted_hash: "b".to_string(),
        max_diff: 0.1,
        diff_indices: vec![0, 1, 2, 3, 4],
        source_format: Format::Gguf,
        target_format: Format::Apr,
        backend: Backend::Cpu,
        failure_type: None,
        quant_type: None,
    };
    assert_eq!(evidence.diff_indices.len(), 5);
}

#[test]
fn test_round_trip_test_full_cycle() {
    let rt = RoundTripTest::new(
        vec![Format::Gguf, Format::Apr, Format::SafeTensors, Format::Gguf],
        Backend::Cpu,
        ModelId::new("test", "model"),
    );
    assert_eq!(rt.formats.len(), 4);
    assert_eq!(rt.formats[0], Format::Gguf);
    assert_eq!(rt.formats[3], Format::Gguf);
}

#[test]
fn test_conversion_config_clone_equality() {
    let config1 = ConversionConfig::default();
    let config2 = config1.clone();
    assert_eq!(config1.test_all_pairs, config2.test_all_pairs);
    assert_eq!(config1.test_round_trips, config2.test_round_trips);
    assert_eq!(config1.no_gpu, config2.no_gpu);
    assert_eq!(config1.backends.len(), config2.backends.len());
}

#[test]
fn test_generate_conversion_tests_contains_all_backends() {
    let model_id = ModelId::new("test", "model");
    let tests = generate_conversion_tests(&model_id);

    let cpu_backend_present = tests.iter().any(|t| t.backend == Backend::Cpu);
    let gpu_backend_present = tests.iter().any(|t| t.backend == Backend::Gpu);

    assert!(cpu_backend_present);
    assert!(gpu_backend_present);
}

#[test]
fn test_garbage_patterns_constant() {
    assert!(!GARBAGE_PATTERNS.is_empty());
    assert!(GARBAGE_PATTERNS.contains(&"PAD"));
    assert!(GARBAGE_PATTERNS.contains(&"<pad>"));
}
