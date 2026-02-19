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
    let mut test =
        ConversionTest::new(Format::Gguf, Format::Apr, Backend::Cpu, model_id).with_output_dir(out);
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
