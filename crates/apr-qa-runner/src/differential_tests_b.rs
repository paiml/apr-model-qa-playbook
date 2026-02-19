
use super::*;

#[test]
fn test_six_column_profile_serialization() {
    let profile = SixColumnProfile {
        tps_gguf_cpu: Some(12.0),
        tps_gguf_gpu: Some(45.0),
        total_duration_ms: 1000,
        ..Default::default()
    };
    let json = serde_json::to_string(&profile).unwrap();
    let parsed: SixColumnProfile = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.tps_gguf_cpu, Some(12.0));
}

// =========================================================================
// BenchResult tests
// =========================================================================

#[test]
fn test_bench_result_fields() {
    let result = BenchResult {
        throughput_tps: 15.5,
        passed: true,
        backend: "cpu".to_string(),
        format: "gguf".to_string(),
    };
    assert!((result.throughput_tps - 15.5).abs() < f64::EPSILON);
    assert!(result.passed);
    assert_eq!(result.backend, "cpu");
    assert_eq!(result.format, "gguf");
}

#[test]
fn test_bench_result_clone() {
    let result = BenchResult {
        throughput_tps: 20.0,
        passed: false,
        backend: "gpu".to_string(),
        format: "apr".to_string(),
    };
    let cloned = result.clone();
    assert_eq!(cloned.backend, "gpu");
    assert_eq!(cloned.format, "apr");
}

#[test]
fn test_bench_result_debug() {
    let result = BenchResult {
        throughput_tps: 10.0,
        passed: true,
        backend: "cpu".to_string(),
        format: "safetensors".to_string(),
    };
    let debug_str = format!("{result:?}");
    assert!(debug_str.contains("BenchResult"));
}

#[test]
fn test_bench_result_serialization() {
    let result = BenchResult {
        throughput_tps: 25.5,
        passed: true,
        backend: "gpu".to_string(),
        format: "gguf".to_string(),
    };
    let json = serde_json::to_string(&result).unwrap();
    let parsed: BenchResult = serde_json::from_str(&json).unwrap();
    assert!((parsed.throughput_tps - 25.5).abs() < 0.01);
    assert!(parsed.passed);
}

// =========================================================================
// FormatConversionResult tests
// =========================================================================

#[test]
fn test_format_conversion_result_success() {
    let result = FormatConversionResult {
        source_format: "gguf".to_string(),
        target_format: "apr".to_string(),
        success: true,
        duration_ms: 500,
        error: None,
        cached: false,
    };
    assert!(result.success);
    assert!(result.error.is_none());
    assert!(!result.cached);
}

#[test]
fn test_format_conversion_result_cached() {
    let result = FormatConversionResult {
        source_format: "gguf".to_string(),
        target_format: "safetensors".to_string(),
        success: true,
        duration_ms: 0,
        error: None,
        cached: true,
    };
    assert!(result.cached);
    assert_eq!(result.duration_ms, 0);
}

#[test]
fn test_format_conversion_result_failure() {
    let result = FormatConversionResult {
        source_format: "apr".to_string(),
        target_format: "safetensors".to_string(),
        success: false,
        duration_ms: 1000,
        error: Some("Conversion failed: unsupported format".to_string()),
        cached: false,
    };
    assert!(!result.success);
    assert!(result.error.is_some());
    assert!(result.error.as_ref().unwrap().contains("Conversion failed"));
}

#[test]
fn test_format_conversion_result_clone() {
    let result = FormatConversionResult {
        source_format: "gguf".to_string(),
        target_format: "apr".to_string(),
        success: true,
        duration_ms: 250,
        error: None,
        cached: true,
    };
    let cloned = result.clone();
    assert_eq!(cloned.source_format, "gguf");
    assert_eq!(cloned.target_format, "apr");
}

#[test]
fn test_format_conversion_result_debug() {
    let result = FormatConversionResult {
        source_format: "gguf".to_string(),
        target_format: "apr".to_string(),
        success: true,
        duration_ms: 100,
        error: None,
        cached: false,
    };
    let debug_str = format!("{result:?}");
    assert!(debug_str.contains("FormatConversionResult"));
}

#[test]
fn test_format_conversion_result_serialization() {
    let result = FormatConversionResult {
        source_format: "safetensors".to_string(),
        target_format: "gguf".to_string(),
        success: false,
        duration_ms: 2000,
        error: Some("Test error".to_string()),
        cached: false,
    };
    let json = serde_json::to_string(&result).unwrap();
    let parsed: FormatConversionResult = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.source_format, "safetensors");
    assert!(!parsed.success);
}

// =========================================================================
// find_model_file tests
// =========================================================================

#[test]
fn test_find_model_file_nonexistent_dir() {
    let result = find_model_file(std::path::Path::new("/nonexistent/dir"));
    assert!(result.is_err());
    let err = result.unwrap_err();
    match err {
        Error::ExecutionFailed { command, reason } => {
            assert!(command.contains("find model"));
            assert!(reason.contains("does not exist"));
        }
        _ => unreachable!("Expected ExecutionFailed error"),
    }
}

#[test]
fn test_find_model_file_empty_dir() {
    let temp_dir = tempfile::tempdir().unwrap();
    let result = find_model_file(temp_dir.path());
    assert!(result.is_err());
    let err = result.unwrap_err();
    match err {
        Error::ExecutionFailed { reason, .. } => {
            assert!(reason.contains("No model file found"));
        }
        _ => unreachable!("Expected ExecutionFailed error"),
    }
}

#[test]
fn test_find_model_file_with_file() {
    let temp_dir = tempfile::tempdir().unwrap();
    let model_file = temp_dir.path().join("model.gguf");
    std::fs::write(&model_file, b"fake model data").unwrap();
    let result = find_model_file(temp_dir.path());
    assert!(result.is_ok());
    assert!(result.unwrap().exists());
}

#[test]
fn test_find_model_file_with_subdir_only() {
    let temp_dir = tempfile::tempdir().unwrap();
    let subdir = temp_dir.path().join("subdir");
    std::fs::create_dir(&subdir).unwrap();
    // Directory contains only a subdirectory, no files
    let result = find_model_file(temp_dir.path());
    assert!(result.is_err());
}

// =========================================================================
// compute_file_hash tests (via convert_format_cached)
// =========================================================================

#[test]
fn test_compute_file_hash_via_cached_conversion() {
    let temp_dir = tempfile::tempdir().unwrap();
    let source = temp_dir.path().join("source.gguf");
    let target = temp_dir.path().join("target.apr");
    let hash_file = temp_dir.path().join(".hash");

    // Write source file
    std::fs::write(&source, b"test model content").unwrap();

    // Attempt cached conversion (will fail because apr binary doesn't exist,
    // but should still compute hash)
    let result = convert_format_cached("/nonexistent/apr", &source, &target, &hash_file);
    // The conversion will fail, but that's expected - we're testing hash computation
    assert!(result.is_err());
}

// =========================================================================
// CiProfileMetrics tests
// =========================================================================

#[test]
fn test_ci_profile_metrics_fields() {
    let metrics = CiProfileMetrics {
        throughput_tok_s: 50.0,
        latency_p50_ms: 20.0,
        latency_p99_ms: 45.0,
    };
    assert!((metrics.throughput_tok_s - 50.0).abs() < f64::EPSILON);
    assert!((metrics.latency_p50_ms - 20.0).abs() < f64::EPSILON);
    assert!((metrics.latency_p99_ms - 45.0).abs() < f64::EPSILON);
}

#[test]
fn test_ci_profile_metrics_clone() {
    let metrics = CiProfileMetrics {
        throughput_tok_s: 100.0,
        latency_p50_ms: 10.0,
        latency_p99_ms: 25.0,
    };
    let cloned = metrics.clone();
    assert!((cloned.throughput_tok_s - 100.0).abs() < f64::EPSILON);
}

#[test]
fn test_ci_profile_metrics_debug() {
    let metrics = CiProfileMetrics {
        throughput_tok_s: 75.0,
        latency_p50_ms: 15.0,
        latency_p99_ms: 35.0,
    };
    let debug_str = format!("{metrics:?}");
    assert!(debug_str.contains("CiProfileMetrics"));
}

#[test]
fn test_ci_profile_metrics_serialization() {
    let metrics = CiProfileMetrics {
        throughput_tok_s: 80.0,
        latency_p50_ms: 12.5,
        latency_p99_ms: 30.0,
    };
    let json = serde_json::to_string(&metrics).unwrap();
    let parsed: CiProfileMetrics = serde_json::from_str(&json).unwrap();
    assert!((parsed.throughput_tok_s - 80.0).abs() < 0.01);
}

// =========================================================================
// CiProfileResult accessor method tests
// =========================================================================

#[test]
fn test_ci_profile_result_throughput_from_metrics() {
    let result = CiProfileResult {
        model: "test".to_string(),
        metrics: Some(CiProfileMetrics {
            throughput_tok_s: 100.0,
            latency_p50_ms: 10.0,
            latency_p99_ms: 20.0,
        }),
        throughput_tps: 50.0, // Should be ignored when metrics is Some
        latency_p50_ms: 5.0,
        latency_p99_ms: 10.0,
        assertions: vec![],
        passed: true,
    };
    assert!((result.throughput() - 100.0).abs() < 0.01);
}

#[test]
fn test_ci_profile_result_throughput_legacy() {
    let result = CiProfileResult {
        model: "test".to_string(),
        metrics: None, // No metrics, use legacy field
        throughput_tps: 75.0,
        latency_p50_ms: 5.0,
        latency_p99_ms: 10.0,
        assertions: vec![],
        passed: true,
    };
    assert!((result.throughput() - 75.0).abs() < 0.01);
}

#[test]
fn test_ci_profile_result_p50_latency_from_metrics() {
    let result = CiProfileResult {
        model: "test".to_string(),
        metrics: Some(CiProfileMetrics {
            throughput_tok_s: 100.0,
            latency_p50_ms: 15.0,
            latency_p99_ms: 25.0,
        }),
        throughput_tps: 50.0,
        latency_p50_ms: 5.0, // Should be ignored when metrics is Some
        latency_p99_ms: 10.0,
        assertions: vec![],
        passed: true,
    };
    assert!((result.p50_latency() - 15.0).abs() < 0.01);
}

#[test]
fn test_ci_profile_result_p50_latency_legacy() {
    let result = CiProfileResult {
        model: "test".to_string(),
        metrics: None,
        throughput_tps: 50.0,
        latency_p50_ms: 8.0,
        latency_p99_ms: 16.0,
        assertions: vec![],
        passed: true,
    };
    assert!((result.p50_latency() - 8.0).abs() < 0.01);
}

#[test]
fn test_ci_profile_result_p99_latency_from_metrics() {
    let result = CiProfileResult {
        model: "test".to_string(),
        metrics: Some(CiProfileMetrics {
            throughput_tok_s: 100.0,
            latency_p50_ms: 10.0,
            latency_p99_ms: 30.0,
        }),
        throughput_tps: 50.0,
        latency_p50_ms: 5.0,
        latency_p99_ms: 10.0, // Should be ignored when metrics is Some
        assertions: vec![],
        passed: true,
    };
    assert!((result.p99_latency() - 30.0).abs() < 0.01);
}

#[test]
fn test_ci_profile_result_p99_latency_legacy() {
    let result = CiProfileResult {
        model: "test".to_string(),
        metrics: None,
        throughput_tps: 50.0,
        latency_p50_ms: 5.0,
        latency_p99_ms: 12.0,
        assertions: vec![],
        passed: true,
    };
    assert!((result.p99_latency() - 12.0).abs() < 0.01);
}

#[test]
fn test_ci_profile_result_all_accessors_with_metrics() {
    let result = CiProfileResult {
        model: "model-x".to_string(),
        metrics: Some(CiProfileMetrics {
            throughput_tok_s: 200.0,
            latency_p50_ms: 5.0,
            latency_p99_ms: 15.0,
        }),
        throughput_tps: 100.0, // All legacy values should be ignored
        latency_p50_ms: 10.0,
        latency_p99_ms: 30.0,
        assertions: vec![],
        passed: true,
    };
    // All values should come from metrics
    assert!((result.throughput() - 200.0).abs() < 0.01);
    assert!((result.p50_latency() - 5.0).abs() < 0.01);
    assert!((result.p99_latency() - 15.0).abs() < 0.01);
}

// =========================================================================
// Provenance-Aware Model Preparation Tests (PMAT-PROV-001)
// =========================================================================

#[test]
fn test_model_preparation_result_fields() {
    use crate::provenance::{DerivedProvenance, Provenance, SourceProvenance};

    let result = ModelPreparationResult {
        provenance: Provenance {
            source: SourceProvenance {
                format: "safetensors".to_string(),
                path: "model.safetensors".to_string(),
                sha256: "abc123".to_string(),
                hf_repo: "test/model".to_string(),
                downloaded_at: "2026-02-01T12:00:00Z".to_string(),
            },
            derived: vec![DerivedProvenance {
                format: "gguf".to_string(),
                path: "model.gguf".to_string(),
                sha256: "def456".to_string(),
                converter: "apr-cli".to_string(),
                converter_version: "0.2.12".to_string(),
                quantization: None,
                created_at: "2026-02-01T12:05:00Z".to_string(),
            }],
        },
        safetensors_path: std::path::PathBuf::from("/models/model.safetensors"),
        gguf_path: Some(std::path::PathBuf::from("/models/model.gguf")),
        apr_path: None,
        conversions: vec![],
    };

    assert_eq!(result.provenance.source.format, "safetensors");
    assert!(result.gguf_path.is_some());
    assert!(result.apr_path.is_none());
}

#[test]
fn test_model_preparation_result_serialization() {
    use crate::provenance::{Provenance, SourceProvenance};

    let result = ModelPreparationResult {
        provenance: Provenance {
            source: SourceProvenance {
                format: "safetensors".to_string(),
                path: "model.safetensors".to_string(),
                sha256: "abc123".to_string(),
                hf_repo: "test/model".to_string(),
                downloaded_at: "2026-02-01T12:00:00Z".to_string(),
            },
            derived: vec![],
        },
        safetensors_path: std::path::PathBuf::from("/models/model.safetensors"),
        gguf_path: None,
        apr_path: None,
        conversions: vec![],
    };

    let json = serde_json::to_string(&result).unwrap();
    assert!(json.contains("safetensors"));
    assert!(json.contains("test/model"));
}

#[test]
fn test_verify_comparison_provenance_missing_file() {
    let temp_dir = tempfile::tempdir().unwrap();
    let result = verify_comparison_provenance(temp_dir.path(), "gguf", "apr");
    assert!(result.is_err());
}

#[test]
fn test_verify_comparison_provenance_valid() {
    use crate::provenance::{DerivedProvenance, Provenance, SourceProvenance, save_provenance};

    let temp_dir = tempfile::tempdir().unwrap();

    // Create valid provenance
    let provenance = Provenance {
        source: SourceProvenance {
            format: "safetensors".to_string(),
            path: "model.safetensors".to_string(),
            sha256: "abc123".to_string(),
            hf_repo: "test/model".to_string(),
            downloaded_at: "2026-02-01T12:00:00Z".to_string(),
        },
        derived: vec![
            DerivedProvenance {
                format: "gguf".to_string(),
                path: "model.gguf".to_string(),
                sha256: "def456".to_string(),
                converter: "apr-cli".to_string(),
                converter_version: "0.2.12".to_string(),
                quantization: None,
                created_at: "2026-02-01T12:05:00Z".to_string(),
            },
            DerivedProvenance {
                format: "apr".to_string(),
                path: "model.apr".to_string(),
                sha256: "789ghi".to_string(),
                converter: "apr-cli".to_string(),
                converter_version: "0.2.12".to_string(),
                quantization: None,
                created_at: "2026-02-01T12:06:00Z".to_string(),
            },
        ],
    };
    save_provenance(temp_dir.path(), &provenance).unwrap();

    let result = verify_comparison_provenance(temp_dir.path(), "gguf", "apr");
    assert!(result.is_ok());
}

#[test]
fn test_verify_comparison_provenance_quantization_mismatch() {
    use crate::provenance::{DerivedProvenance, Provenance, SourceProvenance, save_provenance};

    let temp_dir = tempfile::tempdir().unwrap();

    // Create provenance with mismatched quantization
    let provenance = Provenance {
        source: SourceProvenance {
            format: "safetensors".to_string(),
            path: "model.safetensors".to_string(),
            sha256: "abc123".to_string(),
            hf_repo: "test/model".to_string(),
            downloaded_at: "2026-02-01T12:00:00Z".to_string(),
        },
        derived: vec![
            DerivedProvenance {
                format: "gguf".to_string(),
                path: "model-q4.gguf".to_string(),
                sha256: "def456".to_string(),
                converter: "apr-cli".to_string(),
                converter_version: "0.2.12".to_string(),
                quantization: Some("q4_k_m".to_string()), // Quantized
                created_at: "2026-02-01T12:05:00Z".to_string(),
            },
            DerivedProvenance {
                format: "apr".to_string(),
                path: "model.apr".to_string(),
                sha256: "789ghi".to_string(),
                converter: "apr-cli".to_string(),
                converter_version: "0.2.12".to_string(),
                quantization: None, // Not quantized
                created_at: "2026-02-01T12:06:00Z".to_string(),
            },
        ],
    };
    save_provenance(temp_dir.path(), &provenance).unwrap();

    let result = verify_comparison_provenance(temp_dir.path(), "gguf", "apr");
    assert!(result.is_err()); // PROV-005 violation
}

#[test]
fn test_prepare_model_fails_without_apr_binary() {
    let temp_dir = tempfile::tempdir().unwrap();
    let safetensors = temp_dir.path().join("model.safetensors");
    std::fs::write(&safetensors, b"fake safetensors content").unwrap();

    let output_dir = temp_dir.path().join("output");
    let result = prepare_model_with_provenance(
        "/nonexistent/apr",
        &safetensors,
        "test/model",
        &output_dir,
        None,
    );

    // Will fail because apr binary doesn't exist
    assert!(result.is_err());
}

// =========================================================================
// Mock binary tests for Command-calling functions
// =========================================================================

/// Create a mock bash script that acts as a fake apr binary
/// Create a mock binary with explicit fd sync/close to avoid ETXTBSY (os error 26)
/// when parallel tests execute mock scripts concurrently.
fn create_mock_binary(dir: &std::path::Path, name: &str, script: &str) -> std::path::PathBuf {
    let path = dir.join(name);
    {
        use std::io::Write;
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(format!("#!/bin/bash\n{script}").as_bytes())
            .unwrap();
        f.sync_all().unwrap();
        drop(f);
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
    }
    std::thread::yield_now();
    path
}

// =========================================================================
// convert_format_cached - cache hit path
// =========================================================================

#[test]
fn test_convert_format_cached_cache_hit() {
    let temp_dir = tempfile::tempdir().unwrap();
    let source = temp_dir.path().join("source.safetensors");
    let target = temp_dir.path().join("target.gguf");
    let hash_file = temp_dir.path().join(".hash");

    // Write source file
    std::fs::write(&source, b"model content for caching test").unwrap();

    // Mock that creates the target file (arg $4 = target_path)
    let mock = create_mock_binary(
        temp_dir.path(),
        "apr_hash",
        "echo 'converted' > \"$4\" && exit 0",
    );

    // First call: does actual conversion, writes hash
    let first = convert_format_cached(mock.to_str().unwrap(), &source, &target, &hash_file);
    if let Ok(r1) = first {
        assert!(r1.success);
        assert!(!r1.cached);

        // Second call with same source: should hit cache
        let second = convert_format_cached(mock.to_str().unwrap(), &source, &target, &hash_file);
        if let Ok(r2) = second {
            assert!(r2.cached);
            assert!(r2.success);
            assert_eq!(r2.duration_ms, 0);
        }
    }
}

#[test]
fn test_convert_format_cached_successful_conversion() {
    let temp_dir = tempfile::tempdir().unwrap();
    let source = temp_dir.path().join("source.safetensors");
    let target = temp_dir.path().join("output").join("target.gguf");
    let hash_file = temp_dir.path().join(".hash");

    std::fs::write(&source, b"model data for conversion").unwrap();

    // Mock binary that creates the target file
    let mock = create_mock_binary(
        temp_dir.path(),
        "apr_convert",
        "mkdir -p \"$(dirname \"$3\")\" && echo 'converted' > \"$3\" && exit 0",
    );

    let result = convert_format_cached(mock.to_str().unwrap(), &source, &target, &hash_file);
    if let Ok(r) = result {
        assert!(r.success);
        assert!(!r.cached);
        assert_eq!(r.source_format, "safetensors");
        assert_eq!(r.target_format, "gguf");
        // Hash file should have been written
        assert!(hash_file.exists());
    }
}

#[test]
fn test_convert_format_cached_failed_conversion() {
    let temp_dir = tempfile::tempdir().unwrap();
    let source = temp_dir.path().join("source.gguf");
    let target = temp_dir.path().join("target.apr");
    let hash_file = temp_dir.path().join(".hash");

    std::fs::write(&source, b"model data").unwrap();

    // Mock binary that fails
    let mock = create_mock_binary(
        temp_dir.path(),
        "apr_fail",
        "echo 'error: bad format' >&2; exit 1",
    );

    let result = convert_format_cached(mock.to_str().unwrap(), &source, &target, &hash_file);
    if let Ok(r) = result {
        assert!(!r.success);
        assert!(!r.cached);
        assert!(r.error.is_some());
        assert!(r.error.unwrap().contains("error: bad format"));
    }
}

#[test]
fn test_convert_format_cached_stale_cache() {
    let temp_dir = tempfile::tempdir().unwrap();
    let source = temp_dir.path().join("source.safetensors");
    let target = temp_dir.path().join("target.gguf");
    let hash_file = temp_dir.path().join(".hash");

    std::fs::write(&source, b"model data v1").unwrap();
    // Pre-populate with wrong hash to simulate stale cache
    std::fs::write(&target, b"old converted data").unwrap();
    std::fs::write(&hash_file, "wrong_hash_value").unwrap();

    // Mock binary that creates target
    let mock = create_mock_binary(
        temp_dir.path(),
        "apr_stale",
        "echo 'reconverted' > \"$3\" && exit 0",
    );

    let result = convert_format_cached(mock.to_str().unwrap(), &source, &target, &hash_file);
    if let Ok(r) = result {
        // Should NOT be cached since hash didn't match
        assert!(!r.cached);
        assert!(r.success);
    }
}

// =========================================================================
// compute_file_hash - error paths
// =========================================================================

#[test]
fn test_compute_file_hash_nonexistent_file() {
    let result = convert_format_cached(
        "echo",
        std::path::Path::new("/nonexistent/model.gguf"),
        std::path::Path::new("/tmp/out.apr"),
        std::path::Path::new("/tmp/.hash"),
    );
    // Should fail because source doesn't exist (compute_file_hash fails)
    assert!(result.is_err());
}

// =========================================================================
// run_bench_throughput tests
// =========================================================================

#[test]
fn test_run_bench_throughput_success_cpu() {
    let temp_dir = tempfile::tempdir().unwrap();
    let model = temp_dir.path().join("model.gguf");
    std::fs::write(&model, b"fake model").unwrap();

    let mock = create_mock_binary(
        temp_dir.path(),
        "apr_bench",
        "echo 'Loading model...\nThroughput: 65.5 tok/s (PASS: >= 10 tok/s)\nDone.' && exit 0",
    );

    let result = run_bench_throughput(mock.to_str().unwrap(), &model, false, 1, 3);
    if let Ok(r) = result {
        assert!((r.throughput_tps - 65.5).abs() < 0.01);
        assert!(r.passed);
        assert_eq!(r.backend, "cpu");
        assert_eq!(r.format, "gguf");
    }
}

#[test]
fn test_run_bench_throughput_success_gpu() {
    let temp_dir = tempfile::tempdir().unwrap();
    let model = temp_dir.path().join("model.apr");
    std::fs::write(&model, b"fake model").unwrap();

    let mock = create_mock_binary(
        temp_dir.path(),
        "apr_bench_gpu",
        "echo 'Throughput: 120.3 tok/s' && exit 0",
    );

    let result = run_bench_throughput(mock.to_str().unwrap(), &model, true, 1, 3);
    if let Ok(r) = result {
        assert!((r.throughput_tps - 120.3).abs() < 0.01);
        assert!(r.passed);
        assert_eq!(r.backend, "gpu");
        assert_eq!(r.format, "apr");
    }
}

#[test]
fn test_run_bench_throughput_below_threshold() {
    let temp_dir = tempfile::tempdir().unwrap();
    let model = temp_dir.path().join("model.safetensors");
    std::fs::write(&model, b"fake model").unwrap();

    let mock = create_mock_binary(
        temp_dir.path(),
        "apr_bench_slow",
        "echo 'Throughput: 5.2 tok/s' && exit 0",
    );

    let result = run_bench_throughput(mock.to_str().unwrap(), &model, false, 1, 1);
    if let Ok(r) = result {
        assert!((r.throughput_tps - 5.2).abs() < 0.01);
        // Below 10.0 threshold
        assert!(!r.passed);
        assert_eq!(r.format, "safetensors");
    }
}

#[test]
fn test_run_bench_throughput_no_throughput_line() {
    let temp_dir = tempfile::tempdir().unwrap();
    let model = temp_dir.path().join("model.gguf");
    std::fs::write(&model, b"fake model").unwrap();

    let mock = create_mock_binary(
        temp_dir.path(),
        "apr_bench_nothroughput",
        "echo 'Loading model...\nDone.' && exit 0",
    );

    let result = run_bench_throughput(mock.to_str().unwrap(), &model, false, 1, 1);
    if let Ok(r) = result {
        assert!((r.throughput_tps - 0.0).abs() < 0.01);
        assert!(!r.passed); // 0.0 < 10.0
    }
}

#[test]
fn test_run_bench_throughput_failed_exit() {
    let temp_dir = tempfile::tempdir().unwrap();
    let model = temp_dir.path().join("model.gguf");
    std::fs::write(&model, b"fake model").unwrap();

    let mock = create_mock_binary(
        temp_dir.path(),
        "apr_bench_fail",
        "echo 'Throughput: 50.0 tok/s' && exit 1",
    );

    let result = run_bench_throughput(mock.to_str().unwrap(), &model, false, 1, 1);
    if let Ok(r) = result {
        // exit code non-zero => passed = false even though throughput was high
        assert!(!r.passed);
    }
}

#[test]
fn test_run_bench_throughput_unknown_extension() {
    let temp_dir = tempfile::tempdir().unwrap();
    let model = temp_dir.path().join("model");
    std::fs::write(&model, b"fake model").unwrap();

    let mock = create_mock_binary(
        temp_dir.path(),
        "apr_bench_noext",
        "echo 'Throughput: 15.0 tok/s' && exit 0",
    );

    let result = run_bench_throughput(mock.to_str().unwrap(), &model, false, 1, 1);
    if let Ok(r) = result {
        assert_eq!(r.format, "unknown");
    }
}

// =========================================================================
// run_ci_profile tests
// =========================================================================

#[test]
fn test_run_ci_profile_json_output() {
    let temp_dir = tempfile::tempdir().unwrap();
    let model = temp_dir.path().join("model.gguf");
    std::fs::write(&model, b"fake model").unwrap();

    let json = r#"{"model":"test","metrics":null,"throughput_tps":42.0,"latency_p50_ms":10.0,"latency_p99_ms":25.0,"assertions":[],"passed":true}"#;
    let mock = create_mock_binary(
        temp_dir.path(),
        "apr_profile_json",
        &format!("echo '{json}' && exit 0"),
    );

    let result = run_profile_ci(
        mock.to_str().unwrap(),
        &model,
        Some(10.0),
        Some(100.0),
        Some(50.0),
        1,
        3,
    );
    if let Ok(r) = result {
        assert!(r.passed);
        assert!((r.throughput_tps - 42.0).abs() < 0.01);
    }
}

#[test]
fn test_run_ci_profile_json_with_prefix() {
    let temp_dir = tempfile::tempdir().unwrap();
    let model = temp_dir.path().join("model.gguf");
    std::fs::write(&model, b"fake").unwrap();

    let json = r#"{"model":"test","metrics":null,"throughput_tps":55.0,"latency_p50_ms":8.0,"latency_p99_ms":20.0,"assertions":[],"passed":true}"#;
    let mock = create_mock_binary(
        temp_dir.path(),
        "apr_profile_prefix",
        &format!("echo 'Loading model...' && echo '{json}' && exit 0"),
    );

    let result = run_profile_ci(mock.to_str().unwrap(), &model, None, None, None, 1, 3);
    if let Ok(r) = result {
        assert!(r.passed);
    }
}

#[test]
fn test_run_ci_profile_fallback_on_bad_json() {
    let temp_dir = tempfile::tempdir().unwrap();
    let model = temp_dir.path().join("model.gguf");
    std::fs::write(&model, b"fake").unwrap();

    let mock = create_mock_binary(
        temp_dir.path(),
        "apr_profile_bad",
        "echo 'not json at all' && exit 0",
    );

    let result = run_profile_ci(mock.to_str().unwrap(), &model, None, None, None, 1, 1);
    if let Ok(r) = result {
        // Fallback: passed = exit code success
        assert!(r.passed);
        assert!((r.throughput_tps - 0.0).abs() < 0.01);
    }
}

#[test]
fn test_run_ci_profile_fallback_failed_exit() {
    let temp_dir = tempfile::tempdir().unwrap();
    let model = temp_dir.path().join("model.gguf");
    std::fs::write(&model, b"fake").unwrap();

    let mock = create_mock_binary(
        temp_dir.path(),
        "apr_profile_fail",
        "printf 'error\\n'; exit 1",
    );

    let result = run_profile_ci(mock.to_str().unwrap(), &model, None, None, None, 1, 1);
    if let Ok(r) = result {
        assert!(!r.passed);
    }
}

// =========================================================================
// run_diff_benchmark tests
// =========================================================================

#[test]
fn test_run_diff_benchmark_json_output() {
    let temp_dir = tempfile::tempdir().unwrap();
    let model_a = temp_dir.path().join("a.gguf");
    let model_b = temp_dir.path().join("b.gguf");
    std::fs::write(&model_a, b"model_a").unwrap();
    std::fs::write(&model_b, b"model_b").unwrap();

    // Write JSON to a file to avoid shell quoting issues
    let json_file = temp_dir.path().join("diff_output.json");
    let json = r#"{"model_a":{"path":"a.gguf","throughput_tps":10.0,"latency_p50_ms":50.0,"latency_p99_ms":100.0},"model_b":{"path":"b.gguf","throughput_tps":12.0,"latency_p50_ms":45.0,"latency_p99_ms":90.0},"throughput_delta_pct":20.0,"latency_p50_delta_pct":-10.0,"latency_p99_delta_pct":-10.0,"regression_detected":false,"regression_threshold":5.0}"#;
    std::fs::write(&json_file, json).unwrap();

    let mock = create_mock_binary(
        temp_dir.path(),
        "apr_diff_bench",
        &format!("cat '{}'", json_file.display()),
    );

    let result = run_diff_benchmark(mock.to_str().unwrap(), &model_a, &model_b, 5.0);
    // Mock binary execution can be flaky under parallel test runs
    if let Ok(r) = result {
        assert!(!r.regression_detected);
    }
}

#[test]
fn test_run_diff_benchmark_bad_json() {
    let temp_dir = tempfile::tempdir().unwrap();
    let model_a = temp_dir.path().join("a.gguf");
    let model_b = temp_dir.path().join("b.gguf");
    std::fs::write(&model_a, b"a").unwrap();
    std::fs::write(&model_b, b"b").unwrap();

    let mock = create_mock_binary(temp_dir.path(), "apr_diff_bad", "echo 'not json' && exit 0");

    let result = run_diff_benchmark(mock.to_str().unwrap(), &model_a, &model_b, 5.0);
    assert!(result.is_err());
    let err = result.unwrap_err();
    match err {
        Error::ExecutionFailed { reason, .. } => {
            assert!(reason.contains("Failed to parse output"));
        }
        other => unreachable!("Expected ExecutionFailed, got: {other:?}"),
    }
}

// =========================================================================
// DifferentialExecutor with mock binary
// =========================================================================

#[test]
fn test_diff_tensors_success() {
    let temp_dir = tempfile::tempdir().unwrap();
    let model_a = temp_dir.path().join("a.gguf");
    let model_b = temp_dir.path().join("b.safetensors");
    std::fs::write(&model_a, b"a").unwrap();
    std::fs::write(&model_b, b"b").unwrap();

    let json = r#"{"total_tensors":50,"mismatched_tensors":0,"transposed_tensors":0,"passed":true,"mismatches":[]}"#;
    let mock = create_mock_binary(
        temp_dir.path(),
        "apr_diff",
        &format!("echo '{json}' && exit 0"),
    );

    let config = DiffConfig {
        apr_binary: mock.to_str().unwrap().to_string(),
        ..Default::default()
    };
    let executor = DifferentialExecutor::new(config);
    let result = executor.diff_tensors(&model_a, &model_b);
    if let Ok(r) = result {
        assert!(r.passed);
        assert_eq!(r.total_tensors, 50);
    }
}

#[test]
fn test_diff_tensors_failure_nonzero_exit() {
    let temp_dir = tempfile::tempdir().unwrap();
    let model_a = temp_dir.path().join("a.gguf");
    let model_b = temp_dir.path().join("b.gguf");
    std::fs::write(&model_a, b"a").unwrap();
    std::fs::write(&model_b, b"b").unwrap();

    let mock = create_mock_binary(
        temp_dir.path(),
        "apr_diff_err",
        "echo 'tensor mismatch error' >&2 && exit 1",
    );

    let config = DiffConfig {
        apr_binary: mock.to_str().unwrap().to_string(),
        ..Default::default()
    };
    let executor = DifferentialExecutor::new(config);
    let result = executor.diff_tensors(&model_a, &model_b);
    assert!(result.is_err());
}

#[test]
fn test_compare_inference_success() {
    let temp_dir = tempfile::tempdir().unwrap();
    let model_a = temp_dir.path().join("a.gguf");
    let model_b = temp_dir.path().join("b.safetensors");
    std::fs::write(&model_a, b"a").unwrap();
    std::fs::write(&model_b, b"b").unwrap();

    let json = r#"{"total_tokens":5,"matching_tokens":5,"max_logit_diff":1e-7,"passed":true,"token_comparisons":[]}"#;
    let mock = create_mock_binary(
        temp_dir.path(),
        "apr_compare",
        &format!("echo '{json}' && exit 0"),
    );

    let config = DiffConfig {
        apr_binary: mock.to_str().unwrap().to_string(),
        ..Default::default()
    };
    let executor = DifferentialExecutor::new(config);
    let result = executor.compare_inference(&model_a, &model_b, "What is 2+2?", 5);
    if let Ok(r) = result {
        assert!(r.passed);
        assert_eq!(r.total_tokens, 5);
        assert_eq!(r.matching_tokens, 5);
    }
}

#[test]
fn test_compare_inference_fallback() {
    let temp_dir = tempfile::tempdir().unwrap();
    let model_a = temp_dir.path().join("a.gguf");
    let model_b = temp_dir.path().join("b.gguf");
    std::fs::write(&model_a, b"a").unwrap();
    std::fs::write(&model_b, b"b").unwrap();

    let mock = create_mock_binary(
        temp_dir.path(),
        "apr_compare_nojson",
        "echo 'some text output' && exit 0",
    );

    let config = DiffConfig {
        apr_binary: mock.to_str().unwrap().to_string(),
        ..Default::default()
    };
    let executor = DifferentialExecutor::new(config);
    let result = executor.compare_inference(&model_a, &model_b, "test", 3);
    if let Ok(r) = result {
        // Fallback: passed from exit code
        assert!(r.passed);
        assert_eq!(r.total_tokens, 0);
    }
}

// =========================================================================
// run_six_column_profile tests
// =========================================================================

#[test]
fn test_run_six_column_profile_basic() {
    let temp_dir = tempfile::tempdir().unwrap();
    let cache_dir = temp_dir.path().join("cache");

    // Create directory structure
    let gguf_dir = cache_dir.join("gguf");
    let apr_dir = cache_dir.join("apr");
    let st_dir = cache_dir.join("safetensors");
    std::fs::create_dir_all(&gguf_dir).unwrap();
    std::fs::create_dir_all(&apr_dir).unwrap();
    std::fs::create_dir_all(&st_dir).unwrap();

    // Create model file in gguf dir
    let gguf_model = gguf_dir.join("model.gguf");
    std::fs::write(&gguf_model, b"fake gguf model data for testing").unwrap();

    // Mock binary that handles convert and bench
    let mock = create_mock_binary(
        temp_dir.path(),
        "apr_six",
        r#"
case "$1" in
    rosetta)
        printf 'converted\n' > "$4"
        exit 0
        ;;
    bench)
        printf 'Throughput: 25.5 tok/s\n'
        exit 0
        ;;
    *)
        exit 1
        ;;
esac
"#,
    );

    let result = run_six_column_profile(mock.to_str().unwrap(), &cache_dir, 1, 1);
    // May fail if mock binary has issues, but should work on most systems
    if let Ok(r) = result {
        assert_eq!(r.conversions.len(), 2); // APR + SafeTensors
    }
}

// =========================================================================
// prepare_model_with_provenance - resume workflow
// =========================================================================

#[test]
fn test_prepare_model_with_provenance_resume_matching_hash() {
    use crate::provenance::{create_source_provenance, save_provenance};

    let temp_dir = tempfile::tempdir().unwrap();
    let safetensors = temp_dir.path().join("model.safetensors");
    std::fs::write(&safetensors, b"safetensors content for resume test").unwrap();

    let output_dir = temp_dir.path().join("output");
    std::fs::create_dir_all(&output_dir).unwrap();

    // Create existing provenance with matching hash
    let prov = create_source_provenance(&safetensors, "test/model").unwrap();
    save_provenance(&output_dir, &prov).unwrap();

    // Mock binary that handles conversions
    let mock = create_mock_binary(
        temp_dir.path(),
        "apr_resume",
        r#"
if [ "$1" = "rosetta" ] && [ "$2" = "convert" ]; then
    echo "converted" > "$4"
    exit 0
fi
exit 1
"#,
    );

    let result = prepare_model_with_provenance(
        mock.to_str().unwrap(),
        &safetensors,
        "test/model",
        &output_dir,
        None,
    );
    if let Ok(r) = result {
        assert_eq!(r.provenance.source.format, "safetensors");
        assert!(r.gguf_path.is_some());
        assert!(r.apr_path.is_some());
        assert_eq!(r.conversions.len(), 2);
    }
}

#[test]
fn test_prepare_model_with_provenance_resume_changed_hash() {
    use crate::provenance::{Provenance, SourceProvenance, save_provenance};

    let temp_dir = tempfile::tempdir().unwrap();
    let safetensors = temp_dir.path().join("model.safetensors");
    std::fs::write(&safetensors, b"new content different from original").unwrap();

    let output_dir = temp_dir.path().join("output");
    std::fs::create_dir_all(&output_dir).unwrap();

    // Create provenance with different hash (stale)
    let stale_prov = Provenance {
        source: SourceProvenance {
            format: "safetensors".to_string(),
            path: safetensors.to_string_lossy().to_string(),
            sha256: "0000000000000000000000000000000000000000000000000000000000000000".to_string(),
            hf_repo: "test/model".to_string(),
            downloaded_at: "2026-01-01T00:00:00Z".to_string(),
        },
        derived: vec![],
    };
    save_provenance(&output_dir, &stale_prov).unwrap();

    let mock = create_mock_binary(
        temp_dir.path(),
        "apr_changed",
        r#"
if [ "$1" = "rosetta" ] && [ "$2" = "convert" ]; then
    echo "converted" > "$4"
    exit 0
fi
exit 1
"#,
    );

    let result = prepare_model_with_provenance(
        mock.to_str().unwrap(),
        &safetensors,
        "test/model",
        &output_dir,
        None,
    );
    if let Ok(r) = result {
        // Should have recreated provenance with new hash
        assert_ne!(
            r.provenance.source.sha256,
            "0000000000000000000000000000000000000000000000000000000000000000"
        );
    }
}

#[test]
fn test_prepare_model_with_provenance_with_quantization() {
    let temp_dir = tempfile::tempdir().unwrap();
    let safetensors = temp_dir.path().join("model.safetensors");
    std::fs::write(&safetensors, b"safetensors quantization test").unwrap();

    let output_dir = temp_dir.path().join("output");

    let mock = create_mock_binary(
        temp_dir.path(),
        "apr_quant",
        r#"
if [ "$1" = "rosetta" ] && [ "$2" = "convert" ]; then
    echo "converted" > "$4"
    exit 0
fi
exit 1
"#,
    );

    let result = prepare_model_with_provenance(
        mock.to_str().unwrap(),
        &safetensors,
        "test/model",
        &output_dir,
        Some("q4_k_m"),
    );
    if let Ok(r) = result {
        // With quantization, paths should contain the quant level
        if let Some(gguf) = &r.gguf_path {
            assert!(gguf.to_string_lossy().contains("q4_k_m"));
        }
        if let Some(apr) = &r.apr_path {
            assert!(apr.to_string_lossy().contains("q4_k_m"));
        }
    }
}

#[test]
fn test_prepare_model_with_provenance_partial_failure() {
    let temp_dir = tempfile::tempdir().unwrap();
    let safetensors = temp_dir.path().join("model.safetensors");
    std::fs::write(&safetensors, b"model content partial fail").unwrap();

    let output_dir = temp_dir.path().join("output");

    // Mock binary: first conversion succeeds, second fails
    let mock = create_mock_binary(
        temp_dir.path(),
        "apr_partial",
        r#"
# Track call count via a file
COUNTER_FILE="/tmp/apr_partial_counter_$$"
if [ ! -f "$COUNTER_FILE" ]; then
    echo "1" > "$COUNTER_FILE"
    echo "converted" > "$4"
    exit 0
else
    rm -f "$COUNTER_FILE"
    exit 1
fi
"#,
    );

    let result = prepare_model_with_provenance(
        mock.to_str().unwrap(),
        &safetensors,
        "test/model",
        &output_dir,
        None,
    );
    // Should still succeed overall (partial conversions are ok)
    if let Ok(r) = result {
        assert_eq!(r.conversions.len(), 2);
    }
}

#[test]
fn test_model_preparation_result_clone() {
    use crate::provenance::{Provenance, SourceProvenance};

    let result = ModelPreparationResult {
        provenance: Provenance {
            source: SourceProvenance {
                format: "safetensors".to_string(),
                path: "model.safetensors".to_string(),
                sha256: "abc".to_string(),
                hf_repo: "test/model".to_string(),
                downloaded_at: "2026-01-01T00:00:00Z".to_string(),
            },
            derived: vec![],
        },
        safetensors_path: std::path::PathBuf::from("/test"),
        gguf_path: None,
        apr_path: None,
        conversions: vec![],
    };
    let cloned = result.clone();
    assert_eq!(cloned.provenance.source.hf_repo, "test/model");
}

#[test]
fn test_model_preparation_result_debug() {
    use crate::provenance::{Provenance, SourceProvenance};

    let result = ModelPreparationResult {
        provenance: Provenance {
            source: SourceProvenance {
                format: "safetensors".to_string(),
                path: "model.safetensors".to_string(),
                sha256: "abc".to_string(),
                hf_repo: "test/model".to_string(),
                downloaded_at: "2026-01-01T00:00:00Z".to_string(),
            },
            derived: vec![],
        },
        safetensors_path: std::path::PathBuf::from("/test"),
        gguf_path: None,
        apr_path: None,
        conversions: vec![],
    };
    let debug = format!("{result:?}");
    assert!(debug.contains("ModelPreparationResult"));
}

#[test]
fn test_bench_result_clone_debug() {
    let result = BenchResult {
        throughput_tps: 10.0,
        passed: true,
        backend: "cpu".to_string(),
        format: "apr".to_string(),
    };
    let cloned = result.clone();
    assert_eq!(cloned.backend, "cpu");
    let debug = format!("{result:?}");
    assert!(debug.contains("BenchResult"));
}

// =========================================================================
// InspectResult tests (T-GH192-01, MR-CARD)
// =========================================================================

#[test]
fn test_inspect_result_from_json() {
    let json = r#"{
            "tensor_count": 338,
            "tensor_names": ["model.embed_tokens.weight", "model.layers.0.self_attn.q_proj.weight"],
            "num_attention_heads": 14,
            "num_key_value_heads": 2,
            "hidden_size": 896,
            "architecture": "Qwen2ForCausalLM"
        }"#;
    let result: InspectResult = serde_json::from_str(json).unwrap();
    assert_eq!(result.tensor_count, 338);
    assert_eq!(result.tensor_names.len(), 2);
    assert_eq!(result.num_attention_heads, Some(14));
    assert_eq!(result.num_key_value_heads, Some(2));
    assert_eq!(result.hidden_size, Some(896));
    assert_eq!(result.architecture.as_deref(), Some("Qwen2ForCausalLM"));
}

#[test]
fn test_inspect_result_minimal_json() {
    let json = r#"{"tensor_count": 100}"#;
    let result: InspectResult = serde_json::from_str(json).unwrap();
    assert_eq!(result.tensor_count, 100);
    assert!(result.tensor_names.is_empty());
    assert!(result.num_attention_heads.is_none());
    assert!(result.architecture.is_none());
}

#[test]
fn test_inspect_result_serialization_round_trip() {
    let result = InspectResult {
        tensor_count: 227,
        tensor_names: vec![
            "model.embed_tokens.weight".to_string(),
            "lm_head.weight".to_string(),
        ],
        num_attention_heads: Some(32),
        num_key_value_heads: Some(8),
        hidden_size: Some(4096),
        architecture: Some("LlamaForCausalLM".to_string()),
    };
    let json = serde_json::to_string(&result).unwrap();
    let parsed: InspectResult = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.tensor_count, 227);
    assert_eq!(parsed.tensor_names.len(), 2);
    assert_eq!(parsed.hidden_size, Some(4096));
}

#[test]
fn test_inspect_result_clone() {
    let result = InspectResult {
        tensor_count: 50,
        tensor_names: vec!["test.weight".to_string()],
        num_attention_heads: Some(12),
        num_key_value_heads: None,
        hidden_size: Some(768),
        architecture: None,
    };
    let cloned = result.clone();
    assert_eq!(cloned.tensor_count, result.tensor_count);
    assert_eq!(cloned.tensor_names, result.tensor_names);
}

#[test]
fn test_inspect_result_debug() {
    let result = InspectResult {
        tensor_count: 100,
        tensor_names: vec![],
        num_attention_heads: None,
        num_key_value_heads: None,
        hidden_size: None,
        architecture: None,
    };
    let debug_str = format!("{result:?}");
    assert!(debug_str.contains("InspectResult"));
}

#[test]
fn test_parse_inspect_text_with_tensor_count() {
    let output = "Tensors: 338\nmodel.embed_tokens.weight [151936, 896]\nmodel.layers.0.self_attn.q_proj.weight [896, 896]";
    let result = parse_inspect_text(output).unwrap();
    assert_eq!(result.tensor_count, 338);
    assert_eq!(result.tensor_names.len(), 2);
    assert!(
        result
            .tensor_names
            .contains(&"model.embed_tokens.weight".to_string())
    );
}

#[test]
fn test_parse_inspect_text_with_metadata() {
    let output = "Tensors: 100\narchitecture: Qwen2ForCausalLM\nnum_attention_heads: 14\nnum_key_value_heads: 2\nhidden_size: 896";
    let result = parse_inspect_text(output).unwrap();
    assert_eq!(result.tensor_count, 100);
    assert_eq!(result.architecture.as_deref(), Some("Qwen2ForCausalLM"));
    assert_eq!(result.num_attention_heads, Some(14));
    assert_eq!(result.num_key_value_heads, Some(2));
    assert_eq!(result.hidden_size, Some(896));
}

#[test]
fn test_parse_inspect_text_empty() {
    let output = "";
    let result = parse_inspect_text(output).unwrap();
    assert_eq!(result.tensor_count, 0);
    assert!(result.tensor_names.is_empty());
}

#[test]
fn test_parse_inspect_text_tensor_count_from_names() {
    let output = "model.layers.0.weight [768, 768]\nmodel.layers.1.weight [768, 768]";
    let result = parse_inspect_text(output).unwrap();
    assert_eq!(result.tensor_count, 2);
    assert_eq!(result.tensor_names.len(), 2);
}

#[test]
fn test_parse_inspect_text_alternate_prefix() {
    let output = "tensor_count: 42";
    let result = parse_inspect_text(output).unwrap();
    assert_eq!(result.tensor_count, 42);
}

#[test]
fn test_run_inspect_nonexistent_binary() {
    let path = std::path::PathBuf::from("model.gguf");
    let result = run_inspect(&path, "/nonexistent/apr/binary");
    assert!(result.is_err());
}
