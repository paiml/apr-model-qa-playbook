#[test]
fn test_file_integrity_with_hash_prefix_config() {
    let dir = TempDir::new().expect("create temp dir");
    // Simulate pacha cache: <hash>.safetensors + <hash>.config.json
    create_named_config(dir.path(), "abc123.config.json", 24, 896, 151_936);
    create_named_safetensors(dir.path(), "abc123.safetensors", 24, 896, 151_936);

    let model_file = dir.path().join("abc123.safetensors");
    let result = check_safetensors_file_integrity(&model_file);
    assert!(
        result.passed,
        "Should pass with hash-prefixed config: {:?}",
        result.errors
    );
    assert!(result.config_found);
    assert!(result.layer_count_match);
}

#[test]
fn test_file_integrity_ignores_other_models_in_shared_dir() {
    let dir = TempDir::new().expect("create temp dir");
    // Model A: 24 layers (the one we're checking)
    create_named_config(dir.path(), "aaa111.config.json", 24, 896, 151_936);
    create_named_safetensors(dir.path(), "aaa111.safetensors", 24, 896, 151_936);
    // Model B: 28 layers (different model in same dir — must be ignored)
    create_named_config(dir.path(), "bbb222.config.json", 28, 3584, 151_936);
    create_named_safetensors(dir.path(), "bbb222.safetensors", 28, 3584, 151_936);

    let model_file = dir.path().join("aaa111.safetensors");
    let result = check_safetensors_file_integrity(&model_file);
    assert!(
        result.passed,
        "Must use only aaa111's config and tensors, not bbb222's: {:?}",
        result.errors
    );
    assert_eq!(
        result.tensor_values.as_ref().unwrap().layer_count,
        Some(24),
        "Should see 24 layers from aaa111, not 28 from bbb222"
    );
}

#[test]
fn test_file_integrity_no_config_found() {
    let dir = TempDir::new().expect("create temp dir");
    // Safetensors file with no matching config
    create_named_safetensors(dir.path(), "orphan.safetensors", 12, 768, 30_000);

    let model_file = dir.path().join("orphan.safetensors");
    let result = check_safetensors_file_integrity(&model_file);
    assert!(!result.passed);
    assert!(!result.config_found);
    assert!(
        result
            .errors
            .iter()
            .any(|e| e.contains("G0-INTEGRITY-CONFIG"))
    );
}

#[test]
fn test_file_integrity_falls_back_to_plain_config() {
    let dir = TempDir::new().expect("create temp dir");
    // No hash-prefixed config, but plain config.json exists
    create_test_config(dir.path(), 24, 896, 151_936);
    create_named_safetensors(dir.path(), "model.safetensors", 24, 896, 151_936);

    let model_file = dir.path().join("model.safetensors");
    let result = check_safetensors_file_integrity(&model_file);
    assert!(
        result.passed,
        "Should fall back to config.json: {:?}",
        result.errors
    );
}

#[test]
fn test_file_integrity_layer_mismatch() {
    let dir = TempDir::new().expect("create temp dir");
    create_named_config(dir.path(), "bad.config.json", 14, 896, 151_936);
    create_named_safetensors(dir.path(), "bad.safetensors", 24, 896, 151_936);

    let model_file = dir.path().join("bad.safetensors");
    let result = check_safetensors_file_integrity(&model_file);
    assert!(!result.passed);
    assert!(!result.layer_count_match);
    assert!(result.errors.iter().any(|e| e.contains("LAYERS")));
}

#[test]
fn test_find_config_for_model_file_hash_prefix() {
    let dir = TempDir::new().expect("create temp dir");
    create_named_config(dir.path(), "d71534cb.config.json", 24, 896, 151_936);
    create_named_safetensors(dir.path(), "d71534cb.safetensors", 24, 896, 151_936);

    let result = find_config_for_model_file(&dir.path().join("d71534cb.safetensors"));
    assert!(result.is_some());
    assert!(
        result
            .unwrap()
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .contains("d71534cb.config.json")
    );
}

#[test]
fn test_find_config_for_model_file_no_match() {
    let dir = TempDir::new().expect("create temp dir");
    create_named_safetensors(dir.path(), "noconf.safetensors", 2, 768, 30_000);

    let result = find_config_for_model_file(&dir.path().join("noconf.safetensors"));
    assert!(result.is_none());
}
