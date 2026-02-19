#[test]
fn test_is_2d_shape() {
    assert!(is_2d_shape("[vocab, hidden]"));
    assert!(is_2d_shape("[hidden, vocab]"));
    assert!(!is_2d_shape("[hidden]"));
    assert!(!is_2d_shape("[a, b, c]"));
}

#[test]
fn test_parse_shape_dims() {
    let dims = parse_shape_dims("[vocab, hidden]");
    assert_eq!(dims, vec!["vocab", "hidden"]);

    let dims = parse_shape_dims("[hidden]");
    assert_eq!(dims, vec!["hidden"]);
}

#[test]
fn test_load_contract_missing_file() {
    let result = load_contract_from("/nonexistent/path.yaml");
    assert!(result.is_err());
}

#[test]
fn test_validate_model_missing_file() {
    // Create a minimal contract for testing
    let contract = TensorLayoutContract {
        metadata: ContractMetadata {
            version: "1.0".to_string(),
            created: "2026-01-01".to_string(),
            updated: "2026-01-01".to_string(),
            author: "test".to_string(),
            description: "test".to_string(),
        },
        formats: HashMap::new(),
        kernel: KernelConvention {
            signature: "test".to_string(),
            weight_shape: "[out, in]".to_string(),
            computation: "y = Wx".to_string(),
            byte_calculation: "out * in".to_string(),
            block_sizes: HashMap::new(),
            qk_k: 256,
        },
        tensors: HashMap::new(),
        validation_rules: vec![],
        semantic_validation: None,
    };

    let result = validate_model(Path::new("/nonexistent/model.apr"), &contract);
    assert!(result.is_ok());
    let result = result.unwrap();
    assert!(!result.passed);
    assert!(!result.critical_failures.is_empty());
}

#[test]
fn test_get_critical_tensors() {
    let mut tensors = HashMap::new();
    tensors.insert(
        "lm_head".to_string(),
        TensorSpec {
            gguf_name: "output.weight".to_string(),
            apr_name: "lm_head.weight".to_string(),
            gguf_shape: "[hidden, vocab]".to_string(),
            apr_shape: "[vocab, hidden]".to_string(),
            transpose: true,
            kernel: "matmul".to_string(),
            kernel_out_dim: Some("vocab_size".to_string()),
            kernel_in_dim: Some("hidden_dim".to_string()),
            validation: None,
            critical: true,
            note: Some("GH-202".to_string()),
        },
    );
    tensors.insert(
        "embedding".to_string(),
        TensorSpec {
            gguf_name: "token_embd.weight".to_string(),
            apr_name: "model.embed_tokens.weight".to_string(),
            gguf_shape: "[hidden, vocab]".to_string(),
            apr_shape: "[vocab, hidden]".to_string(),
            transpose: true,
            kernel: "lookup".to_string(),
            kernel_out_dim: None,
            kernel_in_dim: None,
            validation: None,
            critical: false,
            note: None,
        },
    );

    let contract = TensorLayoutContract {
        metadata: ContractMetadata {
            version: "1.0".to_string(),
            created: "2026-01-01".to_string(),
            updated: "2026-01-01".to_string(),
            author: "test".to_string(),
            description: "test".to_string(),
        },
        formats: HashMap::new(),
        kernel: KernelConvention {
            signature: "test".to_string(),
            weight_shape: "[out, in]".to_string(),
            computation: "y = Wx".to_string(),
            byte_calculation: "out * in".to_string(),
            block_sizes: HashMap::new(),
            qk_k: 256,
        },
        tensors,
        validation_rules: vec![],
        semantic_validation: None,
    };

    let critical = get_critical_tensors(&contract);
    assert_eq!(critical.len(), 1);
    assert_eq!(critical[0].apr_name, "lm_head.weight");
}

// ========================================================================
// Helper: create a minimal TensorLayoutContract for testing
// ========================================================================

fn make_contract() -> TensorLayoutContract {
    TensorLayoutContract {
        metadata: ContractMetadata {
            version: "1.0".to_string(),
            created: "2026-01-01".to_string(),
            updated: "2026-01-01".to_string(),
            author: "test".to_string(),
            description: "test".to_string(),
        },
        formats: HashMap::new(),
        kernel: KernelConvention {
            signature: "test".to_string(),
            weight_shape: "[out, in]".to_string(),
            computation: "y = Wx".to_string(),
            byte_calculation: "out * in".to_string(),
            block_sizes: HashMap::new(),
            qk_k: 256,
        },
        tensors: HashMap::new(),
        validation_rules: vec![],
        semantic_validation: None,
    }
}

fn make_spec(apr_name: &str, apr_shape: &str, transpose: bool) -> TensorSpec {
    TensorSpec {
        gguf_name: "test".to_string(),
        apr_name: apr_name.to_string(),
        gguf_shape: "[x, y]".to_string(),
        apr_shape: apr_shape.to_string(),
        transpose,
        kernel: "matmul".to_string(),
        kernel_out_dim: None,
        kernel_in_dim: None,
        validation: None,
        critical: false,
        note: None,
    }
}

fn make_config_full() -> LayoutModelConfig {
    LayoutModelConfig {
        vocab_size: Some(32000),
        hidden_size: Some(4096),
        intermediate_size: Some(11008),
        num_attention_heads: Some(32),
        num_key_value_heads: Some(8),
        num_hidden_layers: Some(2),
    }
}

/// Create a minimal SafeTensors file with the given tensor name/shape pairs.
fn create_test_safetensors(path: &Path, tensors: &[(&str, &[usize])]) {
    use std::io::Write;
    let mut header = serde_json::Map::new();
    header.insert(
        "__metadata__".to_string(),
        serde_json::json!({"format": "pt"}),
    );
    let mut offset = 0usize;
    for (name, shape) in tensors {
        let num_elements: usize = shape.iter().product();
        let byte_size = num_elements * 4; // f32
        header.insert(
            name.to_string(),
            serde_json::json!({
                "dtype": "F32",
                "shape": shape,
                "data_offsets": [offset, offset + byte_size]
            }),
        );
        offset += byte_size;
    }
    let header_json = serde_json::to_string(&header).unwrap();
    let header_bytes = header_json.as_bytes();
    let header_len = header_bytes.len() as u64;
    let mut file = std::fs::File::create(path).unwrap();
    file.write_all(&header_len.to_le_bytes()).unwrap();
    file.write_all(header_bytes).unwrap();
    file.write_all(&vec![0u8; offset]).unwrap();
}

// ========================================================================
// 1. get_usize
// ========================================================================

#[test]
fn test_get_usize_valid() {
    let json = serde_json::json!({"vocab_size": 32000});
    assert_eq!(get_usize(&json, "vocab_size"), Some(32000));
}

#[test]
fn test_get_usize_missing() {
    let json = serde_json::json!({"vocab_size": 32000});
    assert_eq!(get_usize(&json, "hidden_size"), None);
}

#[test]
fn test_get_usize_not_number() {
    let json = serde_json::json!({"vocab_size": "not_a_number"});
    assert_eq!(get_usize(&json, "vocab_size"), None);
}

// ========================================================================
// 2. resolve_dimension
// ========================================================================

#[test]
fn test_resolve_dimension_vocab() {
    let config = make_config_full();
    assert_eq!(resolve_dimension("vocab", &config), Some(32000));
    assert_eq!(resolve_dimension("vocab_size", &config), Some(32000));
}

#[test]
fn test_resolve_dimension_hidden() {
    let config = make_config_full();
    assert_eq!(resolve_dimension("hidden", &config), Some(4096));
    assert_eq!(resolve_dimension("hidden_dim", &config), Some(4096));
    assert_eq!(resolve_dimension("hidden_size", &config), Some(4096));
}

#[test]
fn test_resolve_dimension_intermediate() {
    let config = make_config_full();
    assert_eq!(resolve_dimension("intermediate", &config), Some(11008));
    assert_eq!(resolve_dimension("intermediate_dim", &config), Some(11008));
    assert_eq!(resolve_dimension("intermediate_size", &config), Some(11008));
}

#[test]
fn test_resolve_dimension_heads() {
    let config = make_config_full();
    assert_eq!(resolve_dimension("heads", &config), Some(32));
    assert_eq!(resolve_dimension("num_heads", &config), Some(32));
    assert_eq!(resolve_dimension("num_attention_heads", &config), Some(32));
}

#[test]
fn test_resolve_dimension_kv_heads() {
    let config = make_config_full();
    assert_eq!(resolve_dimension("kv_heads", &config), Some(8));
    assert_eq!(resolve_dimension("num_kv_heads", &config), Some(8));
    assert_eq!(resolve_dimension("num_key_value_heads", &config), Some(8));
}

#[test]
fn test_resolve_dimension_head_dim() {
    let config = make_config_full();
    // head_dim = hidden_size / num_attention_heads = 4096 / 32 = 128
    assert_eq!(resolve_dimension("head_dim", &config), Some(128));
}

#[test]
fn test_resolve_dimension_head_dim_zero_heads() {
    let config = LayoutModelConfig {
        hidden_size: Some(4096),
        num_attention_heads: Some(0),
        ..LayoutModelConfig::default()
    };
    // Division guard: n == 0 => None
    assert_eq!(resolve_dimension("head_dim", &config), None);
}

#[test]
fn test_resolve_dimension_head_dim_missing_fields() {
    // Missing hidden_size
    let config = LayoutModelConfig {
        num_attention_heads: Some(32),
        ..LayoutModelConfig::default()
    };
    assert_eq!(resolve_dimension("head_dim", &config), None);

    // Missing num_attention_heads
    let config2 = LayoutModelConfig {
        hidden_size: Some(4096),
        ..LayoutModelConfig::default()
    };
    assert_eq!(resolve_dimension("head_dim", &config2), None);
}

#[test]
fn test_resolve_dimension_numeric() {
    let config = LayoutModelConfig::default();
    assert_eq!(resolve_dimension("128", &config), Some(128));
    assert_eq!(resolve_dimension("0", &config), Some(0));
}

#[test]
fn test_resolve_dimension_expression_heads_times_head_dim() {
    let config = make_config_full();
    // heads * head_dim = 32 * (4096/32) = 32 * 128 = 4096
    assert_eq!(resolve_dimension("heads*head_dim", &config), Some(32 * 128));
}

#[test]
fn test_resolve_dimension_expression_kv_heads_times_head_dim() {
    let config = make_config_full();
    // kv_heads * head_dim = 8 * 128 = 1024
    assert_eq!(
        resolve_dimension("kv_heads*head_dim", &config),
        Some(8 * 128)
    );
}

#[test]
fn test_resolve_dimension_expression_with_missing() {
    let config = LayoutModelConfig::default();
    // heads * head_dim => None since both are missing
    assert_eq!(resolve_dimension("heads*head_dim", &config), None);
}

#[test]
fn test_resolve_dimension_unknown() {
    let config = LayoutModelConfig::default();
    assert_eq!(resolve_dimension("foobar", &config), None);
}

#[test]
fn test_resolve_dimension_expression_triple_star() {
    // "a*b*c" => 3 parts, not 2, so None
    let config = make_config_full();
    assert_eq!(resolve_dimension("heads*head_dim*kv_heads", &config), None);
}

// ========================================================================
// 3. parse_expected_shape
// ========================================================================

#[test]
fn test_parse_expected_shape_valid() {
    let config = make_config_full();
    let result = parse_expected_shape("[vocab, hidden]", &config);
    assert_eq!(result, Some((32000, 4096)));
}

#[test]
fn test_parse_expected_shape_incomplete() {
    // vocab resolves, but "unknown_dim" does not => None
    let config = make_config_full();
    let result = parse_expected_shape("[vocab, unknown_dim]", &config);
    assert_eq!(result, None);
}

#[test]
fn test_parse_expected_shape_non_2d() {
    let config = make_config_full();
    // Single dim
    let result = parse_expected_shape("[hidden]", &config);
    assert_eq!(result, None);
    // 3 dims
    let result = parse_expected_shape("[a, b, c]", &config);
    assert_eq!(result, None);
}

#[test]
fn test_parse_expected_shape_with_expression() {
    let config = make_config_full();
    // "[heads*head_dim, hidden]" => (4096, 4096)
    let result = parse_expected_shape("[heads*head_dim, hidden]", &config);
    assert_eq!(result, Some((4096, 4096)));
}

// ========================================================================
// 4. validate_lm_head_shape
// ========================================================================

#[test]
fn test_validate_lm_head_shape_not_2d() {
    let config = make_config_full();
    let contract = make_contract();
    let result = validate_lm_head_shape(&[4096], &config, &contract);
    assert!(!result.passed);
    assert_eq!(result.rule_id, "F-LAYOUT-CONTRACT-002");
    assert!(result.details.contains("must be 2D"));
}

#[test]
fn test_validate_lm_head_shape_valid() {
    let config = make_config_full();
    let contract = make_contract();
    let result = validate_lm_head_shape(&[32000, 4096], &config, &contract);
    assert!(result.passed);
    assert!(result.details.contains("shape correct"));
}

#[test]
fn test_validate_lm_head_shape_invalid() {
    let config = make_config_full();
    let contract = make_contract();
    // Transposed: [hidden, vocab] instead of [vocab, hidden]
    let result = validate_lm_head_shape(&[4096, 32000], &config, &contract);
    assert!(!result.passed);
    assert!(result.details.contains("MISMATCH"));
}

#[test]
fn test_validate_lm_head_shape_partial_vocab_only() {
    let config = LayoutModelConfig {
        vocab_size: Some(32000),
        ..LayoutModelConfig::default()
    };
    let contract = make_contract();
    // Only vocab known, dim[0] matches
    let result = validate_lm_head_shape(&[32000, 9999], &config, &contract);
    assert!(result.passed);
    // dim[0] doesn't match vocab
    let result = validate_lm_head_shape(&[9999, 4096], &config, &contract);
    assert!(!result.passed);
}

#[test]
fn test_validate_lm_head_shape_partial_hidden_only() {
    let config = LayoutModelConfig {
        hidden_size: Some(4096),
        ..LayoutModelConfig::default()
    };
    let contract = make_contract();
    // Only hidden known, dim[1] matches
    let result = validate_lm_head_shape(&[9999, 4096], &config, &contract);
    assert!(result.passed);
    // dim[1] doesn't match hidden
    let result = validate_lm_head_shape(&[32000, 9999], &config, &contract);
    assert!(!result.passed);
}
