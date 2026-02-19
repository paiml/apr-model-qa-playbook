    use super::*;

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

    #[test]
    fn test_validate_lm_head_shape_no_config() {
        let config = LayoutModelConfig::default();
        let contract = make_contract();
        // No config => can't validate => passes
        let result = validate_lm_head_shape(&[100, 200], &config, &contract);
        assert!(result.passed);
    }

    // ========================================================================
    // 5. validate_2d_tensor_shape
    // ========================================================================

    #[test]
    fn test_validate_2d_tensor_shape_not_2d() {
        let spec = make_spec("test.weight", "[vocab, hidden]", true);
        let config = make_config_full();
        let result = validate_2d_tensor_shape("test", &[4096], &spec, &config);
        assert!(!result.passed);
        assert_eq!(result.rule_id, "F-LAYOUT-CONTRACT-001");
        assert!(result.details.contains("must be 2D"));
    }

    #[test]
    fn test_validate_2d_tensor_shape_valid() {
        let spec = make_spec("test.weight", "[vocab, hidden]", true);
        let config = make_config_full();
        let result = validate_2d_tensor_shape("test", &[32000, 4096], &spec, &config);
        assert!(result.passed);
        assert!(result.details.contains("shape correct"));
    }

    #[test]
    fn test_validate_2d_tensor_shape_invalid() {
        let spec = make_spec("test.weight", "[vocab, hidden]", true);
        let config = make_config_full();
        // Wrong shape
        let result = validate_2d_tensor_shape("test", &[4096, 32000], &spec, &config);
        assert!(!result.passed);
        assert!(result.details.contains("mismatch"));
    }

    #[test]
    fn test_validate_2d_tensor_shape_unresolvable() {
        // When shape dims can't be resolved, validation passes by default
        let spec = make_spec("test.weight", "[unknown1, unknown2]", true);
        let config = LayoutModelConfig::default();
        let result = validate_2d_tensor_shape("test", &[100, 200], &spec, &config);
        assert!(result.passed);
    }

    // ========================================================================
    // 6. validate_1d_tensor_shape
    // ========================================================================

    #[test]
    fn test_validate_1d_tensor_shape_not_1d() {
        let spec = make_spec("test.bias", "[hidden]", false);
        let config = make_config_full();
        let result = validate_1d_tensor_shape("test.bias", &[4096, 100], &spec, &config);
        assert!(!result.passed);
        assert_eq!(result.rule_id, "F-LAYOUT-CONTRACT-003");
        assert!(result.details.contains("must be 1D"));
    }

    #[test]
    fn test_validate_1d_tensor_shape_valid() {
        let spec = make_spec("test.bias", "[hidden]", false);
        let config = make_config_full();
        let result = validate_1d_tensor_shape("test.bias", &[4096], &spec, &config);
        assert!(result.passed);
        assert!(result.details.contains("shape correct"));
    }

    #[test]
    fn test_validate_1d_tensor_shape_invalid() {
        let spec = make_spec("test.bias", "[hidden]", false);
        let config = make_config_full();
        let result = validate_1d_tensor_shape("test.bias", &[9999], &spec, &config);
        assert!(!result.passed);
        assert!(result.details.contains("shape mismatch"));
    }

    #[test]
    fn test_validate_1d_tensor_shape_no_config() {
        let spec = make_spec("test.bias", "[hidden]", false);
        let config = LayoutModelConfig::default();
        // No hidden_size => passes by default
        let result = validate_1d_tensor_shape("test.bias", &[9999], &spec, &config);
        assert!(result.passed);
    }

    // ========================================================================
    // 7. find_safetensors_files
    // ========================================================================

    #[test]
    fn test_find_safetensors_files_single_file() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("model.safetensors");
        create_test_safetensors(&file_path, &[("x", &[2, 3])]);

        let files = find_safetensors_files(&file_path);
        assert_eq!(files.len(), 1);
        assert_eq!(files[0], file_path);
    }

    #[test]
    fn test_find_safetensors_files_directory() {
        let dir = tempfile::tempdir().unwrap();
        create_test_safetensors(
            &dir.path().join("model-00001-of-00002.safetensors"),
            &[("a", &[2, 3])],
        );
        create_test_safetensors(
            &dir.path().join("model-00002-of-00002.safetensors"),
            &[("b", &[4, 5])],
        );

        let files = find_safetensors_files(dir.path());
        assert_eq!(files.len(), 2);
    }

    #[test]
    fn test_find_safetensors_files_subdir() {
        let dir = tempfile::tempdir().unwrap();
        let st_dir = dir.path().join("safetensors");
        std::fs::create_dir_all(&st_dir).unwrap();
        create_test_safetensors(&st_dir.join("model.safetensors"), &[("x", &[2])]);

        let files = find_safetensors_files(dir.path());
        assert_eq!(files.len(), 1);
    }

    #[test]
    fn test_find_safetensors_files_no_files() {
        let dir = tempfile::tempdir().unwrap();
        let files = find_safetensors_files(dir.path());
        assert!(files.is_empty());
    }

    #[test]
    fn test_find_safetensors_files_non_safetensors_file() {
        let dir = tempfile::tempdir().unwrap();
        let not_st = dir.path().join("model.gguf");
        std::fs::write(&not_st, b"not a safetensors file").unwrap();

        // Passed as file path
        let files = find_safetensors_files(&not_st);
        assert!(files.is_empty());

        // Passed as directory containing only .gguf
        let files = find_safetensors_files(dir.path());
        assert!(files.is_empty());
    }

    // ========================================================================
    // 8. read_safetensors_metadata
    // ========================================================================

    #[test]
    fn test_read_safetensors_metadata_valid() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("model.safetensors");
        create_test_safetensors(
            &file_path,
            &[
                ("lm_head.weight", &[32000, 4096]),
                ("embed_tokens.weight", &[32000, 4096]),
            ],
        );

        let metadata = read_safetensors_metadata(&file_path).unwrap();
        assert_eq!(metadata.len(), 2);
        assert_eq!(metadata["lm_head.weight"], vec![32000, 4096]);
        assert_eq!(metadata["embed_tokens.weight"], vec![32000, 4096]);
    }

    #[test]
    fn test_read_safetensors_metadata_invalid_json() {
        use std::io::Write;
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("bad.safetensors");

        let bad_header = b"this is not json{{{{";
        let header_len = bad_header.len() as u64;
        let mut file = std::fs::File::create(&file_path).unwrap();
        file.write_all(&header_len.to_le_bytes()).unwrap();
        file.write_all(bad_header).unwrap();

        let result = read_safetensors_metadata(&file_path);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("JSON parse error"));
    }

    #[test]
    fn test_read_safetensors_metadata_header_too_large() {
        use std::io::Write;
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("huge.safetensors");

        // Write a header_len that exceeds MAX_HEADER_SIZE
        let huge_len: u64 = (MAX_HEADER_SIZE as u64) + 1;
        let mut file = std::fs::File::create(&file_path).unwrap();
        file.write_all(&huge_len.to_le_bytes()).unwrap();

        let result = read_safetensors_metadata(&file_path);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Header too large"));
    }

    #[test]
    fn test_read_safetensors_metadata_skips_metadata_key() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("model.safetensors");
        // Our helper inserts __metadata__ automatically
        create_test_safetensors(&file_path, &[("weight", &[10, 20])]);

        let metadata = read_safetensors_metadata(&file_path).unwrap();
        // __metadata__ should not appear
        assert!(!metadata.contains_key("__metadata__"));
        assert_eq!(metadata.len(), 1);
    }

    // ========================================================================
    // 9. find_and_load_config
    // ========================================================================

    #[test]
    fn test_find_and_load_config_directory() {
        let dir = tempfile::tempdir().unwrap();
        let config = serde_json::json!({
            "vocab_size": 32000,
            "hidden_size": 4096,
            "intermediate_size": 11008,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "num_hidden_layers": 24
        });
        std::fs::write(
            dir.path().join("config.json"),
            serde_json::to_string(&config).unwrap(),
        )
        .unwrap();

        let mc = find_and_load_config(dir.path());
        assert_eq!(mc.vocab_size, Some(32000));
        assert_eq!(mc.hidden_size, Some(4096));
        assert_eq!(mc.intermediate_size, Some(11008));
        assert_eq!(mc.num_attention_heads, Some(32));
        assert_eq!(mc.num_key_value_heads, Some(8));
        assert_eq!(mc.num_hidden_layers, Some(24));
    }

    #[test]
    fn test_find_and_load_config_file_mode() {
        let dir = tempfile::tempdir().unwrap();
        // Simulate file mode: model file is "model.safetensors", config is "config.json" in same dir
        let model_file = dir.path().join("model.safetensors");
        create_test_safetensors(&model_file, &[("x", &[2, 3])]);

        let config = serde_json::json!({
            "vocab_size": 50000,
            "hidden_size": 2048
        });
        std::fs::write(
            dir.path().join("config.json"),
            serde_json::to_string(&config).unwrap(),
        )
        .unwrap();

        let mc = find_and_load_config(&model_file);
        assert_eq!(mc.vocab_size, Some(50000));
        assert_eq!(mc.hidden_size, Some(2048));
    }

    #[test]
    fn test_find_and_load_config_file_mode_stem_prefix() {
        let dir = tempfile::tempdir().unwrap();
        let model_file = dir.path().join("mymodel.safetensors");
        create_test_safetensors(&model_file, &[("x", &[2])]);

        let config = serde_json::json!({"vocab_size": 12345});
        // Write stem-prefixed config: "mymodel.config.json"
        std::fs::write(
            dir.path().join("mymodel.config.json"),
            serde_json::to_string(&config).unwrap(),
        )
        .unwrap();

        let mc = find_and_load_config(&model_file);
        assert_eq!(mc.vocab_size, Some(12345));
    }

    #[test]
    fn test_find_and_load_config_missing() {
        let dir = tempfile::tempdir().unwrap();
        let mc = find_and_load_config(dir.path());
        assert_eq!(mc.vocab_size, None);
        assert_eq!(mc.hidden_size, None);
    }

    #[test]
    fn test_find_and_load_config_safetensors_subdir() {
        let dir = tempfile::tempdir().unwrap();
        let st_dir = dir.path().join("safetensors");
        std::fs::create_dir_all(&st_dir).unwrap();

        let config = serde_json::json!({"vocab_size": 99999});
        std::fs::write(
            st_dir.join("config.json"),
            serde_json::to_string(&config).unwrap(),
        )
        .unwrap();

        let mc = find_and_load_config(dir.path());
        assert_eq!(mc.vocab_size, Some(99999));
    }

    // ========================================================================
    // 10. validate_model with empty dir (no safetensors)
    // ========================================================================

    #[test]
    fn test_validate_model_empty_dir() {
        let dir = tempfile::tempdir().unwrap();
        let contract = make_contract();

        let result = validate_model(dir.path(), &contract).unwrap();
        // Empty dir -> passed=true, no rules checked
        assert!(result.passed);
        assert_eq!(result.rules_checked, 0);
        assert!(result.critical_failures.is_empty());
    }

    // ========================================================================
    // 11. validate_layer_tensors
    // ========================================================================

    #[test]
    fn test_validate_layer_tensors_with_layers() {
        let config = LayoutModelConfig {
            num_hidden_layers: Some(2),
            vocab_size: Some(32000),
            hidden_size: Some(4096),
            ..LayoutModelConfig::default()
        };

        let spec = make_spec(
            "model.layers.{n}.self_attn.q_proj.weight",
            "[vocab, hidden]",
            true,
        );

        let mut all_tensors = HashMap::new();
        all_tensors.insert(
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            vec![32000, 4096],
        );
        all_tensors.insert(
            "model.layers.1.self_attn.q_proj.weight".to_string(),
            vec![32000, 4096],
        );

        let mut results = Vec::new();
        validate_layer_tensors(
            "model.layers.{n}.self_attn.q_proj.weight",
            &all_tensors,
            &config,
            &spec,
            &mut results,
        );

        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|r| r.passed));
    }

    #[test]
    fn test_validate_layer_tensors_missing_layer() {
        let config = LayoutModelConfig {
            num_hidden_layers: Some(3),
            ..LayoutModelConfig::default()
        };

        let spec = make_spec("model.layers.{n}.weight", "[vocab, hidden]", true);

        let mut all_tensors = HashMap::new();
        // Only layer 0 exists; layers 1 and 2 missing => they are skipped
        all_tensors.insert("model.layers.0.weight".to_string(), vec![10, 20]);

        let mut results = Vec::new();
        validate_layer_tensors(
            "model.layers.{n}.weight",
            &all_tensors,
            &config,
            &spec,
            &mut results,
        );

        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_validate_layer_tensors_zero_layers() {
        let config = LayoutModelConfig::default(); // num_hidden_layers = None => 0
        let spec = make_spec("model.layers.{n}.weight", "[vocab, hidden]", true);
        let all_tensors = HashMap::new();
        let mut results = Vec::new();

        validate_layer_tensors(
            "model.layers.{n}.weight",
            &all_tensors,
            &config,
            &spec,
            &mut results,
        );

        assert!(results.is_empty());
    }

    // ========================================================================
    // 12. validate_1d_layer_tensors
    // ========================================================================

    #[test]
    fn test_validate_1d_layer_tensors_with_layers() {
        let config = LayoutModelConfig {
            num_hidden_layers: Some(2),
            hidden_size: Some(4096),
            ..LayoutModelConfig::default()
        };

        let spec = make_spec("model.layers.{n}.input_layernorm.weight", "[hidden]", false);

        let mut all_tensors = HashMap::new();
        all_tensors.insert(
            "model.layers.0.input_layernorm.weight".to_string(),
            vec![4096],
        );
        all_tensors.insert(
            "model.layers.1.input_layernorm.weight".to_string(),
            vec![4096],
        );

        let mut results = Vec::new();
        validate_1d_layer_tensors(
            "model.layers.{n}.input_layernorm.weight",
            &all_tensors,
            &config,
            &spec,
            &mut results,
        );

        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|r| r.passed));
    }

    #[test]
    fn test_validate_1d_layer_tensors_invalid_shape() {
        let config = LayoutModelConfig {
            num_hidden_layers: Some(1),
            hidden_size: Some(4096),
            ..LayoutModelConfig::default()
        };

        let spec = make_spec("model.layers.{n}.norm.weight", "[hidden]", false);

        let mut all_tensors = HashMap::new();
        all_tensors.insert("model.layers.0.norm.weight".to_string(), vec![9999]);

        let mut results = Vec::new();
        validate_1d_layer_tensors(
            "model.layers.{n}.norm.weight",
            &all_tensors,
            &config,
            &spec,
            &mut results,
        );

        assert_eq!(results.len(), 1);
        assert!(!results[0].passed);
    }

    // ========================================================================
    // 13. get_validation_rules
    // ========================================================================

    #[test]
    fn test_get_validation_rules_returns_rules() {
        let mut contract = make_contract();
        contract.validation_rules = vec![
            ValidationRule {
                id: "F-LAYOUT-CONTRACT-001".to_string(),
                name: "2D transpose".to_string(),
                description: "All 2D weights are transposed".to_string(),
                severity: "P0".to_string(),
                critical: true,
                reference: None,
            },
            ValidationRule {
                id: "F-LAYOUT-CONTRACT-002".to_string(),
                name: "lm_head shape".to_string(),
                description: "lm_head shape matches".to_string(),
                severity: "P0".to_string(),
                critical: true,
                reference: Some("GH-202".to_string()),
            },
        ];

        let rules = get_validation_rules(&contract);
        assert_eq!(rules.len(), 2);
        assert_eq!(rules[0].id, "F-LAYOUT-CONTRACT-001");
        assert_eq!(rules[1].id, "F-LAYOUT-CONTRACT-002");
    }

    #[test]
    fn test_get_validation_rules_empty() {
        let contract = make_contract();
        let rules = get_validation_rules(&contract);
        assert!(rules.is_empty());
    }

    // ========================================================================
    // 14. collect_tensor_metadata with parse error
    // ========================================================================

    #[test]
    fn test_collect_tensor_metadata_with_parse_error() {
        use std::io::Write;
        let dir = tempfile::tempdir().unwrap();

        // Create a corrupt safetensors file
        let bad_file = dir.path().join("corrupt.safetensors");
        let bad_header = b"not valid json at all";
        let header_len = bad_header.len() as u64;
        let mut file = std::fs::File::create(&bad_file).unwrap();
        file.write_all(&header_len.to_le_bytes()).unwrap();
        file.write_all(bad_header).unwrap();

        let mut results = Vec::new();
        let tensors = collect_tensor_metadata(dir.path(), &mut results);

        assert!(tensors.is_empty());
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].rule_id, "PARSE-ERROR");
        assert!(!results[0].passed);
    }

    #[test]
    fn test_collect_tensor_metadata_valid() {
        let dir = tempfile::tempdir().unwrap();
        create_test_safetensors(
            &dir.path().join("model.safetensors"),
            &[("weight.a", &[10, 20]), ("weight.b", &[30])],
        );

        let mut results = Vec::new();
        let tensors = collect_tensor_metadata(dir.path(), &mut results);

        assert!(results.is_empty());
        assert_eq!(tensors.len(), 2);
        assert_eq!(tensors["weight.a"], vec![10, 20]);
        assert_eq!(tensors["weight.b"], vec![30]);
    }

    // ========================================================================
    // 15. validate_lm_head (orchestrator) with/without lm_head in tensors
    // ========================================================================

    #[test]
    fn test_validate_lm_head_with_tensors() {
        let config = make_config_full();
        let contract = make_contract();

        let mut all_tensors = HashMap::new();
        all_tensors.insert("lm_head.weight".to_string(), vec![32000, 4096]);

        let mut results = Vec::new();
        let mut critical_failures = Vec::new();

        validate_lm_head(
            &all_tensors,
            &config,
            &contract,
            &mut results,
            &mut critical_failures,
        );

        assert_eq!(results.len(), 1);
        assert!(results[0].passed);
        assert!(critical_failures.is_empty());
    }

    #[test]
    fn test_validate_lm_head_with_invalid_tensors() {
        let config = make_config_full();
        let contract = make_contract();

        let mut all_tensors = HashMap::new();
        // Transposed shape => mismatch
        all_tensors.insert("lm_head.weight".to_string(), vec![4096, 32000]);

        let mut results = Vec::new();
        let mut critical_failures = Vec::new();

        validate_lm_head(
            &all_tensors,
            &config,
            &contract,
            &mut results,
            &mut critical_failures,
        );

        assert_eq!(results.len(), 1);
        assert!(!results[0].passed);
        assert_eq!(critical_failures.len(), 1);
    }

    #[test]
    fn test_validate_lm_head_without_lm_head() {
        let config = make_config_full();
        let contract = make_contract();

        let all_tensors = HashMap::new(); // no lm_head.weight

        let mut results = Vec::new();
        let mut critical_failures = Vec::new();

        validate_lm_head(
            &all_tensors,
            &config,
            &contract,
            &mut results,
            &mut critical_failures,
        );

        // Nothing happens when lm_head is absent
        assert!(results.is_empty());
        assert!(critical_failures.is_empty());
    }

    // ========================================================================
    // 16. validate_2d_tensors
    // ========================================================================

    #[test]
    fn test_validate_2d_tensors_skips_non_transpose() {
        let mut contract = make_contract();
        // Insert a non-transpose tensor => should be skipped
        contract.tensors.insert(
            "norm".to_string(),
            make_spec("model.norm.weight", "[hidden]", false),
        );

        let all_tensors = HashMap::new();
        let config = make_config_full();
        let mut results = Vec::new();

        validate_2d_tensors(&contract, &all_tensors, &config, &mut results);
        assert!(results.is_empty());
    }

    #[test]
    fn test_validate_2d_tensors_processes_layer_pattern() {
        let mut contract = make_contract();
        contract.tensors.insert(
            "q_proj".to_string(),
            make_spec(
                "model.layers.{n}.self_attn.q_proj.weight",
                "[vocab, hidden]",
                true,
            ),
        );

        let config = LayoutModelConfig {
            num_hidden_layers: Some(1),
            vocab_size: Some(100),
            hidden_size: Some(200),
            ..LayoutModelConfig::default()
        };

        let mut all_tensors = HashMap::new();
        all_tensors.insert(
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            vec![100, 200],
        );

        let mut results = Vec::new();
        validate_2d_tensors(&contract, &all_tensors, &config, &mut results);

        assert_eq!(results.len(), 1);
        assert!(results[0].passed);
    }

    #[test]
    fn test_validate_2d_tensors_processes_single() {
        let mut contract = make_contract();
        contract.tensors.insert(
            "embed".to_string(),
            make_spec("model.embed_tokens.weight", "[vocab, hidden]", true),
        );

        let config = make_config_full();

        let mut all_tensors = HashMap::new();
        all_tensors.insert("model.embed_tokens.weight".to_string(), vec![32000, 4096]);

        let mut results = Vec::new();
        validate_2d_tensors(&contract, &all_tensors, &config, &mut results);

        assert_eq!(results.len(), 1);
        assert!(results[0].passed);
    }

    #[test]
    fn test_validate_2d_tensors_missing_tensor() {
        let mut contract = make_contract();
        contract.tensors.insert(
            "embed".to_string(),
            make_spec("model.embed_tokens.weight", "[vocab, hidden]", true),
        );

        let config = make_config_full();
        let all_tensors = HashMap::new(); // no tensors present

        let mut results = Vec::new();
        validate_2d_tensors(&contract, &all_tensors, &config, &mut results);

        // Tensor not found => not validated (no result added)
        assert!(results.is_empty());
    }

    // ========================================================================
    // 17. validate_1d_tensors
    // ========================================================================

    #[test]
    fn test_validate_1d_tensors_skips_transpose() {
        let mut contract = make_contract();
        // Insert a transpose=true tensor => should be skipped by 1D validation
        contract.tensors.insert(
            "proj".to_string(),
            make_spec("model.proj.weight", "[vocab, hidden]", true),
        );

        let all_tensors = HashMap::new();
        let config = make_config_full();
        let mut results = Vec::new();

        validate_1d_tensors(&contract, &all_tensors, &config, &mut results);
        assert!(results.is_empty());
    }

    #[test]
    fn test_validate_1d_tensors_processes_layer_pattern() {
        let mut contract = make_contract();
        contract.tensors.insert(
            "layernorm".to_string(),
            make_spec("model.layers.{n}.input_layernorm.weight", "[hidden]", false),
        );

        let config = LayoutModelConfig {
            num_hidden_layers: Some(1),
            hidden_size: Some(4096),
            ..LayoutModelConfig::default()
        };

        let mut all_tensors = HashMap::new();
        all_tensors.insert(
            "model.layers.0.input_layernorm.weight".to_string(),
            vec![4096],
        );

        let mut results = Vec::new();
        validate_1d_tensors(&contract, &all_tensors, &config, &mut results);

        assert_eq!(results.len(), 1);
        assert!(results[0].passed);
    }

    #[test]
    fn test_validate_1d_tensors_processes_single() {
        let mut contract = make_contract();
        contract.tensors.insert(
            "norm".to_string(),
            make_spec("model.norm.weight", "[hidden]", false),
        );

        let config = make_config_full();

        let mut all_tensors = HashMap::new();
        all_tensors.insert("model.norm.weight".to_string(), vec![4096]);

        let mut results = Vec::new();
        validate_1d_tensors(&contract, &all_tensors, &config, &mut results);

        assert_eq!(results.len(), 1);
        assert!(results[0].passed);
    }

    #[test]
    fn test_validate_1d_tensors_missing_tensor() {
        let mut contract = make_contract();
        contract.tensors.insert(
            "norm".to_string(),
            make_spec("model.norm.weight", "[hidden]", false),
        );

        let config = make_config_full();
        let all_tensors = HashMap::new();

        let mut results = Vec::new();
        validate_1d_tensors(&contract, &all_tensors, &config, &mut results);

        assert!(results.is_empty());
    }

    // ========================================================================
    // 18. run_all_validations end-to-end
    // ========================================================================

    #[test]
    fn test_run_all_validations_end_to_end() {
        let dir = tempfile::tempdir().unwrap();

        // Create safetensors with lm_head + layer tensors + 1D tensors
        create_test_safetensors(
            &dir.path().join("model.safetensors"),
            &[
                ("lm_head.weight", &[32000, 4096]),
                ("model.layers.0.self_attn.q_proj.weight", &[4096, 4096]),
                ("model.layers.0.input_layernorm.weight", &[4096]),
                ("model.norm.weight", &[4096]),
            ],
        );

        // Create config.json
        let config = serde_json::json!({
            "vocab_size": 32000,
            "hidden_size": 4096,
            "intermediate_size": 11008,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "num_hidden_layers": 1
        });
        std::fs::write(
            dir.path().join("config.json"),
            serde_json::to_string(&config).unwrap(),
        )
        .unwrap();

        // Build a contract with real tensor specs
        let mut contract = make_contract();
        contract.tensors.insert(
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
        contract.tensors.insert(
            "q_proj".to_string(),
            make_spec(
                "model.layers.{n}.self_attn.q_proj.weight",
                "[heads*head_dim, hidden]",
                true,
            ),
        );
        contract.tensors.insert(
            "input_layernorm".to_string(),
            make_spec("model.layers.{n}.input_layernorm.weight", "[hidden]", false),
        );
        contract.tensors.insert(
            "final_norm".to_string(),
            make_spec("model.norm.weight", "[hidden]", false),
        );

        let (results, critical_failures) = run_all_validations(dir.path(), &contract);

        // lm_head should pass
        assert!(critical_failures.is_empty());
        // Results: lm_head via validate_lm_head (1) + lm_head via validate_2d_tensors (1)
        //        + q_proj layer 0 (1) + layernorm layer 0 (1) + final_norm (1) = 5
        assert_eq!(results.len(), 5);
        assert!(
            results.iter().all(|r| r.passed),
            "All results should pass: {:?}",
            results
        );
    }

    #[test]
    fn test_run_all_validations_with_critical_failure() {
        let dir = tempfile::tempdir().unwrap();

        // lm_head shape is transposed => should fail
        create_test_safetensors(
            &dir.path().join("model.safetensors"),
            &[("lm_head.weight", &[4096, 32000])],
        );

        let config = serde_json::json!({
            "vocab_size": 32000,
            "hidden_size": 4096
        });
        std::fs::write(
            dir.path().join("config.json"),
            serde_json::to_string(&config).unwrap(),
        )
        .unwrap();

        let mut contract = make_contract();
        contract.tensors.insert(
            "lm_head".to_string(),
            TensorSpec {
                gguf_name: "output.weight".to_string(),
                apr_name: "lm_head.weight".to_string(),
                gguf_shape: "[hidden, vocab]".to_string(),
                apr_shape: "[vocab, hidden]".to_string(),
                transpose: true,
                kernel: "matmul".to_string(),
                kernel_out_dim: None,
                kernel_in_dim: None,
                validation: None,
                critical: true,
                note: None,
            },
        );

        let (results, critical_failures) = run_all_validations(dir.path(), &contract);

        assert!(!critical_failures.is_empty());
        assert!(results.iter().any(|r| !r.passed));
    }

    // ========================================================================
    // 19. check_model_path_preconditions
    // ========================================================================

    #[test]
    fn test_check_model_path_preconditions_missing_path() {
        let result = check_model_path_preconditions(Path::new("/nonexistent/path"));
        assert!(result.is_some());
        let result = result.unwrap();
        assert!(!result.passed);
        assert_eq!(result.rules_failed, 1);
        assert!(!result.critical_failures.is_empty());
    }

    #[test]
    fn test_check_model_path_preconditions_no_safetensors() {
        let dir = tempfile::tempdir().unwrap();
        // Dir exists but has no .safetensors files
        std::fs::write(dir.path().join("something.txt"), "hello").unwrap();

        let result = check_model_path_preconditions(dir.path());
        assert!(result.is_some());
        let result = result.unwrap();
        // Empty dir -> passed=true (no safetensors to validate is OK)
        assert!(result.passed);
        assert_eq!(result.rules_checked, 0);
    }

    #[test]
    fn test_check_model_path_preconditions_has_safetensors() {
        let dir = tempfile::tempdir().unwrap();
        create_test_safetensors(&dir.path().join("model.safetensors"), &[("x", &[2, 3])]);

        let result = check_model_path_preconditions(dir.path());
        // Should return None (proceed with validation)
        assert!(result.is_none());
    }

    // ========================================================================
    // 20. validate_model full integration
    // ========================================================================

    #[test]
    fn test_validate_model_full_pass() {
        let dir = tempfile::tempdir().unwrap();

        create_test_safetensors(
            &dir.path().join("model.safetensors"),
            &[("lm_head.weight", &[32000, 4096])],
        );

        let config = serde_json::json!({
            "vocab_size": 32000,
            "hidden_size": 4096
        });
        std::fs::write(
            dir.path().join("config.json"),
            serde_json::to_string(&config).unwrap(),
        )
        .unwrap();

        let contract = make_contract();
        let result = validate_model(dir.path(), &contract).unwrap();

        assert!(result.passed);
        assert!(result.critical_failures.is_empty());
        // lm_head validated
        assert!(result.rules_checked > 0);
    }

    #[test]
    fn test_validate_model_full_fail() {
        let dir = tempfile::tempdir().unwrap();

        // Transposed lm_head
        create_test_safetensors(
            &dir.path().join("model.safetensors"),
            &[("lm_head.weight", &[4096, 32000])],
        );

        let config = serde_json::json!({
            "vocab_size": 32000,
            "hidden_size": 4096
        });
        std::fs::write(
            dir.path().join("config.json"),
            serde_json::to_string(&config).unwrap(),
        )
        .unwrap();

        let contract = make_contract();
        let result = validate_model(dir.path(), &contract).unwrap();

        assert!(!result.passed);
        assert!(!result.critical_failures.is_empty());
    }

    #[test]
    fn test_validate_model_no_config_json() {
        let dir = tempfile::tempdir().unwrap();

        create_test_safetensors(
            &dir.path().join("model.safetensors"),
            &[("lm_head.weight", &[32000, 4096])],
        );

        // No config.json => LayoutModelConfig::default() => lm_head passes (no expected dims)
        let contract = make_contract();
        let result = validate_model(dir.path(), &contract).unwrap();

        assert!(result.passed);
    }

    // ========================================================================
    // 21. read_safetensors_metadata - edge cases
    // ========================================================================

    #[test]
    fn test_read_safetensors_metadata_nonexistent_file() {
        let result = read_safetensors_metadata(Path::new("/nonexistent/file.safetensors"));
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Failed to open"));
    }

    #[test]
    fn test_read_safetensors_metadata_truncated_header() {
        use std::io::Write;
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("truncated.safetensors");

        // Write a header_len of 1000 but only provide 5 bytes of header
        let header_len: u64 = 1000;
        let mut file = std::fs::File::create(&file_path).unwrap();
        file.write_all(&header_len.to_le_bytes()).unwrap();
        file.write_all(b"short").unwrap();

        let result = read_safetensors_metadata(&file_path);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Failed to read header"));
    }

    #[test]
    fn test_read_safetensors_metadata_not_json_object() {
        use std::io::Write;
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("array.safetensors");

        // Valid JSON but not an object
        let header = b"[1, 2, 3]";
        let header_len = header.len() as u64;
        let mut file = std::fs::File::create(&file_path).unwrap();
        file.write_all(&header_len.to_le_bytes()).unwrap();
        file.write_all(header).unwrap();

        let result = read_safetensors_metadata(&file_path);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not JSON object"));
    }

    #[test]
    fn test_read_safetensors_metadata_tensor_without_shape() {
        use std::io::Write;
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("noshape.safetensors");

        // Valid JSON object but tensor entry has no "shape" field
        let header_json = serde_json::json!({
            "__metadata__": {"format": "pt"},
            "broken_tensor": {"dtype": "F32", "data_offsets": [0, 100]}
        });
        let header_bytes = serde_json::to_string(&header_json).unwrap();
        let header_len = header_bytes.len() as u64;
        let mut file = std::fs::File::create(&file_path).unwrap();
        file.write_all(&header_len.to_le_bytes()).unwrap();
        file.write_all(header_bytes.as_bytes()).unwrap();

        let result = read_safetensors_metadata(&file_path).unwrap();
        // broken_tensor should be skipped (filter_map returns None)
        assert!(result.is_empty());
    }

    // ========================================================================
    // 22. find_safetensors_files - nonexistent path
    // ========================================================================

    #[test]
    fn test_find_safetensors_files_nonexistent_dir() {
        let files = find_safetensors_files(Path::new("/nonexistent/dir"));
        assert!(files.is_empty());
    }

    // ========================================================================
    // 23. validate_model with file path (not directory)
    // ========================================================================

    #[test]
    fn test_validate_model_with_file_path() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("model.safetensors");

        create_test_safetensors(&file_path, &[("lm_head.weight", &[32000, 4096])]);

        let config = serde_json::json!({
            "vocab_size": 32000,
            "hidden_size": 4096
        });
        std::fs::write(
            dir.path().join("config.json"),
            serde_json::to_string(&config).unwrap(),
        )
        .unwrap();

        let contract = make_contract();
        let result = validate_model(&file_path, &contract).unwrap();

        assert!(result.passed);
        assert!(result.rules_checked > 0);
    }

    // ========================================================================
    // 24. validate_lm_head non-2D through the orchestrator
    // ========================================================================

    #[test]
    fn test_validate_lm_head_not_2d_through_orchestrator() {
        let config = make_config_full();
        let contract = make_contract();

        let mut all_tensors = HashMap::new();
        all_tensors.insert("lm_head.weight".to_string(), vec![4096, 32000, 10]);

        let mut results = Vec::new();
        let mut critical_failures = Vec::new();

        validate_lm_head(
            &all_tensors,
            &config,
            &contract,
            &mut results,
            &mut critical_failures,
        );

        assert_eq!(results.len(), 1);
        assert!(!results[0].passed);
        assert_eq!(critical_failures.len(), 1);
    }

    // ========================================================================
    // 25. Multiple safetensors files merged
    // ========================================================================

    #[test]
    fn test_collect_tensor_metadata_multiple_files() {
        let dir = tempfile::tempdir().unwrap();

        create_test_safetensors(
            &dir.path().join("model-00001-of-00002.safetensors"),
            &[("lm_head.weight", &[32000, 4096])],
        );
        create_test_safetensors(
            &dir.path().join("model-00002-of-00002.safetensors"),
            &[("model.embed_tokens.weight", &[32000, 4096])],
        );

        let mut results = Vec::new();
        let tensors = collect_tensor_metadata(dir.path(), &mut results);

        assert!(results.is_empty());
        assert_eq!(tensors.len(), 2);
        assert!(tensors.contains_key("lm_head.weight"));
        assert!(tensors.contains_key("model.embed_tokens.weight"));
    }

    // ========================================================================
    // 26. read_safetensors_metadata: empty file (< 8 bytes)
    // ========================================================================

    #[test]
    fn test_read_safetensors_metadata_too_short() {
        use std::io::Write;
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("tiny.safetensors");

        let mut file = std::fs::File::create(&file_path).unwrap();
        file.write_all(b"tiny").unwrap(); // Only 4 bytes, need 8

        let result = read_safetensors_metadata(&file_path);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Failed to read header length"));
    }

    // ========================================================================
    // 27. resolve_dimension with spaces in expression
    // ========================================================================

    #[test]
    fn test_resolve_dimension_expression_with_spaces() {
        let config = make_config_full();
        // "heads * head_dim" => trimmed parts should work
        assert_eq!(
            resolve_dimension("heads * head_dim", &config),
            Some(32 * 128)
        );
    }

    // ========================================================================
    // 28. find_and_load_config with invalid JSON
    // ========================================================================

    #[test]
    fn test_find_and_load_config_invalid_json() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("config.json"), "not json at all").unwrap();

        let mc = find_and_load_config(dir.path());
        // Should fall through to default
        assert_eq!(mc.vocab_size, None);
    }

    // ========================================================================
    // 29. validate_2d_tensor_shape 3D tensor
    // ========================================================================

    #[test]
    fn test_validate_2d_tensor_shape_3d() {
        let spec = make_spec("test.weight", "[vocab, hidden]", true);
        let config = make_config_full();
        let result = validate_2d_tensor_shape("test", &[10, 20, 30], &spec, &config);
        assert!(!result.passed);
        assert!(result.details.contains("must be 2D, got 3D"));
    }

    // ========================================================================
    // 30. validate_1d_tensor_shape 0D tensor
    // ========================================================================

    #[test]
    fn test_validate_1d_tensor_shape_0d() {
        let spec = make_spec("test.bias", "[hidden]", false);
        let config = make_config_full();
        let result = validate_1d_tensor_shape("test.bias", &[], &spec, &config);
        assert!(!result.passed);
        assert!(result.details.contains("must be 1D, got 0D"));
    }

    // ========================================================================
    // 31. load_contract_from - valid YAML
    // ========================================================================

    #[test]
    fn test_load_contract_from_valid_yaml() {
        let dir = tempfile::tempdir().unwrap();
        let yaml_path = dir.path().join("contract.yaml");
        let yaml_content = r#"
metadata:
  version: "1.0"
  created: "2026-01-01"
  updated: "2026-01-01"
  author: "test"
  description: "test contract"
formats:
  apr:
    layout: "row-major"
    shape_convention: "[out_dim, in_dim]"
kernel:
  signature: "matmul(W, x, out_dim, in_dim)"
  weight_shape: "[out_dim, in_dim]"
  computation: "y = W @ x"
  byte_calculation: "out * in * block_size / QK_K"
  block_sizes:
    Q4_K: 144
  QK_K: 256
tensors: {}
validation_rules:
  - id: "F-LAYOUT-CONTRACT-001"
    name: "2D Transpose Check"
    description: "All 2D weights are transposed"
    severity: "P0"
    critical: true
"#;
        std::fs::write(&yaml_path, yaml_content).unwrap();

        let contract = load_contract_from(&yaml_path).unwrap();
        assert_eq!(contract.metadata.version, "1.0");
        assert_eq!(contract.metadata.author, "test");
        assert_eq!(contract.kernel.qk_k, 256);
        assert_eq!(contract.validation_rules.len(), 1);
        assert_eq!(contract.validation_rules[0].id, "F-LAYOUT-CONTRACT-001");
    }

    #[test]
    fn test_load_contract_from_invalid_yaml() {
        let dir = tempfile::tempdir().unwrap();
        let yaml_path = dir.path().join("bad.yaml");
        std::fs::write(&yaml_path, "this: is: not: valid: yaml: [[[").unwrap();

        let result = load_contract_from(&yaml_path);
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("Failed to parse"));
    }

    // ========================================================================
    // 32. read_safetensors_metadata - invalid UTF-8
    // ========================================================================

    #[test]
    fn test_read_safetensors_metadata_invalid_utf8() {
        use std::io::Write;
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("badutf8.safetensors");

        // Write invalid UTF-8 bytes as header
        let bad_bytes: &[u8] = &[0xFF, 0xFE, 0x80, 0x81, 0x82, 0x83, 0x84, 0x85];
        let header_len = bad_bytes.len() as u64;
        let mut file = std::fs::File::create(&file_path).unwrap();
        file.write_all(&header_len.to_le_bytes()).unwrap();
        file.write_all(bad_bytes).unwrap();

        let result = read_safetensors_metadata(&file_path);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Invalid UTF-8"));
    }

    // ========================================================================
    // 33. load_contract (uses DEFAULT_CONTRACT_PATH)
    // ========================================================================

    #[test]
    fn test_load_contract_default_path_missing() {
        // The default path is relative and almost certainly not present in test env
        let result = load_contract();
        // Either succeeds (if contract exists) or fails with file-not-found
        if let Err(e) = result {
            let err_msg = format!("{e}");
            assert!(err_msg.contains("Failed to read"));
        }
    }
