#[test]
fn test_contract_tests_with_dotted_workspace_path() {
    use crate::command::MockCommandRunner;

    let runner: Arc<dyn CommandRunner> = Arc::new(MockCommandRunner::new());
    let model_id = ModelId::new("Qwen", "Qwen2.5-Coder-0.5B-Instruct");
    let config = ContractTestConfig::default();

    let evidence = run_contract_tests(
        &runner,
        Path::new("/workspace/Qwen/Qwen2.5-Coder-0.5B-Instruct"),
        &model_id,
        &config,
    );

    // All 4 invariants (I-2 through I-5) should produce evidence
    assert_eq!(evidence.len(), 4, "Expected 4 invariant results");
    for ev in &evidence {
        // None should mention truncated paths like "Coder-0.apr"
        assert!(
            !ev.reason.contains("Coder-0.apr"),
            "Path was truncated by with_extension: {}",
            ev.reason
        );
    }
}

#[test]
fn test_is_valid_tensor_name_edge_cases() {
    let contract = load_format_contract().expect("Failed to load contract");

    // Valid edge cases
    assert!(validate_tensor_name("0.attn.weight", &contract));
    assert!(validate_tensor_name("99.mlp.bias", &contract));

    // Invalid edge cases
    assert!(!validate_tensor_name("weight", &contract));
    assert!(!validate_tensor_name(".q_proj.weight", &contract));
    assert!(!validate_tensor_name("a.q_proj.weight", &contract));
    assert!(!validate_tensor_name("0.q_proj.weight.extra", &contract));
}

#[test]
fn test_naming_convention() {
    let contract = load_format_contract().expect("Failed to load contract");
    assert_eq!(contract.tensor_naming.convention, "gguf-short");
}

#[test]
fn test_invariant_catches_fields() {
    let contract = load_format_contract().expect("Failed to load contract");
    let i1 = contract.invariants.iter().find(|i| i.id == "I-1").unwrap();
    assert!(i1.catches.contains(&"GH-190".to_string()));
    assert!(i1.implemented);

    let i2 = contract.invariants.iter().find(|i| i.id == "I-2").unwrap();
    assert!(i2.catches.contains(&"GH-190".to_string()));
    assert!(!i2.implemented);
}

#[test]
fn test_tolerance_entries_ordered_by_precision() {
    let contract = load_format_contract().expect("Failed to load contract");
    // F32 should have 0 tolerance (exact)
    let f32_tol = lookup_tolerance("F32", &contract).unwrap();
    assert!(f32_tol.0.abs() < f64::EPSILON);

    // Q2_K should have the loosest tolerance
    let q2k_tol = lookup_tolerance("Q2_K", &contract).unwrap();
    assert!(q2k_tol.0 > 0.1);
}

#[test]
fn test_is_word() {
    assert!(is_word("weight"));
    assert!(is_word("q_proj"));
    assert!(is_word("down_proj"));
    assert!(is_word("a"));
    assert!(!is_word(""));
    assert!(!is_word("has.dot"));
    assert!(!is_word("has space"));
}
