#[test]
fn test_oracle_enhancer_default() {
    let enhancer = OracleEnhancer::new();
    assert_eq!(enhancer.timeout, Duration::from_millis(30_000));
    assert!((enhancer.min_relevance - 0.5).abs() < f32::EPSILON);
}

#[test]
fn test_generate_static_checklist_for_conv_failure() {
    let enhancer = OracleEnhancer::new();
    let evidence = Evidence::falsified(
        "F-CONV-G-A",
        make_test_scenario(),
        "Conversion diff: 7.61e-1",
        "output",
        1000,
    );

    let checklist = enhancer.generate_static_checklist(&evidence);
    assert!(!checklist.is_empty());
    assert_eq!(checklist[0].gate_id, "F-LAYOUT-002");
}

#[test]
fn test_generate_static_checklist_for_path_failure() {
    let enhancer = OracleEnhancer::new();
    let evidence = Evidence::falsified(
        "F-CONV-RT-001",
        make_test_scenario(),
        "No file extension found",
        "output",
        1000,
    );

    let checklist = enhancer.generate_static_checklist(&evidence);
    assert!(checklist.iter().any(|c| c.gate_id == "F-PATH-EXT"));
}

#[test]
fn test_check_status_display() {
    assert_eq!(format!("{}", CheckStatus::Pending), "PENDING");
    assert_eq!(
        format!("{}", CheckStatus::Falsified("reason".to_string())),
        "FALSIFIED: reason"
    );
    assert_eq!(format!("{}", CheckStatus::Corroborated), "CORROBORATED");
}

#[test]
fn test_confidence_display() {
    assert_eq!(format!("{}", Confidence::High), "HIGH");
    assert_eq!(format!("{}", Confidence::Medium), "MEDIUM");
    assert_eq!(format!("{}", Confidence::Low), "LOW");
}

#[test]
fn test_generate_checklist_markdown() {
    let context = OracleContext {
        oracle_available: true,
        checklist: vec![FalsificationCheckItem {
            gate_id: "F-LAYOUT-002".to_string(),
            hypothesis: "Row-major layout".to_string(),
            test_procedure: "Check layout flag".to_string(),
            falsified_if: "Garbage output".to_string(),
            status: CheckStatus::Falsified("High diff".to_string()),
            confidence: Confidence::High,
        }],
        hypotheses: vec![RankedHypothesis {
            id: "H1".to_string(),
            description: "Layout bug".to_string(),
            confidence: Confidence::High,
            evidence_for: vec!["High diff".to_string()],
            evidence_against: vec![],
        }],
        cross_references: vec![CrossReference {
            source: "spec.md".to_string(),
            section: "LAYOUT-002".to_string(),
            relevance: 0.95,
        }],
        investigation_commands: vec!["apr inspect model.apr".to_string()],
        query_latency_ms: 1000,
    };

    let md = generate_checklist_markdown("test-model", 320, "F", 24, 13, &context);

    assert!(md.contains("# Falsification Checklist: test-model"));
    assert!(md.contains("F-LAYOUT-002"));
    assert!(md.contains("Row-major layout"));
    assert!(md.contains("H1"));
    assert!(md.contains("apr inspect"));
}

#[test]
fn test_enhance_failure_non_failure() {
    let enhancer = OracleEnhancer::new();
    let evidence = Evidence::corroborated("F-TEST-001", make_test_scenario(), "output", 1000);

    let context = enhancer.enhance_failure(&evidence);
    assert!(!context.oracle_available);
    assert!(context.checklist.is_empty());
}

#[test]
fn test_generate_hypotheses() {
    let enhancer = OracleEnhancer::new();
    let evidence = Evidence::falsified(
        "F-CONV-G-A",
        make_test_scenario(),
        "No file extension found",
        "output",
        1000,
    );

    let hypotheses = enhancer.generate_hypotheses_from_evidence(&evidence);
    assert!(!hypotheses.is_empty());
    assert!(hypotheses.iter().any(|h| h.id == "H1"));
}

#[test]
fn test_generate_cross_references() {
    let enhancer = OracleEnhancer::new();
    let evidence = Evidence::falsified(
        "F-CONV-G-A",
        make_test_scenario(),
        "Conversion diff: 7.61e-1",
        "output",
        1000,
    );

    let refs = enhancer.generate_cross_references(&evidence);
    assert!(!refs.is_empty());
    assert!(refs.iter().any(|r| r.source.contains("spec")));
}

#[test]
fn test_generate_investigation_commands() {
    let enhancer = OracleEnhancer::new();
    let evidence = Evidence::falsified(
        "F-CONV-G-A",
        make_test_scenario(),
        "Conversion failed",
        "output",
        1000,
    );

    let commands = enhancer.generate_investigation_commands(&evidence);
    assert!(!commands.is_empty());
    assert!(commands.iter().any(|c| c.contains("apr")));
}

#[test]
fn test_with_timeout() {
    let enhancer = OracleEnhancer::new().with_timeout(Duration::from_secs(10));
    assert_eq!(enhancer.timeout, Duration::from_secs(10));
}

#[test]
fn test_with_min_relevance() {
    let enhancer = OracleEnhancer::new().with_min_relevance(0.8);
    assert!((enhancer.min_relevance - 0.8).abs() < f32::EPSILON);
}

#[test]
fn test_is_available() {
    // batuta is unlikely to be available in CI, so just verify it returns a bool
    let available = OracleEnhancer::is_available();
    // The function should return without panicking
    let _ = available;
}

#[test]
fn test_enhance_failures_filters_non_failures() {
    let enhancer = OracleEnhancer::new();
    let evidences = vec![
        Evidence::corroborated("F-TEST-001", make_test_scenario(), "ok", 100),
        Evidence::corroborated("F-TEST-002", make_test_scenario(), "ok", 200),
    ];

    let results = enhancer.enhance_failures(&evidences);
    assert!(results.is_empty(), "No failures means no enhanced results");
}

#[test]
fn test_enhance_failures_includes_only_failures() {
    let enhancer = OracleEnhancer::new();
    let evidences = vec![
        Evidence::corroborated("F-TEST-001", make_test_scenario(), "ok", 100),
        Evidence::falsified(
            "F-CONV-G-A",
            make_test_scenario(),
            "diff too high",
            "output",
            1000,
        ),
        Evidence::corroborated("F-TEST-003", make_test_scenario(), "ok", 300),
    ];

    let results = enhancer.enhance_failures(&evidences);
    assert_eq!(results.len(), 1, "Only one failure should be enhanced");
}

#[test]
fn test_build_query_format() {
    let enhancer = OracleEnhancer::new();
    let evidence = Evidence::falsified(
        "F-CONV-G-A",
        make_test_scenario(),
        "Conversion diff: 7.61e-1",
        "output",
        1000,
    );

    let query = enhancer.build_query(&evidence);
    assert!(query.contains("falsification checklist"));
    assert!(query.contains("F-CONV-G-A"));
    assert!(query.contains("Conversion diff: 7.61e-1"));
    assert!(query.contains("LAYOUT-002"));
    assert!(query.contains(&format!("{}", evidence.scenario.format)));
}

#[test]
fn test_parse_oracle_output() {
    let enhancer = OracleEnhancer::new();
    let evidence = Evidence::falsified(
        "F-CONV-G-A",
        make_test_scenario(),
        "Conversion diff: 7.61e-1",
        "output",
        1000,
    );

    let context = enhancer.parse_oracle_output("oracle says something", &evidence, 42);
    assert!(context.oracle_available);
    assert_eq!(context.query_latency_ms, 42);
    assert!(!context.checklist.is_empty());
    assert!(!context.cross_references.is_empty());
}

#[test]
fn test_generate_checklist_inference_gate() {
    let enhancer = OracleEnhancer::new();
    let evidence = Evidence::falsified(
        "F-INF-001",
        make_test_scenario(),
        "Inference output mismatch",
        "output",
        1000,
    );

    let checklist = enhancer.generate_checklist_from_gate(&evidence);
    assert!(
        checklist.iter().any(|c| c.gate_id == "F-CONV-INF-EQ"),
        "Should generate inference equivalence check for INF gate"
    );
    let inf_item = checklist
        .iter()
        .find(|c| c.gate_id == "F-CONV-INF-EQ")
        .unwrap();
    assert!(inf_item.hypothesis.contains("Inference output identical"));
    assert!(matches!(inf_item.status, CheckStatus::Pending));
    assert_eq!(inf_item.confidence, Confidence::Medium);
}

#[test]
fn test_generate_checklist_conv_transpose() {
    let enhancer = OracleEnhancer::new();
    let evidence = Evidence::falsified(
        "F-CONV-G-A-001",
        make_test_scenario(),
        "Conversion failed",
        "output",
        1000,
    );

    let checklist = enhancer.generate_checklist_from_gate(&evidence);
    assert!(
        checklist.iter().any(|c| c.gate_id == "F-CONV-TRANSPOSE"),
        "Should generate transpose check for CONV + G-A gate"
    );
    let transpose_item = checklist
        .iter()
        .find(|c| c.gate_id == "F-CONV-TRANSPOSE")
        .unwrap();
    assert!(transpose_item.hypothesis.contains("Q4K tensor transpose"));
    assert!(matches!(transpose_item.status, CheckStatus::Pending));
    assert_eq!(transpose_item.confidence, Confidence::Medium);
}

#[test]
fn test_generate_checklist_conv_with_diff_falsifies_layout() {
    let enhancer = OracleEnhancer::new();
    let evidence = Evidence::falsified(
        "F-CONV-G-A",
        make_test_scenario(),
        "diff 0.76",
        "output",
        1000,
    );

    let checklist = enhancer.generate_checklist_from_gate(&evidence);
    let layout_item = checklist
        .iter()
        .find(|c| c.gate_id == "F-LAYOUT-002")
        .expect("Should have LAYOUT-002 item");
    assert!(
        matches!(layout_item.status, CheckStatus::Falsified(_)),
        "Diff in reason should falsify the LAYOUT-002 hypothesis"
    );
}

#[test]
fn test_generate_checklist_conv_without_diff_is_pending() {
    let enhancer = OracleEnhancer::new();
    let evidence = Evidence::falsified(
        "F-CONV-001",
        make_test_scenario(),
        "Conversion timeout",
        "output",
        1000,
    );

    let checklist = enhancer.generate_checklist_from_gate(&evidence);
    let layout_item = checklist
        .iter()
        .find(|c| c.gate_id == "F-LAYOUT-002")
        .expect("Should have LAYOUT-002 item for F-CONV gate");
    assert!(
        matches!(layout_item.status, CheckStatus::Pending),
        "No diff in reason should leave status as Pending"
    );
}

#[test]
fn test_generate_hypotheses_diff_in_reason() {
    let enhancer = OracleEnhancer::new();
    let evidence = Evidence::falsified(
        "F-CONV-G-A",
        make_test_scenario(),
        "diff 7.61e-1",
        "output",
        1000,
    );

    let hypotheses = enhancer.generate_hypotheses_from_evidence(&evidence);
    assert!(
        hypotheses.iter().any(|h| h.id == "H2"),
        "Should generate LAYOUT-002 hypothesis when diff in reason"
    );
    let h2 = hypotheses.iter().find(|h| h.id == "H2").unwrap();
    assert!(h2.description.contains("LAYOUT-002"));
    assert_eq!(h2.confidence, Confidence::Medium);
    assert!(!h2.evidence_for.is_empty());
    assert!(!h2.evidence_against.is_empty());
}

#[test]
fn test_generate_hypotheses_conv_gate() {
    let enhancer = OracleEnhancer::new();
    let evidence = Evidence::falsified(
        "F-CONV-001",
        make_test_scenario(),
        "Some conversion error",
        "output",
        1000,
    );

    let hypotheses = enhancer.generate_hypotheses_from_evidence(&evidence);
    assert!(
        hypotheses.iter().any(|h| h.id == "H3"),
        "Should generate quantization mismatch hypothesis for CONV gate"
    );
    let h3 = hypotheses.iter().find(|h| h.id == "H3").unwrap();
    assert!(h3.description.contains("Quantization mismatch"));
    assert_eq!(h3.confidence, Confidence::Low);
}

#[test]
fn test_generate_hypotheses_conv_gate_no_file_ext_no_diff() {
    // CONV gate with neither "No file extension" nor "diff" — only H3
    let enhancer = OracleEnhancer::new();
    let evidence = Evidence::falsified(
        "F-CONV-001",
        make_test_scenario(),
        "Unknown conversion error",
        "output",
        1000,
    );

    let hypotheses = enhancer.generate_hypotheses_from_evidence(&evidence);
    assert!(!hypotheses.iter().any(|h| h.id == "H1"));
    assert!(!hypotheses.iter().any(|h| h.id == "H2"));
    assert!(hypotheses.iter().any(|h| h.id == "H3"));
}

#[test]
fn test_generate_hypotheses_non_conv_non_special_reason() {
    let enhancer = OracleEnhancer::new();
    let evidence = Evidence::falsified(
        "F-LOAD-001",
        make_test_scenario(),
        "Model failed to load",
        "output",
        1000,
    );

    let hypotheses = enhancer.generate_hypotheses_from_evidence(&evidence);
    assert!(
        hypotheses.is_empty(),
        "Non-CONV gate with no special reason should produce no hypotheses"
    );
}

#[test]
fn test_generate_cross_references_garbage_reason() {
    let enhancer = OracleEnhancer::new();
    let evidence = Evidence::falsified(
        "F-LOAD-001",
        make_test_scenario(),
        "garbage output detected",
        "output",
        1000,
    );

    let refs = enhancer.generate_cross_references(&evidence);
    assert!(
        refs.iter().any(|r| r.source == "GH-190"),
        "Should reference GH-190 for garbage in reason"
    );
}

#[test]
fn test_generate_cross_references_no_conv_no_garbage_no_diff() {
    let enhancer = OracleEnhancer::new();
    let evidence = Evidence::falsified(
        "F-LOAD-001",
        make_test_scenario(),
        "Model failed to load",
        "output",
        1000,
    );

    let refs = enhancer.generate_cross_references(&evidence);
    // Should only contain the always-present spec reference
    assert_eq!(refs.len(), 1);
    assert_eq!(refs[0].source, "apr-playbook-spec.md");
}

#[test]
fn test_generate_cross_references_high_min_relevance() {
    let enhancer = OracleEnhancer::new().with_min_relevance(0.99);
    let evidence = Evidence::falsified(
        "F-CONV-G-A",
        make_test_scenario(),
        "diff 0.76",
        "output",
        1000,
    );

    let refs = enhancer.generate_cross_references(&evidence);
    // All references have relevance <= 0.95, so everything should be filtered out
    assert!(
        refs.is_empty(),
        "With min_relevance 0.99, all refs should be filtered out"
    );
}

