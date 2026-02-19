#[test]
fn test_generate_cross_references_low_min_relevance() {
    let enhancer = OracleEnhancer::new().with_min_relevance(0.0);
    let evidence = Evidence::falsified(
        "F-CONV-G-A",
        make_test_scenario(),
        "diff 0.76",
        "output",
        1000,
    );

    let refs = enhancer.generate_cross_references(&evidence);
    // With min_relevance 0.0, all three refs should pass (spec, aprender, GH-190)
    assert_eq!(refs.len(), 3);
}

#[test]
fn test_oracle_error_display_execution_failed() {
    let err = OracleError::ExecutionFailed("not found".to_string());
    let display = format!("{err}");
    assert_eq!(display, "Failed to execute batuta: not found");
}

#[test]
fn test_oracle_error_display_query_failed() {
    let err = OracleError::QueryFailed("bad query".to_string());
    let display = format!("{err}");
    assert_eq!(display, "Oracle query failed: bad query");
}

#[test]
fn test_oracle_error_display_timeout() {
    let err = OracleError::Timeout;
    let display = format!("{err}");
    assert_eq!(display, "Oracle query timed out");
}

#[test]
fn test_oracle_error_is_error_trait() {
    let err: Box<dyn std::error::Error> =
        Box::new(OracleError::ExecutionFailed("test".to_string()));
    assert!(err.source().is_none());
    // Verify it implements std::error::Error by using it as a trait object
    let display = format!("{err}");
    assert!(display.contains("batuta"));
}

#[test]
fn test_oracle_error_debug() {
    let err = OracleError::Timeout;
    let debug = format!("{err:?}");
    assert!(debug.contains("Timeout"));
}

#[test]
fn test_generate_static_commands_conv() {
    let enhancer = OracleEnhancer::new();
    let evidence = Evidence::falsified(
        "F-CONV-001",
        make_test_scenario(),
        "Conversion error",
        "output",
        1000,
    );

    let commands = enhancer.generate_static_commands(&evidence);
    assert!(!commands.is_empty());
    assert!(commands.iter().any(|c| c.contains("layout")));
    assert!(commands.iter().any(|c| c.contains("apr inspect")));
}

#[test]
fn test_generate_static_commands_non_conv() {
    let enhancer = OracleEnhancer::new();
    let evidence = Evidence::falsified(
        "F-LOAD-001",
        make_test_scenario(),
        "Load failed",
        "output",
        1000,
    );

    let commands = enhancer.generate_static_commands(&evidence);
    assert!(
        commands.is_empty(),
        "Non-CONV gate should produce no static commands"
    );
}

#[test]
fn test_enhance_failure_on_actual_failure() {
    // enhance_failure on a falsified evidence should produce a non-empty context
    // regardless of whether batuta is available (live oracle or fallback)
    let enhancer = OracleEnhancer::new();
    let evidence = Evidence::falsified(
        "F-CONV-G-A",
        make_test_scenario(),
        "Conversion diff: 7.61e-1",
        "output",
        1000,
    );

    let context = enhancer.enhance_failure(&evidence);
    // Either path should produce a checklist for CONV gate
    assert!(
        !context.checklist.is_empty(),
        "Both oracle and fallback should generate checklist for CONV gate"
    );
    assert!(
        !context.investigation_commands.is_empty(),
        "Both oracle and fallback should generate investigation commands"
    );

    if context.oracle_available {
        // Live oracle path: parse_oracle_output was called
        // query_latency_ms can be any non-negative value
    } else {
        // Fallback path: static generators used
        assert_eq!(context.query_latency_ms, 0);
        assert!(context.hypotheses.is_empty());
        assert!(context.cross_references.is_empty());
    }
}

#[test]
fn test_enhance_failure_on_timeout_evidence() {
    // Timeout evidence is also a failure, so it should be enhanced
    let enhancer = OracleEnhancer::new();
    let evidence = Evidence::timeout("F-CONV-001", make_test_scenario(), 30000);

    let context = enhancer.enhance_failure(&evidence);
    // The gate starts with F-CONV so checklist should have LAYOUT-002
    // (both oracle path via generate_checklist_from_gate and fallback via generate_static_checklist)
    assert!(
        context
            .checklist
            .iter()
            .any(|c| c.gate_id == "F-LAYOUT-002")
    );
}

#[test]
fn test_enhance_failure_on_crashed_evidence() {
    let enhancer = OracleEnhancer::new();
    let evidence = Evidence::crashed("F-LOAD-001", make_test_scenario(), "segfault", 139, 100);

    let context = enhancer.enhance_failure(&evidence);
    // Both oracle and fallback paths produce empty checklist for F-LOAD (no branches match)
    assert!(context.checklist.is_empty());
}

#[test]
fn test_enhance_failure_fallback_path() {
    // Force the fallback path by using a very short timeout with an enhancer
    // that will cause query_oracle to fail (batuta with bad args would fail,
    // but we can't control that). Instead, test the static generators directly.
    let enhancer = OracleEnhancer::new();

    // Test generate_static_checklist for CONV gate
    let conv_evidence = Evidence::falsified(
        "F-CONV-001",
        make_test_scenario(),
        "extension error",
        "output",
        1000,
    );
    let static_checklist = enhancer.generate_static_checklist(&conv_evidence);
    assert!(static_checklist.iter().any(|c| c.gate_id == "F-LAYOUT-002"));
    assert!(static_checklist.iter().any(|c| c.gate_id == "F-PATH-EXT"));

    // Test generate_static_commands for CONV gate
    let static_commands = enhancer.generate_static_commands(&conv_evidence);
    assert!(!static_commands.is_empty());
    assert!(static_commands.iter().any(|c| c.contains("layout")));
}

#[test]
fn test_generate_checklist_from_gate_no_matches() {
    let enhancer = OracleEnhancer::new();
    let evidence = Evidence::falsified(
        "F-LOAD-001",
        make_test_scenario(),
        "Load failed, no special keywords",
        "output",
        1000,
    );

    let checklist = enhancer.generate_checklist_from_gate(&evidence);
    assert!(
        checklist.is_empty(),
        "Gate that matches no branch should produce empty checklist"
    );
}

#[test]
fn test_generate_investigation_commands_non_conv() {
    let enhancer = OracleEnhancer::new();
    let evidence = Evidence::falsified(
        "F-LOAD-001",
        make_test_scenario(),
        "Load failed",
        "output",
        1000,
    );

    let commands = enhancer.generate_investigation_commands(&evidence);
    // Should have rosetta command (always present) but no CONV-specific commands
    assert!(commands.iter().any(|c| c.contains("rosetta")));
    assert!(!commands.iter().any(|c| c.contains("grep")));
    assert!(!commands.iter().any(|c| c.contains("apr inspect")));
}

#[test]
fn test_generate_investigation_commands_conv_with_ga() {
    let enhancer = OracleEnhancer::new();
    let evidence = Evidence::falsified(
        "F-CONV-G-A",
        make_test_scenario(),
        "Conversion failed",
        "output",
        1000,
    );

    let commands = enhancer.generate_investigation_commands(&evidence);
    // Should have CONV commands, rosetta, and G-A specific convert command
    assert!(commands.iter().any(|c| c.contains("apr inspect")));
    assert!(commands.iter().any(|c| c.contains("transpose_q4k")));
    assert!(commands.iter().any(|c| c.contains("rosetta")));
    assert!(commands.iter().any(|c| c.contains("apr convert")));
}

#[test]
fn test_generate_investigation_commands_conv_without_ga() {
    let enhancer = OracleEnhancer::new();
    let evidence = Evidence::falsified(
        "F-CONV-001",
        make_test_scenario(),
        "Conversion failed",
        "output",
        1000,
    );

    let commands = enhancer.generate_investigation_commands(&evidence);
    // CONV commands present but no G-A specific convert command
    assert!(commands.iter().any(|c| c.contains("apr inspect")));
    assert!(!commands.iter().any(|c| c.contains("apr convert")));
}

#[test]
fn test_builder_chaining() {
    let enhancer = OracleEnhancer::new()
        .with_timeout(Duration::from_secs(5))
        .with_min_relevance(0.75);
    assert_eq!(enhancer.timeout, Duration::from_secs(5));
    assert!((enhancer.min_relevance - 0.75).abs() < f32::EPSILON);
}

#[test]
fn test_generate_checklist_markdown_empty_context() {
    let context = OracleContext::default();
    let md = generate_checklist_markdown("empty-model", 0, "F", 0, 0, &context);
    assert!(md.contains("# Falsification Checklist: empty-model"));
    assert!(md.contains("MQS Score:** 0/1000"));
    assert!(md.contains("Failures:** 0/0"));
    // No sections for empty lists
    assert!(!md.contains("## Root Cause Hypotheses"));
    assert!(!md.contains("## Investigation Commands"));
    assert!(!md.contains("## Cross-References"));
}

#[test]
fn test_generate_checklist_markdown_with_evidence_against() {
    let context = OracleContext {
        oracle_available: true,
        checklist: vec![],
        hypotheses: vec![RankedHypothesis {
            id: "H1".to_string(),
            description: "Test hypothesis".to_string(),
            confidence: Confidence::Medium,
            evidence_for: vec!["Some evidence".to_string()],
            evidence_against: vec!["Counter evidence".to_string()],
        }],
        cross_references: vec![],
        investigation_commands: vec![],
        query_latency_ms: 0,
    };

    let md = generate_checklist_markdown("test-model", 500, "C", 10, 3, &context);
    assert!(md.contains("Evidence For:"));
    assert!(md.contains("Some evidence"));
    assert!(md.contains("Evidence Against:"));
    assert!(md.contains("Counter evidence"));
}

#[test]
fn test_enhance_failures_multiple_failures() {
    let enhancer = OracleEnhancer::new();
    let evidences = vec![
        Evidence::falsified("F-CONV-001", make_test_scenario(), "err1", "out1", 100),
        Evidence::corroborated("F-TEST-001", make_test_scenario(), "ok", 200),
        Evidence::falsified("F-LOAD-001", make_test_scenario(), "err2", "out2", 300),
        Evidence::timeout("F-INF-001", make_test_scenario(), 5000),
    ];

    let results = enhancer.enhance_failures(&evidences);
    assert_eq!(results.len(), 3, "Should have 3 failure enhancements");
}

#[test]
fn test_generate_static_checklist_no_match() {
    let enhancer = OracleEnhancer::new();
    let evidence = Evidence::falsified(
        "F-LOAD-001",
        make_test_scenario(),
        "Load failed",
        "output",
        1000,
    );

    let checklist = enhancer.generate_static_checklist(&evidence);
    assert!(
        checklist.is_empty(),
        "Non-CONV, non-extension failure should produce empty static checklist"
    );
}

#[test]
fn test_generate_static_checklist_both_conv_and_extension() {
    let enhancer = OracleEnhancer::new();
    let evidence = Evidence::falsified(
        "F-CONV-001",
        make_test_scenario(),
        "No file extension found",
        "output",
        1000,
    );

    let checklist = enhancer.generate_static_checklist(&evidence);
    assert_eq!(
        checklist.len(),
        2,
        "Both CONV and extension branches should match"
    );
    assert!(checklist.iter().any(|c| c.gate_id == "F-LAYOUT-002"));
    assert!(checklist.iter().any(|c| c.gate_id == "F-PATH-EXT"));
}

#[test]
fn test_oracle_context_default() {
    let context = OracleContext::default();
    assert!(!context.oracle_available);
    assert!(context.checklist.is_empty());
    assert!(context.hypotheses.is_empty());
    assert!(context.cross_references.is_empty());
    assert!(context.investigation_commands.is_empty());
    assert_eq!(context.query_latency_ms, 0);
}

#[test]
fn test_generate_checklist_all_branches_simultaneously() {
    // Craft evidence that hits CONV + G-A + INF + "No file extension" + "diff"
    let enhancer = OracleEnhancer::new();
    let evidence = Evidence::falsified(
        "F-CONV-G-A-INF-001",
        make_test_scenario(),
        "No file extension diff found",
        "output",
        1000,
    );

    let checklist = enhancer.generate_checklist_from_gate(&evidence);
    // F-CONV → LAYOUT-002 (with diff → Falsified)
    assert!(checklist.iter().any(|c| c.gate_id == "F-LAYOUT-002"));
    // "No file extension" → PATH-EXT
    assert!(checklist.iter().any(|c| c.gate_id == "F-PATH-EXT"));
    // CONV + G-A → CONV-TRANSPOSE
    assert!(checklist.iter().any(|c| c.gate_id == "F-CONV-TRANSPOSE"));
    // INF → CONV-INF-EQ
    assert!(checklist.iter().any(|c| c.gate_id == "F-CONV-INF-EQ"));
    assert_eq!(checklist.len(), 4);
}
