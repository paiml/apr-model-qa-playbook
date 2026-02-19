#[test]
fn test_falsify_cert_003_grade_derivation() {
    // Helper to derive grade from score
    let grade_for = |score: u32| -> String {
        CertificationRow {
            mqs_score: score,
            ..CertificationRow::default()
        }
        .derive_grade()
    };

    // A grade: 900-1000
    assert_eq!(grade_for(1000), "A", "1000 should be A");
    assert_eq!(grade_for(950), "A", "950 should be A");
    assert_eq!(grade_for(900), "A", "900 (lower bound) should be A");

    // B grade: 800-899
    assert_eq!(grade_for(899), "B", "899 (upper bound of B) should be B");
    assert_eq!(grade_for(850), "B", "850 should be B");
    assert_eq!(grade_for(800), "B", "800 (lower bound) should be B");

    // C grade: 600-799
    assert_eq!(grade_for(799), "C", "799 (upper bound of C) should be C");
    assert_eq!(grade_for(700), "C", "700 should be C");
    assert_eq!(grade_for(600), "C", "600 (lower bound) should be C");

    // D grade: 400-599
    assert_eq!(grade_for(599), "D", "599 (upper bound of D) should be D");
    assert_eq!(grade_for(500), "D", "500 should be D");
    assert_eq!(grade_for(400), "D", "400 (lower bound) should be D");

    // F grade: 0-399
    assert_eq!(grade_for(399), "F", "399 (upper bound of F) should be F");
    assert_eq!(grade_for(200), "F", "200 should be F");
    assert_eq!(grade_for(0), "F", "0 should be F");
}
