//! RAG-Optimized Markdown Export
//!
//! Generates markdown reports optimized for semantic chunking and RAG retrieval.
//! Uses headers and structure that align with batuta's SemanticChunker separators.
//!
//! # RAG Integration
//!
//! The generated markdown uses:
//! - `## ` and `### ` headers for semantic boundaries
//! - Structured data tables for easy extraction
//! - Code blocks for reproducible commands
//! - Consistent gate ID references for cross-linking

use apr_qa_runner::{Evidence, EvidenceCollector};

use crate::mqs::MqsScore;
use crate::popperian::PopperianScore;

/// Generate RAG-optimized markdown report for a model qualification
///
/// # Arguments
///
/// * `mqs` - Model Qualification Score
/// * `popperian` - Popperian falsification score
/// * `collector` - Evidence collector with test results
///
/// # Returns
///
/// Markdown string optimized for RAG indexing
#[must_use]
pub fn generate_rag_markdown(
    mqs: &MqsScore,
    popperian: &PopperianScore,
    collector: &EvidenceCollector,
) -> String {
    let mut md = String::with_capacity(8192);

    // Title with model ID (RAG: searchable by model name)
    md.push_str(&format!("# Model Qualification: {}\n\n", mqs.model_id));

    // Summary section (RAG: high-level overview chunk)
    md.push_str("## Summary\n\n");
    md.push_str(&format!(
        "- **MQS Score**: {}/1000 ({:.1} normalized, {})\n",
        mqs.raw_score, mqs.normalized_score, mqs.grade
    ));
    md.push_str(&format!("- **Status**: {}\n", qualification_status(mqs)));
    md.push_str(&format!(
        "- **Tests**: {} passed / {} failed / {} total\n",
        mqs.tests_passed, mqs.tests_failed, mqs.total_tests
    ));
    md.push_str(&format!(
        "- **Black Swans**: {}\n",
        popperian.black_swan_count
    ));
    md.push_str(&format!(
        "- **Corroboration Rate**: {:.1}%\n\n",
        popperian.corroboration_ratio * 100.0
    ));

    // Gateway section (RAG: critical pass/fail info)
    md.push_str("## Gateway Checks\n\n");
    md.push_str("| Gateway | Status | Description |\n");
    md.push_str("|---------|--------|-------------|\n");
    for gw in &mqs.gateways {
        let status = if gw.passed { "✓ PASS" } else { "✗ FAIL" };
        let desc = if let Some(reason) = &gw.failure_reason {
            format!("{} - {}", gw.description, reason)
        } else {
            gw.description.clone()
        };
        md.push_str(&format!("| {} | {} | {} |\n", gw.id, status, desc));
    }
    md.push('\n');

    // Category breakdown (RAG: detailed scoring)
    md.push_str("## Category Scores\n\n");
    md.push_str("| Category | Score | Max | Percentage |\n");
    md.push_str("|----------|-------|-----|------------|\n");
    for (cat, (score, max)) in mqs.categories.breakdown() {
        let pct = if max > 0 {
            (score as f64 / max as f64) * 100.0
        } else {
            0.0
        };
        md.push_str(&format!(
            "| {} | {} | {} | {:.1}% |\n",
            cat, score, max, pct
        ));
    }
    md.push('\n');

    // Falsifications section (RAG: failure analysis)
    if !popperian.falsifications.is_empty() {
        md.push_str("## Falsifications\n\n");
        for (i, falsification) in popperian.falsifications.iter().enumerate() {
            md.push_str(&format!("### {}: {}\n\n", i + 1, falsification.gate_id));
            md.push_str(&format!("- **Hypothesis**: {}\n", falsification.hypothesis));
            md.push_str(&format!("- **Evidence**: {}\n", falsification.evidence));
            md.push_str(&format!("- **Severity**: {}/5\n", falsification.severity));
            if falsification.is_black_swan {
                md.push_str("- **Black Swan**: Yes (rare, high-impact failure)\n");
            }
            md.push_str(&format!(
                "- **Occurrences**: {}\n\n",
                falsification.occurrence_count
            ));
        }
    }

    // Test Results by Category (RAG: detailed gate results)
    md.push_str("## Test Results by Category\n\n");
    for category in &["QUAL", "PERF", "STAB", "COMP", "EDGE", "REGR"] {
        let category_evidence: Vec<&Evidence> = collector
            .all()
            .iter()
            .filter(|e| extract_category(&e.gate_id) == *category)
            .collect();

        if category_evidence.is_empty() {
            continue;
        }

        md.push_str(&format!("### {} Tests\n\n", category));

        let passed = category_evidence
            .iter()
            .filter(|e| e.outcome.is_pass())
            .count();
        let total = category_evidence.len();
        md.push_str(&format!(
            "Pass rate: {}/{} ({:.1}%)\n\n",
            passed,
            total,
            (passed as f64 / total as f64) * 100.0
        ));

        // Show failures first (more relevant for RAG queries about problems)
        let failures: Vec<_> = category_evidence
            .iter()
            .filter(|e| e.outcome.is_fail())
            .collect();

        if !failures.is_empty() {
            md.push_str("**Failures:**\n\n");
            for e in failures.iter().take(10) {
                // Limit to avoid huge files
                md.push_str(&format!(
                    "- `{}`: {} ({:?}, {}ms)\n",
                    e.gate_id, e.reason, e.outcome, e.metrics.duration_ms
                ));
            }
            if failures.len() > 10 {
                md.push_str(&format!(
                    "- ... and {} more failures\n",
                    failures.len() - 10
                ));
            }
            md.push('\n');
        }
    }

    // Penalties section
    if !mqs.penalties.is_empty() {
        md.push_str("## Penalties Applied\n\n");
        md.push_str("| Code | Description | Points |\n");
        md.push_str("|------|-------------|--------|\n");
        for penalty in &mqs.penalties {
            md.push_str(&format!(
                "| {} | {} | -{} |\n",
                penalty.code, penalty.description, penalty.points
            ));
        }
        md.push_str(&format!(
            "\n**Total Penalty**: -{} points\n\n",
            mqs.total_penalty
        ));
    }

    // Popperian Analysis (RAG: scientific methodology)
    md.push_str("## Popperian Analysis\n\n");
    md.push_str(&format!(
        "- **Hypotheses Tested**: {}\n",
        popperian.hypotheses_tested
    ));
    md.push_str(&format!("- **Corroborated**: {}\n", popperian.corroborated));
    md.push_str(&format!("- **Falsified**: {}\n", popperian.falsified));
    md.push_str(&format!(
        "- **Severity-Weighted Score**: {:.2}\n",
        popperian.severity_weighted_score
    ));
    md.push_str(&format!(
        "- **Confidence Level**: {:.1}%\n",
        popperian.confidence_level * 100.0
    ));
    md.push_str(&format!(
        "- **Reproducibility Index**: {:.2}\n\n",
        popperian.reproducibility_index
    ));

    // Metadata footer (RAG: versioning and timestamps)
    md.push_str("## Metadata\n\n");
    md.push_str(&format!("- **Model ID**: {}\n", mqs.model_id));
    md.push_str(&format!("- **Gateways Passed**: {}\n", mqs.gateways_passed));
    md.push_str(&format!("- **Qualifies**: {}\n", mqs.qualifies()));
    md.push_str(&format!(
        "- **Production Ready**: {}\n",
        mqs.is_production_ready()
    ));

    md
}

/// Generate a compact summary for index files
#[must_use]
pub fn generate_index_entry(mqs: &MqsScore) -> String {
    format!(
        "| {} | {}/1000 | {} | {} | {} |\n",
        mqs.model_id,
        mqs.raw_score,
        mqs.grade,
        qualification_status(mqs),
        if mqs.is_production_ready() {
            "Yes"
        } else {
            "No"
        }
    )
}

/// Get qualification status string
fn qualification_status(mqs: &MqsScore) -> &'static str {
    if !mqs.gateways_passed {
        "REJECTED (Gateway Failure)"
    } else if mqs.is_production_ready() {
        "CERTIFIED"
    } else if mqs.qualifies() {
        "PROVISIONAL"
    } else {
        "REJECTED"
    }
}

/// Extract category from gate ID
fn extract_category(gate_id: &str) -> String {
    gate_id.split('-').nth(1).unwrap_or("UNKNOWN").to_string()
}

/// Generate evidence detail markdown for a single test
#[must_use]
pub fn generate_evidence_detail(evidence: &Evidence) -> String {
    let mut md = String::with_capacity(512);

    md.push_str(&format!("### {}\n\n", evidence.gate_id));
    md.push_str(&format!("- **Outcome**: {:?}\n", evidence.outcome));
    md.push_str(&format!("- **Reason**: {}\n", evidence.reason));
    md.push_str(&format!(
        "- **Duration**: {}ms\n",
        evidence.metrics.duration_ms
    ));

    if let Some(tps) = evidence.metrics.tokens_per_second {
        md.push_str(&format!("- **Tokens/sec**: {:.1}\n", tps));
    }
    if let Some(ttft) = evidence.metrics.time_to_first_token_ms {
        md.push_str(&format!("- **Time to First Token**: {:.1}ms\n", ttft));
    }
    if let Some(mem) = evidence.metrics.memory_peak_mb {
        md.push_str(&format!("- **Peak Memory**: {} MB\n", mem));
    }

    // Scenario details
    md.push_str("\n**Scenario**:\n");
    md.push_str(&format!("- Model: {}\n", evidence.scenario.model));
    md.push_str(&format!("- Backend: {:?}\n", evidence.scenario.backend));
    md.push_str(&format!("- Format: {:?}\n", evidence.scenario.format));
    md.push_str(&format!("- Seed: {}\n", evidence.scenario.seed));

    if !evidence.output.is_empty() {
        let output_preview: String = evidence.output.chars().take(200).collect();
        md.push_str(&format!(
            "\n**Output Preview**:\n```\n{}\n```\n",
            output_preview
        ));
    }

    md.push('\n');
    md
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mqs::{CategoryScores, GatewayResult, Penalty};
    use crate::popperian::FalsificationDetail;
    use apr_qa_gen::{Backend, Format, Modality, ModelId, QaScenario};

    fn test_scenario() -> QaScenario {
        QaScenario::new(
            ModelId::new("test", "model"),
            Modality::Run,
            Backend::Cpu,
            Format::Gguf,
            "2+2=".to_string(),
            42,
        )
    }

    fn test_mqs_score() -> MqsScore {
        MqsScore {
            model_id: "qwen2.5-coder-7b".to_string(),
            raw_score: 847,
            normalized_score: 84.7,
            grade: "B+".to_string(),
            gateways: vec![
                GatewayResult::passed("G1", "Model loads successfully"),
                GatewayResult::passed("G2", "Basic inference works"),
                GatewayResult::passed("G3", "No crashes"),
                GatewayResult::passed("G4", "Output is not garbage"),
            ],
            gateways_passed: true,
            categories: CategoryScores {
                qual: 180,
                perf: 120,
                stab: 180,
                comp: 130,
                edge: 120,
                regr: 117,
            },
            total_tests: 100,
            tests_passed: 85,
            tests_failed: 15,
            penalties: vec![Penalty {
                code: "TIMEOUT".to_string(),
                description: "3 timeouts detected".to_string(),
                points: 30,
            }],
            total_penalty: 30,
        }
    }

    fn test_popperian_score() -> PopperianScore {
        PopperianScore {
            model_id: "qwen2.5-coder-7b".to_string(),
            hypotheses_tested: 100,
            corroborated: 85,
            falsified: 15,
            inconclusive: 0,
            corroboration_ratio: 0.85,
            severity_weighted_score: 0.82,
            confidence_level: 0.95,
            reproducibility_index: 0.98,
            black_swan_count: 0,
            falsifications: vec![FalsificationDetail {
                gate_id: "F-PERF-042".to_string(),
                hypothesis: "Inference completes under 100ms".to_string(),
                evidence: "Actual: 142ms".to_string(),
                severity: 3,
                is_black_swan: false,
                occurrence_count: 2,
            }],
        }
    }

    #[test]
    fn test_generate_rag_markdown_basic() {
        let mqs = test_mqs_score();
        let popperian = test_popperian_score();
        let collector = EvidenceCollector::new();

        let md = generate_rag_markdown(&mqs, &popperian, &collector);

        assert!(md.contains("# Model Qualification: qwen2.5-coder-7b"));
        assert!(md.contains("## Summary"));
        assert!(md.contains("847/1000"));
        assert!(md.contains("B+"));
    }

    #[test]
    fn test_generate_rag_markdown_contains_gateways() {
        let mqs = test_mqs_score();
        let popperian = test_popperian_score();
        let collector = EvidenceCollector::new();

        let md = generate_rag_markdown(&mqs, &popperian, &collector);

        assert!(md.contains("## Gateway Checks"));
        assert!(md.contains("G1"));
        assert!(md.contains("G2"));
        assert!(md.contains("G3"));
        assert!(md.contains("G4"));
        assert!(md.contains("✓ PASS"));
    }

    #[test]
    fn test_generate_rag_markdown_contains_categories() {
        let mqs = test_mqs_score();
        let popperian = test_popperian_score();
        let collector = EvidenceCollector::new();

        let md = generate_rag_markdown(&mqs, &popperian, &collector);

        assert!(md.contains("## Category Scores"));
        assert!(md.contains("QUAL"));
        assert!(md.contains("PERF"));
        assert!(md.contains("STAB"));
    }

    #[test]
    fn test_generate_rag_markdown_contains_falsifications() {
        let mqs = test_mqs_score();
        let popperian = test_popperian_score();
        let collector = EvidenceCollector::new();

        let md = generate_rag_markdown(&mqs, &popperian, &collector);

        assert!(md.contains("## Falsifications"));
        assert!(md.contains("F-PERF-042"));
        assert!(md.contains("**Severity**: 3/5"));
    }

    #[test]
    fn test_generate_rag_markdown_contains_penalties() {
        let mqs = test_mqs_score();
        let popperian = test_popperian_score();
        let collector = EvidenceCollector::new();

        let md = generate_rag_markdown(&mqs, &popperian, &collector);

        assert!(md.contains("## Penalties Applied"));
        assert!(md.contains("TIMEOUT"));
        assert!(md.contains("-30"));
    }

    #[test]
    fn test_generate_rag_markdown_contains_popperian() {
        let mqs = test_mqs_score();
        let popperian = test_popperian_score();
        let collector = EvidenceCollector::new();

        let md = generate_rag_markdown(&mqs, &popperian, &collector);

        assert!(md.contains("## Popperian Analysis"));
        assert!(md.contains("Hypotheses Tested"));
        assert!(md.contains("Corroboration Rate"));
    }

    #[test]
    fn test_generate_rag_markdown_contains_metadata() {
        let mqs = test_mqs_score();
        let popperian = test_popperian_score();
        let collector = EvidenceCollector::new();

        let md = generate_rag_markdown(&mqs, &popperian, &collector);

        assert!(md.contains("## Metadata"));
        assert!(md.contains("Production Ready"));
    }

    #[test]
    fn test_generate_rag_markdown_with_evidence() {
        let mqs = test_mqs_score();
        let popperian = test_popperian_score();
        let mut collector = EvidenceCollector::new();

        collector.add(Evidence::corroborated(
            "F-QUAL-001",
            test_scenario(),
            "4",
            100,
        ));
        collector.add(Evidence::falsified(
            "F-QUAL-002",
            test_scenario(),
            "Wrong answer",
            "5",
            200,
        ));

        let md = generate_rag_markdown(&mqs, &popperian, &collector);

        assert!(md.contains("## Test Results by Category"));
        assert!(md.contains("QUAL Tests"));
        assert!(md.contains("F-QUAL-002"));
    }

    #[test]
    fn test_generate_index_entry() {
        let mqs = test_mqs_score();
        let entry = generate_index_entry(&mqs);

        assert!(entry.contains("qwen2.5-coder-7b"));
        assert!(entry.contains("847/1000"));
        assert!(entry.contains("B+"));
        assert!(entry.contains("PROVISIONAL"));
    }

    #[test]
    fn test_qualification_status_certified() {
        let mut mqs = test_mqs_score();
        mqs.normalized_score = 95.0;
        mqs.gateways_passed = true;

        assert_eq!(qualification_status(&mqs), "CERTIFIED");
    }

    #[test]
    fn test_qualification_status_provisional() {
        let mqs = test_mqs_score();
        assert_eq!(qualification_status(&mqs), "PROVISIONAL");
    }

    #[test]
    fn test_qualification_status_rejected_gateway() {
        let mut mqs = test_mqs_score();
        mqs.gateways_passed = false;

        assert_eq!(qualification_status(&mqs), "REJECTED (Gateway Failure)");
    }

    #[test]
    fn test_qualification_status_rejected_score() {
        let mut mqs = test_mqs_score();
        mqs.normalized_score = 50.0;

        assert_eq!(qualification_status(&mqs), "REJECTED");
    }

    #[test]
    fn test_extract_category() {
        assert_eq!(extract_category("F-QUAL-001"), "QUAL");
        assert_eq!(extract_category("F-PERF-042"), "PERF");
        assert_eq!(extract_category("F-STAB-100"), "STAB");
        assert_eq!(extract_category("F-COMP-001"), "COMP");
        assert_eq!(extract_category("F-EDGE-001"), "EDGE");
        assert_eq!(extract_category("F-REGR-001"), "REGR");
        assert_eq!(extract_category("UNKNOWN"), "UNKNOWN");
    }

    #[test]
    fn test_generate_evidence_detail() {
        let evidence = Evidence::corroborated("F-QUAL-001", test_scenario(), "output text", 150);
        let md = generate_evidence_detail(&evidence);

        assert!(md.contains("### F-QUAL-001"));
        assert!(md.contains("Outcome"));
        assert!(md.contains("Corroborated"));
        assert!(md.contains("150ms"));
    }

    #[test]
    fn test_generate_evidence_detail_with_metrics() {
        let mut evidence = Evidence::corroborated("F-PERF-001", test_scenario(), "output", 100);
        evidence.metrics.tokens_per_second = Some(150.5);
        evidence.metrics.time_to_first_token_ms = Some(25.3);
        evidence.metrics.memory_peak_mb = Some(4096);

        let md = generate_evidence_detail(&evidence);

        assert!(md.contains("**Tokens/sec**: 150.5"));
        assert!(md.contains("**Time to First Token**: 25.3ms"));
        assert!(md.contains("**Peak Memory**: 4096 MB"));
    }

    #[test]
    fn test_generate_rag_markdown_no_penalties() {
        let mut mqs = test_mqs_score();
        mqs.penalties.clear();
        mqs.total_penalty = 0;

        let popperian = test_popperian_score();
        let collector = EvidenceCollector::new();

        let md = generate_rag_markdown(&mqs, &popperian, &collector);

        // Should not contain penalties section if no penalties
        assert!(!md.contains("## Penalties Applied"));
    }

    #[test]
    fn test_generate_rag_markdown_no_falsifications() {
        let mqs = test_mqs_score();
        let mut popperian = test_popperian_score();
        popperian.falsifications.clear();

        let collector = EvidenceCollector::new();

        let md = generate_rag_markdown(&mqs, &popperian, &collector);

        // Should still have the section header but no falsifications listed
        // Actually, with empty falsifications, the section is skipped
        assert!(!md.contains("### 1:"));
    }

    #[test]
    fn test_generate_rag_markdown_gateway_failure() {
        let mut mqs = test_mqs_score();
        mqs.gateways = vec![
            GatewayResult::passed("G1", "Model loads successfully"),
            GatewayResult::failed("G2", "Basic inference works", "Inference failed"),
            GatewayResult::passed("G3", "No crashes"),
            GatewayResult::passed("G4", "Output is not garbage"),
        ];
        mqs.gateways_passed = false;

        let popperian = test_popperian_score();
        let collector = EvidenceCollector::new();

        let md = generate_rag_markdown(&mqs, &popperian, &collector);

        assert!(md.contains("✗ FAIL"));
        assert!(md.contains("Inference failed"));
    }

    #[test]
    fn test_generate_rag_markdown_black_swan() {
        let mqs = test_mqs_score();
        let mut popperian = test_popperian_score();
        popperian.black_swan_count = 2;
        popperian.falsifications = vec![FalsificationDetail {
            gate_id: "F-STAB-001".to_string(),
            hypothesis: "Model does not crash".to_string(),
            evidence: "SIGSEGV".to_string(),
            severity: 5,
            is_black_swan: true,
            occurrence_count: 1,
        }];

        let collector = EvidenceCollector::new();

        let md = generate_rag_markdown(&mqs, &popperian, &collector);

        assert!(md.contains("Black Swans**: 2"));
        assert!(md.contains("Black Swan**: Yes"));
    }

    #[test]
    fn test_generate_rag_markdown_multiple_categories() {
        let mqs = test_mqs_score();
        let popperian = test_popperian_score();
        let mut collector = EvidenceCollector::new();

        // Add evidence for multiple categories
        collector.add(Evidence::corroborated(
            "F-QUAL-001",
            test_scenario(),
            "ok",
            100,
        ));
        collector.add(Evidence::corroborated(
            "F-PERF-001",
            test_scenario(),
            "ok",
            100,
        ));
        collector.add(Evidence::falsified(
            "F-STAB-001",
            test_scenario(),
            "fail",
            "",
            100,
        ));
        collector.add(Evidence::corroborated(
            "F-COMP-001",
            test_scenario(),
            "ok",
            100,
        ));
        collector.add(Evidence::corroborated(
            "F-EDGE-001",
            test_scenario(),
            "ok",
            100,
        ));
        collector.add(Evidence::corroborated(
            "F-REGR-001",
            test_scenario(),
            "ok",
            100,
        ));

        let md = generate_rag_markdown(&mqs, &popperian, &collector);

        assert!(md.contains("### QUAL Tests"));
        assert!(md.contains("### PERF Tests"));
        assert!(md.contains("### STAB Tests"));
        assert!(md.contains("### COMP Tests"));
        assert!(md.contains("### EDGE Tests"));
        assert!(md.contains("### REGR Tests"));
    }

    #[test]
    fn test_generate_rag_markdown_many_failures() {
        let mqs = test_mqs_score();
        let popperian = test_popperian_score();
        let mut collector = EvidenceCollector::new();

        // Add more than 10 failures to test truncation
        for i in 0..15 {
            collector.add(Evidence::falsified(
                &format!("F-QUAL-{:03}", i),
                test_scenario(),
                &format!("Failure {}", i),
                "",
                100,
            ));
        }

        let md = generate_rag_markdown(&mqs, &popperian, &collector);

        assert!(md.contains("... and 5 more failures"));
    }

    #[test]
    fn test_generate_evidence_detail_falsified() {
        let evidence = Evidence::falsified(
            "F-EDGE-001",
            test_scenario(),
            "Empty input caused crash",
            "",
            50,
        );
        let md = generate_evidence_detail(&evidence);

        assert!(md.contains("Falsified"));
        assert!(md.contains("Empty input caused crash"));
    }

    #[test]
    fn test_generate_evidence_detail_timeout() {
        let evidence = Evidence::timeout("F-PERF-001", test_scenario(), 30000);
        let md = generate_evidence_detail(&evidence);

        assert!(md.contains("Timeout"));
        assert!(md.contains("30000ms"));
    }

    #[test]
    fn test_generate_evidence_detail_crashed() {
        let evidence = Evidence::crashed("F-STAB-001", test_scenario(), "SIGSEGV", -11, 100);
        let md = generate_evidence_detail(&evidence);

        assert!(md.contains("Crashed"));
        assert!(md.contains("-11"));
    }

    #[test]
    fn test_generate_evidence_detail_with_output() {
        let long_output = "a".repeat(300);
        let evidence = Evidence::corroborated("F-QUAL-001", test_scenario(), &long_output, 100);
        let md = generate_evidence_detail(&evidence);

        // Should truncate to 200 chars
        assert!(md.contains("Output Preview"));
        assert!(!md.contains(&long_output)); // Full output should not be present
    }

    #[test]
    fn test_markdown_uses_correct_headers() {
        let mqs = test_mqs_score();
        let popperian = test_popperian_score();
        let collector = EvidenceCollector::new();

        let md = generate_rag_markdown(&mqs, &popperian, &collector);

        // Check for RAG-friendly headers
        assert!(md.contains("# Model Qualification:"));
        assert!(md.contains("## Summary"));
        assert!(md.contains("## Gateway Checks"));
        assert!(md.contains("## Category Scores"));
        assert!(md.contains("## Popperian Analysis"));
        assert!(md.contains("## Metadata"));
    }
}
