//! Popperian Falsification Scoring
//!
//! Implements scientific scoring based on Karl Popper's falsification methodology.
//!
//! ## Popperian Principles
//!
//! 1. **Falsifiability**: A theory must be testable and potentially refutable
//! 2. **Corroboration**: Surviving rigorous testing increases confidence
//! 3. **Severity**: Harder tests provide stronger evidence
//! 4. **Reproducibility**: Results must be independently verifiable

use apr_qa_runner::{EvidenceCollector, Outcome};
use serde::{Deserialize, Serialize};

/// Popperian score with falsification details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PopperianScore {
    /// Model identifier
    pub model_id: String,
    /// Total hypotheses tested
    pub hypotheses_tested: usize,
    /// Hypotheses not falsified (corroborated)
    pub corroborated: usize,
    /// Hypotheses falsified
    pub falsified: usize,
    /// Inconclusive tests (timeout, skip)
    pub inconclusive: usize,
    /// Corroboration ratio (0.0 - 1.0)
    pub corroboration_ratio: f64,
    /// Severity-weighted score (accounts for test difficulty)
    pub severity_weighted_score: f64,
    /// Confidence level (0.0 - 1.0)
    pub confidence_level: f64,
    /// Reproducibility index (based on seed consistency)
    pub reproducibility_index: f64,
    /// Black swan events (rare, high-impact failures)
    pub black_swan_count: usize,
    /// Falsification details
    pub falsifications: Vec<FalsificationDetail>,
}

impl PopperianScore {
    /// Check if the model has strong corroboration
    #[must_use]
    pub fn is_strongly_corroborated(&self) -> bool {
        self.corroboration_ratio >= 0.95 && self.black_swan_count == 0
    }

    /// Check if black swan events were detected
    #[must_use]
    pub fn has_black_swans(&self) -> bool {
        self.black_swan_count > 0
    }

    /// Get falsification summary
    #[must_use]
    pub fn falsification_summary(&self) -> String {
        if self.falsified == 0 {
            "No falsifications - strongly corroborated".to_string()
        } else {
            format!(
                "{} of {} hypotheses falsified ({:.1}%)",
                self.falsified,
                self.hypotheses_tested,
                (self.falsified as f64 / self.hypotheses_tested as f64) * 100.0
            )
        }
    }
}

/// Detail about a specific falsification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FalsificationDetail {
    /// Gate ID that was falsified
    pub gate_id: String,
    /// Hypothesis that was falsified
    pub hypothesis: String,
    /// Evidence of falsification
    pub evidence: String,
    /// Severity (1-5, 5 being most severe)
    pub severity: u8,
    /// Is this a black swan event?
    pub is_black_swan: bool,
    /// Reproducibility (how many times this was observed)
    pub occurrence_count: usize,
}

/// Popperian score calculator
#[derive(Debug, Default)]
pub struct PopperianCalculator {
    /// Weight for high-severity tests
    severity_weights: [f64; 5],
}

impl PopperianCalculator {
    /// Create a new calculator with default severity weights
    #[must_use]
    pub fn new() -> Self {
        Self {
            // Higher severity tests contribute more to confidence
            severity_weights: [1.0, 1.5, 2.0, 2.5, 3.0],
        }
    }

    /// Calculate Popperian score from evidence
    #[must_use]
    pub fn calculate(&self, model_id: &str, evidence: &EvidenceCollector) -> PopperianScore {
        let all_evidence = evidence.all();

        let mut corroborated = 0;
        let mut falsified = 0;
        let mut inconclusive = 0;
        let mut severity_total = 0.0;
        let mut severity_passed = 0.0;
        let mut falsifications = Vec::new();
        let mut black_swan_count = 0;

        // Group failures by gate_id for reproducibility analysis
        let mut failure_counts: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();

        for e in all_evidence {
            let severity = self.determine_severity(&e.gate_id);
            let weight = self.severity_weights[severity.saturating_sub(1) as usize];
            severity_total += weight;

            match e.outcome {
                Outcome::Corroborated => {
                    corroborated += 1;
                    severity_passed += weight;
                }
                Outcome::Falsified | Outcome::Crashed => {
                    falsified += 1;
                    *failure_counts.entry(e.gate_id.clone()).or_insert(0) += 1;

                    // Black swan: crash or severe unexpected failure
                    let is_black_swan = e.outcome == Outcome::Crashed || severity >= 4;
                    if is_black_swan {
                        black_swan_count += 1;
                    }

                    falsifications.push(FalsificationDetail {
                        gate_id: e.gate_id.clone(),
                        hypothesis: self.gate_to_hypothesis(&e.gate_id),
                        evidence: e.reason.clone(),
                        severity,
                        is_black_swan,
                        occurrence_count: 1, // Will be updated later
                    });
                }
                Outcome::Skipped | Outcome::Timeout => {
                    inconclusive += 1;
                }
            }
        }

        // Update occurrence counts
        for falsification in &mut falsifications {
            if let Some(&count) = failure_counts.get(&falsification.gate_id) {
                falsification.occurrence_count = count;
            }
        }

        // Deduplicate falsifications (keep highest severity per gate)
        falsifications.sort_by(|a, b| a.gate_id.cmp(&b.gate_id).then(b.severity.cmp(&a.severity)));
        falsifications.dedup_by(|a, b| a.gate_id == b.gate_id);

        let hypotheses_tested = corroborated + falsified;
        let corroboration_ratio = if hypotheses_tested > 0 {
            corroborated as f64 / hypotheses_tested as f64
        } else {
            0.0
        };

        let severity_weighted_score = if severity_total > 0.0 {
            severity_passed / severity_total
        } else {
            0.0
        };

        // Confidence level based on sample size and consistency
        let confidence_level = self.calculate_confidence(hypotheses_tested, corroboration_ratio);

        // Reproducibility based on failure consistency
        let reproducibility_index =
            self.calculate_reproducibility(&failure_counts, all_evidence.len());

        PopperianScore {
            model_id: model_id.to_string(),
            hypotheses_tested,
            corroborated,
            falsified,
            inconclusive,
            corroboration_ratio,
            severity_weighted_score,
            confidence_level,
            reproducibility_index,
            black_swan_count,
            falsifications,
        }
    }

    /// Determine severity from gate ID
    fn determine_severity(&self, gate_id: &str) -> u8 {
        // P0 gates are highest severity
        if gate_id.contains("-P0-") || gate_id.starts_with("G") {
            5
        } else if gate_id.contains("-P1-") {
            4
        } else if gate_id.contains("-P2-") {
            3
        } else if gate_id.contains("EDGE") || gate_id.contains("STAB") {
            3
        } else if gate_id.contains("PERF") {
            2
        } else {
            1
        }
    }

    /// Convert gate ID to human-readable hypothesis
    fn gate_to_hypothesis(&self, gate_id: &str) -> String {
        if gate_id.contains("QUAL") {
            "Model produces valid output".to_string()
        } else if gate_id.contains("PERF") {
            "Model meets performance requirements".to_string()
        } else if gate_id.contains("STAB") {
            "Model is stable under stress".to_string()
        } else if gate_id.contains("COMP") {
            "Model is compatible with configuration".to_string()
        } else if gate_id.contains("EDGE") {
            "Model handles edge cases correctly".to_string()
        } else if gate_id.contains("REGR") {
            "Model behavior is consistent".to_string()
        } else {
            format!("Hypothesis for {gate_id}")
        }
    }

    /// Calculate confidence level
    fn calculate_confidence(&self, n: usize, ratio: f64) -> f64 {
        if n == 0 {
            return 0.0;
        }

        // Wilson score interval lower bound approximation
        // Provides conservative confidence estimate
        let z = 1.96; // 95% confidence
        let n_f = n as f64;
        let denominator = 1.0 + z * z / n_f;
        let center = ratio + z * z / (2.0 * n_f);
        let spread = z * ((ratio * (1.0 - ratio) / n_f) + (z * z / (4.0 * n_f * n_f))).sqrt();

        ((center - spread) / denominator).clamp(0.0, 1.0)
    }

    /// Calculate reproducibility index
    fn calculate_reproducibility(
        &self,
        failure_counts: &std::collections::HashMap<String, usize>,
        total_tests: usize,
    ) -> f64 {
        if total_tests == 0 || failure_counts.is_empty() {
            return 1.0; // No failures = perfectly reproducible (trivially)
        }

        // Count consistent failures (appeared more than once)
        let consistent_failures: usize = failure_counts.values().filter(|&&count| count > 1).sum();

        let total_failures: usize = failure_counts.values().sum();

        if total_failures == 0 {
            1.0
        } else {
            // Higher ratio of consistent failures = more reproducible
            (consistent_failures as f64 / total_failures as f64).clamp(0.0, 1.0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use apr_qa_gen::{Backend, Format, Modality, ModelId, QaScenario};
    use apr_qa_runner::Evidence;

    fn test_scenario() -> QaScenario {
        QaScenario::new(
            ModelId::new("test", "model"),
            Modality::Run,
            Backend::Cpu,
            Format::Gguf,
            "test prompt".to_string(),
            42,
        )
    }

    #[test]
    fn test_popperian_all_corroborated() {
        let calculator = PopperianCalculator::new();
        let mut collector = EvidenceCollector::new();

        for i in 0..100 {
            collector.add(Evidence::corroborated(
                &format!("F-QUAL-{i:03}"),
                test_scenario(),
                "correct output",
                100,
            ));
        }

        let score = calculator.calculate("test/model", &collector);

        assert_eq!(score.corroborated, 100);
        assert_eq!(score.falsified, 0);
        assert!((score.corroboration_ratio - 1.0).abs() < 0.001);
        assert!(score.is_strongly_corroborated());
    }

    #[test]
    fn test_popperian_with_falsifications() {
        let calculator = PopperianCalculator::new();
        let mut collector = EvidenceCollector::new();

        // 90 corroborated
        for i in 0..90 {
            collector.add(Evidence::corroborated(
                &format!("F-QUAL-{i:03}"),
                test_scenario(),
                "correct",
                100,
            ));
        }

        // 10 falsified
        for i in 90..100 {
            collector.add(Evidence::falsified(
                &format!("F-QUAL-{i:03}"),
                test_scenario(),
                "wrong answer",
                "garbage",
                100,
            ));
        }

        let score = calculator.calculate("test/model", &collector);

        assert_eq!(score.corroborated, 90);
        assert_eq!(score.falsified, 10);
        assert!((score.corroboration_ratio - 0.9).abs() < 0.001);
        assert!(!score.is_strongly_corroborated());
    }

    #[test]
    fn test_popperian_black_swan_detection() {
        let calculator = PopperianCalculator::new();
        let mut collector = EvidenceCollector::new();

        // Normal passes
        for i in 0..95 {
            collector.add(Evidence::corroborated(
                &format!("F-QUAL-{i:03}"),
                test_scenario(),
                "ok",
                100,
            ));
        }

        // One crash (black swan)
        collector.add(Evidence::crashed(
            "F-QUAL-099",
            test_scenario(),
            "SIGSEGV",
            -11,
            0,
        ));

        let score = calculator.calculate("test/model", &collector);

        assert!(score.has_black_swans());
        assert_eq!(score.black_swan_count, 1);
        assert!(!score.is_strongly_corroborated());
    }

    #[test]
    fn test_severity_determination() {
        let calculator = PopperianCalculator::new();

        assert_eq!(calculator.determine_severity("G1-LOAD"), 5);
        assert_eq!(calculator.determine_severity("F-QUAL-P0-001"), 5);
        assert_eq!(calculator.determine_severity("F-QUAL-P1-001"), 4);
        assert_eq!(calculator.determine_severity("F-QUAL-P2-001"), 3);
        assert_eq!(calculator.determine_severity("F-EDGE-001"), 3);
        assert_eq!(calculator.determine_severity("F-PERF-001"), 2);
        assert_eq!(calculator.determine_severity("F-OTHER-001"), 1);
    }

    #[test]
    fn test_gate_to_hypothesis() {
        let calculator = PopperianCalculator::new();

        assert!(
            calculator
                .gate_to_hypothesis("F-QUAL-001")
                .contains("valid output")
        );
        assert!(
            calculator
                .gate_to_hypothesis("F-PERF-001")
                .contains("performance")
        );
        assert!(
            calculator
                .gate_to_hypothesis("F-STAB-001")
                .contains("stable")
        );
    }

    #[test]
    fn test_falsification_summary() {
        let score = PopperianScore {
            model_id: "test".to_string(),
            hypotheses_tested: 100,
            corroborated: 100,
            falsified: 0,
            inconclusive: 0,
            corroboration_ratio: 1.0,
            severity_weighted_score: 1.0,
            confidence_level: 0.95,
            reproducibility_index: 1.0,
            black_swan_count: 0,
            falsifications: vec![],
        };

        assert!(
            score
                .falsification_summary()
                .contains("strongly corroborated")
        );

        let score_with_failures = PopperianScore {
            falsified: 5,
            hypotheses_tested: 100,
            ..score
        };

        assert!(
            score_with_failures
                .falsification_summary()
                .contains("5 of 100")
        );
    }

    #[test]
    fn test_confidence_calculation() {
        let calculator = PopperianCalculator::new();

        // Small sample = lower confidence
        let small_conf = calculator.calculate_confidence(10, 0.9);
        // Large sample = higher confidence
        let large_conf = calculator.calculate_confidence(1000, 0.9);

        assert!(large_conf > small_conf);
    }

    #[test]
    fn test_reproducibility_no_failures() {
        let calculator = PopperianCalculator::new();
        let empty: std::collections::HashMap<String, usize> = std::collections::HashMap::new();

        let index = calculator.calculate_reproducibility(&empty, 100);
        assert!((index - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_reproducibility_with_consistent_failures() {
        let calculator = PopperianCalculator::new();
        let mut failures = std::collections::HashMap::new();
        failures.insert("F-001".to_string(), 5); // Consistent
        failures.insert("F-002".to_string(), 3); // Consistent

        let index = calculator.calculate_reproducibility(&failures, 100);
        // All failures are consistent (appeared more than once)
        assert!((index - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_reproducibility_with_sporadic_failures() {
        let calculator = PopperianCalculator::new();
        let mut failures = std::collections::HashMap::new();
        failures.insert("F-001".to_string(), 1); // Sporadic
        failures.insert("F-002".to_string(), 1); // Sporadic

        let index = calculator.calculate_reproducibility(&failures, 100);
        // No consistent failures
        assert!((index - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_reproducibility_zero_total() {
        let calculator = PopperianCalculator::new();
        let failures = std::collections::HashMap::new();

        let index = calculator.calculate_reproducibility(&failures, 0);
        assert!((index - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_confidence_zero_samples() {
        let calculator = PopperianCalculator::new();
        let conf = calculator.calculate_confidence(0, 0.9);
        assert!((conf - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_gate_to_hypothesis_comp() {
        let calculator = PopperianCalculator::new();
        assert!(
            calculator
                .gate_to_hypothesis("F-COMP-001")
                .contains("compatible")
        );
    }

    #[test]
    fn test_gate_to_hypothesis_edge() {
        let calculator = PopperianCalculator::new();
        assert!(
            calculator
                .gate_to_hypothesis("F-EDGE-001")
                .contains("edge cases")
        );
    }

    #[test]
    fn test_gate_to_hypothesis_regr() {
        let calculator = PopperianCalculator::new();
        assert!(
            calculator
                .gate_to_hypothesis("F-REGR-001")
                .contains("consistent")
        );
    }

    #[test]
    fn test_gate_to_hypothesis_unknown() {
        let calculator = PopperianCalculator::new();
        let result = calculator.gate_to_hypothesis("F-UNKNOWN-001");
        assert!(result.contains("F-UNKNOWN-001"));
    }

    #[test]
    fn test_popperian_score_has_black_swans() {
        let score = PopperianScore {
            model_id: "test".to_string(),
            hypotheses_tested: 100,
            corroborated: 99,
            falsified: 1,
            inconclusive: 0,
            corroboration_ratio: 0.99,
            severity_weighted_score: 0.99,
            confidence_level: 0.95,
            reproducibility_index: 1.0,
            black_swan_count: 1,
            falsifications: vec![],
        };
        assert!(score.has_black_swans());
    }

    #[test]
    fn test_popperian_score_no_black_swans() {
        let score = PopperianScore {
            model_id: "test".to_string(),
            hypotheses_tested: 100,
            corroborated: 90,
            falsified: 10,
            inconclusive: 0,
            corroboration_ratio: 0.9,
            severity_weighted_score: 0.9,
            confidence_level: 0.9,
            reproducibility_index: 1.0,
            black_swan_count: 0,
            falsifications: vec![],
        };
        assert!(!score.has_black_swans());
    }

    #[test]
    fn test_severity_stab() {
        let calculator = PopperianCalculator::new();
        assert_eq!(calculator.determine_severity("F-STAB-001"), 3);
    }

    #[test]
    fn test_falsification_detail_clone() {
        let detail = FalsificationDetail {
            gate_id: "F-001".to_string(),
            hypothesis: "test".to_string(),
            evidence: "failed".to_string(),
            occurrence_count: 1,
            severity: 3,
            is_black_swan: false,
        };
        let cloned = detail.clone();
        assert_eq!(cloned.gate_id, detail.gate_id);
    }

    #[test]
    fn test_popperian_score_serialize() {
        let score = PopperianScore {
            model_id: "test".to_string(),
            hypotheses_tested: 100,
            corroborated: 100,
            falsified: 0,
            inconclusive: 0,
            corroboration_ratio: 1.0,
            severity_weighted_score: 1.0,
            confidence_level: 0.95,
            reproducibility_index: 1.0,
            black_swan_count: 0,
            falsifications: vec![],
        };
        let json = serde_json::to_string(&score).expect("serialize");
        assert!(json.contains("test"));
    }

    #[test]
    fn test_popperian_with_timeout() {
        let calculator = PopperianCalculator::new();
        let mut collector = EvidenceCollector::new();

        collector.add(Evidence::timeout("F-PERF-001", test_scenario(), 30000));

        let score = calculator.calculate("test/model", &collector);
        // Timeout is treated as inconclusive, not falsified
        assert_eq!(score.inconclusive, 1);
        assert_eq!(score.falsified, 0);
    }
}
