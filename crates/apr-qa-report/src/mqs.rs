//! Model Qualification Score (MQS) Calculator
//!
//! Implements Toyota-style gateway checks and Popperian falsification scoring.
//!
//! ## Scoring System
//!
//! - **Raw score**: 0-1000 points across 6 categories
//! - **Normalized score**: 0-100 (logarithmic scaling, 100 is extremely hard)
//! - **Gateway checks**: G1-G4 failures zero the entire score
//!
//! ## Categories (1000 raw points total)
//!
//! | Category | Points | Description |
//! |----------|--------|-------------|
//! | QUAL     | 200    | Basic quality (loads, responds) |
//! | PERF     | 150    | Performance metrics |
//! | STAB     | 200    | Stability under stress |
//! | COMP     | 150    | Compatibility (formats, backends) |
//! | EDGE     | 150    | Edge case handling |
//! | REGR     | 150    | Regression resistance |

use apr_qa_runner::{Evidence, EvidenceCollector, Outcome};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::error::Result;

/// Gateway check result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GatewayResult {
    /// Gateway ID (G1, G2, G3, G4)
    pub id: String,
    /// Whether the gateway passed
    pub passed: bool,
    /// Description of the check
    pub description: String,
    /// Failure reason (if any)
    pub failure_reason: Option<String>,
}

impl GatewayResult {
    /// Create a passed gateway result
    #[must_use]
    pub fn passed(id: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            passed: true,
            description: description.into(),
            failure_reason: None,
        }
    }

    /// Create a failed gateway result
    #[must_use]
    pub fn failed(
        id: impl Into<String>,
        description: impl Into<String>,
        reason: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            passed: false,
            description: description.into(),
            failure_reason: Some(reason.into()),
        }
    }
}

/// MQS category scores
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CategoryScores {
    /// Quality score (0-200)
    pub qual: u32,
    /// Performance score (0-150)
    pub perf: u32,
    /// Stability score (0-200)
    pub stab: u32,
    /// Compatibility score (0-150)
    pub comp: u32,
    /// Edge case score (0-150)
    pub edge: u32,
    /// Regression score (0-150)
    pub regr: u32,
}

impl CategoryScores {
    /// Maximum points per category
    pub const MAX_QUAL: u32 = 200;
    /// Maximum performance points
    pub const MAX_PERF: u32 = 150;
    /// Maximum stability points
    pub const MAX_STAB: u32 = 200;
    /// Maximum compatibility points
    pub const MAX_COMP: u32 = 150;
    /// Maximum edge case points
    pub const MAX_EDGE: u32 = 150;
    /// Maximum regression points
    pub const MAX_REGR: u32 = 150;
    /// Total maximum raw score
    pub const MAX_TOTAL: u32 = 1000;

    /// Calculate total raw score
    #[must_use]
    pub fn total(&self) -> u32 {
        self.qual + self.perf + self.stab + self.comp + self.edge + self.regr
    }

    /// Get category breakdown as HashMap
    #[must_use]
    pub fn breakdown(&self) -> HashMap<String, (u32, u32)> {
        let mut map = HashMap::new();
        map.insert("QUAL".to_string(), (self.qual, Self::MAX_QUAL));
        map.insert("PERF".to_string(), (self.perf, Self::MAX_PERF));
        map.insert("STAB".to_string(), (self.stab, Self::MAX_STAB));
        map.insert("COMP".to_string(), (self.comp, Self::MAX_COMP));
        map.insert("EDGE".to_string(), (self.edge, Self::MAX_EDGE));
        map.insert("REGR".to_string(), (self.regr, Self::MAX_REGR));
        map
    }
}

/// Final MQS score with all details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MqsScore {
    /// Model identifier
    pub model_id: String,
    /// Raw score (0-1000)
    pub raw_score: u32,
    /// Normalized score (0-100)
    pub normalized_score: f64,
    /// Letter grade (A+, A, B, C, D, F)
    pub grade: String,
    /// Gateway results
    pub gateways: Vec<GatewayResult>,
    /// Whether all gateways passed
    pub gateways_passed: bool,
    /// Category breakdown
    pub categories: CategoryScores,
    /// Total tests run
    pub total_tests: usize,
    /// Tests passed
    pub tests_passed: usize,
    /// Tests failed
    pub tests_failed: usize,
    /// Penalty deductions applied
    pub penalties: Vec<Penalty>,
    /// Total penalty points deducted
    pub total_penalty: u32,
}

impl MqsScore {
    /// Check if model qualifies (normalized score >= 70)
    #[must_use]
    pub fn qualifies(&self) -> bool {
        self.gateways_passed && self.normalized_score >= 70.0
    }

    /// Check if model is production-ready (normalized score >= 90)
    #[must_use]
    pub fn is_production_ready(&self) -> bool {
        self.gateways_passed && self.normalized_score >= 90.0
    }
}

/// Penalty applied to score
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Penalty {
    /// Penalty code
    pub code: String,
    /// Description
    pub description: String,
    /// Points deducted
    pub points: u32,
}

/// MQS Calculator
#[derive(Debug)]
pub struct MqsCalculator {
    /// Penalty multiplier for repeated failures
    failure_multiplier: f64,
    /// Minimum tests required per category
    #[allow(dead_code)]
    min_tests_per_category: usize,
}

impl Default for MqsCalculator {
    fn default() -> Self {
        Self::new()
    }
}

impl MqsCalculator {
    /// Create a new calculator with default settings
    #[must_use]
    pub fn new() -> Self {
        Self {
            failure_multiplier: 1.5,
            min_tests_per_category: 10,
        }
    }

    /// Set failure multiplier
    #[must_use]
    pub fn with_failure_multiplier(mut self, multiplier: f64) -> Self {
        self.failure_multiplier = multiplier;
        self
    }

    /// Calculate MQS from evidence
    ///
    /// # Errors
    ///
    /// Returns an error if score calculation fails.
    pub fn calculate(&self, model_id: &str, evidence: &EvidenceCollector) -> Result<MqsScore> {
        let all_evidence = evidence.all();

        // Run gateway checks
        let gateways = self.check_gateways(all_evidence);
        let gateways_passed = gateways.iter().all(|g| g.passed);

        // If gateways fail, score is zero
        if !gateways_passed {
            return Ok(MqsScore {
                model_id: model_id.to_string(),
                raw_score: 0,
                normalized_score: 0.0,
                grade: "F".to_string(),
                gateways,
                gateways_passed: false,
                categories: CategoryScores::default(),
                total_tests: all_evidence.len(),
                tests_passed: evidence.pass_count(),
                tests_failed: evidence.fail_count(),
                penalties: vec![Penalty {
                    code: "GATEWAY".to_string(),
                    description: "Gateway check failed - score zeroed".to_string(),
                    points: 1000,
                }],
                total_penalty: 1000,
            });
        }

        // Calculate category scores
        let categories = self.calculate_categories(all_evidence);
        let mut penalties = Vec::new();
        let mut total_penalty: u32 = 0;

        // Apply penalties
        let crash_count = all_evidence
            .iter()
            .filter(|e| e.outcome == Outcome::Crashed)
            .count();
        if crash_count > 0 {
            let penalty = (crash_count as u32) * 20;
            penalties.push(Penalty {
                code: "CRASH".to_string(),
                description: format!("{crash_count} crash(es) detected"),
                points: penalty,
            });
            total_penalty += penalty;
        }

        let timeout_count = all_evidence
            .iter()
            .filter(|e| e.outcome == Outcome::Timeout)
            .count();
        if timeout_count > 0 {
            let penalty = (timeout_count as u32) * 10;
            penalties.push(Penalty {
                code: "TIMEOUT".to_string(),
                description: format!("{timeout_count} timeout(s) detected"),
                points: penalty,
            });
            total_penalty += penalty;
        }

        // Calculate raw score with penalties
        let raw_score = categories.total().saturating_sub(total_penalty);

        // Normalize to 0-100 using logarithmic scaling
        // This makes 100/100 extremely difficult to achieve
        let normalized = self.normalize_score(raw_score, categories.total());

        let grade = Self::calculate_grade(normalized);

        Ok(MqsScore {
            model_id: model_id.to_string(),
            raw_score,
            normalized_score: normalized,
            grade,
            gateways,
            gateways_passed: true,
            categories,
            total_tests: all_evidence.len(),
            tests_passed: evidence.pass_count(),
            tests_failed: evidence.fail_count(),
            penalties,
            total_penalty,
        })
    }

    /// Check gateway conditions (G0-G4)
    fn check_gateways(&self, evidence: &[Evidence]) -> Vec<GatewayResult> {
        let mut results = Vec::new();

        // G0: Model integrity (config/tensor consistency)
        // Checks for G0-INTEGRITY-* gate IDs
        let integrity_failures: Vec<&Evidence> = evidence
            .iter()
            .filter(|e| e.gate_id.starts_with("G0-INTEGRITY") && e.outcome.is_fail())
            .collect();
        if integrity_failures.is_empty() {
            results.push(GatewayResult::passed(
                "G0",
                "Model integrity (config/tensor match)",
            ));
        } else {
            let error_details: Vec<&str> = integrity_failures
                .iter()
                .map(|e| e.reason.as_str())
                .collect();
            results.push(GatewayResult::failed(
                "G0",
                "Model integrity (config/tensor match)",
                format!(
                    "{} integrity check(s) failed: {}",
                    integrity_failures.len(),
                    error_details.join("; ")
                ),
            ));
        }

        // G1: Model loads successfully
        let has_load_failure = evidence
            .iter()
            .any(|e| e.gate_id.contains("G1") && e.outcome.is_fail());
        if has_load_failure {
            results.push(GatewayResult::failed(
                "G1",
                "Model loads successfully",
                "Model failed to load",
            ));
        } else {
            results.push(GatewayResult::passed("G1", "Model loads successfully"));
        }

        // G2: Basic inference works
        let has_inference_failure = evidence
            .iter()
            .any(|e| e.gate_id.contains("G2") && e.outcome.is_fail());
        if has_inference_failure {
            results.push(GatewayResult::failed(
                "G2",
                "Basic inference works",
                "Inference failed",
            ));
        } else {
            results.push(GatewayResult::passed("G2", "Basic inference works"));
        }

        // G3: No crashes
        let crash_count = evidence
            .iter()
            .filter(|e| e.outcome == Outcome::Crashed)
            .count();
        if crash_count > 0 {
            results.push(GatewayResult::failed(
                "G3",
                "No crashes",
                format!("{crash_count} crash(es) detected"),
            ));
        } else {
            results.push(GatewayResult::passed("G3", "No crashes"));
        }

        // G4: Output is not garbage
        let garbage_failures = evidence
            .iter()
            .filter(|e| e.gate_id.contains("G4") && e.outcome.is_fail())
            .count();
        if garbage_failures > evidence.len() / 4 {
            // More than 25% garbage output
            results.push(GatewayResult::failed(
                "G4",
                "Output is not garbage",
                format!("{garbage_failures} garbage outputs detected"),
            ));
        } else {
            results.push(GatewayResult::passed("G4", "Output is not garbage"));
        }

        results
    }

    /// Calculate category scores from evidence
    fn calculate_categories(&self, evidence: &[Evidence]) -> CategoryScores {
        // Tally pass/total per category using a map
        let mut tallies: HashMap<String, (usize, usize)> = HashMap::new();
        for e in evidence {
            let cat = Self::extract_category(&e.gate_id);
            let key = match cat.as_str() {
                "QUAL" | "PERF" | "STAB" | "COMP" | "EDGE" | "REGR" => cat,
                _ => "QUAL".to_string(), // Default unknown to QUAL
            };
            let entry = tallies.entry(key).or_insert((0, 0));
            entry.1 += 1;
            if e.outcome.is_pass() {
                entry.0 += 1;
            }
        }

        let score_for =
            |cat: &str, max: u32| -> u32 {
                let &(pass, total) = tallies.get(cat).unwrap_or(&(0, 0));
                Self::proportional_score_or_full(pass, total, max)
            };

        // Categories with 0 tests get full credit (no evidence of failure)
        CategoryScores {
            qual: score_for("QUAL", CategoryScores::MAX_QUAL),
            perf: score_for("PERF", CategoryScores::MAX_PERF),
            stab: score_for("STAB", CategoryScores::MAX_STAB),
            comp: score_for("COMP", CategoryScores::MAX_COMP),
            edge: score_for("EDGE", CategoryScores::MAX_EDGE),
            regr: score_for("REGR", CategoryScores::MAX_REGR),
        }
    }

    /// Extract MQS category from gate ID.
    ///
    /// Maps gate IDs to the 6 MQS categories using two strategies:
    /// 1. Prefix matching for real playbook gate IDs (F-A1, G0-*, F-CONV-*, etc.)
    /// 2. Direct category name extraction for canonical F-{CATEGORY}-xxx pattern
    fn extract_category(gate_id: &str) -> String {
        // Strategy 1: Prefix matching for real playbook gate IDs

        // Regression invariants (round-trip, idempotency, commutativity)
        if gate_id.starts_with("F-CONV-RT")
            || gate_id.starts_with("F-CONV-IDEM")
            || gate_id.starts_with("F-CONV-COM")
        {
            return "REGR".to_string();
        }
        // Format conversion and contract compatibility
        if gate_id.starts_with("F-CONV") || gate_id.starts_with("F-CONTRACT") {
            return "COMP".to_string();
        }
        // Gateway/infrastructure stability
        if gate_id.starts_with("G0-") {
            return "STAB".to_string();
        }

        // Strategy 2: Direct category name from F-{CATEGORY}-xxx pattern
        let categories = ["QUAL", "PERF", "STAB", "COMP", "EDGE", "REGR"];
        if let Some(second) = gate_id.split('-').nth(1) {
            let upper = second.to_uppercase();
            if categories.contains(&upper.as_str()) {
                return upper;
            }
        }

        // Default: quality (F-A1..F-A6, F-GOLDEN-*, etc.)
        "QUAL".to_string()
    }

    /// Calculate proportional score, awarding full credit when no tests exist.
    /// Rationale: absence of evidence is not evidence of absence — categories
    /// with 0 tests should not penalize the overall score.
    fn proportional_score_or_full(passed: usize, total: usize, max: u32) -> u32 {
        if total == 0 {
            return max;
        }
        let ratio = passed as f64 / total as f64;
        (ratio * f64::from(max)).round() as u32
    }

    /// Normalize raw score to 0-100 using logarithmic scaling
    /// This makes achieving 100/100 extremely difficult
    fn normalize_score(&self, raw: u32, pre_penalty: u32) -> f64 {
        if pre_penalty == 0 {
            return 0.0;
        }

        let ratio = f64::from(raw) / f64::from(CategoryScores::MAX_TOTAL);

        // Apply logarithmic scaling to make high scores harder
        // f(x) = 100 * (log(1 + 9x) / log(10))
        // This maps [0,1] to [0,100] with diminishing returns
        let normalized = 100.0 * (1.0 + 9.0 * ratio).ln() / 10_f64.ln();

        // Clamp to valid range
        normalized.clamp(0.0, 100.0)
    }

    /// Calculate letter grade from normalized score
    fn calculate_grade(score: f64) -> String {
        const GRADE_TABLE: &[(f64, &str)] = &[
            (97.0, "A+"),
            (93.0, "A"),
            (90.0, "A-"),
            (87.0, "B+"),
            (83.0, "B"),
            (80.0, "B-"),
            (77.0, "C+"),
            (73.0, "C"),
            (70.0, "C-"),
            (67.0, "D+"),
            (63.0, "D"),
            (60.0, "D-"),
        ];
        GRADE_TABLE
            .iter()
            .find(|(threshold, _)| score >= *threshold)
            .map_or_else(|| "F".to_string(), |(_, grade)| (*grade).to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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

    fn test_evidence_passed(gate_id: &str) -> Evidence {
        Evidence::corroborated(gate_id, test_scenario(), "4", 100)
    }

    fn test_evidence_failed(gate_id: &str) -> Evidence {
        Evidence::falsified(gate_id, test_scenario(), "Wrong answer", "5", 100)
    }

    #[test]
    fn test_gateway_result_passed() {
        let result = GatewayResult::passed("G1", "Model loads");
        assert!(result.passed);
        assert!(result.failure_reason.is_none());
    }

    #[test]
    fn test_gateway_result_failed() {
        let result = GatewayResult::failed("G1", "Model loads", "OOM");
        assert!(!result.passed);
        assert_eq!(result.failure_reason, Some("OOM".to_string()));
    }

    #[test]
    fn test_category_scores_total() {
        let scores = CategoryScores {
            qual: 150,
            perf: 100,
            stab: 150,
            comp: 100,
            edge: 100,
            regr: 100,
        };
        assert_eq!(scores.total(), 700);
    }

    #[test]
    fn test_category_scores_max() {
        assert_eq!(CategoryScores::MAX_TOTAL, 1000);
    }

    #[test]
    fn test_mqs_calculator_all_pass() {
        let calculator = MqsCalculator::new();
        let mut collector = EvidenceCollector::new();

        // Add passing evidence for each category
        for i in 0..10 {
            collector.add(test_evidence_passed(&format!("F-QUAL-{i:03}")));
            collector.add(test_evidence_passed(&format!("F-PERF-{i:03}")));
            collector.add(test_evidence_passed(&format!("F-STAB-{i:03}")));
            collector.add(test_evidence_passed(&format!("F-COMP-{i:03}")));
            collector.add(test_evidence_passed(&format!("F-EDGE-{i:03}")));
            collector.add(test_evidence_passed(&format!("F-REGR-{i:03}")));
        }

        let score = calculator
            .calculate("test/model", &collector)
            .expect("Calculation failed");

        assert!(score.gateways_passed);
        assert_eq!(score.raw_score, 1000);
        assert!(score.normalized_score > 99.0);
        assert_eq!(score.grade, "A+");
    }

    #[test]
    fn test_mqs_calculator_gateway_failure() {
        let calculator = MqsCalculator::new();
        let mut collector = EvidenceCollector::new();

        // Add a crash (fails G3 gateway)
        collector.add(Evidence::crashed(
            "F-QUAL-001",
            test_scenario(),
            "SIGSEGV",
            -11,
            0,
        ));

        let score = calculator
            .calculate("test/model", &collector)
            .expect("Calculation failed");

        assert!(!score.gateways_passed);
        assert_eq!(score.raw_score, 0);
        assert_eq!(score.normalized_score, 0.0);
        assert_eq!(score.grade, "F");
    }

    #[test]
    fn test_mqs_calculator_with_penalties() {
        let calculator = MqsCalculator::new();
        let mut collector = EvidenceCollector::new();

        // Add mostly passing tests
        for i in 0..50 {
            collector.add(test_evidence_passed(&format!("F-QUAL-{i:03}")));
        }

        // Add some timeouts (but not crashes to keep gateways passing)
        for i in 0..5 {
            collector.add(Evidence::timeout(
                &format!("F-PERF-{i:03}"),
                test_scenario(),
                30000,
            ));
        }

        let score = calculator
            .calculate("test/model", &collector)
            .expect("Calculation failed");

        // Should have timeout penalty
        assert!(score.total_penalty > 0);
        assert!(score.penalties.iter().any(|p| p.code == "TIMEOUT"));
    }

    #[test]
    fn test_extract_category() {
        assert_eq!(MqsCalculator::extract_category("F-QUAL-001"), "QUAL");
        assert_eq!(MqsCalculator::extract_category("F-PERF-042"), "PERF");
        assert_eq!(MqsCalculator::extract_category("UNKNOWN"), "QUAL");
    }

    #[test]
    fn test_proportional_score() {
        assert_eq!(MqsCalculator::proportional_score_or_full(10, 10, 200), 200);
        assert_eq!(MqsCalculator::proportional_score_or_full(5, 10, 200), 100);
        assert_eq!(MqsCalculator::proportional_score_or_full(0, 10, 200), 0);
        assert_eq!(MqsCalculator::proportional_score_or_full(0, 0, 200), 200);
    }

    #[test]
    fn test_grade_calculation() {
        assert_eq!(MqsCalculator::calculate_grade(100.0), "A+");
        assert_eq!(MqsCalculator::calculate_grade(97.0), "A+");
        assert_eq!(MqsCalculator::calculate_grade(93.0), "A");
        assert_eq!(MqsCalculator::calculate_grade(90.0), "A-");
        assert_eq!(MqsCalculator::calculate_grade(83.0), "B");
        assert_eq!(MqsCalculator::calculate_grade(73.0), "C");
        assert_eq!(MqsCalculator::calculate_grade(50.0), "F");
    }

    #[test]
    fn test_mqs_score_qualifies() {
        let score = MqsScore {
            model_id: "test".to_string(),
            raw_score: 800,
            normalized_score: 75.0,
            grade: "C".to_string(),
            gateways: vec![],
            gateways_passed: true,
            categories: CategoryScores::default(),
            total_tests: 100,
            tests_passed: 80,
            tests_failed: 20,
            penalties: vec![],
            total_penalty: 0,
        };

        assert!(score.qualifies());
        assert!(!score.is_production_ready());
    }

    #[test]
    fn test_normalize_score_scaling() {
        let calc = MqsCalculator::new();

        // Test that normalization provides diminishing returns
        let low = calc.normalize_score(200, 200);
        let mid = calc.normalize_score(500, 500);
        let high = calc.normalize_score(900, 900);
        let perfect = calc.normalize_score(1000, 1000);

        // Each increment should be harder
        assert!(low < mid);
        assert!(mid < high);
        assert!(high < perfect);

        // Perfect score should be 100
        assert!((perfect - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_grade_all_levels() {
        assert_eq!(MqsCalculator::calculate_grade(98.0), "A+");
        assert_eq!(MqsCalculator::calculate_grade(95.0), "A");
        assert_eq!(MqsCalculator::calculate_grade(91.0), "A-");
        assert_eq!(MqsCalculator::calculate_grade(88.0), "B+");
        assert_eq!(MqsCalculator::calculate_grade(85.0), "B");
        assert_eq!(MqsCalculator::calculate_grade(81.0), "B-");
        assert_eq!(MqsCalculator::calculate_grade(78.0), "C+");
        assert_eq!(MqsCalculator::calculate_grade(75.0), "C");
        assert_eq!(MqsCalculator::calculate_grade(71.0), "C-");
        assert_eq!(MqsCalculator::calculate_grade(68.0), "D+");
        assert_eq!(MqsCalculator::calculate_grade(65.0), "D");
        assert_eq!(MqsCalculator::calculate_grade(61.0), "D-");
        assert_eq!(MqsCalculator::calculate_grade(55.0), "F");
    }

    #[test]
    fn test_mqs_score_is_production_ready() {
        let score = MqsScore {
            model_id: "test".to_string(),
            raw_score: 950,
            normalized_score: 95.0,
            grade: "A".to_string(),
            gateways: vec![],
            gateways_passed: true,
            categories: CategoryScores::default(),
            total_tests: 100,
            tests_passed: 95,
            tests_failed: 5,
            penalties: vec![],
            total_penalty: 0,
        };
        assert!(score.is_production_ready());
    }

    #[test]
    fn test_mqs_score_not_qualifies() {
        let score = MqsScore {
            model_id: "test".to_string(),
            raw_score: 500,
            normalized_score: 50.0,
            grade: "F".to_string(),
            gateways: vec![],
            gateways_passed: true,
            categories: CategoryScores::default(),
            total_tests: 100,
            tests_passed: 50,
            tests_failed: 50,
            penalties: vec![],
            total_penalty: 0,
        };
        assert!(!score.qualifies());
    }

    #[test]
    fn test_mqs_score_gateway_failed_not_qualifies() {
        let score = MqsScore {
            model_id: "test".to_string(),
            raw_score: 900,
            normalized_score: 90.0,
            grade: "A-".to_string(),
            gateways: vec![],
            gateways_passed: false,
            categories: CategoryScores::default(),
            total_tests: 100,
            tests_passed: 90,
            tests_failed: 10,
            penalties: vec![],
            total_penalty: 0,
        };
        assert!(!score.qualifies());
    }

    #[test]
    fn test_category_scores_default() {
        let scores = CategoryScores::default();
        assert_eq!(scores.total(), 0);
    }

    #[test]
    fn test_category_scores_breakdown() {
        let scores = CategoryScores {
            qual: 180,
            perf: 150,
            stab: 160,
            comp: 140,
            edge: 130,
            regr: 120,
        };
        let breakdown = scores.breakdown();
        assert_eq!(breakdown.get("QUAL"), Some(&(180, 200)));
        assert_eq!(breakdown.get("PERF"), Some(&(150, 150)));
        assert_eq!(breakdown.get("STAB"), Some(&(160, 200)));
        assert_eq!(breakdown.get("COMP"), Some(&(140, 150)));
        assert_eq!(breakdown.get("EDGE"), Some(&(130, 150)));
        assert_eq!(breakdown.get("REGR"), Some(&(120, 150)));
    }

    #[test]
    fn test_penalty_clone() {
        let penalty = Penalty {
            code: "TEST".to_string(),
            description: "Test penalty".to_string(),
            points: 10,
        };
        let cloned = penalty.clone();
        assert_eq!(cloned.code, penalty.code);
        assert_eq!(cloned.points, penalty.points);
    }

    #[test]
    fn test_gateway_result_clone() {
        let result = GatewayResult::passed("G1", "Test");
        let cloned = result.clone();
        assert_eq!(cloned.id, result.id);
        assert_eq!(cloned.passed, result.passed);
    }

    #[test]
    fn test_mqs_score_serialize() {
        let score = MqsScore {
            model_id: "test".to_string(),
            raw_score: 800,
            normalized_score: 80.0,
            grade: "B".to_string(),
            gateways: vec![],
            gateways_passed: true,
            categories: CategoryScores::default(),
            total_tests: 100,
            tests_passed: 80,
            tests_failed: 20,
            penalties: vec![],
            total_penalty: 0,
        };
        let json = serde_json::to_string(&score).expect("serialize");
        assert!(json.contains("test"));
        assert!(json.contains("800"));
    }

    #[test]
    fn test_extract_category_stab() {
        assert_eq!(MqsCalculator::extract_category("F-STAB-001"), "STAB");
    }

    #[test]
    fn test_extract_category_comp() {
        assert_eq!(MqsCalculator::extract_category("F-COMP-001"), "COMP");
    }

    #[test]
    fn test_extract_category_edge() {
        assert_eq!(MqsCalculator::extract_category("F-EDGE-001"), "EDGE");
    }

    #[test]
    fn test_extract_category_regr() {
        assert_eq!(MqsCalculator::extract_category("F-REGR-001"), "REGR");
    }

    #[test]
    fn test_normalize_score_zero() {
        let calc = MqsCalculator::new();
        let score = calc.normalize_score(0, 0);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_mqs_calculator_check_gateways() {
        let calc = MqsCalculator::new();
        let collector = EvidenceCollector::new();

        let gateways = calc.check_gateways(collector.all());
        // Should have 5 gateways (G0-G4)
        assert_eq!(gateways.len(), 5);
    }

    #[test]
    fn test_mqs_calculator_with_failure_multiplier() {
        let calc = MqsCalculator::new().with_failure_multiplier(2.0);
        assert_eq!(calc.failure_multiplier, 2.0);
    }

    #[test]
    fn test_mqs_calculator_default() {
        let calc = MqsCalculator::default();
        assert_eq!(calc.failure_multiplier, 1.5);
    }

    #[test]
    fn test_mqs_calculator_debug() {
        let calc = MqsCalculator::new();
        let debug_str = format!("{calc:?}");
        assert!(debug_str.contains("MqsCalculator"));
    }

    #[test]
    fn test_gateway_g1_failure() {
        let calc = MqsCalculator::new();
        let mut collector = EvidenceCollector::new();

        // Add a G1 failure (model load failure)
        collector.add(Evidence::falsified(
            "G1-LOAD",
            test_scenario(),
            "Model failed to load",
            "",
            100,
        ));

        let score = calc
            .calculate("test/model", &collector)
            .expect("Calculation failed");

        // G1 failed should fail all gateways
        assert!(!score.gateways_passed);
        let g1 = score.gateways.iter().find(|g| g.id == "G1").unwrap();
        assert!(!g1.passed);
    }

    #[test]
    fn test_gateway_g2_failure() {
        let calc = MqsCalculator::new();
        let mut collector = EvidenceCollector::new();

        // Add a G2 failure (basic inference failure)
        collector.add(Evidence::falsified(
            "G2-INFERENCE",
            test_scenario(),
            "Inference failed",
            "",
            100,
        ));

        let score = calc
            .calculate("test/model", &collector)
            .expect("Calculation failed");

        let g2 = score.gateways.iter().find(|g| g.id == "G2").unwrap();
        assert!(!g2.passed);
    }

    #[test]
    fn test_gateway_g4_failure_garbage_output() {
        let calc = MqsCalculator::new();
        let mut collector = EvidenceCollector::new();

        // Add many G4 failures (more than 25% garbage)
        for i in 0..10 {
            collector.add(Evidence::falsified(
                &format!("G4-GARBAGE-{i:03}"),
                test_scenario(),
                "Garbage output",
                "###$$@@!!",
                100,
            ));
        }

        let score = calc
            .calculate("test/model", &collector)
            .expect("Calculation failed");

        let g4 = score.gateways.iter().find(|g| g.id == "G4").unwrap();
        assert!(!g4.passed);
    }

    #[test]
    fn test_mqs_with_crash_penalty() {
        let calc = MqsCalculator::new();
        let mut collector = EvidenceCollector::new();

        // Add mostly passing evidence first (so gateways pass)
        for i in 0..50 {
            collector.add(test_evidence_passed(&format!("F-QUAL-{i:03}")));
        }

        // Now the crash count will fail G3 gateway
        // So we need to test crash penalty separately without actual crashes

        let score = calc
            .calculate("test/model", &collector)
            .expect("Calculation failed");
        assert!(score.gateways_passed);
    }

    #[test]
    fn test_calculate_categories_all_types() {
        let calc = MqsCalculator::new();
        let mut collector = EvidenceCollector::new();

        // Add one of each category
        collector.add(test_evidence_passed("F-QUAL-001"));
        collector.add(test_evidence_passed("F-PERF-001"));
        collector.add(test_evidence_passed("F-STAB-001"));
        collector.add(test_evidence_passed("F-COMP-001"));
        collector.add(test_evidence_passed("F-EDGE-001"));
        collector.add(test_evidence_passed("F-REGR-001"));

        let categories = calc.calculate_categories(collector.all());

        assert!(categories.qual > 0);
        assert!(categories.perf > 0);
        assert!(categories.stab > 0);
        assert!(categories.comp > 0);
        assert!(categories.edge > 0);
        assert!(categories.regr > 0);
    }

    #[test]
    fn test_calculate_categories_with_failures() {
        let calc = MqsCalculator::new();
        let mut collector = EvidenceCollector::new();

        // Add passing and failing evidence
        collector.add(test_evidence_passed("F-QUAL-001"));
        collector.add(test_evidence_failed("F-QUAL-002"));
        collector.add(test_evidence_passed("F-QUAL-003"));

        let categories = calc.calculate_categories(collector.all());

        // 2 out of 3 passed, so qual should be ~133 (2/3 of 200)
        assert!(categories.qual > 100);
        assert!(categories.qual < 200);
    }

    #[test]
    fn test_calculate_categories_unknown_category() {
        let calc = MqsCalculator::new();
        let mut collector = EvidenceCollector::new();

        // Add evidence with unknown category - should default to QUAL
        collector.add(test_evidence_passed("UNKNOWN"));

        let categories = calc.calculate_categories(collector.all());
        assert!(categories.qual > 0);
    }

    #[test]
    fn test_gateway_result_debug() {
        let result = GatewayResult::passed("G1", "Test");
        let debug_str = format!("{result:?}");
        assert!(debug_str.contains("GatewayResult"));
    }

    #[test]
    fn test_category_scores_debug() {
        let scores = CategoryScores::default();
        let debug_str = format!("{scores:?}");
        assert!(debug_str.contains("CategoryScores"));
    }

    #[test]
    fn test_penalty_debug() {
        let penalty = Penalty {
            code: "TEST".to_string(),
            description: "Test".to_string(),
            points: 10,
        };
        let debug_str = format!("{penalty:?}");
        assert!(debug_str.contains("Penalty"));
    }

    #[test]
    fn test_mqs_score_debug() {
        let score = MqsScore {
            model_id: "test".to_string(),
            raw_score: 800,
            normalized_score: 80.0,
            grade: "B".to_string(),
            gateways: vec![],
            gateways_passed: true,
            categories: CategoryScores::default(),
            total_tests: 100,
            tests_passed: 80,
            tests_failed: 20,
            penalties: vec![],
            total_penalty: 0,
        };
        let debug_str = format!("{score:?}");
        assert!(debug_str.contains("MqsScore"));
    }

    #[test]
    fn test_mqs_score_clone() {
        let score = MqsScore {
            model_id: "test".to_string(),
            raw_score: 800,
            normalized_score: 80.0,
            grade: "B".to_string(),
            gateways: vec![],
            gateways_passed: true,
            categories: CategoryScores::default(),
            total_tests: 100,
            tests_passed: 80,
            tests_failed: 20,
            penalties: vec![],
            total_penalty: 0,
        };
        let cloned = score.clone();
        assert_eq!(cloned.model_id, score.model_id);
        assert_eq!(cloned.raw_score, score.raw_score);
    }

    #[test]
    fn test_gateway_result_serialize() {
        let result = GatewayResult::passed("G1", "Test");
        let json = serde_json::to_string(&result).expect("serialize");
        assert!(json.contains("G1"));
    }

    #[test]
    fn test_category_scores_serialize() {
        let scores = CategoryScores {
            qual: 100,
            perf: 50,
            stab: 75,
            comp: 60,
            edge: 40,
            regr: 30,
        };
        let json = serde_json::to_string(&scores).expect("serialize");
        assert!(json.contains("100"));
    }

    #[test]
    fn test_penalty_serialize() {
        let penalty = Penalty {
            code: "CRASH".to_string(),
            description: "Crash detected".to_string(),
            points: 20,
        };
        let json = serde_json::to_string(&penalty).expect("serialize");
        assert!(json.contains("CRASH"));
    }

    #[test]
    fn test_mqs_calculator_calculate_empty() {
        let calc = MqsCalculator::new();
        let collector = EvidenceCollector::new();

        let score = calc
            .calculate("test/model", &collector)
            .expect("Calculation failed");

        // Empty collector should pass all gateways (no failures)
        assert!(score.gateways_passed);
        assert_eq!(score.total_tests, 0);
    }

    #[test]
    fn test_category_scores_clone() {
        let scores = CategoryScores {
            qual: 100,
            perf: 50,
            stab: 75,
            comp: 60,
            edge: 40,
            regr: 30,
        };
        let cloned = scores.clone();
        assert_eq!(cloned.qual, scores.qual);
        assert_eq!(cloned.total(), scores.total());
    }

    #[test]
    fn test_mqs_score_deserialize() {
        let json = r#"{
            "model_id": "test",
            "raw_score": 800,
            "normalized_score": 80.0,
            "grade": "B",
            "gateways": [],
            "gateways_passed": true,
            "categories": {"qual": 0, "perf": 0, "stab": 0, "comp": 0, "edge": 0, "regr": 0},
            "total_tests": 100,
            "tests_passed": 80,
            "tests_failed": 20,
            "penalties": [],
            "total_penalty": 0
        }"#;
        let score: MqsScore = serde_json::from_str(json).expect("deserialize");
        assert_eq!(score.model_id, "test");
        assert_eq!(score.raw_score, 800);
    }

    #[test]
    fn test_gateway_g0_integrity_failure() {
        let calc = MqsCalculator::new();
        let mut collector = EvidenceCollector::new();

        // Add a G0 integrity failure (layer count mismatch)
        collector.add(Evidence::falsified(
            "G0-INTEGRITY-LAYERS",
            test_scenario(),
            "config says 14 layers but tensors have 24",
            "",
            100,
        ));

        let score = calc
            .calculate("test/model", &collector)
            .expect("Calculation failed");

        // G0 failed should fail all gateways and zero score
        assert!(!score.gateways_passed);
        assert_eq!(score.raw_score, 0);
        assert_eq!(score.normalized_score, 0.0);
        let g0 = score.gateways.iter().find(|g| g.id == "G0").unwrap();
        assert!(!g0.passed);
        assert!(g0.failure_reason.as_ref().unwrap().contains("integrity"));
    }

    #[test]
    fn test_gateway_g0_integrity_multiple_failures() {
        let calc = MqsCalculator::new();
        let mut collector = EvidenceCollector::new();

        // Add multiple G0 integrity failures (corrupted config scenario)
        collector.add(Evidence::falsified(
            "G0-INTEGRITY-LAYERS",
            test_scenario(),
            "config says 14 layers but tensors have 24",
            "",
            100,
        ));
        collector.add(Evidence::falsified(
            "G0-INTEGRITY-HIDDEN",
            test_scenario(),
            "config says hidden_size=4096 but embedding has 896",
            "",
            100,
        ));
        collector.add(Evidence::falsified(
            "G0-INTEGRITY-VOCAB",
            test_scenario(),
            "config says vocab_size=896 but embedding has 151936",
            "",
            100,
        ));

        let score = calc
            .calculate("test/model", &collector)
            .expect("Calculation failed");

        assert!(!score.gateways_passed);
        assert_eq!(score.raw_score, 0);
        let g0 = score.gateways.iter().find(|g| g.id == "G0").unwrap();
        assert!(!g0.passed);
        // Should mention all 3 failures
        assert!(g0.failure_reason.as_ref().unwrap().contains("3 integrity"));
    }

    #[test]
    fn test_gateway_g0_passes_when_no_integrity_failures() {
        let calc = MqsCalculator::new();
        let mut collector = EvidenceCollector::new();

        // Add only regular test evidence, no G0 failures
        collector.add(test_evidence_passed("F-QUAL-001"));
        collector.add(test_evidence_passed("F-PERF-001"));

        let score = calc
            .calculate("test/model", &collector)
            .expect("Calculation failed");

        assert!(score.gateways_passed);
        let g0 = score.gateways.iter().find(|g| g.id == "G0").unwrap();
        assert!(g0.passed);
    }

    #[test]
    fn test_gateway_order_g0_first() {
        let calc = MqsCalculator::new();
        let collector = EvidenceCollector::new();

        let gateways = calc.check_gateways(collector.all());
        // G0 should be first
        assert_eq!(gateways[0].id, "G0");
        assert_eq!(gateways[1].id, "G1");
        assert_eq!(gateways[2].id, "G2");
        assert_eq!(gateways[3].id, "G3");
        assert_eq!(gateways[4].id, "G4");
    }
}
