//! Evidence Export for Oracle Integration (PMAT-261)
//!
//! This module provides the structured evidence export format consumed by
//! aprender's `apr oracle` CLI for certification status lookup.
//!
//! # Theoretical Foundation
//!
//! - **Reproducibility (Hamming, 1962)**: Same evidence → same MQS score
//! - **Contract Programming (Meyer, 1992)**: Schema defines oracle expectations
//! - **Defensive Programming (Hunt & Thomas, 1999)**: Handle missing/malformed data
//!
//! # JSON Schema
//!
//! The export format follows the apr-qa-evidence.schema.json specification:
//! - Model metadata for identification
//! - Playbook metadata for tier/version tracking
//! - Summary statistics for quick lookup
//! - MQS score and breakdown for certification
//! - Gateway results for compliance checking
//! - Full evidence array for reproducibility

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Model metadata in the evidence export.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMeta {
    /// HuggingFace repository ID (e.g., "Qwen/Qwen2.5-Coder-0.5B-Instruct")
    pub hf_repo: String,
    /// Model family (e.g., "qwen2")
    pub family: String,
    /// Size variant (e.g., "0.5b")
    pub size: String,
    /// Ground truth format (e.g., "safetensors")
    pub format: String,
}

/// Playbook metadata in the evidence export.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlaybookMeta {
    /// Playbook name (e.g., "qwen2.5-coder-0.5b-mvp")
    pub name: String,
    /// Playbook version
    pub version: String,
    /// Certification tier (smoke, mvp, full)
    pub tier: String,
}

/// Summary statistics for the evidence export.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportSummary {
    /// Total number of scenarios
    pub total_scenarios: usize,
    /// Number of passed scenarios
    pub passed: usize,
    /// Number of failed scenarios
    pub failed: usize,
    /// Number of skipped scenarios
    pub skipped: usize,
    /// Pass rate (0.0 - 1.0)
    pub pass_rate: f64,
    /// Total duration in milliseconds
    pub duration_ms: u64,
    /// Run timestamp
    pub timestamp: DateTime<Utc>,
}

/// MQS (Model Qualification Score) in the evidence export.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MqsExport {
    /// Raw MQS score (0-1000)
    pub score: u32,
    /// Letter grade (A, B, C, D, F)
    pub grade: String,
    /// Whether all gateways passed
    pub gateway_passed: bool,
    /// Category score breakdown
    pub category_scores: HashMap<String, u32>,
}

/// Gate result in the evidence export.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateResult {
    /// Whether the gate passed
    pub passed: bool,
    /// Human-readable reason
    pub reason: String,
}

/// Complete evidence export structure for oracle consumption.
///
/// This structure is serialized to JSON and consumed by:
/// - `apr oracle` for certification status lookup
/// - CI/CD pipelines for quality gates
/// - Audit trails for reproducibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceExport {
    /// JSON Schema URL
    #[serde(rename = "$schema")]
    pub schema: String,
    /// Format version
    pub version: String,
    /// Model metadata
    pub model: ModelMeta,
    /// Playbook metadata
    pub playbook: PlaybookMeta,
    /// Summary statistics
    pub summary: ExportSummary,
    /// MQS score and breakdown
    pub mqs: MqsExport,
    /// Gateway/gate results
    pub gates: HashMap<String, GateResult>,
    /// Full evidence array
    pub evidence: Vec<serde_json::Value>,
}

impl Default for EvidenceExport {
    fn default() -> Self {
        Self {
            schema: "https://paiml.com/schemas/apr-qa-evidence.schema.json".to_string(),
            version: "1.0.0".to_string(),
            model: ModelMeta {
                hf_repo: String::new(),
                family: String::new(),
                size: String::new(),
                format: "safetensors".to_string(),
            },
            playbook: PlaybookMeta {
                name: String::new(),
                version: "1.0.0".to_string(),
                tier: "mvp".to_string(),
            },
            summary: ExportSummary {
                total_scenarios: 0,
                passed: 0,
                failed: 0,
                skipped: 0,
                pass_rate: 0.0,
                duration_ms: 0,
                timestamp: Utc::now(),
            },
            mqs: MqsExport {
                score: 0,
                grade: "F".to_string(),
                gateway_passed: false,
                category_scores: HashMap::new(),
            },
            gates: HashMap::new(),
            evidence: Vec::new(),
        }
    }
}

impl EvidenceExport {
    /// Create a new evidence export builder.
    #[must_use]
    pub fn builder() -> EvidenceExportBuilder {
        EvidenceExportBuilder::default()
    }

    /// Create an EvidenceExport from MqsScore and evidence.
    ///
    /// This method bridges the internal MQS calculation with the external
    /// evidence export format consumed by the oracle.
    ///
    /// # Arguments
    ///
    /// * `mqs` - The calculated MQS score
    /// * `evidence` - Raw evidence as serialized JSON values
    /// * `model` - Model metadata
    /// * `playbook` - Playbook metadata
    #[must_use]
    pub fn from_mqs_score(
        mqs: &crate::mqs::MqsScore,
        evidence: Vec<serde_json::Value>,
        model: ModelMeta,
        playbook: PlaybookMeta,
    ) -> Self {
        let total_scenarios = mqs.total_tests;
        let passed = mqs.tests_passed;
        let failed = mqs.tests_failed;
        let skipped = total_scenarios
            .saturating_sub(passed)
            .saturating_sub(failed);

        let pass_rate = if total_scenarios > 0 {
            passed as f64 / total_scenarios as f64
        } else {
            0.0
        };

        // Calculate total duration from evidence metrics
        let duration_ms = evidence
            .iter()
            .filter_map(|e| e.get("metrics").and_then(|m| m.get("duration_ms")))
            .filter_map(serde_json::Value::as_u64)
            .sum();

        // Build category scores from MQS categories
        let mut category_scores = HashMap::new();
        let breakdown = mqs.categories.breakdown();
        for (cat, (score, _max)) in breakdown {
            category_scores.insert(cat.to_lowercase(), score);
        }

        // Build gate results from MQS gateways
        let mut gates = HashMap::new();
        for gateway in &mqs.gateways {
            gates.insert(
                match gateway.id.as_str() {
                    "G0" => "G0-INTEGRITY".to_string(),
                    "G1" => "G1-MODEL-LOADS".to_string(),
                    "G2" => "G2-BASIC-INFERENCE".to_string(),
                    "G3" => "G3-NO-CRASHES".to_string(),
                    "G4" => "G4-OUTPUT-QUALITY".to_string(),
                    other => other.to_string(),
                },
                GateResult {
                    passed: gateway.passed,
                    reason: gateway
                        .failure_reason
                        .clone()
                        .unwrap_or_else(|| gateway.description.clone()),
                },
            );
        }

        Self {
            schema: "https://paiml.com/schemas/apr-qa-evidence.schema.json".to_string(),
            version: "1.0.0".to_string(),
            model,
            playbook,
            summary: ExportSummary {
                total_scenarios,
                passed,
                failed,
                skipped,
                pass_rate,
                duration_ms,
                timestamp: Utc::now(),
            },
            mqs: MqsExport {
                score: mqs.raw_score,
                grade: mqs.grade.clone(),
                gateway_passed: mqs.gateways_passed,
                category_scores,
            },
            gates,
            evidence,
        }
    }

    /// Serialize to JSON string.
    ///
    /// # Errors
    ///
    /// Returns an error if serialization fails.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Deserialize from JSON string.
    ///
    /// # Errors
    ///
    /// Returns an error if deserialization fails.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Calculate pass rate from summary.
    #[must_use]
    pub fn calculate_pass_rate(&self) -> f64 {
        if self.summary.total_scenarios == 0 {
            0.0
        } else {
            self.summary.passed as f64 / self.summary.total_scenarios as f64
        }
    }

    /// Check if all mandatory gateways passed.
    #[must_use]
    pub fn all_gateways_passed(&self) -> bool {
        // G1-G4 are mandatory gateways
        let mandatory = [
            "G1-MODEL-LOADS",
            "G2-BASIC-INFERENCE",
            "G3-NO-CRASHES",
            "G4-OUTPUT-QUALITY",
        ];
        mandatory
            .iter()
            .all(|gate| self.gates.get(*gate).is_some_and(|result| result.passed))
    }

    /// Derive certification status from MQS and gateways.
    #[must_use]
    pub fn derive_status(&self) -> &'static str {
        if self.mqs.score >= 800 && self.mqs.gateway_passed {
            "CERTIFIED"
        } else if self.mqs.score == 0 {
            "UNTESTED"
        } else {
            "BLOCKED"
        }
    }
}

/// Builder for `EvidenceExport`.
#[derive(Debug, Clone, Default)]
pub struct EvidenceExportBuilder {
    export: EvidenceExport,
}

impl EvidenceExportBuilder {
    /// Set model metadata.
    #[must_use]
    pub fn model(
        mut self,
        hf_repo: impl Into<String>,
        family: impl Into<String>,
        size: impl Into<String>,
    ) -> Self {
        self.export.model.hf_repo = hf_repo.into();
        self.export.model.family = family.into();
        self.export.model.size = size.into();
        self
    }

    /// Set model format.
    #[must_use]
    pub fn format(mut self, format: impl Into<String>) -> Self {
        self.export.model.format = format.into();
        self
    }

    /// Set playbook metadata.
    #[must_use]
    pub fn playbook(
        mut self,
        name: impl Into<String>,
        version: impl Into<String>,
        tier: impl Into<String>,
    ) -> Self {
        self.export.playbook.name = name.into();
        self.export.playbook.version = version.into();
        self.export.playbook.tier = tier.into();
        self
    }

    /// Set summary statistics.
    #[must_use]
    pub fn summary(
        mut self,
        total: usize,
        passed: usize,
        failed: usize,
        skipped: usize,
        duration_ms: u64,
    ) -> Self {
        self.export.summary.total_scenarios = total;
        self.export.summary.passed = passed;
        self.export.summary.failed = failed;
        self.export.summary.skipped = skipped;
        self.export.summary.pass_rate = if total > 0 {
            passed as f64 / total as f64
        } else {
            0.0
        };
        self.export.summary.duration_ms = duration_ms;
        self.export.summary.timestamp = Utc::now();
        self
    }

    /// Set MQS score.
    #[must_use]
    pub fn mqs(mut self, score: u32, grade: impl Into<String>, gateway_passed: bool) -> Self {
        self.export.mqs.score = score;
        self.export.mqs.grade = grade.into();
        self.export.mqs.gateway_passed = gateway_passed;
        self
    }

    /// Add a category score.
    #[must_use]
    pub fn category_score(mut self, category: impl Into<String>, score: u32) -> Self {
        self.export
            .mqs
            .category_scores
            .insert(category.into(), score);
        self
    }

    /// Add a gate result.
    #[must_use]
    pub fn gate(
        mut self,
        gate_id: impl Into<String>,
        passed: bool,
        reason: impl Into<String>,
    ) -> Self {
        self.export.gates.insert(
            gate_id.into(),
            GateResult {
                passed,
                reason: reason.into(),
            },
        );
        self
    }

    /// Add evidence items.
    #[must_use]
    pub fn evidence(mut self, evidence: Vec<serde_json::Value>) -> Self {
        self.export.evidence = evidence;
        self
    }

    /// Build the export.
    #[must_use]
    pub fn build(self) -> EvidenceExport {
        self.export
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // FALSIFY-EVIDENCE-001: Round-trip integrity
    //
    // Falsification hypothesis: "JSON round-trip corrupts evidence data"
    // If from_json(to_json(export)) != export semantically, implementation is broken.
    #[test]
    fn test_falsify_evidence_001_roundtrip_integrity() {
        let export = EvidenceExport::builder()
            .model("Qwen/Qwen2.5-Coder-0.5B-Instruct", "qwen2", "0.5b")
            .format("safetensors")
            .playbook("qwen2.5-coder-0.5b-mvp", "1.0.0", "mvp")
            .summary(47, 27, 20, 0, 1_134_808)
            .mqs(574, "C", true)
            .category_score("inference", 600)
            .category_score("conversion", 400)
            .gate("G1-MODEL-LOADS", true, "Model loaded")
            .gate("G2-BASIC-INFERENCE", true, "Inference works")
            .gate("G3-NO-CRASHES", true, "No crashes")
            .gate(
                "G4-OUTPUT-QUALITY",
                false,
                "Conversion diffs exceed tolerance",
            )
            .build();

        let json = export.to_json().expect("serialize");
        let roundtrip = EvidenceExport::from_json(&json).expect("deserialize");

        // Verify key fields survive round-trip
        assert_eq!(roundtrip.model.hf_repo, export.model.hf_repo);
        assert_eq!(roundtrip.model.family, export.model.family);
        assert_eq!(roundtrip.model.size, export.model.size);
        assert_eq!(roundtrip.playbook.name, export.playbook.name);
        assert_eq!(roundtrip.playbook.tier, export.playbook.tier);
        assert_eq!(
            roundtrip.summary.total_scenarios,
            export.summary.total_scenarios
        );
        assert_eq!(roundtrip.summary.passed, export.summary.passed);
        assert_eq!(roundtrip.mqs.score, export.mqs.score);
        assert_eq!(roundtrip.mqs.grade, export.mqs.grade);
        assert_eq!(roundtrip.gates.len(), export.gates.len());
    }

    #[test]
    fn test_evidence_export_default() {
        let export = EvidenceExport::default();
        assert!(export.schema.contains("apr-qa-evidence"));
        assert_eq!(export.version, "1.0.0");
        assert!(export.model.hf_repo.is_empty());
        assert_eq!(export.mqs.score, 0);
    }

    #[test]
    fn test_builder_model() {
        let export = EvidenceExport::builder()
            .model("org/model", "family", "1b")
            .build();

        assert_eq!(export.model.hf_repo, "org/model");
        assert_eq!(export.model.family, "family");
        assert_eq!(export.model.size, "1b");
    }

    #[test]
    fn test_builder_playbook() {
        let export = EvidenceExport::builder()
            .playbook("test-playbook", "2.0.0", "full")
            .build();

        assert_eq!(export.playbook.name, "test-playbook");
        assert_eq!(export.playbook.version, "2.0.0");
        assert_eq!(export.playbook.tier, "full");
    }

    #[test]
    fn test_builder_summary() {
        let export = EvidenceExport::builder()
            .summary(100, 80, 15, 5, 50000)
            .build();

        assert_eq!(export.summary.total_scenarios, 100);
        assert_eq!(export.summary.passed, 80);
        assert_eq!(export.summary.failed, 15);
        assert_eq!(export.summary.skipped, 5);
        assert!((export.summary.pass_rate - 0.8).abs() < 0.001);
        assert_eq!(export.summary.duration_ms, 50000);
    }

    #[test]
    fn test_builder_mqs() {
        let export = EvidenceExport::builder()
            .mqs(850, "B", true)
            .category_score("inference", 900)
            .category_score("stability", 800)
            .build();

        assert_eq!(export.mqs.score, 850);
        assert_eq!(export.mqs.grade, "B");
        assert!(export.mqs.gateway_passed);
        assert_eq!(export.mqs.category_scores.get("inference"), Some(&900));
        assert_eq!(export.mqs.category_scores.get("stability"), Some(&800));
    }

    #[test]
    fn test_builder_gates() {
        let export = EvidenceExport::builder()
            .gate("G1-MODEL-LOADS", true, "OK")
            .gate("G2-BASIC-INFERENCE", false, "Failed")
            .build();

        assert_eq!(export.gates.len(), 2);
        assert!(export.gates.get("G1-MODEL-LOADS").unwrap().passed);
        assert!(!export.gates.get("G2-BASIC-INFERENCE").unwrap().passed);
    }

    #[test]
    fn test_calculate_pass_rate() {
        let export = EvidenceExport::builder()
            .summary(100, 75, 25, 0, 1000)
            .build();

        assert!((export.calculate_pass_rate() - 0.75).abs() < 0.001);
    }

    #[test]
    fn test_calculate_pass_rate_empty() {
        let export = EvidenceExport::default();
        assert!((export.calculate_pass_rate() - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_all_gateways_passed() {
        let export = EvidenceExport::builder()
            .gate("G1-MODEL-LOADS", true, "OK")
            .gate("G2-BASIC-INFERENCE", true, "OK")
            .gate("G3-NO-CRASHES", true, "OK")
            .gate("G4-OUTPUT-QUALITY", true, "OK")
            .build();

        assert!(export.all_gateways_passed());
    }

    #[test]
    fn test_all_gateways_one_failed() {
        let export = EvidenceExport::builder()
            .gate("G1-MODEL-LOADS", true, "OK")
            .gate("G2-BASIC-INFERENCE", true, "OK")
            .gate("G3-NO-CRASHES", false, "Crashed")
            .gate("G4-OUTPUT-QUALITY", true, "OK")
            .build();

        assert!(!export.all_gateways_passed());
    }

    #[test]
    fn test_all_gateways_missing() {
        let export = EvidenceExport::builder()
            .gate("G1-MODEL-LOADS", true, "OK")
            .build();

        assert!(!export.all_gateways_passed());
    }

    #[test]
    fn test_derive_status_certified() {
        let export = EvidenceExport::builder().mqs(850, "B", true).build();

        assert_eq!(export.derive_status(), "CERTIFIED");
    }

    #[test]
    fn test_derive_status_blocked_low_score() {
        let export = EvidenceExport::builder().mqs(500, "D", true).build();

        assert_eq!(export.derive_status(), "BLOCKED");
    }

    #[test]
    fn test_derive_status_blocked_gateway_failed() {
        let export = EvidenceExport::builder().mqs(900, "A", false).build();

        assert_eq!(export.derive_status(), "BLOCKED");
    }

    #[test]
    fn test_derive_status_untested() {
        let export = EvidenceExport::default();
        assert_eq!(export.derive_status(), "UNTESTED");
    }

    #[test]
    fn test_to_json_contains_schema() {
        let export = EvidenceExport::default();
        let json = export.to_json().expect("serialize");
        assert!(json.contains("$schema"));
        assert!(json.contains("apr-qa-evidence.schema.json"));
    }

    #[test]
    fn test_evidence_array() {
        let evidence = vec![
            serde_json::json!({"id": "1", "outcome": "Corroborated"}),
            serde_json::json!({"id": "2", "outcome": "Falsified"}),
        ];

        let export = EvidenceExport::builder().evidence(evidence).build();

        assert_eq!(export.evidence.len(), 2);
    }

    #[test]
    fn test_builder_format() {
        let export = EvidenceExport::builder().format("gguf").build();

        assert_eq!(export.model.format, "gguf");
    }

    #[test]
    fn test_summary_zero_total() {
        let export = EvidenceExport::builder().summary(0, 0, 0, 0, 0).build();

        assert_eq!(export.summary.pass_rate, 0.0);
    }

    #[test]
    fn test_serde_gate_result() {
        let gate = GateResult {
            passed: true,
            reason: "Test passed".to_string(),
        };

        let json = serde_json::to_string(&gate).expect("serialize");
        assert!(json.contains("true"));
        assert!(json.contains("Test passed"));

        let deserialized: GateResult = serde_json::from_str(&json).expect("deserialize");
        assert!(deserialized.passed);
        assert_eq!(deserialized.reason, "Test passed");
    }

    #[test]
    fn test_clone_export() {
        let export = EvidenceExport::builder()
            .model("test/model", "test", "1b")
            .mqs(750, "C", true)
            .build();

        let cloned = export.clone();
        assert_eq!(cloned.model.hf_repo, export.model.hf_repo);
        assert_eq!(cloned.mqs.score, export.mqs.score);
    }

    // FALSIFY-MQS-001: MqsScore conversion integrity
    //
    // Falsification hypothesis: "from_mqs_score loses critical data"
    // If converted export doesn't preserve MQS score, grade, and gateway status, broken.
    #[test]
    fn test_falsify_mqs_001_conversion_integrity() {
        use crate::mqs::{CategoryScores, GatewayResult as MqsGateway, MqsScore};

        let mqs = MqsScore {
            model_id: "test/model".to_string(),
            raw_score: 850,
            normalized_score: 85.0,
            grade: "B".to_string(),
            gateways: vec![
                MqsGateway::passed("G1", "Model loads"),
                MqsGateway::passed("G2", "Inference works"),
                MqsGateway::passed("G3", "No crashes"),
                MqsGateway::failed("G4", "Output quality", "Conversion diffs"),
            ],
            gateways_passed: false,
            categories: CategoryScores {
                qual: 180,
                perf: 140,
                stab: 190,
                comp: 130,
                edge: 120,
                regr: 90,
            },
            total_tests: 100,
            tests_passed: 85,
            tests_failed: 15,
            penalties: vec![],
            total_penalty: 0,
        };

        let model = ModelMeta {
            hf_repo: "Qwen/Qwen2.5-Coder-0.5B-Instruct".to_string(),
            family: "qwen2".to_string(),
            size: "0.5b".to_string(),
            format: "safetensors".to_string(),
        };

        let playbook = PlaybookMeta {
            name: "qwen2.5-coder-0.5b-mvp".to_string(),
            version: "1.0.0".to_string(),
            tier: "mvp".to_string(),
        };

        let evidence = vec![serde_json::json!({
            "id": "1",
            "outcome": "Corroborated",
            "metrics": {"duration_ms": 1000}
        })];

        let export = EvidenceExport::from_mqs_score(&mqs, evidence, model, playbook);

        // Verify MQS data preserved
        assert_eq!(export.mqs.score, 850);
        assert_eq!(export.mqs.grade, "B");
        assert!(!export.mqs.gateway_passed);

        // Verify summary computed correctly
        assert_eq!(export.summary.total_scenarios, 100);
        assert_eq!(export.summary.passed, 85);
        assert_eq!(export.summary.failed, 15);

        // Verify model metadata preserved
        assert_eq!(export.model.hf_repo, "Qwen/Qwen2.5-Coder-0.5B-Instruct");
        assert_eq!(export.model.family, "qwen2");

        // Verify playbook metadata preserved
        assert_eq!(export.playbook.name, "qwen2.5-coder-0.5b-mvp");
        assert_eq!(export.playbook.tier, "mvp");

        // Verify gates converted
        assert!(export.gates.contains_key("G1-MODEL-LOADS"));
        assert!(export.gates.get("G1-MODEL-LOADS").unwrap().passed);
        assert!(!export.gates.get("G4-OUTPUT-QUALITY").unwrap().passed);
    }

    #[test]
    fn test_from_mqs_score_category_breakdown() {
        use crate::mqs::{CategoryScores, GatewayResult as MqsGateway, MqsScore};

        let mqs = MqsScore {
            model_id: "test".to_string(),
            raw_score: 500,
            normalized_score: 50.0,
            grade: "D".to_string(),
            gateways: vec![MqsGateway::passed("G1", "OK")],
            gateways_passed: true,
            categories: CategoryScores {
                qual: 100,
                perf: 80,
                stab: 100,
                comp: 70,
                edge: 80,
                regr: 70,
            },
            total_tests: 50,
            tests_passed: 25,
            tests_failed: 25,
            penalties: vec![],
            total_penalty: 0,
        };

        let export = EvidenceExport::from_mqs_score(
            &mqs,
            vec![],
            ModelMeta {
                hf_repo: "test".to_string(),
                family: "test".to_string(),
                size: "1b".to_string(),
                format: "gguf".to_string(),
            },
            PlaybookMeta {
                name: "test".to_string(),
                version: "1.0.0".to_string(),
                tier: "smoke".to_string(),
            },
        );

        // Verify category scores converted (keys are lowercased)
        assert_eq!(export.mqs.category_scores.get("qual"), Some(&100));
        assert_eq!(export.mqs.category_scores.get("perf"), Some(&80));
        assert_eq!(export.mqs.category_scores.get("stab"), Some(&100));
    }

    #[test]
    fn test_from_mqs_score_pass_rate() {
        use crate::mqs::{CategoryScores, MqsScore};

        let mqs = MqsScore {
            model_id: "test".to_string(),
            raw_score: 750,
            normalized_score: 75.0,
            grade: "C".to_string(),
            gateways: vec![],
            gateways_passed: true,
            categories: CategoryScores::default(),
            total_tests: 80,
            tests_passed: 60,
            tests_failed: 20,
            penalties: vec![],
            total_penalty: 0,
        };

        let export = EvidenceExport::from_mqs_score(
            &mqs,
            vec![],
            ModelMeta {
                hf_repo: "t".to_string(),
                family: "t".to_string(),
                size: "1b".to_string(),
                format: "gguf".to_string(),
            },
            PlaybookMeta {
                name: "t".to_string(),
                version: "1".to_string(),
                tier: "smoke".to_string(),
            },
        );

        assert!((export.summary.pass_rate - 0.75).abs() < 0.001);
    }

    // ── PMAT-267: Oracle Integration Tests ──────────────────────────────────
    //
    // These tests verify the contract between EvidenceExport and CertificationRow,
    // ensuring the oracle can correctly consume certification data.

    // FALSIFY-ORACLE-001: Evidence→Certification data contract
    //
    // Falsification hypothesis: "EvidenceExport status derivation differs from CertificationRow"
    // If derive_status() returns different values for equivalent inputs, oracle is broken.
    #[test]
    fn test_falsify_oracle_001_status_contract() {
        use crate::certification_data::{CertificationRow, ModelStatus};

        // Test case 1: CERTIFIED (MQS >= 800, gateways passed)
        let export = EvidenceExport::builder().mqs(850, "B", true).build();

        let cert_row = CertificationRow {
            mqs_score: 850,
            g1: true,
            g2: true,
            g3: true,
            g4: true,
            ..Default::default()
        };

        assert_eq!(export.derive_status(), "CERTIFIED");
        assert_eq!(cert_row.derive_status(), ModelStatus::Certified);

        // Test case 2: BLOCKED (MQS < 800)
        let export_blocked = EvidenceExport::builder().mqs(500, "D", true).build();

        let cert_row_blocked = CertificationRow {
            mqs_score: 500,
            g1: true,
            g2: true,
            g3: true,
            g4: true,
            ..Default::default()
        };

        assert_eq!(export_blocked.derive_status(), "BLOCKED");
        assert_eq!(cert_row_blocked.derive_status(), ModelStatus::Blocked);

        // Test case 3: BLOCKED (gateway failed)
        let export_gw_fail = EvidenceExport::builder().mqs(900, "A", false).build();

        let cert_row_gw_fail = CertificationRow {
            mqs_score: 900,
            g1: true,
            g2: true,
            g3: false, // Gateway failed
            g4: true,
            ..Default::default()
        };

        assert_eq!(export_gw_fail.derive_status(), "BLOCKED");
        assert_eq!(cert_row_gw_fail.derive_status(), ModelStatus::Blocked);
    }

    // FALSIFY-ORACLE-002: Grade derivation consistency
    //
    // Falsification hypothesis: "Grade thresholds differ between modules"
    // Both modules must use identical grade thresholds.
    #[test]
    fn test_falsify_oracle_002_grade_contract() {
        use crate::certification_data::CertificationRow;

        let grade_cases = [
            (950, "A"),
            (900, "A"),
            (850, "B"),
            (800, "B"),
            (700, "C"),
            (600, "C"),
            (500, "D"),
            (400, "D"),
            (300, "F"),
            (0, "F"),
        ];

        for (score, expected_grade) in grade_cases {
            let cert_row = CertificationRow {
                mqs_score: score,
                ..Default::default()
            };

            assert_eq!(
                cert_row.derive_grade(),
                expected_grade,
                "Grade mismatch for score {score}"
            );
        }
    }

    // FALSIFY-ORACLE-003: Evidence JSON schema compliance
    //
    // Falsification hypothesis: "Serialized JSON lacks required oracle fields"
    // Oracle requires: $schema, model.hf_repo, mqs.score, mqs.grade, gates
    #[test]
    fn test_falsify_oracle_003_schema_compliance() {
        let export = EvidenceExport::builder()
            .model("Qwen/Qwen2.5-Coder-0.5B-Instruct", "qwen2", "0.5b")
            .mqs(850, "B", true)
            .gate("G1-MODEL-LOADS", true, "OK")
            .gate("G2-BASIC-INFERENCE", true, "OK")
            .gate("G3-NO-CRASHES", true, "OK")
            .gate("G4-OUTPUT-QUALITY", true, "OK")
            .build();

        let json = export.to_json().expect("serialize");

        // Verify required oracle fields present
        assert!(json.contains("\"$schema\""), "Missing $schema field");
        assert!(json.contains("\"hf_repo\""), "Missing model.hf_repo field");
        assert!(json.contains("\"score\""), "Missing mqs.score field");
        assert!(json.contains("\"grade\""), "Missing mqs.grade field");
        assert!(json.contains("\"gates\""), "Missing gates field");
        assert!(
            json.contains("\"gateway_passed\""),
            "Missing gateway_passed field"
        );

        // Verify specific values
        assert!(json.contains("Qwen/Qwen2.5-Coder-0.5B-Instruct"));
        assert!(json.contains("850"));
        assert!(json.contains("\"B\""));
    }

    // FALSIFY-ORACLE-004: CertificationRow↔EvidenceExport field mapping
    //
    // Falsification hypothesis: "Field names differ between CSV and JSON"
    // Oracle must be able to map between CSV columns and JSON fields.
    #[test]
    fn test_falsify_oracle_004_field_mapping() {
        use crate::certification_data::CertificationRow;

        // Create equivalent data in both formats
        let export = EvidenceExport::builder()
            .model("test/model", "test-family", "1b")
            .playbook("test-1b-mvp", "1.0.0", "mvp")
            .summary(100, 85, 15, 0, 50000)
            .mqs(850, "B", true)
            .gate("G1-MODEL-LOADS", true, "OK")
            .gate("G2-BASIC-INFERENCE", true, "OK")
            .gate("G3-NO-CRASHES", true, "OK")
            .gate("G4-OUTPUT-QUALITY", true, "OK")
            .build();

        let cert_row = CertificationRow {
            model_id: "test/model".to_string(),
            family: "test-family".to_string(),
            parameters: "1B".to_string(),
            mqs_score: 850,
            grade: "B".to_string(),
            certified_tier: "mvp".to_string(),
            g1: true,
            g2: true,
            g3: true,
            g4: true,
            ..Default::default()
        };

        // Verify field mappings
        assert_eq!(export.model.hf_repo, cert_row.model_id);
        assert_eq!(export.model.family, cert_row.family);
        assert_eq!(export.mqs.score, cert_row.mqs_score);
        assert_eq!(export.mqs.grade, cert_row.grade);
        assert_eq!(export.playbook.tier, cert_row.certified_tier);

        // Verify gateway consistency
        assert_eq!(
            export.gates.get("G1-MODEL-LOADS").unwrap().passed,
            cert_row.g1
        );
        assert_eq!(
            export.gates.get("G2-BASIC-INFERENCE").unwrap().passed,
            cert_row.g2
        );
        assert_eq!(
            export.gates.get("G3-NO-CRASHES").unwrap().passed,
            cert_row.g3
        );
        assert_eq!(
            export.gates.get("G4-OUTPUT-QUALITY").unwrap().passed,
            cert_row.g4
        );
    }

    // FALSIFY-ORACLE-005: Evidence export reproducibility
    //
    // Falsification hypothesis: "Same input produces different exports"
    // If two exports from identical MqsScore differ (except timestamp), broken.
    #[test]
    fn test_falsify_oracle_005_reproducibility() {
        use crate::mqs::{CategoryScores, GatewayResult as MqsGateway, MqsScore};

        let mqs = MqsScore {
            model_id: "test/model".to_string(),
            raw_score: 850,
            normalized_score: 85.0,
            grade: "B".to_string(),
            gateways: vec![
                MqsGateway::passed("G1", "Model loads"),
                MqsGateway::passed("G2", "Inference works"),
            ],
            gateways_passed: true,
            categories: CategoryScores {
                qual: 180,
                perf: 140,
                stab: 190,
                comp: 130,
                edge: 120,
                regr: 90,
            },
            total_tests: 100,
            tests_passed: 85,
            tests_failed: 15,
            penalties: vec![],
            total_penalty: 0,
        };

        let model = ModelMeta {
            hf_repo: "test/model".to_string(),
            family: "test".to_string(),
            size: "1b".to_string(),
            format: "safetensors".to_string(),
        };

        let playbook = PlaybookMeta {
            name: "test-1b".to_string(),
            version: "1.0.0".to_string(),
            tier: "mvp".to_string(),
        };

        let export1 = EvidenceExport::from_mqs_score(&mqs, vec![], model.clone(), playbook.clone());
        let export2 = EvidenceExport::from_mqs_score(&mqs, vec![], model, playbook);

        // All fields except timestamp should be identical
        assert_eq!(export1.mqs.score, export2.mqs.score);
        assert_eq!(export1.mqs.grade, export2.mqs.grade);
        assert_eq!(export1.mqs.gateway_passed, export2.mqs.gateway_passed);
        assert_eq!(
            export1.summary.total_scenarios,
            export2.summary.total_scenarios
        );
        assert_eq!(export1.summary.passed, export2.summary.passed);
        assert_eq!(export1.summary.failed, export2.summary.failed);
        assert_eq!(export1.model.hf_repo, export2.model.hf_repo);
        assert_eq!(export1.playbook.name, export2.playbook.name);
    }
}
