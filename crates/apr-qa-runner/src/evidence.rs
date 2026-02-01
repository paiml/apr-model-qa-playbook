//! Evidence collection for falsification results
//!
//! Every test produces evidence that is recorded regardless of outcome.

use apr_qa_gen::QaScenario;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Outcome of a test
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Outcome {
    /// Hypothesis not falsified
    Corroborated,
    /// Hypothesis falsified
    Falsified,
    /// Test skipped
    Skipped,
    /// Test timed out
    Timeout,
    /// Test crashed
    Crashed,
}

impl Outcome {
    /// Check if this is a passing outcome
    #[must_use]
    pub const fn is_pass(&self) -> bool {
        matches!(self, Self::Corroborated | Self::Skipped)
    }

    /// Check if this is a failing outcome
    #[must_use]
    pub const fn is_fail(&self) -> bool {
        matches!(self, Self::Falsified | Self::Timeout | Self::Crashed)
    }
}

/// Performance metrics from a test run
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Tokens per second
    pub tokens_per_second: Option<f64>,
    /// Time to first token in milliseconds
    pub time_to_first_token_ms: Option<f64>,
    /// Total tokens generated
    pub total_tokens: Option<u32>,
    /// Peak memory usage in MB
    pub memory_peak_mb: Option<u64>,
    /// Total duration in milliseconds
    pub duration_ms: u64,
}

/// Host information for reproducibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HostInfo {
    /// Hostname
    pub hostname: String,
    /// Operating system
    pub os: String,
    /// CPU model
    pub cpu: String,
    /// GPU model (if available)
    pub gpu: Option<String>,
    /// apr-cli version
    pub apr_version: String,
}

impl Default for HostInfo {
    fn default() -> Self {
        Self {
            hostname: hostname::get().map_or_else(
                |_| "unknown".to_string(),
                |h| h.to_string_lossy().to_string(),
            ),
            os: std::env::consts::OS.to_string(),
            cpu: "unknown".to_string(),
            gpu: None,
            apr_version: "unknown".to_string(),
        }
    }
}

/// Evidence from a single test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Evidence {
    /// Unique evidence ID
    pub id: String,
    /// Gate ID (e.g., "F-HTTP-001")
    pub gate_id: String,
    /// Scenario that was tested
    pub scenario: QaScenario,
    /// Test outcome
    pub outcome: Outcome,
    /// Human-readable reason
    pub reason: String,
    /// Raw output from the command
    pub output: String,
    /// Standard error output
    pub stderr: Option<String>,
    /// Exit code
    pub exit_code: Option<i32>,
    /// Performance metrics
    pub metrics: PerformanceMetrics,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Host information
    pub host: HostInfo,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl Evidence {
    /// Create new evidence for a corroborated test
    #[must_use]
    pub fn corroborated(
        gate_id: impl Into<String>,
        scenario: QaScenario,
        output: impl Into<String>,
        duration_ms: u64,
    ) -> Self {
        Self {
            id: uuid_v4(),
            gate_id: gate_id.into(),
            scenario,
            outcome: Outcome::Corroborated,
            reason: "Test passed".to_string(),
            output: output.into(),
            stderr: None,
            exit_code: Some(0),
            metrics: PerformanceMetrics {
                duration_ms,
                ..Default::default()
            },
            timestamp: Utc::now(),
            host: HostInfo::default(),
            metadata: HashMap::new(),
        }
    }

    /// Create new evidence for a falsified test
    #[must_use]
    pub fn falsified(
        gate_id: impl Into<String>,
        scenario: QaScenario,
        reason: impl Into<String>,
        output: impl Into<String>,
        duration_ms: u64,
    ) -> Self {
        Self {
            id: uuid_v4(),
            gate_id: gate_id.into(),
            scenario,
            outcome: Outcome::Falsified,
            reason: reason.into(),
            output: output.into(),
            stderr: None,
            exit_code: None,
            metrics: PerformanceMetrics {
                duration_ms,
                ..Default::default()
            },
            timestamp: Utc::now(),
            host: HostInfo::default(),
            metadata: HashMap::new(),
        }
    }

    /// Create new evidence for a timeout
    #[must_use]
    pub fn timeout(gate_id: impl Into<String>, scenario: QaScenario, timeout_ms: u64) -> Self {
        Self {
            id: uuid_v4(),
            gate_id: gate_id.into(),
            scenario,
            outcome: Outcome::Timeout,
            reason: format!("Timed out after {timeout_ms}ms"),
            output: String::new(),
            stderr: None,
            exit_code: None,
            metrics: PerformanceMetrics {
                duration_ms: timeout_ms,
                ..Default::default()
            },
            timestamp: Utc::now(),
            host: HostInfo::default(),
            metadata: HashMap::new(),
        }
    }

    /// Create new evidence for a crash
    #[must_use]
    pub fn crashed(
        gate_id: impl Into<String>,
        scenario: QaScenario,
        stderr: impl Into<String>,
        exit_code: i32,
        duration_ms: u64,
    ) -> Self {
        Self {
            id: uuid_v4(),
            gate_id: gate_id.into(),
            scenario,
            outcome: Outcome::Crashed,
            reason: format!("Process crashed with exit code {exit_code}"),
            output: String::new(),
            stderr: Some(stderr.into()),
            exit_code: Some(exit_code),
            metrics: PerformanceMetrics {
                duration_ms,
                ..Default::default()
            },
            timestamp: Utc::now(),
            host: HostInfo::default(),
            metadata: HashMap::new(),
        }
    }

    /// Create new evidence for a skipped test
    #[must_use]
    pub fn skipped(
        gate_id: impl Into<String>,
        scenario: QaScenario,
        reason: impl Into<String>,
    ) -> Self {
        Self {
            id: uuid_v4(),
            gate_id: gate_id.into(),
            scenario,
            outcome: Outcome::Skipped,
            reason: reason.into(),
            output: String::new(),
            stderr: None,
            exit_code: None,
            metrics: PerformanceMetrics::default(),
            timestamp: Utc::now(),
            host: HostInfo::default(),
            metadata: HashMap::new(),
        }
    }

    /// Add performance metrics
    #[must_use]
    pub const fn with_metrics(mut self, metrics: PerformanceMetrics) -> Self {
        self.metrics = metrics;
        self
    }

    /// Add metadata
    pub fn add_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.metadata.insert(key.into(), value.into());
    }
}

/// Collector for evidence from multiple tests
#[derive(Debug, Clone, Default)]
pub struct EvidenceCollector {
    evidence: Vec<Evidence>,
}

impl EvidenceCollector {
    /// Create a new collector
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add evidence
    pub fn add(&mut self, evidence: Evidence) {
        self.evidence.push(evidence);
    }

    /// Get all evidence
    #[must_use]
    pub fn all(&self) -> &[Evidence] {
        &self.evidence
    }

    /// Get count of each outcome type
    #[must_use]
    pub fn counts(&self) -> HashMap<Outcome, usize> {
        let mut counts = HashMap::new();
        for e in &self.evidence {
            *counts.entry(e.outcome).or_insert(0) += 1;
        }
        counts
    }

    /// Get pass count
    #[must_use]
    pub fn pass_count(&self) -> usize {
        self.evidence.iter().filter(|e| e.outcome.is_pass()).count()
    }

    /// Get fail count
    #[must_use]
    pub fn fail_count(&self) -> usize {
        self.evidence.iter().filter(|e| e.outcome.is_fail()).count()
    }

    /// Get total count
    #[must_use]
    pub fn total(&self) -> usize {
        self.evidence.len()
    }

    /// Get failed evidence
    #[must_use]
    pub fn failures(&self) -> Vec<&Evidence> {
        self.evidence
            .iter()
            .filter(|e| e.outcome.is_fail())
            .collect()
    }

    /// Export to JSON
    ///
    /// # Errors
    ///
    /// Returns an error if serialization fails.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(&self.evidence)
    }
}

/// Generate a simple UUID v4
fn uuid_v4() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("{timestamp:032x}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use apr_qa_gen::{Backend, Format, Modality, ModelId};

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

    #[test]
    fn test_evidence_corroborated() {
        let evidence = Evidence::corroborated("F-TEST-001", test_scenario(), "4", 100);
        assert_eq!(evidence.outcome, Outcome::Corroborated);
        assert!(evidence.outcome.is_pass());
    }

    #[test]
    fn test_evidence_falsified() {
        let evidence = Evidence::falsified("F-TEST-001", test_scenario(), "Wrong answer", "5", 100);
        assert_eq!(evidence.outcome, Outcome::Falsified);
        assert!(evidence.outcome.is_fail());
    }

    #[test]
    fn test_evidence_collector() {
        let mut collector = EvidenceCollector::new();
        collector.add(Evidence::corroborated(
            "F-TEST-001",
            test_scenario(),
            "4",
            100,
        ));
        collector.add(Evidence::falsified(
            "F-TEST-002",
            test_scenario(),
            "Failed",
            "",
            100,
        ));

        assert_eq!(collector.total(), 2);
        assert_eq!(collector.pass_count(), 1);
        assert_eq!(collector.fail_count(), 1);
    }

    #[test]
    fn test_outcome_pass_fail() {
        assert!(Outcome::Corroborated.is_pass());
        assert!(Outcome::Skipped.is_pass());
        assert!(!Outcome::Falsified.is_pass());
        assert!(Outcome::Falsified.is_fail());
        assert!(Outcome::Timeout.is_fail());
        assert!(Outcome::Crashed.is_fail());
    }

    #[test]
    fn test_evidence_timeout() {
        let evidence = Evidence::timeout("F-TEST-001", test_scenario(), 30000);
        assert_eq!(evidence.outcome, Outcome::Timeout);
        assert!(evidence.outcome.is_fail());
        assert!(evidence.reason.contains("30000"));
        assert_eq!(evidence.metrics.duration_ms, 30000);
    }

    #[test]
    fn test_evidence_crashed() {
        let evidence = Evidence::crashed("F-TEST-001", test_scenario(), "segfault", 139, 100);
        assert_eq!(evidence.outcome, Outcome::Crashed);
        assert!(evidence.outcome.is_fail());
        assert!(evidence.reason.contains("139"));
        assert_eq!(evidence.stderr, Some("segfault".to_string()));
        assert_eq!(evidence.exit_code, Some(139));
    }

    #[test]
    fn test_evidence_skipped() {
        let evidence = Evidence::skipped("F-TEST-001", test_scenario(), "Format not available");
        assert_eq!(evidence.outcome, Outcome::Skipped);
        assert!(evidence.outcome.is_pass());
        assert!(!evidence.outcome.is_fail());
        assert!(evidence.reason.contains("Format not available"));
        assert!(evidence.output.is_empty());
        assert!(evidence.exit_code.is_none());
    }

    #[test]
    fn test_evidence_with_metrics() {
        let metrics = PerformanceMetrics {
            tokens_per_second: Some(100.0),
            time_to_first_token_ms: Some(50.0),
            total_tokens: Some(1000),
            memory_peak_mb: Some(512),
            duration_ms: 200,
        };
        let evidence = Evidence::corroborated("F-TEST-001", test_scenario(), "output", 100)
            .with_metrics(metrics);
        assert_eq!(evidence.metrics.tokens_per_second, Some(100.0));
        assert_eq!(evidence.metrics.total_tokens, Some(1000));
        assert_eq!(evidence.metrics.duration_ms, 200);
    }

    #[test]
    fn test_evidence_add_metadata() {
        let mut evidence = Evidence::corroborated("F-TEST-001", test_scenario(), "output", 100);
        evidence.add_metadata("key1", "value1");
        evidence.add_metadata("key2", "value2");
        assert_eq!(evidence.metadata.get("key1"), Some(&"value1".to_string()));
        assert_eq!(evidence.metadata.get("key2"), Some(&"value2".to_string()));
    }

    #[test]
    fn test_collector_counts() {
        let mut collector = EvidenceCollector::new();
        collector.add(Evidence::corroborated("F-001", test_scenario(), "ok", 100));
        collector.add(Evidence::corroborated("F-002", test_scenario(), "ok", 100));
        collector.add(Evidence::falsified(
            "F-003",
            test_scenario(),
            "fail",
            "bad",
            100,
        ));
        collector.add(Evidence::timeout("F-004", test_scenario(), 5000));

        let counts = collector.counts();
        assert_eq!(counts.get(&Outcome::Corroborated), Some(&2));
        assert_eq!(counts.get(&Outcome::Falsified), Some(&1));
        assert_eq!(counts.get(&Outcome::Timeout), Some(&1));
    }

    #[test]
    fn test_collector_failures() {
        let mut collector = EvidenceCollector::new();
        collector.add(Evidence::corroborated("F-001", test_scenario(), "ok", 100));
        collector.add(Evidence::falsified(
            "F-002",
            test_scenario(),
            "fail",
            "bad",
            100,
        ));
        collector.add(Evidence::crashed("F-003", test_scenario(), "err", -1, 100));

        let failures = collector.failures();
        assert_eq!(failures.len(), 2);
        assert!(failures.iter().all(|e| e.outcome.is_fail()));
    }

    #[test]
    fn test_collector_to_json() {
        let mut collector = EvidenceCollector::new();
        collector.add(Evidence::corroborated("F-001", test_scenario(), "ok", 100));

        let json = collector.to_json().expect("Failed to serialize");
        assert!(json.contains("F-001"));
        assert!(json.contains("Corroborated"));
    }

    #[test]
    fn test_host_info_default() {
        let host = HostInfo::default();
        assert!(!host.hostname.is_empty());
        assert!(!host.os.is_empty());
        assert_eq!(host.apr_version, "unknown");
    }

    #[test]
    fn test_performance_metrics_default() {
        let metrics = PerformanceMetrics::default();
        assert_eq!(metrics.duration_ms, 0);
        assert!(metrics.tokens_per_second.is_none());
        assert!(metrics.memory_peak_mb.is_none());
    }

    #[test]
    fn test_outcome_debug() {
        let outcome = Outcome::Corroborated;
        let debug_str = format!("{outcome:?}");
        assert!(debug_str.contains("Corroborated"));
    }

    #[test]
    fn test_outcome_clone_eq() {
        let outcome1 = Outcome::Falsified;
        let outcome2 = outcome1;
        assert_eq!(outcome1, outcome2);
    }

    #[test]
    fn test_evidence_collector_default() {
        let collector = EvidenceCollector::default();
        assert_eq!(collector.total(), 0);
        assert_eq!(collector.pass_count(), 0);
        assert_eq!(collector.fail_count(), 0);
    }

    #[test]
    fn test_uuid_generation() {
        let uuid1 = uuid_v4();
        let uuid2 = uuid_v4();
        // UUIDs should be generated (not asserting uniqueness as they may be same in fast succession)
        assert!(!uuid1.is_empty());
        assert!(!uuid2.is_empty());
    }

    #[test]
    fn test_evidence_has_all_fields() {
        let evidence = Evidence::corroborated("F-001", test_scenario(), "output", 100);
        assert!(!evidence.id.is_empty());
        assert_eq!(evidence.gate_id, "F-001");
        assert_eq!(evidence.output, "output");
        assert!(evidence.exit_code.is_some());
        assert!(evidence.stderr.is_none());
    }

    #[test]
    fn test_evidence_serialization() {
        let evidence = Evidence::falsified("F-001", test_scenario(), "bad", "out", 100);
        let json = serde_json::to_string(&evidence).expect("serialize");
        let parsed: Evidence = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.gate_id, evidence.gate_id);
        assert_eq!(parsed.outcome, evidence.outcome);
    }
}
