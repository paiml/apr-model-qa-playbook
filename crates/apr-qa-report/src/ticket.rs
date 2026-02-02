//! Upstream Ticket Generator
//!
//! Creates tickets for aprender repository when bugs are found.
//! Generates structured issue reports with reproduction steps.

use apr_qa_runner::{Evidence, Outcome, classify_failure};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::popperian::PopperianScore;

/// Ticket priority
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TicketPriority {
    /// Critical - blocks release
    P0,
    /// High - should fix before release
    P1,
    /// Medium - fix in next release
    P2,
    /// Low - nice to have
    P3,
}

impl std::fmt::Display for TicketPriority {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::P0 => write!(f, "P0-Critical"),
            Self::P1 => write!(f, "P1-High"),
            Self::P2 => write!(f, "P2-Medium"),
            Self::P3 => write!(f, "P3-Low"),
        }
    }
}

/// Ticket category
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TicketCategory {
    /// Bug in functionality
    Bug,
    /// Performance issue
    Performance,
    /// Crash or instability
    Crash,
    /// Compatibility issue
    Compatibility,
    /// Edge case handling
    EdgeCase,
    /// Regression from previous version
    Regression,
}

impl std::fmt::Display for TicketCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Bug => write!(f, "bug"),
            Self::Performance => write!(f, "performance"),
            Self::Crash => write!(f, "crash"),
            Self::Compatibility => write!(f, "compatibility"),
            Self::EdgeCase => write!(f, "edge-case"),
            Self::Regression => write!(f, "regression"),
        }
    }
}

/// Upstream ticket for aprender
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpstreamTicket {
    /// Ticket title
    pub title: String,
    /// Ticket body (markdown)
    pub body: String,
    /// Priority
    pub priority: TicketPriority,
    /// Category
    pub category: TicketCategory,
    /// Labels for GitHub
    pub labels: Vec<String>,
    /// Related gate ID
    pub gate_id: String,
    /// Model that triggered this
    pub model_id: String,
    /// Is this a black swan event?
    pub is_black_swan: bool,
    /// Upstream fixture path for reproduction (§3.5)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub upstream_fixture: Option<String>,
    /// Pygmy builder function name (§3.5)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pygmy_builder: Option<String>,
}

impl UpstreamTicket {
    /// Generate GitHub CLI command to create this ticket
    #[must_use]
    pub fn to_gh_command(&self, repo: &str) -> String {
        let labels = self.labels.join(",");
        format!(
            r#"gh issue create --repo {} --title "{}" --body "{}" --label "{}""#,
            repo,
            self.title.replace('"', r#"\""#),
            self.body.replace('"', r#"\""#).replace('\n', "\\n"),
            labels
        )
    }
}

/// Ticket generator
#[derive(Debug, Default)]
pub struct TicketGenerator {
    /// Repository to create tickets in
    repo: String,
    /// Minimum occurrences before creating ticket
    min_occurrences: usize,
    /// Only create tickets for black swans
    black_swans_only: bool,
}

impl TicketGenerator {
    /// Create a new ticket generator
    #[must_use]
    pub fn new(repo: impl Into<String>) -> Self {
        Self {
            repo: repo.into(),
            min_occurrences: 1,
            black_swans_only: false,
        }
    }

    /// Set minimum occurrences before creating ticket
    #[must_use]
    pub fn with_min_occurrences(mut self, min: usize) -> Self {
        self.min_occurrences = min;
        self
    }

    /// Only create tickets for black swan events
    #[must_use]
    pub fn black_swans_only(mut self) -> Self {
        self.black_swans_only = true;
        self
    }

    /// Generate tickets from evidence
    #[must_use]
    pub fn generate_from_evidence(&self, evidence: &[Evidence]) -> Vec<UpstreamTicket> {
        let mut tickets = Vec::new();

        // Group failures by gate_id
        let mut failure_groups: std::collections::HashMap<String, Vec<&Evidence>> =
            std::collections::HashMap::new();

        for e in evidence {
            if e.outcome.is_fail() {
                failure_groups.entry(e.gate_id.clone()).or_default().push(e);
            }
        }

        for (_gate_id, failures) in failure_groups {
            if failures.len() < self.min_occurrences {
                continue;
            }

            let first = failures[0];
            let is_black_swan = first.outcome == Outcome::Crashed;

            if self.black_swans_only && !is_black_swan {
                continue;
            }

            let ticket = self.create_ticket(first, failures.len(), is_black_swan);
            tickets.push(ticket);
        }

        tickets
    }

    /// Generate tickets from Popperian analysis
    #[must_use]
    pub fn generate_from_popperian(&self, popperian: &PopperianScore) -> Vec<UpstreamTicket> {
        let mut tickets = Vec::new();

        for falsification in &popperian.falsifications {
            if falsification.occurrence_count < self.min_occurrences {
                continue;
            }

            if self.black_swans_only && !falsification.is_black_swan {
                continue;
            }

            let priority = if falsification.is_black_swan {
                TicketPriority::P0
            } else if falsification.severity >= 4 {
                TicketPriority::P1
            } else if falsification.severity >= 3 {
                TicketPriority::P2
            } else {
                TicketPriority::P3
            };

            let category = self.determine_category(&falsification.gate_id);

            let title = format!(
                "[QA] {}: {}",
                falsification.gate_id, falsification.hypothesis
            );

            let body = format!(
                r#"## Summary

Automated QA testing discovered a falsification of the hypothesis: **{}**

## Details

- **Gate ID**: `{}`
- **Model**: `{}`
- **Severity**: {}/5
- **Occurrences**: {}
- **Black Swan**: {}

## Evidence

```
{}
```

## Reproduction

This issue was detected by `apr-model-qa-playbook` during automated qualification testing.

## Labels

- `{}` (priority)
- `{}` (category)
- `qa-automated`
"#,
                falsification.hypothesis,
                falsification.gate_id,
                popperian.model_id,
                falsification.severity,
                falsification.occurrence_count,
                if falsification.is_black_swan {
                    "Yes"
                } else {
                    "No"
                },
                falsification.evidence,
                priority,
                category,
            );

            let mut labels = vec![
                format!("priority:{}", priority),
                category.to_string(),
                "qa-automated".to_string(),
            ];

            if falsification.is_black_swan {
                labels.push("black-swan".to_string());
            }

            tickets.push(UpstreamTicket {
                title,
                body,
                priority,
                category,
                labels,
                gate_id: falsification.gate_id.clone(),
                model_id: popperian.model_id.clone(),
                is_black_swan: falsification.is_black_swan,
                upstream_fixture: None,
                pygmy_builder: None,
            });
        }

        tickets
    }

    /// Create a ticket from evidence
    fn create_ticket(
        &self,
        evidence: &Evidence,
        occurrence_count: usize,
        is_black_swan: bool,
    ) -> UpstreamTicket {
        let priority = self.determine_priority(evidence, is_black_swan);
        let category = self.determine_category(&evidence.gate_id);

        let title = format!(
            "[QA] {}: {} failure in {} mode",
            evidence.gate_id,
            match evidence.outcome {
                Outcome::Crashed => "Crash",
                Outcome::Falsified => "Assertion",
                Outcome::Timeout => "Timeout",
                _ => "Test",
            },
            evidence.scenario.modality
        );

        let body = format!(
            r#"## Summary

Automated QA testing discovered a failure in `apr-cli`.

## Details

- **Gate ID**: `{}`
- **Model**: `{}`
- **Modality**: `{}`
- **Backend**: `{}`
- **Format**: `{}`
- **Occurrences**: {}
- **Black Swan**: {}

## Scenario

```
Prompt: {}
Seed: {}
Temperature: {}
Max Tokens: {}
```

## Output

```
{}
```

## Error

```
{}
```

## Reproduction

```bash
{}
```

## Environment

- **Host**: `{}`
- **OS**: `{}`
- **APR Version**: `{}`

## Labels

- `{}` (priority)
- `{}` (category)
- `qa-automated`
"#,
            evidence.gate_id,
            evidence.scenario.model,
            evidence.scenario.modality,
            evidence.scenario.backend,
            evidence.scenario.format,
            occurrence_count,
            if is_black_swan { "Yes" } else { "No" },
            evidence.scenario.prompt,
            evidence.scenario.seed,
            evidence.scenario.temperature,
            evidence.scenario.max_tokens,
            evidence.output,
            evidence.stderr.as_deref().unwrap_or("N/A"),
            evidence.scenario.to_command("model.gguf"),
            evidence.host.hostname,
            evidence.host.os,
            evidence.host.apr_version,
            priority,
            category,
        );

        let mut labels = vec![
            format!("priority:{}", priority),
            category.to_string(),
            "qa-automated".to_string(),
            format!("modality:{}", evidence.scenario.modality),
            format!("backend:{}", evidence.scenario.backend),
        ];

        if is_black_swan {
            labels.push("black-swan".to_string());
        }

        UpstreamTicket {
            title,
            body,
            priority,
            category,
            labels,
            gate_id: evidence.gate_id.clone(),
            model_id: evidence.scenario.model.to_string(),
            is_black_swan,
            upstream_fixture: None,
            pygmy_builder: None,
        }
    }

    /// Determine priority from evidence
    fn determine_priority(&self, evidence: &Evidence, is_black_swan: bool) -> TicketPriority {
        if is_black_swan || evidence.outcome == Outcome::Crashed {
            TicketPriority::P0
        } else if evidence.gate_id.contains("-P0-") || evidence.gate_id.starts_with('G') {
            TicketPriority::P0
        } else if evidence.gate_id.contains("-P1-") {
            TicketPriority::P1
        } else if evidence.gate_id.contains("-P2-") {
            TicketPriority::P2
        } else {
            TicketPriority::P3
        }
    }

    /// Determine category from gate ID
    fn determine_category(&self, gate_id: &str) -> TicketCategory {
        if gate_id.contains("PERF") {
            TicketCategory::Performance
        } else if gate_id.contains("STAB") || gate_id.contains("CRASH") {
            TicketCategory::Crash
        } else if gate_id.contains("COMP") {
            TicketCategory::Compatibility
        } else if gate_id.contains("EDGE") {
            TicketCategory::EdgeCase
        } else if gate_id.contains("REGR") {
            TicketCategory::Regression
        } else {
            TicketCategory::Bug
        }
    }

    /// Get repository name
    #[must_use]
    pub fn repo(&self) -> &str {
        &self.repo
    }
}

/// Generate structured tickets from evidence using the defect-fixture map (§3.6)
///
/// Groups failures by root cause (`ConversionFailureType`), deduplicates,
/// and renders each group as a single ticket with the corresponding fixture template.
#[must_use]
pub fn generate_structured_tickets<S: ::std::hash::BuildHasher>(
    evidence: &[Evidence],
    defect_map: &HashMap<String, crate::defect_map::DefectFixtureEntry, S>,
) -> Vec<UpstreamTicket> {
    // Step 1: Filter to failures only
    let failures: Vec<&Evidence> = evidence.iter().filter(|e| e.outcome.is_fail()).collect();

    if failures.is_empty() {
        return Vec::new();
    }

    // Step 2: Classify each failure and group by root cause key
    let mut groups: HashMap<String, Vec<&Evidence>> = HashMap::new();
    for ev in &failures {
        let stderr = ev.stderr.as_deref().unwrap_or("");
        let exit_code = ev.exit_code.unwrap_or(1);
        let ft = classify_failure(stderr, exit_code);
        let key = ft.key().to_string();
        groups.entry(key).or_default().push(ev);
    }

    // Step 3: One ticket per root cause
    let mut tickets = Vec::new();
    for (key, group) in &groups {
        let first = group[0];
        let is_black_swan = first.outcome == Outcome::Crashed;

        let priority = if is_black_swan {
            TicketPriority::P0
        } else if first.gate_id.contains("-P0-") || first.gate_id.starts_with('G') {
            TicketPriority::P0
        } else {
            TicketPriority::P1
        };

        let (upstream_fixture, pygmy_builder, body) = if let Some(entry) = defect_map.get(key) {
            let mut fields = HashMap::new();
            fields.insert("model_id".to_string(), first.scenario.model.to_string());
            fields.insert(
                "exit_code".to_string(),
                format!("{}", first.exit_code.unwrap_or(1)),
            );
            fields.insert(
                "stderr".to_string(),
                first.stderr.clone().unwrap_or_default(),
            );
            fields.insert("occurrences".to_string(), group.len().to_string());

            let rendered =
                crate::defect_map::render_ticket_template(&entry.ticket_template, &fields);
            (
                Some(entry.upstream_fixture.clone()),
                Some(entry.pygmy_builder.clone()),
                rendered,
            )
        } else {
            let body = format!(
                "## Conversion Failure\n\n- **Type**: `{key}`\n- **Model**: `{}`\n- **Occurrences**: {}\n\n```\n{}\n```",
                first.scenario.model,
                group.len(),
                first.stderr.as_deref().unwrap_or("N/A"),
            );
            (None, None, body)
        };

        let title = format!(
            "[QA] {}: {} ({} occurrence{})",
            first.gate_id,
            key,
            group.len(),
            if group.len() == 1 { "" } else { "s" },
        );

        let labels = vec![
            format!("priority:{priority}"),
            "qa-automated".to_string(),
            format!("failure-type:{key}"),
        ];

        tickets.push(UpstreamTicket {
            title,
            body,
            priority,
            category: TicketCategory::Bug,
            labels,
            gate_id: first.gate_id.clone(),
            model_id: first.scenario.model.to_string(),
            is_black_swan,
            upstream_fixture,
            pygmy_builder,
        });
    }

    tickets
}

#[cfg(test)]
mod tests {
    use super::*;
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

    #[test]
    fn test_ticket_priority_display() {
        assert_eq!(TicketPriority::P0.to_string(), "P0-Critical");
        assert_eq!(TicketPriority::P1.to_string(), "P1-High");
        assert_eq!(TicketPriority::P2.to_string(), "P2-Medium");
        assert_eq!(TicketPriority::P3.to_string(), "P3-Low");
    }

    #[test]
    fn test_ticket_category_display() {
        assert_eq!(TicketCategory::Bug.to_string(), "bug");
        assert_eq!(TicketCategory::Crash.to_string(), "crash");
        assert_eq!(TicketCategory::Performance.to_string(), "performance");
    }

    #[test]
    fn test_generate_from_evidence() {
        let generator = TicketGenerator::new("paiml/aprender");
        let evidence = vec![Evidence::crashed(
            "F-QUAL-001",
            test_scenario(),
            "SIGSEGV",
            -11,
            0,
        )];

        let tickets = generator.generate_from_evidence(&evidence);

        assert_eq!(tickets.len(), 1);
        assert!(tickets[0].title.contains("Crash"));
        assert_eq!(tickets[0].priority, TicketPriority::P0);
        assert!(tickets[0].is_black_swan);
    }

    #[test]
    fn test_generate_from_popperian() {
        let generator = TicketGenerator::new("paiml/aprender");
        let popperian = PopperianScore {
            model_id: "test/model".to_string(),
            hypotheses_tested: 100,
            corroborated: 95,
            falsified: 5,
            inconclusive: 0,
            corroboration_ratio: 0.95,
            severity_weighted_score: 0.93,
            confidence_level: 0.92,
            reproducibility_index: 0.85,
            black_swan_count: 1,
            falsifications: vec![FalsificationDetail {
                gate_id: "F-STAB-001".to_string(),
                hypothesis: "Model is stable".to_string(),
                evidence: "Crash detected".to_string(),
                severity: 5,
                is_black_swan: true,
                occurrence_count: 1,
            }],
        };

        let tickets = generator.generate_from_popperian(&popperian);

        assert_eq!(tickets.len(), 1);
        assert!(tickets[0].title.contains("F-STAB-001"));
        assert_eq!(tickets[0].priority, TicketPriority::P0);
        assert!(tickets[0].is_black_swan);
    }

    #[test]
    fn test_min_occurrences_filter() {
        let generator = TicketGenerator::new("paiml/aprender").with_min_occurrences(3);
        let evidence = vec![
            Evidence::falsified("F-QUAL-001", test_scenario(), "wrong", "5", 100),
            Evidence::falsified("F-QUAL-001", test_scenario(), "wrong", "5", 100),
        ];

        let tickets = generator.generate_from_evidence(&evidence);

        // Should be empty because we need 3 occurrences
        assert!(tickets.is_empty());
    }

    #[test]
    fn test_black_swans_only_filter() {
        let generator = TicketGenerator::new("paiml/aprender").black_swans_only();
        let evidence = vec![Evidence::falsified(
            "F-QUAL-001",
            test_scenario(),
            "wrong",
            "5",
            100,
        )];

        let tickets = generator.generate_from_evidence(&evidence);

        // Should be empty because it's not a black swan
        assert!(tickets.is_empty());
    }

    #[test]
    fn test_gh_command_generation() {
        let ticket = UpstreamTicket {
            title: "Test ticket".to_string(),
            body: "Test body\nLine 2".to_string(),
            priority: TicketPriority::P1,
            category: TicketCategory::Bug,
            labels: vec!["bug".to_string(), "qa-automated".to_string()],
            gate_id: "F-TEST-001".to_string(),
            model_id: "test/model".to_string(),
            is_black_swan: false,
            upstream_fixture: None,
            pygmy_builder: None,
        };

        let cmd = ticket.to_gh_command("paiml/aprender");

        assert!(cmd.contains("gh issue create"));
        assert!(cmd.contains("paiml/aprender"));
        assert!(cmd.contains("Test ticket"));
    }

    #[test]
    fn test_category_determination() {
        let generator = TicketGenerator::new("test");

        assert_eq!(
            generator.determine_category("F-PERF-001"),
            TicketCategory::Performance
        );
        assert_eq!(
            generator.determine_category("F-STAB-001"),
            TicketCategory::Crash
        );
        assert_eq!(
            generator.determine_category("F-COMP-001"),
            TicketCategory::Compatibility
        );
        assert_eq!(
            generator.determine_category("F-EDGE-001"),
            TicketCategory::EdgeCase
        );
        assert_eq!(
            generator.determine_category("F-REGR-001"),
            TicketCategory::Regression
        );
        assert_eq!(
            generator.determine_category("F-QUAL-001"),
            TicketCategory::Bug
        );
    }

    #[test]
    fn test_category_crash_detection() {
        let generator = TicketGenerator::new("test");
        assert_eq!(
            generator.determine_category("F-CRASH-001"),
            TicketCategory::Crash
        );
    }

    #[test]
    fn test_generator_repo() {
        let generator = TicketGenerator::new("owner/repo");
        assert_eq!(generator.repo(), "owner/repo");
    }

    #[test]
    fn test_priority_determination() {
        let generator = TicketGenerator::new("test");

        // Crash is always P0
        let crash_evidence = Evidence::crashed("F-QUAL-001", test_scenario(), "err", -1, 0);
        assert_eq!(
            generator.determine_priority(&crash_evidence, false),
            TicketPriority::P0
        );

        // Black swan is P0
        let regular_evidence = Evidence::falsified("F-QUAL-001", test_scenario(), "bad", "5", 100);
        assert_eq!(
            generator.determine_priority(&regular_evidence, true),
            TicketPriority::P0
        );

        // P0 gate ID
        let p0_evidence = Evidence::falsified("F-QUAL-P0-001", test_scenario(), "bad", "5", 100);
        assert_eq!(
            generator.determine_priority(&p0_evidence, false),
            TicketPriority::P0
        );

        // P1 gate ID
        let p1_evidence = Evidence::falsified("F-QUAL-P1-001", test_scenario(), "bad", "5", 100);
        assert_eq!(
            generator.determine_priority(&p1_evidence, false),
            TicketPriority::P1
        );

        // P2 gate ID
        let p2_evidence = Evidence::falsified("F-QUAL-P2-001", test_scenario(), "bad", "5", 100);
        assert_eq!(
            generator.determine_priority(&p2_evidence, false),
            TicketPriority::P2
        );

        // Default is P3
        let default_evidence = Evidence::falsified("F-QUAL-001", test_scenario(), "bad", "5", 100);
        assert_eq!(
            generator.determine_priority(&default_evidence, false),
            TicketPriority::P3
        );
    }

    #[test]
    fn test_gateway_gate_is_p0() {
        let generator = TicketGenerator::new("test");
        let evidence = Evidence::falsified("G1-LOAD", test_scenario(), "failed", "", 100);
        assert_eq!(
            generator.determine_priority(&evidence, false),
            TicketPriority::P0
        );
    }

    #[test]
    fn test_ticket_category_eq() {
        assert_eq!(TicketCategory::Bug, TicketCategory::Bug);
        assert_ne!(TicketCategory::Bug, TicketCategory::Crash);
    }

    #[test]
    fn test_ticket_priority_eq() {
        assert_eq!(TicketPriority::P0, TicketPriority::P0);
        assert_ne!(TicketPriority::P0, TicketPriority::P1);
    }

    #[test]
    fn test_upstream_ticket_clone() {
        let ticket = UpstreamTicket {
            title: "Test".to_string(),
            body: "Body".to_string(),
            priority: TicketPriority::P1,
            category: TicketCategory::Bug,
            labels: vec!["label".to_string()],
            gate_id: "F-001".to_string(),
            model_id: "test/model".to_string(),
            is_black_swan: false,
            upstream_fixture: None,
            pygmy_builder: None,
        };
        let cloned = ticket.clone();
        assert_eq!(cloned.title, ticket.title);
    }

    #[test]
    fn test_generate_from_evidence_with_timeout() {
        let generator = TicketGenerator::new("paiml/aprender");
        let evidence = vec![Evidence::timeout("F-PERF-001", test_scenario(), 30000)];

        let tickets = generator.generate_from_evidence(&evidence);

        assert_eq!(tickets.len(), 1);
        assert!(tickets[0].title.contains("Timeout"));
    }

    #[test]
    fn test_generate_from_evidence_falsified() {
        let generator = TicketGenerator::new("paiml/aprender");
        let evidence = vec![Evidence::falsified(
            "F-QUAL-001",
            test_scenario(),
            "Wrong answer",
            "5",
            100,
        )];

        let tickets = generator.generate_from_evidence(&evidence);

        assert_eq!(tickets.len(), 1);
        assert!(tickets[0].title.contains("Assertion"));
    }

    #[test]
    fn test_generate_deduplication() {
        let generator = TicketGenerator::new("test");
        let evidence = vec![
            Evidence::falsified("F-QUAL-001", test_scenario(), "err1", "out", 100),
            Evidence::falsified("F-QUAL-001", test_scenario(), "err2", "out", 100),
            Evidence::falsified("F-QUAL-001", test_scenario(), "err3", "out", 100),
        ];

        let tickets = generator.generate_from_evidence(&evidence);

        // Should be deduplicated to 1 ticket
        assert_eq!(tickets.len(), 1);
    }

    #[test]
    fn test_ticket_labels_include_modality() {
        let generator = TicketGenerator::new("test");
        let evidence = vec![Evidence::crashed("F-001", test_scenario(), "err", -1, 0)];

        let tickets = generator.generate_from_evidence(&evidence);

        assert!(tickets[0].labels.iter().any(|l| l.contains("modality:")));
        assert!(tickets[0].labels.iter().any(|l| l.contains("backend:")));
    }

    #[test]
    fn test_black_swan_label_added() {
        let generator = TicketGenerator::new("test");
        let evidence = vec![Evidence::crashed(
            "F-001",
            test_scenario(),
            "SIGSEGV",
            -11,
            0,
        )];

        let tickets = generator.generate_from_evidence(&evidence);

        assert!(tickets[0].is_black_swan);
        assert!(tickets[0].labels.contains(&"black-swan".to_string()));
    }

    fn falsified_with_stderr(gate_id: &str, stderr: &str) -> Evidence {
        let mut ev = Evidence::falsified(gate_id, test_scenario(), "failure", "N/A", 100);
        ev.stderr = Some(stderr.to_string());
        ev
    }

    #[test]
    fn test_structured_tickets_same_cause_dedup() {
        let defect_map = crate::defect_map::load_defect_fixture_map().expect("load map");

        // 12 failures with the same stderr pattern → all classify as same root cause
        let evidence: Vec<Evidence> = (0..12)
            .map(|i| {
                falsified_with_stderr(
                    "F-CONV-001",
                    &format!("tensor name mismatch: layer.{i}.weight"),
                )
            })
            .collect();

        let tickets = generate_structured_tickets(&evidence, &defect_map);

        // Should be 1 ticket (12 same-cause failures → 1 grouped ticket)
        assert_eq!(tickets.len(), 1);
        assert!(tickets[0].title.contains("12 occurrences"));
    }

    #[test]
    fn test_structured_tickets_two_causes() {
        let defect_map = crate::defect_map::load_defect_fixture_map().expect("load map");

        let mut evidence = Vec::new();
        // 3 tensor name mismatches
        for _ in 0..3 {
            evidence.push(falsified_with_stderr(
                "F-CONV-001",
                "tensor name mismatch: layer.0.weight",
            ));
        }
        // 2 missing artifact failures
        for _ in 0..2 {
            evidence.push(falsified_with_stderr(
                "F-CONV-002",
                "file not found: model.safetensors",
            ));
        }

        let tickets = generate_structured_tickets(&evidence, &defect_map);

        // Should be 2 tickets (2 different root causes)
        assert_eq!(tickets.len(), 2);
    }

    #[test]
    fn test_structured_tickets_fixture_in_body() {
        let defect_map = crate::defect_map::load_defect_fixture_map().expect("load map");

        let evidence = vec![falsified_with_stderr(
            "F-CONV-001",
            "tensor name mismatch: layer.0.weight",
        )];

        let tickets = generate_structured_tickets(&evidence, &defect_map);

        assert_eq!(tickets.len(), 1);
        assert!(tickets[0].upstream_fixture.is_some());
        assert!(tickets[0].pygmy_builder.is_some());
        assert_eq!(
            tickets[0].upstream_fixture.as_deref(),
            Some("fixtures/tensor_name_mismatch.py")
        );
    }

    #[test]
    fn test_structured_tickets_no_failures() {
        let defect_map = crate::defect_map::load_defect_fixture_map().expect("load map");

        let evidence = vec![Evidence::corroborated(
            "F-CONV-001",
            test_scenario(),
            "4",
            100,
        )];

        let tickets = generate_structured_tickets(&evidence, &defect_map);
        assert!(tickets.is_empty());
    }

    #[test]
    fn test_structured_tickets_unknown_cause_no_fixture() {
        let defect_map = crate::defect_map::load_defect_fixture_map().expect("load map");

        // Stderr that doesn't match any known pattern
        let evidence = vec![Evidence::falsified(
            "F-CONV-001",
            test_scenario(),
            "some unknown error xyz",
            "N/A",
            100,
        )];

        let tickets = generate_structured_tickets(&evidence, &defect_map);

        assert_eq!(tickets.len(), 1);
        // Unknown cause → no fixture mapping
        assert!(tickets[0].upstream_fixture.is_none());
        assert!(tickets[0].pygmy_builder.is_none());
    }

    #[test]
    fn test_structured_tickets_labels() {
        let defect_map = crate::defect_map::load_defect_fixture_map().expect("load map");

        let evidence = vec![Evidence::falsified(
            "F-CONV-001",
            test_scenario(),
            "tensor name mismatch: layer.0.weight",
            "N/A",
            100,
        )];

        let tickets = generate_structured_tickets(&evidence, &defect_map);

        assert!(!tickets.is_empty());
        assert!(
            tickets[0]
                .labels
                .iter()
                .any(|l| l.starts_with("failure-type:"))
        );
        assert!(tickets[0].labels.contains(&"qa-automated".to_string()));
    }
}
