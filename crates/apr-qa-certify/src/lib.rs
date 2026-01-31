//! Model certification tools and README synchronization.
//!
//! This crate provides utilities for:
//! - Parsing model certification CSV data
//! - Generating markdown tables for README
//! - Synchronizing certification status with documentation

#![forbid(unsafe_code)]

use chrono::{DateTime, Utc};
use std::fmt;
use thiserror::Error;

/// Errors that can occur during certification operations.
#[derive(Error, Debug)]
pub enum CertifyError {
    /// CSV parsing error.
    #[error("CSV parse error at line {line}: {message}")]
    CsvParse {
        /// Line number where error occurred.
        line: usize,
        /// Error message.
        message: String,
    },

    /// README marker not found.
    #[error("README marker not found: {0}")]
    MarkerNotFound(String),

    /// IO error.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Result type for certification operations.
pub type Result<T> = std::result::Result<T, CertifyError>;

/// Certification status for a model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CertificationStatus {
    /// Model passed all tests with MQS >= 850.
    Certified,
    /// Model passed with MQS >= 700 but < 850.
    Provisional,
    /// Model failed tests or MQS < 700.
    Blocked,
    /// Model not yet tested.
    #[default]
    Pending,
}

impl CertificationStatus {
    /// Parse status from string representation.
    #[must_use]
    pub fn parse(s: &str) -> Self {
        match s.to_uppercase().as_str() {
            "CERTIFIED" => Self::Certified,
            "PROVISIONAL" => Self::Provisional,
            "BLOCKED" => Self::Blocked,
            _ => Self::Pending,
        }
    }

    /// Get badge markdown for this status.
    #[must_use]
    pub const fn badge(&self) -> &'static str {
        match self {
            Self::Certified => "![certified](https://img.shields.io/badge/CERTIFIED-brightgreen)",
            Self::Provisional => "![provisional](https://img.shields.io/badge/PROVISIONAL-yellow)",
            Self::Blocked => "![blocked](https://img.shields.io/badge/BLOCKED-red)",
            Self::Pending => "![pending](https://img.shields.io/badge/PENDING-lightgray)",
        }
    }
}

impl fmt::Display for CertificationStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Certified => write!(f, "CERTIFIED"),
            Self::Provisional => write!(f, "PROVISIONAL"),
            Self::Blocked => write!(f, "BLOCKED"),
            Self::Pending => write!(f, "PENDING"),
        }
    }
}

/// Size category for models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SizeCategory {
    /// < 1B parameters.
    Tiny,
    /// 1B - 5B parameters.
    #[default]
    Small,
    /// 5B - 10B parameters.
    Medium,
    /// 10B - 30B parameters.
    Large,
    /// > 30B parameters.
    XLarge,
}

impl SizeCategory {
    /// Parse size category from string representation.
    #[must_use]
    pub fn parse(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "tiny" => Self::Tiny,
            "medium" => Self::Medium,
            "large" => Self::Large,
            "xlarge" => Self::XLarge,
            // "small" and unknown values default to Small
            _ => Self::Small,
        }
    }
}

/// Model certification record.
///
/// Contains certification data for a single model including gateway status.
/// The four gateway bools (g1-g4) are required for the certification protocol.
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone)]
pub struct ModelCertification {
    /// `HuggingFace` model ID.
    pub model_id: String,
    /// Model family (e.g., qwen-coder, llama).
    pub family: String,
    /// Parameter count (e.g., "1.5B").
    pub parameters: String,
    /// Size category.
    pub size_category: SizeCategory,
    /// Certification status.
    pub status: CertificationStatus,
    /// Model Qualification Score (0-1000).
    pub mqs_score: u16,
    /// Letter grade.
    pub grade: String,
    /// Highest passing tier.
    pub certified_tier: String,
    /// Last certification timestamp.
    pub last_certified: Option<DateTime<Utc>>,
    /// Gateway 1 (load) status.
    pub g1: bool,
    /// Gateway 2 (inference) status.
    pub g2: bool,
    /// Gateway 3 (stability) status.
    pub g3: bool,
    /// Gateway 4 (quality) status.
    pub g4: bool,
}

impl ModelCertification {
    /// Get the short model name (without org prefix).
    #[must_use]
    pub fn short_name(&self) -> &str {
        self.model_id
            .split('/')
            .next_back()
            .unwrap_or(&self.model_id)
    }

    /// Get `HuggingFace` URL for this model.
    #[must_use]
    pub fn hf_url(&self) -> String {
        format!("https://huggingface.co/{}", self.model_id)
    }

    /// Get markdown link for this model.
    #[must_use]
    pub fn markdown_link(&self) -> String {
        format!("[{}]({})", self.short_name(), self.hf_url())
    }

    /// Get gateway symbol for display.
    #[must_use]
    pub const fn gateway_symbol(passed: bool, status: CertificationStatus) -> &'static str {
        if matches!(status, CertificationStatus::Pending) {
            "-"
        } else if passed {
            "\u{2713}" // checkmark
        } else {
            "\u{2717}" // x mark
        }
    }

    /// Parse numeric parameter count for sorting.
    #[must_use]
    pub fn param_count(&self) -> f64 {
        self.parameters
            .trim_end_matches('B')
            .parse::<f64>()
            .unwrap_or(0.0)
    }
}

/// Parse CSV content into model certifications.
///
/// # Errors
///
/// Returns `CertifyError::CsvParse` if the CSV is malformed.
pub fn parse_csv(content: &str) -> Result<Vec<ModelCertification>> {
    let mut models = Vec::new();
    let mut lines = content.lines().enumerate();

    // Skip header
    let Some((_, header)) = lines.next() else {
        return Ok(models);
    };

    // Validate header
    let expected_fields = [
        "model_id",
        "family",
        "parameters",
        "size_category",
        "status",
        "mqs_score",
        "grade",
        "certified_tier",
        "last_certified",
        "g1",
        "g2",
        "g3",
        "g4",
    ];
    let header_fields: Vec<&str> = header.split(',').collect();
    if header_fields.len() < expected_fields.len() {
        return Err(CertifyError::CsvParse {
            line: 1,
            message: format!(
                "expected {} fields, got {}",
                expected_fields.len(),
                header_fields.len()
            ),
        });
    }

    for (line_num, line) in lines {
        if line.trim().is_empty() {
            continue;
        }

        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < 13 {
            return Err(CertifyError::CsvParse {
                line: line_num + 1,
                message: format!("expected 13 fields, got {}", fields.len()),
            });
        }

        let last_certified = DateTime::parse_from_rfc3339(fields[8])
            .ok()
            .map(|dt| dt.with_timezone(&Utc));

        models.push(ModelCertification {
            model_id: fields[0].to_string(),
            family: fields[1].to_string(),
            parameters: fields[2].to_string(),
            size_category: SizeCategory::parse(fields[3]),
            status: CertificationStatus::parse(fields[4]),
            mqs_score: fields[5].parse().unwrap_or(0),
            grade: fields[6].to_string(),
            certified_tier: fields[7].to_string(),
            last_certified,
            g1: fields[9].to_lowercase() == "true",
            g2: fields[10].to_lowercase() == "true",
            g3: fields[11].to_lowercase() == "true",
            g4: fields[12].to_lowercase() == "true",
        });
    }

    Ok(models)
}

/// Generate certification summary statistics.
#[must_use]
pub fn generate_summary(models: &[ModelCertification], timestamp: &str) -> String {
    let total = models.len();
    let certified = models
        .iter()
        .filter(|m| matches!(m.status, CertificationStatus::Certified))
        .count();
    let provisional = models
        .iter()
        .filter(|m| matches!(m.status, CertificationStatus::Provisional))
        .count();
    let blocked = models
        .iter()
        .filter(|m| matches!(m.status, CertificationStatus::Blocked))
        .count();
    let pending = models
        .iter()
        .filter(|m| matches!(m.status, CertificationStatus::Pending))
        .count();

    format!(
        r"**Certification Summary** (updated: {timestamp})

| Status | Count |
|--------|-------|
| Certified | {certified}/{total} |
| Provisional | {provisional}/{total} |
| Blocked | {blocked}/{total} |
| Pending | {pending}/{total} |

**Priority Family:** Qwen Coder (see [Certified Testing Spec](docs/specifications/certified-testing.md))"
    )
}

/// Generate markdown table from model certifications.
#[must_use]
pub fn generate_table(models: &[ModelCertification]) -> String {
    let mut lines = Vec::new();

    // Header
    lines.push("| Model | Family | Size | Status | MQS | Grade | G1 | G2 | G3 | G4 |".to_string());
    lines.push("|-------|--------|------|--------|-----|-------|----|----|----|----|".to_string());

    // Sort by family, then by parameter count
    let mut sorted: Vec<_> = models.iter().collect();
    sorted.sort_by(|a, b| {
        a.family.cmp(&b.family).then_with(|| {
            a.param_count()
                .partial_cmp(&b.param_count())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    });

    for m in sorted {
        let g1 = ModelCertification::gateway_symbol(m.g1, m.status);
        let g2 = ModelCertification::gateway_symbol(m.g2, m.status);
        let g3 = ModelCertification::gateway_symbol(m.g3, m.status);
        let g4 = ModelCertification::gateway_symbol(m.g4, m.status);

        lines.push(format!(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |",
            m.markdown_link(),
            m.family,
            m.parameters,
            m.status.badge(),
            m.mqs_score,
            m.grade,
            g1,
            g2,
            g3,
            g4
        ));
    }

    lines.join("\n")
}

/// Markers for README table replacement.
pub const START_MARKER: &str = "<!-- CERTIFICATION_TABLE_START -->";
/// End marker for README table.
pub const END_MARKER: &str = "<!-- CERTIFICATION_TABLE_END -->";

/// Update README content with new certification table.
///
/// # Errors
///
/// Returns `CertifyError::MarkerNotFound` if the markers are not found.
pub fn update_readme(readme: &str, table_content: &str) -> Result<String> {
    let start_idx = readme
        .find(START_MARKER)
        .ok_or_else(|| CertifyError::MarkerNotFound(START_MARKER.to_string()))?;
    let end_idx = readme
        .find(END_MARKER)
        .ok_or_else(|| CertifyError::MarkerNotFound(END_MARKER.to_string()))?;

    let before = &readme[..start_idx + START_MARKER.len()];
    let after = &readme[end_idx..];

    Ok(format!("{before}\n{table_content}\n{after}"))
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    const SAMPLE_CSV: &str = r"model_id,family,parameters,size_category,status,mqs_score,grade,certified_tier,last_certified,g1,g2,g3,g4
Qwen/Qwen2.5-Coder-0.5B-Instruct,qwen-coder,0.5B,tiny,PENDING,0,-,none,2026-01-31T00:00:00Z,false,false,false,false
Qwen/Qwen2.5-Coder-1.5B-Instruct,qwen-coder,1.5B,small,CERTIFIED,920,A,deep,2026-01-31T12:00:00Z,true,true,true,true
meta-llama/Llama-3.2-1B-Instruct,llama,1B,small,BLOCKED,450,F,smoke,2026-01-31T00:00:00Z,true,false,false,false";

    #[test]
    fn test_parse_csv_valid() {
        let models = parse_csv(SAMPLE_CSV).expect("should parse");
        assert_eq!(models.len(), 3);

        assert_eq!(models[0].model_id, "Qwen/Qwen2.5-Coder-0.5B-Instruct");
        assert_eq!(models[0].family, "qwen-coder");
        assert_eq!(models[0].parameters, "0.5B");
        assert!(matches!(models[0].status, CertificationStatus::Pending));

        assert_eq!(models[1].mqs_score, 920);
        assert!(matches!(models[1].status, CertificationStatus::Certified));
        assert!(models[1].g1);
        assert!(models[1].g2);

        assert!(matches!(models[2].status, CertificationStatus::Blocked));
        assert!(models[2].g1);
        assert!(!models[2].g2);
    }

    #[test]
    fn test_parse_csv_empty() {
        let models = parse_csv("").expect("should parse empty");
        assert!(models.is_empty());
    }

    #[test]
    fn test_parse_csv_header_only() {
        let csv = "model_id,family,parameters,size_category,status,mqs_score,grade,certified_tier,last_certified,g1,g2,g3,g4";
        let models = parse_csv(csv).expect("should parse header only");
        assert!(models.is_empty());
    }

    #[test]
    fn test_parse_csv_invalid_fields() {
        let csv = "a,b,c\n1,2,3";
        let result = parse_csv(csv);
        assert!(result.is_err());
    }

    #[test]
    fn test_certification_status_parse() {
        assert!(matches!(
            CertificationStatus::parse("CERTIFIED"),
            CertificationStatus::Certified
        ));
        assert!(matches!(
            CertificationStatus::parse("certified"),
            CertificationStatus::Certified
        ));
        assert!(matches!(
            CertificationStatus::parse("PROVISIONAL"),
            CertificationStatus::Provisional
        ));
        assert!(matches!(
            CertificationStatus::parse("BLOCKED"),
            CertificationStatus::Blocked
        ));
        assert!(matches!(
            CertificationStatus::parse("PENDING"),
            CertificationStatus::Pending
        ));
        assert!(matches!(
            CertificationStatus::parse("unknown"),
            CertificationStatus::Pending
        ));
    }

    #[test]
    fn test_certification_status_badge() {
        assert!(
            CertificationStatus::Certified
                .badge()
                .contains("brightgreen")
        );
        assert!(CertificationStatus::Provisional.badge().contains("yellow"));
        assert!(CertificationStatus::Blocked.badge().contains("red"));
        assert!(CertificationStatus::Pending.badge().contains("lightgray"));
    }

    #[test]
    fn test_model_short_name() {
        let model = ModelCertification {
            model_id: "Qwen/Qwen2.5-Coder-1.5B-Instruct".to_string(),
            family: "qwen-coder".to_string(),
            parameters: "1.5B".to_string(),
            size_category: SizeCategory::Small,
            status: CertificationStatus::Pending,
            mqs_score: 0,
            grade: "-".to_string(),
            certified_tier: "none".to_string(),
            last_certified: None,
            g1: false,
            g2: false,
            g3: false,
            g4: false,
        };
        assert_eq!(model.short_name(), "Qwen2.5-Coder-1.5B-Instruct");
    }

    #[test]
    fn test_model_hf_url() {
        let model = ModelCertification {
            model_id: "Qwen/Qwen2.5-Coder-1.5B-Instruct".to_string(),
            family: String::new(),
            parameters: String::new(),
            size_category: SizeCategory::Small,
            status: CertificationStatus::Pending,
            mqs_score: 0,
            grade: String::new(),
            certified_tier: String::new(),
            last_certified: None,
            g1: false,
            g2: false,
            g3: false,
            g4: false,
        };
        assert_eq!(
            model.hf_url(),
            "https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct"
        );
    }

    #[test]
    fn test_model_param_count() {
        let mut model = ModelCertification {
            model_id: String::new(),
            family: String::new(),
            parameters: "1.5B".to_string(),
            size_category: SizeCategory::Small,
            status: CertificationStatus::Pending,
            mqs_score: 0,
            grade: String::new(),
            certified_tier: String::new(),
            last_certified: None,
            g1: false,
            g2: false,
            g3: false,
            g4: false,
        };
        assert!((model.param_count() - 1.5).abs() < f64::EPSILON);

        model.parameters = "32B".to_string();
        assert!((model.param_count() - 32.0).abs() < f64::EPSILON);

        model.parameters = "invalid".to_string();
        assert!((model.param_count() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_gateway_symbol() {
        assert_eq!(
            ModelCertification::gateway_symbol(true, CertificationStatus::Certified),
            "\u{2713}"
        );
        assert_eq!(
            ModelCertification::gateway_symbol(false, CertificationStatus::Certified),
            "\u{2717}"
        );
        assert_eq!(
            ModelCertification::gateway_symbol(true, CertificationStatus::Pending),
            "-"
        );
        assert_eq!(
            ModelCertification::gateway_symbol(false, CertificationStatus::Pending),
            "-"
        );
    }

    #[test]
    fn test_generate_summary() {
        let models = parse_csv(SAMPLE_CSV).expect("should parse");
        let summary = generate_summary(&models, "2026-01-31 12:00 UTC");

        assert!(summary.contains("Certified | 1/3"));
        assert!(summary.contains("Blocked | 1/3"));
        assert!(summary.contains("Pending | 1/3"));
        assert!(summary.contains("2026-01-31 12:00 UTC"));
    }

    #[test]
    fn test_generate_table() {
        let models = parse_csv(SAMPLE_CSV).expect("should parse");
        let table = generate_table(&models);

        assert!(table.contains("| Model | Family |"));
        assert!(table.contains("Qwen2.5-Coder-0.5B-Instruct"));
        assert!(table.contains("qwen-coder"));
        assert!(table.contains("CERTIFIED-brightgreen"));
        assert!(table.contains("BLOCKED-red"));
    }

    #[test]
    fn test_generate_table_sorting() {
        let models = parse_csv(SAMPLE_CSV).expect("should parse");
        let table = generate_table(&models);
        let lines: Vec<&str> = table.lines().collect();

        // Should be sorted by family, then by param count
        // llama (1B) should come before qwen-coder (0.5B, 1.5B)
        let llama_idx = lines
            .iter()
            .position(|l| l.contains("Llama"))
            .expect("llama found");
        let qwen_05_idx = lines
            .iter()
            .position(|l| l.contains("0.5B"))
            .expect("qwen 0.5 found");

        assert!(
            llama_idx < qwen_05_idx,
            "llama should come before qwen-coder"
        );
    }

    #[test]
    fn test_update_readme_success() {
        let readme = r"# Title

Some content

<!-- CERTIFICATION_TABLE_START -->
old table
<!-- CERTIFICATION_TABLE_END -->

More content";

        let new_table = "new table content";
        let result = update_readme(readme, new_table).expect("should update");

        assert!(result.contains("new table content"));
        assert!(!result.contains("old table"));
        assert!(result.contains("# Title"));
        assert!(result.contains("More content"));
    }

    #[test]
    fn test_update_readme_missing_start_marker() {
        let readme = "no markers here <!-- CERTIFICATION_TABLE_END -->";
        let result = update_readme(readme, "table");
        assert!(matches!(result, Err(CertifyError::MarkerNotFound(_))));
    }

    #[test]
    fn test_update_readme_missing_end_marker() {
        let readme = "<!-- CERTIFICATION_TABLE_START --> no end marker";
        let result = update_readme(readme, "table");
        assert!(matches!(result, Err(CertifyError::MarkerNotFound(_))));
    }

    #[test]
    fn test_size_category_parse() {
        assert!(matches!(SizeCategory::parse("tiny"), SizeCategory::Tiny));
        assert!(matches!(SizeCategory::parse("SMALL"), SizeCategory::Small));
        assert!(matches!(
            SizeCategory::parse("Medium"),
            SizeCategory::Medium
        ));
        assert!(matches!(SizeCategory::parse("large"), SizeCategory::Large));
        assert!(matches!(
            SizeCategory::parse("xlarge"),
            SizeCategory::XLarge
        ));
        assert!(matches!(
            SizeCategory::parse("unknown"),
            SizeCategory::Small
        ));
    }

    #[test]
    fn test_certification_status_display() {
        assert_eq!(format!("{}", CertificationStatus::Certified), "CERTIFIED");
        assert_eq!(
            format!("{}", CertificationStatus::Provisional),
            "PROVISIONAL"
        );
        assert_eq!(format!("{}", CertificationStatus::Blocked), "BLOCKED");
        assert_eq!(format!("{}", CertificationStatus::Pending), "PENDING");
    }

    #[test]
    fn test_short_name_no_slash() {
        let model = ModelCertification {
            model_id: "model-without-org".to_string(),
            family: String::new(),
            parameters: String::new(),
            size_category: SizeCategory::Small,
            status: CertificationStatus::Pending,
            mqs_score: 0,
            grade: String::new(),
            certified_tier: String::new(),
            last_certified: None,
            g1: false,
            g2: false,
            g3: false,
            g4: false,
        };
        assert_eq!(model.short_name(), "model-without-org");
    }

    #[test]
    fn test_markdown_link() {
        let model = ModelCertification {
            model_id: "Org/Model".to_string(),
            family: String::new(),
            parameters: String::new(),
            size_category: SizeCategory::Small,
            status: CertificationStatus::Pending,
            mqs_score: 0,
            grade: String::new(),
            certified_tier: String::new(),
            last_certified: None,
            g1: false,
            g2: false,
            g3: false,
            g4: false,
        };
        assert_eq!(
            model.markdown_link(),
            "[Model](https://huggingface.co/Org/Model)"
        );
    }
}
