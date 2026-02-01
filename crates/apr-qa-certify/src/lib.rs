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
/// Contains certification data for a single model including gateway status
/// and throughput measurements per format.
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
    pub mqs_score: u32,
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
    /// Throughput in tokens/sec for GGUF format (CPU).
    pub tps_gguf_cpu: Option<f64>,
    /// Throughput in tokens/sec for GGUF format (GPU).
    pub tps_gguf_gpu: Option<f64>,
    /// Throughput in tokens/sec for APR format (CPU).
    pub tps_apr_cpu: Option<f64>,
    /// Throughput in tokens/sec for APR format (GPU).
    pub tps_apr_gpu: Option<f64>,
    /// Throughput in tokens/sec for `SafeTensors` format (CPU).
    pub tps_st_cpu: Option<f64>,
    /// Throughput in tokens/sec for `SafeTensors` format (GPU).
    pub tps_st_gpu: Option<f64>,
    /// Provenance verified (PMAT-PROV-001).
    pub provenance_verified: bool,
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
#[allow(clippy::similar_names)]
pub fn parse_csv(content: &str) -> Result<Vec<ModelCertification>> {
    let mut models = Vec::new();
    let mut lines = content.lines().enumerate();

    // Skip header
    let Some((_, header)) = lines.next() else {
        return Ok(models);
    };

    // Validate header (minimum 13 fields for backwards compatibility, 16 with tps)
    let header_fields: Vec<&str> = header.split(',').collect();
    if header_fields.len() < 13 {
        return Err(CertifyError::CsvParse {
            line: 1,
            message: format!("expected at least 13 fields, got {}", header_fields.len()),
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
                message: format!("expected at least 13 fields, got {}", fields.len()),
            });
        }

        let last_certified = DateTime::parse_from_rfc3339(fields[8])
            .ok()
            .map(|dt| dt.with_timezone(&Utc));

        // Parse optional tps fields (backwards compatible) - 6 columns for format × backend
        let tps_gguf_cpu = fields.get(13).and_then(|s| s.parse().ok());
        let tps_gguf_gpu = fields.get(14).and_then(|s| s.parse().ok());
        let tps_apr_cpu = fields.get(15).and_then(|s| s.parse().ok());
        let tps_apr_gpu = fields.get(16).and_then(|s| s.parse().ok());
        let tps_st_cpu = fields.get(17).and_then(|s| s.parse().ok());
        let tps_st_gpu = fields.get(18).and_then(|s| s.parse().ok());
        let provenance_verified = fields.get(19).is_some_and(|s| s.to_lowercase() == "true");

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
            tps_gguf_cpu,
            tps_gguf_gpu,
            tps_apr_cpu,
            tps_apr_gpu,
            tps_st_cpu,
            tps_st_gpu,
            provenance_verified,
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

    // Header with tok/s columns (format × backend = 6 columns) + provenance
    lines.push(
        "| Model | Family | Size | Status | MQS | Grade | G1-4 | Prov | GGUF CPU | GGUF GPU | APR CPU | APR GPU | ST CPU | ST GPU |"
            .to_string(),
    );
    lines.push(
        "|-------|--------|------|--------|-----|-------|------|------|----------|----------|---------|---------|--------|--------|"
            .to_string(),
    );

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
        // Combine gateways into single column (all must pass)
        let gateways = if matches!(m.status, CertificationStatus::Pending) {
            "-".to_string()
        } else if m.g1 && m.g2 && m.g3 && m.g4 {
            "\u{2713}".to_string() // checkmark
        } else {
            "\u{2717}".to_string() // x mark
        };

        // Provenance status
        let prov = if matches!(m.status, CertificationStatus::Pending) {
            "-"
        } else if m.provenance_verified {
            "\u{2713}" // checkmark
        } else {
            "\u{2717}" // x mark
        };

        // Format tok/s values (6 columns)
        let fmt = |v: Option<f64>| v.map_or_else(|| "-".to_string(), |x| format!("{x:.1}"));

        lines.push(format!(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |",
            m.markdown_link(),
            m.family,
            m.parameters,
            m.status.badge(),
            m.mqs_score,
            m.grade,
            gateways,
            prov,
            fmt(m.tps_gguf_cpu),
            fmt(m.tps_gguf_gpu),
            fmt(m.tps_apr_cpu),
            fmt(m.tps_apr_gpu),
            fmt(m.tps_st_cpu),
            fmt(m.tps_st_gpu),
        ));
    }

    lines.join("\n")
}

/// Markers for README table replacement.
pub const START_MARKER: &str = "<!-- CERTIFICATION_TABLE_START -->";
/// End marker for README table.
pub const END_MARKER: &str = "<!-- CERTIFICATION_TABLE_END -->";

/// Write certification records to CSV format.
///
/// Generates a CSV string with headers that can be written to models.csv.
#[must_use]
pub fn write_csv(models: &[ModelCertification]) -> String {
    let mut lines = Vec::new();

    // Header with 6 tps columns (format × backend) + provenance
    lines.push(
        "model_id,family,parameters,size_category,status,mqs_score,grade,certified_tier,last_certified,g1,g2,g3,g4,tps_gguf_cpu,tps_gguf_gpu,tps_apr_cpu,tps_apr_gpu,tps_st_cpu,tps_st_gpu,provenance_verified"
            .to_string(),
    );

    for m in models {
        let size_cat = match m.size_category {
            SizeCategory::Tiny => "tiny",
            SizeCategory::Small => "small",
            SizeCategory::Medium => "medium",
            SizeCategory::Large => "large",
            SizeCategory::XLarge => "xlarge",
        };
        let last_cert = m
            .last_certified
            .map_or_else(|| "2026-01-31T00:00:00Z".to_string(), |dt| dt.to_rfc3339());

        // Format tps values (empty string for None)
        let fmt = |v: Option<f64>| v.map_or(String::new(), |x| format!("{x:.1}"));

        lines.push(format!(
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            m.model_id,
            m.family,
            m.parameters,
            size_cat,
            m.status,
            m.mqs_score,
            m.grade,
            m.certified_tier,
            last_cert,
            m.g1,
            m.g2,
            m.g3,
            m.g4,
            fmt(m.tps_gguf_cpu),
            fmt(m.tps_gguf_gpu),
            fmt(m.tps_apr_cpu),
            fmt(m.tps_apr_gpu),
            fmt(m.tps_st_cpu),
            fmt(m.tps_st_gpu),
            m.provenance_verified,
        ));
    }

    lines.join("\n") + "\n"
}

/// Certification tier for tier-aware scoring.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CertificationTier {
    /// MVP tier: Tests 18 combinations (3 formats × 2 backends × 3 modalities).
    /// Pass = B grade (800 score), PROVISIONAL status.
    #[default]
    Mvp,
    /// Full tier: Complete 170-point verification matrix.
    /// Pass = A+ grade (950+ score), CERTIFIED status.
    Full,
}

/// MVP tier pass threshold (90% pass rate).
pub const MVP_PASS_THRESHOLD: f64 = 0.90;

/// MVP tier pass score (B grade).
pub const MVP_PASS_SCORE: u32 = 800;

/// Full tier pass threshold (95% on verification matrix).
pub const FULL_PASS_THRESHOLD: f64 = 0.95;

/// Full tier pass score (A+ grade).
pub const FULL_PASS_SCORE: u32 = 950;

/// Calculate certification status from MQS score.
#[must_use]
pub const fn status_from_score(mqs_score: u32, has_p0_failure: bool) -> CertificationStatus {
    if has_p0_failure {
        CertificationStatus::Blocked
    } else if mqs_score >= 850 {
        CertificationStatus::Certified
    } else if mqs_score >= 700 {
        CertificationStatus::Provisional
    } else {
        CertificationStatus::Blocked
    }
}

/// Calculate certification status for a specific tier.
///
/// # Arguments
/// * `tier` - The certification tier (MVP or Full)
/// * `pass_rate` - The pass rate from test execution (0.0 to 1.0)
/// * `has_p0_failure` - Whether any P0 (critical) test failed
#[must_use]
pub fn status_from_tier(
    tier: CertificationTier,
    pass_rate: f64,
    has_p0_failure: bool,
) -> CertificationStatus {
    if has_p0_failure {
        return CertificationStatus::Blocked;
    }

    match tier {
        CertificationTier::Mvp => {
            if pass_rate >= MVP_PASS_THRESHOLD {
                CertificationStatus::Provisional
            } else {
                CertificationStatus::Blocked
            }
        }
        CertificationTier::Full => {
            if pass_rate >= FULL_PASS_THRESHOLD {
                CertificationStatus::Certified
            } else if pass_rate >= MVP_PASS_THRESHOLD {
                CertificationStatus::Provisional
            } else {
                CertificationStatus::Blocked
            }
        }
    }
}

/// Convert pass rate to a scaled score, clamping to valid range.
#[inline]
fn scale_to_f_grade(pass_rate: f64) -> u32 {
    // Clamp pass_rate to [0.0, 1.0] and scale to [0, 699]
    let clamped = pass_rate.clamp(0.0, 1.0);
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let score = (clamped * 699.0) as u32;
    score.min(699)
}

/// Calculate MQS score for a specific tier.
///
/// # Arguments
/// * `tier` - The certification tier (MVP or Full)
/// * `pass_rate` - The pass rate from test execution (0.0 to 1.0)
/// * `has_p0_failure` - Whether any P0 (critical) test failed
#[must_use]
pub fn score_from_tier(tier: CertificationTier, pass_rate: f64, has_p0_failure: bool) -> u32 {
    if has_p0_failure {
        // Scale score based on pass rate, max 699 (F grade)
        return scale_to_f_grade(pass_rate);
    }

    match tier {
        CertificationTier::Mvp => {
            if pass_rate >= MVP_PASS_THRESHOLD {
                // MVP pass: B grade (800)
                MVP_PASS_SCORE
            } else {
                // Scale score based on pass rate, max 699 (F grade)
                scale_to_f_grade(pass_rate)
            }
        }
        CertificationTier::Full => {
            if pass_rate >= FULL_PASS_THRESHOLD {
                // Full pass: A+ grade (950+)
                let bonus = ((pass_rate - FULL_PASS_THRESHOLD) * 1000.0).clamp(0.0, 50.0);
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let bonus_u32 = bonus as u32;
                FULL_PASS_SCORE + bonus_u32
            } else if pass_rate >= MVP_PASS_THRESHOLD {
                // Between MVP and Full threshold: B to B+ range (800-899)
                let ratio =
                    (pass_rate - MVP_PASS_THRESHOLD) / (FULL_PASS_THRESHOLD - MVP_PASS_THRESHOLD);
                let bonus = (ratio * 99.0).clamp(0.0, 99.0);
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let bonus_u32 = bonus as u32;
                800 + bonus_u32
            } else {
                // Below MVP threshold: F grade
                scale_to_f_grade(pass_rate)
            }
        }
    }
}

/// Calculate letter grade from MQS score.
#[must_use]
pub const fn grade_from_score(mqs_score: u32) -> &'static str {
    match mqs_score {
        950..=1000 => "A+",
        900..=949 => "A",
        850..=899 => "B+",
        800..=849 => "B",
        700..=799 => "C",
        _ => "F",
    }
}

/// Calculate grade for a specific tier.
///
/// # Arguments
/// * `tier` - The certification tier (MVP or Full)
/// * `pass_rate` - The pass rate from test execution (0.0 to 1.0)
/// * `has_p0_failure` - Whether any P0 (critical) test failed
#[must_use]
pub fn grade_from_tier(
    tier: CertificationTier,
    pass_rate: f64,
    has_p0_failure: bool,
) -> &'static str {
    let score = score_from_tier(tier, pass_rate, has_p0_failure);
    grade_from_score(score)
}

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

    const SAMPLE_CSV: &str = r"model_id,family,parameters,size_category,status,mqs_score,grade,certified_tier,last_certified,g1,g2,g3,g4,tps_gguf_cpu,tps_gguf_gpu,tps_apr_cpu,tps_apr_gpu,tps_st_cpu,tps_st_gpu,provenance_verified
Qwen/Qwen2.5-Coder-0.5B-Instruct,qwen-coder,0.5B,tiny,PENDING,0,-,none,2026-01-31T00:00:00Z,false,false,false,false,,,,,,false,false
Qwen/Qwen2.5-Coder-1.5B-Instruct,qwen-coder,1.5B,small,CERTIFIED,920,A,deep,2026-01-31T12:00:00Z,true,true,true,true,25.5,85.2,22.3,78.1,18.1,62.5,true
meta-llama/Llama-3.2-1B-Instruct,llama,1B,small,BLOCKED,450,F,smoke,2026-01-31T00:00:00Z,true,false,false,false,12.0,45.0,,,,,false";

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
            tps_gguf_cpu: None,
            tps_gguf_gpu: None,
            tps_apr_cpu: None,
            tps_apr_gpu: None,
            tps_st_cpu: None,
            tps_st_gpu: None,
            provenance_verified: false,
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
            tps_gguf_cpu: None,
            tps_gguf_gpu: None,
            tps_apr_cpu: None,
            tps_apr_gpu: None,
            tps_st_cpu: None,
            tps_st_gpu: None,
            provenance_verified: false,
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
            tps_gguf_cpu: None,
            tps_gguf_gpu: None,
            tps_apr_cpu: None,
            tps_apr_gpu: None,
            tps_st_cpu: None,
            tps_st_gpu: None,
            provenance_verified: false,
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
            tps_gguf_cpu: None,
            tps_gguf_gpu: None,
            tps_apr_cpu: None,
            tps_apr_gpu: None,
            tps_st_cpu: None,
            tps_st_gpu: None,
            provenance_verified: false,
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
            tps_gguf_cpu: None,
            tps_gguf_gpu: None,
            tps_apr_cpu: None,
            tps_apr_gpu: None,
            tps_st_cpu: None,
            tps_st_gpu: None,
            provenance_verified: false,
        };
        assert_eq!(
            model.markdown_link(),
            "[Model](https://huggingface.co/Org/Model)"
        );
    }

    #[test]
    fn test_write_csv_roundtrip() {
        let models = parse_csv(SAMPLE_CSV).expect("should parse");
        let csv_output = write_csv(&models);

        // Parse it back
        let reparsed = parse_csv(&csv_output).expect("should reparse");
        assert_eq!(reparsed.len(), models.len());

        // Check first model
        assert_eq!(reparsed[0].model_id, models[0].model_id);
        assert_eq!(reparsed[0].family, models[0].family);
        assert_eq!(reparsed[0].mqs_score, models[0].mqs_score);
    }

    #[test]
    fn test_write_csv_has_header() {
        let models = parse_csv(SAMPLE_CSV).expect("should parse");
        let csv_output = write_csv(&models);
        assert!(csv_output.starts_with("model_id,family,"));
    }

    #[test]
    fn test_status_from_score_certified() {
        assert!(matches!(
            status_from_score(900_u32, false),
            CertificationStatus::Certified
        ));
        assert!(matches!(
            status_from_score(850_u32, false),
            CertificationStatus::Certified
        ));
    }

    #[test]
    fn test_status_from_score_provisional() {
        assert!(matches!(
            status_from_score(800_u32, false),
            CertificationStatus::Provisional
        ));
        assert!(matches!(
            status_from_score(700_u32, false),
            CertificationStatus::Provisional
        ));
    }

    #[test]
    fn test_status_from_score_blocked() {
        assert!(matches!(
            status_from_score(699_u32, false),
            CertificationStatus::Blocked
        ));
        assert!(matches!(
            status_from_score(0_u32, false),
            CertificationStatus::Blocked
        ));
    }

    #[test]
    fn test_status_from_score_p0_failure() {
        // P0 failure always results in BLOCKED regardless of score
        assert!(matches!(
            status_from_score(950_u32, true),
            CertificationStatus::Blocked
        ));
        assert!(matches!(
            status_from_score(900_u32, true),
            CertificationStatus::Blocked
        ));
    }

    #[test]
    fn test_grade_from_score() {
        assert_eq!(grade_from_score(1000_u32), "A+");
        assert_eq!(grade_from_score(950_u32), "A+");
        assert_eq!(grade_from_score(920_u32), "A");
        assert_eq!(grade_from_score(900_u32), "A");
        assert_eq!(grade_from_score(880_u32), "B+");
        assert_eq!(grade_from_score(850_u32), "B+");
        assert_eq!(grade_from_score(820_u32), "B");
        assert_eq!(grade_from_score(800_u32), "B");
        assert_eq!(grade_from_score(750_u32), "C");
        assert_eq!(grade_from_score(700_u32), "C");
        assert_eq!(grade_from_score(699_u32), "F");
        assert_eq!(grade_from_score(0_u32), "F");
    }

    #[test]
    fn test_mvp_tier_pass() {
        // MVP tier with 90%+ pass rate should get B grade (800 score)
        let status = status_from_tier(CertificationTier::Mvp, 0.95, false);
        assert!(matches!(status, CertificationStatus::Provisional));

        let score = score_from_tier(CertificationTier::Mvp, 0.95, false);
        assert_eq!(score, 800);

        let grade = grade_from_tier(CertificationTier::Mvp, 0.95, false);
        assert_eq!(grade, "B");
    }

    #[test]
    fn test_mvp_tier_exactly_90_percent() {
        // MVP tier at exactly 90% should pass
        let status = status_from_tier(CertificationTier::Mvp, 0.90, false);
        assert!(matches!(status, CertificationStatus::Provisional));

        let score = score_from_tier(CertificationTier::Mvp, 0.90, false);
        assert_eq!(score, 800);
    }

    #[test]
    fn test_mvp_tier_fail() {
        // MVP tier below 90% should fail
        let status = status_from_tier(CertificationTier::Mvp, 0.85, false);
        assert!(matches!(status, CertificationStatus::Blocked));

        let score = score_from_tier(CertificationTier::Mvp, 0.85, false);
        assert!(score < 700); // F grade
    }

    #[test]
    fn test_mvp_tier_p0_failure() {
        // MVP tier with P0 failure should always block
        let status = status_from_tier(CertificationTier::Mvp, 0.99, true);
        assert!(matches!(status, CertificationStatus::Blocked));

        let score = score_from_tier(CertificationTier::Mvp, 0.99, true);
        assert!(score < 700); // F grade even with high pass rate
    }

    #[test]
    fn test_full_tier_pass() {
        // Full tier with 95%+ should get A+ (950+ score)
        let status = status_from_tier(CertificationTier::Full, 0.98, false);
        assert!(matches!(status, CertificationStatus::Certified));

        let score = score_from_tier(CertificationTier::Full, 0.98, false);
        assert!(score >= 950);

        let grade = grade_from_tier(CertificationTier::Full, 0.98, false);
        assert_eq!(grade, "A+");
    }

    #[test]
    fn test_full_tier_provisional() {
        // Full tier between 90% and 95% should get PROVISIONAL
        let status = status_from_tier(CertificationTier::Full, 0.92, false);
        assert!(matches!(status, CertificationStatus::Provisional));

        let score = score_from_tier(CertificationTier::Full, 0.92, false);
        assert!((800..900).contains(&score)); // B to B+ range
    }

    #[test]
    fn test_full_tier_fail() {
        // Full tier below 90% should fail
        let status = status_from_tier(CertificationTier::Full, 0.85, false);
        assert!(matches!(status, CertificationStatus::Blocked));

        let score = score_from_tier(CertificationTier::Full, 0.85, false);
        assert!(score < 700);
    }

    #[test]
    fn test_certification_tier_default() {
        let tier = CertificationTier::default();
        assert!(matches!(tier, CertificationTier::Mvp));
    }

    #[test]
    fn test_parse_csv_with_empty_lines() {
        let csv = r"model_id,family,parameters,size_category,status,mqs_score,grade,certified_tier,last_certified,g1,g2,g3,g4

Qwen/Qwen2.5-Coder-0.5B-Instruct,qwen-coder,0.5B,tiny,PENDING,0,-,none,2026-01-31T00:00:00Z,false,false,false,false

";
        let models = parse_csv(csv).expect("should parse with empty lines");
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].model_id, "Qwen/Qwen2.5-Coder-0.5B-Instruct");
    }

    #[test]
    fn test_parse_csv_insufficient_fields_in_line() {
        let csv = r"model_id,family,parameters,size_category,status,mqs_score,grade,certified_tier,last_certified,g1,g2,g3,g4
only,a,few,fields";
        let result = parse_csv(csv);
        assert!(result.is_err());
        let err = result.expect_err("Should be an error");
        assert!(
            matches!(err, CertifyError::CsvParse { line: 2, .. }),
            "Error should indicate line 2"
        );
    }

    #[test]
    fn test_write_csv_all_size_categories() {
        // Test that write_csv correctly handles all size categories
        let models = vec![
            ModelCertification {
                model_id: "tiny-model".to_string(),
                family: "test".to_string(),
                parameters: "0.5B".to_string(),
                size_category: SizeCategory::Tiny,
                status: CertificationStatus::Pending,
                mqs_score: 0,
                grade: "-".to_string(),
                certified_tier: "none".to_string(),
                last_certified: None,
                g1: false,
                g2: false,
                g3: false,
                g4: false,
                tps_gguf_cpu: None,
                tps_gguf_gpu: None,
                tps_apr_cpu: None,
                tps_apr_gpu: None,
                tps_st_cpu: None,
                tps_st_gpu: None,
                provenance_verified: false,
            },
            ModelCertification {
                model_id: "medium-model".to_string(),
                family: "test".to_string(),
                parameters: "7B".to_string(),
                size_category: SizeCategory::Medium,
                status: CertificationStatus::Pending,
                mqs_score: 0,
                grade: "-".to_string(),
                certified_tier: "none".to_string(),
                last_certified: None,
                g1: false,
                g2: false,
                g3: false,
                g4: false,
                tps_gguf_cpu: None,
                tps_gguf_gpu: None,
                tps_apr_cpu: None,
                tps_apr_gpu: None,
                tps_st_cpu: None,
                tps_st_gpu: None,
                provenance_verified: false,
            },
            ModelCertification {
                model_id: "large-model".to_string(),
                family: "test".to_string(),
                parameters: "34B".to_string(),
                size_category: SizeCategory::Large,
                status: CertificationStatus::Pending,
                mqs_score: 0,
                grade: "-".to_string(),
                certified_tier: "none".to_string(),
                last_certified: None,
                g1: false,
                g2: false,
                g3: false,
                g4: false,
                tps_gguf_cpu: None,
                tps_gguf_gpu: None,
                tps_apr_cpu: None,
                tps_apr_gpu: None,
                tps_st_cpu: None,
                tps_st_gpu: None,
                provenance_verified: false,
            },
            ModelCertification {
                model_id: "xlarge-model".to_string(),
                family: "test".to_string(),
                parameters: "70B".to_string(),
                size_category: SizeCategory::XLarge,
                status: CertificationStatus::Pending,
                mqs_score: 0,
                grade: "-".to_string(),
                certified_tier: "none".to_string(),
                last_certified: None,
                g1: false,
                g2: false,
                g3: false,
                g4: false,
                tps_gguf_cpu: None,
                tps_gguf_gpu: None,
                tps_apr_cpu: None,
                tps_apr_gpu: None,
                tps_st_cpu: None,
                tps_st_gpu: None,
                provenance_verified: false,
            },
        ];

        let csv_output = write_csv(&models);
        assert!(csv_output.contains(",tiny,"));
        assert!(csv_output.contains(",medium,"));
        assert!(csv_output.contains(",large,"));
        assert!(csv_output.contains(",xlarge,"));
    }
}
