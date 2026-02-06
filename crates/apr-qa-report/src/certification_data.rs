//! Certification Data for Oracle Integration (PMAT-260)
//!
//! This module provides the data structures and CSV parsing for the certification
//! lookup table consumed by aprender's `apr oracle` CLI command.
//!
//! # Theoretical Foundation
//!
//! This implementation follows:
//! - **Toyota Production System (Ohno, 1988)**: Jidoka - automatic stop on malformed data
//! - **Poka-Yoke (Shingo, 1986)**: Schema validation prevents invalid certification states
//! - **Popperian Falsification (Popper, 1959)**: Round-trip integrity tests verify correctness
//!
//! # CSV Schema
//!
//! The `models.csv` file uses this schema:
//! ```csv
//! model_id,family,parameters,size_category,status,mqs_score,grade,certified_tier,last_certified,g1,g2,g3,g4,tps_gguf_cpu,tps_gguf_gpu,tps_apr_cpu,tps_apr_gpu,tps_st_cpu,tps_st_gpu,provenance_verified
//! ```

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::error::{Error, Result};

/// Certification status for a model.
///
/// Status definitions follow the specification:
/// - **CERTIFIED**: MQS >= 800, all gateway gates passed, tier requirements met
/// - **BLOCKED**: MQS < 800 or gateway gate failure, cannot be used in production
/// - **PENDING**: No certification run completed, awaiting testing
/// - **UNTESTED**: Legacy status for models never tested
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ModelStatus {
    /// MQS >= 800, all gateways passed
    Certified,
    /// MQS < 800 or gateway failure
    Blocked,
    /// Awaiting certification run
    #[default]
    Pending,
    /// Never tested (legacy)
    Untested,
}

impl std::fmt::Display for ModelStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Certified => write!(f, "CERTIFIED"),
            Self::Blocked => write!(f, "BLOCKED"),
            Self::Pending => write!(f, "PENDING"),
            Self::Untested => write!(f, "UNTESTED"),
        }
    }
}

impl std::str::FromStr for ModelStatus {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self> {
        match s.to_uppercase().as_str() {
            "CERTIFIED" => Ok(Self::Certified),
            "BLOCKED" => Ok(Self::Blocked),
            "PENDING" => Ok(Self::Pending),
            "UNTESTED" => Ok(Self::Untested),
            other => Err(Error::Validation(format!("Invalid status: {other}"))),
        }
    }
}

/// Size category for resource-aware scheduling.
///
/// Matches the `SizeCategory` enum in `apr-qa-runner::playbook`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum SizeCategory {
    /// < 1B params, 4 workers
    #[default]
    Tiny,
    /// 1-2B params, 4 workers
    Small,
    /// 2-7B params, 2 workers
    Medium,
    /// 7-14B params, 1 worker
    Large,
    /// 14-32B params, 1 worker
    Xlarge,
    /// > 32B params, 1 worker
    Huge,
}

impl std::fmt::Display for SizeCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Tiny => write!(f, "tiny"),
            Self::Small => write!(f, "small"),
            Self::Medium => write!(f, "medium"),
            Self::Large => write!(f, "large"),
            Self::Xlarge => write!(f, "xlarge"),
            Self::Huge => write!(f, "huge"),
        }
    }
}

impl std::str::FromStr for SizeCategory {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "tiny" => Ok(Self::Tiny),
            "small" => Ok(Self::Small),
            "medium" => Ok(Self::Medium),
            "large" => Ok(Self::Large),
            "xlarge" => Ok(Self::Xlarge),
            "huge" => Ok(Self::Huge),
            other => Err(Error::Validation(format!("Invalid size category: {other}"))),
        }
    }
}

/// A single row from the certification lookup table (models.csv).
///
/// This struct represents the complete certification state for a model variant,
/// including MQS score, gateway results, and performance metrics.
///
/// The boolean fields (g1-g4, provenance_verified) match the CSV schema
/// and represent gateway pass/fail state directly from test results.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(clippy::struct_excessive_bools)]
pub struct CertificationRow {
    /// HuggingFace model ID (e.g., "Qwen/Qwen2.5-Coder-0.5B-Instruct")
    pub model_id: String,

    /// Model family (e.g., "qwen-coder", "llama", "mistral")
    pub family: String,

    /// Parameter count string (e.g., "0.5B", "1.5B", "7B")
    pub parameters: String,

    /// Size category for resource scheduling
    pub size_category: SizeCategory,

    /// Certification status
    pub status: ModelStatus,

    /// Model Qualification Score (0-1000)
    pub mqs_score: u32,

    /// Letter grade (A, B, C, D, F, or "-" for ungraded)
    pub grade: String,

    /// Highest certified tier (quick, smoke, mvp, full, or "none")
    pub certified_tier: String,

    /// Last certification timestamp (ISO8601)
    pub last_certified: DateTime<Utc>,

    // Gateway results (G1-G4)
    /// G1: Model loads successfully
    pub g1: bool,
    /// G2: Basic inference works
    pub g2: bool,
    /// G3: No crashes or panics
    pub g3: bool,
    /// G4: Output is not garbage
    pub g4: bool,

    // Performance metrics (tokens per second)
    /// GGUF format, CPU backend
    pub tps_gguf_cpu: Option<f64>,
    /// GGUF format, GPU backend
    pub tps_gguf_gpu: Option<f64>,
    /// APR format, CPU backend
    pub tps_apr_cpu: Option<f64>,
    /// APR format, GPU backend
    pub tps_apr_gpu: Option<f64>,
    /// SafeTensors format, CPU backend
    pub tps_st_cpu: Option<f64>,
    /// SafeTensors format, GPU backend
    pub tps_st_gpu: Option<f64>,

    /// Whether model provenance has been verified
    pub provenance_verified: bool,
}

impl Default for CertificationRow {
    fn default() -> Self {
        Self {
            model_id: String::new(),
            family: String::new(),
            parameters: String::new(),
            size_category: SizeCategory::default(),
            status: ModelStatus::default(),
            mqs_score: 0,
            grade: "-".to_string(),
            certified_tier: "none".to_string(),
            last_certified: Utc::now(),
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
        }
    }
}

impl CertificationRow {
    /// Create a new certification row for a model.
    #[must_use]
    pub fn new(model_id: impl Into<String>, family: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
            family: family.into(),
            ..Default::default()
        }
    }

    /// Check if all gateway checks passed.
    #[must_use]
    pub const fn all_gateways_passed(&self) -> bool {
        self.g1 && self.g2 && self.g3 && self.g4
    }

    /// Derive status from MQS score and gateway results.
    ///
    /// Follows the specification:
    /// - CERTIFIED: MQS >= 800 AND all gateways passed
    /// - BLOCKED: otherwise
    #[must_use]
    pub fn derive_status(&self) -> ModelStatus {
        if self.mqs_score >= 800 && self.all_gateways_passed() {
            ModelStatus::Certified
        } else if self.mqs_score == 0 && !self.g1 {
            ModelStatus::Pending
        } else {
            ModelStatus::Blocked
        }
    }

    /// Derive grade from MQS score.
    ///
    /// Grade thresholds:
    /// - A: 900-1000
    /// - B: 800-899
    /// - C: 600-799
    /// - D: 400-599
    /// - F: 0-399
    #[must_use]
    pub fn derive_grade(&self) -> String {
        match self.mqs_score {
            900..=1000 => "A".to_string(),
            800..=899 => "B".to_string(),
            600..=799 => "C".to_string(),
            400..=599 => "D".to_string(),
            0..=399 => "F".to_string(),
            _ => "-".to_string(),
        }
    }
}

/// Read certification rows from a CSV file.
///
/// # Errors
///
/// Returns an error if:
/// - The file cannot be read
/// - The CSV is malformed
/// - A row contains invalid data
pub fn read_models_csv<P: AsRef<Path>>(path: P) -> Result<Vec<CertificationRow>> {
    let file = std::fs::File::open(path.as_ref()).map_err(|e| {
        Error::Io(format!(
            "Failed to open models.csv at {}: {e}",
            path.as_ref().display()
        ))
    })?;

    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .flexible(true)
        .from_reader(file);

    let mut rows = Vec::new();

    for (idx, result) in reader.records().enumerate() {
        let record =
            result.map_err(|e| Error::Validation(format!("CSV parse error at row {idx}: {e}")))?;

        let row = parse_csv_record(&record, idx)?;
        rows.push(row);
    }

    Ok(rows)
}

/// Parse a single CSV record into a CertificationRow.
fn parse_csv_record(record: &csv::StringRecord, idx: usize) -> Result<CertificationRow> {
    // Helper for getting field with context
    let get_field = |i: usize, name: &str| -> Result<&str> {
        record
            .get(i)
            .ok_or_else(|| Error::Validation(format!("Missing field '{name}' at row {idx}")))
    };

    let model_id = get_field(0, "model_id")?.to_string();
    let family = get_field(1, "family")?.to_string();
    let parameters = get_field(2, "parameters")?.to_string();
    let size_category: SizeCategory = get_field(3, "size_category")?.parse()?;
    let status: ModelStatus = get_field(4, "status")?.parse()?;
    let mqs_score: u32 = get_field(5, "mqs_score")?
        .parse()
        .map_err(|e| Error::Validation(format!("Invalid mqs_score at row {idx}: {e}")))?;
    let grade = get_field(6, "grade")?.to_string();
    let certified_tier = get_field(7, "certified_tier")?.to_string();

    let last_certified = get_field(8, "last_certified")?;
    let last_certified: DateTime<Utc> = DateTime::parse_from_rfc3339(last_certified)
        .map_err(|e| Error::Validation(format!("Invalid timestamp at row {idx}: {e}")))?
        .with_timezone(&Utc);

    let parse_bool = |i: usize, name: &str| -> Result<bool> {
        match get_field(i, name)?.to_lowercase().as_str() {
            "true" | "1" | "yes" => Ok(true),
            "false" | "0" | "no" | "" => Ok(false),
            other => Err(Error::Validation(format!(
                "Invalid boolean '{other}' for {name} at row {idx}"
            ))),
        }
    };

    let parse_optional_f64 = |i: usize| -> Option<f64> {
        record.get(i).and_then(|s| {
            let s = s.trim();
            if s.is_empty() { None } else { s.parse().ok() }
        })
    };

    Ok(CertificationRow {
        model_id,
        family,
        parameters,
        size_category,
        status,
        mqs_score,
        grade,
        certified_tier,
        last_certified,
        g1: parse_bool(9, "g1")?,
        g2: parse_bool(10, "g2")?,
        g3: parse_bool(11, "g3")?,
        g4: parse_bool(12, "g4")?,
        tps_gguf_cpu: parse_optional_f64(13),
        tps_gguf_gpu: parse_optional_f64(14),
        tps_apr_cpu: parse_optional_f64(15),
        tps_apr_gpu: parse_optional_f64(16),
        tps_st_cpu: parse_optional_f64(17),
        tps_st_gpu: parse_optional_f64(18),
        provenance_verified: parse_bool(19, "provenance_verified")?,
    })
}

/// Write certification rows to a CSV file.
///
/// # Errors
///
/// Returns an error if the file cannot be written.
pub fn write_models_csv<P: AsRef<Path>>(rows: &[CertificationRow], path: P) -> Result<()> {
    let file = std::fs::File::create(path.as_ref()).map_err(|e| {
        Error::Io(format!(
            "Failed to create models.csv at {}: {e}",
            path.as_ref().display()
        ))
    })?;

    let mut writer = csv::Writer::from_writer(file);

    // Write header
    writer
        .write_record([
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
            "tps_gguf_cpu",
            "tps_gguf_gpu",
            "tps_apr_cpu",
            "tps_apr_gpu",
            "tps_st_cpu",
            "tps_st_gpu",
            "provenance_verified",
        ])
        .map_err(|e| Error::Io(format!("Failed to write CSV header: {e}")))?;

    // Write rows
    for row in rows {
        let format_optional_f64 =
            |opt: Option<f64>| -> String { opt.map_or_else(String::new, |v| format!("{v:.1}")) };

        writer
            .write_record([
                &row.model_id,
                &row.family,
                &row.parameters,
                &row.size_category.to_string(),
                &row.status.to_string(),
                &row.mqs_score.to_string(),
                &row.grade,
                &row.certified_tier,
                &row.last_certified.to_rfc3339(),
                &row.g1.to_string(),
                &row.g2.to_string(),
                &row.g3.to_string(),
                &row.g4.to_string(),
                &format_optional_f64(row.tps_gguf_cpu),
                &format_optional_f64(row.tps_gguf_gpu),
                &format_optional_f64(row.tps_apr_cpu),
                &format_optional_f64(row.tps_apr_gpu),
                &format_optional_f64(row.tps_st_cpu),
                &format_optional_f64(row.tps_st_gpu),
                &row.provenance_verified.to_string(),
            ])
            .map_err(|e| Error::Io(format!("Failed to write CSV row: {e}")))?;
    }

    writer
        .flush()
        .map_err(|e| Error::Io(format!("Failed to flush CSV writer: {e}")))?;

    Ok(())
}

/// Lookup a certification row by model ID.
///
/// Returns `None` if the model is not found.
#[must_use]
pub fn lookup_model<'a>(
    rows: &'a [CertificationRow],
    model_id: &str,
) -> Option<&'a CertificationRow> {
    rows.iter().find(|r| r.model_id == model_id)
}

/// Lookup certification rows by family.
#[must_use]
pub fn lookup_family<'a>(rows: &'a [CertificationRow], family: &str) -> Vec<&'a CertificationRow> {
    rows.iter().filter(|r| r.family == family).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    const TEST_CSV: &str = r#"model_id,family,parameters,size_category,status,mqs_score,grade,certified_tier,last_certified,g1,g2,g3,g4,tps_gguf_cpu,tps_gguf_gpu,tps_apr_cpu,tps_apr_gpu,tps_st_cpu,tps_st_gpu,provenance_verified
Qwen/Qwen2.5-Coder-0.5B-Instruct,qwen-coder,0.5B,tiny,BLOCKED,246,F,quick,2026-02-04T13:28:18.663298968+00:00,true,true,true,true,,,,,,,false
Qwen/Qwen2.5-Coder-1.5B-Instruct,qwen-coder,1.5B,small,BLOCKED,415,-,none,2026-02-03T15:50:04.803811188+00:00,true,true,true,true,17.9,129.8,16.2,0.6,2.9,23.8,false
meta-llama/Llama-3.2-1B-Instruct,llama,1B,small,PENDING,0,-,none,2026-01-31T00:00:00+00:00,false,false,false,false,,,,,,,false
"#;

    // FALSIFY-CERT-001: Round-trip integrity
    //
    // Falsification hypothesis: "CSV round-trip corrupts data"
    // If read(write(rows)) != rows, implementation is broken.
    #[test]
    fn test_falsify_cert_001_roundtrip_integrity() {
        let temp_file = NamedTempFile::new().expect("temp file");

        // Write test CSV to temp file
        std::fs::write(temp_file.path(), TEST_CSV).expect("write");

        // Read original
        let original = read_models_csv(temp_file.path()).expect("read original");
        assert_eq!(original.len(), 3, "Expected 3 rows");

        // Write to new temp file
        let temp_file2 = NamedTempFile::new().expect("temp file 2");
        write_models_csv(&original, temp_file2.path()).expect("write");

        // Read back
        let roundtrip = read_models_csv(temp_file2.path()).expect("read roundtrip");
        assert_eq!(roundtrip.len(), original.len(), "Row count mismatch");

        // Verify each row
        for (orig, rt) in original.iter().zip(roundtrip.iter()) {
            assert_eq!(orig.model_id, rt.model_id, "model_id mismatch");
            assert_eq!(orig.family, rt.family, "family mismatch");
            assert_eq!(orig.parameters, rt.parameters, "parameters mismatch");
            assert_eq!(
                orig.size_category, rt.size_category,
                "size_category mismatch"
            );
            assert_eq!(orig.status, rt.status, "status mismatch");
            assert_eq!(orig.mqs_score, rt.mqs_score, "mqs_score mismatch");
            assert_eq!(orig.grade, rt.grade, "grade mismatch");
            assert_eq!(
                orig.certified_tier, rt.certified_tier,
                "certified_tier mismatch"
            );
            assert_eq!(orig.g1, rt.g1, "g1 mismatch");
            assert_eq!(orig.g2, rt.g2, "g2 mismatch");
            assert_eq!(orig.g3, rt.g3, "g3 mismatch");
            assert_eq!(orig.g4, rt.g4, "g4 mismatch");
            assert_eq!(
                orig.provenance_verified, rt.provenance_verified,
                "provenance_verified mismatch"
            );
        }
    }

    #[test]
    fn test_model_status_from_str() {
        assert_eq!(
            "CERTIFIED".parse::<ModelStatus>().unwrap(),
            ModelStatus::Certified
        );
        assert_eq!(
            "BLOCKED".parse::<ModelStatus>().unwrap(),
            ModelStatus::Blocked
        );
        assert_eq!(
            "PENDING".parse::<ModelStatus>().unwrap(),
            ModelStatus::Pending
        );
        assert_eq!(
            "UNTESTED".parse::<ModelStatus>().unwrap(),
            ModelStatus::Untested
        );
        assert_eq!(
            "certified".parse::<ModelStatus>().unwrap(),
            ModelStatus::Certified
        );
        assert!("INVALID".parse::<ModelStatus>().is_err());
    }

    #[test]
    fn test_model_status_display() {
        assert_eq!(format!("{}", ModelStatus::Certified), "CERTIFIED");
        assert_eq!(format!("{}", ModelStatus::Blocked), "BLOCKED");
        assert_eq!(format!("{}", ModelStatus::Pending), "PENDING");
        assert_eq!(format!("{}", ModelStatus::Untested), "UNTESTED");
    }

    #[test]
    fn test_size_category_from_str() {
        assert_eq!("tiny".parse::<SizeCategory>().unwrap(), SizeCategory::Tiny);
        assert_eq!(
            "SMALL".parse::<SizeCategory>().unwrap(),
            SizeCategory::Small
        );
        assert_eq!(
            "Medium".parse::<SizeCategory>().unwrap(),
            SizeCategory::Medium
        );
        assert_eq!(
            "large".parse::<SizeCategory>().unwrap(),
            SizeCategory::Large
        );
        assert_eq!(
            "xlarge".parse::<SizeCategory>().unwrap(),
            SizeCategory::Xlarge
        );
        assert_eq!("huge".parse::<SizeCategory>().unwrap(), SizeCategory::Huge);
        assert!("invalid".parse::<SizeCategory>().is_err());
    }

    #[test]
    fn test_size_category_display() {
        assert_eq!(format!("{}", SizeCategory::Tiny), "tiny");
        assert_eq!(format!("{}", SizeCategory::Small), "small");
        assert_eq!(format!("{}", SizeCategory::Medium), "medium");
        assert_eq!(format!("{}", SizeCategory::Large), "large");
        assert_eq!(format!("{}", SizeCategory::Xlarge), "xlarge");
        assert_eq!(format!("{}", SizeCategory::Huge), "huge");
    }

    #[test]
    fn test_certification_row_default() {
        let row = CertificationRow::default();
        assert!(row.model_id.is_empty());
        assert_eq!(row.status, ModelStatus::Pending);
        assert_eq!(row.mqs_score, 0);
        assert!(!row.g1);
    }

    #[test]
    fn test_certification_row_new() {
        let row = CertificationRow::new("test/model", "test-family");
        assert_eq!(row.model_id, "test/model");
        assert_eq!(row.family, "test-family");
    }

    #[test]
    fn test_all_gateways_passed() {
        // Default has no gateways passed
        let row = CertificationRow::default();
        assert!(!row.all_gateways_passed());

        // All gateways passed
        let row = CertificationRow {
            g1: true,
            g2: true,
            g3: true,
            g4: true,
            ..Default::default()
        };
        assert!(row.all_gateways_passed());

        // One gateway failed
        let row = CertificationRow {
            g1: true,
            g2: true,
            g3: false,
            g4: true,
            ..Default::default()
        };
        assert!(!row.all_gateways_passed());
    }

    #[test]
    fn test_derive_status() {
        // Test CERTIFIED: MQS >= 800 and all gateways passed
        let row = CertificationRow {
            g1: true,
            g2: true,
            g3: true,
            g4: true,
            mqs_score: 850,
            ..Default::default()
        };
        assert_eq!(row.derive_status(), ModelStatus::Certified);

        // Test BLOCKED: MQS < 800
        let row = CertificationRow {
            g1: true,
            g2: true,
            g3: true,
            g4: true,
            mqs_score: 799,
            ..Default::default()
        };
        assert_eq!(row.derive_status(), ModelStatus::Blocked);

        // Test BLOCKED: gateway failure
        let row = CertificationRow {
            g1: true,
            g2: true,
            g3: false,
            g4: true,
            mqs_score: 900,
            ..Default::default()
        };
        assert_eq!(row.derive_status(), ModelStatus::Blocked);

        // Test PENDING: never tested
        let row = CertificationRow {
            g1: false,
            mqs_score: 0,
            ..Default::default()
        };
        assert_eq!(row.derive_status(), ModelStatus::Pending);
    }

    #[test]
    fn test_derive_grade() {
        let row_a = CertificationRow {
            mqs_score: 950,
            ..Default::default()
        };
        assert_eq!(row_a.derive_grade(), "A");

        let row_b = CertificationRow {
            mqs_score: 850,
            ..Default::default()
        };
        assert_eq!(row_b.derive_grade(), "B");

        let row_c = CertificationRow {
            mqs_score: 700,
            ..Default::default()
        };
        assert_eq!(row_c.derive_grade(), "C");

        let row_d = CertificationRow {
            mqs_score: 500,
            ..Default::default()
        };
        assert_eq!(row_d.derive_grade(), "D");

        let row_f = CertificationRow {
            mqs_score: 200,
            ..Default::default()
        };
        assert_eq!(row_f.derive_grade(), "F");
    }

    #[test]
    fn test_lookup_model() {
        let rows = vec![
            CertificationRow::new("test/model-1", "family-a"),
            CertificationRow::new("test/model-2", "family-b"),
            CertificationRow::new("test/model-3", "family-a"),
        ];

        let found = lookup_model(&rows, "test/model-2");
        assert!(found.is_some());
        assert_eq!(found.unwrap().family, "family-b");

        let not_found = lookup_model(&rows, "nonexistent");
        assert!(not_found.is_none());
    }

    #[test]
    fn test_lookup_family() {
        let rows = vec![
            CertificationRow::new("test/model-1", "family-a"),
            CertificationRow::new("test/model-2", "family-b"),
            CertificationRow::new("test/model-3", "family-a"),
        ];

        let family_a = lookup_family(&rows, "family-a");
        assert_eq!(family_a.len(), 2);

        let family_b = lookup_family(&rows, "family-b");
        assert_eq!(family_b.len(), 1);

        let family_c = lookup_family(&rows, "family-c");
        assert!(family_c.is_empty());
    }

    #[test]
    fn test_read_missing_file() {
        let result = read_models_csv("/nonexistent/path/models.csv");
        assert!(result.is_err());
    }

    #[test]
    fn test_read_malformed_csv() {
        let temp_file = NamedTempFile::new().expect("temp file");
        std::fs::write(
            temp_file.path(),
            "model_id,family\ntest,test,extra,fields,here",
        )
        .expect("write");

        // Should handle flexible field count gracefully
        let result = read_models_csv(temp_file.path());
        // This may error due to missing required fields
        assert!(result.is_err());
    }

    #[test]
    fn test_optional_tps_fields() {
        let temp_file = NamedTempFile::new().expect("temp file");
        std::fs::write(temp_file.path(), TEST_CSV).expect("write");

        let cert_rows = read_models_csv(temp_file.path()).expect("read");

        // First row has no TPS values
        let first_row = &cert_rows[0];
        assert!(first_row.tps_gguf_cpu.is_none());
        assert!(first_row.tps_gguf_gpu.is_none());

        // Second row has TPS values
        let second_row = &cert_rows[1];
        assert!(second_row.tps_gguf_cpu.is_some());
        assert!((second_row.tps_gguf_cpu.unwrap() - 17.9).abs() < 0.1);
    }

    #[test]
    fn test_write_models_csv_creates_file() {
        let temp_file = NamedTempFile::new().expect("temp file");

        let rows = vec![CertificationRow {
            model_id: "test/model".to_string(),
            family: "test-family".to_string(),
            parameters: "1B".to_string(),
            size_category: SizeCategory::Small,
            status: ModelStatus::Blocked,
            mqs_score: 500,
            grade: "D".to_string(),
            certified_tier: "mvp".to_string(),
            g1: true,
            g2: true,
            g3: false,
            g4: true,
            tps_gguf_cpu: Some(10.5),
            tps_gguf_gpu: Some(100.0),
            tps_apr_cpu: None,
            tps_apr_gpu: None,
            tps_st_cpu: None,
            tps_st_gpu: None,
            provenance_verified: true,
            ..Default::default()
        }];

        write_models_csv(&rows, temp_file.path()).expect("write");

        // Verify file exists and can be read back
        let read_back = read_models_csv(temp_file.path()).expect("read");
        assert_eq!(read_back.len(), 1);
        assert_eq!(read_back[0].model_id, "test/model");
        assert_eq!(read_back[0].tps_gguf_cpu.unwrap(), 10.5);
        assert!(read_back[0].provenance_verified);
    }

    #[test]
    fn test_write_to_nonexistent_dir() {
        let result = write_models_csv(&[], "/nonexistent/dir/models.csv");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("Failed to create"));
    }

    #[test]
    fn test_model_status_serde() {
        let status = ModelStatus::Certified;
        let json = serde_json::to_string(&status).expect("serialize");
        assert_eq!(json, "\"CERTIFIED\"");

        let deserialized: ModelStatus = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deserialized, ModelStatus::Certified);
    }

    #[test]
    fn test_size_category_serde() {
        let size = SizeCategory::Medium;
        let json = serde_json::to_string(&size).expect("serialize");
        assert_eq!(json, "\"medium\"");

        let deserialized: SizeCategory = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deserialized, SizeCategory::Medium);
    }

    #[test]
    fn test_certification_row_serde() {
        let row = CertificationRow {
            model_id: "test/model".to_string(),
            family: "test".to_string(),
            status: ModelStatus::Certified,
            mqs_score: 850,
            ..Default::default()
        };

        let json = serde_json::to_string(&row).expect("serialize");
        assert!(json.contains("\"model_id\":\"test/model\""));
        assert!(json.contains("\"status\":\"CERTIFIED\""));

        let deserialized: CertificationRow = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deserialized.model_id, "test/model");
        assert_eq!(deserialized.status, ModelStatus::Certified);
    }

    #[test]
    fn test_invalid_status_parse() {
        let result = "GARBAGE".parse::<ModelStatus>();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Invalid status"));
    }

    #[test]
    fn test_invalid_size_category_parse() {
        let result = "massive".parse::<SizeCategory>();
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Invalid size category")
        );
    }

    #[test]
    fn test_model_status_default() {
        let status = ModelStatus::default();
        assert_eq!(status, ModelStatus::Pending);
    }

    #[test]
    fn test_size_category_default() {
        let size = SizeCategory::default();
        assert_eq!(size, SizeCategory::Tiny);
    }

    // ── FALSIFY-CERT-002: Status derivation from MQS score ────────────────────
    //
    // Prediction: status is deterministically derived from mqs_score and g1-g4 gateways.
    // Per Popper (1959), this test attempts to falsify the status derivation algorithm.

    #[test]
    fn test_falsify_cert_002_status_derivation() {
        // All gateways passed, high score -> CERTIFIED
        let certified = CertificationRow {
            mqs_score: 850,
            g1: true,
            g2: true,
            g3: true,
            g4: true,
            ..CertificationRow::default()
        };
        assert_eq!(
            certified.derive_status(),
            ModelStatus::Certified,
            "All gateways passed + score >= 800 should be CERTIFIED"
        );

        // All gateways passed, low score -> BLOCKED
        let blocked_low = CertificationRow {
            mqs_score: 500,
            g1: true,
            g2: true,
            g3: true,
            g4: true,
            ..CertificationRow::default()
        };
        assert_eq!(
            blocked_low.derive_status(),
            ModelStatus::Blocked,
            "All gateways passed + score < 800 should be BLOCKED"
        );

        // Gateway G3 failed, high score -> BLOCKED
        let blocked_gw = CertificationRow {
            mqs_score: 950,
            g1: true,
            g2: true,
            g3: false, // Gateway failure
            g4: true,
            ..CertificationRow::default()
        };
        assert_eq!(
            blocked_gw.derive_status(),
            ModelStatus::Blocked,
            "Gateway failed should always be BLOCKED"
        );

        // Score 0 with g1=false -> PENDING (never tested)
        let pending = CertificationRow {
            mqs_score: 0,
            g1: false,
            g2: false,
            g3: false,
            g4: false,
            ..CertificationRow::default()
        };
        assert_eq!(
            pending.derive_status(),
            ModelStatus::Pending,
            "Score 0 with g1=false should be PENDING (not yet tested)"
        );
    }

    // ── FALSIFY-CERT-003: Grade derivation from MQS score ─────────────────────
    //
    // Prediction: grade is deterministically derived from mqs_score using fixed thresholds.
    // Per Popper (1959), this test attempts to falsify the grade derivation algorithm.
    //
    // Grade thresholds (from derive_grade):
    // A: 900-1000
    // B: 800-899
    // C: 600-799
    // D: 400-599
    // F: 0-399

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
}
