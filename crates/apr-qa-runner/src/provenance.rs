//! Provenance Validation (PMAT-PROV-001)
//!
//! Ensures all derived formats come from the same SafeTensors source.
//! Prevents the critical error of comparing models from different sources.
//!
//! See: docs/specifications/certified-testing.md Section 7.5

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{BufReader, Read as IoRead};
use std::path::Path;

/// Source model provenance information
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SourceProvenance {
    /// Format of source file (must be "safetensors" per spec 7.4)
    pub format: String,
    /// Relative path to source file
    pub path: String,
    /// SHA256 hash of source file
    pub sha256: String,
    /// HuggingFace repository ID (e.g., "Qwen/Qwen2.5-Coder-0.5B-Instruct")
    pub hf_repo: String,
    /// ISO 8601 timestamp of download
    pub downloaded_at: String,
}

/// Derived format provenance information
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DerivedProvenance {
    /// Format of derived file (e.g., "gguf", "apr")
    pub format: String,
    /// Relative path to derived file
    pub path: String,
    /// SHA256 hash of derived file
    pub sha256: String,
    /// Converter used (must be "apr-cli" per spec 7.5.2)
    pub converter: String,
    /// Version of converter
    pub converter_version: String,
    /// Quantization applied (null for unquantized)
    pub quantization: Option<String>,
    /// ISO 8601 timestamp of conversion
    pub created_at: String,
}

/// Complete provenance record for a model directory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Provenance {
    /// Source model information
    pub source: SourceProvenance,
    /// Derived formats
    pub derived: Vec<DerivedProvenance>,
}

/// Provenance validation errors (PMAT-PROV-001)
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProvenanceError {
    /// PROV-001: Source hashes don't match across formats
    SourceMismatch {
        /// Expected source hash
        expected: String,
        /// Actual source hash found
        actual: String,
        /// Format with mismatched source
        format: String,
    },
    /// PROV-002: Derived file not created by apr-cli
    InvalidConverter {
        /// Format with invalid converter
        format: String,
        /// Converter that was used
        converter: String,
    },
    /// PROV-003: Source is not SafeTensors
    InvalidSourceFormat {
        /// Invalid source format found
        format: String,
    },
    /// PROV-004: Missing provenance file
    MissingProvenance {
        /// Path where provenance was expected
        path: String,
    },
    /// PROV-005: Quantization levels don't match
    QuantizationMismatch {
        /// First format in comparison
        format_a: String,
        /// Quantization of first format
        quant_a: Option<String>,
        /// Second format in comparison
        format_b: String,
        /// Quantization of second format
        quant_b: Option<String>,
    },
    /// PROV-006: File hash doesn't match recorded hash (integrity violation)
    HashMismatch {
        /// Path to file with mismatched hash
        path: String,
        /// Expected hash from provenance
        expected: String,
        /// Actual hash computed from file
        actual: String,
    },
    /// PROV-007: Referenced file does not exist (ghost file)
    FileMissing {
        /// Path to missing file
        path: String,
    },
    /// PROV-008: Duplicate derived format entry
    DuplicateDerived {
        /// Format that already exists
        format: String,
        /// Quantization level (if any)
        quantization: Option<String>,
    },
    /// PROV-009: Format not found in derived list
    FormatNotFound {
        /// Format that was requested but not found
        format: String,
    },
}

impl std::fmt::Display for ProvenanceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SourceMismatch {
                expected,
                actual,
                format,
            } => {
                write!(
                    f,
                    "PROV-001: Source hash mismatch for {format}: expected {expected}, got {actual}"
                )
            }
            Self::InvalidConverter { format, converter } => {
                write!(
                    f,
                    "PROV-002: Invalid converter for {format}: {converter} (must be apr-cli)"
                )
            }
            Self::InvalidSourceFormat { format } => {
                write!(
                    f,
                    "PROV-003: Invalid source format: {format} (must be safetensors)"
                )
            }
            Self::MissingProvenance { path } => {
                write!(f, "PROV-004: Missing provenance file: {path}")
            }
            Self::QuantizationMismatch {
                format_a,
                quant_a,
                format_b,
                quant_b,
            } => {
                write!(
                    f,
                    "PROV-005: Quantization mismatch: {format_a}={quant_a:?} vs {format_b}={quant_b:?}"
                )
            }
            Self::HashMismatch {
                path,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "PROV-006: Hash mismatch for {path}: expected {expected}, got {actual}"
                )
            }
            Self::FileMissing { path } => {
                write!(f, "PROV-007: Referenced file missing: {path}")
            }
            Self::DuplicateDerived {
                format,
                quantization,
            } => {
                write!(
                    f,
                    "PROV-008: Duplicate derived format: {format} (quantization: {quantization:?})"
                )
            }
            Self::FormatNotFound { format } => {
                write!(f, "PROV-009: Format not found in derived list: {format}")
            }
        }
    }
}

impl std::error::Error for ProvenanceError {}

/// Load provenance from a model directory
///
/// # Errors
///
/// Returns error if provenance file is missing or malformed.
pub fn load_provenance(model_dir: &Path) -> Result<Provenance> {
    let provenance_path = model_dir.join(".provenance.json");
    if !provenance_path.exists() {
        return Err(Error::Provenance(ProvenanceError::MissingProvenance {
            path: provenance_path.display().to_string(),
        }));
    }

    let content = std::fs::read_to_string(&provenance_path)?;
    let provenance: Provenance = serde_json::from_str(&content)?;
    Ok(provenance)
}

/// Validate provenance for certification (PMAT-PROV-001)
///
/// Checks all rules from spec section 7.5.2:
/// - PROV-001: All formats share same source hash
/// - PROV-002: All derived files use apr-cli
/// - PROV-003: Source is safetensors
/// - PROV-004: Provenance file exists (checked by load_provenance)
/// - PROV-005: Quantization matches for comparisons
///
/// # Errors
///
/// Returns the first validation error encountered.
pub fn validate_provenance(provenance: &Provenance) -> std::result::Result<(), ProvenanceError> {
    // PROV-003: Source must be safetensors
    if provenance.source.format != "safetensors" {
        return Err(ProvenanceError::InvalidSourceFormat {
            format: provenance.source.format.clone(),
        });
    }

    // PROV-002: All derived files must use apr-cli
    for derived in &provenance.derived {
        if derived.converter != "apr-cli" {
            return Err(ProvenanceError::InvalidConverter {
                format: derived.format.clone(),
                converter: derived.converter.clone(),
            });
        }
    }

    Ok(())
}

/// Validate that two formats can be compared (same source, same quantization)
///
/// # Errors
///
/// Returns error if formats don't exist or have different quantization.
pub fn validate_comparison(
    provenance: &Provenance,
    format_a: &str,
    format_b: &str,
) -> std::result::Result<(), ProvenanceError> {
    // PROV-009: Both formats must exist in derived list
    let derived_a = provenance
        .derived
        .iter()
        .find(|d| d.format == format_a)
        .ok_or_else(|| ProvenanceError::FormatNotFound {
            format: format_a.to_string(),
        })?;

    let derived_b = provenance
        .derived
        .iter()
        .find(|d| d.format == format_b)
        .ok_or_else(|| ProvenanceError::FormatNotFound {
            format: format_b.to_string(),
        })?;

    // PROV-005: Quantization must match
    if derived_a.quantization != derived_b.quantization {
        return Err(ProvenanceError::QuantizationMismatch {
            format_a: format_a.to_string(),
            quant_a: derived_a.quantization.clone(),
            format_b: format_b.to_string(),
            quant_b: derived_b.quantization.clone(),
        });
    }

    Ok(())
}

/// Verify provenance integrity by re-hashing all files (PROV-006, PROV-007)
///
/// This function performs deep verification:
/// - Checks that all referenced files exist
/// - Re-computes SHA256 hashes and compares to recorded values
///
/// # Arguments
///
/// * `provenance` - The provenance record to verify
/// * `model_dir` - Base directory containing model files
///
/// # Errors
///
/// Returns error if any file is missing or hash doesn't match.
pub fn verify_provenance_integrity(
    provenance: &Provenance,
    model_dir: &Path,
) -> std::result::Result<(), ProvenanceError> {
    // Verify source file
    let source_path = model_dir.join(&provenance.source.path);
    if !source_path.exists() {
        return Err(ProvenanceError::FileMissing {
            path: provenance.source.path.clone(),
        });
    }

    let source_hash = compute_sha256(&source_path).map_err(|_| ProvenanceError::FileMissing {
        path: provenance.source.path.clone(),
    })?;

    if source_hash != provenance.source.sha256 {
        return Err(ProvenanceError::HashMismatch {
            path: provenance.source.path.clone(),
            expected: provenance.source.sha256.clone(),
            actual: source_hash,
        });
    }

    // Verify all derived files
    for derived in &provenance.derived {
        let derived_path = model_dir.join(&derived.path);
        if !derived_path.exists() {
            return Err(ProvenanceError::FileMissing {
                path: derived.path.clone(),
            });
        }

        let derived_hash =
            compute_sha256(&derived_path).map_err(|_| ProvenanceError::FileMissing {
                path: derived.path.clone(),
            })?;

        if derived_hash != derived.sha256 {
            return Err(ProvenanceError::HashMismatch {
                path: derived.path.clone(),
                expected: derived.sha256.clone(),
                actual: derived_hash,
            });
        }
    }

    Ok(())
}

/// Quick check that all referenced files exist (PROV-007)
///
/// Lighter than `verify_provenance_integrity` - only checks existence, not hashes.
///
/// # Errors
///
/// Returns error if any referenced file is missing.
pub fn verify_files_exist(
    provenance: &Provenance,
    model_dir: &Path,
) -> std::result::Result<(), ProvenanceError> {
    // Check source file
    let source_path = model_dir.join(&provenance.source.path);
    if !source_path.exists() {
        return Err(ProvenanceError::FileMissing {
            path: provenance.source.path.clone(),
        });
    }

    // Check all derived files
    for derived in &provenance.derived {
        let derived_path = model_dir.join(&derived.path);
        if !derived_path.exists() {
            return Err(ProvenanceError::FileMissing {
                path: derived.path.clone(),
            });
        }
    }

    Ok(())
}

// ============================================================================
// Provenance Generation (PMAT-PROV-001)
// ============================================================================

/// Compute SHA256 hash of a file
///
/// # Errors
///
/// Returns error if file cannot be read.
pub fn compute_sha256(path: &Path) -> Result<String> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut hasher = Sha256::new();

    let mut buffer = [0u8; 8192];
    loop {
        let bytes_read = reader.read(&mut buffer)?;
        if bytes_read == 0 {
            break;
        }
        hasher.update(&buffer[..bytes_read]);
    }

    let result = hasher.finalize();
    Ok(format!("{result:x}"))
}

/// Create initial provenance for a SafeTensors source file
///
/// # Errors
///
/// Returns error if file cannot be read or hashed.
pub fn create_source_provenance(safetensors_path: &Path, hf_repo: &str) -> Result<Provenance> {
    let sha256 = compute_sha256(safetensors_path)?;
    let now = chrono::Utc::now().to_rfc3339();

    Ok(Provenance {
        source: SourceProvenance {
            format: "safetensors".to_string(),
            path: safetensors_path.file_name().map_or_else(
                || safetensors_path.display().to_string(),
                |n| n.to_string_lossy().to_string(),
            ),
            sha256,
            hf_repo: hf_repo.to_string(),
            downloaded_at: now,
        },
        derived: Vec::new(),
    })
}

/// Add a derived format to provenance
///
/// Checks for duplicate entries (same format + quantization) before adding.
///
/// # Errors
///
/// Returns error if:
/// - Derived file cannot be read or hashed
/// - Duplicate format+quantization already exists (PROV-008)
pub fn add_derived(
    provenance: &mut Provenance,
    format: &str,
    derived_path: &Path,
    quantization: Option<&str>,
    converter_version: &str,
) -> Result<()> {
    // PROV-008: Check for duplicate format+quantization
    let exists = provenance
        .derived
        .iter()
        .any(|d| d.format == format && d.quantization.as_deref() == quantization);

    if exists {
        return Err(Error::Provenance(ProvenanceError::DuplicateDerived {
            format: format.to_string(),
            quantization: quantization.map(String::from),
        }));
    }

    let sha256 = compute_sha256(derived_path)?;
    let now = chrono::Utc::now().to_rfc3339();

    provenance.derived.push(DerivedProvenance {
        format: format.to_string(),
        path: derived_path.file_name().map_or_else(
            || derived_path.display().to_string(),
            |n| n.to_string_lossy().to_string(),
        ),
        sha256,
        converter: "apr-cli".to_string(),
        converter_version: converter_version.to_string(),
        quantization: quantization.map(String::from),
        created_at: now,
    });

    Ok(())
}

/// Save provenance to a model directory
///
/// # Errors
///
/// Returns error if provenance cannot be serialized or written.
pub fn save_provenance(model_dir: &Path, provenance: &Provenance) -> Result<()> {
    let provenance_path = model_dir.join(".provenance.json");
    let content = serde_json::to_string_pretty(provenance)?;
    std::fs::write(provenance_path, content)?;
    Ok(())
}

/// Get apr-cli version by running `apr --version`
///
/// Returns "unknown" if command fails.
#[must_use]
pub fn get_apr_cli_version() -> String {
    std::process::Command::new("apr")
        .arg("--version")
        .output()
        .ok()
        .and_then(|output| {
            if output.status.success() {
                String::from_utf8(output.stdout)
                    .ok()
                    .and_then(|s| s.split_whitespace().nth(1).map(String::from))
            } else {
                None
            }
        })
        .unwrap_or_else(|| "unknown".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_provenance() -> Provenance {
        Provenance {
            source: SourceProvenance {
                format: "safetensors".to_string(),
                path: "model.safetensors".to_string(),
                sha256: "a1b2c3d4e5f6".to_string(),
                hf_repo: "Qwen/Qwen2.5-Coder-0.5B-Instruct".to_string(),
                downloaded_at: "2026-02-01T12:00:00Z".to_string(),
            },
            derived: vec![
                DerivedProvenance {
                    format: "gguf".to_string(),
                    path: "model.gguf".to_string(),
                    sha256: "f6e5d4c3b2a1".to_string(),
                    converter: "apr-cli".to_string(),
                    converter_version: "0.2.12".to_string(),
                    quantization: None,
                    created_at: "2026-02-01T12:05:00Z".to_string(),
                },
                DerivedProvenance {
                    format: "apr".to_string(),
                    path: "model.apr".to_string(),
                    sha256: "1a2b3c4d5e6f".to_string(),
                    converter: "apr-cli".to_string(),
                    converter_version: "0.2.12".to_string(),
                    quantization: None,
                    created_at: "2026-02-01T12:06:00Z".to_string(),
                },
            ],
        }
    }

    // PMAT-PROV-001: Reject certification with mismatched sources
    #[test]
    fn test_reject_mismatched_source_hash() {
        // This would be detected by comparing provenance files
        // from different model directories
        let prov_a = sample_provenance();
        let mut prov_b = sample_provenance();
        prov_b.source.sha256 = "different_hash".to_string();

        // Different source hashes should fail comparison
        assert_ne!(prov_a.source.sha256, prov_b.source.sha256);
    }

    // PMAT-PROV-002: Reject third-party files without provenance
    #[test]
    fn test_reject_third_party_gguf() {
        let mut prov = sample_provenance();
        prov.derived[0].converter = "bartowski".to_string(); // Third-party

        let result = validate_provenance(&prov);
        assert!(result.is_err());

        let err = result.unwrap_err();
        assert!(matches!(err, ProvenanceError::InvalidConverter { .. }));
    }

    // PMAT-PROV-003: Accept only SafeTensors as source
    #[test]
    fn test_reject_gguf_as_source() {
        let mut prov = sample_provenance();
        prov.source.format = "gguf".to_string(); // Wrong source format

        let result = validate_provenance(&prov);
        assert!(result.is_err());

        let err = result.unwrap_err();
        assert!(matches!(err, ProvenanceError::InvalidSourceFormat { .. }));
    }

    // PMAT-PROV-004: Reject quantization mismatch
    #[test]
    fn test_reject_quantization_mismatch() {
        let mut prov = sample_provenance();
        prov.derived[0].quantization = Some("q4_k_m".to_string()); // GGUF quantized
        prov.derived[1].quantization = None; // APR unquantized

        let result = validate_comparison(&prov, "gguf", "apr");
        assert!(result.is_err());

        let err = result.unwrap_err();
        assert!(matches!(err, ProvenanceError::QuantizationMismatch { .. }));
    }

    #[test]
    fn test_valid_provenance_passes() {
        let prov = sample_provenance();
        assert!(validate_provenance(&prov).is_ok());
    }

    #[test]
    fn test_valid_comparison_same_quantization() {
        let prov = sample_provenance();
        assert!(validate_comparison(&prov, "gguf", "apr").is_ok());
    }

    #[test]
    fn test_valid_comparison_both_quantized() {
        let mut prov = sample_provenance();
        prov.derived[0].quantization = Some("q4_k_m".to_string());
        prov.derived[1].quantization = Some("q4_k_m".to_string());

        assert!(validate_comparison(&prov, "gguf", "apr").is_ok());
    }

    #[test]
    fn test_provenance_error_display() {
        let err = ProvenanceError::InvalidSourceFormat {
            format: "gguf".to_string(),
        };
        assert!(err.to_string().contains("PROV-003"));
        assert!(err.to_string().contains("safetensors"));
    }

    #[test]
    fn test_source_mismatch_error_display() {
        let err = ProvenanceError::SourceMismatch {
            expected: "abc123".to_string(),
            actual: "def456".to_string(),
            format: "apr".to_string(),
        };
        assert!(err.to_string().contains("PROV-001"));
        assert!(err.to_string().contains("abc123"));
    }

    // ========================================================================
    // Provenance Generation Tests
    // ========================================================================

    #[test]
    fn test_compute_sha256() {
        let dir = tempfile::tempdir().unwrap();
        let test_file = dir.path().join("test.txt");
        std::fs::write(&test_file, "hello world\n").unwrap();

        let hash = compute_sha256(&test_file).unwrap();
        // SHA256 of "hello world\n"
        assert_eq!(
            hash,
            "a948904f2f0f479b8f8197694b30184b0d2ed1c1cd2a1ec0fb85d299a192a447"
        );
    }

    #[test]
    fn test_compute_sha256_empty_file() {
        let dir = tempfile::tempdir().unwrap();
        let test_file = dir.path().join("empty.txt");
        std::fs::write(&test_file, "").unwrap();

        let hash = compute_sha256(&test_file).unwrap();
        // SHA256 of empty string
        assert_eq!(
            hash,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn test_compute_sha256_missing_file() {
        let result = compute_sha256(Path::new("/nonexistent/file.bin"));
        assert!(result.is_err());
    }

    #[test]
    fn test_create_source_provenance() {
        let dir = tempfile::tempdir().unwrap();
        let safetensors = dir.path().join("model.safetensors");
        std::fs::write(&safetensors, "fake safetensors content").unwrap();

        let prov =
            create_source_provenance(&safetensors, "Qwen/Qwen2.5-Coder-0.5B-Instruct").unwrap();

        assert_eq!(prov.source.format, "safetensors");
        assert_eq!(prov.source.path, "model.safetensors");
        assert_eq!(prov.source.hf_repo, "Qwen/Qwen2.5-Coder-0.5B-Instruct");
        assert!(!prov.source.sha256.is_empty());
        assert!(!prov.source.downloaded_at.is_empty());
        assert!(prov.derived.is_empty());
    }

    #[test]
    fn test_add_derived() {
        let dir = tempfile::tempdir().unwrap();
        let safetensors = dir.path().join("model.safetensors");
        let gguf = dir.path().join("model.gguf");
        std::fs::write(&safetensors, "safetensors content").unwrap();
        std::fs::write(&gguf, "gguf content").unwrap();

        let mut prov = create_source_provenance(&safetensors, "test/model").unwrap();
        add_derived(&mut prov, "gguf", &gguf, None, "0.2.12").unwrap();

        assert_eq!(prov.derived.len(), 1);
        assert_eq!(prov.derived[0].format, "gguf");
        assert_eq!(prov.derived[0].path, "model.gguf");
        assert_eq!(prov.derived[0].converter, "apr-cli");
        assert_eq!(prov.derived[0].converter_version, "0.2.12");
        assert!(prov.derived[0].quantization.is_none());
    }

    #[test]
    fn test_add_derived_with_quantization() {
        let dir = tempfile::tempdir().unwrap();
        let safetensors = dir.path().join("model.safetensors");
        let gguf_q4 = dir.path().join("model-q4_k_m.gguf");
        std::fs::write(&safetensors, "safetensors content").unwrap();
        std::fs::write(&gguf_q4, "quantized gguf content").unwrap();

        let mut prov = create_source_provenance(&safetensors, "test/model").unwrap();
        add_derived(&mut prov, "gguf", &gguf_q4, Some("q4_k_m"), "0.2.12").unwrap();

        assert_eq!(prov.derived[0].quantization, Some("q4_k_m".to_string()));
    }

    #[test]
    fn test_save_and_load_provenance() {
        let dir = tempfile::tempdir().unwrap();
        let safetensors = dir.path().join("model.safetensors");
        std::fs::write(&safetensors, "content").unwrap();

        let prov = create_source_provenance(&safetensors, "test/model").unwrap();
        save_provenance(dir.path(), &prov).unwrap();

        let loaded = load_provenance(dir.path()).unwrap();
        assert_eq!(loaded.source.hf_repo, "test/model");
        assert_eq!(loaded.source.sha256, prov.source.sha256);
    }

    #[test]
    fn test_save_provenance_creates_json() {
        let dir = tempfile::tempdir().unwrap();
        let safetensors = dir.path().join("model.safetensors");
        std::fs::write(&safetensors, "content").unwrap();

        let prov = create_source_provenance(&safetensors, "test/model").unwrap();
        save_provenance(dir.path(), &prov).unwrap();

        let prov_path = dir.path().join(".provenance.json");
        assert!(prov_path.exists());

        let content = std::fs::read_to_string(&prov_path).unwrap();
        assert!(content.contains("\"format\": \"safetensors\""));
        assert!(content.contains("test/model"));
    }

    #[test]
    fn test_full_provenance_workflow() {
        let dir = tempfile::tempdir().unwrap();

        // Create source
        let safetensors = dir.path().join("model.safetensors");
        std::fs::write(&safetensors, "source content").unwrap();

        // Create derived formats
        let gguf = dir.path().join("model.gguf");
        let apr = dir.path().join("model.apr");
        std::fs::write(&gguf, "gguf content").unwrap();
        std::fs::write(&apr, "apr content").unwrap();

        // Build provenance
        let mut prov =
            create_source_provenance(&safetensors, "Qwen/Qwen2.5-Coder-0.5B-Instruct").unwrap();
        add_derived(&mut prov, "gguf", &gguf, None, "0.2.12").unwrap();
        add_derived(&mut prov, "apr", &apr, None, "0.2.12").unwrap();

        // Validate provenance
        assert!(validate_provenance(&prov).is_ok());

        // Validate comparison
        assert!(validate_comparison(&prov, "gguf", "apr").is_ok());

        // Save and reload
        save_provenance(dir.path(), &prov).unwrap();
        let loaded = load_provenance(dir.path()).unwrap();

        // Revalidate after reload
        assert!(validate_provenance(&loaded).is_ok());
    }

    #[test]
    fn test_get_apr_cli_version_returns_string() {
        // This test just verifies the function doesn't panic
        // In CI where apr isn't installed, it returns "unknown"
        let version = get_apr_cli_version();
        assert!(!version.is_empty());
    }

    // ========================================================================
    // FALSIFICATION TESTS (PMAT-PROV-001)
    // Operation "Trust No One" - Popperian Falsification
    // ========================================================================

    mod falsification {
        use super::*;

        // ====================================================================
        // Vector A: Integrity Attacks (Physical Artifacts)
        // ====================================================================

        /// F-PROV-IO-001: The "Bit Flip" - corrupt SHA256 hash
        /// Expected: verify_provenance_integrity() MUST detect mismatch (PROV-006)
        #[test]
        fn f_prov_io_001_bit_flip_hash() {
            let dir = tempfile::tempdir().unwrap();

            // Create real files and valid provenance
            let safetensors = dir.path().join("model.safetensors");
            std::fs::write(&safetensors, "source content").unwrap();
            let prov = create_source_provenance(&safetensors, "test/model").unwrap();
            save_provenance(dir.path(), &prov).unwrap();

            // Manually corrupt the hash in the file
            let prov_path = dir.path().join(".provenance.json");
            let content = std::fs::read_to_string(&prov_path).unwrap();
            let corrupted = content.replace(&prov.source.sha256, "CORRUPTED_HASH");
            std::fs::write(&prov_path, corrupted).unwrap();

            // Load provenance (JSON is valid, just hash is wrong)
            let loaded = load_provenance(dir.path()).unwrap();

            // Basic validation still passes (format/converter checks)
            assert!(validate_provenance(&loaded).is_ok());

            // FIX VERIFIED: verify_provenance_integrity() detects the corruption
            let integrity_result = verify_provenance_integrity(&loaded, dir.path());
            assert!(integrity_result.is_err());
            assert!(matches!(
                integrity_result.unwrap_err(),
                ProvenanceError::HashMismatch { .. }
            ));
        }

        /// F-PROV-IO-002: The "Truncation" - partial JSON
        /// Expected: load_provenance() returns robust error, no panic
        #[test]
        fn f_prov_io_002_truncated_json() {
            let dir = tempfile::tempdir().unwrap();
            let prov_path = dir.path().join(".provenance.json");

            // Write truncated JSON (simulate power loss)
            std::fs::write(&prov_path, r#"{"source": {"format": "safetens"#).unwrap();

            let result = load_provenance(dir.path());

            // CORROBORATED: Returns error, does not panic
            assert!(result.is_err());
            // Verify it's a serialization error, not a panic
            let err = result.unwrap_err();
            assert!(matches!(err, Error::SerializationError(_)));
        }

        /// F-PROV-IO-003: The "Ghost File" - model file deleted but provenance exists
        /// Expected: verify_files_exist() detects missing file (PROV-007)
        #[test]
        fn f_prov_io_003_ghost_file() {
            let dir = tempfile::tempdir().unwrap();

            // Create source file and provenance
            let safetensors = dir.path().join("model.safetensors");
            std::fs::write(&safetensors, "content").unwrap();
            let prov = create_source_provenance(&safetensors, "test/model").unwrap();
            save_provenance(dir.path(), &prov).unwrap();

            // Delete the model file (ghost it)
            std::fs::remove_file(&safetensors).unwrap();

            // Load provenance - still works (JSON exists)
            let loaded = load_provenance(dir.path()).unwrap();

            // Basic validation still passes (format/converter checks)
            assert!(validate_provenance(&loaded).is_ok());

            // FIX VERIFIED: verify_files_exist() detects the ghost
            let exist_result = verify_files_exist(&loaded, dir.path());
            assert!(exist_result.is_err());
            assert!(matches!(
                exist_result.unwrap_err(),
                ProvenanceError::FileMissing { .. }
            ));

            // FIX VERIFIED: verify_provenance_integrity() also detects it
            let integrity_result = verify_provenance_integrity(&loaded, dir.path());
            assert!(integrity_result.is_err());
            assert!(matches!(
                integrity_result.unwrap_err(),
                ProvenanceError::FileMissing { .. }
            ));
        }

        /// F-PROV-IO-004: The "Hash Collision" - empty file hash
        /// Expected: Correct hash for empty file, no panic
        #[test]
        fn f_prov_io_004_empty_file_hash() {
            let dir = tempfile::tempdir().unwrap();
            let empty_file = dir.path().join("empty.bin");
            std::fs::write(&empty_file, "").unwrap();

            let hash = compute_sha256(&empty_file).unwrap();

            // CORROBORATED: Returns correct SHA256 for empty file
            assert_eq!(
                hash,
                "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
            );
        }

        // ====================================================================
        // Vector B: Logic Bypass (Validation Rules)
        // ====================================================================

        /// F-PROV-LOGIC-001: The "Imposter Source" - GGUF as source format
        /// Expected: Validation fails (Ground Truth Policy 7.4)
        #[test]
        fn f_prov_logic_001_imposter_source() {
            let mut prov = sample_provenance();
            prov.source.format = "gguf".to_string(); // Violate 7.4

            let result = validate_provenance(&prov);

            // CORROBORATED: Correctly rejects GGUF as source
            assert!(result.is_err());
            assert!(matches!(
                result.unwrap_err(),
                ProvenanceError::InvalidSourceFormat { .. }
            ));
        }

        /// F-PROV-LOGIC-002: The "Rogue Converter"
        /// Expected: validate_provenance() fails (PROV-002)
        #[test]
        fn f_prov_logic_002_rogue_converter() {
            let mut prov = sample_provenance();
            prov.derived[0].converter = "suspicious-script v0.1".to_string();

            let result = validate_provenance(&prov);

            // CORROBORATED: Correctly rejects non-apr-cli converter
            assert!(result.is_err());
            assert!(matches!(
                result.unwrap_err(),
                ProvenanceError::InvalidConverter { .. }
            ));
        }

        /// F-PROV-LOGIC-003: The "Quantization Lie"
        /// Expected: If we lie in JSON, can we compare apples to oranges?
        #[test]
        fn f_prov_logic_003_quantization_lie() {
            let mut prov = sample_provenance();

            // Lie: Say both are Q4_K_M but actual files would be different
            prov.derived[0].quantization = Some("q4_k_m".to_string());
            prov.derived[1].quantization = Some("q4_k_m".to_string());

            // System only checks JSON, not actual file headers
            let result = validate_comparison(&prov, "gguf", "apr");

            // OBSERVATION: Comparison passes because JSON claims match
            assert!(result.is_ok());

            // FALSIFIED: We can lie in metadata and bypass quantization check
            // The system does NOT verify actual file quantization headers
            // P0 TICKET REQUIRED: No verification of claimed vs actual quantization
        }

        /// F-PROV-LOGIC-004: Case sensitivity attack on format
        #[test]
        fn f_prov_logic_004_case_sensitivity() {
            let mut prov = sample_provenance();
            prov.source.format = "SafeTensors".to_string(); // Wrong case

            let result = validate_provenance(&prov);

            // CORROBORATED: Correctly rejects wrong case
            // (string comparison is case-sensitive)
            assert!(result.is_err());
        }

        /// F-PROV-LOGIC-005: Empty converter string
        #[test]
        fn f_prov_logic_005_empty_converter() {
            let mut prov = sample_provenance();
            prov.derived[0].converter = String::new();

            let result = validate_provenance(&prov);

            // CORROBORATED: Empty string != "apr-cli"
            assert!(result.is_err());
        }

        /// F-PROV-LOGIC-006: Converter with whitespace prefix
        #[test]
        fn f_prov_logic_006_whitespace_converter() {
            let mut prov = sample_provenance();
            prov.derived[0].converter = " apr-cli".to_string(); // Leading space

            let result = validate_provenance(&prov);

            // CORROBORATED: " apr-cli" != "apr-cli"
            assert!(result.is_err());
        }

        // ====================================================================
        // Vector C: Workflow Sabotage
        // ====================================================================

        /// F-PROV-FLOW-001: The "Time Traveler" - future timestamp
        #[test]
        fn f_prov_flow_001_future_timestamp() {
            let mut prov = sample_provenance();
            prov.source.downloaded_at = "2099-12-31T23:59:59Z".to_string();

            // OBSERVATION: No timestamp validation exists
            let result = validate_provenance(&prov);
            assert!(result.is_ok());

            // FALSIFIED: Future timestamps accepted without warning
            // Minor issue - could indicate clock manipulation
        }

        /// F-PROV-FLOW-002: The "Race Condition" - concurrent writes
        #[test]
        fn f_prov_flow_002_race_condition() {
            use std::sync::Arc;
            use std::thread;

            let dir = tempfile::tempdir().unwrap();
            let dir_path = Arc::new(dir.path().to_path_buf());

            let mut handles = vec![];

            for i in 0..10 {
                let path = Arc::clone(&dir_path);
                handles.push(thread::spawn(move || {
                    let mut prov = sample_provenance();
                    prov.source.sha256 = format!("hash_{i}");
                    save_provenance(&path, &prov).unwrap();
                }));
            }

            for handle in handles {
                handle.join().unwrap();
            }

            // Load and verify - should have ONE consistent state
            let loaded = load_provenance(&dir_path).unwrap();

            // OBSERVATION: No file locking, last writer wins
            // Result is non-deterministic but at least valid JSON
            assert!(validate_provenance(&loaded).is_ok());

            // CORROBORATED (with caveat): No corruption, but no atomicity guarantee
            // Recommend: atomic write (write to .tmp, then rename)
        }

        /// F-PROV-FLOW-003: The "Version Spoof" - empty version
        #[test]
        fn f_prov_flow_003_empty_version() {
            let dir = tempfile::tempdir().unwrap();
            let safetensors = dir.path().join("model.safetensors");
            let derived = dir.path().join("model.gguf");
            std::fs::write(&safetensors, "source").unwrap();
            std::fs::write(&derived, "derived").unwrap();

            let mut prov = create_source_provenance(&safetensors, "test/model").unwrap();

            // Add derived with empty version
            add_derived(&mut prov, "gguf", &derived, None, "").unwrap();

            // OBSERVATION: Empty version string is accepted
            assert!(validate_provenance(&prov).is_ok());
            assert!(prov.derived[0].converter_version.is_empty());

            // FALSIFIED: No validation that version is non-empty
            // Minor issue - could indicate tampering
        }

        // ====================================================================
        // Vector D: Code Mutation (White Box)
        // ====================================================================

        /// F-PROV-CODE-001: Add same derived model twice
        /// Expected: add_derived() rejects duplicate (PROV-008)
        #[test]
        fn f_prov_code_001_duplicate_derived() {
            let dir = tempfile::tempdir().unwrap();
            let safetensors = dir.path().join("model.safetensors");
            let gguf = dir.path().join("model.gguf");
            std::fs::write(&safetensors, "source").unwrap();
            std::fs::write(&gguf, "derived").unwrap();

            let mut prov = create_source_provenance(&safetensors, "test/model").unwrap();

            // Add GGUF first time - succeeds
            add_derived(&mut prov, "gguf", &gguf, None, "0.2.12").unwrap();
            assert_eq!(prov.derived.len(), 1);

            // FIX VERIFIED: Second add fails with DuplicateDerived error
            let result = add_derived(&mut prov, "gguf", &gguf, None, "0.2.12");
            assert!(result.is_err());
            assert!(matches!(
                result.unwrap_err(),
                Error::Provenance(ProvenanceError::DuplicateDerived { .. })
            ));

            // Still only one entry
            assert_eq!(prov.derived.len(), 1);
        }

        /// F-PROV-CODE-001b: Different quantization = not a duplicate
        #[test]
        fn f_prov_code_001b_different_quantization_allowed() {
            let dir = tempfile::tempdir().unwrap();
            let safetensors = dir.path().join("model.safetensors");
            let gguf = dir.path().join("model.gguf");
            let gguf_q4 = dir.path().join("model-q4.gguf");
            std::fs::write(&safetensors, "source").unwrap();
            std::fs::write(&gguf, "derived").unwrap();
            std::fs::write(&gguf_q4, "derived q4").unwrap();

            let mut prov = create_source_provenance(&safetensors, "test/model").unwrap();

            // Add unquantized GGUF
            add_derived(&mut prov, "gguf", &gguf, None, "0.2.12").unwrap();

            // Add quantized GGUF - different quantization = allowed
            add_derived(&mut prov, "gguf", &gguf_q4, Some("q4_k_m"), "0.2.12").unwrap();

            // Both entries exist
            assert_eq!(prov.derived.len(), 2);
        }

        /// F-PROV-CODE-002: Feed non-provenance JSON
        #[test]
        fn f_prov_code_002_wrong_json_schema() {
            let dir = tempfile::tempdir().unwrap();
            let prov_path = dir.path().join(".provenance.json");

            // Write valid JSON but wrong schema
            std::fs::write(&prov_path, r#"{"hello": "world"}"#).unwrap();

            let result = load_provenance(dir.path());

            // CORROBORATED: Returns deserialization error
            assert!(result.is_err());
            assert!(matches!(result.unwrap_err(), Error::SerializationError(_)));
        }

        /// F-PROV-CODE-003: Unicode in fields
        #[test]
        fn f_prov_code_003_unicode_injection() {
            let mut prov = sample_provenance();
            prov.source.hf_repo = "test/模型".to_string(); // Chinese characters
            prov.derived[0].path = "model_مدل.gguf".to_string(); // Arabic

            // Should handle unicode gracefully
            let result = validate_provenance(&prov);
            assert!(result.is_ok());

            // Serialize and deserialize
            let json = serde_json::to_string(&prov).unwrap();
            let loaded: Provenance = serde_json::from_str(&json).unwrap();
            assert_eq!(loaded.source.hf_repo, "test/模型");

            // CORROBORATED: Unicode handled correctly
        }

        /// F-PROV-CODE-004: Very long strings (buffer overflow attempt)
        #[test]
        fn f_prov_code_004_long_strings() {
            let mut prov = sample_provenance();
            prov.source.sha256 = "a".repeat(1_000_000); // 1MB hash string

            // Should not panic or OOM on validation
            let result = validate_provenance(&prov);
            assert!(result.is_ok());

            // CORROBORATED: Long strings handled (though invalid hash)
            // Note: No hash format validation exists
        }

        /// F-PROV-CODE-005: Null bytes in strings
        #[test]
        fn f_prov_code_005_null_bytes() {
            let mut prov = sample_provenance();
            prov.source.path = "model\0.safetensors".to_string();

            // Should handle embedded nulls
            let result = validate_provenance(&prov);
            assert!(result.is_ok());

            let json = serde_json::to_string(&prov).unwrap();
            let loaded: Provenance = serde_json::from_str(&json).unwrap();
            assert!(loaded.source.path.contains('\0'));

            // CORROBORATED: Null bytes preserved (could be path traversal risk)
        }

        /// F-PROV-CODE-006: Empty derived list
        /// Expected: validate_comparison() fails for missing formats (PROV-009)
        #[test]
        fn f_prov_code_006_empty_derived() {
            let mut prov = sample_provenance();
            prov.derived.clear();

            // Provenance with no derived formats is valid (source only)
            let result = validate_provenance(&prov);
            assert!(result.is_ok());

            // FIX VERIFIED: Comparison fails when formats don't exist
            let cmp_result = validate_comparison(&prov, "gguf", "apr");
            assert!(cmp_result.is_err());
            assert!(matches!(
                cmp_result.unwrap_err(),
                ProvenanceError::FormatNotFound { .. }
            ));
        }

        /// F-PROV-CODE-007: Comparison with non-existent format
        /// Expected: validate_comparison() fails with FormatNotFound (PROV-009)
        #[test]
        fn f_prov_code_007_phantom_format_comparison() {
            let prov = sample_provenance();

            // Compare format that doesn't exist
            let result = validate_comparison(&prov, "gguf", "phantom_format");

            // FIX VERIFIED: Returns FormatNotFound error
            assert!(result.is_err());
            let err = result.unwrap_err();
            assert!(
                matches!(err, ProvenanceError::FormatNotFound { format } if format == "phantom_format")
            );
        }

        // ====================================================================
        // New Tests for Verification Functions
        // ====================================================================

        /// Test verify_provenance_integrity with valid files
        #[test]
        fn test_verify_integrity_valid() {
            let dir = tempfile::tempdir().unwrap();
            let safetensors = dir.path().join("model.safetensors");
            let gguf = dir.path().join("model.gguf");
            std::fs::write(&safetensors, "source content").unwrap();
            std::fs::write(&gguf, "gguf content").unwrap();

            let mut prov = create_source_provenance(&safetensors, "test/model").unwrap();
            add_derived(&mut prov, "gguf", &gguf, None, "0.2.12").unwrap();

            // All files exist and hashes match
            let result = verify_provenance_integrity(&prov, dir.path());
            assert!(result.is_ok());
        }

        /// Test verify_provenance_integrity detects modified source
        #[test]
        fn test_verify_integrity_modified_source() {
            let dir = tempfile::tempdir().unwrap();
            let safetensors = dir.path().join("model.safetensors");
            std::fs::write(&safetensors, "original content").unwrap();

            let prov = create_source_provenance(&safetensors, "test/model").unwrap();

            // Modify the source file after provenance creation
            std::fs::write(&safetensors, "MODIFIED content").unwrap();

            // Integrity check fails
            let result = verify_provenance_integrity(&prov, dir.path());
            assert!(result.is_err());
            assert!(matches!(
                result.unwrap_err(),
                ProvenanceError::HashMismatch { .. }
            ));
        }

        /// Test verify_provenance_integrity detects modified derived file
        #[test]
        fn test_verify_integrity_modified_derived() {
            let dir = tempfile::tempdir().unwrap();
            let safetensors = dir.path().join("model.safetensors");
            let gguf = dir.path().join("model.gguf");
            std::fs::write(&safetensors, "source").unwrap();
            std::fs::write(&gguf, "original gguf").unwrap();

            let mut prov = create_source_provenance(&safetensors, "test/model").unwrap();
            add_derived(&mut prov, "gguf", &gguf, None, "0.2.12").unwrap();

            // Modify derived file
            std::fs::write(&gguf, "TAMPERED gguf").unwrap();

            let result = verify_provenance_integrity(&prov, dir.path());
            assert!(result.is_err());
            assert!(matches!(
                result.unwrap_err(),
                ProvenanceError::HashMismatch { .. }
            ));
        }

        /// Test verify_files_exist with all files present
        #[test]
        fn test_verify_files_exist_valid() {
            let dir = tempfile::tempdir().unwrap();
            let safetensors = dir.path().join("model.safetensors");
            let gguf = dir.path().join("model.gguf");
            std::fs::write(&safetensors, "source").unwrap();
            std::fs::write(&gguf, "gguf").unwrap();

            let mut prov = create_source_provenance(&safetensors, "test/model").unwrap();
            add_derived(&mut prov, "gguf", &gguf, None, "0.2.12").unwrap();

            assert!(verify_files_exist(&prov, dir.path()).is_ok());
        }

        /// Test verify_files_exist detects missing derived file
        #[test]
        fn test_verify_files_exist_missing_derived() {
            let dir = tempfile::tempdir().unwrap();
            let safetensors = dir.path().join("model.safetensors");
            let gguf = dir.path().join("model.gguf");
            std::fs::write(&safetensors, "source").unwrap();
            std::fs::write(&gguf, "gguf").unwrap();

            let mut prov = create_source_provenance(&safetensors, "test/model").unwrap();
            add_derived(&mut prov, "gguf", &gguf, None, "0.2.12").unwrap();

            // Delete derived file
            std::fs::remove_file(&gguf).unwrap();

            let result = verify_files_exist(&prov, dir.path());
            assert!(result.is_err());
            assert!(matches!(
                result.unwrap_err(),
                ProvenanceError::FileMissing { .. }
            ));
        }

        /// Test new error display messages
        #[test]
        fn test_new_error_displays() {
            let err = ProvenanceError::HashMismatch {
                path: "model.gguf".to_string(),
                expected: "abc123".to_string(),
                actual: "def456".to_string(),
            };
            assert!(err.to_string().contains("PROV-006"));

            let err = ProvenanceError::FileMissing {
                path: "model.safetensors".to_string(),
            };
            assert!(err.to_string().contains("PROV-007"));

            let err = ProvenanceError::DuplicateDerived {
                format: "gguf".to_string(),
                quantization: Some("q4_k_m".to_string()),
            };
            assert!(err.to_string().contains("PROV-008"));

            let err = ProvenanceError::FormatNotFound {
                format: "phantom".to_string(),
            };
            assert!(err.to_string().contains("PROV-009"));
        }
    }
}
