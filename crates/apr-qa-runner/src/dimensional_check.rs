//! Metadata-only dimensional verification for dim-smoke tier
//!
//! Verifies model dimensional correctness by parsing `config.json` and
//! SafeTensors headers without loading model weights into memory.
//! Target: complete in under 5 seconds per model.

use crate::layout_contract::{
    LayoutModelConfig, find_and_load_config, find_safetensors_files, read_safetensors_metadata,
};
use crate::playbook::Playbook;
use std::path::Path;
use std::time::Instant;

/// Result of a single dimensional check
#[derive(Debug, Clone)]
pub struct DimensionalCheck {
    /// Check name (e.g., "config_parse", "hidden_size", "num_layers")
    pub name: String,
    /// Expected value
    pub expected: String,
    /// Actual value found
    pub actual: String,
    /// Whether the check passed
    pub passed: bool,
}

/// Aggregated result of all dimensional checks for a model
#[derive(Debug, Clone)]
pub struct DimensionalCheckResult {
    /// Model identifier
    pub model_id: String,
    /// Whether all checks passed
    pub passed: bool,
    /// Individual check results
    pub checks: Vec<DimensionalCheck>,
    /// Total duration in milliseconds
    pub duration_ms: u64,
}

/// Run metadata-only dimensional verification against a model directory.
///
/// Checks:
/// 1. `config.json` exists and parses successfully
/// 2. Architecture dimensions match playbook expectations
/// 3. SafeTensors files exist and headers parse
/// 4. Key tensors have correct shapes
#[must_use]
pub fn run_dimensional_check(model_path: &Path, playbook: &Playbook) -> DimensionalCheckResult {
    let start = Instant::now();
    let model_id = playbook.model.hf_repo.clone();
    let mut checks = Vec::new();

    let config = find_and_load_config(model_path);
    let config_parsed = config.hidden_size.is_some() || config.num_hidden_layers.is_some();
    checks.push(DimensionalCheck {
        name: "config_parse".to_string(),
        expected: "config.json parseable".to_string(),
        actual: if config_parsed {
            "parsed successfully".to_string()
        } else {
            "no config.json or empty".to_string()
        },
        passed: config_parsed,
    });

    check_architecture_dims(&playbook.model, &config, &mut checks);
    check_safetensors(model_path, &config, &mut checks);

    let all_passed = checks.iter().all(|c| c.passed);
    DimensionalCheckResult {
        model_id,
        passed: all_passed,
        checks,
        duration_ms: start.elapsed().as_millis() as u64,
    }
}

/// Format an optional u32 value as a string, or "missing" if None.
fn fmt_opt(v: Option<u32>) -> String {
    v.map_or_else(|| "missing".to_string(), |v| v.to_string())
}

/// Check architecture dimensions from config.json against playbook expectations.
fn check_architecture_dims(
    model: &crate::playbook::ModelConfig,
    config: &LayoutModelConfig,
    checks: &mut Vec<DimensionalCheck>,
) {
    let dim_checks: &[(&str, Option<u32>, Option<usize>)] = &[
        ("hidden_size", model.expected_hidden_dim, config.hidden_size),
        (
            "num_layers",
            model.expected_num_layers,
            config.num_hidden_layers,
        ),
        (
            "num_heads",
            model.expected_num_heads,
            config.num_attention_heads,
        ),
        (
            "num_kv_heads",
            model.expected_num_kv_heads,
            config.num_key_value_heads,
        ),
        ("vocab_size", model.expected_vocab_size, config.vocab_size),
    ];

    for &(name, expected, actual_raw) in dim_checks {
        if let Some(expected_val) = expected {
            let actual = actual_raw.map(|v| v as u32);
            checks.push(DimensionalCheck {
                name: name.to_string(),
                expected: expected_val.to_string(),
                actual: fmt_opt(actual),
                passed: actual == Some(expected_val),
            });
        }
    }
}

/// Check SafeTensors file existence and header tensor shapes.
fn check_safetensors(
    model_path: &Path,
    config: &LayoutModelConfig,
    checks: &mut Vec<DimensionalCheck>,
) {
    let st_files = find_safetensors_files(model_path);
    checks.push(DimensionalCheck {
        name: "safetensors_found".to_string(),
        expected: ">= 1 file".to_string(),
        actual: format!("{} file(s)", st_files.len()),
        passed: !st_files.is_empty(),
    });

    let Some(first_file) = st_files.first() else {
        return;
    };

    if let Ok(tensors) = read_safetensors_metadata(first_file) {
        checks.push(DimensionalCheck {
            name: "safetensors_header".to_string(),
            expected: ">= 1 tensor".to_string(),
            actual: format!("{} tensor(s)", tensors.len()),
            passed: !tensors.is_empty(),
        });

        check_key_tensor(&tensors, "model.embed_tokens.weight", config, checks);
        check_key_tensor(&tensors, "lm_head.weight", config, checks);
    } else {
        checks.push(DimensionalCheck {
            name: "safetensors_header".to_string(),
            expected: "parseable header".to_string(),
            actual: "parse error".to_string(),
            passed: false,
        });
    }
}

/// Check that a key tensor has expected shape [dim0, dim1].
fn check_key_tensor(
    tensors: &std::collections::HashMap<String, Vec<usize>>,
    name: &str,
    config: &LayoutModelConfig,
    checks: &mut Vec<DimensionalCheck>,
) {
    let short_name = name.rsplit('.').nth(1).unwrap_or(name);
    let Some(shape) = tensors.get(name) else {
        // Tensor not found is not a failure — sharded models may split tensors across files
        return;
    };

    if shape.len() != 2 {
        checks.push(DimensionalCheck {
            name: format!("tensor_{short_name}"),
            expected: "2D tensor".to_string(),
            actual: format!("{}D tensor: {shape:?}", shape.len()),
            passed: false,
        });
        return;
    }

    let mut passed = true;
    let mut expected_parts = Vec::new();

    if let Some(d0) = config.vocab_size {
        expected_parts.push(format!("dim0={d0}"));
        if shape[0] != d0 {
            passed = false;
        }
    }
    if let Some(d1) = config.hidden_size {
        expected_parts.push(format!("dim1={d1}"));
        if shape[1] != d1 {
            passed = false;
        }
    }

    let expected_str = if expected_parts.is_empty() {
        "2D tensor".to_string()
    } else {
        expected_parts.join(", ")
    };

    checks.push(DimensionalCheck {
        name: format!("tensor_{short_name}"),
        expected: expected_str,
        actual: format!("{shape:?}"),
        passed,
    });
}


#[cfg(test)]
#[path = "dimensional_check_tests.rs"]
mod dimensional_check_tests;
