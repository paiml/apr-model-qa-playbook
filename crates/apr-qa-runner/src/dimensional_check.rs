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
mod tests {
    use super::*;
    use crate::playbook::Playbook;
    use std::io::Write;
    use tempfile::TempDir;

    fn make_minimal_playbook(hf_repo: &str) -> Playbook {
        let yaml = format!(
            r#"
name: test-playbook
version: "1.0"
model:
  hf_repo: "{hf_repo}"
  expected_hidden_dim: 896
  expected_num_layers: 24
  expected_num_heads: 14
  expected_num_kv_heads: 2
  expected_vocab_size: 151936
test_matrix:
  modalities: [run]
  backends: [cpu]
  formats: [safetensors]
  prompts:
    - "hello"
"#
        );
        Playbook::from_yaml(&yaml).expect("valid test playbook")
    }

    fn write_config_json(dir: &Path, config: &serde_json::Value) {
        let path = dir.join("config.json");
        let mut f = std::fs::File::create(path).unwrap();
        f.write_all(serde_json::to_string(config).unwrap().as_bytes())
            .unwrap();
    }

    fn write_minimal_safetensors(dir: &Path, tensors: &[(&str, &[usize])]) {
        use std::collections::HashMap;
        let path = dir.join("model.safetensors");

        let mut header_map: HashMap<&str, serde_json::Value> = HashMap::new();
        let mut offset: u64 = 0;
        for &(name, shape) in tensors {
            let num_elements: usize = shape.iter().product();
            let byte_size = num_elements * 4;
            let tensor_info = serde_json::json!({
                "dtype": "F32",
                "shape": shape,
                "data_offsets": [offset, offset + byte_size as u64]
            });
            header_map.insert(name, tensor_info);
            offset += byte_size as u64;
        }

        let header_json = serde_json::to_string(&header_map).unwrap();
        let header_bytes = header_json.as_bytes();
        let header_len = header_bytes.len() as u64;

        let mut f = std::fs::File::create(path).unwrap();
        f.write_all(&header_len.to_le_bytes()).unwrap();
        f.write_all(header_bytes).unwrap();
        f.write_all(&vec![0u8; offset as usize]).unwrap();
    }

    #[test]
    fn test_check_valid_config() {
        let dir = TempDir::new().unwrap();
        let config = serde_json::json!({
            "hidden_size": 896,
            "num_hidden_layers": 24,
            "num_attention_heads": 14,
            "num_key_value_heads": 2,
            "vocab_size": 151_936
        });
        write_config_json(dir.path(), &config);
        write_minimal_safetensors(
            dir.path(),
            &[
                ("model.embed_tokens.weight", &[151_936, 896]),
                ("lm_head.weight", &[151_936, 896]),
            ],
        );

        let playbook = make_minimal_playbook("Qwen/Qwen2.5-Coder-0.5B-Instruct");
        let result = run_dimensional_check(dir.path(), &playbook);

        assert!(
            result.passed,
            "all checks should pass: {:#?}",
            result.checks
        );
        assert!(result.duration_ms < 5000, "should complete quickly");
        assert!(
            result.checks.len() >= 8,
            "expected at least 8 checks, got {}",
            result.checks.len()
        );
    }

    #[test]
    fn test_check_mismatched_hidden_size() {
        let dir = TempDir::new().unwrap();
        let config = serde_json::json!({
            "hidden_size": 512,
            "num_hidden_layers": 24,
            "num_attention_heads": 14,
            "num_key_value_heads": 2,
            "vocab_size": 151_936
        });
        write_config_json(dir.path(), &config);

        let playbook = make_minimal_playbook("Qwen/Qwen2.5-Coder-0.5B-Instruct");
        let result = run_dimensional_check(dir.path(), &playbook);

        assert!(!result.passed);
        let hidden_check = result
            .checks
            .iter()
            .find(|c| c.name == "hidden_size")
            .unwrap();
        assert!(!hidden_check.passed);
        assert_eq!(hidden_check.expected, "896");
        assert_eq!(hidden_check.actual, "512");
    }

    #[test]
    fn test_check_missing_config() {
        let dir = TempDir::new().unwrap();

        let playbook = make_minimal_playbook("Qwen/Qwen2.5-Coder-0.5B-Instruct");
        let result = run_dimensional_check(dir.path(), &playbook);

        assert!(!result.passed);
        let config_check = result
            .checks
            .iter()
            .find(|c| c.name == "config_parse")
            .unwrap();
        assert!(!config_check.passed);
    }

    #[test]
    fn test_check_safetensors_header() {
        let dir = TempDir::new().unwrap();
        let config = serde_json::json!({
            "hidden_size": 896,
            "num_hidden_layers": 24,
            "num_attention_heads": 14,
            "num_key_value_heads": 2,
            "vocab_size": 151_936
        });
        write_config_json(dir.path(), &config);
        write_minimal_safetensors(
            dir.path(),
            &[
                ("model.embed_tokens.weight", &[151_936, 896]),
                ("lm_head.weight", &[151_936, 896]),
                ("model.layers.0.self_attn.q_proj.weight", &[896, 896]),
            ],
        );

        let playbook = make_minimal_playbook("Qwen/Qwen2.5-Coder-0.5B-Instruct");
        let result = run_dimensional_check(dir.path(), &playbook);

        assert!(
            result.passed,
            "all checks should pass: {:#?}",
            result.checks
        );
        let header_check = result
            .checks
            .iter()
            .find(|c| c.name == "safetensors_header")
            .unwrap();
        assert!(header_check.passed);
        assert_eq!(header_check.actual, "3 tensor(s)");
    }

    #[test]
    fn test_check_wrong_tensor_shape() {
        let dir = TempDir::new().unwrap();
        let config = serde_json::json!({
            "hidden_size": 896,
            "num_hidden_layers": 24,
            "vocab_size": 151_936
        });
        write_config_json(dir.path(), &config);
        write_minimal_safetensors(
            dir.path(),
            &[("model.embed_tokens.weight", &[151_936, 512])],
        );

        let playbook = make_minimal_playbook("Qwen/Qwen2.5-Coder-0.5B-Instruct");
        let result = run_dimensional_check(dir.path(), &playbook);

        assert!(!result.passed);
        let tensor_check = result
            .checks
            .iter()
            .find(|c| c.name == "tensor_embed_tokens")
            .unwrap();
        assert!(!tensor_check.passed);
    }

    #[test]
    fn test_check_no_safetensors_files() {
        let dir = TempDir::new().unwrap();
        let config = serde_json::json!({
            "hidden_size": 896,
            "num_hidden_layers": 24
        });
        write_config_json(dir.path(), &config);

        let playbook = make_minimal_playbook("Qwen/Qwen2.5-Coder-0.5B-Instruct");
        let result = run_dimensional_check(dir.path(), &playbook);

        assert!(!result.passed);
        let st_check = result
            .checks
            .iter()
            .find(|c| c.name == "safetensors_found")
            .unwrap();
        assert!(!st_check.passed);
    }

    #[test]
    fn test_check_no_expected_params() {
        let dir = TempDir::new().unwrap();
        let config = serde_json::json!({
            "hidden_size": 896,
            "num_hidden_layers": 24
        });
        write_config_json(dir.path(), &config);
        write_minimal_safetensors(dir.path(), &[("some.tensor", &[10, 20])]);

        let yaml = r#"
name: test-playbook
version: "1.0"
model:
  hf_repo: "test/model"
test_matrix:
  modalities: [run]
  backends: [cpu]
  formats: [safetensors]
  prompts:
    - "hello"
"#;
        let playbook = Playbook::from_yaml(yaml).expect("valid test playbook");
        let result = run_dimensional_check(dir.path(), &playbook);

        assert!(
            result.passed,
            "should pass with no expected params: {:#?}",
            result.checks
        );
    }

    #[test]
    fn test_result_model_id() {
        let dir = TempDir::new().unwrap();
        let config = serde_json::json!({"hidden_size": 896});
        write_config_json(dir.path(), &config);

        let playbook = make_minimal_playbook("Qwen/Qwen2.5-Coder-0.5B-Instruct");
        let result = run_dimensional_check(dir.path(), &playbook);
        assert_eq!(result.model_id, "Qwen/Qwen2.5-Coder-0.5B-Instruct");
    }
}
