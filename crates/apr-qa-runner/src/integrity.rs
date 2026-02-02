//! Model Integrity Checker (G0 Gateway)
//!
//! Pre-flight check that validates config.json matches tensor metadata in SafeTensors models.
//! This catches corrupted configs that would pass G1 (model loads) but cause silent inference failures.
//!
//! ## Background
//!
//! A corrupted config.json was found with:
//! - `num_hidden_layers: 14` (should be 24)
//! - `hidden_size: 4096` (should be 896)
//! - `vocab_size: 896` (should be 151_936)
//!
//! This passed G1 (model loads) but would cause silent inference failures.
//!
//! ## Checks
//!
//! - G0-INTEGRITY-CONFIG: config.json exists
//! - G0-INTEGRITY-LAYERS: layer count matches tensors
//! - G0-INTEGRITY-HIDDEN: hidden_size matches embedding shape
//! - G0-INTEGRITY-VOCAB: vocab_size matches embedding shape

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

/// Result of model integrity check
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(clippy::struct_excessive_bools)]
pub struct IntegrityResult {
    /// Whether all integrity checks passed
    pub passed: bool,
    /// Whether config.json was found
    pub config_found: bool,
    /// Whether layer count matches
    pub layer_count_match: bool,
    /// Whether hidden_size matches
    pub hidden_size_match: bool,
    /// Whether vocab_size matches
    pub vocab_size_match: bool,
    /// Detailed error messages
    pub errors: Vec<String>,
    /// Config values found (for diagnostics)
    pub config_values: Option<ConfigValues>,
    /// Tensor-derived values (for diagnostics)
    pub tensor_values: Option<TensorDerivedValues>,
}

/// Values parsed from config.json
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigValues {
    /// Number of hidden layers from config
    pub num_hidden_layers: Option<usize>,
    /// Hidden size from config
    pub hidden_size: Option<usize>,
    /// Vocabulary size from config
    pub vocab_size: Option<usize>,
    /// Number of attention heads from config
    pub num_attention_heads: Option<usize>,
}

/// Values derived from tensor metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorDerivedValues {
    /// Layer count from tensor names (max layer index + 1)
    pub layer_count: Option<usize>,
    /// Hidden size from embedding tensor shape[1]
    pub hidden_size: Option<usize>,
    /// Vocab size from embedding tensor shape[0]
    pub vocab_size: Option<usize>,
}

/// HuggingFace config.json structure (partial)
#[derive(Debug, Deserialize)]
struct HfConfig {
    num_hidden_layers: Option<usize>,
    hidden_size: Option<usize>,
    vocab_size: Option<usize>,
    num_attention_heads: Option<usize>,
}

/// Check integrity of a SafeTensors model directory
///
/// Validates that config.json metadata matches actual tensor shapes.
///
/// # Arguments
///
/// * `model_dir` - Path to the model directory containing config.json and .safetensors files
///
/// # Returns
///
/// `IntegrityResult` with pass/fail status and detailed error messages
#[must_use]
pub fn check_safetensors_integrity(model_dir: &Path) -> IntegrityResult {
    let mut result = IntegrityResult {
        passed: true,
        config_found: false,
        layer_count_match: true,
        hidden_size_match: true,
        vocab_size_match: true,
        errors: Vec::new(),
        config_values: None,
        tensor_values: None,
    };

    // Step 1: Check for config.json
    let config_path = model_dir.join("config.json");
    let config = match read_config(&config_path) {
        Ok(cfg) => {
            result.config_found = true;
            result.config_values = Some(ConfigValues {
                num_hidden_layers: cfg.num_hidden_layers,
                hidden_size: cfg.hidden_size,
                vocab_size: cfg.vocab_size,
                num_attention_heads: cfg.num_attention_heads,
            });
            cfg
        }
        Err(e) => {
            result.config_found = false;
            result.passed = false;
            result.errors.push(format!("G0-INTEGRITY-CONFIG: {e}"));
            return result;
        }
    };

    // Step 2: Find and parse SafeTensors files
    let safetensors_files = find_safetensors_files(model_dir);
    if safetensors_files.is_empty() {
        result.passed = false;
        result
            .errors
            .push("G0-INTEGRITY-CONFIG: No .safetensors files found".to_string());
        return result;
    }

    // Step 3: Extract tensor metadata from all files
    let mut all_tensors: HashMap<String, Vec<usize>> = HashMap::new();
    for st_path in &safetensors_files {
        match read_safetensors_metadata(st_path) {
            Ok(tensors) => {
                all_tensors.extend(tensors);
            }
            Err(e) => {
                result.passed = false;
                result.errors.push(format!(
                    "G0-INTEGRITY-CONFIG: Failed to read {}: {e}",
                    st_path.display()
                ));
                return result;
            }
        }
    }

    // Step 4: Derive values from tensors
    let tensor_values = derive_values_from_tensors(&all_tensors);
    result.tensor_values = Some(tensor_values.clone());

    // Step 5: Validate layer count
    if let (Some(config_layers), Some(tensor_layers)) =
        (config.num_hidden_layers, tensor_values.layer_count)
    {
        if config_layers != tensor_layers {
            result.layer_count_match = false;
            result.passed = false;
            result.errors.push(format!(
                "G0-INTEGRITY-LAYERS: config says {config_layers} layers but tensors have {tensor_layers}"
            ));
        }
    }

    // Step 6: Validate hidden_size
    if let (Some(config_hidden), Some(tensor_hidden)) =
        (config.hidden_size, tensor_values.hidden_size)
    {
        if config_hidden != tensor_hidden {
            result.hidden_size_match = false;
            result.passed = false;
            result.errors.push(format!(
                "G0-INTEGRITY-HIDDEN: config says hidden_size={config_hidden} but embedding has {tensor_hidden}"
            ));
        }
    }

    // Step 7: Validate vocab_size
    if let (Some(config_vocab), Some(tensor_vocab)) = (config.vocab_size, tensor_values.vocab_size)
    {
        if config_vocab != tensor_vocab {
            result.vocab_size_match = false;
            result.passed = false;
            result.errors.push(format!(
                "G0-INTEGRITY-VOCAB: config says vocab_size={config_vocab} but embedding has {tensor_vocab}"
            ));
        }
    }

    result
}

/// Read and parse config.json
fn read_config(path: &Path) -> Result<HfConfig, String> {
    let file = File::open(path).map_err(|e| format!("config.json not found or unreadable: {e}"))?;
    let reader = BufReader::new(file);
    serde_json::from_reader(reader).map_err(|e| format!("config.json parse error: {e}"))
}

/// Find all .safetensors files in a directory
fn find_safetensors_files(dir: &Path) -> Vec<std::path::PathBuf> {
    let mut files = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().is_some_and(|ext| ext == "safetensors") {
                files.push(path);
            }
        }
    }
    files.sort(); // Ensure consistent ordering
    files
}

/// Maximum header size for SafeTensors files (100MB)
const MAX_HEADER_SIZE: usize = 100 * 1024 * 1024;

/// Read SafeTensors metadata header and extract tensor name -> shape mapping
fn read_safetensors_metadata(path: &Path) -> Result<HashMap<String, Vec<usize>>, String> {
    let mut file = File::open(path).map_err(|e| format!("Failed to open safetensors file: {e}"))?;

    // SafeTensors format: first 8 bytes are header length (little endian u64)
    let mut header_len_bytes = [0u8; 8];
    file.read_exact(&mut header_len_bytes)
        .map_err(|e| format!("Failed to read header length: {e}"))?;
    let header_len = u64::from_le_bytes(header_len_bytes) as usize;

    // Safety check: header shouldn't be unreasonably large
    if header_len > MAX_HEADER_SIZE {
        return Err(format!(
            "Header size {header_len} exceeds maximum {MAX_HEADER_SIZE}"
        ));
    }

    // Read the JSON header
    let mut header_bytes = vec![0u8; header_len];
    file.read_exact(&mut header_bytes)
        .map_err(|e| format!("Failed to read header: {e}"))?;

    let header_str = std::str::from_utf8(&header_bytes)
        .map_err(|e| format!("Header is not valid UTF-8: {e}"))?;

    // Parse as JSON object
    let header: serde_json::Value =
        serde_json::from_str(header_str).map_err(|e| format!("Header JSON parse error: {e}"))?;

    let obj = header.as_object().ok_or("Header is not a JSON object")?;

    let mut tensors = HashMap::new();
    for (name, value) in obj {
        // Skip __metadata__ key
        if name == "__metadata__" {
            continue;
        }
        if let Some(tensor_info) = value.as_object() {
            if let Some(shape) = tensor_info.get("shape") {
                if let Some(shape_arr) = shape.as_array() {
                    let dims: Vec<usize> = shape_arr
                        .iter()
                        .filter_map(|v| v.as_u64().map(|n| n as usize))
                        .collect();
                    tensors.insert(name.clone(), dims);
                }
            }
        }
    }

    Ok(tensors)
}

/// Derive model configuration values from tensor metadata
fn derive_values_from_tensors(tensors: &HashMap<String, Vec<usize>>) -> TensorDerivedValues {
    let mut result = TensorDerivedValues {
        layer_count: None,
        hidden_size: None,
        vocab_size: None,
    };

    // Find layer count from tensor names like "model.layers.N.*" or "layers.N.*"
    let mut max_layer: Option<usize> = None;
    for name in tensors.keys() {
        if let Some(layer_num) = extract_layer_number(name) {
            max_layer = Some(max_layer.map_or(layer_num, |m| m.max(layer_num)));
        }
    }
    result.layer_count = max_layer.map(|n| n + 1); // Layer indices are 0-based

    // Find hidden_size and vocab_size from embedding tensor
    // Common names: "model.embed_tokens.weight", "embed_tokens.weight", "lm_head.weight"
    let embedding_names = [
        "model.embed_tokens.weight",
        "embed_tokens.weight",
        "transformer.wte.weight",
        "wte.weight",
    ];

    for name in embedding_names {
        if let Some(shape) = tensors.get(name) {
            if shape.len() >= 2 {
                result.vocab_size = Some(shape[0]);
                result.hidden_size = Some(shape[1]);
                break;
            }
        }
    }

    // If no embed_tokens found, try lm_head (output projection)
    if result.vocab_size.is_none() {
        let lm_head_names = ["lm_head.weight", "model.lm_head.weight"];
        for name in lm_head_names {
            if let Some(shape) = tensors.get(name) {
                if shape.len() >= 2 {
                    result.vocab_size = Some(shape[0]);
                    result.hidden_size = Some(shape[1]);
                    break;
                }
            }
        }
    }

    result
}

/// Extract layer number from tensor name
/// Matches patterns like "model.layers.23.self_attn.q_proj.weight" -> 23
fn extract_layer_number(name: &str) -> Option<usize> {
    // Try different layer naming conventions
    let patterns = ["layers.", "h.", "transformer.h."];

    for pattern in patterns {
        if let Some(idx) = name.find(pattern) {
            let rest = &name[idx + pattern.len()..];
            let num_str: String = rest.chars().take_while(char::is_ascii_digit).collect();
            if let Ok(num) = num_str.parse::<usize>() {
                return Some(num);
            }
        }
    }
    None
}

/// Gate IDs for G0 integrity checks
pub mod gate_ids {
    /// Config.json exists and is readable
    pub const CONFIG: &str = "G0-INTEGRITY-CONFIG";
    /// Layer count in config matches tensor count
    pub const LAYERS: &str = "G0-INTEGRITY-LAYERS";
    /// Hidden size in config matches tensor shape
    pub const HIDDEN: &str = "G0-INTEGRITY-HIDDEN";
    /// Vocab size in config matches tensor shape
    pub const VOCAB: &str = "G0-INTEGRITY-VOCAB";
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    fn create_test_config(dir: &Path, layers: usize, hidden: usize, vocab: usize) {
        let config = format!(
            r#"{{
                "num_hidden_layers": {layers},
                "hidden_size": {hidden},
                "vocab_size": {vocab},
                "num_attention_heads": 12
            }}"#
        );
        let path = dir.join("config.json");
        std::fs::write(path, config).expect("write config");
    }

    fn create_mock_safetensors(dir: &Path, layers: usize, hidden: usize, vocab: usize) {
        // Create a minimal valid safetensors header
        let mut header_obj = serde_json::Map::new();

        // Add embedding tensor
        let mut embed_info = serde_json::Map::new();
        embed_info.insert("shape".to_string(), serde_json::json!([vocab, hidden]));
        embed_info.insert(
            "dtype".to_string(),
            serde_json::Value::String("F32".to_string()),
        );
        embed_info.insert(
            "data_offsets".to_string(),
            serde_json::json!([0, vocab * hidden * 4]),
        );
        header_obj.insert(
            "model.embed_tokens.weight".to_string(),
            serde_json::Value::Object(embed_info),
        );

        // Add layer tensors
        for i in 0..layers {
            let mut layer_info = serde_json::Map::new();
            layer_info.insert("shape".to_string(), serde_json::json!([hidden, hidden]));
            layer_info.insert(
                "dtype".to_string(),
                serde_json::Value::String("F32".to_string()),
            );
            layer_info.insert("data_offsets".to_string(), serde_json::json!([0, 0]));
            header_obj.insert(
                format!("model.layers.{i}.self_attn.q_proj.weight"),
                serde_json::Value::Object(layer_info),
            );
        }

        let header_json = serde_json::to_string(&header_obj).expect("serialize header");
        let header_bytes = header_json.as_bytes();
        let header_len = header_bytes.len() as u64;

        let path = dir.join("model.safetensors");
        let mut file = File::create(path).expect("create safetensors");
        file.write_all(&header_len.to_le_bytes())
            .expect("write len");
        file.write_all(header_bytes).expect("write header");
        // Write minimal tensor data (just zeros to satisfy offsets)
        file.write_all(&[0u8; 1024]).expect("write data");
    }

    #[test]
    fn test_integrity_check_all_match() {
        let dir = TempDir::new().expect("create temp dir");
        create_test_config(dir.path(), 24, 896, 151_936);
        create_mock_safetensors(dir.path(), 24, 896, 151_936);

        let result = check_safetensors_integrity(dir.path());
        assert!(
            result.passed,
            "Should pass when all values match: {:?}",
            result.errors
        );
        assert!(result.config_found);
        assert!(result.layer_count_match);
        assert!(result.hidden_size_match);
        assert!(result.vocab_size_match);
        assert!(result.errors.is_empty());
    }

    #[test]
    fn test_integrity_check_layer_mismatch() {
        let dir = TempDir::new().expect("create temp dir");
        // Config says 14 layers but tensors have 24
        create_test_config(dir.path(), 14, 896, 151_936);
        create_mock_safetensors(dir.path(), 24, 896, 151_936);

        let result = check_safetensors_integrity(dir.path());
        assert!(!result.passed, "Should fail on layer mismatch");
        assert!(!result.layer_count_match);
        assert!(
            result
                .errors
                .iter()
                .any(|e| e.contains("G0-INTEGRITY-LAYERS"))
        );
    }

    #[test]
    fn test_integrity_check_hidden_size_mismatch() {
        let dir = TempDir::new().expect("create temp dir");
        // Config says hidden=4096 but tensors have 896
        create_test_config(dir.path(), 24, 4096, 151_936);
        create_mock_safetensors(dir.path(), 24, 896, 151_936);

        let result = check_safetensors_integrity(dir.path());
        assert!(!result.passed, "Should fail on hidden_size mismatch");
        assert!(!result.hidden_size_match);
        assert!(
            result
                .errors
                .iter()
                .any(|e| e.contains("G0-INTEGRITY-HIDDEN"))
        );
    }

    #[test]
    fn test_integrity_check_vocab_size_mismatch() {
        let dir = TempDir::new().expect("create temp dir");
        // Config says vocab=896 (corrupted) but tensors have 151_936
        create_test_config(dir.path(), 24, 896, 896);
        create_mock_safetensors(dir.path(), 24, 896, 151_936);

        let result = check_safetensors_integrity(dir.path());
        assert!(!result.passed, "Should fail on vocab_size mismatch");
        assert!(!result.vocab_size_match);
        assert!(
            result
                .errors
                .iter()
                .any(|e| e.contains("G0-INTEGRITY-VOCAB"))
        );
    }

    #[test]
    fn test_integrity_check_missing_config() {
        let dir = TempDir::new().expect("create temp dir");
        // No config.json, only safetensors
        create_mock_safetensors(dir.path(), 24, 896, 151_936);

        let result = check_safetensors_integrity(dir.path());
        assert!(!result.passed, "Should fail when config.json missing");
        assert!(!result.config_found);
        assert!(
            result
                .errors
                .iter()
                .any(|e| e.contains("G0-INTEGRITY-CONFIG"))
        );
    }

    #[test]
    fn test_integrity_check_no_safetensors() {
        let dir = TempDir::new().expect("create temp dir");
        // Only config.json, no safetensors files
        create_test_config(dir.path(), 24, 896, 151_936);

        let result = check_safetensors_integrity(dir.path());
        assert!(!result.passed, "Should fail when no .safetensors files");
        assert!(
            result
                .errors
                .iter()
                .any(|e| e.contains("No .safetensors files"))
        );
    }

    #[test]
    fn test_integrity_check_multiple_mismatches() {
        let dir = TempDir::new().expect("create temp dir");
        // All values wrong (the corrupted config case)
        create_test_config(dir.path(), 14, 4096, 896);
        create_mock_safetensors(dir.path(), 24, 896, 151_936);

        let result = check_safetensors_integrity(dir.path());
        assert!(!result.passed, "Should fail on multiple mismatches");
        assert!(!result.layer_count_match);
        assert!(!result.hidden_size_match);
        assert!(!result.vocab_size_match);
        assert_eq!(result.errors.len(), 3, "Should have 3 error messages");
    }

    #[test]
    fn test_extract_layer_number() {
        assert_eq!(
            extract_layer_number("model.layers.23.self_attn.q_proj.weight"),
            Some(23)
        );
        assert_eq!(
            extract_layer_number("layers.0.mlp.gate_proj.weight"),
            Some(0)
        );
        assert_eq!(extract_layer_number("h.15.attn.c_attn.weight"), Some(15));
        assert_eq!(extract_layer_number("transformer.h.7.mlp.weight"), Some(7));
        assert_eq!(extract_layer_number("model.embed_tokens.weight"), None);
        assert_eq!(extract_layer_number("lm_head.weight"), None);
    }

    #[test]
    fn test_config_values_serialization() {
        let values = ConfigValues {
            num_hidden_layers: Some(24),
            hidden_size: Some(896),
            vocab_size: Some(151_936),
            num_attention_heads: Some(14),
        };
        let json = serde_json::to_string(&values).expect("serialize");
        assert!(json.contains("24"));
        assert!(json.contains("896"));
    }

    #[test]
    fn test_tensor_derived_values_serialization() {
        let values = TensorDerivedValues {
            layer_count: Some(24),
            hidden_size: Some(896),
            vocab_size: Some(151_936),
        };
        let json = serde_json::to_string(&values).expect("serialize");
        assert!(json.contains("24"));
        assert!(json.contains("151936"));
    }

    #[test]
    fn test_integrity_result_serialization() {
        let result = IntegrityResult {
            passed: false,
            config_found: true,
            layer_count_match: false,
            hidden_size_match: true,
            vocab_size_match: true,
            errors: vec!["G0-INTEGRITY-LAYERS: mismatch".to_string()],
            config_values: None,
            tensor_values: None,
        };
        let json = serde_json::to_string(&result).expect("serialize");
        assert!(json.contains("G0-INTEGRITY-LAYERS"));
    }

    #[test]
    fn test_gate_ids() {
        assert_eq!(gate_ids::CONFIG, "G0-INTEGRITY-CONFIG");
        assert_eq!(gate_ids::LAYERS, "G0-INTEGRITY-LAYERS");
        assert_eq!(gate_ids::HIDDEN, "G0-INTEGRITY-HIDDEN");
        assert_eq!(gate_ids::VOCAB, "G0-INTEGRITY-VOCAB");
    }

    #[test]
    fn test_integrity_result_debug() {
        let result = IntegrityResult {
            passed: true,
            config_found: true,
            layer_count_match: true,
            hidden_size_match: true,
            vocab_size_match: true,
            errors: vec![],
            config_values: None,
            tensor_values: None,
        };
        let debug_str = format!("{result:?}");
        assert!(debug_str.contains("IntegrityResult"));
    }

    #[test]
    fn test_config_values_debug() {
        let values = ConfigValues {
            num_hidden_layers: Some(24),
            hidden_size: Some(896),
            vocab_size: Some(151_936),
            num_attention_heads: Some(14),
        };
        let debug_str = format!("{values:?}");
        assert!(debug_str.contains("ConfigValues"));
    }

    #[test]
    fn test_tensor_derived_values_debug() {
        let values = TensorDerivedValues {
            layer_count: Some(24),
            hidden_size: Some(896),
            vocab_size: Some(151_936),
        };
        let debug_str = format!("{values:?}");
        assert!(debug_str.contains("TensorDerivedValues"));
    }

    #[test]
    fn test_integrity_result_clone() {
        let result = IntegrityResult {
            passed: true,
            config_found: true,
            layer_count_match: true,
            hidden_size_match: true,
            vocab_size_match: true,
            errors: vec!["test".to_string()],
            config_values: Some(ConfigValues {
                num_hidden_layers: Some(24),
                hidden_size: Some(896),
                vocab_size: Some(151_936),
                num_attention_heads: Some(14),
            }),
            tensor_values: Some(TensorDerivedValues {
                layer_count: Some(24),
                hidden_size: Some(896),
                vocab_size: Some(151_936),
            }),
        };
        let cloned = result.clone();
        assert_eq!(cloned.passed, result.passed);
        assert_eq!(cloned.errors.len(), result.errors.len());
    }

    // =========================================================================
    // Additional coverage tests for uncovered paths
    // =========================================================================

    #[test]
    fn test_read_safetensors_corrupted_file() {
        let dir = TempDir::new().expect("create temp dir");
        // Write a file that's too short to contain a valid header
        let path = dir.path().join("corrupt.safetensors");
        std::fs::write(&path, b"short").expect("write corrupt");
        let result = read_safetensors_metadata(&path);
        assert!(result.is_err());
    }

    #[test]
    fn test_read_safetensors_oversized_header() {
        let dir = TempDir::new().expect("create temp dir");
        let path = dir.path().join("oversize.safetensors");
        let mut file = std::fs::File::create(&path).expect("create file");
        // Header length of 200MB (exceeds MAX_HEADER_SIZE)
        let huge: u64 = 200_000_000;
        file.write_all(&huge.to_le_bytes()).expect("write len");
        drop(file);
        let result = read_safetensors_metadata(&path);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("exceeds maximum"));
    }

    #[test]
    fn test_read_safetensors_with_metadata_key() {
        let dir = TempDir::new().expect("create temp dir");
        // Create safetensors header that includes __metadata__ key
        let mut header_obj = serde_json::Map::new();

        // Add __metadata__ key (should be skipped)
        header_obj.insert(
            "__metadata__".to_string(),
            serde_json::json!({"format": "pt"}),
        );

        // Add a real tensor
        let mut tensor_info = serde_json::Map::new();
        tensor_info.insert("shape".to_string(), serde_json::json!([100, 50]));
        tensor_info.insert(
            "dtype".to_string(),
            serde_json::Value::String("F32".to_string()),
        );
        tensor_info.insert("data_offsets".to_string(), serde_json::json!([0, 20000]));
        header_obj.insert(
            "model.weight".to_string(),
            serde_json::Value::Object(tensor_info),
        );

        let header_json = serde_json::to_string(&header_obj).expect("serialize header");
        let header_bytes = header_json.as_bytes();
        let header_len = header_bytes.len() as u64;

        let path = dir.path().join("model.safetensors");
        let mut file = std::fs::File::create(&path).expect("create file");
        file.write_all(&header_len.to_le_bytes())
            .expect("write len");
        file.write_all(header_bytes).expect("write header");
        file.write_all(&[0u8; 128]).expect("write data padding");
        drop(file);

        let tensors = read_safetensors_metadata(&path).expect("should parse");
        // __metadata__ should NOT appear as a tensor
        assert!(!tensors.contains_key("__metadata__"));
        // But model.weight should
        assert!(tensors.contains_key("model.weight"));
        assert_eq!(tensors["model.weight"], vec![100, 50]);
    }

    #[test]
    fn test_derive_values_from_lm_head_fallback() {
        // No embed_tokens, only lm_head.weight — exercises the fallback path
        let mut tensors = HashMap::new();
        tensors.insert("lm_head.weight".to_string(), vec![32000, 4096]);
        tensors.insert(
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            vec![4096, 4096],
        );
        tensors.insert(
            "model.layers.1.self_attn.q_proj.weight".to_string(),
            vec![4096, 4096],
        );

        let values = derive_values_from_tensors(&tensors);
        assert_eq!(values.vocab_size, Some(32000));
        assert_eq!(values.hidden_size, Some(4096));
        assert_eq!(values.layer_count, Some(2));
    }

    #[test]
    fn test_derive_values_model_lm_head_fallback() {
        // No embed_tokens, uses model.lm_head.weight
        let mut tensors = HashMap::new();
        tensors.insert("model.lm_head.weight".to_string(), vec![50_000, 768]);

        let values = derive_values_from_tensors(&tensors);
        assert_eq!(values.vocab_size, Some(50_000));
        assert_eq!(values.hidden_size, Some(768));
    }

    #[test]
    fn test_check_safetensors_integrity_read_error() {
        let dir = TempDir::new().expect("create temp dir");
        create_test_config(dir.path(), 12, 768, 30_000);

        // Create a corrupt safetensors file (too short for header)
        let path = dir.path().join("model.safetensors");
        std::fs::write(&path, b"bad").expect("write corrupt");

        let result = check_safetensors_integrity(dir.path());
        assert!(!result.passed);
        assert!(
            result
                .errors
                .iter()
                .any(|e| e.contains("G0-INTEGRITY-CONFIG"))
        );
    }

    #[test]
    fn test_check_safetensors_integrity_hidden_size_mismatch() {
        let dir = TempDir::new().expect("create temp dir");
        // Config says hidden_size=1024 but tensor has 768
        create_test_config(dir.path(), 2, 1024, 30_000);
        create_mock_safetensors(dir.path(), 2, 768, 30_000);

        let result = check_safetensors_integrity(dir.path());
        assert!(!result.passed);
        assert!(result.errors.iter().any(|e| e.contains("HIDDEN")));
    }

    #[test]
    fn test_check_safetensors_integrity_vocab_size_mismatch() {
        let dir = TempDir::new().expect("create temp dir");
        // Config says vocab=50000 but tensor has 30000
        create_test_config(dir.path(), 2, 768, 50_000);
        create_mock_safetensors(dir.path(), 2, 768, 30_000);

        let result = check_safetensors_integrity(dir.path());
        assert!(!result.passed);
        assert!(result.errors.iter().any(|e| e.contains("VOCAB")));
    }
}
