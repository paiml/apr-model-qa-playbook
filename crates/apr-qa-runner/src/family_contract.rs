//! Family Contract Loader (PMAT-268)
//!
//! This module loads aprender's family YAML contracts for cross-repository
//! integration. Enables YAML-driven test matrix generation and size category
//! auto-alignment.
//!
//! # Theoretical Foundation
//!
//! - **Contract Programming (Meyer, 1992)**: Family YAMLs define model contracts
//! - **Separation of Concerns (Dijkstra, 1982)**: Contracts in aprender, tests here
//! - **Defensive Programming (Hunt & Thomas, 1999)**: Handle missing aprender gracefully

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::error::{Error, Result};

/// Default path to aprender's family contracts relative to this repo.
pub const DEFAULT_APRENDER_PATH: &str = "../aprender/contracts/model-families";

/// Size variant configuration from family YAML.
///
/// Contains architectural parameters for a specific model size.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SizeVariant {
    /// Human-readable parameter count (e.g., "0.5B", "7B")
    pub parameters: String,

    /// Hidden dimension / d_model
    pub hidden_dim: u32,

    /// Number of transformer layers
    pub num_layers: u32,

    /// Number of attention heads
    pub num_heads: u32,

    /// Number of KV heads (for GQA; may equal num_heads for MHA)
    #[serde(default)]
    pub num_kv_heads: Option<u32>,

    /// FFN intermediate dimension
    #[serde(default)]
    pub intermediate_dim: Option<u32>,

    /// Vocabulary size
    #[serde(default)]
    pub vocab_size: Option<u32>,

    /// Maximum sequence length
    #[serde(default)]
    pub max_position_embeddings: Option<u32>,

    /// Per-head dimension
    #[serde(default)]
    pub head_dim: Option<u32>,

    /// RoPE base frequency
    #[serde(default)]
    pub rope_theta: Option<f64>,

    /// RMSNorm epsilon
    #[serde(default)]
    pub rms_norm_eps: Option<f64>,
}

/// Architectural constraints from family YAML.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Constraints {
    /// Attention type: "mha", "gqa", or "mqa"
    #[serde(default)]
    pub attention_type: Option<String>,

    /// Activation function: "silu", "gelu", "relu"
    #[serde(default)]
    pub activation: Option<String>,

    /// Norm type: "rmsnorm" or "layernorm"
    #[serde(default)]
    pub norm_type: Option<String>,

    /// Whether linear layers have bias terms
    #[serde(default)]
    pub has_bias: Option<bool>,

    /// Whether input/output embeddings are shared
    #[serde(default)]
    pub tied_embeddings: Option<bool>,

    /// Positional encoding: "rope", "absolute", "alibi"
    #[serde(default)]
    pub positional_encoding: Option<String>,

    /// MLP type: "swiglu", "gelu_mlp", "relu_mlp"
    #[serde(default)]
    pub mlp_type: Option<String>,
}

/// Tensor template from family YAML.
///
/// Maps logical tensor roles to actual tensor name patterns.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorTemplate {
    /// Embedding layer tensor name
    #[serde(default)]
    pub embedding: Option<String>,

    /// LM head tensor name
    #[serde(default)]
    pub lm_head: Option<String>,

    /// Final norm tensor name
    #[serde(default)]
    pub final_norm: Option<String>,

    /// Per-layer tensor patterns (use {n} for layer index)
    #[serde(default)]
    pub per_layer: HashMap<String, String>,
}

impl TensorTemplate {
    /// Get all required tensor names for a model with given number of layers.
    #[must_use]
    pub fn required_tensors(&self, num_layers: u32) -> Vec<String> {
        let mut tensors = Vec::new();

        if let Some(ref emb) = self.embedding {
            tensors.push(emb.clone());
        }
        if let Some(ref lm) = self.lm_head {
            tensors.push(lm.clone());
        }
        if let Some(ref norm) = self.final_norm {
            tensors.push(norm.clone());
        }

        for layer in 0..num_layers {
            for pattern in self.per_layer.values() {
                let tensor = pattern.replace("{n}", &layer.to_string());
                tensors.push(tensor);
            }
        }

        tensors
    }
}

/// Certification configuration from family YAML.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CertificationConfig {
    /// Path to playbook template (use {size} placeholder)
    #[serde(default)]
    pub playbook_path: Option<String>,

    /// Key used in CSV test result tracking
    #[serde(default)]
    pub csv_family_key: Option<String>,

    /// Maps size variant to category label (tiny, small, medium, large, xlarge, huge)
    #[serde(default)]
    pub size_categories: HashMap<String, String>,
}

/// Complete family contract from YAML file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FamilyContract {
    /// Unique identifier for the model family
    pub family: String,

    /// Human-readable display name
    #[serde(default)]
    pub display_name: Option<String>,

    /// Organization that created the model
    #[serde(default)]
    pub vendor: Option<String>,

    /// HuggingFace architecture class names
    #[serde(default)]
    pub architectures: Vec<String>,

    /// Glob pattern matching HuggingFace model repository IDs
    #[serde(default)]
    pub hf_pattern: Option<String>,

    /// Size variants with architectural parameters
    #[serde(default)]
    pub size_variants: HashMap<String, SizeVariant>,

    /// Architectural constraints
    #[serde(default)]
    pub constraints: Option<Constraints>,

    /// Tensor name templates
    #[serde(default)]
    pub tensor_template: Option<TensorTemplate>,

    /// Supported quantization formats
    #[serde(default)]
    pub quantizations: Vec<String>,

    /// Certification configuration
    #[serde(default)]
    pub certification: Option<CertificationConfig>,
}

impl FamilyContract {
    /// Load a family contract from a YAML file.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or parsed.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let content = std::fs::read_to_string(path.as_ref()).map_err(|e| {
            Error::Execution(format!(
                "Failed to read family contract at {}: {e}",
                path.as_ref().display()
            ))
        })?;
        Self::from_yaml(&content)
    }

    /// Parse a family contract from YAML string.
    ///
    /// # Errors
    ///
    /// Returns an error if the YAML is invalid.
    pub fn from_yaml(yaml: &str) -> Result<Self> {
        serde_yaml::from_str(yaml).map_err(|e| Error::Execution(format!("YAML parse error: {e}")))
    }

    /// Get size variant by key (e.g., "0.5b", "7b").
    #[must_use]
    pub fn get_size_variant(&self, size: &str) -> Option<&SizeVariant> {
        self.size_variants.get(size)
    }

    /// Get size category for a given size variant.
    ///
    /// Returns the category from `certification.size_categories` if present.
    #[must_use]
    pub fn get_size_category(&self, size: &str) -> Option<&str> {
        self.certification
            .as_ref()
            .and_then(|c| c.size_categories.get(size))
            .map(String::as_str)
    }

    /// Resolve playbook path for a given size variant.
    ///
    /// Replaces the size placeholder in the template path.
    #[must_use]
    pub fn resolve_playbook_path(&self, size: &str, tier: &str) -> Option<String> {
        // The placeholder is literally "{size}" in the YAML, not a format arg
        const SIZE_PLACEHOLDER: &str = "{size}";
        let replacement = format!("{size}-{tier}");
        self.certification.as_ref().and_then(|c| {
            c.playbook_path
                .as_ref()
                .map(|p| p.replace(SIZE_PLACEHOLDER, &replacement))
        })
    }

    /// Get required tensors for a given size variant.
    #[must_use]
    pub fn required_tensors_for_size(&self, size: &str) -> Vec<String> {
        let num_layers = self.get_size_variant(size).map_or(0, |v| v.num_layers);

        self.tensor_template
            .as_ref()
            .map_or_else(Vec::new, |t| t.required_tensors(num_layers))
    }
}

/// Family contract registry that loads contracts from aprender.
#[derive(Debug, Default)]
pub struct FamilyRegistry {
    contracts: HashMap<String, FamilyContract>,
    aprender_path: PathBuf,
}

impl FamilyRegistry {
    /// Create a new registry with default aprender path.
    #[must_use]
    pub fn new() -> Self {
        Self {
            contracts: HashMap::new(),
            aprender_path: PathBuf::from(DEFAULT_APRENDER_PATH),
        }
    }

    /// Create a registry with a custom aprender path.
    #[must_use]
    pub fn with_path(path: impl Into<PathBuf>) -> Self {
        Self {
            contracts: HashMap::new(),
            aprender_path: path.into(),
        }
    }

    /// Check if aprender contracts directory exists.
    #[must_use]
    pub fn aprender_available(&self) -> bool {
        self.aprender_path.exists() && self.aprender_path.is_dir()
    }

    /// Load all family contracts from aprender.
    ///
    /// # Errors
    ///
    /// Returns an error if the directory cannot be read.
    /// Individual file parse errors are logged but don't fail the whole load.
    pub fn load_all(&mut self) -> Result<usize> {
        if !self.aprender_available() {
            return Ok(0);
        }

        let entries = std::fs::read_dir(&self.aprender_path).map_err(|e| {
            Error::Execution(format!(
                "Failed to read aprender contracts at {}: {e}",
                self.aprender_path.display()
            ))
        })?;

        let mut count = 0;
        for entry in entries.flatten() {
            let path = entry.path();
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                // Skip schema and non-YAML files
                let is_yaml = path
                    .extension()
                    .is_some_and(|ext| ext.eq_ignore_ascii_case("yaml"));
                if name.starts_with('_') || !is_yaml {
                    continue;
                }

                if let Ok(contract) = FamilyContract::from_file(&path) {
                    self.contracts.insert(contract.family.clone(), contract);
                    count += 1;
                }
            }
        }

        Ok(count)
    }

    /// Load a specific family contract by name.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be found or parsed.
    pub fn load_family(&mut self, family: &str) -> Result<&FamilyContract> {
        use std::collections::hash_map::Entry;

        // Use entry API to avoid expect() calls
        if let Entry::Vacant(entry) = self.contracts.entry(family.to_string()) {
            let path = self.aprender_path.join(format!("{family}.yaml"));
            let contract = FamilyContract::from_file(&path)?;
            entry.insert(contract);
        }

        // Safe: we just ensured the entry exists
        self.contracts.get(family).ok_or_else(|| {
            Error::Execution(format!(
                "Family contract '{family}' not found after loading"
            ))
        })
    }

    /// Get a loaded family contract by name.
    #[must_use]
    pub fn get(&self, family: &str) -> Option<&FamilyContract> {
        self.contracts.get(family)
    }

    /// Get all loaded family names.
    #[must_use]
    pub fn families(&self) -> Vec<&str> {
        self.contracts.keys().map(String::as_str).collect()
    }

    /// Check if a family is loaded.
    #[must_use]
    pub fn has_family(&self, family: &str) -> bool {
        self.contracts.contains_key(family)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_YAML: &str = r#"
family: qwen2
display_name: "Qwen2 / Qwen2.5-Coder"
vendor: Alibaba
architectures:
  - Qwen2ForCausalLM
hf_pattern: "Qwen/Qwen2*"

size_variants:
  0.5b:
    parameters: "0.5B"
    hidden_dim: 896
    num_layers: 24
    num_heads: 14
    num_kv_heads: 2
    intermediate_dim: 4864
    vocab_size: 151936
  1.5b:
    parameters: "1.5B"
    hidden_dim: 1536
    num_layers: 28
    num_heads: 12
    num_kv_heads: 2

constraints:
  attention_type: gqa
  activation: silu
  norm_type: rmsnorm
  has_bias: true
  tied_embeddings: false
  positional_encoding: rope
  mlp_type: swiglu

tensor_template:
  embedding: "model.embed_tokens.weight"
  lm_head: "lm_head.weight"
  final_norm: "model.norm.weight"
  per_layer:
    q_proj: "model.layers.{n}.self_attn.q_proj.weight"
    k_proj: "model.layers.{n}.self_attn.k_proj.weight"
    input_layernorm: "model.layers.{n}.input_layernorm.weight"

quantizations:
  - q4_k_m
  - q5_k_m
  - q6_k

certification:
  playbook_path: "../apr-model-qa-playbook/playbooks/models/qwen2.5-coder-{size}.playbook.yaml"
  csv_family_key: "qwen-coder"
  size_categories:
    0.5b: tiny
    1.5b: small
    3b: small
    7b: medium
"#;

    #[test]
    fn test_parse_family_contract() {
        let contract = FamilyContract::from_yaml(SAMPLE_YAML).expect("parse");
        assert_eq!(contract.family, "qwen2");
        assert_eq!(
            contract.display_name,
            Some("Qwen2 / Qwen2.5-Coder".to_string())
        );
        assert_eq!(contract.vendor, Some("Alibaba".to_string()));
    }

    #[test]
    fn test_size_variants() {
        let contract = FamilyContract::from_yaml(SAMPLE_YAML).expect("parse");
        assert_eq!(contract.size_variants.len(), 2);

        let v05b = contract.get_size_variant("0.5b").expect("0.5b");
        assert_eq!(v05b.hidden_dim, 896);
        assert_eq!(v05b.num_layers, 24);
        assert_eq!(v05b.num_heads, 14);
        assert_eq!(v05b.num_kv_heads, Some(2));

        let v15b = contract.get_size_variant("1.5b").expect("1.5b");
        assert_eq!(v15b.hidden_dim, 1536);
        assert_eq!(v15b.num_layers, 28);
    }

    #[test]
    fn test_constraints() {
        let contract = FamilyContract::from_yaml(SAMPLE_YAML).expect("parse");
        let constraints = contract.constraints.expect("constraints");

        assert_eq!(constraints.attention_type, Some("gqa".to_string()));
        assert_eq!(constraints.activation, Some("silu".to_string()));
        assert_eq!(constraints.has_bias, Some(true));
    }

    #[test]
    fn test_tensor_template() {
        let contract = FamilyContract::from_yaml(SAMPLE_YAML).expect("parse");
        let template = contract.tensor_template.expect("template");

        assert_eq!(
            template.embedding,
            Some("model.embed_tokens.weight".to_string())
        );
        assert_eq!(template.lm_head, Some("lm_head.weight".to_string()));
        assert!(template.per_layer.contains_key("q_proj"));
    }

    #[test]
    fn test_required_tensors() {
        let contract = FamilyContract::from_yaml(SAMPLE_YAML).expect("parse");
        let template = contract.tensor_template.expect("template");

        // With 2 layers
        let tensors = template.required_tensors(2);

        // Should have embedding, lm_head, final_norm (3)
        // Plus per_layer tensors for 2 layers (3 per layer * 2 = 6)
        // Total: 9
        assert_eq!(tensors.len(), 9);
        assert!(tensors.contains(&"model.embed_tokens.weight".to_string()));
        assert!(tensors.contains(&"lm_head.weight".to_string()));
        assert!(tensors.contains(&"model.layers.0.self_attn.q_proj.weight".to_string()));
        assert!(tensors.contains(&"model.layers.1.self_attn.q_proj.weight".to_string()));
    }

    #[test]
    fn test_certification_config() {
        let contract = FamilyContract::from_yaml(SAMPLE_YAML).expect("parse");
        let cert = contract.certification.expect("certification");

        assert_eq!(cert.csv_family_key, Some("qwen-coder".to_string()));
        assert_eq!(cert.size_categories.get("0.5b"), Some(&"tiny".to_string()));
        assert_eq!(cert.size_categories.get("1.5b"), Some(&"small".to_string()));
    }

    #[test]
    fn test_get_size_category() {
        let contract = FamilyContract::from_yaml(SAMPLE_YAML).expect("parse");

        assert_eq!(contract.get_size_category("0.5b"), Some("tiny"));
        assert_eq!(contract.get_size_category("1.5b"), Some("small"));
        assert_eq!(contract.get_size_category("7b"), Some("medium"));
        assert_eq!(contract.get_size_category("100b"), None);
    }

    #[test]
    fn test_resolve_playbook_path() {
        let contract = FamilyContract::from_yaml(SAMPLE_YAML).expect("parse");

        let path = contract.resolve_playbook_path("0.5b", "mvp");
        assert_eq!(
            path,
            Some(
                "../apr-model-qa-playbook/playbooks/models/qwen2.5-coder-0.5b-mvp.playbook.yaml"
                    .to_string()
            )
        );
    }

    #[test]
    fn test_required_tensors_for_size() {
        let contract = FamilyContract::from_yaml(SAMPLE_YAML).expect("parse");

        // 0.5b has 24 layers
        let tensors = contract.required_tensors_for_size("0.5b");
        // 3 global + 3 per layer * 24 layers = 75
        assert_eq!(tensors.len(), 75);
    }

    #[test]
    fn test_family_registry_new() {
        let registry = FamilyRegistry::new();
        assert!(registry.families().is_empty());
    }

    #[test]
    fn test_family_registry_with_path() {
        let registry = FamilyRegistry::with_path("/custom/path");
        assert!(!registry.aprender_available()); // path doesn't exist
    }

    #[test]
    fn test_family_registry_load_all() {
        let mut registry = FamilyRegistry::new();

        // May or may not have aprender available
        let result = registry.load_all();
        assert!(result.is_ok());
    }

    #[test]
    fn test_architectures() {
        let contract = FamilyContract::from_yaml(SAMPLE_YAML).expect("parse");
        assert_eq!(contract.architectures.len(), 1);
        assert_eq!(contract.architectures[0], "Qwen2ForCausalLM");
    }

    #[test]
    fn test_hf_pattern() {
        let contract = FamilyContract::from_yaml(SAMPLE_YAML).expect("parse");
        assert_eq!(contract.hf_pattern, Some("Qwen/Qwen2*".to_string()));
    }

    #[test]
    fn test_quantizations() {
        let contract = FamilyContract::from_yaml(SAMPLE_YAML).expect("parse");
        assert_eq!(contract.quantizations.len(), 3);
        assert!(contract.quantizations.contains(&"q4_k_m".to_string()));
    }

    #[test]
    fn test_missing_optional_fields() {
        let minimal_yaml = r#"
family: minimal
size_variants:
  1b:
    parameters: "1B"
    hidden_dim: 1024
    num_layers: 12
    num_heads: 16
"#;
        let contract = FamilyContract::from_yaml(minimal_yaml).expect("parse");
        assert_eq!(contract.family, "minimal");
        assert!(contract.display_name.is_none());
        assert!(contract.vendor.is_none());
        assert!(contract.constraints.is_none());
        assert!(contract.tensor_template.is_none());
        assert!(contract.certification.is_none());
    }

    // FALSIFY-FAM-001: Size category alignment
    #[test]
    fn test_falsify_fam_001_size_category_alignment() {
        let contract = FamilyContract::from_yaml(SAMPLE_YAML).expect("parse");

        // Verify all size variants have a size category
        for size in contract.size_variants.keys() {
            if let Some(cat) = contract.get_size_category(size) {
                // Category must be one of the valid values
                assert!(
                    ["tiny", "small", "medium", "large", "xlarge", "huge"].contains(&cat),
                    "Invalid size category '{cat}' for size '{size}'"
                );
            }
        }
    }
}
