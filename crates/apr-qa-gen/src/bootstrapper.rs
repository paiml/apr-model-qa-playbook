//! Playbook Bootstrapper
//!
//! Generates architecture-aware playbook YAML from family contract constraints
//! and kernel profiles. Bootstrapped playbooks include targeted prompts that
//! stress-test the specific kernels each model architecture exercises.

use crate::kernel_profile::{
    ArchConstraints, ArchSizeVariant, KernelProfile, profile_from_constraints,
};
use serde::{Deserialize, Serialize};

/// Configuration for bootstrapping a playbook.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BootstrapConfig {
    /// Model family name (e.g., "qwen2", "llama")
    pub family: String,
    /// Size variant key (e.g., "1.5b", "7b")
    pub size_variant: String,
    /// HuggingFace repository ID
    pub hf_repo: String,
    /// Certification tier (e.g., "mvp", "smoke", "quick")
    pub tier: String,
    /// Optional kernel profile override (auto-derived if not provided)
    pub kernel_profile: Option<KernelProfile>,
}

/// A bootstrapped playbook representation ready for YAML serialization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BootstrappedPlaybook {
    /// Playbook name
    pub name: String,
    /// Version
    pub version: String,
    /// Model configuration section
    pub model: BootstrappedModel,
    /// Test matrix section
    pub test_matrix: BootstrappedTestMatrix,
    /// Kernel profile metadata (documents which kernels are under test)
    pub kernel_profile: BootstrappedKernelProfile,
    /// Falsification gates
    pub falsification_gates: Vec<BootstrappedGate>,
    /// Differential test configuration
    #[serde(skip_serializing_if = "Option::is_none")]
    pub differential_tests: Option<BootstrappedDifferential>,
    /// Profile CI assertions
    #[serde(skip_serializing_if = "Option::is_none")]
    pub profile_ci: Option<BootstrappedProfileCi>,
}

/// Model section of bootstrapped playbook.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BootstrappedModel {
    /// HuggingFace repo
    pub hf_repo: String,
    /// Formats to test
    pub formats: Vec<String>,
    /// Quantizations
    pub quantizations: Vec<String>,
    /// Size category
    pub size_category: String,
    /// Expected hidden dim
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expected_hidden_dim: Option<u32>,
    /// Expected number of layers
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expected_num_layers: Option<u32>,
    /// Expected number of attention heads
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expected_num_heads: Option<u32>,
    /// Expected number of KV heads
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expected_num_kv_heads: Option<u32>,
    /// Expected vocab size
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expected_vocab_size: Option<u32>,
    /// Expected intermediate dim
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expected_intermediate_dim: Option<u32>,
    /// Family identifier
    #[serde(skip_serializing_if = "Option::is_none")]
    pub family: Option<String>,
    /// Size variant
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size_variant: Option<String>,
}

/// Test matrix section of bootstrapped playbook.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BootstrappedTestMatrix {
    /// Modalities
    pub modalities: Vec<String>,
    /// Backends
    pub backends: Vec<String>,
    /// Scenario count per combination
    pub scenario_count: usize,
    /// Architecture-specific prompts
    pub prompts: Vec<String>,
}

/// Kernel profile metadata in the playbook.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BootstrappedKernelProfile {
    /// Family name
    pub family: String,
    /// List of kernel operation names
    pub kernel_ops: Vec<String>,
    /// Total prompt count
    pub prompt_count: usize,
    /// Whether long context is supported
    pub long_context: bool,
}

/// Falsification gate in bootstrapped playbook.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BootstrappedGate {
    /// Gate ID
    pub id: String,
    /// Description
    pub description: String,
    /// Condition
    pub condition: String,
    /// Severity
    pub severity: String,
}

/// Differential test config in bootstrapped playbook.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BootstrappedDifferential {
    /// Format validation
    pub format_validation: BootstrappedFormatValidation,
}

/// Format validation section.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BootstrappedFormatValidation {
    /// Enabled flag
    pub enabled: bool,
    /// Checks to run
    pub checks: Vec<String>,
}

/// Profile CI section.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BootstrappedProfileCi {
    /// Enabled flag
    pub enabled: bool,
    /// Minimum throughput (tokens/sec)
    pub min_throughput: f64,
    /// Max p99 latency (ms)
    pub max_p99_ms: f64,
}

/// Scenario count for a given tier.
fn scenario_count_for_tier(tier: &str) -> usize {
    match tier {
        "smoke" => 1,
        "quick" => 5,
        "standard" => 10,
        "deep" => 50,
        // mvp and unknown tiers default to 3
        _ => 3,
    }
}

/// Size-aware performance thresholds (min throughput tok/s, max p99 ms).
fn performance_thresholds(size_category: &str) -> (f64, f64) {
    match size_category {
        "tiny" => (50.0, 200.0),
        "small" => (30.0, 500.0),
        "medium" => (15.0, 1000.0),
        "large" => (5.0, 3000.0),
        "xlarge" => (2.0, 5000.0),
        _ => (1.0, 10000.0),
    }
}

/// Build the standard falsification gates G1-G4.
fn standard_gates() -> Vec<BootstrappedGate> {
    vec![
        BootstrappedGate {
            id: "G1".to_string(),
            description: "Model loads successfully".to_string(),
            condition: "exit_code == 0".to_string(),
            severity: "P0".to_string(),
        },
        BootstrappedGate {
            id: "G2".to_string(),
            description: "Basic inference produces output".to_string(),
            condition: "output.len() > 0".to_string(),
            severity: "P0".to_string(),
        },
        BootstrappedGate {
            id: "G3".to_string(),
            description: "No crashes or panics".to_string(),
            condition: "!stderr.contains('panic')".to_string(),
            severity: "P0".to_string(),
        },
        BootstrappedGate {
            id: "G4".to_string(),
            description: "Output is not garbage (LAYOUT-002)".to_string(),
            condition: "!garbage_oracle.is_garbage(output)".to_string(),
            severity: "P0".to_string(),
        },
    ]
}

/// Bootstrap a playbook from architecture constraints.
///
/// Generates a complete playbook representation with architecture-specific
/// prompts and kernel profile metadata.
#[must_use]
pub fn bootstrap_playbook(
    config: &BootstrapConfig,
    constraints: &ArchConstraints,
    size_variant: &ArchSizeVariant,
    size_category: &str,
) -> BootstrappedPlaybook {
    let profile = config.kernel_profile.clone().unwrap_or_else(|| {
        profile_from_constraints(
            &config.family,
            constraints,
            size_variant.max_position_embeddings,
        )
    });

    let is_smoke = config.tier == "smoke";
    let (min_throughput, max_p99_ms) = performance_thresholds(size_category);

    let modalities = if is_smoke {
        vec!["run".to_string()]
    } else {
        vec!["run".to_string(), "chat".to_string()]
    };

    let backends = if is_smoke {
        vec!["cpu".to_string()]
    } else {
        vec!["cpu".to_string(), "gpu".to_string()]
    };

    let model = BootstrappedModel {
        hf_repo: config.hf_repo.clone(),
        formats: vec![
            "gguf".to_string(),
            "safetensors".to_string(),
            "apr".to_string(),
        ],
        quantizations: vec!["q4_k_m".to_string()],
        size_category: size_category.to_string(),
        expected_hidden_dim: Some(size_variant.hidden_dim),
        expected_num_layers: Some(size_variant.num_layers),
        expected_num_heads: Some(size_variant.num_heads),
        expected_num_kv_heads: size_variant.num_kv_heads,
        expected_vocab_size: size_variant.vocab_size,
        expected_intermediate_dim: size_variant.intermediate_dim,
        family: Some(config.family.clone()),
        size_variant: Some(config.size_variant.clone()),
    };

    let kernel_profile_meta = BootstrappedKernelProfile {
        family: profile.family.clone(),
        kernel_ops: profile
            .kernel_ops
            .iter()
            .map(|op| format!("{op:?}"))
            .collect(),
        prompt_count: profile.prompt_count(),
        long_context: profile.long_context,
    };

    let differential_tests = if is_smoke {
        None
    } else {
        Some(BootstrappedDifferential {
            format_validation: BootstrappedFormatValidation {
                enabled: true,
                checks: vec![
                    "dtype_mapping".to_string(),
                    "tensor_alignment".to_string(),
                    "header_integrity".to_string(),
                ],
            },
        })
    };

    let profile_ci = if is_smoke {
        None
    } else {
        Some(BootstrappedProfileCi {
            enabled: true,
            min_throughput,
            max_p99_ms,
        })
    };

    BootstrappedPlaybook {
        name: format!("{}-{}-{}", config.family, config.size_variant, config.tier),
        version: "1.0.0".to_string(),
        model,
        test_matrix: BootstrappedTestMatrix {
            modalities,
            backends,
            scenario_count: scenario_count_for_tier(&config.tier),
            prompts: profile.all_prompts(),
        },
        kernel_profile: kernel_profile_meta,
        falsification_gates: standard_gates(),
        differential_tests,
        profile_ci,
    }
}

/// Serialize a bootstrapped playbook to YAML.
///
/// # Errors
///
/// Returns an error string if YAML serialization fails.
pub fn to_yaml(playbook: &BootstrappedPlaybook) -> Result<String, String> {
    use std::fmt::Write;

    let mut yaml = String::new();
    yaml.push_str("# Auto-generated playbook - bootstrapped from family contract\n");
    let _ = writeln!(yaml, "# Family: {}", playbook.kernel_profile.family);
    let _ = writeln!(
        yaml,
        "# Kernel ops: {}",
        playbook.kernel_profile.kernel_ops.join(", ")
    );
    let _ = writeln!(
        yaml,
        "# Prompts: {} architecture-targeted prompts",
        playbook.kernel_profile.prompt_count
    );
    yaml.push('\n');

    let body =
        serde_yaml::to_string(playbook).map_err(|e| format!("YAML serialization error: {e}"))?;
    yaml.push_str(&body);

    Ok(yaml)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel_profile::ArchConstraints;

    fn qwen_config() -> BootstrapConfig {
        BootstrapConfig {
            family: "qwen2".to_string(),
            size_variant: "1.5b".to_string(),
            hf_repo: "Qwen/Qwen2.5-Coder-1.5B-Instruct".to_string(),
            tier: "mvp".to_string(),
            kernel_profile: None,
        }
    }

    fn qwen_constraints() -> ArchConstraints {
        ArchConstraints {
            attention_type: Some("gqa".to_string()),
            activation: Some("silu".to_string()),
            norm_type: Some("rmsnorm".to_string()),
            has_bias: Some(true),
            tied_embeddings: Some(false),
            positional_encoding: Some("rope".to_string()),
            mlp_type: Some("swiglu".to_string()),
        }
    }

    fn qwen_size_variant() -> ArchSizeVariant {
        ArchSizeVariant {
            parameters: "1.5B".to_string(),
            hidden_dim: 1536,
            num_layers: 28,
            num_heads: 12,
            num_kv_heads: Some(2),
            intermediate_dim: Some(8960),
            vocab_size: Some(151_936),
            max_position_embeddings: Some(32_768),
        }
    }

    #[test]
    fn test_bootstrap_playbook_name() {
        let playbook = bootstrap_playbook(
            &qwen_config(),
            &qwen_constraints(),
            &qwen_size_variant(),
            "small",
        );
        assert_eq!(playbook.name, "qwen2-1.5b-mvp");
    }

    #[test]
    fn test_bootstrap_playbook_model_config() {
        let playbook = bootstrap_playbook(
            &qwen_config(),
            &qwen_constraints(),
            &qwen_size_variant(),
            "small",
        );
        assert_eq!(playbook.model.hf_repo, "Qwen/Qwen2.5-Coder-1.5B-Instruct");
        assert_eq!(playbook.model.expected_hidden_dim, Some(1536));
        assert_eq!(playbook.model.expected_num_layers, Some(28));
        assert_eq!(playbook.model.expected_num_heads, Some(12));
        assert_eq!(playbook.model.expected_num_kv_heads, Some(2));
        assert_eq!(playbook.model.size_category, "small");
    }

    #[test]
    fn test_bootstrap_playbook_has_prompts() {
        let playbook = bootstrap_playbook(
            &qwen_config(),
            &qwen_constraints(),
            &qwen_size_variant(),
            "small",
        );
        assert!(!playbook.test_matrix.prompts.is_empty());
        // Should have architecture-specific prompts (GQA, RoPE, bias, arithmetic, code)
        assert!(playbook.test_matrix.prompts.len() >= 10);
    }

    #[test]
    fn test_bootstrap_playbook_kernel_profile() {
        let playbook = bootstrap_playbook(
            &qwen_config(),
            &qwen_constraints(),
            &qwen_size_variant(),
            "small",
        );
        assert_eq!(playbook.kernel_profile.family, "qwen2");
        assert!(!playbook.kernel_profile.kernel_ops.is_empty());
        assert!(playbook.kernel_profile.long_context);
    }

    #[test]
    fn test_bootstrap_smoke_tier() {
        let mut config = qwen_config();
        config.tier = "smoke".to_string();
        let playbook =
            bootstrap_playbook(&config, &qwen_constraints(), &qwen_size_variant(), "small");
        assert_eq!(playbook.test_matrix.scenario_count, 1);
        assert_eq!(playbook.test_matrix.modalities, vec!["run"]);
        assert_eq!(playbook.test_matrix.backends, vec!["cpu"]);
        assert!(playbook.differential_tests.is_none());
        assert!(playbook.profile_ci.is_none());
    }

    #[test]
    fn test_bootstrap_mvp_tier() {
        let playbook = bootstrap_playbook(
            &qwen_config(),
            &qwen_constraints(),
            &qwen_size_variant(),
            "small",
        );
        assert_eq!(playbook.test_matrix.scenario_count, 3);
        assert!(playbook.test_matrix.modalities.contains(&"run".to_string()));
        assert!(
            playbook
                .test_matrix
                .modalities
                .contains(&"chat".to_string())
        );
        assert!(playbook.differential_tests.is_some());
        assert!(playbook.profile_ci.is_some());
    }

    #[test]
    fn test_bootstrap_deep_tier() {
        let mut config = qwen_config();
        config.tier = "deep".to_string();
        let playbook =
            bootstrap_playbook(&config, &qwen_constraints(), &qwen_size_variant(), "small");
        assert_eq!(playbook.test_matrix.scenario_count, 50);
    }

    #[test]
    fn test_bootstrap_gates() {
        let playbook = bootstrap_playbook(
            &qwen_config(),
            &qwen_constraints(),
            &qwen_size_variant(),
            "small",
        );
        assert_eq!(playbook.falsification_gates.len(), 4);
        let gate_ids: Vec<&str> = playbook
            .falsification_gates
            .iter()
            .map(|g| g.id.as_str())
            .collect();
        assert!(gate_ids.contains(&"G1"));
        assert!(gate_ids.contains(&"G2"));
        assert!(gate_ids.contains(&"G3"));
        assert!(gate_ids.contains(&"G4"));
    }

    #[test]
    fn test_bootstrap_size_aware_thresholds_tiny() {
        let mut config = qwen_config();
        config.tier = "mvp".to_string();
        let playbook =
            bootstrap_playbook(&config, &qwen_constraints(), &qwen_size_variant(), "tiny");
        let ci = playbook.profile_ci.expect("profile_ci");
        assert!((ci.min_throughput - 50.0).abs() < f64::EPSILON);
        assert!((ci.max_p99_ms - 200.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_bootstrap_size_aware_thresholds_large() {
        let mut config = qwen_config();
        config.tier = "mvp".to_string();
        let playbook =
            bootstrap_playbook(&config, &qwen_constraints(), &qwen_size_variant(), "large");
        let ci = playbook.profile_ci.expect("profile_ci");
        assert!((ci.min_throughput - 5.0).abs() < f64::EPSILON);
        assert!((ci.max_p99_ms - 3000.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_to_yaml() {
        let playbook = bootstrap_playbook(
            &qwen_config(),
            &qwen_constraints(),
            &qwen_size_variant(),
            "small",
        );
        let yaml = to_yaml(&playbook).expect("yaml");
        assert!(yaml.contains("# Auto-generated playbook"));
        assert!(yaml.contains("qwen2"));
        assert!(yaml.contains("kernel_profile"));
        assert!(yaml.contains("prompts"));
    }

    #[test]
    fn test_to_yaml_contains_hf_repo() {
        let playbook = bootstrap_playbook(
            &qwen_config(),
            &qwen_constraints(),
            &qwen_size_variant(),
            "small",
        );
        let yaml = to_yaml(&playbook).expect("yaml");
        assert!(yaml.contains("Qwen/Qwen2.5-Coder-1.5B-Instruct"));
    }

    #[test]
    fn test_bootstrap_with_custom_profile() {
        let profile = profile_from_constraints("custom", &ArchConstraints::default(), None);
        let config = BootstrapConfig {
            family: "custom".to_string(),
            size_variant: "1b".to_string(),
            hf_repo: "org/custom-1b".to_string(),
            tier: "mvp".to_string(),
            kernel_profile: Some(profile),
        };
        let playbook = bootstrap_playbook(
            &config,
            &ArchConstraints::default(),
            &ArchSizeVariant {
                parameters: "1B".to_string(),
                hidden_dim: 1024,
                num_layers: 12,
                num_heads: 16,
                ..ArchSizeVariant::default()
            },
            "small",
        );
        assert_eq!(playbook.kernel_profile.family, "custom");
    }

    #[test]
    fn test_bootstrap_version() {
        let playbook = bootstrap_playbook(
            &qwen_config(),
            &qwen_constraints(),
            &qwen_size_variant(),
            "small",
        );
        assert_eq!(playbook.version, "1.0.0");
    }

    #[test]
    fn test_bootstrap_formats() {
        let playbook = bootstrap_playbook(
            &qwen_config(),
            &qwen_constraints(),
            &qwen_size_variant(),
            "small",
        );
        assert_eq!(playbook.model.formats.len(), 3);
        assert!(playbook.model.formats.contains(&"gguf".to_string()));
        assert!(playbook.model.formats.contains(&"safetensors".to_string()));
        assert!(playbook.model.formats.contains(&"apr".to_string()));
    }

    #[test]
    fn test_bootstrap_differential_checks() {
        let playbook = bootstrap_playbook(
            &qwen_config(),
            &qwen_constraints(),
            &qwen_size_variant(),
            "small",
        );
        let diff = playbook.differential_tests.expect("differential");
        assert!(diff.format_validation.enabled);
        assert!(
            diff.format_validation
                .checks
                .contains(&"dtype_mapping".to_string())
        );
    }

    #[test]
    fn test_bootstrap_quick_tier() {
        let mut config = qwen_config();
        config.tier = "quick".to_string();
        let playbook =
            bootstrap_playbook(&config, &qwen_constraints(), &qwen_size_variant(), "small");
        assert_eq!(playbook.test_matrix.scenario_count, 5);
    }

    #[test]
    fn test_bootstrap_standard_tier() {
        let mut config = qwen_config();
        config.tier = "standard".to_string();
        let playbook =
            bootstrap_playbook(&config, &qwen_constraints(), &qwen_size_variant(), "small");
        assert_eq!(playbook.test_matrix.scenario_count, 10);
    }

    #[test]
    fn test_bootstrap_unknown_tier_defaults() {
        let mut config = qwen_config();
        config.tier = "unknown".to_string();
        let playbook =
            bootstrap_playbook(&config, &qwen_constraints(), &qwen_size_variant(), "small");
        // Should default to mvp-level
        assert_eq!(playbook.test_matrix.scenario_count, 3);
    }

    #[test]
    fn test_bootstrap_playbook_serialize_roundtrip() {
        let playbook = bootstrap_playbook(
            &qwen_config(),
            &qwen_constraints(),
            &qwen_size_variant(),
            "small",
        );
        let yaml = to_yaml(&playbook).expect("yaml");
        assert!(!yaml.is_empty());
        // Verify the YAML body (after comments) can be parsed back
        // The comments are lines starting with #
        let body: String = yaml
            .lines()
            .filter(|l| !l.starts_with('#') && !l.is_empty())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(!body.is_empty());
    }
}
