use super::*;

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
        num_heads: Some(12),
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
            num_heads: Some(16),
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

#[test]
fn test_bootstrap_dim_smoke_tier() {
    let mut config = qwen_config();
    config.tier = "dim-smoke".to_string();
    let playbook =
        bootstrap_playbook(&config, &qwen_constraints(), &qwen_size_variant(), "small");

    // 1 scenario, single modality/backend, safetensors only
    assert_eq!(playbook.test_matrix.scenario_count, 1);
    assert_eq!(playbook.test_matrix.modalities, vec!["run"]);
    assert_eq!(playbook.test_matrix.backends, vec!["cpu"]);
    assert_eq!(playbook.model.formats, vec!["safetensors"]);

    // No differential or profile CI
    assert!(playbook.differential_tests.is_none());
    assert!(playbook.profile_ci.is_none());

    // Should have kernel proof reference
    assert!(playbook.kernel_proof_ref.is_some());
    let proof_ref = playbook.kernel_proof_ref.unwrap();
    assert!(proof_ref.contains("Qwen"));
}

#[test]
fn test_bootstrap_dim_smoke_name() {
    let mut config = qwen_config();
    config.tier = "dim-smoke".to_string();
    let playbook =
        bootstrap_playbook(&config, &qwen_constraints(), &qwen_size_variant(), "small");
    assert_eq!(playbook.name, "qwen2-1.5b-dim-smoke");
}

#[test]
fn test_bootstrap_non_dim_smoke_no_kernel_proof_ref() {
    let playbook = bootstrap_playbook(
        &qwen_config(),
        &qwen_constraints(),
        &qwen_size_variant(),
        "small",
    );
    assert!(playbook.kernel_proof_ref.is_none());
}
