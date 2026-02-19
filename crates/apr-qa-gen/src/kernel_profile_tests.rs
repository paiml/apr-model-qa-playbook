use super::*;

use super::*;

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

fn falcon_constraints() -> ArchConstraints {
    ArchConstraints {
        attention_type: Some("mha".to_string()),
        activation: Some("gelu".to_string()),
        norm_type: Some("layernorm".to_string()),
        has_bias: Some(false),
        tied_embeddings: Some(false),
        positional_encoding: Some("alibi".to_string()),
        mlp_type: Some("gelu_mlp".to_string()),
    }
}

#[test]
fn test_qwen_profile_kernel_ops() {
    let profile = profile_from_constraints("qwen2", &qwen_constraints(), Some(32768));

    assert!(
        profile
            .kernel_ops
            .contains(&KernelOp::GroupedQueryAttention)
    );
    assert!(profile.kernel_ops.contains(&KernelOp::RmsNorm));
    assert!(profile.kernel_ops.contains(&KernelOp::Silu));
    assert!(profile.kernel_ops.contains(&KernelOp::SwiGlu));
    assert!(profile.kernel_ops.contains(&KernelOp::Rope));
    assert!(profile.kernel_ops.contains(&KernelOp::BiasAdd));
    assert!(profile.kernel_ops.contains(&KernelOp::FusedQ4kMatvec));
    // GQA model should not have MHA
    assert!(!profile.kernel_ops.contains(&KernelOp::MultiHeadAttention));
}

#[test]
fn test_falcon_profile_kernel_ops() {
    let profile = profile_from_constraints("falcon", &falcon_constraints(), Some(2048));

    assert!(profile.kernel_ops.contains(&KernelOp::MultiHeadAttention));
    assert!(profile.kernel_ops.contains(&KernelOp::LayerNorm));
    assert!(profile.kernel_ops.contains(&KernelOp::Gelu));
    assert!(profile.kernel_ops.contains(&KernelOp::Alibi));
    // Falcon should not have GQA, RMSNorm, SiLU, RoPE
    assert!(
        !profile
            .kernel_ops
            .contains(&KernelOp::GroupedQueryAttention)
    );
    assert!(!profile.kernel_ops.contains(&KernelOp::RmsNorm));
    assert!(!profile.kernel_ops.contains(&KernelOp::Silu));
    assert!(!profile.kernel_ops.contains(&KernelOp::Rope));
}

#[test]
fn test_qwen_long_context() {
    let profile = profile_from_constraints("qwen2", &qwen_constraints(), Some(32768));
    assert!(profile.long_context);
}

#[test]
fn test_falcon_no_long_context() {
    let profile = profile_from_constraints("falcon", &falcon_constraints(), Some(2048));
    assert!(!profile.long_context);
}

/// Helper: check if profile has a prompt category by name.
fn has_category(profile: &KernelProfile, name: &str) -> bool {
    profile.prompt_categories.iter().any(|c| c.name == name)
}

#[test]
fn test_qwen_has_gqa_prompts() {
    let profile = profile_from_constraints("qwen2", &qwen_constraints(), Some(32768));
    assert!(has_category(&profile, "gqa_multi_turn"));
    assert!(!has_category(&profile, "mha_long_dependency"));
}

#[test]
fn test_falcon_has_mha_prompts() {
    let profile = profile_from_constraints("falcon", &falcon_constraints(), Some(2048));
    assert!(has_category(&profile, "mha_long_dependency"));
    assert!(!has_category(&profile, "gqa_multi_turn"));
}

#[test]
fn test_rope_long_context_prompts_added() {
    let profile = profile_from_constraints("qwen2", &qwen_constraints(), Some(32768));
    assert!(has_category(&profile, "rope_long_context"));
}

#[test]
fn test_rope_short_context_no_long_prompts() {
    let mut constraints = qwen_constraints();
    constraints.positional_encoding = Some("rope".to_string());
    let profile = profile_from_constraints("qwen2-small", &constraints, Some(2048));
    assert!(!has_category(&profile, "rope_long_context"));
}

#[test]
fn test_bias_prompts_when_has_bias() {
    let profile = profile_from_constraints("qwen2", &qwen_constraints(), Some(4096));
    assert!(has_category(&profile, "bias_precision"));
}

#[test]
fn test_no_bias_prompts_when_no_bias() {
    let profile = profile_from_constraints("falcon", &falcon_constraints(), Some(2048));
    assert!(!has_category(&profile, "bias_precision"));
}

#[test]
fn test_always_has_arithmetic_and_code() {
    let profile = profile_from_constraints("test", &ArchConstraints::default(), None);
    assert!(has_category(&profile, "arithmetic_verification"));
    assert!(has_category(&profile, "code_completion"));
}

#[test]
fn test_all_prompts_flattened() {
    let profile = profile_from_constraints("qwen2", &qwen_constraints(), Some(32768));
    let all = profile.all_prompts();
    assert!(!all.is_empty());
    assert_eq!(all.len(), profile.prompt_count());
}

#[test]
fn test_prompt_count() {
    let profile = profile_from_constraints("qwen2", &qwen_constraints(), Some(32768));
    assert!(profile.prompt_count() > 0);
    // Should have prompts from: gqa, rope_long_context, bias, arithmetic, code
    assert!(profile.prompt_count() >= 15);
}

#[test]
fn test_default_constraints_profile() {
    let profile = profile_from_constraints("unknown", &ArchConstraints::default(), None);
    // Should default to MHA, RMSNorm, SiLU
    assert!(profile.kernel_ops.contains(&KernelOp::MultiHeadAttention));
    assert!(profile.kernel_ops.contains(&KernelOp::RmsNorm));
    assert!(profile.kernel_ops.contains(&KernelOp::Silu));
    assert!(!profile.long_context);
}

#[test]
fn test_tied_embeddings() {
    let constraints = ArchConstraints {
        tied_embeddings: Some(true),
        ..ArchConstraints::default()
    };
    let profile = profile_from_constraints("test", &constraints, None);
    assert!(profile.kernel_ops.contains(&KernelOp::TiedEmbeddings));
}

#[test]
fn test_no_tied_embeddings() {
    let constraints = ArchConstraints {
        tied_embeddings: Some(false),
        ..ArchConstraints::default()
    };
    let profile = profile_from_constraints("test", &constraints, None);
    assert!(!profile.kernel_ops.contains(&KernelOp::TiedEmbeddings));
}

#[test]
fn test_mqa_attention() {
    let constraints = ArchConstraints {
        attention_type: Some("mqa".to_string()),
        ..ArchConstraints::default()
    };
    let profile = profile_from_constraints("falcon40b", &constraints, None);
    assert!(profile.kernel_ops.contains(&KernelOp::MultiQueryAttention));
    assert!(has_category(&profile, "mqa_kv_efficiency"));
}

#[test]
fn test_kernel_op_display() {
    assert_eq!(
        format!("{}", KernelOp::GroupedQueryAttention),
        "Grouped-query attention (GQA)"
    );
    assert_eq!(format!("{}", KernelOp::RmsNorm), "RMS normalization");
}

#[test]
fn test_kernel_op_description() {
    assert_eq!(
        KernelOp::FusedQ4kMatvec.description(),
        "Fused Q4K quantized matrix-vector multiply"
    );
    assert_eq!(
        KernelOp::TiedEmbeddings.description(),
        "Tied input/output embeddings"
    );
}

#[test]
fn test_profile_family_name() {
    let profile = profile_from_constraints("qwen2", &qwen_constraints(), Some(32768));
    assert_eq!(profile.family, "qwen2");
}

#[test]
fn test_suggested_max_tokens_long_context() {
    let profile = profile_from_constraints("qwen2", &qwen_constraints(), Some(32768));
    assert_eq!(profile.suggested_max_tokens, 128);
}

#[test]
fn test_suggested_max_tokens_short_context() {
    let profile = profile_from_constraints("falcon", &falcon_constraints(), Some(2048));
    assert_eq!(profile.suggested_max_tokens, 64);
}

#[test]
fn test_kernel_op_serialize() {
    let op = KernelOp::GroupedQueryAttention;
    let json = serde_json::to_string(&op).expect("serialize");
    assert_eq!(json, "\"grouped_query_attention\"");
}

#[test]
fn test_kernel_op_deserialize() {
    let op: KernelOp = serde_json::from_str("\"fused_q4k_matvec\"").expect("deserialize");
    assert_eq!(op, KernelOp::FusedQ4kMatvec);
}

#[test]
fn test_arch_constraints_default() {
    let c = ArchConstraints::default();
    assert!(c.attention_type.is_none());
    assert!(c.activation.is_none());
    assert!(c.norm_type.is_none());
    assert!(c.has_bias.is_none());
    assert!(c.tied_embeddings.is_none());
    assert!(c.positional_encoding.is_none());
    assert!(c.mlp_type.is_none());
}

#[test]
fn test_arch_size_variant_default() {
    let v = ArchSizeVariant::default();
    assert_eq!(v.hidden_dim, 0);
    assert_eq!(v.num_layers, 0);
    assert_eq!(v.num_heads, None);
    assert!(v.parameters.is_empty());
}

#[test]
fn test_prompt_category_oracle_types() {
    let profile = profile_from_constraints("qwen2", &qwen_constraints(), Some(32768));
    for cat in &profile.prompt_categories {
        assert!(
            ["arithmetic", "response", "code_syntax"].contains(&cat.oracle_type.as_str()),
            "Unexpected oracle type: {}",
            cat.oracle_type
        );
    }
}

#[test]
fn test_prompt_category_max_tokens_positive() {
    let profile = profile_from_constraints("qwen2", &qwen_constraints(), Some(32768));
    for cat in &profile.prompt_categories {
        assert!(cat.max_tokens > 0, "max_tokens must be positive");
    }
}

#[test]
fn test_prompt_category_has_prompts() {
    let profile = profile_from_constraints("qwen2", &qwen_constraints(), Some(32768));
    for cat in &profile.prompt_categories {
        assert!(
            !cat.prompts.is_empty(),
            "Category '{}' must have prompts",
            cat.name
        );
    }
}

#[test]
fn test_absolute_position_encoding() {
    let constraints = ArchConstraints {
        positional_encoding: Some("absolute".to_string()),
        ..ArchConstraints::default()
    };
    let profile = profile_from_constraints("gpt2", &constraints, None);
    assert!(profile.kernel_ops.contains(&KernelOp::AbsolutePosition));
    assert!(!profile.long_context);
}

#[test]
fn test_kernel_profile_serialize_roundtrip() {
    let profile = profile_from_constraints("qwen2", &qwen_constraints(), Some(32768));
    let json = serde_json::to_string(&profile).expect("serialize");
    let deserialized: KernelProfile = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(deserialized.family, profile.family);
    assert_eq!(deserialized.kernel_ops.len(), profile.kernel_ops.len());
    assert_eq!(
        deserialized.prompt_categories.len(),
        profile.prompt_categories.len()
    );
}
