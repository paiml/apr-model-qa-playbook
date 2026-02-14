//! Kernel Profile-Driven Playbook Bootstrapping
//!
//! Maps architecture constraints from family contracts to kernel operations
//! and targeted test prompts. This enables architecture-aware playbook generation
//! that stress-tests the specific kernels each model family exercises.
//!
//! # Design Philosophy
//!
//! HuggingFace model families exercise different kernel operations:
//! - LLaMA/Qwen: GQA + RMSNorm + SiLU + RoPE
//! - Falcon: MHA + LayerNorm + GELU
//! - GPT-NeoX: MHA + LayerNorm + GELU + RoPE
//!
//! By connecting family contract constraints to kernel ops to targeted prompts,
//! we bootstrap playbooks that exercise the exact code paths each model uses.

use serde::{Deserialize, Serialize};

/// Kernel operation exercised by a model architecture.
///
/// Each variant maps to a specific SIMD kernel in the trueno/realizar stack.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KernelOp {
    /// Fused Q4K matrix-vector multiply (quantized inference)
    FusedQ4kMatvec,
    /// Fused Q5K matrix-vector multiply
    FusedQ5kMatvec,
    /// Fused Q6K matrix-vector multiply
    FusedQ6kMatvec,
    /// RMS normalization (LLaMA, Qwen, Mistral families)
    RmsNorm,
    /// Layer normalization (Falcon, GPT-NeoX families)
    LayerNorm,
    /// SiLU activation (LLaMA, Qwen)
    Silu,
    /// GELU activation (Falcon, GPT-NeoX)
    Gelu,
    /// SwiGLU MLP gate (LLaMA, Qwen, Mistral)
    SwiGlu,
    /// Rotary positional encoding
    Rope,
    /// Grouped-query attention (Qwen, LLaMA 3.x, Mistral)
    GroupedQueryAttention,
    /// Multi-head attention (Falcon, GPT-NeoX, older models)
    MultiHeadAttention,
    /// Multi-query attention (Falcon-40B)
    MultiQueryAttention,
    /// Bias addition in linear layers
    BiasAdd,
    /// Tied input/output embeddings (shared weight matrix)
    TiedEmbeddings,
    /// ALiBi positional encoding (Falcon)
    Alibi,
    /// Absolute positional encoding (GPT-2, BERT)
    AbsolutePosition,
}

impl KernelOp {
    /// Human-readable description of this kernel operation.
    #[must_use]
    pub const fn description(&self) -> &'static str {
        match self {
            Self::FusedQ4kMatvec => "Fused Q4K quantized matrix-vector multiply",
            Self::FusedQ5kMatvec => "Fused Q5K quantized matrix-vector multiply",
            Self::FusedQ6kMatvec => "Fused Q6K quantized matrix-vector multiply",
            Self::RmsNorm => "RMS normalization",
            Self::LayerNorm => "Layer normalization",
            Self::Silu => "SiLU activation function",
            Self::Gelu => "GELU activation function",
            Self::SwiGlu => "SwiGLU gated MLP",
            Self::Rope => "Rotary positional encoding",
            Self::GroupedQueryAttention => "Grouped-query attention (GQA)",
            Self::MultiHeadAttention => "Multi-head attention (MHA)",
            Self::MultiQueryAttention => "Multi-query attention (MQA)",
            Self::BiasAdd => "Bias addition in linear layers",
            Self::TiedEmbeddings => "Tied input/output embeddings",
            Self::Alibi => "ALiBi positional encoding",
            Self::AbsolutePosition => "Absolute positional encoding",
        }
    }
}

impl std::fmt::Display for KernelOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.description())
    }
}

/// A category of prompts targeting specific kernel behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptCategory {
    /// Category name (e.g., "gqa_multi_turn", "rope_long_context")
    pub name: String,
    /// Why these prompts target specific kernels
    pub rationale: String,
    /// The actual test prompts
    pub prompts: Vec<String>,
    /// Oracle type for evaluating outputs
    pub oracle_type: String,
    /// Suggested max tokens for this category
    pub max_tokens: u32,
}

/// Complete kernel profile for a model family.
///
/// Describes which kernel operations a model architecture exercises
/// and provides targeted prompts to stress-test those operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelProfile {
    /// Model family name (e.g., "qwen2", "llama", "falcon")
    pub family: String,
    /// Kernel operations exercised by this architecture
    pub kernel_ops: Vec<KernelOp>,
    /// Architecture-targeted prompt categories
    pub prompt_categories: Vec<PromptCategory>,
    /// Suggested max tokens based on architecture
    pub suggested_max_tokens: u32,
    /// Whether this architecture supports long context (>4K tokens)
    pub long_context: bool,
}

impl KernelProfile {
    /// Get all prompts from all categories, flattened.
    #[must_use]
    pub fn all_prompts(&self) -> Vec<String> {
        self.prompt_categories
            .iter()
            .flat_map(|c| c.prompts.clone())
            .collect()
    }

    /// Total number of prompts across all categories.
    #[must_use]
    pub fn prompt_count(&self) -> usize {
        self.prompt_categories.iter().map(|c| c.prompts.len()).sum()
    }
}

/// Mirror of `Constraints` from `apr-qa-runner::family_contract`.
///
/// Defined here to avoid circular dependency (runner depends on gen).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ArchConstraints {
    /// Attention type: "mha", "gqa", or "mqa"
    pub attention_type: Option<String>,
    /// Activation function: "silu", "gelu", "relu"
    pub activation: Option<String>,
    /// Norm type: "rmsnorm" or "layernorm"
    pub norm_type: Option<String>,
    /// Whether linear layers have bias terms
    pub has_bias: Option<bool>,
    /// Whether input/output embeddings are shared
    pub tied_embeddings: Option<bool>,
    /// Positional encoding: "rope", "absolute", "alibi"
    pub positional_encoding: Option<String>,
    /// MLP type: "swiglu", "gelu_mlp", "relu_mlp"
    pub mlp_type: Option<String>,
}

/// Mirror of `SizeVariant` from `apr-qa-runner::family_contract`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ArchSizeVariant {
    /// Human-readable parameter count (e.g., "0.5B", "7B")
    pub parameters: String,
    /// Hidden dimension / d_model
    pub hidden_dim: u32,
    /// Number of transformer layers
    pub num_layers: u32,
    /// Number of attention heads
    pub num_heads: u32,
    /// Number of KV heads (for GQA)
    pub num_kv_heads: Option<u32>,
    /// FFN intermediate dimension
    pub intermediate_dim: Option<u32>,
    /// Vocabulary size
    pub vocab_size: Option<u32>,
    /// Maximum sequence length
    pub max_position_embeddings: Option<u32>,
}

/// Build a kernel profile from architecture constraints.
///
/// Maps family contract constraints to kernel operations and generates
/// targeted prompts that exercise the specific kernels each model uses.
#[must_use]
pub fn profile_from_constraints(
    family: &str,
    constraints: &ArchConstraints,
    max_position_embeddings: Option<u32>,
) -> KernelProfile {
    let mut kernel_ops = Vec::new();
    let mut prompt_categories = Vec::new();

    // Always include quantized matvec ops (all models use these)
    kernel_ops.push(KernelOp::FusedQ4kMatvec);
    kernel_ops.push(KernelOp::FusedQ6kMatvec);

    // Attention type -> kernel ops + prompts
    match constraints.attention_type.as_deref() {
        Some("gqa") => {
            kernel_ops.push(KernelOp::GroupedQueryAttention);
            prompt_categories.push(gqa_prompts());
        }
        Some("mqa") => {
            kernel_ops.push(KernelOp::MultiQueryAttention);
            prompt_categories.push(mqa_prompts());
        }
        _ => {
            kernel_ops.push(KernelOp::MultiHeadAttention);
            prompt_categories.push(mha_prompts());
        }
    }

    // Normalization type
    match constraints.norm_type.as_deref() {
        Some("layernorm") => kernel_ops.push(KernelOp::LayerNorm),
        // Default to RMSNorm (most common in modern architectures)
        _ => kernel_ops.push(KernelOp::RmsNorm),
    }

    // Activation function
    match constraints.activation.as_deref() {
        Some("gelu") => kernel_ops.push(KernelOp::Gelu),
        // Default to SiLU (most common in modern architectures)
        _ => kernel_ops.push(KernelOp::Silu),
    }

    // MLP type
    if constraints.mlp_type.as_deref() == Some("swiglu") {
        kernel_ops.push(KernelOp::SwiGlu);
    }

    // Positional encoding
    let long_context = match constraints.positional_encoding.as_deref() {
        Some("rope") => {
            kernel_ops.push(KernelOp::Rope);
            let max_pos = max_position_embeddings.unwrap_or(4096);
            if max_pos > 4096 {
                prompt_categories.push(rope_long_context_prompts());
            }
            max_pos > 4096
        }
        Some("alibi") => {
            kernel_ops.push(KernelOp::Alibi);
            false
        }
        Some("absolute") => {
            kernel_ops.push(KernelOp::AbsolutePosition);
            false
        }
        _ => false,
    };

    // Bias in linear layers
    if constraints.has_bias == Some(true) {
        kernel_ops.push(KernelOp::BiasAdd);
        prompt_categories.push(bias_stress_prompts());
    }

    // Tied embeddings
    if constraints.tied_embeddings == Some(true) {
        kernel_ops.push(KernelOp::TiedEmbeddings);
    }

    // Always include arithmetic verification prompts
    prompt_categories.push(arithmetic_prompts());

    // Always include code completion prompts (exercises token generation paths)
    prompt_categories.push(code_prompts());

    let suggested_max_tokens = if long_context { 128 } else { 64 };

    KernelProfile {
        family: family.to_string(),
        kernel_ops,
        prompt_categories,
        suggested_max_tokens,
        long_context,
    }
}

/// Prompts targeting grouped-query attention (GQA).
///
/// GQA shares KV heads across query head groups. Multi-turn and
/// context-dependent prompts stress the KV cache sharing logic.
fn gqa_prompts() -> PromptCategory {
    PromptCategory {
        name: "gqa_multi_turn".to_string(),
        rationale: "GQA shares KV heads across query groups; multi-turn prompts \
                    stress KV cache sharing and head group boundaries"
            .to_string(),
        prompts: vec![
            "Given x=5 and y=3, what is x*y? Then what is the result plus 10?".to_string(),
            "List the first 5 prime numbers. Now sum them.".to_string(),
            "Define a function add(a,b) that returns a+b. What does add(3,4) return?".to_string(),
        ],
        oracle_type: "arithmetic".to_string(),
        max_tokens: 64,
    }
}

/// Prompts targeting multi-head attention (MHA).
///
/// MHA has independent KV heads per query head. Long dependency
/// prompts stress full attention computation.
fn mha_prompts() -> PromptCategory {
    PromptCategory {
        name: "mha_long_dependency".to_string(),
        rationale: "MHA computes independent KV per head; long-range dependency \
                    prompts test full attention matrix computation"
            .to_string(),
        prompts: vec![
            "The capital of France is Paris. The capital of Germany is Berlin. \
             What is the capital of France?"
                .to_string(),
            "Alice has 3 apples. Bob gives her 2 more. Carol takes 1. \
             How many apples does Alice have?"
                .to_string(),
            "If x=10, y=x+5, z=y*2, what is z?".to_string(),
        ],
        oracle_type: "response".to_string(),
        max_tokens: 64,
    }
}

/// Prompts targeting multi-query attention (MQA).
fn mqa_prompts() -> PromptCategory {
    PromptCategory {
        name: "mqa_kv_efficiency".to_string(),
        rationale: "MQA uses a single KV head for all query heads; prompts test \
                    that shared KV computation produces correct results"
            .to_string(),
        prompts: vec![
            "What is 7*8? Answer with just the number.".to_string(),
            "Complete: The sum of 15 and 25 is".to_string(),
            "Translate 'hello' to Spanish in one word.".to_string(),
        ],
        oracle_type: "arithmetic".to_string(),
        max_tokens: 32,
    }
}

/// Prompts for RoPE long-context models (>4K tokens).
///
/// Tests that rotary position encoding correctly handles
/// positions beyond the standard 2K-4K range.
fn rope_long_context_prompts() -> PromptCategory {
    PromptCategory {
        name: "rope_long_context".to_string(),
        rationale: "RoPE position encoding must correctly extrapolate to long \
                    sequences; these prompts test position-dependent accuracy"
            .to_string(),
        prompts: vec![
            "Write a detailed step-by-step solution to: What is 123 * 456? \
             Show all intermediate multiplication steps."
                .to_string(),
            "List the numbers from 1 to 20, then sum them all. \
             What is the final sum?"
                .to_string(),
            "Explain the Fibonacci sequence, list the first 10 numbers, \
             then tell me what the 10th Fibonacci number is."
                .to_string(),
        ],
        oracle_type: "response".to_string(),
        max_tokens: 256,
    }
}

/// Prompts that stress bias addition in linear layers.
///
/// Models with bias terms have additional addition operations
/// in every linear projection. Precision-sensitive prompts
/// can reveal bias accumulation errors.
fn bias_stress_prompts() -> PromptCategory {
    PromptCategory {
        name: "bias_precision".to_string(),
        rationale: "Bias terms add to every linear projection output; \
                    arithmetic prompts can reveal floating-point accumulation errors"
            .to_string(),
        prompts: vec![
            "What is 0.1 + 0.2? Give a precise answer.".to_string(),
            "Calculate 999 + 1.".to_string(),
            "What is 1000000 - 999999?".to_string(),
        ],
        oracle_type: "arithmetic".to_string(),
        max_tokens: 32,
    }
}

/// Standard arithmetic verification prompts.
fn arithmetic_prompts() -> PromptCategory {
    PromptCategory {
        name: "arithmetic_verification".to_string(),
        rationale: "Arithmetic prompts provide deterministic verification \
                    of model output correctness across all architectures"
            .to_string(),
        prompts: vec![
            "What is 2+2?".to_string(),
            "Calculate 7*8".to_string(),
            "What is 15-7?".to_string(),
            "What is 100/4?".to_string(),
        ],
        oracle_type: "arithmetic".to_string(),
        max_tokens: 32,
    }
}

/// Code completion prompts that exercise token generation paths.
fn code_prompts() -> PromptCategory {
    PromptCategory {
        name: "code_completion".to_string(),
        rationale: "Code completion exercises the full token generation pipeline \
                    including vocabulary lookup and sampling"
            .to_string(),
        prompts: vec![
            "def fibonacci(n):".to_string(),
            "fn main() {".to_string(),
            "Write a Python function that checks if a number is prime.".to_string(),
        ],
        oracle_type: "code_syntax".to_string(),
        max_tokens: 64,
    }
}

#[cfg(test)]
mod tests {
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
        assert_eq!(v.num_heads, 0);
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
}
