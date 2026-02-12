//! Model registry and metadata
//!
//! Defines the `HuggingFace` model registry for qualification testing.

#![allow(clippy::struct_excessive_bools)]

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Model size category
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SizeCategory {
    /// 0.5B parameters
    Tiny,
    /// 1-2B parameters
    Small,
    /// 3-4B parameters
    Medium,
    /// 7-8B parameters
    Large,
    /// 13-14B parameters
    XLarge,
    /// 70B+ parameters
    Huge,
}

impl SizeCategory {
    /// Get the approximate parameter count
    #[must_use]
    pub const fn approx_params(&self) -> u64 {
        match self {
            Self::Tiny => 500_000_000,
            Self::Small => 1_500_000_000,
            Self::Medium => 3_500_000_000,
            Self::Large => 7_500_000_000,
            Self::XLarge => 13_500_000_000,
            Self::Huge => 70_000_000_000,
        }
    }

    /// Get memory estimate for F32 (4 bytes per param)
    #[must_use]
    pub const fn memory_f32_gb(&self) -> u64 {
        self.approx_params() * 4 / 1_000_000_000
    }

    /// Get memory estimate for `Q4_K` (0.5 bytes per param)
    #[must_use]
    pub const fn memory_q4k_gb(&self) -> u64 {
        self.approx_params() / 2 / 1_000_000_000
    }
}

/// Unique model identifier
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ModelId {
    /// `HuggingFace` organization
    pub org: String,
    /// Model name
    pub name: String,
    /// Optional variant (e.g., "Instruct", "Chat")
    pub variant: Option<String>,
}

impl ModelId {
    /// Create a new model ID
    #[must_use]
    pub fn new(org: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            org: org.into(),
            name: name.into(),
            variant: None,
        }
    }

    /// Create a model ID with variant
    #[must_use]
    pub fn with_variant(
        org: impl Into<String>,
        name: impl Into<String>,
        variant: impl Into<String>,
    ) -> Self {
        Self {
            org: org.into(),
            name: name.into(),
            variant: Some(variant.into()),
        }
    }

    /// Get the full `HuggingFace` repo ID
    #[must_use]
    pub fn hf_repo(&self) -> String {
        self.variant.as_ref().map_or_else(
            || format!("{}/{}", self.org, self.name),
            |v| format!("{}/{}-{}", self.org, self.name, v),
        )
    }
}

impl std::fmt::Display for ModelId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.hf_repo())
    }
}

/// Model metadata for qualification testing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// Model identifier
    pub id: ModelId,
    /// Size category
    pub size: SizeCategory,
    /// Architecture family (e.g., "qwen2", "llama", "mistral")
    pub architecture: String,
    /// Available quantizations
    pub quantizations: Vec<String>,
    /// Has chat template
    pub has_chat_template: bool,
    /// Supports system prompt
    pub supports_system_prompt: bool,
    /// Expected capabilities
    pub capabilities: ModelCapabilities,
}

/// Expected model capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCapabilities {
    /// Can do arithmetic (2+2=4)
    pub arithmetic: bool,
    /// Can do code completion
    pub code_completion: bool,
    /// Can follow instructions
    pub instruction_following: bool,
    /// Supports multi-turn conversation
    pub multi_turn: bool,
}

impl Default for ModelCapabilities {
    fn default() -> Self {
        Self {
            arithmetic: true,
            instruction_following: true,
            code_completion: false,
            multi_turn: true,
        }
    }
}

/// Registry of models for qualification testing
#[derive(Debug, Clone, Default)]
pub struct ModelRegistry {
    models: HashMap<String, ModelMetadata>,
}

impl ModelRegistry {
    /// Create a new empty registry
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create registry with default `HuggingFace` top models
    #[must_use]
    pub fn with_defaults() -> Self {
        let mut registry = Self::new();
        registry.add_default_models();
        registry
    }

    /// Add a model to the registry
    pub fn add(&mut self, metadata: ModelMetadata) {
        self.models.insert(metadata.id.hf_repo(), metadata);
    }

    /// Get model metadata by ID
    #[must_use]
    pub fn get(&self, id: &str) -> Option<&ModelMetadata> {
        self.models.get(id)
    }

    /// Get all models
    #[must_use]
    pub fn all(&self) -> Vec<&ModelMetadata> {
        self.models.values().collect()
    }

    /// Get models by size category
    #[must_use]
    pub fn by_size(&self, size: SizeCategory) -> Vec<&ModelMetadata> {
        self.models.values().filter(|m| m.size == size).collect()
    }

    /// Number of registered models
    #[must_use]
    pub fn len(&self) -> usize {
        self.models.len()
    }

    /// Check if registry is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.models.is_empty()
    }

    fn add_default_models(&mut self) {
        self.add_qwen25_general_models();
        self.add_qwen25_coder_models();
        self.add_qwen3_models();
        self.add_llama_models();
        self.add_mistral_models();
        self.add_gemma_models();
        self.add_phi_models();
        self.add_deepseek_coder_models();
        self.add_deepseek_r1_models();
        self.add_starcoder_models();
        self.add_yi_models();
        self.add_small_models();
        self.add_falcon_models();
        self.add_internlm_models();
        self.add_granite_models();
        self.add_olmo_models();
        self.add_nvidia_models();
        self.add_community_models();
    }

    fn add_qwen25_general_models(&mut self) {
        self.add(ModelMetadata {
            id: ModelId::with_variant("Qwen", "Qwen2.5-0.5B", "Instruct"),
            size: SizeCategory::Tiny,
            architecture: "qwen2".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        });

        self.add(ModelMetadata {
            id: ModelId::with_variant("Qwen", "Qwen2.5-1.5B", "Instruct"),
            size: SizeCategory::Small,
            architecture: "qwen2".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string(), "f16".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        });

        self.add(ModelMetadata {
            id: ModelId::with_variant("Qwen", "Qwen2.5-3B", "Instruct"),
            size: SizeCategory::Medium,
            architecture: "qwen2".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string(), "f16".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        });

        self.add(ModelMetadata {
            id: ModelId::with_variant("Qwen", "Qwen2.5-7B", "Instruct"),
            size: SizeCategory::Large,
            architecture: "qwen2".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string(), "f16".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        });

        self.add(ModelMetadata {
            id: ModelId::with_variant("Qwen", "Qwen2.5-14B", "Instruct"),
            size: SizeCategory::XLarge,
            architecture: "qwen2".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        });

        self.add(ModelMetadata {
            id: ModelId::with_variant("Qwen", "Qwen2.5-32B", "Instruct"),
            size: SizeCategory::Huge,
            architecture: "qwen2".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        });

        self.add(ModelMetadata {
            id: ModelId::with_variant("Qwen", "Qwen2.5-72B", "Instruct"),
            size: SizeCategory::Huge,
            architecture: "qwen2".to_string(),
            quantizations: vec!["q4_k_m".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        });

        // QwQ reasoning model
        self.add(ModelMetadata {
            id: ModelId::new("Qwen", "QwQ-32B"),
            size: SizeCategory::Huge,
            architecture: "qwen2".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        });
    }

    fn add_qwen25_coder_models(&mut self) {
        self.add(ModelMetadata {
            id: ModelId::with_variant("Qwen", "Qwen2.5-Coder-0.5B", "Instruct"),
            size: SizeCategory::Tiny,
            architecture: "qwen2".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities {
                code_completion: true,
                ..Default::default()
            },
        });

        self.add(ModelMetadata {
            id: ModelId::with_variant("Qwen", "Qwen2.5-Coder-1.5B", "Instruct"),
            size: SizeCategory::Small,
            architecture: "qwen2".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string(), "f16".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities {
                code_completion: true,
                ..Default::default()
            },
        });

        self.add(ModelMetadata {
            id: ModelId::with_variant("Qwen", "Qwen2.5-Coder-3B", "Instruct"),
            size: SizeCategory::Medium,
            architecture: "qwen2".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string(), "f16".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities {
                code_completion: true,
                ..Default::default()
            },
        });

        self.add(ModelMetadata {
            id: ModelId::with_variant("Qwen", "Qwen2.5-Coder-7B", "Instruct"),
            size: SizeCategory::Large,
            architecture: "qwen2".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string(), "f16".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities {
                code_completion: true,
                ..Default::default()
            },
        });

        self.add(ModelMetadata {
            id: ModelId::with_variant("Qwen", "Qwen2.5-Coder-14B", "Instruct"),
            size: SizeCategory::XLarge,
            architecture: "qwen2".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities {
                code_completion: true,
                ..Default::default()
            },
        });

        self.add(ModelMetadata {
            id: ModelId::with_variant("Qwen", "Qwen2.5-Coder-32B", "Instruct"),
            size: SizeCategory::Huge,
            architecture: "qwen2".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities {
                code_completion: true,
                ..Default::default()
            },
        });
    }

    fn add_qwen3_models(&mut self) {
        self.add(ModelMetadata {
            id: ModelId::new("Qwen", "Qwen3-0.6B"),
            size: SizeCategory::Tiny,
            architecture: "qwen3".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        });

        self.add(ModelMetadata {
            id: ModelId::new("Qwen", "Qwen3-1.7B"),
            size: SizeCategory::Small,
            architecture: "qwen3".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string(), "f16".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        });

        self.add(ModelMetadata {
            id: ModelId::new("Qwen", "Qwen3-4B"),
            size: SizeCategory::Medium,
            architecture: "qwen3".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string(), "f16".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        });

        self.add(ModelMetadata {
            id: ModelId::new("Qwen", "Qwen3-8B"),
            size: SizeCategory::Large,
            architecture: "qwen3".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string(), "f16".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        });

        self.add(ModelMetadata {
            id: ModelId::new("Qwen", "Qwen3-14B"),
            size: SizeCategory::XLarge,
            architecture: "qwen3".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        });

        self.add(ModelMetadata {
            id: ModelId::new("Qwen", "Qwen3-32B"),
            size: SizeCategory::Huge,
            architecture: "qwen3".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        });

        self.add(ModelMetadata {
            id: ModelId::new("Qwen", "Qwen3-30B-A3B"),
            size: SizeCategory::Huge,
            architecture: "qwen3_moe".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        });

        self.add(ModelMetadata {
            id: ModelId::with_variant("Qwen", "Qwen3-Coder-30B-A3B", "Instruct"),
            size: SizeCategory::Huge,
            architecture: "qwen3_moe".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities {
                code_completion: true,
                ..Default::default()
            },
        });
    }

    fn add_llama_models(&mut self) {
        self.add(ModelMetadata {
            id: ModelId::with_variant("meta-llama", "Llama-3.2-1B", "Instruct"),
            size: SizeCategory::Small,
            architecture: "llama".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        });

        self.add(ModelMetadata {
            id: ModelId::with_variant("meta-llama", "Llama-3.2-3B", "Instruct"),
            size: SizeCategory::Medium,
            architecture: "llama".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        });

        self.add(ModelMetadata {
            id: ModelId::with_variant("meta-llama", "Llama-3.1-8B", "Instruct"),
            size: SizeCategory::Large,
            architecture: "llama".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string(), "f16".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        });

        self.add(ModelMetadata {
            id: ModelId::with_variant("meta-llama", "Llama-3.1-70B", "Instruct"),
            size: SizeCategory::Huge,
            architecture: "llama".to_string(),
            quantizations: vec!["q4_k_m".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        });

        self.add(ModelMetadata {
            id: ModelId::with_variant("meta-llama", "Llama-3.3-70B", "Instruct"),
            size: SizeCategory::Huge,
            architecture: "llama".to_string(),
            quantizations: vec!["q4_k_m".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        });

        // CodeLlama family
        self.add(ModelMetadata {
            id: ModelId::new("meta-llama", "CodeLlama-7b-Instruct-hf"),
            size: SizeCategory::Large,
            architecture: "llama".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: false,
            capabilities: ModelCapabilities {
                code_completion: true,
                ..Default::default()
            },
        });

        self.add(ModelMetadata {
            id: ModelId::new("meta-llama", "CodeLlama-13b-Instruct-hf"),
            size: SizeCategory::XLarge,
            architecture: "llama".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: false,
            capabilities: ModelCapabilities {
                code_completion: true,
                ..Default::default()
            },
        });

        self.add(ModelMetadata {
            id: ModelId::new("meta-llama", "CodeLlama-34b-Instruct-hf"),
            size: SizeCategory::Huge,
            architecture: "llama".to_string(),
            quantizations: vec!["q4_k_m".to_string()],
            has_chat_template: true,
            supports_system_prompt: false,
            capabilities: ModelCapabilities {
                code_completion: true,
                ..Default::default()
            },
        });

        self.add(ModelMetadata {
            id: ModelId::new("meta-llama", "CodeLlama-70b-Instruct-hf"),
            size: SizeCategory::Huge,
            architecture: "llama".to_string(),
            quantizations: vec!["q4_k_m".to_string()],
            has_chat_template: true,
            supports_system_prompt: false,
            capabilities: ModelCapabilities {
                code_completion: true,
                ..Default::default()
            },
        });
    }

    fn add_mistral_models(&mut self) {
        self.add(ModelMetadata {
            id: ModelId::with_variant("mistralai", "Mistral-7B", "Instruct-v0.3"),
            size: SizeCategory::Large,
            architecture: "mistral".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        });

        self.add(ModelMetadata {
            id: ModelId::new("mistralai", "Mistral-Nemo-Instruct-2407"),
            size: SizeCategory::XLarge,
            architecture: "mistral".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        });

        self.add(ModelMetadata {
            id: ModelId::new("mistralai", "Mistral-Small-24B-Instruct-2501"),
            size: SizeCategory::Huge,
            architecture: "mistral".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        });

        self.add(ModelMetadata {
            id: ModelId::new("mistralai", "Codestral-22B-v0.1"),
            size: SizeCategory::Huge,
            architecture: "mistral".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities {
                code_completion: true,
                ..Default::default()
            },
        });
    }

    fn add_gemma_models(&mut self) {
        // Gemma 2 family
        self.add(ModelMetadata {
            id: ModelId::with_variant("google", "gemma-2-2b", "it"),
            size: SizeCategory::Small,
            architecture: "gemma2".to_string(),
            quantizations: vec!["q4_k_m".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        });

        self.add(ModelMetadata {
            id: ModelId::with_variant("google", "gemma-2-9b", "it"),
            size: SizeCategory::Large,
            architecture: "gemma2".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        });

        self.add(ModelMetadata {
            id: ModelId::with_variant("google", "gemma-2-27b", "it"),
            size: SizeCategory::Huge,
            architecture: "gemma2".to_string(),
            quantizations: vec!["q4_k_m".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        });

        self.add(ModelMetadata {
            id: ModelId::with_variant("google", "codegemma-7b", "it"),
            size: SizeCategory::Large,
            architecture: "gemma".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities {
                code_completion: true,
                ..Default::default()
            },
        });

        // Gemma 3 family
        self.add(ModelMetadata {
            id: ModelId::with_variant("google", "gemma-3-1b", "it"),
            size: SizeCategory::Small,
            architecture: "gemma3".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        });

        self.add(ModelMetadata {
            id: ModelId::with_variant("google", "gemma-3-4b", "it"),
            size: SizeCategory::Medium,
            architecture: "gemma3".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string(), "f16".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        });

        self.add(ModelMetadata {
            id: ModelId::with_variant("google", "gemma-3-12b", "it"),
            size: SizeCategory::XLarge,
            architecture: "gemma3".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        });

        self.add(ModelMetadata {
            id: ModelId::with_variant("google", "gemma-3-27b", "it"),
            size: SizeCategory::Huge,
            architecture: "gemma3".to_string(),
            quantizations: vec!["q4_k_m".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        });
    }

    fn add_phi_models(&mut self) {
        self.add(ModelMetadata {
            id: ModelId::new("microsoft", "Phi-3-mini-4k-instruct"),
            size: SizeCategory::Medium,
            architecture: "phi3".to_string(),
            quantizations: vec!["q4_k_m".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities {
                code_completion: true,
                ..Default::default()
            },
        });

        self.add(ModelMetadata {
            id: ModelId::new("microsoft", "Phi-3-small-8k-instruct"),
            size: SizeCategory::Large,
            architecture: "phi3".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities {
                code_completion: true,
                ..Default::default()
            },
        });

        self.add(ModelMetadata {
            id: ModelId::new("microsoft", "Phi-3-medium-4k-instruct"),
            size: SizeCategory::XLarge,
            architecture: "phi3".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities {
                code_completion: true,
                ..Default::default()
            },
        });

        self.add(ModelMetadata {
            id: ModelId::new("microsoft", "Phi-3.5-mini-instruct"),
            size: SizeCategory::Medium,
            architecture: "phi3".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities {
                code_completion: true,
                ..Default::default()
            },
        });

        self.add(ModelMetadata {
            id: ModelId::new("microsoft", "Phi-4-mini-instruct"),
            size: SizeCategory::Medium,
            architecture: "phi4".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities {
                code_completion: true,
                ..Default::default()
            },
        });
    }

    fn add_deepseek_coder_models(&mut self) {
        self.add(ModelMetadata {
            id: ModelId::new("deepseek-ai", "deepseek-coder-1.3b-instruct"),
            size: SizeCategory::Small,
            architecture: "deepseek".to_string(),
            quantizations: vec![
                "q4_k_m".to_string(),
                "q5_k_m".to_string(),
                "q8_0".to_string(),
            ],
            has_chat_template: true,
            supports_system_prompt: false,
            capabilities: ModelCapabilities {
                code_completion: true,
                ..Default::default()
            },
        });

        self.add(ModelMetadata {
            id: ModelId::new("deepseek-ai", "deepseek-coder-6.7b-instruct"),
            size: SizeCategory::Large,
            architecture: "deepseek".to_string(),
            quantizations: vec![
                "q4_k_m".to_string(),
                "q5_k_m".to_string(),
                "q8_0".to_string(),
            ],
            has_chat_template: true,
            supports_system_prompt: false,
            capabilities: ModelCapabilities {
                code_completion: true,
                ..Default::default()
            },
        });

        self.add(ModelMetadata {
            id: ModelId::new("deepseek-ai", "deepseek-coder-7b-instruct"),
            size: SizeCategory::Large,
            architecture: "deepseek".to_string(),
            quantizations: vec![
                "q4_k_m".to_string(),
                "q5_k_m".to_string(),
                "q8_0".to_string(),
            ],
            has_chat_template: true,
            supports_system_prompt: false,
            capabilities: ModelCapabilities {
                code_completion: true,
                ..Default::default()
            },
        });

        self.add(ModelMetadata {
            id: ModelId::new("deepseek-ai", "deepseek-coder-33b-instruct"),
            size: SizeCategory::Huge,
            architecture: "deepseek".to_string(),
            quantizations: vec!["q4_k_m".to_string()],
            has_chat_template: true,
            supports_system_prompt: false,
            capabilities: ModelCapabilities {
                code_completion: true,
                ..Default::default()
            },
        });

        self.add(ModelMetadata {
            id: ModelId::new("deepseek-ai", "DeepSeek-Coder-V2-Lite-Instruct"),
            size: SizeCategory::XLarge,
            architecture: "deepseek2".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities {
                code_completion: true,
                ..Default::default()
            },
        });
    }

    fn add_deepseek_r1_models(&mut self) {
        // Qwen architecture distills
        self.add(ModelMetadata {
            id: ModelId::new("deepseek-ai", "DeepSeek-R1-Distill-Qwen-1.5B"),
            size: SizeCategory::Small,
            architecture: "qwen2".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        });

        self.add(ModelMetadata {
            id: ModelId::new("deepseek-ai", "DeepSeek-R1-Distill-Qwen-7B"),
            size: SizeCategory::Large,
            architecture: "qwen2".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        });

        self.add(ModelMetadata {
            id: ModelId::new("deepseek-ai", "DeepSeek-R1-Distill-Qwen-14B"),
            size: SizeCategory::XLarge,
            architecture: "qwen2".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        });

        self.add(ModelMetadata {
            id: ModelId::new("deepseek-ai", "DeepSeek-R1-Distill-Qwen-32B"),
            size: SizeCategory::Huge,
            architecture: "qwen2".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        });

        // Llama architecture distills
        self.add(ModelMetadata {
            id: ModelId::new("deepseek-ai", "DeepSeek-R1-Distill-Llama-8B"),
            size: SizeCategory::Large,
            architecture: "llama".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        });

        self.add(ModelMetadata {
            id: ModelId::new("deepseek-ai", "DeepSeek-R1-Distill-Llama-70B"),
            size: SizeCategory::Huge,
            architecture: "llama".to_string(),
            quantizations: vec!["q4_k_m".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        });
    }

    fn add_starcoder_models(&mut self) {
        self.add(ModelMetadata {
            id: ModelId::new("bigcode", "starcoder2-3b"),
            size: SizeCategory::Medium,
            architecture: "starcoder2".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: false,
            supports_system_prompt: false,
            capabilities: ModelCapabilities {
                code_completion: true,
                arithmetic: false,
                instruction_following: false,
                multi_turn: false,
            },
        });

        self.add(ModelMetadata {
            id: ModelId::new("bigcode", "starcoder2-7b"),
            size: SizeCategory::Large,
            architecture: "starcoder2".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: false,
            supports_system_prompt: false,
            capabilities: ModelCapabilities {
                code_completion: true,
                arithmetic: false,
                instruction_following: false,
                multi_turn: false,
            },
        });

        self.add(ModelMetadata {
            id: ModelId::new("bigcode", "starcoder2-15b"),
            size: SizeCategory::XLarge,
            architecture: "starcoder2".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: false,
            supports_system_prompt: false,
            capabilities: ModelCapabilities {
                code_completion: true,
                arithmetic: false,
                instruction_following: false,
                multi_turn: false,
            },
        });
    }

    fn add_yi_models(&mut self) {
        self.add(ModelMetadata {
            id: ModelId::new("01-ai", "Yi-1.5-6B-Chat"),
            size: SizeCategory::Large,
            architecture: "yi".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        });

        self.add(ModelMetadata {
            id: ModelId::new("01-ai", "Yi-1.5-9B-Chat"),
            size: SizeCategory::Large,
            architecture: "yi".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        });

        self.add(ModelMetadata {
            id: ModelId::new("01-ai", "Yi-1.5-34B-Chat"),
            size: SizeCategory::Huge,
            architecture: "yi".to_string(),
            quantizations: vec!["q4_k_m".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        });
    }

    fn add_small_models(&mut self) {
        // SmolLM2 family
        self.add(ModelMetadata {
            id: ModelId::new("HuggingFaceTB", "SmolLM2-135M-Instruct"),
            size: SizeCategory::Tiny,
            architecture: "smollm".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities {
                arithmetic: false,
                ..Default::default()
            },
        });

        self.add(ModelMetadata {
            id: ModelId::new("HuggingFaceTB", "SmolLM2-360M-Instruct"),
            size: SizeCategory::Tiny,
            architecture: "smollm".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities {
                arithmetic: false,
                ..Default::default()
            },
        });

        self.add(ModelMetadata {
            id: ModelId::new("HuggingFaceTB", "SmolLM2-1.7B-Instruct"),
            size: SizeCategory::Small,
            architecture: "smollm".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        });

        // TinyLlama
        self.add(ModelMetadata {
            id: ModelId::new("TinyLlama", "TinyLlama-1.1B-Chat-v1.0"),
            size: SizeCategory::Small,
            architecture: "llama".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: false,
            capabilities: ModelCapabilities {
                arithmetic: false,
                ..Default::default()
            },
        });

        // StableLM family
        self.add(ModelMetadata {
            id: ModelId::new("stabilityai", "stablelm-2-zephyr-1_6b"),
            size: SizeCategory::Small,
            architecture: "stablelm".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        });

        self.add(ModelMetadata {
            id: ModelId::new("stabilityai", "stablelm-zephyr-3b"),
            size: SizeCategory::Medium,
            architecture: "stablelm".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        });
    }

    fn add_falcon_models(&mut self) {
        self.add(ModelMetadata {
            id: ModelId::new("tiiuae", "falcon-7b-instruct"),
            size: SizeCategory::Large,
            architecture: "falcon".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: false,
            supports_system_prompt: false,
            capabilities: ModelCapabilities::default(),
        });

        self.add(ModelMetadata {
            id: ModelId::new("tiiuae", "falcon-40b-instruct"),
            size: SizeCategory::Huge,
            architecture: "falcon".to_string(),
            quantizations: vec!["q4_k_m".to_string()],
            has_chat_template: false,
            supports_system_prompt: false,
            capabilities: ModelCapabilities::default(),
        });
    }

    fn add_internlm_models(&mut self) {
        self.add(ModelMetadata {
            id: ModelId::new("internlm", "internlm2_5-7b-chat"),
            size: SizeCategory::Large,
            architecture: "internlm2".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities {
                code_completion: true,
                ..Default::default()
            },
        });

        self.add(ModelMetadata {
            id: ModelId::new("internlm", "internlm2_5-20b-chat"),
            size: SizeCategory::Huge,
            architecture: "internlm2".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities {
                code_completion: true,
                ..Default::default()
            },
        });
    }

    fn add_granite_models(&mut self) {
        self.add(ModelMetadata {
            id: ModelId::new("ibm-granite", "granite-3.1-2b-instruct"),
            size: SizeCategory::Small,
            architecture: "granite".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities {
                code_completion: true,
                ..Default::default()
            },
        });

        self.add(ModelMetadata {
            id: ModelId::new("ibm-granite", "granite-3.1-8b-instruct"),
            size: SizeCategory::Large,
            architecture: "granite".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities {
                code_completion: true,
                ..Default::default()
            },
        });

        self.add(ModelMetadata {
            id: ModelId::new("ibm-granite", "granite-3b-code-instruct-128k"),
            size: SizeCategory::Medium,
            architecture: "granite".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities {
                code_completion: true,
                ..Default::default()
            },
        });
    }

    fn add_olmo_models(&mut self) {
        self.add(ModelMetadata {
            id: ModelId::new("allenai", "OLMo-2-1124-7B-Instruct"),
            size: SizeCategory::Large,
            architecture: "olmo".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        });

        self.add(ModelMetadata {
            id: ModelId::new("allenai", "OLMo-2-1124-13B-Instruct"),
            size: SizeCategory::XLarge,
            architecture: "olmo".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        });
    }

    fn add_nvidia_models(&mut self) {
        self.add(ModelMetadata {
            id: ModelId::new("nvidia", "Llama-3.1-Nemotron-Nano-4B-v1.1"),
            size: SizeCategory::Medium,
            architecture: "llama".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        });

        self.add(ModelMetadata {
            id: ModelId::new("nvidia", "Llama-3.1-Nemotron-70B-Instruct-HF"),
            size: SizeCategory::Huge,
            architecture: "llama".to_string(),
            quantizations: vec!["q4_k_m".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        });
    }

    fn add_community_models(&mut self) {
        // NousResearch Hermes
        self.add(ModelMetadata {
            id: ModelId::new("NousResearch", "Hermes-3-Llama-3.1-8B"),
            size: SizeCategory::Large,
            architecture: "llama".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        });

        // OpenChat
        self.add(ModelMetadata {
            id: ModelId::new("openchat", "openchat-3.5-0106"),
            size: SizeCategory::Large,
            architecture: "mistral".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        });

        // Zephyr
        self.add(ModelMetadata {
            id: ModelId::new("HuggingFaceH4", "zephyr-7b-beta"),
            size: SizeCategory::Large,
            architecture: "mistral".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        });

        // Dolphin
        self.add(ModelMetadata {
            id: ModelId::new("cognitivecomputations", "dolphin-2.6-mistral-7b"),
            size: SizeCategory::Large,
            architecture: "mistral".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        });

        self.add(ModelMetadata {
            id: ModelId::new("cognitivecomputations", "Dolphin3.0-Llama3.1-8B"),
            size: SizeCategory::Large,
            architecture: "llama".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        });

        // Vicuna
        self.add(ModelMetadata {
            id: ModelId::new("lmsys", "vicuna-7b-v1.5"),
            size: SizeCategory::Large,
            architecture: "llama".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        });

        self.add(ModelMetadata {
            id: ModelId::new("lmsys", "vicuna-13b-v1.5"),
            size: SizeCategory::XLarge,
            architecture: "llama".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        });

        // OpenHermes
        self.add(ModelMetadata {
            id: ModelId::new("teknium", "OpenHermes-2.5-Mistral-7B"),
            size: SizeCategory::Large,
            architecture: "mistral".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        });

        // WizardCoder
        self.add(ModelMetadata {
            id: ModelId::new("WizardLMTeam", "WizardCoder-15B-V1.0"),
            size: SizeCategory::XLarge,
            architecture: "starcoder".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: false,
            supports_system_prompt: false,
            capabilities: ModelCapabilities {
                code_completion: true,
                ..Default::default()
            },
        });

        self.add(ModelMetadata {
            id: ModelId::new("WizardLMTeam", "WizardCoder-33B-V1.1"),
            size: SizeCategory::Huge,
            architecture: "deepseek".to_string(),
            quantizations: vec!["q4_k_m".to_string()],
            has_chat_template: false,
            supports_system_prompt: false,
            capabilities: ModelCapabilities {
                code_completion: true,
                ..Default::default()
            },
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_id_display() {
        let id = ModelId::new("Qwen", "Qwen2.5-Coder-1.5B");
        assert_eq!(id.to_string(), "Qwen/Qwen2.5-Coder-1.5B");

        let id_with_variant = ModelId::with_variant("Qwen", "Qwen2.5-Coder-1.5B", "Instruct");
        assert_eq!(
            id_with_variant.to_string(),
            "Qwen/Qwen2.5-Coder-1.5B-Instruct"
        );
    }

    #[test]
    fn test_size_category_memory() {
        assert_eq!(SizeCategory::Tiny.memory_f32_gb(), 2);
        assert_eq!(SizeCategory::Small.memory_f32_gb(), 6);
        assert_eq!(SizeCategory::Large.memory_f32_gb(), 30);
    }

    #[test]
    fn test_registry_with_defaults() {
        let registry = ModelRegistry::with_defaults();
        assert!(!registry.is_empty());
        assert!(registry.len() >= 90);
    }

    #[test]
    fn test_registry_by_size() {
        let registry = ModelRegistry::with_defaults();
        let small_models = registry.by_size(SizeCategory::Small);
        assert!(!small_models.is_empty());
        for model in small_models {
            assert_eq!(model.size, SizeCategory::Small);
        }
    }

    #[test]
    fn test_size_category_approx_params() {
        assert_eq!(SizeCategory::Tiny.approx_params(), 500_000_000);
        assert_eq!(SizeCategory::Small.approx_params(), 1_500_000_000);
        assert_eq!(SizeCategory::Medium.approx_params(), 3_500_000_000);
        assert_eq!(SizeCategory::Large.approx_params(), 7_500_000_000);
        assert_eq!(SizeCategory::XLarge.approx_params(), 13_500_000_000);
        assert_eq!(SizeCategory::Huge.approx_params(), 70_000_000_000);
    }

    #[test]
    fn test_size_category_memory_q4k() {
        assert_eq!(SizeCategory::Tiny.memory_q4k_gb(), 0);
        assert_eq!(SizeCategory::Small.memory_q4k_gb(), 0);
        assert_eq!(SizeCategory::Medium.memory_q4k_gb(), 1);
        assert_eq!(SizeCategory::Large.memory_q4k_gb(), 3);
        assert_eq!(SizeCategory::XLarge.memory_q4k_gb(), 6);
        assert_eq!(SizeCategory::Huge.memory_q4k_gb(), 35);
    }

    #[test]
    fn test_model_id_hf_repo() {
        let id = ModelId::new("org", "model");
        assert_eq!(id.hf_repo(), "org/model");
    }

    #[test]
    fn test_model_id_clone() {
        let id = ModelId::new("org", "model");
        let cloned = id.clone();
        assert_eq!(cloned.org, id.org);
        assert_eq!(cloned.name, id.name);
    }

    #[test]
    fn test_model_id_eq() {
        let id1 = ModelId::new("org", "model");
        let id2 = ModelId::new("org", "model");
        assert_eq!(id1, id2);
    }

    #[test]
    fn test_model_capabilities_default() {
        let caps = ModelCapabilities::default();
        assert!(caps.arithmetic);
        assert!(caps.instruction_following);
        assert!(!caps.code_completion);
        assert!(caps.multi_turn);
    }

    #[test]
    fn test_registry_new() {
        let registry = ModelRegistry::new();
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);
    }

    #[test]
    fn test_registry_add_and_get() {
        let mut registry = ModelRegistry::new();
        let metadata = ModelMetadata {
            id: ModelId::new("test", "model"),
            size: SizeCategory::Small,
            architecture: "test".to_string(),
            quantizations: vec!["q4_k_m".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        };

        registry.add(metadata);
        assert_eq!(registry.len(), 1);

        let retrieved = registry.get("test/model");
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().architecture, "test");
    }

    #[test]
    fn test_registry_get_nonexistent() {
        let registry = ModelRegistry::new();
        assert!(registry.get("nonexistent/model").is_none());
    }

    #[test]
    fn test_registry_all() {
        let registry = ModelRegistry::with_defaults();
        let all = registry.all();
        assert_eq!(all.len(), registry.len());
    }

    #[test]
    fn test_model_metadata_clone() {
        let metadata = ModelMetadata {
            id: ModelId::new("test", "model"),
            size: SizeCategory::Small,
            architecture: "test".to_string(),
            quantizations: vec!["q4_k_m".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        };
        let cloned = metadata.clone();
        assert_eq!(cloned.id, metadata.id);
        assert_eq!(cloned.architecture, metadata.architecture);
    }

    #[test]
    fn test_size_category_debug() {
        let size = SizeCategory::Large;
        let debug_str = format!("{size:?}");
        assert!(debug_str.contains("Large"));
    }

    #[test]
    fn test_model_id_serialize() {
        let id = ModelId::new("test", "model");
        let json = serde_json::to_string(&id).expect("serialize");
        assert!(json.contains("test"));
        assert!(json.contains("model"));
    }

    #[test]
    fn test_size_category_eq_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(SizeCategory::Small);
        set.insert(SizeCategory::Medium);
        set.insert(SizeCategory::Small); // duplicate
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_registry_default() {
        let registry = ModelRegistry::default();
        assert!(registry.is_empty());
    }

    #[test]
    fn test_model_id_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(ModelId::new("org1", "model1"));
        set.insert(ModelId::new("org2", "model2"));
        set.insert(ModelId::new("org1", "model1")); // duplicate
        assert_eq!(set.len(), 2);
    }

    // --- Mutation-killing tests for hf_repo ---
    #[test]
    fn test_hf_repo_format_with_slash() {
        let id = ModelId::new("MyOrg", "MyModel");
        let repo = id.hf_repo();
        assert!(!repo.is_empty());
        assert!(repo.contains('/'));
        assert_eq!(repo, "MyOrg/MyModel");
        assert!(repo.starts_with("MyOrg"));
        assert!(repo.ends_with("MyModel"));
    }

    #[test]
    fn test_hf_repo_with_variant_format() {
        let id = ModelId::with_variant("Org", "Model", "Variant");
        let repo = id.hf_repo();
        assert!(!repo.is_empty());
        assert!(repo.contains('/'));
        assert!(repo.contains('-'));
        assert_eq!(repo, "Org/Model-Variant");
    }

    #[test]
    fn test_hf_repo_variant_none_uses_name_only() {
        let id = ModelId::new("X", "Y");
        assert!(id.variant.is_none());
        assert_eq!(id.hf_repo(), "X/Y");
        assert!(!id.hf_repo().contains('-'));
    }

    #[test]
    fn test_hf_repo_variant_some_appends_dash() {
        let id = ModelId::with_variant("X", "Y", "Z");
        assert!(id.variant.is_some());
        assert_eq!(id.hf_repo(), "X/Y-Z");
        assert!(id.hf_repo().contains('-'));
    }

    // --- Mutation-killing tests for SizeCategory memory calculations ---
    #[test]
    fn test_size_memory_f32_nonzero() {
        // Verify calculations don't return zero for all
        assert!(SizeCategory::Tiny.memory_f32_gb() >= 1);
        assert!(SizeCategory::Small.memory_f32_gb() > SizeCategory::Tiny.memory_f32_gb());
        assert!(SizeCategory::Large.memory_f32_gb() > SizeCategory::Medium.memory_f32_gb());
    }

    #[test]
    fn test_size_memory_q4k_ordering() {
        // Larger models should have more memory
        assert!(SizeCategory::Huge.memory_q4k_gb() > SizeCategory::XLarge.memory_q4k_gb());
        assert!(SizeCategory::XLarge.memory_q4k_gb() > SizeCategory::Large.memory_q4k_gb());
    }

    #[test]
    fn test_approx_params_ordering() {
        // Strict ordering from tiny to huge
        let tiny = SizeCategory::Tiny.approx_params();
        let small = SizeCategory::Small.approx_params();
        let medium = SizeCategory::Medium.approx_params();
        let large = SizeCategory::Large.approx_params();
        let xlarge = SizeCategory::XLarge.approx_params();
        let huge = SizeCategory::Huge.approx_params();

        assert!(tiny < small);
        assert!(small < medium);
        assert!(medium < large);
        assert!(large < xlarge);
        assert!(xlarge < huge);
    }

    // --- Mutation-killing tests for ModelCapabilities ---
    #[test]
    fn test_capabilities_default_values_explicit() {
        let caps = ModelCapabilities::default();
        // Explicitly test all boolean values
        assert!(caps.arithmetic, "arithmetic should be true");
        assert!(
            caps.instruction_following,
            "instruction_following should be true"
        );
        assert!(!caps.code_completion, "code_completion should be false");
        assert!(caps.multi_turn, "multi_turn should be true");
    }
}
