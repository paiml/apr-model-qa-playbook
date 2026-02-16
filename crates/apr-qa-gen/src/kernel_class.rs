//! Kernel Equivalence Classes
//!
//! Maps model families to kernel equivalence classes. Models sharing the same
//! kernel class use identical compute kernels (attention, normalization, activation,
//! positional encoding). Once kernels are proven via full MVP on one representative
//! model, other models in the class only need dimensional smoke verification.

use serde::{Deserialize, Serialize};
use std::str::FromStr;

/// Kernel equivalence class identifier.
///
/// Each class groups model families that share identical kernel pipelines.
/// Class A covers ~70% of all models (GQA+RMSNorm+SiLU+SwiGLU+RoPE).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum KernelClass {
    /// GQA + RMSNorm + SiLU + SwiGLU + RoPE (LLaMA 3, Qwen2, Mistral, Yi, DeepSeek, InternLM2, Gemma)
    A,
    /// MHA + LayerNorm + GELU (GPT-NeoX, GPT-J, Falcon 7B)
    B,
    /// MQA + LayerNorm + GELU + ALiBi (Falcon-40B)
    C,
    /// GQA + LayerNorm + GELU/SiLU (Phi-3, StableLM)
    D,
    /// MoE + GQA + RMSNorm + SwiGLU (Mixtral, Qwen-MoE)
    E,
}

impl KernelClass {
    /// Map a model family string to its kernel equivalence class.
    ///
    /// Returns `None` for unknown families.
    #[must_use]
    pub fn from_family(family: &str) -> Option<Self> {
        match family.to_lowercase().as_str() {
            // Class A: GQA + RMSNorm + SiLU + SwiGLU + RoPE
            "llama" | "llama3" | "llama-3" | "llama3.2" | "codellama" | "tinyllama"
            | "qwen" | "qwen2" | "qwen2.5" | "qwen3" | "qwen-coder"
            | "mistral" | "yi"
            | "deepseek" | "deepseek-v2" | "deepseek-coder" | "deepseek-r1"
            | "internlm2" | "internlm"
            | "gemma" | "gemma2" | "gemma3" | "codegemma"
            | "smollm" | "olmo"
            | "granite" | "granite-code"
            | "starcoder2"
            | "nemotron" => Some(Self::A),

            // Class B: MHA + LayerNorm + GELU
            "gpt-neox" | "gptneox"
            | "gpt-j" | "gptj"
            | "falcon-7b" | "falcon7b" => Some(Self::B),

            // Class C: MQA + LayerNorm + GELU + ALiBi
            "falcon-40b" | "falcon40b" | "falcon" => Some(Self::C),

            // Class D: GQA + LayerNorm + GELU/SiLU
            "phi" | "phi-3" | "phi3" | "phi4"
            | "stablelm" | "stable-lm" => Some(Self::D),

            // Class E: MoE + GQA + RMSNorm + SwiGLU
            "mixtral"
            | "qwen-moe" | "qwenmoe" => Some(Self::E),

            _ => None,
        }
    }

    /// Get the representative model for this kernel class.
    ///
    /// This is the model that should have full MVP certification to prove
    /// kernel correctness for the entire class.
    #[must_use]
    pub const fn representative_model(&self) -> &'static str {
        match self {
            Self::A => "Qwen/Qwen2.5-Coder-0.5B-Instruct",
            Self::B => "EleutherAI/gpt-neox-20b",
            Self::C => "tiiuae/falcon-40b-instruct",
            Self::D => "microsoft/Phi-3-mini-4k-instruct",
            Self::E => "mistralai/Mixtral-8x7B-Instruct-v0.1",
        }
    }

    /// Human-readable label for this kernel class.
    #[must_use]
    pub const fn label(&self) -> &'static str {
        match self {
            Self::A => "GQA+RMSNorm+SiLU+SwiGLU+RoPE",
            Self::B => "MHA+LayerNorm+GELU",
            Self::C => "MQA+LayerNorm+GELU+ALiBi",
            Self::D => "GQA+LayerNorm+GELU/SiLU",
            Self::E => "MoE+GQA+RMSNorm+SwiGLU",
        }
    }

    /// Return all kernel class variants.
    #[must_use]
    pub const fn all() -> &'static [Self] {
        &[Self::A, Self::B, Self::C, Self::D, Self::E]
    }
}

impl FromStr for KernelClass {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_uppercase().as_str() {
            "A" => Ok(Self::A),
            "B" => Ok(Self::B),
            "C" => Ok(Self::C),
            "D" => Ok(Self::D),
            "E" => Ok(Self::E),
            _ => Err(format!("Unknown kernel class: {s}. Use: A, B, C, D, E")),
        }
    }
}

impl std::fmt::Display for KernelClass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::A => write!(f, "A"),
            Self::B => write!(f, "B"),
            Self::C => write!(f, "C"),
            Self::D => write!(f, "D"),
            Self::E => write!(f, "E"),
        }
    }
}

/// Filter certification records by kernel class.
///
/// Returns model IDs from the certifications list whose family maps to the given class.
#[must_use]
pub fn models_in_class(class: KernelClass, families_and_ids: &[(String, String)]) -> Vec<String> {
    families_and_ids
        .iter()
        .filter(|(family, _)| KernelClass::from_family(family) == Some(class))
        .map(|(_, model_id)| model_id.clone())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_family_class_a() {
        assert_eq!(KernelClass::from_family("llama"), Some(KernelClass::A));
        assert_eq!(KernelClass::from_family("qwen"), Some(KernelClass::A));
        assert_eq!(KernelClass::from_family("qwen-coder"), Some(KernelClass::A));
        assert_eq!(KernelClass::from_family("qwen3"), Some(KernelClass::A));
        assert_eq!(KernelClass::from_family("mistral"), Some(KernelClass::A));
        assert_eq!(KernelClass::from_family("yi"), Some(KernelClass::A));
        assert_eq!(KernelClass::from_family("deepseek"), Some(KernelClass::A));
        assert_eq!(KernelClass::from_family("gemma"), Some(KernelClass::A));
        assert_eq!(KernelClass::from_family("internlm2"), Some(KernelClass::A));
        assert_eq!(KernelClass::from_family("codellama"), Some(KernelClass::A));
        assert_eq!(KernelClass::from_family("tinyllama"), Some(KernelClass::A));
        assert_eq!(KernelClass::from_family("deepseek-coder"), Some(KernelClass::A));
        assert_eq!(KernelClass::from_family("deepseek-r1"), Some(KernelClass::A));
        assert_eq!(KernelClass::from_family("gemma3"), Some(KernelClass::A));
        assert_eq!(KernelClass::from_family("codegemma"), Some(KernelClass::A));
        assert_eq!(KernelClass::from_family("smollm"), Some(KernelClass::A));
        assert_eq!(KernelClass::from_family("olmo"), Some(KernelClass::A));
        assert_eq!(KernelClass::from_family("internlm"), Some(KernelClass::A));
        assert_eq!(KernelClass::from_family("granite"), Some(KernelClass::A));
        assert_eq!(KernelClass::from_family("granite-code"), Some(KernelClass::A));
        assert_eq!(KernelClass::from_family("starcoder2"), Some(KernelClass::A));
        assert_eq!(KernelClass::from_family("nemotron"), Some(KernelClass::A));
    }

    #[test]
    fn test_from_family_class_b() {
        assert_eq!(KernelClass::from_family("gpt-neox"), Some(KernelClass::B));
        assert_eq!(KernelClass::from_family("gpt-j"), Some(KernelClass::B));
        assert_eq!(
            KernelClass::from_family("falcon-7b"),
            Some(KernelClass::B)
        );
    }

    #[test]
    fn test_from_family_class_c() {
        assert_eq!(
            KernelClass::from_family("falcon-40b"),
            Some(KernelClass::C)
        );
        assert_eq!(KernelClass::from_family("falcon"), Some(KernelClass::C));
    }

    #[test]
    fn test_from_family_class_d() {
        assert_eq!(KernelClass::from_family("phi"), Some(KernelClass::D));
        assert_eq!(KernelClass::from_family("phi-3"), Some(KernelClass::D));
        assert_eq!(KernelClass::from_family("phi4"), Some(KernelClass::D));
        assert_eq!(KernelClass::from_family("stablelm"), Some(KernelClass::D));
    }

    #[test]
    fn test_from_family_class_e() {
        assert_eq!(KernelClass::from_family("mixtral"), Some(KernelClass::E));
        assert_eq!(KernelClass::from_family("qwen-moe"), Some(KernelClass::E));
    }

    #[test]
    fn test_from_family_unknown() {
        assert_eq!(KernelClass::from_family("unknown-model"), None);
        assert_eq!(KernelClass::from_family(""), None);
    }

    #[test]
    fn test_from_family_case_insensitive() {
        assert_eq!(KernelClass::from_family("LLAMA"), Some(KernelClass::A));
        assert_eq!(KernelClass::from_family("Qwen"), Some(KernelClass::A));
        assert_eq!(KernelClass::from_family("Mixtral"), Some(KernelClass::E));
    }

    #[test]
    fn test_representative_models() {
        assert!(KernelClass::A.representative_model().contains("Qwen"));
        assert!(KernelClass::B.representative_model().contains("neox"));
        assert!(KernelClass::C.representative_model().contains("falcon"));
        assert!(KernelClass::D.representative_model().contains("Phi"));
        assert!(KernelClass::E.representative_model().contains("Mixtral"));
    }

    #[test]
    fn test_labels() {
        assert!(KernelClass::A.label().contains("GQA"));
        assert!(KernelClass::B.label().contains("MHA"));
        assert!(KernelClass::C.label().contains("MQA"));
        assert!(KernelClass::D.label().contains("LayerNorm"));
        assert!(KernelClass::E.label().contains("MoE"));
    }

    #[test]
    fn test_from_str() {
        assert_eq!("A".parse::<KernelClass>().unwrap(), KernelClass::A);
        assert_eq!("b".parse::<KernelClass>().unwrap(), KernelClass::B);
        assert_eq!("C".parse::<KernelClass>().unwrap(), KernelClass::C);
        assert_eq!("d".parse::<KernelClass>().unwrap(), KernelClass::D);
        assert_eq!("E".parse::<KernelClass>().unwrap(), KernelClass::E);
        assert!("X".parse::<KernelClass>().is_err());
    }

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", KernelClass::A), "A");
        assert_eq!(format!("{}", KernelClass::E), "E");
    }

    #[test]
    fn test_models_in_class() {
        let data = vec![
            ("qwen".to_string(), "Qwen/Qwen2.5-0.5B".to_string()),
            ("qwen-coder".to_string(), "Qwen/Qwen2.5-Coder-1.5B".to_string()),
            ("llama".to_string(), "meta-llama/Llama-3-8B".to_string()),
            ("phi".to_string(), "microsoft/Phi-3-mini".to_string()),
            ("mixtral".to_string(), "mistralai/Mixtral-8x7B".to_string()),
        ];

        let class_a = models_in_class(KernelClass::A, &data);
        assert_eq!(class_a.len(), 3);
        assert!(class_a.contains(&"Qwen/Qwen2.5-0.5B".to_string()));
        assert!(class_a.contains(&"Qwen/Qwen2.5-Coder-1.5B".to_string()));
        assert!(class_a.contains(&"meta-llama/Llama-3-8B".to_string()));

        let class_d = models_in_class(KernelClass::D, &data);
        assert_eq!(class_d.len(), 1);
        assert!(class_d.contains(&"microsoft/Phi-3-mini".to_string()));

        let class_e = models_in_class(KernelClass::E, &data);
        assert_eq!(class_e.len(), 1);

        let class_b = models_in_class(KernelClass::B, &data);
        assert!(class_b.is_empty());
    }

    #[test]
    fn test_all_classes() {
        let all = KernelClass::all();
        assert_eq!(all.len(), 5);
        assert_eq!(all[0], KernelClass::A);
        assert_eq!(all[4], KernelClass::E);
    }

    #[test]
    fn test_serialize_deserialize() {
        let class = KernelClass::A;
        let json = serde_json::to_string(&class).unwrap();
        assert_eq!(json, "\"A\"");
        let deserialized: KernelClass = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, KernelClass::A);
    }
}
