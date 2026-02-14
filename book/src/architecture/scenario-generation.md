# Scenario Generation

The `apr-qa-gen` crate generates test scenarios using property-based testing with proptest.

## QaScenario

Each scenario represents a falsifiable hypothesis about model behavior:

```rust
pub struct QaScenario {
    pub model_id: ModelId,       // HuggingFace model identifier
    pub modality: Modality,      // run, chat, or serve
    pub backend: Backend,        // cpu or gpu
    pub format: Format,          // gguf, safetensors, or apr
    pub prompt: String,          // Input prompt
    pub seed: u64,               // For reproducibility
}
```

## Proptest Integration

Scenarios implement `Arbitrary` for property-based generation:

```rust
impl Arbitrary for QaScenario {
    type Strategy = BoxedStrategy<Self>;

    fn arbitrary() -> Self::Strategy {
        // Generates random but valid scenarios
    }
}
```

## Prompt Categories

| Category | Example | Oracle |
|----------|---------|--------|
| Arithmetic | `"2+2="` | Arithmetic oracle |
| Code | `"def hello():"` | CodeSyntax oracle |
| Instruction | `"List 3 colors"` | Response oracle |
| Edge | `""`, very long | Garbage oracle |

## Model Registry

The `ModelRegistry` contains the top 100 HuggingFace models:

```rust
let registry = ModelRegistry::new();
let models = registry.by_architecture("llama");
let small_models = registry.by_size(SizeCategory::Small);
```

## Scenario Generator

```rust
let generator = ScenarioGenerator::new(42); // seed
let scenarios = generator.generate(100);    // count
```

## Kernel Profile-Driven Generation

The `kernel_profile` module maps architecture constraints from family contracts to kernel operations and targeted prompts. Instead of generic prompts for all models, this produces architecture-specific prompts that stress-test the exact kernels each model uses.

### Kernel Operations

Each model family exercises a specific set of kernel operations:

| Family | Attention | Norm | Activation | Positional |
|--------|-----------|------|------------|------------|
| LLaMA/Qwen | GQA | RMSNorm | SiLU/SwiGLU | RoPE |
| Falcon | MHA/MQA | LayerNorm | GELU | ALiBi |
| GPT-NeoX | MHA | LayerNorm | GELU | RoPE |

### Architecture Constraints to Kernel Profile

```rust
use apr_qa_gen::{ArchConstraints, profile_from_constraints};

let constraints = ArchConstraints {
    attention_type: Some("gqa".to_string()),
    activation: Some("silu".to_string()),
    norm_type: Some("rmsnorm".to_string()),
    positional_encoding: Some("rope".to_string()),
    mlp_type: Some("swiglu".to_string()),
    ..Default::default()
};

let profile = profile_from_constraints("qwen2", &constraints, Some(32_768));

// Profile contains kernel ops and targeted prompts
assert!(profile.kernel_ops.len() > 5);
assert!(profile.long_context); // 32K > 4K threshold
```

### Playbook Bootstrapping

The `bootstrapper` module generates complete playbook YAML from constraints:

```rust
use apr_qa_gen::{BootstrapConfig, bootstrap_playbook, to_yaml};

let config = BootstrapConfig {
    family: "qwen2".to_string(),
    size_variant: "1.5b".to_string(),
    hf_repo: "Qwen/Qwen2.5-Coder-1.5B-Instruct".to_string(),
    tier: "mvp".to_string(),
    kernel_profile: None, // auto-derived from constraints
};

let playbook = bootstrap_playbook(&config, &constraints, &size_variant, "small");
let yaml = to_yaml(&playbook).unwrap();
```

Or via CLI:

```bash
apr-qa bootstrap qwen2 1.5b \
    --hf-repo Qwen/Qwen2.5-Coder-1.5B-Instruct \
    --tier mvp \
    --output playbooks/models/qwen2.5-coder-1.5b-mvp.playbook.yaml
```
