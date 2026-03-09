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

The `kernel_profile` module maps architecture constraints from family
contracts to kernel operations and targeted prompts. Instead of generic
prompts for all models, this produces architecture-specific prompts
that stress-test the exact kernels each model uses.

### Kernel Operations

Each model family exercises a specific set of kernel operations (17 total):

| Family | Attention | Norm | Activation | MLP | Positional |
|--------|-----------|------|------------|-----|------------|
| LLaMA/Qwen | GQA | RMSNorm | SiLU | SwiGLU | RoPE |
| Falcon-7B | MHA | LayerNorm | GELU | - | RoPE |
| Falcon-40B | MQA | LayerNorm | GELU | - | ALiBi |
| Gemma | GQA | RMSNorm | GELU | GatedMlp | RoPE |
| GPT-NeoX | MHA | LayerNorm | GELU | - | RoPE |
| Mamba/RWKV | none | RMSNorm | SiLU | SwiGLU | none |

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

## Kernel Coverage Verification

The `kernel_coverage` module verifies that the sovereign stack (trueno/realizar)
implements all kernel operations required by each HuggingFace architecture. It loads
constraints from `arch-constraints-v1.yaml` (provable-contracts) and bindings from
`kernel-bindings.yaml` — zero hardcoded data.

### CoverageContext

```rust
use apr_qa_gen::CoverageContext;

// Load from YAML files
let ctx = CoverageContext::load(
    "../provable-contracts/contracts",
    "playbooks/kernel-bindings.yaml",
)?;

// Verify a single architecture
let result = ctx.verify_architecture("qwen2");

// Verify all architectures (canonical names only, no aliases)
let results = ctx.verify_all_architectures();
```

Key semantics:
- `fully_covered` is `false` when using LLaMA defaults (actual kernel requirements unknown)
- `gap_count` excludes models using defaults (separates unverified from real gaps)
- `canonical_names` prevents alias duplication in `--all` output

### Kernel Equivalence Classes (A-F)

| Class | Pipeline | Representative |
|-------|----------|---------------|
| A | GQA+RMSNorm+SiLU+SwiGLU+RoPE | Qwen2.5-Coder-0.5B |
| B | MHA+LayerNorm+GELU | GPT-NeoX-20B |
| C | MQA+LayerNorm+GELU+ALiBi | Falcon-40B |
| D | GQA+LayerNorm+GELU/SiLU | Phi-3 |
| E | MoE+GQA+RMSNorm+SwiGLU | Mixtral-8x7B |
| F | RMSNorm+GELU+GatedMlp+RoPE | Gemma-2-2B |
