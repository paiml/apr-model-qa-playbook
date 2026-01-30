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
