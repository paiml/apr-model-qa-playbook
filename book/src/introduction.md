# APR Model QA Playbook

A **property-based model qualification testing framework** for HuggingFace models.

## Philosophy

This framework embodies two complementary quality philosophies:

1. **Toyota Production System (TPS):** Zero-defect manufacturing through Jidoka (autonomation), Poka-Yoke (error-proofing), and Genchi Genbutsu (go and see).

2. **Popperian Falsificationism:** Solving the *demarcation problem* of model quality by defining "correctness" not as the accumulation of passing tests, but as the survival of rigorous attempts at refutation.

> "The criterion of the scientific status of a theory is its falsifiability, or refutability, or testability."
> — Karl Popper, *Conjectures and Refutations* (1963)

## Key Features

- **Property-based testing** with proptest for comprehensive scenario generation
- **Parallel execution** with Rayon worker pools
- **Gateway checks (G1-G4)** that zero the score on critical failures
- **Model Qualification Score (MQS)** 0-1000 with grade mapping
- **JUnit XML and HTML reports** for CI/CD integration
- **Playbook YAML format** with JSON Schema validation

## Test Matrix

The framework tests models across multiple dimensions:

| Dimension | Options |
|-----------|---------|
| **Modality** | run, chat, serve |
| **Backend** | cpu, gpu |
| **Format** | gguf, safetensors, apr |
| **Quantization** | q4_k_m, q5_k_m, q8_0, etc. |

With 100 scenarios per combination, this yields comprehensive coverage of model behavior.

## Quick Start

```bash
# Run all tests
make test

# Generate coverage report
make coverage

# Run a specific playbook
cargo run --bin apr-qa -- run playbooks/models/qwen2.5-coder-1.5b.playbook.yaml
```

## Project Structure

```
apr-model-qa-playbook/
├── crates/
│   ├── apr-qa-gen/      # Scenario generation + oracles
│   ├── apr-qa-runner/   # Playbook execution
│   ├── apr-qa-report/   # MQS scoring + reports
│   └── apr-qa-cli/      # CLI binary
├── playbooks/
│   ├── models/          # Per-model playbooks
│   ├── templates/       # Reusable templates
│   ├── verify/          # Ticket verification
│   └── spec/            # Executable specifications
└── docs/
    └── specifications/  # Full specification document
```
