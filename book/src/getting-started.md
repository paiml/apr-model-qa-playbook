# Getting Started

## Prerequisites

- Rust 1.85+ (edition 2024)
- cargo-llvm-cov (for coverage)

## Installation

```bash
# Clone the repository
git clone https://github.com/paiml/apr-model-qa-playbook
cd apr-model-qa-playbook

# Install development dependencies
make dev-deps

# Build all crates
make build
```

## Running Tests

```bash
# Run all tests
make test

# Run with verbose output
make test-verbose

# Run a single test
cargo test --package apr-qa-gen -- test_arithmetic_oracle
```

## Coverage

```bash
# Generate HTML coverage report
make coverage

# Check PMAT compliance (>= 95%)
make coverage-check
```

## Creating a Playbook

Create a YAML file following the playbook schema:

```yaml
name: my-model-test
version: "1.0.0"

model:
  hf_repo: "org/model-name"
  formats:
    - gguf
  quantizations:
    - q4_k_m

test_matrix:
  modalities:
    - run
  backends:
    - cpu
  scenario_count: 10

gates:
  g1_model_loads: true
  g2_basic_inference: true
  g3_no_crashes: true
  g4_output_quality: true
```

## Running a Playbook

```bash
cargo run --bin apr-qa -- run playbooks/models/my-model.playbook.yaml
```

## Running Examples

Run the included examples to see the libraries in action:

```bash
# Generate QA scenarios
cargo run --example generate_scenarios -p apr-qa-gen

# Collect test evidence
cargo run --example collect_evidence -p apr-qa-runner

# Calculate MQS scores
cargo run --example calculate_mqs -p apr-qa-report
```

See the [Code Examples](./examples.md) chapter for detailed documentation.
