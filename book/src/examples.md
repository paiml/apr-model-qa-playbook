# Code Examples

This chapter demonstrates how to use the APR QA libraries programmatically.
These examples can be run with `cargo run --example <name> -p <crate>`.

## Available Examples

| Example | Crate | Command |
|---------|-------|---------|
| `generate_scenarios` | apr-qa-gen | `cargo run --example generate_scenarios -p apr-qa-gen` |
| `collect_evidence` | apr-qa-runner | `cargo run --example collect_evidence -p apr-qa-runner` |
| `calculate_mqs` | apr-qa-report | `cargo run --example calculate_mqs -p apr-qa-report` |

## Generating QA Scenarios

The `generate_scenarios` example shows how to create test scenarios
using the ScenarioGenerator:

```rust
use apr_qa_gen::{Backend, Format, Modality, ModelId, ScenarioGenerator};

fn main() {
    // Create a model ID (org, name)
    let model = ModelId::new("meta-llama", "Llama-3.2-1B-Instruct");

    // Create a scenario generator
    let generator = ScenarioGenerator::new(model);

    // Generate scenarios for a specific configuration
    let scenarios = generator.generate_for(
        Modality::Run,
        Backend::Cpu,
        Format::Gguf
    );

    println!("Generated {} scenarios", scenarios.len());
}
```

### Output

```
Generated 18 scenarios for meta-llama/Llama-3.2-1B-Instruct:

Scenario 1:
  ID:       Llama-3.2-1B-Instruct_run_cpu_gguf_0000000000000000
  Modality: Run
  Backend:  Cpu
  Format:   Gguf
  Prompt:   What is 2+2?
  Seed:     0
```

## Collecting Test Evidence

The `collect_evidence` example demonstrates how to record test outcomes:

```rust
use apr_qa_gen::{Backend, Format, Modality, ModelId, QaScenario};
use apr_qa_runner::{Evidence, EvidenceCollector, Outcome};

fn main() {
    let model = ModelId::new("microsoft", "phi-3-mini-4k-instruct");
    let scenario = QaScenario::new(
        model,
        Modality::Run,
        Backend::Cpu,
        Format::Gguf,
        "What is 2+2?".to_string(),
        42,
    );

    let mut collector = EvidenceCollector::new();

    // Add corroborated evidence (test passed)
    collector.add(Evidence::corroborated(
        "G1-LOAD",
        scenario,
        "Model loaded successfully",
        1200,  // duration in ms
    ));

    // Print results
    for evidence in collector.all() {
        println!("[{}] Gate: {}",
            match evidence.outcome {
                Outcome::Corroborated => "PASS",
                Outcome::Falsified => "FAIL",
                _ => "SKIP",
            },
            evidence.gate_id
        );
    }
}
```

### Output

```
Evidence Collector Summary
==========================

[PASS] Gate: G1-LOAD
  Scenario: phi-3-mini-4k-instruct_run_cpu_gguf_000000000000002a
  Reason:   Test passed
  Duration: 1200ms

Summary: 1 passed, 0 failed
```

## Calculating MQS Scores

The `calculate_mqs` example shows how to compute Model Qualification Scores:

```rust
use apr_qa_gen::{Backend, Format, Modality, ModelId, QaScenario};
use apr_qa_report::MqsCalculator;
use apr_qa_runner::{Evidence, EvidenceCollector};

fn main() {
    let model_id = "meta-llama/Llama-3.2-1B-Instruct";
    let model = ModelId::new("meta-llama", "Llama-3.2-1B-Instruct");

    let mut collector = EvidenceCollector::new();
    // ... add evidence ...

    let calculator = MqsCalculator::new();

    match calculator.calculate(model_id, &collector) {
        Ok(score) => {
            println!("Model: {}", score.model_id);
            println!("Grade: {}", score.grade);
            println!("Score: {:.1}%", score.normalized_score);
            println!("Qualified: {}", score.qualifies());
        }
        Err(e) => eprintln!("Error: {}", e),
    }
}
```

### Output

```
Model Qualification Score (MQS) Report
======================================

Model: meta-llama/Llama-3.2-1B-Instruct
Grade: A-

Gateway Status:
  [PASS] G1: Model loads successfully
  [PASS] G2: Basic inference works
  [PASS] G3: No crashes
  [PASS] G4: Output is not garbage

Score Breakdown:
  Raw Score:         336
  Normalized Score:  93.8%
  Tests Passed:      15/16
  Qualification:     QUALIFIED
```

## Using with Popperian Analysis

For philosophical analysis using Popperian falsification methodology:

```rust
use apr_qa_report::popperian::PopperianCalculator;

let popperian_calc = PopperianCalculator::new();
let popperian = popperian_calc.calculate(model_id, &collector);

println!("Corroborated: {}", popperian.corroborated);
println!("Falsified:    {}", popperian.falsified);
println!("Black Swans:  {}", popperian.black_swan_count);
println!("{}", popperian.falsification_summary());
```

### Output

```
Popperian Analysis:
  Corroborated:      15
  Falsified:         1
  Inconclusive:      0
  Corroboration:     93.8%
  Black Swans:       0

Summary: 1 of 16 hypotheses falsified (6.2%)
```

## YAML Playbook Examples

The `examples/` directory in the repository root contains YAML playbook examples:

- `llama-3.2.yaml` - Llama 3.2 1B Instruct qualification
- `phi-3.yaml` - Phi-3 Mini qualification
- `qwen-coder.yaml` - Qwen Coder qualification

These can be run with the CLI:

```bash
cargo run --bin apr-qa -- run examples/llama-3.2.yaml
```
