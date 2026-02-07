# Code Examples

This chapter demonstrates how to use the APR QA libraries programmatically.
These examples can be run with `cargo run --example <name> -p <crate>`.

## Available Examples

| Example | Crate | Command |
|---------|-------|---------|
| `generate_scenarios` | apr-qa-gen | `cargo run --example generate_scenarios -p apr-qa-gen` |
| `collect_evidence` | apr-qa-runner | `cargo run --example collect_evidence -p apr-qa-runner` |
| `rosetta_testing` | apr-qa-runner | `cargo run --example rosetta_testing -p apr-qa-runner` |
| `calculate_mqs` | apr-qa-report | `cargo run --example calculate_mqs -p apr-qa-report` |
| `generate_certificate` | apr-qa-report | `cargo run --example generate_certificate -p apr-qa-report` |
| `generate_rag_markdown` | apr-qa-report | `cargo run --example generate_rag_markdown -p apr-qa-report` |
| `fail_fast_demo` | apr-qa-cli | `cargo run --example fail_fast_demo -p apr-qa-cli` |
| `integrity_lock_demo` | apr-qa-cli | `cargo run --example integrity_lock_demo -p apr-qa-cli` |
| `contract_demo` | apr-qa-runner | `cargo run --example contract_demo -p apr-qa-runner` |

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

## Generating Certificates

The `generate_certificate` example shows how to create CERTIFICATE.md files:

```rust
use apr_qa_report::{CertificateGenerator, MqsCalculator};
use apr_qa_report::popperian::PopperianCalculator;

fn main() {
    let model_id = "Qwen/Qwen2.5-Coder-0.5B-Instruct";
    let version = "1.0.0";

    // Calculate scores from evidence
    let mqs = mqs_calc.calculate(model_id, &collector).unwrap();
    let popperian = popperian_calc.calculate(model_id, &collector);

    // Generate certificate
    let generator = CertificateGenerator::new("APR QA Framework");
    let certificate = generator.generate(
        model_id,
        version,
        &mqs,
        &popperian,
        evidence_hash,
    );

    // Output status
    println!("Status: {}", certificate.status);  // CERTIFIED/PROVISIONAL/REJECTED
    println!("Grade:  {}", certificate.grade);   // A+, A, B+, B, C, F

    // Generate CERTIFICATE.md
    let markdown = generator.to_markdown(&certificate);
}
```

### Output

```
Certificate Generated
=====================

Model:    Qwen/Qwen2.5-Coder-0.5B-Instruct
Version:  1.0.0
Status:   CERTIFIED
Grade:    A
MQS:      920/1000
Score:    156/170 (91.8%)
Black Swans: 0
```

## Generating RAG-Optimized Markdown

The `generate_rag_markdown` example shows how to generate markdown reports
optimized for batuta's RAG oracle indexing:

```rust
use apr_qa_report::{generate_rag_markdown, generate_index_entry, generate_evidence_detail};
use apr_qa_report::mqs::MqsScore;
use apr_qa_report::popperian::PopperianScore;
use apr_qa_runner::EvidenceCollector;

fn main() {
    // Create or calculate MQS and Popperian scores
    let mqs: MqsScore = /* ... */;
    let popperian: PopperianScore = /* ... */;
    let collector: EvidenceCollector = /* ... */;

    // Generate full RAG-optimized markdown report
    let full_report = generate_rag_markdown(&mqs, &popperian, &collector);
    println!("{}", full_report);

    // Generate compact index entry for summary tables
    let entry = generate_index_entry(&mqs);
    println!("| Model | Score | Grade | Status | Prod Ready |");
    println!("{}", entry);

    // Generate individual evidence detail
    for evidence in collector.all().iter().take(3) {
        println!("{}", generate_evidence_detail(evidence));
    }
}
```

### Output

```
# Model Qualification: qwen/Qwen2.5-Coder-7B-Instruct

## Summary

- **MQS Score**: 892/1000 (89.2 normalized, A-)
- **Status**: PROVISIONAL
- **Tests**: 142 passed / 8 failed / 150 total
- **Black Swans**: 0
- **Corroboration Rate**: 94.7%

## Gateway Checks

| Gateway | Status | Description |
|---------|--------|-------------|
| G1 | ✓ PASS | Model loads successfully |
| G2 | ✓ PASS | Basic inference works |
| G3 | ✓ PASS | No crashes during testing |
| G4 | ✓ PASS | Output is coherent text |

## Falsifications

### 1: F-PERF-023

- **Hypothesis**: Inference completes under 500ms for short prompts
- **Evidence**: Measured: 623ms (exceeded by 24.6%)
- **Severity**: 2/5
- **Occurrences**: 1
```

The generated markdown uses semantic headers (`##`, `###`) that align with
batuta's SemanticChunker, enabling effective RAG retrieval:

```bash
# Index the QA playbook documentation
batuta oracle --rag-index

# Query for qualification information
batuta oracle --rag "Popperian falsification scoring"
batuta oracle --rag "MQS gateway checks"
```

## Rosetta Testing (Metamorphic Relations)

The `rosetta_testing` example demonstrates the conversion testing gates
added by the PMAT-ROSETTA-002/003 spec. These implement metamorphic relations
that catch silent conversion bugs.

```rust
use apr_qa_gen::{Backend, Format, ModelId};
use apr_qa_runner::{
    CommutativityTest, ConversionConfig, IdempotencyTest,
    InspectResult, RoundTripTest,
};

fn main() {
    let model = ModelId::new("Qwen", "Qwen2.5-Coder-0.5B-Instruct");

    // Parse inspect output to get tensor metadata
    let inspect: InspectResult = serde_json::from_str(json).unwrap();
    println!("Tensors: {}", inspect.tensor_count);

    // Multi-hop chain: ST → APR → GGUF → APR → ST
    let rt = RoundTripTest::new(
        vec![Format::SafeTensors, Format::Apr, Format::Gguf,
             Format::Apr, Format::SafeTensors],
        Backend::Cpu,
        model.clone(),
    );

    // Idempotency: convert twice, compare
    let idem = IdempotencyTest::new(
        Format::Gguf, Format::Apr, Backend::Cpu, model.clone(),
    );

    // Commutativity: GGUF→APR vs GGUF→ST→APR
    let com = CommutativityTest::new(Backend::Cpu, model);

    // All enabled by default in ConversionConfig
    let config = ConversionConfig::default();
    assert!(config.test_multi_hop);
    assert!(config.test_cardinality);
    assert!(config.test_idempotency);
    assert!(config.test_commutativity);
}
```

### Output

```
=== InspectResult (T-GH192-01) ===

Tensor count:          338
Attention heads:       Some(14)
KV heads:              Some(2)
Hidden size:           Some(896)
Architecture:          Some("Qwen2ForCausalLM")

=== Multi-Hop Round-Trip Chains ===

F-CONV-RT-002: [SafeTensors, Apr, Gguf, SafeTensors] (3 hops)
F-CONV-RT-003: [SafeTensors, Apr, Gguf, Apr, SafeTensors] (4 hops)

=== New Gate IDs (PMAT-ROSETTA-002/003 + GH-6) ===

| Gate ID              | Source     | Description                          |
|----------------------+------------+--------------------------------------|
| F-CONV-CARD-001      | MR-CARD    | Silent tensor loss (QKV fusion)      |
| F-CONV-NAME-001      | T-QKV-02   | Unexpected tensor renaming           |
| F-CONV-RT-002        | T-QKV-03   | ST→APR→GGUF→ST round-trip failure    |
| F-CONV-RT-003        | T-QKV-04   | Multi-hop chain failure              |
| F-CONV-RT-004        | GH-6/AC-3  | ST→APR→GGUF→APR round-trip (3 hops)  |
| F-CONV-RT-BYTE-001   | GH-6/AC-3  | Byte-level tensor diff after RT      |
| F-CONV-IDEM-001      | MR-IDEM    | Double-convert instability           |
| F-CONV-COM-001       | MR-COM     | Path-dependent conversion bugs       |
| F-INSPECT-META-001   | T-GH192    | Missing/wrong model metadata         |
| F-OLLAMA-001         | GH-6/AC-2  | APR/Ollama output parity             |
| F-OLLAMA-002         | GH-6/AC-2  | APR/Ollama performance ratio         |
```

## Model Certification Workflow

The recommended workflow for certifying models:

### 1. MVP Certification (Quick Surface Coverage)

```bash
# Certify all models in a family with MVP tier
apr-qa certify --family qwen-coder --tier mvp

# Results:
# - Tests 18 combinations (3 formats × 2 backends × 3 modalities)
# - On pass: Grade B, Status PROVISIONAL
# - Time limit: ≤10 minutes per model
```

### 2. Full Certification (Production Release)

```bash
# Certify for production release
apr-qa certify --family qwen-coder --tier full

# Results:
# - Runs complete 170-point Verification Matrix
# - On pass: Grade A+, Status CERTIFIED
# - Time limit: ≤1 hour per model
```

### 3. Certification Results

Results are stored in:
- `docs/certifications/models.csv` - Central certification database
- `certifications/<model>/evidence.json` - Raw test evidence
- `README.md` - Certification table (via `apr-qa-readme-sync`)

## Playbook Integrity Locking

Lock playbook hashes to detect unauthorized modifications:

```bash
# Generate lock file
cargo run --bin apr-qa -- lock-playbooks

# Output: playbooks/playbook.lock.yaml
# Contains SHA-256 hashes for each .playbook.yaml file
```

During certification, the executor automatically verifies playbooks against the
lock file. If a playbook has been modified since locking, certification refuses
to proceed (Jidoka: stop the line).

## Auto-Ticket Generation from Failures

Generate structured upstream tickets that group failures by root cause:

```bash
# Certify with auto-ticket generation
cargo run --bin apr-qa -- certify --family qwen-coder --tier mvp --auto-ticket

# Output:
# === Auto-Generated Tickets (2) ===
#   [QA] F-CONV-001: tensor_name_mismatch (12 occurrences) [P1-High]
#     Fixture: fixtures/tensor_name_mismatch.py
#   [QA] F-CONV-002: missing_artifact (3 occurrences) [P1-High]
#     Fixture: fixtures/artifact_completeness.py
```

The auto-ticket system:
1. Classifies failures by stderr pattern matching (tensor mismatch, dequantization, missing artifact, etc.)
2. Groups by root cause (12 same-type failures become 1 ticket)
3. Renders ticket body from the defect-fixture map with reproduction steps
4. Attaches upstream fixture paths and pygmy builder names

### Programmatic Usage

```rust
use apr_qa_report::defect_map::load_defect_fixture_map;
use apr_qa_report::ticket::generate_structured_tickets;

let defect_map = load_defect_fixture_map().expect("load map");
let tickets = generate_structured_tickets(&evidence, &defect_map);

for ticket in &tickets {
    println!("{} [{}]", ticket.title, ticket.priority);
    if let Some(ref fixture) = ticket.upstream_fixture {
        println!("  Fixture: {fixture}");
    }
}
```

## Isolated Output Directories (ISO-OUT-001)

Conversion test artifacts are written to an isolated directory to prevent
pollution of the HuggingFace cache:

```
output/conversions/{org}/{repo}/{test_type}/model.{tag}.{ext}
```

Example structure after running tests:

```
output/
├── evidence.json
├── report.html
└── conversions/
    └── Qwen/
        └── Qwen2.5-Coder-0.5B-Instruct/
            ├── basic/
            │   └── model.converted.apr
            ├── semantic/
            │   └── model.semantic_test.apr
            └── idempotency/
                ├── model.idem1.apr
                └── model.idem2.apr
```

This keeps `~/.cache/huggingface/` clean and makes cleanup easy (`rm -rf output/`).

### Programmatic Usage

```rust
use apr_qa_gen::ModelId;
use apr_qa_runner::ConversionOutputDir;
use std::path::Path;

let model_id = ModelId::new("Qwen", "Qwen2.5-Coder-0.5B-Instruct");
let output_dir = ConversionOutputDir::new(Path::new("output"), &model_id);

// Get paths for different test types
let basic_path = output_dir.basic_dir();
let semantic_path = output_dir.semantic_dir();
let idempotency_path = output_dir.idempotency_dir();

// Clean up after testing
output_dir.cleanup().expect("cleanup failed");
```

## Fail-Fast Diagnostic Reports (FF-REPORT-001)

The `fail_fast_demo` example demonstrates the diagnostic report generation
when --fail-fast mode detects a test failure:

```rust
use apr_qa_runner::{
    DiagnosticResult, DiagnosticsBundle, EnvironmentContext,
    FailFastReport, FailFastReporter, FailureDetails, FailurePolicy,
    ReproductionInfo,
};
use std::path::Path;

fn main() {
    // Create reporter pointing to output directory
    let reporter = FailFastReporter::new(Path::new("output"));

    // When a test fails, the executor calls:
    // reporter.generate_report(&evidence, model_path, Some("playbook.yaml"))
    //
    // This generates:
    //   output/fail-fast-report/
    //   ├── summary.md           # GitHub-ready markdown
    //   ├── diagnostics.json     # Full report
    //   ├── check.json           # apr check output
    //   ├── inspect.json         # apr inspect output
    //   ├── trace.json           # apr trace output
    //   ├── tensors.json         # apr tensors output
    //   ├── environment.json     # System context
    //   └── stderr.log           # Raw stderr

    // FailurePolicy::FailFast enables diagnostic generation
    let policy = FailurePolicy::FailFast;
    assert!(policy.emit_diagnostic());      // true
    assert!(policy.stops_on_any_failure()); // true
}
```

### Output

```
[FAIL-FAST] Gate G3-STABLE FALSIFIED
[FAIL-FAST] Model: Qwen/Qwen2.5-Coder-0.5B-Instruct
[FAIL-FAST] Format: Apr
[FAIL-FAST] Backend: Cpu
[FAIL-FAST] Outcome: Crashed
[FAIL-FAST] Reason: Process crashed with exit code -1
[FAIL-FAST] Generating diagnostic report...
[FAIL-FAST] Running apr check... done (2.3s)
[FAIL-FAST] Running apr inspect... done (0.4s)
[FAIL-FAST] Running apr trace... done (5.1s)
[FAIL-FAST] Running apr tensors... done (0.3s)
[FAIL-FAST] Report saved to: output/fail-fast-report/
[FAIL-FAST] Summary: output/fail-fast-report/summary.md
[FAIL-FAST] GitHub issue body ready for paste
```

The generated `summary.md` is formatted for direct paste into GitHub issues:

```markdown
# Fail-Fast Report: G3-STABLE

## Failure Summary

| Field | Value |
|-------|-------|
| Gate | `G3-STABLE` |
| Model | `Qwen/Qwen2.5-Coder-0.5B-Instruct` |
| Format | Apr |
| Backend | Cpu |
| Outcome | Crashed |
| Exit Code | -1 |
| Duration | 52740ms |

## Environment

| Field | Value |
|-------|-------|
| OS | linux x86_64 |
| apr-qa | 0.1.0 |
| apr-cli | 0.2.12 |
| Git | abc123 (main) |
| Rust | 1.93.0 |

## Reproduction

```bash
# Reproduce this failure
apr-qa run playbook.yaml --fail-fast

# Run diagnostics manually
apr check /path/to/model.apr
apr trace /path/to/model.apr --payload -v
apr explain G3-STABLE
```
```

## Playbook Integrity Lock (§3.1)

The `integrity_lock_demo` example demonstrates the playbook integrity lock system
that prevents unauthorized modification of test specifications:

```bash
cargo run --example integrity_lock_demo -p apr-qa-cli
```

### Key Concepts

The integrity lock system uses SHA-256 hashes to detect playbook modifications:

```rust
use apr_qa_runner::{
    PlaybookLockFile, compute_playbook_hash, generate_lock_entry,
    verify_playbook_integrity,
};

fn main() {
    // Compute hash of a playbook file
    let hash = compute_playbook_hash("playbook.yaml").unwrap();
    println!("SHA-256: {hash}"); // 64 hex characters

    // Generate a lock entry
    let (name, entry) = generate_lock_entry("playbook.yaml").unwrap();
    println!("Locked fields: {:?}", entry.locked_fields);

    // Verify integrity against lock file
    let lock_file = PlaybookLockFile::default();
    match verify_playbook_integrity("playbook.yaml", &lock_file, &name) {
        Ok(()) => println!("Integrity: PASSED"),
        Err(e) => println!("Integrity: BLOCKED - {e}"),
    }
}
```

### Lock File Structure

```yaml
# playbook.lock.yaml
entries:
  qwen2.5-coder-1.5b-mvp:
    sha256: 8dbb1f48ca93a0948a560fb32a6febc37e2569040fc2aac2581dd5668cd3d7d2
    locked_fields:
    - model.hf_repo
    - model.formats
    - test_matrix
    - falsification_gates
```

### Example Output

```
=== Playbook Integrity Lock Demo (§3.1) ===

=== Hash Computation ===

Playbook: demo-model-mvp.playbook.yaml
SHA-256:  52ab204274e7f1170939a74bd48d73766425331b9c7daa2b22f4e6be400c9324

=== Integrity Verification ===

Case 1: Unmodified playbook
  Result: PASSED ✓

Case 2: Modified playbook
  Result: BLOCKED ✗
  Error: Integrity check failed for 'demo-model-mvp': expected 52ab20..., got 55b585...
```

### CLI Usage

```bash
# Generate lock file
apr-qa lock-playbooks playbooks/models -o playbooks/playbook.lock.yaml

# Run with integrity check (default)
apr-qa run playbook.yaml
# Output: Integrity check: PASSED

# Bypass check if needed (not recommended)
apr-qa run playbook.yaml --no-integrity-check
```

## Format Contract (GH-190/191)

The `contract_demo` example demonstrates the shared format contract system.
This contract defines behavioral invariants between the writer (aprender)
and reader (realizar), preventing the class of conversion bugs identified
in the GH-190 and GH-191 Five-Whys analyses.

```bash
cargo run --example contract_demo -p apr-qa-runner
```

### Contract Overview

The contract is defined in `apr_format_contract.yaml` and embedded at
compile time. It is the single source of truth for:

- **Tensor naming** — GGUF-short convention with canonical/forbidden examples
- **Dtype-byte mappings** — GGML type values that writer and reader must agree on
- **Tolerances** — Per-dtype absolute and relative tolerance for statistical comparison
- **Invariants** — Five behavioral invariants (I-1 through I-5)

### Loading and Validating the Contract

```rust
use apr_qa_runner::{
    load_format_contract, validate_dtype_bytes,
    validate_tensor_name, lookup_tolerance,
};

fn main() {
    let contract = load_format_contract().unwrap();

    // Validate no duplicate GGML byte values
    validate_dtype_bytes(&contract).unwrap();

    // Check tensor names against the naming convention
    assert!(validate_tensor_name("0.q_proj.weight", &contract));
    assert!(validate_tensor_name("token_embd.weight", &contract));
    assert!(!validate_tensor_name("model.layers.0.self_attn.q_proj.weight", &contract));

    // Look up per-dtype tolerances
    let (atol, rtol) = lookup_tolerance("Q4_K", &contract).unwrap();
    println!("Q4_K tolerance: atol={atol}, rtol={rtol}");
}
```

### Contract Invariants

| ID  | Name                     | Description | Gate ID |
|-----|--------------------------|-------------|---------|
| I-1 | Round-trip Identity      | `inference(convert(model)) == inference(model)` | `F-CONTRACT-I1-001` |
| I-2 | Tensor Name Bijection    | Writer tensor names == Reader tensor names | `F-CONTRACT-I2-001` |
| I-3 | No Silent Fallbacks      | Unknown dtype/tensor must error, never default to F32 | `F-CONTRACT-I3-001` |
| I-4 | Statistical Preservation | Tensor stats preserved within dtype tolerance | `F-CONTRACT-I4-001` |
| I-5 | Tokenizer Roundtrip      | `encode(decode(tokens)) == tokens` | `F-CONTRACT-I5-001` |

I-1 is implemented as the Golden Rule Test in `executor.rs`. I-2 through I-5
are implemented in `contract.rs` and run during certification when enabled.

### Enabling Contract Tests in Playbooks

Add the `contract_tests` section to any playbook YAML:

```yaml
contract_tests:
  invariants: ["I-2", "I-3", "I-4", "I-5"]
```

The executor runs these invariants after the Golden Rule Test and before
HuggingFace parity checks. Each invariant produces `Evidence` with
`Corroborated` or `Falsified` outcome, following the Popperian methodology.

### Programmatic Configuration

```rust
use apr_qa_runner::{ContractTestConfig, InvariantId};

// Default: all invariants enabled
let config = ContractTestConfig::default();
assert_eq!(config.invariants, vec!["I-2", "I-3", "I-4", "I-5"]);

// Custom: only tensor name and fallback checks
let config = ContractTestConfig {
    invariants: vec!["I-2".to_string(), "I-3".to_string()],
};

// Type-safe invariant dispatch
let id = InvariantId::from_label("I-2").unwrap();
assert_eq!(id.gate_id(), "F-CONTRACT-I2-001");
```

### Output

```
=== Format Contract Demo (GH-190/191) ===

=== Loading Contract ===

Version:     1.0
Invariants:  5
Dtypes:      11
Tolerances:  10

=== Dtype-Byte Validation ===

[PASS] No duplicate GGML byte values

=== Tensor Naming Convention ===

Convention: gguf-short

Tensor name validation:
  0.q_proj.weight                VALID
  token_embd.weight              VALID
  model.layers.0.self_attn.q_proj.weight INVALID

=== Contract Invariants ===

| ID    | Name                      | Status      | Gate ID            |
|-------|---------------------------|-------------|--------------------|
| I-1   | Round-trip Identity       | Implemented | F-CONTRACT-I1-001  |
| I-2   | Tensor Name Bijection     | Contract    | F-CONTRACT-I2-001  |
| I-3   | No Silent Fallbacks       | Contract    | F-CONTRACT-I3-001  |
| I-4   | Statistical Preservation  | Contract    | F-CONTRACT-I4-001  |
| I-5   | Tokenizer Roundtrip       | Contract    | F-CONTRACT-I5-001  |
```

## YAML Playbook Examples

The `playbooks/` directory contains YAML playbooks:

- `playbooks/models/qwen2.5-coder-*-mvp.playbook.yaml` - MVP tier playbooks
- `playbooks/models/qwen2.5-coder-7b-full.playbook.yaml` - 7B full qualification (5-quant ladder)
- `playbooks/models/qwen2.5-coder-1.5b-smoke.playbook.yaml` - CI smoke test
- `playbooks/templates/` - Reusable templates

Run a specific playbook:

```bash
# CI smoke (fastest)
cargo run --bin apr-qa -- run playbooks/models/qwen2.5-coder-1.5b-smoke.playbook.yaml

# MVP (medium)
cargo run --bin apr-qa -- run playbooks/models/qwen2.5-coder-1.5b-mvp.playbook.yaml

# Full qualification with 5-quant ladder (comprehensive)
cargo run --bin apr-qa -- run playbooks/models/qwen2.5-coder-7b-full.playbook.yaml
```
