# Certified Testing

The APR Model QA Framework implements a rigorous **Popperian Falsification** methodology
for model certification. This chapter explains the 170-point Verification Matrix and
certification process.

## Philosophy

Following Karl Popper's critical rationalism, we don't "verify" that models work.
Instead, we attempt to **falsify** specific hypotheses about model behavior. A model
earns certification by **surviving** these refutation attempts.

> "Certified" means we tried very hard to break it in specific ways, and failed.

## Testing Tiers

The framework provides four testing tiers mapped to playbook templates.
All tiers build on the same base grid of **3 formats × 2 backends × 3
modalities = 18 combinations**, scaled by a per-combination scenario count.

| Tier | Scenarios | Formula | Time Limit | Pass → Grade / Status |
|------|-----------|---------|------------|----------------------|
| **Quick-Check** | 10 | 1×1×1×10 | ~1 min | Dev feedback only |
| **MVP** | 18 | 3×2×3×1 | ≤10 min | ≥90% → B / PROVISIONAL |
| **CI-Pipeline** | 150 | 2×1×3×25 | ~15 min | CI gate |
| **Full** | 1,800 | 3×2×3×100 | ≤1 hour | ≥95% → A+ / CERTIFIED |

Only **MVP** and **Full** produce formal certification results. Quick-Check
and CI-Pipeline are for development and continuous integration feedback.

### Quick-Check (Dev Feedback)

Single format, single backend, single modality — 10 scenarios. Use during
development for fast iteration. No certification output.

```bash
cargo run --bin apr-qa -- run playbooks/templates/quick-check.yaml
```

### MVP Tier (Minimum Viable Product)

Tests all 18 format×backend×modality combinations with 1 scenario each.

**Pass Criteria:**
- ≥90% pass rate across all 18 combinations
- All P0 gateways (G1-G4) must pass

**On Pass:** MQS Score = 800, Grade = **B**, Status = **PROVISIONAL**

```bash
apr-qa certify --family qwen-coder --tier mvp
```

### CI-Pipeline (Continuous Integration)

2 formats × 1 backend × 3 modalities × 25 scenarios = 150 tests. Designed
to run in CI on every merge. Not a formal certification tier.

```bash
cargo run --bin apr-qa -- run playbooks/templates/ci-pipeline.yaml
```

### Full Tier (Production Qualification)

Runs the complete 170-point Verification Matrix with 100 scenarios per
combination (3×2×3×100 = 1,800 tests).

**Pass Criteria:**
- ≥95% pass rate on verification matrix
- All P0 gates must pass

**On Pass:** MQS Score = 950+, Grade = **A+**, Status = **CERTIFIED**

```bash
apr-qa certify --family qwen-coder --tier full
```

### Status Summary

| Status | Requirements |
|--------|-------------|
| **CERTIFIED** | Full tier pass (≥95% AND zero P0 failures) |
| **PROVISIONAL** | MVP tier pass (≥90% AND zero P0 failures) |
| **BLOCKED** | Pass rate < 90% OR any P0 failure |

A P0 gateway failure always results in BLOCKED, regardless of tier or
overall pass rate.

## The Verification Matrix (170 Points)

### Class I: Fundamental Integrity (P0 - 50 pts)

Any failure here results in immediate **REJECTED** status.

| Gate | Points | Hypothesis |
|------|--------|------------|
| F-INT-001 | 10 | Memory Safety: No segfaults, buffer overflows |
| F-INT-002 | 10 | Process Termination: Clean exit, no hangs |
| F-INT-003 | 10 | Tensor Validity: No NaN/Inf values |
| F-INT-004 | 10 | Format Fidelity: Round-trip conversion is bitwise identical |
| F-INT-005 | 10 | Determinism: Same seed = same output |

### Class II: Interface Compliance (P1 - 25 pts)

| Gate | Points | Hypothesis |
|------|--------|------------|
| F-API-001 | 5 | JSON Compliance: Valid JSON responses |
| F-API-002 | 5 | Chat Template: No control token leakage |
| F-API-003 | 5 | Health Check: /health returns 200 in <1s |
| F-API-004 | 5 | Error Handling: Invalid input returns 400, not crash |
| F-API-005 | 5 | Streaming: Valid SSE format |

### Class III: Numerical Stability (P1 - 20 pts)

| Gate | Points | Hypothesis |
|------|--------|------------|
| F-NUM-001 | 5 | Attention Entropy: Not collapsed or exploded |
| F-NUM-002 | 5 | LayerNorm Drift: Mean < 1e-3, std dev ~1.0 |
| F-NUM-003 | 5 | Softmax Sum: Outputs sum to 1.0 +/- 1e-6 |
| F-NUM-004 | 5 | Token Probability: Valid range [0, 1] |

### Class IV: Cross-Platform Parity (P2 - 15 pts)

| Gate | Points | Hypothesis |
|------|--------|------------|
| F-PAR-001 | 5 | CPU/GPU Equivalence: Diff < epsilon (1e-5) |
| F-PAR-002 | 5 | Format Parity: GGUF = SafeTensors output |
| F-PAR-003 | 5 | Quantization Impact: Perplexity < 10% degradation |

### Class V: Performance Boundaries (P2 - 20 pts)

| Gate | Points | Hypothesis |
|------|--------|------------|
| F-PERF-001 | 5 | Minimum TPS: >= 10 tokens/second |
| F-PERF-002 | 5 | TTFT: Time to first token < 2000ms |
| F-PERF-003 | 5 | Memory Leak: RSS growth < 5% over 100 requests |
| F-PERF-004 | 5 | GPU Utilization: >= 50% occupancy |

### Class VI: Security & Safety (P0 - 30 pts)

| Gate | Points | Hypothesis |
|------|--------|------------|
| F-SEC-001 | 10 | Path Traversal: No ../ access |
| F-SEC-002 | 10 | Prompt Injection: System prompt protected |
| F-SEC-003 | 10 | DoS Protection: Zip bomb/token flood rejected |

## Throughput Tracking (tok/s)

MVP certification includes throughput measurements for all format × backend combinations:

| Column | Format | Backend | Description |
|--------|--------|---------|-------------|
| GGUF CPU | GGUF | CPU | Tokens/sec with GGUF on CPU |
| GGUF GPU | GGUF | GPU | Tokens/sec with GGUF on GPU |
| APR CPU | APR | CPU | Tokens/sec with APR on CPU |
| APR GPU | APR | GPU | Tokens/sec with APR on GPU |
| ST CPU | SafeTensors | CPU | Tokens/sec with SafeTensors on CPU |
| ST GPU | SafeTensors | GPU | Tokens/sec with SafeTensors on GPU |

These metrics are tracked in `models.csv` and displayed in the README certification table.

### Running Real Profiling

To get actual tok/s measurements, use subprocess mode:

```bash
# Run certification with real profiling (cache auto-populated)
apr-qa certify --family qwen-coder --tier mvp \
  --subprocess \
  --apr-binary apr

# Or with an explicit cache directory
apr-qa certify --family qwen-coder --tier mvp \
  --subprocess \
  --model-cache ~/.cache/apr-models \
  --apr-binary apr
```

When `--model-cache` is omitted, the certifier defaults to `~/.cache/apr-models`
and auto-populates it for each model before execution:

1. Creates `gguf/`, `apr/`, `safetensors/` subdirectories
2. Runs `apr pull <model_id>` to ensure the model is in the pacha cache
3. Symlinks the GGUF from `~/.cache/pacha/models/` into `gguf/model.gguf`
4. Symlinks SafeTensors from `~/.cache/huggingface/hub/` into `safetensors/model.safetensors`
5. Copies `config.json` from the HuggingFace snapshot
6. The `apr/` format is populated during 6-column profiling (GGUF -> APR conversion)

If the cache directory already contains a `.gguf` file, auto-population is skipped.

**Model cache structure:**
```
~/.cache/apr-models/
├── qwen2-5-coder-0-5b-instruct/
│   ├── gguf/
│   │   └── model.gguf -> ~/.cache/pacha/models/<hash>.gguf
│   ├── apr/
│   │   └── model.apr          (created during profiling)
│   └── safetensors/
│       ├── model.safetensors -> ~/.cache/huggingface/hub/.../model.safetensors
│       └── config.json
└── qwen2-5-coder-1-5b-instruct/
    └── ...
```

The `--subprocess` flag tells the certifier to:
1. Auto-populate the model cache (if needed)
2. Find model files in the cache directory
3. Run `apr profile --ci --json` for each format
4. Parse throughput from JSON output
5. Store tok/s values in the certification record

### Playbook Configuration

Enable throughput measurement in playbooks:

```yaml
profile_ci:
  enabled: true
  warmup: 1
  measure: 3
  assertions:
    min_throughput: 1.0  # tok/s minimum threshold
```

## Certification Artifacts

A certified build produces three immutable artifacts:

1. **evidence.json** - Complete raw log of every probe
2. **popperian_report.html** - Human-readable dashboard
3. **CERTIFICATE.md** - Official certification document

## Example: Generate Certificate

```rust
use apr_qa_report::{CertificateGenerator, MqsCalculator};
use apr_qa_report::popperian::PopperianCalculator;

let mqs = mqs_calc.calculate(model_id, &collector)?;
let popperian = popperian_calc.calculate(model_id, &collector);

let generator = CertificateGenerator::new("APR QA Framework");
let certificate = generator.generate(
    model_id,
    version,
    &mqs,
    &popperian,
    evidence_hash,
);

// Generate CERTIFICATE.md
let markdown = generator.to_markdown(&certificate);
std::fs::write("CERTIFICATE.md", markdown)?;
```

Run the example:

```bash
cargo run --example generate_certificate -p apr-qa-report
```

## Using SpecGate IDs

The framework provides a `SpecGate` enum with all 25 gate IDs:

```rust
use apr_qa_runner::SpecGate;

// Get gate ID
assert_eq!(SpecGate::IntMemorySafety.id(), "F-INT-001");

// Get point value
assert_eq!(SpecGate::IntMemorySafety.points(), 10);

// Get priority
assert_eq!(SpecGate::IntMemorySafety.priority(), "P0");

// Total verification matrix points
assert_eq!(SpecGate::total_points(), 160);
```

## Gate Checkers

Use the built-in checkers for each gate class:

```rust
use apr_qa_runner::{
    IntegrityChecker,      // F-INT-*
    ApiComplianceChecker,  // F-API-*
    PerformanceValidator,  // F-PERF-*
    ParityChecker,         // F-PAR-*
    PatternDetector,       // F-NUM-*, F-SEC-*
};

// Check memory safety
let result = IntegrityChecker::check_memory_safety(exit_signal, stderr);
if !result.passed {
    println!("FALSIFIED: {}", result.description);
}

// Check API compliance
let result = ApiComplianceChecker::check_json_compliance(response);
assert!(result.passed, "JSON validation failed");

// Check performance
let result = PerformanceValidator::check_tps(measured_tps, 10.0);
assert!(result.passed, "TPS below threshold");
```

## Validity Period

Certificates expire after **90 days**. Re-certification is required:

- After any model update
- After configuration changes
- Before production deployment
- Quarterly for maintained models

## References

- [Popper, K. R. (1959). *The Logic of Scientific Discovery*](https://en.wikipedia.org/wiki/The_Logic_of_Scientific_Discovery)
- [Taleb, N. N. (2007). *The Black Swan*](https://en.wikipedia.org/wiki/The_Black_Swan:_The_Impact_of_the_Highly_Improbable)
- [Ohno, T. (1988). *Toyota Production System*](https://en.wikipedia.org/wiki/Toyota_Production_System)
