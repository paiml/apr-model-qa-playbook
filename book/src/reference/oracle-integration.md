# Oracle Integration

This chapter describes how the APR QA framework integrates with aprender's Oracle system for certification tracking and model lookup.

## Overview

The Oracle system provides a unified interface for querying model certification status. The QA framework produces evidence and certification data that the Oracle consumes.

```
┌─────────────────────┐
│   Run Playbook      │  cargo run --bin apr-qa -- run playbook.yaml
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Collect Evidence   │  certifications/{model}/evidence.json
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Calculate MQS      │  0-1000 score with G0-G4 gateways
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Update models.csv  │  docs/certifications/models.csv
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Oracle Lookup      │  apr oracle <model_id> → certification status
└─────────────────────┘
```

## Key Files

| File | Purpose |
|------|---------|
| `certifications/{model}/evidence.json` | Raw test evidence (JSON array) |
| `docs/certifications/models.csv` | Certification lookup table |
| `playbooks/evidence.schema.json` | Evidence JSON schema |
| `scripts/validate-schemas.sh` | Schema validation script |
| `scripts/validate-aprender-alignment.sh` | Cross-repo consistency check |

## Gateway System

The gateway system implements Toyota-style quality gates that must all pass for certification.

| Gate | Name | Description | Failure Impact |
|------|------|-------------|----------------|
| G0 | Integrity | config.json matches tensor metadata | Score zeroed |
| G1 | Load | Model loads successfully | Score zeroed |
| G2 | Inference | Basic inference works | Score zeroed |
| G3 | Stability | No crashes or panics | Score zeroed |
| G4 | Quality | Output is not garbage | Score zeroed |

**Any gateway failure zeros the MQS score.** This is the Jidoka principle: stop the line immediately when a defect is detected.

## Evidence Format

Evidence files are JSON arrays containing test results:

```json
[
  {
    "id": "000000000000000018912864878b4460",
    "gate_id": "G0-INTEGRITY-CONFIG",
    "scenario": {
      "model": {"org": "Qwen", "name": "Qwen2.5-Coder-0.5B-Instruct"},
      "modality": "run",
      "backend": "cpu",
      "format": "safetensors"
    },
    "outcome": "Corroborated",
    "reason": "Test passed",
    "output": "G0 PASS: config.json matches tensor metadata",
    "timestamp": "2026-02-04T21:29:00.769208033Z"
  }
]
```

### Outcome Values

| Outcome | Meaning |
|---------|---------|
| `Corroborated` | Hypothesis survived refutation attempt |
| `Falsified` | Hypothesis was refuted by evidence |
| `Timeout` | Test exceeded time limit |
| `Crashed` | Process crashed during execution |
| `Skipped` | Test skipped due to policy or precondition |

## models.csv Schema

The certification lookup table uses this schema:

```csv
model_id,family,parameters,size_category,status,mqs_score,grade,certified_tier,last_certified,g1,g2,g3,g4,...
```

| Column | Type | Description |
|--------|------|-------------|
| `model_id` | string | HuggingFace model ID (org/name) |
| `family` | string | Model family (qwen-coder, llama, etc.) |
| `parameters` | string | Parameter count (0.5B, 7B, etc.) |
| `size_category` | enum | tiny/small/medium/large/xlarge/huge |
| `status` | enum | CERTIFIED/BLOCKED/PENDING/UNTESTED |
| `mqs_score` | int | 0-1000 Model Qualification Score |
| `grade` | enum | A/B/C/D/F/- |
| `g1-g4` | bool | Gateway pass/fail status |

## Family Contract Integration

Playbooks can auto-populate from aprender's family YAML contracts:

```yaml
# playbook.yaml
model:
  hf_repo: "Qwen/Qwen2.5-Coder-0.5B-Instruct"
  family: qwen2
  size_variant: "0.5b"
  # These are auto-populated from family YAML:
  # size_category: tiny
  # expected_hidden_dim: 896
  # expected_num_layers: 24
```

The family contract at `aprender/contracts/model-families/qwen2.yaml` defines:

```yaml
certification:
  size_categories:
    0.5b: tiny
    1.5b: small
    7b: medium
    14b: large
    32b: xlarge

size_variants:
  0.5b:
    hidden_dim: 896
    num_layers: 24
    num_heads: 14
```

## CI Validation

### Schema Validation (PMAT-273)

Validates evidence.json and models.csv:

```bash
./scripts/validate-schemas.sh
```

Checks:
- Evidence JSON syntax and required fields
- Valid outcome values
- models.csv header and constraints
- Valid status, grade, size_category values

### Cross-Repo Alignment (PMAT-274)

Validates playbook alignment with aprender family YAMLs:

```bash
./scripts/validate-aprender-alignment.sh ../aprender
```

Checks:
- size_category matches family YAML
- expected dimensions match size_variants
- Model ID follows family naming conventions

## Certification Workflows

### Certify a New Model

```bash
# 1. Create playbook
cp playbooks/templates/mvp-template.yaml playbooks/models/my-model-mvp.playbook.yaml

# 2. Edit playbook with model details

# 3. Run certification
cargo run --bin apr-qa -- run playbooks/models/my-model-mvp.playbook.yaml \
    --output certifications/my-model/evidence.json

# 4. Update models.csv

# 5. Commit and push
```

### Re-run After Fixes

```bash
# Clear old evidence
rm -rf certifications/my-model/

# Re-run certification
cargo run --bin apr-qa -- run playbooks/models/my-model-mvp.playbook.yaml \
    --output certifications/my-model/evidence.json
```

## Troubleshooting

### Gateway G0 Failed

**Symptom:** `G0 FAIL: config.json says 14 layers but tensors have 24`

**Solution:** Re-download model or verify config.json matches tensor metadata.

### size_category Mismatch

**Symptom:** Cross-repo validation error about size_category

**Solution:** Update playbook's `size_category` to match family YAML's `certification.size_categories`.

### Evidence Validation Failed

**Symptom:** `ERROR: evidence.json: Invalid JSON syntax`

**Solution:** Re-run certification to regenerate evidence, or fix JSON manually.

## API Reference

### Rust Types

```rust
use apr_qa_report::certification_data::{CertificationRow, ModelStatus};
use apr_qa_report::evidence_export::EvidenceExport;

// Read models.csv
let rows = read_models_csv("docs/certifications/models.csv")?;

// Lookup by model ID
if let Some(row) = lookup_model(&rows, "Qwen/Qwen2.5-Coder-0.5B-Instruct") {
    println!("Status: {:?}", row.status);
    println!("MQS: {}", row.mqs_score);
}

// Derive status from evidence
let status = row.derive_status();  // CERTIFIED/BLOCKED/PENDING
let grade = row.derive_grade();    // A/B/C/D/F
```

### CLI Commands

```bash
# Run playbook
apr-qa run playbook.yaml --output evidence.json

# Certify family
apr-qa certify --family qwen-coder --tier mvp

# Validate schemas
./scripts/validate-schemas.sh

# Check aprender alignment
./scripts/validate-aprender-alignment.sh ../aprender
```
