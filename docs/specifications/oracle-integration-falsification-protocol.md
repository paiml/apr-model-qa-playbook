# Oracle Integration & Compiler-Enforced Falsification Protocol

**Version**: 1.0.0
**Status**: Draft
**Created**: 2026-02-06
**Author**: PAIML Engineering
**Tickets**: PMAT-260 through PMAT-275
**Cross-Reference**: aprender `compiler-enforced-model-types-model-oracle.md`

> This specification defines the integration between apr-model-qa-playbook and aprender's
> `apr oracle` CLI command, establishing a bidirectional certification cross-reference
> system with compiler-enforced falsification protocols.

---

## Table of Contents

1. [Abstract](#1-abstract)
2. [Motivation](#2-motivation)
3. [Theoretical Foundation](#3-theoretical-foundation)
4. [Certification Data Architecture](#4-certification-data-architecture)
   - [4.1 Directory Structure](#41-directory-structure)
   - [4.2 models.csv Schema](#42-modelscsv-schema)
   - [4.3 Evidence JSON Schema](#43-evidence-json-schema)
   - [4.4 Playbook Naming Convention](#44-playbook-naming-convention)
5. [Oracle Integration Protocol](#5-oracle-integration-protocol)
   - [5.1 CertificationInfo Contract](#51-certificationinfo-contract)
   - [5.2 Cross-Repository Resolution](#52-cross-repository-resolution)
   - [5.3 MQS Score Calculation](#53-mqs-score-calculation)
   - [5.4 Grade Assignment](#54-grade-assignment)
6. [Family Contract Consumption](#6-family-contract-consumption)
   - [6.1 YAML-Driven Test Matrix Generation](#61-yaml-driven-test-matrix-generation)
   - [6.2 Size Category Alignment](#62-size-category-alignment)
   - [6.3 Tensor Template Validation](#63-tensor-template-validation)
7. [Popperian Falsification Protocol](#7-popperian-falsification-protocol)
   - [7.1 Certification Data Falsifications](#71-certification-data-falsifications)
   - [7.2 Oracle Integration Falsifications](#72-oracle-integration-falsifications)
   - [7.3 Family Contract Falsifications](#73-family-contract-falsifications)
   - [7.4 Evidence Export Falsifications](#74-evidence-export-falsifications)
8. [Implementation Roadmap](#8-implementation-roadmap)
   - [8.1 Phase 1: Certification Data (PMAT-260..263)](#81-phase-1-certification-data)
   - [8.2 Phase 2: Oracle Integration (PMAT-264..267)](#82-phase-2-oracle-integration)
   - [8.3 Phase 3: Family Contract Consumption (PMAT-268..271)](#83-phase-3-family-contract-consumption)
   - [8.4 Phase 4: Compiler Enforcement (PMAT-272..275)](#84-phase-4-compiler-enforcement)
9. [Architectural Decisions](#9-architectural-decisions)
10. [References](#10-references)
11. [Appendix A: PMAT Ticket Descriptions](#appendix-a-pmat-ticket-descriptions)

---

## 1. Abstract

The aprender `apr oracle` command provides model introspection capabilities including
family detection, contract compliance verification, and certification status lookup.
This certification status is sourced from apr-model-qa-playbook, creating a bidirectional
dependency that requires formal specification.

This document defines:

- **Certification data formats** (models.csv, evidence JSON) that oracle consumes
- **Naming conventions** that enable cross-repository model→playbook resolution
- **Falsification protocols** ensuring certification data integrity
- **Family contract consumption** enabling YAML-driven test matrix generation
- **PMAT work items** for complete implementation with quality gates

The integration follows Toyota Production System principles (Ohno, 1988) and Popperian
falsification methodology (Popper, 1959), making certification data integrity a
compile-time guarantee rather than a runtime hope.

---

## 2. Motivation

### 2.1 Current State

The apr-model-qa-playbook currently operates in isolation:

| Component | Current State | Gap |
|-----------|---------------|-----|
| Evidence files | JSON in `output/` and `certifications/` | No standard schema, not consumable by oracle |
| Certification status | Implicit in pass/fail counts | No `models.csv` for oracle lookup |
| Playbook naming | Ad-hoc (`qwen2.5-coder-0.5b-mvp.playbook.yaml`) | Pattern may not match oracle expectations |
| Family contracts | Not consumed | Test matrices hardcoded, not derived from YAML |
| MQS scores | Calculated but not persisted | Oracle cannot report scores |

### 2.2 Target State

After implementation:

1. **Oracle queries certification**: `apr oracle model.gguf` returns MQS score, grade, tier
2. **Bidirectional linking**: Family YAML → playbook path → models.csv → evidence JSON
3. **YAML-driven test matrices**: Playbooks derive configurations from family contracts
4. **Falsification coverage**: Every integration point has explicit falsification tests

### 2.3 Design Goals

1. Make certification data **machine-readable** by oracle and other tools
2. Make playbook↔family contract mapping **deterministic** and **verifiable**
3. Make MQS calculation **reproducible** from evidence JSON alone
4. Make integration errors **detectable at build time** where possible

---

## 3. Theoretical Foundation

This specification draws on established works in quality engineering, scientific
methodology, and systems thinking.

### 3.1 Toyota Production System: Jidoka

> "Automation with a human touch... the machine detects defects and stops itself,
> preventing defective products from being produced."

**Citation**: Ohno, T. (1988). *Toyota Production System: Beyond Large-Scale
Production*. Productivity Press. ISBN 0-915299-14-3.

**Application**: The certification pipeline implements jidoka through:
- **Automatic stop**: If evidence JSON is malformed, oracle reports "UNTESTED" not garbage
- **Andon signal**: Falsification test failures block CI, alerting humans
- **Root cause**: Each falsification test identifies the specific contract violated

### 3.2 Toyota Production System: Poka-Yoke

> "A poka-yoke device is any mechanism that helps an equipment operator avoid mistakes."

**Citation**: Shingo, S. (1986). *Zero Quality Control: Source Inspection and the
Poka-Yoke System*. Productivity Press. ISBN 0-915299-07-0.

**Application**: The naming convention (`{family}-{size}.playbook.yaml`) is a poka-yoke:
- **Mistake-proofing**: Oracle cannot mis-resolve a playbook if the name encodes family+size
- **Source inspection**: Validation at file creation time, not oracle query time
- **Immediate feedback**: Invalid playbook names fail schema validation

### 3.3 Popperian Falsification

> "The criterion of the scientific status of a theory is its falsifiability."

**Citation**: Popper, K.R. (1959). *The Logic of Scientific Discovery*. Hutchinson
& Co. ISBN 0-415-27844-9.

**Application**: Every integration contract has explicit falsification criteria:
- **Prediction**: "If models.csv exists and contains row X, oracle returns CertificationInfo Y"
- **Test**: Construct scenario where prediction fails → implementation is broken
- **Non-tautology**: Predictions are non-trivial and could be false

### 3.4 Scientific Method: Reproducibility

> "The goal of scientific computing is not numbers, but insight."

**Citation**: Hamming, R.W. (1962). *Numerical Methods for Scientists and Engineers*.
McGraw-Hill. ISBN 0-486-65241-6.

**Application**: MQS scores must be reproducible:
- **Same evidence → same score**: `calculate_mqs(evidence.json)` is deterministic
- **Audit trail**: Evidence JSON contains all inputs needed to reproduce the score
- **No hidden state**: No external dependencies (timestamps, random seeds) affect scores

### 3.5 Contract Programming

> "A software system is correct if it does what its contracts say it does."

**Citation**: Meyer, B. (1992). *Applying "Design by Contract"*. IEEE Computer,
25(10), 40-51. DOI: 10.1109/2.161279.

**Application**: The oracle↔playbook integration is a contract:
- **Precondition**: Playbook exists at `playbook_path`, models.csv has matching row
- **Postcondition**: Oracle returns CertificationInfo with correct status, score, grade
- **Invariant**: `csv_family_key` + size uniquely identifies one certification row

### 3.6 Defensive Programming

> "Write code as if the person who ends up maintaining your code is a violent
> psychopath who knows where you live."

**Citation**: Hunt, A. & Thomas, D. (1999). *The Pragmatic Programmer*. Addison-Wesley.
ISBN 0-201-61622-X.

**Application**: Handle missing/malformed data gracefully:
- **Missing models.csv**: Oracle reports "Certification data not available"
- **Missing playbook**: Oracle reports "Playbook not found at {path}"
- **Malformed evidence**: Oracle reports "Evidence parse error" with details

### 3.7 Separation of Concerns

> "The technique of separating a program into distinct sections, such that each
> section addresses a separate concern."

**Citation**: Dijkstra, E.W. (1982). *On the Role of Scientific Thought*. In
*Selected Writings on Computing*. Springer. ISBN 0-387-90652-5.

**Application**: Clear responsibility boundaries:
- **apr-model-qa-playbook**: Produces certification data (models.csv, evidence.json)
- **aprender**: Consumes certification data (oracle queries)
- **Neither**: Duplicates the other's data or logic

---

## 4. Certification Data Architecture

### 4.1 Directory Structure

```
apr-model-qa-playbook/
├── docs/
│   └── certifications/
│       ├── models.csv                    # Oracle lookup table
│       └── evidence/                     # Per-model evidence snapshots
│           ├── qwen2.5-coder-0.5b.json
│           ├── qwen2.5-coder-1.5b.json
│           └── ...
├── playbooks/
│   └── models/
│       ├── qwen2.5-coder-0.5b-smoke.playbook.yaml
│       ├── qwen2.5-coder-0.5b-mvp.playbook.yaml
│       ├── qwen2.5-coder-1.5b-mvp.playbook.yaml
│       └── ...
└── certifications/                       # Legacy location (deprecated)
    └── {model}/evidence.json
```

### 4.2 models.csv Schema

The models.csv file is the authoritative certification lookup table consumed by oracle.

```csv
hf_repo,family,size,status,mqs_score,grade,tier,playbook_path,evidence_path,last_updated
Qwen/Qwen2.5-Coder-0.5B-Instruct,qwen2,0.5b,CERTIFIED,847,A,mvp,playbooks/models/qwen2.5-coder-0.5b-mvp.playbook.yaml,docs/certifications/evidence/qwen2.5-coder-0.5b.json,2026-02-06T12:00:00Z
Qwen/Qwen2.5-Coder-1.5B-Instruct,qwen2,1.5b,BLOCKED,523,C,mvp,playbooks/models/qwen2.5-coder-1.5b-mvp.playbook.yaml,docs/certifications/evidence/qwen2.5-coder-1.5b.json,2026-02-06T12:00:00Z
Qwen/Qwen2.5-Coder-3B-Instruct,qwen2,3b,BLOCKED,231,F,mvp,playbooks/models/qwen2.5-coder-3b-mvp.playbook.yaml,docs/certifications/evidence/qwen2.5-coder-3b.json,2026-02-06T12:00:00Z
```

**Column Definitions**:

| Column | Type | Description | Constraint |
|--------|------|-------------|------------|
| `hf_repo` | string | HuggingFace repository ID | Primary key, format `org/name` |
| `family` | string | Model family from aprender contracts | Must match `csv_family_key` in family YAML |
| `size` | string | Size variant (0.5b, 1.5b, 3b, etc.) | Must match variant in family YAML |
| `status` | enum | Certification status | One of: CERTIFIED, BLOCKED, UNTESTED |
| `mqs_score` | int | Model Qualification Score | 0-1000 |
| `grade` | enum | Letter grade | One of: A, B, C, D, F |
| `tier` | enum | Certification tier achieved | One of: smoke, mvp, full |
| `playbook_path` | string | Relative path to playbook | Must exist |
| `evidence_path` | string | Relative path to evidence JSON | Must exist |
| `last_updated` | ISO8601 | Last certification run timestamp | UTC timezone |

**Status Definitions**:

- **CERTIFIED**: MQS ≥ 800, all gateway gates passed, tier requirements met
- **BLOCKED**: MQS < 800 or gateway gate failure, cannot be used in production
- **UNTESTED**: No certification run completed, evidence missing

**Grade Thresholds**:

| Grade | MQS Range | Description |
|-------|-----------|-------------|
| A | 900-1000 | Excellent - production ready, all tests pass |
| B | 800-899 | Good - minor issues, production ready with caveats |
| C | 600-799 | Marginal - significant issues, not recommended |
| D | 400-599 | Poor - major issues, many test failures |
| F | 0-399 | Failing - critical issues, do not use |

### 4.3 Evidence JSON Schema

Evidence JSON files contain the complete test results for reproducibility.

```json
{
  "$schema": "https://paiml.com/schemas/apr-qa-evidence.schema.json",
  "version": "1.0.0",
  "model": {
    "hf_repo": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    "family": "qwen2",
    "size": "0.5b",
    "format": "safetensors"
  },
  "playbook": {
    "name": "qwen2.5-coder-0.5b-mvp",
    "version": "1.0.0",
    "tier": "mvp"
  },
  "summary": {
    "total_scenarios": 47,
    "passed": 27,
    "failed": 20,
    "skipped": 0,
    "pass_rate": 0.574,
    "duration_ms": 1134808,
    "timestamp": "2026-02-06T12:00:00Z"
  },
  "mqs": {
    "score": 574,
    "grade": "C",
    "gateway_passed": true,
    "category_scores": {
      "inference": 600,
      "conversion": 400,
      "stability": 700,
      "performance": 600
    }
  },
  "gates": {
    "G0-PULL-001": {"passed": true, "reason": "Model acquired"},
    "G0-VALIDATE-001": {"passed": true, "reason": "Physics validated"},
    "G0-INTEGRITY-CONFIG": {"passed": true, "reason": "Config valid"},
    "G1-MODEL-LOADS": {"passed": true, "reason": "Model loaded"},
    "G2-BASIC-INFERENCE": {"passed": true, "reason": "Inference works"},
    "G3-NO-CRASHES": {"passed": true, "reason": "No crashes"},
    "G4-OUTPUT-QUALITY": {"passed": false, "reason": "Conversion diffs exceed tolerance"}
  },
  "evidence": [
    // Full evidence array from test run
  ]
}
```

### 4.4 Playbook Naming Convention

Playbooks follow a deterministic naming pattern enabling oracle resolution:

```
{family}-{size}[-{tier}].playbook.yaml
```

**Examples**:
- `qwen2.5-coder-0.5b-smoke.playbook.yaml` → family=qwen2, size=0.5b, tier=smoke
- `qwen2.5-coder-1.5b-mvp.playbook.yaml` → family=qwen2, size=1.5b, tier=mvp
- `llama-3.2-1b-full.playbook.yaml` → family=llama, size=1b, tier=full

**Resolution Algorithm** (implemented by oracle):

```python
def resolve_playbook(family: str, size: str, tier: str = "mvp") -> str:
    """
    Resolve family YAML's playbook_path template to concrete path.

    Example:
      playbook_path: "../apr-model-qa-playbook/playbooks/models/qwen2.5-coder-{size}.playbook.yaml"
      family=qwen2, size=0.5b, tier=mvp
      → "../apr-model-qa-playbook/playbooks/models/qwen2.5-coder-0.5b-mvp.playbook.yaml"
    """
    template = family_yaml.certification.playbook_path
    # Replace {size} placeholder
    path = template.replace("{size}", f"{size}-{tier}")
    return path
```

---

## 5. Oracle Integration Protocol

### 5.1 CertificationInfo Contract

The oracle expects this data structure (defined in aprender):

```rust
#[derive(Debug, Clone, Serialize)]
pub struct CertificationInfo {
    /// Certification status: "CERTIFIED", "BLOCKED", "UNTESTED"
    pub status: String,
    /// Model Qualification Score (0-1000)
    pub mqs_score: Option<u32>,
    /// Letter grade: "A", "B", "C", "D", "F"
    pub grade: Option<String>,
    /// Certification tier achieved: "smoke", "mvp", "full"
    pub certified_tier: Option<String>,
    /// Relative path to playbook YAML
    pub playbook_path: Option<String>,
}
```

**Contract**:

| Precondition | Postcondition |
|--------------|---------------|
| `models.csv` exists and is valid CSV | Oracle parses without error |
| Row exists for `(hf_repo)` | `CertificationInfo` populated from row |
| Row missing for `(hf_repo)` | `status = "UNTESTED"`, other fields `None` |
| `models.csv` missing | Message: "Certification data not available" |

### 5.2 Cross-Repository Resolution

Oracle resolves certification data through this chain:

```
User query: apr oracle model.gguf
    │
    ▼
1. Detect family from tensor names/metadata
    │ Result: family = "qwen2"
    ▼
2. Detect size from config (hidden_dim, num_layers)
    │ Result: size = "0.5b"
    ▼
3. Load family YAML (contracts/model-families/qwen2.yaml)
    │ Result: csv_family_key = "qwen-coder"
    ▼
4. Search models.csv for matching (family, size)
    │ Query: family="qwen2" AND size="0.5b"
    ▼
5. Return CertificationInfo from matched row
    │ Result: status=CERTIFIED, mqs=847, grade=A
    ▼
Output: Model introspection report with certification status
```

### 5.3 MQS Score Calculation

MQS (Model Qualification Score) is calculated from evidence:

```rust
pub fn calculate_mqs(evidence: &EvidenceCollection) -> MqsScore {
    // Gateway check (G1-G4 must all pass)
    let gateway_passed = evidence.gates.iter()
        .filter(|(id, _)| id.starts_with("G"))
        .all(|(_, gate)| gate.passed);

    if !gateway_passed {
        return MqsScore { score: 0, grade: "F", gateway_passed: false, .. };
    }

    // Category scores (weighted average)
    let inference_score = calculate_inference_score(evidence);    // 40%
    let conversion_score = calculate_conversion_score(evidence);  // 30%
    let stability_score = calculate_stability_score(evidence);    // 20%
    let performance_score = calculate_performance_score(evidence); // 10%

    let score = (inference_score * 0.4
               + conversion_score * 0.3
               + stability_score * 0.2
               + performance_score * 0.1) as u32;

    let grade = match score {
        900..=1000 => "A",
        800..=899 => "B",
        600..=799 => "C",
        400..=599 => "D",
        _ => "F",
    };

    MqsScore { score, grade, gateway_passed: true, category_scores: .. }
}
```

**Category Definitions**:

| Category | Weight | Inputs | Description |
|----------|--------|--------|-------------|
| Inference | 40% | F-INF-* gates, run/chat/serve scenarios | Core inference correctness |
| Conversion | 30% | F-CONV-* gates, round-trip tests | Format conversion fidelity |
| Stability | 20% | G3, timeout/crash counts | Runtime stability |
| Performance | 10% | Profile CI assertions, tok/s | Throughput thresholds |

### 5.4 Grade Assignment

Grades map MQS scores to certification decisions:

| Grade | MQS | Status | Implications |
|-------|-----|--------|--------------|
| A | 900+ | CERTIFIED | Production ready, recommended |
| B | 800-899 | CERTIFIED | Production ready, minor caveats |
| C | 600-799 | BLOCKED | Not production ready, needs fixes |
| D | 400-599 | BLOCKED | Significant issues, major rework |
| F | <400 | BLOCKED | Critical failures, unsafe to use |

---

## 6. Family Contract Consumption

### 6.1 YAML-Driven Test Matrix Generation

Playbooks can derive configurations from aprender's family YAML contracts:

```yaml
# In aprender: contracts/model-families/qwen2.yaml
size_variants:
  0.5b:
    hidden_dim: 896
    num_layers: 24
    num_heads: 14
    num_kv_heads: 2
  1.5b:
    hidden_dim: 1536
    num_layers: 28
    num_heads: 12
    num_kv_heads: 2
```

**Generated Test Matrix** (derived from YAML):

```yaml
# In apr-model-qa-playbook: playbooks/models/qwen2.5-coder-0.5b-mvp.playbook.yaml
# AUTO-GENERATED from contracts/model-families/qwen2.yaml
model:
  hf_repo: "Qwen/Qwen2.5-Coder-0.5B-Instruct"
  # These values derived from qwen2.yaml size_variants.0.5b
  expected_hidden_dim: 896
  expected_num_layers: 24
  expected_num_heads: 14
  expected_num_kv_heads: 2
```

### 6.2 Size Category Alignment

The `SizeCategory` enum in apr-model-qa-playbook MUST align with aprender's family YAML:

| aprender YAML | apr-model-qa-playbook | Max Workers |
|---------------|----------------------|-------------|
| `0.5b` | `SizeCategory::Tiny` | 4 |
| `1.5b` | `SizeCategory::Small` | 4 |
| `3b` | `SizeCategory::Medium` | 2 |
| `7b` | `SizeCategory::Large` | 1 |
| `14b` | `SizeCategory::Large` | 1 |
| `32b` | `SizeCategory::Xlarge` | 1 |

**Contract**: `family_yaml.certification.size_categories[variant]` MUST equal
`playbook.model.size_category.to_string().to_lowercase()`.

### 6.3 Tensor Template Validation

Family YAMLs define expected tensor patterns that playbooks validate:

```yaml
# In aprender: contracts/model-families/qwen2.yaml
tensor_templates:
  embedding: "model.embed_tokens.weight"
  lm_head: "lm_head.weight"
  layer_pattern: "model.layers.{layer}"
  layer_tensors:
    - "input_layernorm.weight"
    - "self_attn.q_proj.weight"
    - "self_attn.q_proj.bias"
    - "self_attn.k_proj.weight"
    - "self_attn.k_proj.bias"
    # ... etc
```

**Playbook Validation** (G0-VALIDATE):

```rust
fn validate_tensor_names(model: &Model, family_yaml: &FamilyContract) -> Result<()> {
    for expected in family_yaml.tensor_templates.required_tensors() {
        if !model.has_tensor(&expected) {
            return Err(Error::MissingTensor(expected));
        }
    }
    Ok(())
}
```

---

## 7. Popperian Falsification Protocol

Per Popper (1959), each validation rule must make a prediction that could be proven
false. If a falsification test finds a counterexample, the implementation is broken.

### 7.1 Certification Data Falsifications

**FALSIFY-CERT-001**: models.csv round-trip integrity

```
Prediction: Writing models.csv and reading it back produces identical data.

Falsification test:
  fn falsify_cert_001() {
      let original = vec![CertificationRow { hf_repo: "Qwen/...", mqs_score: 847, .. }];
      write_models_csv(&original, "test.csv")?;
      let parsed = read_models_csv("test.csv")?;
      assert_eq!(original, parsed);
  }

If test fails: CSV serialization/deserialization is lossy.
```

**FALSIFY-CERT-002**: Status derivation from MQS score

```
Prediction: status is deterministically derived from mqs_score and gateway_passed.

Falsification test:
  fn falsify_cert_002() {
      // Gateway passed, high score → CERTIFIED
      assert_eq!(derive_status(850, true), "CERTIFIED");
      // Gateway passed, low score → BLOCKED
      assert_eq!(derive_status(500, true), "BLOCKED");
      // Gateway failed, any score → BLOCKED
      assert_eq!(derive_status(950, false), "BLOCKED");
  }

If test fails: Status derivation logic is inconsistent.
```

**FALSIFY-CERT-003**: Grade derivation from MQS score

```
Prediction: grade is deterministically derived from mqs_score using fixed thresholds.

Falsification test:
  fn falsify_cert_003() {
      assert_eq!(derive_grade(950), "A");
      assert_eq!(derive_grade(850), "B");
      assert_eq!(derive_grade(700), "C");
      assert_eq!(derive_grade(500), "D");
      assert_eq!(derive_grade(300), "F");
      // Boundary cases
      assert_eq!(derive_grade(900), "A");  // Lower bound of A
      assert_eq!(derive_grade(899), "B");  // Upper bound of B
  }

If test fails: Grade thresholds do not match specification.
```

**FALSIFY-CERT-004**: Playbook path existence validation

```
Prediction: All playbook_path values in models.csv point to existing files.

Falsification test:
  fn falsify_cert_004() {
      let rows = read_models_csv("docs/certifications/models.csv")?;
      for row in rows {
          let path = Path::new(&row.playbook_path);
          assert!(path.exists(), "Playbook not found: {}", row.playbook_path);
      }
  }

If test fails: models.csv contains stale/invalid playbook references.
```

**FALSIFY-CERT-005**: Evidence path existence validation

```
Prediction: All evidence_path values in models.csv point to existing files.

Falsification test:
  fn falsify_cert_005() {
      let rows = read_models_csv("docs/certifications/models.csv")?;
      for row in rows {
          let path = Path::new(&row.evidence_path);
          assert!(path.exists(), "Evidence not found: {}", row.evidence_path);
      }
  }

If test fails: models.csv contains stale/invalid evidence references.
```

### 7.2 Oracle Integration Falsifications

**FALSIFY-ORC-INT-001**: Oracle returns correct CertificationInfo

```
Prediction: Oracle returns CertificationInfo matching models.csv row.

Falsification test (integration):
  $ apr oracle hf://Qwen/Qwen2.5-Coder-0.5B-Instruct --json | jq '.certification'
  Expected: {"status": "CERTIFIED", "mqs_score": 847, "grade": "A", "tier": "mvp", ...}

  $ grep "Qwen/Qwen2.5-Coder-0.5B-Instruct" models.csv
  Expected: Same values as oracle output

If values differ: Oracle↔CSV sync is broken.
```

**FALSIFY-ORC-INT-002**: Oracle handles missing certification gracefully

```
Prediction: Oracle returns status="UNTESTED" for models not in models.csv.

Falsification test (integration):
  $ apr oracle hf://SomeUnknown/UncertifiedModel --json | jq '.certification.status'
  Expected: "UNTESTED"

If returns error or crashes: Missing certification handling is broken.
```

**FALSIFY-ORC-INT-003**: Oracle handles missing playbook repo gracefully

```
Prediction: Oracle reports informative message when apr-model-qa-playbook not found.

Falsification test (integration):
  $ cd /tmp && apr oracle test.gguf 2>&1 | grep -i "certification"
  Expected: Contains "not available" or "not found"

If crashes or returns garbage: Missing repo handling is broken.
```

### 7.3 Family Contract Falsifications

**FALSIFY-FAM-001**: Size category alignment

```
Prediction: Playbook size_category matches family YAML size_categories mapping.

Falsification test:
  fn falsify_fam_001() {
      let family_yaml = load_family_yaml("qwen2.yaml")?;
      let playbook = load_playbook("qwen2.5-coder-0.5b-mvp.playbook.yaml")?;

      let expected = family_yaml.certification.size_categories.get("0.5b");
      let actual = playbook.model.size_category.to_string().to_lowercase();

      assert_eq!(expected, Some(&actual));
  }

If test fails: Size category mapping is misaligned.
```

**FALSIFY-FAM-002**: Tensor template coverage

```
Prediction: G0-VALIDATE checks all tensors in family YAML tensor_templates.

Falsification test:
  fn falsify_fam_002() {
      let family_yaml = load_family_yaml("qwen2.yaml")?;
      let validate_checks = extract_tensor_checks_from_g0_validate();

      for tensor in family_yaml.tensor_templates.required_tensors() {
          assert!(validate_checks.contains(&tensor),
              "Missing tensor check: {}", tensor);
      }
  }

If test fails: G0-VALIDATE does not cover all family-required tensors.
```

**FALSIFY-FAM-003**: Hidden dimensions match

```
Prediction: Playbook expected_hidden_dim matches family YAML size_variants.

Falsification test:
  fn falsify_fam_003() {
      let family_yaml = load_family_yaml("qwen2.yaml")?;
      let playbook = load_playbook("qwen2.5-coder-0.5b-mvp.playbook.yaml")?;

      let expected = family_yaml.size_variants.get("0.5b").hidden_dim;
      let actual = playbook.model.expected_hidden_dim;

      assert_eq!(expected, actual);
  }

If test fails: Playbook hardcodes wrong dimensions.
```

### 7.4 Evidence Export Falsifications

**FALSIFY-EXP-001**: Evidence JSON schema compliance

```
Prediction: Generated evidence.json validates against schema.

Falsification test:
  fn falsify_exp_001() {
      let evidence = generate_evidence_json(&test_results)?;
      let schema = load_schema("apr-qa-evidence.schema.json")?;

      assert!(schema.validate(&evidence).is_ok());
  }

If test fails: Evidence JSON does not match schema.
```

**FALSIFY-EXP-002**: MQS reproducibility from evidence

```
Prediction: calculate_mqs(evidence) produces same score as stored in evidence.

Falsification test:
  fn falsify_exp_002() {
      let evidence = load_evidence_json("qwen2.5-coder-0.5b.json")?;
      let recalculated = calculate_mqs(&evidence.evidence);

      assert_eq!(evidence.mqs.score, recalculated.score);
  }

If test fails: MQS calculation is non-deterministic or evidence is incomplete.
```

**FALSIFY-EXP-003**: Evidence contains all gate results

```
Prediction: evidence.gates contains results for all defined gates.

Falsification test:
  fn falsify_exp_003() {
      let evidence = load_evidence_json("qwen2.5-coder-0.5b.json")?;
      let required_gates = ["G0-PULL-001", "G0-VALIDATE-001", "G1-MODEL-LOADS",
                           "G2-BASIC-INFERENCE", "G3-NO-CRASHES", "G4-OUTPUT-QUALITY"];

      for gate in required_gates {
          assert!(evidence.gates.contains_key(gate), "Missing gate: {}", gate);
      }
  }

If test fails: Evidence export skips gates.
```

---

## 8. Implementation Roadmap

### 8.1 Phase 1: Certification Data (PMAT-260..263)

| Ticket | Pri | Title | Deliverables |
|--------|-----|-------|--------------|
| PMAT-260 | 1 | models.csv schema and parser | `CertificationRow` struct, CSV read/write, schema validation |
| PMAT-261 | 1 | Evidence JSON schema | JSON Schema file, `EvidenceExport` struct, serialization |
| PMAT-262 | 2 | MQS calculation from evidence | `calculate_mqs()` function, category score functions |
| PMAT-263 | 2 | Grade and status derivation | `derive_grade()`, `derive_status()`, threshold constants |

**Dependencies**: None (foundation layer)
**Estimated effort**: 2-3 days total

### 8.2 Phase 2: Oracle Integration (PMAT-264..267)

| Ticket | Pri | Title | Deliverables |
|--------|-----|-------|--------------|
| PMAT-264 | 2 | models.csv export command | `apr-qa export-csv` command, updates models.csv from evidence |
| PMAT-265 | 2 | Evidence JSON export command | `apr-qa export-evidence` command, schema-compliant JSON |
| PMAT-266 | 2 | Playbook naming convention enforcement | Schema validation, naming pattern regex, migration script |
| PMAT-267 | 3 | Oracle integration tests | End-to-end tests: oracle queries playbook data correctly |

**Dependencies**: PMAT-260..263
**Estimated effort**: 3-4 days total

### 8.3 Phase 3: Family Contract Consumption (PMAT-268..271)

| Ticket | Pri | Title | Deliverables |
|--------|-----|-------|--------------|
| PMAT-268 | 3 | Family YAML loader | Read aprender's family YAML, `FamilyContract` struct |
| PMAT-269 | 3 | Test matrix generation from YAML | Generate playbook configs from family YAML size_variants |
| PMAT-270 | 3 | Size category auto-alignment | Auto-set `size_category` from family YAML mapping |
| PMAT-271 | 3 | Tensor template validation | G0-VALIDATE uses family YAML tensor_templates |

**Dependencies**: PMAT-264..267, aprender family YAMLs
**Estimated effort**: 4-5 days total

### 8.4 Phase 4: Compiler Enforcement (PMAT-272..275)

| Ticket | Pri | Title | Deliverables |
|--------|-----|-------|--------------|
| PMAT-272 | 4 | Falsification test suite | All FALSIFY-* tests implemented, CI integration |
| PMAT-273 | 4 | Schema validation in CI | JSON Schema validation for evidence, CSV validation |
| PMAT-274 | 4 | Cross-repo consistency checks | CI job validates apr-model-qa-playbook ↔ aprender alignment |
| PMAT-275 | 4 | Documentation and examples | Updated CLAUDE.md, example workflows, troubleshooting |

**Dependencies**: PMAT-268..271
**Estimated effort**: 3-4 days total

---

## 9. Architectural Decisions

### 9.1 CSV vs JSON for models.csv

**Decision**: Use CSV format for the certification lookup table.

**Rationale**:
- **Human readable**: CSV is trivially viewable in any text editor or spreadsheet
- **Diff-friendly**: Line-based format makes git diffs meaningful
- **Oracle compatibility**: aprender's oracle expects CSV (per their spec)
- **Simplicity**: No nested structures needed for lookup table

**Trade-off**: JSON would allow nested structures and better typing, but CSV is
sufficient for a flat lookup table and matches aprender's expectations.

### 9.2 Evidence in docs/certifications/ vs output/

**Decision**: Publish stable evidence snapshots to `docs/certifications/evidence/`.

**Rationale**:
- **Separation**: `output/` is ephemeral (gitignored), `docs/` is committed
- **Reproducibility**: Published evidence enables audit and reproduction
- **Oracle access**: Oracle needs committed files, not ephemeral test outputs
- **History**: Git tracks certification evolution over time

**Trade-off**: Larger repository size, but evidence files are typically <100KB each.

### 9.3 Family YAML in aprender vs duplicated

**Decision**: Read family YAMLs from `../aprender/contracts/model-families/`.

**Rationale**:
- **Single source of truth**: Family contracts defined once, consumed everywhere
- **No drift**: Changes to family contracts automatically reflected in playbooks
- **Separation of concerns**: apr-model-qa-playbook tests, aprender defines contracts

**Trade-off**: Cross-repo dependency, but this is intentional and specified in
aprender's oracle integration section.

### 9.4 MQS Calculation: Playbook vs Report Crate

**Decision**: Implement MQS calculation in apr-qa-report crate.

**Rationale**:
- **Existing responsibility**: Report crate already handles scoring
- **Reusability**: Both CLI and evidence export use the same calculation
- **Testing**: Report crate has existing test infrastructure

---

## 10. References

1. Ohno, T. (1988). *Toyota Production System: Beyond Large-Scale Production*.
   Productivity Press. ISBN 0-915299-14-3.

2. Shingo, S. (1986). *Zero Quality Control: Source Inspection and the Poka-Yoke
   System*. Productivity Press. ISBN 0-915299-07-0.

3. Popper, K.R. (1959). *The Logic of Scientific Discovery*. Hutchinson & Co.
   ISBN 0-415-27844-9.

4. Meyer, B. (1992). Applying "Design by Contract". *IEEE Computer*, 25(10), 40-51.
   DOI: 10.1109/2.161279.

5. Hamming, R.W. (1962). *Numerical Methods for Scientists and Engineers*.
   McGraw-Hill. ISBN 0-486-65241-6.

6. Hunt, A. & Thomas, D. (1999). *The Pragmatic Programmer*. Addison-Wesley.
   ISBN 0-201-61622-X.

7. Dijkstra, E.W. (1982). On the Role of Scientific Thought. In *Selected Writings
   on Computing*. Springer. ISBN 0-387-90652-5.

8. PAIML Engineering. (2026). Compiler-Enforced Model Types & Model Oracle
   [Specification]. aprender repository.

---

## Appendix A: PMAT Ticket Descriptions

### PMAT-260: models.csv schema and parser

**Priority**: 1 (Foundation)
**Estimate**: 4 hours
**Dependencies**: None

**Description**:

Define the `CertificationRow` struct matching the models.csv schema (§4.2). Implement
CSV parsing with serde_csv. Add schema validation ensuring required columns exist and
values conform to constraints.

**Acceptance Criteria**:
- [ ] `CertificationRow` struct with all fields from §4.2
- [ ] `read_models_csv(path) -> Result<Vec<CertificationRow>>`
- [ ] `write_models_csv(rows, path) -> Result<()>`
- [ ] Schema validation (status enum, grade enum, paths exist)
- [ ] FALSIFY-CERT-001 test passes

**Files**:
- `crates/apr-qa-report/src/certification.rs` (new)
- `crates/apr-qa-report/src/lib.rs` (export)

---

### PMAT-261: Evidence JSON schema

**Priority**: 1 (Foundation)
**Estimate**: 4 hours
**Dependencies**: None

**Description**:

Define JSON Schema for evidence export (§4.3). Implement `EvidenceExport` struct with
serde serialization. Schema file at `schemas/apr-qa-evidence.schema.json`.

**Acceptance Criteria**:
- [ ] JSON Schema file with all fields from §4.3
- [ ] `EvidenceExport` struct matching schema
- [ ] `to_json()` and `from_json()` methods
- [ ] FALSIFY-EXP-001 test passes

**Files**:
- `schemas/apr-qa-evidence.schema.json` (new)
- `crates/apr-qa-runner/src/evidence_export.rs` (new)

---

### PMAT-262: MQS calculation from evidence

**Priority**: 2
**Estimate**: 6 hours
**Dependencies**: PMAT-261

**Description**:

Implement `calculate_mqs()` function per §5.3. Calculate category scores (inference,
conversion, stability, performance) with documented weights. Ensure deterministic
calculation from evidence alone.

**Acceptance Criteria**:
- [ ] `calculate_mqs(evidence) -> MqsScore` function
- [ ] Category score functions for each category
- [ ] Weighted average calculation with documented weights
- [ ] FALSIFY-EXP-002 test passes (reproducibility)

**Files**:
- `crates/apr-qa-report/src/mqs.rs` (new)

---

### PMAT-263: Grade and status derivation

**Priority**: 2
**Estimate**: 2 hours
**Dependencies**: PMAT-262

**Description**:

Implement `derive_grade()` and `derive_status()` functions per §5.4. Define threshold
constants. Ensure deterministic derivation from MQS score and gateway status.

**Acceptance Criteria**:
- [ ] `derive_grade(mqs_score) -> Grade` function
- [ ] `derive_status(mqs_score, gateway_passed) -> Status` function
- [ ] Grade threshold constants matching §4.2
- [ ] FALSIFY-CERT-002 and FALSIFY-CERT-003 tests pass

**Files**:
- `crates/apr-qa-report/src/mqs.rs` (extend)

---

### PMAT-264: models.csv export command

**Priority**: 2
**Estimate**: 4 hours
**Dependencies**: PMAT-260, PMAT-262, PMAT-263

**Description**:

Add `apr-qa export-csv` command that generates/updates models.csv from evidence files.
Scans `docs/certifications/evidence/`, calculates MQS for each, writes CSV.

**Acceptance Criteria**:
- [ ] `apr-qa export-csv` command
- [ ] Reads all evidence JSON from configured directory
- [ ] Calculates MQS, grade, status for each
- [ ] Writes valid models.csv
- [ ] FALSIFY-CERT-004 and FALSIFY-CERT-005 tests pass after export

**Files**:
- `crates/apr-qa-cli/src/commands/export_csv.rs` (new)

---

### PMAT-265: Evidence JSON export command

**Priority**: 2
**Estimate**: 4 hours
**Dependencies**: PMAT-261

**Description**:

Add `apr-qa export-evidence` command that exports test results to schema-compliant
JSON in `docs/certifications/evidence/`.

**Acceptance Criteria**:
- [ ] `apr-qa export-evidence --output path` command
- [ ] Generates schema-compliant JSON from test run
- [ ] Includes all gates, evidence array, MQS calculation
- [ ] Validates against schema before writing

**Files**:
- `crates/apr-qa-cli/src/commands/export_evidence.rs` (new)

---

### PMAT-266: Playbook naming convention enforcement

**Priority**: 2
**Estimate**: 3 hours
**Dependencies**: None

**Description**:

Add schema validation for playbook naming convention (§4.4). Implement regex pattern
matching. Add migration script for existing playbooks.

**Acceptance Criteria**:
- [ ] Naming pattern regex: `{family}-{size}[-{tier}].playbook.yaml`
- [ ] Schema validation rejects non-conforming names
- [ ] Migration script renames existing playbooks
- [ ] All existing playbooks conform to convention

**Files**:
- `playbooks/playbook.schema.yaml` (update)
- `scripts/migrate-playbook-names.sh` (new)

---

### PMAT-267: Oracle integration tests

**Priority**: 3
**Estimate**: 4 hours
**Dependencies**: PMAT-264, PMAT-265, PMAT-266

**Description**:

End-to-end integration tests verifying oracle correctly queries certification data.
Requires aprender oracle CLI.

**Acceptance Criteria**:
- [ ] Integration test: oracle returns correct CertificationInfo
- [ ] Integration test: oracle handles missing certification
- [ ] FALSIFY-ORC-INT-001, 002, 003 tests pass
- [ ] CI job runs integration tests

**Files**:
- `tests/integration/oracle_integration.rs` (new)

---

### PMAT-268: Family YAML loader

**Priority**: 3
**Estimate**: 4 hours
**Dependencies**: PMAT-267

**Description**:

Implement loader for aprender's family YAML contracts. Define `FamilyContract` struct.
Handle missing aprender repo gracefully.

**Acceptance Criteria**:
- [ ] `FamilyContract` struct matching aprender YAML schema
- [ ] `load_family_yaml(path) -> Result<FamilyContract>`
- [ ] Graceful handling when `../aprender` not present
- [ ] Tests with sample YAML

**Files**:
- `crates/apr-qa-runner/src/family_contract.rs` (new)

---

### PMAT-269: Test matrix generation from YAML

**Priority**: 3
**Estimate**: 6 hours
**Dependencies**: PMAT-268

**Description**:

Generate playbook model configurations from family YAML size_variants. Auto-populate
hidden_dim, num_layers, num_heads from family contract.

**Acceptance Criteria**:
- [ ] `generate_model_config(family_yaml, size) -> ModelConfig`
- [ ] Playbook can reference family YAML instead of hardcoding
- [ ] Generated config matches family YAML values

**Files**:
- `crates/apr-qa-runner/src/playbook.rs` (extend)

---

### PMAT-270: Size category auto-alignment

**Priority**: 3
**Estimate**: 2 hours
**Dependencies**: PMAT-268

**Description**:

Auto-set playbook `size_category` from family YAML `certification.size_categories`
mapping.

**Acceptance Criteria**:
- [ ] Playbook loader checks family YAML for size mapping
- [ ] Auto-sets `size_category` if not explicitly set
- [ ] FALSIFY-FAM-001 test passes

**Files**:
- `crates/apr-qa-runner/src/playbook.rs` (extend)

---

### PMAT-271: Tensor template validation

**Priority**: 3
**Estimate**: 4 hours
**Dependencies**: PMAT-268

**Description**:

Enhance G0-VALIDATE to check tensors against family YAML tensor_templates.

**Acceptance Criteria**:
- [ ] G0-VALIDATE loads family YAML
- [ ] Checks all required tensors from tensor_templates
- [ ] Reports missing tensors with family context
- [ ] FALSIFY-FAM-002 test passes

**Files**:
- `crates/apr-qa-runner/src/executor.rs` (extend G0-VALIDATE)

---

### PMAT-272: Falsification test suite

**Priority**: 4
**Estimate**: 6 hours
**Dependencies**: PMAT-267, PMAT-271

**Description**:

Implement all FALSIFY-* tests from §7. Add to CI pipeline.

**Acceptance Criteria**:
- [ ] All FALSIFY-CERT-* tests implemented
- [ ] All FALSIFY-ORC-INT-* tests implemented
- [ ] All FALSIFY-FAM-* tests implemented
- [ ] All FALSIFY-EXP-* tests implemented
- [ ] CI runs falsification tests on every PR

**Files**:
- `tests/falsification/` (new directory with test files)
- `.github/workflows/ci.yml` (extend)

---

### PMAT-273: Schema validation in CI

**Priority**: 4
**Estimate**: 3 hours
**Dependencies**: PMAT-261, PMAT-265

**Description**:

Add CI job that validates all evidence JSON files against schema and models.csv
against column constraints.

**Acceptance Criteria**:
- [ ] CI validates evidence JSON schema compliance
- [ ] CI validates models.csv format and constraints
- [ ] Failures block merge
- [ ] Clear error messages for violations

**Files**:
- `.github/workflows/schema-validation.yml` (new)
- `scripts/validate-schemas.sh` (new)

---

### PMAT-274: Cross-repo consistency checks

**Priority**: 4
**Estimate**: 4 hours
**Dependencies**: PMAT-270, PMAT-271

**Description**:

CI job that validates apr-model-qa-playbook data is consistent with aprender family
YAMLs. Checks size category alignment, tensor templates, naming conventions.

**Acceptance Criteria**:
- [ ] CI clones aprender (or uses cached version)
- [ ] Validates size_category matches family YAML
- [ ] Validates playbook paths match family YAML templates
- [ ] Reports misalignments with actionable messages

**Files**:
- `.github/workflows/cross-repo-validation.yml` (new)
- `scripts/validate-aprender-alignment.sh` (new)

---

### PMAT-275: Documentation and examples

**Priority**: 4
**Estimate**: 4 hours
**Dependencies**: PMAT-272, PMAT-274

**Description**:

Update CLAUDE.md with oracle integration documentation. Add example workflows for
certification pipeline. Create troubleshooting guide.

**Acceptance Criteria**:
- [ ] CLAUDE.md updated with oracle integration section
- [ ] Example: "Certify a new model" workflow
- [ ] Example: "Update certification after fixes" workflow
- [ ] Troubleshooting: common integration errors
- [ ] Links to aprender oracle documentation

**Files**:
- `CLAUDE.md` (extend)
- `docs/workflows/certification.md` (new)
- `docs/troubleshooting/oracle-integration.md` (new)

---

## Appendix B: PMAT Compliance Checklist

All tickets MUST meet PMAT quality gates:

- [ ] **Coverage**: ≥95% line coverage for new code
- [ ] **Mutation**: ≥80% mutation score for critical paths
- [ ] **Clippy**: Zero warnings with pedantic + nursery lints
- [ ] **Documentation**: All public APIs documented
- [ ] **Tests**: Unit tests for all functions, integration tests for commands
- [ ] **Falsification**: At least one FALSIFY-* test per major component
