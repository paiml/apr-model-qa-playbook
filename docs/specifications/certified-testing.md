# Certified Testing & Popperian Falsification

**Version:** 2.0.0
**Status:** APPROVED
**Author:** PAIML Engineering
**Date:** 2026-01-31
**Philosophy:** Karl Popper (Critical Rationalism) + Nassim Taleb (Black Swan Theory)

---

## 1. Abstract: The Definition of "Certified"

In the context of the APR Model QA Playbook, **"Certified"** does not mean "verified to be bug-free" (an impossibility under the Problem of Induction). Instead, it adopts a strict **Popperian definition**:

> **Certified Status** is assigned to a model version if and only if it has **survived** a specific, finite, and rigorous set of attempted refutations (the Verification Matrix) without a single falsification.

We do not "certify" that the model works; we certify that we tried very hard to break it, in specific ways, and failed.

### 1.1 The Demarcation Problem
We solve the *demarcation problem* (distinguishing scientific testing from "happy path" testing) by requiring every test case to be a **Potential Falsifier**. A test is only valid if it describes an observation that, if made, would logically entail the falsehood of the model's quality claims.

## 2. The Popperian Protocol

Every "Certified" test suite must adhere to the following protocol:

1.  **Hypothesis Formulation ($H$):** Formulate a universal statement (e.g., "For all inputs $x$, output $y$ is valid JSON").
2.  **Severe Testing:** Design an experiment $E$ specifically intended to find an instance where $H$ is false (e.g., "Inject malformed UTF-8 and garbage tokens").
3.  **Binary Outcome:**
    *   **FALSIFIED:** The system produces an invalid state. The hypothesis is false. The model is **REJECTED**.
    *   **CORROBORATED:** The system survives the attack. The hypothesis is tentatively accepted. The model gains **VERISIMILITUDE**.

### 2.1 Black Swans and Anti-Fragility
Standard testing optimizes for the "White Swan" (expected behavior). Certified Testing optimizes for the "Black Swan" (rare, catastrophic events).
*   **Fragile:** Breaks under stress (Standard QA).
*   **Robust:** Resists stress (Certified QA).
*   **Antifragile:** Improves with stress (Future Goal).

## 3. Certification Grades & Tiers

### 3.1 Two-Tier Certification Model

| Tier | Time Limit | Grade on Pass | Status |
|------|------------|---------------|--------|
| **MVP** | ≤10 min | **B** | PROVISIONAL |
| **Full** | ≤1 hour | **A+** | CERTIFIED |

### 3.2 MVP Tier (80% of Models)

The MVP tier answers: *"Does this model work in all supported configurations?"*

**Requirements:**
- Tests all 18 combinations: 3 formats × 2 backends × 3 modalities
- Must pass all 5 gateways (G0-G4)
- Must achieve ≥90% pass rate
- Must complete in ≤10 minutes

**Outcome:**
- **PASS** → Grade **B**, Status **PROVISIONAL** (score 800)
- **FAIL** → Grade **F**, Status **BLOCKED** (score based on actual results)

### 3.3 Full Tier (Production Certification)

The Full tier answers: *"Has this model survived rigorous falsification attempts?"*

**Requirements:**
- Complete 170-point Verification Matrix
- All P0 gates must pass
- Must achieve ≥95% on matrix
- Must complete in ≤1 hour

**Outcome:**
- **PASS** → Grade **A+**, Status **CERTIFIED** (score 950+)
- **PARTIAL** → Grade based on score (A/B+/B/C)
- **FAIL** → Grade **F**, Status **BLOCKED**

## 4. The Verification Matrix (170 Points)

### 4.1 Class I: Fundamental Integrity (P0 - CRITICAL)
*Any failure here results in immediate disqualification.*

| Gate ID | Hypothesis | Falsification Criteria | Points |
| :--- | :--- | :--- | :--- |
| **F-INT-001** | **Memory Safety** | Segmentation fault, buffer overflow, or wild pointer access. | 10 |
| **F-INT-002** | **Process Termination** | Unclean exit (non-zero code without error), zombie process, or hang > timeout. | 10 |
| **F-INT-003** | **Tensor Validity** | Any internal tensor contains NaN or Inf values. | 10 |
| **F-INT-004** | **Format Fidelity** | Round-trip conversion (GGUF→APR→GGUF) alters quantized weights (bitwise mismatch). | 10 |
| **F-INT-005** | **Determinisim** | Fixed seed ($S=42$) produces different output tokens on identical hardware. | 10 |

### 4.2 Class II: Interface Compliance (P1 - HIGH)

| Gate ID | Hypothesis | Falsification Criteria | Points |
| :--- | :--- | :--- | :--- |
| **F-API-001** | **JSON Compliance** | REST API returns malformed JSON or non-conforming schema. | 5 |
| **F-API-002** | **Chat Template** | Output contains raw control tokens (e.g., `<|im_end|>`) or template leakage. | 5 |
| **F-API-003** | **Health Check** | `/health` endpoint returns non-200 or blocks for >1s. | 5 |
| **F-API-004** | **Error Handling** | Invalid input crashes server instead of returning 400 Bad Request. | 5 |
| **F-API-005** | **Streaming** | SSE stream breaks format (missing `data:` prefix) or hangs. | 5 |

### 4.3 Class III: Numerical Stability (P1 - HIGH)

| Gate ID | Hypothesis | Falsification Criteria | Points |
| :--- | :--- | :--- | :--- |
| **F-NUM-001** | **Attention Entropy** | Attention distribution collapses (entropy $\approx 0$) or explodes (uniform). | 5 |
| **F-NUM-002** | **LayerNorm Drift** | LayerNorm output mean > $10^{-3}$ or std dev diverges from 1.0 by > 5%. | 5 |
| **F-NUM-003** | **Softmax Sum** | Softmax outputs do not sum to $1.0 \pm 10^{-6}$. | 5 |
| **F-NUM-004** | **Token Probability** | Logits result in invalid probabilities (<0 or >1). | 5 |

### 4.4 Class IV: Cross-Platform Parity (P2 - MEDIUM)

| Gate ID | Hypothesis | Falsification Criteria | Points |
| :--- | :--- | :--- | :--- |
| **F-PAR-001** | **CPU/GPU Equivalence** | CPU vs GPU outputs differ by > $\epsilon$ (1e-5) for same FP32 input. | 5 |
| **F-PAR-002** | **Format Parity** | GGUF vs SafeTensors inference output differs logically (different tokens). | 5 |
| **F-PAR-003** | **Quantization Impact** | Q4_K_M Perplexity degrades > 10% vs F16 reference on calibration set. | 5 |

### 4.5 Class V: Performance Boundaries (P2 - MEDIUM)

| Gate ID | Hypothesis | Falsification Criteria | Points |
| :--- | :--- | :--- | :--- |
| **F-PERF-001** | **Minimum TPS** | Inference throughput < 10 tok/s on reference hardware (CPU). | 5 |
| **F-PERF-002** | **TTFT** | Time To First Token > 2000ms (Cold Start). | 5 |
| **F-PERF-003** | **Memory Leak** | RSS memory grows monotonically > 5% over 100 requests. | 5 |
| **F-PERF-004** | **GPU Utilization** | Compute-bound kernels achieve < 50% theoretical occupancy. | 5 |

### 4.6 Class VI: Security & Safety (P0 - CRITICAL)

| Gate ID | Hypothesis | Falsification Criteria | Points |
| :--- | :--- | :--- | :--- |
| **F-SEC-001** | **Path Traversal** | Model path allows access to parent directories (`../`). | 10 |
| **F-SEC-002** | **Prompt Injection** | System prompt is overridden by user input containing control sequences. | 10 |
| **F-SEC-003** | **Denial of Service** | "Zip bomb" or "Token flood" input causes OOM or infinite loop. | 10 |

## 5. Certification Artifacts

A **Certified** build must produce the following immutable artifacts:

1.  **`evidence.json`**: The complete, raw log of every probe, timestamped and hashed.
2.  **`popperian_report.html`**: A human-readable dashboard highlighting the "Survivor" tests and any falsifications.
3.  **`CERTIFICATE.md`**: A generated file in the release package stating:
    *   **Version:** X.Y.Z
    *   **Score:** N/170
    *   **Status:** CERTIFIED / PROVISIONAL / REJECTED
    *   **Black Swans Caught:** (Count of regression tests added)

> **Note:** Validation of this specification's implementation is governed by the [100-Point QA Checklist](./certified-testing-qa-checklist.md).

## 6. Certification CLI

The `apr-qa certify` command executes the Popperian certification protocol:

```bash
# Certify all models in a family with MVP tier (recommended)
apr-qa certify --family qwen-coder --tier mvp

# Certify specific models for production release
apr-qa certify Qwen/Qwen2.5-Coder-1.5B-Instruct --tier full

# Dry run to preview certification plan
apr-qa certify --family qwen-coder --dry-run

# Certify all registered models
apr-qa certify --all --tier mvp
```

### 6.1 Certification Tiers (Two-Tier Model)

The certification system uses a **two-tier model** aligned with section 3:

| Tier | Time Limit | Grade on Pass | Status | Use Case |
|------|------------|---------------|--------|----------|
| **MVP** | ≤10 min | **B** | PROVISIONAL | 80% of models - surface coverage |
| **Full** | ≤1 hour | **A+** | CERTIFIED | Production qualification |

#### 6.1.1 MVP Tier (Minimum Viable Product)

The **MVP tier** answers: *"Does this model work in all supported configurations?"*

```
Formats:    GGUF, APR, SafeTensors     (3)
Backends:   CPU, GPU                    (2)
Modalities: run, chat, serve            (3)
                                        ───
Total:      18 test combinations
```

**Pass Criteria:**
- ≥90% pass rate across all 18 combinations
- All P0 gateways (G0-G4) must pass

**On Pass:** MQS Score = 800, Grade = **B**, Status = **PROVISIONAL**

```bash
# Run MVP certification (recommended for most models)
apr-qa certify --family qwen-coder --tier mvp

# MVP is ideal for:
# - Pre-release sanity check
# - New format/backend support verification
# - Quick regression after infrastructure changes
```

#### 6.1.2 Full Tier (Production Qualification)

The **Full tier** answers: *"Has this model survived rigorous falsification attempts?"*

**Requirements:**
- Complete 170-point Verification Matrix
- All P0 gates must pass
- ≥95% pass rate on matrix

**On Pass:** MQS Score = 950+, Grade = **A+**, Status = **CERTIFIED**

```bash
# Run Full certification (for production release)
apr-qa certify --family qwen-coder --tier full
```

#### 6.1.3 Tier Comparison

| Aspect | MVP | Full |
|--------|-----|------|
| **Time Limit** | ≤10 min | ≤1 hour |
| **Pass Threshold** | 90% | 95% |
| **Grade on Pass** | B | A+ |
| **Status on Pass** | PROVISIONAL | CERTIFIED |
| Formats | 3 | 3 |
| Backends | 2 | 2 |
| Modalities | 3 | 3 |
| Scenarios/combo | 1 | 50+ |
| Verification Matrix | No | Yes (170 pts) |
| Conversion tests | No | Yes |
| Differential tests | No | Yes |
| **Total tests** | 18 | ~900+ |

### 6.2 Certification Results

Results are written to:
- `docs/certifications/models.csv`: Central certification database
- `certifications/<model>/evidence.json`: Raw test evidence
- README.md certification table (via `apr-qa-readme-sync`)

## 7. Performance Profiling Matrix

### 7.1 Six-Column Throughput Requirement

Every certified model **MUST** report throughput (tok/s) across all 6 format×backend combinations:

| Column | Format | Backend | Required |
|--------|--------|---------|----------|
| `tps_gguf_cpu` | GGUF | CPU | **YES** |
| `tps_gguf_gpu` | GGUF | GPU | **YES** |
| `tps_apr_cpu` | APR | CPU | **YES** |
| `tps_apr_gpu` | APR | GPU | **YES** |
| `tps_st_cpu` | SafeTensors | CPU | **YES** |
| `tps_st_gpu` | SafeTensors | GPU | **YES** |

### 7.2 Profiling Protocol

For each format×backend combination:

1. **Model File Resolution:**
   - GGUF: `<cache>/<model>/gguf/model.gguf`
   - APR: `<cache>/<model>/apr/model.apr`
   - SafeTensors: `<cache>/<model>/safetensors/model.safetensors`

2. **Profiling Command:**
   ```bash
   apr profile <model-file> --backend <cpu|gpu> --format json --warmup 1 --measure 3
   ```

3. **Metric Extraction:**
   - Parse JSON output for `metrics.throughput_tok_s`
   - Record value in corresponding column
   - Record `-` if format/backend unavailable

### 7.3 Backend Selection

The `apr profile` command must support explicit backend selection:

```bash
# CPU profiling (default)
apr profile model.gguf --backend cpu --format json

# GPU profiling (requires CUDA/Metal)
apr profile model.gguf --backend gpu --format json
```

If `--backend gpu` is unavailable (no GPU hardware), the GPU columns record `-`.

### 7.4 Ground Truth Policy

**SafeTensors is the canonical source of truth.** All format comparisons derive from the original HuggingFace SafeTensors model.

#### 7.4.1 Unquantized Testing (Precision Parity)

For testing format handling at the same precision:

```bash
# Source: HuggingFace SafeTensors (BF16/FP16)
apr pull Qwen/Qwen2.5-Coder-0.5B-Instruct

# Derive GGUF from SafeTensors (same precision)
apr convert model.safetensors --to gguf -o model.gguf

# Derive APR from SafeTensors (same precision)
apr import model.safetensors -o model.apr
```

**Never use third-party GGUF files** (e.g., bartowski, TheBloke) for qualification. These have unknown provenance and different quantization.

#### 7.4.2 Quantized Testing (Quantization Parity)

For testing quantization impact, apr-cli performs the quantization:

```bash
# Quantize SafeTensors → APR Q4_K_M
apr import model.safetensors -o model_q4km.apr --quantize q4_k_m

# Quantize SafeTensors → GGUF Q4_K_M
apr convert model.safetensors --to gguf -o model_q4km.gguf --quantize q4_k_m

# Compare: APR Q4_K_M vs GGUF Q4_K_M (same quantization, same source)
```

**Rule:** Quantized-to-quantized comparisons must use the same:
1. Source model (SafeTensors)
2. Quantization method (apr-cli)
3. Quantization level (e.g., Q4_K_M)

#### 7.4.3 Format Conversion Matrix

| Source | Target | Command | Use Case |
|--------|--------|---------|----------|
| SafeTensors | GGUF (FP16) | `apr convert --to gguf` | Precision parity |
| SafeTensors | APR (FP16) | `apr import` | Precision parity |
| SafeTensors | GGUF (Q4_K_M) | `apr convert --to gguf --quantize q4_k_m` | Quantization parity |
| SafeTensors | APR (Q4_K_M) | `apr import --quantize q4_k_m` | Quantization parity |

**Prohibited:**
- Using pre-quantized GGUF from third parties
- Comparing different quantization levels
- Comparing different source models

### 7.5 Provenance Validation (PMAT-PROV-001)

**All derived formats MUST include provenance metadata** linking back to the SafeTensors source. This prevents the critical error of comparing models from different sources.

#### 7.5.1 Provenance File Format

Each model directory contains a `.provenance.json` file:

```json
{
  "source": {
    "format": "safetensors",
    "path": "model.safetensors",
    "sha256": "a1b2c3d4e5f6...",
    "hf_repo": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    "downloaded_at": "2026-02-01T12:00:00Z"
  },
  "derived": [
    {
      "format": "gguf",
      "path": "model.gguf",
      "sha256": "f6e5d4c3b2a1...",
      "converter": "apr-cli",
      "converter_version": "0.2.12",
      "quantization": null,
      "created_at": "2026-02-01T12:05:00Z"
    },
    {
      "format": "apr",
      "path": "model.apr",
      "sha256": "1a2b3c4d5e6f...",
      "converter": "apr-cli",
      "converter_version": "0.2.12",
      "quantization": null,
      "created_at": "2026-02-01T12:06:00Z"
    }
  ]
}
```

#### 7.5.2 Validation Rules

| Rule ID | Description | Failure Action |
|---------|-------------|----------------|
| **PROV-001** | All formats in certification must share same `source.sha256` | REJECT certification |
| **PROV-002** | Derived files must have `converter: "apr-cli"` | REJECT file |
| **PROV-003** | Source must be `format: "safetensors"` | REJECT certification |
| **PROV-004** | Third-party files (no provenance) are prohibited | REJECT file |
| **PROV-005** | Quantization levels must match for comparisons | REJECT comparison |
| **PROV-006** | File hash must match recorded hash (integrity) | REJECT file |
| **PROV-007** | Referenced files must exist (no ghost files) | REJECT provenance |
| **PROV-008** | No duplicate format+quantization entries | REJECT entry |
| **PROV-009** | Compared formats must exist in derived list | REJECT comparison |

#### 7.5.3 Test Requirements

The following tests MUST exist to enforce provenance:

```rust
// PMAT-PROV-001: Reject certification with mismatched sources
#[test]
fn test_reject_mismatched_source_hash() {
    // GGUF from bartowski, SafeTensors from HuggingFace
    // Must fail with ProvenanceMismatch error
}

// PMAT-PROV-002: Reject third-party files without provenance
#[test]
fn test_reject_third_party_gguf() {
    // GGUF downloaded directly (no .provenance.json)
    // Must fail with MissingProvenance error
}

// PMAT-PROV-003: Accept only SafeTensors as source
#[test]
fn test_reject_gguf_as_source() {
    // Attempt to use GGUF as ground truth
    // Must fail with InvalidSourceFormat error
}

// PMAT-PROV-004: Reject quantization mismatch
#[test]
fn test_reject_quantization_mismatch() {
    // Compare Q4_K_M APR with FP16 GGUF
    // Must fail with QuantizationMismatch error
}
```

#### 7.5.4 Provenance Integrity Falsification Matrix (P0)

These tests constitute permanent regression gates for the Chain of Custody system.

| Gate ID | Hypothesis | Falsification Criteria | Points |
|---------|------------|------------------------|--------|
| **F-PROV-001** | **Chain of Custody** | Provenance record allows mismatch between recorded hash and actual file content. | 10 |
| **F-PROV-002** | **Referential Integrity** | System proceeds with comparison when referenced source/derived files are missing (ghost files). | 10 |
| **F-PROV-003** | **Uniqueness** | System allows duplicate format/quantization entries in the derived list. | 5 |
| **F-PROV-004** | **Format Existence** | System allows comparison against non-existent formats in derived list. | 5 |

**Verification Workflow (3-Stage Defense):**

```
validate_provenance()     →  verify_files_exist()    →  verify_provenance_integrity()
     (Logic Check)              (Availability Check)        (Integrity Check)
     PROV-002, PROV-003         PROV-007                     PROV-006, PROV-007
```

**Known Limitation (F-PROV-LOGIC-003):**

The Quantization Lie attack (claiming Q8_0 file is Q4_K_M) is not detected by hash verification alone. Future mitigation: Header Sample check reading first 4KB of GGUF/SafeTensors to verify internal metadata matches provenance record.

#### 7.5.5 Provenance Generation

The `apr-qa` tool automatically generates provenance during conversion:

```bash
# This creates .provenance.json automatically
apr-qa prepare-model Qwen/Qwen2.5-Coder-0.5B-Instruct

# Verify provenance before certification
apr-qa verify-provenance ~/.cache/pacha/models/qwen2-5-coder-0-5b-instruct/
```

### 7.6 Certification Database Schema

The `docs/certifications/models.csv` schema:

```csv
model_id,family,parameters,size_category,status,mqs_score,grade,certified_tier,last_certified,g1,g2,g3,g4,tps_gguf_cpu,tps_gguf_gpu,tps_apr_cpu,tps_apr_gpu,tps_st_cpu,tps_st_gpu
```

All 6 throughput columns are **mandatory fields**. Empty values indicate the combination was not tested.

### 7.7 Conversion Caching

To keep MVP certification under 5 minutes, format conversions are cached:

```
<cache>/<model>/
├── safetensors/
│   ├── model.safetensors    # Ground truth (from HuggingFace)
│   └── config.json          # Model config (required for inference)
├── gguf/
│   └── model.gguf           # Derived from SafeTensors
├── apr/
│   └── model.apr            # Derived from SafeTensors
└── .conversion_hash         # SHA256 of source SafeTensors
```

**Ground truth flow:**
1. `apr pull` downloads SafeTensors from HuggingFace
2. `apr convert` derives GGUF from SafeTensors
3. `apr import` derives APR from SafeTensors

**Cache invalidation**: Conversion is skipped if:
1. Target file exists
2. `.conversion_hash` matches current SafeTensors file hash

**First run**: ~4 minutes (includes conversion)
**Subsequent runs**: ~2 minutes (benchmarks only)

### 7.8 Minimum Throughput Gates

From the Verification Matrix (F-PERF-001):

| Backend | Minimum Throughput |
|---------|-------------------|
| CPU | 10 tok/s |
| GPU | 50 tok/s |

Failure to meet minimum throughput results in gate failure (5 points deducted).

## 8. References

*   **Popper, K. R.** (1959). *The Logic of Scientific Discovery*.
*   **Popper, K. R.** (1963). *Conjectures and Refutations*.
*   **Taleb, N. N.** (2007). *The Black Swan*.
*   **Ohno, T.** (1988). *Toyota Production System*.