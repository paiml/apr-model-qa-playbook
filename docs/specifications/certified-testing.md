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
- Must pass all 4 gateways (G1-G4)
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
- All P0 gateways (G1-G4) must pass

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

## 7. References

*   **Popper, K. R.** (1959). *The Logic of Scientific Discovery*.
*   **Popper, K. R.** (1963). *Conjectures and Refutations*.
*   **Taleb, N. N.** (2007). *The Black Swan*.
*   **Ohno, T.** (1988). *Toyota Production System*.