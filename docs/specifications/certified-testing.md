# Certified Testing & Popperian Falsification

**Version:** 1.1.0
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

## 3. The Verification Matrix (170 Points)

To achieve **Certified** status, a model/system must achieve a score of **95% or higher** on the Verification Matrix.
*   **Certified:** Score ≥ 95% (e.g., ≥ 162/170) AND Zero P0 Failures.
*   **Provisional:** Score ≥ 80% AND Zero P0 Failures.
*   **Rejected:** Score < 80% OR Any P0 Failure.

### 3.1 Class I: Fundamental Integrity (P0 - CRITICAL)
*Any failure here results in immediate disqualification.*

| Gate ID | Hypothesis | Falsification Criteria | Points |
| :--- | :--- | :--- | :--- |
| **F-INT-001** | **Memory Safety** | Segmentation fault, buffer overflow, or wild pointer access. | 10 |
| **F-INT-002** | **Process Termination** | Unclean exit (non-zero code without error), zombie process, or hang > timeout. | 10 |
| **F-INT-003** | **Tensor Validity** | Any internal tensor contains NaN or Inf values. | 10 |
| **F-INT-004** | **Format Fidelity** | Round-trip conversion (GGUF→APR→GGUF) alters quantized weights (bitwise mismatch). | 10 |
| **F-INT-005** | **Determinisim** | Fixed seed ($S=42$) produces different output tokens on identical hardware. | 10 |

### 3.2 Class II: Interface Compliance (P1 - HIGH)

| Gate ID | Hypothesis | Falsification Criteria | Points |
| :--- | :--- | :--- | :--- |
| **F-API-001** | **JSON Compliance** | REST API returns malformed JSON or non-conforming schema. | 5 |
| **F-API-002** | **Chat Template** | Output contains raw control tokens (e.g., `<|im_end|>`) or template leakage. | 5 |
| **F-API-003** | **Health Check** | `/health` endpoint returns non-200 or blocks for >1s. | 5 |
| **F-API-004** | **Error Handling** | Invalid input crashes server instead of returning 400 Bad Request. | 5 |
| **F-API-005** | **Streaming** | SSE stream breaks format (missing `data:` prefix) or hangs. | 5 |

### 3.3 Class III: Numerical Stability (P1 - HIGH)

| Gate ID | Hypothesis | Falsification Criteria | Points |
| :--- | :--- | :--- | :--- |
| **F-NUM-001** | **Attention Entropy** | Attention distribution collapses (entropy $\approx 0$) or explodes (uniform). | 5 |
| **F-NUM-002** | **LayerNorm Drift** | LayerNorm output mean > $10^{-3}$ or std dev diverges from 1.0 by > 5%. | 5 |
| **F-NUM-003** | **Softmax Sum** | Softmax outputs do not sum to $1.0 \pm 10^{-6}$. | 5 |
| **F-NUM-004** | **Token Probability** | Logits result in invalid probabilities (<0 or >1). | 5 |

### 3.4 Class IV: Cross-Platform Parity (P2 - MEDIUM)

| Gate ID | Hypothesis | Falsification Criteria | Points |
| :--- | :--- | :--- | :--- |
| **F-PAR-001** | **CPU/GPU Equivalence** | CPU vs GPU outputs differ by > $\epsilon$ (1e-5) for same FP32 input. | 5 |
| **F-PAR-002** | **Format Parity** | GGUF vs SafeTensors inference output differs logically (different tokens). | 5 |
| **F-PAR-003** | **Quantization Impact** | Q4_K_M Perplexity degrades > 10% vs F16 reference on calibration set. | 5 |

### 3.5 Class V: Performance Boundaries (P2 - MEDIUM)

| Gate ID | Hypothesis | Falsification Criteria | Points |
| :--- | :--- | :--- | :--- |
| **F-PERF-001** | **Minimum TPS** | Inference throughput < 10 tok/s on reference hardware (CPU). | 5 |
| **F-PERF-002** | **TTFT** | Time To First Token > 2000ms (Cold Start). | 5 |
| **F-PERF-003** | **Memory Leak** | RSS memory grows monotonically > 5% over 100 requests. | 5 |
| **F-PERF-004** | **GPU Utilization** | Compute-bound kernels achieve < 50% theoretical occupancy. | 5 |

### 3.6 Class VI: Security & Safety (P0 - CRITICAL)

| Gate ID | Hypothesis | Falsification Criteria | Points |
| :--- | :--- | :--- | :--- |
| **F-SEC-001** | **Path Traversal** | Model path allows access to parent directories (`../`). | 10 |
| **F-SEC-002** | **Prompt Injection** | System prompt is overridden by user input containing control sequences. | 10 |
| **F-SEC-003** | **Denial of Service** | "Zip bomb" or "Token flood" input causes OOM or infinite loop. | 10 |

## 4. Certification Artifacts

A **Certified** build must produce the following immutable artifacts:

1.  **`evidence.json`**: The complete, raw log of every probe, timestamped and hashed.
2.  **`popperian_report.html`**: A human-readable dashboard highlighting the "Survivor" tests and any falsifications.
3.  **`CERTIFICATE.md`**: A generated file in the release package stating:
    *   **Version:** X.Y.Z
    *   **Score:** N/170
    *   **Status:** CERTIFIED / PROVISIONAL / REJECTED
    *   **Black Swans Caught:** (Count of regression tests added)

> **Note:** Validation of this specification's implementation is governed by the [100-Point QA Checklist](./certified-testing-qa-checklist.md).

## 5. Certification CLI

The `apr-qa certify` command executes the Popperian certification protocol:

```bash
# Certify all models in a family (0.5B to 32B)
apr-qa certify --family qwen-coder --tier quick

# Certify specific models
apr-qa certify Qwen/Qwen2.5-Coder-1.5B-Instruct --tier deep

# Dry run to preview certification plan
apr-qa certify --family qwen-coder --dry-run

# Certify all registered models
apr-qa certify --all --tier smoke
```

### 5.1 Certification Tiers

| Tier | Duration | Use Case |
|------|----------|----------|
| **smoke** | ~1s/model | Quick sanity check |
| **quick** | ~30s/model | Development iteration |
| **standard** | ~1m/model | CI/CD pipeline |
| **deep** | ~10m/model | Production qualification |

### 5.2 Certification Results

Results are written to:
- `docs/certifications/models.csv`: Central certification database
- `certifications/<model>/evidence.json`: Raw test evidence
- README.md certification table (via `apr-qa-readme-sync`)

## 6. References

*   **Popper, K. R.** (1959). *The Logic of Scientific Discovery*.
*   **Popper, K. R.** (1963). *Conjectures and Refutations*.
*   **Taleb, N. N.** (2007). *The Black Swan*.
*   **Ohno, T.** (1988). *Toyota Production System*.