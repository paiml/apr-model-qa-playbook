# QA Checklist: Certified Testing Specification Validation

**Target Spec:** `docs/specifications/certified-testing.md` (v1.0.0)
**Auditor:** External QA Team
**Total Points:** 100
**Passing Score:** 100/100 (Strict Compliance)

---

## 1. Meta-Compliance & Documentation (10 Points)
*Verify that the specification itself is properly integrated into the project lifecycle.*

| ID | Check | Audit Instruction | Points | Pass/Fail |
|:---|:---|:---|:---:|:---:|
| **QA-META-01** | **Spec Existence** | Confirm `docs/specifications/certified-testing.md` exists and matches v1.0.0. | 2 | [ ] |
| **QA-META-02** | **Reference Integration** | Check `docs/specifications/apr-playbook-spec.md` Appendix C for a link to the new spec. | 2 | [ ] |
| **QA-META-03** | **Philosophy Citations** | Verify spec references Popper (1959) and Taleb (2007) correctly. | 2 | [ ] |
| **QA-META-04** | **Definitions** | Confirm "Certified," "Corroborated," and "Falsified" are explicitly defined in the abstract. | 2 | [ ] |
| **QA-META-05** | **Versioning** | Verify spec has a valid semantic version and date. | 2 | [ ] |

## 2. Execution Harness Capabilities (20 Points)
*Verify that the test runner (`apr-qa`) has the necessary features to execute the spec.*

| ID | Check | Audit Instruction | Points | Pass/Fail |
|:---|:---|:---|:---:|:---:|
| **QA-EXEC-01** | **Jidoka (Stop-the-line)** | Inject a P0 failure (e.g., `panic!`). Verify runner halts immediately (exit code != 0). | 5 | [ ] |
| **QA-EXEC-02** | **Isolation** | Run two tests in parallel. Verify no state leakage (e.g., shared temp files). | 5 | [ ] |
| **QA-EXEC-03** | **Determinism** | Run the suite twice with seed `42`. Verify `evidence.json` hashes match exactly. | 5 | [ ] |
| **QA-EXEC-04** | **Timeout Enforcement** | Simulate a hang (>60s). Verify runner kills the process and marks **F-INT-002** as FALSIFIED. | 5 | [ ] |

## 3. Gate Implementation Verification (40 Points)
*Verify that the Gates defined in the Spec Verification Matrix exist in the codebase.*
*Method: `grep -r "GATE_ID" src/`*

### Class I: Fundamental Integrity (10 pts)
| ID | Check | Audit Instruction | Points | Pass/Fail |
|:---|:---|:---|:---:|:---:|
| **QA-GATE-01** | **F-INT-001 (Mem)** | Verify code checks for segfault signals / unsafe blocks. | 2 | [ ] |
| **QA-GATE-02** | **F-INT-002 (Term)** | Verify code handles non-zero exit codes. | 2 | [ ] |
| **QA-GATE-03** | **F-INT-003 (Tensor)** | Verify recursive NaN/Inf check on all tensor buffers. | 2 | [ ] |
| **QA-GATE-04** | **F-INT-004 (Fidelity)** | Verify round-trip conversion logic exists. | 2 | [ ] |
| **QA-GATE-05** | **F-INT-005 (Det)** | Verify seed propagation to inference engine. | 2 | [ ] |

### Class II: Interface Compliance (5 pts)
| ID | Check | Audit Instruction | Points | Pass/Fail |
|:---|:---|:---|:---:|:---:|
| **QA-GATE-06** | **F-API-001..005** | Verify logic for JSON validation, Health check, and SSE parsing. | 5 | [ ] |

### Class III: Numerical Stability (5 pts)
| ID | Check | Audit Instruction | Points | Pass/Fail |
|:---|:---|:---|:---:|:---:|
| **QA-GATE-07** | **F-NUM-001..004** | Verify entropy calc, LayerNorm std-dev checks, Softmax sum checks. | 5 | [ ] |

### Class IV: Cross-Platform Parity (5 pts)
| ID | Check | Audit Instruction | Points | Pass/Fail |
|:---|:---|:---|:---:|:---:|
| **QA-GATE-08** | **F-PAR-001..003** | Verify differential testing logic (CPU vs GPU, GGUF vs SafeTensors). | 5 | [ ] |

### Class V: Performance Boundaries (5 pts)
| ID | Check | Audit Instruction | Points | Pass/Fail |
|:---|:---|:---|:---:|:---:|
| **QA-GATE-09** | **F-PERF-001..004** | Verify TPS measurement, TTFT timers, and RSS memory monitoring. | 5 | [ ] |

### Class VI: Security & Safety (10 pts)
| ID | Check | Audit Instruction | Points | Pass/Fail |
|:---|:---|:---|:---:|:---:|
| **QA-GATE-10** | **F-SEC-001 (Path)** | Verify path sanitization logic forbids `../`. | 3 | [ ] |
| **QA-GATE-11** | **F-SEC-002 (Inject)** | Verify prompt sanitizer checks for control tokens. | 3 | [ ] |
| **QA-GATE-12** | **F-SEC-003 (DoS)** | Verify input length limits and expansion safeguards. | 4 | [ ] |

## 4. Artifact Verification (20 Points)
*Verify the system produces the required "Certified" artifacts.*

| ID | Check | Audit Instruction | Points | Pass/Fail |
|:---|:---|:---|:---:|:---:|
| **QA-ART-01** | **evidence.json Schema** | Validate generated `evidence.json` against a JSON Schema requiring `timestamp`, `gate_id`, `outcome`. | 5 | [ ] |
| **QA-ART-02** | **HTML Report** | Open `popperian_report.html` in a headless browser. Verify 0 console errors and "Certified" badge visibility. | 5 | [ ] |
| **QA-ART-03** | **Certificate Gen** | Verify `CERTIFICATE.md` is created with correct Score (N/170) and Status. | 5 | [ ] |
| **QA-ART-04** | **Traceability** | Verify every entry in `evidence.json` links back to a specific Gate ID in the spec. | 5 | [ ] |

## 5. Negative Validation (Red Teaming) (10 Points)
*Prove the falsifier works by trying to fool it.*

| ID | Check | Audit Instruction | Points | Pass/Fail |
|:---|:---|:---|:---:|:---:|
| **QA-NEG-01** | **The "Bad Math" Test** | Modify a model to return `2+2=5`. Verify **F-ORACLE-001** reports FALSIFIED. | 4 | [ ] |
| **QA-NEG-02** | **The "Zip Bomb" Test** | Feed a 1GB input string. Verify **F-SEC-003** triggers (catch OOM/Timeout). | 3 | [ ] |
| **QA-NEG-03** | **The "Silent Fail" Test** | Mock a crash that exits with code 0 but empty output. Verify **F-RUN-001** (Empty Output) catches it. | 3 | [ ] |

---

## Audit Summary
*   **Total Score:** ______ / 100
*   **Auditor Signature:** ____________________
*   **Date:** ____________________

### Outcome
- [ ] **PASS** (100/100) - Specification is fully implemented.
- [ ] **FAIL** (<100) - Remediation required.
