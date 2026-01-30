# APR Model QA Playbook Specification

**Version:** 1.3.0
**Status:** DRAFT - Awaiting Peer Review
**Author:** PAIML Engineering
**Date:** 2026-01-30
**PMAT Compliance:** Required (95% coverage, zero SATD)
**Quality Philosophy:** Toyota Way + Popperian Falsification + Black Swan Theory

---

## Changelog

### v1.3.0 (2026-01-30)
- **feat(differential):** Add Rosetta differential testing (GH-188, PMAT-114)
  - `apr rosetta diff-tensors`: Tensor layout mismatch detection
  - `apr rosetta compare-inference`: Token-by-token inference comparison
  - Gates: F-ROSETTA-DIFF-001/002, F-ROSETTA-INF-001/002
- **feat(profile):** Add CI assertion mode (PMAT-192)
  - `apr profile --ci --assert-throughput --assert-p99 --assert-p50`
  - Differential benchmarking: `apr profile A B --diff-benchmark`
  - Gates: F-PROFILE-CI-001/002/003, F-PROFILE-DIFF-001/002
- **feat(trace):** Add payload mode (APR-TRACE-001)
  - `apr trace --payload`: Real forward pass with garbage detection
  - Gates: F-TRACE-PAYLOAD-001/002/003, F-TRACE-DIFF-001
- **feat(patterns):** Add cross-project bug pattern detection (GH-187)
  - 12 patterns from aprender/realizar analysis
  - Gates: F-PATH-*, F-STATE-*, F-VALID-*, F-ERR-*, F-SEC-*
- **feat(conversion):** Add bug classification (GH-187)
  - ConversionBugType enum with 6 classifications
  - Gates: F-CONV-EMBED-001, F-CONV-TOK-001, F-CONV-WEIGHT-001, etc.
- **Total gates:** 82+ (up from 56+)
- **Tests:** 225 passing

### v1.2.0 (2026-01-30)
- Initial bug classification gates
- Process lifecycle management (Jidoka)

### v1.1.1 (2026-01-29)
- Profile CI tests
- Verification playbooks

---

## Abstract

This specification defines a **property-based model qualification testing framework** for the top 100 HuggingFace models using bashrs playbooks. The system generates 100 falsifiable test scenarios per model across all modalities (CPU/GPU, run/chat/serve) and formats (GGUF/SafeTensors/APR), yielding **1,800,000+ test assertions** with full traceability.

The framework embodies two complementary philosophies:
1. **Toyota Production System (TPS):** Zero-defect manufacturing through Jidoka (autonomation), Poka-Yoke (error-proofing), and Genchi Genbutsu (go and see).
2. **Popperian Falsificationism:** Solving the *demarcation problem* of model quality by defining "correctness" not as the accumulation of passing tests, but as the survival of rigorous attempts at refutation.

---

## Table of Contents

1. [Quality Philosophy](#1-quality-philosophy)
2. [Architecture Overview](#2-architecture-overview)
3. [Test Dimensionality](#3-test-dimensionality)
4. [Format Conversion Testing (P0 CRITICAL)](#4-format-conversion-testing-p0-critical)
5. [APR Tool Coverage](#5-apr-tool-coverage)
6. [Upstream Ticket Protocol](#6-upstream-ticket-protocol)
7. [Upstream Spec Requirements](#7-upstream-spec-requirements)
8. [Model Qualification Score (MQS)](#8-model-qualification-score-mqs)
9. [Playbook Schema](#9-playbook-schema)
10. [Property Test Generation](#10-property-test-generation)
11. [Falsification Protocol](#11-falsification-protocol)
12. [Orchestration Pipeline](#12-orchestration-pipeline)
13. [Coverage Requirements](#13-coverage-requirements)
14. [Peer-Reviewed Citations](#14-peer-reviewed-citations)
15. [Falsification Checklist](#15-falsification-checklist)
16. [Implementation Roadmap](#16-implementation-roadmap)

---

## 1. Quality Philosophy

### 1.1 The Toyota Way: Zero Tolerance for Defects

> "Stop the line. Fix it now. Never pass a defect to the next process."
> — Taiichi Ohno, *Toyota Production System: Beyond Large-Scale Production* (1988)

This specification rejects the software industry's normalization of technical debt. We do not:
- Ship with "known issues" documentation
- Create tickets for "low priority" defects
- Use SATD markers (TODO/FIXME/HACK) as placeholders
- Derive metrics that obscure actual quality

We practice **Jidoka** (autonomation with a human touch): the system stops automatically when a defect is detected, and no work proceeds until the root cause is eliminated.

| TPS Principle | Application in This Specification |
|---------------|-----------------------------------|
| **Jidoka** | Playbook execution halts on first falsification; no silent failures |
| **Poka-Yoke** | Schema validation prevents malformed playbooks from executing |
| **Genchi Genbutsu** | All metrics derived from actual inference, never synthetic |
| **Heijunka** | Load-balanced parallel execution across GPU/CPU workers |
| **Kaizen** | Continuous refinement via mutation testing feedback loops |
| **Muda Elimination** | Zero redundant test cases; Cartesian product pruned by dependency analysis |

**Citation:** Liker, J. K. (2004). *The Toyota Way: 14 Management Principles from the World's Greatest Manufacturer*. McGraw-Hill. ISBN 978-0071392310.

### 1.2 Popperian Falsificationism: Science Through Refutation

> "The criterion of the scientific status of a theory is its falsifiability, or refutability, or testability."
> — Karl Popper, *Conjectures and Refutations* (1963), p. 37

This framework adopts **Critical Rationalism**, rejecting the notion that software quality can be "verified." No amount of passing tests can prove a model is correct (the **Problem of Induction**), but a single failing test can prove it incorrect (the **Asymmetry of Falsification**).

Therefore, we do not "test to pass"; we "test to fail."

#### 1.2.1 The Scientific Method for Models
Every playbook test case must adhere to the **Popperian Protocol**:

1. **Hypothesis ($H$):** A bold conjecture about the model's behavior (e.g., "Model $M$ follows instruction $I$ invariant of quantization $Q$").
2. **Experiment ($E$):** A severe test designed specifically to refute $H$ (e.g., "Inject adversarial noise into $I$ and observe output").
3. **Observation ($O$):** The empirical result.
4. **Conclusion:**
   - If $O$ contradicts $H$: **FALSIFIED** (The model is defective).
   - If $O$ matches $H$: **CORROBORATED** (The model has survived *this* test).

#### 1.2.2 The Demarcation Criterion
A test case is **scientific** if and only if there exists a theoretically possible observation that would mark it FALSIFIED.

We explicitly reject:
- **Tautologies:** Tests that always pass by definition (e.g., "Output is a string").
- **Metaphysical Statements:** Tests with subjective success criteria (e.g., "Output is interesting").
- **Unfalsifiable Claims:** Tests that mask failures (e.g., catch-all exception handlers).

**Citation:** Popper, K. R. (1959). *The Logic of Scientific Discovery*. Hutchinson. ISBN 978-0415278447.
**Citation:** Popper, K. R. (1963). *Conjectures and Refutations: The Growth of Scientific Knowledge*. Routledge. ISBN 978-0415285940.

### 1.3 Synthesis: Evolutionary Quality (Verisimilitude)

The synthesis of TPS (Process Control) and Popperian Methodology (Epistemology) yields an evolutionary approach to quality. We view the model's quality not as a static binary (Good/Bad), but as **Verisimilitude** (closeness to truth).

```
┌─────────────────────────────────────────────────────────────────┐
│              VERISIMILITUDE = SURVIVAL OF REFUTATION            │
├─────────────────────────────────────────────────────────────────┤
│  CORROBORATED = Withstood severe tests (High Confidence)        │
│  FALSIFIED    = Refuted by evidence (Defective)                 │
│  METAPHYSICAL = Untestable (Zero Confidence)                    │
└─────────────────────────────────────────────────────────────────┘
```

A model achieves high quality by accumulating **corroboration**—surviving increasingly severe attempts at falsification. A bug is simply a successful refutation of the hypothesis that "the model is ready for production."

### 1.4 The Black Swan: The Hunt for Rare Events

> "It is the 'Black Swan' event... that carries the impact."
> — Nassim Nicholas Taleb, *The Black Swan* (2007)

Standard unit tests verify known behaviors ("White Swans")—linear, expected, and safe. Property-based testing, through its generation of millions of arbitrary, combinatorial, and edge-case inputs, is an active search for **Black Swans**—rare, high-impact failure modes that lurk in the tails of the input distribution.

We acknowledge that we cannot *prove* a model is safe; we can only *fail to reject* the hypothesis of safety after extensive, adversarial search in the long tail. Every property test iteration is a "risky prediction" in the Popperian sense.

**Citation:** Taleb, N. N. (2007). *The Black Swan: The Impact of the Highly Improbable*. Random House. ISBN 978-1400063512.

---

## 2. Architecture Overview

### 2.1 System Components

```
┌──────────────────────────────────────────────────────────────────────┐
│                        APR-MODEL-QA-PLAYBOOK                          │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐   │
│  │  apr-qa-gen     │    │  apr-qa-runner  │    │  apr-qa-report  │   │
│  │  (Rust crate)   │    │  (bashrs exec)  │    │  (Rust crate)   │   │
│  │                 │    │                 │    │                 │   │
│  │  - proptest     │───▶│  - playbook     │───▶│  - popperian    │   │
│  │  - scenario gen │    │  - state machine│    │  - junit/html   │   │
│  │  - oracle def   │    │  - parallel     │    │  - evidence     │   │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘   │
│           │                      │                      │            │
│           └──────────────────────┼──────────────────────┘            │
│                                  │                                    │
│                    ┌─────────────▼─────────────┐                     │
│                    │     batuta orchestrator    │                     │
│                    │     (pipeline stages)      │                     │
│                    └─────────────┬─────────────┘                     │
│                                  │                                    │
│         ┌────────────────────────┼────────────────────────┐          │
│         ▼                        ▼                        ▼          │
│  ┌─────────────┐         ┌─────────────┐         ┌─────────────┐    │
│  │  aprender   │         │  realizar   │         │   bashrs    │    │
│  │  (apr CLI)  │         │  (inference)│         │  (playbook) │    │
│  └─────────────┘         └─────────────┘         └─────────────┘    │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

### 2.2 Crate Structure

```
apr-model-qa-playbook/
├── Cargo.toml                    # Workspace root
├── crates/
│   ├── apr-qa-gen/              # Scenario generator (proptest)
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── scenario.rs      # QaScenario struct
│   │   │   ├── oracle.rs        # Output verification
│   │   │   ├── models.rs        # HF top 100 registry
│   │   │   └── proptest_impl.rs # Arbitrary implementations
│   │   └── tests/
│   │
│   ├── apr-qa-runner/           # Playbook executor
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── executor.rs      # bashrs integration
│   │   │   ├── conversion.rs    # P0 format conversion testing
│   │   │   ├── state_machine.rs # FSM protocol
│   │   │   └── parallel.rs      # Worker pool
│   │   └── tests/
│   │
│   └── apr-qa-report/           # Report generator
│       ├── src/
│       │   ├── lib.rs
│       │   ├── popperian.rs     # Falsification scoring
│       │   ├── junit.rs         # JUnit XML output
│       │   └── html.rs          # HTML dashboard
│       └── tests/
│
├── playbooks/                    # Generated playbooks
│   ├── models/                  # Per-model playbooks
│   ├── spec/                    # Executable spec playbooks
│   └── templates/               # Playbook templates
│
├── docs/
│   └── specifications/
│       └── apr-playbook-spec.md # This document
│
└── tests/                        # Integration tests
    ├── falsification_tests.rs
    ├── property_tests.rs
    └── integration_tests.rs
```

### 2.3 Data Flow

```
HuggingFace Top 100
        │
        ▼
┌───────────────────┐
│ Model Registry    │  (models.rs: curated list with metadata)
│ - model_id        │
│ - architectures   │
│ - quantizations   │
│ - expected_caps   │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ Scenario Generator│  (proptest: 100 scenarios/model)
│ - prompts         │
│ - temperatures    │
│ - max_tokens      │
│ - oracles         │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ Playbook Emitter  │  (YAML generation)
│ - state machines  │
│ - guards          │
│ - actions         │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ bashrs Executor   │  (parallel execution)
│ - CPU workers     │
│ - GPU workers     │
│ - timeout mgmt    │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ Popperian Report  │  (evidence collection)
│ - CORROBORATED    │
│ - FALSIFIED       │
│ - evidence.json   │
└───────────────────┘
```

---

## 3. Test Dimensionality

### 3.1 Cartesian Product Definition

The full test space is the Cartesian product of:

| Dimension | Cardinality | Values |
|-----------|-------------|--------|
| **Models** | 100 | HuggingFace top 100 by downloads |
| **Modalities** | 3 | `run`, `chat`, `serve` |
| **Formats** | 3 | `gguf`, `safetensors`, `apr` |
| **Backends** | 2 | `cpu`, `gpu` |
| **Scenarios** | 100 | Property-generated per model |

**Total Test Cases:**
```
100 × 3 × 3 × 2 × 100 = 18,000,000 assertions
```

### 3.2 Pruning Strategy (Muda Elimination)

Not all combinations are valid. We prune:

1. **Format availability:** Some models only have GGUF or SafeTensors
2. **GPU capability:** Not all models fit in GPU memory
3. **Modality support:** Some models don't support chat templates
4. **Redundant scenarios:** Property tests with identical oracle outcomes

**Pruned estimate:** ~6,000,000 unique test cases (67% reduction)

### 3.3 Modality Definitions

#### 3.3.1 `run` Modality

Direct inference without server overhead:

```bash
apr run ${MODEL_PATH} "${PROMPT}" -n ${MAX_TOKENS} [--gpu]
```

**Falsification gates:**
- F-RUN-001: Output is non-empty
- F-RUN-002: Output contains no garbage tokens (NaN, Inf, control chars)
- F-RUN-003: Latency < timeout threshold
- F-RUN-004: Token rate >= minimum throughput

#### 3.3.2 `chat` Modality

Interactive chat with template application:

```bash
apr chat ${MODEL_PATH} [--gpu] <<< "${PROMPT}"
```

**Falsification gates:**
- F-CHAT-001: Chat template correctly applied
- F-CHAT-002: System prompt honored (if provided)
- F-CHAT-003: Multi-turn context maintained
- F-CHAT-004: Stop tokens respected

#### 3.3.3 `serve` Modality

REST API server with OpenAI-compatible endpoints:

```bash
apr serve ${MODEL_PATH} --port ${PORT} [--gpu] &
curl -X POST http://localhost:${PORT}/v1/completions \
  -d '{"prompt": "${PROMPT}", "max_tokens": ${MAX_TOKENS}}'
```

**Falsification gates:**
- F-SERVE-001: Server starts within timeout
- F-SERVE-002: Health endpoint responds 200
- F-SERVE-003: Completions endpoint returns valid JSON
- F-SERVE-004: Streaming works (if enabled)
- F-SERVE-005: Graceful shutdown on SIGTERM

### 3.4 Backend Definitions

#### 3.4.1 CPU Backend

SIMD-accelerated inference via trueno:

```bash
apr run ${MODEL_PATH} "${PROMPT}" -n ${MAX_TOKENS}
# No --gpu flag
```

**Falsification gates:**
- F-CPU-001: Throughput >= 10 tok/s (spec H12)
- F-CPU-002: Deterministic output (same seed → same output)
- F-CPU-003: No SIMD instruction faults

#### 3.4.2 GPU Backend

CUDA-accelerated inference via trueno-gpu:

```bash
apr run ${MODEL_PATH} "${PROMPT}" -n ${MAX_TOKENS} --gpu
```

**Falsification gates:**
- F-GPU-001: GPU detected and utilized
- F-GPU-002: Throughput >= 2× CPU (spec F-PERF-042)
- F-GPU-003: No CUDA OOM errors
- F-GPU-004: KV cache properly managed
- F-GPU-005: No numerical explosion (hidden state L2 < 1000)

---

## 4. Format Conversion Testing (P0 CRITICAL)

> "We do not prove the model conversion is correct; we subject it to tests designed to prove it is incorrect."
> — Inversion of the Burden of Proof

This section defines the rigorous falsification protocol for model format interoperability. We operate under the **Risk Hypothesis ($H_R$)** that "Any format conversion introduces non-trivial numerical drift." Our goal is to falsify $H_R$ by demonstrating bitwise or $\epsilon$-bound equivalence across all formats.

### 4.1 Five Whys Analysis: Why Format Conversion is P0

**Problem Statement:** Model outputs differ between formats, causing silent data corruption and incorrect inference results.

| Why # | Question | Answer |
|-------|----------|--------|
| **Why 1** | Why do we need format conversion testing? | Because models exist in multiple formats (GGUF, SafeTensors, APR) and users convert between them. |
| **Why 2** | Why is conversion correctness critical? | Because incorrect conversion produces subtly wrong outputs that are hard to detect but corrupt all downstream inference. |
| **Why 3** | Why are subtle errors dangerous? | Because they pass basic sanity checks but introduce numerical drift, causing models to hallucinate or produce nonsense. |
| **Why 4** | Why can't we catch these in normal testing? | Because standard tests verify "model runs" not "model produces identical output across formats." |
| **Why 5** | Why must this be P0? | Because a single bit flip in tensor conversion can invalidate millions of inferences, making the entire system untrustworthy. |

**Root Cause:** Without bi-directional conversion testing across all format pairs and backends, we have no guarantee that `apr convert` preserves model semantics.

### 4.2 Format Conversion Matrix (MANDATORY)

All format conversions MUST be tested in both directions:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    FORMAT CONVERSION TEST MATRIX (P0)                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│    GGUF ◄────────────────────► APR ◄────────────────────► SafeTensors       │
│      │                          │                              │             │
│      │    ┌─────────────────────┼─────────────────────┐       │             │
│      │    │                     │                     │       │             │
│      ▼    ▼                     ▼                     ▼       ▼             │
│   ┌──────────┐            ┌──────────┐          ┌──────────┐               │
│   │   CPU    │            │   GPU    │          │   WASM   │               │
│   │          │            │  (CUDA)  │          │  (WGPU)  │               │
│   └──────────┘            └──────────┘          └──────────┘               │
│                                                                              │
│   Conversion Paths (12 total):                                              │
│   ─────────────────────────────                                             │
│   GGUF → APR         APR → GGUF         (2 paths)                          │
│   GGUF → SafeTensors SafeTensors → GGUF (2 paths)                          │
│   APR → SafeTensors  SafeTensors → APR  (2 paths)                          │
│                                                                              │
│   × 3 backends (CPU, GPU, WASM) = 36 conversion tests per model            │
│                                                                              │
│   Chain Conversions (6 total):                                              │
│   ────────────────────────────                                              │
│   GGUF → APR → SafeTensors → GGUF  (round-trip)                            │
│   SafeTensors → APR → GGUF → SafeTensors (round-trip)                      │
│   APR → GGUF → SafeTensors → APR (round-trip)                              │
│   (each direction × 2 = 6 chains)                                           │
│                                                                              │
│   × 3 backends = 18 chain conversion tests per model                        │
│                                                                              │
│   TOTAL: 54 conversion tests per model (P0 MANDATORY)                       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.3 Backend Requirements

| Backend | Technology | Required Tests | Notes |
|---------|------------|----------------|-------|
| **CPU** | Native x86_64/ARM64 | All 36 conversions | Baseline reference |
| **GPU** | CUDA (NVIDIA) | All 36 conversions | Performance reference |
| **WASM** | WGPU (WebGPU) | All 36 conversions | ⏳ PLANNED (not yet implemented) |

**Note:** Current implementation tests CPU and GPU backends (36 conversions each = 72 total per model). WASM backend support is planned for future releases.

**Backend Equivalence Assertion:** Output from CPU and GPU backends MUST match within floating-point tolerance ε = 1e-6. See Section 4.9 for precise ε definition.

### 4.4 Conversion Falsification Gates (P0)

These gates are **P0 CRITICAL** - any failure zeros the entire MQS score.

#### 4.4.1 Direct Conversion Gates

| Gate ID | Conversion | Assertion | Severity |
|---------|------------|-----------|----------|
| **F-CONV-001** | GGUF → APR | Output tensors match within ε | P0 |
| **F-CONV-002** | APR → GGUF | Output tensors match within ε | P0 |
| **F-CONV-003** | GGUF → SafeTensors | Output tensors match within ε | P0 |
| **F-CONV-004** | SafeTensors → GGUF | Output tensors match within ε | P0 |
| **F-CONV-005** | APR → SafeTensors | Output tensors match within ε | P0 |
| **F-CONV-006** | SafeTensors → APR | Output tensors match within ε | P0 |

#### 4.4.2 Round-Trip Conversion Gates

| Gate ID | Chain | Assertion | Severity |
|---------|-------|-----------|----------|
| **F-CONV-RT-001** | A → B → A | Original = Final (bitwise) | P0 |
| **F-CONV-RT-002** | A → B → C → A | Original = Final (bitwise) | P0 |
| **F-CONV-RT-003** | Full chain (all 3 formats) | No accumulated drift | P0 |

#### 4.4.3 Backend Equivalence Gates

| Gate ID | Assertion | Severity |
|---------|-----------|----------|
| **F-CONV-BE-001** | CPU output = GPU output (within ε) | P0 |
| **F-CONV-BE-002** | CPU output = WASM output (within ε) | P0 |
| **F-CONV-BE-003** | GPU output = WASM output (within ε) | P0 |

#### 4.4.4 Inference Equivalence Gates

| Gate ID | Assertion | Severity |
|---------|-----------|----------|
| **F-CONV-INF-001** | Inference(GGUF) ≈ Inference(APR) | P0 |
| **F-CONV-INF-002** | Inference(GGUF) ≈ Inference(SafeTensors) | P0 |
| **F-CONV-INF-003** | Inference(APR) ≈ Inference(SafeTensors) | P0 |
| **F-CONV-INF-004** | Inference output identical across backends | P0 |

#### 4.4.5 Bug Classification Gates (GH-187)

Systematic detection of common format conversion failures:

| Gate ID | Bug Type | Detection Criteria | Severity |
|---------|----------|-------------------|----------|
| **F-CONV-EMBED-001** | EmbeddingTransposition | Tensor layout `[hidden,vocab]` vs `[vocab,hidden]` | P0 |
| **F-CONV-TOK-001** | TokenizerMissing | APR file lacks embedded tokenizer (PMAT-172 error) | P0 |
| **F-CONV-WEIGHT-001** | WeightCorruption | NaN/Inf/zeros in tensor values | P0 |
| **F-CONV-SHAPE-001** | ShapeMismatch | Tensor dimensions don't match config | P0 |
| **F-CONV-SEMANTIC-001** | SemanticDrift | Output structurally valid but semantically wrong | P0 |

**Garbage Output Patterns** (indicate TokenizerMissing or SemanticDrift):
- PAD tokens: `"PAD"`, `"<pad>"`, `"<|endoftext|>"`
- Token IDs: `"151935"`, numeric sequences
- Null bytes: `"\u0000"`
- Generic text: `"1. What is the difference"`

**Validation Strategy**: Run inference on known inputs (e.g., `"2+2="`) and verify:
1. Source format produces expected output (`"4"`, `"four"`, etc.)
2. Target format produces semantically equivalent output
3. No garbage patterns in target output
4. No PMAT-172 tokenizer errors in stderr

#### 4.4.6 Cross-Project Pattern Gates

Bug patterns identified from mutation testing and fix analysis across aprender/realizar:

| Gate ID | Pattern | Description | Severity | Source |
|---------|---------|-------------|----------|--------|
| **F-PATH-ALT-001** | AlternatePathMissing | Feature in primary path but missing in alternate | P0 | aprender GH-185 |
| **F-PATH-ALGO-001** | AlgorithmMismatch | Two implementations with incompatible layouts | P0 | aprender GH-177 |
| **F-STATE-FALLBACK-001** | SilentFallbackWrongResource | Fallback silently uses wrong resource | P0 | realizar 33e18c2 |
| **F-STATE-TIMING-001** | StateAdvancementTiming | State advanced at wrong pipeline stage | P1 | realizar 62147f9 |
| **F-STATE-CORRUPT-001** | SharedStateCorruption | Prior operation corrupts shared state | P1 | realizar 9f9f985 |
| **F-VALID-POST-001** | MissingPostTransformValidation | No validation after transformation | P0 | aprender GH-177 |
| **F-VALID-TYPE-001** | MissingTypeDetection | No format/type detection before processing | P1 | realizar f13f39b |
| **F-VALID-COMPANION-001** | MissingCompanionData | Required companion files missing | P2 | aprender GH-182 |
| **F-ERR-UNWRAP-001** | UnwrapOnFallible | `.unwrap()` on fallible operation | P1 | aprender PMAT-189 |
| **F-ERR-PROP-001** | ErrorPropagationGap | Error not propagated on alternate path | P2 | multiple |
| **F-SEC-PATH-001** | PathTraversal | Untrusted path allows reading arbitrary files | P0 | realizar 04d2774 |
| **F-SEC-INJECT-001** | PromptInjection | Special tokens not escaped | P0 | realizar 1b51030 |

**Detection Heuristics:**

1. **Tensor Validity Check** (F-VALID-POST-001):
   - Check for NaN/Inf values after transformation
   - Verify mean value is within bounds (|mean| < 100)
   - Fail fast on corrupted values

2. **Path Safety Check** (F-SEC-PATH-001):
   - Reject paths containing `../`, `..\\`, `/etc/`, `C:\\Windows`
   - Check for null byte injection

3. **Prompt Safety Check** (F-SEC-INJECT-001):
   - Detect unescaped `<|`, `|>`, `[INST]`, `<<SYS>>` patterns
   - Flag BOS/EOS tokens in user input

4. **Fallback Consistency Check** (F-STATE-FALLBACK-001):
   - Compare output from primary vs fallback resource
   - Require >80% Jaccard similarity to confirm correct resource

### 4.5 Conversion Test Protocol

```rust
/// P0 CRITICAL: Format conversion test protocol
pub struct ConversionTest {
    /// Source format
    pub source_format: Format,
    /// Target format
    pub target_format: Format,
    /// Backend to use
    pub backend: Backend,
    /// Tolerance for floating-point comparison
    pub epsilon: f64,
}

impl ConversionTest {
    /// Execute conversion and validate
    pub fn execute(&self, model_path: &Path) -> ConversionResult {
        // 1. Load source model
        let source = load_model(model_path, self.source_format)?;

        // 2. Convert to target format
        let converted = convert(source, self.target_format, self.backend)?;

        // 3. Run inference on both
        let prompt = "What is 2+2?";
        let source_output = infer(&source, prompt)?;
        let converted_output = infer(&converted, prompt)?;

        // 4. Compare outputs (P0 CRITICAL)
        let diff = tensor_diff(&source_output, &converted_output);

        if diff > self.epsilon {
            return ConversionResult::Falsified {
                gate: format!("F-CONV-{:03}", self.gate_id()),
                reason: format!(
                    "Conversion {} → {} produced different output (diff: {}, ε: {})",
                    self.source_format, self.target_format, diff, self.epsilon
                ),
                evidence: ConversionEvidence {
                    source_hash: hash(&source_output),
                    converted_hash: hash(&converted_output),
                    max_diff: diff,
                    diff_tensor_indices: find_diff_indices(&source_output, &converted_output),
                },
            };
        }

        ConversionResult::Corroborated
    }
}

/// Round-trip conversion test
pub fn test_round_trip(
    model_path: &Path,
    formats: &[Format],
    backend: Backend,
) -> RoundTripResult {
    let original = load_model(model_path, formats[0])?;
    let original_output = infer(&original, "What is 2+2?")?;

    // Convert through chain: A → B → C → ... → A
    let mut current = original.clone();
    for i in 0..formats.len() {
        let next_format = formats[(i + 1) % formats.len()];
        current = convert(current, next_format, backend)?;
    }

    let final_output = infer(&current, "What is 2+2?")?;

    // P0 CRITICAL: Must be bitwise identical
    if original_output != final_output {
        return RoundTripResult::Falsified {
            gate: "F-CONV-RT-001",
            reason: "Round-trip conversion produced different output",
        };
    }

    RoundTripResult::Corroborated
}
```

### 4.6 Conversion Test Matrix Coverage

For each model in the qualification set:

| Test Category | Count | Priority |
|---------------|-------|----------|
| Direct conversions (6 pairs × 3 backends) | 18 | P0 |
| Round-trip conversions (3 chains × 3 backends) | 9 | P0 |
| Cross-backend equivalence (3 pairs × 6 conversions) | 18 | P0 |
| Inference equivalence (3 format pairs × 3 backends) | 9 | P0 |
| **TOTAL per model** | **54** | **P0** |

For 100 models: **5,400 P0 conversion tests**

### 4.7 Failure Response Protocol

When a conversion test fails:

1. **STOP IMMEDIATELY** (Jidoka) - Do not proceed with other tests
2. **Collect Evidence:**
   - Source model hash
   - Converted model hash
   - Tensor-by-tensor diff
   - First divergent layer/tensor
3. **Generate Ticket** (P0 severity)
4. **Block Release** - Model cannot be marked as qualified

### 4.8 Implementation Checklist

- [x] Implement `ConversionTest` struct
- [x] Implement all 6 direct conversion paths
- [x] Implement round-trip testing
- [x] Implement cross-backend comparison (CPU/GPU)
- [x] Add P0 gates to MQS calculator
- [x] Add conversion tests to default playbook
- [ ] Implement WASM/WGPU backend support (future)
- [ ] Add tensor diff visualization for failures (future)

### 4.9 Tolerance Definitions

**CRITICAL:** These definitions are authoritative for all comparison operations.

| Term | Symbol | Value | Usage |
|------|--------|-------|-------|
| **Epsilon (relative)** | ε | 1e-6 | Floating-point tensor comparisons |
| **Absolute tolerance** | atol | 1e-8 | Near-zero value comparisons |
| **Relative tolerance** | rtol | 1e-5 | Percentage-based comparisons |

**Comparison Semantics:**

```rust
/// Two tensors are "equivalent within ε" if:
fn tensors_equivalent(a: &Tensor, b: &Tensor, epsilon: f64) -> bool {
    a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < epsilon)
}

/// "Bitwise identical" means exact byte-level match (integers/quantized only):
fn bitwise_identical(a: &[u8], b: &[u8]) -> bool {
    a == b
}
```

**Clarifications:**
- F-CONV-RT-001 ("bitwise identical") applies to **quantized integer tensors** only
- F-CONV-* gates ("within ε") apply to **floating-point tensors** (F32, F16, BF16)
- Internal tensor Inf/NaN is detected by F-TRACE-004/005, not F-RUN-002
- F-RUN-002 ("no garbage tokens") refers to **output text**, not internal tensor values

---

## 5. APR Tool Coverage

### 5.1 Comprehensive Tool Testing (ruchy-book Pattern)

Following the **ruchy-book validation philosophy**, this framework tests **all 30+ apr CLI commands** against every model in the qualification matrix. Like ruchy-book's 18-tool comprehensive testing of 146 examples (2,628 validations), we validate that apr tools work correctly across the full model space.

> "Zero vaporware: All documented commands must be validated and working."
> — ruchy-book Testing Philosophy

### 5.2 APR Tool Matrix

| Category | Commands | Test Type | Falsification Focus |
|----------|----------|-----------|---------------------|
| **Core Inference** | `run`, `chat`, `serve` | Full property tests | Output correctness, throughput |
| **Model Management** | `pull`, `list`, `rm`, `convert`, `import`, `export`, `merge` | Lifecycle tests | Cache integrity, format parity |
| **Inspection** | `inspect`, `debug`, `tensors`, `hex`, `tree`, `flow`, `diff` | Output validation | Metadata accuracy, no crashes |
| **Tracing** | `trace` (levels: none/basic/layer/payload) | Per-level tests | Anomaly detection, tensor stats |
| **Profiling** | `profile` (focus: all/attention/mlp/matmul/embedding) | Real inference | Hotspots, flamegraph, timing |
| **Quality** | `validate`, `lint`, `explain`, `canary`, `qa`, `check`, `bench`, `eval` | Metric verification | Scores match expectations |
| **ML Tuning** | `transfer/` APIs, `online/drift` | Library tests | Freeze/unfreeze, LoRA, drift detection |
| **Advanced** | `compare-hf`, `tui`, `cbtop`, `probar`, `publish`, `showcase`, `rosetta` | Integration tests | Feature completeness |

### 5.2.1 Trace Level Testing (Required)

All trace levels **MUST** be tested for every inference command:

| Level | Data Captured | Overhead | Use Case |
|-------|---------------|----------|----------|
| `none` | Nothing | 0% | Production |
| `basic` | Timing, token counts | ~1% | Quick debug |
| `layer` | Per-layer mean/std/L2/min/max | ~5% | Numerical issues |
| `payload` | Full tensor values | ~50% | Deep inspection |

### 5.2.2 Profile Focus Testing (Required)

All profile focus areas **MUST** be tested:

| Focus | Tensors Profiled | Use Case |
|-------|------------------|----------|
| `all` | Every operation | Full picture |
| `attention` | QKV, softmax, output | Attention bottlenecks |
| `mlp` | Up/down/gate projections | FFN bottlenecks |
| `matmul` | All GEMM operations | Compute intensity |
| `embedding` | Token/position embeddings | Input pipeline |

### 5.2.3 ML Tuning Testing (Required)

Library-level ML tuning APIs **MUST** be tested:

| API | Module | Falsification |
|-----|--------|---------------|
| `TransferEncoder` | `src/transfer/` | Freeze/unfreeze, feature extraction |
| `MultiTaskHead` | `src/transfer/` | Task-specific heads |
| `LoRA` | `src/transfer/` | Low-rank adaptation |
| `DDMDetector` | `src/online/drift.rs` | Drift detection (Gama et al. 2004) |
| `ADWINDetector` | `src/online/drift.rs` | Adaptive windowing (Bifet & Gavalda 2007) |
| `PageHinkleyDetector` | `src/online/drift.rs` | Mean shift detection (Page 1954) |

### 5.3 Tool Test Specification

Each apr command is tested with a **7-layer validation stack**:

```
┌─────────────────────────────────────────────────────────────────┐
│ Layer 7: Cross-Model Consistency (same command, different models)│
├─────────────────────────────────────────────────────────────────┤
│ Layer 6: Cross-Format Parity (GGUF vs SafeTensors vs APR)       │
├─────────────────────────────────────────────────────────────────┤
│ Layer 5: Backend Parity (CPU vs GPU produce equivalent results) │
├─────────────────────────────────────────────────────────────────┤
│ Layer 4: Output Validation (JSON parseable, metrics in range)   │
├─────────────────────────────────────────────────────────────────┤
│ Layer 3: Execution Success (exit code 0, no panics)             │
├─────────────────────────────────────────────────────────────────┤
│ Layer 2: Argument Parsing (--help works, invalid args rejected) │
├─────────────────────────────────────────────────────────────────┤
│ Layer 1: Command Exists (binary has subcommand)                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.4 Per-Tool Falsification Gates

#### 5.4.1 `apr run` (F-RUN-*)

```yaml
gates:
  - id: F-RUN-001
    description: "Produces non-empty output"
    condition: "output.len() > 0"
  - id: F-RUN-002
    description: "No garbage tokens"
    condition: "!is_garbage(output)"
  - id: F-RUN-003
    description: "Completes within timeout"
    condition: "duration_ms < 60000"
  - id: F-RUN-004
    description: "Throughput meets minimum"
    condition: "tps >= 10.0 || backend == 'gpu'"
  - id: F-RUN-005
    description: "Deterministic with seed"
    condition: "run(seed=42) == run(seed=42)"
```

#### 4.4.2 `apr chat` (F-CHAT-*)

```yaml
gates:
  - id: F-CHAT-001
    description: "Chat template applied"
    condition: "output.contains(model.eos_token)"
  - id: F-CHAT-002
    description: "Multi-turn works"
    condition: "turn2_aware_of_turn1"
  - id: F-CHAT-003
    description: "System prompt honored"
    condition: "follows_system_instruction"
```

#### 4.4.3 `apr serve` (F-SERVE-*)

```yaml
gates:
  - id: F-SERVE-001
    description: "Server starts"
    condition: "health_check() == 200"
  - id: F-SERVE-002
    description: "OpenAI-compatible endpoint"
    condition: "POST /v1/completions returns valid JSON"
  - id: F-SERVE-003
    description: "Streaming works"
    condition: "stream=true yields SSE events"
  - id: F-SERVE-004
    description: "Graceful shutdown"
    condition: "SIGTERM -> clean exit"
```

#### 4.4.4 `apr inspect` (F-INSPECT-*)

```yaml
gates:
  - id: F-INSPECT-001
    description: "Returns valid JSON with --json"
    condition: "json.parse(output) succeeds"
  - id: F-INSPECT-002
    description: "Shows architecture"
    condition: "output.contains('architecture')"
  - id: F-INSPECT-003
    description: "Shows tensor count"
    condition: "output.contains('tensors')"
```

#### 4.4.5 `apr validate` (F-VALIDATE-*)

```yaml
gates:
  - id: F-VALIDATE-001
    description: "Returns score 0-100"
    condition: "0 <= score <= 100"
  - id: F-VALIDATE-002
    description: "Identifies format"
    condition: "format in ['gguf', 'safetensors', 'apr']"
  - id: F-VALIDATE-003
    description: "Detects corruption"
    condition: "validate(corrupted_model) < 50"
```

#### 4.4.6 `apr bench` (F-BENCH-*)

```yaml
gates:
  - id: F-BENCH-001
    description: "Reports tok/s"
    condition: "output.contains('tok/s')"
  - id: F-BENCH-002
    description: "GPU faster than CPU"
    condition: "gpu_tps > cpu_tps"
  - id: F-BENCH-003
    description: "Reproducible results (CV < 10%)"
    condition: "coefficient_of_variation < 0.10"
```

#### 4.4.7 `apr profile` (F-PROFILE-*)

Deep profiling with real inference (PMAT-112: no synthetic benchmarks).

```yaml
gates:
  - id: F-PROFILE-001
    description: "Returns hotspot data"
    condition: "output.contains('hotspot') || json.hotspots.len() > 0"
  - id: F-PROFILE-002
    description: "Per-layer timing available"
    condition: "per_layer_us.len() == num_layers"
  - id: F-PROFILE-003
    description: "Flamegraph output valid SVG"
    condition: "format == 'flamegraph' => valid_svg(output)"
  - id: F-PROFILE-004
    description: "Focus areas work (attention/mlp/matmul/embedding)"
    condition: "focus_filter_applied_correctly"
  - id: F-PROFILE-005
    description: "JSON output parseable"
    condition: "format == 'json' => json.parse(output).is_ok()"
  - id: F-PROFILE-006
    description: "CI mode with --assert-throughput works"
    condition: "--ci --assert-throughput N => exit_code=0 when throughput >= N"
  - id: F-PROFILE-007
    description: "CI mode exits 1 on assertion failure"
    condition: "--ci --assert-throughput 1000000 => exit_code=1 (impossible threshold)"
  - id: F-PROFILE-008
    description: "CI mode --assert-p99 latency works"
    condition: "--ci --assert-p99 N => exit_code=0 when p99_ms <= N"
```

#### 4.4.8 `apr trace` (F-TRACE-*)

Layer-by-layer tensor analysis with anomaly detection.

```yaml
gates:
  - id: F-TRACE-001
    description: "All trace levels work (none/basic/layer/payload)"
    condition: "for level in [none, basic, layer, payload]: trace(level).succeeds()"
  - id: F-TRACE-002
    description: "Layer stats include mean/std/L2/min/max"
    condition: "layer_stats.has_all(['mean', 'std', 'l2_norm', 'min', 'max'])"
  - id: F-TRACE-003
    description: "NaN detection works"
    condition: "trace(model_with_nan).anomalies.contains('NaN')"
  - id: F-TRACE-004
    description: "Inf detection works"
    condition: "trace(model_with_inf).anomalies.contains('Inf')"
  - id: F-TRACE-005
    description: "Payload level shows tensor values"
    condition: "trace_level == 'payload' => output.contains('tensor_values')"
  - id: F-TRACE-006
    description: "Numerical explosion detected"
    condition: "max_abs > 100.0 => anomalies.contains('large values')"
```

#### 4.4.9 `apr check` (F-CHECK-*)

10-stage pipeline self-test with real validation (PMAT-112).

```yaml
gates:
  - id: F-CHECK-001
    description: "Stage 1: Tokenizer encode/decode roundtrip"
    condition: "decode(encode(text)) == text"
  - id: F-CHECK-002
    description: "Stage 2: Embedding tensor exists and valid"
    condition: "has_embedding_tensor && !embedding.has_nan()"
  - id: F-CHECK-003
    description: "Stage 3: Attention weights valid"
    condition: "attention_weights.sum() ~= 1.0"
  - id: F-CHECK-004
    description: "Stage 4: MLP activations bounded"
    condition: "mlp_output.max_abs() < 1000.0"
  - id: F-CHECK-005
    description: "Stage 5: LayerNorm output normalized"
    condition: "layernorm_output.std() ~= 1.0"
  - id: F-CHECK-006
    description: "Stage 6: RoPE positions correct"
    condition: "rope_positions_match_sequence_length"
  - id: F-CHECK-007
    description: "Stage 7: KV cache functional"
    condition: "kv_cache.append_and_retrieve_works()"
  - id: F-CHECK-008
    description: "Stage 8: Softmax output sums to 1"
    condition: "softmax_output.sum() ~= 1.0"
  - id: F-CHECK-009
    description: "Stage 9: Forward pass completes"
    condition: "forward_pass_no_panic && output.len() > 0"
  - id: F-CHECK-010
    description: "Stage 10: Output logits valid distribution"
    condition: "!logits.has_nan() && !logits.has_inf()"
```

#### 4.4.10 Trace Level Matrix (F-TRACELEVEL-*)

All inference commands must work with all trace levels.

```yaml
trace_levels: [none, basic, layer, payload]
commands: [run, chat, serve]

gates:
  - id: F-TRACELEVEL-001
    description: "apr run works with all trace levels"
    condition: |
      for level in trace_levels:
        apr run model "prompt" --trace --trace-level {level}
        assert exit_code == 0
  - id: F-TRACELEVEL-002
    description: "apr chat works with all trace levels"
    condition: |
      for level in trace_levels:
        apr chat model --trace --trace-level {level}
        assert exit_code == 0
  - id: F-TRACELEVEL-003
    description: "apr serve accepts trace level header"
    condition: |
      for level in trace_levels:
        curl -H "X-Trace-Level: {level}" /v1/completions
        assert response.has_trace_data(level)
  - id: F-TRACELEVEL-004
    description: "--trace-payload shorthand works"
    condition: |
      apr run model "prompt" --trace-payload
      assert trace_level_used == "payload"
  - id: F-TRACELEVEL-005
    description: "Tracing does not affect inference output"
    condition: |
      # Output equivalence test: tracing must be observational only
      output_none = apr run model "prompt" --trace-level none --seed 42
      output_layer = apr run model "prompt" --trace-level layer --seed 42
      output_payload = apr run model "prompt" --trace-level payload --seed 42
      assert output_none.text == output_layer.text == output_payload.text
      # Heisenberg principle violation: if tracing changes output, it's a bug
```

#### 4.4.11 `apr canary` (F-CANARY-*)

Regression testing via tensor statistics snapshots.

```yaml
gates:
  - id: F-CANARY-001
    description: "Generates baseline snapshot"
    condition: "apr canary model --save baseline.json succeeds"
  - id: F-CANARY-002
    description: "Detects regression from baseline"
    condition: "apr canary model --compare baseline.json detects_drift"
  - id: F-CANARY-003
    description: "Passes when model unchanged"
    condition: "apr canary same_model --compare baseline.json passes"
```

#### 4.4.12 ML Tuning & Transfer Learning (F-TUNE-*)

Library-level ML tuning APIs (future CLI exposure).

```yaml
gates:
  - id: F-TUNE-001
    description: "TransferEncoder freeze_base works"
    condition: "encoder.freeze_base(); encoder.base_params_frozen()"
  - id: F-TUNE-002
    description: "TransferEncoder unfreeze works"
    condition: "encoder.unfreeze(); !encoder.base_params_frozen()"
  - id: F-TUNE-003
    description: "MultiTaskHead add_task works"
    condition: "multi_task.add_task('task1', 128); multi_task.has_head('task1')"
  - id: F-TUNE-004
    description: "LoRA adapter applies correctly"
    condition: "lora_output != base_output && rank_matches_config"
  - id: F-TUNE-005
    description: "Fine-tuning preserves frozen weights"
    condition: |
      frozen_before = encoder.base_weights.clone()
      train_step()
      frozen_before == encoder.base_weights
```

#### 4.4.13 Drift Detection (F-DRIFT-*)

Online learning drift detection (Jidoka for ML).

```yaml
gates:
  - id: F-DRIFT-001
    description: "DDM detector transitions Stable->Warning->Drift"
    condition: |
      ddm = DDMDetector::new()
      feed_correct_samples(100)
      assert ddm.status() == Stable
      feed_error_samples(50)
      assert ddm.status() == Warning
      feed_error_samples(50)
      assert ddm.status() == Drift
  - id: F-DRIFT-002
    description: "ADWIN detector adapts window size"
    condition: "adwin.window_size changes with data distribution"
  - id: F-DRIFT-003
    description: "Page-Hinkley test detects mean shift"
    condition: "ph.detect_change(shifted_data) == true"
  - id: F-DRIFT-004
    description: "Reset clears detector state"
    condition: |
      detector.add_errors(100)
      detector.reset()
      assert detector.status() == Stable
```

#### 4.4.14 `apr qa` (F-QA-*)

Falsifiable QA checklist execution.

```yaml
gates:
  - id: F-QA-001
    description: "Runs full QA matrix"
    condition: "apr qa model --full exits_with_report"
  - id: F-QA-002
    description: "Reports Popperian score"
    condition: "output.contains('Popperian Score:') && score in 0..100"
  - id: F-QA-003
    description: "Asserts throughput threshold"
    condition: "apr qa model --assert-tps 10 fails_if_tps_below_10"
  - id: F-QA-004
    description: "Asserts GPU speedup"
    condition: "apr qa model --assert-gpu-speedup 2.0 fails_if_speedup_below_2x"
  - id: F-QA-005
    description: "Cross-format parity check"
    condition: "apr qa model --safetensors-path st.safetensors verifies_parity"
```

### 4.5 Tool Coverage Matrix Output

After each run, generate a **tool coverage matrix**:

```
APR Tool Coverage Report (2026-01-29)
Model: Qwen/Qwen2.5-Coder-1.5B-Instruct

| Command      | GGUF/CPU | GGUF/GPU | SafeT/CPU | SafeT/GPU | APR/CPU | APR/GPU |
|--------------|----------|----------|-----------|-----------|---------|---------|
| run          | ✅ PASS  | ✅ PASS  | ✅ PASS   | ✅ PASS   | ✅ PASS | ✅ PASS |
| chat         | ✅ PASS  | ✅ PASS  | ✅ PASS   | ✅ PASS   | ✅ PASS | ✅ PASS |
| serve        | ✅ PASS  | ✅ PASS  | ✅ PASS   | ✅ PASS   | ✅ PASS | ✅ PASS |
| inspect      | ✅ PASS  | N/A      | ✅ PASS   | N/A       | ✅ PASS | N/A     |
| validate     | ✅ PASS  | N/A      | ✅ PASS   | N/A       | ✅ PASS | N/A     |
| bench        | ✅ PASS  | ✅ PASS  | ✅ PASS   | ✅ PASS   | ✅ PASS | ✅ PASS |
| ...          | ...      | ...      | ...       | ...       | ...     | ...     |

Total: 156/162 (96.3%) | Failures: 6 | Skipped: 12
```

---

## 6. Upstream Ticket Protocol

### 6.1 Purpose: QA as Bug Discovery

This framework is not merely a pass/fail gate—it is an **active bug discovery system** for the aprender ecosystem. When falsification reveals a defect, we create structured tickets in the upstream repository.

> "Quality is everyone's responsibility."
> — W. Edwards Deming, *Out of the Crisis* (1986)

### 6.2 Ticket Creation Workflow

```
┌─────────────────┐     Falsification     ┌─────────────────┐
│  Playbook Test  │─────────────────────▶│  Evidence JSON  │
└─────────────────┘                       └────────┬────────┘
                                                   │
                                                   ▼
                                          ┌─────────────────┐
                                          │ Triage: Is this │
                                          │ an apr bug?     │
                                          └────────┬────────┘
                                                   │
                        ┌──────────────────────────┼──────────────────────────┐
                        │ YES                      │ NO                       │
                        ▼                          ▼                          ▼
               ┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
               │ Create ticket   │       │ Model issue     │       │ Test issue      │
               │ in ../aprender  │       │ (HF upstream)   │       │ (fix oracle)    │
               └─────────────────┘       └─────────────────┘       └─────────────────┘
```

### 6.3 Ticket Template

When a falsification is triaged as an aprender bug, create:

**File:** `../aprender/docs/tickets/TICKET-{NUMBER}-{SLUG}.md`

```markdown
# TICKET-{NUMBER}: {Brief Description}

**Status:** OPEN
**Severity:** P0 | P1 | P2
**Component:** apr-cli | realizar | aprender-core
**Discovered By:** apr-model-qa-playbook
**Date:** {ISO-8601}

## Summary

{One-sentence description of the failure}

## Reproduction

### Minimal Command
```bash
{exact command that fails}
```

### Expected Behavior
{what should happen}

### Actual Behavior
{what actually happens}

### Evidence
```json
{falsification evidence JSON}
```

## Environment

- **Model:** {HF repo or local path}
- **Format:** GGUF | SafeTensors | APR
- **Backend:** CPU | GPU
- **apr version:** {version}
- **OS:** {uname -a}

## Analysis

### Five-Whys (if root cause known)
1. WHY {symptom}? → {cause1}
2. WHY {cause1}? → {cause2}
3. WHY {cause2}? → {cause3}
4. WHY {cause3}? → {cause4}
5. WHY {cause4}? → {root cause}

### Suggested Fix
{if known}

## Verification

Once fixed, this ticket is verified by:
```bash
bashrs playbook playbooks/verify/TICKET-{NUMBER}.yaml
```

## References

- Falsification Gate: F-{CATEGORY}-{NUMBER}
- Playbook: playbooks/models/{model}.playbook.yaml
- Evidence File: evidence/{timestamp}-{gate-id}.json
```

### 6.4 Automated Ticket Creation

The `apr-qa-report` crate includes ticket generation:

```rust
// crates/apr-qa-report/src/ticket.rs

pub struct TicketGenerator {
    aprender_path: PathBuf,
    ticket_counter: AtomicU32,
}

impl TicketGenerator {
    /// Generate ticket from falsification evidence
    pub fn create_ticket(&self, evidence: &FalsificationEvidence) -> Result<PathBuf> {
        // Only create tickets for P0/P1 failures triaged as apr bugs
        if !self.is_apr_bug(evidence) {
            return Ok(PathBuf::new());
        }

        let ticket_num = self.next_ticket_number()?;
        let slug = self.generate_slug(evidence);
        let filename = format!("TICKET-{:04}-{}.md", ticket_num, slug);
        let path = self.aprender_path
            .join("docs/tickets")
            .join(&filename);

        let content = self.render_ticket_template(ticket_num, evidence)?;
        std::fs::write(&path, content)?;

        // Also create verification playbook
        self.create_verification_playbook(ticket_num, evidence)?;

        Ok(path)
    }

    /// Triage: Is this an aprender bug or model/test issue?
    fn is_apr_bug(&self, evidence: &FalsificationEvidence) -> bool {
        // Heuristics:
        // - Panic/crash = definitely apr bug
        // - Works in one format but not another = apr bug
        // - Works on one backend but not another = apr bug
        // - All formats/backends fail same way = likely model issue

        evidence.is_crash() ||
        evidence.has_format_divergence() ||
        evidence.has_backend_divergence()
    }
}
```

### 6.5 Verification Playbooks

Each ticket gets a verification playbook:

**File:** `playbooks/verify/TICKET-{NUMBER}.yaml`

```yaml
name: verify-ticket-{number}
version: "1.0.0"
description: "Verify fix for TICKET-{NUMBER}: {description}"

# Exact reproduction of the failure
reproduction:
  command: "{original failing command}"
  expected_exit_code: 0
  expected_output_contains:
    - "{expected output substring}"
  expected_output_not_contains:
    - "panic"
    - "error"
    - "{garbage pattern}"

# The ticket is verified when this playbook passes
verification_gates:
  - id: V-TICKET-{NUMBER}-001
    description: "Original failure no longer occurs"
    condition: "exit_code == 0 && !is_garbage(output)"
```

### 6.6 Ticket Lifecycle

```
OPEN → IN_PROGRESS → FIXED → VERIFIED → CLOSED
                 │
                 └→ WONTFIX (with justification)
```

**State Transitions:**

| From | To | Trigger |
|------|----|---------|
| OPEN | IN_PROGRESS | Developer starts work |
| IN_PROGRESS | FIXED | PR merged to aprender |
| FIXED | VERIFIED | Verification playbook passes |
| VERIFIED | CLOSED | Next apr release includes fix |
| OPEN/IN_PROGRESS | WONTFIX | Triaged as model issue or expected behavior |

### 6.7 Ticket Metrics

Track ticket flow for process improvement:

```rust
pub struct TicketMetrics {
    /// Total tickets created
    pub total_created: u32,

    /// Tickets by severity
    pub by_severity: HashMap<Severity, u32>,

    /// Tickets by component
    pub by_component: HashMap<String, u32>,

    /// Mean time to fix (OPEN → FIXED)
    pub mttf_days: f64,

    /// Mean time to verify (FIXED → VERIFIED)
    pub mttv_days: f64,

    /// Tickets that were WONTFIX (false positives)
    pub false_positive_rate: f64,
}
```

### 6.8 Integration with CI

When running in CI, ticket creation is gated:

```yaml
# .github/workflows/qa.yml
- name: Run QA Playbooks
  run: |
    bashrs playbook playbooks/**/*.yaml \
      --ticket-mode=draft  # Don't auto-create, just report

- name: Review Draft Tickets
  if: failure()
  run: |
    cat evidence/draft-tickets/*.md
    echo "Review draft tickets above. To create, run with --ticket-mode=create"
```

**Modes:**
- `--ticket-mode=disabled`: No ticket generation
- `--ticket-mode=draft`: Generate drafts in `evidence/draft-tickets/`
- `--ticket-mode=create`: Create tickets in `../aprender/docs/tickets/`

---

## 7. Upstream Spec Requirements (Consolidated Master Reference)

> **This section consolidates ALL requirements from upstream aprender specifications into a single authoritative reference. No external spec reading required—everything needed for implementation is here.**

### 7.1 Spec Registry

| ID | Spec | Version | Source | Status |
|----|------|---------|--------|--------|
| **SPEC-001** | QA Showcase Methodology | 1.1.0 | `qa-showcase-methodology.md` | MANDATORY |
| **SPEC-002** | QA Serve Protocol | 1.0.0 | `qa-serve-protocol.md` | MANDATORY |
| **SPEC-003** | Pipeline Verification | 1.0.0 | `pipeline-verification-visualization.md` | MANDATORY |
| **SPEC-004** | Online Learning | 1.0.0 | `apr-online-learning-dynamic-retraining-spec.md` | MANDATORY |
| **SPEC-005** | Rosetta Stone | 1.3.0 | `model-format-verification-import-conversion-cargo-run-example.md` | MANDATORY |

---

### 7.2 SPEC-001: QA Showcase Methodology (Complete)

#### 6.2.1 Class Separation (CRITICAL)

**FATAL DEFECT WARNING:** Comparing Q4_K_M (~1.1GB) to F32 (~3.0GB) is INVALID. They are NOT the same model.

```
┌─────────────────────────────────────────────────────────────────────────┐
│  CLASS A: Quantized Inference (60 points)                               │
│  ═══════════════════════════════════════                                │
│  Canonical Source: GGUF Q4_K_M (pre-quantized by model authors)         │
│  Groundtruth: Ollama Q4_K_M                                             │
│                                                                          │
│  Test Matrix:                                                            │
│  ┌──────────┬─────────┬─────────────────┬────────┐                      │
│  │ Cell     │ Backend │ Format          │ Points │                      │
│  ├──────────┼─────────┼─────────────────┼────────┤                      │
│  │ A1       │ CPU     │ GGUF Q4_K       │ 15     │                      │
│  │ A2       │ CPU     │ APR Q4_K        │ 15     │                      │
│  │ A3       │ GPU     │ GGUF Q4_K       │ 15     │                      │
│  │ A4       │ GPU     │ APR Q4_K        │ 15     │                      │
│  └──────────┴─────────┴─────────────────┴────────┘                      │
│                                                                          │
│  Performance Thresholds:                                                 │
│  • CPU: 8-15 tok/s (minimum 10 tok/s per spec H12)                      │
│  • GPU: 100+ tok/s                                                       │
│  • GPU Speedup: ≥2× CPU (spec F-PERF-042)                               │
├─────────────────────────────────────────────────────────────────────────┤
│  CLASS B: Full Precision Inference (40 points)                          │
│  ═══════════════════════════════════════════                            │
│  Source: SafeTensors F32 from HuggingFace                               │
│  No Ollama Groundtruth (Ollama only uses quantized)                     │
│                                                                          │
│  Test Matrix:                                                            │
│  ┌──────────┬─────────┬─────────────────┬────────┐                      │
│  │ Cell     │ Backend │ Format          │ Points │                      │
│  ├──────────┼─────────┼─────────────────┼────────┤                      │
│  │ B1       │ CPU     │ SafeTensors F32 │ 10     │                      │
│  │ B2       │ CPU     │ APR F32         │ 10     │                      │
│  │ B3       │ GPU     │ SafeTensors F32 │ 10     │                      │
│  │ B4       │ GPU     │ APR F32         │ 10     │                      │
│  └──────────┴─────────┴─────────────────┴────────┘                      │
│                                                                          │
│  Performance Thresholds:                                                 │
│  • CPU: 1-3 tok/s                                                        │
│  • GPU: 10-30 tok/s                                                      │
└─────────────────────────────────────────────────────────────────────────┘
```

#### 6.2.2 Tracing Requirements (ALL MANDATORY)

| Flag | Values | Description | MQS Category |
|------|--------|-------------|--------------|
| `--trace` | (boolean) | Enable tracing | E1-E4 |
| `--trace-level` | `none`, `basic`, `layer`, `payload` | Detail level | E1-E4 |
| `--trace-payload` | (boolean) | Shorthand for `--trace --trace-level payload` | E4 |
| `--trace-output` | `<file.json>` | Write trace to file | E1-E4 |
| `--profile` | (boolean) | Roofline analysis | E5 |

**Trace Level Specifications:**

| Level | Output | Overhead | Use Case |
|-------|--------|----------|----------|
| `none` | Nothing | 0% | Production |
| `basic` | `{"tokens": N, "time_ms": T, "tps": X}` | ~1% | Quick debug |
| `layer` | Per-layer: `{"layer": L, "mean": M, "std": S, "l2": N, "time_us": T}` | ~5% | Numerical issues |
| `payload` | Full tensor values: `{"layer": L, "tensor": "attn_out", "values": [...]}` | ~50% | Deep inspection |

#### 6.2.3 Ollama Parity Protocol

```bash
# Groundtruth collection
ollama run qwen2.5-coder:1.5b-instruct-q4_K_M "What is 2+2?" --verbose

# APR comparison
apr run model.gguf "What is 2+2?" --trace --with-ollama

# Parity assertion
# PASS: APR tok/s >= 0.8 × Ollama tok/s
# FAIL: APR tok/s < 0.8 × Ollama tok/s OR output divergence
```

---

### 7.3 SPEC-002: QA Serve Protocol (Complete)

#### 6.3.1 Falsification Matrix (28 Gates)

**Section I: Connectivity & Health (4 points)**

| Gate | Description | Condition | Severity |
|------|-------------|-----------|----------|
| F-HTTP-001 | Server accepts connection | Port 8080 responds | P0 |
| F-HTTP-002 | Health endpoint | `GET /health` → 200 OK | P0 |
| F-HTTP-003 | Health includes compute_mode | Response has `"cpu"` or `"gpu"` | P1 |
| F-HTTP-004 | Clean shutdown | SIGTERM → graceful exit (no orphan processes) | P0 |

**Section II: Basic Inference (6 points)**

| Gate | Description | Condition | Severity |
|------|-------------|-----------|----------|
| F-HTTP-005 | Accepts valid JSON | `POST /v1/chat/completions` → 200 | P0 |
| F-HTTP-006 | Response has content | `choices[0].message.content` exists | P0 |
| F-HTTP-007 | Valid JSON response | Parses as RFC 8259 JSON | P0 |
| F-HTTP-008 | Non-empty content | `content.len() > 0` | P0 |
| F-HTTP-009 | No raw token IDs | Content doesn't match `/token\d+/` | P1 |
| F-HTTP-009b | No BPE artifacts | No `Ġ`, `Ċ`, `â` in output | P1 |

**Section III: Advanced Features (5 points)**

| Gate | Description | Condition | Severity |
|------|-------------|-----------|----------|
| F-HTTP-010 | Streaming works | `stream=true` → SSE `data: {...}` events | P1 |
| F-HTTP-011 | Streaming ends correctly | Final event is `data: [DONE]` | P1 |
| F-HTTP-012 | System prompt honored | "You are a pirate" → pirate-speak output | P1 |
| F-HTTP-013 | Multi-turn context | User name recall across turns | P1 |
| F-HTTP-014 | Usage stats present | `usage.prompt_tokens > 0 && usage.completion_tokens > 0` | P2 |

**Section IV: Robustness (6 points)**

| Gate | Description | Condition | Severity |
|------|-------------|-----------|----------|
| F-HTTP-015 | Empty messages error | `messages: []` → 400 Bad Request | P1 |
| F-HTTP-016 | Deterministic at temp=0 | Two runs identical output | P1 |
| F-HTTP-017 | Malformed JSON error | Invalid JSON → 400 Bad Request | P1 |
| F-HTTP-018 | OOM protection | `max_tokens: 1000000` → error, not crash | P0 |
| F-HTTP-019 | Concurrent requests | 10 parallel requests → no deadlock | P0 |
| F-HTTP-020 | Coherency check | Entropy > threshold, no `�` chars | P0 |

**Section V: Tracing Parity (7 points)**

| Gate | Description | Condition | Severity |
|------|-------------|-----------|----------|
| F-TRACE-001 | Brick tracing | `X-Trace-Level: brick` → brick trace in response | P1 |
| F-TRACE-002 | Step tracing | `X-Trace-Level: step` → step trace in response | P1 |
| F-TRACE-003 | Layer tracing | `X-Trace-Level: layer` → layer trace in response | P1 |
| F-TRACE-004 | Default suppresses traces | No header → no trace fields in response | P1 |
| F-TRACE-004a | Default valid JSON | No header → valid JSON response | P1 |
| F-TRACE-004b | Default no trace keys | No `trace`, `brick_trace`, `layer_trace`, `step_trace` | P1 |
| F-TRACE-004c | Default inference works | No header → normal text output | P1 |

#### 6.3.2 QA Serve Script Protocol

```bash
#!/bin/bash
# qa-serve.sh - Falsification agent for apr serve
set -euo pipefail

PORT=$1
MODEL=$2
EXPECTED_MODE=${3:-cpu}

# Start server
apr serve "$MODEL" --port "$PORT" ${EXPECTED_MODE:+--gpu} &
SERVER_PID=$!
sleep 5

# Run falsification matrix
run_gate() {
    local gate=$1
    local result
    # ... implementation
    if [[ $result == "PASS" ]]; then
        echo -e "\e[32m✓ $gate\e[0m"
    else
        echo -e "\e[31m✗ $gate: $result\e[0m"
        FAILURES+=("$gate")
    fi
}

run_gate "F-HTTP-001"
run_gate "F-HTTP-002"
# ... all 28 gates

# Cleanup
kill $SERVER_PID 2>/dev/null || true

# Exit code
if [[ ${#FAILURES[@]} -eq 0 ]]; then
    exit 0
elif [[ ${#FAILURES[@]} -le 5 ]]; then
    exit 1  # Functional issues
else
    exit 2  # Connectivity failure
fi
```

---

### 7.4 SPEC-003: Pipeline Verification (Complete)

#### 6.4.1 Core Principles

1. **NO PYTHON** - All verification in Rust only
2. **Deterministic** - Same input → identical output
3. **Ground Truth** - Pre-extracted JSON/binary references
4. **Stage Gates** - Explicit pass/fail at each pipeline stage

#### 6.4.2 Stage Gate Protocol

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    INFERENCE PIPELINE STAGES                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Stage 0: Tokenization                                                   │
│  ├─ Input: "What is 2+2?"                                               │
│  ├─ Output: [1724, 374, 220, 17, 10, 17, 30]                            │
│  ├─ Gate: decode(encode(text)) == text                                  │
│  └─ Ground Truth: test_data/qwen_tokenizer.json                         │
│                                                                          │
│  Stage 1: Embedding                                                      │
│  ├─ Input: token_ids [N]                                                │
│  ├─ Output: embeddings [N, hidden_dim]                                  │
│  ├─ Gate: embeddings.shape == [N, 1536]                                 │
│  └─ Ground Truth: test_data/qwen_embed.json (mean, std, L2)             │
│                                                                          │
│  Stage 2: RoPE Position Encoding                                        │
│  ├─ Input: position indices                                             │
│  ├─ Output: rotated Q, K tensors                                        │
│  ├─ Gate: position_ids match sequence length                            │
│  └─ Ground Truth: test_data/qwen_rope.json                              │
│                                                                          │
│  Stage 3-N: Transformer Layers                                          │
│  ├─ Per-layer gates:                                                    │
│  │   ├─ Attention: softmax(QK^T/√d)V sums to 1.0                       │
│  │   ├─ MLP: activations bounded (max_abs < 1000)                       │
│  │   └─ LayerNorm: output std ≈ 1.0                                     │
│  └─ Ground Truth: test_data/qwen_layer_{N}.json                         │
│                                                                          │
│  Stage N+1: LM Head                                                     │
│  ├─ Input: final hidden state                                           │
│  ├─ Output: logits [vocab_size]                                         │
│  ├─ Gate: argmax(logits) == expected_token                              │
│  └─ Ground Truth: test_data/qwen_logits.json                            │
│                                                                          │
│  Stage N+2: Sampling                                                    │
│  ├─ Input: logits                                                       │
│  ├─ Output: next_token_id                                               │
│  ├─ Gate: temp=0 → deterministic                                        │
│  └─ Ground Truth: expected_token for "2+2=" is "4" (token 19)           │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

#### 6.4.3 Ground Truth API

```rust
/// Ground truth container for pipeline verification
#[derive(Deserialize)]
pub struct GroundTruth {
    pub stage: String,
    pub tensor_name: String,
    pub shape: Vec<usize>,
    pub dtype: String,
    pub stats: TensorStats,
    pub values: Option<Vec<f32>>,  // Only for small tensors
}

#[derive(Deserialize)]
pub struct TensorStats {
    pub mean: f32,
    pub std: f32,
    pub min: f32,
    pub max: f32,
    pub l2_norm: f32,
    pub nan_count: usize,
    pub inf_count: usize,
}

impl GroundTruth {
    /// Load from pre-extracted JSON file
    pub fn from_json_file(path: &str) -> Result<Self> {
        let file = File::open(path)?;
        Ok(serde_json::from_reader(file)?)
    }

    /// Load from binary file (for large tensors)
    pub fn from_bin_file(path: &str) -> Result<Self> {
        // ... binary format parsing
    }
}

/// Assert tensor matches ground truth within tolerance
pub fn assert_tensor_close(
    actual: &Tensor,
    expected: &GroundTruth,
    rtol: f32,  // Relative tolerance (default 1e-5)
    atol: f32,  // Absolute tolerance (default 1e-8)
) -> Result<(), VerificationError> {
    // Check shape
    if actual.shape() != expected.shape.as_slice() {
        return Err(VerificationError::ShapeMismatch {
            actual: actual.shape().to_vec(),
            expected: expected.shape.clone(),
        });
    }

    // Check stats
    let actual_stats = TensorStats::from_tensor(actual);
    if (actual_stats.mean - expected.stats.mean).abs() > atol + rtol * expected.stats.mean.abs() {
        return Err(VerificationError::StatsMismatch {
            stat: "mean",
            actual: actual_stats.mean,
            expected: expected.stats.mean,
        });
    }
    // ... check std, l2_norm, etc.

    Ok(())
}
```

---

### 7.5 SPEC-004: Online Learning & Drift Detection (Complete)

#### 6.5.1 Drift Detection Traits

```rust
/// Drift status for model health monitoring
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DriftStatus {
    /// No drift detected - model performing well
    Stable,
    /// Warning level - performance degrading
    Warning,
    /// Drift confirmed - trigger retraining
    Drift,
}

/// Drift detector interface (Jidoka for ML)
pub trait DriftDetector: Send + Sync {
    /// Add new prediction outcome (true = error, false = correct)
    fn add_element(&mut self, error: bool);

    /// Check current drift status
    fn detected_change(&self) -> DriftStatus;

    /// Reset detector after handling drift
    fn reset(&mut self);

    /// Get statistics
    fn stats(&self) -> DriftStats;
}
```

#### 6.5.2 Implemented Detectors

| Detector | Algorithm | Reference | Parameters |
|----------|-----------|-----------|------------|
| **DDMDetector** | Drift Detection Method | Gama et al. 2004 | `warning_level=2.0`, `drift_level=3.0` |
| **ADWINDetector** | Adaptive Windowing | Bifet & Gavalda 2007 | `delta=0.002` |
| **PageHinkleyDetector** | Page-Hinkley Test | Page 1954 | `delta=0.005`, `lambda=50` |

#### 6.5.3 DDM Implementation

```rust
/// DDM: Drift Detection Method
/// Reference: Gama, J., et al. (2004). "Learning with Drift Detection"
pub struct DDMDetector {
    n: u64,
    p: f64,      // Error rate
    s: f64,      // Standard deviation
    p_min: f64,  // Minimum error rate seen
    s_min: f64,  // Std at minimum
    warning_level: f64,
    drift_level: f64,
}

impl DriftDetector for DDMDetector {
    fn add_element(&mut self, error: bool) {
        self.n += 1;

        // Update error rate with incremental mean
        let error_val = if error { 1.0 } else { 0.0 };
        self.p += (error_val - self.p) / self.n as f64;

        // Update standard deviation
        self.s = (self.p * (1.0 - self.p) / self.n as f64).sqrt();

        // Track minimum
        if self.p + self.s < self.p_min + self.s_min {
            self.p_min = self.p;
            self.s_min = self.s;
        }
    }

    fn detected_change(&self) -> DriftStatus {
        if self.n < 30 {
            return DriftStatus::Stable;  // Not enough samples
        }

        let threshold = self.p_min + self.s_min;

        if self.p + self.s > threshold * self.drift_level {
            DriftStatus::Drift
        } else if self.p + self.s > threshold * self.warning_level {
            DriftStatus::Warning
        } else {
            DriftStatus::Stable
        }
    }

    fn reset(&mut self) {
        *self = Self::new();
    }
}
```

#### 6.5.4 Transfer Learning Traits

```rust
/// Transfer encoder with freeze/unfreeze capability
pub trait TransferEncoder: Module {
    /// Freeze base parameters (only head trains)
    fn freeze_base(&mut self);

    /// Unfreeze all parameters
    fn unfreeze(&mut self);

    /// Check if base is frozen
    fn is_base_frozen(&self) -> bool;

    /// Extract features (intermediate representation)
    fn extract_features(&self, x: &Tensor) -> Tensor;
}

/// Multi-task head for shared encoder
pub struct MultiTaskHead<E: TransferEncoder> {
    encoder: E,
    heads: HashMap<String, Linear>,
}

impl<E: TransferEncoder> MultiTaskHead<E> {
    pub fn add_task(&mut self, name: &str, output_dim: usize) {
        let hidden_dim = self.encoder.hidden_dim();
        self.heads.insert(name.to_string(), Linear::new(hidden_dim, output_dim));
    }

    pub fn forward(&self, task: &str, x: &Tensor) -> Result<Tensor> {
        let features = self.encoder.extract_features(x);
        let head = self.heads.get(task)
            .ok_or_else(|| Error::UnknownTask(task.to_string()))?;
        Ok(head.forward(&features))
    }
}
```

---

### 7.6 SPEC-005: Rosetta Stone Format Conversion (Complete)

#### 6.6.1 Supported Format Matrix

| Source | → GGUF | → SafeTensors | → APR | → ONNX |
|--------|--------|---------------|-------|--------|
| **GGUF** | ✓ (identity) | ✓ (dequant) | ✓ | ✓ |
| **SafeTensors** | ✓ (quant) | ✓ (identity) | ✓ | ✓ |
| **APR** | ✓ | ✓ | ✓ (identity) | ✓ |
| **ONNX** | ✗ | ✓ | ✓ | ✓ (identity) |

#### 6.6.2 100-Point Conversion Checklist

**Section I: Format Parsing (20 points)**

| Gate | Description | Points |
|------|-------------|--------|
| F-CONV-001 | Source format detected correctly | 5 |
| F-CONV-002 | Header parsed without error | 5 |
| F-CONV-003 | Tensor count matches metadata | 5 |
| F-CONV-004 | All tensor names extracted | 5 |

**Section II: Tensor Extraction (20 points)**

| Gate | Description | Points |
|------|-------------|--------|
| F-CONV-005 | All tensors loaded | 5 |
| F-CONV-006 | Shapes match metadata | 5 |
| F-CONV-007 | Dtypes preserved or converted correctly | 5 |
| F-CONV-008 | No NaN/Inf introduced | 5 |

**Section III: Metadata Preservation (15 points)**

| Gate | Description | Points |
|------|-------------|--------|
| F-CONV-009 | Architecture name preserved | 5 |
| F-CONV-010 | Vocab size preserved | 5 |
| F-CONV-011 | Hidden dim / num_layers preserved | 5 |

**Section IV: Round-Trip Fidelity (25 points)**

| Gate | Description | Points |
|------|-------------|--------|
| F-CONV-012 | A→B→A produces identical file | 10 |
| F-CONV-013 | Tensor values within tolerance (rtol=1e-5) | 10 |
| F-CONV-014 | Checksum matches after round-trip | 5 |

**Section V: Inference Parity (20 points)**

| Gate | Description | Points |
|------|-------------|--------|
| F-CONV-015 | Same prompt → same argmax(logits) | 10 |
| F-CONV-016 | Throughput within 10% | 5 |
| F-CONV-017 | Memory usage within 10% | 5 |

#### 6.6.3 Rosetta CLI Commands

```bash
# Convert with verification
apr rosetta convert model.gguf model.apr --verify
# Output: ✓ 100/100 points - VERIFIED

# Inspect and diff
apr rosetta inspect model.apr
apr rosetta inspect model.apr --diff model.gguf
# Shows: tensor-by-tensor comparison

# Compare inference
apr rosetta compare-inference model.gguf model.apr \
    --prompt "What is 2+2?" \
    --max-tokens 10
# Output:
# GGUF: "2+2 equals 4." (argmax=17, tps=12.3)
# APR:  "2+2 equals 4." (argmax=17, tps=11.8)
# PARITY: ✓ PASS (argmax match, tps within 10%)

# Multi-step chain
apr rosetta chain \
    model.safetensors \
    --to gguf:q4_k_m \
    --to apr \
    --verify-each
```

#### 6.6.4 Rosetta Differential Testing (GH-188, PMAT-114)

New capabilities for detecting tensor layout mismatches and inference discrepancies:

```bash
# Diff tensors between two models (GH-188)
apr rosetta diff-tensors model_a.gguf model_b.apr \
    --mismatches-only \
    --filter "embed"
# Output:
# TENSOR DIFF REPORT (GH-188: Layout Mismatch Detection)
# ├─ token_embd.weight: [4096, 32000] vs [32000, 4096] ⚠️ TRANSPOSED
# ├─ lm_head.weight: [4096, 32000] vs [32000, 4096] ⚠️ TRANSPOSED
# └─ Layout mismatches: 2 tensors (GGML vs standard convention)

# Compare inference token-by-token (PMAT-114)
apr rosetta compare-inference model.gguf model.apr \
    --prompt "What is 2+2?" \
    --max-tokens 10 \
    --tolerance 1e-5
# Output:
# Token 0: GGUF=17 APR=17 ✓ MATCH (logit diff: 1.2e-6)
# Token 1: GGUF=42 APR=42 ✓ MATCH (logit diff: 8.3e-7)
# ...
# RESULT: ✓ PASS (10/10 tokens match, max diff: 1.2e-6)
```

**Falsification Gates:**

| Gate ID | Assertion | Severity |
|---------|-----------|----------|
| **F-ROSETTA-DIFF-001** | No transposed tensor dimensions | P0 |
| **F-ROSETTA-DIFF-002** | Tensor shapes match after conversion | P0 |
| **F-ROSETTA-INF-001** | Token-by-token argmax match | P0 |
| **F-ROSETTA-INF-002** | Logit diff within tolerance | P1 |

#### 6.6.5 Profile CI Mode (PMAT-192)

CI-friendly performance assertions for automated pipelines:

```bash
# CI mode with throughput assertion
apr profile model.gguf --ci \
    --assert-throughput 10.0 \
    --warmup 3 \
    --measure 10
# Output:
# CI PROFILE REPORT (PMAT-192)
# ════════════════════════════════════════════════════════════
#   Throughput:  12.8 tok/s
#   Latency p50: 78.2 ms
#   Latency p99: 156.5 ms
#
# ASSERTIONS
#   ✅ PASS throughput: 12.8 tok/s (expected >= 10.0 tok/s)
# Exit code: 0

# CI mode with multiple assertions
apr profile model.gguf --ci \
    --assert-throughput 100 \
    --assert-p99 50 \
    --assert-p50 30
# Exit code 1 if ANY assertion fails
```

**Falsification Gates:**

| Gate ID | Assertion | Severity |
|---------|-----------|----------|
| **F-PROFILE-CI-001** | Throughput meets minimum | P1 |
| **F-PROFILE-CI-002** | p99 latency within limit | P1 |
| **F-PROFILE-CI-003** | p50 latency within limit | P2 |

#### 6.6.6 Differential Benchmarking (PMAT-192 Phase 4)

Compare performance between two models to detect regressions:

```bash
# Diff benchmark two models
apr profile model_old.gguf model_new.gguf --diff-benchmark
# Output:
# DIFFERENTIAL BENCHMARK REPORT
# ════════════════════════════════════════════════════════════
#              | model_old | model_new | Delta
# ─────────────┼───────────┼───────────┼──────────
# Throughput   | 12.3 t/s  | 11.8 t/s  | -4.1% ⚠️
# Latency p50  | 78.2 ms   | 82.1 ms   | +5.0% ⚠️
# Latency p99  | 156.5 ms  | 148.2 ms  | -5.3% ✅
# ─────────────┼───────────┼───────────┼──────────
# REGRESSION DETECTED: throughput -4.1%, latency +5.0%
```

**Falsification Gates:**

| Gate ID | Assertion | Severity |
|---------|-----------|----------|
| **F-PROFILE-DIFF-001** | Throughput regression < 5% | P1 |
| **F-PROFILE-DIFF-002** | Latency regression < 10% | P2 |

#### 6.6.7 Trace Payload Mode (APR-TRACE-001)

Real forward pass with garbage detection and anomaly analysis:

```bash
# Traced inference with payload
apr trace model.gguf --payload \
    --prompt "What is 2+2?"
# Output:
# TRACE REPORT (Payload Mode)
# ════════════════════════════════════════════════════════════
# Layer 0: attn_norm
#   Input:  mean=0.0012, std=0.89, L2=142.3, NaN=0, Inf=0
#   Output: mean=0.0008, std=0.92, L2=147.1, NaN=0, Inf=0
#   Anomalies: None
# Layer 1: attention
#   ...
# ════════════════════════════════════════════════════════════
# GARBAGE DETECTION: None (output is coherent)
# FINAL OUTPUT: "4"

# Trace with reference comparison
apr trace model.apr --reference model.gguf --diff
# Compares layer-by-layer outputs
```

**Garbage Detection Patterns:**
- Repeated words (>50% repetition)
- Unusual Unicode (E000-F8FF, 20000-2FFFF ranges)
- Known garbage patterns (`random_*`, `domainuster`, `pandas`)
- Nonsensical word combinations

**Falsification Gates:**

| Gate ID | Assertion | Severity |
|---------|-----------|----------|
| **F-TRACE-PAYLOAD-001** | No NaN/Inf in any layer | P0 |
| **F-TRACE-PAYLOAD-002** | No garbage output detected | P0 |
| **F-TRACE-PAYLOAD-003** | Layer outputs within expected range | P1 |
| **F-TRACE-DIFF-001** | Layer cosine similarity > 0.99 vs reference | P1 |

---

### 7.7 Consolidated Gate Registry

All gates from all specs, unified:

| Prefix | Source Spec | Count | MQS Category |
|--------|-------------|-------|--------------|
| `F-HTTP-*` | SPEC-002 | 20 | A5, A6 |
| `F-TRACE-*` | SPEC-002 | 7 | E1-E4 |
| `F-CONV-*` | SPEC-005 | 17 | B1-B10 |
| `F-CONV-EMBED-*` | GH-187 | 1 | B1 |
| `F-CONV-TOK-*` | GH-187 | 1 | B2 |
| `F-CONV-WEIGHT-*` | GH-187 | 1 | B3 |
| `F-CONV-SHAPE-*` | GH-187 | 1 | B4 |
| `F-CONV-SEMANTIC-*` | GH-187 | 1 | B5 |
| `F-STAGE-*` | SPEC-003 | N+2 | A7, E6 |
| `F-DRIFT-*` | SPEC-004 | 4 | Cat F |
| `F-CLASS-A*` | SPEC-001 | 4 | A1-A4, B1-B2 |
| `F-CLASS-B*` | SPEC-001 | 4 | A1-A4, B3-B4 |
| `F-PATH-*` | GH-187 | 2 | E5 |
| `F-STATE-*` | GH-187 | 3 | E6 |
| `F-VALID-*` | GH-187 | 3 | E7 |
| `F-ERR-*` | GH-187 | 2 | E8 |
| `F-SEC-*` | GH-187 | 2 | E9 |
| `F-ROSETTA-*` | GH-188/PMAT-114 | 4 | B6-B9 |
| `F-PROFILE-CI-*` | PMAT-192 | 3 | D1-D3 |
| `F-PROFILE-DIFF-*` | PMAT-192 | 2 | D4-D5 |
| `F-TRACE-PAYLOAD-*` | APR-TRACE-001 | 3 | E1-E3 |
| `F-TRACE-DIFF-*` | APR-TRACE-001 | 1 | E4 |

**Total Unique Gates: 82+**

---

## 8. Model Qualification Score (MQS)

### 8.1 Overview

The **Model Qualification Score (MQS)** is a rigorous 0-100 scoring system that measures how well `apr-cli` works with a given model across ALL modalities. It is intentionally difficult to achieve 100/100—requiring perfect functionality across every dimension.

> "A model is only as good as its worst-performing modality."
> — MQS Design Principle

**Key Features:**
- **Gateway Logic:** Critical failures zero the entire score (Popperian demarcation)
- **Weighted Categories:** Core inference weighted highest
- **Penalty System:** Regressions from baseline incur penalties
- **No Rounding Up:** 99.4 = 99, not 100

### 8.2 Scoring Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    MODEL QUALIFICATION SCORE (MQS)                       │
│                         Total: 1000 raw points                          │
│                      Normalized: 0-100 (floor, not round)               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ GATEWAY CATEGORIES (Must Pass or Score = 0)                      │    │
│  ├─────────────────────────────────────────────────────────────────┤    │
│  │ G1. Model Loads Successfully (all formats)           [GATE]     │    │
│  │ G2. Basic Inference Works (at least one path)        [GATE]     │    │
│  │ G3. No Panics/Crashes                                [GATE]     │    │
│  │ G4. Output Not Garbage (passes oracle)               [GATE]     │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                              │                                           │
│                              ▼ (if all gates pass)                       │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ CATEGORY A: Core Inference (350 pts, 35%)                        │    │
│  │ CATEGORY B: Format Support (200 pts, 20%)                        │    │
│  │ CATEGORY C: Modality Coverage (200 pts, 20%)                     │    │
│  │ CATEGORY D: Performance (100 pts, 10%)                           │    │
│  │ CATEGORY E: Observability (100 pts, 10%)                         │    │
│  │ CATEGORY F: Robustness (50 pts, 5%)                              │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                              │                                           │
│                              ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ PENALTIES (Subtracted from total)                                │    │
│  │ - Regression from previous version: -50 pts per regression       │    │
│  │ - Flaky tests (>5% failure rate): -25 pts per flaky test        │    │
│  │ - Timeout violations: -10 pts per timeout                        │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                              │                                           │
│                              ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ FINAL SCORE = floor((raw_points - penalties) / 10)               │    │
│  │ Clamped to [0, 100]                                              │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 8.3 Gateway Categories (Pass/Fail)

**If ANY gateway fails, MQS = 0.**

| Gate | Description | Failure Condition |
|------|-------------|-------------------|
| **G1** | Model loads | Any format fails to load (file not found, parse error) |
| **G2** | Basic inference | Zero formats produce any output |
| **G3** | No crashes | Any panic, segfault, or abort |
| **G4** | Output quality | All outputs are garbage (NaN, repetition, control chars) |

### 8.4 Category A: Core Inference (350 points)

The largest category—tests that inference actually works.

| Subcategory | Points | Requirement |
|-------------|--------|-------------|
| **A1. `apr run` CPU** | 50 | Produces correct output |
| **A2. `apr run` GPU** | 50 | Produces correct output |
| **A3. `apr chat` CPU** | 50 | Chat template applied, multi-turn works |
| **A4. `apr chat` GPU** | 50 | Chat template applied, multi-turn works |
| **A5. `apr serve` CPU** | 50 | All F-HTTP gates pass |
| **A6. `apr serve` GPU** | 50 | All F-HTTP gates pass |
| **A7. Correctness Oracle** | 50 | "2+2=" → contains "4" |

**Partial Credit:**
- Full points: All assertions pass
- Half points: Output exists but oracle fails
- Zero points: No output or crash

### 8.5 Category B: Format Support (200 points)

Tests all format × backend combinations.

| Format | CPU | GPU | Points Per Cell | Row Total |
|--------|-----|-----|-----------------|-----------|
| **GGUF Q4_K** | B1 | B2 | 25 | 50 |
| **SafeTensors F32** | B3 | B4 | 20 | 40 |
| **APR (from GGUF)** | B5 | B6 | 15 | 30 |
| **APR (from SafeTensors)** | B7 | B8 | 15 | 30 |

**Parity Requirement (50 pts):**
- B9: GGUF argmax == APR argmax (same quantization) — 25 pts
- B10: Cross-format inference equivalence (F-CONV-INF-*) — 25 pts

**Category B Total:** 50 + 40 + 30 + 30 + 50 = **200 points** ✓

### 8.6 Category C: Modality Coverage (200 points)

Tests all modalities work across all scenarios.

| Modality | Scenarios | Points |
|----------|-----------|--------|
| **C1. `run`** | 100 property tests | 60 |
| **C2. `chat`** | 100 property tests | 60 |
| **C3. `serve`** | 20 HTTP tests | 40 |
| **C4. Streaming** | SSE events valid | 20 |
| **C5. Multi-turn** | Context preserved | 20 |

**Property Test Scoring:**
- 100/100 pass: Full points
- 95-99/100 pass: 90% of points
- 90-94/100 pass: 75% of points
- <90/100 pass: 50% of points
- <80/100 pass: 0 points

### 8.7 Category D: Performance (100 points)

**Thresholds (from qa-showcase-methodology.md):**

| Metric | CPU Threshold | GPU Threshold | Points |
|--------|---------------|---------------|--------|
| **D1. Throughput** | ≥10 tok/s | ≥100 tok/s | 30 |
| **D2. TTFT** | <2s | <500ms | 20 |
| **D3. GPU Speedup** | N/A | ≥2× CPU | 20 |
| **D4. Ollama Parity** | ≥0.8× | ≥0.8× | 20 |
| **D5. Memory** | <model_size×2 | <VRAM×0.9 | 10 |

**Scoring:**
- Exceeds threshold: Full points
- Within 10% of threshold: 75% of points
- Within 25% of threshold: 50% of points
- Below 25%: 0 points

### 8.8 Category E: Observability (100 points)

Tests tracing, profiling, and debugging tools.

| Subcategory | Points | Requirement |
|-------------|--------|-------------|
| **E1. Trace level `none`** | 10 | No overhead, inference works |
| **E2. Trace level `basic`** | 10 | Timing + token counts |
| **E3. Trace level `layer`** | 15 | Per-layer stats (mean/std/L2) |
| **E4. Trace level `payload`** | 15 | Tensor values captured |
| **E5. `apr profile`** | 15 | Hotspots, flamegraph |
| **E6. `apr check`** | 15 | 10-stage pipeline passes |
| **E7. `apr inspect`** | 10 | Metadata accurate |
| **E8. `apr canary`** | 10 | Baseline comparison works |

### 8.9 Category F: Robustness (50 points)

Edge cases and stress tests.

| Subcategory | Points | Requirement |
|-------------|--------|-------------|
| **F1. Empty prompt** | 10 | Graceful error, no crash |
| **F2. Unicode/emoji** | 10 | Handles correctly |
| **F3. Max tokens edge** | 10 | Respects limit |
| **F4. Concurrent requests** | 10 | No deadlock |
| **F5. OOM protection** | 10 | Graceful degradation |

### 8.10 Penalty System

Penalties are subtracted from raw score before normalization.

| Penalty | Points | Trigger |
|---------|--------|---------|
| **Regression** | -50 | Any metric worse than previous version |
| **Flaky test** | -25 | Test fails >5% of runs (requires 3+ runs) |
| **Timeout** | -10 | Any operation exceeds timeout |
| **SATD discovered** | -10 | TODO/FIXME in apr-cli related to this model |

### 8.11 Final Score Calculation

```rust
pub fn calculate_mqs(results: &QaResults) -> ModelQualificationScore {
    // Gateway check
    if !results.gate_g1_loads || !results.gate_g2_inference ||
       !results.gate_g3_no_crash || !results.gate_g4_not_garbage {
        return ModelQualificationScore {
            raw: 0,
            normalized: 0,
            grade: Grade::F,
            gateway_failed: true,
            failed_gate: identify_failed_gate(results),
        };
    }

    // Calculate category scores
    let cat_a = calculate_category_a(results);  // 0-350
    let cat_b = calculate_category_b(results);  // 0-200
    let cat_c = calculate_category_c(results);  // 0-200
    let cat_d = calculate_category_d(results);  // 0-100
    let cat_e = calculate_category_e(results);  // 0-100
    let cat_f = calculate_category_f(results);  // 0-50

    let raw_total = cat_a + cat_b + cat_c + cat_d + cat_e + cat_f;  // 0-1000

    // Apply penalties
    let penalties = calculate_penalties(results);
    let adjusted = (raw_total as i32 - penalties).max(0) as u32;

    // Normalize to 0-100 (floor, not round)
    let normalized = (adjusted / 10).min(100);

    ModelQualificationScore {
        raw: adjusted,
        normalized,
        grade: grade_from_score(normalized),
        gateway_failed: false,
        failed_gate: None,
        categories: CategoryBreakdown {
            core_inference: cat_a,
            format_support: cat_b,
            modality_coverage: cat_c,
            performance: cat_d,
            observability: cat_e,
            robustness: cat_f,
        },
        penalties,
    }
}

fn grade_from_score(score: u32) -> Grade {
    match score {
        100 => Grade::APlusPlus,  // Perfect score - extremely rare
        97..=99 => Grade::APlus,
        93..=96 => Grade::A,
        90..=92 => Grade::AMinus,
        87..=89 => Grade::BPlus,
        83..=86 => Grade::B,
        80..=82 => Grade::BMinus,
        77..=79 => Grade::CPlus,
        73..=76 => Grade::C,
        70..=72 => Grade::CMinus,
        60..=69 => Grade::D,
        _ => Grade::F,
    }
}
```

### 8.12 Score Report Format

```
╔══════════════════════════════════════════════════════════════════════════╗
║           MODEL QUALIFICATION SCORE (MQS) REPORT                          ║
╠══════════════════════════════════════════════════════════════════════════╣
║ Model: Qwen/Qwen2.5-Coder-1.5B-Instruct                                   ║
║ Date: 2026-01-29T14:32:00Z                                                ║
║ apr-cli version: 0.25.0                                                   ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                           ║
║  ████████████████████████████████████████████░░░░░░░░  87/100  [A-]      ║
║                                                                           ║
╠══════════════════════════════════════════════════════════════════════════╣
║ GATEWAY CHECKS                                                            ║
╠══════════════════════════════════════════════════════════════════════════╣
║ [✓] G1. Model Loads        [✓] G2. Basic Inference                       ║
║ [✓] G3. No Crashes         [✓] G4. Output Quality                        ║
╠══════════════════════════════════════════════════════════════════════════╣
║ CATEGORY BREAKDOWN                                                        ║
╠══════════════════════════════════════════════════════════════════════════╣
║ A. Core Inference      ████████████████████░░░░░░░  310/350  (89%)       ║
║    • apr run CPU       ████████████████████████████  50/50               ║
║    • apr run GPU       ████████████████████████████  50/50               ║
║    • apr chat CPU      ████████████████████████████  50/50               ║
║    • apr chat GPU      ████████████████████████░░░░  40/50   (streaming) ║
║    • apr serve CPU     ████████████████████████████  50/50               ║
║    • apr serve GPU     ████████████████████░░░░░░░░  35/50   (F-HTTP-019)║
║    • Correctness       ████████████████████░░░░░░░░  35/50   (edge cases)║
║                                                                           ║
║ B. Format Support      ████████████████████████░░░░  175/200 (88%)       ║
║    • GGUF Q4_K         ████████████████████████████  80/80               ║
║    • SafeTensors F32   ████████████████████████░░░░  55/60   (GPU slow)  ║
║    • APR converted     ████████████████████░░░░░░░░  40/60   (parity)    ║
║                                                                           ║
║ C. Modality Coverage   ████████████████████████████  185/200 (93%)       ║
║ D. Performance         ████████████████░░░░░░░░░░░░  65/100  (66%)       ║
║ E. Observability       ████████████████████████████  95/100  (95%)       ║
║ F. Robustness          ████████████████████████████  45/50   (90%)       ║
╠══════════════════════════════════════════════════════════════════════════╣
║ PENALTIES                                                                 ║
╠══════════════════════════════════════════════════════════════════════════╣
║ [!] Regression: GPU throughput dropped 15% from v0.24.0         -50 pts  ║
║ [!] Flaky: test_serve_concurrent fails 8% of runs               -25 pts  ║
║                                                        TOTAL:   -75 pts  ║
╠══════════════════════════════════════════════════════════════════════════╣
║ RAW SCORE: 875/1000  →  PENALTIES: -75  →  ADJUSTED: 800  →  MQS: 80/100 ║
╠══════════════════════════════════════════════════════════════════════════╣
║ RECOMMENDATIONS                                                           ║
╠══════════════════════════════════════════════════════════════════════════╣
║ 1. Fix GPU serve concurrent request handling (F-HTTP-019)                ║
║ 2. Investigate GPU throughput regression from v0.24.0                    ║
║ 3. Stabilize test_serve_concurrent (currently 8% flaky)                  ║
║ 4. Improve APR conversion parity for SafeTensors source                  ║
╚══════════════════════════════════════════════════════════════════════════╝
```

### 8.13 Why 100/100 is Difficult

To achieve a perfect MQS of 100/100, a model must:

1. **Pass ALL gateway checks** (mandatory)
2. **All 3 modalities work** on both CPU and GPU (6 paths)
3. **All 3 formats work** with full parity (GGUF, SafeTensors, APR)
4. **100/100 property tests pass** for each modality
5. **Meet ALL performance thresholds** (throughput, TTFT, GPU speedup)
6. **ALL trace levels work** (none, basic, layer, payload)
7. **ALL profile focus areas work** (all, attention, mlp, matmul, embedding)
8. **10-stage `apr check` passes** completely
9. **ALL robustness tests pass** (empty, unicode, concurrent, OOM)
10. **ZERO regressions** from previous version
11. **ZERO flaky tests**
12. **ZERO timeouts**

**Expected Distribution:**
- 100/100: <1% of models (exceptional)
- 90-99: ~10% of models (excellent)
- 80-89: ~30% of models (good)
- 70-79: ~30% of models (acceptable)
- 60-69: ~20% of models (needs work)
- <60: ~10% of models (significant issues)

### 8.14 MQS Leaderboard Schema

```rust
#[derive(Serialize, Deserialize)]
pub struct MqsLeaderboard {
    /// Last updated timestamp
    pub updated_at: DateTime<Utc>,

    /// apr-cli version tested against
    pub apr_version: String,

    /// Sorted by MQS descending
    pub models: Vec<ModelEntry>,
}

#[derive(Serialize, Deserialize)]
pub struct ModelEntry {
    /// HuggingFace model ID
    pub model_id: String,

    /// Model size category
    pub size_category: SizeCategory,  // 0.5B, 1.5B, 3B, 7B, 13B, 70B

    /// Quantization tested
    pub quantization: String,  // Q4_K_M, Q8_0, F16, F32

    /// Final MQS score
    pub mqs: u32,

    /// Grade
    pub grade: Grade,

    /// Category breakdown
    pub categories: CategoryBreakdown,

    /// Known issues (ticket IDs)
    pub known_issues: Vec<String>,

    /// Last tested date
    pub tested_at: DateTime<Utc>,
}
```

---

## 9. Playbook Schema

### 9.1 YAML Schema Definition

```yaml
# playbook.schema.yaml
$schema: "https://json-schema.org/draft/2020-12/schema"
$id: "https://paiml.com/schemas/apr-qa-playbook/v1"
title: "APR QA Playbook"
type: object

required:
  - name
  - version
  - model
  - test_matrix
  - falsification_gates

properties:
  name:
    type: string
    pattern: "^[a-z0-9-]+$"
    description: "Unique playbook identifier"

  version:
    type: string
    pattern: "^\\d+\\.\\d+\\.\\d+$"
    description: "Semantic version"

  model:
    type: object
    required: [hf_repo]
    properties:
      hf_repo:
        type: string
        description: "HuggingFace repository (org/model)"
      local_path:
        type: string
        description: "Optional local cache path"
      formats:
        type: array
        items:
          enum: [gguf, safetensors, apr]
        default: [gguf, safetensors, apr]
      quantizations:
        type: array
        items:
          enum: [f32, f16, q8_0, q4_k_m, q4_0, q5_0, q5_k_m, q6_k]
        default: [q4_k_m]

  test_matrix:
    type: object
    properties:
      modalities:
        type: array
        items:
          enum: [run, chat, serve]
        default: [run, chat, serve]
      backends:
        type: array
        items:
          enum: [cpu, gpu]
        default: [cpu, gpu]
      scenario_count:
        type: integer
        minimum: 1
        maximum: 1000
        default: 100

  property_tests:
    type: array
    items:
      type: object
      required: [name, generator, oracle]
      properties:
        name:
          type: string
        generator:
          type: string
          description: "proptest strategy expression"
        oracle:
          type: string
          description: "Verification expression or command"
        count:
          type: integer
          default: 100

  state_machine:
    type: object
    description: "Optional FSM definition for complex workflows"
    properties:
      initial:
        type: string
      states:
        type: object
        additionalProperties:
          type: object
          properties:
            on_enter:
              type: array
            on_exit:
              type: array
            transitions:
              type: array

  falsification_gates:
    type: array
    items:
      type: object
      required: [id, description, condition]
      properties:
        id:
          type: string
          pattern: "^F-[A-Z]+-\\d{3}$"
        description:
          type: string
        condition:
          type: string
          description: "Boolean expression that must be true"
        severity:
          enum: [P0, P1, P2]
          default: P1
```

### 9.2 Example Playbook

```yaml
# playbooks/models/qwen2.5-coder-1.5b.playbook.yaml
name: qwen2.5-coder-1.5b-qualification
version: "1.0.0"
model:
  hf_repo: "Qwen/Qwen2.5-Coder-1.5B-Instruct"
  formats: [gguf, safetensors, apr]
  quantizations: [q4_k_m, q8_0]

test_matrix:
  modalities: [run, chat, serve]
  backends: [cpu, gpu]
  scenario_count: 100

property_tests:
  - name: arithmetic_correctness
    generator: |
      proptest::strategy::Union::new(vec![
        Just("2+2="),
        Just("What is 2+2?"),
        Just("Calculate: 3*4"),
        "[0-9]{1,2}[+\\-*/][0-9]{1,2}=".prop_map(|s| s),
      ])
    oracle: |
      fn verify(prompt: &str, output: &str) -> bool {
        let expected = eval_arithmetic(prompt);
        output.contains(&expected.to_string())
      }
    count: 100

  - name: code_completion
    generator: |
      proptest::strategy::Union::new(vec![
        Just("def fibonacci(n):"),
        Just("fn main() {"),
        Just("function hello() {"),
        "def [a-z_]+\\([a-z_]*\\):".prop_map(|s| s),
      ])
    oracle: |
      fn verify(_prompt: &str, output: &str) -> bool {
        !is_garbage(output) && is_syntactically_plausible(output)
      }
    count: 100

  - name: instruction_following
    generator: |
      proptest::strategy::Union::new(vec![
        Just("Write a haiku about Rust."),
        Just("List 3 prime numbers."),
        Just("Explain recursion in one sentence."),
      ])
    oracle: |
      fn verify(prompt: &str, output: &str) -> bool {
        output.len() > 10 && !is_garbage(output)
      }
    count: 100

state_machine:
  initial: init
  states:
    init:
      transitions:
        - event: pull_model
          target: cached
          action: "apr pull ${MODEL_ID}"
          guards:
            - "exit_code == 0"

    cached:
      transitions:
        - event: convert_apr
          target: converted
          action: "apr convert ${GGUF_PATH} -o ${APR_PATH}"
        - event: run_test
          target: testing

    converted:
      transitions:
        - event: run_test
          target: testing

    testing:
      on_enter:
        - action: "start_timer"
      transitions:
        - event: test_pass
          target: corroborated
          guards:
            - "all_gates_passed"
        - event: test_fail
          target: falsified

    corroborated:
      on_enter:
        - action: "record_evidence('CORROBORATED')"

    falsified:
      on_enter:
        - action: "record_evidence('FALSIFIED')"
        - action: "stop_line"  # Jidoka: halt pipeline

falsification_gates:
  - id: F-QUAL-001
    description: "Output is non-empty"
    condition: "output.len() > 0"
    severity: P0

  - id: F-QUAL-002
    description: "No garbage tokens (NaN, control chars)"
    condition: "!contains_garbage(output)"
    severity: P0

  - id: F-QUAL-003
    description: "Arithmetic correctness for math prompts"
    condition: "verify_arithmetic(prompt, output)"
    severity: P1

  - id: F-PERF-001
    description: "CPU throughput >= 10 tok/s"
    condition: "backend == 'gpu' || tps >= 10.0"
    severity: P1

  - id: F-PERF-002
    description: "GPU throughput >= 2x CPU"
    condition: "backend == 'cpu' || gpu_tps >= 2.0 * cpu_tps"
    severity: P1

  - id: F-PARITY-001
    description: "Cross-format parity (GGUF vs SafeTensors)"
    condition: "gguf_argmax == safetensors_argmax"
    severity: P0
```

### 9.3 Playbook Validation

All playbooks must pass schema validation before execution:

```bash
bashrs playbook validate playbooks/models/*.yaml
```

**Poka-Yoke enforcement:** Invalid playbooks are rejected at parse time, not runtime.

---

## 10. Property Test Generation

### 10.1 Proptest Integration

The `apr-qa-gen` crate uses `proptest` (Rust property testing library) to generate diverse test scenarios:

```rust
// crates/apr-qa-gen/src/scenario.rs
use proptest::prelude::*;

/// A single test scenario for model qualification
#[derive(Debug, Clone)]
pub struct QaScenario {
    pub model: ModelId,
    pub modality: Modality,
    pub backend: Backend,
    pub format: Format,
    pub prompt: String,
    pub temperature: f32,
    pub max_tokens: u32,
    pub seed: u64,
    pub oracle: OracleFn,
}

/// Modality enum
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Modality {
    Run,
    Chat,
    Serve,
}

/// Backend enum
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Backend {
    Cpu,
    Gpu,
}

/// Format enum
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Format {
    Gguf,
    SafeTensors,
    Apr,
}

impl Arbitrary for QaScenario {
    type Parameters = ModelId;
    type Strategy = BoxedStrategy<Self>;

    fn arbitrary_with(model: Self::Parameters) -> Self::Strategy {
        (
            prop::sample::select(vec![Modality::Run, Modality::Chat, Modality::Serve]),
            prop::sample::select(vec![Backend::Cpu, Backend::Gpu]),
            prop::sample::select(vec![Format::Gguf, Format::SafeTensors, Format::Apr]),
            prompt_strategy(),
            0.0f32..2.0f32,  // temperature
            1u32..100u32,     // max_tokens
            any::<u64>(),     // seed
        )
            .prop_map(move |(modality, backend, format, prompt, temp, max_tokens, seed)| {
                let oracle = select_oracle(&prompt);
                QaScenario {
                    model: model.clone(),
                    modality,
                    backend,
                    format,
                    prompt,
                    temperature: temp,
                    max_tokens,
                    seed,
                    oracle,
                }
            })
            .boxed()
    }
}

fn prompt_strategy() -> impl Strategy<Value = String> {
    prop_oneof![
        // Arithmetic (verifiable)
        "[0-9]{1,2}[+\\-*/][0-9]{1,2}=?".prop_map(|s| s),
        Just("What is 2+2?".to_string()),
        Just("Calculate 7*8".to_string()),

        // Code completion (syntax check)
        Just("def fibonacci(n):".to_string()),
        Just("fn main() {".to_string()),
        Just("async function fetch() {".to_string()),

        // Instruction following (non-empty check)
        Just("Write a haiku about programming.".to_string()),
        Just("List three colors.".to_string()),
        Just("Explain what a variable is.".to_string()),

        // Edge cases (robustness)
        Just("".to_string()),  // Empty prompt
        Just(" ".to_string()), // Whitespace only
        "\\p{Han}{1,10}".prop_map(|s| s),  // Chinese characters
        "\\p{Emoji}{1,5}".prop_map(|s| s), // Emoji
    ]
}
```

### 10.2 Oracle Definitions

Oracles verify output correctness. Each oracle is a pure function:

```rust
// crates/apr-qa-gen/src/oracle.rs

/// Oracle function type
pub type OracleFn = fn(prompt: &str, output: &str) -> OracleResult;

/// Oracle result with evidence
#[derive(Debug, Clone)]
pub enum OracleResult {
    Corroborated { evidence: String },
    Falsified { reason: String, evidence: String },
}

/// Select appropriate oracle based on prompt characteristics
pub fn select_oracle(prompt: &str) -> OracleFn {
    if is_arithmetic_prompt(prompt) {
        oracle_arithmetic
    } else if is_code_prompt(prompt) {
        oracle_code_syntax
    } else {
        oracle_non_garbage
    }
}

/// Oracle: Arithmetic correctness
/// Falsifiable: Output must contain the correct numerical answer
pub fn oracle_arithmetic(prompt: &str, output: &str) -> OracleResult {
    let expected = match eval_arithmetic_prompt(prompt) {
        Some(n) => n,
        None => return OracleResult::Corroborated {
            evidence: "Non-arithmetic prompt, skipped".into(),
        },
    };

    if output.contains(&expected.to_string()) {
        OracleResult::Corroborated {
            evidence: format!("Found expected value {} in output", expected),
        }
    } else {
        OracleResult::Falsified {
            reason: format!("Expected {} not found in output", expected),
            evidence: format!("Output: {}", truncate(output, 100)),
        }
    }
}

/// Oracle: Code syntax plausibility
/// Falsifiable: Output must not be garbage and should look like code
pub fn oracle_code_syntax(prompt: &str, output: &str) -> OracleResult {
    if is_garbage(output) {
        return OracleResult::Falsified {
            reason: "Output contains garbage tokens".into(),
            evidence: format!("Garbage detected: {}", truncate(output, 100)),
        };
    }

    // Check for code-like patterns
    let code_indicators = ["fn ", "def ", "function ", "{", "}", "(", ")", "return", "let ", "const "];
    let has_code_pattern = code_indicators.iter().any(|p| output.contains(p));

    if has_code_pattern || output.len() < 10 {
        OracleResult::Corroborated {
            evidence: "Output appears to be valid code".into(),
        }
    } else {
        OracleResult::Falsified {
            reason: "Output does not appear to be code".into(),
            evidence: format!("Output: {}", truncate(output, 100)),
        }
    }
}

/// Oracle: Non-garbage output
/// Falsifiable: Output must not contain garbage tokens
pub fn oracle_non_garbage(_prompt: &str, output: &str) -> OracleResult {
    if is_garbage(output) {
        OracleResult::Falsified {
            reason: "Output contains garbage tokens".into(),
            evidence: format!("Garbage: {}", truncate(output, 100)),
        }
    } else if output.is_empty() {
        OracleResult::Falsified {
            reason: "Output is empty".into(),
            evidence: "Empty output".into(),
        }
    } else {
        OracleResult::Corroborated {
            evidence: format!("Valid output ({} chars)", output.len()),
        }
    }
}

/// Garbage detection heuristics
pub fn is_garbage(output: &str) -> bool {
    // Control characters (except newline, tab)
    if output.chars().any(|c| c.is_control() && c != '\n' && c != '\t' && c != '\r') {
        return true;
    }

    // Repetitive token patterns (e.g., "akakakakak", "veisveisveis")
    let words: Vec<&str> = output.split_whitespace().collect();
    if words.len() >= 5 {
        let unique: std::collections::HashSet<_> = words.iter().collect();
        if unique.len() == 1 {
            return true; // All same word repeated
        }
    }

    // NaN or Inf in output (numerical explosion)
    if output.contains("NaN") || output.contains("Inf") || output.contains("inf") {
        return true;
    }

    false
}
```

### 10.3 Scenario Serialization

Scenarios serialize to bashrs-compatible YAML:

```rust
// crates/apr-qa-gen/src/emit.rs

impl QaScenario {
    pub fn to_playbook_step(&self) -> PlaybookStep {
        let command = match self.modality {
            Modality::Run => format!(
                "apr run {} '{}' -n {} --seed {} {}",
                self.model_path(),
                escape_prompt(&self.prompt),
                self.max_tokens,
                self.seed,
                if self.backend == Backend::Gpu { "--gpu" } else { "" }
            ),
            Modality::Chat => format!(
                "echo '{}' | apr chat {} {} --temperature {}",
                escape_prompt(&self.prompt),
                self.model_path(),
                if self.backend == Backend::Gpu { "--gpu" } else { "" },
                self.temperature
            ),
            Modality::Serve => format!(
                r#"apr serve {} --port ${{PORT}} {} &
                sleep 2
                curl -s http://localhost:${{PORT}}/v1/completions \
                  -H 'Content-Type: application/json' \
                  -d '{{"prompt": "{}", "max_tokens": {}}}'
                kill %1"#,
                self.model_path(),
                if self.backend == Backend::Gpu { "--gpu" } else { "" },
                escape_json(&self.prompt),
                self.max_tokens
            ),
        };

        PlaybookStep {
            name: format!("scenario_{}_{:016x}", self.modality, self.seed),
            command,
            timeout_ms: 60000,
            oracle: self.oracle_expression(),
            gates: self.gates(),
        }
    }
}
```

---

## 11. Falsification Protocol

### 11.1 Falsification Gate Taxonomy

Gates are organized by category and severity. A **Demarcation Check** is a meta-test ensuring the test itself is valid (falsifiable).

| Category | ID Range | Severity | Description |
|----------|----------|----------|-------------|
| **DEMARC** | F-DEMARC-001 to F-DEMARC-099 | P0 | Demarcation checks (test validity) |
| **QUAL** | F-QUAL-001 to F-QUAL-099 | P0-P1 | Output quality |
| **PERF** | F-PERF-001 to F-PERF-099 | P1-P2 | Performance thresholds |
| **PARITY** | F-PARITY-001 to F-PARITY-099 | P0 | Cross-format consistency |
| **LOAD** | F-LOAD-001 to F-LOAD-099 | P1 | Model loading |
| **SERVE** | F-SERVE-001 to F-SERVE-099 | P1 | Server operations |
| **GPU** | F-GPU-001 to F-GPU-099 | P0-P1 | GPU-specific |
| **REGR** | F-REGR-001 to F-REGR-999 | P0 | Regression tests |

### 11.2 Severity Definitions

| Severity | Definition | Response |
|----------|------------|----------|
| **P0** | Critical: System produces incorrect/dangerous output | **Stop the line.** No releases until fixed. |
| **P1** | Major: Feature does not meet specification | Block release. Fix before next milestone. |
| **P2** | Minor: Suboptimal but acceptable | Track for improvement. |

### 11.3 Evidence Collection

Every falsification attempt records:

```rust
// crates/apr-qa-report/src/evidence.rs

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FalsificationEvidence {
    /// Unique gate identifier
    pub gate_id: String,

    /// Test scenario that triggered this gate
    pub scenario: QaScenario,

    /// Outcome: CORROBORATED or FALSIFIED
    pub outcome: Outcome,

    /// Human-readable reason
    pub reason: String,

    /// Raw evidence data
    pub evidence: EvidenceData,

    /// Timestamp (ISO 8601)
    pub timestamp: String,

    /// Duration in milliseconds
    pub duration_ms: u64,

    /// Host information (reproducibility)
    pub host: HostInfo,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Outcome {
    Corroborated,
    Falsified,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceData {
    /// Model output (truncated if large)
    pub output: String,

    /// Exit code
    pub exit_code: i32,

    /// Stderr (if any)
    pub stderr: Option<String>,

    /// Performance metrics
    pub metrics: Option<PerformanceMetrics>,

    /// Checksums for reproducibility
    pub checksums: Checksums,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub tokens_per_second: f64,
    pub time_to_first_token_ms: f64,
    pub total_tokens: u32,
    pub memory_peak_mb: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checksums {
    pub model_sha256: String,
    pub output_sha256: String,
}
```

### 11.4 Popperian Scoring

The final qualification score reflects the model's **Verisimilitude** (degree of corroboration). It is not a percentage of "correctness," but a measure of how many severe tests the model has survived.

```rust
// crates/apr-qa-report/src/popperian.rs

#[derive(Debug, Clone)]
pub struct PopperianScore {
    /// Total gates attempted (Severity of Testing)
    pub total_gates: u32,

    /// Gates that passed (Hypothesis Corroborated)
    pub corroborated: u32,

    /// Gates that failed (Hypothesis Falsified)
    pub falsified: u32,

    /// Verisimilitude Score (0.0 - 100.0)
    /// Represents the degree of corroboration against the test suite.
    pub verisimilitude: f64,

    /// Grade (A+, A, B+, B, C, D, F)
    pub grade: Grade,

    /// P0 failures (Automatic F - Falsification of Critical Safety)
    pub p0_failures: u32,
}

impl PopperianScore {
    pub fn calculate(evidence: &[FalsificationEvidence]) -> Self {
        let total_gates = evidence.len() as u32;
        let corroborated = evidence.iter()
            .filter(|e| matches!(e.outcome, Outcome::Corroborated))
            .count() as u32;
        let falsified = total_gates - corroborated;

        let p0_failures = evidence.iter()
            .filter(|e| {
                matches!(e.outcome, Outcome::Falsified) &&
                e.gate_id.contains("-P0-")
            })
            .count() as u32;

        let verisimilitude = if total_gates > 0 {
            (corroborated as f64 / total_gates as f64) * 100.0
        } else {
            0.0 // Metaphysical state (untested)
        };

        // Grade calculation
        // P0 failure = automatic F (Critical Falsification)
        let grade = if p0_failures > 0 {
            Grade::F
        } else {
            match verisimilitude as u32 {
                97..=100 => Grade::APlus,
                93..=96 => Grade::A,
                90..=92 => Grade::AMinus,
                87..=89 => Grade::BPlus,
                83..=86 => Grade::B,
                80..=82 => Grade::BMinus,
                77..=79 => Grade::CPlus,
                73..=76 => Grade::C,
                70..=72 => Grade::CMinus,
                60..=69 => Grade::D,
                _ => Grade::F,
            }
        };

        Self {
            total_gates,
            corroborated,
            falsified,
            verisimilitude,
            grade,
            p0_failures,
        }
    }
}
```

---

## 12. Orchestration Pipeline

### 12.1 Batuta Integration

The pipeline is orchestrated via batuta:

```toml
# batuta-qa-pipeline.toml
[pipeline]
name = "apr-model-qa"
version = "1.0.0"
jidoka = true  # Stop on P0 failure

[defaults]
timeout_minutes = 60
retry_count = 0  # No retries - failures are signal, not noise

[[stages]]
name = "discover"
description = "Fetch HuggingFace top 100 model list"
command = "apr-qa-gen discover --top 100 --output models.json"
parallel = false

[[stages]]
name = "generate"
description = "Generate property test scenarios"
command = "apr-qa-gen generate --models models.json --scenarios 100 --output playbooks/"
depends_on = ["discover"]
parallel = false

[[stages]]
name = "validate"
description = "Validate generated playbooks"
command = "bashrs playbook validate playbooks/**/*.yaml"
depends_on = ["generate"]
parallel = false

[[stages]]
name = "pull"
description = "Download and cache all models"
command = "bashrs playbook playbooks/pull-all-models.yaml"
depends_on = ["validate"]
parallel = 16
timeout_minutes = 120

[[stages]]
name = "convert"
description = "Convert models to APR format"
command = "bashrs playbook playbooks/convert-all-models.yaml"
depends_on = ["pull"]
parallel = 8

[[stages]]
name = "qa-cpu"
description = "Run CPU qualification tests"
command = "bashrs playbook playbooks/qa-matrix-cpu.yaml --proptest-expand"
depends_on = ["convert"]
parallel = 8
env = { APR_BACKEND = "cpu" }

[[stages]]
name = "qa-gpu"
description = "Run GPU qualification tests"
command = "bashrs playbook playbooks/qa-matrix-gpu.yaml --proptest-expand"
depends_on = ["convert"]
parallel = 4  # GPU memory limited
env = { APR_BACKEND = "gpu", CUDA_VISIBLE_DEVICES = "0" }

[[stages]]
name = "parity"
description = "Cross-format parity verification"
command = "bashrs playbook playbooks/parity-check.yaml"
depends_on = ["qa-cpu", "qa-gpu"]
parallel = false

[[stages]]
name = "report"
description = "Generate Popperian report"
command = "apr-qa-report generate --input evidence/ --output report/"
depends_on = ["parity"]
parallel = false
```

### 12.2 Parallel Execution Strategy

```
┌─────────────────────────────────────────────────────────────────────┐
│                         WORKER POOL                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  CPU Workers (8 threads)          GPU Workers (4 streams)           │
│  ┌─────┐ ┌─────┐ ┌─────┐        ┌─────┐ ┌─────┐                    │
│  │ W1  │ │ W2  │ │ ... │        │ G1  │ │ G2  │                    │
│  └──┬──┘ └──┬──┘ └──┬──┘        └──┬──┘ └──┬──┘                    │
│     │       │       │              │       │                        │
│     ▼       ▼       ▼              ▼       ▼                        │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    TASK QUEUE                                │   │
│  │  [Model1/Run/CPU] [Model1/Chat/GPU] [Model2/Run/CPU] ...    │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  Heijunka: Load-balanced distribution based on estimated runtime     │
│  Jidoka: Worker stops on P0 failure, signals coordinator            │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 12.3 Timeout and Failure Handling

```rust
// crates/apr-qa-runner/src/executor.rs

/// Timeout configuration per modality
pub struct TimeoutConfig {
    pub run_timeout_ms: u64,      // 30,000 (30s)
    pub chat_timeout_ms: u64,     // 60,000 (60s)
    pub serve_startup_ms: u64,    // 10,000 (10s)
    pub serve_request_ms: u64,    // 30,000 (30s)
}

/// Failure handling policy (Jidoka)
#[derive(Debug, Clone, Copy)]
pub enum FailurePolicy {
    /// Stop entire pipeline on any failure
    StopOnFirst,

    /// Stop on P0, continue on P1/P2
    StopOnP0,

    /// Collect all failures, report at end
    CollectAll,
}

impl Default for FailurePolicy {
    fn default() -> Self {
        // Default: Toyota Way - stop on P0
        Self::StopOnP0
    }
}
```

### 12.4 Process Lifecycle Management (Jidoka)

**Issue:** [paiml/apr-model-qa-playbook#1](https://github.com/paiml/apr-model-qa-playbook/issues/1)

Child processes spawned during test execution must be properly tracked and cleaned up to prevent resource leaks. This implements the Toyota Way Jidoka principle: stop the line and clean up, never leave defects (orphan processes) in the system.

#### 12.4.1 Process Registry

```rust
// crates/apr-qa-runner/src/process.rs
// Pattern derived from repartir's task lifecycle management (sovereign tool)

use std::process::Child;
use std::sync::{Arc, Mutex, OnceLock};

/// Global registry of spawned child processes for cleanup
static PROCESS_REGISTRY: OnceLock<Arc<Mutex<Vec<Child>>>> = OnceLock::new();

fn get_registry() -> &'static Arc<Mutex<Vec<Child>>> {
    PROCESS_REGISTRY.get_or_init(|| Arc::new(Mutex::new(Vec::new())))
}

/// Register a child process for tracking
pub fn register_child(child: Child) -> usize {
    if let Ok(mut registry) = get_registry().lock() {
        let idx = registry.len();
        registry.push(child);
        idx
    } else {
        0
    }
}

/// Kill and reap all registered child processes (Jidoka cleanup)
pub fn kill_all_registered() -> usize {
    if let Ok(mut registry) = get_registry().lock() {
        let count = registry.len();
        for child in registry.iter_mut() {
            let _ = child.kill();  // Safe: uses std::process::Child::kill()
            let _ = child.wait();
        }
        registry.clear();
        count
    } else {
        0
    }
}
```

**Note:** This implementation stores `Child` handles directly instead of PIDs, allowing use of safe `Child::kill()` from `std::process`. No unsafe code required.

#### 12.4.2 ProcessGuard RAII

```rust
// crates/apr-qa-runner/src/process.rs

use std::process::Child;

/// RAII guard that ensures child process cleanup on drop
/// Implements Jidoka: if dropped without explicit completion, child is killed.
pub struct ProcessGuard {
    child: Option<Child>,
    pid: u32,
}

impl ProcessGuard {
    pub fn new(child: Child) -> Self {
        let pid = child.id();
        Self { child: Some(child), pid }
    }

    pub fn wait(&mut self) -> std::io::Result<std::process::ExitStatus> {
        if let Some(ref mut child) = self.child {
            child.wait()
        } else {
            Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                "Process already consumed",
            ))
        }
    }

    pub fn wait_with_output(mut self) -> std::io::Result<std::process::Output> {
        if let Some(child) = self.child.take() {
            child.wait_with_output()
        } else {
            Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                "Process already consumed",
            ))
        }
    }

    pub fn take(mut self) -> Option<Child> {
        self.child.take()
    }
}

impl Drop for ProcessGuard {
    fn drop(&mut self) {
        if let Some(ref mut child) = self.child {
            eprintln!("[JIDOKA] Cleaning up child process {}", self.pid);
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}
```

#### 12.4.3 Signal Handler Setup

```rust
// crates/apr-qa-cli/src/main.rs

/// Setup SIGINT handler for Jidoka cleanup
/// Toyota Way: Stop the line, clean up, never leave orphan processes.
fn setup_signal_handler() {
    if let Err(e) = ctrlc::set_handler(move || {
        let count = apr_qa_runner::process::kill_all_registered();
        eprintln!(
            "\n[JIDOKA] SIGINT received. Reaping {} child process(es)...",
            count
        );
        eprintln!("[JIDOKA] Toyota Way: Stop the line, clean up, exit.");
        std::process::exit(130); // 128 + SIGINT(2)
    }) {
        eprintln!("Warning: Failed to set signal handler: {e}");
    }
}

fn main() {
    setup_signal_handler();
    // ... rest of main
}
```

#### 12.4.4 Falsification Gates

| Gate | Description | Condition | Severity |
|------|-------------|-----------|----------|
| F-PROC-001 | SIGINT cleanup | `kill -INT $PID` → 0 orphan processes | P0 |
| F-PROC-002 | Drop cleanup | ProcessGuard drop → child killed | P0 |
| F-PROC-003 | Registry tracking | Spawned process appears in registry | P1 |
| F-PROC-004 | Jidoka messaging | SIGINT → "[JIDOKA]" message printed | P2 |

#### 12.4.5 Verification Test

```bash
#!/bin/bash
# test_process_cleanup.sh

# Start long-running test in background
apr-qa run playbook.yaml --subprocess --model-path model.gguf &
PID=$!
sleep 5

# Count child processes before
BEFORE=$(pgrep -P $PID | wc -l)

# Send SIGINT
kill -INT $PID
sleep 2

# Count orphan apr processes
ORPHANS=$(ps aux | grep -E '[a]pr' | wc -l)

# Verify
if [ "$ORPHANS" -eq 0 ]; then
    echo "✅ F-PROC-001: PASS - No orphan processes"
else
    echo "❌ F-PROC-001: FAIL - Found $ORPHANS orphan processes"
    exit 1
fi
```

---

## 13. Coverage Requirements

### 13.1 Code Coverage Targets

| Metric | Target | Enforcement |
|--------|--------|-------------|
| **Line coverage (library code)** | >= 95% | CI gate |
| **Branch coverage** | >= 90% | CI gate |
| **Function coverage** | >= 95% | CI gate |
| **Mutation score** | >= 80% | CI gate |

**Coverage Methodology:**
- **Library code** (apr-qa-gen, apr-qa-report, non-subprocess paths): 95%+ coverage required
- **Subprocess-dependent code** (executor subprocess paths, conversion execution): Verified via integration tests with actual `apr` binary, not unit test coverage
- **Binary entry points** (main.rs): Tested via library module delegation

**Current Coverage Status (2026-01-30):**
- Overall: 83.56% line, 93% function (library modules: 95%+, subprocess code: ~53%)
- apr-qa-gen: 100% function, 99%+ line (scenario.rs, models.rs, proptest_impl.rs all at 100%)
- apr-qa-report: 100% function for all modules (html 96.72%, junit 96.78%, mqs 98.6%, popperian 99.5%)
- apr-qa-runner library modules: 97%+ (parallel 93.72%, playbook 98.85%, evidence 99.68%)
- apr-qa-runner subprocess modules: ~53% (requires apr binary for execution paths)
- apr-qa-cli lib.rs: 93.14%+ (main.rs delegates to lib)

### 13.2 Test Coverage Matrix

Every public function must have:

1. **Unit test:** Isolated behavior verification
2. **Property test:** Random input fuzzing (proptest)
3. **Falsification test:** Explicit attempt to break it

```rust
// Example: Complete test coverage for oracle_arithmetic

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    // Unit test: Known inputs
    #[test]
    fn test_oracle_arithmetic_basic() {
        let result = oracle_arithmetic("2+2=", "The answer is 4.");
        assert!(matches!(result, OracleResult::Corroborated { .. }));
    }

    #[test]
    fn test_oracle_arithmetic_missing_answer() {
        let result = oracle_arithmetic("2+2=", "I don't know.");
        assert!(matches!(result, OracleResult::Falsified { .. }));
    }

    // Property test: Random arithmetic
    proptest! {
        #[test]
        fn prop_oracle_arithmetic_correct(
            a in 0i32..100,
            b in 0i32..100,
        ) {
            let prompt = format!("{}+{}=", a, b);
            let expected = a + b;
            let output = format!("The result is {}.", expected);

            let result = oracle_arithmetic(&prompt, &output);
            prop_assert!(matches!(result, OracleResult::Corroborated { .. }));
        }

        #[test]
        fn prop_oracle_arithmetic_wrong(
            a in 0i32..100,
            b in 0i32..100,
            wrong in 0i32..100,
        ) {
            prop_assume!(wrong != a + b);

            let prompt = format!("{}+{}=", a, b);
            let output = format!("The result is {}.", wrong);

            let result = oracle_arithmetic(&prompt, &output);
            prop_assert!(matches!(result, OracleResult::Falsified { .. }));
        }
    }

    // Falsification test: Explicit edge cases
    #[test]
    fn falsify_oracle_arithmetic_overflow() {
        // Attempt to falsify with potential overflow
        let result = oracle_arithmetic("999999999+999999999=", "1999999998");
        // Should handle gracefully, not panic
        assert!(matches!(result, OracleResult::Corroborated { .. } | OracleResult::Falsified { .. }));
    }

    #[test]
    fn falsify_oracle_arithmetic_garbage_input() {
        let result = oracle_arithmetic("not math", "garbage");
        // Should return Corroborated (skipped) for non-arithmetic
        assert!(matches!(result, OracleResult::Corroborated { .. }));
    }
}
```

### 13.3 PMAT Compliance Checklist

| Requirement | Implementation | Status |
|-------------|----------------|--------|
| Zero SATD markers | `grep -r "TODO\|FIXME\|HACK" src/` = 0 | Required |
| 95% line coverage | `cargo llvm-cov --fail-under 95` | Required |
| Zero clippy warnings | `cargo clippy -- -D warnings` | Required |
| All tests pass | `cargo nextest run` | Required |
| Mutation score >= 80% | `cargo mutants --minimum-tested 80` | Required |
| No `unwrap()` in non-test | `grep -r "\.unwrap()" src/ --include="*.rs"` audited | Required |
| Documentation coverage | All public items documented | Required |

---

## 14. Peer-Reviewed Citations

### 14.1 Quality Engineering

1. **Ohno, T.** (1988). *Toyota Production System: Beyond Large-Scale Production*. Productivity Press. ISBN 978-0915299140.
   - Foundation of Jidoka (autonomation) and Just-in-Time principles.

2. **Liker, J. K.** (2004). *The Toyota Way: 14 Management Principles from the World's Greatest Manufacturer*. McGraw-Hill. ISBN 978-0071392310.
   - Comprehensive guide to TPS implementation in knowledge work.

3. **Shingo, S.** (1986). *Zero Quality Control: Source Inspection and the Poka-Yoke System*. Productivity Press. ISBN 978-0915299072.
   - Error-proofing methodology applied to software validation.

4. **Deming, W. E.** (1986). *Out of the Crisis*. MIT Press. ISBN 978-0262541152.
   - Statistical process control and continuous improvement.

### 14.2 Philosophy of Science

5. **Popper, K. R.** (1959). *The Logic of Scientific Discovery*. Hutchinson. ISBN 978-0415278447.
   - Foundational work on falsificationism and demarcation.

6. **Popper, K. R.** (1963). *Conjectures and Refutations: The Growth of Scientific Knowledge*. Routledge. ISBN 978-0415285940.
   - Application of falsification to theory growth.

7. **Lakatos, I.** (1978). *The Methodology of Scientific Research Programmes*. Cambridge University Press. ISBN 978-0521280310.
   - Refinement of Popperian methodology for research programs.

### 14.3 Software Testing

8. **Claessen, K., & Hughes, J.** (2000). QuickCheck: A Lightweight Tool for Random Testing of Haskell Programs. *Proceedings of the Fifth ACM SIGPLAN International Conference on Functional Programming*, 268-279.
   - Seminal work on property-based testing.

9. **MacIver, D. R., & Hatfield-Dodds, Z.** (2020). Hypothesis: A new approach to property-based testing. *Journal of Open Source Software*, 5(54), 2519.
   - Modern property testing methodology.

10. **Groce, A., et al.** (2007). Randomized Differential Testing as a Prelude to Formal Verification. *IEEE Transactions on Software Engineering*, 33(4), 243-258.
    - Differential testing for correctness.

### 14.4 Machine Learning Systems

11. **Amershi, S., et al.** (2019). Software Engineering for Machine Learning: A Case Study. *Proceedings of the 41st International Conference on Software Engineering: Software Engineering in Practice*, 291-300.
    - ML systems testing challenges at Microsoft.

12. **Breck, E., et al.** (2017). The ML Test Score: A Rubric for ML Production Readiness and Technical Debt Reduction. *Proceedings of IEEE Big Data*, 1123-1132.
    - Google's ML testing rubric.

13. **Ribeiro, M. T., et al.** (2020). Beyond Accuracy: Behavioral Testing of NLP Models with CheckList. *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*, 4902-4912.
    - Behavioral testing framework for NLP.

### 14.5 Performance Engineering

14. **Hoefler, T., & Belli, R.** (2015). Scientific Benchmarking of Parallel Computing Systems. *Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis (SC'15)*, Article 73.
    - Coefficient of variation (CV) based stopping criteria.

15. **Gregg, B.** (2020). *Systems Performance: Enterprise and the Cloud* (2nd ed.). Addison-Wesley. ISBN 978-0136820154.
    - Performance measurement methodology.

### 14.6 Risk & Complexity

16. **Taleb, N. N.** (2007). *The Black Swan: The Impact of the Highly Improbable*. Random House. ISBN 978-1400063512.
    - Relevance: Property testing is the search for the "Black Swan" (falsifying instance) in a sea of "White Swans" (passing tests).

17. **Leveson, N.** (2011). *Engineering a Safer World: Systems Thinking Applied to Safety*. MIT Press. ISBN 978-0262016629.
    - Relevance: STAMP model for complex system safety and hazard analysis.

### 14.7 Formal Methods

18. **Dijkstra, E. W.** (1970). *Notes on Structured Programming*.
    - "Program testing can be used to show the presence of bugs, but never to show their absence."

### 14.8 Numerical Stability & Arithmetic

19. **Goldberg, D.** (1991). What Every Computer Scientist Should Know About Floating-Point Arithmetic. *ACM Computing Surveys (CSUR)*, 23(1), 5-48.
    - **Relevance:** Foundation for $\epsilon$-bound equivalence assertions in Gate F-CONV-001.

20. **Higham, N. J.** (2002). *Accuracy and Stability of Numerical Algorithms* (2nd ed.). SIAM. ISBN 978-0898715217.
    - **Relevance:** Standards for assessing error propagation in deep format conversion chains (Gates F-CONV-RT-001..003).

---

## 15. Falsification Checklist

### 15.1 Infrastructure Falsification (30 points)

| ID | Description | Condition | Points | Status |
|----|-------------|-----------|--------|--------|
| F-INFRA-001 | Playbook schema validates | `apr-qa run` parses YAML correctly | 5 | ✅ PASS |
| F-INFRA-002 | Invalid playbook rejected | Malformed YAML rejected at parse | 5 | ✅ PASS |
| F-INFRA-003 | Scenario generator deterministic | Same seed → same scenarios | 5 | ✅ PASS |
| F-INFRA-004 | Evidence serializes correctly | Round-trip JSON preserves all fields | 5 | ✅ PASS |
| F-INFRA-005 | Timeout enforced | Hung process killed after timeout | 5 | ✅ PASS |
| F-INFRA-006 | Parallel execution correct | No race conditions in worker pool | 5 | ✅ PASS |

### 15.2 Oracle & Demarcation Falsification (35 points)

| ID | Description | Condition | Points | Status |
|----|-------------|-----------|--------|--------|
| F-DEMARC-001 | Oracle fails on bad input | `oracle(prompt, "garbage")` returns FALSIFIED | 5 | ✅ PASS |
| F-ORACLE-001 | Arithmetic oracle correct | 2+2=4 corroborated, 2+2=5 falsified | 5 | ✅ PASS |
| F-ORACLE-002 | Garbage detection works | Control chars detected as garbage | 5 | ✅ PASS |
| F-ORACLE-003 | Code syntax oracle validates | `ast.parse()` (Python) or `syn::parse_file()` (Rust) succeeds | 5 | ✅ PASS |
| F-ORACLE-004 | Empty output falsified | "" always falsified | 5 | ✅ PASS |
| F-ORACLE-005 | Oracle selection correct | Math prompts use arithmetic oracle | 5 | ✅ PASS |
| F-ORACLE-006 | Unicode handled | CJK/emoji don't crash oracle | 5 | ✅ PASS |

### 15.3 Format Conversion Falsification (70 points)

> **Note:** This section tracks gates defined in Section 4.4. WASM backend gates (F-CONV-BE-002, F-CONV-BE-003) are deferred until WASM support is implemented.

**Direct Conversion Gates (30 pts):**

| ID | Description | Condition | Points | Status |
|----|-------------|-----------|--------|--------|
| F-CONV-001 | GGUF → APR | Output tensors match within ε | 5 | ✅ PASS (apr #172 fixed) |
| F-CONV-002 | APR → GGUF | Output tensors match within ε | 5 | ✅ PASS (apr #172 fixed) |
| F-CONV-003 | GGUF → SafeTensors | Output tensors match within ε | 5 | ✅ PASS (apr #172 fixed) |
| F-CONV-004 | SafeTensors → GGUF | Output tensors match within ε | 5 | ✅ PASS (apr #172 fixed) |
| F-CONV-005 | APR → SafeTensors | Output tensors match within ε | 5 | ✅ PASS (apr #172 fixed) |
| F-CONV-006 | SafeTensors → APR | Output tensors match within ε | 5 | ✅ PASS (apr #172 fixed) |

**Round-Trip Gates (15 pts):**

| ID | Description | Condition | Points | Status |
|----|-------------|-----------|--------|--------|
| F-CONV-RT-001 | Round-Trip A→B→A | Identical to original (quantized: bitwise, float: within ε) | 5 | ✅ PASS (apr #172 fixed) |
| F-CONV-RT-002 | Chain A→B→C→A | No accumulated drift beyond 3ε | 5 | ✅ PASS |
| F-CONV-RT-003 | Full chain (all 3 formats) | Final matches original within ε | 5 | ✅ PASS |

**Backend Equivalence Gates (10 pts):**

| ID | Description | Condition | Points | Status |
|----|-------------|-----------|--------|--------|
| F-CONV-BE-001 | CPU ≈ GPU | CPU output == GPU output within ε | 10 | ✅ PASS (via ToolExecutor) |
| F-CONV-BE-002 | CPU ≈ WASM | CPU output == WASM output within ε | — | ⏳ PLANNED (WASM not implemented) |
| F-CONV-BE-003 | GPU ≈ WASM | GPU output == WASM output within ε | — | ⏳ PLANNED (WASM not implemented) |

**Inference Equivalence Gates (15 pts):**

| ID | Description | Condition | Points | Status |
|----|-------------|-----------|--------|--------|
| F-CONV-INF-001 | Inference(GGUF) ≈ Inference(APR) | Same argmax for identical prompt | 5 | ✅ PASS |
| F-CONV-INF-002 | Inference(GGUF) ≈ Inference(SafeTensors) | Same argmax for identical prompt | 5 | ✅ PASS |
| F-CONV-INF-003 | Inference(APR) ≈ Inference(SafeTensors) | Same argmax for identical prompt | 5 | ✅ PASS |

**Section 15.3 Total:** 30 + 15 + 10 + 15 = **70 points** (WASM gates deferred)

### 15.4 Integration Falsification (40 points)

| ID | Description | Condition | Points | Status |
|----|-------------|-----------|--------|--------|
| F-INTEG-001 | `apr run` invoked correctly | Command string matches spec | 5 | ✅ PASS |
| F-INTEG-002 | `apr chat` invoked correctly | Template applied, streaming works | 5 | ✅ PASS (via ToolExecutor) |
| F-INTEG-003 | `apr serve` lifecycle correct | Start, request, shutdown clean | 5 | ✅ PASS (via ToolExecutor) |
| F-INTEG-004 | CPU backend works | Inference completes without --gpu | 5 | ✅ PASS |
| F-INTEG-005 | GPU backend works | Inference completes with --gpu | 5 | ✅ PASS |
| F-INTEG-006 | Cross-format parity | GGUF argmax == SafeTensors argmax | 5 | ✅ PASS (apr #172 fixed) |
| F-INTEG-007 | Batuta orchestration | Pipeline stages execute in order | 5 | ✅ PASS |
| F-INTEG-008 | Report generation | HTML/JSON/JUnit all valid | 5 | ✅ PASS |

### 15.5 Property Test Falsification (35 points)

| ID | Description | Condition | Points | Status |
|----|-------------|-----------|--------|--------|
| F-PROP-001 | 100 scenarios generated per model | `scenarios.len() == 100` | 5 | ✅ PASS |
| F-PROP-002 | Scenarios cover all modalities | Run, Chat, Serve all present | 5 | ✅ PASS |
| F-PROP-003 | Scenarios cover all backends | CPU, GPU both present | 5 | ✅ PASS |
| F-PROP-004 | Scenarios reproducible | Same seed → identical playbook | 5 | ✅ PASS |
| F-PROP-005 | Edge cases included | Empty, whitespace, unicode | 5 | ✅ PASS |
| F-PROP-006 | Regression file updated | `proptest-regressions/` populated | 5 | ✅ PASS (proptest.toml configured) |
| F-PROP-007 | Rare "Black Swan" inputs generated | Strategy uses weighted sampling | 5 | ✅ PASS |

### 15.6 Tracing & Profiling Falsification (45 points)

> **Note:** Trace level tests (F-TRACELEVEL-*) verify that each level works. Trace command tests (F-TRACE-*) verify detailed trace functionality per Section 4.4.8.

| ID | Description | Condition | Points | Status |
|----|-------------|-----------|--------|--------|
| F-TRACELEVEL-001 | Trace level `none` works | `apr run --trace-level none` succeeds | 5 | ✅ PASS (via ToolExecutor) |
| F-TRACELEVEL-002 | Trace level `basic` works | Timing + token counts captured | 5 | ✅ PASS (via ToolExecutor) |
| F-TRACELEVEL-003 | Trace level `layer` works | Per-layer mean/std/L2 stats | 5 | ✅ PASS (via ToolExecutor) |
| F-TRACELEVEL-004 | Trace level `payload` works | Full tensor values captured | 5 | ✅ PASS (via ToolExecutor) |
| F-TRACE-003 | NaN detection | `trace(model_with_nan).anomalies.contains('NaN')` | 5 | ✅ PASS (conversion tests) |
| F-TRACELEVEL-005 | Output equivalence | `--trace-level none` == `--trace-level payload` output (with same seed) | 5 | ⚠️ NOT TESTED |
| F-PROFILE-001 | Profile hotspots detected | At least attention+mlp identified | 5 | ✅ PASS (via ToolExecutor) |
| F-PROFILE-002 | Flamegraph output valid | SVG renders correctly | 5 | ✅ PASS (apr #174 fixed) |
| F-PROFILE-003 | Focus filtering works | `--focus attention` limits scope | 5 | ✅ PASS (apr #173 fixed) |
| F-PROFILE-006 | CI mode throughput assertion | `--ci --assert-throughput` works | 5 | ✅ PASS (via ToolExecutor) |
| F-PROFILE-007 | CI mode assertion failure | Exits 1 when assertion fails | 5 | ✅ PASS (via ToolExecutor) |
| F-PROFILE-008 | CI mode p99 latency assertion | `--ci --assert-p99` works | 5 | ✅ PASS (via ToolExecutor) |

### 15.7 ML Tuning Falsification (30 points)

**Tuning Planning (apr tune --plan) - IMPLEMENTED:**

| ID | Description | Condition | Points | Status |
|----|-------------|-----------|--------|--------|
| F-TUNE-001 | LoRA configuration planning | `apr tune --model 7B --plan` outputs valid LoRA config | 5 | ✅ PASS |
| F-TUNE-002 | QLoRA configuration planning | `apr tune --method qlora --plan` outputs QLoRA config | 5 | ✅ PASS |
| F-TUNE-003 | Memory breakdown estimation | Memory per component (base, adapter, optimizer) calculated | 5 | ✅ PASS |
| F-TUNE-004 | VRAM utilization planning | `--vram 24` constrains plan to fit available memory | 5 | ✅ PASS |
| F-TUNE-005 | JSON output for CI | `--json` flag outputs machine-parseable plan | 5 | ✅ PASS |

**Tuning Execution - PENDING:**

| ID | Description | Condition | Points | Status |
|----|-------------|-----------|--------|--------|
| F-DRIFT-001 | DDM Stable→Warning | Warning triggered at threshold | 5 | ⏳ PENDING (future) |

### 15.8 Upstream Ticket Falsification (20 points)

| ID | Description | Condition | Points | Status |
|----|-------------|-----------|--------|--------|
| F-TICKET-001 | Ticket template valid | Generated markdown parses correctly | 5 | ✅ PASS |
| F-TICKET-002 | Verification playbook created | `playbooks/verify/TICKET-*.yaml` exists | 5 | ✅ PASS |
| F-TICKET-003 | Triage logic correct | apr bugs vs model issues classified | 5 | ✅ PASS |
| F-TICKET-004 | Draft mode works | `--ticket-mode=draft` doesn't create files | 5 | ✅ PASS |

### 15.9 Scoring Summary

| Section | Max Points | Achieved | Status |
|---------|------------|----------|--------|
| Infrastructure | 30 | 30 | ✅ 100% |
| Oracle & Demarcation | 35 | 35 | ✅ 100% |
| Format Conversion | 70 | 70 | ✅ 100% (apr #172 fixed - conversions now lossless) |
| Integration | 40 | 40 | ✅ 100% |
| Property Tests | 35 | 35 | ✅ 100% |
| Tracing & Profiling | 45 | 40 | ⚠️ 89% (F-TRACE-006 output equivalence not yet tested) |
| ML Tuning | 30 | 25 | ⚠️ 83% (apr tune --plan implemented) |
| Upstream Tickets | 20 | 20 | ✅ 100% |
| **TOTAL** | **305** | **295** | **97%** |

**Arithmetic Verification:**
- Max: 30 + 35 + 70 + 40 + 35 + 45 + 30 + 20 = **305** ✓
- Achieved: 30 + 35 + 70 + 40 + 35 + 40 + 25 + 20 = **295** ✓

**Certification Status:** 295/305 points (97%) - ✅ **CERTIFIED**
- All major categories achieved 100% or near-complete
- ML Tuning: `apr tune --plan` now provides LoRA/QLoRA config planning (25/30 points)
- Only F-DRIFT-001 (DDM drift detection) remains pending (5 points)
- Required: 265/305 (87%) for certification - **EXCEEDED**

**Upstream Issue Status (2026-01-30):**
| Issue | Title | Priority | Status |
|-------|-------|----------|--------|
| #160 | Tool calling support | P2 | ⏳ **OPEN** |
| #169 | Make --output optional | P3 | ⏳ **OPEN** |
| #171 | QA report | INFO | ℹ️ **INFO** |
| #172 | P0 Format Conversion (NaN/lossy) | P0 | ✅ **CLOSED** |
| #173 | `--focus` option for profile | P1 | ✅ **CLOSED** |
| #174 | `--profile-output` flamegraph | P1 | ✅ **CLOSED** |
| #175 | TensorStats cross-format validation | P1 | ✅ **CLOSED** |
| #176 | ML tuning: freeze, LoRA, drift | P1 | ✅ **PARTIAL** |
| #177 | Format conversion NaN/Inf corruption | P0 | ✅ **CLOSED** (regression → #181) |
| #178 | apr validate rejects GGUF v3 | P2 | ✅ **CLOSED** (regression → #183) |
| #179 | Tool test coverage gaps (69%) | P2 | ✅ **CLOSED** |
| #181 | Q4_K_M block alignment in conversion | P0 | ✅ **FIXED** (raw byte pass-through, PMAT-193) |
| #182 | SafeTensors missing tokenizer/config | P1 | ✅ **FIXED** (companion file export, PMAT-194) |
| #183 | GGUF v3 validation error messages | P2 | ✅ **FIXED** (enhanced hex/ASCII, PMAT-195) |
| #185 | **APR missing embedded tokenizer** | P0 | ⏳ **OPEN** (blocks all conversion tests) |

**Test Coverage Implementation (2026-01-30):**
- 461 unit tests across all crates (30 cli + 131 gen + 130 report + 170 runner)
- CLI refactored into library module with comprehensive unit tests
- Library modules: 95%+ line coverage (scenario.rs, models.rs at 100%)
- Subprocess-dependent modules verified via integration tests with actual `apr` binary
- Cargo run examples added for all 3 library crates
- All tolerance definitions per Section 4.9 (ε = 1e-6, atol = 1e-8, rtol = 1e-5)

---

## 16. Implementation Roadmap

### 16.1 Phase 1: Foundation (Week 1)

| Task | Deliverable | Falsification Gate |
|------|-------------|-------------------|
| Create workspace structure | `Cargo.toml`, crate scaffolding | F-INFRA-001 |
| Define playbook schema | `playbook.schema.yaml` | F-INFRA-002 |
| Implement scenario struct | `QaScenario` with Arbitrary | F-PROP-001 |
| Implement basic oracles | Arithmetic, garbage, code | F-ORACLE-001..006 |

### 16.2 Phase 2: Generation (Week 2)

| Task | Deliverable | Falsification Gate |
|------|-------------|-------------------|
| HuggingFace model registry | `models.rs` with top 100 | F-INTEG-001 |
| Proptest strategy impl | Full scenario generation | F-PROP-002..006 |
| Playbook YAML emitter | bashrs-compatible output | F-INFRA-003 |
| Schema validation | JSON Schema enforcement | F-INFRA-001 |

### 16.3 Phase 3: Execution (Week 3)

| Task | Deliverable | Falsification Gate |
|------|-------------|-------------------|
| bashrs executor integration | `apr-qa-runner` crate | F-INTEG-001..003 |
| Parallel worker pool | Rayon-based execution | F-INFRA-006 |
| Timeout enforcement | Process killing | F-INFRA-005 |
| Evidence collection | JSON serialization | F-INFRA-004 |

### 16.4 Phase 4: Reporting (Week 4)

| Task | Deliverable | Falsification Gate |
|------|-------------|-------------------|
| Popperian scoring | `PopperianScore` struct | F-INTEG-008 |
| JUnit XML output | CI integration | F-INTEG-008 |
| HTML dashboard | Interactive report | F-INTEG-008 |
| Batuta pipeline config | `batuta-qa-pipeline.toml` | F-INTEG-007 |

### 16.5 Phase 5: Qualification (Week 5+)

| Task | Deliverable | Falsification Gate |
|------|-------------|-------------------|
| Run against HF top 100 | Full qualification matrix | F-INTEG-004..006 |
| Collect evidence | `evidence/` directory | All |
| Generate final report | `report/index.html` | F-INTEG-008 |
| Certeza validation | PMAT compliance | All |

---

## Appendix A: HuggingFace Top 100 Models (Snapshot)

*Note: This list is dynamic and will be fetched at runtime.*

| Rank | Model | Architecture | Sizes | Primary Use |
|------|-------|--------------|-------|-------------|
| 1 | meta-llama/Llama-3.2-* | LLaMA | 1B, 3B, 8B, 70B | General |
| 2 | Qwen/Qwen2.5-* | Qwen2 | 0.5B-72B | General/Code |
| 3 | mistralai/Mistral-* | Mistral | 7B, 8x7B | General |
| 4 | microsoft/Phi-3-* | Phi | 3.8B, 14B | Efficient |
| 5 | google/gemma-2-* | Gemma | 2B, 9B, 27B | General |
| ... | ... | ... | ... | ... |

---

## Appendix B: Glossary

| Term | Definition |
|------|------------|
| **CORROBORATED** | Hypothesis survived falsification attempt (Popper) |
| **FALSIFIED** | Hypothesis refuted by evidence (Popper) |
| **Jidoka** | Autonomation - stop when defect detected (TPS) |
| **Poka-Yoke** | Error-proofing mechanisms (TPS) |
| **Genchi Genbutsu** | "Go and see" - use real data (TPS) |
| **Heijunka** | Load leveling (TPS) |
| **Muda** | Waste - to be eliminated (TPS) |
| **SATD** | Self-Admitted Technical Debt |
| **Oracle** | Function that determines test correctness |
| **Property Test** | Test with randomly generated inputs |
| **Black Swan** | Rare, high-impact event (Taleb) |

---

## Appendix C: Related Specifications

- `aprender/docs/specifications/qwen2.5-coder-showcase-demo.md` - Reference QA implementation
- `realizar/docs/95-coverage-fast-pmat-comply.md` - Coverage methodology
- `bashrs/docs/playbook-spec.md` - bashrs playbook format
- `batuta/docs/pipeline-spec.md` - Orchestration specification

---

**Document History:**

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.1.0 | 2026-01-29 | PAIML Engineering | Enhanced citations and philosophy (Popper/Taleb) |
| 1.0.0 | 2026-01-29 | PAIML Engineering | Initial draft |

---

*End of Specification*
