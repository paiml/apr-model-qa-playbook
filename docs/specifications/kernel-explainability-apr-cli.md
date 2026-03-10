# Kernel Explainability for `apr explain --kernel`

**Status**: Implemented (aprender commit 07ae637a)
**Date**: 2026-03-10
**Author**: PAIML Engineering
**Depends on**: aprender/contracts/model-families/*.yaml, provable-contracts/contracts/arch-constraints-v1.yaml
**Consumers**: aprender/crates/apr-cli/src/commands/explain.rs

## 1. Problem Statement

A user runs a model through the sovereign stack and it fails (garbage
output, crash, wrong scores). Today they see cryptic error messages:
"Output is highly repetitive", "G4 failed: 3/4 garbage-oracle tests
failed". They have no way to understand *which kernels* fired, *why*
those kernels were selected, and *whether the kernel pipeline is
provably correct* for their model architecture.

The kernel pipeline decision is fully deterministic from config.json
metadata. It can be explained without loading the model, without
running inference, and without downloading weights. This is pure
static analysis over contract data.

## 2. Prior Art and Research

### 2.1 Formal Kernel Equivalence (Volta)

Dubey et al. (2025) introduce Volta, the first equivalence checker for
ML GPU kernels. Volta symbolically executes kernel pairs, checks data
race freedom, then verifies output tensor equivalence. Key properties:
**soundness** (no false positives) and **completeness** (for a
well-defined class of ML kernels). Covers convolutions, matrix
multiplications, and attention mechanisms.

**Relevance**: Volta proves two kernel *implementations* produce
identical outputs. Our kernel equivalence classes prove two model
*architectures* dispatch to identical kernel *pipelines*. These are
complementary: Volta verifies the kernel code, we verify the kernel
selection. Together they form a complete chain: architecture →
kernel selection (us) → kernel correctness (Volta).

Reference: [Equivalence Checking of ML GPU Kernels](https://arxiv.org/abs/2511.12638)

### 2.2 Popperian Sequential Falsification (Popper Framework)

Huang et al. (2025) apply Karl Popper's falsification principle to
hypothesis validation via LLM agents that design and execute
falsification experiments. Their sequential testing framework ensures
strict Type-I error control while gathering evidence from diverse
observations.

**Relevance**: Our kernel explainability adopts the same falsification
structure. Each kernel dispatch decision is a falsifiable hypothesis:
"Given config.json field `hidden_act=silu`, the stack MUST dispatch
the SiLU activation kernel." The explain command reports which
hypotheses have been *corroborated* (contract + test evidence) vs
which are *unfalsified* (contract exists but no test coverage) vs
which are *unknown* (no contract at all).

Reference: [Automated Hypothesis Validation with Agentic Sequential Falsifications](https://arxiv.org/abs/2502.09858)

### 2.3 Deep Kernel Fusion (DeepFusionKernel)

Recent work on kernel fusion for transformer inference demonstrates
that fused kernels reduce memory traffic but introduce verification
complexity. Speedup variability is explained by communication jitter
and system-level noise.

**Relevance**: The existing `kernel-fusion-v1.yaml` contract in
aprender documents fusion decisions. The explain command MUST surface
fusion status: which ops are fused, which are sequential, and what the
contract says about why.

Reference: [Deep Kernel Fusion for Transformers](https://arxiv.org/html/2602.11808)

### 2.4 HuggingFace Architecture Metadata (config.json)

HuggingFace's `PreTrainedConfig` standardizes architecture metadata
fields that determine kernel dispatch:

| config.json field | Kernel decision | Example values |
|-------------------|----------------|----------------|
| `model_type` | Architecture class dispatch | `llama`, `qwen2`, `gpt2` |
| `num_key_value_heads` | GQA vs MHA vs MQA | `num_kv_heads < num_heads` → GQA |
| `hidden_act` | Activation kernel | `silu`, `gelu`, `gelu_new` |
| `rms_norm_eps` | RMSNorm (not LayerNorm) | presence implies RMSNorm |
| `rope_theta` | RoPE positional encoding | presence implies RoPE |
| `max_position_embeddings` | Context length / ALiBi | `131072` for long-context |
| `intermediate_size` | MLP width → SwiGLU detection | `2.67x hidden_size` → SwiGLU |
| `num_experts` | MoE routing dispatch | presence → Class E |

**Relevance**: These fields are the input to kernel class resolution.
The explain command reads them from config.json (or APR metadata) and
maps to kernel equivalence class.

Reference: [HuggingFace Configuration](https://huggingface.co/docs/transformers/en/main_classes/configuration)

### 2.5 HuggingFace Kernel Hub

HuggingFace's Kernel Hub provides pre-compiled optimized kernels
(FlashAttention, Triton LayerNorm/RMSNorm, activation kernels,
MoE routing) loadable via `get_kernel()`. The `kernels` library
auto-detects hardware (CUDA version, GPU architecture) and downloads
matching binaries.

**Relevance**: The HF Kernel Hub organizes kernels by the same
categories we use (attention, normalization, activation). Our kernel
equivalence classes map directly to HF kernel selections. The explain
command can show which HF Kernel Hub kernels correspond to each op.

Reference: [Learn the Hugging Face Kernel Hub in 5 Minutes](https://huggingface.co/blog/hello-hf-kernels)

### 2.6 V&V for Scientific ML

Maupin et al. (2025) survey verification and validation for
trustworthy scientific ML, establishing consensus on good practices
for predictive models. They emphasize that verification (does the
implementation match the specification?) and validation (does the
specification match reality?) are distinct concerns.

**Relevance**: Kernel explainability is verification: does the
kernel dispatch match the architecture contract? Model certification
(MQS scoring) is validation: does the model actually produce good
output? The explain command handles the former without needing the
latter.

Reference: [Verification and Validation for Trustworthy Scientific Machine Learning](https://arxiv.org/abs/2502.15496)

## 3. Design

### 3.1 Command Interface

```
apr explain --kernel <MODEL_OR_FAMILY> [--json] [--verbose] [--proof-status]

  MODEL_OR_FAMILY:
    model.apr          → read architecture from APR metadata
    model.gguf         → read architecture from GGUF metadata
    path/to/config.json → read config.json directly
    qwen2              → resolve from family contract
    hf://Qwen/Qwen2.5-Coder-1.5B → resolve via HF repo metadata

  --json              Machine-readable JSON output
  --verbose           Show kernel contract details and proof obligations
  --proof-status      Show per-kernel proof status from contract tests
```

### 3.2 Output Structure

**Default (human-readable):**

```
Kernel Explainability Report: Qwen2.5-Coder-1.5B
═════════════════════════════════════════════════

Architecture:  Qwen2ForCausalLM
Kernel Class:  A (GQA + RMSNorm + SiLU + SwiGLU + RoPE)
Family:        qwen2
Source:        config.json → model_type="qwen2"

┌─────────────────────────────────────────────────────────┐
│ Kernel Pipeline (8 ops)                                 │
├─────────────────────┬───────────────┬───────────────────┤
│ Operation           │ Kernel        │ Contract          │
├─────────────────────┼───────────────┼───────────────────┤
│ MatVec (Q4K)        │ fused_q4k_*   │ matvec-kernel-v1  │
│ MatVec (Q5K)        │ fused_q5k_*   │ matvec-kernel-v1  │
│ MatVec (Q6K)        │ fused_q6k_*   │ matvec-kernel-v1  │
│ Attention (GQA)     │ gqa_forward    │ attention-v1      │
│ Normalization       │ rms_norm       │ normalization-v1  │
│ Activation          │ silu           │ activation-v1     │
│ MLP                 │ swiglu         │ element-wise-v1   │
│ Position Encoding   │ rope_forward   │ rope-kernel-v1    │
└─────────────────────┴───────────────┴───────────────────┘

Config.json → Kernel Mapping:
  hidden_act=silu           → SiLU activation (not GELU)
  rms_norm_eps=1e-6         → RMSNorm (not LayerNorm)
  num_key_value_heads=2     → GQA (2 KV heads < 12 Q heads)
  rope_theta=1e6            → RoPE positional encoding
  intermediate_size=8960    → SwiGLU MLP (8960/1536 ≈ 5.83x)

Layout: Row-major (LAYOUT-002 compliant)
  GGUF→APR conversion transposes at import time.
  Direct GGUF inference uses column-major kernels.
```

**With `--proof-status`:**

```
Proof Status:
  ✓ matvec-kernel-v1      Proven (Kani harness + 2209 quantize tests)
  ✓ attention-kernel-v1   Proven (contract tests + CI gates)
  ✓ normalization-v1      Proven (contract tests)
  ✓ activation-v1         Proven (contract tests)
  ✓ rope-kernel-v1        Proven (contract tests)
  ✓ element-wise-v1       Proven (contract tests)
  ○ kernel-fusion-v1      Documented (no fused FFN — PAR-077)

Kernel Class A: Fully proven. 6/7 contracts verified.
```

**With `--json`:**

```json
{
  "architecture": "Qwen2ForCausalLM",
  "kernel_class": "A",
  "family": "qwen2",
  "kernel_ops": [
    { "op": "fused_q4k_matvec", "contract": "matvec-kernel-v1", "status": "proven" },
    { "op": "grouped_query_attention", "contract": "attention-kernel-v1", "status": "proven" },
    ...
  ],
  "config_mapping": {
    "hidden_act": { "value": "silu", "kernel": "SiLU", "rationale": "config.json hidden_act field" },
    ...
  },
  "proof_summary": { "proven": 6, "documented": 1, "unknown": 0, "total": 7 },
  "layout": "row_major",
  "equivalence_class_models": ["llama", "qwen2", "mistral", "yi", "deepseek", "internlm2"]
}
```

### 3.3 Resolution Chain

```
┌──────────────────────┐
│ User input           │  "qwen2" or "model.apr" or "config.json"
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Resolve architecture │  Read model_type from config.json / APR metadata
└──────────┬───────────┘  / GGUF metadata, or use family string directly
           │
           ▼
┌──────────────────────┐
│ Load constraints     │  From contracts/model-families/{family}.yaml
└──────────┬───────────┘  Fields: attention_type, norm_type, activation,
           │              mlp_type, positional_encoding, has_bias, tied_emb
           ▼
┌──────────────────────┐
│ Derive kernel class  │  Pure function of constraints → class A-F
└──────────┬───────────┘  (build.rs codegen from family YAML)
           │
           ▼
┌──────────────────────┐
│ Map to kernel ops    │  Class → [KernelOp] (FusedQ4kMatvec, GQA, ...)
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Resolve contracts    │  Each KernelOp → contract YAML path
└──────────┬───────────┘  (matvec-kernel-v1.yaml, rope-kernel-v1.yaml, ...)
           │
           ▼
┌──────────────────────┐
│ Query proof status   │  Read contract YAML: kani_harnesses, test refs
└──────────┬───────────┘  falsification_tests, enforcement sections
           │
           ▼
┌──────────────────────┐
│ Format and display   │  Human table or --json
└──────────────────────┘
```

### 3.4 Implementation Strategy: build.rs Codegen

The implementation avoids a cross-repo dependency from aprender →
apr-model-qa-playbook. Instead, it derives kernel class membership
from aprender's own data:

**Source data** (already in aprender):
- `contracts/model-families/*.yaml` → constraints sections
- `contracts/arch-constraints-v1.yaml` → architecture → constraint mapping
- `contracts/kernel-fusion-v1.yaml` → fusion decisions
- `contracts/matvec-kernel-v1.yaml`, `rope-kernel-v1.yaml`, etc. → per-kernel contracts

**Codegen** (new build.rs rules):
```rust
// build.rs — generates kernel_class_lookup.rs
// Input: contracts/model-families/*.yaml
// Output: const fn kernel_class(family: &str) -> Option<KernelClass>
//         const fn kernel_ops(class: KernelClass) -> &'static [KernelOp]
//         fn kernel_contracts(op: KernelOp) -> &'static str

fn derive_kernel_class(constraints: &Constraints) -> KernelClass {
    match (constraints.attention_type, constraints.norm_type,
           constraints.activation, constraints.mlp_type) {
        ("gqa", "rmsnorm", "silu", "swiglu")  => KernelClass::A,
        ("mha", "layernorm", "gelu", _)        => KernelClass::B,
        ("mqa", "layernorm", "gelu", _)        => KernelClass::C,
        ("gqa", "layernorm", _, _)             => KernelClass::D,
        _ if constraints.num_experts.is_some() => KernelClass::E,
        ("gqa", "rmsnorm", "gelu", "gated_mlp") => KernelClass::F,
        _ => KernelClass::Unknown,
    }
}
```

This pattern follows the precedent set by `arch-constraints-v1.yaml`
codegen (provable-contracts GH-323). The family YAMLs are the single
source of truth. The codegen is deterministic and testable.

### 3.5 Cross-Repo Alignment Validation

Add kernel class consistency to the existing
`scripts/validate-aprender-alignment.sh`:

```bash
# Verify kernel class derived from aprender family YAMLs matches
# apr-model-qa-playbook KernelClass::from_family()
for family in $(ls contracts/model-families/*.yaml | xargs -I{} basename {} .yaml); do
    aprender_class=$(apr explain --kernel "$family" --json | jq -r .kernel_class)
    qa_class=$(apr-qa kernel-coverage --family "$family" --json | jq -r .kernel_class)
    if [ "$aprender_class" != "$qa_class" ]; then
        echo "DRIFT: $family aprender=$aprender_class qa=$qa_class"
        exit 1
    fi
done
```

## 4. Falsification Protocol

Following Popperian methodology, the explain command's correctness is
subject to falsification:

### 4.1 Falsification Tests

| ID | Hypothesis | Falsification Experiment | If Fails |
|----|-----------|-------------------------|----------|
| FALSIFY-KE-001 | config.json → kernel class is deterministic | Same config.json always produces same kernel class | Nondeterministic codegen |
| FALSIFY-KE-002 | Kernel class matches actual kernel dispatch | Run model, trace kernels, compare to predicted | Constraint → kernel mapping error |
| FALSIFY-KE-003 | Unknown family returns Unknown, not panic | Pass "nonexistent-family" to explain --kernel | Missing fallback handling |
| FALSIFY-KE-004 | Cross-repo alignment holds | Run validate-aprender-alignment.sh | Contract drift |
| FALSIFY-KE-005 | Proof status reflects reality | Compare contract kani_harnesses/tests to claimed | Stale proof claims |
| FALSIFY-KE-006 | LAYOUT-002 warning for GGUF input | Explain a .gguf file, verify layout warning | Missing layout context |

### 4.2 Proof Levels

Each kernel operation has a proof level derived from its contract:

| Level | Meaning | Evidence |
|-------|---------|----------|
| **Proven** | Formally verified or exhaustively tested | Kani harness, proptest exhaustive, >1000 tests |
| **Tested** | Covered by contract tests | Falsification tests pass, CI gates green |
| **Documented** | Contract exists but no automated tests | YAML contract with proof_obligations section |
| **Unknown** | No contract exists | KernelOp has no corresponding contract YAML |

### 4.3 What This Does NOT Prove

The explain command provides kernel-level verification, NOT model-level
validation. It answers:
- "Which kernels will fire for this architecture?" (deterministic)
- "Are those kernels provably correct?" (contract-based)
- "Is the kernel selection consistent with config.json?" (traceable)

It does NOT answer:
- "Will the model produce good output?" (requires inference → MQS)
- "Are the weights correct?" (requires numerical validation)
- "Is the model safe to deploy?" (requires full certification)

## 5. Integration Points

### 5.1 apr explain (existing command)

Extend `commands/explain.rs` with `--kernel` flag. The `run()` function
already dispatches on positional argument type. Add kernel path:

```rust
if kernel_flag {
    explain_kernel(resolved_file.as_deref(), code.as_deref(), json, verbose, proof_status);
} else if let Some(c) = code { ... }
```

### 5.2 apr-qa kernel-coverage (existing command)

The QA playbook's `kernel-coverage` subcommand already verifies kernel
operation coverage per architecture. The explain command provides the
user-facing surface; kernel-coverage provides the CI gate.

### 5.3 Gateway G0 (DIM-SMOKE)

When `apr explain --kernel` reports a model's kernel class, it provides
the same information that G0 DIM-SMOKE uses for metadata-only
verification. An unexplainable model (Unknown class) should correlate
with G0 failures.

## 6. Effort Estimate

| Component | Lines | Risk |
|-----------|-------|------|
| build.rs codegen (family YAML → kernel class) | ~150 | Low (pattern exists) |
| explain_kernel() CLI handler | ~200 | Low (data formatting) |
| JSON output mode | ~50 | Low (serde) |
| Proof status reader (parse contract YAMLs) | ~100 | Medium (YAML parsing) |
| Cross-repo alignment script | ~30 | Low (bash) |
| Tests | ~150 | Low (unit tests) |
| **Total** | **~680** | **Low-Medium** |

The codegen pattern already exists (arch-constraints). The data already
exists (family YAMLs + kernel contracts). The CLI surface already exists
(`apr explain`). This is primarily a wiring exercise.

## 7. References

### Academic

- Dubey et al. "Equivalence Checking of ML GPU Kernels" (2025) — [arXiv:2511.12638](https://arxiv.org/abs/2511.12638)
- Huang et al. "Automated Hypothesis Validation with Agentic Sequential Falsifications" (2025) — [arXiv:2502.09858](https://arxiv.org/abs/2502.09858)
- "Deep Kernel Fusion for Transformers" (2026) — [arXiv:2602.11808](https://arxiv.org/html/2602.11808)
- Maupin et al. "Verification and Validation for Trustworthy Scientific ML" (2025) — [arXiv:2502.15496](https://arxiv.org/abs/2502.15496)

### HuggingFace

- [PreTrainedConfig Documentation](https://huggingface.co/docs/transformers/en/main_classes/configuration) — config.json field semantics
- [HF Kernel Hub](https://huggingface.co/blog/hello-hf-kernels) — kernel dispatch and organization
- [Qwen3 Config](https://huggingface.co/docs/transformers/en/model_doc/qwen3) — hidden_act, rms_norm_eps, rope_theta fields
- [LLaMA Config](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/configuration_llama.py) — reference implementation

### Stack Internal

- `aprender/contracts/model-families/*.yaml` — family constraint contracts (17 families)
- `aprender/contracts/kernel-fusion-v1.yaml` — fusion decisions with five-whys root cause
- `provable-contracts/contracts/arch-constraints-v1.yaml` — codegen source for ArchConstraints
- `apr-model-qa-playbook/crates/apr-qa-gen/src/kernel_class.rs` — KernelClass A-F taxonomy
- `apr-model-qa-playbook/crates/apr-qa-gen/src/kernel_profile.rs` — KernelProfile struct
- `apr-model-qa-playbook/book/src/architecture/scenario-generation.md` — kernel coverage docs
