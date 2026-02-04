# HuggingFace Parity Oracle Specification

**Version:** 1.1.0
**Status:** IMPLEMENTED
**Author:** PAIML Engineering
**Date:** 2026-02-03
**Philosophy:** Popperian Falsification + Toyota Jidoka + Numerical Reproducibility

---

## Implementation Status

| Component | Status | Notes |
|-----------|--------|-------|
| **Core Oracle** (`hf_parity.rs`) | ✅ Implemented | Tolerance configs, tensor comparison, bias detection |
| **CLI Command** (`apr-qa parity`) | ✅ Implemented | List, self-check, and verification modes |
| **Playbook Integration** | ✅ Implemented | `--hf-parity` flag in `apr-qa run` |
| **Golden Generator** (`generate_golden.py`) | ✅ Implemented | uv script with GPU support |
| **Make Targets** | ✅ Implemented | `parity-check`, `parity-list`, `golden-generate` |
| **Golden Corpus** | ✅ Available | 3 model families, 60 total golden outputs |

**Current Golden Corpus:**
- `qwen2.5-coder-1.5b/v1` - 20 prompts
- `qwen2.5-7b/v1` - 20 prompts
- `mistral-7b-v0.3/v1` - 20 prompts

---

## 1. Abstract

The HuggingFace Parity Oracle implements **cross-implementation validation** for the Sovereign AI Stack by treating HuggingFace transformers as an external **ground truth oracle**. This follows Popper's methodology of **severe testing** [1]: we attempt to falsify the hypothesis that our implementation produces equivalent outputs to the reference implementation.

> **Parity Hypothesis ($H_{parity}$):** For a given model $M$, input $x$, and tolerance $\epsilon$, the Sovereign Stack output $y_s$ is equivalent to the HuggingFace output $y_h$ such that $\|y_s - y_h\|_\infty < \epsilon$.

A **falsification** of $H_{parity}$ indicates either:
1. An implementation bug in the Sovereign Stack
2. A numerical precision difference requiring investigation
3. A layout/format incompatibility (cf. LAYOUT-002)

This oracle embodies Toyota's **Jidoka** principle [2]: automatic detection of defects with immediate stoppage, preventing defective outputs from propagating downstream.

## 2. Theoretical Foundation

### 2.1 Popperian Severe Testing

Popper argues that theories gain **corroboration** not through verification, but through surviving **severe tests** designed to refute them [1]. The HF Parity Oracle instantiates this by:

1. **Formulating a Bold Conjecture:** "Our implementation is numerically equivalent to HuggingFace"
2. **Designing Severe Tests:** Comparing outputs across diverse inputs, edge cases, and numerical regimes
3. **Accepting Falsification:** Any divergence beyond tolerance immediately halts certification

> "The wrong view of science betrays itself in the craving to be right; for it is not his possession of knowledge, of irrefutable truth, that makes the man of science, but his persistent and recklessly critical quest for truth." — Popper [1, p. 281]

### 2.2 Toyota Production System: Jidoka

**Jidoka** (自働化, "automation with a human touch") is the TPS principle of building quality into the process through automatic defect detection [2, 3]. The Parity Oracle implements Jidoka through:

| TPS Principle | Oracle Implementation |
|---------------|----------------------|
| **Andon** (signal) | `OracleResult::Falsified` raises immediate alert |
| **Stop-the-line** | Certification halts on first parity failure |
| **Root cause** | `TensorDiff` reports exact location of divergence |
| **Poka-yoke** | Tolerance thresholds prevent false negatives |

Ohno emphasizes that "stopping production" is not waste—it is the **prevention of greater waste** [3]. A model that passes certification with hidden numerical bugs will cause downstream failures in production.

### 2.3 Numerical Reproducibility in ML

Reproducibility in machine learning is a well-documented challenge [4, 5]. Sources of divergence include:

1. **Floating-point non-associativity:** $(a + b) + c \neq a + (b + c)$ in IEEE 754 [6]
2. **Non-deterministic GPU reductions:** cuBLAS atomics cause ordering variations [7]
3. **Quantization error:** INT4/INT8 introduces systematic bias [8]
4. **Tensor layout:** Row-major vs column-major affects numerical stability [9]

The Parity Oracle accounts for these by:
- Using **configurable tolerance** ($\epsilon$) per precision level
- Reporting **statistical divergence metrics** (max, mean, percentile)
- Flagging **systematic bias** vs random noise

## 3. Oracle Design

### 3.1 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     HF Ground Truth Corpus                      │
│  ~/src/hf-ground-truth-corpus/oracle/{model}/{hash}.safetensors │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      HfParityOracle                             │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────┐    │
│  │ GoldenLoader│→ │TensorCompare │→ │ DivergenceReporter  │    │
│  └─────────────┘  └──────────────┘  └─────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      OracleResult                               │
│           Corroborated { evidence } | Falsified { reason }      │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Golden Output Format

Golden outputs are stored as SafeTensors [10] with metadata:

```
{filename}.safetensors
├── Tensor: "logits" [batch, seq, vocab] F32
├── Tensor: "hidden_states" [batch, seq, hidden] F32 (optional)
├── Tensor: "attentions" [batch, heads, seq, seq] F32 (optional)
└── Metadata:
    ├── "prompt": original input text
    ├── "model": HuggingFace model ID
    ├── "transformers_version": e.g., "4.38.0"
    ├── "torch_version": e.g., "2.2.0"
    └── "generation_config": JSON serialized config
```

### 3.3 Tolerance Specification

Following IEEE 754 analysis [6] and ML reproducibility guidelines [4]:

| Precision | Absolute Tolerance ($\epsilon_a$) | Relative Tolerance ($\epsilon_r$) | Rationale |
|-----------|----------------------------------|----------------------------------|-----------|
| FP32 | $10^{-5}$ | $10^{-4}$ | ~6 significant digits |
| FP16 | $10^{-3}$ | $10^{-2}$ | ~3 significant digits |
| BF16 | $10^{-2}$ | $10^{-1}$ | ~2 significant digits |
| INT8 | $10^{-1}$ | N/A | Quantization error dominates |
| INT4 | $5 \times 10^{-1}$ | N/A | High quantization error expected |

Comparison uses the **allclose** criterion [11]:

$$|y_s - y_h| \leq \epsilon_a + \epsilon_r \cdot |y_h|$$

### 3.4 Divergence Metrics

When falsification occurs, the oracle reports:

| Metric | Description | Diagnostic Value |
|--------|-------------|------------------|
| `max_diff` | Maximum absolute difference | Identifies outliers |
| `max_diff_idx` | Index of maximum divergence | Locates problem tensor element |
| `mean_diff` | Mean absolute difference | Systematic vs random error |
| `num_mismatches` | Count of elements exceeding tolerance | Severity assessment |
| `mismatch_ratio` | `num_mismatches / total_elements` | Pass/fail threshold |

## 4. Verification Matrix Extension

The HF Parity Oracle adds a new verification class to the existing matrix:

### 4.1 Class VII: Cross-Implementation Parity (P1 - HIGH)

| Gate ID | Hypothesis | Falsification Criteria | Points |
|---------|------------|------------------------|--------|
| **F-HFP-001** | **Logit Parity** | Final logits diverge from HF by > $\epsilon$ on >1% of elements | 10 |
| **F-HFP-002** | **Token Parity** | Greedy-decoded tokens differ from HF reference | 10 |
| **F-HFP-003** | **Attention Parity** | Attention weights diverge by > $10\epsilon$ (accumulated error) | 5 |
| **F-HFP-004** | **Hidden State Parity** | Per-layer hidden states diverge by > $\epsilon$ | 5 |
| **F-HFP-005** | **Embedding Parity** | Token embeddings diverge from HF embedding layer | 5 |
| **F-HFP-006** | **KV Cache Parity** | Cached keys/values diverge after N tokens | 5 |

**Total: 40 points** (added to existing 170-point matrix → 210 points)

### 4.2 Systematic Bias Detection

Beyond element-wise comparison, the oracle detects **systematic bias** [12]:

| Bias Type | Detection Method | Indicates |
|-----------|------------------|-----------|
| **Mean shift** | $|\mu_s - \mu_h| > 3\sigma$ | Normalization bug |
| **Scale drift** | $|\sigma_s / \sigma_h - 1| > 0.1$ | Scaling factor error |
| **Correlation loss** | $\rho(y_s, y_h) < 0.99$ | Layout/transpose bug |
| **Truncation** | $y_s$ clipped while $y_h$ not | Overflow handling |

## 5. Ground Truth Corpus Generation

### 5.1 Input Selection Strategy

Following **adversarial testing** principles [13], inputs are selected to maximize falsification potential:

| Category | Examples | Rationale |
|----------|----------|-----------|
| **Boundary** | Empty string, max context length | Edge cases |
| **Numerical** | "2+2=", "Calculate 1/3" | Precision-sensitive |
| **Unicode** | CJK, RTL, emoji | Tokenizer edge cases |
| **Code** | Multi-language snippets | Structured output |
| **Repetitive** | "the the the..." | Attention pattern stress |
| **Adversarial** | Known jailbreak prompts | Safety alignment |

### 5.2 Model Coverage

Initial corpus covers:

| Family | Models | Priority |
|--------|--------|----------|
| **Llama** | Llama-2-7B, Llama-3-8B | P0 |
| **Qwen** | Qwen2.5-Coder-1.5B | P0 |
| **Mistral** | Mistral-7B-v0.1 | P1 |
| **Phi** | Phi-3-mini | P1 |
| **Whisper** | whisper-tiny, whisper-base | P0 |
| **BERT** | bert-base-uncased | P2 |

### 5.3 Corpus Maintenance

The corpus is **append-only** with versioning:

```
oracle/
├── llama-2-7b/
│   ├── v1/           # transformers 4.38.0
│   │   ├── {hash1}.safetensors
│   │   └── manifest.json
│   └── v2/           # transformers 4.40.0
│       └── ...
└── corpus-metadata.json
```

**Regeneration triggers:**
1. HuggingFace transformers major version bump
2. Model architecture update (e.g., Llama 2 → Llama 3)
3. Discovered bug in golden generation

## 6. Integration with Playbook Infrastructure

### 6.1 Playbook Configuration

```yaml
# playbooks/models/llama-2-7b-parity.playbook.yaml
metadata:
  name: "Llama-2-7B HF Parity"
  tier: full

oracles:
  - type: hf_parity
    config:
      corpus_path: "${HF_GROUND_TRUTH}/oracle/llama-2-7b"
      tolerance:
        fp32: { atol: 1e-5, rtol: 1e-4 }
        quantized: { atol: 1e-2 }

scenarios:
  - prompt_set: "standard_bench"
    formats: [safetensors, apr, gguf]
    backends: [cpu, gpu]
```

### 6.2 CLI Usage

**Standalone Parity Command:**

```bash
# List available golden outputs
apr-qa parity --model-family qwen2.5-coder-1.5b/v1 --list

# Self-check: verify golden outputs match themselves (sanity check)
apr-qa parity --model-family qwen2.5-coder-1.5b/v1 --self-check

# Verify actual logits against golden
apr-qa parity --model-family qwen2.5-coder-1.5b/v1 \
  --logits-file output.safetensors \
  --prompt "def fibonacci(n):" \
  --tolerance fp16
```

**Integrated with Playbook Runs:**

```bash
# Run playbook with HF parity verification
apr-qa run playbooks/models/qwen-coder.yaml \
  --hf-parity \
  --hf-model-family qwen2.5-coder-1.5b/v1

# Full certification with parity (uses default corpus path)
apr-qa run playbooks/models/qwen-coder.yaml \
  --hf-parity \
  --hf-corpus-path ../hf-ground-truth-corpus/oracle \
  --hf-model-family qwen2.5-coder-1.5b/v1
```

**Make Targets:**

```bash
# Self-check all golden corpora (all model families)
make parity-check

# List available golden outputs
make parity-list

# Generate golden outputs (requires GPU + HuggingFace)
make golden-generate MODEL=Qwen/Qwen2.5-Coder-1.5B-Instruct VERSION=v1
```

**Generate golden outputs (requires uv + transformers):**

```bash
uv run scripts/generate_golden.py \
  --model Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --prompts prompts/code_bench.txt \
  --output ../hf-ground-truth-corpus/oracle/qwen2.5-coder-1.5b/v1/
```

### 6.3 Python Toolchain

**POLICY:** Only `uv`, `ty`, and `ruff` are permitted for Python tooling. No `pip`, `venv`, `virtualenv`, or `conda`.

| Tool | Purpose |
|------|---------|
| `uv` | Package management, script execution, lockfiles |
| `ty` | Type checking |
| `ruff` | Linting and formatting |

```bash
# Install dependencies
uv sync

# Run script with inline dependencies
uv run --with torch,transformers,safetensors scripts/generate_golden.py

# Type check
ty check scripts/

# Lint and format
ruff check scripts/
ruff format scripts/
```

### 6.3 Evidence Artifacts

Parity checks produce additional artifacts:

```
certifications/{model}/parity/
├── divergence_report.json    # Per-element diff statistics
├── correlation_matrix.png    # Visual correlation heatmap
├── layer_drift.csv           # Per-layer divergence tracking
└── falsification_log.json    # Detailed failure records
```

## 7. Failure Response Protocol

Following Jidoka's **5 Whys** methodology [3]:

### 7.1 Immediate Response (Andon)

1. **STOP:** Certification immediately halts
2. **CALL:** Alert generated with `TensorDiff` details
3. **WAIT:** No promotion to next tier until resolved

### 7.2 Root Cause Analysis

| Divergence Pattern | Likely Cause | Investigation |
|--------------------|--------------|---------------|
| Single large outlier | Numerical edge case | Check inf/nan handling |
| Systematic offset | Layout mismatch | Verify LAYOUT-002 compliance |
| Growing per-layer | Accumulated error | Check normalization |
| Random scatter | Non-determinism | Fix random seeds |
| GPU-only failure | Kernel bug | Compare GPU vs CPU path |

### 7.3 Resolution Workflow

```
Falsification
     │
     ▼
┌─────────────┐
│ Reproduce   │ ← Minimal failing test case
└─────────────┘
     │
     ▼
┌─────────────┐
│ Bisect      │ ← Layer-by-layer diff
└─────────────┘
     │
     ▼
┌─────────────┐
│ Fix         │ ← Code change
└─────────────┘
     │
     ▼
┌─────────────┐
│ Verify      │ ← Re-run parity oracle
└─────────────┘
     │
     ▼
┌─────────────┐
│ Add to CI   │ ← Regression test
└─────────────┘
```

## 8. Limitations and Future Work

### 8.1 Known Limitations

1. **Oracle Trust:** Assumes HuggingFace implementation is correct
2. **Coverage:** Cannot test all possible inputs (Popper's problem of induction)
3. **Version Drift:** HF updates may change "correct" outputs
4. **Performance:** Golden generation requires significant compute

### 8.2 Future Extensions

1. **Bidirectional Oracle:** Test HF against our outputs (mutual falsification)
2. **Fuzzing Integration:** Property-based input generation
3. **Continuous Monitoring:** Production divergence alerting
4. **Multi-Oracle Consensus:** Compare against vLLM, llama.cpp, etc.

## 9. References

[1] K. Popper, *The Logic of Scientific Discovery*, Routledge, 1959. (Original: *Logik der Forschung*, 1934). ISBN: 978-0415278447.

[2] T. Ohno, *Toyota Production System: Beyond Large-Scale Production*, Productivity Press, 1988. ISBN: 978-0915299140.

[3] J. K. Liker, *The Toyota Way: 14 Management Principles from the World's Greatest Manufacturer*, McGraw-Hill, 2004. ISBN: 978-0071392310.

[4] J. Pineau et al., "Improving Reproducibility in Machine Learning Research," *Journal of Machine Learning Research*, vol. 22, no. 164, pp. 1-20, 2021.

[5] O. E. Gundersen and S. Kjensmo, "State of the Art: Reproducibility in Artificial Intelligence," *Proceedings of the AAAI Conference on Artificial Intelligence*, vol. 32, no. 1, 2018.

[6] D. Goldberg, "What Every Computer Scientist Should Know About Floating-Point Arithmetic," *ACM Computing Surveys*, vol. 23, no. 1, pp. 5-48, 1991.

[7] NVIDIA, "Floating Point and IEEE 754 Compliance for NVIDIA GPUs," NVIDIA Developer Documentation, 2024.

[8] B. Jacob et al., "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference," *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 2704-2713, 2018.

[9] K. Chellapilla et al., "High Performance Convolutional Neural Networks for Document Processing," *Tenth International Workshop on Frontiers in Handwriting Recognition*, 2006.

[10] HuggingFace, "SafeTensors: A Simple, Safe Way to Store and Distribute Tensors," 2023. https://github.com/huggingface/safetensors

[11] NumPy Documentation, "numpy.allclose," NumPy Reference Guide, 2024.

[12] P. Henderson et al., "Deep Reinforcement Learning that Matters," *Proceedings of the AAAI Conference on Artificial Intelligence*, vol. 32, no. 1, 2018.

[13] I. J. Goodfellow et al., "Explaining and Harnessing Adversarial Examples," *International Conference on Learning Representations (ICLR)*, 2015.

---

## Appendix A: Popperian Glossary

| Term | Definition | Oracle Application |
|------|------------|-------------------|
| **Falsifiability** | A theory is scientific iff it can be refuted by observation | Parity hypothesis can be falsified by divergent output |
| **Corroboration** | Surviving severe tests without falsification | Model passes parity checks |
| **Verisimilitude** | Degree of truth-likeness | Higher coverage → higher confidence |
| **Severe Test** | Test designed to likely fail if hypothesis is false | Edge cases, adversarial inputs |
| **Basic Statement** | Singular observation statement | "Output tensor differs by 0.003 at index 4521" |

## Appendix B: Toyota Glossary

| Term | Japanese | Definition | Oracle Application |
|------|----------|------------|-------------------|
| **Jidoka** | 自働化 | Automation with human touch | Auto-detect divergence |
| **Andon** | 行灯 | Signal lamp for problems | `Falsified` result |
| **Poka-yoke** | ポカヨケ | Mistake-proofing | Tolerance thresholds |
| **Genchi Genbutsu** | 現地現物 | Go and see | Inspect actual tensors |
| **Kaizen** | 改善 | Continuous improvement | Expand corpus over time |
