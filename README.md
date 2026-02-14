# APR Model QA Playbook

<p align="center">
  <img src="assets/hero.svg" alt="APR Model QA Playbook" width="800">
</p>

<p align="center">
  <strong>Property-Based Model Qualification Testing for HuggingFace Models</strong>
</p>

<p align="center">
  <a href="#philosophy">Philosophy</a> •
  <a href="#features">Features</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#test-matrix">Test Matrix</a> •
  <a href="#mqs-scoring">MQS Scoring</a>
</p>

---

## Philosophy

This framework synthesizes two complementary quality paradigms:

### Toyota Production System (TPS)

> *"Stop the line. Fix it now. Never pass a defect to the next process."*
> — Taiichi Ohno

| Principle | Application |
|-----------|-------------|
| **Jidoka** | Execution halts on first P0 failure |
| **Poka-Yoke** | Schema validation prevents malformed playbooks |
| **Genchi Genbutsu** | All metrics from actual inference |
| **Heijunka** | Load-balanced parallel execution |
| **Kaizen** | Continuous refinement via mutation testing |

### Popperian Falsificationism

> *"The criterion of the scientific status of a theory is its falsifiability."*
> — Karl Popper

We don't test to pass—we **test to fail**. No amount of passing tests proves correctness, but a single failure proves a defect.

| Outcome | Meaning |
|---------|---------|
| `Corroborated` | Hypothesis survived refutation attempt |
| `Falsified` | Hypothesis refuted by evidence |
| `Timeout` | Execution exceeded time limit |
| `Crashed` | Process terminated abnormally |

## Features

- **Property-based testing** via proptest for comprehensive scenario generation
- **Parallel execution** with Rayon worker pools
- **Gateway checks (G0-G4)** that zero the score on critical failures
- **Model Qualification Score (MQS)** 0-1000 with grade mapping
- **JUnit XML and HTML reports** for CI/CD integration
- **Playbook YAML format** with JSON Schema validation
- **1.8M+ test assertions** across all model/format/backend combinations
- **217 falsification gates** across conversion, inference, patterns, and security domains

### New in v2.0.0

| Feature | Description |
|---------|-------------|
| **Two-Tier Certification** | MVP (≤10min, Grade B) and Full (≤1hr, Grade A+) tiers |
| **Tier-Aware Scoring** | `score_from_tier()`, `status_from_tier()`, `grade_from_tier()` |
| **Certify CLI Command** | `apr-qa certify --family qwen-coder --tier mvp` |
| **Rosetta Differential Testing** | Tensor layout mismatch, token comparison, fingerprint, stats validation |
| **Profile CI Mode** | Performance assertions for CI/CD (`--assert-throughput`, `--assert-p99`) |
| **Trace Payload Mode** | Real forward pass with NaN/Inf and garbage output detection |
| **Bug Pattern Detection** | 12 cross-project patterns from aprender/realizar analysis |

## Model Certifications

<!-- CERTIFICATION_TABLE_START -->
**Certification Summary** (updated: 2026-02-13 16:08 UTC)

| Status | Count |
|--------|-------|
| Certified | 9/92 |
| Provisional | 0/92 |
| Blocked | 4/92 |
| Pending | 79/92 |

**Priority Family:** Qwen Coder (see [Certified Testing Spec](docs/specifications/certified-testing.md))

| Model | Family | Size | Status | MQS | Grade | G1-4 | Prov | GGUF CPU | GGUF GPU | APR CPU | APR GPU | ST CPU | ST GPU |
|-------|--------|------|--------|-----|-------|------|------|----------|----------|---------|---------|--------|--------|
| [codegemma-7b-it](https://huggingface.co/google/codegemma-7b-it) | codegemma | 7B | ![pending](https://img.shields.io/badge/PENDING-lightgray) | 0 | - | - | - | - | - | - | - | - | - |
| [CodeLlama-7b-Instruct-hf](https://huggingface.co/meta-llama/CodeLlama-7b-Instruct-hf) | codellama | 7B | ![pending](https://img.shields.io/badge/PENDING-lightgray) | 0 | - | - | - | - | - | - | - | - | - |
| [CodeLlama-13b-Instruct-hf](https://huggingface.co/meta-llama/CodeLlama-13b-Instruct-hf) | codellama | 13B | ![pending](https://img.shields.io/badge/PENDING-lightgray) | 0 | - | - | - | - | - | - | - | - | - |
| [CodeLlama-34b-Instruct-hf](https://huggingface.co/meta-llama/CodeLlama-34b-Instruct-hf) | codellama | 34B | ![pending](https://img.shields.io/badge/PENDING-lightgray) | 0 | - | - | - | - | - | - | - | - | - |
| [CodeLlama-70b-Instruct-hf](https://huggingface.co/meta-llama/CodeLlama-70b-Instruct-hf) | codellama | 70B | ![pending](https://img.shields.io/badge/PENDING-lightgray) | 0 | - | - | - | - | - | - | - | - | - |
| [deepseek-coder-1.3b-instruct](https://huggingface.co/deepseek-ai/deepseek-coder-1.3b-instruct) | deepseek-coder | 1.3B | ![pending](https://img.shields.io/badge/PENDING-lightgray) | 0 | - | - | - | - | - | - | - | - | - |
| [deepseek-coder-6.7b-instruct](https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-instruct) | deepseek-coder | 6.7B | ![pending](https://img.shields.io/badge/PENDING-lightgray) | 0 | - | - | - | - | - | - | - | - | - |
| [deepseek-coder-7b-instruct](https://huggingface.co/deepseek-ai/deepseek-coder-7b-instruct) | deepseek-coder | 7B | ![pending](https://img.shields.io/badge/PENDING-lightgray) | 0 | - | - | - | - | - | - | - | - | - |
| [deepseek-coder-33b-instruct](https://huggingface.co/deepseek-ai/deepseek-coder-33b-instruct) | deepseek-coder | 33B | ![pending](https://img.shields.io/badge/PENDING-lightgray) | 0 | - | - | - | - | - | - | - | - | - |
| [DeepSeek-Coder-V2-Lite-Instruct](https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct) | deepseek-coder-v2 | 16B | ![pending](https://img.shields.io/badge/PENDING-lightgray) | 0 | - | - | - | - | - | - | - | - | - |
| [DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) | deepseek-r1 | 1.5B | ![certified](https://img.shields.io/badge/CERTIFIED-brightgreen) | 1000 | A | ✗ | ✗ | - | - | - | - | - | - |
| [DeepSeek-R1-Distill-Qwen-7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B) | deepseek-r1 | 7B | ![pending](https://img.shields.io/badge/PENDING-lightgray) | 0 | - | - | - | - | - | - | - | - | - |
| [DeepSeek-R1-Distill-Llama-8B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B) | deepseek-r1 | 8B | ![pending](https://img.shields.io/badge/PENDING-lightgray) | 0 | - | - | - | - | - | - | - | - | - |
| [DeepSeek-R1-Distill-Qwen-14B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B) | deepseek-r1 | 14B | ![pending](https://img.shields.io/badge/PENDING-lightgray) | 0 | - | - | - | - | - | - | - | - | - |
| [DeepSeek-R1-Distill-Qwen-32B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B) | deepseek-r1 | 32B | ![pending](https://img.shields.io/badge/PENDING-lightgray) | 0 | - | - | - | - | - | - | - | - | - |
| [DeepSeek-R1-Distill-Llama-70B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B) | deepseek-r1 | 70B | ![pending](https://img.shields.io/badge/PENDING-lightgray) | 0 | - | - | - | - | - | - | - | - | - |
| [dolphin-2.6-mistral-7b](https://huggingface.co/cognitivecomputations/dolphin-2.6-mistral-7b) | dolphin | 7B | ![pending](https://img.shields.io/badge/PENDING-lightgray) | 0 | - | - | - | - | - | - | - | - | - |
| [Dolphin3.0-Llama3.1-8B](https://huggingface.co/cognitivecomputations/Dolphin3.0-Llama3.1-8B) | dolphin | 8B | ![pending](https://img.shields.io/badge/PENDING-lightgray) | 0 | - | - | - | - | - | - | - | - | - |
| [falcon-7b-instruct](https://huggingface.co/tiiuae/falcon-7b-instruct) | falcon | 7B | ![pending](https://img.shields.io/badge/PENDING-lightgray) | 0 | - | - | - | - | - | - | - | - | - |
| [falcon-40b-instruct](https://huggingface.co/tiiuae/falcon-40b-instruct) | falcon | 40B | ![pending](https://img.shields.io/badge/PENDING-lightgray) | 0 | - | - | - | - | - | - | - | - | - |
| [gemma-2-2b-it](https://huggingface.co/google/gemma-2-2b-it) | gemma | 2B | ![pending](https://img.shields.io/badge/PENDING-lightgray) | 0 | - | - | - | - | - | - | - | - | - |
| [gemma-2-9b-it](https://huggingface.co/google/gemma-2-9b-it) | gemma2 | 9B | ![pending](https://img.shields.io/badge/PENDING-lightgray) | 0 | - | - | - | - | - | - | - | - | - |
| [gemma-2-27b-it](https://huggingface.co/google/gemma-2-27b-it) | gemma2 | 27B | ![pending](https://img.shields.io/badge/PENDING-lightgray) | 0 | - | - | - | - | - | - | - | - | - |
| [gemma-3-1b-it](https://huggingface.co/google/gemma-3-1b-it) | gemma3 | 1B | ![pending](https://img.shields.io/badge/PENDING-lightgray) | 0 | F | - | - | - | - | - | - | - | - |
| [gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it) | gemma3 | 4B | ![pending](https://img.shields.io/badge/PENDING-lightgray) | 0 | - | - | - | - | - | - | - | - | - |
| [gemma-3-12b-it](https://huggingface.co/google/gemma-3-12b-it) | gemma3 | 12B | ![pending](https://img.shields.io/badge/PENDING-lightgray) | 0 | - | - | - | - | - | - | - | - | - |
| [gemma-3-27b-it](https://huggingface.co/google/gemma-3-27b-it) | gemma3 | 27B | ![pending](https://img.shields.io/badge/PENDING-lightgray) | 0 | - | - | - | - | - | - | - | - | - |
| [granite-3.1-2b-instruct](https://huggingface.co/ibm-granite/granite-3.1-2b-instruct) | granite | 2B | ![pending](https://img.shields.io/badge/PENDING-lightgray) | 892 | C | - | - | - | - | - | - | - | - |
| [granite-3.1-8b-instruct](https://huggingface.co/ibm-granite/granite-3.1-8b-instruct) | granite | 8B | ![pending](https://img.shields.io/badge/PENDING-lightgray) | 0 | - | - | - | - | - | - | - | - | - |
| [granite-3b-code-instruct-128k](https://huggingface.co/ibm-granite/granite-3b-code-instruct-128k) | granite-code | 3B | ![pending](https://img.shields.io/badge/PENDING-lightgray) | 0 | - | - | - | - | - | - | - | - | - |
| [Hermes-3-Llama-3.1-8B](https://huggingface.co/NousResearch/Hermes-3-Llama-3.1-8B) | hermes | 8B | ![pending](https://img.shields.io/badge/PENDING-lightgray) | 0 | - | - | - | - | - | - | - | - | - |
| [internlm2_5-7b-chat](https://huggingface.co/internlm/internlm2_5-7b-chat) | internlm | 7B | ![pending](https://img.shields.io/badge/PENDING-lightgray) | 0 | - | - | - | - | - | - | - | - | - |
| [internlm2_5-20b-chat](https://huggingface.co/internlm/internlm2_5-20b-chat) | internlm | 20B | ![pending](https://img.shields.io/badge/PENDING-lightgray) | 0 | - | - | - | - | - | - | - | - | - |
| [Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) | llama | 1B | ![pending](https://img.shields.io/badge/PENDING-lightgray) | 0 | - | - | - | - | - | - | - | - | - |
| [Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) | llama | 3B | ![pending](https://img.shields.io/badge/PENDING-lightgray) | 0 | - | - | - | - | - | - | - | - | - |
| [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | llama | 8B | ![pending](https://img.shields.io/badge/PENDING-lightgray) | 0 | - | - | - | - | - | - | - | - | - |
| [Llama-3.1-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct) | llama | 70B | ![pending](https://img.shields.io/badge/PENDING-lightgray) | 0 | - | - | - | - | - | - | - | - | - |
| [Llama-3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) | llama | 70B | ![pending](https://img.shields.io/badge/PENDING-lightgray) | 0 | - | - | - | - | - | - | - | - | - |
| [Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) | mistral | 7B | ![pending](https://img.shields.io/badge/PENDING-lightgray) | 0 | - | - | - | - | - | - | - | - | - |
| [Mistral-Nemo-Instruct-2407](https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407) | mistral | 12B | ![pending](https://img.shields.io/badge/PENDING-lightgray) | 0 | - | - | - | - | - | - | - | - | - |
| [Mistral-Small-24B-Instruct-2501](https://huggingface.co/mistralai/Mistral-Small-24B-Instruct-2501) | mistral | 24B | ![pending](https://img.shields.io/badge/PENDING-lightgray) | 0 | - | - | - | - | - | - | - | - | - |
| [Codestral-22B-v0.1](https://huggingface.co/mistralai/Codestral-22B-v0.1) | mistral-code | 22B | ![pending](https://img.shields.io/badge/PENDING-lightgray) | 0 | - | - | - | - | - | - | - | - | - |
| [Llama-3.1-Nemotron-Nano-4B-v1.1](https://huggingface.co/nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1) | nemotron | 4B | ![pending](https://img.shields.io/badge/PENDING-lightgray) | 122 | F | - | - | - | - | - | - | - | - |
| [Llama-3.1-Nemotron-70B-Instruct-HF](https://huggingface.co/nvidia/Llama-3.1-Nemotron-70B-Instruct-HF) | nemotron | 70B | ![pending](https://img.shields.io/badge/PENDING-lightgray) | 0 | - | - | - | - | - | - | - | - | - |
| [OLMo-2-1124-7B-Instruct](https://huggingface.co/allenai/OLMo-2-1124-7B-Instruct) | olmo | 7B | ![pending](https://img.shields.io/badge/PENDING-lightgray) | 0 | - | - | - | - | - | - | - | - | - |
| [OLMo-2-1124-13B-Instruct](https://huggingface.co/allenai/OLMo-2-1124-13B-Instruct) | olmo | 13B | ![pending](https://img.shields.io/badge/PENDING-lightgray) | 0 | - | - | - | - | - | - | - | - | - |
| [openchat-3.5-0106](https://huggingface.co/openchat/openchat-3.5-0106) | openchat | 7B | ![pending](https://img.shields.io/badge/PENDING-lightgray) | 0 | - | - | - | - | - | - | - | - | - |
| [OpenHermes-2.5-Mistral-7B](https://huggingface.co/teknium/OpenHermes-2.5-Mistral-7B) | openhermes | 7B | ![pending](https://img.shields.io/badge/PENDING-lightgray) | 0 | - | - | - | - | - | - | - | - | - |
| [Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) | phi | 3.8B | ![pending](https://img.shields.io/badge/PENDING-lightgray) | 0 | - | - | - | - | - | - | - | - | - |
| [Phi-3.5-mini-instruct](https://huggingface.co/microsoft/Phi-3.5-mini-instruct) | phi | 3.8B | ![pending](https://img.shields.io/badge/PENDING-lightgray) | 0 | - | - | - | - | - | - | - | - | - |
| [Phi-3-small-8k-instruct](https://huggingface.co/microsoft/Phi-3-small-8k-instruct) | phi | 7B | ![pending](https://img.shields.io/badge/PENDING-lightgray) | 0 | - | - | - | - | - | - | - | - | - |
| [Phi-3-medium-4k-instruct](https://huggingface.co/microsoft/Phi-3-medium-4k-instruct) | phi | 14B | ![pending](https://img.shields.io/badge/PENDING-lightgray) | 0 | - | - | - | - | - | - | - | - | - |
| [Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct) | phi4 | 3.8B | ![pending](https://img.shields.io/badge/PENDING-lightgray) | 607 | D | - | - | - | - | - | - | - | - |
| [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) | qwen | 0.5B | ![pending](https://img.shields.io/badge/PENDING-lightgray) | 0 | - | - | - | - | - | - | - | - | - |
| [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) | qwen | 1.5B | ![certified](https://img.shields.io/badge/CERTIFIED-brightgreen) | 1000 | A | ✗ | ✗ | - | - | - | - | - | - |
| [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) | qwen | 3B | ![certified](https://img.shields.io/badge/CERTIFIED-brightgreen) | 964 | A | ✗ | ✗ | - | - | - | - | - | - |
| [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | qwen | 7B | ![certified](https://img.shields.io/badge/CERTIFIED-brightgreen) | 900 | B | ✗ | ✗ | - | - | - | - | - | - |
| [Qwen2.5-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct) | qwen | 14B | ![pending](https://img.shields.io/badge/PENDING-lightgray) | 0 | - | - | - | - | - | - | - | - | - |
| [Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct) | qwen | 32B | ![pending](https://img.shields.io/badge/PENDING-lightgray) | 0 | - | - | - | - | - | - | - | - | - |
| [QwQ-32B](https://huggingface.co/Qwen/QwQ-32B) | qwen | 32B | ![pending](https://img.shields.io/badge/PENDING-lightgray) | 0 | - | - | - | - | - | - | - | - | - |
| [Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct) | qwen | 72B | ![pending](https://img.shields.io/badge/PENDING-lightgray) | 0 | - | - | - | - | - | - | - | - | - |
| [Qwen2.5-Coder-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct) | qwen-coder | 0.5B | ![pending](https://img.shields.io/badge/PENDING-lightgray) | 500 | F | - | - | - | - | - | - | - | - |
| [Qwen2.5-Coder-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct) | qwen-coder | 1.5B | ![pending](https://img.shields.io/badge/PENDING-lightgray) | 551 | F | - | - | 17.9 | 129.8 | 16.2 | 0.6 | 2.9 | 23.8 |
| [Qwen2.5-Coder-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct) | qwen-coder | 3B | ![blocked](https://img.shields.io/badge/BLOCKED-red) | 283 | - | ✗ | ✗ | 11.2 | 66.2 | 10.7 | 0.4 | 0.0 | 0.0 |
| [Qwen2.5-Coder-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct) | qwen-coder | 7B | ![blocked](https://img.shields.io/badge/BLOCKED-red) | 152 | - | ✗ | ✗ | 8.2 | 30.2 | 8.5 | 0.0 | 0.0 | 0.0 |
| [Qwen2.5-Coder-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct) | qwen-coder | 14B | ![blocked](https://img.shields.io/badge/BLOCKED-red) | 152 | - | ✗ | ✗ | 4.7 | 15.1 | 2.3 | 0.0 | 0.0 | 0.0 |
| [Qwen2.5-Coder-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct) | qwen-coder | 32B | ![blocked](https://img.shields.io/badge/BLOCKED-red) | 0 | F | ✗ | ✗ | - | - | - | - | - | - |
| [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | qwen3 | 0.6B | ![pending](https://img.shields.io/badge/PENDING-lightgray) | 159 | F | - | - | - | - | - | - | - | - |
| [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) | qwen3 | 1.7B | ![certified](https://img.shields.io/badge/CERTIFIED-brightgreen) | 964 | A | ✗ | ✗ | - | - | - | - | - | - |
| [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) | qwen3 | 4B | ![pending](https://img.shields.io/badge/PENDING-lightgray) | 127 | F | - | - | - | - | - | - | - | - |
| [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | qwen3 | 8B | ![pending](https://img.shields.io/badge/PENDING-lightgray) | 0 | - | - | - | - | - | - | - | - | - |
| [Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B) | qwen3 | 14B | ![pending](https://img.shields.io/badge/PENDING-lightgray) | 0 | - | - | - | - | - | - | - | - | - |
| [Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) | qwen3 | 32B | ![pending](https://img.shields.io/badge/PENDING-lightgray) | 0 | - | - | - | - | - | - | - | - | - |
| [Qwen3-Coder-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct) | qwen3-coder-moe | 30B | ![pending](https://img.shields.io/badge/PENDING-lightgray) | 0 | - | - | - | - | - | - | - | - | - |
| [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) | qwen3-moe | 30B | ![pending](https://img.shields.io/badge/PENDING-lightgray) | 0 | - | - | - | - | - | - | - | - | - |
| [SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) | smollm | 135M | ![certified](https://img.shields.io/badge/CERTIFIED-brightgreen) | 925 | B | ✗ | ✗ | - | - | - | - | - | - |
| [SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) | smollm | 360M | ![certified](https://img.shields.io/badge/CERTIFIED-brightgreen) | 925 | B | ✗ | ✗ | - | - | - | - | - | - |
| [SmolLM2-1.7B-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct) | smollm | 1.7B | ![certified](https://img.shields.io/badge/CERTIFIED-brightgreen) | 925 | B | ✗ | ✗ | - | - | - | - | - | - |
| [stablelm-2-zephyr-1_6b](https://huggingface.co/stabilityai/stablelm-2-zephyr-1_6b) | stablelm | 1.6B | ![pending](https://img.shields.io/badge/PENDING-lightgray) | 0 | - | - | - | - | - | - | - | - | - |
| [stablelm-zephyr-3b](https://huggingface.co/stabilityai/stablelm-zephyr-3b) | stablelm | 3B | ![pending](https://img.shields.io/badge/PENDING-lightgray) | 0 | - | - | - | - | - | - | - | - | - |
| [starcoder2-3b](https://huggingface.co/bigcode/starcoder2-3b) | starcoder2 | 3B | ![pending](https://img.shields.io/badge/PENDING-lightgray) | 0 | - | - | - | - | - | - | - | - | - |
| [starcoder2-7b](https://huggingface.co/bigcode/starcoder2-7b) | starcoder2 | 7B | ![pending](https://img.shields.io/badge/PENDING-lightgray) | 0 | - | - | - | - | - | - | - | - | - |
| [starcoder2-15b](https://huggingface.co/bigcode/starcoder2-15b) | starcoder2 | 15B | ![pending](https://img.shields.io/badge/PENDING-lightgray) | 0 | - | - | - | - | - | - | - | - | - |
| [TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) | tinyllama | 1.1B | ![certified](https://img.shields.io/badge/CERTIFIED-brightgreen) | 1000 | A | ✗ | ✗ | - | - | - | - | - | - |
| [vicuna-7b-v1.5](https://huggingface.co/lmsys/vicuna-7b-v1.5) | vicuna | 7B | ![pending](https://img.shields.io/badge/PENDING-lightgray) | 0 | - | - | - | - | - | - | - | - | - |
| [vicuna-13b-v1.5](https://huggingface.co/lmsys/vicuna-13b-v1.5) | vicuna | 13B | ![pending](https://img.shields.io/badge/PENDING-lightgray) | 0 | - | - | - | - | - | - | - | - | - |
| [WizardCoder-15B-V1.0](https://huggingface.co/WizardLMTeam/WizardCoder-15B-V1.0) | wizardcoder | 15B | ![pending](https://img.shields.io/badge/PENDING-lightgray) | 0 | - | - | - | - | - | - | - | - | - |
| [WizardCoder-33B-V1.1](https://huggingface.co/WizardLMTeam/WizardCoder-33B-V1.1) | wizardcoder | 33B | ![pending](https://img.shields.io/badge/PENDING-lightgray) | 0 | - | - | - | - | - | - | - | - | - |
| [Yi-1.5-6B-Chat](https://huggingface.co/01-ai/Yi-1.5-6B-Chat) | yi | 6B | ![pending](https://img.shields.io/badge/PENDING-lightgray) | 0 | - | - | - | - | - | - | - | - | - |
| [Yi-1.5-9B-Chat](https://huggingface.co/01-ai/Yi-1.5-9B-Chat) | yi | 9B | ![pending](https://img.shields.io/badge/PENDING-lightgray) | 0 | - | - | - | - | - | - | - | - | - |
| [Yi-1.5-34B-Chat](https://huggingface.co/01-ai/Yi-1.5-34B-Chat) | yi | 34B | ![pending](https://img.shields.io/badge/PENDING-lightgray) | 0 | - | - | - | - | - | - | - | - | - |
| [zephyr-7b-beta](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta) | zephyr | 7B | ![pending](https://img.shields.io/badge/PENDING-lightgray) | 0 | - | - | - | - | - | - | - | - | - |
<!-- CERTIFICATION_TABLE_END -->

## Quick Start

```bash
# Build all crates
make build

# Run all tests
make test

# Generate coverage report
make coverage

# Certify models (recommended)
cargo run --bin apr-qa -- certify --family qwen-coder --tier mvp

# Run a specific playbook
cargo run --bin apr-qa -- run playbooks/models/qwen2.5-coder-1.5b-mvp.playbook.yaml
```

### Certification Tiers

| Tier | Time | Description | Pass → Grade / Status |
|------|------|-------------|----------------------|
| **Smoke** | ~1-2 min | Sanity check (minimal matrix) | Dev feedback only |
| **MVP** | ~5-10 min | All formats × backends × modalities (18 combos) | ≥90% → B / PROVISIONAL |
| **Quick** | ~10-30 min | Dev iteration with broader coverage | Dev feedback |
| **Standard** | ~1-2 hr | CI/CD gate | CI gate |
| **Deep** | ~8-24 hr | Production qualification (full matrix) | ≥95% → A+ / CERTIFIED |

```bash
# Smoke check (fastest)
cargo run --bin apr-qa -- certify --family qwen-coder --tier smoke

# MVP certification (quick surface coverage)
cargo run --bin apr-qa -- certify --family qwen-coder --tier mvp

# Deep certification (production qualification)
cargo run --bin apr-qa -- certify --family qwen-coder --tier deep
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          APR-MODEL-QA-PLAYBOOK                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│  │ apr-qa-gen   │    │ apr-qa-runner│    │apr-qa-report │                  │
│  │              │───▶│              │───▶│              │                  │
│  │ • proptest   │    │ • parallel   │    │ • MQS score  │                  │
│  │ • scenarios  │    │ • execution  │    │ • JUnit XML  │                  │
│  │ • oracles    │    │ • evidence   │    │ • HTML/MD    │                  │
│  │ • kernels    │    │              │    │              │                  │
│  │ • bootstrap  │    │              │    │              │                  │
│  └──────────────┘    └──────────────┘    └──────────────┘                  │
│         │                    │                    │                          │
│         └────────────────────┼────────────────────┘                          │
│                              ▼                                               │
│  ┌──────────────┐    ┌──────────────┐                                       │
│  │apr-qa-certify│    │ apr-qa-cli   │                                       │
│  │              │◀───│              │                                       │
│  │ • tier score │    │ • certify    │                                       │
│  │ • README sync│    │ • run/report │                                       │
│  │ • CSV export │    │ • Jidoka sigs│                                       │
│  └──────────────┘    └──────────────┘                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Crate Structure

| Crate | Purpose |
|-------|---------|
| `apr-qa-gen` | Scenario generation with proptest, oracle definitions, kernel profiles, playbook bootstrapping |
| `apr-qa-runner` | Playbook execution, differential testing, bug patterns |
| `apr-qa-report` | MQS scoring, JUnit/HTML report generation |
| `apr-qa-certify` | Two-tier certification, README sync, tier-aware scoring |
| `apr-qa-cli` | Command-line interface |

### Key Modules (apr-qa-runner)

| Module | Purpose |
|--------|---------|
| `executor.rs` | Scenario execution engine |
| `parallel.rs` | Rayon-based parallel execution with Jidoka enforcement |
| `playbook.rs` | YAML playbook parsing and validation |
| `conversion.rs` | Format conversion testing with bug classification |
| `differential.rs` | Rosetta diff-tensors, compare-inference, profile CI |
| `patterns.rs` | Cross-project bug pattern detection (12 patterns) |
| `contract.rs` | Generic contract validation |
| `family_contract.rs` | Family YAML alignment checks |
| `layout_contract.rs` | LAYOUT-002 row-major tensor validation |
| `integrity.rs` | config.json and model integrity (G0 gateway) |
| `provenance.rs` | Git/file provenance tracking |
| `evidence.rs` | Evidence collection and serialization |
| `oracle.rs` | Oracle execution layer |
| `command.rs` | Process execution wrapper |
| `diagnostics.rs` | Debugging and diagnostic output |
| `process.rs` | Jidoka process lifecycle management |

## Test Matrix

The framework tests models across multiple dimensions:

| Dimension | Options |
|-----------|---------|
| **Modality** | `run`, `chat`, `serve` |
| **Backend** | `cpu`, `gpu` |
| **Format** | `safetensors` (ground truth), `apr`, `gguf` |
| **Quantization** | `q4_k_m`, `q5_k_m`, `q8_0`, `f16`, `f32` |

> **Ground Truth**: SafeTensors is the source of truth for model weights (native HuggingFace format). APR is our optimized native format. GGUF is a supported third-party format.

With 100 scenarios per combination across 100 HuggingFace models:
- 3 modalities × 2 backends × 3 formats × 100 models × 100 scenarios = **1,800,000 tests**

## MQS Scoring

The **Model Qualification Score (MQS)** ranges from 0-1000:

### Gateway Checks (G0-G4)

Any gateway failure **zeros the entire score**:

| Gateway | Check | Failure Impact |
|---------|-------|----------------|
| **G0** | config.json matches tensor metadata | MQS = 0 |
| **G1** | Model loads successfully | MQS = 0 |
| **G2** | Basic inference works | MQS = 0 |
| **G3** | No crashes or panics | MQS = 0 |
| **G4** | Output is not garbage | MQS = 0 |

### Tier-Aware Scoring

The scoring system uses tier-aware functions:

| Tier | Pass Threshold | Score on Pass | Grade | Status |
|------|----------------|---------------|-------|--------|
| **MVP** | ≥90% | 800 | B | PROVISIONAL |
| **Full** | ≥95% | 950+ | A+ | CERTIFIED |

### Grade Mapping

| Score | Grade | Status |
|-------|-------|--------|
| 950-1000 | A+ | CERTIFIED |
| 900-949 | A | CERTIFIED |
| 850-899 | B+ | CERTIFIED |
| 800-849 | B | PROVISIONAL |
| 700-799 | C | PROVISIONAL |
| 0-699 | F | BLOCKED |

## Playbook Format

```yaml
version: "1.0"
model:
  id: "Qwen/Qwen2.5-Coder-1.5B"
  revision: "main"

test_matrix:
  modalities: [run, chat]
  backends: [cpu, gpu]
  formats: [safetensors, apr, gguf]  # safetensors is ground truth

scenarios:
  - name: "arithmetic_basic"
    prompt: "What is 2 + 2?"
    oracle: arithmetic
    expected: 4

  - name: "code_generation"
    prompt: "Write a Python function to reverse a string"
    oracle: code_syntax
    language: python

# Differential Testing (v1.3.0)
differential_tests:
  tensor_diff:
    enabled: true
    filter: "embed,lm_head"
    gates: ["F-ROSETTA-DIFF-001"]
  inference_compare:
    enabled: true
    prompt: "What is 2+2?"
    tolerance: 1e-5

# Profile CI Assertions (v1.3.0)
profile_ci:
  enabled: true
  assertions:
    min_throughput: 10.0  # tok/s
    max_p99_ms: 500       # ms

# Trace Payload (v1.3.0)
trace_payload:
  enabled: true
  gates: ["F-TRACE-PAYLOAD-001", "F-TRACE-PAYLOAD-002"]
```

## Project Structure

```
apr-model-qa-playbook/
├── crates/
│   ├── apr-qa-gen/        # Scenario generation + oracles + kernel profiles + bootstrapper
│   ├── apr-qa-runner/     # Playbook execution (Rayon parallel, 16 modules)
│   ├── apr-qa-report/     # MQS scoring + JUnit/HTML/Markdown reports
│   ├── apr-qa-certify/    # Tier-aware scoring, README sync, CSV export
│   └── apr-qa-cli/        # CLI binary (14 subcommands)
├── certifications/        # Model certification evidence (39 models)
│   └── <model>/evidence.json
├── playbooks/
│   ├── models/            # Per-model playbooks (117 YAML files)
│   ├── templates/         # Reusable templates (smoke, mvp, quick, standard, deep)
│   ├── verify/            # Ticket verification
│   └── spec/              # Executable specifications
├── book/                  # mdBook documentation
├── scripts/               # Validation and golden output generation
└── docs/
    ├── certifications/    # models.csv certification database (92 models)
    ├── specifications/    # Full specification (10 docs)
    ├── tickets/           # Ticket analysis (GH-190, GH-191)
    ├── five-whys/         # Root cause analysis
    ├── workflows/         # Certification workflow guides
    └── troubleshooting/   # Debugging guides
```

## Development

```bash
# Run tests with coverage
make coverage

# Verify PMAT compliance (>= 95%)
make coverage-check

# Lint with clippy
make lint

# Full check (fmt + lint + test)
make check
```

## License

MIT License - see [LICENSE](LICENSE) for details.

---

<p align="center">
  Built with Rust • Powered by proptest • Inspired by Toyota & Popper
</p>
