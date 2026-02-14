# HuggingFace Cache Resolution Specification

**Version:** 1.0.0
**Status:** IMPLEMENTED
**Author:** PAIML Engineering
**Date:** 2026-02-04
**Specification IDs:** HF-CACHE-001, HF-CACHE-002

---

## 1. Abstract

This specification defines automatic resolution of HuggingFace model repository IDs to local cache directories. The goal is **zero-setup model qualification**: users should be able to run playbooks without manually specifying `--model-path` if the model has already been downloaded via `huggingface-cli` or other HuggingFace tools.

## 2. Problem Statement

Playbooks declare models using HuggingFace repository IDs:

```yaml
model:
  hf_repo: "Qwen/Qwen2.5-Coder-0.5B-Instruct"
```

Previously, the runner required explicit `--model-path` flags, violating the zero-setup requirement. Users who had already downloaded models via HuggingFace tools had to manually locate cache directories and pass them as arguments.

## 3. Specification

### 3.1 HF-CACHE-001: Automatic HuggingFace Cache Resolution

**When:** `--model-path` is not provided to `apr-qa run`

**Then:** Resolve `playbook.model.hf_repo` to an actual model directory by searching (in order):

1. **HuggingFace cache:** `$HF_CACHE/models--{org}--{repo}/snapshots/*/model.safetensors`
2. **APR cache:** `~/.cache/apr-models/{org}/{repo}/`

**Returns:** The snapshot directory (for HF cache) or APR cache directory containing the model files.

**Error if:** Model not found in any cache location. The error message lists all searched paths for debugging.

### 3.2 HF-CACHE-002: Environment Variable Support

Per HuggingFace convention, the cache directory is determined by environment variables in priority order:

| Priority | Variable | Description |
|----------|----------|-------------|
| 1 (highest) | `$HUGGINGFACE_HUB_CACHE` | Direct cache path override |
| 2 | `$HF_HOME/hub` | HuggingFace home directory with `/hub` suffix |
| 3 (default) | `~/.cache/huggingface/hub` | Standard HuggingFace cache location |

## 4. Implementation

### 4.1 Public API

```rust
/// Get the HuggingFace cache directory respecting environment variables.
pub fn get_hf_cache_dir() -> PathBuf;

/// Split a HuggingFace repo ID into (org, repo).
pub fn split_hf_repo(hf_repo: &str) -> (&str, &str);

/// Resolve a HuggingFace repo ID to a local cache directory.
pub fn resolve_hf_repo_to_cache(hf_repo: &str) -> Result<PathBuf>;
```

### 4.2 Files Modified

| File | Change |
|------|--------|
| `crates/apr-qa-runner/src/conversion.rs` | Added `get_hf_cache_dir()`, `split_hf_repo()`, `resolve_hf_repo_to_cache()` |
| `crates/apr-qa-runner/src/lib.rs` | Exported new functions |
| `crates/apr-qa-cli/src/main.rs` | Auto-resolution in `run_playbook()` |

### 4.3 Cache Directory Structure

**HuggingFace cache:**
```
~/.cache/huggingface/hub/
└── models--{org}--{repo}/
    └── snapshots/
        └── {commit_hash}/
            ├── model.safetensors
            ├── config.json
            └── tokenizer.json
```

**APR cache:**
```
~/.cache/apr-models/
└── {org}/
    └── {repo}/
        ├── safetensors/
        ├── gguf/
        └── apr/
```

## 5. Usage Examples

### 5.1 Zero-Setup Workflow

```bash
# 1. Download model via HuggingFace (one-time)
huggingface-cli download Qwen/Qwen2.5-Coder-0.5B-Instruct

# 2. Run playbook - model auto-resolved from HF cache
cargo run --bin apr-qa -- run playbooks/models/qwen2.5-coder-0.5b-mvp.playbook.yaml

# Output:
# Loading playbook: playbooks/models/qwen2.5-coder-0.5b-mvp.playbook.yaml
#   Auto-resolved model: /home/user/.cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-0.5B-Instruct/snapshots/abc123
# Running playbook: qwen2.5-coder-0.5b-mvp
```

### 5.2 Explicit Path (Override)

```bash
# Explicit path takes precedence over auto-resolution
cargo run --bin apr-qa -- run playbook.yaml --model-path /custom/path/to/model
```

### 5.3 Error Case

```bash
# If model not downloaded, helpful error with searched paths
cargo run --bin apr-qa -- run playbook.yaml

# Output:
# Warning: Model not found in cache: Qwen/Qwen2.5-Coder-0.5B-Instruct
# Searched:
#   - /home/user/.cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-0.5B-Instruct/snapshots
#   - /home/user/.cache/apr-models/Qwen/Qwen2.5-Coder-0.5B-Instruct
# Hint: Download model with `huggingface-cli download Qwen/Qwen2.5-Coder-0.5B-Instruct` or use --model-path
```

## 6. Testing

Tests cover:
- `split_hf_repo()` with and without org prefix
- `find_hf_snapshot()` finding snapshots with `model.safetensors`
- `find_apr_cache()` finding APR cache directories
- `resolve_hf_repo_with_dirs()` priority order (HF cache before APR cache)
- Error message formatting with searched paths

All tests use `tempfile::TempDir` to create isolated test environments without modifying global environment variables.

## 7. References

- [HuggingFace Hub Cache Documentation](https://huggingface.co/docs/huggingface_hub/guides/manage-cache)
- [LAYOUT-002: Row-Major Mandate](../../CLAUDE.md)
- Related: HF-PARITY-001 (HuggingFace Parity Oracle)
