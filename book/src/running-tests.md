# Running Tests

> **CRITICAL**: Always use `apr-qa` commands for model qualification. Never bypass the playbook infrastructure with manual `apr` commands.

## CLI Usage

```bash
# Certify models (RECOMMENDED)
cargo run --bin apr-qa -- certify --family qwen-coder --tier mvp

# Run a playbook
cargo run --bin apr-qa -- run <playbook.yaml>

# Generate report for a model
cargo run --bin apr-qa -- report <model-id>

# Validate playbook schema
cargo run --bin apr-qa -- validate <playbook.yaml>
```

## Execution

The playbook runner internally invokes the `apr` binary for inference. This is managed by the playbook infrastructure—do not call `apr` directly:

```bash
# Zero-setup: model auto-resolved from HuggingFace cache (HF-CACHE-001)
cargo run --bin apr-qa -- run playbook.yaml

# Or with explicit model path
cargo run --bin apr-qa -- run playbook.yaml --model-path /path/to/model
```

### HuggingFace Cache Resolution

When `--model-path` is not provided, the runner automatically resolves `playbook.model.hf_repo` to your local cache:

1. **HuggingFace cache** (checked first):
   - `$HUGGINGFACE_HUB_CACHE` or `$HF_HOME/hub` or `~/.cache/huggingface/hub`
   - Looks for `models--{org}--{repo}/snapshots/*/model.safetensors`

2. **APR cache** (fallback):
   - `~/.cache/apr-models/{org}/{repo}/`

Example output:
```
Loading playbook: playbooks/models/qwen2.5-coder-0.5b-mvp.playbook.yaml
  Auto-resolved model: /home/user/.cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-0.5B-Instruct/snapshots/abc123
Running playbook: qwen2.5-coder-0.5b-mvp
```

If the model is not found, you'll see searched paths and a hint:
```
Warning: Model not found in cache: Qwen/Qwen2.5-Coder-0.5B-Instruct
Searched:
  - /home/user/.cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-0.5B-Instruct/snapshots
  - /home/user/.cache/apr-models/Qwen/Qwen2.5-Coder-0.5B-Instruct
Hint: Download model with `huggingface-cli download Qwen/Qwen2.5-Coder-0.5B-Instruct` or use --model-path
```

## Parallel Execution

The runner uses Rayon for parallel execution:

```yaml
# In batuta-qa-pipeline.toml
[parallel]
max_workers = 8
timeout_ms = 60000
mode = "rayon"
```

## Failure Policies

| Policy | Behavior |
|--------|----------|
| `stop_on_first` | Stop immediately on any failure |
| `stop_on_p0` | Stop on P0 (gateway) failures only (default) |
| `collect_all` | Run all tests, collect all failures |
| `fail-fast` | Stop on first failure with enhanced diagnostics |

### Fail-Fast Mode (§12.5.3, FF-REPORT-001)

The `--fail-fast` flag is designed for debugging and GitHub ticket creation. It stops on the first failure and generates a comprehensive diagnostic report using apr's rich tooling.

```bash
# Basic fail-fast
apr-qa run playbook.yaml --fail-fast

# With full tracing for GitHub ticket
RUST_LOG=debug apr-qa run playbook.yaml --fail-fast 2>&1 | tee failure.log

# View generated report
cat output/fail-fast-report/summary.md
```

**On failure, generates `output/fail-fast-report/`:**

```
output/fail-fast-report/
├── summary.md           # GitHub-ready markdown
├── diagnostics.json     # Full machine-readable report
├── check.json           # apr check output (pipeline integrity)
├── inspect.json         # apr inspect output (model metadata)
├── trace.json           # apr trace output (layer analysis, .apr only)
├── tensors.json         # apr tensors output (tensor inventory)
├── environment.json     # OS, versions, git state
└── stderr.log           # Raw stderr capture
```

**Diagnostic commands run (with timeouts):**

| Stage | Command | Timeout | Purpose |
|-------|---------|---------|---------|
| 1 | `apr check <model> --json` | 30s | 10-stage pipeline integrity |
| 2 | `apr inspect <model> --json` | 10s | Metadata, vocab, structure |
| 3 | `apr trace <model> --payload --json` | 60s | Layer-by-layer analysis |
| 4 | `apr tensors <model> --json` | 10s | Tensor names and shapes |
| 5 | `apr explain <error-code>` | 5s | Human-readable explanation |

**Use cases:**
- Debugging a specific test failure
- Bisecting a regression
- Creating GitHub issues with full context (copy summary.md)
- CI pipelines needing immediate failure notification

## Playbook Integrity Lock (§3.1)

The playbook lock system prevents accidental or malicious modification of test specifications. When enabled (default), the runner verifies that playbook files haven't been modified since they were last locked.

### Generating the Lock File

```bash
# Lock all playbooks in the models directory
apr-qa lock-playbooks playbooks/models -o playbooks/playbook.lock.yaml

# Output: Locked 27 playbook(s) → playbooks/playbook.lock.yaml
```

The lock file contains SHA-256 hashes for each playbook:

```yaml
entries:
  qwen2.5-coder-1.5b-mvp:
    sha256: 8dbb1f48ca93a0948a560fb32a6febc37e2569040fc2aac2581dd5668cd3d7d2
    locked_fields:
    - model.hf_repo
    - model.formats
    - test_matrix
    - falsification_gates
```

### Integrity Verification

When running or certifying:

```bash
# Passes integrity check
apr-qa run playbook.yaml
# Output: Integrity check: PASSED

# If playbook was modified:
# [INTEGRITY] Playbook hash does not match lock file.
# [INTEGRITY] Either:
#   1. Run `apr-qa lock-playbooks` to regenerate the lock file
#   2. Use --no-integrity-check to bypass (NOT RECOMMENDED)
```

### When to Regenerate the Lock

Regenerate the lock file after intentional playbook changes:

```bash
# After modifying playbook specifications
apr-qa lock-playbooks playbooks/models

# Then commit both the playbook and lock file
git add playbooks/models/*.yaml playbooks/playbook.lock.yaml
git commit -m "feat(playbook): update test matrix"
```

**Why this matters:** The integrity lock implements Poka-Yoke (mistake-proofing) from the Toyota Production System. It prevents operators from silently weakening test specifications to make failures "go away."

## Evidence Collection

Test results are collected as `Evidence`:

```rust
pub enum Outcome {
    Corroborated,  // Hypothesis survived
    Falsified,     // Hypothesis refuted
    Timeout,       // Exceeded time limit
    Crashed,       // Process crashed
    Skipped,       // Test skipped
}
```

## Isolated Output Directories (ISO-OUT-001)

Conversion test artifacts are written to `output/conversions/` instead of polluting the HuggingFace cache:

```
output/
├── evidence.json          # Test results
├── report.html            # Dashboard
└── conversions/           # Conversion test artifacts
    └── {org}/{repo}/
        ├── basic/         # Basic conversion tests
        ├── semantic/      # Semantic comparison tests
        ├── idempotency/   # Double-convert stability
        └── round-trip/    # Multi-hop chain tests
```

This keeps `~/.cache/huggingface/` clean. Clean up with `rm -rf output/`.

## Output Formats

### JUnit XML

For CI/CD integration (Jenkins, GitHub Actions):

```bash
cargo run --bin apr-qa -- report --format junit -o report.xml
```

### HTML Dashboard

Interactive report with MQS visualization:

```bash
cargo run --bin apr-qa -- report --format html -o report/
```

### JSON Evidence

Raw evidence for further processing:

```bash
cargo run --bin apr-qa -- run playbook.yaml --evidence-dir evidence/
```
