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
cargo run --bin apr-qa -- run playbook.yaml --model-path /path/to/model
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

### Fail-Fast Mode (§12.5.3)

The `--fail-fast` flag is designed for debugging and GitHub ticket creation. It stops on the first failure and emits comprehensive diagnostic output:

```bash
# Basic fail-fast
apr-qa run playbook.yaml --fail-fast

# With full tracing for GitHub ticket
RUST_LOG=debug apr-qa run playbook.yaml --fail-fast 2>&1 | tee failure.log
```

**Output includes:**
- Gate ID and failure reason
- Model ID, format, and backend
- Full stderr capture
- Exit code
- Environment context (OS, versions, git commit)

**Use cases:**
- Debugging a specific test failure
- Bisecting a regression
- Creating GitHub issues with full context
- CI pipelines needing immediate failure notification

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
