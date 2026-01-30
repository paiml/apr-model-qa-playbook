# Running Tests

## CLI Usage

```bash
# Run a playbook
cargo run --bin apr-qa -- run <playbook.yaml>

# Generate report for a model
cargo run --bin apr-qa -- report <model-id>

# Validate playbook schema
cargo run --bin apr-qa -- validate <playbook.yaml>
```

## Execution Modes

### Simulate Mode (Default)

Fast execution without actual model inference. Useful for testing playbook structure.

### Subprocess Mode

Actual model inference via subprocess execution:

```bash
cargo run --bin apr-qa -- run playbook.yaml --mode subprocess
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
| `stop_on_p0` | Stop on P0 (gateway) failures only |
| `collect_all` | Run all tests, collect all failures |

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
