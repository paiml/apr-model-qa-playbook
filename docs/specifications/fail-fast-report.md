# Fail-Fast Diagnostic Report Specification

## Overview

When `--fail-fast` is enabled and a test fails, generate a comprehensive diagnostic report using apr's rich tooling. This report is designed for immediate GitHub issue creation with full reproduction context.

## Specification

### FF-REPORT-001: Structured Diagnostic Collection

On first failure, collect diagnostics using apr CLI tools:

| Stage | Command | Purpose |
|-------|---------|---------|
| 1. Check | `apr check <model> --json` | 10-stage pipeline integrity |
| 2. Inspect | `apr inspect <model> --json` | Metadata, vocab, structure |
| 3. Trace | `apr trace <model> --payload --json` | Layer-by-layer analysis |
| 4. Tensors | `apr tensors <model> --json` | Tensor names and shapes |
| 5. Explain | `apr explain <error-code>` | Human-readable error explanation |

### FF-REPORT-002: Report Structure

```
output/fail-fast-report/
├── summary.md           # Human-readable summary for GitHub
├── diagnostics.json     # Machine-readable full diagnostics
├── check.json           # apr check output
├── inspect.json         # apr inspect output
├── trace.json           # apr trace output (if applicable)
├── tensors.json         # apr tensors output
├── stderr.log           # Raw stderr capture
└── environment.json     # System context
```

### FF-REPORT-003: Summary Markdown Format

```markdown
# Fail-Fast Report: {gate_id}

## Failure Summary

| Field | Value |
|-------|-------|
| Gate | {gate_id} |
| Model | {model_hf_repo} |
| Format | {format} |
| Backend | {backend} |
| Exit Code | {exit_code} |
| Duration | {duration_ms}ms |

## Environment

| Field | Value |
|-------|-------|
| OS | {os} {arch} |
| apr-qa | {version} |
| apr-cli | {apr_version} |
| Git | {commit} ({branch}) |
| Rust | {rustc_version} |

## Pipeline Check Results

{apr_check_summary}

## Model Metadata

{apr_inspect_summary}

## Error Analysis

{apr_explain_output}

## Stderr Capture

```
{stderr}
```

## Reproduction

```bash
# Reproduce this failure
apr-qa run {playbook} --fail-fast --model-path {model_path}

# Run diagnostics manually
apr check {model_path}
apr trace {model_path} --payload -v
apr explain {error_code}
```
```

### FF-REPORT-004: Diagnostic Stages

#### Stage 1: apr check (Pipeline Integrity)

The 10-stage check validates:
1. File header magic
2. Version compatibility
3. Tensor count
4. Tensor metadata
5. Tokenizer presence
6. Config validity
7. Weight shapes
8. Quantization metadata
9. Embedding dimensions
10. Inference smoke test

On failure, identify which stage failed and include in report.

#### Stage 2: apr inspect (Metadata)

Capture:
- Architecture (e.g., "Qwen2ForCausalLM")
- Hidden size, attention heads, KV heads
- Vocabulary size
- Tokenizer type
- Quantization type (if any)

#### Stage 3: apr trace (Layer Analysis)

If failure is inference-related:
- Trace first 3 layers
- Capture activation statistics
- Identify NaN/Inf values
- Detect shape mismatches

#### Stage 4: apr tensors (Weight Inventory)

Capture:
- Total tensor count
- Missing expected tensors
- Unexpected tensor names
- Shape anomalies

#### Stage 5: apr explain (Error Interpretation)

Map error codes to explanations:
- `E-LOAD-001`: Tokenizer missing
- `E-LOAD-002`: Config missing
- `E-INFER-001`: Shape mismatch
- `E-CONV-001`: Conversion corruption
- etc.

### FF-REPORT-005: JSON Diagnostics Schema

```json
{
  "version": "1.0.0",
  "timestamp": "2024-02-04T18:00:00Z",
  "failure": {
    "gate_id": "G3-STABLE",
    "model": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    "format": "Apr",
    "backend": "Cpu",
    "outcome": "Crashed",
    "reason": "Process crashed with exit code -1",
    "exit_code": -1,
    "duration_ms": 52740
  },
  "environment": {
    "os": "linux",
    "arch": "x86_64",
    "apr_qa_version": "0.1.0",
    "apr_cli_version": "0.2.12",
    "git_commit": "408d121",
    "git_branch": "main",
    "rustc_version": "1.93.0"
  },
  "diagnostics": {
    "check": { /* apr check --json output */ },
    "inspect": { /* apr inspect --json output */ },
    "trace": { /* apr trace --json output */ },
    "tensors": { /* apr tensors --json output */ }
  },
  "stderr": "...",
  "reproduction": {
    "command": "apr-qa run playbook.yaml --fail-fast",
    "model_path": "/path/to/model"
  }
}
```

### FF-REPORT-006: Timeout Handling

Each diagnostic command has a timeout:
- `apr check`: 30s
- `apr inspect`: 10s
- `apr trace`: 60s
- `apr tensors`: 10s
- `apr explain`: 5s

If a diagnostic times out, note it in the report and continue.

### FF-REPORT-007: GitHub Issue Template

Generate issue body suitable for direct paste:

```markdown
### Bug Report: {gate_id} Failure

**Model:** {model}
**Format:** {format}
**Backend:** {backend}

#### Summary
{one_line_summary}

#### Environment
- OS: {os}
- apr-cli: {version}
- Git: {commit}

#### Reproduction
\`\`\`bash
apr-qa run {playbook} --fail-fast
\`\`\`

#### Diagnostics
<details>
<summary>apr check output</summary>

\`\`\`json
{check_output}
\`\`\`
</details>

<details>
<summary>Full stderr</summary>

\`\`\`
{stderr}
\`\`\`
</details>

#### Attachments
- [diagnostics.json](link)
- [full report](link)
```

## Implementation

### Files to Modify

| File | Change |
|------|--------|
| `crates/apr-qa-runner/src/executor.rs` | Add `FailFastReporter` struct |
| `crates/apr-qa-runner/src/diagnostics.rs` | New module for diagnostic collection |
| `crates/apr-qa-cli/src/main.rs` | Wire up report generation |

### FailFastReporter API

```rust
pub struct FailFastReporter {
    output_dir: PathBuf,
    binary: String,
}

impl FailFastReporter {
    pub fn new(output_dir: &Path) -> Self;

    /// Generate full diagnostic report on failure
    pub fn generate_report(
        &self,
        evidence: &Evidence,
        model_path: &Path,
    ) -> Result<FailFastReport>;

    /// Run apr check and capture output
    fn run_check(&self, model_path: &Path) -> Result<CheckResult>;

    /// Run apr inspect and capture output
    fn run_inspect(&self, model_path: &Path) -> Result<InspectResult>;

    /// Run apr trace and capture output
    fn run_trace(&self, model_path: &Path) -> Result<TraceResult>;

    /// Generate markdown summary
    fn generate_markdown(&self, report: &FailFastReport) -> String;

    /// Generate GitHub issue body
    fn generate_issue_body(&self, report: &FailFastReport) -> String;
}
```

## Verification

```bash
# Trigger a fail-fast report
cargo run --bin apr-qa -- run playbook.yaml --fail-fast

# Check report was generated
ls output/fail-fast-report/
# summary.md  diagnostics.json  check.json  ...

# Validate JSON schema
jq . output/fail-fast-report/diagnostics.json

# Copy issue body to clipboard (Linux)
cat output/fail-fast-report/summary.md | xclip -selection clipboard
```

## Example Output

```
[FAIL-FAST] Gate G3-STABLE FALSIFIED
[FAIL-FAST] Generating diagnostic report...
[FAIL-FAST] Running apr check... done (2.3s)
[FAIL-FAST] Running apr inspect... done (0.4s)
[FAIL-FAST] Running apr trace... done (5.1s)
[FAIL-FAST] Running apr tensors... done (0.3s)
[FAIL-FAST] Report saved to: output/fail-fast-report/
[FAIL-FAST] Summary: output/fail-fast-report/summary.md
[FAIL-FAST] GitHub issue body ready for paste
```
