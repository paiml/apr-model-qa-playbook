# CLI Reference

> **CRITICAL**: Always use `apr-qa` commands (this project's CLI) for model qualification. Never use manual `apr` commands (`apr qa`, `apr run`, `apr pull`) to bypass the playbook infrastructure. The playbook system ensures consistent testing, proper evidence collection, and SafeTensors ground truth validation.

## apr-qa

```
apr-qa [COMMAND] [OPTIONS]
```

### Commands

#### certify

Certify models against the verification matrix:

```bash
apr-qa certify [OPTIONS] [MODEL...]
```

Options:
- `--all` - Certify all models in registry
- `--family <NAME>` - Certify by model family (e.g., "qwen-coder", "llama")
- `--tier <TIER>` - Certification tier: `smoke`, `quick` (default), `standard`, `deep`
- `--output <DIR>` - Output directory for certification artifacts
- `--dry-run` - Preview what would be certified
- `--model-cache <DIR>` - Model cache directory (defaults to `~/.cache/apr-models`)
- `--apr-binary <PATH>` - Path to apr binary for inference (default: `apr`)
- `--auto-ticket` - Auto-generate structured tickets from failures (groups by root cause)
- `--ticket-repo <REPO>` - Repository for auto-ticket creation (default: `paiml/aprender`)
- `--no-integrity-check` - Disable playbook integrity verification against lock file
- `--fail-fast` - Stop certification on first model failure with enhanced diagnostics (§12.5.3)

Examples:
```bash
# MVP certification for a model family (recommended)
apr-qa certify --family qwen-coder --tier mvp

# Real profiling with auto-resolved cache
apr-qa certify --family qwen-coder --tier mvp --apr-binary apr

# Full certification for production release
apr-qa certify --family qwen-coder --tier full

# Certify specific model
apr-qa certify Qwen/Qwen2.5-Coder-1.5B-Instruct --tier mvp --apr-binary apr

# Dry run to preview plan
apr-qa certify --family qwen-coder --dry-run

# Explicit cache directory
apr-qa certify --family qwen-coder --tier mvp \
  --model-cache /custom/cache --apr-binary apr

# Auto-generate upstream tickets from failures
apr-qa certify --family qwen-coder --tier mvp --auto-ticket

# Fail-fast mode for debugging (stops after first model failure)
apr-qa certify --family qwen-coder --tier mvp --fail-fast

# Skip integrity checks (not recommended)
apr-qa certify --family qwen-coder --tier mvp --no-integrity-check
```

#### run

Execute a playbook:

```bash
apr-qa run <playbook.yaml> [OPTIONS]
```

Options:
- `--workers <N>` - Number of parallel workers (default: 4)
- `--timeout <MS>` - Per-test timeout in milliseconds (default: 60000)
- `-o, --output <DIR>` - Output directory for evidence (default: `output`)
- `--failure-policy <POLICY>` - Failure handling: `stop-on-first`, `stop-on-p0` (default), `collect-all`, `fail-fast`
- `--fail-fast` - Stop on first failure with enhanced diagnostics (§12.5.3)
- `--dry-run` - Preview execution without running tests
- `--model-path <PATH>` - Path to model file
- `--no-gpu` - Disable GPU acceleration
- `--skip-conversion-tests` - Skip P0 format conversion tests (NOT RECOMMENDED)
- `--run-tool-tests` - Run APR tool coverage tests
- `--profile-ci` - Run profile CI assertions
- `--no-differential` - Skip differential tests
- `--no-trace-payload` - Skip trace payload tests
- `--hf-parity` - Enable HuggingFace parity verification
- `--hf-corpus-path <PATH>` - Path to HF golden corpus
- `--hf-model-family <FAMILY>` - HF parity model family

Examples:
```bash
# Zero-setup: model auto-resolved from HuggingFace cache (HF-CACHE-001)
apr-qa run playbooks/models/qwen2.5-coder-1.5b-mvp.playbook.yaml

# With explicit model path
apr-qa run playbook.yaml --model-path ~/.cache/apr-models/model/gguf/model.gguf

# Fail-fast mode for debugging
apr-qa run playbook.yaml --fail-fast

# Full tracing for GitHub ticket creation
RUST_LOG=debug apr-qa run playbook.yaml --fail-fast 2>&1 | tee failure.log

# CI mode with all assertions
apr-qa run playbook.yaml --profile-ci --run-tool-tests
```

**Model Auto-Resolution (HF-CACHE-001):** When `--model-path` is omitted, the runner automatically resolves `playbook.model.hf_repo` from your local cache. It searches HuggingFace cache first (`~/.cache/huggingface/hub/`), then APR cache (`~/.cache/apr-models/`). Environment variables `HUGGINGFACE_HUB_CACHE` and `HF_HOME` are respected per HuggingFace convention.

#### report

Generate qualification report:

```bash
apr-qa report <model-id> [OPTIONS]
```

Options:
- `--format <junit|html|json>` - Output format
- `-o, --output <PATH>` - Output file/directory
- `--evidence-dir <DIR>` - Evidence input directory

#### validate

Validate playbook against schema:

```bash
apr-qa validate <playbook.yaml>
```

#### lock-playbooks

Lock playbook hashes for integrity verification:

```bash
apr-qa lock-playbooks [DIR] [OPTIONS]
```

Options:
- `DIR` - Directory containing playbook YAML files (default: `playbooks`)
- `-o, --output <PATH>` - Output lock file path (default: `playbooks/playbook.lock.yaml`)

Examples:
```bash
# Lock all playbooks in the default directory
apr-qa lock-playbooks

# Lock playbooks in a custom directory
apr-qa lock-playbooks playbooks/models -o playbooks/models.lock.yaml
```

The lock file records SHA-256 hashes of each `.playbook.yaml` file. During certification,
the executor verifies playbooks haven't been modified since locking (unless `--no-integrity-check`
is passed).

#### list-models

List available models in registry:

```bash
apr-qa list-models [OPTIONS]
```

Options:
- `--architecture <ARCH>` - Filter by architecture
- `--size <SIZE>` - Filter by size category

### Global Options

- `-v, --verbose` - Increase verbosity
- `-q, --quiet` - Suppress output
- `--color <auto|always|never>` - Color output
- `-h, --help` - Show help
- `-V, --version` - Show version

## APR CLI Commands (Upstream)

> **WARNING**: The commands below are upstream `apr` CLI commands that this QA framework **tests**. Do NOT use these commands directly for model qualification. Always use `apr-qa certify` or `apr-qa run` with a playbook instead.

The following apr commands are tested by this QA framework:

### apr tune

Plan ML tuning configurations:

```bash
# Plan LoRA tuning for a 7B model with 24GB VRAM
apr tune --model 7B --vram 24 --plan

# Plan QLoRA for memory-constrained setup
apr tune --model 1.5B --vram 8 --method qlora --plan

# Output as JSON for CI integration
apr tune --model 7B --vram 24 --json
```

Options:
- `--model <SIZE>` - Model size (e.g., 1.5B, 7B, 13B)
- `--vram <GB>` - Available VRAM in gigabytes
- `--method <lora|qlora>` - Tuning method (default: lora)
- `--plan` - Output tuning plan
- `--json` - Output in JSON format for CI

### apr profile

Profile model execution:

```bash
apr profile <model> --output flamegraph.svg
apr profile <model> --focus attention
```

### apr trace

Trace model inference:

```bash
apr trace <model> --level basic
apr trace <model> --level layer
apr trace <model> --level payload
```

## Make Targets

```bash
make build          # Build all crates
make test           # Run all tests
make lint           # Run clippy
make coverage       # Generate coverage report
make coverage-check # Verify >= 95% coverage
make check          # fmt + lint + test
make doc            # Generate documentation
```
