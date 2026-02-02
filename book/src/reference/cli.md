# CLI Reference

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
```

#### run

Execute a playbook:

```bash
apr-qa run <playbook.yaml> [OPTIONS]
```

Options:
- `--workers <N>` - Number of parallel workers
- `--timeout <MS>` - Per-test timeout in milliseconds
- `--evidence-dir <DIR>` - Output directory for evidence
- `--fail-fast` - Stop on first failure

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
