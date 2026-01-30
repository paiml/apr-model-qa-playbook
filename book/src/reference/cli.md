# CLI Reference

## apr-qa

```
apr-qa [COMMAND] [OPTIONS]
```

### Commands

#### run

Execute a playbook:

```bash
apr-qa run <playbook.yaml> [OPTIONS]
```

Options:
- `--mode <simulate|subprocess>` - Execution mode
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
