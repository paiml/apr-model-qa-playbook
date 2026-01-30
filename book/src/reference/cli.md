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
