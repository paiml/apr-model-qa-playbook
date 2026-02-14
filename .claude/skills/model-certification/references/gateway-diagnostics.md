# Gateway Diagnostics

Quick reference for diagnosing gateway failures. For detailed explanations, see the `gateway-debug` skill.

## Diagnostic Flowchart

```
MQS = 0
│
├── grep "G0-" evidence.json → failures?
│   ├── G0-PULL: Model download failed
│   │   └── Check network, HF token, repo ID
│   ├── G0-FORMAT: Invalid model file
│   │   └── Re-download, check file integrity
│   ├── G0-VALIDATE: Physics failure (NaN/Inf/zeros)
│   │   └── Re-download, try different quantization
│   ├── G0-TENSOR: Tensor count/shape mismatch
│   │   └── Check family contract, inspect model
│   ├── G0-INTEGRITY: config.json mismatch
│   │   └── Verify config.json matches actual tensors
│   └── G0-LAYOUT: Not row-major (LAYOUT-002)
│       └── Convert GGUF -> APR via apr import
│
├── grep "G1" evidence.json → failures?
│   └── Model failed to load
│       ├── File not found → check --model-cache path
│       ├── OOM → use --no-gpu or smaller quant
│       └── Bad format → re-download
│
├── grep "G2" evidence.json → failures?
│   └── Inference produced no output
│       ├── Timeout → increase --timeout
│       └── Empty output → check model compatibility
│
├── any "Crashed" in evidence.json?
│   └── G3 failure (stability)
│       ├── Exit 139 (SIGSEGV) → LAYOUT-002 violation
│       ├── Exit 137 (SIGKILL) → OOM killer
│       └── Exit 134 (SIGABRT) → assertion failure
│
└── grep "G4" evidence.json → >25% failures?
    └── G4 failure (quality/garbage)
        ├── Repetitive output → weak model or layout bug
        ├── NaN in output → numerical instability
        ├── Empty output → check G2 first
        └── Non-ASCII gibberish → LAYOUT-002 violation
```

## Quick Commands

```bash
# Which gateways failed?
cargo run --bin apr-qa -- score evidence.json

# Find all failures
grep -c '"Falsified"\|"Crashed"\|"Timeout"' evidence.json

# G0 failures specifically
grep '"G0-' evidence.json | grep -c '"Falsified"\|"Crashed"'

# G4 garbage details
grep -A5 '"G4-GARBAGE' evidence.json

# Crash stderr
grep -A10 '"Crashed"' evidence.json | grep '"stderr"'

# Run conversion diagnostics
./scripts/diagnose-conversion.sh /path/to/model.gguf

# Validate contract
cargo run --bin apr-qa -- validate-contract /path/to/model
```

## Common Fixes

| Symptom | Gateway | Fix |
|---------|---------|-----|
| Score = 0, all formats fail | G1 | Check model path, re-download |
| Score = 0, only GGUF fails | G0-LAYOUT | Convert: `apr import model.gguf -o model.apr` |
| Score = 0, crashes on GPU | G3 | Use `--no-gpu`, check CUDA drivers |
| Score = 0, garbage output | G4 | Check LAYOUT-002, re-download model |
| Score > 0 but low | - | Check category breakdown in score output |
| Timeouts everywhere | G2 | Increase `--timeout`, check system load |
