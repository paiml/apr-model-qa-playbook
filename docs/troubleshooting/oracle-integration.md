# Oracle Integration Troubleshooting Guide

This guide helps diagnose and fix common issues with the APR QA oracle integration.

## Gateway Failures

### G0: Integrity Check Failed

**Symptom:**
```
G0 FAIL: config.json says 14 layers but tensors have 24
```

**Causes:**
1. Corrupted or outdated `config.json`
2. Model weights don't match config
3. Mixed files from different model versions

**Solutions:**

```bash
# Re-download model files
apr pull --force Qwen/Qwen2.5-Coder-0.5B-Instruct

# Verify config matches tensors manually
python -c "
import json
from safetensors import safe_open

with open('config.json') as f:
    config = json.load(f)

with safe_open('model.safetensors', framework='pt') as f:
    keys = f.keys()
    layer_keys = [k for k in keys if 'layers.' in k]
    num_layers = len(set(k.split('.')[2] for k in layer_keys if k.startswith('model.layers.')))

print(f'Config says: {config.get(\"num_hidden_layers\")} layers')
print(f'Tensors have: {num_layers} layers')
"
```

### G1: Model Load Failed

**Symptom:**
```
G1 FAIL: Model failed to load - OutOfMemory
G1 FAIL: Model failed to load - FileNotFound
```

**Causes:**
1. Insufficient memory for model size
2. Missing model files (safetensors, tokenizer.json)
3. Corrupted weight files

**Solutions:**

```bash
# Check available memory
free -h

# Verify all required files exist
ls -la ~/.cache/apr-models/{model}/safetensors/
# Should contain: model.safetensors, config.json, tokenizer.json

# Check file integrity
sha256sum model.safetensors
```

### G2: Inference Failed

**Symptom:**
```
G2 FAIL: Inference failed - TokenizerError
G2 FAIL: Inference failed - ShapeMismatch
```

**Causes:**
1. Tokenizer configuration issues
2. Input/output shape mismatches
3. Missing special tokens

**Solutions:**

```bash
# Test tokenizer independently
python -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-Coder-0.5B-Instruct')
print(tok.encode('Hello world'))
print(tok.decode([1, 2, 3]))
"

# Check tokenizer.json exists and is valid
jq '.' tokenizer.json > /dev/null && echo "Valid JSON"
```

### G3: Crash Detected

**Symptom:**
```
G3 FAIL: Process crashed with signal 11 (SIGSEGV)
G3 FAIL: Process crashed with signal 6 (SIGABRT)
```

**Causes:**
1. Memory corruption (null pointer dereference)
2. Stack overflow
3. Assertion failures in native code

**Solutions:**

```bash
# Run with debug symbols
RUST_BACKTRACE=1 cargo run --bin apr-qa -- run playbook.yaml

# Check system limits
ulimit -s  # Stack size

# Test with smaller batch/context
# Edit playbook to reduce max_tokens
```

### G4: Garbage Output

**Symptom:**
```
G4 FAIL: Output contains garbage - non-ASCII ratio 45%
G4 FAIL: Output is empty
```

**Causes:**
1. **LAYOUT-002 violation** - Column-major vs row-major mismatch
2. Incorrect quantization
3. Corrupted weights

**Solutions:**

```bash
# CRITICAL: Check tensor layout
# GGUF uses column-major, APR uses row-major
# Always convert GGUF → APR before testing

apr convert model.gguf -o model.apr

# Verify output manually
apr run model.apr -p "What is 2+2?" --max-tokens 32
```

---

## Schema Validation Errors

### Evidence JSON Invalid

**Symptom:**
```
ERROR: certifications/my-model/evidence.json: Invalid JSON syntax
ERROR: certifications/my-model/evidence.json: 5 entries missing required fields
```

**Solutions:**

```bash
# Check JSON syntax
jq '.' evidence.json > /dev/null

# Find entries with missing fields
jq '.[] | select(.id == null or .gate_id == null)' evidence.json

# Re-run certification to regenerate evidence
rm evidence.json
cargo run --bin apr-qa -- run playbook.yaml --output evidence.json
```

### Models.csv Validation Failed

**Symptom:**
```
ERROR: models.csv:5: Invalid status 'PASSED'
ERROR: models.csv:8: Invalid mqs_score '1500'
```

**Solutions:**

Valid values:
- `status`: CERTIFIED, BLOCKED, PENDING, UNTESTED
- `mqs_score`: 0-1000
- `grade`: A, B, C, D, F, -
- `size_category`: tiny, small, medium, large, xlarge, huge
- `g1-g4`: true, false

```bash
# Validate locally
./scripts/validate-schemas.sh
```

---

## Cross-Repo Alignment Errors

### Size Category Mismatch

**Symptom:**
```
ERROR: Qwen/Qwen2.5-Coder-3B-Instruct: size_category mismatch -
       playbook has 'medium', family YAML expects 'small'
```

**Causes:**
1. Playbook has incorrect `size_category`
2. Family YAML needs updating

**Solutions:**

```bash
# Check family YAML definition
yq '.certification.size_categories' ../aprender/contracts/model-families/qwen2.yaml

# Update playbook to match
# Edit playbooks/models/{model}.yaml:
#   model:
#     size_category: small  # Match family YAML
```

### Missing Family Contract

**Symptom:**
```
WARNING: Family YAML not found: /path/to/aprender/contracts/model-families/unknown.yaml
```

**Causes:**
1. New model family without contract
2. Incorrect family name mapping

**Solutions:**

```bash
# List available family contracts
ls ../aprender/contracts/model-families/

# Map model to correct family
# Qwen2.5-Coder → qwen2.yaml
# Llama-3.2 → llama.yaml
# Mistral-7B → mistral.yaml
```

---

## Performance Issues

### Certification Too Slow

**Causes:**
1. Too many scenarios
2. Large model on CPU
3. Insufficient parallelism

**Solutions:**

```bash
# Reduce scenario count in playbook
test_matrix:
  scenario_count: 3  # Reduce from 10

# Use GPU backend
test_matrix:
  backends: [gpu]  # Skip CPU

# Increase workers (for small models)
cargo run --bin apr-qa -- run playbook.yaml --workers 4
```

### Out of Memory

**Causes:**
1. Model too large for available RAM
2. GPU VRAM exceeded
3. Memory leak

**Solutions:**

```bash
# Check model size requirements
# tiny: ~2GB, small: ~4GB, medium: ~16GB, large: ~32GB

# Use CPU offloading (if supported)
# Or reduce batch size in playbook

# Monitor memory during run
watch -n 1 free -h
```

---

## Getting Help

1. Check the [certification workflows](../workflows/certification.md)
2. Review the [Oracle Integration spec](../specifications/oracle-integration-falsification-protocol.md)
3. Search stack docs: `batuta oracle --rag "your question"`
4. File an issue: https://github.com/paiml/apr-model-qa-playbook/issues
