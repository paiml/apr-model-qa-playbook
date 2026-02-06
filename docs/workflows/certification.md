# Model Certification Workflows

This document describes the workflows for certifying models using the APR QA framework.

## Prerequisites

- Rust toolchain (1.85+)
- `apr` CLI installed and configured
- Model available on HuggingFace or locally

## Workflow 1: Certify a New Model

### Step 1: Create Playbook

Start from a template matching your certification tier:

```bash
# For MVP tier (recommended for new models)
cp playbooks/templates/mvp-template.yaml playbooks/models/{family}-{model}-mvp.playbook.yaml

# Example for Qwen2.5-Coder-0.5B
cp playbooks/templates/mvp-template.yaml playbooks/models/qwen2.5-coder-0.5b-mvp.playbook.yaml
```

### Step 2: Edit Playbook

Update the playbook with model-specific details:

```yaml
name: qwen2.5-coder-0.5b-mvp
version: "1.0.0"
model:
  hf_repo: "Qwen/Qwen2.5-Coder-0.5B-Instruct"
  formats: [safetensors, apr, gguf]
  size_category: tiny  # tiny/small/medium/large/xlarge/huge
  family: qwen2
  size_variant: "0.5b"
test_matrix:
  modalities: [run, chat]
  backends: [cpu]
  scenario_count: 3
```

### Step 3: Run Certification

```bash
# Run the playbook
cargo run --bin apr-qa -- run playbooks/models/qwen2.5-coder-0.5b-mvp.playbook.yaml \
    --output certifications/qwen2.5-coder-0.5b-mvp/evidence.json

# Or use make targets
make certify MODEL=qwen2.5-coder-0.5b TIER=mvp
```

### Step 4: Review Results

The execution produces:
- `certifications/{model}/evidence.json` - Raw test evidence
- Console output with MQS score and gateway results

Check the MQS score and gateway status:
- **800+**: CERTIFIED (if all gateways pass)
- **<800**: BLOCKED (needs investigation)
- **Gateway failure**: Score is zeroed

### Step 5: Update Certification Table

Add or update the entry in `docs/certifications/models.csv`:

```csv
model_id,family,parameters,size_category,status,mqs_score,grade,...
Qwen/Qwen2.5-Coder-0.5B-Instruct,qwen-coder,0.5B,tiny,CERTIFIED,850,B,...
```

### Step 6: Commit and Push

```bash
git add certifications/qwen2.5-coder-0.5b-mvp/
git add docs/certifications/models.csv
git commit -m "feat(cert): add qwen2.5-coder-0.5b-mvp certification"
git push
```

---

## Workflow 2: Update Certification After Fixes

When a model fails certification and you've fixed the underlying issue:

### Step 1: Identify the Failure

Review the evidence file or console output:

```bash
# Check evidence for failures
jq '.[] | select(.outcome == "Falsified")' certifications/my-model/evidence.json

# Check gateway status
jq '.[] | select(.gate_id | startswith("G"))' certifications/my-model/evidence.json
```

### Step 2: Fix the Issue

Common fixes:
- **G0 failure**: Fix config.json or tensor metadata
- **G1 failure**: Fix model loading (missing files, corrupt weights)
- **G2 failure**: Fix inference (tokenizer issues, OOM)
- **G3 failure**: Fix crashes (null pointers, assertion failures)
- **G4 failure**: Fix garbage output (layout issues, quantization bugs)

### Step 3: Re-run Certification

```bash
# Option A: Clear and re-run
rm -rf certifications/my-model/
cargo run --bin apr-qa -- run playbooks/models/my-model-mvp.playbook.yaml \
    --output certifications/my-model/evidence.json

# Option B: Append mode (keeps history)
cargo run --bin apr-qa -- run playbooks/models/my-model-mvp.playbook.yaml \
    --output certifications/my-model/evidence.json \
    --append
```

### Step 4: Update CSV and Commit

Update `docs/certifications/models.csv` with new scores and commit.

---

## Workflow 3: Full Certification Pipeline

For production-ready models, run all tiers:

```bash
# Quick check (fast sanity test)
cargo run --bin apr-qa -- run playbooks/models/my-model-quick.playbook.yaml

# Smoke test (basic functionality)
cargo run --bin apr-qa -- run playbooks/models/my-model-smoke.playbook.yaml

# MVP tier (minimum viable)
cargo run --bin apr-qa -- run playbooks/models/my-model-mvp.playbook.yaml

# Full certification (comprehensive)
cargo run --bin apr-qa -- run playbooks/models/my-model-full.playbook.yaml
```

Each tier builds on the previous:
- **quick**: 5 scenarios, CPU only, 1 format
- **smoke**: 15 scenarios, CPU+GPU, 2 formats
- **mvp**: 50 scenarios, all backends, all formats
- **full**: 200+ scenarios, stress tests, edge cases

---

## Validation Commands

### Validate Schemas Locally

```bash
./scripts/validate-schemas.sh
```

### Check Cross-Repo Alignment

```bash
./scripts/validate-aprender-alignment.sh ../aprender
```

### Verify Playbook Syntax

```bash
yq eval '.' playbooks/models/my-model-mvp.playbook.yaml > /dev/null
```

---

## Best Practices

1. **Start with MVP tier** - Don't jump to full certification
2. **Fix gateway failures first** - They zero the entire score
3. **Use SafeTensors as ground truth** - GGUF/APR are derived formats
4. **Keep evidence files** - They're audit trail for certification
5. **Run validation scripts** - Before committing changes
