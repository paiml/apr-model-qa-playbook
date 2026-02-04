# Isolated Test Outputs Specification

## Problem Statement

Conversion tests write output files to the same directory as source models (HuggingFace cache), polluting the cache with test artifacts like:
- `model.converted.apr`
- `model.converted.converted.converted.apr`
- `model.converted.idem1.apr`
- `model.converted.com_direct.apr`

This causes:
1. **Cache pollution**: User's HF cache fills with test artifacts
2. **Test interference**: Subsequent runs may pick up polluted files
3. **Disk bloat**: Large model files accumulate (~2.5GB per artifact)
4. **Cleanup burden**: Users must manually delete test artifacts

## Specification

### ISO-OUT-001: Isolated Output Directory

**Requirement**: All conversion test outputs MUST be written to an isolated directory, never to the source model location.

**Output Location**: `{output_dir}/conversions/{org}/{repo}/{test_type}/`

Where:
- `{output_dir}` is the playbook output directory (default: `output/`)
- `{org}/{repo}` matches the HuggingFace repo structure (e.g., `Qwen/Qwen2.5-Coder-0.5B-Instruct`)
- `{test_type}` is the conversion test category (e.g., `idempotency`, `comparison`, `round-trip`)

**Example**:
```
output/
├── evidence.json
├── report.html
└── conversions/
    └── Qwen/
        └── Qwen2.5-Coder-0.5B-Instruct/
            ├── idempotency/
            │   ├── model.idem1.apr
            │   └── model.idem2.apr
            ├── comparison/
            │   ├── model.direct.apr
            │   └── model.indirect.apr
            └── round-trip/
                └── model.rt.safetensors
```

### ISO-OUT-002: Source Model Read-Only

**Requirement**: The testing framework MUST treat source model directories as read-only.

- Never write files to `~/.cache/huggingface/`
- Never write files to `~/.cache/apr-models/`
- Never modify existing model files

### ISO-OUT-003: Cleanup on Success

**Requirement**: Conversion test artifacts SHOULD be cleaned up after successful test completion.

Options (configurable):
- `--keep-artifacts`: Retain all conversion outputs for debugging
- Default: Delete conversion outputs after evidence is collected

### ISO-OUT-004: Artifact Naming Convention

**Requirement**: Test artifacts MUST use clear naming that indicates their purpose.

| Test Type | Pattern | Example |
|-----------|---------|---------|
| Idempotency | `model.idem{N}.{ext}` | `model.idem1.apr` |
| Comparison | `model.{method}.{ext}` | `model.direct.apr` |
| Round-trip | `model.rt.{ext}` | `model.rt.safetensors` |
| Chain | `model.chain{N}.{ext}` | `model.chain3.apr` |

## Implementation

### Files to Modify

| File | Change |
|------|--------|
| `crates/apr-qa-runner/src/conversion.rs` | Add `ConversionOutputDir` struct, update test functions |
| `crates/apr-qa-runner/src/executor.rs` | Pass output directory to conversion tests |
| `crates/apr-qa-cli/src/main.rs` | Create output directory structure |

### ConversionOutputDir API

```rust
pub struct ConversionOutputDir {
    base: PathBuf,
}

impl ConversionOutputDir {
    pub fn new(output_dir: &Path, model_id: &ModelId) -> Self;
    pub fn idempotency_dir(&self) -> PathBuf;
    pub fn comparison_dir(&self) -> PathBuf;
    pub fn round_trip_dir(&self) -> PathBuf;
    pub fn cleanup(&self) -> std::io::Result<()>;
}
```

## Verification

```bash
# Run conversion tests
cargo run --bin apr-qa -- run playbook.yaml

# Verify no pollution in HF cache
ls ~/.cache/huggingface/hub/models--*/snapshots/*/*.apr
# Should return: No such file or directory

# Verify outputs in correct location
ls output/conversions/
# Should show: Qwen/Qwen2.5-Coder-0.5B-Instruct/...
```

## Migration

Existing polluted caches can be cleaned with:
```bash
find ~/.cache/huggingface -name "*.converted*.apr" -delete
find ~/.cache/huggingface -name "*.idem*.apr" -delete
find ~/.cache/huggingface -name "*.com_*.apr" -delete
find ~/.cache/huggingface -name "*.rt_*.apr" -delete
```
