# Playbook Infrastructure Improvements

**Version:** 1.0.0
**Status:** DRAFT
**Date:** 2026-02-02
**Triggered by:** Qwen2.5-Coder dev-tier qualification failure (0/6 BLOCKED)
**Upstream ticket:** [paiml/aprender#200](https://github.com/paiml/aprender/issues/200)

---

## 1. Problem Statement

On 2026-02-02, dev-tier qualification of the Qwen2.5-Coder family (6 models) resulted
in 0/6 passing. The QA framework correctly detected three upstream defects in `aprender`:
tensor name mapping, `lm_head.weight` dequantization, and conversion infrastructure failures.

However, two process failures occurred during the response:

1. **The QA operator attempted to switch playbook formats from `safetensors` to `gguf`
   to make the qualification pass** — creating Self-Admitted Technical Debt (SATD) by
   destroying the test specification instead of reporting the defect.

2. **The ticket to aprender required manual authoring** — the certifier produced raw
   evidence JSON but no structured failure analysis that maps to upstream test fixture gaps.

Both failures violate Toyota Production System principles: Jidoka (stop the line) and
Poka-Yoke (mistake-proofing).

## 2. Five-Whys Analysis

### 2.1 Why did the QA operator attempt to weaken the test specification?

| # | Why | Finding |
|---|-----|---------|
| 1 | Switched `formats: [safetensors]` → `[gguf]` | SafeTensors inference failed on 3B/7B/14B |
| 2 | Wanted the qualification to pass | Interpreted "qualify" as "make green" not "determine if qualified" |
| 3 | Confused "test to pass" with "test to fail" | No guardrail prevents modifying the format spec mid-certification |
| 4 | Format field is silently editable | Playbook YAML has no provenance or integrity validation |
| **5** | **No design-intent lock on test specifications** | **A certification playbook's format spec is as mutable as a config variable — but it should be as immutable as a gateway definition** |

**Root cause:** The playbook schema treats all fields as configuration. There is no
distinction between "this is the format we are testing" (immutable design intent) and
"this is how many workers to use" (tunable parameter).

### 2.2 Why didn't the feedback loop auto-generate rich tickets?

| # | Why | Finding |
|---|-----|---------|
| 1 | Ticket required manual `gh issue create` | Certifier writes evidence JSON + CSV but has no ticket generation step |
| 2 | No ticket generation step exists | Evidence captures outcomes but doesn't classify failure types |
| 3 | Failures aren't classified | Conversion executor treats all errors as opaque strings |
| 4 | No typed failure modes | No mapping between observed errors and upstream test fixture gaps |
| **5** | **No defect→fixture mapping** | **When the runner sees "tensor name mismatch" it should know that maps to "needs `with_gguf()` harness in aprender"; when it sees "dequantize Q4_K_M failed" it should know "needs `with_q4k()` in PygmyConfig"** |

**Root cause:** The QA framework detects *what* failed but has no ontology of *why*
it failed or *what upstream test* would have prevented it.

## 3. Proposed Improvements

### 3.1 Playbook Integrity Lock (Poka-Yoke)

**Problem:** Playbook format fields can be silently changed to dodge failures.

**Solution:** Add a `playbook_sha256` field to the certification record. The certifier
computes the hash of the playbook file before execution and refuses to certify if it
differs from the committed version.

```yaml
# playbook.lock.yaml (generated, committed)
qwen2.5-coder-1.5b-quick:
  sha256: "a1b2c3d4..."
  locked_fields:
    - model.formats
    - test_matrix.backends
    - test_matrix.modalities
```

The CLI enforces:
- `apr-qa certify` computes playbook hash before execution
- If hash differs from lock file, certification is REFUSED with error
- Lock file is regenerated only by explicit `apr-qa lock-playbooks` command

**Prior art:** HuggingFace `kernels-community` uses `build.toml` as an immutable
contract per kernel. Their `validate-kernel-pr.py` rejects PRs that don't match the
expected structure. `flake.lock` files pin exact dependency versions for reproducibility.

**Falsification test:** Can an operator modify `formats: [safetensors]` to
`formats: [gguf]` and successfully run certification without updating the lock file?
If yes, the Poka-Yoke has failed.

### 3.2 CODEOWNERS for Playbooks

**Problem:** Any operator can modify a playbook's test matrix without review.

**Solution:** Add `CODEOWNERS` entry:

```
playbooks/models/    @paiml/qa-leads
playbooks/templates/ @paiml/qa-leads
```

**Prior art:** `kernels-community` assigns per-kernel maintainers in `.github/CODEOWNERS`.
A change to `flash-attn3/` requires review from the flash-attn maintainer.

**Falsification test:** Can a playbook format change be merged to `main` without
review from a QA lead? If yes, the gate is ineffective.

### 3.3 Explicit Skip Mechanism

**Problem:** When a model genuinely cannot run a format (e.g., 32B too large for GPU
memory), there is no way to document this except silently removing the format.

**Solution:** Require a `skip-<format>.reason` file in the playbook directory:

```
playbooks/models/
├── qwen2.5-coder-32b-quick.playbook.yaml
└── qwen2.5-coder-32b-quick.skip-gpu.reason
```

Contents:
```
32B model exceeds single-GPU VRAM (48GB required, 24GB available).
Tracked: paiml/aprender#200
```

The certifier logs skips as `SKIPPED (explicit)` in evidence, distinct from
`SKIPPED (implicit)` which should trigger a warning.

**Prior art:** `kernels-community` uses `.skip-pr-ci` files — visible, auditable,
deliberate opt-outs from CI.

**Falsification test:** Can a format be absent from a playbook with no corresponding
skip file and no warning in the certification output? If yes, silent skips persist.

### 3.4 Typed Failure Taxonomy

**Problem:** All conversion failures are opaque strings. "Tensor not found:
model.embed_tokens.weight" and "Failed to dequantize lm_head.weight" both produce
the same `Falsified` outcome with a reason string.

**Solution:** Add structured failure variants to the evidence schema:

```rust
pub enum ConversionFailureType {
    TensorNameMismatch {
        expected_convention: TensorNaming,  // HuggingFace, GGUF, APR
        actual_convention: TensorNaming,
        example_expected: String,
        example_actual: String,
    },
    DequantizationFailure {
        tensor_name: String,
        quant_type: String,  // Q4_K_M, Q6_K, F16, F32
    },
    ConfigMetadataMismatch {
        field: String,
        expected: String,
        actual: String,
    },
    MissingArtifact {
        artifact: String,  // tokenizer.json, config.json
        expected_location: PathBuf,
    },
    InferenceFailure {
        exit_code: i32,
        stderr_first_line: String,
    },
}
```

**Prior art:** `kernels-community`'s `assert_close()` in
`gpt-oss-triton-kernels/torch-ext/gpt_oss_triton_kernels/testing.py` produces
structured diagnostics: tensor index, actual vs expected, RMS error, count of
mismatched elements. The `paiml-mcp-agent-toolkit` CUDA TDG module implements an
18-category `DefectSeverity` taxonomy with P0–P3 classification
(source: batuta oracle, `paiml-mcp-agent-toolkit/src/cli/handlers/cuda_tdg_handlers.rs`).

**Falsification test:** Run a certification that produces a `TensorNameMismatch`
failure. Does the evidence JSON contain the structured variant with both conventions
and example tensor names? If it only contains a reason string, the taxonomy is not
implemented.

### 3.5 Defect→Fixture Mapping

**Problem:** When the QA framework detects a failure, it doesn't know what upstream
test fixture would have prevented it.

**Solution:** A declarative mapping from failure types to aprender pygmy fixtures:

```yaml
# defect-fixture-map.yaml
tensor_name_mismatch:
  upstream_fixture: "ConversionTestHarness::with_gguf()"
  pygmy_builder: "build_pygmy_gguf_hf_names()"
  ticket_template: |
    ## Conversion Bug: {source}→{target} retains {actual_convention} tensor naming

    Expected: `{example_expected}`
    Got: `{example_actual}`

    ### Required Fixture
    Add `with_gguf()` to `ConversionTestHarness` that generates tensors
    with GGUF-convention names and verifies the converter maps them to
    {expected_convention} conventions.

dequantization_failure:
  upstream_fixture: "PygmyConfig::with_q4k() / with_q6k()"
  pygmy_builder: "build_pygmy_quantized({quant_type})"
  ticket_template: |
    ## Dequantization Bug: {tensor_name} fails for {quant_type}

    ### Required Fixture
    Add `with_{quant_type_snake}()` to `PygmyConfig` that generates
    quantized tensor blocks. Add dequantization round-trip test.

missing_artifact:
  upstream_fixture: "auto_populate_cache()"
  pygmy_builder: null
  ticket_template: |
    ## Cache Bug: {artifact} missing from {expected_location}

    The auto-populate cache step does not copy {artifact} from the
    HuggingFace snapshot directory.

config_metadata_mismatch:
  upstream_fixture: "build_pygmy_config_json()"
  pygmy_builder: "build_pygmy_with_real_config()"
  ticket_template: |
    ## Config Bug: {field} = {actual}, expected {expected}

    The converter generates stub config.json with incorrect metadata.

    ### Required Fixture
    Add config.json validation to conversion tests that checks
    {field} matches the source model's metadata.
```

**Prior art:** The `aprender/src/format/test_factory.rs` file (source: batuta oracle)
already implements the "Active Pygmy" pattern — minimal valid model files in memory
for testing. The mapping extends this by telling the factory *what* to generate when
a specific failure class is detected. The `decy` project documents a similar
18-category defect taxonomy for ownership assignment
(source: batuta oracle, `decy/docs/improvements-ml-techniques.md`).

**Falsification test:** Create a synthetic `TensorNameMismatch` failure. Does the
certifier auto-generate a ticket body that includes the exact `PygmyConfig` builder
method needed? If the ticket body is generic (no fixture name), the mapping is broken.

### 3.6 Auto-Generated Tickets

**Problem:** Tickets require manual authoring after certification.

**Solution:** After certification completes, the certifier:

1. Classifies each `Falsified` evidence using the typed taxonomy (§3.4)
2. Groups failures by root cause (not by scenario)
3. Looks up the defect→fixture mapping (§3.5)
4. Generates a GitHub issue body with the template, filling in structured fields
5. Posts via `gh issue create` (or outputs the body for review)

```
apr-qa certify --family qwen-coder --tier quick --auto-ticket
```

The `--auto-ticket` flag enables automatic issue creation. Without it, the certifier
prints the would-be ticket body to stdout for review.

**Prior art:** `kernels-community`'s `scripts/report_kernel_failures.py` implements
discover→check→classify→report with Slack integration. Their nightly pipeline
auto-posts structured failure reports with GitHub Actions links.

**Falsification test:** Run certification with `--auto-ticket` against a model with
known conversion failures. Does the system create exactly one ticket per root cause
(not one per scenario)? If it creates 12 tickets for 12 conversion scenarios that
share one root cause, deduplication is broken.

### 3.7 Dtype-Aware Tolerance Parametrization

**Problem:** Conversion tests treat all quantization types identically. A Q4_K_M
failure and an F32 failure produce the same evidence structure, but they have
fundamentally different root causes and require different pygmy fixtures.

**Solution:** Parametrize conversion tests by quantization type with expected
tolerances:

```rust
pub struct ConversionTolerance {
    pub quant_type: QuantType,
    pub atol: f64,
    pub rtol: f64,
    pub expected_pygmy_fixture: &'static str,
}

const TOLERANCES: &[ConversionTolerance] = &[
    ConversionTolerance { quant_type: QuantType::F32,   atol: 1e-6, rtol: 1e-6, expected_pygmy_fixture: "build_pygmy_f32()" },
    ConversionTolerance { quant_type: QuantType::F16,   atol: 1e-3, rtol: 1e-3, expected_pygmy_fixture: "build_pygmy_f16()" },
    ConversionTolerance { quant_type: QuantType::Q4KM,  atol: 1e-1, rtol: 1e-2, expected_pygmy_fixture: "build_pygmy_q4k()" },
    ConversionTolerance { quant_type: QuantType::Q6K,   atol: 5e-2, rtol: 1e-2, expected_pygmy_fixture: "build_pygmy_q6k()" },
];
```

**Prior art:** `kernels-community` parametrizes every kernel test with
`(dtype, atol, rtol)` triples. Their rotary embedding tests use:
```python
@pytest.mark.parametrize("dtype, atol, rtol", [
    (torch.float32, 1e-5, 1e-5),
    (torch.bfloat16, 1e-1, 1e-5),
])
```

**Falsification test:** Run a round-trip conversion with Q4_K_M quantized weights.
Does the evidence include the quant type and the tolerance expectation? Does the
auto-generated ticket reference `build_pygmy_q4k()`? If the quant type is absent
from evidence, the parametrization is not propagated.

## 4. Citations

### Methodological Foundations

1. Popper, K. R. (1959). *The Logic of Scientific Discovery*. Routledge.
   — Falsificationism as the demarcation criterion. A test is only valuable if it
   can fail. The playbook integrity lock ensures tests cannot be weakened to avoid
   failure.

2. Taleb, N. N. (2007). *The Black Swan: The Impact of the Highly Improbable*.
   Random House.
   — Rare catastrophic events (tensor name mismatches in converted models) are the
   events that matter most. The failure taxonomy classifies these rare events instead
   of treating them as opaque strings.

3. Ohno, T. (1988). *Toyota Production System: Beyond Large-Scale Production*.
   Productivity Press.
   — Jidoka: stop the line on defect. The playbook integrity lock prevents
   certification from proceeding with a weakened specification. Poka-Yoke: mistake-
   proofing. The lock file makes it physically impossible to silently change test
   parameters.

4. Ishikawa, K. (1985). *What Is Total Quality Control? The Japanese Way*.
   Prentice Hall.
   — Five-whys root cause analysis as applied in §2. The method traces symptoms
   to structural causes rather than treating each failure independently.

### Software Engineering

5. Potvin, R., & Levenberg, J. (2016). "Why Google Stores Billions of Lines of Code
   in a Single Repository." *Communications of the ACM*, 59(7), 78–87.
   doi:10.1145/2854146
   — CODEOWNERS and monorepo-scale code review. The pattern of per-directory ownership
   for playbooks derives from Google's approach to large-scale code governance.

6. Sierra, G., Shihab, E., & Kamei, Y. (2019). "A Survey on Self-Admitted Technical
   Debt." *Journal of Systems and Software*, 152, 70–82.
   doi:10.1016/j.jss.2019.02.056
   — SATD taxonomy. The simulate executor removal (prior commit) and the attempted
   GGUF workaround are both instances of SATD. The playbook integrity lock prevents
   new SATD introduction via test weakening.

7. Zeller, A. (2009). *Why Programs Fail: A Guide to Systematic Debugging*.
   Morgan Kaufmann.
   — Defect classification and the delta debugging algorithm. The defect→fixture
   mapping (§3.5) applies Zeller's principle that debugging is most effective when
   the failure is classified into a known taxonomy before attempting repair.

### Testing Infrastructure

8. Claessen, K., & Hughes, J. (2000). "QuickCheck: A Lightweight Tool for Random
   Testing of Haskell Programs." *Proceedings of the Fifth ACM SIGPLAN International
   Conference on Functional Programming (ICFP '00)*, 268–279.
   doi:10.1145/351240.351266
   — Property-based testing. The QA framework's proptest-based scenario generation
   follows this lineage. The dtype-aware tolerance parametrization (§3.7) extends
   property-based testing with quantization-type-specific invariants.

9. Barr, E. T., Harman, M., McMinn, P., Shahbaz, M., & Yoo, S. (2015). "The Oracle
   Problem in Software Testing: A Survey." *IEEE Transactions on Software Engineering*,
   41(5), 507–525. doi:10.1109/TSE.2014.2372785
   — The oracle problem. Our conversion tests currently lack precise oracles for
   quantized tensor round-trips. The tolerance parametrization (§3.7) defines
   format-specific oracles rather than using a single threshold.

## 5. Batuta Oracle Cross-References

The following stack documentation informed this specification (queried 2026-02-02):

| Source | Relevance |
|--------|-----------|
| `aprender/src/format/test_factory.rs` | "Active Pygmy" pattern — the existing fixture factory that §3.5 extends with failure-driven generation |
| `aprender/docs/APR-SPEC.md §10` | Tensor scaling error diagnostics — the error format that §3.4's taxonomy must parse |
| `paiml-mcp-agent-toolkit/.../cuda_tdg_handlers.rs` | 18-category `DefectSeverity` taxonomy (P0–P3) — model for §3.4's `ConversionFailureType` |
| `decy/docs/improvements-ml-techniques.md §3.4` | "Defect Taxonomy for Ownership" — prior art for mapping failure classes to responsible fixtures |
| `paiml-mcp-agent-toolkit/docs/improve-pmat-comply.md` | "Trend G: Data Corruption in Model I/O" — documents the Q4_K/Q6_K transpose bugs that §3.7's tolerances must catch |
| `trueno/docs/simulation-testing-spec.md §8` | Jidoka guards and Poka-Yoke type safety in simulation testing — architectural pattern for §3.1 |
| `apr-model-qa-playbook/docs/apr-playbook-spec.md` | Existing F-PARITY-001 gate definition ("gguf_argmax == safetensors_argmax") — the cross-format parity gate that conversion fixes must satisfy |

## 6. Falsification Summary

Each improvement has a falsification test. If any test passes (i.e., the defect is
still reproducible), the improvement has failed:

| § | Improvement | Falsification |
|---|-------------|---------------|
| 3.1 | Playbook integrity lock | Operator modifies `formats:` and certifies without lock update → **must be refused** |
| 3.2 | CODEOWNERS for playbooks | Format change merged without QA lead review → **must be blocked** |
| 3.3 | Explicit skip mechanism | Format absent from playbook with no skip file and no warning → **must warn** |
| 3.4 | Typed failure taxonomy | `TensorNameMismatch` failure produces only a string reason, no structured variant → **must have typed fields** |
| 3.5 | Defect→fixture mapping | Auto-ticket for `TensorNameMismatch` omits `with_gguf()` fixture name → **must reference exact builder** |
| 3.6 | Auto-generated tickets | 12 conversion scenarios with one root cause produce 12 tickets → **must deduplicate to 1** |
| 3.7 | Dtype-aware tolerances | Q4_K_M failure evidence lacks quant type field → **must include quant_type** |

---

*This specification follows the Popperian protocol defined in
[certified-testing.md](certified-testing.md): every proposed improvement includes
a falsification test that, if it passes, proves the improvement is not yet effective.*
