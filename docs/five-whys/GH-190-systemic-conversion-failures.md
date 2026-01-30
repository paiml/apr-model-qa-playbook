# Five Whys: Why Do Conversion Bugs Keep Recurring?

**Date:** 2026-01-30
**Scope:** GH-185, GH-186, GH-189, GH-190 (4 P0 conversion bugs in one sprint)
**Method:** Toyota Production System Five Whys Root Cause Analysis

---

## The Problem Statement

We have filed **four P0 conversion pipeline bugs in a single sprint**, each causing silent data corruption where models pass validation but produce garbage output. This is despite having:
- 220 falsification gates
- Rosetta differential testing (diff-tensors, compare-inference, fingerprint, validate-stats)
- Automated playbook-driven qualification
- A 4,741-line specification

**The bugs keep coming back in different forms.** Why?

---

## Bug Inventory

| Bug | What Broke | Root Cause | Silent? |
|-----|-----------|-----------|---------|
| GH-185 | Missing tokenizer in APR | Alternate code path not tested | Yes |
| GH-186 | DType 12 (Q4K) read as F32 | No bidirectional contract test | Yes |
| GH-189 | LayerNorm weights zeroed | No role-specific validation | Yes |
| GH-190 | Tensor names mismatched | Writer/loader naming disagreement | Yes |

**Common thread:** All four are **silent failures at format boundaries**.

---

## Five Whys

### Why #1: Why does GH-190 exist?

**Because the APR converter writes HuggingFace tensor names (`model.layers.0.self_attn.q_proj.weight`) but the APR loader expects GGUF-style names (`0.q_proj.weight`).**

The writer and reader were developed independently and never tested together with a round-trip assertion.

### Why #2: Why were writer and reader never tested together?

**Because there is no integration test that converts GGUF→APR and then runs inference on the result, comparing output token-by-token against the original.**

We have *unit tests* for the converter (does it produce a valid APR file?) and *unit tests* for the loader (does it load a valid APR file?). But no test answers: "does converting and then loading produce the same inference output?"

The converter tests check: "did I write 339 tensors?" (yes)
The loader tests check: "can I find tensors?" (yes, via pattern matching)
Nobody checks: "do the names agree?"

### Why #3: Why is there no end-to-end integration test?

**Because the converter (`aprender`) and the loader (`realizar`) are separate crates in separate repositories with separate CI pipelines.**

- `aprender` tests: "I can write APR files" ✅
- `realizar` tests: "I can load APR files" ✅
- Nobody tests: "What `aprender` writes, `realizar` can load correctly" ❌

This is the **contract boundary problem**. Each side tests its own implementation, but nobody tests the *contract between them*.

### Why #4: Why is there no cross-repository contract test?

**Because we treat format conversion as a serialization problem (structural) instead of a behavioral contract (semantic).**

Our testing philosophy has been:
1. Write the file ✅
2. Validate the file structure ✅
3. Load the file ✅
4. **Run inference and compare output** ❌ ← THIS IS THE MISSING STEP

We validate *syntax* (can we parse it?) but not *semantics* (does it produce correct output?). This is why `apr check` reports 10/10 PASS on a model that outputs garbage — it checks structure, not behavior.

### Why #5: Why do we validate syntax instead of semantics?

**Because semantic validation is expensive (requires running inference) and we optimized for fast CI feedback over correctness.**

Running actual inference on a 1.5B model takes 5-10 seconds. Structural validation takes milliseconds. We chose speed over safety.

This is the **fundamental mistake**: we treated the conversion pipeline as a *data transformation* when it's actually a *behavioral contract*. Data transformations can be validated structurally. Behavioral contracts can only be validated by observing behavior.

---

## The Systemic Pattern

Every conversion bug follows the same pattern:

```
Writer assumes convention X
           ↕ (no contract test)
Reader assumes convention Y
           ↕ (no behavioral test)
Validator checks structure, not behavior
           ↕
User gets garbage output from "validated" model
```

| Bug | Writer Assumes | Reader Assumes |
|-----|---------------|----------------|
| GH-186 | dtype 12 = Q4K | dtype 12 = F32 (fallback) |
| GH-189 | LayerNorm written as-is | LayerNorm must be non-zero |
| GH-190 | HuggingFace tensor names | GGUF tensor names |
| GH-185 | Tokenizer in metadata | Tokenizer in alternate path |

**It's the same bug four times wearing different masks.**

---

## Why Our Tooling Didn't Catch It

We built detection *after* each bug, not prevention *before*:

```
Timeline:
  Bug found → Write gate → Add to spec → Next bug found → Write gate → ...

What we should have:
  Contract test → Blocks merge → Bug never ships
```

Our 220 gates are **reactive** (detect known bugs) not **proactive** (prevent unknown bugs). We're playing whack-a-mole with a taxonomy of failure modes instead of testing the actual invariant:

> **"Converting a model and running inference MUST produce the same output as the original."**

This single invariant would have caught all four bugs. Instead, we wrote 22 format-specific gates that each catch one symptom.

---

## What We Must Change

### 1. The Golden Rule Test (Mandatory, Non-Negotiable)

```bash
# This MUST run in CI before any conversion code merges
apr run original.gguf -p "What is 2+2?" --max-tokens 10 > /tmp/expected.txt
apr convert original.gguf -o converted.apr
apr run converted.apr -p "What is 2+2?" --max-tokens 10 > /tmp/actual.txt
diff /tmp/expected.txt /tmp/actual.txt  # MUST be identical
```

**Cost:** 20 seconds per CI run
**Benefit:** Would have caught GH-186, GH-189, GH-190 before merge

This is not optional. This is the *definition* of correct conversion.

### 2. Cross-Repository Contract Tests

Create a shared test harness that runs in *both* `aprender` and `realizar` CI:

```rust
#[test]
fn contract_gguf_to_apr_roundtrip() {
    let gguf = download_test_model("tinyllama-1.1b-q4k.gguf");
    let apr = convert(&gguf, Format::Apr);

    let gguf_output = inference(&gguf, "2+2=", 5);
    let apr_output = inference(&apr, "2+2=", 5);

    assert_eq!(gguf_output.tokens, apr_output.tokens,
        "Converted model MUST produce identical tokens");

    for (i, (a, b)) in gguf_output.logits.iter()
        .zip(apr_output.logits.iter()).enumerate()
    {
        assert!((a - b).abs() < 1e-5,
            "Token {i}: logit diff {}", (a - b).abs());
    }
}
```

### 3. Behavioral Validation in `apr check`

`apr check` currently validates structure. Add a **Stage 11**:

```
Stage 11: Behavioral Validation
  Run inference with canonical prompt "2+2="
  Compare against known-good reference output
  FAIL if tokens differ
```

This is the difference between "model is structurally valid" and "model actually works."

### 4. Tensor Name Contract (APR Format Spec)

The APR format spec must define a **canonical tensor naming convention**:

```yaml
# APR v2 Tensor Naming Convention (MANDATORY)
# All tensors MUST use short GGUF-style names:
#   {layer}.{component}.{param}
# Examples:
#   0.q_proj.weight      (NOT model.layers.0.self_attn.q_proj.weight)
#   0.down_proj.weight   (NOT model.layers.0.mlp.down_proj.weight)
#   token_embd.weight    (NOT model.embed_tokens.weight)
```

Both writer and reader import this convention from a shared crate. No more independent assumptions.

### 5. Kill the Fallback Pattern

GH-186's root cause was a `_ => F32` default match arm that silently converted unknown dtypes. GH-190's root cause is pattern matching that "finds something" instead of requiring exact names.

**New rule:** At format boundaries, **fail loudly on unknown input**. Never default, never fallback, never pattern-match when you should exact-match.

```rust
// BEFORE (GH-186 pattern - WRONG)
match dtype_byte {
    0 => F32,
    1 => F16,
    _ => F32,  // Silent fallback → garbage
}

// AFTER (correct)
match dtype_byte {
    0 => Ok(F32),
    1 => Ok(F16),
    12 => Ok(Q4K),
    _ => Err(UnknownDType(byte)),  // Fail fast
}
```

### 6. Reduce Gate Count, Increase Invariant Count

We have 220 gates testing specific symptoms. We need **5 invariants** testing fundamental properties:

| Invariant | What It Tests | Catches |
|-----------|--------------|---------|
| **I-1**: Round-trip identity | convert(model) produces same inference | ALL conversion bugs |
| **I-2**: Tensor name bijection | writer names == loader names | GH-190 |
| **I-3**: No silent fallbacks | unknown input → error, never default | GH-186 |
| **I-4**: Statistical preservation | tensor stats unchanged after conversion | GH-189 |
| **I-5**: Tokenizer roundtrip | encode(decode(tokens)) == tokens | GH-185 |

5 invariants > 220 symptom-specific gates.

---

## Cost of Inaction

Every week we don't implement the Golden Rule Test:
- Engineering time debugging silent failures: ~40 hours per bug
- User trust erosion: models "validated" by our tooling produce garbage
- Specification bloat: we add more gates to catch more symptoms
- False confidence: "220 gates" sounds safe but catches 0 unknown bugs

---

## Action Items

| # | Action | Owner | Priority |
|---|--------|-------|----------|
| 1 | Add Golden Rule Test to aprender CI | aprender team | P0 (this week) |
| 2 | Add Golden Rule Test to realizar CI | realizar team | P0 (this week) |
| 3 | Define canonical tensor naming in APR v2 spec | spec owner | P0 |
| 4 | Add Stage 11 (behavioral) to `apr check` | apr-cli team | P1 |
| 5 | Audit all `_ =>` match arms in format code | aprender team | P1 |
| 6 | Create shared contract test crate | platform team | P1 |
| 7 | Replace 22 symptom gates with 5 invariants | apr-qa team | P2 |

---

## References

- Toyota Production System: "Build quality in, don't inspect it in"
- Popper: Test the invariant, not the instance
- Deming: "Cease dependence on inspection to achieve quality"
- Hyrum's Law: "With a sufficient number of users, all observable behaviors of your system will be depended on by somebody" — including the naming convention of tensors

---

*"We don't have a testing problem. We have an architecture problem. The conversion pipeline has no single source of truth for tensor naming, dtype mapping, or behavioral correctness. Until writer and reader share a contract, every fix creates the conditions for the next bug."*
