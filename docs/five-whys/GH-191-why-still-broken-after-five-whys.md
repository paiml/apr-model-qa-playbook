# Five Whys: Why Are We STILL Broken After a Five Whys?

**Date:** 2026-01-30
**Scope:** GH-191 (discovered immediately after GH-190 fix)
**Predecessor:** [GH-190 Five Whys](GH-190-systemic-conversion-failures.md) — written hours earlier
**Method:** Toyota Production System — Meta Root Cause Analysis
**Root Cause:** CONFIRMED — dtype byte mapping mismatch in `realizar/src/gguf/loader.rs`

---

## The Problem Statement

At 18:00 today we wrote a thorough Five Whys analysis for GH-190. It correctly identified:
- The systemic root cause (no behavioral contract between writer and reader)
- The single test that would catch all conversion bugs (Golden Rule Test)
- 7 concrete action items with owners

At 20:33 PMAT-205 fixed the naming bug. We re-ran the Golden Rule Test.

**It still fails.** A second bug (GH-191: quantization data loss) was hiding underneath.

We diagnosed the disease, prescribed the cure, and the patient is still sick. **Why?**

---

## The Timeline (Today)

```
14:00  MQS score 24.2 — discover conversion produces garbage
15:00  File GH-190 with detailed evidence
16:00  Write Five Whys — identify Golden Rule Test as the fix
17:00  Build Golden Rule Test into executor.rs
18:00  Push Five Whys, Golden Rule Test, 5 new playbooks
20:33  PMAT-205 lands (naming fix) — 7 regression tests, 8135 total pass
20:45  Re-run Golden Rule Test — STILL FAILS
20:50  Discover GH-191: 0 quantized tensors, 308 F32, 10550 MB
21:00  File GH-191 ticket
```

**6 hours of analysis, 1 fix landed, and the pipeline is still broken.**

---

## Confirmed Root Cause: dtype Byte Mapping Mismatch in realizar

**Location:** `/home/noah/src/realizar/src/gguf/loader.rs`

The writer and reader of APR dtype bytes live **in the same file** and **disagree with each other**.

### The Writer: `dtype_to_byte()` (lines 1711-1730)

```rust
fn dtype_to_byte(dtype: &str) -> u8 {
    match dtype {
        "F32" => 0,
        "F16" => 1,
        "BF16" => 2,
        "I8" => 3,   "I16" => 4,   "I32" => 5,   "I64" => 6,   "U8" => 7,
        "Q4_K" => 8,        // ← writes 8 for Q4_K
        "Q6_K" => 9,
        "Q8_0" => 10,
        "Q4_0" => 11,
        "Q5_K" => 12,
        "Q3_K" => 13,
        "Q2_K" => 14,
        _ => 0,             // ← SILENT FALLBACK TO F32
    }
}
```

### The Reader: `TensorEntry::from_binary()` (around line 781)

Maps byte 8 back to `"Q4"` — **NOT** `"Q4_K"`. The forward and reverse mappings are inconsistent within the same file.

### The Matcher: `dtype_to_ggml_qtype()` (around line 537)

Matches against `"Q4_K"` but receives `"Q4"` from the reader. No match → `None` → treated as F32.

### The Full Chain

```
Writer: "Q4_K" → dtype_to_byte() → byte 8       ← WRITES correctly
Reader: byte 8 → from_binary()   → "Q4"         ← READS incorrectly
Matcher: "Q4" vs "Q4_K"          → NO MATCH      ← FALLS THROUGH
Fallback: None                    → F32           ← SILENT CORRUPTION
```

### Complete Mismatch Table

```
┌───────────┬───────────────────────────┬───────────────────────────────┬────────────────────────────┐
│ GGML type │ dtype_to_byte() WRITES    │ from_binary() READS BACK AS  │ dtype_to_ggml_qtype match? │
├───────────┼───────────────────────────┼───────────────────────────────┼────────────────────────────┤
│ F32       │ 0                         │ "F32"                         │ ✅ Correct                 │
│ F16       │ 1                         │ "F16"                         │ ✅ Correct                 │
│ Q4_K      │ 8                         │ "Q4"                          │ ❌ NO ("Q4_K" expected)     │
│ Q5_K      │ 12                        │ ???                           │ ❌ NO                       │
│ Q6_K      │ 9                         │ "Q8_0"                        │ ❌ NO ("Q6_K" expected)     │
│ Q8_0      │ 10                        │ unknown → "F32"               │ ❌ NO                       │
│ Q4_0      │ 11                        │ ???                           │ ❌ NO                       │
│ Q3_K      │ 13                        │ ???                           │ ❌ NO                       │
│ Q2_K      │ 14                        │ ???                           │ ❌ NO                       │
└───────────┴───────────────────────────┴───────────────────────────────┴────────────────────────────┘
```

**Every quantized type except F32/F16 is broken.** The writer and reader were written independently and never round-trip tested.

### Three Silent Fallbacks in One File

1. **`dtype_to_byte()` line 1728:** `_ => 0` — unknown string → F32 byte
2. **`qtype_to_dtype()` line 1706:** `_ => "F32"` — unknown GGML type → F32 string
3. **`dtype_to_ggml_qtype()`:** `None` → treated as F32 by caller

Three fallbacks, all to F32, all silent, all in the same file. Any one of them triggers the entire corruption chain.

### Why This Is GH-186 Redux

GH-186 was: "GGML dtype 12 (Q4_K) read as F32 via silent `_ => F32` fallback."
GH-191 is: "APR byte 8 (Q4_K) read as `"Q4"` via inconsistent reverse mapping → F32 fallback."

Same pattern. Same file. Same fallback. Different code path.

---

## Five Whys

### Why #1: Why is the pipeline still broken after PMAT-205?

**Because there are TWO independent bugs, and PMAT-205 only fixed one.**

- GH-190: Tensor naming — `model.` prefix (FIXED)
- GH-191: Quantization data — Q4_K_M bytes loaded as F32 (UNFIXED)

Both bugs produce the same symptom (garbage output). Fixing one leaves the other. The Golden Rule Test correctly fails for both — but we only discovered GH-191 after GH-190 was fixed.

### Why #2: Why didn't we discover GH-191 at the same time as GH-190?

**Because GH-190 masked GH-191. When tensor names are wrong, you can't diagnose whether tensor *data* is also wrong.**

```
GH-190 active (names wrong):
  → Loader can't find tensors → uninitialized memory → garbage
  → Can't tell if data would also be wrong if names were right

GH-190 fixed (names correct):
  → Loader finds tensors by name → reads their data → data is garbage
  → NOW we can see GH-191
```

Bug masking is a compound failure mode. The first bug hides the second. You peel the onion one layer at a time. Each fix reveals the next bug.

### Why #3: Why does our diagnostic process discover bugs sequentially instead of all at once?

**Because we debug symptoms, not invariants.**

Our diagnostic loop:
```
1. Observe symptom (garbage output)
2. Find A cause (wrong names)
3. Fix THAT cause
4. Retest
5. Same symptom persists → find ANOTHER cause
6. Fix THAT cause
7. Retest
8. ...repeat until symptom gone
```

This is O(n) where n = number of co-occurring bugs. If there are 5 bugs stacked, we cycle through the fix-retest loop 5 times.

What we should do:
```
1. Observe symptom (garbage output)
2. Enumerate ALL invariants that could produce this symptom
3. Test EVERY invariant independently
4. Fix ALL failures
5. Retest once
```

We already know the 5 invariants from the first Five Whys:

| Invariant | Tests | Would Catch | Did We Check? |
|-----------|-------|-------------|---------------|
| I-1: Round-trip identity | convert → inference → diff | GH-190 + GH-191 | Only after fix |
| I-2: Tensor name bijection | writer names == loader names | GH-190 | Yes (diff-tensors) |
| I-3: No silent fallbacks | unknown → error | GH-186 | No |
| **I-4: Statistical preservation** | **tensor stats match after convert** | **GH-191** | **NO** |
| I-5: Tokenizer roundtrip | encode(decode(x)) == x | GH-185 | No |

**We checked I-2 (names) and filed GH-190. We never checked I-4 (data integrity) so we missed GH-191.**

If we had run `apr rosetta fingerprint` on both the GGUF source and APR output and diffed the tensor checksums, we would have seen the data corruption immediately — independent of whether names were right.

### Why #4: Why did we only check one invariant when we had identified five?

**Because we diagnosed reactively (chasing the symptom) instead of proactively (testing all invariants).**

The Five Whys we wrote at 16:00 identifies 5 invariants. The Golden Rule Test we built at 17:00 tests I-1. But we didn't run all 5 invariants as a diagnostic checklist against the current model. We found the naming bug (I-2 violation), stopped looking, and waited for the fix.

This is the **premature root cause fallacy**: we found *a* cause and assumed it was *the* cause.

```
What we did:
  Symptom → First cause found → "That must be it" → Wait for fix → Still broken → Surprise

What we should do:
  Symptom → Test ALL invariants → List ALL violations → Fix ALL of them → Verify
```

### Why #5: Why do we keep falling into the same diagnostic pattern even after writing an analysis that warns against it?

**Because analysis is not implementation. We wrote the right prescription but didn't take the medicine.**

The GH-190 Five Whys says:

> *"5 invariants > 220 symptom-specific gates"*

And then we went right back to chasing a single symptom. The Five Whys is a document sitting in `docs/`. It's not a checklist in our diagnostic runbook. It's not automated. It's not enforced. It's prose wisdom that gets forgotten under the urgency of "the pipeline is broken, find the bug."

**Insight is necessary but not sufficient.** The insight must be encoded into process:

```
Insight (what we had):     "Test all invariants, not just symptoms"
Process (what we lacked):   A diagnostic runbook that REQUIRES testing all 5 invariants
Automation (what we need):  A single command that tests all 5 invariants in parallel
```

---

## The Meta-Pattern

We've now done Five Whys twice. Both times we identified the correct systemic fix. Both times the fix wasn't implemented fast enough to prevent the next incident. This is itself a pattern:

```
Five Whys #1 (GH-190):
  Insight: "Need behavioral contract tests (Golden Rule Test)"
  Status: Built in executor.rs, NOT in aprender/realizar CI
  Result: PMAT-205 merged without Golden Rule gate → GH-191 shipped

Five Whys #2 (GH-191, this document):
  Insight: "Need to test ALL invariants, not chase symptoms one at a time"
  Status: Written in docs
  Risk: Next fix will be merged without invariant checklist → GH-192 ships
```

The pattern is: **we analyze faster than we implement the fixes our analysis prescribes.**

---

## The Onion Model

Conversion bugs stack like onion layers. Each fix peels one layer, revealing the next:

```
Layer 0 (outermost): Names wrong → can't find tensors → garbage     [GH-190 — FIXED by PMAT-205]
Layer 1:             dtype byte mapping inconsistent → F32 fallback  [GH-191 — ROOT CAUSE FOUND]
                     realizar/src/gguf/loader.rs:
                       dtype_to_byte() writes 8 for Q4_K
                       from_binary() reads 8 as "Q4" (not "Q4_K")
                       Three silent _ => F32 fallbacks in same file
Layer 2 (unknown):   ??? → ??? → garbage?
```

We've peeled Layer 0 (GH-190) and root-caused Layer 1 (GH-191). The fix is straightforward: make `from_binary()` the exact inverse of `dtype_to_byte()`, and replace all three `_ => F32` fallbacks with explicit errors.

We won't know if Layer 2 exists until GH-191 is fixed and the Golden Rule Test runs again.

**This is why I-1 (round-trip identity) is the supreme invariant.** It's the only test that proves ALL layers are correct simultaneously, without having to discover and test each one individually.

---

## Diagnosis vs. Treatment Gap

| What We Prescribed (GH-190 Five Whys) | Current Status |
|---------------------------------------|---------------|
| Golden Rule Test in aprender CI | NOT IMPLEMENTED |
| Golden Rule Test in realizar CI | NOT IMPLEMENTED |
| Shared contract crate | NOT IMPLEMENTED |
| Stage 11 behavioral in `apr check` | NOT IMPLEMENTED |
| Audit all `_ =>` fallbacks | NOT IMPLEMENTED |
| Canonical tensor naming spec | NOT IMPLEMENTED |
| 5 invariants replace symptom gates | NOT IMPLEMENTED |

**0/7 action items from the first Five Whys are implemented.** The analysis was correct. The implementation hasn't happened. This is why we're still broken.

---

## What We Must Change (Upgraded From GH-190 Five Whys)

### 1. Diagnostic Runbook: All 5 Invariants (IMMEDIATE)

When any conversion bug is suspected, run ALL of these — not just the first one that fails:

```bash
#!/bin/bash
# CONVERSION DIAGNOSTIC RUNBOOK — ALL 5 INVARIANTS
set -euo pipefail

GGUF="$1"
APR="/tmp/diag-$(date +%s).apr"

echo "=== INVARIANT DIAGNOSTIC: Testing all 5 invariants ==="

# Convert
apr convert "$GGUF" -o "$APR" --verbose

# I-1: Round-trip identity (THE supreme test)
echo "--- I-1: Round-trip identity ---"
EXPECTED=$(apr run "$GGUF" -p "What is 2+2?" --max-tokens 10 2>/dev/null)
ACTUAL=$(apr run "$APR" -p "What is 2+2?" --max-tokens 10 2>/dev/null)
if [ "$EXPECTED" = "$ACTUAL" ]; then
    echo "I-1: PASS"
else
    echo "I-1: FAIL"
    echo "  Expected: $EXPECTED"
    echo "  Actual:   $ACTUAL"
fi

# I-2: Tensor name bijection
echo "--- I-2: Tensor name bijection ---"
apr rosetta diff-tensors "$GGUF" "$APR" 2>&1 | head -20

# I-3: No silent fallbacks (check for F32 fallback on quantized model)
echo "--- I-3: Silent fallback check ---"
apr run "$APR" -p "test" --max-tokens 1 2>&1 | grep -E "quantized|F32|fallback"

# I-4: Statistical preservation
echo "--- I-4: Statistical preservation ---"
apr rosetta fingerprint "$GGUF" --json > /tmp/diag-fp-gguf.json
apr rosetta fingerprint "$APR" --json > /tmp/diag-fp-apr.json
apr rosetta validate-stats /tmp/diag-fp-gguf.json /tmp/diag-fp-apr.json 2>&1 | head -20

# I-5: Tokenizer roundtrip
echo "--- I-5: Tokenizer roundtrip ---"
apr rosetta compare-inference "$GGUF" "$APR" \
    --prompt "Hello" --max-tokens 1 2>&1 | head -20

echo "=== DIAGNOSTIC COMPLETE ==="
```

This runs in <60 seconds and tests EVERYTHING. No more peeling one layer at a time.

### 2. Definition of Done for Conversion Fixes

A conversion fix is NOT done when:
- ~~Its regression test passes~~ (PMAT-205 had 7 regression tests — still broken)
- ~~`apr check` passes~~ (10/10 on garbage model)
- ~~CI is green~~ (8135 tests pass — none test round-trip)

A conversion fix IS done when:
- **The Golden Rule Test passes:** `convert(gguf) → inference == gguf inference`
- **All 5 invariants pass** on at least one real model (not synthetic)
- **The diagnostic runbook** (above) produces 5x PASS

### 3. Gate the Fix, Not Just the Feature

PMAT-205 was merged because:
- 7 new regression tests pass ✅
- 8135 total tests pass ✅
- Code review approved ✅

PMAT-205 should not have been merged until:
- Golden Rule Test passes on Qwen2.5-Coder-1.5B-Q4_K_M ❌
- All 5 invariants pass ❌

**The merge criteria was structural (tests pass) when it should have been behavioral (model works).**

### 4. Parallel Invariant Testing in CI

Don't run invariants sequentially (fix → test → fail → fix → test → ...).
Run ALL invariants on EVERY PR:

```yaml
# CI pipeline for conversion code
conversion-invariants:
  runs-on: gpu-runner
  steps:
    - name: Convert test model
      run: apr convert test-model.gguf -o test.apr
    - name: I-1 Round-trip identity
      run: ./scripts/test-invariant-1.sh
    - name: I-2 Tensor name bijection
      run: ./scripts/test-invariant-2.sh
    - name: I-3 No silent fallbacks
      run: ./scripts/test-invariant-3.sh
    - name: I-4 Statistical preservation
      run: ./scripts/test-invariant-4.sh
    - name: I-5 Tokenizer roundtrip
      run: ./scripts/test-invariant-5.sh
```

One failing invariant blocks merge. ALL invariants run, even if the first fails. You see the full picture on every PR.

---

## The Deeper Lesson

### Why Does This Keep Happening?

```
We analyze well.
We implement slowly.
Bugs ship in the gap between insight and enforcement.
```

The GH-190 Five Whys is a good document. It identifies the right problems. But it's a **document**, not a **gate**. Documents inform humans. Gates block code. We have enough documents. We need gates.

### The Three Speeds

```
Speed of bugs:        Minutes (one bad mapping → garbage)
Speed of analysis:    Hours (Five Whys, tickets, invariants)
Speed of enforcement: Days/weeks (CI gates, shared crates, runbooks)

Bugs > Analysis > Enforcement

We're losing because enforcement is slowest.
```

The fix is not more analysis. It's faster enforcement:

1. **Today:** Diagnostic runbook script checked into repo
2. **This PR:** Golden Rule Test as a merge gate in aprender CI
3. **This sprint:** All 5 invariants as CI gates
4. **Never again:** Conversion fix merged without behavioral verification

### The One-Sentence Summary

> *"We keep diagnosing correctly and implementing slowly. The bugs ship in the gap. Close the gap: make the Golden Rule Test a merge gate, not a post-mortem insight."*

---

## Updated Action Items (Supersedes GH-190 Five Whys)

| # | Action | Owner | Priority | Status |
|---|--------|-------|----------|--------|
| 1 | **Diagnostic runbook script** in repo | apr-qa team | P0 TODAY | ✅ DONE (`scripts/diagnose-conversion.sh`) |
| 2 | **Fix `from_binary()` reverse mapping** in realizar | realizar team | P0 THIS PR | ROOT CAUSE FOUND |
| 3 | **Kill 3 silent `_ => F32` fallbacks** in `loader.rs` | realizar team | P0 THIS PR | AUDIT COMPLETE (lines 1706, 1728, and dtype_to_ggml_qtype) |
| 4 | **Golden Rule Test as merge gate** in realizar CI | realizar team | P0 THIS PR | BLOCKED (need items 2+3) |
| 5 | **All 5 invariants in CI** | platform team | P0 THIS SPRINT | NOT STARTED |
| 6 | **Definition of Done** for conversion fixes = behavioral | eng leads | P0 | NOT STARTED |
| 7 | Shared contract crate | platform team | P1 | NOT STARTED |
| 8 | Stage 11 behavioral in `apr check` | apr-cli team | P1 | NOT STARTED |
| 9 | Round-trip test for `dtype_to_byte` ↔ `from_binary` | realizar team | P0 | NOT STARTED |

### Specific Fix for GH-191

**File:** `realizar/src/gguf/loader.rs`

**Fix 1:** Make `from_binary()` the exact inverse of `dtype_to_byte()`:
```rust
// byte → string (MUST be exact inverse of dtype_to_byte)
fn byte_to_dtype(byte: u8) -> Result<&'static str, RealizarError> {
    match byte {
        0 => Ok("F32"),
        1 => Ok("F16"),
        2 => Ok("BF16"),
        3 => Ok("I8"),    4 => Ok("I16"),   5 => Ok("I32"),   6 => Ok("I64"),   7 => Ok("U8"),
        8 => Ok("Q4_K"),   // NOT "Q4"
        9 => Ok("Q6_K"),   // NOT "Q8_0"
        10 => Ok("Q8_0"),
        11 => Ok("Q4_0"),
        12 => Ok("Q5_K"),
        13 => Ok("Q3_K"),
        14 => Ok("Q2_K"),
        _ => Err(RealizarError::UnsupportedOperation {
            operation: "byte_to_dtype".into(),
            reason: format!("Unknown APR dtype byte: {} — refusing silent F32 fallback", byte),
        }),
    }
}
```

**Fix 2:** Kill silent fallbacks at lines 1706 and 1728:
```rust
// qtype_to_dtype line 1706: _ => "F32"  →  _ => panic!("Unknown GGML qtype: {}", qtype)
// dtype_to_byte line 1728:  _ => 0      →  _ => panic!("Unknown dtype string: {}", dtype)
```

**Fix 3:** Add round-trip invariant test:
```rust
#[test]
fn dtype_byte_roundtrip() {
    for dtype in ["F32", "F16", "BF16", "Q4_K", "Q5_K", "Q6_K", "Q8_0", "Q4_0", "Q3_K", "Q2_K"] {
        let byte = dtype_to_byte(dtype);
        let back = byte_to_dtype(byte).unwrap();
        assert_eq!(dtype, back, "Round-trip failed: {} → {} → {}", dtype, byte, back);
    }
}
```

---

## References

- [GH-190 Five Whys](GH-190-systemic-conversion-failures.md) — the predecessor analysis (correct but un-enforced)
- [GH-190 Ticket](../tickets/GH-190-GGUF-APR-CONVERSION-GARBAGE-OUTPUT.md) — naming bug (FIXED)
- [GH-191 Ticket](../tickets/GH-191-APR-QUANTIZATION-DATA-LOSS.md) — quantization bug (OPEN)
- Deming: *"It is not enough to do your best; you must know what to do, and then do your best."*
- Taiichi Ohno: *"Having no problems is the biggest problem of all."* — We had the analysis. We didn't have the enforcement.

---

*"We don't have an analysis problem. We have an implementation-of-analysis problem. The Five Whys we wrote six hours ago correctly predicts everything that happened since. The prescription was right. We just didn't fill it fast enough."*
