# Clippy Landmines Reference

Full clippy configuration for the APR Model QA Playbook workspace. Knowing which lints are active (and where they're allowed) prevents wasted time fighting the linter.

## Workspace-Level Configuration

**Source:** Root `Cargo.toml` `[workspace.lints.clippy]`

```toml
[workspace.lints.rust]
unsafe_code = "deny"
missing_docs = "warn"

[workspace.lints.clippy]
all = { level = "warn", priority = -1 }
pedantic = { level = "warn", priority = -1 }
nursery = { level = "warn", priority = -1 }
unwrap_used = "deny"
expect_used = "warn"
panic = "deny"
```

All crates inherit via `[lints] workspace = true`.

## Per-Crate Allow Lists

### apr-qa-gen
```rust
#![forbid(unsafe_code)]
#![allow(clippy::missing_const_for_fn)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::needless_pass_by_value)]
#![allow(clippy::needless_raw_string_hashes)]
#![cfg_attr(test, allow(clippy::expect_used, clippy::unwrap_used))]
#![cfg_attr(test, allow(clippy::redundant_closure_for_method_calls))]
#![cfg_attr(test, allow(clippy::redundant_clone))]
```

### apr-qa-runner
```rust
#![forbid(unsafe_code)]
#![allow(clippy::missing_const_for_fn)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::unnecessary_wraps)]
#![allow(clippy::unused_self)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::needless_pass_by_value)]
#![allow(clippy::needless_borrows_for_generic_args)]
#![allow(clippy::single_char_pattern)]
#![cfg_attr(test, allow(clippy::expect_used, clippy::unwrap_used))]
#![cfg_attr(test, allow(clippy::redundant_closure_for_method_calls))]
#![cfg_attr(test, allow(clippy::redundant_clone))]
#![cfg_attr(test, allow(clippy::uninlined_format_args))]
#![cfg_attr(test, allow(clippy::cast_sign_loss))]
```

### apr-qa-report
```rust
#![forbid(unsafe_code)]
#![allow(clippy::missing_const_for_fn)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::needless_pass_by_value)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::unused_self)]
#![allow(clippy::format_push_string)]
#![allow(clippy::needless_raw_string_hashes)]
#![allow(clippy::needless_borrows_for_generic_args)]
#![allow(clippy::if_same_then_else)]
#![allow(clippy::manual_let_else)]
#![allow(clippy::single_match_else)]
#![allow(clippy::single_char_pattern)]
#![allow(clippy::suboptimal_flops)]
#![allow(clippy::imprecise_flops)]
#![allow(clippy::items_after_statements)]
#![allow(clippy::match_same_arms)]
#![allow(clippy::uninlined_format_args)]
#![allow(clippy::option_if_let_else)]
#![cfg_attr(test, allow(clippy::expect_used, clippy::unwrap_used))]
#![cfg_attr(test, allow(clippy::redundant_closure_for_method_calls))]
#![cfg_attr(test, allow(clippy::redundant_clone))]
#![cfg_attr(test, allow(clippy::float_cmp))]
```

### apr-qa-cli
```rust
#![allow(clippy::doc_markdown)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::struct_excessive_bools)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::missing_const_for_fn)]
#![cfg_attr(test, allow(clippy::expect_used, clippy::unwrap_used))]
```

### apr-qa-certify
```rust
#![forbid(unsafe_code)]
// Minimal allows - most restrictive crate
```

## Most Commonly Triggered Lints

These are the lints most likely to catch you when writing new code or tests.

### 1. `float_cmp` (pedantic)
**Triggers:** `==` comparison on floats
```rust
// BAD
if score == 0.95 { ... }
assert_eq!(score, 0.95);

// GOOD
if (score - 0.95).abs() < f64::EPSILON { ... }
assert!((score - 0.95).abs() < 1e-10);
// Or use range assertions
assert!(score >= 0.9 && score <= 1.0);
```
**Allowed in tests:** Only in `apr-qa-report` via `cfg_attr(test, allow(clippy::float_cmp))`

### 2. `option_if_let_else` (nursery)
**Triggers:** `if let Some(x) = opt { a } else { b }`
```rust
// BAD
let result = if let Some(val) = option {
    process(val)
} else {
    default_value()
};

// GOOD
let result = option.map_or_else(default_value, process);
// Or
let result = option.map_or(fallback, |val| process(val));
```
**Allowed in:** `apr-qa-report`

### 3. `manual_let_else` (pedantic)
**Triggers:** Pattern match that returns/continues on failure
```rust
// BAD
let val = match option {
    Some(v) => v,
    None => return Err(Error::Execution("missing".into())),
};

// GOOD
let Some(val) = option else {
    return Err(Error::Execution("missing".into()));
};
```
**Allowed in:** `apr-qa-report`

### 4. `doc_link_with_quotes` (pedantic)
**Triggers:** Quoted strings in doc comments that look like links
```rust
// BAD
/// See ["quoted name"] for details.

// GOOD
/// See `"quoted name"` for details.
```

### 5. `or_fun_call` (nursery)
**Triggers:** `.unwrap_or()` with a function call argument
```rust
// BAD
let s = option.unwrap_or(String::new());
let v = option.unwrap_or(Vec::new());

// GOOD
let s = option.unwrap_or_default();
let v = option.unwrap_or_default();
```

### 6. `unwrap_used` (DENIED)
**Triggers:** Any `.unwrap()` in library code
```rust
// BAD (compile error in library code)
let value = map.get("key").unwrap();

// GOOD
let value = map.get("key").ok_or(Error::Execution("key missing".into()))?;

// OK in tests
#[test]
fn my_test() {
    let value = map.get("key").unwrap(); // Allowed via cfg_attr
}
```

### 7. `panic` (DENIED)
**Triggers:** `panic!()`, `todo!()`, `unimplemented!()` in library code
```rust
// BAD
panic!("unexpected state");
todo!("implement later");

// GOOD
return Err(Error::Execution("unexpected state".into()));
```

### 8. `too_many_lines` (pedantic)
**Triggers:** Functions > ~100 lines
```rust
// Fix with function-level allow
#[allow(clippy::too_many_lines)]
fn long_function() { ... }
```

### 9. `struct_excessive_bools` (pedantic)
**Triggers:** Structs with many bool fields
```rust
// Fix with struct-level allow (already done for ExecutionConfig)
#[allow(clippy::struct_excessive_bools)]
pub struct Config { ... }
```

### 10. `missing_docs` (WARNED)
**Triggers:** Public items without doc comments
```rust
// BAD (warning)
pub fn my_function() { ... }

// GOOD
/// Does the thing.
pub fn my_function() { ... }
```

## Function-Level Allows in Executor

The executor uses several function-level allows. Follow this pattern when needed:

```rust
#[allow(clippy::too_many_lines)]
pub fn execute(&mut self, playbook: &Playbook) -> Result<ExecutionResult> { ... }

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn run_extended_tests(&mut self, playbook: &Playbook) -> (u32, u32) { ... }
```

## Quick Decision Matrix

| Writing... | Key Concern | Action |
|-----------|-------------|--------|
| Library code | `unwrap_used` denied | Use `?` and `ok_or()` |
| Test code | `unwrap`/`expect` allowed | Use freely |
| Scoring math | `float_cmp` | Use range assertions |
| Long functions | `too_many_lines` | Add `#[allow]` attribute |
| Struct with bools | `struct_excessive_bools` | Add `#[allow]` attribute |
| Doc comments | `doc_link_with_quotes` | Use backticks for quoted strings |
| String defaults | `or_fun_call` | Use `unwrap_or_default()` |
| Option matching | `option_if_let_else` / `manual_let_else` | Use `map_or` or `let-else` |
