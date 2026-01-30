# Toyota Way

> "Stop the line. Fix it now. Never pass a defect to the next process."
> — Taiichi Ohno, *Toyota Production System* (1988)

This framework rejects the software industry's normalization of technical debt.

## TPS Principles Applied

| Principle | Application |
|-----------|-------------|
| **Jidoka** | Execution halts on first P0 failure; no silent failures |
| **Poka-Yoke** | Schema validation prevents malformed playbooks |
| **Genchi Genbutsu** | All metrics from actual inference, never synthetic |
| **Heijunka** | Load-balanced parallel execution |
| **Kaizen** | Continuous refinement via mutation testing |
| **Muda Elimination** | Zero redundant test cases |

## Jidoka in Practice

When a gateway check fails, the entire system stops:

```rust
pub enum FailurePolicy {
    StopOnFirst,   // Stop on any failure
    StopOnP0,      // Stop on gateway failures only
    CollectAll,    // Continue, collect all failures
}
```

The default `StopOnP0` policy implements Jidoka — the system autonomously stops when a critical defect is detected.

## Poka-Yoke: Error-Proofing

Playbook schema validation prevents invalid configurations from executing:

```yaml
# This will fail schema validation
test_matrix:
  modalities: []  # Error: minItems is 1
```

## Zero SATD

This codebase contains no:
- TODO markers
- FIXME markers
- HACK markers
- "Known issues" documentation

If something needs fixing, it gets fixed now.

## Reference

Liker, J. K. (2004). *The Toyota Way: 14 Management Principles from the World's Greatest Manufacturer*. McGraw-Hill. ISBN 978-0071392310.
