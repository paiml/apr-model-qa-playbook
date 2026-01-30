# Model Qualification Score (MQS)

The MQS is a 0-1000 point score (normalized to 0-100) that measures model readiness for production.

## Gateway Logic

Four gateways must pass or the **entire score is zeroed**:

| Gateway | Check | Failure Impact |
|---------|-------|----------------|
| G1 | Model loads successfully | Score = 0 |
| G2 | Basic inference works | Score = 0 |
| G3 | No crashes or panics | Score = 0 |
| G4 | Output is not garbage | Score = 0 |

```rust
if !gateways_passed {
    return MqsScore { raw_score: 0, .. };
}
```

## Category Scoring

After gateways pass, points are awarded by category:

| Category | Weight | Max Points | Gate Prefix |
|----------|--------|------------|-------------|
| Quality | 30% | 300 | F-QUAL |
| Performance | 25% | 250 | F-PERF |
| Stability | 20% | 200 | F-STAB |
| Compatibility | 15% | 150 | F-COMP |
| Edge Cases | 10% | 100 | F-EDGE |

## Grade Mapping

| Grade | Normalized Score |
|-------|------------------|
| A+ | ≥95 |
| A | ≥90 |
| A- | ≥85 |
| B+ | ≥80 |
| B | ≥75 |
| B- | ≥70 |
| C | ≥60 |
| D | ≥50 |
| F | <50 |

## Penalty System

Penalties are applied for:

| Violation | Penalty |
|-----------|---------|
| P0 failure (gateway) | Score = 0 |
| P1 failure | -50 points |
| P2 failure | -20 points |
| Timeout | -30 points |

## Example Score

```
Gateways: G1 ✓, G2 ✓, G3 ✓, G4 ✓

Quality:       280/300 (93%)
Performance:   220/250 (88%)
Stability:     190/200 (95%)
Compatibility: 140/150 (93%)
Edge Cases:     90/100 (90%)

Raw Score:     920/1000
Normalized:    92.0
Grade:         A
```
