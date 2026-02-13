# Model Qualification Score (MQS)

The MQS is a 0-1000 point score (normalized to 0-100) that measures model readiness for production.

## Gateway Logic

Five gateways must pass or the **entire score is zeroed**:

| Gateway | Check | Failure Impact |
|---------|-------|----------------|
| G0 | config.json matches tensor metadata | Score = 0 |
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

| Grade | Raw Score | Status |
|-------|-----------|--------|
| A+ | 950-1000 | CERTIFIED |
| A | 900-949 | CERTIFIED |
| B+ | 850-899 | CERTIFIED |
| B | 800-849 | PROVISIONAL |
| C | 700-799 | PROVISIONAL |
| F | <700 | BLOCKED |

## Tier-Aware Scoring

The certification system uses tier-aware scoring:

| Tier | Pass Threshold | Score on Pass | Grade | Status |
|------|----------------|---------------|-------|--------|
| MVP | ≥90% | 800 | B | PROVISIONAL |
| Full | ≥95% | 950+ | A+ | CERTIFIED |

```rust
// Tier-aware scoring functions (apr-qa-certify)
use apr_qa_certify::{CertificationTier, score_from_tier, status_from_tier};

// MVP tier with 95% pass rate
let score = score_from_tier(CertificationTier::Mvp, 0.95, false);
assert_eq!(score, 800); // B grade

// Full tier with 98% pass rate
let score = score_from_tier(CertificationTier::Full, 0.98, false);
assert!(score >= 950); // A+ grade
```

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
Gateways: G0 ✓, G1 ✓, G2 ✓, G3 ✓, G4 ✓

Quality:       280/300 (93%)
Performance:   220/250 (88%)
Stability:     190/200 (95%)
Compatibility: 140/150 (93%)
Edge Cases:     90/100 (90%)

Raw Score:     920/1000
Normalized:    92.0
Grade:         A
```
