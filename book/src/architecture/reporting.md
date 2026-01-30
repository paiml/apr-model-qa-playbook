# Reporting

The `apr-qa-report` crate calculates MQS scores and generates reports.

## MQS Calculation

```rust
let calculator = MqsCalculator::new();
let score = calculator.calculate(&evidence_collector);
```

### Scoring Formula

```
Raw Score = Σ(category_points) - Σ(penalties)
Normalized = (raw_score / max_possible) × 100
```

### Category Weights

| Category | Weight | Prefix |
|----------|--------|--------|
| Quality | 30% | F-QUAL |
| Performance | 25% | F-PERF |
| Stability | 20% | F-STAB |
| Compatibility | 15% | F-COMP |
| Edge Cases | 10% | F-EDGE |

### Grade Mapping

| Grade | Threshold |
|-------|-----------|
| A+ | ≥95 |
| A | ≥90 |
| A- | ≥85 |
| B+ | ≥80 |
| B | ≥75 |
| B- | ≥70 |
| C | ≥60 |
| D | ≥50 |
| F | <50 |

## Report Formats

### JUnit XML

```rust
let report = JunitReport::new("TestSuite");
let xml = report.generate(&evidence, &score)?;
```

Output includes:
- Test counts (passed, failed, errors, skipped)
- MQS properties
- Individual test cases with timing

### HTML Dashboard

```rust
let report = HtmlReport::new();
let html = report.generate(&evidence, &score)?;
```

Features:
- MQS gauge visualization
- Category breakdown charts
- Gateway status indicators
- Searchable test results

## Popperian Scoring

```rust
pub struct PopperianScore {
    pub corroborated: u32,    // Hypotheses that survived
    pub falsified: u32,       // Hypotheses refuted
    pub corroboration_ratio: f64,
}
```

The Popperian score measures the model's "verisimilitude" — how well it survives attempts at refutation.
