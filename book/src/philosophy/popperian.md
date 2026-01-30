# Popperian Falsification

> "The criterion of the scientific status of a theory is its falsifiability, or refutability, or testability."
> — Karl Popper, *Conjectures and Refutations* (1963)

## The Problem of Induction

No amount of passing tests can prove a model is correct. A million successful inferences don't guarantee the next one will work.

But a single failing test can prove incorrectness.

This is the **Asymmetry of Falsification**.

## Test to Fail, Not to Pass

Traditional testing asks: "Does this work?"

Popperian testing asks: "How can I break this?"

Every test scenario is a **falsifiable hypothesis**:

```
H: Model M follows instruction I regardless of quantization Q
E: Test with adversarial input
O: Observe output
```

## Outcomes

| Outcome | Meaning |
|---------|---------|
| **Corroborated** | Hypothesis survived this attempt at refutation |
| **Falsified** | Hypothesis refuted by evidence |

Note: "Corroborated" does not mean "proven correct" — only that the hypothesis has not yet been refuted.

## The Demarcation Criterion

A test is **scientific** if and only if there exists a possible observation that would mark it FALSIFIED.

We reject:
- **Tautologies**: Tests that always pass (`"output is a string"`)
- **Metaphysical statements**: Tests with subjective criteria (`"output is interesting"`)
- **Unfalsifiable claims**: Tests with catch-all exception handlers

## Verisimilitude

Quality is not binary (good/bad) but a measure of **verisimilitude** — closeness to truth.

A model achieves high quality by surviving increasingly severe attempts at falsification.

## Reference

- Popper, K. R. (1959). *The Logic of Scientific Discovery*. Hutchinson.
- Popper, K. R. (1963). *Conjectures and Refutations*. Routledge.
