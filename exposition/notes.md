# Experiment Notes

## Exp 01 — Confounder mirage

**Setup:** Z → F, Z → R. F has zero causal effect on R.
**Result:** Naive R ~ F: beta = 0.482, p ≈ 5e-242. Adding Z: beta ≈ 0.01, p = 0.37.
**Lesson:** A hidden common cause makes a useless factor look hugely significant.
Controlling for the confounder removes the fake premium.
**Twist:** The chain F → Z → R (F truly causal, Z a mediator) gives the same
regression output. Data alone can't tell fork from chain (Markov equivalence),
so controlling for Z can either expose a fake factor or erase a real one.

## Exp 02 — Collider mirage

**Setup:** X → Y (true beta = 0.5), and X → C ← Y. C is a common effect.
**Result:** Correct Y ~ X: beta = 0.514, adjR² = 0.21.

| a (X→C) | b (Y→C) | beta_x with C | adjR² |
| -------- | -------- | ------------- | ------ |
| 1        | 0        | 0.502         | 0.207  |
| 0        | 1        | 0.270         | 0.608  |
| 1        | 1        | −0.230       | 0.608  |
| 2        | 2        | −0.682       | 0.843  |

**Lessons:**

- Sign flip needs both arrows (a true collider). Y → C alone only shrinks beta.
- The R² gain comes only from b: C contains a copy of Y, so the model
  "predicts" returns with information from the returns themselves.
  Higher R² here is leakage, not skill, and can't be traded.
- Every standard metric (p-value, adjR²) prefers the wrong model.
- The Primer's Exhibit 5 shows this on Barra data (proprietary). This experiment
  simulates the mechanism; the public-data replication comes next.
