# Experiment Notes

## Exp 01 — Confounder mirage

**Setup:** Z → F, Z → R. F has zero effect on R.
**Result:** R ~ F: beta = 0.482, p ≈ 5e-242. Add Z: beta ≈ 0.01, p = 0.37.
**Lesson:** A hidden common cause makes a useless factor look significant.
Controlling for it removes the fake premium.
**Twist:** The chain F → Z → R (F truly works through Z) gives the same output.
Data can't tell them apart (Markov equivalence), so controlling for Z can
expose a fake factor or erase a real one.

## Exp 02 — Collider mirage

**Setup:** X → Y (true beta = 0.5), X → C ← Y.
**Result:** Y ~ X: beta = 0.514, adjR² = 0.21.

| a (X→C) | b (Y→C) | beta_x with C | adjR² |
| -------- | -------- | ------------- | ------ |
| 1        | 0        | 0.502         | 0.207  |
| 0        | 1        | 0.270         | 0.608  |
| 1        | 1        | −0.230       | 0.608  |
| 2        | 2        | −0.682       | 0.843  |

**Lessons:**

- A sign flip needs both arrows (a true collider).
- Higher R² comes from C containing a copy of Y: leakage, not skill.
- p-values and adjR² both prefer the wrong model.

## Exp 03 — PC vs. my graph on real factor data

**Data:** Ken French Data Library, CRSP 202608 release, downloaded 2026-10-05.
8 daily factors (Mkt-RF, SMB, HML, RMW, CMA, Mom, ST_Rev, LT_Rev),
1963-07-01 to 2026-08-31, 15,897 days, decimals, no missing values.

**Key idea:** a factor here is the daily return of a long-short portfolio,
not a stock trait. An arrow is a claim about same-day returns.

**My graph** (drafted by me, refined with AI help):

- Direct: Mkt-RF → Mom, Mkt-RF → SMB, Mkt-RF → RMW, RMW → HML (low confidence)
- Hidden causes:
  - U1 Rate surprises → Mkt-RF, HML
  - U2 Macro news → Mkt-RF, Mom
  - U3 Style rotation (money moving between cheap and expensive stocks) → HML, CMA, LT_Rev, Mom
  - U4 Liquidity → SMB, ST_Rev
- Known gap: SMB–CMA share construction but have no path.

**Toy check:** PC leaves a chain undirected (a—b—c) but orients a collider
(a → b ← c). Every arrow PC draws comes from a collider pattern.

**Predictions (written before running PC):**

1. Mom and ST_Rev are uncorrelated.
2. ST_Rev and Mkt-RF are uncorrelated (SMB is a collider between them).
3. Controlling for SMB creates an ST_Rev–Mkt-RF correlation.
4. SMB and CMA are uncorrelated.
5. PC (assumes no hidden causes) will draw direct edges among HML, CMA, LT_Rev, Mom.
6. With 15,897 days, PC will find many edges, since tiny links pass a 5% test.

### Step 1 result — PC on full sample (alpha = 0.05, Fisher-z)

**17 of 28 possible edges:**

- Into LT_Rev: SMB, HML, RMW, CMA
- Into Mkt-RF: ST_Rev, SMB, RMW
- Into CMA: ST_Rev, Mkt-RF, RMW
- Into SMB: HML, RMW
- Into ST_Rev: Mom, RMW
- Into Mom: Mkt-RF, HML
- Undirected: HML — CMA

**Cycle:** Mkt-RF → Mom → ST_Rev → Mkt-RF. Not a valid DAG.
**Why:** each arrow comes from a separate test. When PC's assumptions fail
(hidden causes, fat tails, non-faithfulness), the tests contradict each other.
A cycle is a sign the assumptions don't hold.

**Prediction scorecard:**

| # | Prediction                               | Result                                            |
| - | ---------------------------------------- | ------------------------------------------------- |
| 1 | Mom ⟂ ST_Rev                            | Wrong (Mom → ST_Rev)                             |
| 2 | ST_Rev ⟂ Mkt-RF                         | Wrong (ST_Rev → Mkt-RF)                          |
| 3 | SMB opens ST_Rev–Mkt-RF                 | Can't test (already adjacent)                     |
| 4 | SMB ⟂ CMA                               | Consistent (no edge); still check raw correlation |
| 5 | Direct edges among HML, CMA, LT_Rev, Mom | Right                                             |
| 6 | Many edges                               | Right (17/28)                                     |

**Lessons:**

- My graph's biggest gap is ST_Rev: I linked it to almost nothing; PC links it to Mom, Mkt-RF, RMW, CMA.
- A missing PC edge means "independent given some other factors," not "uncorrelated."
- Don't tidy the output by tuning alpha: the inconsistency is the finding.

**Roadmap:**

- [X] 1. Run PC on the 8 factors; compare with my graph.
- [ ] 2. Stability: run PC per decade; does the cycle persist? Keep edges in ≥ 1/3 of runs (Primer's method).
- [ ] 3. Replicate Exhibit 5 under both graphs (correct controls vs. collider).
- [ ] 4. Add a rate-change proxy (10y Treasury, FRED); see which PC edges vanish.
- [ ] 5. Write up: where PC's assumptions failed; how conclusions depend on the graph.
