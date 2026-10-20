# Experiment 04 — Predictions

*Committed before any Phase 5 run.*

## Fixed choices

| Choice                 | Value                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| ---------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| LLM runs               | One run = Sokolov's full pipeline: 10 exploration chats (C1), one direction step (C2), validation rounds (C3). **gpt-oss:20b: 10 runs** (free, automatic). **Claude Opus 5.5: 3 runs** (~12 chats each, by hand). Results reported as mean and range across runs                                                                                                                                                                                                                                                                                                      |
| Hybrid A3 rule         | Start from the LLM's role for each variable; if B1's top role differs and B1's probability for it is ≥ **τ**, use B1's role                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| τ                     | **0.70** (`b5_runs/choose_tau.py`, table in `results/exp04/b5_runs/tau.csv`): the lowest B1 confidence at which B1 is right ≥ 85% of the time **averaged equally over the 8 roles**, on the 1,880 generator-check datasets, which are not part of the test. At 0.70: balanced precision 86.6%, covering 70% of variables; by role from 77% (mediator) to 97% (cause of Y). Refined once before any test run: pooling all variables gave τ = 0.55 but hid the mediator's 65%. A3 is also shown at τ = 0.5 and 0.9 as a sensitivity check, never as the headline |
| Regressions compared   | **Everything**: all variables measured by the end of the return year (drops next-year ownership change). **Start of year**: leverage, past 5-year return, index membership, fiscal month. **X alone.** **Oracle**: the true confounders (leverage). Each method's own graph: its confounders                                                                                                                                                                                                                                                              |
| Wrong sign, hedge leak | Counted only in worlds with \|true beta\| ≥ 0.10 (near zero, sign is a coin flip and leak explodes); excluded worlds reported                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| False effect           | A regression reports a beta that is significant (OLS p < 0.05) **and** meaningful (\|β̂\| ≥ 0.10) when the true beta is 0. Measured on S2b                                                                                                                                                                                                                                                                                                                                                                                                                               |
| Uncertainty            | 95% intervals by bootstrapping over graphs within each agreement level                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| Already seen           | `generator.py`'s self-check printed betas for literature seeds 0–7 (= `lit_000`–`lit_007`, their S1, S2, S2b and S3 twins). Claim 1 is also reported without them                                                                                                                                                                                                                                                                                                                                                                                                         |

## Predictions

### Claim 1: no causal graph → wrong beta, or a beta that isn't there

- **P1.** "Everything": wrong sign in ≥ 20% of literature worlds; median hedge leak ≥ 50%.
- **P1b.** "Start of year": median hedge leak ≤ 35% (timing discipline alone avoids the collider and mediator).
- **P2.** "X alone": wrong sign in ≥ 10%; median hedge leak ≥ 30%.
- **P3.** Oracle: median hedge leak ≤ 25%.
- **P3b.** S2b (true beta = 0, leverage drives both): "X alone" and "Everything" report a false effect in ≥ 50% of worlds; the oracle in ≤ 20%.

### Claim 2: role accuracy across the dial (balanced accuracy)

- **P4.** B1 is flat: 65–78% at every level, no two levels more than 6 points apart.
- **P5.** A1 Claude: ≥ 75% at 100%; ≤ 25% at 0%; falls at every step.
- **P6.** A1 gpt-oss: below A1 Claude at 100% by ≥ 10 points.
- **P7.** A1 Claude and B1 cross between 75% and 100% agreement.
- **P8.** A2 beats A1 by ≥ 3 points at 50% and below; no gain at 100%.
- **P9.** PC: 33–45% at every level, below B1 everywhere.

### Claim 4: the hybrid

- **P10.** A3 is at or above the better of A1 and B1 (within 1 point) at every level, and strictly above both at 75%.

### Claim 3: stress tests

- **P11.** S1 (leverage hidden): every method's median hedge leak at least doubles vs. the same 100 worlds in the main test.
- **P12.** S2 (no direct X → Y): B1 and A1 keep X → Y in ≥ 90% of datasets; only A2 ever removes it.
- **P13.** S3 (realistic): A1's role accuracy is unchanged (it never sees data); B1 falls ≥ 5 points and PC falls vs. the paired main datasets; the oracle's hedge leak rises.

### Claim 5: cost

- **P14.** A1 costs $0 and minutes per run; B1 costs ~23 h of one-off training plus ~10 min per 500 datasets; A3 costs both.
