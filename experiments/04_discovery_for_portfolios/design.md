# Experiment 04 — Causal Discovery for Portfolios: LLM vs. Supervised ML vs. Both

## 1. Question

> **Factor models built without a causal graph can report the wrong exposure: leaving out a common cause, or controlling for a common effect, biases a factor's beta, so a hedge built on that beta leaves behind risk the manager believes is gone. This experiment measures how large that error is, then asks which way of drawing the causal graph gets the beta closest to the truth: an LLM reading the literature, a supervised model reading the data, or the two combined. It tests each under clean and realistic data conditions and records its cost, against synthetic worlds where the true graph and true effect are known.**

**Sources tested**

- **López de Prado (2023); López de Prado & Zoonekynd (2025, the Primer):** a wrong graph gives a wrong beta (Primer, Exhibit 5).
- **Sokolov et al. (2025):** an LLM draws the graph from variable names and definitions, then checks each edge against data.
- **ADIA challenge (Olivetti et al., 2026):** supervised ML learns to read graphs from data alone. Top solutions scored ~74–77%; PC scored ~40%.

Neither discovery paper compared itself with the other, or measured the cost of a wrong graph in beta terms. This experiment does both.

## 2. Claims and evidence

Every claim in the write-up must point to one row of this table. A reader can check any claim by opening the listed result.

| # | Claim the write-up will make                                                                                        | Evidence                                                          | Where                 |
| - | ------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- | --------------------- |
| 1 | A factor regression without a causal graph gives a wrong beta, and a hedge built on it leaks risk                   | Beta and hedge error of the naive regressions vs. the true effect | Phase 6, figure 1     |
| 2 | Which discovery method gets closest to the true graph and beta, and how that depends on how right the literature is | Role accuracy and beta error per method across the agreement dial | Phase 6, figure 2     |
| 3 | Whether the answer survives realistic data (fat-tailed, with a regime break) and hidden causes                      | Stress tests S1–S3                                               | Phase 6, figure 3     |
| 4 | Whether combining LLM and data beats either alone                                                                   | Hybrid arm A3 vs. A1 and B1                                       | Phase 6, figures 2–3 |
| 5 | What each method costs                                                                                              | Time, hardware, dollars, manual work                              | Phase 6, cost table   |

## 3. The idea: an agreement dial

The LLM knows the literature, and the supervised model knows data. Whoever builds the test decides how far the true graph matches the literature, and that choice decides the winner. No setting is neutral.

So I don't pick one setting. I build true graphs that agree with the literature **100%, 75%, 50%, 25% and 0%**, and plot each method's score across them.

**Fairness rules**

1. Every method is scored against the same true graphs, with the same metrics.
2. Data is generated with ADIA's published recipe, not one I invent. Departures from it are stress tests, reported separately.
3. Every edge in the literature graph has a citation. No source, no edge.
4. Variable definitions say *what* is measured and *when*, never what causes what.
5. The main test follows ADIA's rules: X → Y is always present and there are no hidden causes.
6. Each method must reproduce its own paper's results before it is used here.
7. Predictions, including the hybrid's combination rule, are committed before any run.
8. LLM prompts come from Sokolov et al., not written for this experiment, and are frozen after Phase 2.

## 4. Methods

| Arm                           | What it does                                                                                                                                                                                                                          | Uses names | Uses data |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- | --------- |
| **A1 LLM**              | Sokolov's three steps: list possible edges (10 fresh runs, combined), give them directions, then check and correct (up to 5 rounds)                                                                                                   | Yes        | No        |
| **A2 LLM + refutation** | A1, then test every edge on data and remove edges that fail                                                                                                                                                                           | Yes        | Yes       |
| **A3 Hybrid**           | Starts from the LLM's roles; where B1 is at least τ confident a variable plays a different role, B1's answer replaces it (τ and rule fixed in `predictions.md`)                                                                   | Yes        | Yes       |
| **B1 Supervised ML**    | 3rd-place ADIA solution (74.06%), the only top solution with full public code                                                                                                                                                         | No         | Yes       |
| **C1 PC**               | Classic algorithm, as a reference (links to Exp 03)                                                                                                                                                                                   | No         | Yes       |
| **Naive regressions**   | No graph: (a) return on X alone; (b) on X and everything measured by the end of the return year (the factor-model habit; includes the collider and mediator); (c) on X and start-of-year variables only (timing discipline, no graph) | No         | Yes       |
| **Oracle**              | The true graph: the best any method can do                                                                                                                                                                                            | —         | —        |

**How τ is set (`b5_runs/choose_tau.py`).** τ is calibrated on B1's predictions for the 1,880 generator-check datasets, never on the test: it is the lowest confidence at which B1 is right ≥ 85% of the time, with precision computed per role and averaged equally over the 8 roles, the same weighting as balanced accuracy. Pooling all variables instead would let the easy roles (independent, cause of Y) hide B1's weak ones (mediator, collider). The hybrid is also reported at τ = 0.5 and 0.9, so the conclusion can't hinge on one threshold.

**A2's limits.** A2 can only *remove* edges; it can't add a missing edge or flip a wrong one. It tests each edge's *total* effect, so a shortcut edge across a real chain survives. Its adjustment is linear, so nonlinear confounding can leak through.

**LLMs ($0 beyond an existing subscription).** Both run at medium reasoning effort.

- **Frontier:** Claude Opus 5.5, run by hand in claude.ai (Pro plan). Every call in a fresh chat outside any project, with memory, preferences and web search off.
- **Free:** `gpt-oss:20b`, run locally on the RTX 5080 through Ollama. This answers: *can this be done for free?*

The LLM never sees data, so it answers once per set of variable names, not once per dataset.

## 5. Scoring

**Graph score (ADIA).** Each of the 8 other variables has one role relative to X and Y: confounder, collider, mediator, cause of X, cause of Y, consequence of X, consequence of Y, or independent. Main score: balanced accuracy over the 8 roles (`common/scoring.py`). Extra: accuracy per role.

**Beta score (new: what a portfolio manager cares about).**

- **True beta:** in each synthetic world, run a 20,000-row randomized experiment: assign X at random, recompute everything downstream, and take the slope of Y on X. It's the beta a perfect regression should recover, and it's possible only because we built the world.
- **Estimated effect:** regress Y on X, controlling for the variables a method labels **confounder**. That is the adjustment its graph implies, and it works the same for every arm, since B1 only outputs roles.
- **Beta error** = estimated − true. Also reported: how often the sign is wrong.
- **Hedge leak** = |beta error| / |true effect|: the share of the exposure left unhedged by someone who trusts the estimate. Under a 2-standard-deviation move in X, this is the loss the hedge failed to cover.
- **False effect:** when the true beta is exactly 0 (S2b), a regression reports a beta that is significant (p < 0.05) and meaningful (|β̂| ≥ 0.10): the factor mirage. Wrong sign and hedge leak only count worlds with |true beta| ≥ 0.10.

**Costs recorded:** run time, hardware, dollars, setup hours, manual work, and inputs needed.

## 6. The literature graph (100% agreement)

**Timing.** Everything is placed around one **return year**. "Start" = the first day of it, "end" = the last day. Causes can only point forward in time.

| Variable                        | Measured                      | Role             | Edges                        | Source                                                       |
| ------------------------------- | ----------------------------- | ---------------- | ---------------------------- | ------------------------------------------------------------ |
| **BookToMarket (X)**      | Start                         | —               | X → Return                  | Fama & French (1992, 1993)                                   |
| **Return (Y)**            | Over the year                 | —               | —                           | —                                                           |
| Leverage                        | Start                         | Confounder       | Leverage → X, Leverage → Y | Primer p. 4; Bhandari (1988); Fama & French (1992), abstract |
| Quality (QMJ score)             | End                           | Collider         | X → Quality, Y → Quality   | Primer p. 4; Fama & French (1995) p. 132                     |
| EarningsSurprise                | During the year               | Mediator         | X → Surprise → Y           | La Porta et al. (1997), abstract                             |
| Past5yReturn                    | 5 years before the start      | Cause of X       | Past5y → X                  | Fama & French (1996), abstract                               |
| MarketReturn (all other stocks) | Over the year                 | Cause of Y       | Market → Y                  | Sharpe (1964), CAPM                                          |
| ValueIndexMember                | At the index's annual rebuild | Consequence of X | X → Member                  | FTSE Russell methodology                                     |
| InstOwnershipChange             | The year *after*            | Consequence of Y | Y → OwnChange               | Nofsinger & Sias (1999), abstract                            |
| FiscalYearEndMonth              | Fixed                         | Independent      | none                         | —                                                           |

**Two more edges among the other variables:**

| Edge                     | Source                                                                   |
| ------------------------ | ------------------------------------------------------------------------ |
| Leverage → Quality      | Asness, Frazzini & Pedersen (2019) §3.2: safety includes low leverage   |
| Past5yReturn → Leverage | Welch (2004), abstract: stock returns explain ~40% of debt-ratio changes |

**13 edges in total.** ADIA's 10-variable training graphs have 13.5 on average (range 3–34, 9,216 graphs), so B1 sees a typical graph. Files: `finance_variables.json` (what the LLM sees) and `finance_truth.json` (edges, sources, roles).

**Why this graph suits claim 1.** It holds both traps from the Primer: a confounder (leverage), which biases the "X alone" regression, and a collider (quality) plus a mediator (earnings surprise), which bias the "control for everything" regression.

**Caveats from these choices** (each one goes in the write-up):

1. **X → Quality is the weakest edge.** The Primer uses it as a hypothetical ("*if* quality is influenced by…"), and Fama & French (1995) say a high book-to-market "signals" low profitability. The real mechanism is likely a common cause (persistent earning power), simplified here to a direct arrow.
2. **Y → Quality** also rests on the Primer's example. Partial mechanism: QMJ's safety score uses market beta, which is estimated from returns.
3. **InstOwnershipChange is measured the following year.** Measured over the same year, causality runs both ways (Nofsinger & Sias), which a DAG can't hold.
4. **No Past5y → Return edge.** Fama & French (1996) find long-term reversal is absorbed by the three-factor model.
5. **No ValueIndexMember → Return edge**, although index-inclusion effects exist. A simplification.
6. **MarketReturn excludes the firm.** Otherwise Sokolov's prompt rule ("a component causes the aggregate") would point the arrow backwards.
7. **EarningsSurprise is a partial mediator** ("a significant portion" of the value premium, La Porta et al.), so the direct X → Y edge stays.
8. **Past5yReturn → Leverage → Return** skips book-to-market. Not a direct edge, but it stretches Fama & French's "no separate effect".
9. **Several sources are cited from the abstract only**, and the Primer's examples are illustrative, not evidence. The dial exists because the literature graph may be wrong.

## 7. Worlds and data

- **The dial (`dial.py` → `graphs/finance_dial.json`):** agreement = share of the 12 literature edges (all but X → Y) left untouched: 75% breaks 3, 50% breaks 6, 25% breaks 9, 0% breaks all 12. Each broken edge is removed, reversed or swapped for a new edge with equal odds. Reversals and new edges must respect timing (an edge points forward in time or within one period) and keep the graph acyclic; a reversal that isn't allowed becomes a swap, so reversals are rare (~0.5 per graph). 20 graphs per level, because which edges break matters as much as how many: at 75%, 1 to 4 of the 8 roles change.
- **Data recipe (ADIA Appendix C, pp. 20–24):** each variable is a random nonlinear function of its causes: a one-hidden-layer ReLU net (hidden size 5 × parents), a heteroskedastic "Bayesian" net, or a Gaussian process (Matérn 3/2). Gaussian noise with sd drawn from [0, 1]; 1,000 rows; every column rank-uniformized to [−0.999, 0.999], which closes the "sort by variance" loophole (Reisach et al. 2021).
- **Choices Appendix C leaves open** (checked by the generator check): root variables are standard normal before uniformizing; the three function types are equally likely; every node is uniformized before its children use it; columns saved in alphabetical order, so their order reveals nothing about causal order.
- **Calibration (`calibrate_generator.py`):** three knobs Appendix C leaves open (Gaussian-process smoothness, noise level, how kinked the nets are) were set by matching properties of the *data*, not any method's score, on 300 ADIA *training* graphs. ADIA's data: signal R² 0.87, linear share 0.965, |corr| on edges 0.49. Chosen: smoothness 4–16, noise × 0.6, net biases × 1 → 0.87, 0.95, 0.62. R² and linear share were weighted first, because they decide whether arrow directions are detectable; |corr| on edges stays too high (0.62 vs 0.49), a known gap.
- **Size (`make_datasets.py`):** 100% agreement: one graph × 100 data draws. Each other level: 20 graphs × 5 draws. Total: **500 datasets**, each saved with its true graph, true beta and seed (`results/exp04/b3_worlds/datasets_meta.csv`). Columns are renamed the ADIA way (X, Y, 0–7), so no method sees the finance names except through the LLM's variable file.
- **The true beta's sign is random across worlds** (mean ≈ 0): the generator draws random functions, so it doesn't build in a positive value premium. That doesn't matter here, because every score compares a method's estimate with each world's own truth.
- **Generator check (the gate):** take all 1,880 of ADIA's test graphs (B1 never trained on them) and regenerate their data with my generator. B1 and PC then score both versions: same graphs, only the data recipe differs. **Pass:** B1 on my data within 3 points of B1 on ADIA's data. Scores are also reported by edge count. These datasets are only for the check, never part of the test. B1 on ADIA's own version must also reproduce 71.64%, which validates the predict-only tool.
- **The LLM's graph is the same at every level**; only the true graph it is scored against changes.

**Stress tests (reported separately):**

- **S1 hidden confounder:** exactly the 100 literature datasets with the leverage column deleted, so S1 vs. the main test is a paired comparison: same worlds, only visibility differs.
- **S2 no direct effect:** the literature graph without X → Y (100 datasets). X still reaches Y through earnings surprise, so the true beta isn't zero; S2 tests whether methods wrongly insist on a direct edge.
- **S2b true zero:** the literature graph without X → Y and X → earnings surprise, so X has exactly zero effect on returns while leverage still drives both (100 datasets): Exp 01's mirage inside the full finance world.
- **S3 realistic data:** the main test's 500 datasets again (same worlds, same seeds, so every S3 dataset has a twin), made the way firm data really looks: (1) **fat tails:** Student-t noise with 3 degrees of freedom, same variance as before; (2) **raw values:** variables pass standardized raw values to each other, not ranks, so an extreme value really propagates; (3) **a regime break:** halfway through the rows, about a third of the variables switch to a new random function (like the value premium fading after 2007). True beta is reported per regime and pooled. Every method gets the same raw data and does its own standard preparation: B1 rank-transforms it first (`predict_third_place.py --rank`), as any user of B1 would; PC and the regressions use it raw. Rows stay 1,000: a row is a firm-year, and a real cross-section has thousands of firms, so a short sample isn't the realistic problem here. The full dial is kept, because a portfolio manager never knows how well the literature fits their portfolio.

## 8. Predictions

All predictions, with numbers, are in [`predictions.md`](predictions.md), committed before any Phase 5 run: claim 1 (P1–P3b: wrong betas, hedge leak, false effects), claim 2 (P4–P9: the dial), claim 4 (P10: the hybrid), claim 3 (P11–P13: stress tests), claim 5 (P14: cost).

## 9. Roadmap

### Part A — Validation (done): do the tools work?

**Phase 0 — Setup** ✅ Design file, folders, environments, GPU check, Ollama, bibliography

**Phase 1 — Reproduce B1** ✅

- PC baseline: **38.30%** (5th-place report: 37.79%)
- 3rd-place solution: **71.64%** vs. 74.06% published (−2.4 pts, within the 3-point gate)

**Phase 2 — Reproduce A1 and A2** ✅

- Prompts from Sokolov's Appendix B–C; LLM pipeline and refutation built
- Gate on Sprinkler and Asia. Asia: Claude 8/8 edges (memorized); gpt-oss 7/8 with 8 extra; refutation removed only 1 extra

### Part B — Evidence: the claims in Section 2

**Phase 3 — Build the worlds**

- [X] Literature graph: 10 variables, 13 sourced edges, `finance_variables.json` and `finance_truth.json`
- [X] Data generator, `b3_worlds/generator.py` (ADIA Appendix C; rank-uniformized; true beta from a randomized experiment)
- [X] **Gate:** generator check on all 1,880 test graphs: **B1 71.97% on ADIA's data vs. 71.99% on ours (+0.02 pts), PASS**; PC 38.30% vs. 39.49% (+1.2). B1 on ADIA's data is 71.97%, not 71.64%: features recomputed from scratch differ slightly (the 3rd-place feature code has unseeded randomness), so B1 varies by ~0.3 pts run to run
- [X] Dial graphs: 20 per level at 75 / 50 / 25 / 0% (`dial.py`)
- [X] Generate the 500 datasets, plus S1 and S2 (`make_datasets.py`)
- [X] S3 realistic data: 500 datasets paired with the main test (`generator.py realistic=True`; main, S1 and S2 unchanged byte for byte)

**Phase 4 — Commit predictions**

- [X] `predictions.md`: a number (or range) for every prediction
- [X] The hybrid's combination rule (A3), exactly; τ from `choose_tau.py`
- [X] The number of LLM runs per model (gpt-oss 10, Claude 3)
- [ ] **Gate:** committed before any Phase 5 run

**Phase 5 — Run every arm**

- [ ] A1: both LLMs on the finance variables (every prompt and reply saved)
- [ ] A2: refute each LLM graph on every dataset
- [ ] B1: the Phase 1 model predicts on every dataset (no retraining)
- [ ] A3: combine A1 and B1 with the committed rule
- [ ] C1 PC and the naive regressions on every dataset
- [ ] S1, S2, S2b and S3 for every arm
- [ ] **Gate:** every dataset has a result from every arm; failures logged

**Phase 6 — Score: one output per claim**

- [ ] Claim 1: beta error, hedge leak and false effects of the naive regressions (figure 1)
- [ ] Claims 2 and 4: role accuracy and beta error vs. agreement, one line per arm, with confidence intervals (figure 2)
- [ ] Claim 3: the same under S1–S3 and S2b (figure 3)
- [ ] Claim 5: cost table
- [ ] Results vs. `predictions.md`, including the predictions that failed

## 10. Folders

`a` = Part A (validation), `b` = Part B (evidence); the number is the roadmap phase. Phase 4 has no folder: its output is `predictions.md`.

```
experiments/04_discovery_for_portfolios/
  README.md            reader entry point: claim → figure → result file → script
  design.md  predictions.md
  common/              paths.py, scoring.py
  graphs/              sprinkler, asia, finance (_variables.json = what the LLM sees, _truth.json = answer key)
  responses/           every LLM prompt and reply, by graph / model / run
  a1_reproduce_adia/   data check, PC baseline, 3rd-place reproduction, requirements_third_place.txt
  a2_reproduce_llm/    Modelfile, prompts/, run_llm.py, refute.py
  b3_worlds/           data generator, dial, stress-test data
  b5_runs/             run every arm
  b6_analysis/         scoring, figures, cost table
  figures/
data/adia/  data/synthetic/<phase folder>/   (gitignored)
external/                                    (gitignored; 3rd-place code, pinned commit)
models/                                      (gitignored)
results/exp04/<phase folder>/                same names as the code folders
```

## 11. Limitations

- One literature graph, with some weak edges (Section 6 caveats).
- ADIA's rules (always X → Y, no hidden causes) don't hold in real markets; S1 and S3 test only part of that.
- The literature graph borrows the Primer's own examples, so a frontier LLM may reproduce it from memory (as Claude did with Asia).
- The beta score uses a linear regression in a nonlinear world. The oracle shows how much error that alone causes.
- Hedge leak is measured in a synthetic world, in relative terms, not in dollars of a real portfolio.
- LLM results depend on the model, the date and the prompts; claude.ai doesn't let me set temperature.
- B1 is 3rd place, not the winner, and was reproduced on the public test set (the private one was never released).
- S3 ranks B1's input over all rows at once; practitioners rank within each period, so S3's two regimes are mixed in B1's ranks.

## Change log (methodology only)

| Date       | Change                                                                                                                                                                                                                                                                                  | Why                                                                                                                                                                                                                                     |
| ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2026-10-13 | Agreement dial replaces World A/B; ADIA data recipe; B1 added                                                                                                                                                                                                                           | Fair comparison                                                                                                                                                                                                                         |
| 2026-10-14 | Dropped the hidden-names LLM arm                                                                                                                                                                                                                                                        | Without names or data it only repeats random guessing                                                                                                                                                                                   |
| 2026-10-14 | LLMs: Claude by hand (Pro plan) and local `gpt-oss:20b`; no paid API                                                                                                                                                                                                                  | $0 budget; tests whether a free model is enough                                                                                                                                                                                         |
| 2026-10-15 | B1 reproduced on ADIA's public test set                                                                                                                                                                                                                                                 | The private test set was never released                                                                                                                                                                                                 |
| 2026-10-17 | Medium reasoning effort for both LLMs                                                                                                                                                                                                                                                   | Isolates model quality from thinking budget                                                                                                                                                                                             |
| 2026-10-17 | Pipeline deviations from Sokolov: no clustering (10 variables); JSON instruction replaces function calling; system text inside the user message; 3 cut-off prompt fragments reconstructed; validation capped at 5 rounds                                                                | Needed to run without the OpenAI API; identical for both models                                                                                                                                                                         |
| 2026-10-18 | Refutation: DoWhy finds the adjustment set; OLS gives effect and p-value; subset refuter (50 × 80%) rebuilt with DoWhy's test                                                                                                                                                          | DoWhy 0.14's estimator reported the p-value of the wrong coefficient                                                                                                                                                                    |
| 2026-10-18 | Literature graph settled (Section 6): ownership change moved to the following year; market return replaces industry return; earnings surprise as mediator; no Past5y → Return or index → Return edges; 2 sourced edges among the others (13 total); fairness rule 4 added             | Every edge sourced; no two-way edges; definitions can't leak the answer; typical density for B1                                                                                                                                         |
| 2026-10-18 | Refocused on portfolios: new question; claims-and-evidence table; beta error and hedge leak scores; naive-regression and oracle references; hybrid arm A3; stress test S3 (realistic data); roadmap split into validation and evidence                                                  | The write-up must show why a wrong graph costs money, and whether the answer holds on realistic data                                                                                                                                    |
| 2026-10-19 | Refutation picks the smallest valid adjustment set (ties alphabetical)                                                                                                                                                                                                                  | DoWhy's choice among equally valid sets depended on Python's hash seed, so effect sizes changed between runs                                                                                                                            |
| 2026-10-19 | True beta defined by a randomized experiment in each world (not a small nudge of X); generator choices Appendix C leaves open written down (Section 7)                                                                                                                                  | A regression slope is a weighted average over X, so it must be compared with a slope; on 8 test worlds the nudge definition disagreed with a correctly adjusted regression by up to 0.3                                                 |
| 2026-10-19 | Generator calibrated to ADIA's data (smoothness 4–16, noise × 0.6); gate rerun on all 1,880 test graphs                                                                                                                                                                               | First gate (300 graphs, uncalibrated): B1 69.1% on ADIA's data vs. 73.5% on ours, FAIL. Our links were too curved, giving B1 direction clues ADIA's near-linear data doesn't have; 300 graphs also left ±2–3 points of sampling noise |
| 2026-10-20 | S3 redesigned: full dial (500, paired with main) instead of two levels; 1,000 rows instead of 240; Student-t (3 df) noise, raw values between variables, regime break at mid-sample (a third of functions redrawn); raw data to every method                                            | A row is a firm-year, so 240 rows mixed in a time-series problem; ranks between variables would erase the fat tails; the dial stands for not knowing how well the literature fits a real portfolio                                      |
| 2026-10-20 | Hybrid threshold τ: B1 must be right ≥ 85% of the time averaged equally over the 8 roles, on generator-check data (refined from a pooled average, which gave 0.55)                                                                                                                    | Pooled precision was 86.6% but only 65% for mediators; equal role weights match balanced accuracy. Decided before any test run                                                                                                          |
| 2026-10-20 | Predictions written (`predictions.md`). Added S2b (true beta exactly 0) and the false-effect rate; naive regressions split into "everything by year-end" and "start of year only"; wrong sign and hedge leak counted only where \|true beta\| ≥ 0.10; LLM runs: gpt-oss 10, Claude 3 | The factor mirage (a significant beta that doesn't exist) needed its own measure and worlds where the truth is zero; "everything" no longer includes a future variable, so it's the realistic trap, not a straw man                     |

**Implementation notes.** Hardware: RTX 5080 Laptop (16 GB), Core Ultra 9 285H, 31 GB RAM. Envs: `adia-3rd` (Python 3.11, Oct-2024 package versions so the 3rd-place code runs unpatched; xgboost 3.2.0 for GPU) and `causal-factor` (everything else). Local LLM: `gpt-oss-exp04` (gpt-oss:20b, 16K context, Ollama 0.40.4). Run logs and package issues are in the commit history.
