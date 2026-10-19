# Experiment 04 — Causal Discovery for Portfolios: LLM vs. Supervised ML vs. Both

*Nothing in Phase 5 runs until this file and `predictions.md` are committed. Later changes go in the Change log.*

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
| 3 | Whether the answer survives realistic data (short, fat-tailed, with a regime break) and hidden causes               | Stress tests S1–S3                                               | Phase 6, figure 3     |
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

| Arm                           | What it does                                                                                                                                    | Uses names | Uses data |
| ----------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- | ---------- | --------- |
| **A1 LLM**              | Sokolov's three steps: list possible edges (10 fresh runs, combined), give them directions, then check and correct (up to 5 rounds)             | Yes        | No        |
| **A2 LLM + refutation** | A1, then test every edge on data and remove edges that fail                                                                                     | Yes        | Yes       |
| **A3 Hybrid**           | Starts from the LLM's graph; where B1 is confident a variable plays a different role, B1's answer replaces it. Rule fixed in `predictions.md` | Yes        | Yes       |
| **B1 Supervised ML**    | 3rd-place ADIA solution (74.06%), the only top solution with full public code                                                                   | No         | Yes       |
| **C1 PC**               | Classic algorithm, as a reference (links to Exp 03)                                                                                             | No         | Yes       |
| **Naive regressions**   | No graph: (a) regress return on X alone; (b) regress on X and all 8 other variables ("control for everything")                                  | No         | Yes       |
| **Oracle**              | The true graph: the best any method can do                                                                                                      | —         | —        |

**A2's limits.** A2 can only *remove* edges; it can't add a missing edge or flip a wrong one. It tests each edge's *total* effect, so a shortcut edge across a real chain survives. Its adjustment is linear, so nonlinear confounding can leak through.

**LLMs ($0 beyond an existing subscription).** Both run at medium reasoning effort.

- **Frontier:** Claude Opus 5.5, run by hand in claude.ai (Pro plan). Every call in a fresh chat outside any project, with memory, preferences and web search off.
- **Free:** `gpt-oss:20b`, run locally on the RTX 5080 through Ollama. This answers: *can this be done for free?*

The LLM never sees data, so it answers once per set of variable names, not once per dataset.

## 5. Scoring

**Graph score (ADIA).** Each of the 8 other variables has one role relative to X and Y: confounder, collider, mediator, cause of X, cause of Y, consequence of X, consequence of Y, or independent. Main score: balanced accuracy over the 8 roles (`common/scoring.py`). Extra: accuracy per role.

**Beta score (new: what a portfolio manager cares about).**

- **True effect:** in each synthetic world, nudge X up by a small amount, hold every noise term fixed, and measure the average change in Y. Possible only because we built the world.
- **Estimated effect:** regress Y on X, controlling for the variables a method labels **confounder**. That is the adjustment its graph implies, and it works the same for every arm, since B1 only outputs roles.
- **Beta error** = estimated − true. Also reported: how often the sign is wrong.
- **Hedge leak** = |beta error| / |true effect|: the share of the exposure left unhedged by someone who trusts the estimate. Under a 2-standard-deviation move in X, this is the loss the hedge failed to cover.

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

- **The dial:** at lower agreement, a share of the literature edges (never X → Y) is randomly removed, reversed or swapped for a new edge. The graph never has cycles.
- **Data recipe (ADIA Appendix C):** each variable is a random nonlinear function of its causes (small neural net, Bayesian model or Gaussian process) plus Gaussian noise; 1,000 rows; every column rescaled to [−1, 1].
- **Size:** 100% agreement: one graph × 100 data draws. Each other level: 20 graphs × 5 draws. Total: **500 datasets**, each saved with its true graph and true effect.
- **Generator check:** on random graphs from my generator, B1 should score near its ADIA level. If it doesn't, my generator isn't ADIA's. The random graphs span ADIA's range of edge counts (3–34), and B1's and PC's scores are also reported by edge count.
- **The LLM's graph is the same at every level**; only the true graph it is scored against changes.

**Stress tests (reported separately):**

- **S1 hidden confounder:** leverage generates the data, then is removed from it.
- **S2 no effect:** no X → Y edge.
- **S3 realistic data:** 240 rows (20 years of monthly data), fat-tailed noise, and a regime break halfway through where some edge functions change. 100% and 50% levels, 50 datasets each. B1 never saw data like this in training.

## 8. Predictions (numbers finalized in Phase 4)

1. **Naive regressions:** "X alone" overstates the beta (leverage confounds); "control for everything" understates it, and can flip its sign (collider and mediator).
2. A1 is best at 100% agreement and falls as agreement falls.
3. B1 stays roughly flat, near its ADIA score.
4. The A1 and B1 curves cross somewhere between 50% and 100% agreement.
5. A2 beats A1 at low agreement, but can't fix reversed arrows.
6. A3 matches or beats both A1 and B1 at middle agreement levels.
7. C1 is below B1 at every level.
8. **S1:** every method keeps a false link between X and Y.
9. **S2:** only A2 can remove the X → Y edge.
10. **S3:** B1 and PC lose more than A1 does.

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
- [ ] Data generator (ADIA Appendix C; check how ADIA rescales columns), saving each dataset's true effect
- [ ] **Gate:** generator check, plus B1 and PC scores by edge count
- [ ] Dial graphs: 20 per level at 75 / 50 / 25 / 0%
- [ ] Generate the 500 datasets, plus S1, S2 and S3

**Phase 4 — Commit predictions**

- [ ] `predictions.md`: a number (or range) for every prediction in Section 8
- [ ] The hybrid's combination rule (A3), exactly
- [ ] The number of LLM runs per model
- [ ] **Gate:** committed before any Phase 5 run

**Phase 5 — Run every arm**

- [ ] A1: both LLMs on the finance variables (every prompt and reply saved)
- [ ] A2: refute each LLM graph on every dataset
- [ ] B1: the Phase 1 model predicts on every dataset (no retraining)
- [ ] A3: combine A1 and B1 with the committed rule
- [ ] C1 PC and the naive regressions on every dataset
- [ ] S1, S2 and S3 for every arm
- [ ] **Gate:** every dataset has a result from every arm; failures logged

**Phase 6 — Score: one output per claim**

- [ ] Claim 1: beta error and hedge leak of the naive regressions (figure 1)
- [ ] Claims 2 and 4: role accuracy and beta error vs. agreement, one line per arm, with confidence intervals (figure 2)
- [ ] Claim 3: the same under S1–S3 (figure 3)
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

## Change log (methodology only)

| Date       | Change                                                                                                                                                                                                                                                                      | Why                                                                                                  |
| ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| 2026-10-13 | Agreement dial replaces World A/B; ADIA data recipe; B1 added                                                                                                                                                                                                               | Fair comparison                                                                                      |
| 2026-10-14 | Dropped the hidden-names LLM arm                                                                                                                                                                                                                                            | Without names or data it only repeats random guessing                                                |
| 2026-10-14 | LLMs: Claude by hand (Pro plan) and local `gpt-oss:20b`; no paid API                                                                                                                                                                                                      | $0 budget; tests whether a free model is enough                                                      |
| 2026-10-15 | B1 reproduced on ADIA's public test set                                                                                                                                                                                                                                     | The private test set was never released                                                              |
| 2026-10-17 | Medium reasoning effort for both LLMs                                                                                                                                                                                                                                       | Isolates model quality from thinking budget                                                          |
| 2026-10-17 | Pipeline deviations from Sokolov: no clustering (10 variables); JSON instruction replaces function calling; system text inside the user message; 3 cut-off prompt fragments reconstructed; validation capped at 5 rounds                                                    | Needed to run without the OpenAI API; identical for both models                                      |
| 2026-10-18 | Refutation: DoWhy finds the adjustment set; OLS gives effect and p-value; subset refuter (50 × 80%) rebuilt with DoWhy's test                                                                                                                                              | DoWhy 0.14's estimator reported the p-value of the wrong coefficient                                 |
| 2026-10-18 | Literature graph settled (Section 6): ownership change moved to the following year; market return replaces industry return; earnings surprise as mediator; no Past5y → Return or index → Return edges; 2 sourced edges among the others (13 total); fairness rule 4 added | Every edge sourced; no two-way edges; definitions can't leak the answer; typical density for B1      |
| 2026-10-18 | Refocused on portfolios: new question; claims-and-evidence table; beta error and hedge leak scores; naive-regression and oracle references; hybrid arm A3; stress test S3 (realistic data); roadmap split into validation and evidence                                      | The write-up must show why a wrong graph costs money, and whether the answer holds on realistic data |
| 2026-10-19 | Refutation picks the smallest valid adjustment set (ties alphabetical) | DoWhy's choice among equally valid sets depended on Python's hash seed, so effect sizes changed between runs |

**Implementation notes.** Hardware: RTX 5080 Laptop (16 GB), Core Ultra 9 285H, 31 GB RAM. Envs: `adia-3rd` (Python 3.11, Oct-2024 package versions so the 3rd-place code runs unpatched; xgboost 3.2.0 for GPU) and `causal-factor` (everything else). Local LLM: `gpt-oss-exp04` (gpt-oss:20b, 16K context, Ollama 0.40.4). Run logs and package issues are in the commit history.
