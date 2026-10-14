# Experiment 04 — LLM vs. Supervised ML for Causal Discovery

*Draft. Nothing runs until this file is final and committed. Later changes go in the Change log.*

## 1. Question

**When should you trust an LLM's knowledge to draw a causal graph, and when should you trust a model that learns the graph from data?**

- **Sokolov et al. (2025):** an LLM draws the graph from variable names and definitions, then checks each edge against data.
- **ADIA challenge (Olivetti et al.):** supervised ML learns to read graphs from data alone. The best solutions scored ~74–77%; classic algorithms like PC scored ~40%.

Neither paper compared the two. This experiment does, on synthetic data where the true graph is known. The end product is a **decision guide**: the costs and benefits of each approach, and when to use which.

## 2. The idea: an agreement dial

The LLM knows the literature, and the supervised model knows data. Whoever builds the test decides how far the true graph matches the literature, and that choice decides the winner. No setting is neutral.

So I don't pick one setting. I build true graphs that agree with the literature **100%, 75%, 50%, 25% and 0%**, and plot each method's score across them. The result is where the curves cross.

**Fairness rules**

1. Every method is scored against the same true graphs, with the same metric.
2. Data is generated with ADIA's published recipe, not one I invent.
3. Every edge in the literature graph has a citation. No source, no edge.
4. The main test follows ADIA's rules: X → Y is always present and there are no hidden causes. Breaking either rule is a separate stress test.
5. Each method must reproduce its own paper's results before it is used here.
6. Predictions are committed before any run.

## 3. Methods

| Arm                           | What it does                                                                                       | Uses names | Uses data |
| ----------------------------- | -------------------------------------------------------------------------------------------------- | ---------- | --------- |
| **A1 LLM**              | Sokolov's three steps: list possible edges (10 runs), give them directions, then check and correct | Yes        | No        |
| **A2 LLM + refutation** | A1, then test every edge on data (DoWhy) and remove edges that aren't significant                  | Yes        | Yes       |
| **B1 Supervised ML**    | 3rd-place ADIA solution (74.06%), the only top solution with full public code                      | No         | Yes       |
| **C1 PC**               | Classic algorithm, as a reference (links to Exp 03)                                                | No         | Yes       |
| **Random guessing**     | The floor: 12.5%                                                                                   | No         | No        |

A2 can only *remove* edges. It can't add a missing edge or flip a wrong direction. The LLM runs on two current models from two different providers.

## 4. Task and scoring (from ADIA)

- 10 variables: **X** = book-to-market, **Y** = annual stock return, plus 8 others.
- Each of the 8 has one role relative to X and Y: confounder, collider, mediator, cause of X, cause of Y, consequence of X, consequence of Y, or independent.
- **Main score:** balanced accuracy over the 8 roles, using ADIA's `scoring.py`.
- **Extra scores:** accuracy per role; edge precision and recall; share of arrows pointing the right way.
- **Costs recorded:** run time, hardware, API dollars, setup hours, and inputs needed.

## 5. Worlds and data

- **Literature graph (100%):** the 10 variables, with a citation for every edge.
- **The dial:** at lower agreement, a share of the literature edges (never X → Y) is randomly removed, reversed or swapped for a new edge. The graph never has cycles.
- **Data recipe (ADIA Appendix C):**
  - Each variable is a random nonlinear function of its causes: a small neural net, a Bayesian model or a Gaussian process.
  - Gaussian noise; 1,000 rows; every column rescaled to [−1, 1].
- **Size:**
  - 100% agreement: one graph × 100 data draws.
  - Each other level: 20 graphs × 5 draws.
  - Total: **500 datasets**.
- **Generator check:** on random graphs from my generator, B1 should score near its ADIA level. If it doesn't, my generator isn't ADIA's.
- **The LLM never sees data.** Its A1 graph is the same at every agreement level; only the true graph it's scored against changes.

**Stress tests (reported separately):**

- **S1 hidden confounder:** leverage is used to generate the data, then removed from it.
- **S2 no effect:** no X → Y edge.

## 6. Predictions (numbers finalized before Phase 5)

1. A1 is best at 100% agreement and falls as agreement falls.
2. B1 stays roughly flat, near its ADIA score.
3. The A1 and B1 curves cross somewhere between 50% and 100% agreement.
4. A2 beats A1 at low agreement, but can't fix reversed arrows.
5. C1 is below B1 at every level.
6. **S1:** every method keeps a false link between X and Y.
7. **S2:** only A2 can remove the X → Y edge.

## 7. Roadmap

**Phase 0 — Setup**

- [ ] Finalize and commit this file
- [ ] Folders and `.gitignore`
- [ ] Environment, including a GPU check
- [ ] API keys in `.env`, with spending caps
- [ ] Add new sources to the bibliography

**Phase 1 — Reproduce B1**

- [ ] Download the ADIA data
- [ ] Run the PC baseline (~40%)
- [ ] Run the 3rd-place solution (~74%, within 3 points)
- [ ] **Gate:** don't continue until it does

**Phase 2 — Reproduce A1 and A2**

- [ ] Prompts from Sokolov's Appendix B–C
- [ ] Build the pipeline
- [ ] **Gate:** test on a small known graph

**Phase 3 — Build the worlds**

- [ ] Literature graph
- [ ] Data generator
- [ ] **Gate:** generator check
- [ ] Generate the 500 datasets

**Phase 4 — Commit predictions**

**Phase 5 — Run all arms and stress tests** (every LLM prompt and reply saved)

**Phase 6 — Score:** score-vs-agreement plot and cost table

**Phase 7 — Write-up:** results vs. predictions, the decision guide, limitations

## 8. Folders

```
experiments/04_llm_vs_supervised/
  design.md
  llm/prompts/  llm/responses/
  figures/
data/adia/  data/synthetic/      (gitignored)
models/                          (gitignored)
results/exp04/
```

## 9. Open decisions (settle before Phase 3)

1. **Past 5-year return → stock return.** Long-term reversal supports the edge; Fama & French (1996) argue it runs through book-to-market.
2. **Book-to-market → quality.** Justified only by timing (quality is measured at year-end).
3. **Institutional ownership change.** Measure it after the return year, or drop it?
4. **Value index membership.** Cite an index methodology document, or drop it?
5. **Industry return and fiscal year-end month.** They need sources, or replacements.
6. **LLM choice:** which two models, and the budget.

## 10. Limitations

- One literature graph.
- ADIA's rules (always X → Y, no hidden causes) don't hold in real markets.
- The LLM may have read the Primer.
- LLM results depend on the model, the date and the prompts.
- B1 is 3rd place, not the winner.
- Static synthetic data: no regime changes.

## Change log

| Date       | Change                                                                 | Why                                                   |
| ---------- | ---------------------------------------------------------------------- | ----------------------------------------------------- |
| 2026-10-13 | Replaced World A/B with the agreement dial; ADIA data recipe; added B1 | Fair comparison                                       |
| 2026-10-14 | Dropped the hidden-names LLM arm; shortened the file                   | Without names or data it only repeats random guessing |
