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
7. LLM prompts come from Sokolov et al., not written for this experiment.

## 3. Methods

| Arm                           | What it does                                                                                       | Uses names | Uses data |
| ----------------------------- | -------------------------------------------------------------------------------------------------- | ---------- | --------- |
| **A1 LLM**              | Sokolov's three steps: list possible edges (10 runs), give them directions, then check and correct | Yes        | No        |
| **A2 LLM + refutation** | A1, then test every edge on data (DoWhy) and remove edges that aren't significant                  | Yes        | Yes       |
| **B1 Supervised ML**    | 3rd-place ADIA solution (74.06%), the only top solution with full public code                      | No         | Yes       |
| **C1 PC**               | Classic algorithm, as a reference (links to Exp 03)                                                | No         | Yes       |
| **Random guessing**     | The floor: 12.5%                                                                                   | No         | No        |

A2 can only *remove* edges. It can't add a missing edge or flip a wrong direction.

**LLMs (budget: $0 beyond an existing subscription)**

A1 and A2 each run on two models:

- **Frontier:** Claude, run by hand in a claude.ai chat (Pro plan).
  - One fresh chat per run: no project, memory off, personal preferences off, web search off. Nothing from my other chats should leak into the answers.
  - Record the model name, date and settings for every run.
- **Free:** a mid-size open-weight model, run locally on the RTX 5080 through Ollama. This answers: *can this be done for free?*

This is cheap because the LLM never sees the data. It answers once per set of variable names, not once per dataset.

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

- [X] Finalize and commit this file
- [X] Folders and `.gitignore`
- [X] Environment, including a GPU check
- [X] Install Ollama; pick the local model
- [X] Add new sources to the bibliography

**Phase 1 — Reproduce B1**

- [X] Download the ADIA data
- [X] Run the PC baseline (~40%)
- [X] Run the 3rd-place solution (~74%, within 3 points)
- [X] **Gate:** don't continue until it does

**Phase 2 — Reproduce A1 and A2**

- [X] Prompts from Sokolov's Appendix B–C
- [X] Build the pipeline
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

One folder per roadmap phase; shared code in `common/`.

```
experiments/04_llm_vs_supervised/
  design.md  predictions.md  requirements_b1.txt
  common/          paths.py, scoring.py (shared by every phase)
  phase1_adia/     data check, baselines, 3rd-place reproduction
  phase2_llm/      Modelfile, prompts/, responses/, LLM pipeline
  phase3_worlds/   literature graph, dial, data generator
  phase5_runs/     run every arm on the 500 datasets
  phase6_analysis/ scoring tables, plots
  figures/
data/adia/  data/synthetic/      (gitignored)
external/                        (gitignored; 3rd-place code, pinned commit)
models/                          (gitignored)
results/exp04/phase1/ ... phase6/
```

## 9. Open decisions (settle before Phase 3)

1. **Past 5-year return → stock return.** Long-term reversal supports the edge; Fama & French (1996) argue it runs through book-to-market.
2. **Book-to-market → quality.** Justified only by timing (quality is measured at year-end).
3. **Institutional ownership change.** Measure it after the return year, or drop it?
4. **Value index membership.** Cite an index methodology document, or drop it?
5. **Industry return and fiscal year-end month.** They need sources, or replacements.
6. **LLM Choice:** Claude (claude.ai Pro, by hand) and a local open-weight model (Ollama). The exact local model is picked in Phase 0.

## 10. Limitations

- One literature graph.
- ADIA's rules (always X → Y, no hidden causes) don't hold in real markets.
- The LLM may have read the Primer.
- LLM results depend on the model, the date and the prompts.
- Claude is run through the chat window, so settings like temperature can't be controlled.
- B1 is 3rd place, not the winner.
- Static synthetic data: no regime changes.

## Change log

| Date       | Change                                                                                                                                                                                                                     | Why                                                                                                                                                                                     |
| ---------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2026-10-13 | Replaced World A/B with the agreement dial; ADIA data recipe; added B1                                                                                                                                                     | Fair comparison                                                                                                                                                                         |
| 2026-10-14 | Dropped the hidden-names LLM arm; shortened the file                                                                                                                                                                       | Without names or data it only repeats random guessing                                                                                                                                   |
| 2026-10-14 | LLMs: Claude by hand (Pro plan) and a local model through Ollama; no paid API                                                                                                                                              | $0 budget; the local model tests whether a free model is enough                                                                                                                         |
| 2026-10-14 | Hardware recorded: RTX 5080 Laptop (16 GB, CUDA 13.4), Core Ultra 9 285H (16 cores), 31 GB RAM                                                                                                                             | Reproducibility                                                                                                                                                                         |
| 2026-10-14 | Local model fixed before any results: `gpt-oss:20b` via Ollama 0.40.4, saved as `gpt-oss-exp04` (`phase2_llm/Modelfile`: num_ctx 16384, default temperature). Runs 100% on GPU at ~82–96 tokens/s                   | Strongest general model that fits in 16 GB; 16K context so long Sokolov prompts aren't cut off. Switching is allowed only if it can't run                                               |
| 2026-10-14 | B1 environment: separate conda env `adia-b1`; xgboost 3.4.1 with CUDA confirmed                                                                                                                                          | Keeps B1's package versions away from the Exp 03 env                                                                                                                                    |
| 2026-10-15 | ADIA data checked: 23,500 train + 1,880 reduced-test datasets; 1,000 rows, 3–10 variables, values in [−0.999, 0.999]; X → Y in every graph, all acyclic, 0 failures                                                     | Matches the paper. The private test set behind the official 74.06% was never released, so B1 is reproduced on the public reduced test set (5th place: 70.69% public vs. 70.37% private) |
| 2026-10-15 | Folders reorganized by phase; shared paths in `common/paths.py`; Modelfile moved to `phase2_llm/`                                                                                                                      | Easier to navigate; one place for paths                                                                                                                                                 |
| 2026-10-16 | PC baseline reproduced: 38.30% balanced accuracy on 1,880 reduced-test datasets (5th-place report: 37.79%); 0.4 min on 16 cores                                                                                            | Data, scorer and pipeline validated before the 3rd-place run                                                                                                                            |
| 2026-10-16 | 3rd-place features run in a separate env `adia-3rd` with Oct-2024 package versions (requirements_third_place.txt); their code unpatched. Only xgboost is current (GPU support)                                           | 2026 versions of pingouin and pandas crash on their code; matching the competition-era versions is more faithful than patching                                                          |
| 2026-10-16 | 3rd-place features run in env `adia-3rd`: Python 3.11, Oct-2024 package versions (requirements_third_place.txt), their code unpatched; xgboost 3.2.0 for GPU                                                             | 2026 pingouin/pandas crash on their code; econml 0.15.1 needs shap 0.43, which has no Windows build for Python 3.12                                                                     |
| 2026-10-16 | 3rd-place pilot (1,000 train, no augmentation): 59.34% balanced accuracy on 1,880 test datasets, 51 min                                                                                                                    | Pipeline works on real data; +21 pts over PC with 4% of the training data. Full run next (est. 15–20 h)                                                                                |
| 2026-10-17 | 3rd-place full run (23,500 train, augment 3): 71.64% balanced accuracy on 1,880 public test datasets vs. 74.06% published (−2.4 pts, within the 3-pt gate); 1,355 min (22.6 h). Largest gaps: Mediator −10, Collider −6 | Phase 1 gate passed. Gap likely from their final run using a larger precomputed augmented feature table, plus public vs. private test set and unseeded feature randomness               |
| 2026-10-17 | Env `adia-b1` retired; PC baseline, scorer and Phase 2+ run in `causal-factor` (added gcastle, torch, dowhy, ollama, pyperclip). PC baseline re-run there: 0.3830, identical                                           | Two envs: `adia-3rd` for 3rd-place code, `causal-factor` for everything else                                                                                                        |
| 2026-10-17 | Reasoning effort fixed at medium for both LLMs (Claude Opus 5.5 in claude.ai; gpt-oss default)                                                                                                                             | Same thinking budget for both; isolates model quality from reasoning budget; Pro usage limits                                                                                           |
| 2026-10-17 | Phase 2 plumbing check passed: Sprinkler + Asia, both LLMs; all calls parsed, final graphs acyclic, C3 loop exercised (gpt-oss Asia: 3 rounds). Asia: Claude 8/8, 0 extra; gpt-oss 7/8, 8 extra | Pipeline works; Claude's perfect Asia reflects memorization of a famous graph |
| 2026-10-17 | LLM pipeline deviations: clustering (B1–B7) skipped (10 variables); JSON instruction replaces function calling (same for both models); system text inside user message; 3 reconstructed fragments in C1/C3; C3 capped at 5 rounds | Necessary to run without the OpenAI API; all identical across models |
| 2026-10-18 | refute.py: DoWhy for the backdoor adjustment set; OLS for effect and p-value; subset refuter (50 × 80%) reimplemented with DoWhy's normal test | DoWhy 0.14's linear estimator silently added interaction terms and reported the p-value of the wrong coefficient (verified against OLS) |
