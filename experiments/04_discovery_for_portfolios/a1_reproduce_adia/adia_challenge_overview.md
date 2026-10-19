# ADIA Lab Causal Discovery Challenge — quick reference

Sources: Olivetti et al. (2026), SSRN 6125566 (§2–3, Appendix B.3, Appendix C); Hong & Gu's code
(github.com/Ori-Replication/ADIA-Lab-Causal-Discovery, commit 7fd401d); our Phase 1 runs.

## The challenge in one paragraph

ADIA Lab and CrunchDAO ran it Aug–Oct 2024. 1,904 people entered. Each dataset is a table of about
1,000 rows and 3–10 columns, generated from a known causal graph. Two columns are always **X**
(treatment) and **Y** (outcome), and **X → Y always exists**. The task: for every other column *v*,
say which of 8 roles it plays relative to X and Y.

## The 8 roles (only direct arrows to X and Y count)

| Role             | Arrows             |
| ---------------- | ------------------ |
| Confounder       | v → X and v → Y  |
| Collider         | X → v and Y → v  |
| Mediator         | X → v → Y        |
| Cause of X       | v → X only        |
| Cause of Y       | v → Y only        |
| Consequence of X | X → v only        |
| Consequence of Y | Y → v only        |
| Independent      | no arrow to X or Y |

## Scoring

**Balanced accuracy**: the average of the 8 per-role recall rates. Every role counts equally,
so always guessing "Independent" scores only 12.5%.

## The data

|                       | Datasets | Labels?                                   |
| --------------------- | -------- | ----------------------------------------- |
| Train                 | 23,500   | yes, true graphs                          |
| Public test (reduced) | 1,880    | yes (we have them)                        |
| Private test          | ~22,000  | never released; decided the final ranking |

**How it was generated (Appendix C):** random graphs from 6 families (Erdős–Rényi, scale-free,
bipartite, hierarchical, low-rank, preferential attachment) plus subgraphs of a real food web.
Each variable = a random nonlinear function of its parents (small neural net, Gaussian process,
or a heteroskedastic "Bayesian" model) + Gaussian noise. Every column rescaled to [−1, 1].
**No hidden confounders.**

## Results

|                                                     | Balanced accuracy |
| --------------------------------------------------- | ----------------- |
| Random guessing                                     | 12.5%             |
| Official baselines (PC, random forest, neural net)  | ~40%              |
| 1st place (Hallak, deep learning)                   | 76.70%            |
| 2nd place (Franca & Gassmann, AVICI + exact search) | 74.47%            |
| **3rd place (Hong & Gu, features + XGBoost)** | **74.06%**  |
| Our PC baseline reproduction                        | 38.30%            |
| **Our 3rd-place reproduction**                | **71.64%**  |

Hardest roles for everyone: Mediator, Confounder, Collider. Easiest: Independent, Cause of Y.

## How the 3rd-place solution works

**Core idea:** turn causal discovery into ordinary classification.
One row = one variable *v* in one dataset; label = its role; predict the role from statistics of the data.

1. **Augmentation.** Every arrow A → B in a training graph can be relabelled as a new "X → Y" question.
   This multiplies the training data (we used ×3: 23,500 → ~92,000 datasets; the paper says up to
   ×11 and about +5 points).
2. **Features (~360 per variable, 25 sets).** Mostly variations of four questions:
   is *v* related to X? to Y? does the *v*–X link survive once Y is known? does the *v*–Y link
   survive once X is known? Plus "direction" clues (regression residuals, PPS, Cloud), structure fits
   (SEM), and the opinions of classic algorithms (PC, FCI, GRaSP, ExactSearch) as votes.
3. **Model.** 3 XGBoost models (7,000 / 7,500 / 8,000 trees, different seeds), probabilities
   averaged. Class weights so rare roles count as much as common ones.

## Why the triangle roles are hard

Confounder, Mediator and Collider all link *v*, X and Y to each other. Independence tests give the
same answers for all three (Markov equivalence, Exp 01). Only weak asymmetry clues can separate them.

## Caveats to remember

- The model learns **the habits of ADIA's simulator**. Whether that transfers to data from another
  process is exactly what Exp 04 tests.
- It assumes X → Y exists and there are no hidden confounders — both false in real markets.
- Our 71.64% is on the public test set; the published 74.06% is on the private one.
- Their final submission used a precomputed feature table we don't have; we ran the from-scratch path.
- Their feature code has unseeded randomness, so rebuilding features gives slightly different scores.
