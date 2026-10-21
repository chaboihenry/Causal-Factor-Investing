# Causal Factor Investing

Factor models report betas: how much a stock or portfolio moves with value, momentum or the market. Hedges and risk reports are built on those betas.

López de Prado argues that a factor regression built without a causal graph can report the wrong beta, even when it fits the data well.[^1] This project tests that claim in four experiments, reproducing each source before testing it. It then asks who should draw the graph:

- a language model reading the literature,[^2]
- a supervised model trained to read graphs from data,[^3]
- or both together.

## The experiments

| #  | Question                                                                                            | Finding                                                                                                                                                                                                                                                                                              | Read more                                                                                                                                                                |
| -- | --------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 01 | Can a factor with no effect look significant?                                                       | Yes. A factor with zero true effect gets β = 0.48 with p ≈ 10⁻²⁴²; controlling for the hidden common cause drops it to 0.01. A chain (factor → middle variable → return) gives the same regression output, so data alone can't tell a confounder from a mediator                             | [notes](exposition/notes.md#exp-01--confounder-mirage) · [script](experiments/01_confounder_mirage/confounder_mirage.py)                                                  |
| 02 | What does controlling for a common effect (a collider) do?                                          | It can flip the sign of a true β = 0.5 to −0.68, while adjusted R² *rises* from 0.21 to 0.84. The wrong model looks better                                                                                                                                                                      | [notes](exposition/notes.md#exp-02--collider-mirage) · [script](experiments/02_collider_mirage/collider_mirage.py)                                                        |
| 03 | Do the Primer's methods hold on real factor data?                                                   | PC on 8 Fama-French factors (1963–2026) drew graphs with cycles, and the arrows flipped direction between decades. Factor correlations reversed sign around 2000 and 2008, and the proxies for hidden causes explained little. Real data can show that methods disagree, but not which one is right | [summary](exposition/experiment03_summary.md) · [notes](exposition/notes.md#exp-03--pc-vs-my-graph-on-real-factor-data)                                                   |
| 04 | How wrong does a beta get without the right graph, and which way of drawing the graph gets closest? | Below                                                                                                                                                                                                                                                                                                | [summary](exposition/experiment04_summary.md) · [README](experiments/04_discovery_for_portfolios/README.md) · [design](experiments/04_discovery_for_portfolios/design.md) |

## Experiment 04: the answer

The test uses synthetic worlds built with ADIA Lab's published data recipe, so the true graph and the true beta are known. The factor is book-to-market, and eight other variables come from the finance literature. The true graphs agree with the literature 100%, 75%, 50%, 25% or 0%.

**How large is the error?** In worlds that follow the literature:

- **Controlling for everything measured by year-end** (the usual factor-model habit) left a median **84%** of the true beta unhedged, and got its sign wrong 25% of the time.
- **Regressing on the factor alone** left **46%** unhedged.
- **Controlling for the true common cause** (leverage) left **17%**.
- **When the factor's true effect was exactly zero,** regressing on it alone still reported a significant, meaningful beta in **81%** of worlds.

**Which way of drawing the graph gets the beta closest?** None got close.

- An LLM (Claude Opus 5.5 or the free gpt-oss:20b), the ADIA 3rd-place supervised model and a hybrid of the two all left **32–40%** of the beta unhedged.
- That is about twice the oracle's error, and no better than simply controlling for what is measured before the factor (**30%**).
- All of them missed the same fact: leverage drives both book-to-market and returns.

**Which reads the graph best?** The supervised model, at every level of agreement with the literature.

- It got 68–75% of variable roles right. Claude got 50% even on the literature's own graph; gpt-oss got 33%.
- The LLMs were only as good as the literature, and fell below random guessing when it was wrong.
- Combining the LLM and the model never beat the model alone.
- But the hedges leaked the same either way, because **getting most roles right is not the same as getting the confounder right**.

**Clean vs. realistic data.** With fat tails and a regime break, the supervised model lost 19 points of role accuracy and ended level with Claude on the literature graph. The oracle's leak rose from 17% to 30%.

**Cost.**

- LLM: minutes per run, at $0.
- Supervised model: 22.6 hours of one-off GPU training, then about 12 minutes per 500 datasets.

**For risk management.** A causal graph changes a hedge far more than any choice between discovery methods does. Whichever method draws the graph, the confounders are the part a person still has to check.

10 of the 18 predictions committed before the runs held. The ones that failed are reported and explained in [design.md, Section 12](experiments/04_discovery_for_portfolios/design.md#12-results).

<p align="center"><img src="experiments/04_discovery_for_portfolios/figures/fig1_beta_error.png" width="100%"></p>

## Repository

```
experiments/   one folder per experiment (code, design, figures)
results/       outputs, one folder per experiment
exposition/    notes, experiment summaries, bibliography
data/ models/ external/ papers/   not committed (downloads, trained models, third-party code, PDFs)
```

**Environments.**

- `causal-factor`: everything.
- `adia-3rd`: only the ADIA 3rd-place code. Setup is in its [requirements file](experiments/04_discovery_for_portfolios/a1_reproduce_adia/requirements_third_place.txt).

**Sources.** Full list in [`exposition/bibliography.md`](exposition/bibliography.md).

[^1]: Marcos López de Prado, *Causal Factor Investing* (Cambridge University Press, 2023); Marcos López de Prado and Vincent Zoonekynd, *Causality and Factor Investing: A Primer* (CFA Institute Research Foundation, 2025).
    
[^2]: Alik Sokolov et al., "Toward Automating Causal Discovery in Financial Markets and Beyond," *Transactions of ADIA Lab* (2025).
    
[^3]: Emanuele Olivetti et al., "Can AI Learn Causal Structure? Evidence from ADIA Lab's Causal Discovery Challenge," working paper (2026); Mutian Hong and Guoqin Gu, ADIA Lab Causal Discovery Challenge rank-3 solution (GitHub).
