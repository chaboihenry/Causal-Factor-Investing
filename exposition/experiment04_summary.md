# Experiment 4 Summary

Experiments 01 and 02 showed that a regression's beta depends on which variables it controls for. Controlling for a confounder removes bias, while controlling for a collider or a mediator creates it.[^1] Experiment 03 showed the difficulty on real data. A theory-first graph and a data-driven graph of the Fama-French factors disagreed, and with no true graph to check against, nothing could say which was right. Experiment 04 moves to synthetic worlds, where the true graph and the true effect are known, and asks two questions. First, how wrong does a factor's beta get without the right causal graph, and how much of a hedge built on that beta leaks? Second, which way of drawing the graph gets the beta closest to the truth? Four methods are compared:

- a large language model reading only variable names and definitions, following the pipeline of Sokolov et al.;[^2]
- a supervised model trained to read graphs from data, the third-place solution of ADIA Lab's Causal Discovery Challenge;[^3]
- a hybrid of the two;
- the PC algorithm, as a reference.[^4]

The worlds are built around one factor, book-to-market, and the next year's stock return, plus eight other variables. Together they form a literature graph of 13 edges, each with a source. Following the Primer's own example, it contains a confounder (leverage), a collider (a quality score) and a mediator (earnings surprise).[^5] The literature could be wrong, so I also built true graphs that keep 75%, 50%, 25% and 0% of its edges. The data follow ADIA's published recipe,[^6] with every variable rank-transformed so that simple variance patterns don't give the answer away.[^7] I checked that the supervised model scores the same on my data as on ADIA's: 71.99% against 71.97%, over 1,880 test graphs. The true beta in each world comes from a randomized experiment run inside it.

Two LLMs were tested: Claude Opus 5.5, run three times by hand in fresh chats, and the free gpt-oss:20b, run ten times locally. The refutation step used DoWhy.[^8] Eighteen predictions were committed before any run; ten held.

The first question has a clear answer. In the 100 worlds that follow the literature, controlling for everything measured by the end of the year left a median 84% of the true beta unhedged and got its sign wrong in a quarter of the worlds. That is the factor-model habit of adding every available variable. Regressing on the factor alone left 46%, and controlling for the true confounder left 17%. In worlds where the factor's true effect was exactly zero, the regression on the factor alone reported a significant and meaningful beta 81% of the time: the factor mirage of Experiment 01, inside a full finance graph.[^9]

The second answer is less comfortable. None of the discovery methods recovered the beta. The LLM, the refuted LLM, the supervised model and the hybrid all left 32–40% of the beta unhedged. That is about twice the oracle's error, and no better than a rule needing no graph at all: control only for what is measured before the factor (30%). Where the true effect was zero, that simple rule reported a false effect 20% of the time; the discovery methods did so 38–58% of the time.

A diagnostic run after the results, and labelled as such, shows why. No method usually controlled for leverage. Claude's graph controlled for the past five-year return instead, and the supervised model's most common adjustment set was empty. This also explains the failed prediction about a hidden confounder. Deleting leverage from the data barely changed any method's error, since none was using it. Only the oracle, which did use it, saw its error nearly triple (17% → 46%).

On reading the graph as a whole, the supervised model was best at every level of agreement, with 68–75% balanced accuracy over the eight roles. Claude scored 50% even on the literature's own graph, and gpt-oss 33%. All four of Claude's misses involved an arrow into or out of book-to-market. It treated leverage and earnings surprise as causes of the return only, and quality as a consequence of the return only. That graph is my reading of the literature, though: in these worlds it is true by construction, but some of its arrows are a judgement call.

The LLMs were only as good as the literature, and below random guessing when every edge had been changed. Refutation against data helped gpt-oss by about 13 points at every level, mostly by pruning the 25–32 edges it drew against 13 true ones. For Claude, it helped only where the literature was wrong. PC beat both LLMs everywhere except Claude on the literature graph.

The hybrid never beat the supervised model alone. When the supervised model overruled the LLM, it was right 88–90% of the time, so its confidence threshold was well calibrated. The loss came from starting with the LLM's answer, which stood wherever the supervised model was less sure, even though it was right about half the time there.

The most practical result is that **reading the graph well is not the same as hedging well**. The supervised model got 71% of roles right and Claude 50%, yet both leaked the same 38% of the beta. Only the confounder role enters the regression.

Realistic data changed the ranking. I added fat tails, raw values passed between variables, and a regime break halfway through each sample. The supervised model lost 19 points and ended level with Claude on the literature graph (52.9% against 50.0%). This is the model-risk lesson of Olivetti et al.: a model trained on simulations is only as good as their resemblance to the data it meets. The oracle's error rose from 17% to 30%. The simple timing rule collapsed from 30% to 65%, worse than the factor alone. One possible reason, not tested, is that its control set includes index membership, which the factor itself determines.

As for cost, an LLM run took 7–11 minutes and cost nothing beyond a subscription. The supervised model took 22.6 hours of GPU training once, then about 12 minutes per 500 datasets.

These conclusions have limits. Everything here is synthetic, so it shows what happens inside one data recipe built around one literature graph, which is my own reading of the literature. The literature level of the dial is a single graph. Hedge error is measured relative to the true beta, not in dollars. The beta comes from a linear regression in a nonlinear world, which is why even the oracle leaks 17%. The LLM results rest on two models, a fixed set of prompts and a small number of runs.

Within those limits, the experiment answers its question in two parts. A causal graph changes a hedge far more than the choice of discovery method does. And no method, whether the language model, the supervised model or both together, reliably found the one arrow that mattered most for the hedge. For a portfolio, that means the confounders still need to be argued and checked by a person, with any method's graph treated as a draft rather than an answer.

Figures: [beta error without a graph](../experiments/04_discovery_for_portfolios/figures/fig1_beta_error.png) · [role accuracy across the dial](../experiments/04_discovery_for_portfolios/figures/fig2_dial.png) · [hidden confounder and realistic data](../experiments/04_discovery_for_portfolios/figures/fig3_stress.png). Results: [`results/exp04/b6_analysis/`](../results/exp04/b6_analysis/). Design and predictions: [`design.md`](../experiments/04_discovery_for_portfolios/design.md), [`predictions.md`](../experiments/04_discovery_for_portfolios/predictions.md).

[^1]: Marcos López de Prado, _Causal Factor Investing: Can Factor Investing Become Scientific?_ (Cambridge: Cambridge University Press, 2023), https://doi.org/10.1017/9781009397315.
    
[^2]: Alik Sokolov, Fabrizzio Sabelli, Behzad Azadie Faraz, Wuding Li, and Luis Seco, "Toward Automating Causal Discovery in Financial Markets and Beyond," in _Transactions of ADIA Lab: Interdisciplinary Advances in Data and Computational Science_, ed. Horst Simon (Singapore: World Scientific, 2025), 359–92, https://doi.org/10.1142/9789819813049_0014.
    
[^3]: Mutian Hong and Guoqin Gu, "ADIA Lab Causal Discovery Challenge: Rank 3 Solution," GitHub repository, accessed October 13, 2026, https://github.com/Ori-Replication/ADIA-Lab-Causal-Discovery.
    
[^4]: Peter Spirtes, Clark Glymour, and Richard Scheines, _Causation, Prediction, and Search_, 2nd ed. (Cambridge, MA: MIT Press, 2000).
    
[^5]: Marcos López de Prado and Vincent Zoonekynd, _Causality and Factor Investing: A Primer_ (Charlottesville, VA: CFA Institute Research Foundation, 2025), https://doi.org/10.56227/25.1.30.
    
[^6]: Emanuele Olivetti, Vincent Zoonekynd, Patrick Yam, Marcos López de Prado, Guido W. Imbens, and Miguel A. Hernán, "Can AI Learn Causal Structure? Evidence from ADIA Lab's Causal Discovery Challenge," working paper, January 24, 2026, https://doi.org/10.2139/ssrn.6125566.
    
[^7]: Alexander Reisach, Christof Seiler, and Sebastian Weichwald, "Beware of the Simulated DAG! Causal Discovery Benchmarks May Be Easy to Game," in _Advances in Neural Information Processing Systems 34_ (Red Hook, NY: Curran Associates, 2021), 27772–84.
    
[^8]: Amit Sharma and Emre Kiciman, "DoWhy: An End-to-End Library for Causal Inference," preprint, arXiv, 2020, https://arxiv.org/abs/2011.04216.
    
[^9]: López de Prado and Zoonekynd, _Causality and Factor Investing_.
