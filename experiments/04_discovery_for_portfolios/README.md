# Experiment 04 — Causal Discovery for Portfolios

**Question.** A factor regression without a causal graph can report the wrong beta, and a hedge built on it leaks risk. How large is that error, and which way of drawing the graph gets the beta closest to the truth: an LLM reading the literature, a supervised model reading the data, or both combined?

Full plan, rules and predictions: [`design.md`](design.md) · Predictions committed before any run: [`predictions.md`](predictions.md)

## Check a claim yourself

| # | Claim                                                     | Figure      | Result file | Script      |
| - | --------------------------------------------------------- | ----------- | ----------- | ----------- |
| 1 | No graph → wrong beta → hedge leaks risk                | *pending* | *pending* | *pending* |
| 2 | Which method gets closest, across the agreement dial      | *pending* | *pending* | *pending* |
| 3 | Does it survive realistic data and hidden causes (S1–S3) | *pending* | *pending* | *pending* |
| 4 | Does combining LLM + data beat either alone               | *pending* | *pending* | *pending* |
| 5 | What each method costs                                    | *pending* | *pending* | *pending* |

## Validation: the tools reproduce their papers

| Check                            | Our result                                    | Published                       | Result file                                                                                                                       | Script                                                                                                    |
| -------------------------------- | --------------------------------------------- | ------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| PC baseline on ADIA public test  | 38.30%                                        | ~40% (5th-place report: 37.79%) | [`results/exp04/a1_reproduce_adia/pc_baseline_score.csv`](../../results/exp04/a1_reproduce_adia/pc_baseline_score.csv)           | [`a1_reproduce_adia/pc_baseline.py`](a1_reproduce_adia/pc_baseline.py)                                   |
| ADIA 3rd-place solution          | 71.64%                                        | 74.06% (private test)           | [`results/exp04/a1_reproduce_adia/third_place_full_score.csv`](../../results/exp04/a1_reproduce_adia/third_place_full_score.csv) | [`a1_reproduce_adia/run_third_place.py`](a1_reproduce_adia/run_third_place.py)                           |
| LLM pipeline on Asia / Sprinkler | Claude 8/8 (memorized), gpt-oss 7/8 + 8 extra | —                              | [`a2_reproduce_llm/responses/`](a2_reproduce_llm/responses/), [`results/exp04/a2_reproduce_llm/refute/`](../../results/exp04/a2_reproduce_llm/refute/)            | [`a2_reproduce_llm/run_llm.py`](a2_reproduce_llm/run_llm.py), [`refute.py`](a2_reproduce_llm/refute.py) |

## Inputs

- The finance graph, with a source for every edge: [`graphs/finance_truth.json`](graphs/finance_truth.json)
- Exactly what the LLM sees: [`graphs/finance_variables.json`](graphs/finance_variables.json)
- Every LLM prompt and reply: [`a2_reproduce_llm/responses/`](a2_reproduce_llm/responses/)
- ADIA data: download from the challenge (not redistributed); expected under `data/adia/`.
