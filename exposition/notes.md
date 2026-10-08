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

### Step 1 — PC on full sample (alpha = 0.05, Fisher-z)

**17 of 28 possible edges**, plus a **cycle**: Mkt-RF → Mom → ST_Rev → Mkt-RF.
Not a valid DAG. Each arrow comes from a separate test; when PC's assumptions
fail, the tests contradict each other.

| # | Prediction                               | Result                        |
| - | ---------------------------------------- | ----------------------------- |
| 1 | Mom ⟂ ST_Rev                            | Wrong                         |
| 2 | ST_Rev ⟂ Mkt-RF                         | Wrong                         |
| 3 | SMB opens ST_Rev–Mkt-RF                 | Can't test (already adjacent) |
| 4 | SMB ⟂ CMA                               | Consistent                    |
| 5 | Direct edges among HML, CMA, LT_Rev, Mom | Right                         |
| 6 | Many edges                               | Right (17/28)                 |

### Step 2 — Stability across 6 decades

**What the Primer did:** PC on 85 Barra models (different markets), kept edges in ≥ 1/3
of graphs. **What I did:** same vote, but across 6 decades of US data.
So I test stability over *time*, they tested it across *markets*.

**Results:**

- **Edges are stable, directions are not.** 4 pairs appear in all 6 decades
  (CMA–LT_Rev, RMW–SMB, Mkt-RF–ST_Rev, Mkt-RF–SMB), but most pairs flip direction.
- **The 1/3 rule barely filters:** 22 of 28 pairs pass. With lots of data, almost
  every pair is "significant" somewhere.
- **Cycles in 3 of 7 runs**, always through ST_Rev, Mom or SMB.
- **The voted graph still has a cycle:** Mkt-RF → RMW → SMB → Mkt-RF.
- **PC often makes the market an effect** (SMB → Mkt-RF in 5 of 6), which is
  economically hard to believe.

**Lesson:** voting removes random errors, not systematic ones. A hidden cause
present in every decade produces the same wrong edge every time, and the vote keeps it.

### Step 3 — Exhibit 5 replication

For each collider triple in the voted PC graph (X → Y, X → C ← Y):
compare Y ~ X ("correct") with Y ~ X + C ("biased").

| Factor | Cause | Collider | Correct β | adjR² | Biased β | adjR² | Flip |
| ------ | ----- | -------- | ---------- | ------ | --------- | ------ | ---- |
| Mom    | HML   | LT_Rev   | −0.360    | 0.072  | −0.498   | 0.101  |      |
| Mom    | HML   | Mkt-RF   | −0.360    | 0.072  | −0.398   | 0.097  |      |
| Mom    | CMA   | LT_Rev   | 0.037      | 0.000  | 0.039     | 0.000  |      |
| Mom    | CMA   | Mkt-RF   | 0.037      | 0.000  | −0.048   | 0.013  | ✓   |
| Mom    | CMA   | ST_Rev   | 0.037      | 0.000  | 0.001     | 0.007  |      |
| Mkt-RF | CMA   | ST_Rev   | −0.914    | 0.119  | −0.773   | 0.179  |      |
| Mkt-RF | Mom   | ST_Rev   | −0.144    | 0.012  | −0.112   | 0.105  |      |
| Mkt-RF | HML   | RMW      | −0.298    | 0.029  | −0.271   | 0.072  |      |

**Lessons:**

- **1 flip in 8.** It starts from a beta near zero (adjR² 0.0003), the same weak
  pattern as many rows in the Primer's table.
- **adjR² rises in 7 of 8.** Standard metrics reward adding the "collider."
- **Flips aren't the only damage:** betas move a lot without flipping
  (−0.36 → −0.50), and significance can vanish (p 0.024 → 0.94).
- **The key point:** the flip only counts as collider bias if Mkt-RF really is
  *caused by* Mom and CMA. If the market is a *cause* (the usual view), Mkt-RF is
  a confounder, controlling for it is correct, and the "biased" beta is the right one.
  **The same regression is a mistake or a fix depending on arrow direction, and
  arrow direction is exactly what PC can't pin down.**

### Step 4 — Confounder-proxy test (U1: rate surprises)

**Idea:** my graph says HML and Mkt-RF are linked two ways:

1. a hidden common cause: Mkt-RF ← U1 → HML (confounder)
2. a chain: Mkt-RF → RMW → HML (RMW is a **mediator**, not a confounder)

Block both and the link should disappear. A proxy can only block part of U1.

**Proxy:** daily change in the 10-year Treasury yield (FRED DGS10, in bp).
Missing days dropped *before* differencing, so each change runs from the
previous trading day. Merged with the factors: 15,736 days (bond holidays dropped).

**Prediction (written before running):** |β| on Mkt-RF falls step by step.

| Model                        | Paths left open      | Predicted |β| |
| ---------------------------- | -------------------- | --------------- |
| HML ~ Mkt-RF                 | U1 + RMW chain       | largest         |
| HML ~ Mkt-RF + dY10          | (rest of U1) + chain | smaller         |
| HML ~ Mkt-RF + dY10 + RMW    | (rest of U1)         | close to zero   |

**How to judge it:** by how much β shrinks, not by p-values
(with ~15,700 days, even a tiny β stays "significant").

**Caveats:**

- **The proxy is noisy**, so some of U1 always stays. Expect shrinking, not a clean zero.
- **Model 3 removes a real effect.** Blocking the RMW chain erases the part of
  the market's effect on HML that runs through RMW (Exp 01's mediator lesson).
  Model 3 tests whether there's a *direct* link, which my graph says there isn't.
- **Yields may react to stocks the same day** (flight to quality: stocks fall,
  investors buy Treasuries, yields fall). If so, dY10 is partly an *effect* of
  Mkt-RF, not a pure confounder. A shrinking β fits my story but doesn't prove it.
- **Removing confounding doesn't always shrink β.** If the confounding pushed β
  in the opposite direction to the real effect, removing it makes |β| *bigger*.

**Roadmap:**

- [X] 1. Run PC on the 8 factors; compare with my graph.
- [X] 2. Stability across decades (Primer-style vote).
- [X] 3. Replicate Exhibit 5.
- [ ] 4. Confounder-proxy test (U1, 10y Treasury). Data ready; regressions next.
- [ ] 5. Write up: where PC's assumptions failed; how conclusions depend on the graph.

## Exp 04 — Synthetic benchmark: LLM vs. data-driven discovery

**Why:** real data shows the methods *disagree*, but not who is *right*.
Only a world where I write the true graph can score them.

**Design rules:**

- The LLM is tested in a fresh chat (Gemini or ChatGPT) with no project context,
  so it hasn't seen the true graph.
- The LLM gets only variable names and descriptions, never the true graph.
- I write the true graph, so I can't also be the "theory-first" guesser.
