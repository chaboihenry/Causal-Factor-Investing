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

### Step 4 — Confounder-proxy tests

**The idea:** if two factors are linked only through a hidden common cause,
controlling for that cause makes the link disappear (Exp 01). A proxy is a noisy
stand-in, so expect the link to **shrink toward zero**, not vanish.
Judge by how much β shrinks, not by p-values (with thousands of days, even a
tiny β stays "significant").

**General caveats:**

- **Removing confounding doesn't always shrink β.** If the confounding pushed β
  the opposite way to a real effect, removing it makes |β| *bigger*.
- **A proxy may also be an effect** of the factors (it reacts the same day).
  Then controlling for it isn't pure confounder adjustment.

#### 4a — U1: rate surprises (HML–Mkt-RF)

**My graph:** two links between HML and Mkt-RF:

1. hidden common cause: Mkt-RF ← U1 → HML (confounder)
2. chain: Mkt-RF → RMW → HML (RMW is a **mediator**, not a confounder)

**Proxy:** daily change in the 10-year Treasury yield (FRED DGS10, in bp).
File: `factors_u1_yield.csv`, 15,736 days (bond holidays dropped).

**Prediction (written before running):** |β| on Mkt-RF falls step by step.

| Model                     | Paths left open      | Predicted |β| |
| ------------------------- | -------------------- | --------------- |
| HML ~ Mkt-RF              | U1 + RMW chain       | largest         |
| HML ~ Mkt-RF + dY10       | (rest of U1) + chain | smaller         |
| HML ~ Mkt-RF + dY10 + RMW | (rest of U1)         | close to zero   |

Model 3 blocks a *real* effect (the RMW chain), so it tests whether a *direct*
link is left. My graph says there isn't.

**Result — full sample (1963–2026):**

| Model           | Mkt-RF β | 95% CI             | adjR² | Change |
| --------------- | --------- | ------------------ | ------ | ------ |
| 1: Mkt-RF       | −0.097   | [−0.106, −0.088] | 0.028  | —     |
| 2: + dY10       | −0.096   | [−0.105, −0.088] | 0.037  | 0.4%   |
| 3: + dY10 + RMW | −0.091   | [−0.100, −0.081] | 0.039  | 6.4%   |

**Prediction failed.** β barely moves and is nowhere near zero.

**Why:** a confounder must be linked to *both* variables. Over the full sample
the yield change is linked to HML (corr +0.10) but not to the market (corr −0.007).

| Period     | Corr(dY10, Mkt-RF) | Model 1 β | Model 2 β | Model 3 β |
| ---------- | ------------------ | ---------- | ---------- | ---------- |
| 1963–1999 | −0.26             | −0.259    | −0.268    | −0.257    |
| 2000–2026 | +0.28              | 0.006      | −0.019    | 0.031      |

- **The stock–bond link flipped sign around 2000.** Before: rates up → stocks down
  (inflation fears). After: rates up → stocks up (growth optimism). Over 60 years
  the two eras cancel out.
- **The HML–Mkt-RF link changed too:** −0.26 before 2000, about zero after.
- **Before 2000, the proxy still doesn't explain the link** (β moves < 4%).
- **After 2000, adding the proxy makes β significantly negative** (from ~0):
  confounding may have been hiding a real negative link.

**Lessons:**

- **A causal arrow can change sign over time.** One fixed graph can't show that.
- **Test causal claims within stable regimes**, not only over the full sample.
- **% change from a near-zero β is meaningless** (±400% after 2000).
- **A failed prediction is a finding:** my U1 story, measured by 10y yields,
  doesn't explain the HML–Mkt-RF link.

#### 4b — U4: liquidity (SMB–ST_Rev)

**My graph:** SMB and ST_Rev share a hidden cause, ST_Rev ← U4 → SMB.
**Update from Step 2:** PC linked Mkt-RF–ST_Rev in all 6 decades, and my graph
has Mkt-RF → SMB. So the market may be a second common cause of the pair.
I control for it too. (This uses what PC found; my original graph had no
Mkt-RF–ST_Rev link.)

**Proxies** (both partly reflect liquidity; using both checks if they agree):

- **dBAA10Y** (main): daily change in the corporate credit spread, in bp. Rises when
  borrowing gets harder. Contains the 10y yield change with opposite sign.
- **dVIX** (check): daily change in the VIX, in points. Closer to fear than
  liquidity, and jumps the same day the market falls.

File: `factors_u4_liquidity.csv`, from 1990 (where both proxies exist).
All models use the same days, so differences come from the controls, not the sample.

**Prediction (written before running):** |β| on SMB falls step by step.

| Model                                   | Paths left open      | Predicted |β| |
| --------------------------------------- | -------------------- | --------------- |
| 1: ST_Rev ~ SMB                         | U4 + market          | largest         |
| 2: + Mkt-RF                             | U4                   | smaller         |
| 3a: + Mkt-RF + dBAA10Y                  | rest of U4           | smaller still   |
| 3b: + Mkt-RF + dVIX                     | rest of U4           | smaller still   |
| 4: + Mkt-RF + dBAA10Y + dVIX            | least of U4          | closest to zero |

**Also checked:** the same models before and after 2008 (the financial crisis was
a liquidity shock and may have changed the relationship, as 2000 did for rates).

**Result** (SMB β; same days within each period):

| Model                     | 1990–2026 | 1990–2007   | 2008–2026 |
| ------------------------- | ---------- | ------------ | ---------- |
| 1: SMB                    | 0.114      | 0.020 (n.s.) | 0.178      |
| 2: + Mkt-RF               | 0.085      | 0.118        | 0.067      |
| 3a: + Mkt-RF + dBAA       | 0.084      | 0.112        | 0.067      |
| 3b: + Mkt-RF + dVIX       | 0.086      | 0.118        | 0.071      |
| 4: + Mkt-RF + dBAA + dVIX | 0.085      | 0.111        | 0.071      |
| Days                      | 9,151      | 4,492        | 4,659      |

All betas from model 2 on are significant (CIs exclude zero).

| Proxy correlation | with SMB (90–26 / 90–07 / 08–26) | with ST_Rev (90–26 / 90–07 / 08–26) |
| ----------------- | ----------------------------------- | -------------------------------------- |
| dBAA10Y           | −0.06 / −0.10 / −0.03            | −0.07 / −0.08 / −0.07               |
| dVIX              | −0.04 / +0.15 / −0.13             | −0.27 / −0.27 / −0.27               |

**Prediction mostly failed.**

- **The market does the work, not the liquidity proxies.** Adding Mkt-RF moves β
  a lot (−26% overall, −62% after 2008). Adding the proxies after that moves it
  by at most 0.007. Whatever the proxies know about SMB and ST_Rev, the market
  already knows.
- **β never gets close to zero.** About 0.07–0.11 remains in every period, so
  something besides the market and these proxies links SMB and ST_Rev.
- **Before 2008, the market was hiding the link.** β goes from ~0 to 0.118 once
  Mkt-RF is controlled: the "opposite pushes cancel out" case again.
- **The proxies are weakly linked to SMB.** The credit spread barely moves with
  either factor. The VIX moves with ST_Rev (−0.27, every period) but its link to
  SMB **flips sign around 2008** (+0.15 → −0.13), like the yield did around 2000.
- **% change is useless here too** (+505% from a near-zero base before 2008).

**A possible explanation for the leftover link:** ST_Rev is built from portfolios
sorted on size and past return, and small stocks reverse more. So SMB and ST_Rev
may share stocks by construction, the same kind of link as the SMB–CMA gap.
That would be a mechanical link, not a hidden cause. Not tested.

**Step 4 overall:**

- **Neither hidden-cause story held up with these proxies.** For U1 and U4 the
  proxies explained almost nothing; for U4 an *observed* factor (the market)
  explained most of what could be explained.
- **Relationships change by regime** (2000 for rates, 2008 for the VIX).
  Full-sample results can hide two opposite worlds.
- **Twice, controlling for a variable made a link appear** (U1 after 2000,
  U4 before 2008). A near-zero β doesn't mean no link; effects can cancel.
- **Limits:** a weak proxy can't prove a confounder is absent. These tests show
  that *these measures* don't explain the links, not that U1 and U4 don't exist.

**Roadmap:**

- [X] 1. Run PC on the 8 factors; compare with my graph.
- [X] 2. Stability across decades (Primer-style vote).
- [X] 3. Replicate Exhibit 5.
- [X] 4a. Confounder-proxy test U1 (10y Treasury), full sample and two regimes.
- [X] 4b. Confounder-proxy test U4 (credit spread, VIX), 1990–2026 and two regimes.
- [X] 5. Write up: where PC's assumptions failed; how conclusions depend on the graph.

**Untested confounders:** U2 (macro news) and U3 (style rotation) have no clean
proxy. Possible indirect tests: announcement-day correlations for U2; FCI
(which allows hidden causes) for U3.

## Exp 04 — Synthetic benchmark: LLM vs. data-driven discovery

**Why:** real data shows the methods *disagree*, but not who is *right*.
Only a world where I write the true graph can score them.

**Design rules:**

- The LLM is tested in a fresh chat (Gemini or ChatGPT) with no project context,
  so it hasn't seen the true graph.
- The LLM gets only variable names and descriptions, never the true graph.
- I write the true graph, so I can't also be the "theory-first" guesser.
