"""
Replicate the Primer's Exhibit 5 on public Fama-French factors.

Finds collider triples (X -> Y and X -> C <- Y) in the aggregated PC graph,
then regresses Y on X with and without the collider C. Reports X's beta,
p-value and adjusted R² in both models, and flags sign flips.
Saves results/exp03/exhibit5_replication.csv.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / 'results' / 'exp03' / 'pc_edge_stability.csv'
FACTORS = ROOT / 'data' / 'processed' / 'factors_daily.csv'
OUT = ROOT / 'results' / 'exp03' / 'exhibit5_replication.csv'

df = pd.read_csv(DATA)
df = df[(df['keep'] == True) & (df['tie'] == False) & (df['majority'].str.contains('->'))]

edges = set()
for m in df['majority']:
    src, dst = m.split("->")
    edges.add((src, dst))
nodes = sorted({n for e in edges for n in e})
print(f"{len(edges)} directed edges, {len(nodes)} nodes")

triples = []
for x, y in sorted(edges):
    for c in nodes:
        if c in (x, y):
            continue
        if (x, c) in edges and (y, c) in edges:
            triples.append((x, y, c))
print(f"{len(triples)} collider triples")

fac_df = pd.read_csv(FACTORS, parse_dates=True, index_col=0)

rows = []
for x, y, c in triples:
    correct = sm.OLS(fac_df[y], sm.add_constant(fac_df[[x]])).fit()
    biased = sm.OLS(fac_df[y], sm.add_constant(fac_df[[x, c]])).fit()
    rows.append({
        'factor': y,
        'cause': x,
        'collider': c,
        'correct_beta': correct.params[x],
        'correct_p': correct.pvalues[x],
        'correct_adjr2': correct.rsquared_adj,
        'biased_beta': biased.params[x],
        'biased_p': biased.pvalues[x],
        'biased_adjr2': biased.rsquared_adj,
    })

ex5_df = pd.DataFrame(rows)
ex5_df['sign_flip'] = np.sign(ex5_df['correct_beta']) != np.sign(ex5_df['biased_beta'])
print(ex5_df.round(4))
ex5_df.to_csv(OUT, index=False)