"""PC causal discovery on Fama-French factors, full sample and by decade.

Replicates the Primer's approach (PC on factor returns) on public Ken French
data, and tests stability: which edges and directions survive across periods.
Saves every period's edges to results/pc_by_period.csv.
"""

import pandas as pd
from pathlib import Path
from causallearn.search.ConstraintBased.PC import pc

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / 'data' / 'processed' / 'factors_daily.csv'
OUT = ROOT / 'results' / 'pc_by_period.csv'

ALPHA = 0.05
TEST = 'fisherz'

PERIODS = [
    ('1963', '2026'),   # full sample
    ('1963', '1972'),
    ('1973', '1982'),
    ('1983', '1992'),
    ('1993', '2002'),
    ('2003', '2012'),
    ('2013', '2026'),
]

fac_df = pd.read_csv(DATA, parse_dates=True, index_col=0)
names = list(fac_df.columns)

records = []
for start, end in PERIODS:
    df = fac_df.loc[start:end]
    print(f"{start}-{end}: {df.shape[0]} days")

    cg = pc(df.to_numpy(), alpha=ALPHA, indep_test=TEST,
            node_names=names, show_progress=False)
    g = cg.G.graph   # g[i,j] = -1 and g[j,i] = 1 means i -> j

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if g[i, j] == 0 and g[j, i] == 0:
                continue                                  # no edge
            if g[i, j] == -1 and g[j, i] == 1:
                src, dst, kind = names[i], names[j], '->'
            elif g[i, j] == 1 and g[j, i] == -1:
                src, dst, kind = names[j], names[i], '->'
            elif g[i, j] == -1 and g[j, i] == -1:
                src, dst, kind = names[i], names[j], '--'
            else:
                src, dst, kind = names[i], names[j], 'other'
            records.append({'period': f"{start}-{end}", 'n_days': df.shape[0],
                            'source': src, 'target': dst, 'type': kind})

edges = pd.DataFrame(records)
edges.to_csv(OUT, index=False)
print(edges.groupby('period').size())