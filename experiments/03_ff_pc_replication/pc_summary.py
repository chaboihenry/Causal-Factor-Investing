"""
Summarize how stable PC's edges are across decades.

Reads results/pc_by_period.csv, ignores the full-sample run, and counts how many
decades each pair of factors is linked in, plus its most common direction.
Primer-style aggregation: they voted across 85 Barra models (markets); this
votes across 6 decades of US data. Saves results/pc_edge_stability.csv.
"""

import pandas as pd
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / 'results' / 'pc_by_period.csv'
OUT = ROOT / 'results' / 'pc_edge_stability.csv'

pc_df = pd.read_csv(DATA)
pc_df = pc_df[pc_df['period'] != '1963-2026'].copy()
pc_df['pair_key'] = pc_df.apply(
    lambda r: " - ".join(sorted([r.source, r.target])), axis=1
)
pc_df['direction'] = np.where(
    pc_df['type'] == '->',
    pc_df['source'] + '->' + pc_df['target'], 
    pc_df['pair_key']
)

n_periods = pc_df['period'].nunique()
threshold = n_periods / 3

summary = pc_df.groupby('pair_key').agg(
    n_decades=('period', 'nunique'),
)

summary['keep'] = summary['n_decades'] >= threshold
summary = summary.sort_values(by='n_decades', ascending=False)

print(f"{n_periods} decades, threshold = {threshold:.2f}")
print(summary)
print(f"pairs seen: {len(summary)}, kept: {summary['keep'].sum()}")


def majority(s):
    return s.value_counts().index[0]


def is_tie(s):
    vc = s.value_counts()
    return len(vc) > 1 and vc.iloc[0] == vc.iloc[1]


directions = pc_df.groupby('pair_key').agg(
    directions=('direction', lambda s: s.value_counts().to_dict()),
    majority=('direction', majority),
    tie=('direction', is_tie),
)
summary = summary.join(directions)
summary.reset_index().to_csv(OUT, index=False)


