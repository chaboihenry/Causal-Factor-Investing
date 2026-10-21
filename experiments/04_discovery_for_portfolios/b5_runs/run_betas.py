"""
Step 4: the beta of Y (return) on X (book-to-market) in every test dataset, for the naive regressions (X alone;
everything known by year-end; start-of-year variables), the oracle (the true confounders) and every arm (controlling
for the variables it labels confounder). Adds the true beta and predictions.md's measures: beta error, wrong sign and
hedge leak (only where |true beta| >= 0.10), false effect (true beta exactly 0: p < 0.05 and |beta| >= 0.10).
Saves results/exp04/b5_runs/betas_all.csv.gz; scores nothing.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import datasets
from common.paths import RESULTS, SYNTHETIC
from run_refute import ols

X, Y = 'BookToMarket', 'AnnualReturn'
SETS = ['main', 's1', 's2', 's2b', 's3']
EVERYTHING_EXCEPT = {'InstOwnershipChange'}             # measured after the return year: a look-ahead
START_OF_YEAR = ['FiscalYearEndMonth', 'Leverage', 'PastFiveYearReturn', 'ValueIndexMember']
MIN_TRUE, MIN_EFFECT, ALPHA = 0.10, 0.10, 0.05
B5 = RESULTS / 'b5_runs'
META = RESULTS / 'b3_worlds' / 'datasets_meta.csv'


def names_of(meta_row):
    return {col[4:]: name for col, name in meta_row.items() if col.startswith('col_') and isinstance(name, str)}


def beta(data, controls, cache):
    """OLS beta of Y on X with these controls, and its p-value; cached per control set."""
    key = tuple(sorted(controls))
    if key not in cache:
        design = np.column_stack([np.ones(len(data)), data[X]] + [data[c] for c in key])
        cache[key] = ols(design, data[Y].to_numpy())
    return cache[key]


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    meta = pd.read_csv(META).set_index('id')
    roles = pd.read_csv(B5 / 'roles_all.csv.gz')
    roles['model'] = roles['model'].fillna('')
    confounders = (roles[roles.pred == 'Confounder'].groupby(['dataset', 'arm', 'model', 'run'])['variable']
                   .apply(list).to_dict())
    true_conf = roles[(roles.arm == 'B1') & (roles.true == 'Confounder')].groupby('dataset')['variable'].apply(list).to_dict()
    arms = roles[['arm', 'model', 'run']].drop_duplicates().itertuples(index=False)
    arms = list(arms)

    rows = []
    for s in SETS:
        for key, frame in sorted(datasets.load(SYNTHETIC / 'b3_worlds' / s / 'X.pkl').items()):
            data, cache = frame.rename(columns=names_of(meta.loc[key])), {}
            others = [c for c in data.columns if c not in (X, Y)]
            specs = [('X alone', '', 0, []),
                     ('Everything', '', 0, [c for c in others if c not in EVERYTHING_EXCEPT]),
                     ('Start of year', '', 0, [c for c in START_OF_YEAR if c in others]),
                     ('Oracle', '', 0, true_conf.get(key, []))]
            specs += [(a.arm, a.model, a.run, confounders.get((key, a.arm, a.model, a.run), [])) for a in arms]
            for arm, model, run, controls in specs:
                b, p = beta(data, controls, cache)
                rows.append({'set': s, 'dataset': key, 'agreement': meta.loc[key, 'agreement'], 'arm': arm,
                             'model': model, 'run': run, 'controls': ' '.join(sorted(controls)), 'beta': b, 'p_value': p})

    table = pd.DataFrame(rows)
    table['true_beta'] = table.dataset.map(meta['true_beta'])
    table['beta_error'] = table.beta - table.true_beta
    counted = table.true_beta.abs() >= MIN_TRUE
    table['wrong_sign'] = np.where(counted, np.sign(table.beta) != np.sign(table.true_beta), np.nan)
    table['hedge_leak'] = np.where(counted, table.beta_error.abs() / table.true_beta.abs(), np.nan)
    table['false_effect'] = np.where(table.true_beta == 0, (table.p_value < ALPHA) & (table.beta.abs() >= MIN_EFFECT), np.nan)
    table.to_csv(B5 / 'betas_all.csv.gz', index=False)
    print(table.groupby(['set']).agg(datasets=('dataset', 'nunique'), regressions=('beta', 'size'),
                                     counted_for_sign=('wrong_sign', 'count')).to_string())
    print(f"saved {B5 / 'betas_all.csv.gz'}: {len(table):,} rows")