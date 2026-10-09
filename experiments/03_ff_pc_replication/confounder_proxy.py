"""
Confounder-proxy tests for two hidden causes in my graph.

U1 (rate surprises) on HML–Mkt-RF: proxy = daily 10-year yield change.
U4 (liquidity) on ST_Rev–SMB: proxies = daily credit-spread and VIX changes,
with Mkt-RF as an extra control. Each test adds controls step by step and
runs on its full sample and two regimes. Reports the beta of interest, its
95% CI, adjusted R², % change versus model 1, and each proxy's correlation
with both factors. Saves results/confounder_proxy.csv.
"""

import pandas as pd
import statsmodels.api as sm
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROC = ROOT / 'data' / 'processed'
OUT = ROOT / 'results' / 'confounder_proxy.csv'

TESTS = [
    {
        'name': 'U1',
        'file': 'factors_u1_yield.csv',
        'y': 'HML',
        'x': 'Mkt-RF',
        'proxies': ['dY10_bp'],
        'models': [
            ('1: Mkt-RF', []),
            ('2: + dY10', ['dY10_bp']),
            ('3: + dY10 + RMW', ['dY10_bp', 'RMW']),
        ],
        'periods': [('1963', '2026'), ('1963', '1999'), ('2000', '2026')],
    },
    {
        'name': 'U4',
        'file': 'factors_u4_liquidity.csv',
        'y': 'ST_Rev',
        'x': 'SMB',
        'proxies': ['dBAA10Y_bp', 'dVIX_pts'],
        'models': [
            ('1: SMB', []),
            ('2: + Mkt-RF', ['Mkt-RF']),
            ('3a: + Mkt-RF + dBAA', ['Mkt-RF', 'dBAA10Y_bp']),
            ('3b: + Mkt-RF + dVIX', ['Mkt-RF', 'dVIX_pts']),
            ('4: + Mkt-RF + dBAA + dVIX', ['Mkt-RF', 'dBAA10Y_bp', 'dVIX_pts']),
        ],
        'periods': [('1990', '2026'), ('1990', '2007'), ('2008', '2026')],
    },
]

rows = []
for test in TESTS:
    df = pd.read_csv(PROC / test['file'], parse_dates=True, index_col=0)
    y, x = test['y'], test['x']
    print(f"\n=== {test['name']}: {y} ~ {x} ===")

    for start, end in test['periods']:
        sub = df.loc[start:end]
        corrs = {f"corr_{p}_{v}": sub[p].corr(sub[v])
                 for p in test['proxies'] for v in (x, y)}
        print(f"{start}-{end}: {len(sub)} days, "
              + ", ".join(f"{k} = {v:.3f}" for k, v in corrs.items()))

        base = None
        for name, controls in test['models']:
            fit = sm.OLS(sub[y], sm.add_constant(sub[[x] + controls])).fit()
            low, high = fit.conf_int().loc[x]
            beta = fit.params[x]
            if base is None:
                base = beta
            rows.append({
                'test': test['name'],
                'period': f"{start}-{end}",
                'n_days': len(sub),
                'model': name,
                'beta': beta,
                'ci_low': low,
                'ci_high': high,
                'adjr2': fit.rsquared_adj,
                'pct_change': (beta - base) / abs(base) * 100,
                **corrs,
            })

res_df = pd.DataFrame(rows)
pd.set_option('display.width', 200)
print()
print(res_df[['test', 'period', 'model', 'beta', 'ci_low', 'ci_high', 'adjr2', 'pct_change']].round(4).to_string(index=False))
res_df.to_csv(OUT, index=False)