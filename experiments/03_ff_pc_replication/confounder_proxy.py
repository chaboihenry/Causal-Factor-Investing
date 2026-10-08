"""
Confounder-proxy test for U1 (rate surprises) on the HML–Mkt-RF link.

Regresses HML on Mkt-RF three ways: alone, adding the daily 10-year yield
change (proxy for U1), and adding RMW (blocks the Mkt-RF -> RMW -> HML chain).
Runs on the full sample and on two regimes (before/after 2000, when the
stock-bond correlation changed sign). Reports the Mkt-RF beta, its 95% CI,
adjusted R², the % change in beta versus model 1, and the regime's
correlation between the yield change and Mkt-RF.
Saves results/confounder_proxy.csv.
"""

import pandas as pd
import statsmodels.api as sm
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / 'data' / 'processed' / 'factors_daily_ext.csv'
OUT = ROOT / 'results' / 'confounder_proxy.csv'

Y = 'HML'
X = 'Mkt-RF'
PROXY = 'dY10_bp'
MODELS = [
    ('1: Mkt-RF', [X]),
    ('2: + dY10', [X, PROXY]),
    ('3: + dY10 + RMW', [X, PROXY, 'RMW']),
]
PERIODS = [
    ('1963', '2026'),   # full sample
    ('1963', '1999'),
    ('2000', '2026'),
]

df = pd.read_csv(DATA, parse_dates=True, index_col=0)

rows = []
for start, end in PERIODS:
    sub = df.loc[start:end]
    proxy_corr = sub[PROXY].corr(sub[X])
    print(f"{start}-{end}: {len(sub)} days, corr({PROXY}, {X}) = {proxy_corr:.3f}")

    base = None
    for name, regressors in MODELS:
        fit = sm.OLS(sub[Y], sm.add_constant(sub[regressors])).fit()
        low, high = fit.conf_int().loc[X]
        beta = fit.params[X]
        if base is None:
            base = beta
        rows.append({
            'period': f"{start}-{end}",
            'model': name,
            'beta': beta,
            'ci_low': low,
            'ci_high': high,
            'adjr2': fit.rsquared_adj,
            'pct_change': (beta - base) / abs(base) * 100,
            'proxy_corr_mkt': proxy_corr,
        })

res_df = pd.DataFrame(rows)
print(res_df.round(4))
res_df.to_csv(OUT, index=False)