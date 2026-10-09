"""
Build the factor files used in the two confounder-proxy tests.

Loads three FRED series (DGS10, BAA10Y, VIXCLS), drops missing days, then
computes daily changes (dropping first, so each change runs from the previous
trading day): bp for the two yields, index points for the VIX.
Inner-joins with factors_daily.csv into two files in data/processed:
factors_u1_yield.csv (U1 test, from 1963) and
factors_u4_liquidity.csv (U4 test, from 1990, where both proxies exist).
"""

import pandas as pd
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RAW = ROOT / 'data' / 'raw'
FACTORS = ROOT / 'data' / 'processed' / 'factors_daily.csv'
OUT_D = ROOT / 'data' / 'processed'

PROXIES = [
    # (FRED series, level column name, change column name, scale)
    ('DGS10', 'DGS10_pct', 'dY10_bp', 100),
    ('BAA10Y', 'BAA10Y_pct', 'dBAA10Y_bp', 100),
    ('VIXCLS', 'VIX', 'dVIX_pts', 1),
]


def load_proxy(series, level_name, change_name, scale):
    df = pd.read_csv(RAW / f'{series}.csv', parse_dates=['observation_date'], index_col=0)
    level = df[series].dropna()
    change = level.diff() * scale
    out = pd.concat([level.rename(level_name), change.rename(change_name)], axis=1)
    return out.dropna()


proxies = {series: load_proxy(series, lvl, chg, scale) for series, lvl, chg, scale in PROXIES}

factor_df = pd.read_csv(FACTORS, index_col=0)
factor_df.index = pd.to_datetime(factor_df.index)

files = {
    'factors_u1_yield.csv': [proxies['DGS10']],
    'factors_u4_liquidity.csv': [proxies['BAA10Y'], proxies['VIXCLS']],
}

for fname, parts in files.items():
    merged_df = pd.concat(parts + [factor_df], axis=1, join='inner')
    merged_df.to_csv(OUT_D / fname)
    print(f"{fname}: {merged_df.shape}, "
          f"{merged_df.index.min().date()} to {merged_df.index.max().date()}, "
          f"NaNs: {merged_df.isna().sum().sum()}")