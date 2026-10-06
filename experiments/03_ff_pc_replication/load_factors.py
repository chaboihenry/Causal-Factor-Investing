"""
Process the csv's manually downloaded of Fama & Frenchs factors into one clean
usable csv.
"""

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
DATA_DIR = ROOT / 'data' / 'raw'

LT_FACTOR_CSV = DATA_DIR / 'F-F_LT_Reversal_Factor_Daily.csv'
MOM_FACTOR_CSV = DATA_DIR / 'F-F_Momentum_Factor_daily.csv'
FIVE_FACTOR_CSV =  DATA_DIR / 'F-F_Research_Data_5_Factors_2x3_daily.csv'
ST_FACTOR_CSV = DATA_DIR / 'F-F_ST_Reversal_Factor_daily.csv'

results = []
for path in (LT_FACTOR_CSV, MOM_FACTOR_CSV, ST_FACTOR_CSV):
    df = pd.read_csv(path, skiprows=12, index_col=0)
    df.index = pd.to_datetime(df.index.astype(str), format='%Y%m%d', errors='coerce')
    df = df[df.index.notna()]
    results.append(df)
merged_df = pd.concat([results[0], results[1], results[2]], axis=1, join='inner')

five_df = pd.read_csv(FIVE_FACTOR_CSV, skiprows=4, index_col=0)
five_df.index = pd.to_datetime(five_df.index.astype(str), format='%Y%m%d', errors='coerce')
five_df = five_df[five_df.index.notna()]

complete_df = pd.concat([merged_df, five_df], axis=1, join='inner')
complete_df = complete_df / 100
complete_df = complete_df.drop(columns="RF")
complete_df.to_csv(ROOT / 'data' / 'processed' / 'factors_daily.csv', float_format="%.4f")
