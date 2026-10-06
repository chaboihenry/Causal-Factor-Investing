"""
doc description
"""

import pandas as pd
import numpy as np
from pathlib import Path
from causallearn.search.ConstraintBased.PC import pc

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / 'data' / 'processed' / 'factors_daily.csv'

ALPHA = 0.05
TEST = 'fisherz'

fac_df = pd.read_csv(DATA, parse_dates=True, index_col=0)
df = fac_df.loc['1963':'1972']
data = df.to_numpy()
names = list(df.columns)

cg = pc(data, alpha=ALPHA, indep_test=TEST, node_names=names)
print(cg.G)



