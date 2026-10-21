"""
Arm C1: the official ADIA PC baseline (gCastle PC + fix_DAG, as in a1_reproduce_adia/pc_baseline.py) on every test
dataset: main, S1, S2, S2b, S3 (S3 raw, as decided). fix_DAG always adds X -> Y, as the baseline does, so in S2/S2b
PC keeps X -> Y by construction. Saves results/exp04/b5_runs/pc_edges.csv.gz: one row per (set, dataset, edge),
with variables in finance names. A few minutes on 16 cores.
"""

import sys
import warnings
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'a1_reproduce_adia'))
from common import datasets
from common.paths import RESULTS, SYNTHETIC
from pc_baseline import SEED, pc_graph

SETS = ['main', 's1', 's2', 's2b', 's3']
OUT = RESULTS / 'b5_runs' / 'pc_edges.csv.gz'
META = RESULTS / 'b3_worlds' / 'datasets_meta.csv'


def finance_names(meta_row):
    return {col[4:]: name for col, name in meta_row.items() if col.startswith('col_') and isinstance(name, str)}


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    meta = pd.read_csv(META).set_index('id')
    rows = []
    for s in SETS:
        frames = datasets.load(SYNTHETIC / 'b3_worlds' / s / 'X.pkl')
        results = Parallel(n_jobs=-1)(delayed(pc_graph)(k, frames[k], SEED + i) for i, k in enumerate(sorted(frames)))
        for key, adj in results:
            names = finance_names(meta.loc[key])
            rows += [{'set': s, 'dataset': key, 'source': names[a], 'target': names[b]}
                     for a in adj.index for b in adj.columns if adj.loc[a, b]]
        print(f"{s}: PC on {len(frames)} datasets")
    table = pd.DataFrame(rows)
    table.to_csv(OUT, index=False)
    print(f"saved {OUT}: {table.groupby('dataset').size().mean():.1f} edges per dataset on average")