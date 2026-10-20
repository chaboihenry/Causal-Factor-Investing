"""
Generator check, step 3 (env causal-factor): run PC on both data versions, read B1's two scores (step 2), and
compare ADIA's data with ours on the same 1,880 test graphs, overall and by edge count. PASS if B1's balanced accuracy
on our data is within 3 points of its score on ADIA's data. Saves results/exp04/b3_worlds/generator_check.csv.
"""

import sys
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'a1_reproduce_adia'))
from common import datasets
from common.paths import RESULTS, SYNTHETIC
from common.scoring import role_table, score
from pc_baseline import SEED, pc_graph

DATA = SYNTHETIC / 'b3_worlds' / 'generator_check'
OUT = RESULTS / 'b3_worlds'
VERSIONS = ['adia', 'ours']
EDGE_BINS, EDGE_LABELS = [0, 5, 10, 15, 100], ['<=5', '6-10', '11-15', '16+']
TOLERANCE = 0.03


def pc_roles(frames, graphs):
    results = Parallel(n_jobs=-1)(delayed(pc_graph)(k, frames[k], SEED + i) for i, k in enumerate(sorted(frames)))
    return role_table(graphs, dict(results))


def b1_roles(version):
    path = OUT / f"third_place_check_{version}_roles.csv"
    if not path.exists():
        sys.exit(f"missing {path}: run step 2 (predict_third_place.py) for '{version}' first")
    return pd.read_csv(path, dtype={'dataset': str, 'variable': str})


def summary(roles, edges):
    """Balanced accuracy overall and by the graph's number of edges."""
    ids = roles['dataset'] if 'dataset' in roles else roles['dataset_id']      # B1 and PC name the column differently
    roles = roles.assign(edges=pd.cut(ids.astype(str).map(edges), EDGE_BINS, labels=EDGE_LABELS))
    by_size = {f"edges {b}": score(g)[0] for b, g in roles.groupby('edges', observed=True)}
    return {'all': score(roles)[0], **by_size}


if __name__ == '__main__':
    graphs = datasets.load(DATA / 'graphs.pkl')
    edges = {str(k): int(g.to_numpy().sum()) for k, g in graphs.items()}
    rows = {}
    for version in VERSIONS:
        frames = datasets.load(DATA / f"X_{version}.pkl")
        rows[f"B1 on {version}"] = summary(b1_roles(version), edges)
        rows[f"PC on {version}"] = summary(pc_roles(frames, graphs), edges)

    table = pd.DataFrame(rows).T
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / 'generator_check.csv')
    print(table.round(4).to_string())
    gap = table.loc['B1 on ours', 'all'] - table.loc['B1 on adia', 'all']
    print(f"\nB1 gap (ours - ADIA): {gap:+.4f} -> {'PASS' if abs(gap) <= TOLERANCE else 'FAIL'} "
          f"(tolerance {TOLERANCE:.2f}); PC gap: {table.loc['PC on ours', 'all'] - table.loc['PC on adia', 'all']:+.4f}")