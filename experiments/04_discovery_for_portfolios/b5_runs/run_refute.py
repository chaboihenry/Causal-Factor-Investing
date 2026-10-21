"""
Arm A2: refute every LLM graph (gpt-oss 10 runs, Claude 3) on every test dataset with refute.py's rule: per edge,
the smallest valid backdoor set (DoWhy, computed once per graph), OLS effect and p-value, 50 x 80% subset refuter;
kept if p < 0.05 and refuter p >= 0.05. S1 drops leverage from the LLM graph (the analyst doesn't have it).
Saves results/exp04/b5_runs/a2/<model>_run<NN>.csv.gz (one row per dataset and edge). Resumable; ~1 h on 16 cores.
"""

import json
import sys
import warnings
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'a2_reproduce_llm'))
from common import datasets
from common.paths import RESPONSES, RESULTS, SYNTHETIC
from refute import ALPHA, N_REFUTE, SEED, SUBSET, adjustment_set

RUNS = {'gpt-oss-exp04': range(1, 11), 'claude-opus-5.5-medium': range(1, 4)}
SETS = ['main', 's1', 's2', 's2b', 's3']
HIDDEN = 'Leverage'
OUT = RESULTS / 'b5_runs' / 'a2'
META = RESULTS / 'b3_worlds' / 'datasets_meta.csv'


def ols(X, y):
    """Coefficient and two-sided p-value of column 1 (after the constant), as statsmodels OLS reports them."""
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    dof = len(y) - X.shape[1]
    sigma2 = np.sum((y - X @ coef) ** 2) / dof
    se = np.sqrt(sigma2 * np.linalg.inv(X.T @ X)[1, 1])
    return coef[1], 2 * stats.t.sf(abs(coef[1] / se), dof)


def test_edges(key, data, edges):
    """refute.py's test_edge for every (source, target, adjustment set) on one dataset."""
    rows = []
    for source, target, adjust in edges:
        design = lambda d: np.column_stack([np.ones(len(d)), d[source]] + [d[a] for a in adjust])
        effect, p = ols(design(data), data[target].to_numpy())
        rng = np.random.default_rng(SEED)
        subs = [data.sample(frac=SUBSET, random_state=int(rng.integers(1 << 31))) for _ in range(N_REFUTE)]
        sims = np.array([ols(design(s), s[target].to_numpy())[0] for s in subs])
        z = (effect - sims.mean()) / sims.std(ddof=1) if sims.std(ddof=1) > 0 else 0.0
        refute_p = 2 * stats.norm.sf(abs(z))
        rows.append({'dataset': key, 'source': source, 'target': target, 'adjusted_for': ' '.join(adjust),
                     'effect': effect, 'p_value': p, 'refute_p_value': refute_p,
                     'kept': p < ALPHA and refute_p >= ALPHA})
    return rows


def plan_edges(dag):
    """Each edge with its adjustment set, from the graph alone (DoWhy needs a data frame, not the data)."""
    dummy = pd.DataFrame(0.0, index=range(5), columns=sorted(dag.nodes))
    return [(s, t, adjustment_set(dummy, dag, s, t)) for s, t in sorted(dag.edges)]


def finance_names(meta_row):
    """ADIA column name -> finance variable name, for this dataset."""
    return {col[4:]: name for col, name in meta_row.items() if col.startswith('col_') and isinstance(name, str)}


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    meta = pd.read_csv(META).set_index('id')
    data = {s: datasets.load(SYNTHETIC / 'b3_worlds' / s / 'X.pkl') for s in SETS}
    OUT.mkdir(parents=True, exist_ok=True)

    for model, runs in RUNS.items():
        for run in runs:
            path = OUT / f"{model}_run{run:02d}.csv.gz"
            if path.exists():
                print(f"{path.name}: done, skipped")
                continue
            final = json.loads((RESPONSES / 'finance' / model / f"run{run:02d}" / 'final_dag.json').read_text())
            dag = nx.DiGraph([(e['source'], e['target']) for e in final['edges']])
            plans = {'full': plan_edges(dag), 'no_leverage': plan_edges(dag.subgraph(n for n in dag if n != HIDDEN).copy())}
            rows = []
            for s in SETS:
                edges = plans['no_leverage' if s == 's1' else 'full']
                jobs = [(k, f.rename(columns=finance_names(meta.loc[k])), edges) for k, f in sorted(data[s].items())]
                results = Parallel(n_jobs=-1)(delayed(test_edges)(*job) for job in jobs)
                rows += [{'set': s, **r} for part in results for r in part]
            table = pd.DataFrame(rows)
            table.to_csv(path, index=False)
            print(f"{path.name}: {len(dag.edges)} edges x {table.dataset.nunique()} datasets, "
                  f"{table.kept.mean():.0%} of edge tests kept")