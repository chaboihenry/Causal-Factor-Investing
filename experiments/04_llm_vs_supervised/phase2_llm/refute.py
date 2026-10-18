"""
Sokolov et al. refutation step (arm A2): test every edge of an LLM's final DAG on data with DoWhy.
Per edge: DoWhy finds the backdoor adjustment set; OLS gives the effect and p-value; a data-subset refuter
(50 x 80% subsamples) checks stability. Kept if p < 0.05 and refuter p >= 0.05. Total effect, as in the paper.
Saves results/exp04/phase2/refute/<graph>_<label>_runNN.csv and refuted_dag.json beside final_dag.json.
"""

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import statsmodels.api as sm
from dowhy import CausalModel
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common.paths import RESULTS, SYNTHETIC

GRAPHS = HERE / 'graphs'
RESPONSES = HERE / 'responses'
OUT = RESULTS / 'phase2' / 'refute'
ALPHA = 0.05
N_REFUTE, SUBSET = 50, 0.8
N_ROWS, SEED = 1000, 0


def simulate_textbook(graph):
    """Binary data from a textbook graph's true edges: each node = 1 if a noisy sum of its parents crosses 0.
    TubOrCancer (Asia) is an exact OR of its parents, as defined."""
    truth = nx.DiGraph(json.loads((GRAPHS / f"{graph}_truth.json").read_text())['edges'])
    names = [v['name'] for v in json.loads((GRAPHS / f"{graph}_variables.json").read_text())['variables']]
    truth.add_nodes_from(names)
    rng = np.random.default_rng(SEED)
    data = {}
    for node in nx.topological_sort(truth):
        parents = list(truth.predecessors(node))
        if node == 'TubOrCancer':
            data[node] = np.maximum.reduce([data[p] for p in parents])
            continue
        latent = rng.normal(-0.5, 1, N_ROWS) + sum(1.5 * data[p] for p in parents)
        data[node] = (latent > 0).astype(int)
    return pd.DataFrame(data)[names]


def adjustment_set(data, dag, source, target):
    """DoWhy's backdoor adjustment set for source -> target under the LLM's graph (the do-calculus step)."""
    model = CausalModel(data=data, treatment=source, outcome=target, graph=nx.DiGraph(dag))
    return list(model.identify_effect(proceed_when_unidentifiable=True).get_backdoor_variables())


def ols_effect(data, source, target, adjust):
    fit = sm.OLS(data[target], sm.add_constant(data[[source] + adjust], has_constant='add')).fit()
    return float(fit.params[source]), float(fit.pvalues[source])


def test_edge(data, dag, source, target):
    """Effect and p-value by OLS on the backdoor set; refuter: re-estimate on 80% subsets and test whether the
    full-sample estimate is consistent with them (normal test, as DoWhy's data_subset_refuter does)."""
    adjust = adjustment_set(data, dag, source, target)
    effect, p_value = ols_effect(data, source, target, adjust)
    rng = np.random.default_rng(SEED)
    sims = np.array([ols_effect(data.sample(frac=SUBSET, random_state=int(rng.integers(1 << 31))), source, target, adjust)[0]
                     for _ in range(N_REFUTE)])
    z = (effect - sims.mean()) / sims.std(ddof=1) if sims.std(ddof=1) > 0 else 0.0
    refute_p = float(2 * stats.norm.sf(abs(z)))
    return effect, p_value, refute_p, adjust


def refute(graph, label, run_no, data):
    run_dir = RESPONSES / graph / label / f"run{run_no:02d}"
    final = json.loads((run_dir / 'final_dag.json').read_text(encoding='utf-8'))
    dag = nx.DiGraph([(e['source'], e['target']) for e in final['edges']])
    dag.add_nodes_from(data.columns)

    rows = []
    for source, target in dag.edges:
        try:
            effect, p, refute_p, adjust = test_edge(data, dag, source, target)
            kept, note = p < ALPHA and refute_p >= ALPHA, ''
        except Exception as err:                     # estimation failed: keep the edge, flag it
            effect = p = refute_p = np.nan
            adjust, kept, note = [], True, f"error: {type(err).__name__}: {err}"[:200]
        rows.append({'source': source, 'target': target, 'adjusted_for': ' '.join(adjust), 'effect': effect,
                     'p_value': p, 'refute_p_value': refute_p, 'kept': kept, 'note': note})
    table = pd.DataFrame(rows)

    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / f"{graph}_{label}_run{run_no:02d}.csv", index=False)
    kept_edges = [e for e in final['edges'] if (e['source'], e['target']) in
                  set(zip(table.loc[table.kept, 'source'], table.loc[table.kept, 'target']))]
    result = {**{k: final[k] for k in ['graph', 'backend', 'run']}, 'rule': f"p<{ALPHA} and refute_p>={ALPHA}",
              'n_before': len(final['edges']), 'n_after': len(kept_edges), 'edges': kept_edges}
    (run_dir / 'refuted_dag.json').write_text(json.dumps(result, indent=1), encoding='utf-8')
    print(table.round(3).to_string(index=False))
    print(f"{graph} / {label}: {len(final['edges'])} -> {len(kept_edges)} edges kept, "
          f"{int(table.note.ne('').sum())} estimation errors")
    return kept_edges


def check(graph, before, after):
    truth = {tuple(e) for e in json.loads((GRAPHS / f"{graph}_truth.json").read_text())['edges']}
    for name, edges in [('before refutation', before), ('after refutation ', after)]:
        found = {(e['source'], e['target']) for e in edges}
        print(f"{name}: {len(found & truth)}/{len(truth)} true edges, {len(found - truth)} extra")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('graph')
    parser.add_argument('--label', required=True, help="model folder name under responses/<graph>/")
    parser.add_argument('--run', type=int, default=1)
    parser.add_argument('--data', help="CSV to test on; default: simulate from the textbook graph")
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    warnings.filterwarnings('ignore')
    logging.getLogger('dowhy').setLevel(logging.ERROR)

    if args.data:
        data = pd.read_csv(args.data)
    else:
        path = SYNTHETIC / 'phase2' / f"{args.graph}.csv"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            simulate_textbook(args.graph).to_csv(path, index=False)
        data = pd.read_csv(path)

    before = json.loads((RESPONSES / args.graph / args.label / f"run{args.run:02d}" / 'final_dag.json').read_text())['edges']
    after = refute(args.graph, args.label, args.run, data)
    if args.check:
        check(args.graph, before, after)