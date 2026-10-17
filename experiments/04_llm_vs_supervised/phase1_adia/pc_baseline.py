"""
Reproduce the official ADIA PC baseline on the 1,880 reduced-test datasets.
gCastle PC (Fisher-z, alpha 0.05), then the notebook's fix_DAG with a seeded RNG per dataset.
Saves roles and score to results/exp04/phase1/. Target: ~38% (5th-place report).
"""

import sys
import time
import castle.algorithms
import networkx as nx
import numpy as np
import pandas as pd
import torch  # noqa: F401  gCastle needs torch imported first
from joblib import Parallel, delayed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import ADIA, RESULTS
from common.scoring import role_table, score

OUT = RESULTS / 'phase1'
SEED = 0


def fix_DAG(g, rng):
    """Official baseline: add X -> Y, then break each cycle by removing a random edge (never X -> Y)."""
    gg = g.copy()
    if ('X', 'Y') not in gg.edges:
        gg.add_edge('X', 'Y')
    while not nx.is_directed_acyclic_graph(gg):
        h = gg.copy()
        while True:                                   # strip sources and sinks
            finished = True
            for i, v in nx.in_degree_centrality(h).items():
                if v == 0:
                    h.remove_node(i)
                    finished = False
            for i, v in nx.out_degree_centrality(h).items():
                if v == 0:
                    h.remove_node(i)
                    finished = False
            if finished:
                break
        node = list(h.nodes)[0]                       # random walk until it repeats
        cycle = [node]
        while True:
            edges = list(h.out_edges(node))
            _, node = edges[rng.integers(len(edges))]
            if node in cycle:
                break
            cycle.append(node)
        cycle = cycle[cycle.index(node):] + [node]
        edges = [e for e in zip(cycle[:-1], cycle[1:]) if e != ('X', 'Y')]
        gg.remove_edge(*edges[rng.integers(len(edges))])
    return gg


def pc_graph(dataset_id, data, seed):
    nodes = data.columns
    model = castle.algorithms.PC()
    model.learn(data)
    a_hat = pd.DataFrame(model.causal_matrix, index=nodes, columns=nodes)
    g_hat = fix_DAG(nx.from_pandas_adjacency(a_hat, create_using=nx.DiGraph), np.random.default_rng(seed))
    return dataset_id, nx.to_pandas_adjacency(g_hat, nodelist=nodes, dtype=int)


if __name__ == '__main__':
    X_test = pd.read_pickle(ADIA / 'X_test_reduced.pickle')
    y_test = pd.read_pickle(ADIA / 'y_test_reduced.pickle')

    start = time.time()
    results = Parallel(n_jobs=-1)(delayed(pc_graph)(k, data, SEED + i)
                                  for i, (k, data) in enumerate(X_test.items()))
    minutes = (time.time() - start) / 60
    pred_graphs = dict(results)

    roles_df = role_table(y_test, pred_graphs)
    overall, per_role = score(roles_df)

    OUT.mkdir(parents=True, exist_ok=True)
    roles_df.to_csv(OUT / 'pc_baseline_roles.csv', index=False)
    per_role.loc['ALL (balanced accuracy)'] = [overall, len(roles_df)]
    per_role.to_csv(OUT / 'pc_baseline_score.csv')

    print(f"PC baseline on {len(X_test)} datasets in {minutes:.1f} min")
    print(per_role.round(4))
    print(f"balanced accuracy: {overall:.4f}  (target ~0.38)")