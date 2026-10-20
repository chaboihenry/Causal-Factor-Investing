"""
Generator check, step 1 (env causal-factor): take ADIA's 1,880 test graphs and regenerate their data with our
generator, so the same graphs exist twice: ADIA's data and ours. Saves X_adia.pkl, X_ours.pkl and graphs.pkl to
data/synthetic/b3_worlds/generator_check/, and prints how strongly linked variables look in each version.
Expected: similar link strengths; B1 then scores both versions (step 2) and generator_check.py compares (step 3).
"""

import random
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import datasets
from common.paths import ADIA, SYNTHETIC
from generator import generate

OUT = SYNTHETIC / 'b3_worlds' / 'generator_check'
N_CHECK = None                     # None = all 1,880 test graphs
SEED = 0


def regenerate(data, adj, seed):
    """Our data for ADIA's graph: same variables, same number of rows, same column order."""
    graph = nx.from_pandas_adjacency(adj, create_using=nx.DiGraph)      # rows are parents
    ours, _, _ = generate(graph, 'X', 'Y', seed, n_rows=len(data))
    return ours[list(data.columns)]


def link_strength(frames, graphs):
    """Median |Spearman correlation| over pairs joined by an edge, and over pairs that aren't."""
    linked, unlinked = [], []
    for k, data in frames.items():
        corr, adj = data.corr(method='spearman').abs(), graphs[k]
        for i, a in enumerate(data.columns):
            for b in data.columns[i + 1:]:
                (linked if adj.loc[a, b] or adj.loc[b, a] else unlinked).append(corr.loc[a, b])
    return np.median(linked), np.median(unlinked)


if __name__ == '__main__':
    X_test, y_test = pd.read_pickle(ADIA / 'X_test_reduced.pickle'), pd.read_pickle(ADIA / 'y_test_reduced.pickle')
    keys = sorted(X_test) if N_CHECK is None else sorted(random.Random(SEED).sample(sorted(X_test), N_CHECK))
    adia = {k: X_test[k] for k in keys}
    graphs = {k: y_test[k] for k in keys}
    ours = {k: regenerate(adia[k], graphs[k], seed) for seed, k in enumerate(keys)}

    datasets.save(OUT / 'X_adia.pkl', adia)
    datasets.save(OUT / 'X_ours.pkl', ours)
    datasets.save(OUT / 'graphs.pkl', graphs)
    for name, frames in [('ADIA', adia), ('ours', ours)]:
        on, off = link_strength(frames, graphs)
        print(f"{name:>4}: median |corr| on true edges {on:.3f}, on non-edges {off:.3f}")
    print(f"saved {len(keys)} graphs x 2 data versions to {OUT}")