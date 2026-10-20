"""
Compare our generator's data with ADIA's on 300 ADIA *training* graphs (never the generator-check test graphs), for
several settings of three open knobs: GP smoothness, noise scale and how kinked the nets are. Per graph-and-node it measures
signal strength (flexible R^2 of a node from its parents), linearity (linear R^2 / flexible R^2) and |corr| on edges.
The setting whose numbers sit closest to ADIA's is fixed in generator.py before the gate is rerun. No method is run.
"""

import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import KNeighborsRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import ADIA, RESULTS
import generator
from make_check_data import regenerate

OUT = RESULTS / 'b3_worlds' / 'generator_calibration.csv'
N_GRAPHS = 300
SEED = 1
SETTINGS = [  # (label, GP lengthscale range, noise scale, net bias scale); round 1 pointed to smoother + less noise
    ('current', (0.5, 2.0), 1.0, 1.0),
] + [(f"ls {ls[0]:g}-{ls[1]:g}, noise {noise}, bias {bias}", ls, noise, bias)
     for ls in [(2.0, 8.0), (4.0, 16.0)] for noise in [0.4, 0.5, 0.6] for bias in [1.0, 3.0]]


def r2(model, X, y):
    pred = cross_val_predict(model, X, y, cv=5)
    return 1 - np.mean((y - pred) ** 2) / np.var(y)


def statistics(frames, graphs):
    """Medians over all nodes with parents: flexible R^2, linear share of it, and |Spearman| over all edges."""
    flexible, linear_share, corr = [], [], []
    for k, data in frames.items():
        adj = graphs[k]
        spearman = data.corr(method='spearman').abs()
        for child in data.columns:
            parents = [p for p in data.columns if adj.loc[p, child]]
            if not parents:
                continue
            X, y = data[parents].to_numpy(), data[child].to_numpy()
            flex = r2(KNeighborsRegressor(n_neighbors=25), X, y)
            flexible.append(flex)
            linear_share.append(r2(LinearRegression(), X, y) / flex if flex > 0.05 else np.nan)
            corr += [spearman.loc[p, child] for p in parents]
    return {'flexible R2': np.nanmedian(flexible), 'linear share': np.nanmedian(linear_share),
            '|corr| on edges': np.median(corr)}


if __name__ == '__main__':
    X_train, y_train = pd.read_pickle(ADIA / 'X_train.pickle'), pd.read_pickle(ADIA / 'y_train.pickle')
    keys = sorted(random.Random(SEED).sample(sorted(X_train), N_GRAPHS))
    adia, graphs = {k: X_train[k] for k in keys}, {k: y_train[k] for k in keys}

    rows = {'ADIA': statistics(adia, graphs)}
    for label, lengthscale, noise, bias in SETTINGS:
        generator.LENGTHSCALE, generator.NOISE_SCALE, generator.NN_BIAS_SCALE = lengthscale, noise, bias
        ours = {k: regenerate(adia[k], graphs[k], seed) for seed, k in enumerate(keys)}
        rows[label] = statistics(ours, graphs)
        print(f"{label}: {rows[label]}")

    table = pd.DataFrame(rows).T
    target = table.loc['ADIA']
    table['distance to ADIA'] = ((table - target).abs() / target).sum(axis=1)   # relative gaps, summed
    OUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT)
    print(f"\n{table.round(3).to_string()}\n\nclosest to ADIA: {table.drop('ADIA')['distance to ADIA'].idxmin()}")