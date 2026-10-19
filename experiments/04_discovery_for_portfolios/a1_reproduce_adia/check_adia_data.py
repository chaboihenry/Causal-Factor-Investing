"""
Check the ADIA challenge data against the paper before any modelling.

Loads the train and reduced-test pickles from data/adia. For every dataset,
records its size, its value range, and whether its graph matches the paper:
X and Y present, X -> Y present, no cycles, and the same variable names in the
data and the adjacency matrix (rows = parents). A split whose graph file is not
a dict of adjacency matrices is reported and skipped, not guessed at.
Prints one summary per split and saves the per-dataset checks to
results/exp04/a1_reproduce_adia/adia_data_check.csv.
"""

import sys
import networkx as nx
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import ADIA, RESULTS

OUT = RESULTS / 'a1_reproduce_adia' / 'adia_data_check.csv'

SPLITS = [
    # (split name, data file, graph file)
    ('train', 'X_train.pickle', 'y_train.pickle'),
    ('test', 'X_test_reduced.pickle', 'y_test_reduced.pickle'),
]

NUM_CHECKS = ['n_rows', 'n_vars', 'min_value', 'max_value']
BOOL_CHECKS = ['has_X_and_Y', 'x_causes_y', 'is_dag', 'names_match']


def describe(obj):
    """Type of the object and, for a dict, the type of its first value."""
    if isinstance(obj, dict):
        first = next(iter(obj.values()))
        return f"dict of {len(obj)} x {type(first).__name__}"
    return f"{type(obj).__name__} {getattr(obj, 'shape', '')}"


def is_graph_dict(obj):
    return isinstance(obj, dict) and isinstance(next(iter(obj.values())), pd.DataFrame)


def check_split(split, X_dict, y_dict):
    same_ids = set(X_dict) == set(y_dict)
    print(f"{split}: {len(X_dict)} datasets, same IDs in X and y: {same_ids}")
    rows = []
    for dataset_id, data in X_dict.items():
        adj = y_dict[dataset_id]
        graph = nx.from_pandas_adjacency(adj, create_using=nx.DiGraph)  # edge row -> column
        cols = set(data.columns)
        rows.append({
            'split': split,
            'dataset_id': dataset_id,
            'n_rows': len(data),
            'n_vars': data.shape[1],
            'min_value': data.to_numpy().min(),
            'max_value': data.to_numpy().max(),
            'has_X_and_Y': {'X', 'Y'} <= cols,
            'x_causes_y': 'X' in adj.index and 'Y' in adj.columns and adj.loc['X', 'Y'] == 1,
            'is_dag': nx.is_directed_acyclic_graph(graph),
            'names_match': cols == set(adj.index) == set(adj.columns),
        })
    return pd.DataFrame(rows)


raw = {split: (pd.read_pickle(ADIA / x_file), pd.read_pickle(ADIA / y_file))
       for split, x_file, y_file in SPLITS}

for split, (X_obj, y_obj) in raw.items():
    print(f"{split}: X is {describe(X_obj)}, y is {describe(y_obj)}")

checkable = {split: pair for split, pair in raw.items() if is_graph_dict(pair[1])}
for split in raw.keys() - checkable.keys():
    print(f"{split}: graph file is not a dict of adjacency matrices, skipped")

checks_df = pd.concat([check_split(split, X_dict, y_dict) for split, (X_dict, y_dict) in checkable.items()],
                      ignore_index=True)

print(checks_df.groupby('split')[NUM_CHECKS].agg(['min', 'mean', 'max']).T.round(3))
print(checks_df.groupby('split')[BOOL_CHECKS].all())
print(checks_df.groupby('split')['n_vars'].value_counts().sort_index())

failures = checks_df[~checks_df[BOOL_CHECKS].all(axis=1)]
print(f"datasets failing a check: {len(failures)}")
if len(failures):
    print(failures.head())

OUT.parent.mkdir(parents=True, exist_ok=True)
checks_df.to_csv(OUT, index=False)
print(f"saved {OUT}")

