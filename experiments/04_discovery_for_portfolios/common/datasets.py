"""
Save and load collections of datasets or graphs ({id: DataFrame}, ADIA's shape) as plain Python lists in a
pickle, with no numpy or pandas objects inside, so files written in one conda env load in the other.
"""

import pickle

import pandas as pd


def save(path, frames):
    path.parent.mkdir(parents=True, exist_ok=True)
    plain = {k: {'columns': f.columns.tolist(), 'index': f.index.tolist(), 'values': f.to_numpy().tolist()}
             for k, f in frames.items()}
    path.write_bytes(pickle.dumps(plain, protocol=4))


def load(path):
    plain = pickle.loads(path.read_bytes())
    return {k: pd.DataFrame(p['values'], index=p['index'], columns=p['columns']) for k, p in plain.items()}