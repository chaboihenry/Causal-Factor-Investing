"""
Predict roles with the trained 3rd-place ensemble (models/third_place/full/xgb_*.json): no retraining (env adia-3rd).
Usage: predict_third_place.py <name> <data.pkl> [--graphs graphs.pkl] [--out b3_worlds]. Features are cached in
models/third_place/predict/<name>/. Saves every variable's predicted role and the 8 role probabilities (for the
hybrid arm) to results/exp04/<out>/third_place_<name>_roles.csv; with --graphs, also scores them (_score.csv).
"""

import argparse
import hashlib
import random
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import datasets
from common.paths import MODELS, RESULTS
from common.scoring import roles_from_adjacency, score
from run_third_place import CLASSES, ENSEMBLE, SEED, features, preprocess, tp

TRAINED = MODELS / 'third_place' / 'full'


def load_models():
    models = []
    for _, seed in ENSEMBLE:
        model = xgb.XGBClassifier()
        model.load_model(TRAINED / f"xgb_{seed}.json")
        models.append(model)
    return models


def fingerprint(data):
    h = hashlib.sha256()
    for k in data:
        h.update(str(k).encode() + data[k].to_numpy().tobytes())
    return h.hexdigest()


def predict(name, data, models):
    cache = MODELS / 'third_place' / 'predict' / name
    cache.mkdir(parents=True, exist_ok=True)
    stamp, digest = cache / 'data_hash.txt', fingerprint(data)
    if stamp.exists() and stamp.read_text() != digest:            # cached feature chunks belong to other data
        sys.exit(f"{cache} holds features for different data: delete that folder or use another name")
    stamp.write_text(digest)
    test_df, X_te = preprocess(features('test', data, None, cache))
    X_te = tp.align_features(tp.filter_features(X_te, models[0]), models[0])
    proba = np.mean([m.predict_proba(X_te) for m in models], axis=0)
    out = test_df[['dataset', 'variable']].astype({'variable': str}).copy()
    out['pred'] = CLASSES[proba.argmax(axis=1)]
    for i, role in enumerate(CLASSES):
        out[f"p_{role}"] = proba[:, i]
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('name')
    parser.add_argument('data', type=Path)
    parser.add_argument('--graphs', type=Path)
    parser.add_argument('--out', default='b3_worlds')
    args = parser.parse_args()
    warnings.filterwarnings('ignore')
    random.seed(SEED)
    np.random.seed(SEED)

    data = datasets.load(args.data)
    preds = predict(args.name, data, load_models())
    out = RESULTS / args.out
    out.mkdir(parents=True, exist_ok=True)

    if args.graphs:
        graphs = datasets.load(args.graphs)
        truth = [{'dataset': k, 'variable': str(v), 'true': r}
                 for k in sorted(data) for v, r in roles_from_adjacency(graphs[k]).items()]
        preds = preds.merge(pd.DataFrame(truth), on=['dataset', 'variable'], how='right')
        if preds['pred'].isna().any():
            sys.exit(f"{preds['pred'].isna().sum()} variables have no prediction")
        overall, per_role = score(preds)
        per_role.loc['ALL (balanced accuracy)'] = [overall, len(preds)]
        per_role.to_csv(out / f"third_place_{args.name}_score.csv")
        print(per_role.round(4))
        print(f"{args.name}: balanced accuracy {overall:.4f} on {len(data)} datasets")
    preds.to_csv(out / f"third_place_{args.name}_roles.csv", index=False)
    print(f"saved {out / f'third_place_{args.name}_roles.csv'}")