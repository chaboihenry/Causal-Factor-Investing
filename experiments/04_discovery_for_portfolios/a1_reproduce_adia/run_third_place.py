"""
Reproduce the ADIA 3rd-place model (Hong & Gu, 74.06%) on the 1,880 reduced-test datasets.
Their pipeline: edge-relabel augmentation, 25 feature sets, 3 XGBoost models, soft vote.
Features are cached in chunks under models/third_place/<mode>/ so a crash resumes.
Modes: pilot (1,000 train, no augmentation), full (23,500 train, augment 3), smoke (tiny, CPU).
Saves roles and score to results/exp04/a1_reproduce_adia/third_place_<mode>_*.csv. Target: ~74%.
"""

import argparse
import random
import sys
import time
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import ADIA, EXTERNAL, MODELS, RESULTS
from common.scoring import roles_from_adjacency, score

sys.path.insert(0, str(EXTERNAL))
import third_place_features as tp

MODES = {
    # n_train=None means all; tree_scale shrinks the XGBoost models (smoke test only)
    'pilot': dict(n_train=1000, n_test=None, augment=0.0, tree_scale=1.0),
    'full': dict(n_train=None, n_test=None, augment=3.0, tree_scale=1.0),
    'smoke': dict(n_train=40, n_test=40, augment=1.0, tree_scale=0.01),
}
FEATURES = [
    'pearson_correlation', 'rolling_abs_pearson_correlation', 'spearman_correlation', 'kendall_correlation',
    'mutual_information', 'conditional_mutual_information', 'distance_correlation', 'partial_correlation',
    'copula_entropy', 'PPS_feature', 'linear_regression_feature', 'ridge_regression_feature',
    'ExactSearch_feature', 'PC_feature', 'FCI_feature', 'GRaSP_feature', 'DML_feature',
    'piecewise_linear_regression_feature', 'piecewise_quadratic_regression_feature',
    'piecewise_quadratic_regression_feature_4_improved', 'sliding_window_linear_regression_feature',
    'sem_features', 'grid_feature', 'discrete_statistic_features', 'Cloud_feature',
]
CLASSES = np.array(['Confounder', 'Collider', 'Mediator', 'Independent',
                    'Cause of X', 'Consequence of X', 'Cause of Y', 'Consequence of Y'])
BLACKLIST = ['ttest(v,X)', 'pvalue(ttest(v,X))<=0.05', 'ttest(v,Y)', 'pvalue(ttest(v,Y))<=0.05',
             'ttest(X,Y)', 'pvalue(ttest(X,Y))<=0.05', 'square_dimension', 'max(PPS(v,others))',
             'TLI_Collider', 'TLI_Confounder', 'RMSEA_Collider', 'RMSEA_Confounder']
XGB_PARAMS = dict(learning_rate=0.05, objective='multi:softmax', num_class=8, eval_metric='mlogloss',
                  max_depth=6, subsample=0.7, colsample_bytree=0.7, min_child_weight=1, gamma=0.3,
                  reg_lambda=18.0, reg_alpha=18.0, max_delta_step=0.5, tree_method='hist', verbosity=0)
ENSEMBLE = [(7000, 42), (7500, 77), (8000, 2)]    # (n_estimators, random_state) of their 3 models
CHUNK = 500
SEED = 0


def features(split, datasets, graphs, cache):
    """Their create_all_columns, run chunk by chunk; label first for train, as in their train()."""
    keys = list(datasets)
    parts = []
    for i in range(0, len(keys), CHUNK):
        path = cache / f"{split}_{i // CHUNK:04d}.pkl"
        if not path.exists():
            chunk = {k: datasets[k] for k in keys[i:i + CHUNK]}
            funcs = {tp.label: {k: graphs[k] for k in chunk}} if graphs else {}
            funcs |= {getattr(tp, name): chunk for name in FEATURES}
            t0 = time.time()
            tp.create_all_columns(funcs, n_jobs=-1, create_dimension_feature=True).to_pickle(path)
            print(f"{split} chunk {i // CHUNK + 1}/{-(-len(keys) // CHUNK)}: {time.time() - t0:.0f}s")
        parts.append(pd.read_pickle(path))
    df = pd.concat(parts, ignore_index=True)
    for col in ['MI(v,X)', 'MI(v,Y)', 'MI(X,Y)', 'max(MI(v, others))', 'min(MI(v, others))']:
        df[f'{col}^2'] = df[col] ** 2
    return df.loc[:, ~df.columns.duplicated()]


def preprocess(df):
    """Their cleaning: drop blacklist, inf -> NaN -> column mean of this split, clean names, encode categoricals."""
    df = df.drop(columns=[c for c in BLACKLIST if c in df.columns]).replace([np.inf, -np.inf], np.nan)
    num = df.select_dtypes(include=[np.number]).columns
    df[num] = df[num].fillna(df[num].mean())
    df = tp.clean_feature_names(df)
    X = df.drop(columns=[c for c in ['dataset', 'variable', 'label'] if c in df.columns])
    _, _, X = tp.process_categorical_features(X)
    return df, X


def train_ensemble(X, y, weights, scale, device, out_dir):
    models = []
    for n_trees, seed in ENSEMBLE:
        stop = xgb.callback.EarlyStopping(rounds=20, save_best=True, maximize=False, metric_name='mlogloss')
        model = xgb.XGBClassifier(n_estimators=max(10, int(n_trees * scale)), random_state=seed,
                                  device=device, callbacks=[stop], **XGB_PARAMS)
        t0 = time.time()
        model.fit(X, y, sample_weight=weights, eval_set=[(X, y)], verbose=500)
        model.save_model(out_dir / f"xgb_{seed}.json")
        print(f"xgb seed {seed}: {model.best_iteration + 1} trees, {time.time() - t0:.0f}s")
        models.append(model)
    return models


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=MODES)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    cfg = MODES[args.mode]
    warnings.filterwarnings('ignore')
    random.seed(SEED)
    np.random.seed(SEED)
    cache = MODELS / 'third_place' / args.mode
    cache.mkdir(parents=True, exist_ok=True)
    start = time.time()

    X_train, y_train = pd.read_pickle(ADIA / 'X_train.pickle'), pd.read_pickle(ADIA / 'y_train.pickle')
    X_test, y_test = pd.read_pickle(ADIA / 'X_test_reduced.pickle'), pd.read_pickle(ADIA / 'y_test_reduced.pickle')
    train_keys = sorted(X_train) if cfg['n_train'] is None else sorted(random.sample(sorted(X_train), cfg['n_train']))
    test_keys = sorted(X_test)[:cfg['n_test']]
    X_train, y_train = {k: X_train[k] for k in train_keys}, {k: y_train[k] for k in train_keys}
    X_test = {k: X_test[k] for k in test_keys}

    if cfg['augment'] > 0:
        X_train, y_train, failures = tp.augment_data(X_train, y_train, augment_factor=cfg['augment'])
        print(f"augmented: {len(train_keys)} -> {len(X_train)} datasets, failures by size: {failures}")

    train_df, X_tr = preprocess(features('train', X_train, y_train, cache))
    y_tr = pd.Series(pd.Categorical(train_df['label'], categories=CLASSES).codes, index=train_df.index)
    raw = train_df['dataset'].isin(train_keys)                 # class weights from un-augmented rows only
    class_w = dict(zip(range(8), compute_class_weight('balanced', classes=np.arange(8), y=y_tr[raw])))
    models = train_ensemble(X_tr, y_tr, y_tr.map(class_w), cfg['tree_scale'], args.device, cache)

    test_df, X_te = preprocess(features('test', X_test, None, cache))
    X_te = tp.align_features(tp.filter_features(X_te, models[0]), models[0])
    proba = np.mean([m.predict_proba(X_te) for m in models], axis=0)
    test_df['pred'] = CLASSES[proba.argmax(axis=1)]

    truth = pd.DataFrame([{'dataset': k, 'variable': str(v), 'true': r}
                          for k in test_keys for v, r in roles_from_adjacency(y_test[k]).items()])
    roles_df = truth.merge(test_df[['dataset', 'variable', 'pred']].astype({'variable': str}),
                           on=['dataset', 'variable'], how='left')
    if roles_df['pred'].isna().any():
        sys.exit(f"{roles_df['pred'].isna().sum()} test variables have no prediction")
    overall, per_role = score(roles_df)

    out = RESULTS / 'a1_reproduce_adia'
    out.mkdir(parents=True, exist_ok=True)
    roles_df.to_csv(out / f"third_place_{args.mode}_roles.csv", index=False)
    per_role.loc['ALL (balanced accuracy)'] = [overall, len(roles_df)]
    per_role.to_csv(out / f"third_place_{args.mode}_score.csv")
    print(per_role.round(4))
    print(f"{args.mode}: balanced accuracy {overall:.4f} on {len(test_keys)} test datasets "
          f"(target ~0.74), total {(time.time() - start) / 60:.0f} min")