"""
Every arm's predicted role for every variable in every test dataset, next to the true role, in one long table:
B1, C1 (PC), and per LLM run A1 (its graph), A2 (its graph after refutation on that dataset) and A3 (A1's roles,
overruled by B1 where B1's top role differs and its probability >= τ; τ = 0.70, plus 0.5 and 0.9 as sensitivity).
Saves results/exp04/b5_runs/roles_all.csv.gz; scores nothing (that's Phase 6). Roles use ADIA's rule, but are also
defined without X -> Y (S2, S2b); a pattern outside the 8 roles is labelled 'Other'.
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import datasets
from common.paths import RESPONSES, RESULTS, SYNTHETIC
from common.scoring import ROLES

X, Y = 'BookToMarket', 'AnnualReturn'
SETS = ['main', 's1', 's2', 's2b', 's3']
RUNS = {'gpt-oss-exp04': range(1, 11), 'claude-opus-5.5-medium': range(1, 4)}
TAUS = {'A3': 0.70, 'A3_tau0.5': 0.5, 'A3_tau0.9': 0.9}
B5 = RESULTS / 'b5_runs'
META = RESULTS / 'b3_worlds' / 'datasets_meta.csv'


def roles(edges, variables):
    """ADIA's role for each variable from its direct edges with X and Y; works with or without X -> Y."""
    e = set(edges)
    return {v: ROLES.get(((X, v) in e) << 3 | ((Y, v) in e) << 2 | ((v, Y) in e) << 1 | ((v, X) in e), 'Other')
            for v in variables}


def names_of(meta_row):
    """ADIA column name -> finance name for one dataset."""
    return {col[4:]: name for col, name in meta_row.items() if col.startswith('col_') and isinstance(name, str)}


def rows_for(arm, model, run, predicted, base):
    """predicted: {(dataset, variable): role}; base: one row per (set, dataset, variable) with the true role."""
    return base.assign(arm=arm, model=model, run=run, pred=[predicted[(d, v)] for d, v in zip(base.dataset, base.variable)])


if __name__ == '__main__':
    meta = pd.read_csv(META).set_index('id')
    names = {k: names_of(meta.loc[k]) for k in meta.index}

    base, variables, true_graphs = [], {}, {}
    for s in SETS:
        for key, adj in sorted(datasets.load(SYNTHETIC / 'b3_worlds' / s / 'graphs.pkl').items()):
            n = names[key]
            edges = [(n[a], n[b]) for a in adj.index for b in adj.columns if adj.loc[a, b]]
            variables[key] = [n[c] for c in adj.columns if n[c] not in (X, Y)]
            base += [{'set': s, 'dataset': key, 'agreement': meta.loc[key, 'agreement'], 'variable': v, 'true': r}
                     for v, r in roles(edges, variables[key]).items()]
    base = pd.DataFrame(base)
    tables = []

    b1 = pd.concat([pd.read_csv(B5 / f"third_place_b1_{s}_roles.csv", dtype={'variable': str}) for s in SETS])
    b1['variable'] = [names[d][v] for d, v in zip(b1.dataset, b1.variable)]
    b1['confidence'] = b1.filter(like='p_').max(axis=1)
    b1 = b1.set_index(['dataset', 'variable'])
    tables.append(rows_for('B1', '', 0, b1['pred'].to_dict(), base))

    pc = pd.read_csv(B5 / 'pc_edges.csv.gz')
    pc_edges = pc.groupby('dataset').apply(lambda g: list(zip(g.source, g.target))).to_dict()
    tables.append(rows_for('C1', '', 0, {(d, v): r for d in variables
                                           for v, r in roles(pc_edges.get(d, []), variables[d]).items()}, base))

    for model, runs in RUNS.items():
        for run in runs:
            final = json.loads((RESPONSES / 'finance' / model / f"run{run:02d}" / 'final_dag.json').read_text())
            a1_edges = [(e['source'], e['target']) for e in final['edges']]
            a1 = {(d, v): r for d in variables for v, r in roles(a1_edges, variables[d]).items()}
            tables.append(rows_for('A1', model, run, a1, base))

            a2 = pd.read_csv(B5 / 'a2' / f"{model}_run{run:02d}.csv.gz")
            kept = a2[a2.kept].groupby('dataset').apply(lambda g: list(zip(g.source, g.target))).to_dict()
            tables.append(rows_for('A2', model, run, {(d, v): r for d in variables
                                                        for v, r in roles(kept.get(d, []), variables[d]).items()}, base))

            for arm, tau in TAUS.items():
                a3 = {k: (b1.at[k, 'pred'] if b1.at[k, 'pred'] != r and b1.at[k, 'confidence'] >= tau else r)
                      for k, r in a1.items()}
                tables.append(rows_for(arm, model, run, a3, base))

    table = pd.concat(tables, ignore_index=True)
    table.to_csv(B5 / 'roles_all.csv.gz', index=False)
    print(table.groupby(['arm', 'set']).size().unstack().to_string())
    print(f"saved {B5 / 'roles_all.csv.gz'}: {len(table):,} rows; 'Other' predictions: {(table.pred == 'Other').sum():,}")