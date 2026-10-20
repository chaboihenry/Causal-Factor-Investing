"""
Test datasets: literature graph x 100 draws + 80 dial graphs x 5 draws (= 500, 'main'); S1 (main's literature data
with leverage hidden; 100); S2 (no direct X -> Y; 100); S3 (main's 500 again, same seeds, made realistic: fat tails,
raw values, regime break). Columns renamed the ADIA way (X, Y, 0-7). Saves data/synthetic/b3_worlds/<set>/{X,graphs}.pkl
and results/exp04/b3_worlds/datasets_meta.csv (id, world, agreement, seed, true beta, column names, regime betas).
"""

import json
import sys
from pathlib import Path

import networkx as nx
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import datasets
from common.paths import GRAPHS, RESULTS, SYNTHETIC
from generator import generate

OUT = SYNTHETIC / 'b3_worlds'
META = RESULTS / 'b3_worlds' / 'datasets_meta.csv'
DRAWS = {'lit': 100, 'dial': 5, 's1': 100, 's2': 100}            # S3 mirrors main
HIDDEN = 'Leverage'                 # S1: the confounder that is generated, then removed


def adia_names(names, x, y):
    """X and Y keep their roles; the other variables become '0'-'7' in alphabetical order (same map every time)."""
    others = sorted(n for n in names if n not in (x, y))
    return {x: 'X', y: 'Y', **{n: str(i) for i, n in enumerate(others)}}


def build(world, edges, nodes, x, y, draws, seed0, hide=None, realistic=False, prefix=''):
    """draws datasets from one graph; returns {id: data}, {id: adjacency}, meta rows."""
    graph = nx.DiGraph([tuple(e) for e in edges])
    graph.add_nodes_from(nodes)
    kept = [n for n in nodes if n != hide]
    names = adia_names(kept, x, y)
    adj = nx.to_pandas_adjacency(graph.subgraph(kept), nodelist=kept, dtype=int).rename(index=names, columns=names)
    order = ['X', 'Y'] + sorted((names[n] for n in kept if n not in (x, y)), key=int)
    frames, graphs, meta = {}, {}, []
    for d in range(draws):
        seed, key = seed0 + d, f"{prefix}{world['id']}_{d:03d}"
        data, beta, info = generate(graph, x, y, seed, realistic=realistic)
        frames[key] = data[kept].rename(columns=names)[order]
        graphs[key] = adj.loc[order, order]
        meta.append({'id': key, 'world': world['id'], 'agreement': world['agreement'], 'seed': seed,
                     'true_beta': beta, **{f"col_{v}": k for k, v in names.items()},
                     'beta_regime1': info.get('beta_regime1'), 'beta_regime2': info.get('beta_regime2'),
                     'regime_changed': ' '.join(info.get('changed', []))})
    return frames, graphs, meta


if __name__ == '__main__':
    dial = json.loads((GRAPHS / 'finance_dial.json').read_text(encoding='utf-8'))
    nodes = [v['name'] for v in json.loads((GRAPHS / 'finance_variables.json').read_text(encoding='utf-8'))['variables']]
    x, y, worlds = dial['x'], dial['y'], dial['worlds']
    lit = worlds[0]

    main = [(w, w['edges'], DRAWS['lit'] if w['id'] == 'lit' else DRAWS['dial'], 1000 * i, None)
            for i, w in enumerate(worlds)]
    plan = {  # set -> list of (world, edges, draws, first seed, hidden variable[, realistic, id prefix])
        'main': main,
        's1': [({**lit, 'id': 's1'}, lit['edges'], DRAWS['s1'], 0, HIDDEN)],      # same seeds: S1 = lit data minus leverage
        's2': [({**lit, 'id': 's2'}, [e for e in lit['edges'] if e != [x, y]], DRAWS['s2'], 0, None)],
        's3': [job + (True, 's3_') for job in main],                 # same worlds and seeds as main: paired
    }
    all_meta = []
    for name, jobs in plan.items():
        frames, graphs = {}, {}
        for job in jobs:
            f, g, m = build(*job[:2], nodes, x, y, *job[2:])
            frames |= f
            graphs |= g
            all_meta += [{'set': name, **row} for row in m]
        datasets.save(OUT / name / 'X.pkl', frames)
        datasets.save(OUT / name / 'graphs.pkl', graphs)
        print(f"{name}: {len(frames)} datasets -> {OUT / name}")

    meta = pd.DataFrame(all_meta)
    META.parent.mkdir(parents=True, exist_ok=True)
    meta.to_csv(META, index=False)
    print(meta.groupby(['set', 'agreement'])['true_beta'].agg(['count', 'mean', 'std', 'min', 'max']).round(3))