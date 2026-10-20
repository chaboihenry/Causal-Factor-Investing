"""
The agreement dial: 20 altered copies of the finance graph at each of 75 / 50 / 25 / 0% agreement (= share of the
12 literature edges, all but X -> Y, left untouched). Each altered edge is removed, reversed or swapped for a new
edge with equal odds; reversals and new edges must respect the variables' timing and keep the graph acyclic.
Saves graphs/finance_dial.json (every graph with what changed) and prints edge counts and roles changed per level.
"""

import json
import random
import sys
from pathlib import Path

import networkx as nx
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import GRAPHS
from common.scoring import roles_from_adjacency

OUT = GRAPHS / 'finance_dial.json'
LEVELS = {75: 3, 50: 6, 25: 9, 0: 12}      # agreement % -> number of literature edges altered
PER_LEVEL = 20
SEED = 0
TIME = {  # when each variable is fixed (design.md section 6); an edge may only point forward or within a period
    'FiscalYearEndMonth': -1, 'PastFiveYearReturn': 0, 'Leverage': 1, 'BookToMarket': 1, 'ValueIndexMember': 1,
    'EarningsSurprise': 2, 'MarketReturn': 2, 'AnnualReturn': 2, 'QualityScore': 3, 'InstOwnershipChange': 4,
}


def allowed(graph, a, b):
    """a -> b can be added: timing allows it, it's new in both directions, and no cycle appears."""
    if a == b or TIME[a] > TIME[b] or graph.has_edge(a, b) or graph.has_edge(b, a):
        return False
    return not nx.has_path(graph, b, a)


def swap_in(graph, rng, literature):
    """Add one new edge that the literature graph doesn't have, chosen at random among those allowed."""
    options = [(a, b) for a in sorted(TIME) for b in sorted(TIME)
               if (a, b) not in literature and allowed(graph, a, b)]
    edge = options[rng.randrange(len(options))]
    graph.add_edge(*edge)
    return edge


def alter(literature, x, y, n_changes, rng):
    graph = nx.DiGraph(sorted(literature))
    graph.add_nodes_from(TIME)
    changes = []
    for a, b in rng.sample(sorted(e for e in literature if e != (x, y)), n_changes):
        action = rng.choice(['remove', 'reverse', 'swap'])
        graph.remove_edge(a, b)
        if action == 'reverse' and allowed(graph, b, a):
            graph.add_edge(b, a)
            changes.append({'edge': [a, b], 'action': 'reverse'})
        elif action == 'remove':
            changes.append({'edge': [a, b], 'action': 'remove'})
        else:                                   # swap, or a reversal the timing or a cycle doesn't allow
            changes.append({'edge': [a, b], 'action': 'swap', 'new': list(swap_in(graph, rng, literature))})
    return sorted(graph.edges), changes


def roles(edges, x, y):
    nodes = sorted(TIME)
    adj = pd.DataFrame(0, index=nodes, columns=nodes)
    for a, b in edges:
        adj.loc[a, b] = 1
    return roles_from_adjacency(adj.rename(index={x: 'X', y: 'Y'}, columns={x: 'X', y: 'Y'}))


if __name__ == '__main__':
    truth = json.loads((GRAPHS / 'finance_truth.json').read_text(encoding='utf-8'))
    x, y, literature = truth['x'], truth['y'], {tuple(e) for e in truth['edges']}
    base_roles = roles(literature, x, y)
    rng = random.Random(SEED)

    worlds = [{'id': 'lit', 'agreement': 100, 'edges': sorted(map(list, literature)), 'changes': []}]
    for level, n_changes in LEVELS.items():
        seen = set()
        while len(seen) < PER_LEVEL:
            edges, changes = alter(literature, x, y, n_changes, rng)
            if tuple(edges) in seen:
                continue
            seen.add(tuple(edges))
            worlds.append({'id': f"a{level:02d}_{len(seen):02d}", 'agreement': level,
                           'edges': [list(e) for e in edges], 'changes': changes})

    OUT.write_text(json.dumps({'graph': 'finance', 'x': x, 'y': y, 'seed': SEED, 'worlds': worlds}, indent=1),
                   encoding='utf-8')
    summary = pd.DataFrame([{
        'agreement': w['agreement'], 'edges': len(w['edges']),
        'roles changed': sum(r != base_roles[v] for v, r in roles(w['edges'], x, y).items()),
        **{a: sum(c['action'] == a for c in w['changes']) for a in ['remove', 'reverse', 'swap']},
    } for w in worlds])
    print(summary.groupby('agreement').agg(['mean', 'min', 'max']).round(1).sort_index(ascending=False).to_string())
    print(f"saved {len(worlds)} graphs to {OUT}")