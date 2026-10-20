"""
ADIA-style data generator (Olivetti et al. 2026, Appendix C): each node is a random function of its parents
(neural net, heteroskedastic 'Bayesian' net or Gaussian process) plus Gaussian noise, then rank-uniformized to
[-0.999, 0.999]. Also returns the true beta of Y on X: the slope from a 20,000-row randomized experiment in the same world.
Self-check on the finance graph: ranges +-0.999, same seed -> same data, true vs. regression betas for 8 seeds.
"""

import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import GRAPHS

N_ROWS = 1000
FUNCTIONS = ('nn', 'bayes', 'gp')    # Appendix C.4; equal odds (our choice, not stated in the paper)
HIDDEN_PER_INPUT = 5                 # C.4.1: hidden layer = 5 x number of parents
N_FOURIER = 200                      # size of the Gaussian-process approximation
LENGTHSCALE = (4.0, 16.0)            # GP smoothness range: not in the paper, calibrated to ADIA's data
NOISE_SCALE = 0.6                    # multiplies the noise sd drawn from [0, 1]: calibrated to ADIA's data
NN_BIAS_SCALE = 1.0                  # spread of the nets' hidden biases; larger = fewer kinks (calibrated: 1.0)
N_TRUTH = 20_000                     # rows in the randomized experiment that defines the true beta


def rank_grid(n):
    """The n evenly spaced values that rank-uniformizing produces: 1,000 rows -> -0.999, ..., 0.999."""
    return (np.arange(n) + 0.5) / n * 2 - 1


def random_net(rng, k):
    """C.4.1: x -> B relu(Ax + a) + b with random weights, hidden size 5k."""
    h = HIDDEN_PER_INPUT * k
    A, a, B, b = rng.normal(size=(h, k)), rng.normal(scale=NN_BIAS_SCALE, size=h), rng.normal(size=h), rng.normal()
    return lambda x: np.maximum(x @ A.T + a, 0) @ B + b


def random_gp(rng, k):
    """C.4.3: a random smooth function with a Matern 3/2 kernel, drawn with random Fourier features
    (the kernel's spectrum is a Student-t with 3 degrees of freedom), so it can be evaluated anywhere."""
    ell = rng.uniform(*LENGTHSCALE)
    freq = rng.normal(size=(N_FOURIER, k)) / np.sqrt(rng.chisquare(3, size=(N_FOURIER, 1)) / 3) / ell
    phase, weight = rng.uniform(0, 2 * np.pi, N_FOURIER), rng.normal(size=N_FOURIER)
    return lambda x: np.sqrt(2 / N_FOURIER) * np.cos(x @ freq.T + phase) @ weight


def random_node(rng, parents):
    """Draw one node's function type, function and noise size (C.5: noise sd uniform in [0, 1])."""
    kind = 'root' if not parents else FUNCTIONS[rng.integers(len(FUNCTIONS))]
    node = {'parents': parents, 'kind': kind, 'noise_sd': NOISE_SCALE * rng.uniform(0, 1)}
    if kind != 'root':
        node['f'] = (random_gp if kind == 'gp' else random_net)(rng, len(parents))
    if kind == 'bayes':
        node['g'] = random_net(rng, len(parents))     # C.4.2: a second net sets the noise size (heteroskedastic)
    return node


def random_scm(graph, rng):
    order = list(nx.lexicographical_topological_sort(graph))
    return {'order': order, 'nodes': {v: random_node(rng, sorted(graph.predecessors(v))) for v in order}}


def standardize(values, key, frozen, record):
    if record:
        frozen[key] = (values.mean(), values.std() or 1.0)
    mean, std = frozen[key]
    return (values - mean) / std


def simulate(scm, eps, frozen=None, setting=None):
    """Every node in causal order. The first call (frozen=None) records each node's scaling and rank map; later
    calls reuse them, so an intervention changes only what it causes. setting=(node, values) forces that node."""
    record, frozen, data = frozen is None, frozen or {}, {}
    for v in scm['order']:
        node = scm['nodes'][v]
        if setting and setting[0] == v:
            data[v] = setting[1]
            continue
        if node['kind'] == 'root':
            raw = eps[v]
        else:
            x = np.column_stack([data[p] for p in node['parents']])
            sd = node['noise_sd']
            if node['kind'] == 'bayes':
                sd = sd * np.exp(0.5 * standardize(node['g'](x), (v, 'g'), frozen, record))
            raw = standardize(node['f'](x), (v, 'f'), frozen, record) + sd * eps[v]
        if record:
            frozen[(v, 'rank')] = np.sort(raw)
        ranked = frozen[(v, 'rank')]
        data[v] = np.interp(raw, ranked, rank_grid(len(ranked)))     # C.4.4: uniformize every node
    return data, frozen


def true_beta(scm, frozen, x, y, rng):
    """The beta a randomized experiment would find: x assigned at random (same uniform spread), everything
    downstream recomputed with fresh noise, then the slope of y on x. This is what an adjusted regression targets."""
    eps = {v: rng.standard_normal(N_TRUTH) for v in scm['order']}
    data, _ = simulate(scm, eps, frozen, setting=(x, rng.uniform(-0.999, 0.999, N_TRUTH)))
    return slope(pd.DataFrame(data), x, y)


def generate(graph, x, y, seed, n_rows=N_ROWS):
    """One dataset from one seed: (data with columns in alphabetical order, true beta, function type per node)."""
    rng = np.random.default_rng(seed)
    scm = random_scm(graph, rng)
    eps = {v: rng.standard_normal(n_rows) for v in scm['order']}
    data, frozen = simulate(scm, eps)
    kinds = {v: scm['nodes'][v]['kind'] for v in scm['order']}
    return pd.DataFrame(data)[sorted(graph.nodes)], true_beta(scm, frozen, x, y, rng), kinds


def load_graph(name):
    truth = json.loads((GRAPHS / f"{name}_truth.json").read_text(encoding='utf-8'))
    names = [v['name'] for v in json.loads((GRAPHS / f"{name}_variables.json").read_text(encoding='utf-8'))['variables']]
    graph = nx.DiGraph(truth['edges'])
    graph.add_nodes_from(names)
    return graph, truth['x'], truth['y']


def slope(data, x, y, controls=()):
    design = np.column_stack([np.ones(len(data)), data[x]] + [data[c] for c in controls])
    return float(np.linalg.lstsq(design, data[y], rcond=None)[0][1])


if __name__ == '__main__':
    graph, x, y = load_graph('finance')
    data, effect, kinds = generate(graph, x, y, seed=0)
    again, _, _ = generate(graph, x, y, seed=0)
    other, _, _ = generate(graph, x, y, seed=1)
    print(f"range: {data.min().min():.3f} to {data.max().max():.3f}; "
          f"column sd: {data.std().min():.3f}-{data.std().max():.3f}")
    print(f"same seed identical: {data.equals(again)}; different seed differs: {not data.equals(other)}")
    print(f"function types (seed 0): {kinds}")
    others = [v for v in sorted(graph.nodes) if v not in (x, y)]
    print(f"\n{'seed':>4} {'true beta':>9} {'X alone':>8} {'all controls':>13} {'confounder only (oracle)':>25}")
    for seed in range(8):
        data, effect, _ = generate(graph, x, y, seed)
        print(f"{seed:>4} {effect:>9.3f} {slope(data, x, y):>8.3f} {slope(data, x, y, others):>13.3f} "
              f"{slope(data, x, y, ['Leverage']):>25.3f}")