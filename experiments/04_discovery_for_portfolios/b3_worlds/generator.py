"""
ADIA-style data generator (Olivetti et al. 2026, Appendix C): each node is a random function of its parents (neural
net, heteroskedastic 'Bayesian' net or Gaussian process) plus Gaussian noise, rank-uniformized to [-0.999, 0.999].
realistic=True (stress test S3): Student-t noise, raw standardized values, a regime break halfway through the rows.
Also returns the true beta of Y on X: the slope from a 20,000-row randomized experiment in the same world.
Self-check on the finance graph: ranges, same seed -> same data, true vs. regression betas, ADIA vs. realistic.
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
T_DF = 3                             # S3: Student-t noise, 3 degrees of freedom (fat tails, finite variance)
BREAK_SHARE = 1 / 3                  # S3: share of non-root variables whose function changes at the regime break


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


def add_regime_break(scm, rng):
    """S3: about a third of the non-root variables get a second, independently drawn function for regime 2."""
    for v in scm['order']:
        node = scm['nodes'][v]
        if node['kind'] != 'root' and rng.random() < BREAK_SHARE:
            k = len(node['parents'])
            node['f2'] = (random_gp if node['kind'] == 'gp' else random_net)(rng, k)
            if node['kind'] == 'bayes':
                node['g2'] = random_net(rng, k)


def t_noise(rng, n):
    """Student-t noise rescaled to variance 1, so only the tails differ from the Gaussian case."""
    return rng.standard_t(T_DF, n) * np.sqrt((T_DF - 2) / T_DF)


def standardize(values, key, frozen, record):
    if record:
        frozen[key] = (values.mean(), values.std() or 1.0)
    mean, std = frozen[key]
    return (values - mean) / std


def simulate(scm, eps, frozen=None, setting=None, regime=None, raw=False):
    """Every node in causal order. The first call (frozen=None) records each node's scaling and rank map; later
    calls reuse them, so an intervention changes only what it causes. setting=(node, values) forces that node;
    regime marks rows in regime 2 (S3); raw=True passes standardized values on instead of ranks (S3)."""
    record, frozen, data = frozen is None, frozen or {}, {}

    def part(node, v, name, x):
        values = standardize(node[name](x), (v, name), frozen, record)
        if regime is not None and f"{name}2" in node:
            values = np.where(regime, standardize(node[f"{name}2"](x), (v, f"{name}2"), frozen, record), values)
        return values

    for v in scm['order']:
        node = scm['nodes'][v]
        if setting and setting[0] == v:
            data[v] = setting[1]
            continue
        if node['kind'] == 'root':
            out = eps[v]
        else:
            x = np.column_stack([data[p] for p in node['parents']])
            sd = node['noise_sd']
            if node['kind'] == 'bayes':
                g = part(node, v, 'g', x)
                sd = sd * np.exp(0.5 * (np.clip(g, -3, 3) if raw else g))   # raw: cap fat-tailed inputs at 4.5x
            out = part(node, v, 'f', x) + sd * eps[v]
        if raw:
            data[v] = standardize(out, (v, 'out'), frozen, record)
            continue
        if record:
            frozen[(v, 'rank')] = np.sort(out)
        ranked = frozen[(v, 'rank')]
        data[v] = np.interp(out, ranked, rank_grid(len(ranked)))     # C.4.4: uniformize every node
    return data, frozen


def true_beta(scm, frozen, x, y, rng, realistic=False, x_pool=None, regime=None):
    """The beta a randomized experiment would find: x assigned at random (same spread as in the data), everything
    downstream recomputed with fresh noise, then the slope of y on x. This is what an adjusted regression targets."""
    noise = t_noise if realistic else (lambda r, n: r.standard_normal(n))
    eps = {v: noise(rng, N_TRUTH) for v in scm['order']}
    xs = rng.choice(x_pool, N_TRUTH) if realistic else rng.uniform(-0.999, 0.999, N_TRUTH)
    flags = None if regime is None else np.full(N_TRUTH, regime)
    data, _ = simulate(scm, eps, frozen, setting=(x, xs), regime=flags, raw=realistic)
    return slope(pd.DataFrame(data), x, y)


def generate(graph, x, y, seed, n_rows=N_ROWS, realistic=False):
    """One dataset from one seed: (data, columns in alphabetical order; true beta; info: function types and, for
    realistic data, the true beta in each regime). realistic=True reuses the same functions and Gaussian draws as
    the ADIA-style dataset with that seed, then adds fat tails and the regime break, so the two are paired."""
    rng = np.random.default_rng(seed)
    scm = random_scm(graph, rng)
    eps = {v: rng.standard_normal(n_rows) for v in scm['order']}
    info = {'kinds': {v: scm['nodes'][v]['kind'] for v in scm['order']}}
    if not realistic:
        data, frozen = simulate(scm, eps)
        return pd.DataFrame(data)[sorted(graph.nodes)], true_beta(scm, frozen, x, y, rng), info
    scale = np.sqrt((T_DF - 2) / T_DF)
    eps = {v: eps[v] / np.sqrt(rng.chisquare(T_DF, n_rows) / T_DF) * scale for v in scm['order']}   # normal -> t
    add_regime_break(scm, rng)
    regime = np.arange(n_rows) >= n_rows // 2
    data, frozen = simulate(scm, eps, regime=regime, raw=True)
    betas = [true_beta(scm, frozen, x, y, rng, True, data[x], r) for r in (False, True)]
    info |= {'beta_regime1': betas[0], 'beta_regime2': betas[1],
             'changed': sorted(v for v in scm['order'] if 'f2' in scm['nodes'][v])}
    return pd.DataFrame(data)[sorted(graph.nodes)], float(np.mean(betas)), info


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
    data, effect, info = generate(graph, x, y, seed=0)
    again, _, _ = generate(graph, x, y, seed=0)
    other, _, _ = generate(graph, x, y, seed=1)
    print(f"range: {data.min().min():.3f} to {data.max().max():.3f}; "
          f"column sd: {data.std().min():.3f}-{data.std().max():.3f}")
    print(f"same seed identical: {data.equals(again)}; different seed differs: {not data.equals(other)}")
    print(f"function types (seed 0): {info['kinds']}")
    others = [v for v in sorted(graph.nodes) if v not in (x, y)]
    for realistic in (False, True):
        print(f"\n{'realistic (S3)' if realistic else 'ADIA-style'}:")
        print(f"{'seed':>4} {'true beta':>9} {'X alone':>8} {'all controls':>13} {'confounder only (oracle)':>25}"
              + (f" {'regime 1':>9} {'regime 2':>9} {'kurtosis':>9}" if realistic else ''))
        for seed in range(8):
            data, effect, info = generate(graph, x, y, seed, realistic=realistic)
            line = (f"{seed:>4} {effect:>9.3f} {slope(data, x, y):>8.3f} {slope(data, x, y, others):>13.3f} "
                    f"{slope(data, x, y, ['Leverage']):>25.3f}")
            if realistic:
                line += f" {info['beta_regime1']:>9.3f} {info['beta_regime2']:>9.3f} {data.kurt().median():>9.1f}"
            print(line)