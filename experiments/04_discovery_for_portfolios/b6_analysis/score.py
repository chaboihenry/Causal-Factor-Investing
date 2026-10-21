"""
Phase 6: score every arm against predictions.md in one pass, from the frozen Phase 5 outputs (roles_all, betas_all,
A2 edge tests, PC edges, costs). Balanced accuracy per arm and agreement level (LLM arms: mean over runs, with range);
beta measures per regression and arm; 95% intervals by bootstrapping graphs within a level (datasets where a level has one
graph). Saves results/exp04/b6_analysis/{verdicts,scores_roles,scores_betas}.csv and figures/fig1-3; prints P1-P14 verdicts.
"""

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import FIGURES, RESULTS

B5, OUT, META = RESULTS / 'b5_runs', RESULTS / 'b6_analysis', RESULTS / 'b3_worlds' / 'datasets_meta.csv'
X, Y = 'BookToMarket', 'AnnualReturn'
LEVELS = [100, 75, 50, 25, 0]
CLAUDE, OSS = 'claude-opus-5.5-medium', 'gpt-oss-exp04'
RUNS = {OSS: range(1, 11), CLAUDE: range(1, 4)}
ROLES8 = ['Cause of X', 'Cause of Y', 'Collider', 'Confounder', 'Consequence of X', 'Consequence of Y', 'Independent',
          'Mediator']
N_BOOT, SEED = 1000, 0
SEEN = [f"lit_{i:03d}" for i in range(8)]         # printed by generator.py's self-check before predictions
COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']          # validated categorical slots 1-5
INK, MUTED = '#0b0b0b', '#52514e'


# ---------- balanced accuracy with a bootstrap over graphs ----------
def ba_summary(df, unit):
    """df: one arm/model/set/level, all runs. Mean over runs of balanced accuracy, run range, and a 95% interval."""
    runs, units = sorted(df.run.unique()), sorted(df[unit].unique())
    r_idx, u_idx = df.run.map({r: i for i, r in enumerate(runs)}), df[unit].map({u: i for i, u in enumerate(units)})
    k_idx = df.true.map({k: i for i, k in enumerate(ROLES8)})
    total, right = np.zeros((len(runs), len(units), 8)), np.zeros((len(runs), len(units), 8))
    np.add.at(total, (r_idx, u_idx, k_idx), 1)
    np.add.at(right, (r_idx, u_idx, k_idx), (df.true == df.pred).to_numpy())

    def ba(sel):
        t, c = total[:, sel].sum(1), right[:, sel].sum(1)
        with np.errstate(invalid='ignore'):
            return np.nanmean(np.where(t > 0, c / t, np.nan), axis=1)          # one value per run

    per_run = ba(np.arange(len(units)))
    rng = np.random.default_rng(SEED)
    boots = [ba(rng.integers(len(units), size=len(units))).mean() for _ in range(N_BOOT)]
    return {'ba': per_run.mean(), 'lo': np.percentile(boots, 2.5), 'hi': np.percentile(boots, 97.5),
            'run_min': per_run.min(), 'run_max': per_run.max(), 'runs': len(runs)}


def score_roles(roles, meta):
    roles = roles.assign(world=roles.dataset.map(meta.world))
    rows = []
    for (arm, model, s, level), g in roles.groupby(['arm', 'model', 'set', 'agreement']):
        unit = 'world' if g.world.nunique() > 1 else 'dataset'
        rows.append({'arm': arm, 'model': model, 'set': s, 'agreement': level, **ba_summary(g, unit)})
    for (arm, model, s), g in roles.groupby(['arm', 'model', 'set']):
        rows.append({'arm': arm, 'model': model, 'set': s, 'agreement': 'all', **ba_summary(g, 'world')})
    return pd.DataFrame(rows)


# ---------- beta measures with a bootstrap over datasets ----------
def stat(values, how):
    return np.nanmedian(values) if how == 'median' else np.nanmean(values)


def beta_summary(df, column, how):
    """Pooled over runs; bootstrap resamples datasets (each with all its runs)."""
    m = df.pivot_table(index='dataset', columns='run', values=column, aggfunc='first').to_numpy(dtype=float)
    if np.isnan(m).all():
        return {f'{column}': np.nan, f'{column}_lo': np.nan, f'{column}_hi': np.nan, f'{column}_n': 0}
    rng = np.random.default_rng(SEED)
    boots = [stat(m[rng.integers(len(m), size=len(m))], how) for _ in range(N_BOOT)]
    return {column: stat(m, how), f'{column}_lo': np.nanpercentile(boots, 2.5),
            f'{column}_hi': np.nanpercentile(boots, 97.5), f'{column}_n': int(np.isfinite(m).sum())}


def score_betas(betas):
    betas = betas.assign(subset=np.where((betas.set == 'main') & (betas.agreement == 100), 'lit', betas.set))
    unseen = betas[(betas.subset == 'lit') & ~betas.dataset.isin(SEEN)].assign(subset='lit (without seen 0-7)')
    s3_lit = betas[(betas.set == 's3') & (betas.agreement == 100)].assign(subset='s3 lit')
    rows = []
    for (arm, model, subset), g in pd.concat([betas, unseen, s3_lit]).groupby(['arm', 'model', 'subset']):
        rows.append({'arm': arm, 'model': model, 'subset': subset, **beta_summary(g, 'wrong_sign', 'mean'),
                     **beta_summary(g, 'hedge_leak', 'median'), **beta_summary(g, 'false_effect', 'mean')})
    return pd.DataFrame(rows)


# ---------- verdicts ----------
def verdicts(R, Bt, a2, pc, costs):
    ba = lambda arm, model='', s='main', level='all': R.set_index(['arm', 'model', 'set', 'agreement']).loc[
        (arm, model, s, level), 'ba'] * 100
    bt = lambda arm, subset, col, model='': Bt.set_index(['arm', 'model', 'subset']).loc[(arm, model, subset), col] * 100
    out = []

    def add(p, claim, rule, measured, ok):
        out.append({'prediction': p, 'claim': claim, 'rule': rule, 'measured': measured, 'verdict': 'PASS' if ok else 'FAIL'})

    for p, arm, sign_min, leak_rule in [('P1', 'Everything', 20, ('>=', 50)), ('P2', 'X alone', 10, ('>=', 30))]:
        ws, hl = bt(arm, 'lit', 'wrong_sign'), bt(arm, 'lit', 'hedge_leak')
        add(p, 1, f'"{arm}": wrong sign >= {sign_min}%, median hedge leak >= {leak_rule[1]}%',
            f'wrong sign {ws:.0f}%, leak {hl:.0f}%', ws >= sign_min and hl >= leak_rule[1])
    hl = bt('Start of year', 'lit', 'hedge_leak')
    add('P1b', 1, '"Start of year": median hedge leak <= 35%', f'leak {hl:.0f}%', hl <= 35)
    hl = bt('Oracle', 'lit', 'hedge_leak')
    add('P3', 1, 'Oracle: median hedge leak <= 25%', f'leak {hl:.0f}%', hl <= 25)
    fe = {a: bt(a, 's2b', 'false_effect') for a in ['X alone', 'Everything', 'Oracle']}
    add('P3b', 1, 'S2b false effect: X alone and Everything >= 50%; Oracle <= 20%',
        ', '.join(f'{a} {v:.0f}%' for a, v in fe.items()), fe['X alone'] >= 50 and fe['Everything'] >= 50 and fe['Oracle'] <= 20)

    b1 = [ba('B1', level=lv) for lv in LEVELS]
    add('P4', 2, 'B1 65-78% at every level, spread <= 6 pts', ' / '.join(f'{v:.1f}' for v in b1),
        all(65 <= v <= 78 for v in b1) and max(b1) - min(b1) <= 6)
    cl = [ba('A1', CLAUDE, level=lv) for lv in LEVELS]
    add('P5', 2, 'A1 Claude >= 75% at 100, <= 25% at 0, falls at every step', ' / '.join(f'{v:.1f}' for v in cl),
        cl[0] >= 75 and cl[-1] <= 25 and all(a > b for a, b in zip(cl, cl[1:])))
    oss100 = ba('A1', OSS, level=100)
    add('P6', 2, 'A1 gpt-oss at 100% below Claude by >= 10 pts', f'gpt-oss {oss100:.1f} vs Claude {cl[0]:.1f}',
        cl[0] - oss100 >= 10)
    d100, d75 = cl[0] - b1[0], cl[1] - b1[1]
    add('P7', 2, 'A1 Claude and B1 cross between 75% and 100%', f'Claude - B1: {d100:+.1f} at 100, {d75:+.1f} at 75',
        d100 > 0 > d75)
    for model, tag in [(CLAUDE, 'Claude'), (OSS, 'gpt-oss')]:
        gain = [ba('A2', model, level=lv) - ba('A1', model, level=lv) for lv in LEVELS]
        add(f'P8 ({tag})', 2, 'A2 - A1 >= 3 pts at 50, 25, 0; < 1 pt at 100', ' / '.join(f'{v:+.1f}' for v in gain),
            all(g >= 3 for g in gain[2:]) and gain[0] < 1)
    pcv = [ba('C1', level=lv) for lv in LEVELS]
    add('P9', 2, 'PC 33-45% at every level, below B1 everywhere', ' / '.join(f'{v:.1f}' for v in pcv),
        all(33 <= v <= 45 for v in pcv) and all(p < b for p, b in zip(pcv, b1)))
    for model, tag in [(CLAUDE, 'Claude'), (OSS, 'gpt-oss')]:
        a3 = [ba('A3', model, level=lv) for lv in LEVELS]
        best = [max(ba('A1', model, level=lv), b) for lv, b in zip(LEVELS, b1)]
        add(f'P10 ({tag})', 4, 'A3 >= best of A1, B1 - 1 pt at every level; strictly above both at 75%',
            ' / '.join(f'{a - b:+.1f}' for a, b in zip(a3, best)) + ' (A3 minus best)',
            all(a >= b - 1 for a, b in zip(a3, best)) and a3[1] > best[1])

    arms = [('B1', ''), ('C1', '')] + [(a, m) for a in ['A1', 'A2', 'A3'] for m in (CLAUDE, OSS)]
    ratios = {f"{a}{' ' + m.split('-')[0] if m else ''}": bt(a, 's1', 'hedge_leak', m) / bt(a, 'lit', 'hedge_leak', m)
              for a, m in arms}
    add('P11', 3, 'S1: every arm\'s median hedge leak at least doubles vs. the same worlds in main',
        ', '.join(f'{k} x{v:.1f}' for k, v in ratios.items()), all(v >= 2 for v in ratios.values()))
    xy = a2[(a2.set == 's2') & (a2.source == X) & (a2.target == Y)]
    a1_keep = len(xy) / (sum(len(r) for r in RUNS.values()) * a2[a2.set == 's2'].dataset.nunique())
    a2_removes = 1 - xy.kept.mean() if len(xy) else 0.0
    pc_keep = pc[(pc.set == 's2') & (pc.source == X) & (pc.target == Y)].dataset.nunique() / 100
    add('P12', 3, 'S2: B1 and A1 keep X->Y in >= 90%; only A2 ever removes it',
        f'A1 keeps {a1_keep:.0%}, B1 100% (by design), PC {pc_keep:.0%}, A2 removes {a2_removes:.0%}',
        a1_keep >= 0.9 and pc_keep == 1 and a2_removes > 0)
    a1_same = all(abs(ba('A1', m, 's3') - ba('A1', m)) < 1e-9 for m in RUNS)
    b1_drop, pc_drop = ba('B1') - ba('B1', s='s3'), ba('C1') - ba('C1', s='s3')
    leak_main, leak_s3 = bt('Oracle', 'lit', 'hedge_leak'), bt('Oracle', 's3 lit', 'hedge_leak')
    add('P13', 3, 'S3: A1 unchanged; B1 falls >= 5 pts; PC falls; oracle leak rises',
        f'A1 unchanged: {a1_same}; B1 {-b1_drop:+.1f} pts; PC {-pc_drop:+.1f} pts; oracle leak {leak_main:.0f}% -> {leak_s3:.0f}%',
        a1_same and b1_drop >= 5 and pc_drop > 0 and leak_s3 > leak_main)
    llm = costs.dropna(subset=['cost per run (min, median x calls)'])
    add('P14', 5, 'A1 $0 and minutes per run; B1 ~23 h training + ~10 min per 500 datasets; A3 both',
        '; '.join(f"{r.arm}: {r['cost per run (min, median x calls)']:.1f} min/run" for _, r in llm.iterrows())
        + '; B1 22.6 h training, ~12 min per 500 to predict', (llm['cost per run (min, median x calls)'] < 60).all())
    return pd.DataFrame(out)


# ---------- figures ----------
def style(ax, title):
    ax.set_title(title, loc='left', fontsize=11, color=INK)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis='y', color='#e4e3df', linewidth=0.8)
    ax.set_axisbelow(True)


def bars(ax, labels, values, lo, hi, color, fmt='{:.0f}%'):
    x = np.arange(len(labels))
    ax.bar(x, values, color=color, width=0.6, edgecolor='#fcfcfb', linewidth=2)
    ax.errorbar(x, values, yerr=[np.subtract(values, lo), np.subtract(hi, values)], fmt='none', ecolor=MUTED, capsize=3, lw=1)
    for xi, v, top in zip(x, values, hi):
        ax.text(xi, top, fmt.format(v), ha='center', va='bottom', fontsize=9, color=INK)
    ax.set_xticks(x, labels, fontsize=9)


def fig1(Bt):
    regs = ['X alone', 'Everything', 'Start of year', 'Oracle']
    get = lambda subset, col: Bt.set_index(['arm', 'model', 'subset']).loc[[(r, '', subset) for r in regs], [col, f'{col}_lo', f'{col}_hi']] * 100
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), facecolor='#fcfcfb')
    for ax, (subset, col, title) in zip(axes, [('lit', 'hedge_leak', 'Median hedge leak (100 literature worlds)'),
                                               ('lit', 'wrong_sign', 'Wrong sign of beta'),
                                               ('s2b', 'false_effect', 'False effect when the true beta is 0 (S2b)')]):
        v = get(subset, col)
        bars(ax, regs, v[col], v[f'{col}_lo'], v[f'{col}_hi'], COLORS[0])
        ax.set_ylabel('% of worlds' if col != 'hedge_leak' else '% of true beta left unhedged', fontsize=9, color=MUTED)
        style(ax, title)
    fig.suptitle('Claim 1: a regression without the right causal graph reports the wrong beta', x=0.01, ha='left', fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES / 'fig1_beta_error.png', dpi=160)


def fig2(R):
    arms = [('B1', '', 'B1 supervised ML', 'o'), ('C1', '', 'PC', 's'), ('A1', None, 'A1 LLM', '^'),
            ('A2', None, 'A2 LLM + refutation', 'v'), ('A3', None, 'A3 hybrid', 'D')]
    main = R[(R.set == 'main') & (R.agreement != 'all')].copy()
    main['agreement'] = main.agreement.astype(int)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True, facecolor='#fcfcfb')
    for ax, (model, tag) in zip(axes, [(CLAUDE, 'Claude Opus 5.5'), (OSS, 'gpt-oss:20b')]):
        for color, (arm, m, label, marker) in zip(COLORS, arms):
            g = main[(main.arm == arm) & (main.model == (model if m is None else m))].sort_values('agreement')
            ax.plot(g.agreement, g.ba * 100, color=color, lw=2, marker=marker, ms=8, label=label)
            ax.fill_between(g.agreement, g.lo * 100, g.hi * 100, color=color, alpha=0.12, lw=0)
        ax.axhline(12.5, color=MUTED, lw=1, ls='--')
        ax.text(62, 11.5, 'random guessing (12.5%)', fontsize=8, color=MUTED, ha='center', va='top')
        ax.set_xlim(103, -3)
        ax.set_xticks(LEVELS, [f'{lv}%' for lv in LEVELS])
        ax.set_xlabel('agreement of the true graph with the literature', fontsize=9, color=MUTED)
        style(ax, f'LLM = {tag}')
    axes[0].set_ylabel('balanced accuracy of roles (%)', fontsize=9, color=MUTED)
    for ax in axes:
        ax.legend(frameon=False, fontsize=8, loc='upper right')
    fig.suptitle('Claims 2 and 4: who finds the right roles as the literature becomes less right', x=0.01, ha='left', fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES / 'fig2_dial.png', dpi=160)


def fig3(R, Bt):
    arms = [('B1', '', 'B1'), ('C1', '', 'PC'), ('A1', CLAUDE, 'A1 Claude'), ('A2', CLAUDE, 'A2 Claude'),
            ('A3', CLAUDE, 'A3 Claude'), ('Oracle', '', 'Oracle')]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), facecolor='#fcfcfb')
    bi = Bt.set_index(['arm', 'model', 'subset'])
    for ax, (base, stress, title) in zip(axes, [('lit', 's1', 'S1 leverage hidden: median hedge leak'),
                                                ('lit', 's3 lit', 'S3 realistic data: median hedge leak')]):
        x = np.arange(len(arms))
        for i, (subset, label) in enumerate([(base, 'main test'), (stress, title.split(':')[0])]):
            v = [bi.loc[(a, m, subset), 'hedge_leak'] * 100 for a, m, _ in arms]
            ax.bar(x + (i - 0.5) * 0.38, v, width=0.38, color=COLORS[i], edgecolor='#fcfcfb', linewidth=2, label=label)
        ax.set_xticks(x, [lab for *_, lab in arms], fontsize=9)
        ax.set_ylabel('% of true beta left unhedged', fontsize=9, color=MUTED)
        ax.legend(frameon=False, fontsize=8)
        style(ax, title)
    fig.suptitle('Claim 3: does the answer survive hidden causes and realistic data?', x=0.01, ha='left', fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES / 'fig3_stress.png', dpi=160)


if __name__ == '__main__':
    meta = pd.read_csv(META).set_index('id')
    roles = pd.read_csv(B5 / 'roles_all.csv.gz').fillna({'model': ''})
    betas = pd.read_csv(B5 / 'betas_all.csv.gz').fillna({'model': ''})
    a2 = pd.concat([pd.read_csv(B5 / 'a2' / f"{m}_run{r:02d}.csv.gz").assign(model=m, run=r) for m, rs in RUNS.items() for r in rs])
    pc, costs = pd.read_csv(B5 / 'pc_edges.csv.gz'), pd.read_csv(OUT / 'costs.csv')

    R, Bt = score_roles(roles, meta), score_betas(betas)
    V = verdicts(R, Bt, a2, pc, costs)
    OUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    R.to_csv(OUT / 'scores_roles.csv', index=False)
    Bt.to_csv(OUT / 'scores_betas.csv', index=False)
    V.to_csv(OUT / 'verdicts.csv', index=False)
    fig1(Bt), fig2(R), fig3(R, Bt)
    pd.set_option('display.width', 250, 'display.max_colwidth', 120)
    print(V[['prediction', 'verdict', 'measured', 'rule']].to_string(index=False))
    print(f"\n{(V.verdict == 'PASS').sum()} of {len(V)} predictions passed; tables in {OUT}, figures in {FIGURES}")