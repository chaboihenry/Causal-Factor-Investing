"""
Post figure: median hedge leak in the 100 literature worlds, from no graph to the true graph (Claude-based LLM arms).
Reads results/exp04/b6_analysis/scores_betas.csv (subset 'lit'); saves figures/fig4_hedge_leak.png.
Expected bars: 84 / 46 / 38 / 38 / 38 / 30 / 17.
"""

import sys
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import FIGURES, RESULTS

CLAUDE = 'claude-opus-5.5-medium'
BARS = [  # arm, model (None = no model), label, group
    ('Everything', None, 'Control for everything\nmeasured by year-end', 'none'),
    ('X alone', None, 'Factor alone', 'none'),
    ('A1', CLAUDE, 'LLM (Claude Opus 5.5)', 'ai'),
    ('B1', None, 'Supervised ML\n(ADIA 3rd-place solution)', 'ai'),
    ('A3', CLAUDE, 'LLM + ML hybrid', 'ai'),
    ('Start of year', None, "Simple rule: control only for\nwhat's known before the factor", 'rule'),
    ('Oracle', None, 'True causal graph', 'truth'),
]
GROUPS = {'none': ('No causal graph', '#a3a29b'), 'ai': ('AI draws the graph', '#2a78d6'),
          'rule': ('Timing rule', '#86b6ef'), 'truth': ('Knowing the truth', '#104281')}
INK, INK2, MUTED, GRID, AXIS, SURF = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#fcfcfb'


def leak(scores, arm, model):
    """Median hedge leak (%) of one arm in the literature worlds."""
    rows = scores[(scores['arm'] == arm) & (scores['model'].isna() if model is None else scores['model'] == model)]
    assert len(rows) == 1, f"expected 1 row for {arm} / {model}, got {len(rows)}"
    return 100 * rows['hedge_leak'].iloc[0]


def plot(values):
    """Horizontal bars, top to bottom in BARS order, with value labels and a group legend."""
    fig, ax = plt.subplots(figsize=(10, 6.2), dpi=200, facecolor=SURF)
    ax.set_facecolor(SURF)
    ys = range(len(BARS) - 1, -1, -1)
    for y, (_, _, label, group), v in zip(ys, BARS, values):
        ax.barh(y, v, height=0.64, color=GROUPS[group][1])
        ax.text(v + 1.2, y, f"{v:.0f}%", va='center', fontsize=13, weight='bold', color=INK)
    ax.set_yticks(list(ys), [b[2] for b in BARS], fontsize=11.5, color=INK)
    ax.set_xlim(0, 100)
    ax.set_xticks(range(0, 101, 25), [f"{t}%" for t in range(0, 101, 25)], color=MUTED)
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right', 'bottom']].set_visible(False)
    ax.spines['left'].set_color(AXIS)
    ax.tick_params(length=0)
    fig.text(0.02, 0.965, "How much of a factor's true risk does the hedge miss?", fontsize=17, weight='bold', color=INK)
    fig.text(0.02, 0.918, 'Median share of the true beta left unhedged, 100 synthetic worlds built from the finance '
             'literature', fontsize=11, color=INK2)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c in GROUPS.values()]
    fig.legend(handles, [n for n, _ in GROUPS.values()], loc='upper left', bbox_to_anchor=(0.012, 0.9), ncol=4,
               frameon=False, fontsize=10.5, labelcolor=INK2, handlelength=1, columnspacing=3)
    fig.text(0.02, 0.02, 'Source: Causal Factor Investing on GitHub, Experiment 04 (scores_betas.csv, literature worlds)',
             fontsize=9, color=MUTED)
    fig.subplots_adjust(left=0.30, right=0.97, top=0.83, bottom=0.08)
    return fig


def main():
    scores = pd.read_csv(RESULTS / 'b6_analysis' / 'scores_betas.csv')
    scores = scores[scores['subset'] == 'lit']
    values = [leak(scores, arm, model) for arm, model, _, _ in BARS]
    print(' / '.join(f"{v:.0f}" for v in values))
    out = FIGURES / 'fig4_hedge_leak.png'
    plot(values).savefig(out, facecolor=SURF)
    print(f"saved {out}")


if __name__ == '__main__':
    main()