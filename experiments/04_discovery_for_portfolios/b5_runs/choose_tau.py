"""
Choose τ, the hybrid's (A3) confidence threshold, before any test run: the lowest B1 confidence at which B1's top
role is right >= 85% of the time *averaged equally over the 8 roles* (as balanced accuracy weighs them), measured on
the 1,880 generator-check datasets (our generator, not the test). Also prints the pooled precision (every variable one
vote) for comparison. Saves results/exp04/b5_runs/tau.csv; the chosen τ goes into predictions.md before it is committed.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import RESULTS

CHECK = RESULTS / 'b3_worlds' / 'third_place_check_ours_roles.csv'
OUT = RESULTS / 'b5_runs' / 'tau.csv'
TARGET = 0.85
GRID = np.round(np.append(np.arange(0.30, 1.00, 0.05), 0.99), 2)


def precision(confident):
    """Pooled: share right over all variables. Balanced: share right per predicted role, then averaged over roles."""
    by_role = confident.groupby('pred')['right'].mean()
    return confident['right'].mean(), by_role.mean(), len(by_role)


if __name__ == '__main__':
    roles = pd.read_csv(CHECK)
    roles['confidence'] = roles.filter(like='p_').max(axis=1)
    roles['right'] = roles['pred'] == roles['true']

    rows = []
    for t in GRID:
        confident = roles[roles.confidence >= t]
        pooled, balanced, n_roles = precision(confident)
        rows.append({'tau': t, 'balanced precision': balanced, 'pooled precision': pooled,
                     'coverage': len(confident) / len(roles), 'roles present': n_roles})
    table = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT, index=False)
    print(table.round(3).to_string(index=False))

    passing = table[(table['balanced precision'] >= TARGET) & (table['roles present'] == 8)]
    if passing.empty:
        sys.exit(f"no threshold reaches {TARGET:.0%} balanced precision with all 8 roles present")
    tau = passing['tau'].min()
    confident = roles[roles.confidence >= tau]
    print(f"\nτ = {tau}: balanced precision {precision(confident)[1]:.1%} on the {len(confident)} variables "
          f"({len(confident) / len(roles):.0%}) where B1 is at least that confident")
    print(confident.groupby('pred')['right'].agg(['mean', 'size']).round(3).rename(columns={'mean': 'precision'}))