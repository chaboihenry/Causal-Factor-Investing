"""
Post-hoc diagnostics (NOT pre-registered; explain results, never change verdicts): (1) the role each LLM run gives each
variable vs the true literature role; (2) per arm, which variables it controls for and its median hedge leak in the
literature worlds vs S1; (3) how often A3's τ rule lets B1 overrule the LLM, and how often the overrule is right.
Saves results/exp04/b6_analysis/diagnostics.txt.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import RESULTS

B5, OUT = RESULTS / 'b5_runs', RESULTS / 'b6_analysis'

if __name__ == '__main__':
    roles = pd.read_csv(B5 / 'roles_all.csv.gz').fillna({'model': ''})
    betas = pd.read_csv(B5 / 'betas_all.csv.gz').fillna({'model': ''})
    lines = []

    lit = roles[(roles.set == 'main') & (roles.agreement == 100)]
    a1 = lit[lit.arm == 'A1'].groupby(['variable', 'true', 'model'])['pred'].agg(lambda p: ', '.join(
        f"{r} x{n}" for r, n in p.value_counts().items() if n))
    lines += ['(1) A1 roles in the literature worlds (role x number of dataset-runs)', a1.unstack('model').to_string(), '']

    bl = betas[(betas.agreement == 100) & betas.set.isin(['main', 's1'])]
    leak = bl.groupby(['arm', 'model', 'set'])['hedge_leak'].median().unstack('set').mul(100).round(1)
    ctrl = bl[bl.set == 'main'].groupby(['arm', 'model'])['controls'].agg(lambda c: c.fillna('(none)').value_counts().index[0])
    lines += ['(2) Median hedge leak (%) in literature worlds: main vs S1, and the most common control set in main',
              leak.join(ctrl.rename('usual controls (main)')).to_string(), '']

    b1 = roles[roles.arm == 'B1'][['set', 'dataset', 'variable', 'pred']].rename(columns={'pred': 'b1'})
    a1all = roles[roles.arm == 'A1'][['set', 'dataset', 'variable', 'model', 'run', 'true', 'pred']]
    a3 = roles[roles.arm == 'A3'][['set', 'dataset', 'variable', 'model', 'run', 'pred']].rename(columns={'pred': 'a3'})
    m = a1all.merge(a3, on=['set', 'dataset', 'variable', 'model', 'run']).merge(b1, on=['set', 'dataset', 'variable'])
    m = m[m.set == 'main']
    m['overruled'] = m.a3 != m.pred
    m['disagree'] = m.b1 != m.pred
    summary = m.groupby('model').agg(disagree=('disagree', 'mean'), overruled=('overruled', 'mean'))
    summary['overrule right'] = m[m.overruled].groupby('model').apply(lambda g: (g.a3 == g.true).mean())
    summary['LLM right where overruled'] = m[m.overruled].groupby('model').apply(lambda g: (g.pred == g.true).mean())
    summary['B1 right where not overruled but disagreeing'] = m[m.disagree & ~m.overruled].groupby('model').apply(
        lambda g: (g.b1 == g.true).mean())
    lines += ['(3) A3 in the main test: share of variables where B1 disagrees with the LLM / overrules it, and who is right',
              summary.round(3).to_string()]

    text = '\n'.join(lines)
    (OUT / 'diagnostics.txt').write_text(text, encoding='utf-8')
    print(text)