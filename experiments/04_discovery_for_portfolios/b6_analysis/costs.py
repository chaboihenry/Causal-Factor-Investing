"""
P14's cost table, built from the saved terminal logs rather than typed in. Each LLM call's time is read from the
run log; the cost per run is the MEDIAN call time x the number of calls, so a break before pasting a prompt (two
Claude calls took ~32 min) can't inflate it; the total wall-clock time is reported next to it. B1, A2 and PC times
come from their logged runs. Saves results/exp04/b6_analysis/costs.csv. Rule fixed before scoring.
"""

import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import RESULTS

LOGS = RESULTS / 'b5_runs' / 'logs'
OUT = RESULTS / 'b6_analysis' / 'costs.csv'
CALL = re.compile(r'^\s+(c\d(?:_\d+)?): (\d+)s, parsed=')
LLM = {'A1 gpt-oss:20b (local)': ('a1_gpt-oss-exp04.log', 10, '$0 (local GPU)', 'none'),
       'A1 Claude Opus 5.5 (claude.ai)': ('a1_claude-opus-5.5-medium.log', 3, '$0 beyond the Pro plan', 'paste each prompt by hand')}
OTHER = [  # (arm, one-off cost, cost on the 1,300 test datasets, source)
    ('B1 3rd-place ADIA', 'training: 22.6 h wall clock (23,500 datasets, augment x3)', '~2-12 min per set to predict',
     'design.md Phase 1; predict_third_place.py logs'),
    ('A2 refutation (per LLM run)', 'none beyond A1', '~71 min for 12 runs on 16 cores (~6 min per run)', 'run_refute.py log'),
    ('C1 PC', 'none', 'a few minutes on 16 cores', 'run_pc.py log'),
    ('A3 hybrid', 'A1 + B1', 'seconds (combines saved outputs)', 'collect_roles.py'),
]


def calls(log):
    times = [int(m.group(2)) for line in log.read_text(encoding='utf-8').splitlines() if (m := CALL.match(line))]
    return pd.Series(times, dtype=float)


if __name__ == '__main__':
    rows = []
    for arm, (log, runs, dollars, human) in LLM.items():
        t = calls(LOGS / log)
        rows.append({'arm': arm, 'runs': runs, 'calls': len(t), 'calls per run': len(t) / runs,
                     'median call (s)': t.median(), 'cost per run (min, median x calls)': t.median() * len(t) / runs / 60,
                     'total wall clock (min)': t.sum() / 60, 'longest call (s)': t.max(), 'dollars': dollars, 'human work': human})
    table = pd.concat([pd.DataFrame(rows), pd.DataFrame(OTHER, columns=['arm', 'one-off cost', 'test-set cost', 'source'])])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT, index=False)
    print(table.round(1).to_string(index=False))