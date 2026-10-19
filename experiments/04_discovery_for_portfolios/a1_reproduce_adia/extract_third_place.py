"""
Turn the 3rd-place notebook (Hong & Gu, pinned commit) into an importable module, unchanged.
Copies its helper and feature cells into external/third_place_features.py (gitignored,
never committed: the repo has no license), dropping only CrunchDAO-only lines.
Run in the adia-3rd env (requirements_third_place.txt: 2024 package versions).
"""

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import EXTERNAL

REPO = EXTERNAL / 'adia-3rd'
NOTEBOOK = REPO / 'supervised_baseline_test_sub_online.ipynb'
OUT = EXTERNAL / 'third_place_features.py'
COMMIT = '7fd401dff92e7093724963fcdf06528107439ffb'

# code-cell indices: imports, graph utils, augmentation, copula entropy, PPS, regression utils,
# discretisation, Cloud, feature engineering, label, create_*_columns, model imports, training utils
CELLS = [0, 2, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]

head = subprocess.run(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
if head != COMMIT:
    sys.exit(f"external/adia-3rd is at {head}, expected {COMMIT}")

code_cells = [c for c in json.loads(NOTEBOOK.read_text(encoding='utf-8'))['cells'] if c['cell_type'] == 'code']
keep = lambda line: not line.lstrip().startswith(('%', '!')) and 'crunch' not in line
source = '\n\n'.join(f"# ---- notebook code cell {i}\n" + '\n'.join(filter(keep, ''.join(code_cells[i]['source']).splitlines()))
                     for i in CELLS)

header = (f'"""Auto-generated from Hong & Gu, ADIA Lab Causal Discovery 3rd place, commit {COMMIT}.\n'
          'Do not edit or commit. Rebuild with a1_reproduce_adia/extract_third_place.py."""\n\n')
OUT.write_text(header + source, encoding='utf-8')
print(f"wrote {OUT} ({len(source.splitlines())} lines from {len(CELLS)} cells, unpatched)")