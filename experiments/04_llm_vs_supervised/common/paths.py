"""
Shared paths for Experiment 04.

Every script imports its locations from here, so moving a folder means
changing one line instead of every file.
"""

from pathlib import Path

EXP = Path(__file__).resolve().parents[1]      # experiments/04_llm_vs_supervised
ROOT = EXP.parents[1]                          # repo root

ADIA = ROOT / 'data' / 'adia'
SYNTHETIC = ROOT / 'data' / 'synthetic'
MODELS = ROOT / 'models'
EXTERNAL = ROOT / 'external'
RESULTS = ROOT / 'results' / 'exp04'
FIGURES = EXP / 'figures'
