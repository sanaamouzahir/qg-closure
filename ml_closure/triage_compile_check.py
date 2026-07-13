"""Syntax/import gate for the 2026-07-13 triage scripts (runs on all.q; the
GPU jobs hold on this so a typo never burns a GPU slot)."""
import importlib
import py_compile
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = ['triage_ab_filter.py', 'replot_eval_fields.py',
           'triage_mask_audit.py', 'triage_sigma_decomp.py']

for s in SCRIPTS:
    py_compile.compile(str(HERE / s), doraise=True)
    print(f'[compile-check] {s}: OK')

# import the frozen deps the tools rely on (no model construction, no data)
for mod in ['dataset_piff', 'model_piff', 'qg._output.filter']:
    importlib.import_module(mod)
    print(f'[compile-check] import {mod}: OK')
print('[compile-check] all clear')
sys.exit(0)
