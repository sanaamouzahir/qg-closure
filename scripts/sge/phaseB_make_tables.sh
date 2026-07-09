#!/bin/bash
# phaseB_make_tables.sh - CPU job generating the Phase-B wave-1 inlet table:
#   MOD-const @ dt=2.5e-4, T=120, t-wait 30 (charter S4.1; Amendment 02
#   cylinder-only). Modulated-case tables (sine/ramp/ou/telegraph) are
#   generated when the 4 modulated runs are RELEASED by Audit A (theory doc
#   S8/A decision rule), not before. Gate D-1's Re=200 table is HELD pending
#   Sanaa's ruling on the [QG][FLAG][SGS] timescale arithmetic (T=120 at
#   U=0.1026 is 9.8 convective units; see BRANCH_LOG 2026-07-09).
#
# Pattern: gate1_make_tables.sh (job 1826260). CPU-only; queue all.q.
# Usage:
#   qsub -q all.q -N phaseB_tab -o logs/phaseB_tab.$JOB_ID.log -j y -cwd -V \
#        scripts/sge/phaseB_make_tables.sh

#$ -S /bin/bash
#$ -cwd
#$ -V

set -e

QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
QG_DIR="$QG_ROOT/qg-simple-package-stable/src/qg"

source "$QG_ROOT/qg-env/bin/activate"
export TMPDIR="$QG_ROOT/tmp"
export MPLCONFIGDIR="$QG_ROOT/mplcache"
export PYTHONUNBUFFERED=1

MODULATION_PY="$QG_ROOT/qg-sgs-closure/training/modulation.py"
[ -f "$MODULATION_PY" ] || { echo "ERROR: $MODULATION_PY missing"; exit 1; }

TABLES_DIR="$QG_DIR/outputs/SGS_closure_ensemble/tables"
mkdir -p "$TABLES_DIR"

echo "[phaseB_tables] host $HOSTNAME  date $(date -u +%FT%TZ)"

# MOD-const, production: dt equals the solver dt exactly (bc rule: no
# runtime interpolation), T=120, T_wait=30 (charter S4.1). YAML-trap rule:
# explicit decimal mantissa.
python -u "$MODULATION_PY" --signal const --dt 2.5e-4 --T 120 --t-wait 30 \
    --out "$TABLES_DIR/const_dt2p5e-4_T120.npz"

ls -la "$TABLES_DIR"
echo "[phaseB_tables] done $(date -u +%FT%TZ)"
