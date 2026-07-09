#!/bin/bash
# phaseB_make_tables_wave2.sh - CPU job generating the Phase-B wave-2 inlet
# tables: MOD-sine/ramp/ou/telegraph @ dt=2.5e-4, T=120, t-wait 30, seed
# 20260707 (charter S3.1/S4.1). Pattern: phaseB_make_tables.sh (wave 1).
# Queue: all.q. Released by Sanaa's 2026-07-09 full-ensemble approval.
#
# Usage:
#   qsub -q all.q -N pB2_tab -o logs/pB2_tab.$JOB_ID.log -j y -cwd -V \
#        scripts/sge/phaseB_make_tables_wave2.sh

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

echo "[pB2_tables] host $HOSTNAME  date $(date -u +%FT%TZ)"

for sig in sine ramp ou telegraph; do
    python -u "$MODULATION_PY" --signal "$sig" --dt 2.5e-4 --T 120 --t-wait 30 \
        --seed 20260707 --out "$TABLES_DIR/${sig}_dt2p5e-4_T120.npz"
done

ls -la "$TABLES_DIR"
echo "[pB2_tables] done $(date -u +%FT%TZ)"
