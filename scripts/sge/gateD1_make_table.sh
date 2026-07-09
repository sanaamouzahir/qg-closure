#!/bin/bash
# gateD1_make_table.sh - CPU job: Gate D-1 option-B inlet table.
#   const, Re=200, dt=2.5e-3 (Sanaa-approved run dt, [red-approved]
#   2026-07-09), T=1500, t-wait 250. Arg 1 = output npz path.
# Usage: qsub -q all.q -N gd1_tab -o logs/... -j y -cwd -V \
#            gateD1_make_table.sh <out.npz>

#$ -S /bin/bash
#$ -cwd
#$ -V

set -e

QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
source "$QG_ROOT/qg-env/bin/activate"
export TMPDIR="$QG_ROOT/tmp"
export MPLCONFIGDIR="$QG_ROOT/mplcache"
export PYTHONUNBUFFERED=1

OUT="${1:?usage: gateD1_make_table.sh <out.npz>}"
mkdir -p "$(dirname "$OUT")"

echo "[gd1_table] host $HOSTNAME  date $(date -u +%FT%TZ)  out $OUT"
python -u "$QG_ROOT/qg-sgs-closure/training/modulation.py" \
    --signal const --re-const 200 --dt 2.5e-3 --T 1500 --t-wait 250 \
    --out "$OUT"
ls -la "$(dirname "$OUT")"
echo "[gd1_table] done $(date -u +%FT%TZ)"
