#!/bin/bash
# phaseC_make_tables.sh - CPU job generating the Phase-C convergence-tier
# inlet tables: MOD-const at each tier dt (bc rule: direct index lookup, NO
# interpolation -> table dt must equal solver dt to 1e-12; charter S5.3).
#   const_dt1p25e-4_T60.npz   (grid study + finest dt rung)
#   const_dt2p5e-4_T60.npz    (dt study)
#   const_dt5p0e-4_T60.npz    (dt study)
# T=60 from the shared developed-flow IC (true IC time t=29.97, frame 111 —
# NOT exactly 30.0; carried in the run manifests). --t-wait 0: the const
# signal is Re=3900 at every index regardless, and the tier starts from
# developed flow, so no hold window applies.
#
# Pattern: phaseB_make_tables.sh (gate1_make_tables.sh lineage). CPU-only;
# queue all.q. Amendment 02 S3: python runs INSIDE this batch job only.
# Usage:
#   qsub -q all.q -N cnv_tab -o logs/cnv_tab.$JOB_ID.log -j y -cwd -V \
#        scripts/sge/phaseC_make_tables.sh

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

echo "[phaseC_tables] host $HOSTNAME  date $(date -u +%FT%TZ)"

# YAML-trap rule: explicit decimal mantissa on every dt.
for DT in 1.25e-4 2.5e-4 5.0e-4; do
    case "$DT" in
        1.25e-4) TAG=dt1p25e-4 ;;
        2.5e-4)  TAG=dt2p5e-4  ;;
        5.0e-4)  TAG=dt5p0e-4  ;;
    esac
    OUT="$TABLES_DIR/const_${TAG}_T60.npz"
    if [ -f "$OUT" ]; then
        echo "[SKIP] table exists: $OUT"
        continue
    fi
    python -u "$MODULATION_PY" --signal const --dt "$DT" --T 60 --t-wait 0 \
        --out "$OUT"
done

ls -la "$TABLES_DIR"
echo "[phaseC_tables] done $(date -u +%FT%TZ)"
