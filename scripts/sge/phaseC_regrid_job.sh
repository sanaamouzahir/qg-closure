#!/bin/bash
# phaseC_regrid_job.sh - CPU job producing the Phase-C shared-IC set by
# spectral regrid of the FPC-const t=29.97 restart IC (charter S5.1; Sanaa
# 2026-07-08 five-grid directive adds 256^2 and 512^2 coarse anchors).
#   1. spectral_regrid.py --self-test on the 2048^2 source: prints the
#      charter-required 2048 -> 4096 -> 2048 round-trip number (machine
#      precision expected) + the source Nyquist content. Goes in the
#      [QG][CONV][SGS-CLOSURE] report.
#   2. 2048 -> {256, 512, 1024} spectral truncation (dropped-energy fraction
#      printed per target), 2048 -> 4096 spectral zero-padding.
# The two 2048^2 tier runs consume the SOURCE file directly (no regrid).
# All ICs are float64 (rule 3); ~145 MB total.
#
# CPU-only; queue all.q. Amendment 02 S3: python runs INSIDE this batch job.
# Usage:
#   qsub -q all.q -N cnv_ics -o logs/cnv_ics.$JOB_ID.log -j y -cwd -V \
#        scripts/sge/phaseC_regrid_job.sh

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

REGRID_PY="$QG_ROOT/qg-sgs-closure/training/spectral_regrid.py"
SRC="$QG_DIR/outputs/SGS_closure_ensemble/FPC-const/restart_ic_t30.npy"
ICS_DIR="$QG_DIR/outputs/SGS_closure_ensemble/convergence/ics"

[ -f "$REGRID_PY" ] || { echo "ERROR: $REGRID_PY missing"; exit 1; }
[ -f "$SRC" ]       || { echo "ERROR: shared IC missing: $SRC"; exit 1; }
mkdir -p "$ICS_DIR"

echo "[phaseC_regrid] host $HOSTNAME  date $(date -u +%FT%TZ)"
echo "[phaseC_regrid] source: $SRC (true time t=29.97, frame 111 — see manifest)"

# 1. Charter round-trip check (2048 -> 4096 -> 2048), reported number.
python -u "$REGRID_PY" --source "$SRC" --self-test

# 2. Tier ICs. Skip-if-done per file (never silently overwrite).
for N in 256 512 1024 4096; do
    OUT="$ICS_DIR/ic_t29p97_N${N}.npy"
    if [ -f "$OUT" ]; then
        echo "[SKIP] IC exists: $OUT"
        continue
    fi
    python -u "$REGRID_PY" --source "$SRC" --out "$OUT" --N-out "$N"
done

# Provenance note (plain shell; no python on top of the above).
cat > "$ICS_DIR/README.txt" <<EOF
Phase-C convergence-tier shared ICs (charter S5.1 + 5-grid directive 2026-07-08)
source: $SRC
  true time t=29.97 (frame 111; NOT exactly 30.0 — see restart_ic_t30_manifest.txt)
method: training/spectral_regrid.py — spectral truncation (2048->256/512/1024),
  spectral zero-padding (2048->4096); rfft2, Nyquist row/col zeroed both ways.
2048^2 tier runs consume the source file directly.
Round-trip 2048->4096->2048 + dropped-energy fractions: see cnv_ics job log.
Generated: $(date -u +%FT%TZ) by job ${JOB_ID:-manual} on $HOSTNAME
EOF

ls -la "$ICS_DIR"
echo "[phaseC_regrid] done $(date -u +%FT%TZ)"
