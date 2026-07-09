#!/bin/bash
# ic_extract_job.sh - CPU worker: extract a single-time vorticity IC from a
# landed run's DNS.npy (5D B,T,C,Ny,Nx float64), for the convergence tier's
# shared developed-flow IC (charter S4 / Amendment 02; rule 5: dt sweeps
# restart from a shared developed-flow snapshot, never t=0).
#
# Two chained steps, both float64 end-to-end (repo rule 3):
#   1. training/extract_omega_from_dns_npy.py --run-dir <dir>
#        DNS.npy (B,T,4,Ny,Nx) -> DNS_FR_omega.npy (B,T,Ny,Nx) + DNS_FR_times.npy
#   2. training/extract_restart_ic.py --source DNS_FR_omega.npy
#        --time-index <i> --out <dir>/restart_ic_<tag>.npy  (B,Ny,Nx)
#
# Amendment 02 S3 (absolute): no .py executes on the frontend -- this worker
# is the batch vehicle. CPU-only (memmapped reads; no GPU). Queue: all.q.
# Never the forbidden queue/memory-reservation flags (scripts/sge/CLAUDE.md).
#
# Usage (submit from the branch root so logs land in <branch>/logs/):
#   qsub -q all.q -N icx_<tag> \
#        -o "$PWD/logs/\$JOB_NAME.\$JOB_ID.log" \
#        -e "$PWD/logs/\$JOB_NAME.\$JOB_ID.err" \
#        scripts/sge/ic_extract_job.sh <run-dir> <time-index> <out-basename>

#$ -S /bin/bash
#$ -cwd
#$ -V

set -e

QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
BRANCH="$QG_ROOT/qg-sgs-closure"

source "$QG_ROOT/qg-env/bin/activate"
export TMPDIR="$QG_ROOT/tmp"
export MPLCONFIGDIR="$QG_ROOT/mplcache"
export PYTHONUNBUFFERED=1

RUN_DIR="$1"
TIME_INDEX="$2"
OUT_BASE="$3"

echo "[ic_extract_job] hostname: $HOSTNAME"
echo "[ic_extract_job] date: $(date -u +%FT%TZ)"
echo "[ic_extract_job] run-dir: $RUN_DIR  time-index: $TIME_INDEX  out: $OUT_BASE"
echo "----------------------------------------------------------------------"

# training/ scripts run flat with sibling imports (repo rule 2)
cd "$BRANCH/training"

if [ -f "$RUN_DIR/DNS_FR_omega.npy" ]; then
    echo "[SKIP] $RUN_DIR/DNS_FR_omega.npy exists"
else
    python -u extract_omega_from_dns_npy.py --run-dir "$RUN_DIR"
fi

python -u extract_restart_ic.py \
    --source "$RUN_DIR/DNS_FR_omega.npy" \
    --time-index "$TIME_INDEX" \
    --out "$RUN_DIR/${OUT_BASE}.npy"

# record the exact extracted time from DNS_FR_times.npy for the tier manifest
python -u - "$RUN_DIR" "$TIME_INDEX" "$OUT_BASE" <<'EOF'
import sys, numpy as np
run_dir, idx, base = sys.argv[1], int(sys.argv[2]), sys.argv[3]
t = np.load(f"{run_dir}/DNS_FR_times.npy")
print(f"[ic_extract_job] extracted index {idx} -> t = {t[idx]:.6f} "
      f"(dt_save = {np.median(np.diff(t)):.6f}, T range [{t[0]}, {t[-1]}])")
with open(f"{run_dir}/{base}_manifest.txt", "w") as fh:
    fh.write(f"source=DNS.npy channel=0 (omega) float64\n"
             f"time_index={idx}\nt={t[idx]:.12g}\n")
EOF

echo "----------------------------------------------------------------------"
echo "[ic_extract_job] done at $(date -u +%FT%TZ)"
