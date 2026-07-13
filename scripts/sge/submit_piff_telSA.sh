#!/bin/bash
# submit_piff_telSA.sh - Pi_FF s{2,4,8} + Audit-A + Step-0 chain for FPC-telS-A,
# mirroring submit_piff_wave2.sh (FPC-const precedent 1828716-19). telS-A landed
# 2026-07-12 but its Pi_FF landing chain was never fired (gap found 2026-07-13
# during the ensemble-training data audit). Also submits step0 (manifest +
# canonical DNS_LES_s4.npz + U_of_t.npz) held on the Pi_FF jobs — telS-A is the
# 5th member of the FPC ensemble-training pool (Sanaa ORDER 3, 2026-07-13).
#
# Dry-run by default; pass --go to submit. Standing green light: Sanaa
# 2026-07-09 (DECISIONS: all remaining FPC-ensemble steps approved) + the
# 2026-07-13 ORDER 3 (FPC ensemble training on all usable members).

set -euo pipefail

QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
BRANCH="$QG_ROOT/qg-sgs-closure"
QG_DIR="$QG_ROOT/qg-simple-package-stable/src/qg"
RUN="$QG_DIR/outputs/SGS_closure_ensemble/FPC-telS-A"
LOGS="$BRANCH/logs"

SCALES=(2 4 8)

GO=0
[ "${1:-}" = "--go" ] && GO=1

# ---- preflight: landed inputs ------------------------------------------ #
for f in DNS_FR.npz DNS_FR_params.yaml scalars.npz shedding/shedding_summary.npz; do
    [ -e "$RUN/$f" ] || { echo "MISSING: $RUN/$f" >&2; exit 1; }
done
# never-overwrite guard (wave-2 convention)
for s in "${SCALES[@]}"; do
    [ -e "$RUN/piff_s$s/DNS_LES.npz" ] && { echo "EXISTS: $RUN/piff_s$s/DNS_LES.npz — refusing to overwrite" >&2; exit 1; }
done
echo "[preflight] FPC-telS-A inputs complete; no existing Pi_FF products"

if [ "$GO" -ne 1 ]; then
    echo "DRY RUN - would submit 3x Pi_FF (ibgpu.q gpu=1) + Audit-A (all.q, held) + step0 (all.q, held)."
    echo "Re-run with --go to submit."
    exit 0
fi

cd "$BRANCH"
mkdir -p "$LOGS"

HOLD_IDS=""
for s in "${SCALES[@]}"; do
    SDIR="$RUN/piff_s$s"
    mkdir -p "$SDIR"
    ln -sfn ../DNS_FR.npz          "$SDIR/DNS_FR.npz"
    ln -sfn ../DNS_FR_params.yaml  "$SDIR/DNS_FR_params.yaml"

    OUT=$(qsub -q ibgpu.q -l gpu=1 -N "piff_s${s}_teSA" -j y \
          -o "$LOGS/\$JOB_NAME.\$JOB_ID.log" \
          scripts/sge/compute_pi_ff_job.sh "$SDIR" \
          --scale "$s" --alpha 1.5 --device cuda --no-videos)
    echo "$OUT"
    JID=$(echo "$OUT" | grep -oE '[0-9]+' | head -1)
    HOLD_IDS="${HOLD_IDS:+$HOLD_IDS,}$JID"
done

mkdir -p "$RUN/piff"
for s in "${SCALES[@]}"; do
    ln -sfn "../piff_s$s/DNS_LES.npz" "$RUN/piff/DNS_s${s}_LES.npz"
done

qsub -q all.q -N "audA_teSA" -hold_jid "$HOLD_IDS" -j y \
     -o "$LOGS/\$JOB_NAME.\$JOB_ID.log" \
     scripts/sge/audit_A_job.sh \
     --scalars "$RUN/scalars.npz" \
     --shedding-summary "$RUN/shedding/shedding_summary.npz" \
     --piff-dir "$RUN/piff" \
     --outdir "$RUN/audit_A" \
     --t-min 30.0

qsub -q all.q -N "step0_teSA" -hold_jid "$HOLD_IDS" -j y \
     -o "$LOGS/\$JOB_NAME.\$JOB_ID.log" \
     scripts/sge/piff_step0_job.sh "$RUN" --scale 4

echo "[submit_piff_telSA] chain submitted."
