#!/bin/bash
# submit_piff_wave2.sh - Pi_FF s{2,4,8} + Audit-A chain for the four Phase-B
# wave-2 modulated runs (FPC-sine/ramp/ou/tel), mirroring the FPC-const
# landing chain of 2026-07-09 (jobs 1828716-18 -> 1828719).
#
# Per run:
#   piff_s{2,4,8}/  symlink dirs (DNS_FR.npz + DNS_FR_params.yaml -> parent)
#   3x compute_pi_ff_job.sh on ibgpu.q gpu=1 (--scale N --alpha 1.5 --no-videos)
#   piff/           collection symlinks DNS_s{N}_LES.npz -> ../piff_s{N}/DNS_LES.npz
#   audit_A_job.sh on all.q, -hold_jid on the three Pi_FF jobs, --t-min 30.0
#
# Dry-run by default; pass --go to submit. Standing green light: Sanaa
# 2026-07-09 (DECISIONS: all remaining FPC-ensemble steps approved).

set -euo pipefail

QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
BRANCH="$QG_ROOT/qg-sgs-closure"
QG_DIR="$QG_ROOT/qg-simple-package-stable/src/qg"
ENS="$QG_DIR/outputs/SGS_closure_ensemble"
LOGS="$BRANCH/logs"

MODS=(sine ramp ou tel)
SCALES=(2 4 8)

GO=0
[ "${1:-}" = "--go" ] && GO=1

# ---- preflight: every run dir must have landed inputs ------------------- #
for mod in "${MODS[@]}"; do
    RUN="$ENS/FPC-$mod"
    for f in DNS_FR.npz DNS_FR_params.yaml scalars.npz shedding/shedding_summary.npz; do
        [ -e "$RUN/$f" ] || { echo "MISSING: $RUN/$f" >&2; exit 1; }
    done
done
echo "[preflight] all 4 run dirs complete (DNS_FR + params + scalars + shedding)"

if [ "$GO" -ne 1 ]; then
    echo "DRY RUN - would submit 12x Pi_FF (ibgpu.q gpu=1) + 4x Audit-A (all.q, held)."
    echo "Re-run with --go to submit."
    exit 0
fi

cd "$BRANCH"
mkdir -p "$LOGS"

for mod in "${MODS[@]}"; do
    RUN="$ENS/FPC-$mod"
    HOLD_IDS=""

    for s in "${SCALES[@]}"; do
        SDIR="$RUN/piff_s$s"
        mkdir -p "$SDIR"
        ln -sfn ../DNS_FR.npz          "$SDIR/DNS_FR.npz"
        ln -sfn ../DNS_FR_params.yaml  "$SDIR/DNS_FR_params.yaml"

        OUT=$(qsub -q ibgpu.q -l gpu=1 -N "piff_s${s}_$mod" -j y \
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

    qsub -q all.q -N "audA_$mod" -hold_jid "$HOLD_IDS" -j y \
         -o "$LOGS/\$JOB_NAME.\$JOB_ID.log" \
         scripts/sge/audit_A_job.sh \
         --scalars "$RUN/scalars.npz" \
         --shedding-summary "$RUN/shedding/shedding_summary.npz" \
         --piff-dir "$RUN/piff" \
         --outdir "$RUN/audit_A" \
         --t-min 30.0
done

echo "[submit_piff_wave2] all chains submitted."
