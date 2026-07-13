#!/bin/bash
# submit_piff_capeA.sh - Pi_FF s{2,4,8} + Audit-A chain for the 5 landed
# CAPE-A wave runs (FPCape-const/sine/ramp/ou/tel), mirroring the FPC
# wave-2 landing chain (submit_piff_wave2.sh, jobs 1829661-1829671 etc.,
# 2026-07-10).
#
# Deviation from FPC wave-2 precedent: with 5 members x 3 scales = 15
# Pi_FF jobs, running all scales concurrently per member would put up to
# 15 GPU jobs on ibgpu.q at once. Per task ceiling (max 6 concurrent GPU
# jobs), scales are chained SEQUENTIALLY per member (s2 -> s4 -> s8 via
# -hold_jid), giving at most 5 concurrent GPU jobs (one per member).
# Audit-A is held on the s8 job id (last link in the per-member chain);
# since s8 itself depends on s4 depends on s2, this is equivalent to (and
# required by, since audit_A reads all 3 scales from piff/) holding on
# all three scale jobs -- which is what the FPC wave-2 precedent actually
# did (confirmed via qacct start-time inspection of 1829668-71: audA_ou
# started only after ALL THREE of its scale jobs had finished).
#
# Per run:
#   piff_s{2,4,8}/  symlink dirs (DNS_FR.npz + DNS_FR_params.yaml -> parent)
#   3x compute_pi_ff_job.sh on ibgpu.q gpu=1, chained s2->s4->s8
#   piff/           collection symlinks DNS_s{N}_LES.npz -> ../piff_s{N}/DNS_LES.npz
#   audit_A_job.sh on all.q, -hold_jid on the member's s8 job, --t-min 30.0
#
# Dry-run by default; pass --go to submit.

set -euo pipefail

QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
BRANCH="$QG_ROOT/qg-sgs-closure"
QG_DIR="$QG_ROOT/qg-simple-package-stable/src/qg"
ENS="$QG_DIR/outputs/SGS_closure_ensemble"
LOGS="$BRANCH/logs"

# member -> short tag (unique in qstat's 10-char name column)
declare -A TAGS=( [const]=CAco [sine]=CAsi [ramp]=CAra [ou]=CAou [tel]=CAte )
MODS=(const sine ramp ou tel)
SCALES=(2 4 8)

GO=0
[ "${1:-}" = "--go" ] && GO=1

# ---- preflight: every run dir must have landed inputs, no pre-existing LES ----
for mod in "${MODS[@]}"; do
    RUN="$ENS/FPCape-$mod"
    for f in DNS_FR.npz DNS_FR_params.yaml scalars.npz shedding/shedding_summary.npz; do
        [ -e "$RUN/$f" ] || { echo "MISSING: $RUN/$f" >&2; exit 1; }
    done
    if find "$RUN" -iname "DNS_LES.npz" | grep -q .; then
        echo "REFUSING: $RUN already has a DNS_LES.npz" >&2
        exit 1
    fi
done
echo "[preflight] all 5 FPCape run dirs complete; no pre-existing DNS_LES.npz"

if [ "$GO" -ne 1 ]; then
    echo "DRY RUN - would submit 15x Pi_FF (ibgpu.q gpu=1, chained s2->s4->s8 per member)"
    echo "          + 5x Audit-A (all.q, held on each member's s8 job)."
    echo "Re-run with --go to submit."
    exit 0
fi

cd "$BRANCH"
mkdir -p "$LOGS"

for mod in "${MODS[@]}"; do
    RUN="$ENS/FPCape-$mod"
    TAG="${TAGS[$mod]}"
    PREV_HOLD=""
    S8_JID=""

    for s in "${SCALES[@]}"; do
        SDIR="$RUN/piff_s$s"
        mkdir -p "$SDIR"
        ln -sfn ../DNS_FR.npz          "$SDIR/DNS_FR.npz"
        ln -sfn ../DNS_FR_params.yaml  "$SDIR/DNS_FR_params.yaml"

        HOLD_FLAGS=()
        [ -n "$PREV_HOLD" ] && HOLD_FLAGS=(-hold_jid "$PREV_HOLD")

        OUT=$(qsub -q ibgpu.q -l gpu=1 -N "piff${s}_${TAG}" "${HOLD_FLAGS[@]+"${HOLD_FLAGS[@]}"}" -j y \
              -o "$LOGS/\$JOB_NAME.\$JOB_ID.log" \
              scripts/sge/compute_pi_ff_job.sh "$SDIR" \
              --scale "$s" --alpha 1.5 --device cuda --no-videos)
        echo "$OUT"
        JID=$(echo "$OUT" | grep -oE '[0-9]+' | head -1)
        PREV_HOLD="$JID"
        [ "$s" = "8" ] && S8_JID="$JID"
    done

    mkdir -p "$RUN/piff"
    for s in "${SCALES[@]}"; do
        ln -sfn "../piff_s$s/DNS_LES.npz" "$RUN/piff/DNS_s${s}_LES.npz"
    done

    qsub -q all.q -N "audA_${TAG}" -hold_jid "$S8_JID" -j y \
         -o "$LOGS/\$JOB_NAME.\$JOB_ID.log" \
         scripts/sge/audit_A_job.sh \
         --scalars "$RUN/scalars.npz" \
         --shedding-summary "$RUN/shedding/shedding_summary.npz" \
         --piff-dir "$RUN/piff" \
         --outdir "$RUN/audit_A" \
         --t-min 30.0
done

echo "[submit_piff_capeA] all chains submitted."
