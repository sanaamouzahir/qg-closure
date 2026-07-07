#!/bin/bash
# Gate 1 (charter §3.4 + AMENDMENT_01 §C.3). Authored by sge-runner under
# branch-supervisor direction; TBDs filled 2026-07-07 after the Fable Phase A
# handoff (BRANCH_LOG 1b). Released for submission per Sanaa's resume order.
#
# submit_gate1_smokes.sh - Submits the 5 Gate-1 flow-past-cylinder smoke runs
# (charter §3.4, AMENDMENT_01 §C.1/§C.3) via gate1_job.sh:
#
#   legacy       - no bc.inlet_table, no diag key, bc.inlet_velocity=2.0
#                  (bit-identity baseline for the const-table regression)
#   table_const  - const inlet_table, no diag key
#                  (bit-identity arm; max|Δω| vs legacy must be exactly 0.0)
#   const_rec    - const inlet_table + diag.scalar_rate=10
#                  (recorder smoke: Cd/Cl shedding, U_inlet-vs-table overlay)
#   sine         - sine inlet_table + diag.scalar_rate=10
#   ou           - ou inlet_table   + diag.scalar_rate=10
#
# Depends on the solver hooks (+qg.bc.inlet_table, +qg.diag.scalar_rate —
# LIVE in qg-simple-package-stable since commit 7a743d1's mirror) and on
# gate1_make_tables.sh having produced the const/sine/ou tables under
# $QG_DIR/outputs/SGS_closure_gate1/tables/. The GATE1_RELEASE guard below
# stays as a two-step safety; a SUBMIT email accompanies every release.
#
# Usage (once released):
#   GATE1_RELEASE=1 ./submit_gate1_smokes.sh

set -e

# ---------------------------------------------------------------------- #
# HOLD guard — do not remove without branch-supervisor + Sanaa sign-off. #
# ---------------------------------------------------------------------- #
echo "HOLD: Fable code handoff pending — set GATE1_RELEASE=1 to submit"
[ "${GATE1_RELEASE:-0}" = "1" ] || exit 1

# ---------------------------------------------------------------------- #
# Paths                                                                  #
# ---------------------------------------------------------------------- #
QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
QG_DIR="$QG_ROOT/qg-simple-package-stable/src/qg"
JOB_SCRIPT="$QG_ROOT/qg-sgs-closure/scripts/sge/gate1_job.sh"
TABLES_DIR="$QG_DIR/outputs/SGS_closure_gate1/tables"
RUN_ROOT="outputs/SGS_closure_gate1"   # relative to $QG_DIR (hydra cwd)
LOG_DIR="$QG_DIR/logs"

CONST_TABLE="$TABLES_DIR/const_dt2p5e-4.npz"
SINE_TABLE="$TABLES_DIR/sine_dt2p5e-4.npz"
OU_TABLE="$TABLES_DIR/ou_dt2p5e-4.npz"

if [ -z "${QG_NOTIFY_EMAIL:-}" ]; then
    echo "ERROR: QG_NOTIFY_EMAIL is unset. Ask the supervisor for the notify"
    echo "       address rather than guessing; do not submit without it."
    exit 1
fi

mkdir -p "$LOG_DIR"

echo "==================================================================="
echo "Gate 1 smoke submission (5 runs)"
echo "  job script : $JOB_SCRIPT"
echo "  run root   : $QG_DIR/$RUN_ROOT/<id>"
echo "  tables dir : $TABLES_DIR"
echo "  notify     : $QG_NOTIFY_EMAIL"
echo "==================================================================="
echo

# ---------------------------------------------------------------------- #
# Case definitions: id -> extra hydra overrides (beyond hydra.run.dir)    #
# ---------------------------------------------------------------------- #
CASE_IDS=(legacy table_const const_rec sine ou)

extra_overrides_for () {
    case "$1" in
        legacy)
            echo "qg.bc.inlet_velocity=2.0"
            ;;
        table_const)
            echo "+qg.bc.inlet_table=$CONST_TABLE"
            ;;
        const_rec)
            echo "+qg.bc.inlet_table=$CONST_TABLE +qg.diag.scalar_rate=10"
            ;;
        sine)
            echo "+qg.bc.inlet_table=$SINE_TABLE +qg.diag.scalar_rate=10"
            ;;
        ou)
            echo "+qg.bc.inlet_table=$OU_TABLE +qg.diag.scalar_rate=10"
            ;;
        *)
            echo "ERROR: unknown case id '$1'" >&2
            exit 1
            ;;
    esac
}

SUBMITTED=0
SKIPPED=0

for ID in "${CASE_IDS[@]}"; do
    JOBNAME="sgs_gate1_${ID}"
    OUT_DIR_ABS="$QG_DIR/$RUN_ROOT/$ID"
    LOG="$LOG_DIR/${JOBNAME}.log"

    echo "-----  case = $ID  -----"
    echo "  jobname : $JOBNAME"
    echo "  out_dir : $OUT_DIR_ABS"

    # Existence check / skip-if-done: DNS_FR.npz marks a completed run,
    # per the convention used across scripts/sge (e.g. submit_pi_ff_sweep.sh,
    # submit_decay_sweep_v2.sh).
    if [ -f "$OUT_DIR_ABS/DNS_FR.npz" ]; then
        echo "  [SKIP] $OUT_DIR_ABS/DNS_FR.npz already exists"
        echo
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    # Case sanity: table-consuming cases need their table already generated.
    if [ "$ID" != "legacy" ]; then
        TABLE_PATH=$(extra_overrides_for "$ID" | grep -oE 'bc.inlet_table=[^ ]+' | cut -d= -f2)
        if [ ! -f "$TABLE_PATH" ]; then
            echo "  [SKIP] required table not found: $TABLE_PATH"
            echo "         (run gate1_make_tables.sh first)"
            echo
            SKIPPED=$((SKIPPED + 1))
            continue
        fi
    fi

    read -ra EXTRA_ARGS <<< "$(extra_overrides_for "$ID")"

    echo "  overrides : ${EXTRA_ARGS[*]}"
    echo "  qsub cmd  : qsub -N $JOBNAME -o $LOG -e $LOG -j y -cwd -V"
    echo "              -q ibgpu.q -l gpu=1 -m ea -M \$QG_NOTIFY_EMAIL"
    echo "              $JOB_SCRIPT ${EXTRA_ARGS[*]} hydra.run.dir=$RUN_ROOT/$ID"

    qsub -N "$JOBNAME" -o "$LOG" -e "$LOG" -j y -cwd -V \
         -q ibgpu.q -l gpu=1 -m ea -M "$QG_NOTIFY_EMAIL" \
         "$JOB_SCRIPT" "${EXTRA_ARGS[@]}" hydra.run.dir="$RUN_ROOT/$ID"

    echo
    SUBMITTED=$((SUBMITTED + 1))
done

echo "==================================================================="
echo "Submitted: $SUBMITTED   Skipped: $SKIPPED"
echo
echo "Monitor with:"
echo "  qstat -u \$USER"
echo
echo "Tail any single log:"
echo "  tail -f $LOG_DIR/sgs_gate1_<id>.log"
echo
echo "NOTE: the bit-identity diff (legacy vs table_const, max|Δω| must be"
echo "      0.0), the Cd/Cl + inlet-overlay smoke plots, and the"
echo "      dt-consistency overlay are a SEPARATE analysis step run by the"
echo "      branch supervisor after these jobs complete — not performed by"
echo "      this script."
echo "==================================================================="
