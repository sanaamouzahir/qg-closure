#!/bin/bash
# Gate 1 (charter §3.4 + AMENDMENT_01 §C.3). Authored by sge-runner under
# branch-supervisor direction; TBDs filled 2026-07-07 after the Fable Phase A
# handoff (BRANCH_LOG 1b). Released for submission per Sanaa's resume order.
#
# gate1_make_tables.sh - SGE job that generates the U(t)/Re(t) inlet tables
# needed for the Gate 1 smoke tests (charter §3.4, AMENDMENT_01 §C.3):
#   - const, sine, ou @ dt=2.5e-4, T=15           (Gate-1 smoke inputs)
#   - ou, telegraph   @ dt=2.5e-4 AND dt=1.25e-4  (dt-consistency check, §3.2)
#
# Table generator: training/modulation.py, handed off in branch commit
# 9b5b9bf ([fable-authored]); path confirmed 2026-07-07 (BRANCH_LOG 1b).
#
# CLI form per charter §3.2:
#   python modulation.py --signal {const,sine,ramp,ou,telegraph} \
#       --dt DT --T T --t-wait TWAIT --out U_of_t.npz [--seed N]
#
# This job is CPU-only (table generation is cheap arithmetic, no GPU needed).
# Submit as a plain CPU job with no queue flag at all (let SGE pick the
# default queue) and no per-job memory-reservation flag (see project hard
# rules in CLAUDE.md / scripts/sge/CLAUDE.md for the exact forbidden forms —
# not repeated here). If a GPU queue is ever genuinely required, the only
# permitted queue+resource pair on this project is documented in CLAUDE.md.
#
# Usage (HOLD — do not qsub until released):
#   qsub -N sgs_gate1_tables -o logs/sgs_gate1_tables.log -e logs/sgs_gate1_tables.log -j y -cwd -V \
#        gate1_make_tables.sh

#$ -S /bin/bash
#$ -cwd
#$ -V

set -e

# ---- 1. Activate venv & set caches -------------------------------------- #
QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
QG_DIR="$QG_ROOT/qg-simple-package-stable/src/qg"

source "$QG_ROOT/qg-env/bin/activate"
export TMPDIR="$QG_ROOT/tmp"
export PIP_CACHE_DIR="$QG_ROOT/pip-cache"
export MPLCONFIGDIR="$QG_ROOT/mplcache"
export PATH="$QG_ROOT/qg-env/bin-extra:$PATH"
export PYTHONUNBUFFERED=1

# modulation.py lives in the branch worktree (training/, flat — CLAUDE.md
# rule 2), NOT in the solver package. Confirmed at Fable handoff (commit
# 9b5b9bf on exp/sgs-closure).
MODULATION_PY="$QG_ROOT/qg-sgs-closure/training/modulation.py"

# ---- 2. Table generation config ----------------------------------------- #
OUT_DIR="$QG_DIR/outputs/SGS_closure_gate1/tables"
mkdir -p "$OUT_DIR"

DT_FINE=2.5e-4     # Gate-1 solver dt
DT_HALF=1.25e-4    # second dt for the dt-consistency check (§3.2)
T_TOTAL=15
T_WAIT=5.0         # confirmed at Fable handoff (BRANCH_LOG 1b: t-wait 5.0 approved)
SEED=20260707      # per charter §3.1 MOD-ou / MOD-tel; same seed for dt-consistency pair

echo "[gate1_make_tables] hostname: $HOSTNAME"
echo "[gate1_make_tables] date: $(date -u +%FT%TZ)"
if [ ! -f "$MODULATION_PY" ]; then
    echo "[gate1_make_tables] ERROR: modulation.py not found at $MODULATION_PY"
    exit 1
fi
echo "[gate1_make_tables] modulation.py: $MODULATION_PY"
echo "[gate1_make_tables] out_dir: $OUT_DIR"
echo "----------------------------------------------------------------------"

cd "$QG_DIR"

make_table () {
    local signal="$1"
    local dt="$2"
    local dt_tag="$3"
    local seed_args=("${@:4}")
    local out="$OUT_DIR/${signal}_dt${dt_tag}.npz"

    if [ -f "$out" ]; then
        echo "[gate1_make_tables] [SKIP] $out already exists"
        return 0
    fi

    echo "[gate1_make_tables] generating $out"
    echo "[gate1_make_tables] cmd: python -u \"$MODULATION_PY\" --signal $signal --dt $dt --T $T_TOTAL --t-wait $T_WAIT --out $out ${seed_args[*]}"
    python -u "$MODULATION_PY" --signal "$signal" --dt "$dt" --T "$T_TOTAL" \
        --t-wait "$T_WAIT" --out "$out" "${seed_args[@]}"
}

# ---- 3. Gate-1 smoke tables @ dt=2.5e-4, T=15 --------------------------- #
make_table const "$DT_FINE" 2p5e-4
make_table sine  "$DT_FINE" 2p5e-4
make_table ou    "$DT_FINE" 2p5e-4 --seed "$SEED"

# ---- 4. dt-consistency pair: ou + telegraph @ dt=2.5e-4 AND dt=1.25e-4 -- #
# (ou @ 2.5e-4 already made above; reused for the overlay comparison)
make_table telegraph "$DT_FINE" 2p5e-4 --seed "$SEED"
make_table ou        "$DT_HALF" 1p25e-4 --seed "$SEED"
make_table telegraph "$DT_HALF" 1p25e-4 --seed "$SEED"

echo "----------------------------------------------------------------------"
echo "[gate1_make_tables] done at $(date -u +%FT%TZ)"
echo "[gate1_make_tables] tables written under: $OUT_DIR"
echo "[gate1_make_tables] dt-consistency overlay plot (ou, telegraph @ both dt)"
echo "                    is a separate analysis step (charter §3.2/§3.4),"
echo "                    not performed by this job."
