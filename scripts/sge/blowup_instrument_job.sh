#!/bin/bash
# blowup_instrument_job.sh -- INSTRUMENT ONE BLOWING RUN (Sanaa GO 2026-07-19).
#
# WHY: p3 drove the frozen-coefficient von Neumann certificate from |G_eff|~1.7
# to ~0.96 on developed states and the a-posteriori blowup moved only +1..+3
# steps (kf4 ic532 42->45, ic837 37->38, ic912 35->37; the bare arm stays clean
# at CFL 0.26 while the closure arm hits CFL 4.2). The 2026-07-13 dissipative
# projection (removing radial enstrophy injection) also moved blowup by <=1
# step. Two interventions aimed at LINEAR/spectral mechanisms, two ~1-step
# nudges: we do not know the mechanism. No further training arms until we do.
#
# WHAT: rollout_aposteriori.py --instrument-blowup records, EVERY step and per
# integer mode-radius shell, the state enstrophy Z_k, the APPLIED closure
# correction split into analytic / NN / implicit-L^3 (Ca_k / Cn_k / Ci_k), and
# the bare rhs increment R_k. diagnostics/analyze_blowup_modes.py then reports
# WHICH modes grow, AT WHAT RATE, FROM WHAT STEP, and whether the correction
# LEADS or LAGS the growth. The BARE arm is instrumented in the same pass as
# the CONTROL (it does not blow up -- its growth signature is the baseline).
#
# CPU only (Sanaa standing order 2026-07-16: no diagnostics/rollouts on GPU).
# NaN policy: this job's SUBJECT is a blowup, so non-finite state is the
# expected outcome, not a failure -- there is nothing to abort. The driver's
# own blowup detection stops the arm; the analyzer runs regardless and the
# report mail is spooled either way (that IS the STOP>CHECK reaction here).
#
# Usage (defaults = the recommended draw: FRC-kf4 ic912, blows at 37):
#   qsub -q all.q -N blowup_instr scripts/sge/blowup_instrument_job.sh \
#       [MEMBER] [IC] [HORIZON]
# [fable-authored 2026-07-20]
#$ -S /bin/bash
#$ -q all.q
#$ -cwd
#$ -j y
#$ -o /gdata/projects/ml_scope/Closure_modeling/QG-closure/qg-wiener-conditioning/logs/
#$ -e /gdata/projects/ml_scope/Closure_modeling/QG-closure/qg-wiener-conditioning/logs/
#$ -m ea
#$ -M sanaamz@mit.edu
set -uo pipefail
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg

QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
BRANCH=$QG_ROOT/qg-wiener-conditioning
REP=$BRANCH/diagnostics/Results/apost_opt2_rep_20260711
D=$QG_ROOT/qg-simple-package-stable/src/qg/training/data/ensemble_N5_7lag/training_runs
CKPT=${BLOWUP_CKPT:-$D/rollout_ft_w31p3_certv2/best.pt}
SP=$QG_ROOT/reporting/pending_mail
DIGEST="$BRANCH/diagnostics/digest_writer.py"
RUN_NAME=blowup_instrument

MEMBER=${1:-FRC-kf4}
IC=${2:-912}
# Horizon 64 (not 45): the saved truth refs exist on the 64-step / 24-checkpoint
# grid, and --load-refs HARD-FAILS on a grid mismatch. The closure arm blows at
# ~37 and stops itself there, so the extra budget costs nothing on that arm; the
# bare control runs the full 64 and gives a longer clean baseline.
H=${3:-64}

OUT=${BLOWUP_OUT:-$BRANCH/diagnostics/Results/blowup_instrument_20260720}
CD=$OUT/${MEMBER}_ic${IC}
NPZ=$CD/blowup_instr_${MEMBER}_ic${IC}_5em3_h${H}.npz
TXT=$CD/blowup_modes_${MEMBER}_ic${IC}.txt
PNG=$CD/blowup_modes_${MEMBER}_ic${IC}.png
RH=$REP/${MEMBER}_ic${IC}/apost_refs_ic${IC}_5em3_h${H}.npz

export MPLCONFIGDIR=$QG_ROOT/.mplcache; mkdir -p "$MPLCONFIGDIR"
mkdir -p "$CD" "$SP"

digest_event() {  # I23b; no-op if digest_writer is not on this checkout
    [[ -f "$DIGEST" && -n "$RUN_NAME" ]] && \
        python "$DIGEST" --repo-dir "$BRANCH" --run-name "$RUN_NAME" \
            --event "$1" --job-id "${JOB_ID:-}" --note "$2" || true
}
trap 'digest_event fail "blowup instrumentation exited rc=$?"' ERR

source "$QG_ROOT/qg-env/bin/activate"
cd "$BRANCH/training"
echo "[blowup] host=$HOSTNAME date=$(date -u +%FT%TZ) member=$MEMBER ic=$IC h=$H"
echo "[blowup] ckpt=$CKPT"
[ -f "$CKPT" ] || { echo "[blowup] HARD-FAIL: ckpt $CKPT not found"; \
    digest_event fail "ckpt missing"; exit 4; }
[ -f "$RH" ] || { echo "[blowup] HARD-FAIL: refs $RH not found -- truth is \
NEVER recomputed on CPU"; digest_event fail "refs missing"; exit 4; }
digest_event start "instrument $MEMBER ic$IC h$H, arms bare(control)+closure, CPU"

# ---- 1. instrumented rollout: closure (blows) + bare (control) ------------ #
python -u rollout_aposteriori.py \
    --root-dir "data/ensemble_N5_7lag/$MEMBER/sweep_dT_5em3" \
    --ckpt "$CKPT" \
    --ic-index "$IC" --K 500 --n-steps "$H" --n-checkpoints 24 \
    --arms bare,closure \
    --device cpu --out-dir "$CD" \
    --tag "ic${IC}_5em3_h${H}_blowupinstr" \
    --load-refs "$RH" \
    --instrument-blowup "$NPZ" || echo "[blowup] ROLL_RC=$? (arm blowup is expected; continuing)"

[ -f "$NPZ" ] || { echo "[blowup] HARD-FAIL: no instrumentation npz written"; \
    digest_event fail "no npz"; exit 5; }

# ---- 2. mechanism analysis (onset table + lead/lag + verdict + figure) ---- #
python -u ../diagnostics/analyze_blowup_modes.py \
    --npz "$NPZ" --arm closure --control-arm bare \
    --g-thresh 0.05 --sustain 3 --fig "$PNG" > "$TXT" 2>&1 \
    || echo "[blowup] ANALYZE_RC=$? (see $TXT)"
cat "$TXT"

# ---- 3. [QG][REPORT] mail: plain English, table inline, directory list ---- #
{
  echo "To: ${QG_NOTIFY_EMAIL:-sanaamz@mit.edu}"
  echo "Subject: [QG][REPORT][wiener-conditioning] blowup mechanism -- ${MEMBER} ic${IC}: onset table + verdict"
  echo
  echo "WHAT THIS IS"
  echo "One blowing a-posteriori run, instrumented per step and per radial"
  echo "shell, with the non-blowing bare arm recorded in the same pass as the"
  echo "control. This answers WHICH modes grow, HOW FAST, and FROM WHAT STEP --"
  echo "the question left open after two linear/spectral interventions (the p3"
  echo "von Neumann certificate, |G_eff| 1.7 -> 0.96, and the 2026-07-13"
  echo "dissipative projection) each moved blowup by only 1-3 steps. No new"
  echo "training arm is proposed until the mechanism is named."
  echo
  echo "RUN: member=$MEMBER ic=$IC dT=5e-3 K=500 horizon=$H steps, arms"
  echo "bare(control)+closure, ckpt $(basename "$(dirname "$CKPT")")/$(basename "$CKPT"),"
  echo "truth refs REUSED from $RH (never recomputed on CPU)."
  echo
  echo "The full analyzer output -- ONSET TABLE, LEAD/LAG, and the VERDICT"
  echo "block -- is inline below."
  echo
  echo "================================================================"
  cat "$TXT"
  echo
  echo "DIRECTORIES / FILES"
  echo "  results dir : $CD"
  echo "  instrument  : $NPZ"
  echo "  analysis    : $TXT"
  echo "  figure      : $PNG"
  echo "  rollout npz : $CD/rollout_apost_ic${IC}_5em3_h${H}_blowupinstr.npz"
  echo "  job log     : $BRANCH/logs/${JOB_NAME:-blowup_instrument}.o${JOB_ID:-}"
  echo
  echo "REPRODUCE"
  echo "  qsub -q all.q -N blowup_instr scripts/sge/blowup_instrument_job.sh $MEMBER $IC $H"
} > "$SP/$(date +%Y%m%dT%H%M%S)_blowup_instrument_${MEMBER}_ic${IC}.mail"

digest_event done "instrumented $MEMBER ic$IC; onset table + verdict mailed"
echo "[blowup] done $(date -u +%FT%TZ) -- report mail spooled to $SP"
