#!/bin/bash
# submit_step1.sh  (FLOW PAST CYLINDER)
#
# Submit the step1 convergence analysis (decaying turbulence) as a CPU SGE
# job. Two phases:
#
#   PHASE A: convert each DNS_FR.npz -> DNS_FR_omega.npy + DNS_FR_times.npy
#            (one-time preprocessing; mmap-able formats)
#   PHASE B: run step1_convergence_plot.py against the .npy files
#
# By default this script does PHASE A (only if .npy files don't already exist),
# then PHASE B. Use flags to control:
#
#   ./submit_step1.sh                        # auto: convert if needed, then analyze
#   ./submit_step1.sh --analyze-only         # skip conversion
#   ./submit_step1.sh --convert-only         # only do the conversion
#   ./submit_step1.sh --interactive          # run on the login node, all phases
#   ./submit_step1.sh --sweep-root /path/... # override sweep dir
#   ./submit_step1.sh --threshold 0.20 --t-spinup 12.0 --chunk 50
#
# Any other args are forwarded to step1_convergence_plot.py.

set -e

# ---- Defaults ------------------------------------------------------------- #
QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
QG_DIR="$QG_ROOT/qg-simple-package-stable/src/qg"
CONV_DIR="$QG_DIR/Convergence_studies/Decaying_Turbulence/Step1"

JOBNAME="step1_convergence_decay"
SWEEP_ROOT="$QG_DIR/outputs/decaying_turb_dt_sweep"
OUT_DIR="$CONV_DIR/figures"
INTERACTIVE=0
DO_CONVERT=1
DO_ANALYZE=1
FORCE_CONVERT=0

EXTRA_ARGS=()

# ---- Parse flags ---------------------------------------------------------- #
while [ $# -gt 0 ]; do
    case "$1" in
        --interactive)        INTERACTIVE=1; shift ;;
        --analyze-only)       DO_CONVERT=0; shift ;;
        --convert-only)       DO_ANALYZE=0; shift ;;
        --force-convert)      FORCE_CONVERT=1; shift ;;
        --jobname)            JOBNAME="$2"; shift 2 ;;
        --sweep-root)         SWEEP_ROOT="$2"; shift 2 ;;
        --out-dir)            OUT_DIR="$2"; shift 2 ;;
        --threshold)          EXTRA_ARGS+=(--rel-error-threshold "$2"); shift 2 ;;
        --t-spinup)           EXTRA_ARGS+=(--t-spinup "$2"); shift 2 ;;
        --chunk)              EXTRA_ARGS+=(--chunk "$2"); shift 2 ;;
        -h|--help)            sed -n '2,22p' "$0"; exit 0 ;;
        *)                    EXTRA_ARGS+=("$1"); shift ;;
    esac
done

mkdir -p "$OUT_DIR"
LOG_DIR="$QG_ROOT/qg-wiener-conditioning/logs"
mkdir -p "$LOG_DIR"

# ---- Sanity check the sweep root ------------------------------------------ #
if [ ! -d "$SWEEP_ROOT" ]; then
    echo "ERROR: sweep root does not exist:"
    echo "  $SWEEP_ROOT"
    echo "Pass --sweep-root <dir> if your data lives elsewhere."
    exit 1
fi

# Verify either .npz or .npy exists in the expected decaying turbulence subdirs
HAS_DATA=0
for sub in dt_2em3 dt_1em3 dt_5em4 dt_2p5em4 dt_1p25em4 dt_2em5 dt_1em5; do
    if [ -f "$SWEEP_ROOT/$sub/DNS_FR.npz" ] || [ -f "$SWEEP_ROOT/$sub/DNS_FR_omega.npy" ]; then
        HAS_DATA=1
        break
    fi
done
if [ "$HAS_DATA" -eq 0 ]; then
    echo "ERROR: no DNS_FR.{npz,_omega.npy} found in $SWEEP_ROOT/dt_*/"
    echo "Did the decaying turbulence dt sweep complete?"
    exit 1
fi

# ---- Phase A: conversion -------------------------------------------------- #
if [ "$DO_CONVERT" -eq 1 ]; then
    CONVERT_FLAGS=(--sweep-root "$SWEEP_ROOT")
    [ "$FORCE_CONVERT" -eq 1 ] && CONVERT_FLAGS+=(--force)

    if [ "$INTERACTIVE" -eq 1 ]; then
        echo "==========================================================="
        echo "[interactive] PHASE A: prepare_npz_for_mmap.py"
        echo "==========================================================="
        source "$QG_ROOT/qg-env/bin/activate"
        export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
        cd "$CONV_DIR"
        python -u prepare_npz_for_mmap.py "${CONVERT_FLAGS[@]}"
    else
        CONVERT_LOG="$LOG_DIR/${JOBNAME}_convert.log"
        CONVERT_JOB_SCRIPT="$CONV_DIR/prepare_npz_for_mmap_job.sh"
        chmod +x "$CONVERT_JOB_SCRIPT" 2>/dev/null || true

        # Sanity-check the job script before submission. SGE's error message
        # truncates paths and can be misleading when the script doesn't exist.
        if [ ! -f "$CONVERT_JOB_SCRIPT" ]; then
            echo "ERROR: job script not found:"
            echo "  $CONVERT_JOB_SCRIPT"
            echo "Did you copy prepare_npz_for_mmap_job.sh into $CONV_DIR ?"
            exit 1
        fi
        if [ ! -x "$CONVERT_JOB_SCRIPT" ]; then
            echo "ERROR: job script is not executable:"
            echo "  $CONVERT_JOB_SCRIPT"
            echo "Run: chmod +x $CONVERT_JOB_SCRIPT"
            exit 1
        fi
        # Detect Windows-style line endings (CRLF), which silently break
        # bash scripts and cause cryptic 'No such file' errors with qsub.
        if head -1 "$CONVERT_JOB_SCRIPT" | grep -q $'\r'; then
            echo "ERROR: $CONVERT_JOB_SCRIPT has Windows-style (CRLF) line endings."
            echo "Fix with:  sed -i 's/\\r\$//' $CONVERT_JOB_SCRIPT"
            exit 1
        fi

        echo "==========================================================="
        echo "submitting PHASE A (convert npz -> npy)"
        echo "  jobname: ${JOBNAME}_convert"
        echo "  log:     $CONVERT_LOG"
        echo "  script:  $CONVERT_JOB_SCRIPT"
        echo "  args:    ${CONVERT_FLAGS[*]}"
        echo "==========================================================="

        # Decaying turb runs at 256^2 x 1200 frames x float32 ~ 80 MB; pad to 2 GB.
        CONVERT_QSUB_FLAGS=(
            -N "${JOBNAME}_convert"
            -o "$CONVERT_LOG"
            -e "$CONVERT_LOG"
            -j y
            -cwd
            -V
            -q "ibfdr.q"
        )

        CONVERT_QSUB_OUT=$(qsub -terse "${CONVERT_QSUB_FLAGS[@]}" \
            "$CONVERT_JOB_SCRIPT" "${CONVERT_FLAGS[@]}")
        CONVERT_JOB_ID=$(echo "$CONVERT_QSUB_OUT" | head -n 1)
        echo "  submitted: job id $CONVERT_JOB_ID"
        echo
    fi
fi

# ---- Phase B: analysis ---------------------------------------------------- #
if [ "$DO_ANALYZE" -eq 1 ]; then
    ANALYZE_LOG="$LOG_DIR/${JOBNAME}.log"
    ANALYZE_JOB_SCRIPT="$CONV_DIR/step1_convergence_job.sh"
    chmod +x "$ANALYZE_JOB_SCRIPT" 2>/dev/null || true

    if [ "$INTERACTIVE" -eq 0 ]; then
        # Same checks as PHASE A: catch missing-file and CRLF problems before
        # qsub does so with a cryptic message.
        if [ ! -f "$ANALYZE_JOB_SCRIPT" ]; then
            echo "ERROR: job script not found:"
            echo "  $ANALYZE_JOB_SCRIPT"
            echo "Did you copy step1_convergence_job.sh into $CONV_DIR ?"
            exit 1
        fi
        if [ ! -x "$ANALYZE_JOB_SCRIPT" ]; then
            echo "ERROR: job script is not executable:"
            echo "  $ANALYZE_JOB_SCRIPT"
            echo "Run: chmod +x $ANALYZE_JOB_SCRIPT"
            exit 1
        fi
        if head -1 "$ANALYZE_JOB_SCRIPT" | grep -q $'\r'; then
            echo "ERROR: $ANALYZE_JOB_SCRIPT has Windows-style (CRLF) line endings."
            echo "Fix with:  sed -i 's/\\r\$//' $ANALYZE_JOB_SCRIPT"
            exit 1
        fi
    fi

    PYTHON_ARGS=(
        --sweep-root "$SWEEP_ROOT"
        --out-dir    "$OUT_DIR"
        "${EXTRA_ARGS[@]}"
    )

    if [ "$INTERACTIVE" -eq 1 ]; then
        echo "==========================================================="
        echo "[interactive] PHASE B: step1_convergence_plot.py (decaying turbulence)"
        echo "==========================================================="
        source "$QG_ROOT/qg-env/bin/activate"
        export MPLCONFIGDIR="$QG_ROOT/.mplcache"
        mkdir -p "$MPLCONFIGDIR"
        export MPLBACKEND=Agg
        export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
        cd "$CONV_DIR"
        python -u step1_convergence_plot.py "${PYTHON_ARGS[@]}"
        echo
        echo "Results:"
        ls -la "$OUT_DIR"
    else
        echo "==========================================================="
        echo "submitting PHASE B (analyze)"
        echo "  jobname:    $JOBNAME"
        echo "  log:        $ANALYZE_LOG"
        echo "  sweep root: $SWEEP_ROOT"
        echo "  out dir:    $OUT_DIR"
        echo "==========================================================="

        ANALYZE_QSUB_FLAGS=(
            -N "$JOBNAME"
            -o "$ANALYZE_LOG"
            -e "$ANALYZE_LOG"
            -j y
            -cwd
            -V
            
        )
        if [ "$DO_CONVERT" -eq 1 ] && [ -n "${CONVERT_JOB_ID:-}" ]; then
            ANALYZE_QSUB_FLAGS+=(-hold_jid "$CONVERT_JOB_ID")
            echo "  (held until job $CONVERT_JOB_ID finishes)"
        fi

        qsub "${ANALYZE_QSUB_FLAGS[@]}" "$ANALYZE_JOB_SCRIPT" "${PYTHON_ARGS[@]}"
        echo
    fi
fi

# ---- Help text on how to monitor ----------------------------------------- #
if [ "$INTERACTIVE" -eq 0 ]; then
    cat <<EOF
==========================================================
Watch progress with:
EOF
    if [ "$DO_CONVERT" -eq 1 ]; then
        echo "  tail -f $LOG_DIR/${JOBNAME}_convert.log"
    fi
    if [ "$DO_ANALYZE" -eq 1 ]; then
        echo "  tail -f $LOG_DIR/${JOBNAME}.log"
    fi
    cat <<EOF

Check job status:
  qstat -u \$USER

Output figures and CSVs will appear in:
  $OUT_DIR
==========================================================
EOF
fi