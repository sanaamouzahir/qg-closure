#!/bin/bash
# restart_decaying_turb.sh
# 1. Extract last frame from the existing dt=2e-5 DNS run as a restart IC.
# 2. Submit a new continuation run starting from that IC.
#
# What you should expect from this run:
#   - Vortex merger continues; by t=100 (60 baseline + 40 here) the field
#     should look noticeably "coarser" -- fewer, larger coherent vortices.
#   - If we want the "single big vortex" final state we may need T~500-1000
#     more; cf. ν*k_typical^2 ~ 1e-3 timescale.
set -e

QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
QG_DIR="$QG_ROOT/qg-simple-package-stable/src/qg"
SOURCE_DIR="$QG_DIR/outputs/decaying_turb_dt_sweep_float64/dt_2em5"

# Step 1: extract IC from last time index of DNS_FR_omega.npy
IC_PATH="$SOURCE_DIR/restart_ic_t60.npy"
if [ ! -f "$IC_PATH" ]; then
    echo "[restart] extracting last frame as IC..."
    python "$QG_DIR/extract_restart_ic.py" \
        --source "$SOURCE_DIR/DNS_FR_omega.npy" \
        --out    "$IC_PATH"
else
    echo "[restart] IC already exists: $IC_PATH"
fi

# Step 2: submit the continuation run.
# Output goes to a NEW directory so we don't clobber the original.
RUN_DIR="$QG_DIR/outputs/decaying_turb_restart_t60_dt1em5"
mkdir -p "$RUN_DIR"

# Use the project's submit_qg.sh wrapper.  Override the scenario YAML and
# the output dir.  Adjust the flags below to match how your submit_qg.sh
# parses arguments -- this assumes Hydra-style scenario override.
cd "$QG_DIR/submit_scripts/Decaying_Turbulence" || cd "$QG_DIR"
"$QG_DIR/submit_qg.sh" \
    --scenario decaying_turbulence_restart \
    --output-dir "$RUN_DIR" \
    --jobname    qg_restart_t60_dt1em5

echo
echo "Restart submitted. Output: $RUN_DIR"
echo "Watch with: tail -f $QG_DIR/logs/qg_restart_t60_dt1em5.log"
