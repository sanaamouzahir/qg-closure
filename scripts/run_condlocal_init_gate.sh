#!/bin/bash
# run_condlocal_init_gate.sh -- runs the cond_local init acceptance gate.
# MUST execute on a compute node (qrsh/qlogin -q ibgpu.q -l gpu=1); the
# guard hook blocks direct login-node execution of the diagnostic by design.
set -e
QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
source "$QG_ROOT/qg-env/bin/activate"
export MPLBACKEND=Agg
cd "$QG_ROOT/qg-wiener-conditioning/training"
exec python -u "../diagnostics/diagnose_condlocal_init.py" \
    --sweep-roots data/ensemble_N5_7lag/FRC-{256,b25,b2,combo,kf4}/sweep_dT_* \
                  data/ensemble_N5_7lag/FRC-Re25k/sweep_dT_{1em2,5em3} \
    --n-snapshots 7 --grad-kernel 15 --max-val-samples 96 --device cuda "$@"
