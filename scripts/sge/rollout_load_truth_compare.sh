#!/bin/bash
# rollout_load_truth_compare.sh
# Submit the load-truth/compute-closure rollout to the GPU queue, or run
# interactively.  Lives in the same directory as rollout_load_truth_compare.py.
#
# Truth (dt fine) and bare (dt coarse) trajectories are LOADED; only the
# closure rollout is computed.
#
# Usage:
#   ./rollout_load_truth_compare.sh [--interactive] [--jobname NAME] -- <py args>
#   ./rollout_load_truth_compare.sh --jobname rollout_loaded_t60 -- \
#       --run-dir   <run> --root-dir <root> \
#       --truth-omega <...> --truth-times <...> \
#       --bare-omega  <...> --bare-times  <...> \
#       --batch-index 0 --ic-tag restart_t60 --out-dir .

set -euo pipefail

INTERACTIVE=0
JOBNAME=rollout_loaded
PYARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --interactive) INTERACTIVE=1; shift ;;
        --jobname)     JOBNAME="$2"; shift 2 ;;
        --) shift; while [[ $# -gt 0 ]]; do PYARGS+=("$1"); shift; done ;;
        *) PYARGS+=("$1"); shift ;;
    esac
done

# ---- Hard-coded paths ----
QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
VENV=$QG_ROOT/qg-env
QG_DIR=$QG_ROOT/qg-simple-package-stable/src/qg
LOG_DIR=/gdata/projects/ml_scope/Closure_modeling/QG-closure/qg-wiener-conditioning/logs

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT=$HERE/rollout_load_truth_compare.py

mkdir -p "$LOG_DIR"

if [[ ! -f "$SCRIPT" ]]; then
    echo "ERROR: cannot find $SCRIPT" >&2
    exit 1
fi
if [[ ! -f "$VENV/bin/activate" ]]; then
    echo "ERROR: venv not found at $VENV/bin/activate" >&2
    exit 1
fi

if [[ "$INTERACTIVE" == "1" ]]; then
    echo "[rollout] interactive mode"
    source "$VENV/bin/activate"
    cd "$HERE"
    exec python "$SCRIPT" "${PYARGS[@]}"
fi

# Build the qsub job script
JOB_SCRIPT=$(mktemp /tmp/rollout_loaded_job_XXXXXX.sh)
cat > "$JOB_SCRIPT" <<EOF
#!/bin/bash
#\$ -N $JOBNAME
#\$ -q ibgpu.q
#\$ -l gpu=1
#\$ -j y
#\$ -o $LOG_DIR/\$JOB_NAME.\$JOB_ID.log
#\$ -e $LOG_DIR/\$JOB_NAME.\$JOB_ID.err
#\$ -cwd

source $VENV/bin/activate
cd $HERE
python $SCRIPT ${PYARGS[@]}
EOF

chmod +x "$JOB_SCRIPT"
echo "[rollout] submitting $JOBNAME ..."
qsub "$JOB_SCRIPT"
echo "[rollout] log: $LOG_DIR/${JOBNAME}.*.log"
