#!/bin/bash
# rollout_multistep_comparison.sh
# Submit the multi-step rollout to the GPU queue, or run interactively.
# Lives in the same directory as rollout_multistep_comparison.py.

set -euo pipefail

INTERACTIVE=0
JOBNAME=rollout_multistep
PYARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --interactive) INTERACTIVE=1; shift ;;
        --jobname)     JOBNAME="$2"; shift 2 ;;
        *) PYARGS+=("$1"); shift ;;
    esac
done

# ---- Hard-coded paths ----
QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
VENV=$QG_ROOT/qg-env
QG_DIR=$QG_ROOT/qg-simple-package-stable/src/qg
LOG_DIR=$QG_DIR/logs

# Resolve SCRIPT next to this shell script (works whether invoked from
# plotting/trainvD/ or anywhere else).
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT=$HERE/rollout_multistep_comparison.py

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
JOB_SCRIPT=$(mktemp /tmp/rollout_job_XXXXXX.sh)
cat > "$JOB_SCRIPT" <<EOF
#!/bin/bash
#\$ -N $JOBNAME
#\$ -q ibgpu.q
#\$ -l gpu=1
#\$ -j y
#\$ -o $LOG_DIR/${JOBNAME}.log
#\$ -cwd

source $VENV/bin/activate
cd $HERE
python $SCRIPT ${PYARGS[@]}
EOF

chmod +x "$JOB_SCRIPT"
echo "[rollout] submitting $JOBNAME ..."
qsub "$JOB_SCRIPT"
echo "[rollout] log: $LOG_DIR/${JOBNAME}.log"