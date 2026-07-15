#!/bin/bash
# submit_reflection_probe.sh -- Sanaa GO 2026-07-15: is the near-wall
# missed-peak error the upstream-reflection problem? One CPU job per geometry
# (worst member each: FPC-const, FPCape-sine). [fable-authored]
# Same v1.4 contract: logs -> <branch>/logs/, digests -> reports/, all.q CPU.
set -euo pipefail
[[ -d /opt/rocks/bin ]] && export PATH=/opt/rocks/bin:$PATH

BRANCH=$(git rev-parse --show-toplevel)
EMAIL=${QG_NOTIFY_EMAIL:-sanaamz@mit.edu}
mkdir -p "$BRANCH/logs"
cd "$BRANCH/ml_closure"

submit_one() {  # $1 geom  $2 member  $3 ratios
    local CKPT="runs_piff/piff_$1_gjs_ylp75/best.pt"
    [[ -e "$CKPT" ]] || { echo "MISSING $CKPT" >&2; exit 1; }
    local JID
    JID=$(qsub -terse -q all.q -m ea -M "$EMAIL" \
        -N "refl_$1" -j y \
        -o "$BRANCH/logs/refl_$1.\$JOB_ID.log" -cwd -V \
        -v "QG_DIGEST_RUN=reflection_probe_$1,OMP_NUM_THREADS=8,MKL_NUM_THREADS=8,OPENBLAS_NUM_THREADS=8" \
        ../scripts/sge/piff_tool_job.sh probe_reflection_hypothesis.py \
        --ckpt "$CKPT" --config "conf_piff_$1_gjs_ylp75.yaml" \
        --member "$2" --ratios "$3" --device cpu \
        --report-run "reflection_probe_$1")
    JID=${JID%%.*}
    echo "refl_$1: job $JID ($2)  raw log $BRANCH/logs/refl_$1.$JID.log"
}

submit_one fpc  FPC-const   "0.7,0.8,0.859,0.9,1.0"
submit_one cape FPCape-sine "0.3,0.5,0.7,0.859,1.0"
echo "verdicts land in reports/reflection_probe_{fpc,cape}/summary.md"
