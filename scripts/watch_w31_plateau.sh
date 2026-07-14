#!/bin/bash
# watch_w31_plateau.sh - wake the supervisor when deriv7_cond_local_w31 val
# plateaus (Sanaa 2026-07-14: fire rollout+vN as soon as val plateaus below
# cond_v2; don't wait for 150 ep). Exit codes: 0 plateau+improved (FIRE),
# 2 trainer gone (triage), 3 plateau without improvement (REPORT, no fire).
LOG=/gdata/projects/ml_scope/Closure_modeling/QG-closure/qg-wiener-conditioning/training/data/ensemble_N5_7lag/training_runs/deriv7_cond_local_w31/log.csv
REF_NDDOT=0.1375   # cond_v2 plateau val_Nddot (mean); warm start begins here
PLATEAU=6          # epochs without a new best
for i in $(seq 1 192); do   # 192 x 15 min = 48 h
    if ! qstat -u $USER 2>/dev/null | grep -q 1833569; then
        echo "TRAINER GONE"; tail -3 "$LOG" 2>/dev/null; exit 2
    fi
    if [ -f "$LOG" ]; then
        n=$(($(wc -l < "$LOG") - 1))
        if [ "$n" -ge 10 ]; then
            stale=$(awk -F, 'NR>1{if($5!=prev){prev=$5;last=$1}} END{print $1-last}' "$LOG")
            best_nddot=$(awk -F, 'NR>1&&$7<m||NR==2{m=$7} END{print m}' "$LOG")
            improved=$(awk -v b="$best_nddot" -v r="$REF_NDDOT" 'BEGIN{print (b<r*0.98)?1:0}')
            if [ "$stale" -ge "$PLATEAU" ]; then
                echo "PLATEAU at epoch $(awk -F, 'END{print $1}' "$LOG"), stale=$stale, best val_Nddot=$best_nddot (ref $REF_NDDOT)"
                tail -3 "$LOG"
                [ "$improved" = "1" ] && exit 0 || exit 3
            fi
        fi
    fi
    sleep 900
done
echo "48h without plateau"; exit 4
