#!/bin/bash
# watch_w31_autofire_job.sh - LAPTOP-INDEPENDENT plateau watcher (Sanaa
# 2026-07-14 night: "set up any reporting that can survive even if my laptop
# dies"). SGE-resident version of scripts/watch_w31_plateau.sh: same plateau
# logic, but on PLATEAU+IMPROVED it FIRES submit_w31_p1.sh --go ITSELF and
# queues a pending_mail. Safe to coexist with the in-session watcher:
# submit_w31_p1.sh refuses to double-fire (EXISTS guard on the run dirs).
# Submit: qsub -q all.q -N w31autoF -o logs/w31autoF.$JOB_ID.log -j y -cwd -V \
#         scripts/sge/watch_w31_autofire_job.sh

#$ -S /bin/bash
#$ -cwd
#$ -V

set -uo pipefail

QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
W="$QG_ROOT/qg-wiener-conditioning"
LOG="$W/training/data/ensemble_N5_7lag/training_runs/deriv7_cond_local_w31/log.csv"
PEND="$QG_ROOT/reporting/pending_mail"
REF_NDDOT=0.1375   # cond_v2 plateau val_Nddot mean (go/no-go GATE only)
PLATEAU=6          # epochs without a new best_val (col 5) = the TRIGGER

queue_mail () {  # $1=subject  $2=body
    local f="$PEND/w31_autofire_$(date +%s).mail"
    { echo "To: sanaamz@mit.edu"; echo "Subject: $1"; echo; echo "$2"; } > "$f"
}

for i in $(seq 1 192); do   # 192 x 15 min = 48 h
    if ! qstat -u "$USER" 2>/dev/null | grep -q 1833569; then
        queue_mail "[QG][TRIAGE][wiener] w31 trainer 1833569 GONE before plateau -- autofire watcher exiting" \
            "$(tail -3 "$LOG" 2>/dev/null)"
        exit 2
    fi
    if [ -f "$LOG" ]; then
        n=$(($(wc -l < "$LOG") - 1))
        if [ "$n" -ge 10 ]; then
            stale=$(awk -F, 'NR>1{if($5!=prev){prev=$5;last=$1}} END{print $1-last}' "$LOG")
            best_nddot=$(awk -F, 'NR>1&&$7<m||NR==2{m=$7} END{print m}' "$LOG")
            improved=$(awk -v b="$best_nddot" -v r="$REF_NDDOT" 'BEGIN{print (b<r*0.98)?1:0}')
            if [ "$stale" -ge "$PLATEAU" ]; then
                ep=$(awk -F, 'END{print $1}' "$LOG")
                if [ "$improved" = "1" ]; then
                    cd "$W"
                    out=$(bash scripts/sge/submit_w31_p1.sh --go 2>&1) || {
                        queue_mail "[QG][TRIAGE][wiener] w31 PLATEAU at ep $ep but auto-fire FAILED (likely already fired by the in-session watcher -- fine)" "$out"
                        exit 0
                    }
                    queue_mail "[QG][SUBMIT][wiener] w31 PLATEAU at ep $ep (best val_Nddot $best_nddot < gate $REF_NDDOT) -- ANCHORED ARMS AUTO-FIRED (laptop-independent watcher)" \
"PARAMETERS: two arms w31p1a (anchor 3e-2) / w31p1b (anchor 3e-1), p1-prod recipe, warm w31 best.pt, FIXED certificate, ~6-9 GPU-h each.

$out

Last log rows:
$(tail -3 "$LOG")"
                    exit 0
                else
                    queue_mail "[QG][REPORT][wiener] w31 plateaued at ep $ep WITHOUT beating cond_v2 gate ($best_nddot >= $REF_NDDOT) -- NOT firing; decision needed" \
                        "$(tail -5 "$LOG")"
                    exit 3
                fi
            fi
        fi
    fi
    sleep 900
done
queue_mail "[QG][TRIAGE][wiener] w31 autofire watcher: 48h without plateau -- resubmit me" ""
exit 4
