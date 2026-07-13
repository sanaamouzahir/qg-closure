#!/bin/bash
# gaussian_wait_job.sh - tiny all.q gate job: exits 0 when every listed member
# has its DNS_LES_s4_gaussian.npz (the rebuild fleet 1832329-38 writes them,
# scale 4 FIRST). Trainers -hold_jid on this job, so they start the moment the
# training-scale data exists instead of waiting for the full s2/s8 rebuilds.
# Exits 1 after MAX_WAIT_H hours so a stuck rebuild can't hold GPUs hostage
# silently (the held trainers then die on the hold error -> visible).
#
# Usage: qsub -N gwait_<tag> -q all.q -o <logs>/... -j y -cwd -V \
#             gaussian_wait_job.sh <member_dir> [<member_dir> ...]

#$ -S /bin/bash
#$ -cwd
#$ -V

MAX_WAIT_H=${MAX_WAIT_H:-14}
DEADLINE=$(( $(date +%s) + MAX_WAIT_H * 3600 ))

echo "[gwait] waiting for DNS_LES_s4_gaussian.npz in: $*"
while true; do
    missing=0
    for d in "$@"; do
        [ -s "$d/DNS_LES_s4_gaussian.npz" ] || { missing=$((missing+1)); }
    done
    if [ "$missing" -eq 0 ]; then
        echo "[gwait] all present at $(date -u +%FT%TZ)"
        exit 0
    fi
    if [ "$(date +%s)" -gt "$DEADLINE" ]; then
        echo "[gwait] TIMEOUT after ${MAX_WAIT_H}h with $missing member(s) missing" >&2
        exit 1
    fi
    sleep 120
done
