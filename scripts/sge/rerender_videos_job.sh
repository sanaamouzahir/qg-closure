#!/bin/bash
# rerender_videos_job.sh - CPU worker: re-render + re-encode FR videos for
# one or more save_path dirs under outputs/SGS_closure_ensemble/ via
# rerender_videos.py (jpcm.draw for frames, ffmpeg for H.264/yuv420p
# re-encode). CPU-only, no GPU. Queue: all.q. Never the forbidden
# queue/memory-reservation flags (scripts/sge/CLAUDE.md).
#
# CRITICAL TRAP (do not remove this block): rerender_videos.py silently
# SKIPS the re-encode step if `ffmpeg` is not literally on PATH
# (shutil.which check, no error, no nonzero exit) -- this is exactly how
# the previously-unplayable videos happened. This job therefore explicitly
# prepends qg-env/bin-extra (imageio_ffmpeg's static ffmpeg binary,
# symlinked as `ffmpeg`) to PATH and prints `which ffmpeg` + `ffmpeg
# -version` into the log before doing anything else, so a silent-skip is
# visible in the log even if it recurs.
#
# After each save_path, every {name}*.mp4 actually present is verified by
# parsing `ffmpeg -i <file>` stderr for "Video: h264" and "yuv420p" (no
# ffprobe shipped alongside imageio_ffmpeg) and the verdict is printed.
#
# Usage (submit from qg-sgs-closure root so logs land in logs/):
#   qsub -q all.q -N rerender_<tag> \
#        -o "$PWD/logs/\$JOB_NAME.\$JOB_ID.log" -j y -cwd -V \
#        scripts/sge/rerender_videos_job.sh [--name DNS] [--fps 20] [--clamp 0.3] \
#        <save_path_1> [<save_path_2> ...]
#
# save_path args and rerender_videos.py flags (--name/--fps/--clamp/etc.)
# may be interleaved in any order; anything starting with `outputs/` or
# `/` or matching an existing directory is treated as a save_path, all
# other leading `--flag value` pairs are forwarded to rerender_videos.py
# for EVERY save_path in this invocation.

#$ -S /bin/bash
#$ -cwd
#$ -V

set -e

QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
QG_DIR="$QG_ROOT/qg-simple-package-stable/src/qg"
RERENDER="$QG_DIR/rerender_videos.py"
QG_OUT="$QG_DIR/outputs/SGS_closure_ensemble"

source "$QG_ROOT/qg-env/bin/activate"
# Defensive: explicitly prepend the known-good ffmpeg dir, do not rely on
# an inherited PATH (the -V export is a login-shell artifact, not a
# guarantee for every future caller of this script).
export PATH="$QG_ROOT/qg-env/bin-extra:$PATH"
export TMPDIR="$QG_ROOT/tmp"
export MPLCONFIGDIR="$QG_ROOT/mplcache"
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

echo "[rerender_videos_job] hostname: $HOSTNAME"
echo "[rerender_videos_job] date: $(date -u +%FT%TZ)"
echo "[rerender_videos_job] args: $*"
echo "[rerender_videos_job] which ffmpeg: $(which ffmpeg || echo NOT_FOUND)"
ffmpeg -version 2>&1 | head -1
echo "----------------------------------------------------------------------"

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "[rerender_videos_job] FATAL: ffmpeg not on PATH -- aborting before" \
         "any silent-skip re-encodes can happen."
    exit 1
fi

# ---- split argv into rerender flags vs save_path targets ---------------- #
FLAGS=()
PATHS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --*)
            FLAGS+=("$1" "$2"); shift 2
            ;;
        *)
            if [[ "$1" = /* ]]; then
                PATHS+=("$1")
            else
                PATHS+=("$QG_OUT/$1")
            fi
            shift
            ;;
    esac
done

if [ ${#PATHS[@]} -eq 0 ]; then
    echo "[rerender_videos_job] ERROR: no save_path arguments given"
    exit 1
fi

echo "[rerender_videos_job] flags forwarded: ${FLAGS[*]:-<none>}"
echo "[rerender_videos_job] save_paths (${#PATHS[@]}): ${PATHS[*]}"
echo "----------------------------------------------------------------------"

verify_mp4() {
    local f="$1"
    if [ ! -f "$f" ]; then
        return
    fi
    local out
    out="$(ffmpeg -i "$f" 2>&1 || true)"
    local h264 yuv420
    h264=$(echo "$out" | grep -c "Video: h264")
    yuv420=$(echo "$out" | grep -c "yuv420p")
    if [ "$h264" -ge 1 ] && [ "$yuv420" -ge 1 ]; then
        echo "  [verify OK]   $f  (h264, yuv420p, mtime=$(date -r "$f" +%FT%T))"
    else
        echo "  [verify FAIL] $f  -- codec line: $(echo "$out" | grep 'Video:' | head -1)"
    fi
}

N_OK=0
N_FAIL=0
for sp in "${PATHS[@]}"; do
    echo "===== $sp ====="
    if [ ! -d "$sp" ]; then
        echo "  [skip] directory does not exist: $sp"
        N_FAIL=$((N_FAIL + 1))
        continue
    fi
    if python -u "$RERENDER" "$sp" "${FLAGS[@]}"; then
        echo "  [render] rerender_videos.py exited 0"
    else
        echo "  [render] rerender_videos.py FAILED (nonzero exit)"
        N_FAIL=$((N_FAIL + 1))
        continue
    fi
    for name_base in DNS DNS_clamped DNS_seismic; do
        verify_mp4 "$sp/${name_base}.mp4"
    done
    N_OK=$((N_OK + 1))
    echo
done

echo "----------------------------------------------------------------------"
echo "[rerender_videos_job] done at $(date -u +%FT%TZ)  ok=$N_OK fail=$N_FAIL"
