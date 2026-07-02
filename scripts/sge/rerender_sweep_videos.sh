#!/bin/bash
# rerender_sweep_videos.sh - Re-render all FR videos in a sweep directory.
#
# Walks every immediate subdirectory under <sweep-root> and runs
# rerender_videos.py on each one that contains the expected data file
# (default: DNS.npy or DNS_FR.npz). Subdirs without simulation output
# are skipped silently.
#
# Usage:
#   ./rerender_sweep_videos.sh <sweep-root> [extra args forwarded to rerender_videos.py]
#
# Examples:
#   ./rerender_sweep_videos.sh outputs/cylinder_dt_sweep_re200_v2
#   ./rerender_sweep_videos.sh outputs/decaying_turb_dt_sweep
#   ./rerender_sweep_videos.sh outputs/forced_turb_dt_sweep --les-only
#   ./rerender_sweep_videos.sh outputs/cylinder_dt_sweep_re200_v2 --name DNS --clamp 0.5
#   ./rerender_sweep_videos.sh outputs/my_sweep --only dt_1em3,dt_5em4
#
# Wrapper-specific flags (must come before any --name/--fps/--clamp etc.):
#   --only <a,b,c>   process only these subdir names (comma-separated)
#   --name <prefix>  override the data-file prefix to look for (default: DNS)
#                    Same flag is also forwarded to rerender_videos.py.
#
# All other flags are forwarded as-is to rerender_videos.py:
#   --fps <int>       (default in rerender: 20)
#   --clamp <float>   (default in rerender: 0.3)
#   --no-reencode     skip the H.264/yuv420p reencode step
#   --fr-only         skip LES rendering
#   --les-only        skip FR rendering

set -e

# ---- Locate the venv and rerender script -------------------------------- #
QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
QG_DIR="$QG_ROOT/qg-simple-package-stable/src/qg"
RERENDER="$QG_DIR/rerender_videos.py"

if [ ! -f "$RERENDER" ]; then
    echo "ERROR: rerender_videos.py not found at $RERENDER"
    exit 1
fi

# ---- Parse args --------------------------------------------------------- #
if [ $# -lt 1 ]; then
    sed -n '2,30p' "$0"
    exit 1
fi

SWEEP_ROOT="$1"
shift

NAME="DNS"          # data-file prefix (will also be passed to rerender_videos.py)
ONLY=""             # comma-separated list of subdirs to restrict to
PASSTHRU=()         # everything else gets forwarded

while [ $# -gt 0 ]; do
    case "$1" in
        --only)
            ONLY="$2"; shift 2
            ;;
        --name)
            # capture the prefix for our existence check, AND forward
            NAME="$2"
            PASSTHRU+=(--name "$2"); shift 2
            ;;
        *)
            PASSTHRU+=("$1"); shift
            ;;
    esac
done

if [ ! -d "$SWEEP_ROOT" ]; then
    echo "ERROR: sweep root not found: $SWEEP_ROOT"
    exit 1
fi
SWEEP_ROOT_ABS="$(cd "$SWEEP_ROOT" && pwd)"

# Always pass --name through if the user didn't (so default DNS still works
# explicitly when forwarded; this is a no-op if user already passed it).
if [[ ! " ${PASSTHRU[*]} " =~ " --name " ]]; then
    PASSTHRU+=(--name "$NAME")
fi

# ---- Activate venv ------------------------------------------------------ #
source "$QG_ROOT/qg-env/bin/activate"
export MPLCONFIGDIR="$QG_ROOT/.mplcache"
mkdir -p "$MPLCONFIGDIR"
export MPLBACKEND=Agg

# ---- Build list of subdirs to process ----------------------------------- #
if [ -n "$ONLY" ]; then
    # explicit comma-separated list
    IFS=',' read -ra SUBDIRS <<< "$ONLY"
    for i in "${!SUBDIRS[@]}"; do
        SUBDIRS[$i]="${SUBDIRS[$i]// /}"   # trim spaces
    done
else
    # all immediate subdirectories of sweep root, sorted
    SUBDIRS=()
    while IFS= read -r -d '' d; do
        SUBDIRS+=("$(basename "$d")")
    done < <(find "$SWEEP_ROOT_ABS" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)
fi

if [ ${#SUBDIRS[@]} -eq 0 ]; then
    echo "No subdirectories found under $SWEEP_ROOT_ABS"
    exit 1
fi

# ---- Run --------------------------------------------------------------- #
echo "==========================================================="
echo "Re-rendering sweep videos"
echo "  sweep root : $SWEEP_ROOT_ABS"
echo "  subdirs    : ${SUBDIRS[*]}"
echo "  prefix     : $NAME (looking for ${NAME}.npy or ${NAME}_FR.npz)"
echo "  fwd args   : ${PASSTHRU[*]}"
echo "==========================================================="
echo

N_OK=0
N_FAIL=0
N_SKIP=0
N_NOFILE=0

for sub in "${SUBDIRS[@]}"; do
    SAVE_PATH="$SWEEP_ROOT_ABS/$sub"
    echo "----- $sub -----"

    if [ ! -d "$SAVE_PATH" ]; then
        echo "  [skip] $SAVE_PATH does not exist"
        N_SKIP=$((N_SKIP + 1))
        echo
        continue
    fi

    # rerender_videos.py reads <NAME>.npy first (4-channel state) or falls
    # back to <NAME>_FR.npz. Skip subdirs that have neither.
    if [ ! -f "$SAVE_PATH/$NAME.npy" ] && [ ! -f "$SAVE_PATH/${NAME}_FR.npz" ]; then
        echo "  [skip] no ${NAME}.npy or ${NAME}_FR.npz in $SAVE_PATH"
        N_NOFILE=$((N_NOFILE + 1))
        echo
        continue
    fi

    if python -u "$RERENDER" "$SAVE_PATH" "${PASSTHRU[@]}"; then
        N_OK=$((N_OK + 1))
        echo "  -> done"
    else
        N_FAIL=$((N_FAIL + 1))
        echo "  -> FAILED"
    fi
    echo
done

echo "==========================================================="
echo "Summary:"
echo "  ok       : $N_OK"
echo "  failed   : $N_FAIL"
echo "  no data  : $N_NOFILE"
echo "  skipped  : $N_SKIP"
echo
echo "To copy the videos to your laptop:"
echo "  scp 'sanaamz@<cluster>:$SWEEP_ROOT_ABS/*/${NAME}*.mp4' ~/Downloads/"
echo "==========================================================="
