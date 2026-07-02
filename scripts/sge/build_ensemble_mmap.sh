#!/bin/bash
# build_ensemble_mmap.sh
# ----------------------------------------------------------------------------
# Build the temporal-closure dataset (7-snapshot mmap builder, N^(1)..N^(5)
# targets) for EVERY member folder under the step-size/resolution ensemble:
#
#   $ENSEMBLE_DIR/<member>/  ->  $OUT_DIR/<member>/<scenario>_dT_<tag>/packed/
#
# One GPU job is submitted PER member (parallel across the queue; one member's
# failure does not abort the others), matching submit_ensemble.sh conventions.
#
# Differences vs build_training_data_mmap_ft.sh (the single-file FT build):
#   * Loops over folders instead of iterating batches in one DNS_FR_omega.npy.
#     Each folder is a single hydra run (one trajectory) -> split-mode by_time.
#   * Per-member regime: by default the build reads each member's OWN resolved
#     config (.hydra/config.yaml) so beta / nu / mu / forcing / grid are correct
#     without hand-maintaining one yaml per member. Falls back to $DEFAULT_YAML.
#   * Source omega/times filenames are DISCOVERED per folder (glob), since the
#     hydra output name may differ from DNS_FR_omega.npy. Override at the top.
#   * --max-order 5  (Ndot..N5dot; needed to assemble R6 for AB2CN2 and to give
#     headroom for the AB4CN2 / RK4 / stability operators). Default in the new
#     builder, kept explicit here.
#
# Usage:
#   ./build_ensemble_mmap.sh --dry-run         # resolve + PRINT every command, submit nothing
#   ./build_ensemble_mmap.sh                   # submit one GPU job per member
#   ./build_ensemble_mmap.sh --interactive     # run ONLY the first member inline (smoke test)
#   ./build_ensemble_mmap.sh --members FRC-b2,FRC-256   # restrict to a subset
#   ./build_ensemble_mmap.sh --force           # rebuild members whose packed/ already exists
#   ./build_ensemble_mmap.sh -- --n-seeds 20   # forward extra args to the builder (after --)
# ----------------------------------------------------------------------------
set -uo pipefail

# ---- Paths ---------------------------------------------------------------- #
QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
VENV=$QG_ROOT/qg-env
QG_DIR=$QG_ROOT/qg-simple-package-stable/src/qg
SCRIPT_DIR=$QG_DIR/training
LOG_DIR=$QG_DIR/logs
PY=build_training_data_mmap.py

ENSEMBLE_DIR=$QG_DIR/outputs/Step_size_resolution_closure_ensemble
OUT_DIR=$SCRIPT_DIR/data/ensemble_N5_7lag            # per-member subdir appended below
DEFAULT_YAML=$QG_DIR/conf/scenario/forced_turbulence.yaml   # fallback regime yaml

# ---- Per-member source discovery ------------------------------------------ #
# Each member folder holds DNS_FR.npz (fine reference: omega + times in ONE npz,
# self-contained -> no separate --source-times) and DNS.npy (coarse run). We seed
# from the fine reference, mirroring the old DNS_FR_omega.npy source.
# First match wins. Run --dry-run to confirm what resolves.
OMEGA_CANDIDATES=(DNS_FR.npz DNS.npy)
TIMES_CANDIDATES=(DNS_FR_times.npy times.npy)   # usually none -> times read from the npz
# Per-member hydra config, if the run saved one (often absent here):
YAML_CANDIDATES=(.hydra/config.yaml config.yaml)
USE_MEMBER_YAML=1                              # 0 -> skip hydra-config autodetect

# ---- Per-member REGIME yaml (CORRECTNESS-CRITICAL) ------------------------- #
# The build reads nu/mu/beta/forcing/grid from --source-yaml. Members differ in
# regime (beta, forcing wavenumber, Re), so a single default yaml would compute
# the WRONG L_hat and WRONG N-derivative targets for off-default members. If the
# folders have no .hydra/config.yaml, map each member to its regime yaml here.
# Leave empty to rely on hydra autodetect / DEFAULT_YAML (a loud warning fires
# whenever a member falls back to the default).
declare -A MEMBER_YAML=(
    # [FRC-b2]=$QG_DIR/conf/scenario/forced_turbulence_b2.yaml
    # [FRC-b25]=$QG_DIR/conf/scenario/forced_turbulence_b25.yaml
    # [FRC-kf4]=$QG_DIR/conf/scenario/forced_turbulence_kf4.yaml
    # [FRC-Re25k]=$QG_DIR/conf/scenario/forced_turbulence_Re25k.yaml
    # [FRC-combo]=$QG_DIR/conf/scenario/forced_turbulence_combo.yaml
    # [FRC-256]=$QG_DIR/conf/scenario/forced_turbulence_256.yaml
)

# ---- Build knobs (override after `--`) ------------------------------------ #
SCENARIO=forced_turbulence     # only affects validation + output subdir name
DELTA_T=5.0e-3
H_FINE=1.0e-5                  # K = DT/h_fine = 100
H_ULTRAFINE=5.0e-6            # RK4 warmup for the 7 snapshot levels
N_SEEDS=500                   # snapshots per member (capped at unique snapshots in t-range)
T_START=15.0                 # developed-flow window start; LOWER if a member's run is short
SPLIT_MODE=by_time           # single trajectory per folder -> split by time
DEVICE=cuda
MAX_ORDER=5                  # save N_dot..N5dot
INPUT_DTYPE=float32          # build computes f64; this is storage dtype only

# ---- CLI ------------------------------------------------------------------ #
INTERACTIVE=0
DRY_RUN=0
FORCE=0
JOBNAME_PREFIX=build_mmap
MEMBERS_CSV=""
EXTRA=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --interactive) INTERACTIVE=1; shift ;;
        --dry-run)     DRY_RUN=1; shift ;;
        --force)       FORCE=1; shift ;;
        --jobname-prefix) JOBNAME_PREFIX="$2"; shift 2 ;;
        --members)     MEMBERS_CSV="$2"; shift 2 ;;
        --) shift; while [[ $# -gt 0 ]]; do EXTRA+=("$1"); shift; done ;;
        *) EXTRA+=("$1"); shift ;;
    esac
done

# ---- Pre-flight ----------------------------------------------------------- #
[[ -f "$VENV/bin/activate" ]] || { echo "ERROR: venv not at $VENV" >&2; exit 1; }
[[ -f "$SCRIPT_DIR/$PY"    ]] || { echo "ERROR: $PY not in $SCRIPT_DIR" >&2; exit 1; }
[[ -d "$ENSEMBLE_DIR"      ]] || { echo "ERROR: ensemble dir not found: $ENSEMBLE_DIR" >&2; exit 1; }
mkdir -p "$LOG_DIR" "$OUT_DIR"

# ---- Member list ---------------------------------------------------------- #
if [[ -n "$MEMBERS_CSV" ]]; then
    IFS=',' read -r -a MEMBERS <<< "$MEMBERS_CSV"
else
    MEMBERS=()
    for d in "$ENSEMBLE_DIR"/*/; do
        [[ -d "$d" ]] || continue
        MEMBERS+=("$(basename "$d")")
    done
fi
[[ ${#MEMBERS[@]} -gt 0 ]] || { echo "ERROR: no member folders found in $ENSEMBLE_DIR" >&2; exit 1; }
echo "[ensemble-build] $((${#MEMBERS[@]})) members: ${MEMBERS[*]}"

# ---- Resolver: first existing candidate (supports globs) ------------------ #
resolve_one() {  # $1=member_dir ; shift ; remaining = candidate patterns
    local mdir="$1"; shift
    local pat hit
    for pat in "$@"; do
        for hit in "$mdir"/$pat; do
            [[ -e "$hit" ]] && { echo "$hit"; return 0; }
        done
    done
    return 1
}

# ---- Per-member launch ---------------------------------------------------- #
submit_or_run() {  # $1=member
    local member="$1"
    local mdir="$ENSEMBLE_DIR/$member"
    local out_member="$OUT_DIR/$member"
    local jobname="${JOBNAME_PREFIX}_${member}"

    local omega times yaml
    omega=$(resolve_one "$mdir" "${OMEGA_CANDIDATES[@]}") || {
        echo "  [skip] $member: no omega file (tried: ${OMEGA_CANDIDATES[*]})" >&2; return 0; }
    times=$(resolve_one "$mdir" "${TIMES_CANDIDATES[@]}") || times=""   # else read from npz
    # regime yaml: explicit map -> hydra autodetect -> default (with a warning)
    if [[ -n "${MEMBER_YAML[$member]:-}" ]]; then
        yaml="${MEMBER_YAML[$member]}"
    elif [[ "$USE_MEMBER_YAML" == "1" ]] && yaml=$(resolve_one "$mdir" "${YAML_CANDIDATES[@]}"); then
        :   # found a per-member hydra/config yaml
    else
        yaml="$DEFAULT_YAML"
        echo "  [WARN] $member: no member yaml -> using DEFAULT_YAML; regime " \
             "(beta/nu/forcing) may be WRONG for this member. Map it in MEMBER_YAML." >&2
    fi
    [[ -f "$yaml" ]] || { echo "  [skip] $member: yaml not found: $yaml" >&2; return 0; }

    local packed="$out_member/${SCENARIO}_dT_unstable/packed/inputs.npy"
    if [[ "$FORCE" != "1" && -f "$packed" ]]; then
        echo "  [skip] $member: packed exists ($packed); use --force to rebuild"; return 0
    fi
    mkdir -p "$out_member"

    local PYARGS=(
        --scenario     "$SCENARIO"
        --source-omega "$omega"
        --source-yaml  "$yaml"
        --out-dir      "$out_member"
        --Delta-T      "$DELTA_T"
        --h-fine       "$H_FINE"
        --h-ultrafine  "$H_ULTRAFINE"
        --n-seeds      "$N_SEEDS"
        --t-start      "$T_START"
        --split-mode   "$SPLIT_MODE"
        --device       "$DEVICE"
        --max-order    "$MAX_ORDER"
        --input-dtype  "$INPUT_DTYPE"
    )
    [[ -n "$times" ]] && PYARGS+=(--source-times "$times")
    PYARGS+=(${EXTRA[@]+"${EXTRA[@]}"})

    echo "  [$member]"
    echo "     omega : $omega"
    echo "     times : ${times:-<none / embedded>}"
    echo "     yaml  : $yaml"
    echo "     out   : $out_member/${SCENARIO}_dT_5em3/packed/"

    if [[ "$DRY_RUN" == "1" ]]; then
        echo "     cmd   : python -u $PY ${PYARGS[*]}"
        return 0
    fi

    if [[ "$INTERACTIVE" == "1" ]]; then
        source "$VENV/bin/activate"
        export MPLCONFIGDIR="$QG_ROOT/.mplcache"; mkdir -p "$MPLCONFIGDIR"; export MPLBACKEND=Agg
        cd "$SCRIPT_DIR"
        python -u "$PY" "${PYARGS[@]}"
        return $?
    fi

    # SGE: GPU queue ONLY -- -q ibgpu.q -l gpu=1. No -l h_vmem, no other queues.
    local JOB_SCRIPT
    JOB_SCRIPT=$(mktemp /tmp/${jobname}_XXXXXX.sh)
    cat > "$JOB_SCRIPT" <<EOF
#!/bin/bash
#\$ -N $jobname
#\$ -q ibgpu.q
#\$ -l gpu=1
#\$ -j y
#\$ -o $LOG_DIR/${jobname}.log
#\$ -cwd
set -e
source $VENV/bin/activate
export MPLCONFIGDIR=$QG_ROOT/.mplcache; mkdir -p "\$MPLCONFIGDIR"; export MPLBACKEND=Agg
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd $SCRIPT_DIR
echo "[mmap-build] member=$member host=\$HOSTNAME gpu=\${CUDA_VISIBLE_DEVICES:-none} $(date -u +%FT%TZ)"
echo "[mmap-build] cmd: python -u $PY ${PYARGS[*]}"
python -u $PY ${PYARGS[@]}
echo "[mmap-build] done $(date -u +%FT%TZ)"
EOF
    chmod +x "$JOB_SCRIPT"
    qsub "$JOB_SCRIPT"
}

# ---- Drive ---------------------------------------------------------------- #
if [[ "$INTERACTIVE" == "1" ]]; then
    echo "[ensemble-build] interactive: building ONLY the first member (${MEMBERS[0]})"
    submit_or_run "${MEMBERS[0]}"
    exit $?
fi

for m in "${MEMBERS[@]}"; do
    submit_or_run "$m"
done

if [[ "$DRY_RUN" == "1" ]]; then
    echo "[ensemble-build] dry run complete -- nothing submitted."
else
    echo "[ensemble-build] submitted; logs in $LOG_DIR/${JOBNAME_PREFIX}_<member>.log"
fi