#!/bin/bash
# slice_new8_job.sh -- STAGE (a) of the DEC extension post-chain.
# Slices 7-lag derivative-target sweeps for the 8 newly deep-built members
# (FRC-b0,b05,b075,b1 + DEC-base,loRe,hiRe,512) from their deep 28-mark builds.
# Held on the 8 build_mmap_<member> job ids so it fires only after ALL builds
# finish. Self-guards the forcing-gate correctness check before touching data.
#$ -N deriv7_slice_new8
#$ -q ibgpu.q
#$ -l gpu=1
#$ -j y
#$ -o /gdata/projects/ml_scope/Closure_modeling/QG-closure/qg-simple-package-stable/src/qg/logs/deriv7_slice_new8.$JOB_ID.log
#$ -cwd
set -euo pipefail

QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
QG_DIR=$QG_ROOT/qg-simple-package-stable/src/qg
TRAIN_DIR=$QG_DIR/training
LOG_DIR=$QG_DIR/logs
DATA_ROOT=$TRAIN_DIR/data/ensemble_N5_7lag

FRC_MEMBERS=(FRC-b0 FRC-b05 FRC-b075 FRC-b1)
DEC_MEMBERS=(DEC-base DEC-loRe DEC-hiRe DEC-512)
ALL_MEMBERS=("${FRC_MEMBERS[@]}" "${DEC_MEMBERS[@]}")

echo "[slice_new8] host=$HOSTNAME date=$(date -u +%FT%TZ)"

# ---- Correctness gate (rule from supervisor): FRC must show static forcing, --- #
# ---- DEC must show no forcing. Abort loudly, do NOT slice, if violated. ------- #
GATE_FAIL=0
for m in "${FRC_MEMBERS[@]}"; do
    log="$LOG_DIR/build_mmap_${m}.log"
    if [[ ! -f "$log" ]]; then
        echo "[slice_new8] GATE ERROR: missing build log for $m ($log)"; GATE_FAIL=1; continue
    fi
    if grep -q '\[build\] no forcing' "$log"; then
        echo "[slice_new8] GATE ERROR: FRC member $m printed 'no forcing' -- STOP."
        GATE_FAIL=1
    fi
    if ! grep -q '\[build\] static forcing' "$log"; then
        echo "[slice_new8] GATE ERROR: FRC member $m never printed 'static forcing' -- STOP."
        GATE_FAIL=1
    fi
done
for m in "${DEC_MEMBERS[@]}"; do
    log="$LOG_DIR/build_mmap_${m}.log"
    if [[ ! -f "$log" ]]; then
        echo "[slice_new8] GATE ERROR: missing build log for $m ($log)"; GATE_FAIL=1; continue
    fi
    if grep -q '\[build\] static forcing' "$log"; then
        echo "[slice_new8] GATE ERROR: DEC member $m printed 'static forcing' -- STOP."
        GATE_FAIL=1
    fi
    if ! grep -q '\[build\] no forcing' "$log"; then
        echo "[slice_new8] GATE ERROR: DEC member $m never printed 'no forcing' -- STOP."
        GATE_FAIL=1
    fi
done
if [[ "$GATE_FAIL" != "0" ]]; then
    echo "[slice_new8] ABORTING -- forcing gate failed. No slicing performed."
    exit 1
fi
echo "[slice_new8] forcing gate OK for all 8 members."

source "$QG_ROOT/qg-env/bin/activate"
export MPLCONFIGDIR=$QG_ROOT/.mplcache; mkdir -p "$MPLCONFIGDIR"; export MPLBACKEND=Agg
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd "$TRAIN_DIR"

# ---- Verify each deep-build source exists (build job may have failed) -------- #
SOURCES=()
for m in "${ALL_MEMBERS[@]}"; do
    src="data/ensemble_N5_7lag/$m/forced_turbulence_dT_5em3"
    if [[ ! -f "$src/manifest.json" ]] || [[ ! -f "$src/packed/inputs.npy" && ! -f "$src/inputs.npy" ]]; then
        echo "[slice_new8] ERROR: deep build output missing/incomplete for $m at $src"
        exit 1
    fi
    SOURCES+=("$src")
done

TMP_ROOT="data/ensemble_N5_7lag/_slice_stage_tmp_${JOB_ID}"
if [[ -e "$TMP_ROOT" ]]; then
    echo "[slice_new8] ERROR: stale tmp root $TMP_ROOT already exists"; exit 1
fi

echo "[slice_new8] sources: ${SOURCES[*]}"
echo "[slice_new8] tmp out-root: $TMP_ROOT"
python -u slice_deriv_from_deep.py \
    --sources "${SOURCES[@]}" \
    --out-root "$TMP_ROOT" \
    --n-snapshots 7 \
    --target-dts 5e-3 1e-2 1.5e-2 \
    --max-anchors 3 \
    --device cuda \
    --dtype float64

# ---- Relocate sweep_dT_* dirs into each member's existing (empty) home dir --- #
for m in "${ALL_MEMBERS[@]}"; do
    dst="data/ensemble_N5_7lag/$m"
    shopt -s nullglob
    moved=0
    for d in "$TMP_ROOT/$m"/sweep_dT_*; do
        mv "$d" "$dst/"
        moved=$((moved+1))
    done
    shopt -u nullglob
    echo "[slice_new8] $m: relocated $moved sweep_dT_* dir(s) -> $dst/"
    if [[ "$moved" == "0" ]]; then
        echo "[slice_new8] ERROR: no sweep_dT_* produced for $m"
        exit 1
    fi
    rmdir "$TMP_ROOT/$m" 2>/dev/null || true
done
rmdir "$TMP_ROOT" 2>/dev/null || echo "[slice_new8] WARNING: tmp root $TMP_ROOT not empty after relocation, left in place for inspection"

echo "[slice_new8] done at $(date -u +%FT%TZ)"
