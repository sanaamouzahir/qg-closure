#!/bin/bash
# migrate_results_tree.sh -- one-shot migration of existing SGS Pi_FF result
# artifacts into the results-organization STANDARD (Sanaa direct order,
# 2026-07-17):
#
#     ml_closure/results/<geometry>/<model>/<suite>/<member_modulation>/
#
# Moves (mv -- same filesystem, GBs of pngs/npz never copied and NEVER on the
# front end; this is an all.q job):
#   runs_piff/<model>/{eval*, mean_prediction_diag*, error_tails_diag*,
#                      reflection_probe, sigma_at_events}
#   pngs/{mean_prediction_diag, error_tails_diag, reflection_probe}/<model*>/
#   pngs/nearwall_variants/<member>/   (attributed to the ylp75 model of its
#                                       geometry -- those events fed it)
# then, inside the new tree, renames member subdirs to the plain modulation
# names and leaves RELATIVE symlinks at EVERY old path (old suite dirs AND
# codename member dirs) so nothing existing breaks. Idempotent: symlinks are
# skipped on re-run. SGS side only -- the wiener branch owns no Pi_FF
# artifacts and nothing outside qg-sgs-closure is touched.
#
# Manifest: full old->new list to this job's log AND to
# reports/results_migration/summary.md, pushed via diagnostics/digest_writer.py.
#
# Submit (front end, submission only):
#   cd $BRANCH && qsub -q all.q -N migrate_results -m ea -M sanaamz@mit.edu \
#       -j y -cwd -V -o logs/migrate_results.\$JOB_ID.log \
#       scripts/sge/migrate_results_tree.sh

#$ -S /bin/bash
#$ -q all.q
#$ -j y
#$ -cwd
#$ -V
#$ -o logs/migrate_results_tree.$JOB_ID.log
#$ -e logs/migrate_results_tree.$JOB_ID.err
#$ -m ea
#$ -M sanaamz@mit.edu

set -uo pipefail

QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
BRANCH="$QG_ROOT/qg-sgs-closure"
[[ -d /opt/rocks/bin ]] && export PATH=/opt/rocks/bin:$PATH
source "$QG_ROOT/qg-env-piff/bin/activate"
export PYTHONUNBUFFERED=1
# polite CPU share on all.q (pure filesystem work; no BLAS needed)
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$BRANCH/ml_closure"

echo "[migrate] host $HOSTNAME date $(date -u +%FT%TZ)"
python "$BRANCH/diagnostics/digest_writer.py" --repo-dir "$BRANCH" \
    --run-name results_migration --event start --job-id "${JOB_ID:-}" \
    --note "results-tree migration started on $HOSTNAME" \
    || echo "[migrate] WARNING: start digest failed (continuing)"

python - <<'PY'
import os
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, '.')
import member_naming as mn

ROOT = Path.cwd()                      # <branch>/ml_closure
RESULTS = ROOT / 'results'
REPORT = ROOT.parent / 'reports' / 'results_migration'
manifest = []                          # (kind, old, new)

# suite-dir mapping (runs_piff side); prefix match keeps _snap* suffixes:
# eval_snap1856 -> evaluation_snap1856, error_tails_diag_snap1903q ->
# error_tails_snap1903q, ...
SUITE_MAP = [('mean_prediction_diag', 'mean_prediction'),
             ('error_tails_diag', 'error_tails'),
             ('eval', 'evaluation'),
             ('reflection_probe', 'reflection_probe'),
             ('sigma_at_events', 'sigma_at_events')]


def suite_target(dirname):
    for old, new in SUITE_MAP:
        if dirname == old:
            return new
        if dirname.startswith(old + '_'):
            return new + dirname[len(old):]
    return None


def log(kind, old, new):
    manifest.append((kind, str(old), str(new)))
    print(f"[migrate] {kind:14s} {old} -> {new}", flush=True)


def merge_move(src, dst):
    """Move every child of src dir into dst dir (dst exists); codename
    symlinks inside dst are followed so pngs merge into the already-renamed
    modulation dirs. src is removed when emptied."""
    dst.mkdir(parents=True, exist_ok=True)
    for child in sorted(src.iterdir()):
        tgt = dst / child.name
        if tgt.is_symlink():           # e.g. FPC-const -> constant_inflow
            tgt = tgt.resolve()
        if child.is_dir() and not child.is_symlink():
            if tgt.exists() and tgt.is_dir():
                merge_move(child, tgt)
            else:
                child.rename(tgt)
                log('move', child, tgt)
        else:
            if tgt.exists():
                log('SKIP-exists', child, tgt)
            else:
                child.rename(tgt)
                log('move', child, tgt)
    try:
        src.rmdir()
    except OSError:
        pass


def rename_members(root):
    """Bottom-up: any directory named like a member codename becomes its
    modulation name, with a same-dir relative compat symlink left behind."""
    dirs = [p for p in root.rglob('*') if p.is_dir() and not p.is_symlink()]
    for d in sorted(dirs, key=lambda p: -len(p.parts)):
        if not mn.is_member_name(d.name):
            continue
        sibs = [x.name for x in d.parent.iterdir()]
        new = d.parent / mn.modulation_name(d.name, sibs)
        if new.exists():
            merge_move(d, new)
            if d.exists():
                continue               # not emptied; leave as-is, no symlink
        else:
            d.rename(new)
        os.symlink(new.name, d)        # compat: codename -> modulation
        log('member-rename', d, new)


def move_suite(src, target):
    """Move src dir to the standard-tree target, rename members inside,
    leave a relative symlink at src."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        merge_move(src, target)
    else:
        src.rename(target)
    log('suite-move', src, target)
    rename_members(target)
    if not src.exists():
        os.symlink(os.path.relpath(target, src.parent), src)
        log('symlink', src, target)


def model_of_png_dir(name):
    """pngs/<diag>/<name> -> (model, tag): exact runs_piff dir, or
    <model>_snap<...> split."""
    if (ROOT / 'runs_piff' / name).is_dir():
        return name, ''
    m = re.match(r'^(.*)(_snap\w+)$', name)
    if m and (ROOT / 'runs_piff' / m.group(1)).is_dir():
        return m.group(1), m.group(2)
    return None, None


# ---- 1. runs_piff/<model>/<suite dirs> ----------------------------------- #
for model_dir in sorted((ROOT / 'runs_piff').iterdir()):
    if not model_dir.is_dir() or model_dir.is_symlink():
        continue
    model = model_dir.name
    geom = mn.geometry_name(model)
    for sd in sorted(model_dir.iterdir()):
        if sd.is_symlink() or not sd.is_dir():
            continue
        suite = suite_target(sd.name)
        if suite is None:
            continue
        move_suite(sd, RESULTS / geom / model / suite)

# ---- 2. pngs/{mean_prediction_diag,error_tails_diag,reflection_probe} ---- #
for diag, suite_base in [('mean_prediction_diag', 'mean_prediction'),
                         ('error_tails_diag', 'error_tails'),
                         ('reflection_probe', 'reflection_probe')]:
    base = ROOT / 'pngs' / diag
    if not base.is_dir():
        continue
    for md in sorted(base.iterdir()):
        if md.is_symlink() or not md.is_dir():
            continue
        model, tag = model_of_png_dir(md.name)
        if model is None:
            log('SKIP-unmapped', md, '?')
            continue
        move_suite(md, RESULTS / mn.geometry_name(model) / model
                   / (suite_base + tag))

# ---- 3. pngs/nearwall_variants/<member> ---------------------------------- #
# data-only diagnostic keyed by member; attributed to the ylp75 model of its
# geometry (its events came from that model's error_tails run -- see the
# 2026-07-15 usage in diagnose_nearwall_variants.py's docstring).
base = ROOT / 'pngs' / 'nearwall_variants'
if base.is_dir():
    sibs = [x.name for x in base.iterdir()]
    for md in sorted(base.iterdir()):
        if md.is_symlink() or not md.is_dir() or not mn.is_member_name(md.name):
            continue
        model = ('piff_cape_gjs_ylp75' if 'cape' in md.name.lower()
                 else 'piff_fpc_gjs_ylp75')
        target = (RESULTS / mn.geometry_name(md.name) / model
                  / 'nearwall_variants' / mn.modulation_name(md.name, sibs))
        move_suite(md, target)

# ---- manifest ------------------------------------------------------------ #
REPORT.mkdir(parents=True, exist_ok=True)
stamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
counts = {}
for kind, _, _ in manifest:
    counts[kind] = counts.get(kind, 0) + 1
lines = [
    '# results-tree migration -- old -> new manifest',
    f'run {stamp} on the cluster (job '
    f"{os.environ.get('JOB_ID', 'interactive')}), Sanaa STANDARD 2026-07-17.",
    '',
    'Standard tree: ml_closure/results/<geometry>/<model>/<suite>/'
    '<member_modulation>/',
    'geometry: flow_past_cylinder | flow_past_cape;  suites: evaluation,',
    'mean_prediction, error_tails, sigma_at_events, reflection_probe,',
    'nearwall_variants (+ _snap* mid-training variants);  member dirs:',
    'constant_inflow, sine_modulation, ramp_modulation,',
    'ornstein_uhlenbeck_modulation, telegraph_modulation(_smoothed_A).',
    'Every old path keeps a RELATIVE symlink (incl. codename member dirs).',
    '',
    'counts: ' + ', '.join(f'{k}={v}' for k, v in sorted(counts.items())),
    '',
    '| kind | old | new |',
    '|---|---|---|',
]
lines += [f'| {k} | {o} | {n} |' for k, o, n in manifest]
(REPORT / 'summary.md').write_text('\n'.join(lines) + '\n')
print(f"[migrate] {len(manifest)} manifest entries; summary at "
      f"{REPORT / 'summary.md'}", flush=True)
PY
rc=$?

EV=done; [[ $rc -ne 0 ]] && EV=fail
python "$BRANCH/diagnostics/digest_writer.py" --repo-dir "$BRANCH" \
    --run-name results_migration --event "$EV" --job-id "${JOB_ID:-}" \
    --note "results-tree migration rc=$rc (manifest in reports/results_migration/summary.md)" \
    || echo "[migrate] WARNING: digest push failed"

echo "[migrate] done rc=$rc at $(date -u +%FT%TZ)"
exit $rc
