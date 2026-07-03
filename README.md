# qg-closure

ML-based temporal (δR) and spatial (Π) closure modeling for coarse-grid 2D QG turbulence.
**Read `CLAUDE.md` first** — it is the authoritative project brief (math conventions, model lineage, hard rules, pipelines). `docs/REPO_AUDIT.md` documents where every file came from and why.


### Cloning this repo fresh (e.g. on the cluster)
```bash
git clone --recurse-submodules https://github.com/<YOU>/qg-closure.git
# or, if already cloned:  git submodule update --init
```

## Porting the local solver modifications
The Project's solver files were 0.2.1-era and carry local changes (LES filter, FR dataset
extraction, `run_qg.py` driver, `from_file` IC restart, cape/submarine masks, scenario
YAMLs). Upstream 0.2.3 refactored BCs/integrator, so these must be *ported*, not copied.
Follow `solver_patches/PORTING.md`, commit each port to the fork's `closure` branch, then
delete `solver_patches/`.

## Quick pipeline map
Simulate → `run_qg.py` via `scripts/sge/submit_qg.sh` · Build ensemble data →
`scripts/sge/build_ensemble_mmap.sh` + `training/add_deriv_targets.py` · Train →
`scripts/sge/train_deriv_job.sh` (see CLAUDE.md §Main pipelines for exact commands) ·
Evaluate → `training/rollout_perfect_closure.py`, `training/rollout_timed_pareto.py`,
`analysis/rollout_multistep_comparison.py`.
