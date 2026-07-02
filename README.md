# qg-closure

ML-based temporal (δR) and spatial (Π) closure modeling for coarse-grid 2D QG turbulence.
**Read `CLAUDE.md` first** — it is the authoritative project brief (math conventions, model lineage, hard rules, pipelines). `docs/REPO_AUDIT.md` documents where every file came from and why.

## Layout

```
external/qg-simple/   QG solver — git SUBMODULE of your fork of akhilsadam/qg-simple
                      (branch package-stable; your modifications on a `closure` branch)
solver_patches/       Local 0.2.1-era solver modifications awaiting port onto the fork
                      (see solver_patches/PORTING.md — this dir DELETES itself once ported)
training/             Temporal-closure pipeline (v2 derivative-loss, ACTIVE WIP). FLAT —
                      scripts do sibling imports and must run from this dir.
staging/multigrid/    Next step after v2 — parallel deposit area. Editable; never merge
                      into training/ before v2 is done.
analysis/             Convergence, truncation-operator validation, stability, rollout figures.
spatial_closure/      Π_FF pipeline.
scripts/sge/          SGE submission wrappers/workers (submit_X.sh → X_job.sh).
docs/                 THEORETICAL_GUARANTEES.md, repo audit, slide sources.
paper/                Manuscript .tex (PDF is a build artifact, gitignored).
legacy/               Superseded generations + legacy/snapshots/ (read-only locks).
```

## One-time setup: fork the solver and wire it in

The solver is developed upstream at `akhilsadam/qg-simple` (now v0.2.3 — refactored BCs,
integrator, operator splitting). You **fork** it (a copy under your GitHub account that
can receive your pushes and still pull upstream), then add your fork as a **submodule**:

```bash
# 0. prerequisites: gh CLI authenticated (gh auth login), or fork via the GitHub web UI

# 1. fork upstream to your account and note the URL
gh repo fork akhilsadam/qg-simple --remote=false
#    -> creates https://github.com/<YOU>/qg-simple

# 2. from the root of THIS repo, add the fork as a submodule tracking package-stable
git submodule add -b package-stable https://github.com/<YOU>/qg-simple.git external/qg-simple
git commit -m "Add qg-simple solver submodule (fork, package-stable)"

# 3. inside the submodule: keep upstream reachable and create your modification branch
cd external/qg-simple
git remote add upstream https://github.com/akhilsadam/qg-simple.git
git fetch upstream
git checkout -b closure origin/package-stable
git push -u origin closure
cd ../..
# then point the submodule at the closure branch:
git config -f .gitmodules submodule.external/qg-simple.branch closure
git commit -am "Track closure branch of qg-simple fork"

# 4. install (uv env)
pip install uv && uv venv && source .venv/bin/activate
uv pip install -e external/qg-simple
```

### Pulling upstream solver updates later
```bash
cd external/qg-simple
git fetch upstream
git merge upstream/package-stable      # into your closure branch; resolve conflicts
git push
cd ../..
git add external/qg-simple && git commit -m "Bump qg-simple to upstream package-stable"
```

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
