# qg-closure

**Machine-learned temporal and spatial closures for coarse simulations of 2D quasi-geostrophic turbulence.**

PhD research (MIT MSEAS, Mechanical Engineering + Center for Computational Science and Engineering). Two closure tracks, one pseudo-spectral GPU solver, one paper in preparation, and a fully documented multi-agent development workflow.

## What this is

Time integration of turbulence is expensive because accuracy — not stability — sets the step size. This repo develops **step-size resolution closures**: a compact (~10³-parameter) physics-initialized network corrects each coarse AB2–CN2 IMEX step so it matches the accuracy *and stability* of fine-step RK4, at linear-multistep cost.

**Temporal closure (δR):** closed-form local truncation error of AB2–CN2 / AB4–CN2 / RK4 to O(ΔT⁶); an analytical/learned split where powers of the Fourier-diagonal linear operator are free and only the time-derivatives of the advective Jacobian are learned; an exact Wiener error analysis that *derives* physics conditioning of the learned stencils on a measurable per-shell decorrelation rate; and a differentiable frozen-coefficient von Neumann stability certificate used as a training penalty. Full manuscript: [`paper/main.tex`](paper/) (branch `exp/wiener-conditioning`).

**Spatial closure (Π):** CNN + sparse-variational-GP sub-grid-scale forcing model for coarse-grid flow past obstacles (cylinder, cape), with FiLM conditioning on Reynolds number and calibrated conformal uncertainty (branch `exp/sgs-closure`).

## Repository map

| Path | Contents |
|---|---|
| `paper/` | Manuscript source (current version on `exp/wiener-conditioning`) |
| `Theoretical_guarantees/` | Derivations + verification scripts: LTE operators, convergence radius, Taylor-microscale law, Wiener floors, vN certificate |
| `analysis/` | A posteriori rollout comparisons, stability regions, truncation validation |
| `training/` | Estimator architecture, offline + rollout fine-tune training, evaluation |
| `diagnostics/` | Blow-up forensics, certificate evaluation, a posteriori stability studies |
| `scripts/` | SGE cluster submission scripts (GPU ensemble runs, data builds) |
| `spatial_closure/` | The Π-closure track (CNN+SVGP) |
| `docs/` | Agent-team design, repo audit, reports, slides |
| `legacy/`, `staging/`, `solver_patches/` | Archived earlier iterations, WIP multigrid, solver port queue |
| `external/` | `qg-simple` pseudo-spectral solver (fork, as submodule) |

## How this is built

Development runs as a supervised hierarchy of Claude agents under an explicit contract:

- [`CLAUDE.md`](CLAUDE.md) — authoritative project brief: math conventions, model lineage, hard rules, pipelines
- [`OPERATING_CHARTER.md`](OPERATING_CHARTER.md) — the agent operating charter: risk tiers, invariants (float64-only closure math, FFT budgets, SGE queue rules), monitoring triggers, day/night modes, handoff protocol
- [`DECISIONS.md`](DECISIONS.md) / [`HANDOFF.md`](HANDOFF.md) — running decision log and open rulings
- [`docs/AGENT_TEAM.md`](docs/AGENT_TEAM.md) — the agent hierarchy (portfolio supervisor → branch supervisors → runners/checkers/reviewers)

Every experiment lives on its own `exp/*` branch as a git worktree; agents are barred from pushing to `main`.

## Quick start

```bash
git clone --recurse-submodules https://github.com/sanaamouzahir/qg-closure.git
```

Pipelines (details in `CLAUDE.md`): simulate → `run_qg.py` via `scripts/sge/submit_qg.sh` · build ensemble data → `scripts/sge/build_ensemble_mmap.sh` + `training/add_deriv_targets.py` · train → `scripts/sge/train_deriv_job.sh` · evaluate → `training/rollout_timed_pareto.py`, `analysis/rollout_multistep_comparison.py`.

Compute: SGE GPU cluster (float64 throughout — the closure target is O(ΔT³) ≈ 10⁻⁹, below float32 machine epsilon).

## Status

- Temporal-closure manuscript in active revision (`exp/wiener-conditioning`)
- Certificate-regularized training sweep (λ-sweep) in flight
- Spatial closure: cylinder-ensemble phase, conformal calibration complete

---
*Sanaa Mouzahir · MIT MSEAS ([mseas.mit.edu](https://mseas.mit.edu)) · advisor: Prof. Pierre F. J. Lermusiaux*
