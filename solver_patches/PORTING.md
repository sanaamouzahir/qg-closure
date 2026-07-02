# solver_patches — port queue onto the qg-simple fork

These are the **local 0.2.1-era solver files** recovered from the working copy, laid out
in the reconstructed package structure. Upstream `qg-simple` is now **0.2.3** (refactored
boundary conditions, integrator, and operator splitting — see its changelog), so DO NOT
copy these files over the fork wholesale. Port each item by diffing against the fork's
`closure` branch, keeping upstream's 0.2.3 structure and re-applying the local change.

Once every item below is committed to the fork, **delete this directory**.

## Port list (local change → likely upstream target)

| Local file | What the local change is | Port notes |
|---|---|---|
| `src/qg/solver/opt/filter.py` | `LESFilter` — Eq. 5 composite (Gaussian + cutoff + avg-pool), independent `scale`/`width` | Likely absent upstream; add as new module. Spatial-closure (Π_FF) depends on it. |
| `src/qg/solver/dataset.py` | Solve-time FR extraction (obstacle/sponge masks, params, ω) for Π_FF post-processing | Likely local-only; add. Imports `qg._input.sources.bc.Region` — verify that path survived the 0.2.3 BC refactor. |
| `src/qg/run_qg.py` | Minimal Hydra driver without wandb/mura/git overhead | Add; trivial port. |
| `src/qg/_input/sources/ic.py` | `from_file` restart IC (broadcast single snapshot to n_batch) | Diff against 0.2.3 ic.py; re-apply if not upstreamed. Restart sweeps depend on it. |
| `src/qg/_input/sources/submarine_mask.py` | Streamlined-hull obstacle mask | Add to the mask library. |
| `src/qg/_input/sources/mask.py` | Check for local `cape` mask params (x_support etc.) vs upstream | Diff carefully. |
| `src/qg/conf/scenario/*.yaml` | `flow_past_cape`, `decaying_turbulence_restart`, tuned decay/FT/cylinder presets | Diff against upstream conf/ ("check default configs to see what needs updating" — 0.2.3 changelog). `decaying_turbulence_restart.yaml` has a hardcoded cluster IC path — parameterize via `${oc.env:...}` when porting. |
| `src/qg/solver/qg.py`, `integrator/imex.py`, `opt/operator/__init__.py`, `bc.py` | 0.2.1-era versions — upstream 0.2.3 refactored exactly these | **Do not port blindly.** Diff for any local fixes (e.g. the colleague's convergence fix referenced in run_akhil_t10_sweep.sh) and re-apply onto 0.2.3 only if still relevant. |
| everything else (`basis.py`, `derivative.py`, `jacobian.py`, `cartesian.py`, `obstacle.py`, `forcing.py`, `config.py`, `draw.py`, `train.py`, `test.py`, `__init__.py`) | Probably unmodified 0.2.1 | Diff; expect no-ops. Discard if identical to upstream history. |

## Verification after porting
1. `uv pip install -e external/qg-simple` clean.
2. `python run_qg.py +scenario=decaying_turbulence qg.time.T=0.5` runs on CPU.
3. One member of the decay dt-sweep reproduces the archived convergence slope (~2).
4. `training/add_deriv_targets.py` imports resolve (`qg` package paths intact).
