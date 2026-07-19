# vendor_patches/qg-simple — solver checkout delta (reproducibility record)

The cluster checkout `$QG_ROOT/qg-simple-package-stable/` (upstream
quasigeostrophic-flow 0.2.x install, git-init'd locally, NO remote) carries
local modifications that are not in any pushed repo. This directory is the
version-controlled copy of that delta (git format-patch --root, 2026-07-15).

- 0001: SGS hooks baseline — inlet_table BC + diag.scalar_rate recorder
  (pre-edit versions: qg-sgs-closure/solver_patches/sgs_hooks_2026-07-07.patch).
- 0002: ScalarRecorder flush fix — np.savez appends `.npz` to any filename
  not ending in it, so the tmp+os.replace atomic write died at first flush
  (jobs 1828232-34). Fixed by writing through an open file handle.

These are 0.2.x-era edits awaiting port onto the fork's `closure` branch
(external/qg-simple), same workflow as solver_patches/PORTING.md: diff,
don't overwrite — upstream 0.2.3 refactored BCs/integrator/operator
splitting. The checkout itself stays as-is on the cluster; this directory
only puts the code delta under version control.
