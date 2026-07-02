# training/ — sub-agent brief
Temporal-closure (δR) pipeline. **v2 derivative-loss, single-grid — ACTIVE WIP.**
Root CLAUDE.md is authoritative; local rules:
- FLAT module layout is load-bearing: scripts do sibling imports (concat_dataset,
  build_training_data_fixD_v2) plus the installed `qg` package. Run everything FROM
  this directory. Never introduce subpackages or relative-import refactors.
- float64 everywhere in data building and training. rel_l2 per-sample/per-channel loss.
- N̈ (Nddot) val rel-L2 = the rollout error floor. It is THE metric.
- TimeFD self.weight is inert by design (frozen-exact W_unit/dt^k path) — not a bug.
- Multigrid concerns do NOT belong here yet — deposit them in staging/multigrid/.
- Canonical model: model_deriv_closure.py (cheap_deriv, factored dx-independent).
  model.py / model_fixD.py are earlier baselines, kept deliberately.
