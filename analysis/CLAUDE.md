# analysis/ — sub-agent brief
Validation & figures: convergence (step1–3), truncation operators R3–R6, stability,
rollout comparisons. Root CLAUDE.md rules apply, especially:
- Convergence sweeps restart from a shared developed-flow snapshot, never t=0.
- Plots: cmap='seismic', aspect-preserving centered fit, no \tfrac in mathtext.
- replot_rollout_multistep.py exists so figures iterate WITHOUT re-running rollouts —
  prefer it over recompute.
