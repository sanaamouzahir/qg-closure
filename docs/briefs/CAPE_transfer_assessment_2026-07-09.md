# FPC -> CAPE transfer assessment (2026-07-09, read-only; cape SUSPENDED per A02)

Requested by Sanaa for the 2026-07-10 morning report. Full table in the report email;
this file is the branch-side record.

HEADLINE: run pipeline ~90% cape-ready; audit/theory layer is not. Zero solver code
changes (scenario yaml + cape mask + bc hook + recorder cape-guards + Pi_FF all
pre-built). Code files needing edits: ~6 — phaseB_job.sh + phaseC_job.sh (1 line each:
scenario hardcode), shedding_tracker.py (S: cape theory block / no-lit mode),
audit_decorrelation.py (S: lee wake box + cape constants), audit_resolution.py
(L: polar theta-ray layer -> arc-length y=h(x) wall-normal layer = the one real
rewrite), diagnostics_wake.py (M: R-and-centerline geometry).

NEW THEORY ITEMS (Sanaa's deliverable, theory-doc extension + Amendment 03): (1)
normalization length ruling L_cape=1 vs W_cape=4 (Cd/Cl/Re convention); (2) cape
timescale/N_eff inventory (S2/S4); (3) wall-layer + separation spec for a smooth bump
(S7); (4) shedding reference / gate policy (no cape literature analog of St=0.21);
(5) dt_save-commensurability bootstrap.

CHOREOGRAPHY INVERSION: cape has no a-priori T_sh -> order flips to CAPE-const FIRST,
measure f_sh/T_sh, derive dt_save/bands/N_eff from it (Audit A becomes the theory
SOURCE), then release the 4 modulated runs. Existing FPC U(t) tables are literally
reusable (same nu, same signals, same dt).

EFFORT: ~3-4 agent sessions code+submission + Sanaa's theory extension; cape runs
~4x cheaper than FPC (1024^2 FR). Details + file:line evidence in the morning report
and the coordinator session record.
