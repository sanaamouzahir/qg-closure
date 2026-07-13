# Implementation audit — error_analysis_shallow_nn.tex (2026-07-13)

Sanaa's question: "has the doc been implemented?" Claim-by-claim answer.

Scope: `Theoretical_guarantees/error_analysis_shallow_nn.tex` (546 lines; the copy in
the main repo `qg-closure/Theoretical_guarantees/` is BYTE-IDENTICAL to this branch's,
both 20,859 bytes, dated Jul 8) + the companion `THEORETICAL_GUARANTEES.md` Props 1-5.

Statuses: IMPLEMENTED-AND-VERIFIED / IMPLEMENTED-NOT-RUN / NOT-IMPLEMENTED / SUPERSEDED.

## A. Claims in error_analysis_shallow_nn.tex

| # | Claim (plain English) | Tex | Implemented where | Verified | Status |
|---|---|---|---|---|---|
| 1 | Binomial N-derivative recursion is the target generator and the model init | S1 ln 20-37 | training/slice_deriv_from_deep.py; training/model_deriv_closure.py (physics-init mix); training/closure_operators.py | analysis/validate_ab2cn2_vs_truth.py (R3-R6); every target build | IMPLEMENTED-AND-VERIFIED |
| 2 | Backward-FD Vandermonde rows; time error eps_k ~ dT^(S-k) omega^(S) | S2 ln 41-71 | model_deriv_closure.py TimeFD (frozen W_unit/dt^k, ln 116-143); temporal_fd_floor_deep.py | Results/fd_floor/ (2026-06-29); CHECK-1 dT-collapse plots (Jul 6) | IMPLEMENTED-AND-VERIFIED |
| 3 | Explicit constant C_k = (1/S!) sum W_ki (-i)^S from the Vandermonde remainder | S2 eq (9) | NONE — no script computes C_k itself; checks measure eps directly | — | NOT-IMPLEMENTED (open item already in CLAUDE.md; trivial, absent) |
| 4 | inv-Laplacian commutes with the time stencil (eps_psi = inv_lap eps_omega) | S2 eq (12) | design fact in slicer + model psi path; probed by diagnostics/diagnose_condlocal_triage.py (psi0 consistency ~3e-8 = f32 disk) | BRANCH_LOG session 5, D2 verdict ALIGNED | IMPLEMENTED-AND-VERIFIED |
| 5 | Error split Delta N^(m) = time part + stencil part; leading term driven by omega^(S) | S3 ln 81-141 | Theoretical_guarantees/check_wiener_floor.py (builds e = FD - truth; CHECK-3 full-slot response basis) | Results/wiener_floor/ (Jul 6): floor_decomposition.png + CSV | IMPLEMENTED-AND-VERIFIED |
| 6 | Pooled-LSQ floor decomposition (i) cascade + (ii) pooled variance + (iii) width deficit | S4 ln 143-206 | check_wiener_floor.py | wiener_floor_summary.csv (pooled_var_ii, shape_deficit_iii) | IMPLEMENTED-AND-VERIFIED |
| 7 | Measured constants: diag-absorbable 0.21-0.29, full-slot 0.36-0.44, (ii)=0.72, (iii)=0.31 | S4 eq (29) | same script, CHECK 3 | v3 pass Jul 6 19:30 — plots + conditioned_parameterization_note.md ONLY; archived CSV is the v2 pass ((ii)=0.862, (iii)=0.238) | IMPLEMENTED-AND-VERIFIED, HYGIENE CAVEAT: v3 console never archived |
| 8 | Diagonal sweeping ansatz omega_hat^(S) = sigma^(S-k) e^(i phi) omega_hat^(k) + r | S5 ln 214-319 | check_decorrelation_time.py (sigma(kappa) per member); check_wiener_floor.py measured transfers | Results/decorrelation/ + r_family/g_family plots | IMPLEMENTED-AND-VERIFIED (verified via measured transfers, not via assumptions A1-A4) |
| 9 | Shell-average formula <Omega^2> = k^2 U^2/2 + beta^2/(2k^2) - U beta cos(alpha) | S5 eq (44) | NONE — sigma(kappa) measured empirically instead; analytic U/beta split never fit | — | NOT-IMPLEMENTED (theory-only; low consequence, covered by measured sigma) |
| 10 | Cascade remainder r1 is incoherent: 1-coh^2 = 0.71-0.79 on healthy tiers | S6 ln 323-332 | check_wiener_floor.py complex shell coherence | CSV irr_frac 0.706-0.791 on the 7 healthy tiers — exact match | IMPLEMENTED-AND-VERIFIED |
| 11 | r2 coefficient-drift scaling, prefactor (S-k)(S+k-1)/2 (=20 at S=7,k=2) | S6 ln 334-385; S8 | formula NOT implemented; empirical proxy diagnostics/diagnose_sigma_drift.py (anchor drift 0.5-0.85%) | BRANCH_LOG session 1 | NOT-IMPLEMENTED (formula/scaling untested; proxy says drift small) |
| 12 | r3 anisotropy: <cos^5 theta> = 0; only cos3/cos5 harmonics reachable | S6 ln 387-420 | NONE — no angular-decomposition diagnostic exists | — | NOT-IMPLEMENTED (pure analytic result) |
| 13 | Reality/parity constraint D(-k) = conj D(k) | S6 eq (57) | encoded by construction in training/cond_grad.py (i k_d in-phase / quadrature split) | init exactness 5e-16 (diagnose_cond_init_sanity.py, session 1) | IMPLEMENTED-AND-VERIFIED |
| 14 | Conditioning factorization r_j = dT^(S-k)[exact] x g^(k)(kappa); dT-collapse 1.6x | S7 eqs (58-59) | check_wiener_floor.py CHECK 1 | dT_collapse_{Re25k,kf4,256}.png (Jul 6); note: spread 1.64x | IMPLEMENTED-AND-VERIFIED |
| 15 | sigma-hat estimator (2-mark FD + arcsin de-bias), fidelity 0.0-0.1% | S7 eqs (60-61) | training/cond_grad.py sigma_hat (ln 68-110); reused in model_cond_local.py | CHECK-2 fidelity plots; bit-identical from stepper (session 3) | IMPLEMENTED-AND-VERIFIED |
| 16 | F_spec conditioned SPECTRAL family, deficiency 0 | S7 eq (62) | cond_grad.SpectralCondGrad + CondDerivClosureNet (--model cond_deriv) | acceptance PASS 5e-16 (session 1); never trained | SUPERSEDED (24 FFTs/forward, I3; ceiling instrument only — cond_local is the deliverable) |
| 17 | F_loc conditioned LOCAL family (0-FFT), deficiency = 0.31 | S7 eqs (63-64) | training/model_cond_local.py (amp (dT/dT_ref)^(S-k), session-6 fix) | deriv7_cond_local_v2 ep63 pooled 0.2139, Nddot 0.138 vs control 0.186; per-root eval 1828724-26: cond beats control Nddot 16/17 roots | IMPLEMENTED-AND-VERIFIED |
| 18 | Unconditioned floor dominated by pooled variance (ii)=0.72; net absorption ~0.03 | S8 eq (65) | check_wiener_floor.py pooled analysis | pooled raw floor 0.175 vs control plateau 0.19 (CSV observed_floor) | IMPLEMENTED-AND-VERIFIED |
| 19 | Conditioned per-tier ceiling formula floor_j = raw_j sqrt(irr + coh (def + h.o.t.)) | S8 eq (67) | check_wiener_floor.py ln 333-336 (ceiling_cond) | v3 pass Jul 6 | IMPLEMENTED-AND-VERIFIED (as computation; see #20) |
| 20 | Predicted conditioned ceilings: kf4@1.5e-2 0.031->0.023; 256@1.5e-2 0.047->0.037; 256@1e-2 0.0068->0.0055; Re25k@1e-2 0.252->0.243 | S8 eq (68) | raw floors match archived CSV exactly; ceilings from CHECK 3 | TESTED BY TRAINING AND NOT MET: cond_local_v2 kf4@1.5e-2 = 0.065 vs target <=0.023 (killed at plateau ep78) | IMPLEMENTED, prediction OPEN — fell 2-3x short; not formally falsified (run killed pre-convergence) |
| 21 | Pooled raw floor 0.175 ~ observed 0.19; conditioned-pooled companion ~0.05 | S8 eq (69) | check_wiener_floor.py + control pooled val | 0.175/0.19 verified; conditioned pooled ~0.05 NOT achieved (0.214 at kill) | IMPLEMENTED-AND-VERIFIED (uncond part); conditioned-pooled prediction OPEN |

## B. Companion THEORETICAL_GUARANTEES.md Props 1-5 (run_all_guarantees.sh)

| # | Claim | Implemented where | Verified | Status |
|---|---|---|---|---|
| 22 | Prop 1 convergence radius dT* (Re25k 0.066 / combo 0.139 / kf4 0.199) | convergence_radius.py | Results/convergence_radius/ 20260629 + logs/tg_all.log | IMPLEMENTED-AND-VERIFIED |
| 23 | Prop 2 stencil depth helps inside dT*, hurts past it (Re25k crossover 1.5e-2) | fd_depth_check.py | Results/fd_depth_check/ 20260629 | IMPLEMENTED-AND-VERIFIED |
| 24 | Prop 3 decorrelation rho(tau) = 1 - tau^2 sigma^2/2, tau_lambda sigma = 1 | check_decorrelation_time.py | Results/decorrelation/ CSV (ratios 1.148/0.978, 1.092/0.968) | IMPLEMENTED-AND-VERIFIED |
| 25 | Prop 4 dT* = C tau_lambda, C = 2.08 +/- 0.15 universal | combination of Props 1+3 outputs (no single script) | both Results CSVs | IMPLEMENTED-AND-VERIFIED |
| 26 | Prop 5 closure error = eps_Nddot 1:1 un-amplified across the envelope | closure_error_propagation.py (+ run_error_prop_3dt.sh) | Results/error_propagation/ 20260629, eps=1 geometry pass | IMPLEMENTED-AND-VERIFIED |

## Summary

- 26 rows: 20 IMPLEMENTED-AND-VERIFIED, 4 NOT-IMPLEMENTED (#3 explicit C_k, #9
  shell-average Omega^2 formula, #11 r2 drift-coefficient scaling, #12 r3 angular
  harmonics), 1 SUPERSEDED (#16 F_spec), 1 implemented-but-prediction-OPEN (#20).
- Most important gap: the tex's central quantitative payoff — the conditioned
  ceilings (0.023/0.037/0.0055 per tier; pooled ~0.05) — was implemented and tested
  but NEVER REACHED: cond_local_v2 plateaued at kf4@1.5e-2 = 0.065 and pooled 0.214
  before the plateau kill. The headline Wiener prediction is open, not falsified.
- Hygiene gap: the v3 CHECK 1-3 console (source of (ii)=0.72, (iii)=0.31,
  full-slot 0.36-0.44) was never archived; the stored wiener_floor_summary.csv is
  the earlier v2 pass ((ii)=0.862, (iii)=0.238). Tex constants currently trace to
  plots + conditioned_parameterization_note.md only.

Audited 2026-07-13 (fresh-context subagent read of tex + all guarantee scripts +
Results trees + BRANCH_LOG/DECISIONS; both repo copies of the tex byte-identical).
