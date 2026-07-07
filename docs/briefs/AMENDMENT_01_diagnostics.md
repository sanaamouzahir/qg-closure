# [QG][BRIEF][SGS-CLOSURE] AMENDMENT 01 — rulings + team-authored diagnostics scope

Global supervisor: your restatement is approved and this message is the ack that releases Phase A. First action: commit this text verbatim as `docs/briefs/AMENDMENT_01_diagnostics.md` on the branch. It amends the charter; where they conflict, this amendment wins.

## A. Rulings on your three conflicts

1. **bc.py hook location: your recommendation is approved.** Strictly-additive opt-in key in the shared `qg-simple-package-stable`, absent-key behavior bit-identical (enforced by the Gate 1 max|Δω| = 0.0 test), diff mirrored into `solver_patches/` per PORTING.md, other branch supervisors notified. The same mechanism and the same bit-identical requirement apply to the new in-run scalar recorder defined in §C below (opt-in key `diag.scalar_rate`; absent ⇒ recorder fully inert).
2. **Email enum: your merge is approved.** Charter codes (GATE1, RUN, CONV, PIFF, MILESTONE) for the named reports; global codes for state changes; ISSUE→FLAG. Branch tag is SGS-CLOSURE everywhere; the wiener-conditioning mention was indeed a copy-paste artifact.
3. **§7.5: the substance you acked (qlogin-only rerendering, milestone emails) is exactly right and stands.** Treat the 2026-07-06 governance amendments as charter-equivalent; committing this amendment closes the paper trail.

## B. Scope change: diagnostics are now TEAM-AUTHORED (charter §7 superseded)

`diagnostics_wake.py` will NOT be delivered by Sanaa. You (Fable) author all diagnostics code below — it is mathematical/numerical code, so per the §8 matrix it is yours alone. The branch supervisor integrates, dispatches, and may author the plotting layers on top. Sanaa's authorship on this branch is now limited to the LATER ML closure code; every dataset and diagnostic deliverable below must be ML-training-ready without her touching pipeline code.

## C. In-run scalar recorder (NEW — part of Phase A, gated by Gate 1)

Must exist BEFORE any production run: shedding-frequency tracking needs dense force time series, and 0.25-t.u. snapshots are far too coarse for clean spectra.

**C.1 Recorded every `diag.scalar_rate` steps (default 10 steps = 2.5e-3 t.u.):**
- `t`, step index.
- Brinkman force on the obstacle: `Fx`, `Fy` from the penalty term, i.e. the discrete momentum sink the mask actually applies this step. Derive the physical force from the discrete update as implemented (the YAML's eta = factor*dt convention means you must extract the effective physical damping from the code, not assume it) and document the derivation in the module docstring.
- `Cd_inst`, `Cl_inst` = 2F/(U(t)^2 D) normalized by INSTANTANEOUS inlet speed, and `Cd_mid`, `Cl_mid` normalized by U_mid = 2.0. Record both; raw Fx, Fy are the ground truth.
- `U_inlet` read back from the state (not the table), and `Re_inlet = U_inlet * D / nu`.
- `U_cyl` and `Re_cyl = U_cyl * D / nu`: the incident velocity actually seen by the body, defined as the mean streamwise velocity over a probe window centered 1.5 D upstream of the obstacle center, spanning |y - y_c| <= D/2, excluding masked points. This differs from U_inlet through blockage and wake feedback; recording both is mandatory.
- Domain-integral `E` (energy) and `Z` (enstrophy).
- Wake velocity probes: (u, v) at (x_c + {1, 2, 3} D, y_c) and (x_c + 1 D, y_c +/- 0.5 D). For the cape, place the equivalent probe set in the lee; propose coordinates in the Gate 1 report.
- Output: one `scalars.npz` per run (append-safe, flushed periodically so a killed job loses at most one flush interval), plus a `meta` block with all definitions.

**C.2 Sampling-rate justification (this is the save-rate logic; reproduce it in the code comments and the Gate 1 report):**
- T_shed(Re) = D/(St U): 5.30 at Re 2200, 2.99 at 3900, 2.08 at 5600.
- Scalars at 2.5e-3 t.u. -> ~830 samples per FASTEST shedding period: spectra and instantaneous-frequency estimates are sampling-noise-free. Storage is trivial (~48k samples x ~20 scalars x 8 B < 8 MB/run).
- Full fields stay at save_rate = 1000 (0.25 t.u.) -> 8.3 snapshots per fastest period, 481 per run: sufficient for Pi_FF a priori work and mean/rms wake statistics, and keeps the ensemble at ~100 GB. Field snapshots are NOT used for St.
- If Gate 1 reveals the scalar hook costs > 2% walltime at rate 10, report and propose rate 20 before changing anything.

**C.3 Gate 1 additions:** the bit-identical regression (recorder key absent), a recorder-on smoke whose Cd/Cl traces show clean periodic shedding after transient, and an overlay of U_inlet vs the prescribed table (must match at every recorded step).

## D. Shedding-frequency tracking (Fable-authored, post-processing on scalars.npz)

`shedding_tracker.py`:
1. Global St over the usable window: Welch PSD of Cl(t) (Hann, window 5 T_shed_mid ~ 15 t.u., 75% overlap); peak frequency f_shed; St(t)-mean via f_shed D / mean(U_inlet).
2. INSTANTANEOUS f_shed(t): band-pass Cl around the expected shedding band (0.15–0.55 given the Re range and St ~ 0.21), Hilbert transform, unwrapped-phase derivative, lightly smoothed. Cross-checks: zero-crossing intervals of Cl, and the same analysis on the wake-probe v(t) at x_c + 1D.
3. Key deliverable plot per run, on shared time axis: Re_inlet(t) and Re_cyl(t); f_shed(t) measured vs. the quasi-steady prediction f_qs(t) = St_ref U_inlet(t)/D with St_ref from the MOD-const run; instantaneous St(t) = f_shed(t) D / U_inlet(t). The lag/deviation of f_shed from f_qs under modulation is exactly the non-stationarity we will later ask the closure to cope with — treat this figure as a first-class result for the group meeting.

## E. diagnostics_wake.py (Fable-authored, post-processing on snapshots + scalars)

Per run, over the usable window t in [30, 120]:
1. Force statistics: mean Cd, rms Cl (both normalizations), tabulated per case.
2. Pressure and Cp: pressure Poisson solve from the streamfunction on snapshots; surface Cp(theta) evaluated on the ring r = R + 2 dx with a sensitivity check at R + {1,2,3} dx (penalized boundaries smear the surface; report the spread). Mean and rms Cp(theta). Cylinder only; for the cape, report the pressure field and along-boundary Cp as feasible.
3. Wake statistics: time-mean and rms fields; mean-U deficit profiles at x/D in {1.06, 1.54, 2.02, 3, 5}; recirculation length L_r from the mean centerline U; v'-spectra at the x_c + 1D probe.
4. Spectra: KE and enstrophy spectra E(k), Z(k) averaged over the window, per case, overlaid across modulations per geometry.
5. EVERY multi-panel diagnostic figure includes a Re_inlet(t) / Re_cyl(t) trace panel so all quantities are readable against the forcing history.

## F. Diagnostics validation gate (NEW: Gate D-1, before trusting any Re-3900 numbers)

One dedicated validation run: FPC, CONSTANT Re = 200 (set U via the same table mechanism; nu unchanged; grid 1024^2 is ample), T = 120. At Re = 200 the true flow IS two-dimensional (laminar shedding regime), so published 2D values are legitimate targets — this is the one place literature comparison is allowed:
- mean Cd ~ 1.3–1.4, St ~ 0.195–0.20, rms Cl ~ 0.4–0.7 (cite the specific 2D references you compare against, e.g. Henderson-type benchmarks).
- Pass ⇒ the force/St/Cp pipeline is trusted and the same code runs unmodified at Re 3900 (where results are compared only against our own fine-grid truth, per the charter's framing constraint). Fail ⇒ FLAG email, stop Phase E.
- Report: [QG][GATE-D1][SGS-CLOSURE].

## G. ML-readiness deliverable (extends Phase D/E)

Per case, the team produces `DATASET_MANIFEST.md` in the run directory: paths and shapes of DNS_FR / DNS_LES_s{2,4,8} / U_of_t / scalars.npz, the filter definition (scale, alpha, width), the Pi_FF sign/normalization convention as implemented, dtypes, and the usable-window bounds. Target: Sanaa opens the manifest and starts writing training code with zero archaeology.

## H. Sequencing update

- Phase A now = modulation.py + bc hook + scalar recorder + spectral_regrid.py, all Fable-authored; extended Gate 1 per §C.3.
- Gate D-1 (§F) runs immediately after Gate 1 approval, in parallel with the FPC-const/CAPE-const first pair.
- shedding_tracker.py and diagnostics_wake.py are authored while Phase B runs; they must be ready when the first-pair milestone fires.
- Everything else in the charter stands, including all gates, the §8 matrix, qlogin-only rerendering, and milestone emails.

Proceed.
