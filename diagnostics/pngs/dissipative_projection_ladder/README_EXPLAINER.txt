DISSIPATIVE PROJECTION LADDER -- what these figures show (plain English)

The trained network adds a small correction to the coarse solver at every
step. Most of that correction fixes real errors, but a part of it can PUMP
energy (enstrophy) into the simulation -- and at large time steps that pumped
energy is what makes runs explode. The new inference safety valve
(--nn-dissipative-proj) looks at the correction one wavenumber shell at a
time and deletes ONLY the piece that pumps energy into that shell; everything
else passes through untouched. It costs no extra Fourier transforms and does
not touch the physics or the solver -- it acts on the network output alone.

The experiment: 10 independent starting states (4 from FRC-kf4, 3 from
FRC-256, 3 from FRC-combo, the fine-tune's held-out member) are rolled out
16 steps at three step sizes (5e-3, 1e-2, 1.5e-2), each once WITH the valve
and once WITHOUT -- everything else identical, truth references reused from
the 07-11 replication ladder. At the smallest step size we additionally run
much longer horizons (64 and 128 steps) to check the valve holds up over
time. Pre-registered success bars, fixed BEFORE the runs: (1) with the
valve, all 10 runs at step 1e-2 survive; (2) at 1.5e-2 things improve; (3)
at 5e-3 the valve costs less than 5% accuracy.

f1_ab_final_relL2.png  Each marker is one starting state: horizontal = final
    error without the valve, vertical = with it. On the dashed diagonal the
    valve changed nothing. BELOW the diagonal the valve helped. Open markers
    on the dotted line are runs that EXPLODED without the valve but finished
    with it -- the valve's whole purpose.

f2_proj_activity.png  How many wavenumber shells the valve actually touched
    at each step. Theory says it should be nearly idle at the small step
    size (nothing to fix) and busy at the large one (where explosions
    happened). Median line, shaded band = spread across the 10 draws.

f3_horizon_curves.png  Error over time for the long runs (64 and 128 steps
    at 5e-3), with and without the valve, against the no-closure baseline.
    Shows whether the valve keeps the closure's advantage over long
    horizons.

Numbers: Results/apost_dissproj_20260713/dissproj_verdict.txt (bars verdict)
and dissproj_AB_table.csv (every case). One consolidated npz per case in the
same directory tree. Extended truth references were saved next to the
existing ones under Results/apost_opt2_rep_20260711/ for future reuse.
