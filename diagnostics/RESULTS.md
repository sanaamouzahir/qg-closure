# RESULTS — wiener-conditioning (LIVING DOC)

**Convention (Sanaa, 2026-07-09): this is THE results file. Overwrite it in place
every session — never create a new dated RESULTS_*.md. Per diagnostic run: ONE
consolidated .npz + ONE readable text block here. Dated RESULTS_2026-07-*.md files
are frozen archives; git history preserves every prior state of this file.**

_Last overwrite: 2026-07-12 (session 11: replication triage of job 1830720 + psw3
verdict ladder 1830760). Prior state (sessions 7f–7g audit + scoreboard + 4-arm
table): git history of this file._

## Session context

The session-10 agent submitted apost_rep 1830720 (replication ladder) and the ro2p
three-job unit 1830721/22/23 (per-stride free-weight run rollout_ft_opt2_psw3), then
wedged and was killed. Both jobs landed exit 0 overnight, untriaged. This session
(Sanaa 07-12 autonomy window, chat): triage both, run the psw3 verdict ladder,
combined verdict.

## 1. Replication of the M=16 opt2 ladder (job 1830720, ckpt = opt2_cond ep33)

27 cases: {kf4, FRC-256, combo (HELD OUT of the fine-tune)} x 3 ICs x 3 dT, identical
code path/flags as the 07-11 IC837 headline ladder (1830550). Truth refs saved per
(member,IC,dT) for reuse. Consolidated: `Results/apost_opt2_rep_20260711/`
(one npz per case + `ladder_matrix_summary_ALL.csv`).

Including the IC837 headline row (10 cases per dT):

| dT     | stable | improvement over bare: median [min, max] |
|--------|--------|------------------------------------------|
| 5.0e-3 | 10/10  | **15.96x** [0.92, 23.57]                 |
| 1.0e-2 | 9/10   | 0.50x [0.01, 6.17]                       |
| 1.5e-2 | 6/10   | 0.65x [0.33, 1.89]                       |

- **The 16.6x REPLICATES at 5e-3**: all 10 cases stable; 9/10 in 6.4–23.6x; IC837
  (16.6x) sits at the median. One weak case: combo IC527 at 0.92x (parity, not
  degradation) — combo is the held-out member.
- **Stability does NOT hold everywhere at large dT — IC837 was a favorable draw**:
  1.5e-2 blows on 4/10 (256_ic1357@s15, combo_ic527@s10, combo_ic884@s14,
  kf4_ic912@s16); 1e-2 blows on 1/10 (combo_ic527@s13). Blow-ups concentrate on
  the held-out member (combo: 3 of the 5) but are NOT combo-exclusive.
- Large-dT accuracy where stable sits at/below parity (medians 0.50x/0.65x) —
  the single-IC picture confirmed: stability delivered, large-dT accuracy is the gap.

## 2. Per-stride free-weight run (ro2p 1830721, rollout_ft_opt2_psw3)

Warm start ep33 best; hinge lambda = (1e-3, 1e-3, 2.5e-4) — ONLY s3 relaxed 4x;
schedule 12:4,16:6,21:10, 20 ep. LANDED 52.7 min, best ep17 val 4.8677e-05.
- Supervised metric: rf mean 0.047/0.042/0.055 vs ep33's 0.050/0.048/0.078 —
  **rf_s3 fully recovered** (0.078 → 0.055, better than the warm baseline 0.065)
  with fb = 0.00 all strides. By the session-10 outcome test this read as
  hinge over-damping → ladder the ckpt (section 3 overturns the rollout half).
- Two transient train-loss explosions (ep2: 60 blown windows, train 4.6e121;
  ep6: train inf) absorbed by the session-9 guards; val never poisoned, n_skip 0.
  The LIVE monitor's ep6 EXPLODE mail delivered via the pending_mail spool —
  the session-10 delivery fix verified end-to-end (3 mails spooled AND relayed).

## 3. psw3 verdict ladder (job 1830760, 9 min, all 30 truth refs REUSED)

Identical code path/flags as sections 1 and the 07-11 ladder; ONLY the ckpt differs
(`rollout_ft_opt2_psw3/best.pt`). Full grid = IC837 + the 9 replication pairs.
`Results/apost_psw3_20260712/` (one npz per case + merged CSV).

| dT     | ep33: stable, median [range]   | psw3: stable, median [range]    |
|--------|--------------------------------|---------------------------------|
| 5.0e-3 | 10/10, 15.96x [0.92, 23.57]    | 10/10, **17.02x** [1.75, 25.62] |
| 1.0e-2 | 9/10, 0.50x [0.01, 6.17]       | 9/10, 0.49x [0.04, 7.11]        |
| 1.5e-2 | 6/10, 0.65x [0.33, 1.89]       | 7/10, **0.18x** [0.04, 1.53]    |

Per-case: psw3 improves the 5e-3 ratio in **10/10 cases** (combo IC527 0.92→1.75x,
combo IC884 6.4→10.3x). 1e-2 is a wash. At 1.5e-2 psw3 un-blows kf4_ic912
(6/10 → 7/10 stable) but degrades accuracy in EVERY previously-stable case
(IC837 0.33→0.18x, 256_ic549 0.85→0.17x; IC837 low-k drift 0.23 → 3.19).

## VERDICT (combined)

1. **Replication: YES at 5e-3.** The headline gain is real across ICs and members
   (median ~16x; held-out combo weakest but ≥ parity). **Large-dT stability does
   NOT replicate**: 4/10 blow at 1.5e-2 — "first NN closure to survive the 16-step
   ladder at 1.5e-2" downgrades to "survives on ~60% of draws".
2. **psw3 is a TRADE, not the accuracy lever.** Buys uniform 5e-3 improvement
   (median 15.96 → 17.02x) and one un-blown 1.5e-2 case; costs ~3.6x median
   1.5e-2 accuracy among stable cases. The offline rf_s3 recovery did NOT transfer
   to 16-step rollout: relaxing the s3 hinge lets low-k drift grow — the annulus
   damping was protective (the p170 lesson in a third form). The s3 rollout-accuracy
   deficit is NOT hinge over-damping; treat as intrinsic to the data limit
   (M_max = 3 supervised steps at stride 3).
3. **Best-ckpt ruling: ep33 (rollout_ft_opt2_cond) REMAINS the reference.**
   psw3 recorded as the 5e-3 specialist, not promoted. The remaining large-dT
   accuracy levers are (a) deeper builds extending s3 supervision (option 1 —
   needs Sanaa) and (b) the physics-conditioned model per the Wiener roadmap;
   hinge tuning is exhausted.

Artifacts: `Results/apost_opt2_rep_20260711/ladder_matrix_summary_ALL.csv`,
`Results/apost_psw3_20260712/ladder_matrix_summary_ALL.csv`; one consolidated npz
per case; all 30 truth refs kept for future ckpt ladders.

## Open items (carried)

- Deeper builds (option 1) still NOT authorized; the psw3 result strengthens the
  case that s3 supervision range is the binding constraint.
- Wiener filter theory formalization (iPad) before the conditioned model build.
- σ̂-drift CSV, frozen-σ̂ A/B, control-as-5th-arm, pareto, profile-step D-item
  ports: proposed with costs, awaiting per-item GO.
- DEC-loRe N3dot: input-information-limited (f32 disk floor), not a model bug —
  re-slice f64 only if a clean number is needed.

## Archive pointers (frozen)

RESULTS_2026-07-03.md (quiescent windows) · RESULTS_2026-07-08_smoke3.md ·
RESULTS_2026-07-09_apost_matrix.md (ladder matrix + dealias/FFT audit).
Sessions 7f–7g scoreboard, 4-arm table, ε(k) profile, 2/3-world tests: this file's
git history (state of 2026-07-09). Ladder npz roots: Results/apost_ladder_20260709*/,
Results/apost_opt2_20260711/, Results/apost_opt2_rep_20260711/,
Results/apost_psw3_20260712/, Results/spectral_error_profile_20260709/.
