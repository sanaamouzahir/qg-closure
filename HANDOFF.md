# HANDOFF — cluster state at agent yield (I17: ONE living document, overwrite in place, never duplicate)

Written 2026-07-15 ~11:55 EDT by the final cluster session, on Sanaa's YIELD-AND-EXIT order.
From now on Sanaa works on her LOCAL machine; cluster access via ssh (mseas = the ONLY SGE
submit host). `QG_ROOT = /gdata/projects/ml_scope/Closure_modeling/QG-closure`. All paths
below are under `QG_ROOT` unless absolute.

---

## RUNNING (qstat -u sanaamz at 11:45 EDT, verified)

```
1833569  w31_TRN     r    ibgpu.q   the width-31 conditioned retrain (deriv7_cond_local_w31)
1833570  w31_L       r    all.q     ... its LIVE monitor (emails ~2-hourly, agent-free via relay cron)
1833571  w31_F       hqw            ... its FINALIZE monitor (fires at trainer exit)
1834629  w31p1a_TRN  r    ibgpu.q   anchored rollout FT arm A (anchor lambda 3e-2, 20 ep)
1834631  w31p1a_F    hqw            ... FINALIZE monitor
1834632  w31p1b_TRN  r    ibgpu.q   anchored rollout FT arm B (anchor lambda 3e-1, 20 ep)
1834634  w31p1b_F    hqw            ... FINALIZE monitor
1834691  w31p1a_L2   qw (-a 12:50)  replacement LIVE monitor for arm A (see monitor note)
1834692  w31p1b_L2   qw (-a 12:50)  replacement LIVE monitor for arm B
```

- **w31 trainer (1833569)**: at ep~46 of 150, ~1550 s/ep -> would run to ~2026-07-16 evening.
  best.pt is FROZEN at ep32 (pooled val 0.0605); the plateau criterion fired and the arms
  warm-started from it. It is running only in case a later best appears. Sanaa may qdel it
  at will; nothing chained depends on its exit. Log: `qg-wiener-conditioning/logs/` (o-file
  of 1833569) + run log `.../training_runs/deriv7_cond_local_w31/log.csv` (path map below).
- **Arms (1834629 / 1834632)**: ~3630 s/ep at ep0, 20 epochs -> land ~2026-07-16 07:00-08:00.
  FINALIZE monitors email the verdict with NO agent present. Acceptance gate (Sanaa's option
  4): per-member a-priori Nddot before/after via `eval_deriv_by_root.py`, tolerance 10%
  (`training/accept_ft_gate.py`, exit 3 on FAIL) — run AFTER landing, from the wiener
  worktree `training/`.
- Digest of what the arms are: rollout fine-tune of w31 best.pt with von Neumann penalty 0.1
  (FIRST arms with the FIXED tap-read certificate), a-priori accuracy anchor on all 41 roots
  (lam 3e-2 balanced / 3e-1 hard), rollout pool 7 roots, combo+b25 HELD OUT for OOD.

## MONITOR-COVERAGE NOTE (Y3 honesty)

The arms' ORIGINAL LIVE monitors (1834630/1834633) emailed `ep0 EXPLODE` at 11:30 and exited
BY DESIGN (EXPLODE = email + exit). The trigger is a STARTUP ARTIFACT, not a blow-up: the
rollout FT's ep0 train column is `inf` because the warm model is unstable in free rollout
BEFORE fine-tuning (n_blown 1163 at ep0 — this is exactly what the FT exists to fix); val and
all anchor terms are finite and sane (arm A val 1.89e-03, anc_Nddot 0.116). Replacements
1834691/1834692 start at 12:50, by which time ep1's row (finite train expected) is the latest
the monitor judges. IF ep1 is also non-finite they will email EXPLODE and exit again — in that
case the arms run LIVE-BLIND (no periodic emails) but the FINALIZE verdict emails still fire
at exit. Treat a second EXPLODE email pair as "check the arms by hand".

## QUEUED

Only the held/delayed monitors listed above. No other jobs in the scheduler.

## AGENT-FREE AUTOMATION STILL ACTIVE (mseas crontab, survives everything)

- `reporting/send_pending.sh` every 10 min — relays `reporting/pending_mail/*.mail`;
  `reporting/relay.log` is the only proof of delivery.
- `reporting/daily_report.sh` 08:00 daily; `reporting/status_report.sh` 12/16/20h.
- `reporting/autofire_check.sh` every 10 min — INERT now (markers `w31_fired` and
  `ylp75_fired` both set in `reporting/.autofire_markers/`).

## AWAITING SANAA'S RULING

1. **Promote ylp75 to production reference** (both geometries). Evidence table below.
2. **Re-fit the CRPS head (sigma stage 2) on the ylp75 models** — current heads were fit on
   the pre-ylp75 gjs checkpoints (cov1 0.713 fpc / 0.749 cape vs nominal 0.68). Cheap GPU job.
3. **Sigma stage 3: retire the GP posterior-variance pathway** (tau_pv -> ~0.017/0.006 says it
   carries almost no calibration info). Proposal only — nothing built or run.
4. Non-blocking, offered 07-15: (a) definitive DEC-loRe@5e-3 re-score on the 13 shared val
   windows (the one w31 regression cell; evidence says sampling artifact); (b) quiescent-filter
   audit at 5e-3 for the other DEC members before the conditioned-model design freezes;
   (c) if w31's trainer later beats ep32: re-arm the FT from the new best (manual — the
   autofire marker is set and will NOT refire; remove `rollout_ft_w31_p1a/b` EXISTS-guard
   dirs and run `qg-wiener-conditioning/scripts/sge/submit_w31_p1.sh --go` from that worktree).

## CLUSTER-ONLY ARTIFACTS (the ssh map — not in git by .gitignore policy)

Checkpoints (wiener lane; `training/data` in the worktree is a SYMLINK to the real store):
```
REAL STORE: QG_ROOT/qg-simple-package-stable/src/qg/training/data/ensemble_N5_7lag/training_runs/
   deriv7_cond_local_w31/{best.pt (ep32 = THE model), last.pt, log.csv, config.json,
                          eval_by_root_val.csv (the 41-root eval)}
   deriv7_cond_local_v2/{best.pt, eval_by_root_val.csv}        (width-15 baseline + gate)
   rollout_ft_w31_p1a/, rollout_ft_w31_p1b/                    (filling now, land 07-16 AM)
   rollout_ft_p1_prod/, rollout_ft_p1_lam01/ ...               (prior FT generations)
   deriv7_filtered_floor0.1/ ...                               (unconditioned control)
```
Datasets (same store, per member): `.../training/data/ensemble_N5_7lag/<MEMBER>/`
(deep 28-mark builds `forced_turbulence_dT_*` + training slices `sweep_dT_{5em3,1em2,1p5em2}/`
with packed/ memmaps, split.npz, split_prefilter.npz). Legacy S=4: `.../data/ensemble_N5/`.

Rollout / a-posteriori arrays (multi-GB npz, wiener):
`QG_ROOT/qg-wiener-conditioning/diagnostics/Results/apost_*/` (dissproj, indist, p1lam01, ...).

SGS lane:
```
QG_ROOT/qg-sgs-closure/ml_closure/runs_piff/
   piff_fpc_gjs_ylp75/  piff_cape_gjs_ylp75/    <- THE production candidates:
       best.pt, last.pt, recalibration_structural.yaml, conformal_calibration.yaml,
       eval/ (yamls in git; the .pt are cluster-only)
   piff_fpc_gjs/  piff_cape_gjs/                <- previous finals + crps_head.pt
   (studentt/, ringmask/ = dead ends, kept for the record)
SGS ensemble data: QG_ROOT/qg-simple-package-stable/src/qg/outputs/SGS_closure_ensemble/
   <MEMBER>/DNS_LES_s4*.npz   (incl. the _gaussian_jonly_ylp75.npz filtered targets)
```
Solver checkout: `QG_ROOT/qg-simple-package-stable/` (local git, NO remote; its full code
delta is version-controlled as `vendor_patches/qg-simple/` on main). Venvs: `QG_ROOT/qg-env/`
(closure) and `QG_ROOT/qg-env-piff/` (SGS). Reporting infra: `QG_ROOT/reporting/`
(pending_mail spool, relay.log, next_steps.md, autofire markers).

Figures: the curated review sets ARE in git (un-ignored 07-15): ylp75 eval panels, CRPS
staircases, ring 4-panel, conformal reliability, w31 eval figures. Everything else under
`ml_closure/pngs/` (~118 MB) is cluster-only.

## LAST KNOWN GOOD — the ylp75 promotion evidence (2026-07-15)

| metric                          | gjs final (old) | ylp75 (new) | verdict |
|---------------------------------|-----------------|-------------|---------|
| FPC global R2                   | 0.823           | 0.863       | better  |
| FPC ring-excluded R2 (sdf>1D)   | 0.922           | 0.922       | flat    |
| FPC RMSE                        | 0.300           | 0.265       | better  |
| FPC recal'd test NLL            | -2.744          | -2.847      | better  |
| FPC conformal cov 68/95/99.7    | nominal ±0.6%   | .673/.950/.997 | nominal |
| CAPE global R2                  | 0.935           | 0.950       | better  |
| CAPE ring-excluded R2           | 0.972           | 0.977       | better  |
| CAPE RMSE                       | 0.198           | 0.172       | better  |
| CAPE recal'd test NLL           | -2.324          | -2.334      | better  |
| CAPE conformal cov 68/95/99.7   | nominal ±0.6%   | .687/.959/.998 | nominal |
| ring artefact in truth panel    | visible column  | gone        | fixed   |

Wiener lane last-known-good: w31 ep32 beats cond_v2 in 40/41 (member,dT) cells, median Nddot
ratio 0.665; kf4 Nddot .020/.020/.032. Figures + table:
`qg-wiener-conditioning/diagnostics/Results/w31_eval_20260715/` (in git).

## GIT PARITY AT YIELD (Y1, verified clean before this commit)

| branch                  | HEAD at verification | == origin |
|-------------------------|----------------------|-----------|
| main                    | b6e211c (then this HANDOFF commit) | yes |
| exp/wiener-conditioning | 1358db9              | yes       |
| exp/sgs-closure         | 9a7a70b              | yes       |

Remote: `git@github.com:sanaamouzahir/qg-closure.git`. The narrative record of 07-14/07-15 is
in the two relayed reports (morning narrative 10:30, locations 10:40, ylp75-calibrated 11:30),
the BRANCH_LOGs, and `reporting/next_steps.md`.
