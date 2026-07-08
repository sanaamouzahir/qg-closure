# DECISIONS.md — global (main clone) decision ledger

One line per decision: date | tier | what | gate result | commit. Bootstrapped
2026-07-08 (I17): the global supervisor previously had no ledger; the P1 postmortem
below is its founding entry.

---

## 2026-07-08 | P1 POSTMORTEM — monitoring failure on deriv7_cond_local (job 1827034)

Question (Sanaa's ORDER 2): a training job ran 6+ epochs with an order-inverted val
(Ndot 5.4/1.3/3.9 oscillation) and Sanaa detected it from a raw log before any agent did.

(a) WAS monitor_training.py chained? YES — twice. FINALIZE-style 1827035 (-hold_jid
    1827034) and live watchdog 1827036 (all.q, no hold), both submitted 13:52:xx with
    the trainer. Both killed 16:43 in the queue wipe. "Never-chained" is FALSE.
(b) What did they evaluate at ep1–5 and why no trigger?
    - 1827035: NOTHING during training — -hold_jid starts it only after the trainer
      leaves the queue. Post-run by construction.
    - 1827036 (monitor v1): polled log.csv live with triggers EXPLODE / LR-SANITY /
      OSCILLATE / IMBALANCE / STALL. At ep1–5 NONE of these can fire on the incident
      signature: val 3.48 < EXPLODE_ABS 10 and < 8x ep0 (ep0 was already elevated);
      OSCILLATE needs an 11-epoch window; IMBALANCE starts at ep10 + 20-epoch sustain;
      LR-SANITY needs 10 monotone epochs. v1 also emails NOTHING — verdicts print to
      its own log for the supervising session to read, and that session died.
      (1827036's stdout log was not recoverable post-wipe; the no-fire conclusion is
      by-construction from the v1 trigger code, diagnostics/monitor_training.py@aa0b3da.)
(c) FAILURE MODE: DETECTION-GAP (primary — no absolute per-order baseline check; every
    v1 trigger is statistical and structurally blind before ~ep10; ORDER-INVERSION did
    not exist) + SILENT-VERDICTS (secondary — no email path in v1; delivery depended on
    a live supervisor session). This drives I18(c) (ORDER-INVERSION vs the physics-init
    medians 0.19/0.26/0.33, card-based) as the primary fix and I18(b) (monitor emails
    directly: first val epoch, every 5, on trigger; outbox + qsub-notify fallback since
    compute-node sendmail is broken) as the secondary.

Context fact for the record: the CONTROL run (deriv7_filtered_floor0.1) shows the SAME
pooled val_Ndot mean oscillation (5.4 -> 15.4 -> 0.9 over ep0–4, median ~0.2) — the
pooled unfloored eval mean is dominated by near-zero-denominator val samples in ANY run
of this family. Encoded in baseline_cards/T1_deriv7.json (known_metric_caveat) pending
the F1 eval-metric fix; formal per-member D1 verdict from triage job 1827252.

## 2026-07-08 | YELLOW | I18 monitoring tooling committed (monitor v2 authored pre-
disconnect by the prior session, completed + committed this session): monitor_training.py
v2 (LIVE/FINALIZE, email cadence, ORDER-INVERSION, baseline card), monitor_training_job.sh,
baseline_cards/T1_deriv7.json, sge-checker G5 refusal rule (I18a), this ledger. Gate:
sge-checker audit of monitor_training_job.sh; live monitors wired to 1827216 + 1827225.
Charter v1.3 append text (I18/I19) drafted and emailed — file edit is RED, Sanaa pushes.
