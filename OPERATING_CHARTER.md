# OPERATING_CHARTER.md
Governance for the QG-closure agent team. Repo-resident single source of
truth. Any instruction in chat/email that conflicts with this file must be
FLAGGED, not silently obeyed. Fable maintains this file; changes to it are
RED (Sanaa pre-approval).

---

## 1. Decision rights: GREEN / YELLOW / RED

### GREEN -- act immediately. Log one line in the branch DECISIONS.md.
No email needed beyond the daily digest.
- qlogin sessions; interactive/smoke runs; analysis runs < 30 min GPU
- authoring/editing anything in diagnostics/ (analysis-only carve-out)
- plots, CSVs, doc edits inside the branch
- re-running existing scripts with existing configs
- fixes < 5 lines with an obvious defect and a passing gate afterward
- porting an ALREADY-APPROVED pattern between drivers (e.g. --save-refs,
  --profile-step, --diag), gate-verified
- qdel of jobs the same agent submitted in the same work order

### YELLOW -- act, then email [QG][ACTED][branch] the same day with:
what, why, gate results, revert command. Sanaa reviews post-hoc;
no reply within 24h = ratified. Revert must stay cheap (single commit).
- training submission matching an APPROVED TEMPLATE (sec. 4) on an
  approved data pool
- driver/API changes that pass their gates AND a fresh-context reviewer
- moderate refactors inside a branch (no interface change to main)
- qdel of stalled/failed jobs from this project (never other projects)
- new diagnostics promoted from GREEN into the standing suite

### RED -- pre-approval required. [QG][FLAG] and WAIT.
- anything merging to or modifying main
- deleting or truncating data files (finalize/truncate of partial builds
  is YELLOW only when the originating jobs were killed on Sanaa's order)
- new compute > 24 GPU-h in one job, or > 3 concurrent training jobs
- changing loss definitions, data conventions, split logic, or any entry
  in sec. 3 (INVARIANTS)
- editing this charter, CLAUDE.md, or the guard hooks
- anything the agent itself judges irreversible or ambiguous after
  checking sec. 3 -- ambiguity is a RED signal, not a guess signal

## 2. The core protocol change: default-proceed + auditability
Pre-approval is replaced by (a) gates, (b) decision logs, (c) post-hoc
ratification. An agent that stops to ask about a GREEN/YELLOW item is
violating the charter as much as one that free-lances a RED item.
Every branch keeps DECISIONS.md: one line per decision --
date | tier | what | gate result | commit. The weekly digest quotes it.

## 3. INVARIANTS (self-check before acting; conflicts -> FLAG)
I1  SGE: `-q ibgpu.q -l gpu=1` ONLY. Never ibamd.q, never -l h_vmem.
    Always -m ea -M $QG_NOTIFY_EMAIL. Monitor chained with -hold_jid.
    Never run compute on the head node (qlogin first).
I2  Precision: float64 end-to-end in the closure build/train pipeline.
    Closure targets are O(dT^3) ~ 1e-9: float32 anywhere in the data path
    is a correctness bug, not a performance choice. (NN inference may run
    f32 ONLY in timing runs, per the rollout's documented mixed-precision
    path.)
I3  FFT budget at inference: the NN is conv-only (zero transforms).
    Closure step total = 8 FFTs (5 solver + 1 psi input-stack infra + 2
    NN-output). Any design that adds transforms per step requires a cost
    table in a PROPOSE email BEFORE code is written. (The spectral
    conditioned layer died for this; do not resurrect it as a deliverable.
    It remains a ceiling-measurement instrument only.)
I4  Truth conventions (two, context-bound; confusing them caused a
    correction cycle):
    - TRAINING TARGETS: K-fold AB2CN2 convention, coef includes
      (1 - 1/K^2). "Truth" = AB2CN2 at h_fine.
    - A-POSTERIORI ROLLOUTS (rollout_aposteriori.py,
      rollout_timed_pareto.py): truth = RK4 at h_fine = dT/K;
      coef = dT^3, coef4 = dT^4, NO (1-1/K^n) factor. K = refinement only.
    Any new driver states which convention it uses in its docstring.
I5  Rollout comparisons: all arms share ONE developed-flow IC
    (rule-15 spinup respected); Brinkman/sponge eta fixed across dt;
    convergence/error norms per the established definitions (centerline
    where the historical plots used centerline).
I6  Tier hygiene: Re25k dT=1.5e-2 is EXCLUDED from training pools
    (past-wall). The 5e-3 tiers of pre-fix builds are storage-noise-floored;
    do not tune to them or report them as model quality.
I7  Ckpt-member pairing: never evaluate a checkpoint on a member outside
    its training pool without labeling the result OOD in the same table.
    OOD numbers are diagnostic, not scoreboard.
I8  Conditioned models: zero-init must reproduce the unconditioned model
    to f64 round-off (the init gate). No training submission before the
    gate passes ON THE CURRENT COMMIT.
I9  YAML: write 5.0e-3, never 5e-3 (PyYAML parses the latter as a string).
I10 Authorship: Fable authors model/training/pipeline code
    ("[fable-authored]"); branch supervisors author diagnostics only and
    fixes < 5 lines; runner agents execute. Unchanged.
I11 Data: inputs.npy/targets.npy are append-complete; killed builders leave
    valid arrays. Finalization = metadata generation + truncation to
    n = min(complete records), never deletion.

## 4. APPROVED TEMPLATES (YELLOW submissions must match one; else RED)
T1  train_deriv family: lr 5e-5, 300 ep, --rel-floor 0.1, f64, batch 4,
    filtered splits, pool per I6, run-name deriv7_<variant>.
    Variants currently approved: cond_local, freeW, lean-stencil S-sweep.
T2  a-posteriori evaluation: rollout_aposteriori.py on an approved ckpt,
    horizon <= 100 turnovers, refs reused via --load-refs when available.
T3  ensemble mmap builds: scripts/sge/build_ensemble_mmap.sh with the
    established flags; forcing gate + slicing + filtering pipeline as-is.
Template changes are RED. A submission citing its template letter in the
[QG][ACTED] email needs no further justification for the config.

## 5. GATES (a PASS substitutes for Sanaa's review)
G1  init-exactness gate: diagnose_condlocal_init.py (or the variant's
    equivalent) -- zero-init == unconditioned to f64 round-off, physics-init
    medians reproduce control. Required before ANY training submission of a
    conditioned model (I8).
G2  data gate: 3 random samples per member load end-to-end through the
    dataset class. Required after any metadata/finalization change and
    before any training submission touching those members.
G3  driver gate: after any rollout/driver edit -- the standing smoke
    (FRC-b2 5e-3, 12 steps, K=20, deriv7_filtered ckpt) reproduces the
    documented reference table to <=1% per arm, PLUS one in-distribution
    smoke (kf4@1.5e-2). Reference tables live in
    diagnostics/REFERENCE_SMOKES.md and are updated only via YELLOW.
G4  reviewer gate: any YELLOW code change gets a fresh-context reviewer
    subagent pass (closure-reviewer); its verdict is quoted in the ACTED
    email. The reviewer checks against THIS FILE, sec. 3 first.
G5  sge gate: sge-checker audit on every submit script (I1 enforcement).

## 6. EMAIL PROTOCOL v2
Categories: SUBMIT | LANDED | ACTED | FLAG | PROPOSE | BLOCKED | DIGEST |
BUILD. New: ACTED (post-hoc YELLOW notification, same-day).
Subject grammar: [QG][CATEGORY][BRANCH] <one-line summary>
Body contract (all categories): <=12 lines, numbers first, then paths,
then next action. FLAG bodies additionally: the exact question, the
options considered, the agent's recommendation, and what happens if no
answer arrives in 24h (safe default).
Ratification rule: ACTED items unanswered for 24h are ratified. FLAG items
unanswered for 24h execute their stated safe default ONLY if the default
is GREEN-tier; otherwise they wait.
DIGEST: daily at 18:00 from each active branch supervisor; weekly Sunday
from Fable (global) -- includes the DECISIONS.md ledger since last digest,
malformed-subject enforcement, and a drift check (any behavior that
required Sanaa intervention -> proposed charter/invariant amendment so the
same intervention is never needed twice).

## 7. THE RATCHET (how this gets less manual over time)
Every Sanaa intervention is treated as a defect in this file, not in the
agent. The weekly digest must convert each intervention into a proposed
amendment: a new invariant, a new gate, a new template, or a tier change
for the decision class involved. Sanaa approves amendments (RED), and the
class of decisions that needed her shrinks monotonically. Target steady
state: Sanaa touches RED items and weekly digests only.
# CHARTER AMENDMENT v1.1 (append to OPERATING_CHARTER.md)

## 3. INVARIANTS -- additions

I12 LOG DISCIPLINE. Every .sh submission script in every branch writes its
    SGE stdout/stderr to <branch>/logs/<run-name>.o$JOB_ID / .e$JOB_ID
    (#$ -o / #$ -e lines mandatory; sge-checker G5 now audits their
    presence). Within 5 minutes of any qsub, the submitting supervisor
    emails [QG][SUBMIT][log] <job-name> with the job id, node, and the
    log path; thereafter it appends the last ~40 log lines to the daily
    digest while the job runs, and sends an updated [QG][SUBMIT][log] on
    any state change (start, first checkpoint, warning, finish). Purpose:
    Sanaa monitors from her phone and replies with instructions; treat a
    reply to a [log] email as a direct order to the branch supervisor.

I13 EDIT AUTONOMY. File edits inside a worktree are NEVER submitted to
    Sanaa for approval -- not in chat, not via permission prompts. The
    protections are the gates (G3/G4), DECISIONS.md, and git history, not
    pre-approval. (Mechanical enforcement: the project settings allow
    Edit/Write without prompting; see ops note in the adoption order.)
    RED-tier files (charter, CLAUDE.md, guard hooks, anything on main)
    remain the only exception.

I14 QDEL + RESUBMIT RIGHTS. A branch supervisor may qdel ITS OWN branch's
    running job and resubmit when it has evidence of: loss blow-up
    (EXPLODE per the monitor), a spotted config/code error, a wrong data
    pool, or a violated invariant. Procedure: qdel -> fix -> gates ->
    resubmit under the same template -> [QG][ACTED][branch] email stating
    (a) what was wrong, with the evidence (log lines / monitor verdict),
    (b) the fix, (c) old and new job ids. No pre-approval. Jobs belonging
    to other branches or to Sanaa directly remain RED.

I15 RESULTS HYGIENE. (a) A-posteriori truth refinement K >= 100 for any
    run reported as a RESULT; K < 100 runs are SMOKEs and every mention of
    their numbers must carry the word SMOKE. (b) No number enters an email
    or a doc without its run parameters recoverable from the same email
    (see 6.1). (c) OOD pairings labeled per I7.

## 6. EMAIL PROTOCOL -- v2.1 additions

6.1 PARAMETER HEADER (mandatory, all categories, before any prose).
    First block of every email = the full context of what was run, so a
    mistake is spottable from the phone without opening anything:
    - Simulation/rollout: member, Re (or nu), beta, mu, grid Nx x Ny,
      Lx x Ly, dT, K, h_fine, scheme per arm, truth convention (I4),
      horizon (steps + turnovers), IC (index/restart + tag), ckpt path,
      forcing, special flags (--r4, --dealias-nn, f32-NN timing mode...).
    - Training: model type, run-name, data pool (members x dts actually
      included, exclusions named), lr, epochs, batch, precision, loss/
      floor, input fields, param count, init-gate result.
    - Build/data: generator script, members, dT, n_marks, n_samples,
      split logic.
    A number whose header omits the parameter that would falsify it is a
    protocol violation (this catches the K=20-reported-as-if-deployment
    class of error).

6.2 NEXT STEPS (mandatory, all categories). Every email ends with a
    "NEXT:" block -- 1-3 concrete options with the supervisor's
    recommendation marked. FLAG emails already required this; it now
    applies to LANDED/SUBMIT/ACTED/DIGEST too. An email that reports a
    problem without a proposed fix is incomplete.

6.3 [QG][SUBMIT][log] subformat: subject carries the job name; body =
    parameter header (6.1) + job id/node + log path + current tail
    (~40 lines) + NEXT block. Replies to these emails are orders (I12).

## 1. DECISION RIGHTS -- reclassifications
- "qdel + fix + resubmit own branch job on evidence" : YELLOW (was
  effectively RED in practice). Governed by I14.
- "file edits inside a worktree" : explicitly GREEN, prompts disabled
  (I13). Was GREEN de jure, RED de facto via permission prompts.

## 7. GIT-VISIBLE STATUS (extends the ratchet)
Sanaa reads git, not just email. Therefore:
- Every branch keeps BRANCH_LOG.md at its root: reverse-chronological,
  3-6 lines per day per active branch -- what ran, what landed, what's
  pending, current job ids. Updated at least daily while active; the
  commit touching it is pushed so it is visible from the GitHub UI.
- Commit messages carry the tier tag: [green]/[yellow]/[red-approved]
  prefix after the [fable-authored] marker where applicable, so the git
  log doubles as the decision ledger.
- DECISIONS.md and BRANCH_LOG.md are always pushed same-day; an unpushed
  ledger is a protocol violation (invisible autonomy is not autonomy).
# CHARTER AMENDMENT v1.1 (append to OPERATING_CHARTER.md)

## 3. INVARIANTS -- additions

I12 LOG DISCIPLINE. Every .sh submission script in every branch writes its
    SGE stdout/stderr to <branch>/logs/<run-name>.o$JOB_ID / .e$JOB_ID
    (#$ -o / #$ -e lines mandatory; sge-checker G5 now audits their
    presence). Within 5 minutes of any qsub, the submitting supervisor
    emails [QG][SUBMIT][log] <job-name> with the job id, node, and the
    log path; thereafter it appends the last ~40 log lines to the daily
    digest while the job runs, and sends an updated [QG][SUBMIT][log] on
    any state change (start, first checkpoint, warning, finish). Purpose:
    Sanaa monitors from her phone and replies with instructions; treat a
    reply to a [log] email as a direct order to the branch supervisor.

I13 EDIT AUTONOMY. File edits inside a worktree are NEVER submitted to
    Sanaa for approval -- not in chat, not via permission prompts. The
    protections are the gates (G3/G4), DECISIONS.md, and git history, not
    pre-approval. (Mechanical enforcement: the project settings allow
    Edit/Write without prompting; see ops note in the adoption order.)
    RED-tier files (charter, CLAUDE.md, guard hooks, anything on main)
    remain the only exception.

I14 QDEL + RESUBMIT RIGHTS. A branch supervisor may qdel ITS OWN branch's
    running job and resubmit when it has evidence of: loss blow-up
    (EXPLODE per the monitor), a spotted config/code error, a wrong data
    pool, or a violated invariant. Procedure: qdel -> fix -> gates ->
    resubmit under the same template -> [QG][ACTED][branch] email stating
    (a) what was wrong, with the evidence (log lines / monitor verdict),
    (b) the fix, (c) old and new job ids. No pre-approval. Jobs belonging
    to other branches or to Sanaa directly remain RED.

I15 RESULTS HYGIENE. (a) A-posteriori truth refinement K >= 100 for any
    run reported as a RESULT; K < 100 runs are SMOKEs and every mention of
    their numbers must carry the word SMOKE. (b) No number enters an email
    or a doc without its run parameters recoverable from the same email
    (see 6.1). (c) OOD pairings labeled per I7.

## 6. EMAIL PROTOCOL -- v2.1 additions

6.1 PARAMETER HEADER (mandatory, all categories, before any prose).
    First block of every email = the full context of what was run, so a
    mistake is spottable from the phone without opening anything:
    - Simulation/rollout: member, Re (or nu), beta, mu, grid Nx x Ny,
      Lx x Ly, dT, K, h_fine, scheme per arm, truth convention (I4),
      horizon (steps + turnovers), IC (index/restart + tag), ckpt path,
      forcing, special flags (--r4, --dealias-nn, f32-NN timing mode...).
    - Training: model type, run-name, data pool (members x dts actually
      included, exclusions named), lr, epochs, batch, precision, loss/
      floor, input fields, param count, init-gate result.
    - Build/data: generator script, members, dT, n_marks, n_samples,
      split logic.
    A number whose header omits the parameter that would falsify it is a
    protocol violation (this catches the K=20-reported-as-if-deployment
    class of error).

6.2 NEXT STEPS (mandatory, all categories). Every email ends with a
    "NEXT:" block -- 1-3 concrete options with the supervisor's
    recommendation marked. FLAG emails already required this; it now
    applies to LANDED/SUBMIT/ACTED/DIGEST too. An email that reports a
    problem without a proposed fix is incomplete.

6.3 [QG][SUBMIT][log] subformat: subject carries the job name; body =
    parameter header (6.1) + job id/node + log path + current tail
    (~40 lines) + NEXT block. Replies to these emails are orders (I12).

## 1. DECISION RIGHTS -- reclassifications
- "qdel + fix + resubmit own branch job on evidence" : YELLOW (was
  effectively RED in practice). Governed by I14.
- "file edits inside a worktree" : explicitly GREEN, prompts disabled
  (I13). Was GREEN de jure, RED de facto via permission prompts.

## 7. GIT-VISIBLE STATUS (extends the ratchet)
Sanaa reads git, not just email. Therefore:
- Every branch keeps BRANCH_LOG.md at its root: reverse-chronological,
  3-6 lines per day per active branch -- what ran, what landed, what's
  pending, current job ids. Updated at least daily while active; the
  commit touching it is pushed so it is visible from the GitHub UI.
- Commit messages carry the tier tag: [green]/[yellow]/[red-approved]
  prefix after the [fable-authored] marker where applicable, so the git
  log doubles as the decision ledger.
- DECISIONS.md and BRANCH_LOG.md are always pushed same-day; an unpushed
  ledger is a protocol violation (invisible autonomy is not autonomy).
# CHARTER AMENDMENT v1.4 (append to OPERATING_CHARTER.md)

Scope note: this amendment changes WHERE agents run and HOW reactivity is
achieved. It changes NOTHING about the branch/agent structure, decision
tiers, gates, templates, invariants I1-I20, or the email protocol. Fable
remains the global supervisor; branch supervisors remain per-worktree;
authorship rules (I10) are unchanged.

---

## 3. INVARIANTS -- additions

I21 EXECUTION MODEL: LOCAL AGENTS, SSH SUBMISSION.
    (a) All agents run on Sanaa's local station. All reading, authoring,
        thinking, review, ledgers, docs, and reports happen locally and
        are pushed to origin.
    (b) No persistent agent process on the cluster. The front end is a
        single shared login node; other users' interactive work has
        priority. Violating this is a RED-tier incident.
    (c) Cluster interaction is NON-INTERACTIVE ssh only, one bounded
        command per call, from the local agent:
            ssh mseas "<cd> && git pull --rebase && qsub ..."
            ssh mseas "qstat -u sanaamz"
            ssh mseas "tail -n 40 <log>" | "grep -n <pattern> <log>"
            scp mseas:<artifact> ./tmp/
        The canonical submission sequence is: PULL -> qsub(job) ->
        qsub(monitor, -hold_jid) -> report job ids. Nothing else.
    (d) Interactive cluster sessions (qlogin) are ephemeral and
        task-scoped: GPU queue only (I1), used for a named diagnostic,
        closed on completion. Never a home for an agent session.
    (e) A cluster job's own runtime products (logs, per-epoch digests,
        result artifacts) are produced ON the cluster by the job itself
        -- that is not an agent on the cluster and is permitted.

I22 PATH PARTITION (makes two-host git safe).
    (a) LOCAL writes: everything EXCEPT reports/ and logs/.
    (b) CLUSTER writes: ONLY reports/ and logs/. Job scripts and monitors
        commit with explicit paths (git add reports/<run>/ logs/<job>),
        NEVER `git add -A`, NEVER a code/doc/ledger path.
    (c) Disjoint paths => merges cannot conflict. Every cluster push is
        `git pull --rebase origin <branch> && git push` in a retry loop
        (3 attempts, 10 s backoff) to survive concurrent monitors.
    (d) A cluster commit touching anything outside (b) is a protocol
        violation: Fable reverts it and reports in the digest.

I23 LOGGING GRANULARITY: DIGEST-PUSH, RAW-ON-DEMAND.
    (a) RAW logs (SGE .o/.e, full solver stdout) stay ON the cluster in
        the branch's logs/. They are never committed, never pulled whole,
        never pasted into an agent context.
    (b) Every monitored job writes a DIGEST to reports/<run-name>/:
        - progress.csv : one row per eval epoch -- epoch, train, val,
          per-order val (Ndot/Nddot/N3dot or the run's equivalent), lr,
          best-so-far, seconds, verdict
        - status.md    : <= 20 lines -- 6.1 parameter header, job id,
          node, current verdict, last 3 rows of progress.csv, NEXT
        - on completion: summary.md with the final table.
        The digest is committed and pushed (I22b) at every eval epoch.
    (c) Fable reads DIGESTS by pulling the repo -- default and cheap
        (~KB). It fetches RAW slices over ssh ONLY when a digest verdict
        or Sanaa's question requires it, and ONLY bounded
        (tail -n <= 100, or grep). Unbounded remote reads are a
        protocol violation (context bloat is the failure mode).
    (d) Artifacts (.npz/.png) stay on the cluster; scp only the specific
        file needed, into ./tmp/ (git-ignored).

I24 REACTIVITY: REFLEX vs JUDGMENT.
    Cluster-side monitors act WITHOUT an agent. The decision tree's
    deterministic branches are CODE (reflexes) and fire in seconds; only
    ambiguity waits for a local agent session. Reflexes are pre-approved
    by this charter and require no per-case approval.

    REFLEX LADDER (monitor_training.py, cluster-resident, chained per
    I18a):
      X1 STOP-ON-EXPLODE: NaN/inf loss, or val > 10x its best, or CFL/
         enstrophy blowup where applicable -> qdel the job IMMEDIATELY,
         write status.md verdict=EXPLODE, email [QG][MONITOR][branch].
      X2 STOP-ON-INVERSION: I18c ORDER-INVERSION persisting 3 consecutive
         evals after epoch 2 -> qdel, verdict=ORDER-INVERSION, email.
      X3 STOP-ON-STALL: 60 epochs with no best AND lr > 0.2 x lr_init ->
         qdel, verdict=STALL, email.
      X4 AUTO-DIAGNOSE (fires on ANY of X1-X3): immediately qsub the
         standing diagnostic bundle on the last saved checkpoint
         (per-member per-order MEDIAN vs MEAN, worst-sample ||target||,
         zero-init reference where defined); push its output to
         reports/<run-name>/diag_<trigger>/ ; email the table when done.
         Evidence must be READY, not requested, when Sanaa next looks.
      X5 AUTO-RESUBMIT -- WHITELIST ONLY: node failure, scheduler
         preemption, transient CUDA/driver error. Resubmit the IDENTICAL
         script ONCE, email old+new job ids. NEVER resubmit on a
         scientific failure (X1-X3): those need a code change, code
         changes are authored locally, so they wait for a session --
         by design, not by limitation.
      X6 HEARTBEAT: if a job is running and no digest row has been
         written in 3x the median epoch time, email verdict=SILENT.
    Anything not X1-X6 waits for the next local session. The monitor
    never edits code, never changes configs, never picks a fix.

I25 SESSION-OPEN PROTOCOL (how judgment catches up).
    Every local agent session begins with, in order:
      1. git pull
      2. read reports/*/status.md for all runs marked active
      3. ssh mseas "qstat -u sanaamz"
      4. reconcile: any verdict != OK, or any job in the digest that
         qstat says is gone -> that is the session's first agenda item,
         before anything Sanaa asks for.
    Sanaa's replies to [MONITOR]/[SUBMIT][log] emails while away are
    ORDERS QUEUED for the next session (amends I12: replies are orders,
    but execution is session-bound, not instant).


## 6. EMAIL PROTOCOL -- v2.2 additions

6.4 Emails now originate from TWO sources; both keep the 6.1 header and
    the NEXT block:
    - CLUSTER (monitors, job scripts): [QG][MONITOR][branch] and
      [QG][SUBMIT][log]. Reflex verdicts and their auto-diagnoses. These
      arrive at any hour; they report what the reflex ALREADY did.
    - LOCAL (agents): everything else. These report judgment.
    A MONITOR email must state which reflex fired, what it did (qdel'd /
    resubmitted / diagnosed), and what is WAITING for a session.

## 4. APPROVED TEMPLATES -- addition

T1-T4 submissions are made via the I21c ssh sequence. The template's
job script must, per I18a/I23b/I22b: chain the monitor, write raw logs to
the branch logs/, write+push the digest to reports/<run-name>/, and
carry the reflex ladder (I24) in the chained monitor.

## 7. THE RATCHET -- addition

The reflex ladder is ratchet-eligible: every time a local session takes an
action on a trigger that was mechanical (no judgment used), the weekly
digest must propose promoting that action into an X-rung. The set of
things that wait for Sanaa shrinks; the set of things that fire in seconds
grows.
