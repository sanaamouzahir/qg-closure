# The agent team — how it's wired and how to drive it

## Mental model
- **You** = the PI. Physics-reality authority.
- **Supervisors are sessions, not files.** The agent you talk to is the interactive Claude Code
  session running in a worktree. You set *which* supervisor it is by the model you launch with
  and the brief it reads — not by an agent file.
    * Global supervisor = a session in the **main** checkout, launched as **Fable 5**. Portfolio
      + merges. Reads `.claude/GLOBAL_SUPERVISOR.md`.
    * Branch supervisor = a session in a **worktree**, launched as **Opus 4.8**. Runs one idea.
      Reads its `SUPERVISOR_BRIEF.md`.
- **Subagents are the shared team** in `.claude/agents/` — committed once, inherited by every
  branch. Supervisors delegate to them; they report up, never to you directly.

## The six subagents
| Agent | Model | You'd invoke it for |
|---|---|---|
| sge-runner | Sonnet | "submit the b25 ensemble and chain the post-pipeline + email" |
| sge-checker | Sonnet | auto-runs before submitting a new/edited script |
| pipeline-runner | Sonnet | "rerender the videos and rebuild the pareto figure" |
| results-summarizer | Sonnet | "give me the verdict on last night's run" |
| closure-reviewer | Opus | "review the diff I just made in training/" |
| physics-sanity | Opus | "this val looks too low — is it real?" |

## How to direct a supervisor
Talk in goals, not commands. Examples:
- "Retrain with hidden=128 and a 4th time level. Have sge-checker audit the script, then submit
  with the completion-email chain. Summarize when it lands and flag anything that looks too good."
- "Compare this branch's floor to the beta-feature branch and tell me if they're converging."
The supervisor decomposes and delegates. You approve the yellow/red actions.

## Controlling autonomy — three layers (weakest to strongest)
1. **CLAUDE.md (advisory).** Physics rules, conventions. The agent *should* follow; not enforced.
2. **permissions (`.claude/settings.json`).** `ask` before `qsub`, `git push`, `rm -rf`; `deny`
   force-push and hard-reset. This is where you set "propose vs act."
3. **hooks (`.claude/hooks/guard_bash.sh`) — HARDCODED, cannot be talked around.** Blocks the
   forbidden SGE flags (`ibamd.q`, `h_vmem`), any push to `main`, and float32 in closure commands.

Rule of thumb: **mechanical invariants → hooks** (enforced), **judgment → CLAUDE.md + the brief's
autonomy dial** (advisory). To loosen a branch you trust, relax its `ask` rules; to tighten, add
to `deny` or the hook. Nothing removes the hard SGE/float64/main-push guards without editing the hook.

## The autonomy dial (in every branch brief)
GREEN act freely (read, plot, review, draft) · YELLOW propose + wait (submit jobs, edit train/data
code) · RED never without explicit go (push main, merge, delete, touch legacy/snapshots).
Default is YELLOW. Tell your supervisor "you're green on X for this branch" to widen it.

## Continuity across check-ins
Each branch has a `BRANCH_LOG.md`. The supervisor reads it at session start and updates it at the
end — that's how "here's what I want to see next time" persists when you close your laptop.

## Notify email
sge-runner uses `$QG_NOTIFY_EMAIL`. Set it once on the cluster:
`echo 'export QG_NOTIFY_EMAIL=you@mit.edu' >> ~/.bashrc`. Then sims mail you on end/abort via
`-m ea -M`, and the post-pipeline auto-fires via `-hold_jid`. No agent babysitting a queue.
