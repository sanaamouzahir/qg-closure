#!/usr/bin/env python3
"""digest_writer.py -- I23b digest: progress.csv + status.md (+ summary.md) under
<repo>/reports/<run-name>/, committed and pushed per the I22 path partition.
[fable-authored]

CHARTER v1.4 I22/I23 (adopted 2026-07-15, Sanaa's [QG][GLOBAL] order):
  - CLUSTER git writes touch ONLY reports/ and logs/. This module stages with an
    explicit `git add reports/<run-name>/` -- NEVER `git add -A`, never a code/
    doc/ledger path (I22b). A violation here poisons the two-host partition.
  - Every push is `git pull --rebase origin <branch> && git push` in a 3-attempt
    / 10 s backoff retry loop so concurrent monitors survive each other (I22c).
    A failed rebase is aborted before the retry (never leaves a half-rebase).
  - RAW logs (SGE .o/.e, solver stdout) never enter the repo; the digest is the
    only thing pushed (I23a/b). Keep status.md <= 20 lines.

Two callers:
  1. monitor_training.py imports write_digest()/push_reports() and calls them at
     every eval epoch and on every verdict change (training runs).
  2. Batch jobs (T2 rollouts, T3 builds, X4 diag bundles) call the CLI with
     --event to keep a minimal digest for non-epoch jobs:
       python digest_writer.py --repo-dir "$SGE_O_WORKDIR" --run-name <name> \
           --event start --note "member FRC-b2 dT 5.0e-3" --job-id "$JOB_ID"
     Event rows append to reports/<run-name>/progress.csv (timestamp,event,
     job_id,host,note); status.md is rewritten from the last 3 events.

Standalone by design: stdlib only, no torch, no repo imports -- it must run on
any cluster node with the bare venv python and no agent present (I24).
"""
import argparse
import csv
import math
import os
import socket
import subprocess
import time
from datetime import datetime
from pathlib import Path

PUSH_ATTEMPTS = 3
PUSH_BACKOFF_S = 10
STATUS_MAX_LINES = 20
EVENT_COLS = ['timestamp', 'event', 'job_id', 'host', 'note']

# Cluster git quirk (2026-07-15): /usr/bin/git on mseas is 1.8.3.1, which
# predates linked-worktree support -- it cannot even `rev-parse` inside the
# qg-sgs-closure / qg-wiener-conditioning worktrees. The worktree-capable
# 2.9.2 lives at /opt/rocks/bin/git. Prefer it when present; QG_GIT overrides.
GIT = os.environ.get('QG_GIT') or (
    '/opt/rocks/bin/git' if os.path.exists('/opt/rocks/bin/git') else 'git')


def _run(cmd, cwd, timeout=180):
    if cmd and cmd[0] == 'git':
        cmd = [GIT] + list(cmd[1:])
    return subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True,
                          timeout=timeout)


def _fmt(v):
    if v is None:
        return ''
    try:
        return f"{float(v):.6g}"
    except (TypeError, ValueError):
        return str(v)


def report_dir(repo_dir, run_name):
    d = Path(repo_dir) / 'reports' / run_name
    d.mkdir(parents=True, exist_ok=True)
    return d


# ------------------------------------------------------------ training digest

def write_progress(repo_dir, run_name, rows, orders, verdict='OK'):
    """I23b progress.csv: one row per eval epoch -- epoch, train, val, per-order
    val, lr, best-so-far, seconds, verdict. Rewritten whole each call (KB-sized;
    idempotent under monitor restarts). `verdict` stamps the LAST row; earlier
    rows keep OK (history verdicts live in git, not in the file)."""
    d = report_dir(repo_dir, run_name)
    cols = ['epoch', 'train_relL2', 'val_relL2', *orders,
            'lr', 'best_val', 'elapsed_s', 'verdict']
    best = float('inf')
    with open(d / 'progress.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(cols)
        for i, r in enumerate(rows):
            v = r.get('val_relL2')
            if v is not None and math.isfinite(v):
                best = min(best, v)
            w.writerow([int(r.get('epoch', -1)),
                        _fmt(r.get('train_relL2')), _fmt(v),
                        *[_fmt(r.get(o)) for o in orders],
                        _fmt(r.get('lr')),
                        _fmt(best if math.isfinite(best) else None),
                        _fmt(r.get('elapsed_s')),
                        verdict if i == len(rows) - 1 else 'OK'])
    return d / 'progress.csv'


def _last_rows_lines(rows, orders, n=3):
    out = []
    for r in rows[-n:]:
        cells = ' '.join(f"{o.replace('val_', '')}={_fmt(r.get(o))}" for o in orders)
        out.append(f"  ep{int(r.get('epoch', -1))} val={_fmt(r.get('val_relL2'))} {cells}")
    return out or ['  (no eval rows yet)']


def write_status(repo_dir, run_name, header_lines, job_id, verdict, rows, orders,
                 next_items):
    """I23b status.md, <= 20 lines: 6.1 parameter header, job id, node, current
    verdict, last 3 rows of progress.csv, NEXT."""
    d = report_dir(repo_dir, run_name)
    stamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    lines = [f"# {run_name}",
             f"updated {stamp}  host {socket.gethostname()}  job {job_id or '?'}  "
             f"verdict **{verdict}**"]
    lines += [ln for ln in header_lines if ln.strip()][:6]
    lines.append('last eval rows:')
    lines += _last_rows_lines(rows, orders)
    nxt = '; '.join(next_items) if next_items else 'none'
    lines.append(f"NEXT: {nxt[:400]}")
    (d / 'status.md').write_text('\n'.join(lines[:STATUS_MAX_LINES]) + '\n')
    return d / 'status.md'


def write_summary(repo_dir, run_name, rows, orders, verdict, extra_lines=()):
    """On completion (I23b): summary.md with the final table."""
    d = report_dir(repo_dir, run_name)
    stamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    lines = [f"# {run_name} -- summary ({stamp})", f"final verdict: **{verdict}**",
             f"epochs: {len(rows)}"]
    if rows:
        finite = [r for r in rows
                  if math.isfinite(r.get('val_relL2', float('nan')))]
        if finite:
            bi = min(range(len(finite)), key=lambda i: finite[i]['val_relL2'])
            lines.append('| row | epoch | val | ' +
                         ' | '.join(o.replace('val_', '') for o in orders) + ' |')
            lines.append('|---' * (3 + len(orders)) + '|')
            for tag, r in (('best', finite[bi]), ('last', rows[-1])):
                lines.append(f"| {tag} | {int(r.get('epoch', -1))} | "
                             f"{_fmt(r.get('val_relL2'))} | " +
                             ' | '.join(_fmt(r.get(o)) for o in orders) + ' |')
    lines += list(extra_lines)
    (d / 'summary.md').write_text('\n'.join(lines) + '\n')
    return d / 'summary.md'


def write_digest(repo_dir, run_name, rows, orders, header_lines, job_id,
                 verdict='OK', next_items=(), final=False):
    """Convenience: progress.csv + status.md (+ summary.md when final)."""
    write_progress(repo_dir, run_name, rows, orders, verdict)
    write_status(repo_dir, run_name, header_lines, job_id, verdict, rows, orders,
                 list(next_items))
    if final:
        write_summary(repo_dir, run_name, rows, orders, verdict)


# --------------------------------------------------------------- event digest

def append_event(repo_dir, run_name, event, note='', job_id=''):
    """Batch-job digest (T2/T3/X4): append one event row, rewrite status.md."""
    d = report_dir(repo_dir, run_name)
    p = d / 'progress.csv'
    new = not p.exists()
    with open(p, 'a', newline='') as f:
        w = csv.writer(f)
        if new:
            w.writerow(EVENT_COLS)
        w.writerow([datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), event,
                    job_id, socket.gethostname(), note])
    with open(p, newline='') as f:
        events = list(csv.DictReader(f))
    stamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    lines = [f"# {run_name}",
             f"updated {stamp}  host {socket.gethostname()}  job {job_id or '?'}  "
             f"verdict **{event.upper()}**",
             'last events:']
    lines += [f"  {e['timestamp']} {e['event']} ({e['job_id']}) {e['note']}"[:120]
              for e in events[-3:]]
    lines.append(f"NEXT: {note[:200] if event == 'fail' else 'see run emails'}")
    (d / 'status.md').write_text('\n'.join(lines[:STATUS_MAX_LINES]) + '\n')


# ----------------------------------------------------------------------- push

def push_reports(repo_dir, run_name, message):
    """I22b/I22c: stage ONLY reports/<run-name>/, commit, pull --rebase, push;
    3 attempts, 10 s backoff, rebase aborted on failure. Returns True on a
    successful push (or nothing to push AND remote already has our HEAD)."""
    repo = Path(repo_dir)
    if not (repo / '.git').exists():
        print(f"[digest] {repo} is not a git repo root; digest written, not pushed",
              flush=True)
        return False
    rel = f"reports/{run_name}/"
    br = _run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], repo)
    branch = br.stdout.strip() or 'main'
    for attempt in range(1, PUSH_ATTEMPTS + 1):
        _run(['git', 'add', rel], repo)          # explicit path ONLY (I22b)
        c = _run(['git', 'commit', '-m', message], repo)
        if c.returncode != 0 and 'nothing to commit' not in (c.stdout + c.stderr):
            print(f"[digest] commit failed: {(c.stdout + c.stderr).strip()[:300]}",
                  flush=True)
        pl = _run(['git', 'pull', '--rebase', 'origin', branch], repo)
        if pl.returncode != 0:
            _run(['git', 'rebase', '--abort'], repo)
            print(f"[digest] pull --rebase failed (attempt {attempt}): "
                  f"{pl.stderr.strip()[:200]}", flush=True)
        else:
            ps = _run(['git', 'push', 'origin', branch], repo)
            if ps.returncode == 0:
                return True
            print(f"[digest] push failed (attempt {attempt}): "
                  f"{ps.stderr.strip()[:200]}", flush=True)
        if attempt < PUSH_ATTEMPTS:
            time.sleep(PUSH_BACKOFF_S)
    print(f"[digest] WARNING: push did not complete after {PUSH_ATTEMPTS} attempts; "
          f"digest is committed locally in {repo} (next push will carry it)",
          flush=True)
    return False


# ------------------------------------------------------------------------ CLI

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--repo-dir', default=os.environ.get('SGE_O_WORKDIR', '.'),
                    help='branch checkout root (contains reports/)')
    ap.add_argument('--run-name', required=True)
    ap.add_argument('--event', choices=['start', 'done', 'fail', 'diag'],
                    help='batch-job event mode (T2/T3/X4)')
    ap.add_argument('--note', default='')
    ap.add_argument('--job-id', default=os.environ.get('JOB_ID', ''))
    ap.add_argument('--message', default=None, help='commit message override')
    ap.add_argument('--no-push', action='store_true')
    args = ap.parse_args()

    if not args.event:
        ap.error('--event is required in CLI mode (training digests are written '
                 'by monitor_training.py, which imports this module)')
    append_event(args.repo_dir, args.run_name, args.event, args.note, args.job_id)
    print(f"[digest] {args.run_name}: event {args.event} recorded", flush=True)
    if not args.no_push:
        msg = args.message or (f"[cluster][digest] {args.run_name} "
                               f"{args.event} ({args.job_id or 'no-jid'})")
        push_reports(args.repo_dir, args.run_name, msg)


if __name__ == '__main__':
    main()
