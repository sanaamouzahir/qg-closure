#!/usr/bin/env python
"""gate_piff_events.py -- pre-registered event gate for the wallv2 generation
(bars fixed by Sanaa 2026-07-18, BEFORE the runs): compare a NEW run against a
BASELINE run on the existing per-member diagnostic outputs of
diagnose_sigma_at_events (z_true_median), diagnose_error_tails
(worst0.1pct_SS_share = sq_error_share of the |e| > p99.9 pixels, + r2
fields), and diagnose_mean_prediction (global r2).

Per-member bars (pre-registered -- do NOT tune after seeing results):
  Z   : NEW z_true_median < 3.0            (the model KNOWS its misses;
                                            baseline was 14-17 sigma)
  SS  : NEW worst0.1pct_SS_share <= 0.5 * BASELINE worst0.1pct_SS_share
  R2  : NEW mean-prediction r2 >= BASELINE r2 - 0.005

Verdict tiers:
  PASS             all three bars met on every member          -> exit 0
  PASS-conditional Z and SS both met on a strict majority of members
                   INCLUDING every 'const' member, and R2 met on every
                   member                                       -> exit 2
  REGRESSED        anything else (or missing artifacts)         -> exit 3

Outputs: full per-member table on stdout + gate_wallv2.txt next to the new
run's ckpt; --report-run copies it to <branch>/reports/<name>/ and pushes a
digest event (I23b).

Artifact lookup handles BOTH layouts (results-tree STANDARD 2026-07-17 left
relative symlinks at the old paths):
  runs_piff/<run>/{sigma_at_events,error_tails_diag,mean_prediction_diag}/
  results/<geometry>/<run>/{sigma_at_events,error_tails,mean_prediction}/
with member subdirs named by codename (FPC-const) or modulation
(constant_inflow); members are matched on the 'member' key inside the yamls.

Usage (via piff_tool_job.sh, all.q, chained -hold_jid after the diagnostics):
  python gate_piff_events.py --new-run runs_piff/piff_fpc_gjs_wallv2 \
      --baseline-run runs_piff/piff_fpc_gjs_ylp75 [--report-run gate_wallv2_fpc]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

from member_naming import geometry_name

HERE = Path(__file__).resolve().parent
BRANCH_ROOT = HERE.parent

# ---- pre-registered bars (Sanaa 2026-07-18) -- never tune post hoc ---- #
Z_BAR = 3.0            # NEW z_true_median must be < this
SS_FACTOR = 0.5        # NEW worst0.1pct SS share <= factor * baseline
R2_TOL = 0.005         # NEW mean-prediction r2 >= baseline r2 - tol
TAIL_KEY = 'gt_p99.9'  # diagnose_error_tails TAIL_QS[0] block


def suite_dirs(run_dir, suite_old, suite_new):
    """Candidate directories for one diagnostic suite of one run, covering
    the legacy runs_piff layout and the STANDARD results tree."""
    run_dir = Path(run_dir)
    geom = geometry_name(run_dir.name)
    cands = [run_dir / suite_old,
             run_dir.parent.parent / 'results' / geom / run_dir.name / suite_new,
             HERE / 'results' / geom / run_dir.name / suite_new]
    seen, out = set(), []
    for c in cands:
        r = str(c.resolve()) if c.exists() else str(c)
        if r not in seen:
            seen.add(r)
            out.append(c)
    return out


def _read_yaml(p):
    with open(p) as f:
        d = yaml.safe_load(f)
    return d if isinstance(d, dict) else None


def load_member_metrics(dirs):
    """{member: metrics dict} from <dir>/<membersubdir>/metrics.yaml
    (subdir may be codename or modulation; 'member' key is authoritative)."""
    out = {}
    for d in dirs:
        if not Path(d).is_dir():
            continue
        for p in sorted(Path(d).glob('*/metrics.yaml')):
            m = _read_yaml(p)
            if m and 'member' in m:
                out.setdefault(m['member'], m)
    return out


def load_sigma_metrics(dirs):
    """{member: metrics dict} from diagnose_sigma_at_events output: either
    <dir>/<member>.yaml (default) or <dir>/<modulation>/metrics.yaml
    (--plain-member-names / STANDARD tree)."""
    out = {}
    for d in dirs:
        if not Path(d).is_dir():
            continue
        for p in sorted(Path(d).glob('*.yaml')):
            m = _read_yaml(p)
            if m and 'member' in m and 'z_true_median' in m:
                out.setdefault(m['member'], m)
        for p in sorted(Path(d).glob('*/metrics.yaml')):
            m = _read_yaml(p)
            if m and 'member' in m and 'z_true_median' in m:
                out.setdefault(m['member'], m)
    return out


def tail_ss_share(tails_metrics):
    """worst-0.1% squared-error share from a diagnose_error_tails metrics
    dict (fraction in [0,1])."""
    return float(tails_metrics['exceedance_counts'][TAIL_KEY]['sq_error_share'])


def fmt(v, spec='.4f'):
    if v is None:
        return 'MISSING'
    if isinstance(v, bool):
        return 'ok' if v else 'FAIL'
    return format(v, spec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--new-run', required=True,
                    help='run dir of the NEW model (e.g. runs_piff/piff_fpc_gjs_wallv2)')
    ap.add_argument('--baseline-run', required=True,
                    help='run dir of the BASELINE (e.g. runs_piff/piff_fpc_gjs_ylp75)')
    ap.add_argument('--expect-members', default='',
                    help='comma-separated FULL member list; members absent '
                         'from the new error-tails output are forced into '
                         'the table as missing => REGRESSED (never PASS on '
                         'a shrunken universe; G4 2026-07-19 #2)')
    ap.add_argument('--out-name', default='gate_wallv2.txt',
                    help='verdict file written next to the new run ckpt')
    ap.add_argument('--report-run', default=None)
    args = ap.parse_args()

    new_dir = (HERE / args.new_run) if not Path(args.new_run).is_absolute() \
        else Path(args.new_run)
    base_dir = (HERE / args.baseline_run) if not Path(args.baseline_run).is_absolute() \
        else Path(args.baseline_run)
    if not new_dir.is_dir():
        raise SystemExit(f"new run dir missing: {new_dir}")

    new_sigma = load_sigma_metrics(
        suite_dirs(new_dir, 'sigma_at_events', 'sigma_at_events'))
    new_tails = load_member_metrics(
        suite_dirs(new_dir, 'error_tails_diag', 'error_tails'))
    new_mp = load_member_metrics(
        suite_dirs(new_dir, 'mean_prediction_diag', 'mean_prediction'))
    base_tails = load_member_metrics(
        suite_dirs(base_dir, 'error_tails_diag', 'error_tails'))
    base_mp = load_member_metrics(
        suite_dirs(base_dir, 'mean_prediction_diag', 'mean_prediction'))

    members = sorted(new_tails)
    if not members:
        raise SystemExit(f"no per-member error-tails metrics under {new_dir} — "
                         f"did the diagnostics chain run?")
    # G4 2026-07-19 finding #2: never infer the member universe from the NEW
    # outputs alone — a partially-failed error_tails job would silently shrink
    # n and let a subset PASS. --expect-members pins the ensemble: absent
    # members become all-None rows, which the existing missing-artifacts flow
    # forces to REGRESSED (exit 3) with the absentees named in the table.
    if args.expect_members:
        expected = sorted(m.strip() for m in args.expect_members.split(',')
                          if m.strip())
        members = sorted(set(members) | set(expected))

    rows, missing = [], []
    for m in members:
        r = {'member': m}
        sg = new_sigma.get(m)
        nt, bt = new_tails.get(m), base_tails.get(m)
        nmp, bmp = new_mp.get(m), base_mp.get(m)
        r['z_new'] = float(sg['z_true_median']) if sg else None
        r['ss_new'] = tail_ss_share(nt) if nt else None
        r['ss_base'] = tail_ss_share(bt) if bt else None
        r['r2_new'] = float(nmp['global']['r2']) if nmp else None
        r['r2_base'] = float(bmp['global']['r2']) if bmp else None
        # extra context (not bars): active/quiet-set r2 from error_tails
        for tag, src in (('new', nt), ('base', bt)):
            act = qui = None
            if src:
                for s in src.get('activity_split', []):
                    if s.get('set') == 'active_ge_q90':
                        act = s.get('r2_own_mean')
                    if s.get('set') == 'quiet_le_q50':
                        qui = s.get('r2_own_mean')
            r[f'r2_active_{tag}'] = act
            r[f'r2_quiet_{tag}'] = qui
        for k in ('z_new', 'ss_new', 'ss_base', 'r2_new', 'r2_base'):
            if r[k] is None:
                missing.append(f"{m}: {k}")
        r['Z'] = (r['z_new'] is not None and r['z_new'] < Z_BAR)
        r['SS'] = (r['ss_new'] is not None and r['ss_base'] is not None
                   and r['ss_new'] <= SS_FACTOR * r['ss_base'])
        r['R2'] = (r['r2_new'] is not None and r['r2_base'] is not None
                   and r['r2_new'] >= r['r2_base'] - R2_TOL)
        rows.append(r)

    n = len(rows)
    all_pass = all(r['Z'] and r['SS'] and r['R2'] for r in rows)
    zs_pass = [r for r in rows if r['Z'] and r['SS']]
    const_rows = [r for r in rows if 'const' in r['member'].lower()]
    r2_everywhere = all(r['R2'] for r in rows)
    conditional = (r2_everywhere
                   and len(zs_pass) * 2 > n
                   and all(r['Z'] and r['SS'] for r in const_rows))
    if missing:
        verdict, code = 'REGRESSED (missing artifacts)', 3
    elif all_pass:
        verdict, code = 'PASS', 0
    elif conditional:
        verdict, code = 'PASS-conditional', 2
    else:
        verdict, code = 'REGRESSED', 3

    hdr = ['member', 'z_new', 'Z<3', 'ss_new', 'ss_base', 'SS<=.5b',
           'r2_new', 'r2_base', 'R2>=b-.005', 'r2_act_new', 'r2_act_base']
    lines = [f"wallv2 event gate — NEW {new_dir.name} vs BASELINE {base_dir.name}",
             f"bars (pre-registered, Sanaa 2026-07-18): z_true_median < {Z_BAR}; "
             f"worst0.1pct_SS_share <= {SS_FACTOR}*baseline; "
             f"mean-prediction r2 >= baseline - {R2_TOL}", '',
             ' | '.join(hdr), '-' * 100]
    for r in rows:
        lines.append(' | '.join([
            r['member'],
            fmt(r['z_new'], '.2f'), fmt(r['Z']),
            fmt(r['ss_new'], '.4f'), fmt(r['ss_base'], '.4f'), fmt(r['SS']),
            fmt(r['r2_new'], '.4f'), fmt(r['r2_base'], '.4f'), fmt(r['R2']),
            fmt(r['r2_active_new'], '.3f'), fmt(r['r2_active_base'], '.3f')]))
    lines += ['']
    if missing:
        lines += ['MISSING artifacts (chain incomplete — verdict forced):']
        lines += [f'  {x}' for x in missing]
        lines += ['']
    lines += [f"members: {n}; all-bar pass: "
              f"{sum(1 for r in rows if r['Z'] and r['SS'] and r['R2'])}; "
              f"Z+SS pass: {len(zs_pass)}; R2 pass: "
              f"{sum(1 for r in rows if r['R2'])}",
              f"VERDICT: {verdict} (exit {code})"]
    text = '\n'.join(lines) + '\n'
    print(text, flush=True)
    out_p = new_dir / args.out_name
    out_p.write_text(text)
    print(f"[gate] written {out_p}", flush=True)

    if args.report_run:
        rep = BRANCH_ROOT / 'reports' / args.report_run
        rep.mkdir(parents=True, exist_ok=True)
        (rep / 'summary.md').write_text(
            f"# wallv2 event gate — {new_dir.name} vs {base_dir.name}\n\n"
            f"verdict: **{verdict}**\n\n```\n{text}```\n")
        dw = BRANCH_ROOT / 'diagnostics' / 'digest_writer.py'
        if dw.exists():
            subprocess.run([sys.executable, str(dw), '--repo-dir',
                            str(BRANCH_ROOT), '--run-name', args.report_run,
                            '--event', 'done', '--note',
                            f'{new_dir.name} gate: {verdict}'],
                           capture_output=True)
    sys.exit(code)


if __name__ == '__main__':
    main()
