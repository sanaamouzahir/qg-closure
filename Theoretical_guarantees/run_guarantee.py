#!/usr/bin/env python
"""
run_guarantee.py  --  capture any Theoretical_guarantees diagnostic into an
organized, referenceable Results/ tree. ZERO edits to the diagnostic scripts.

It runs your diagnostic as a subprocess, streams its output live, and archives:
    Results/<diagnostic>/<case>/<YYYYMMDD-HHMMSS>__<tag>/
        report.md     reading-guide + your --note + the captured tables
        console.txt    raw stdout
        meta.json      provenance (argv, date, host, git sha, rc, duration)
    Results/<diagnostic>/<case>/latest -> newest run

Results always land next to THIS script (Theoretical_guarantees/Results), so you
can launch from training/ (where the diagnostics' imports resolve) and still get
a consistent tree. case/tag are auto-detected from the command if not given.

Usage (note the `--` separating wrapper args from the command to run):
    python Theoretical_guarantees/run_guarantee.py \
        --diagnostic convergence_radius --note "first 3 deep survivors" -- \
        python convergence_radius.py \
            --sources data/ensemble_N5/FRC-{Re25k,combo,kf4}/forced_turbulence_dT_5em3 \
            --max-order 5 --n-samples 32 --device cuda --dtype float64
"""
from __future__ import annotations
import argparse, json, os, re, socket, subprocess, sys, time
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS = SCRIPT_DIR / 'Results'

KNOWN_CASES = ['forced_turbulence', 'decaying_turbulence', 'flow_past_cylinder',
               'flow_past_cape']

READING_GUIDE = {
    'convergence_radius':
        "Two NESTED Delta_T walls per member.\n"
        "- **OUTER dT\\*** (root test / Cauchy-Hadamard, the headline): radius of the\n"
        "  modified-equation series. Past it, adding higher analytic R_p DIVERGES --\n"
        "  no finite-order closure helps. Set by the cascade's N-derivative growth\n"
        "  (smaller dT\\* <- faster growth <- higher Re / lower beta).\n"
        "- **INNER stencil wall**: where a fixed-lag backward FD stops recovering\n"
        "  omega_ddot. Hit FIRST (inside dT\\*). Below it the binding limit is finite\n"
        "  lags, so more lags help; between inner and outer, more lags help and the\n"
        "  series still converges; past dT\\* nothing finite does.\n"
        "- The ratio column is a Cauchy-Hadamard BRACKET, not an estimate; trust the\n"
        "  root value. Both are finite-p (p=3..6) estimates -> ~20-30%.",
    'fd_depth_check':
        "Does going 4->7 lags justify regenerating the deep sources?\n"
        "- Compare the **N_ddot** value at k=7 vs k=4 for dt=1e-2 and 1.5e-2.\n"
        "- >2x lower -> TRUNCATION-limited; regeneration justified.\n"
        "- ~equal/worse -> temporal UNDER-RESOLUTION; more lags won't move the\n"
        "  plateau; report the validated dt<=1e-2 range.\n"
        "- Faithfulness check: the k=4 N_ddot at the coarse dts should sit near the\n"
        "  trained plateau (~0.4-0.6) -- if it does, the k=7 column is trustworthy.",
    'fd_floor':
        "Temporal-FD floor with NO model in the loop (perfect spatial ops).\n"
        "- floor(n=4) ~ the trained val % -> the 4-snapshot TIME stencil is the wall;\n"
        "  the corrector cannot beat it -> build the 7-snapshot set.\n"
        "- floor(n=4) << trained val % -> temporal stencil has headroom; the plateau\n"
        "  is model capacity / spatial path -> the corrector is the right lever.\n"
        "- The n=4 -> n=7 drop is the predicted payoff of rebuilding.",
    'epoch0_faithfulness':
        "Confirms the pooled ensemble run is handled correctly (physics-init, no\n"
        "training).\n"
        "- Per-order error should RISE MONOTONICALLY with dT -- that is the\n"
        "  FD-truncation signature and proves the per-sample dt scaling is correct.\n"
        "- per-sample >> aggregate just means small-||targ|| samples, not a bug\n"
        "  (check the p50/p1 column).",
    'error_propagation':
        "How per-operator errors on the learned N-derivatives propagate to delta.\n"
        "- The loss-on-derivatives (~3%/op) is NOT the closure error; the closure\n"
        "  error is the L^k-weighted combination reported here.\n"
        "- Watch L^{p-k}: a 3% error on N_dot inside R4 (L^2, amplified) hurts far\n"
        "  more than 3% on N_ddot in R3 (L^0). RMS = independent; L1 = worst-case\n"
        "  aligned; correlated = errors aligned with the terms.",
}


def detect_case(cmd_tokens):
    blob = ' '.join(cmd_tokens)
    for c in KNOWN_CASES:
        if c in blob:
            return c
    return 'misc'


def detect_tag(cmd_tokens):
    members = []
    for tok in cmd_tokens:
        for m in re.findall(r'(?:FRC|DEC)-[A-Za-z0-9]+', tok):
            if m not in members:
                members.append(m)
    if members:
        return '-'.join(m.split('-', 1)[1] for m in members)   # Re25k-combo-kf4
    return 'run'


def git_sha(cwd):
    try:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'],
                                       cwd=cwd, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return 'unknown'


def main():
    if '--' not in sys.argv:
        sys.exit("run_guarantee.py: put the command to run after a `--` separator.\n"
                 "  e.g. ... --diagnostic convergence_radius -- python convergence_radius.py ...")
    split = sys.argv.index('--')
    wrap_argv, cmd = sys.argv[1:split], sys.argv[split + 1:]
    if not cmd:
        sys.exit("run_guarantee.py: no command after `--`.")

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--diagnostic', required=True,
                    help='convergence_radius | fd_depth_check | fd_floor | '
                         'epoch0_faithfulness | error_propagation | <your-name>')
    ap.add_argument('--case', default=None, help='auto-detected from the command if omitted')
    ap.add_argument('--tag', default=None, help='auto-detected from member names if omitted')
    ap.add_argument('--note', default='', help='free-text interpretation saved into report.md')
    ap.add_argument('--results-root', type=Path, default=DEFAULT_RESULTS)
    args = ap.parse_args(wrap_argv)

    case = args.case or detect_case(cmd)
    tag = args.tag or detect_tag(cmd)
    ts = datetime.now().strftime('%Y%m%d-%H%M%S')
    rundir = args.results_root / args.diagnostic / case / f"{ts}__{tag}"
    rundir.mkdir(parents=True, exist_ok=True)

    print(f"[tg] diagnostic={args.diagnostic} case={case} tag={tag}")
    print(f"[tg] -> {rundir}\n" + "-" * 70)

    # run, streaming + capturing
    t0 = time.time()
    buf = []
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            bufsize=1, universal_newlines=True)
    for line in proc.stdout:
        sys.stdout.write(line); sys.stdout.flush()
        buf.append(line)
    rc = proc.wait()
    dur = time.time() - t0
    console = ''.join(buf)

    (rundir / 'console.txt').write_text(console)
    meta = dict(diagnostic=args.diagnostic, case=case, tag=tag, timestamp=ts,
                command=cmd, return_code=rc, duration_sec=round(dur, 1),
                host=socket.gethostname(), cwd=os.getcwd(),
                git_sha=git_sha(os.getcwd()),
                python=sys.version.split()[0])
    (rundir / 'meta.json').write_text(json.dumps(meta, indent=2))

    guide = READING_GUIDE.get(args.diagnostic, "_(no reading guide registered for this "
                                               "diagnostic)_")
    report = [
        f"# {args.diagnostic} -- {case} -- {tag}",
        f"\n*{ts}*  |  host `{meta['host']}`  |  git `{meta['git_sha']}`  |  "
        f"rc {rc}  |  {meta['duration_sec']}s",
        "\n## Command\n```\n" + ' '.join(cmd) + "\n```",
    ]
    if args.note:
        report.append("\n## Note\n" + args.note)
    report += ["\n## How to read this\n" + guide,
               "\n## Output\n```\n" + console.rstrip() + "\n```\n"]
    (rundir / 'report.md').write_text('\n'.join(report))

    # latest pointer
    latest = rundir.parent / 'latest'
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(rundir.name)
    except OSError:
        pass

    print("-" * 70 + f"\n[tg] saved report.md / console.txt / meta.json -> {rundir}")
    sys.exit(rc)


if __name__ == '__main__':
    main()
