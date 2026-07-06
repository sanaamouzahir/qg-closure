#!/usr/bin/env python3
"""monitor_training.py -- live watcher for a training run's log.csv. [fable-authored]

Runs as a low-resource SGE job (scripts/sge/monitor_training_job.sh) submitted
alongside every training job (sge-runner wiring), or under qlogin. NEVER on the
login node (qlogin rule; guard hook enforces).

Polls <run_dir>/log.csv and emails [QG][FLAG][<branch>] with the offending log
lines when a trigger fires (each trigger fires at most once per run):

  EXPLODE    any per-order val > 10x its epoch-5 value, or any non-finite value.
  OSCILLATE  sign changes of d(val_order)/d(epoch) > 6 in a 10-epoch window
             while the other orders are flat (the Ndot signature).
  IMBALANCE  one order improved >5x more than another since epoch 10
             (each relative to its own epoch-10 value), sustained 20 epochs.
  STALL      best val unimproved for 60 epochs with lr still > 0.2x initial.
  LR_SANITY  val worsens monotonically for the first 10 epochs (lr too high).

On completion with no trigger it exits silently -- the usual [QG][LANDED]
notify chain handles healthy completion; this script only flags.

Usage:
  python monitor_training.py --run-dir <dir-with-log.csv> --branch <branch> \
      [--job-id <sge-id>] [--email you@mit.edu] [--interval 600]
"""
import argparse
import csv
import math
import subprocess
import time
from pathlib import Path

ORDER_PREFIX = 'val_'
NON_ORDER_COLS = {'epoch', 'lr', 'train_relL2', 'val_relL2', 'best_val', 'elapsed_s'}
FLAT_REL_RANGE = 0.05      # "other orders flat" = rel range < 5% over the window
OSC_WINDOW = 10
OSC_SIGN_CHANGES = 6
EXPLODE_FACTOR = 10.0
EXPLODE_BASE_EPOCH = 5
IMBALANCE_FACTOR = 5.0
IMBALANCE_START = 10
IMBALANCE_SUSTAIN = 20
STALL_EPOCHS = 60
STALL_LR_FRAC = 0.2
LR_SANITY_EPOCHS = 10


def read_log(path):
    """Parse log.csv -> (order_names, rows). Rows are dicts of floats keyed by column."""
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        rows = []
        for r in reader:
            try:
                rows.append({k: float(v) for k, v in r.items() if k is not None and v not in (None, '')})
            except (TypeError, ValueError):
                # partially-written last line during an epoch flush -- skip
                continue
    orders = [c for c in cols if c.startswith(ORDER_PREFIX) and c not in NON_ORDER_COLS]
    return orders, rows


def fmt_rows(rows, idxs, orders):
    """Render selected log rows as the offending-lines block for the email body."""
    hdr = ['epoch', 'lr', 'val_relL2', 'best_val'] + orders
    lines = [','.join(hdr)]
    for i in idxs:
        r = rows[i]
        lines.append(','.join(f"{r.get(c, float('nan')):.4g}" if c != 'epoch'
                              else f"{int(r.get(c, -1))}" for c in hdr))
    return '\n'.join(lines)


def check_explode(orders, rows):
    for r_i, r in enumerate(rows):
        vals = [r.get(c) for c in orders + ['val_relL2', 'train_relL2']]
        if any(v is not None and not math.isfinite(v) for v in vals):
            return ('EXPLODE', f"non-finite value at epoch {int(r['epoch'])}", [r_i])
    if len(rows) <= EXPLODE_BASE_EPOCH:
        return None
    base = rows[EXPLODE_BASE_EPOCH]
    for o in orders:
        b = base.get(o)
        if not b or b <= 0:
            continue
        for r_i in range(EXPLODE_BASE_EPOCH + 1, len(rows)):
            v = rows[r_i].get(o, 0.0)
            if v > EXPLODE_FACTOR * b:
                return ('EXPLODE',
                        f"{o} = {v:.3g} at epoch {int(rows[r_i]['epoch'])} "
                        f"> {EXPLODE_FACTOR:g}x its epoch-{EXPLODE_BASE_EPOCH} value {b:.3g}",
                        [EXPLODE_BASE_EPOCH, r_i])
    return None


def check_oscillate(orders, rows):
    if len(rows) < OSC_WINDOW + 1:
        return None
    # examine the most recent 10-epoch window each poll
    w = list(range(len(rows) - OSC_WINDOW - 1, len(rows)))
    for o in orders:
        d = [rows[w[i + 1]].get(o, 0) - rows[w[i]].get(o, 0) for i in range(len(w) - 1)]
        sc = sum(1 for i in range(len(d) - 1) if d[i] * d[i + 1] < 0)
        if sc <= OSC_SIGN_CHANGES:
            continue
        others_flat = True
        for oo in orders:
            if oo == o:
                continue
            vs = [rows[i].get(oo, 0) for i in w]
            mid = sorted(vs)[len(vs) // 2]
            if mid > 0 and (max(vs) - min(vs)) / mid >= FLAT_REL_RANGE:
                others_flat = False
                break
        if others_flat:
            return ('OSCILLATE',
                    f"{o}: {sc} sign changes of d(val)/d(epoch) in the last "
                    f"{OSC_WINDOW}-epoch window while other orders are flat "
                    f"(<{FLAT_REL_RANGE:.0%} rel range)", w)
    return None


def check_imbalance(orders, rows):
    if len(rows) <= IMBALANCE_START + IMBALANCE_SUSTAIN:
        return None
    base = {o: rows[IMBALANCE_START].get(o, 0) for o in orders}
    if any(b <= 0 for b in base.values()):
        return None
    streak = 0
    for r_i in range(IMBALANCE_START + 1, len(rows)):
        gains = {o: base[o] / max(rows[r_i].get(o, float('inf')), 1e-300) for o in orders}
        hi, lo = max(gains, key=gains.get), min(gains, key=gains.get)
        cond = gains[hi] > IMBALANCE_FACTOR * gains[lo]
        streak = streak + 1 if cond else 0
        if streak >= IMBALANCE_SUSTAIN:
            return ('IMBALANCE',
                    f"{hi} improved {gains[hi]:.2f}x vs {lo} {gains[lo]:.2f}x since epoch "
                    f"{IMBALANCE_START} (ratio > {IMBALANCE_FACTOR:g}), sustained "
                    f"{IMBALANCE_SUSTAIN} epochs (through epoch {int(rows[r_i]['epoch'])})",
                    [IMBALANCE_START, r_i - IMBALANCE_SUSTAIN + 1, r_i])
    return None


def check_stall(orders, rows):
    if len(rows) < STALL_EPOCHS + 1:
        return None
    lr0 = rows[0].get('lr', 0)
    last_improve = 0
    best = float('inf')
    for r_i, r in enumerate(rows):
        if r.get('val_relL2', float('inf')) < best:
            best = r['val_relL2']
            last_improve = r_i
    r_i = len(rows) - 1
    if (r_i - last_improve >= STALL_EPOCHS
            and rows[r_i].get('lr', 0) > STALL_LR_FRAC * lr0):
        return ('STALL',
                f"best val {best:.3g} unimproved for {r_i - last_improve} epochs "
                f"(since epoch {int(rows[last_improve]['epoch'])}) with lr "
                f"{rows[r_i]['lr']:.3g} > {STALL_LR_FRAC:g}x initial {lr0:.3g}",
                [last_improve, r_i])
    return None


def check_lr_sanity(orders, rows):
    if len(rows) < LR_SANITY_EPOCHS:
        return None
    w = rows[:LR_SANITY_EPOCHS]
    if all(w[i + 1].get('val_relL2', 0) > w[i].get('val_relL2', 0) for i in range(len(w) - 1)):
        return ('LR_SANITY',
                f"val_relL2 worsens monotonically over epochs 0..{LR_SANITY_EPOCHS - 1} "
                f"({w[0]['val_relL2']:.3g} -> {w[-1]['val_relL2']:.3g}) -- lr likely too high",
                list(range(LR_SANITY_EPOCHS)))
    return None


CHECKS = [check_explode, check_oscillate, check_imbalance, check_stall, check_lr_sanity]


def job_alive(job_id):
    return subprocess.run(['qstat', '-j', str(job_id)],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def send_mail(subject, body, to):
    subprocess.run(['mail', '-s', subject, to], input=body.encode(), check=False)
    print(f"[monitor] MAILED: {subject}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True, help='dir containing log.csv')
    ap.add_argument('--branch', required=True, help='branch tag for the [QG][FLAG][<branch>] subject')
    ap.add_argument('--job-id', default=None, help='SGE id of the training job; monitor exits when it leaves qstat')
    ap.add_argument('--email', default='sanaamz@mit.edu')
    ap.add_argument('--interval', type=float, default=600.0, help='poll seconds')
    ap.add_argument('--max-hours', type=float, default=96.0, help='hard runtime cap')
    ap.add_argument('--once', action='store_true', help='single pass (no loop) -- for qlogin spot checks')
    args = ap.parse_args()

    log = Path(args.run_dir) / 'log.csv'
    fired = set()
    t0 = time.time()
    print(f"[monitor] watching {log}  branch={args.branch}  job={args.job_id}  "
          f"interval={args.interval:g}s")

    while True:
        if log.exists():
            try:
                orders, rows = read_log(log)
            except Exception as e:
                print(f"[monitor] log parse error (transient?): {e}")
                orders, rows = [], []
            for chk in CHECKS:
                if chk.__name__ in fired:
                    continue
                hit = chk(orders, rows) if rows else None
                if hit:
                    name, msg, idxs = hit
                    fired.add(chk.__name__)
                    body = (f"run: {args.run_dir}\n"
                            f"trigger: {name}\n"
                            f"{msg}\n\n"
                            f"offending lines:\n{fmt_rows(rows, idxs, orders)}\n")
                    send_mail(f"[QG][FLAG][{args.branch}] monitor: {name} in "
                              f"{Path(args.run_dir).name}", body, args.email)
        done = args.job_id is not None and not job_alive(args.job_id)
        expired = (time.time() - t0) > args.max_hours * 3600
        if args.once or done or expired:
            break
        time.sleep(args.interval)

    if fired:
        print(f"[monitor] exiting; fired: {sorted(fired)}")
    else:
        # healthy: stay silent -- [QG][LANDED] comes from the usual notify chain
        print("[monitor] exiting; no triggers fired (LANDED handled by the usual chain)")


if __name__ == '__main__':
    main()
