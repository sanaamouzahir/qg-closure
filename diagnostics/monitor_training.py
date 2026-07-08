#!/usr/bin/env python3
"""monitor_training.py -- live watcher + postmortem verdict for a training run. [fable-authored]

v2 (2026-07-08, CHARTER v1.3 I18): monitoring is part of the submission, not an
accessory. One canonical implementation, submitted TWICE per training job by
scripts/sge/monitor_training_job.sh:

  LIVE      submitted at qsub time WITHOUT -hold_jid; polls <run_dir>/log.csv
            while the trainer runs, emails [QG][MONITOR][<branch>] at the first
            val epoch, every --cadence epochs after, and IMMEDIATELY on any
            trigger. Emits the final verdict when the trainer leaves the queue.
  FINALIZE  submitted WITH -hold_jid <trainer> and --finalize; starts after the
            trainer exits. If the live watcher already emailed a final verdict
            (outbox check), exits silently; otherwise it is the safety net that
            produces the postmortem email the live watcher failed to send.

Triggers (each fires at most once; ORDER-INVERSION and the legacy statistical
triggers email and KEEP WATCHING; EXPLODE and FAIL email and exit):

  EXPLODE          non-finite train/val, pooled val > 10 abs or > 8x ep0, or a
                   per-order val > 10x its epoch-5 value. ACTION per I14:
                   qdel + diagnose + resubmit (branch authority; not BLOCKED).
  ORDER-INVERSION  (I18c, new) after epoch 2: any order's val rel-L2 exceeds
                   its baseline-card physics-init median by > inversion_factor,
                   OR the documented easy->hard ordering is violated.
                   Loss-level triggers alone are insufficient -- the 2026-07-08
                   deriv7_cond_local incident (job 1827034) is the proof case.
  LR_SANITY        val worsens monotonically over the first 10 epochs.
  OSCILLATE        >6 derivative sign changes of one order in a 10-ep window
                   while the others are flat (the Ndot signature).
  IMBALANCE        one order improved >5x more than another since epoch 10,
                   sustained 20 epochs.
  STALL            best val unimproved for 60 epochs with lr > 0.2x initial.
  FAIL             traceback in the stdout log, or trainer left the queue
                   before its first val epoch.
  DONE             trainer left the queue after >= 1 val epoch (final verdict
                   email summarises best/last vs the baseline card).

Baseline card (I18d): --baseline-card <json> carries the template's expected
curve -- physics-init medians, expected ordering, control-run reference epochs.
The monitor compares against the card, not just against NaN/explosion. JSON on
purpose (hard rule 4: PyYAML float parsing is banned territory).

Email path: compute nodes on this cluster have a broken sendmail
(libmysqlclient.so.18 missing -- observed 2026-07-08); the head node works.
send_email() therefore ALWAYS writes the body to <run_dir>/monitor_outbox/ and
then tries mailx/mail/sendmail; if all fail it submits a subject-only qsub
notify job (-m e) so the verdict still reaches a phone via qmaster mail, with
the full table in the outbox file. Silence is a violation (I18b).

Usage:
  python monitor_training.py --run-dir <dir-with-log.csv> --branch <branch> \
      [--job-id <sge-id>] [--baseline-card diagnostics/baseline_cards/T1_deriv7.json] \
      [--log <trainer-stdout-log>] [--email you@mit.edu] [--interval 120] \
      [--cadence 5] [--finalize] [--once] [--exit-on-trigger]
"""
import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

ORDER_PREFIX = 'val_'
NON_ORDER_COLS = {'epoch', 'lr', 'train_relL2', 'val_relL2', 'best_val', 'elapsed_s'}
FLAT_REL_RANGE = 0.05      # "other orders flat" = rel range < 5% over the window
OSC_WINDOW = 10
OSC_SIGN_CHANGES = 6
EXPLODE_ABS = 10.0         # pooled val above this = blow-up regardless of ep0
EXPLODE_REL_EP0 = 8.0      # pooled val above 8x its ep0 value
EXPLODE_FACTOR = 10.0      # a per-order val above 10x its epoch-5 value
EXPLODE_BASE_EPOCH = 5
IMBALANCE_FACTOR = 5.0
IMBALANCE_START = 10
IMBALANCE_SUSTAIN = 20
STALL_EPOCHS = 60
STALL_LR_FRAC = 0.2
LR_SANITY_EPOCHS = 10
FATAL = {'EXPLODE', 'FAIL'}   # email + exit; everything else emails + keeps watching

# 6.1 header: config.json keys worth a phone screen, in display order.
HEADER_KEYS = ('model', 'run_name', 'lr', 'epochs', 'batch_size', 'compute_dtype',
               'rel_floor', 'n_snapshots', 'out_orders', 'grad_kernel', 'n_params',
               'seed', 'weight_decay')


# ---------------------------------------------------------------- log parsing

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


def fmt_rows(rows, idxs, orders, medians=None):
    """Render selected log rows; with a card, append (xmed) ratios per order."""
    hdr = ['epoch', 'lr', 'val_relL2', 'best_val'] + orders
    lines = [','.join(hdr) + (',  ratios-to-physics-init-median' if medians else '')]
    for i in idxs:
        r = rows[i]
        cells = [f"{int(r.get('epoch', -1))}"]
        cells += [f"{r.get(c, float('nan')):.4g}" for c in hdr[1:]]
        line = ','.join(cells)
        if medians:
            rat = ' '.join(f"{o.replace(ORDER_PREFIX, '')}x{r.get(o, float('nan')) / medians[o]:.1f}"
                           for o in orders if medians.get(o))
            line += f"   [{rat}]"
        lines.append(line)
    return '\n'.join(lines)


# ------------------------------------------------------------------- triggers

def check_explode(orders, rows, card):
    for r_i, r in enumerate(rows):
        vals = [r.get(c) for c in orders + ['val_relL2', 'train_relL2']]
        if any(v is not None and not math.isfinite(v) for v in vals):
            return ('EXPLODE', f"non-finite value at epoch {int(r['epoch'])}", [r_i])
        v = r.get('val_relL2')
        if v is not None and v > EXPLODE_ABS:
            return ('EXPLODE', f"val_relL2 = {v:.3g} > {EXPLODE_ABS:g} absolute at "
                               f"epoch {int(r['epoch'])}", [0, r_i])
        v0 = rows[0].get('val_relL2') if rows else None
        if v is not None and v0 and v > EXPLODE_REL_EP0 * v0:
            return ('EXPLODE', f"val_relL2 = {v:.3g} > {EXPLODE_REL_EP0:g}x its ep0 "
                               f"value {v0:.3g} at epoch {int(r['epoch'])}", [0, r_i])
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


def check_order_inversion(orders, rows, card):
    """I18c. Needs a baseline card; silent (never fires) without one."""
    if not card or not rows:
        return None
    medians = card_medians(card, orders)
    if not medians:
        return None
    factor = card.get('inversion_factor', 1.5)
    min_ep = card.get('inversion_min_epoch', 2)
    ordering = [ORDER_PREFIX + o for o in card.get('expected_ordering', [])]
    ordering = [o for o in ordering if o in orders]
    for r_i, r in enumerate(rows):
        if r.get('epoch', 0) <= min_ep:
            continue
        for o in orders:
            med, v = medians.get(o), r.get(o)
            if med and v is not None and v > factor * med:
                return ('ORDER-INVERSION',
                        f"{o} = {v:.3g} at epoch {int(r['epoch'])} > {factor:g}x its "
                        f"physics-init median {med:.3g} (baseline card "
                        f"'{card.get('template', '?')}')", [max(0, r_i - 2), r_i])
        vs = [r.get(o) for o in ordering]
        if len(vs) >= 2 and all(v is not None for v in vs):
            for a in range(len(vs) - 1):
                if vs[a] > vs[a + 1]:
                    return ('ORDER-INVERSION',
                            f"easy->hard ordering violated at epoch {int(r['epoch'])}: "
                            f"{ordering[a]} = {vs[a]:.3g} > {ordering[a + 1]} = "
                            f"{vs[a + 1]:.3g} (card expects "
                            f"{' < '.join(card['expected_ordering'])})",
                            [max(0, r_i - 2), r_i])
    return None


def check_oscillate(orders, rows, card):
    if len(rows) < OSC_WINDOW + 1:
        return None
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


def check_imbalance(orders, rows, card):
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


def check_stall(orders, rows, card):
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


def check_lr_sanity(orders, rows, card):
    if len(rows) < LR_SANITY_EPOCHS:
        return None
    w = rows[:LR_SANITY_EPOCHS]
    if all(w[i + 1].get('val_relL2', 0) > w[i].get('val_relL2', 0) for i in range(len(w) - 1)):
        return ('LR_SANITY',
                f"val_relL2 worsens monotonically over epochs 0..{LR_SANITY_EPOCHS - 1} "
                f"({w[0]['val_relL2']:.3g} -> {w[-1]['val_relL2']:.3g}) -- lr likely too high",
                list(range(LR_SANITY_EPOCHS)))
    return None


# ORDER-INVERSION first: it is the absolute-baseline check the 1827034 incident
# proved necessary; the statistical checks are relative and blind to a run that
# is born broken.
CHECKS = [check_explode, check_order_inversion, check_oscillate,
          check_imbalance, check_stall, check_lr_sanity]


# --------------------------------------------------------------- card / header

def load_card(path):
    if not path:
        return None
    try:
        with open(path) as f:
            card = json.load(f)
        card['_path'] = str(path)
        return card
    except Exception as e:
        print(f"[monitor] WARNING: baseline card unreadable ({e}); "
              "ORDER-INVERSION and card comparisons disabled.", flush=True)
        return None


def card_medians(card, orders):
    """Map card physics_init_median keys (Ndot,...) onto log columns (val_Ndot,...)."""
    med = card.get('physics_init_median', {}) if card else {}
    return {o: med[o.replace(ORDER_PREFIX, '')] for o in orders
            if o.replace(ORDER_PREFIX, '') in med}


def param_header(run_dir, branch, job_id, card):
    """6.1 parameter header from <run_dir>/config.json -- first block of every email."""
    lines = [f"run: {Path(run_dir).name}  branch: {branch}  trainer-job: {job_id or '?'}",
             f"run_dir: {run_dir}", f"ckpt: {run_dir}/best.pt (+ last.pt)"]
    cfg_path = Path(run_dir) / 'config.json'
    try:
        with open(cfg_path) as f:
            cfg = json.load(f)
        kv = [f"{k}={cfg[k]}" for k in HEADER_KEYS if k in cfg]
        roots = cfg.get('sweep_roots') or cfg.get('roots')
        if roots:
            kv.append(f"pool={len(roots)} roots ({Path(str(roots[0])).parent.name}, ...)")
        lines.append('  '.join(kv) if kv else "config.json has no known keys")
    except Exception:
        lines.append(f"6.1 HEADER INCOMPLETE: {cfg_path} unreadable -- "
                     "recover parameters from the submit script before quoting numbers")
    if card:
        med = card.get('physics_init_median', {})
        lines.append(f"baseline card: {card.get('template', '?')} ({card.get('_path')})  "
                     f"physics-init medians: "
                     + ' '.join(f"{k}={v}" for k, v in med.items()))
    return '\n'.join(lines)


# --------------------------------------------------------------------- email

def send_email(subject, body, to, run_dir, notify_qsub=True):
    """Outbox first (always), then mailx/mail/sendmail, then a subject-only
    qsub notify job as the last resort (compute-node sendmail is broken on this
    cluster; qmaster mail is not). Returns the delivery channel used."""
    outbox = Path(run_dir) / 'monitor_outbox'
    outbox.mkdir(exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%dT%H%M%S')
    slug = ''.join(ch if ch.isalnum() else '_' for ch in subject)[:80]
    out_file = outbox / f"{stamp}_{slug}.txt"
    out_file.write_text(f"Subject: {subject}\nTo: {to}\nDate: {stamp}\n\n{body}\n")

    text = f"{body}\n\n[full copy: {out_file}]"
    for cmd in (['mailx', '-s', subject, to], ['mail', '-s', subject, to]):
        if shutil.which(cmd[0]):
            r = subprocess.run(cmd, input=text.encode(), capture_output=True)
            if r.returncode == 0 and not r.stderr:
                print(f"[monitor] MAILED via {cmd[0]}: {subject}", flush=True)
                return cmd[0]
    sm = shutil.which('sendmail') or '/usr/sbin/sendmail'
    if os.path.exists(sm):
        msg = f"To: {to}\nSubject: {subject}\n\n{text}"
        r = subprocess.run([sm, '-t'], input=msg.encode(), capture_output=True)
        if r.returncode == 0 and not r.stderr:
            print(f"[monitor] MAILED via sendmail: {subject}", flush=True)
            return 'sendmail'
    if notify_qsub and shutil.which('qsub'):
        # Subject-only phone ping via qmaster mail; body stays in the outbox.
        name = ('QGMON_' + ''.join(ch if ch.isalnum() else '_' for ch in subject)
                .replace('_QG__MONITOR__', ''))[:120]
        r = subprocess.run(['qsub', '-b', 'y', '-N', name, '-m', 'e', '-M', to,
                            '-o', str(outbox), '-e', str(outbox), '/bin/true'],
                           capture_output=True, text=True)
        if r.returncode == 0:
            print(f"[monitor] mail broken on this node -- qsub notify sent "
                  f"({name}); body in {out_file}", flush=True)
            return 'qsub-notify'
    print(f"[monitor] WARNING: NO delivery channel worked; verdict only in "
          f"{out_file}", flush=True)
    return 'outbox-only'


def _indent_block(text, pad='    '):
    return '\n'.join(pad + ln if ln.strip() else ln for ln in text.splitlines())


def _numbered(items, pad='    '):
    """Sanaa's email convention: indented 'N. ' points with a blank line
    between every point."""
    out = []
    for i, it in enumerate(items, 1):
        body = _indent_block(it, pad + '   ').lstrip()
        out.append(f"{pad}{i}. {body}")
    return '\n\n'.join(out)


def build_body(kind, msg, run_dir, branch, job_id, card, orders, rows, idxs, next_items):
    """Body layout per Sanaa's convention (2026-07-08): PARAMETERS OF THE RUN
    first; BOLD + CAPITAL section titles; indented, numbered, blank-line-
    separated points; highlight what matters."""
    medians = card_medians(card, orders) if card else None
    recent = sorted(set((idxs or []) + list(range(max(0, len(rows) - 6), len(rows)))))
    table = fmt_rows(rows, recent, orders, medians) if rows else '(no epochs parsed yet)'
    hdr_items = [ln for ln in param_header(run_dir, branch, job_id, card).splitlines()
                 if ln.strip()]
    return ('**PARAMETERS OF THE RUN**\n\n'
            + _numbered(hdr_items) + '\n\n\n'
            + f'**VERDICT: {kind}**\n\n'
            + _indent_block(msg) + '\n\n\n'
            + '**PER-ORDER TABLE (RECENT + OFFENDING EPOCHS)**\n\n'
            + _indent_block(table) + '\n\n\n'
            + '**NEXT**\n\n'
            + _numbered(next_items) + '\n')


def next_for(kind, run_dir):
    if kind == 'EXPLODE':
        return ["per I14: **QDEL + DIAGNOSE + RESUBMIT** under the same template "
                "(branch authority; not BLOCKED). Attach this table to the ACTED email."]
    if kind == 'ORDER-INVERSION':
        return ["(RECOMMENDED) per-member MEDIAN-vs-MEAN before trusting any pooled number:\n"
                f"python diagnostics/diagnose_error_distribution.py --ckpt {run_dir}/last.pt\n"
                "attach its table to the follow-up (I18c); loss-level triggers are insufficient.",
                "if median ~ card level but mean >> median: unfloored-eval artifact -> F1 "
                "metric fix (floored/median eval); qdel+resubmit per I14 if best-selection "
                "is corrupted.",
                "escalate **[QG][BLOCKED] to Fable SAME DAY** if unresolved in one cycle (I19)."]
    if kind == 'DONE':
        return ["eval per-root breakdown (eval_deriv_by_root.py) BEFORE quoting the pooled val.",
                "compare best/last vs the baseline card plateau in the LANDED email."]
    if kind == 'FAIL':
        return ["read the log tail above, fix, resubmit per I14; if the cause is outside "
                "the branch, **[QG][BLOCKED]** to Fable."]
    return ["no action needed; next cadence email in --cadence epochs."]


# ----------------------------------------------------------------------- main

def job_alive(job_id):
    return subprocess.run(['qstat', '-j', str(job_id)],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True, help='dir containing log.csv')
    ap.add_argument('--branch', required=True, help='branch tag for the [QG][MONITOR][<branch>] subject')
    ap.add_argument('--job-id', default=None, help='SGE id of the training job; monitor exits when it leaves qstat')
    ap.add_argument('--log', default=None, help='trainer stdout log (optional; enables traceback detection)')
    ap.add_argument('--baseline-card', default=None, help='JSON baseline card (I18d); required for ORDER-INVERSION')
    ap.add_argument('--email', default=os.environ.get('QG_NOTIFY_EMAIL', 'sanaamz@mit.edu'))
    ap.add_argument('--interval', type=float, default=120.0, help='poll seconds')
    ap.add_argument('--cadence', type=int, default=5, help='email every N epochs (I18b)')
    ap.add_argument('--max-hours', type=float, default=96.0, help='hard runtime cap')
    ap.add_argument('--once', action='store_true', help='single pass (no loop) -- for qlogin spot checks')
    ap.add_argument('--finalize', action='store_true',
                    help='held postmortem mode (-hold_jid): only email if the live watcher did not')
    ap.add_argument('--exit-on-trigger', action='store_true',
                    help='exit 2 on ANY trigger (supervisor-side background use); default: only EXPLODE/FAIL exit')
    args = ap.parse_args()

    run_dir = str(Path(args.run_dir))
    log = Path(run_dir) / 'log.csv'
    card = load_card(args.baseline_card)
    fired = set()
    last_mailed_ep = None
    t0 = time.time()
    subj = lambda tag, ep: f"[QG][MONITOR][{args.branch}] {Path(run_dir).name} ep{ep} {tag}"

    def email(kind, msg, orders, rows, idxs):
        ep = int(rows[-1]['epoch']) if rows else -1
        body = build_body(kind, msg, run_dir, args.branch, args.job_id, card,
                          orders, rows, idxs, next_for(kind, run_dir))
        send_email(subj(kind, ep), body, args.email, run_dir)

    if args.finalize:
        # Safety net: the trainer has exited (we were held on it). If the live
        # watcher already sent a final verdict, stay silent; else send it now.
        outbox = Path(run_dir) / 'monitor_outbox'
        finals = list(outbox.glob('*_DONE*.txt')) + list(outbox.glob('*_FAIL*.txt')) \
            + list(outbox.glob('*_EXPLODE*.txt')) if outbox.exists() else []
        if finals:
            print(f"[monitor] finalize: live watcher already delivered "
                  f"({finals[-1].name}); exiting silently.", flush=True)
            return
        orders, rows = read_log(log) if log.exists() else ([], [])
        if rows:
            hits = [chk(orders, rows, card) for chk in CHECKS]
            hits = [h for h in hits if h]
            if hits:
                kind, msg, idxs = hits[0]
                email(kind, f"(postmortem -- live watcher missed this) {msg}", orders, rows, idxs)
            else:
                email('DONE', f"postmortem: {len(rows)} epochs, no trigger; live watcher "
                              "sent no final verdict (investigate why -- I18b).", orders, rows, [])
        else:
            email('FAIL', 'postmortem: trainer exited with an empty/absent log.csv '
                          '(died before the first val epoch).', [], [], [])
        return

    print(f"[monitor] watching {log}  branch={args.branch}  job={args.job_id}  "
          f"interval={args.interval:g}s cadence={args.cadence} "
          f"card={args.baseline_card or 'NONE (ORDER-INVERSION disabled)'}", flush=True)
    if card is None:
        print("[monitor] NOTE: I18d expects a baseline card on every training "
              "template; running without one is a detection gap.", flush=True)

    while True:
        orders, rows = [], []
        if log.exists():
            try:
                orders, rows = read_log(log)
            except Exception as e:
                print(f"[monitor] log parse error (transient?): {e}", flush=True)
        if args.log and Path(args.log).exists():
            try:
                text = Path(args.log).read_text(errors='ignore')
                if 'Traceback (most recent call last)' in text and 'FAIL' not in fired:
                    fired.add('FAIL')
                    tail = '\n'.join(text.splitlines()[-15:])
                    email('FAIL', f"python traceback in trainer log:\n{tail}", orders, rows, [])
                    raise SystemExit(3)
            except OSError:
                pass

        if rows:
            cur_ep = int(rows[-1]['epoch'])
            # I18b cadence: first val epoch, then every --cadence epochs.
            if last_mailed_ep is None or cur_ep >= last_mailed_ep + args.cadence:
                tag = 'FIRST-VAL' if last_mailed_ep is None else 'OK'
                email(tag, f"cadence report at epoch {cur_ep} "
                           f"(best so far: {min(r.get('val_relL2', float('inf')) for r in rows):.4g}).",
                      orders, rows, [])
                last_mailed_ep = cur_ep
            for chk in CHECKS:
                if chk.__name__ in fired:
                    continue
                hit = chk(orders, rows, card)
                if hit:
                    kind, msg, idxs = hit
                    fired.add(chk.__name__)
                    email(kind, msg, orders, rows, idxs)
                    if kind in FATAL or args.exit_on_trigger:
                        raise SystemExit(2)

        done = args.job_id is not None and not job_alive(args.job_id)
        expired = (time.time() - t0) > args.max_hours * 3600
        if args.once or done or expired:
            if done and not args.once:
                if rows:
                    email('DONE', f"trainer left the queue after epoch "
                                  f"{int(rows[-1]['epoch'])}.", orders, rows, [])
                else:
                    email('FAIL', 'trainer left the queue before its first val epoch '
                                  '(no log.csv rows).', [], [], [])
                    raise SystemExit(3)
            break
        time.sleep(args.interval)

    print(f"[monitor] exiting; fired: {sorted(fired) if fired else 'none'}", flush=True)


if __name__ == '__main__':
    main()
