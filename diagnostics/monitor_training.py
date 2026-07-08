#!/usr/bin/env python
r"""monitor_training.py -- autonomous watchdog for a deriv-closure SGE run.

Watches the SGE stdout log of a train_deriv.py run, parses per-epoch metrics, and
applies decision-tree D. It EXITS (non-zero) the moment a trigger fires or the job
ends, printing a structured verdict on the last line -- so a supervising agent that
launched it in the background is re-invoked to act. Also exits 0 cleanly on
successful completion.

Decision tree D (thresholds are conservative; tune per run):
  EXPLODE   : any train/val is NaN/Inf, OR val > EXPLODE_ABS, OR val > EXPLODE_REL x ep0-val.
  LR-SANITY : train loss at ep >= LR_CHECK_EP exceeds train at ep0 (lr too hot / diverging).
  OSCILLATE : val rises for >= OSC_RISES consecutive epochs after warmup, OR
              rolling std/mean of val over OSC_WIN epochs > OSC_CV.
  IMBALANCE : any per-order rel-L2 > IMB_REL x its own ep0 value (one order blowing up),
              OR Nddot (the rollout-ceiling metric) rises while train falls (overfit) for
              >= IMB_EP epochs.
  STALL     : best val not improved for >= STALL_PATIENCE epochs.
  DONE      : log shows 'done in' (success) -> exit 0.
  FAIL      : job left the queue without 'done in', or log has a Python traceback.

Usage (typically launched in the background by the branch supervisor):
    python diagnostics/monitor_training.py --job-id 1825720 \
        --log /gdata/.../qg/logs/deriv7_cond.1825720.log \
        --run-dir /gdata/.../training_runs/deriv7_cond \
        --poll 60 --max-hours 12
"""
from __future__ import annotations
import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

# ---- decision-tree D thresholds ----
EXPLODE_ABS = 10.0        # physics-init starts ~0.16; >10 mean-relL2 == blow-up
EXPLODE_REL = 8.0         # or 8x the ep0 value
LR_CHECK_EP = 8           # by this epoch train should be <= ep0 train
OSC_RISES = 4             # consecutive val rises after warmup
OSC_WIN = 8
OSC_CV = 0.35             # rolling coeff-of-variation of val
WARMUP = 5
IMB_REL = 4.0             # a single order exceeding 4x its ep0
IMB_EP = 15               # Nddot-up / train-down sustained this many epochs
STALL_PATIENCE = 40       # epochs without best-val improvement

EP_RE = re.compile(
    r"ep\s+(\d+)\s+\*?\s+train=([\d.eEnaN+-]+)\s+val=([\d.eEnaN+-]+)\s+"
    r"best=([\d.eEnaN+-]+).*?Ndot=([\d.eEnaN+-]+)\s+Nddot=([\d.eEnaN+-]+)\s+"
    r"N3dot=([\d.eEnaN+-]+)")


def fnum(s):
    try:
        return float(s)
    except ValueError:
        return float('nan')


def job_alive(job_id: str) -> bool:
    try:
        out = subprocess.run(['qstat'], capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return True  # can't tell -> assume alive, rely on log
    return any(job_id in ln.split()[0:1] or job_id in ln for ln in out.splitlines())


def verdict(kind: str, msg: str) -> int:
    print(f"\n[MONITOR][{kind}] {msg}", flush=True)
    # exit codes: 0 ok/done, 2 trigger, 3 job failure
    return 0 if kind == 'DONE' else (3 if kind == 'FAIL' else 2)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--job-id', required=True)
    ap.add_argument('--log', type=Path, required=True)
    ap.add_argument('--run-dir', type=Path, default=None)
    ap.add_argument('--poll', type=int, default=60)
    ap.add_argument('--max-hours', type=float, default=12.0)
    ap.add_argument('--start-timeout-min', type=float, default=240.0,
                    help='give up waiting for the job to start after this (qw too long)')
    args = ap.parse_args()

    t0 = time.time()
    deadline = t0 + args.max_hours * 3600
    start_deadline = t0 + args.start_timeout_min * 60

    epochs = {}          # ep -> dict(train,val,best,Ndot,Nddot,N3dot)
    ep0 = None
    best_val = float('inf')
    best_ep = -1
    nddot_up = 0
    val_hist = []
    started = False
    print(f"[MONITOR] watching job {args.job_id} log={args.log} poll={args.poll}s "
          f"max={args.max_hours}h", flush=True)

    while True:
        now = time.time()
        alive = job_alive(args.job_id)
        log_exists = args.log.exists()

        if not started:
            if log_exists:
                started = True
                print("[MONITOR] log appeared -- job running.", flush=True)
            elif not alive and now > t0 + 120:
                sys.exit(verdict('FAIL', f"job {args.job_id} left the queue before "
                                         "writing a log (submit/host error?)."))
            elif now > start_deadline:
                sys.exit(verdict('FAIL', f"job {args.job_id} still not started after "
                                         f"{args.start_timeout_min:.0f} min in qw."))
            time.sleep(args.poll)
            continue

        text = args.log.read_text(errors='ignore') if log_exists else ''
        if 'Traceback (most recent call last)' in text:
            tail = '\n'.join(text.splitlines()[-15:])
            sys.exit(verdict('FAIL', f"python traceback in log:\n{tail}"))

        for m in EP_RE.finditer(text):
            ep = int(m.group(1))
            epochs[ep] = dict(train=fnum(m.group(2)), val=fnum(m.group(3)),
                              best=fnum(m.group(4)), Ndot=fnum(m.group(5)),
                              Nddot=fnum(m.group(6)), N3dot=fnum(m.group(7)))
        if epochs:
            ep0 = epochs[min(epochs)]
            cur_ep = max(epochs)
            cur = epochs[cur_ep]
            val_hist = [epochs[e]['val'] for e in sorted(epochs)]
            if cur['val'] < best_val:
                best_val, best_ep = cur['val'], cur_ep

            # EXPLODE -- action path per CHARTER v1.1 I14: the branch
            # supervisor qdels the run, diagnoses, and RESUBMITS under its
            # own authority (qdel+resubmit rights). Do NOT park it as
            # [QG][BLOCKED]; BLOCKED is only for what the branch cannot fix.
            for key in ('train', 'val'):
                if cur[key] != cur[key] or cur[key] in (float('inf'), float('-inf')):
                    sys.exit(verdict('EXPLODE', f"ep{cur_ep} {key}={cur[key]} "
                                                "(NaN/Inf). ACTION per I14: "
                                                "qdel + diagnose + resubmit "
                                                "(branch authority; not BLOCKED)."))
            if cur['val'] > EXPLODE_ABS or (ep0 and cur['val'] > EXPLODE_REL * ep0['val']):
                sys.exit(verdict('EXPLODE', f"ep{cur_ep} val={cur['val']:.3e} "
                                            f"(ep0 {ep0['val']:.3e}); blow-up. "
                                            "ACTION per I14: qdel + diagnose + "
                                            "resubmit (branch authority; not "
                                            "BLOCKED)."))
            # LR-SANITY
            if cur_ep >= LR_CHECK_EP and ep0 and cur['train'] > ep0['train']:
                sys.exit(verdict('LR-SANITY', f"ep{cur_ep} train={cur['train']:.3e} > "
                                              f"ep0 train={ep0['train']:.3e}; lr too hot."))
            # OSCILLATE
            if cur_ep > WARMUP and len(val_hist) >= OSC_RISES + 1:
                tailv = val_hist[-(OSC_RISES + 1):]
                if all(tailv[i + 1] > tailv[i] for i in range(OSC_RISES)):
                    sys.exit(verdict('OSCILLATE', f"val rose {OSC_RISES} epochs straight "
                                                  f"to {cur['val']:.3e} at ep{cur_ep}."))
                if len(val_hist) >= OSC_WIN:
                    w = val_hist[-OSC_WIN:]
                    mean = sum(w) / len(w)
                    var = sum((x - mean) ** 2 for x in w) / len(w)
                    if mean > 0 and (var ** 0.5) / mean > OSC_CV:
                        sys.exit(verdict('OSCILLATE', f"val CV={(var**0.5)/mean:.2f} over "
                                                      f"{OSC_WIN} ep (>{OSC_CV})."))
            # IMBALANCE
            if ep0:
                for o in ('Ndot', 'Nddot', 'N3dot'):
                    if ep0[o] > 0 and cur[o] > IMB_REL * ep0[o]:
                        sys.exit(verdict('IMBALANCE', f"ep{cur_ep} {o}={cur[o]:.3e} > "
                                                      f"{IMB_REL}x ep0 {ep0[o]:.3e}."))
                if cur_ep >= 1:
                    prev = epochs.get(cur_ep - 1)
                    if prev and cur['Nddot'] > prev['Nddot'] and cur['train'] < prev['train']:
                        nddot_up += 1
                    else:
                        nddot_up = 0
                    if nddot_up >= IMB_EP:
                        sys.exit(verdict('IMBALANCE', f"Nddot rising while train falling "
                                                      f"{nddot_up} ep (overfit ceiling)."))
            # STALL
            if cur_ep - best_ep >= STALL_PATIENCE:
                sys.exit(verdict('STALL', f"best val {best_val:.3e} (ep{best_ep}) not "
                                          f"improved for {cur_ep-best_ep} ep."))
            # DONE
            if 'done in' in text:
                sys.exit(verdict('DONE', f"training finished. best val={best_val:.3e} "
                                         f"@ep{best_ep}; last Nddot="
                                         f"{cur['Nddot']:.3e} N3dot={cur['N3dot']:.3e}."))
            print(f"[MONITOR] ep{cur_ep} val={cur['val']:.4e} best={best_val:.4e}"
                  f"@{best_ep} [Nd={cur['Ndot']:.3e} Ndd={cur['Nddot']:.3e} "
                  f"N3={cur['N3dot']:.3e}]", flush=True)

        # job ended without 'done in' -> failure
        if not alive and 'done in' not in text:
            tail = '\n'.join(text.splitlines()[-15:])
            sys.exit(verdict('FAIL', f"job {args.job_id} left the queue without "
                                     f"completing. log tail:\n{tail}"))
        if now > deadline:
            sys.exit(verdict('STALL', f"monitor hit --max-hours {args.max_hours}; "
                                      f"last best {best_val:.3e}@ep{best_ep}."))
        time.sleep(args.poll)


if __name__ == '__main__':
    main()
