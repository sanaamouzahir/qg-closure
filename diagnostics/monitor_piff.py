"""
I18 monitor for Pi_FF SVGP trainings (template T4). Polls the trainer's stdout
log (the '[ep NNN] ...' grammar of train_piff.py), checks it against a baseline
card (diagnostics/baseline_cards/SGS_piff_ens.json), and reports by spooling
[QG][MONITOR][sgs] mails to reporting/pending_mail/ (the mseas cron relays —
works from ANY node; the 2026-07-09 ibfdr mailx lesson).

LIVE mode (default): report at first parsed epoch, every 5 epochs, and
immediately on any trigger; exit when the trainer job id leaves qstat.
FINALIZE mode (QG_MONITOR_FINALIZE=1): one postmortem pass over the full log —
silent if a final LIVE verdict marker exists, else emits the final verdict.

Usage: python monitor_piff.py <trainer_log> <run_name> <trainer_job_id> <card_json>
"""

import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REP = Path('/gdata/projects/ml_scope/Closure_modeling/QG-closure/reporting/pending_mail')
EP_RE = re.compile(
    r'\[ep (\d+)\] .*val NLL ([\d.eE+-]+|nan|inf) RMSE ([\d.eE+-]+|nan|inf) '
    r'R2 ([\d.eE+-]+|nan|-?inf) .*zeta_ls ([\d.eE+-]+)')


def spool(subject, body):
    REP.mkdir(parents=True, exist_ok=True)
    p = REP / f"monitor_{int(time.time())}_{os.getpid()}.mail"
    p.write_text(f"To: sanaamz@mit.edu\nSubject: {subject}\n\n{body}\n")
    print(f"[monitor] spooled: {subject}")


def parse(log_path):
    rows = []
    try:
        text = Path(log_path).read_text(errors='replace')
    except OSError:
        return rows, False
    for m in EP_RE.finditer(text):
        ep = int(m.group(1))
        vals = [float(m.group(i)) for i in (2, 3, 4)]
        rows.append({'ep': ep, 'nll': vals[0], 'rmse': vals[1], 'r2': vals[2],
                     'zeta_ls': float(m.group(5))})
    return rows, 'PLAN-B SYMPTOM' in text


def verdicts(rows, planb, card, cape):
    out = []
    if not rows:
        return out
    shift = card['cape_deltas']['max_nll_shift'] if cape else 0.0
    if any(not (math.isfinite(r['nll']) and math.isfinite(r['r2'])) for r in rows):
        out.append('EXPLODE: non-finite val metric')
    tail = [r for r in rows if r['ep'] >= 5][-10:]
    if len(tail) == 10 and all(r['r2'] < 0.02 for r in tail):
        out.append('COLLAPSE: val R2 < 0.02 for 10 consecutive epochs (arm C/D signature)')
    if planb:
        out.append('PLAN-B SYMPTOM logged by trainer (feature collapse >10x)')
    late = [r for r in rows if r['ep'] >= 30]
    if late and all(abs(r['zeta_ls'] - 0.6931) < 1e-4 for r in late):
        out.append('CONDITIONING-INERT (P1 falsifier): zeta_ls frozen at init through all epochs >= 30')
    run_min, run = math.inf, 0
    for r in rows:
        run_min = min(run_min, r['nll'])
        run = run + 1 if r['nll'] > run_min + 0.5 else 0
    if run >= 10:
        out.append(f'NLL RUNAWAY: val NLL > running_min+0.5 for {run} consecutive epochs')
    for ms in card['milestones']:
        past = [r for r in rows if r['ep'] >= ms['epoch']]
        if past:
            r0 = past[0]
            if r0['r2'] < ms['min_r2'] or r0['nll'] > ms['max_nll'] + shift:
                out.append(f"MILESTONE MISS ep{ms['epoch']}: R2 {r0['r2']:.3f} (need "
                           f">= {ms['min_r2']}) NLL {r0['nll']:.3f} (need <= "
                           f"{ms['max_nll'] + shift:.2f}){' [' + ms['gate'] + ']' if 'gate' in ms else ''}")
    return out


def fmt(rows, verds, run_name, jid):
    last = rows[-1] if rows else None
    lines = [f"RUN {run_name} (job {jid}) — {len(rows)} epochs parsed", ""]
    if last:
        lines.append(f"    latest: ep {last['ep']}  NLL {last['nll']:.4f}  "
                     f"R2 {last['r2']:.4f}  zeta_ls {last['zeta_ls']:.3f}")
        lines.append("")
    lines.append("**TRIGGERS**" if verds else "**NO TRIGGERS — curve healthy vs card**")
    for i, v in enumerate(verds, 1):
        lines.append(f"\n    {i}. {v}")
    return "\n".join(lines)


def main():
    log_path, run_name, jid, card_path = sys.argv[1:5]
    card = json.loads(Path(card_path).read_text())
    cape = 'cape' in run_name.lower()
    finalize = os.environ.get('QG_MONITOR_FINALIZE') == '1'
    marker = Path(log_path).with_suffix('.monitor_final')

    if finalize:
        if marker.exists():
            print('[monitor] LIVE already delivered final verdict; silent exit')
            return
        rows, planb = parse(log_path)
        verds = verdicts(rows, planb, card, cape)
        spool(f"[QG][MONITOR][sgs] FINAL {run_name}: "
              f"{'CLEAN' if not verds else str(len(verds)) + ' trigger(s)'}",
              fmt(rows, verds, run_name, jid) + "\n\n(finalize postmortem — LIVE monitor did not conclude)")
        return

    reported_ep, reported_verds = -1, set()
    while True:
        rows, planb = parse(log_path)
        verds = verdicts(rows, planb, card, cape)
        new_verds = [v for v in verds if v.split(':')[0] not in reported_verds]
        last_ep = rows[-1]['ep'] if rows else -1
        if new_verds or (last_ep >= 0 and reported_ep < 0) or \
           (last_ep >= reported_ep + 5 and last_ep >= 0):
            tag = 'TRIGGER' if new_verds else 'status'
            spool(f"[QG][MONITOR][sgs] {tag} {run_name} ep{last_ep}", fmt(rows, verds, run_name, jid))
            reported_ep = last_ep
            reported_verds |= {v.split(':')[0] for v in new_verds}
        q = subprocess.run(['qstat'], capture_output=True, text=True).stdout
        if not re.search(rf'^\s*{jid}\s', q, re.M):
            rows, planb = parse(log_path)
            verds = verdicts(rows, planb, card, cape)
            spool(f"[QG][MONITOR][sgs] FINAL {run_name}: "
                  f"{'CLEAN' if not verds else str(len(verds)) + ' trigger(s)'}",
                  fmt(rows, verds, run_name, jid))
            marker.write_text('done\n')
            return
        time.sleep(120)


if __name__ == '__main__':
    main()
