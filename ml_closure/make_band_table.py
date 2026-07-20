#!/usr/bin/env python
"""make_band_table.py -- the three-band verdict + metrics table for the
two-band SGS closure (Sanaa GO 2026-07-20; report contract of the same day;
diagnostics-table convention 2026-07-19).

Reads the CSVs written by `compare_band_metrics.py --out-csv` (one per model)
and emits, in this order:

  1. a PLAIN-ENGLISH verdict -- in words, before any numbers: did the two-band
     model beat BOTH specialists at once?
  2. the metrics TABLE itself: rows = band (NEAR / FAR / ALL), columns =
     ylp75, lap, wallv2, BLENDED, for r2, err % of RMS(truth), and the
     worst-0.1% share of squared error. A model whose CSV is absent keeps its
     COLUMN and reads n/a -- never silently dropped.

Elements 3 (artifact directory list) and 4 (results-standard confirmation) of
the report contract are added by the caller (scripts/sge/bandmodels_gate_job.sh),
which knows the cluster paths.

PRE-REGISTERED BAR (never tuned post hoc): NEAR r2 >= 0.954 (wallv2's measured
near) AND FAR r2 >= 0.934 (lap's measured far), simultaneously.

Aggregation: pixel-count-weighted over frames/members within each band. r2 and
err% are recomputed from the aggregate error/truth energies (sum n*rms^2) --
never averaged as a ratio of ratios. The worst-0.1% share is an n-weighted mean
of the per-frame shares (a share of a share does not pool exactly; it is
reported as context, not as a bar). Bands are compare_band_metrics' convention:
NEAR sdf <= 1 D, FAR sdf > 1 D, ALL every valid pixel.

Exit code: 0 = bar met, 4 = bar not met (loud, like the event gate).

Usage:
  python make_band_table.py --geometry fpc --primary blended \\
      --csv ylp75=<a.csv> --csv lap=<b.csv> --csv wallv2=<c.csv> \\
      --csv blended=<d.csv> --out band_table_fpc.txt
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

BANDS = ('NEAR', 'FAR', 'ALL')
BAR = {'NEAR': 0.954, 'FAR': 0.934}
BAR_SRC = {'NEAR': "wallv2's measured near-wall r2",
           'FAR': "lap's measured far-field r2"}
BAND_WORDS = {'NEAR': 'near-wall band (sdf <= 1 D)',
              'FAR': 'far field (sdf > 1 D)',
              'ALL': 'whole field'}


def load(path):
    """CSV -> {band: pooled sums + the list of per-frame CENTERED r2}.

    ESTIMATOR CONTRACT (G4 MAJOR 1, 2026-07-20). The bars NEAR >= 0.954 /
    FAR >= 0.934 were registered from compare_band_metrics' CENTERED per-frame
    r2 = 1 - var(err)/var(truth), averaged over frames. This module therefore
    AGGREGATES THE CSV's EXISTING `r2` COLUMN by mean over frames. It must
    NEVER recompute r2 from the pooled energies (1 - sum(err^2)/sum(truth^2)):
    that is the UNCENTERED estimator and a different number, which would make
    the table and the bar incomparable. st/se are kept for err% only."""
    agg = defaultdict(lambda: {'n': 0, 'st': 0.0, 'se': 0.0, 'tail': 0.0,
                               'r2': []})
    with open(path, newline='') as fh:
        for r in csv.DictReader(fh):
            n = int(r['n'])
            rms_t = float(r['rms_t'])
            rms_e = float(r['rel']) * rms_t
            a = agg[r['band']]
            a['n'] += n
            a['st'] += n * rms_t * rms_t
            a['se'] += n * rms_e * rms_e
            a['tail'] += n * float(r.get('tail') or 0.0)
            a['r2'].append(float(r['r2']))       # centered, per frame
    return dict(agg)


def stats(agg, band):
    """r2   = MEAN over frames of the per-frame CENTERED r2 -- the registered-
              bar estimator (see load()).
    err%   = POOLED RMSE / RMS(truth), pixel-count-weighted (an energy ratio;
              pooling is its right aggregation, and it carries no bar).
    tail   = n-weighted mean of the per-frame worst-0.1% shares (context).
    err% is deliberately NOT sqrt(1 - r2) here: the two quantities use
    different aggregations on purpose, and the table header says so."""
    if not agg or band not in agg or agg[band]['n'] == 0 or not agg[band]['r2']:
        return None
    a = agg[band]
    mst, mse = a['st'] / a['n'], a['se'] / a['n']
    frac = mse / max(mst, 1e-30)
    return {'n': a['n'], 'n_frames': len(a['r2']),
            'r2': sum(a['r2']) / len(a['r2']),
            'err': 100.0 * frac ** 0.5,
            'tail': a['tail'] / a['n']}


def fmt(s, key, spec):
    return f"{s[key]:{spec}}" if s else 'n/a'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', action='append', default=[], metavar='NAME=PATH',
                    help='repeatable; NAME is the table column label')
    ap.add_argument('--geometry', required=True)
    ap.add_argument('--out', default=None, help='also write the text here')
    ap.add_argument('--primary', default='blended',
                    help='column the pre-registered bar is applied to')
    args = ap.parse_args()

    order, data, notes = [], {}, {}
    for spec in args.csv:
        if '=' not in spec:
            raise SystemExit(f"--csv expects NAME=PATH, got {spec!r}")
        name, path = spec.split('=', 1)
        p = Path(path)
        order.append(name)
        data[name] = load(p) if p.exists() else None
        notes[name] = 'ok' if p.exists() else f'CSV absent ({p})'
    prim = data.get(args.primary)

    L = []
    L.append("=" * 78)
    L.append(f"TWO-BAND SGS CLOSURE -- verdict and metrics [{args.geometry}]")
    L.append("=" * 78)
    L.append("")

    # ---- 1. PLAIN-ENGLISH VERDICT, before any numbers ------------------- #
    L.append("VERDICT (plain English)")
    L.append("-" * 78)
    sn = stats(prim, 'NEAR') if prim else None
    sf = stats(prim, 'FAR') if prim else None
    if sn is None or sf is None:
        L.append("  The two-band model could not be judged: its band metrics are")
        L.append("  missing, so we do not know whether it beat the specialists.")
        L.append("  Treat this run as INCOMPLETE, not as a failure of the idea.")
        ok = False
    else:
        near_ok, far_ok = sn['r2'] >= BAR['NEAR'], sf['r2'] >= BAR['FAR']
        ok = near_ok and far_ok
        if ok:
            L.append("  YES -- the two-band model beat BOTH specialists at the same time.")
            L.append("  It matches the near-wall accuracy of the wall-gated model AND the")
            L.append("  far-field accuracy of the lap model, which no single model has")
            L.append("  managed before. Splitting the domain into two specialists, each")
            L.append("  with its own target normalization, bought the near-wall band")
            L.append("  without giving up the far field. This is the result we wanted.")
        elif near_ok or far_ok:
            good = 'near-wall band' if near_ok else 'far field'
            bad = 'far field' if near_ok else 'near-wall band'
            L.append(f"  PARTLY -- the two-band model met the bar in the {good} but not")
            L.append(f"  in the {bad}. It did NOT beat both specialists at once, so the")
            L.append("  trade we set out to remove is still there, just moved. The split")
            L.append("  helped where it was aimed and cost something on the other side;")
            L.append("  the next question is whether the losing band's specialist was")
            L.append("  under-trained or whether the blend hand-over is placed wrong.")
        else:
            L.append("  NO -- the two-band model met neither bar. It did not beat either")
            L.append("  specialist in that specialist's own region, so on this evidence")
            L.append("  splitting the domain did not buy what we hoped. Check the two")
            L.append("  specialists' own training curves before drawing conclusions about")
            L.append("  the idea itself: a blend can only be as good as its parts.")
        L.append("")
        L.append(f"  In numbers: near-wall r2 {sn['r2']:.4f} against a bar of "
                 f"{BAR['NEAR']:.3f} ({'met' if near_ok else 'NOT met'}), "
                 f"far-field r2")
        L.append(f"  {sf['r2']:.4f} against a bar of {BAR['FAR']:.3f} "
                 f"({'met' if far_ok else 'NOT met'}). Typical error is "
                 f"{sn['err']:.0f}% of the near-wall")
        L.append(f"  signal and {sf['err']:.0f}% of the far-field signal.")
    L.append("")

    # ---- 2. THE TABLE (inline numbers, never a path) -------------------- #
    L.append("METRICS TABLE")
    L.append("-" * 78)
    L.append("  rows = band, columns = model.  NEAR sdf <= 1 D | FAR sdf > 1 D |")
    L.append("  ALL every valid pixel.  Same truth, same frames, same scale for all.")
    L.append("")
    L.append("  AGGREGATION (they differ on purpose -- G4 MAJOR 1, 2026-07-20):")
    L.append("    r2    = MEAN over frames of the per-frame CENTERED r2")
    L.append("            (1 - var(err)/var(truth)) -- the SAME estimator the")
    L.append("            pre-registered bar was set from, so table and bar agree.")
    L.append("    err%  = POOLED RMSE / RMS(truth), pixel-count-weighted.")
    L.append("    tail  = n-weighted mean of the per-frame worst-0.1% shares.")
    L.append("    Hence err% is NOT sqrt(1 - r2): different aggregations.")
    L.append("")
    w = 12
    head = f"  {'band':<8}" + ''.join(f"{n:>{w}}" for n in order)
    for title, key, spec in (
            ('r2  (1 = perfect)', 'r2', '.4f'),
            ('err % of RMS(truth)  (lower is better)', 'err', '.1f'),
            ('worst-0.1% share of squared error', 'tail', '.3f')):
        L.append(f"  {title}")
        L.append(head)
        L.append('  ' + '-' * (8 + w * len(order)))
        for b in BANDS:
            row = f"  {b:<8}"
            for n in order:
                row += f"{fmt(stats(data[n], b), key, spec):>{w}}"
            L.append(row)
        L.append("")
    for n in order:
        if notes[n] != 'ok':
            L.append(f"  note: column '{n}' is n/a -- {notes[n]}")
    L.append("")

    # ---- the bar, spelled out ------------------------------------------- #
    L.append("PRE-REGISTERED BAR (set before the run, never tuned post hoc)")
    L.append("-" * 78)
    L.append(f"  beat BOTH specialists simultaneously:")
    for b, thr in BAR.items():
        L.append(f"    {BAND_WORDS[b]:<28} r2 >= {thr:.3f}   ({BAR_SRC[b]})")
    rc = 0
    if prim is None or sn is None or sf is None:
        L.append(f"  RESULT: n/a -- no usable '{args.primary}' column")
        rc = 4
    else:
        for b, thr in BAR.items():
            s = stats(prim, b)
            L.append(f"    {b:<5} r2 {s['r2']:.4f} vs {thr:.3f} -> "
                     f"{'MET' if s['r2'] >= thr else 'NOT MET'} "
                     f"(margin {s['r2'] - thr:+.4f}, err {s['err']:.1f}%)")
        L.append(f"  RESULT: {'BAR MET' if ok else 'BAR NOT MET'}")
        rc = 0 if ok else 4

    text = '\n'.join(L) + '\n'
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)
        print(f"[table] wrote {args.out}")
    sys.exit(rc)


if __name__ == '__main__':
    main()
