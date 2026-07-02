"""
split_ensemble.py

Resolve the train/test split of the temporal-closure ensemble by reading each
member's manifest.json -- NOT by folder name -- so the split is reproducible and
immune to naming drift.

Generalization design (temporal closure; N-derivative map is regime-universal):
  TEST (held-out regimes), each isolating one axis:
    beta == 0.75                       -> beta INTERPOLATION        (FRC-b075)
    beta >= 2.0                        -> beta/geostrophic EXTRAP   (FRC-b2, FRC-b25)
    nu   <= 2.5e-6                     -> Re EXTRAPOLATION          (DEC-hiRe)
    nu ~= 4e-5 and beta == 0.5         -> COMPOSITIONAL OOD         (FRC-combo)
  TRAIN: everything else.

A member's root (what ConcatClosureDataset wants) is the directory holding
`packed/` + `manifest.json` (here `<member>/forced_turbulence_dT_1em3/`).

Usage
-----
    from split_ensemble import resolve_split
    train_roots, test_roots, table = resolve_split(ENSEMBLE_ROOT)
    # train_roots/test_roots -> feed make_concat_loaders

    python split_ensemble.py /gdata/.../training/data/ensemble_N5
"""
from __future__ import annotations

import json
import math
from pathlib import Path


# ---- tolerances -------------------------------------------------------------- #
_NU_RE_EXTRAP = 2.5e-6          # <= this -> Re extrapolation holdout
_NU_COMBO     = 4.0e-5          # combo viscosity
_ATOL_B       = 1e-6
_RTOL_NU      = 0.10


def _close(a, b, rtol=_RTOL_NU):
    if a is None or b is None or math.isnan(a) or math.isnan(b):
        return False
    return abs(a - b) <= rtol * abs(b)


def _member_name(man: dict, root: Path) -> str:
    """Human label: parent folder of DNS_FR.npz in source_omega_path, else dir."""
    p = man.get('source_omega_path', '')
    if p:
        parts = Path(p).parts
        if len(parts) >= 2:
            return parts[-2]                     # .../<MEMBER>/DNS_FR.npz
    # fall back to the member dir two levels up (…/<MEMBER>/forced_turbulence_dT_1em3)
    return root.parent.name or root.name


def classify(beta: float, nu: float) -> tuple[str, str]:
    """Return (split, reason)."""
    if not math.isnan(beta) and abs(beta - 0.75) <= _ATOL_B:
        return 'test', 'beta-interp(0.75)'
    if not math.isnan(beta) and beta >= 2.0 - _ATOL_B:
        return 'test', f'beta-extrap(geostrophic, beta={beta:g})'
    if not math.isnan(nu) and nu <= _NU_RE_EXTRAP * 1.01:
        return 'test', f'Re-extrap(nu={nu:g})'
    if _close(nu, _NU_COMBO) and abs(beta - 0.5) <= _ATOL_B:
        return 'test', 'compositional(combo)'
    return 'train', 'in-distribution'


def _find_member_roots(ensemble_root: Path) -> list[Path]:
    """Every dir under ensemble_root that has both manifest.json and packed/."""
    roots = []
    for man in ensemble_root.rglob('manifest.json'):
        root = man.parent
        if (root / 'packed').is_dir():
            roots.append(root)
    return sorted(set(roots))


def resolve_split(ensemble_root, verbose: bool = True):
    """-> (train_roots, test_roots, table) with roots as str paths."""
    ensemble_root = Path(ensemble_root)
    member_roots = _find_member_roots(ensemble_root)
    if not member_roots:
        raise FileNotFoundError(
            f"no '<dir>/manifest.json' with a sibling 'packed/' under {ensemble_root}")

    table = []      # (name, split, reason, beta, nu, Nx, Delta_T, root)
    for root in member_roots:
        with open(root / 'manifest.json') as f:
            man = json.load(f)
        beta = float(man.get('beta', float('nan')))
        nu = float(man.get('nu', float('nan')))
        Nx = int(man.get('Nx', -1))
        dT = float(man.get('Delta_T', float('nan')))
        name = _member_name(man, root)
        split, reason = classify(beta, nu)
        table.append(dict(name=name, split=split, reason=reason, beta=beta,
                          nu=nu, Nx=Nx, Delta_T=dT, root=str(root)))

    table.sort(key=lambda r: (r['split'] != 'train', r['name']))
    train_roots = [r['root'] for r in table if r['split'] == 'train']
    test_roots  = [r['root'] for r in table if r['split'] == 'test']

    if verbose:
        print(f"ensemble_root: {ensemble_root}")
        print(f"found {len(table)} members  ->  {len(train_roots)} train / {len(test_roots)} test\n")
        hdr = f"{'member':12}{'split':6}{'beta':>6}{'nu':>11}{'Nx':>6}{'dT':>8}   reason"
        print(hdr); print('-' * len(hdr))
        for r in table:
            print(f"{r['name']:12}{r['split']:6}{r['beta']:>6.3g}{r['nu']:>11.3g}"
                  f"{r['Nx']:>6}{r['Delta_T']:>8.4g}   {r['reason']}")
        # quick coverage sanity for the train pool
        tb = sorted({round(r['beta'], 4) for r in table if r['split'] == 'train'})
        tn = sorted({float(f"{r['nu']:.3g}") for r in table if r['split'] == 'train'})
        tg = sorted({r['Nx'] for r in table if r['split'] == 'train'})
        print(f"\ntrain coverage  beta={tb}  nu={tn}  grids={tg}")
        # warn on anything suspicious
        for r in table:
            if math.isnan(r['beta']) or math.isnan(r['nu']):
                print(f"  [WARN] {r['name']}: missing beta/nu in manifest -> defaulted to train")
    return train_roots, test_roots, table


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print("usage: python split_ensemble.py <ENSEMBLE_ROOT>")
        raise SystemExit(2)
    resolve_split(sys.argv[1])
