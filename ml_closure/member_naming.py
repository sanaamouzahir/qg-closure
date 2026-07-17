"""member_naming.py -- plain-English naming for the results-tree STANDARD
(Sanaa direct order 2026-07-17): every artifact lives under

    ml_closure/results/<geometry>/<model>/<suite>/<member_modulation>/

  geometry ∈ {flow_past_cylinder, flow_past_cape}
  model    = the run codename (piff_fpc_gjs_ylp75, piff_fpc_gjs_lap, ...)
  suite    ∈ {evaluation, mean_prediction, error_tails, sigma_at_events,
              reflection_probe, nearwall_variants, feature_screen, ...}
  member_modulation ∈ {constant_inflow, sine_modulation, ramp_modulation,
              ornstein_uhlenbeck_modulation, telegraph_modulation,
              telegraph_modulation_smoothed_A}

Member codenames (FPC-const, FPCape-sine, FPC-telS-A, ...) map to the
modulation names; telS-A maps to telegraph_modulation_smoothed_A ONLY when a
plain "tel" sibling coexists in the same pool (Sanaa's disambiguation rule);
alone it is just telegraph_modulation.

Also provides member_stamp() -- the mandatory figure-title fragment stating
the modulation function and the Reynolds number (STANDARD rule 2: per-frame
Re(t) where frame-specific, else the member's Re range).

Stdlib-only on purpose: imported by the diagnostics AND by the cluster
migration job (no torch, no repo deps).
"""
from __future__ import annotations

import re

MODULATION = {
    'const': 'constant_inflow',
    'sine': 'sine_modulation',
    'ramp': 'ramp_modulation',
    'ou': 'ornstein_uhlenbeck_modulation',
    'tel': 'telegraph_modulation',
    'telS-A': 'telegraph_modulation_smoothed_A',
}

_MEMBER_RE = re.compile(r'^FPC(ape)?-')


def is_member_name(name):
    """True for ensemble-member codenames (FPC-*, FPCape-*)."""
    return bool(_MEMBER_RE.match(str(name)))


def _suffix(member):
    return str(member).split('-', 1)[1] if '-' in str(member) else str(member)


def modulation_name(member, siblings=()):
    """FPC-const -> constant_inflow, FPCape-ou -> ornstein_uhlenbeck_modulation.
    telS-A -> telegraph_modulation_smoothed_A only if a plain tel sibling
    exists in `siblings` (member codenames); else telegraph_modulation.
    Non-member names pass through unchanged."""
    if not is_member_name(member):
        return str(member)
    sfx = _suffix(member)
    if sfx == 'telS-A':
        has_tel = any(is_member_name(s) and _suffix(s) == 'tel'
                      for s in siblings)
        return MODULATION['telS-A'] if has_tel else MODULATION['tel']
    if sfx in MODULATION:
        return MODULATION[sfx]
    # unknown modulation codename: keep it recognizable, still plain-ish
    return sfx.replace('-', '_') + '_modulation'


def geometry_name(name):
    """Member OR model-run codename -> geometry directory.
    Everything cape-flavored (FPCape-*, piff_cape_*, cape_lomo_*) is
    flow_past_cape; every other Pi_FF run in this branch is the cylinder."""
    return 'flow_past_cape' if 'cape' in str(name).lower() else 'flow_past_cylinder'


def member_dirname(member, plain, siblings=()):
    """Per-member output-subdirectory name: the modulation name when the
    STANDARD tree is requested (--plain-member-names), else the codename
    (backward-compatible default)."""
    return modulation_name(member, siblings) if plain else str(member)


def member_stamp(member, re_lo=None, re_hi=None, siblings=()):
    """Figure-title fragment (STANDARD rule 2): '<codename> [<modulation>]'
    plus the Reynolds number -- a single value when constant (or when only
    one is given), else the member's Re range."""
    mod = modulation_name(member, siblings).replace('_', ' ')
    s = f"{member} [{mod}]"
    if re_lo is not None:
        if re_hi is None or abs(float(re_hi) - float(re_lo)) < 1.0:
            s += f", Re={float(re_lo):.0f}"
        else:
            s += f", Re {float(re_lo):.0f}-{float(re_hi):.0f}"
    return s
