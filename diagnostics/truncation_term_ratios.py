"""
truncation_term_ratios.py
=========================
Sanaa 2026-07-13 item 2: TRUE R3/R4/R5 terms and the ratio test.

Wraps analysis/measure_truncation_magnitudes.py (imports its operators and
R_p assembly unchanged) and, for each replication member
{FRC-kf4, FRC-256, FRC-combo}:

  1. loads 4 DEVELOPED-FLOW omega snapshots (t ~ 30/50/70/90) from the raw
     ensemble DNS (outputs/Step_size_resolution_closure_ensemble/<M>/DNS_FR.npz,
     float32 on disk -- upcast to float64; ALL compute float64),
  2. rebuilds forcing EXACTLY from the member's hydra config
     (F = A cos(Bx) + D cos(Ey)),
  3. computes omega^(k), N^(m) via the recursive chain-rule bootstrap,
     validates each N^(m) against RK4 central finite differences,
  4. records PER-SNAPSHOT ||R_p|| (p=3..6) and per-monomial RMS,
  5. reports MEDIANS over snapshots (Sanaa's medians rule), the dT-weighted
     LTE terms  term_p = (dT^p / D_p) ||R_p||  at dT in {5e-3, 1e-2, 1.5e-2},
     raw ratios ||R_3||/||R_4||, ||R_4||/||R_5||, weighted ratios
     term3/term4, term4/term5, and the predicted improvement ceiling of a
     PERFECT R3-only closure:
        ceiling(dT) = (term3 + term4 + term5) / (term4 + term5)
     (assumption: one-step LTE-dominated error, no accumulation/feedback),
  6. compares against the MEASURED analytic-closure (r3anal) improvements
     read from diagnostics/Results/apost_ladder_20260709_p170/ and
     .../apost_ladder_20260709_third23/ summary CSVs (16-step rollouts,
     FRC-kf4 IC837, truth RK4 h_fine=1e-5 -- I4 rollout convention),
  7. writes ONE consolidated npz + yaml summary + 2 figures (Sanaa layout:
     diagnostics/pngs/<name>/ and diagnostics/yamls/<name>/ + .txt explainer).

Truth convention note (I4): the r3anal rollout applies coef = dT^3 (no
(1-1/K^2)); the LTE terms computed here use the same modified-equation
operators, so the predicted/measured comparison is convention-consistent.

Run (CPU is fine, all float64, ~minutes):
  qsub -q all.q -N dg_rterms scripts/sge/diag_job.sh truncation_term_ratios.py
"""
from __future__ import annotations
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

W = Path("/gdata/projects/ml_scope/Closure_modeling/QG-closure/qg-wiener-conditioning")
sys.path.insert(0, str(W / "analysis"))

from measure_truncation_magnitudes import (  # noqa: E402
    compute_derivatives, fd_validate, assemble_R, build_L_hat, build_F_phys,
    R_DEFS, D_P, _rms,
)

OUT_ROOT = W / "diagnostics"
NAME = "truncation_term_ratios_vs_measured_improvement"
PNG_DIR = OUT_ROOT / "pngs" / NAME
YAML_DIR = OUT_ROOT / "yamls" / NAME
NPZ_PATH = YAML_DIR / "truncation_term_ratios_consolidated.npz"

ENS = Path("/gdata/projects/ml_scope/Closure_modeling/QG-closure/"
           "qg-simple-package-stable/src/qg/outputs/"
           "Step_size_resolution_closure_ensemble")
MEMBERS = ["FRC-kf4", "FRC-256", "FRC-combo"]
SNAP_TIMES = [30.0, 50.0, 70.0, 90.0]   # developed window (training seeds start t=15)
DTS = [5.0e-3, 1.0e-2, 1.5e-2]
MAX_M = 5           # through R_6
FD_DT = 2.0e-3

P170_CSV = OUT_ROOT / "Results/apost_ladder_20260709_p170/ladder_matrix_summary.csv"
T23_CSV = OUT_ROOT / "Results/apost_ladder_20260709_third23/ladder_matrix_summary.csv"


def read_measured(csv_path, case_prefix="case_analytic_full_0p"):
    """Measured r3anal improvement per dT: final_relL2_bare / final_relL2_closure.
    Excludes the *_proj annulus-isolation variants."""
    out = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            c = row["case"]
            if not c.startswith(case_prefix) or c.endswith("_proj") or "_proj_" in c:
                continue
            dT = float(row["Delta_T"])
            bare = float(row["final_relL2_bare"])
            clos = float(row["final_relL2_closure"])
            out[dT] = bare / clos
    return out


def main():
    device = "cpu"
    dtype = torch.float64
    torch.set_num_threads(4)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    YAML_DIR.mkdir(parents=True, exist_ok=True)

    from qg.solver.grid.cartesian import CartesianGrid
    from qg.solver.opt.derivative import Derivative

    results = {}      # member -> dict
    npz_store = {}

    for member in MEMBERS:
        cfg_path = ENS / member / ".hydra/config.yaml"
        npz_path = ENS / member / "DNS_FR.npz"
        cfg = yaml.safe_load(open(cfg_path))
        qg = cfg.get("qg", cfg)
        Nx, Ny = int(qg["grid"]["Nx"]), int(qg["grid"]["Ny"])
        Lx, Ly = float(qg["grid"]["Lx"]), float(qg["grid"]["Ly"])
        nu, mu = float(qg["pde"]["nu"]), float(qg["pde"]["mu"])
        beta = float(qg["pde"]["B"])
        print(f"\n########## {member}: {Nx}x{Ny} L={Lx:.3f} nu={nu} mu={mu} "
              f"beta={beta} ##########", flush=True)

        grid = CartesianGrid(Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly, device=device,
                             precision="float64")
        derivative = Derivative(grid)
        for attr in ("dx", "dy", "laplacian", "inv_laplacian"):
            setattr(derivative, attr, getattr(derivative, attr).to(device))
        L_hat = build_L_hat(derivative, nu=nu, mu=mu, B=beta).to(device)
        F_phys = build_F_phys(cfg, "forced_turbulence", Lx, Ly, Nx, Ny,
                              device, dtype)

        z = np.load(npz_path, mmap_mode="r")
        om_all = z["omega_FR"]                     # (1, T, Ny, Nx) float32 disk
        T = om_all.shape[1]
        # frame index <-> time: 1001 frames over t in [0, 100]
        t_grid = np.linspace(0.0, 100.0, T)
        snaps = []
        for t_want in SNAP_TIMES:
            i = int(np.argmin(np.abs(t_grid - t_want)))
            snaps.append((float(t_grid[i]),
                          np.asarray(om_all[0, i]).astype(np.float64)))

        Rnorm = {p: [] for p in range(3, MAX_M + 2)}     # per-snapshot ||R_p||
        monos = {p: {} for p in range(3, MAX_M + 2)}
        fd_rel = {m: [] for m in range(1, MAX_M + 1)}
        for (t_snap, om_np) in snaps:
            omega = torch.tensor(om_np, dtype=dtype, device=device)[None]
            derivs = compute_derivatives(omega, derivative, L_hat, F_phys,
                                         max_m=MAX_M, dealias=True)
            fd = fd_validate(omega, derivative, L_hat, F_phys, MAX_M, FD_DT,
                             dealias=True)
            print(f"[{member}] t={t_snap:.1f} |omega|={_rms(omega):.4e} "
                  f"|N|={_rms(derivs['N'][0]):.4e}", flush=True)
            for m in range(1, MAX_M + 1):
                rel = _rms(derivs["N"][m] - fd[m]) / max(_rms(fd[m]), 1e-30)
                fd_rel[m].append(rel)
                print(f"    N^({m}) chain-vs-FD rel.diff = {rel:.3e}", flush=True)
            for p in range(3, MAX_M + 2):
                R, mn = assemble_R(p, derivs, L_hat, derivative)
                Rnorm[p].append(_rms(R))
                for lbl, (coeff, rms) in mn.items():
                    monos[p].setdefault(lbl, []).append(rms)

        med = {p: float(np.median(Rnorm[p])) for p in Rnorm}
        results[member] = dict(
            Nx=Nx, Ny=Ny, Lx=Lx, nu=nu, mu=mu, beta=beta,
            snap_times=[t for t, _ in snaps],
            Rnorm_snapshots={p: [float(v) for v in Rnorm[p]] for p in Rnorm},
            Rnorm_median=med,
            fd_rel_median={m: float(np.median(fd_rel[m])) for m in fd_rel},
            monomial_median={p: {lbl: float(np.median(v))
                                 for lbl, v in monos[p].items()}
                             for p in monos},
        )
        for p in Rnorm:
            npz_store[f"{member}_Rnorm_p{p}"] = np.array(Rnorm[p])
        for m in fd_rel:
            npz_store[f"{member}_fdrel_m{m}"] = np.array(fd_rel[m])
        npz_store[f"{member}_snap_times"] = np.array([t for t, _ in snaps])

    measured_p170 = read_measured(P170_CSV)
    measured_t23 = read_measured(T23_CSV)

    # ---- terms, ratios, ceilings ----
    summary = {}
    for member, r in results.items():
        med = r["Rnorm_median"]
        per_dt = {}
        for dT in DTS:
            term = {p: (dT ** p) / D_P[p] * med[p] for p in (3, 4, 5, 6)}
            ceil345 = (term[3] + term[4] + term[5]) / (term[4] + term[5])
            ceil_all = (term[3] + term[4] + term[5] + term[6]) / \
                       (term[4] + term[5] + term[6])
            per_dt[dT] = dict(
                term3=term[3], term4=term[4], term5=term[5], term6=term[6],
                term3_over_term4=term[3] / term[4],
                term4_over_term5=term[4] / term[5],
                predicted_ceiling_345=ceil345,
                predicted_ceiling_3456=ceil_all,
                measured_r3anal_kf4=measured_p170.get(dT),
                measured_r3anal_kf4_third23=measured_t23.get(dT),
            )
        summary[member] = dict(
            raw_R3_over_R4=med[3] / med[4],
            raw_R4_over_R5=med[4] / med[5],
            Rnorm_median=med,
            fd_rel_median=r["fd_rel_median"],
            per_dT=per_dt,
        )

    # ---- print tables ----
    print("\n" + "=" * 78)
    print("MEDIANS OVER 4 DEVELOPED-FLOW SNAPSHOTS PER MEMBER (all float64)")
    print("=" * 78)
    for member, s in summary.items():
        med = s["Rnorm_median"]
        print(f"\n--- {member} ---")
        print("  ||R_p|| medians: " +
              "  ".join(f"R{p}={med[p]:.4e}" for p in sorted(med)))
        print(f"  raw ||R3||/||R4|| = {s['raw_R3_over_R4']:.4f}   "
              f"||R4||/||R5|| = {s['raw_R4_over_R5']:.4f}")
        print("  FD-validation medians: " +
              "  ".join(f"N^({m})={v:.2e}" for m, v in
                        s["fd_rel_median"].items()))
        print(f"  {'dT':>8s} {'term3':>10s} {'term4':>10s} {'term5':>10s} "
              f"{'t3/t4':>8s} {'t4/t5':>8s} {'ceiling':>8s} {'measured':>9s}")
        for dT in DTS:
            d = s["per_dT"][dT]
            meas = d["measured_r3anal_kf4"]
            print(f"  {dT:8.4f} {d['term3']:10.3e} {d['term4']:10.3e} "
                  f"{d['term5']:10.3e} {d['term3_over_term4']:8.2f} "
                  f"{d['term4_over_term5']:8.2f} "
                  f"{d['predicted_ceiling_345']:8.2f} "
                  f"{(f'{meas:8.1f}x' if meas else '   --')}")

    # ---- yaml ----
    yaml_out = {
        "_README": (
            "TRUE R3/R4/R5 truncation terms vs measured analytic-closure "
            "improvements (Sanaa 2026-07-13 item 2). For each replication "
            "member, medians over 4 developed-flow DNS snapshots (t~30/50/"
            "70/90, float64) of the assembled ||R_p|| RMS (p=3..6), the "
            "dT-weighted LTE terms term_p=(dT^p/D_p)||R_p|| (D=12/24/240/"
            "1440), their ratios, and the predicted improvement ceiling of "
            "a PERFECT R3-only closure, ceiling=(t3+t4+t5)/(t4+t5) -- "
            "one-step LTE-dominated assumption, no accumulation. Measured "
            "columns: 16-step r3anal rollout improvements on FRC-kf4 IC837 "
            "(p170 full-world; third23 = 2/3-world variant), truth RK4 "
            "h_fine=1e-5, I4 convention. FD-validation medians quantify "
            "trust in each N^(m): small = chain-rule value confirmed by an "
            "independent RK4 finite difference; N^(4),N^(5) (feeding R5/R6) "
            "are FD-roundoff-limited, so R3/R4 are the trustworthy pair."),
        "snapshots_source": str(ENS),
        "snapshot_times": SNAP_TIMES,
        "fd_dt": FD_DT,
        "measured_source_p170": str(P170_CSV),
        "measured_source_third23": str(T23_CSV),
        "members": summary,
    }
    with open(YAML_DIR / "summary.yaml", "w") as f:
        yaml.safe_dump(yaml_out, f, sort_keys=False, width=100)
    np.savez(NPZ_PATH, **npz_store)
    with open(YAML_DIR / "results_full.json", "w") as f:
        json.dump(dict(results={k: v for k, v in results.items()},
                       summary=summary), f, indent=1, default=float)

    # ---- figures ----
    make_figures(summary, measured_p170, measured_t23)
    write_explainer()
    print("\n[rterms] DONE.", flush=True)


# fixed-order CVD-safe palette (Tol bright)
C_TERM = {3: "#4477AA", 4: "#EE6677", 5: "#228833", 6: "#CCBB44"}
C_PRED = "#4477AA"
C_MEAS = "#EE6677"
C_MEAS23 = "#CCBB44"


def make_figures(summary, measured_p170, measured_t23):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # FIG 1: per-order weighted contribution bars, per dT per member
    fig, axes = plt.subplots(1, len(summary), figsize=(13, 4.2), sharey=True)
    for ax, (member, s) in zip(np.atleast_1d(axes), summary.items()):
        xs = np.arange(len(DTS))
        wid = 0.2
        for i, p in enumerate((3, 4, 5, 6)):
            vals = [s["per_dT"][dT][f"term{p}"] for dT in DTS]
            ax.bar(xs + (i - 1.5) * wid, vals, wid, label=f"$p={p}$",
                   color=C_TERM[p], edgecolor="white", linewidth=0.6)
        ax.set_yscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{dT:g}" for dT in DTS])
        ax.set_xlabel(r"$\Delta T$")
        ax.set_title(member, fontsize=11)
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)
        ax.set_axisbelow(True)
    np.atleast_1d(axes)[0].set_ylabel(
        r"LTE term  $(\Delta T^p/D_p)\,\Vert R_p\Vert$")
    np.atleast_1d(axes)[0].legend(title="order", frameon=False)
    fig.suptitle("Per-order weighted LTE contributions "
                 "(medians over 4 developed-flow snapshots, float64)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(PNG_DIR / "per_order_weighted_lte_contributions.png", dpi=170)
    plt.close(fig)

    # FIG 2: predicted R3-only ceiling vs measured r3anal improvement per dT
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    xs = np.arange(len(DTS))
    members = list(summary.keys())
    n_groups = len(members) + 2
    wid = 0.8 / n_groups
    member_alpha = {0: 1.0, 1: 0.55, 2: 0.28}
    for i, member in enumerate(members):
        vals = [summary[member]["per_dT"][dT]["predicted_ceiling_345"]
                for dT in DTS]
        alpha = member_alpha.get(i, 0.28)
        ax.bar(xs + (i - (n_groups - 1) / 2) * wid, vals, wid,
               label=f"predicted ceiling {member}",
               color=C_PRED, alpha=alpha, edgecolor="white", linewidth=0.6)
    meas = [measured_p170.get(dT, np.nan) for dT in DTS]
    ax.bar(xs + (len(members) - (n_groups - 1) / 2) * wid, meas, wid,
           label="measured r3anal 16-step (kf4, full world)",
           color=C_MEAS, edgecolor="white", linewidth=0.6)
    meas23 = [measured_t23.get(dT, np.nan) for dT in DTS]
    ax.bar(xs + (len(members) + 1 - (n_groups - 1) / 2) * wid, meas23, wid,
           label="measured r3anal 16-step (kf4, 2/3-world)",
           color=C_MEAS23, edgecolor="white", linewidth=0.6)
    for x, v in zip(xs + (len(members) - (n_groups - 1) / 2) * wid, meas):
        if np.isfinite(v):
            ax.annotate(f"{v:.0f}x", (x, v), ha="center", va="bottom",
                        fontsize=8)
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{dT:g}" for dT in DTS])
    ax.set_xlabel(r"$\Delta T$")
    ax.set_ylabel("improvement factor over bare AB2CN2")
    ax.set_title("Perfect-R3 predicted ceiling (one-step LTE) vs measured\n"
                 "16-step analytic-closure improvement", fontsize=11)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(PNG_DIR / "predicted_ceiling_vs_measured_improvement.png",
                dpi=170)
    plt.close(fig)


def write_explainer():
    txt = f"""WHAT THIS FOLDER SHOWS
======================
The TRUE (measured, not assumed) sizes of the AB2CN2 truncation operators
R_3, R_4, R_5, R_6 on developed-flow snapshots of the three replication
members (FRC-kf4, FRC-256, FRC-combo), and whether the truncation series
predicts the improvement factors we MEASURED with the exact analytic R3
closure in 16-step rollouts.

INPUTS
  - omega snapshots: raw ensemble DNS, 4 per member at t~30/50/70/90
    ({ENS}/<member>/DNS_FR.npz, float32 disk, upcast; all compute float64)
  - forcing rebuilt exactly from each member's hydra config
    (F = A cos(Bx) + D cos(Ey))
  - measured improvements: Results/apost_ladder_20260709_p170 (full world)
    and Results/apost_ladder_20260709_third23 (2/3 world) summary CSVs
    (16-step r3anal rollouts, FRC-kf4 IC837, truth RK4 h_fine=1e-5, I4).

FORMULAS
  tau = -(h^3/12)R3 - (h^4/24)R4 - (h^5/240)R5 - (h^6/1440)R6 + O(h^7)
  term_p = (dT^p / D_p) * ||R_p||,  D = {{3:12, 4:24, 5:240, 6:1440}}
  predicted ceiling of a PERFECT R3-only closure (one-step LTE-dominated,
  no accumulation/feedback):
    ceiling(dT) = (term3 + term4 + term5) / (term4 + term5)

OUTPUTS
  pngs/{NAME}/per_order_weighted_lte_contributions.png
      grouped bars, term_p per dT per member, log scale.
  pngs/{NAME}/predicted_ceiling_vs_measured_improvement.png
      predicted ceiling (3 members) vs measured r3anal improvement (kf4,
      full world + 2/3 world), per dT, log scale.
  yamls/{NAME}/summary.yaml  -- all numbers incl. FD-validation medians.
  yamls/{NAME}/truncation_term_ratios_consolidated.npz -- per-snapshot raw.

PURPOSE
Sanaa 2026-07-13 item 2: sanity-check the measured 132.6x/35.1x/71.4x
(1.5e-2/1e-2/5e-3) analytic-closure improvements against what the
truncation series itself allows, and locate where the comparison gaps
(rollout accumulation vs one-step LTE; per-dT bare baselines; FD trust
in N^(4)/N^(5)).
"""
    with open(PNG_DIR / "README_what_this_folder_shows.txt", "w") as f:
        f.write(txt)
    with open(YAML_DIR / "README_what_this_folder_shows.txt", "w") as f:
        f.write(txt)


if __name__ == "__main__":
    main()
