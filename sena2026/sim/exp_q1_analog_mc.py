"""
B3 — Analog stochastic-neuron Monte Carlo: mismatch, temperature, yield.

Uses the comparator + Gaussian-threshold model in analog_neuron.py.
Reports transfer RMSE to the ideal logistic and device yield under a
tolerance band.

Examples:
  python exp_q1_analog_mc.py --profile smoke
  python exp_q1_analog_mc.py --profile full
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analog_neuron import (
    AnalogNeuronParams, AnalogStochasticNeuron, sweep_transfer,
)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / "results"
FIGS = ROOT / "assets" / "figs"
MEDIA = ROOT.parent / "_media"

PROFILES = {
    "smoke": dict(
        n_devices=20, n_samples=128, n_pts=41,
        mismatches=[0.02, 0.05, 0.10],
        temperatures=[250.0, 300.0, 350.0],
        tol=0.08, rmse_tol=0.12, seed=0,
    ),
    "full": dict(
        n_devices=80, n_samples=512, n_pts=81,
        mismatches=[0.01, 0.02, 0.05, 0.08, 0.12],
        temperatures=[200.0, 250.0, 300.0, 350.0, 400.0],
        tol=0.08, rmse_tol=0.10, seed=2026,
    ),
}


def rmse(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def run_mismatch_sweep(cfg):
    vin = np.linspace(-2.0, 2.0, cfg["n_pts"])
    rows = []
    curves_by_mis = {}
    for mis in cfg["mismatches"]:
        p = AnalogNeuronParams(
            mismatch=float(mis), T=300.0, seed=cfg["seed"])
        neuron = AnalogStochasticNeuron(p)
        ideal = neuron.ideal_sigmoid(vin, scale=neuron.sigma)
        curves = neuron.monte_carlo_curves(
            vin, n_devices=cfg["n_devices"], n_samples=cfg["n_samples"])
        curves_by_mis[mis] = curves
        per_rmse = np.sqrt(np.mean((curves - ideal[None, :]) ** 2, axis=1))
        mean_curve = curves.mean(axis=0, keepdims=True)
        match = np.max(np.abs(curves - mean_curve), axis=1) <= cfg["tol"]
        # Ideal-fit yield uses an RMSE budget (MC sampling alone is ~0.06).
        rmse_tol = float(cfg.get("rmse_tol", 0.10))
        fit = per_rmse <= rmse_tol
        rows.append(dict(
            sweep="mismatch", mismatch=float(mis), temperature=300.0,
            rmse_mean=float(per_rmse.mean()),
            rmse_std=float(per_rmse.std(ddof=1)) if len(per_rmse) > 1 else 0.0,
            yield_frac=float(np.mean(fit)),
            match_yield_frac=float(np.mean(match)),
            n_devices=cfg["n_devices"], tol=cfg["tol"], rmse_tol=rmse_tol,
            energy_pJ=float(neuron.energy_pj(0.5)),
            area_um2=float(p.area_um2),
        ))
    return rows, vin, curves_by_mis


def run_temperature_sweep(cfg):
    vin = np.linspace(-2.0, 2.0, cfg["n_pts"])
    rows = []
    for temp in cfg["temperatures"]:
        p = AnalogNeuronParams(
            mismatch=0.05, T=float(temp), seed=cfg["seed"] + 7)
        neuron = AnalogStochasticNeuron(p)
        ideal = neuron.ideal_sigmoid(vin, scale=neuron.sigma)
        curves = neuron.monte_carlo_curves(
            vin, n_devices=cfg["n_devices"], n_samples=cfg["n_samples"])
        per_rmse = np.sqrt(np.mean((curves - ideal[None, :]) ** 2, axis=1))
        mean_curve = curves.mean(axis=0, keepdims=True)
        match = np.max(np.abs(curves - mean_curve), axis=1) <= cfg["tol"]
        rmse_tol = float(cfg.get("rmse_tol", 0.10))
        fit = per_rmse <= rmse_tol
        rows.append(dict(
            sweep="temperature", mismatch=0.05, temperature=float(temp),
            rmse_mean=float(per_rmse.mean()),
            rmse_std=float(per_rmse.std(ddof=1)) if len(per_rmse) > 1 else 0.0,
            yield_frac=float(np.mean(fit)),
            match_yield_frac=float(np.mean(match)),
            n_devices=cfg["n_devices"], tol=cfg["tol"], rmse_tol=rmse_tol,
            sigma_V=float(neuron.sigma),
            energy_pJ=float(neuron.energy_pj(0.5)),
            area_um2=float(p.area_um2),
        ))
    return rows, vin


def render(mis_rows, temp_rows, vin, curves_by_mis, profile, tol):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), dpi=180)

    # (a) transfer family at mid mismatch
    mid = sorted(curves_by_mis)[len(curves_by_mis) // 2]
    curves = curves_by_mis[mid]
    for i in range(min(len(curves), 24)):
        axes[0].plot(vin, curves[i], color="#4472c4", alpha=0.25, lw=0.8)
    mean_c = curves.mean(axis=0)
    axes[0].plot(vin, mean_c, color="#c00000", lw=2, label="MC mean")
    p = AnalogNeuronParams(mismatch=mid, seed=0)
    n = AnalogStochasticNeuron(p)
    axes[0].plot(vin, n.ideal_sigmoid(vin, scale=n.sigma), "--",
                 color="#333333", label="ideal logistic")
    axes[0].set_xlabel("Vin (V)")
    axes[0].set_ylabel("P(fire)")
    axes[0].set_title(f"(a) Mismatch family (σ_mis={mid} V)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)

    # (b) yield vs mismatch
    mis = [r["mismatch"] for r in mis_rows]
    yld = [r["yield_frac"] * 100 for r in mis_rows]
    match = [r.get("match_yield_frac", r["yield_frac"]) * 100 for r in mis_rows]
    axes[1].plot(mis, yld, "o-", color="#70ad47", label="fit yield (RMSE)")
    axes[1].plot(mis, match, "s--", color="#4472c4", label="match yield")
    axes[1].set_xlabel("Threshold mismatch σ (V)")
    axes[1].set_ylabel("Yield (%)")
    axes[1].set_title("(b) Yield vs mismatch @ 300 K")
    axes[1].set_ylim(0, 105)
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25)

    # (c) RMSE vs temperature
    temps = [r["temperature"] for r in temp_rows]
    rmse_m = [r["rmse_mean"] for r in temp_rows]
    rmse_s = [r["rmse_std"] for r in temp_rows]
    axes[2].errorbar(temps, rmse_m, yerr=rmse_s, fmt="o-",
                     color="#ed7d31", capsize=3)
    axes[2].set_xlabel("Temperature (K)")
    axes[2].set_ylabel("Transfer RMSE vs ideal")
    axes[2].set_title("(c) Noise temperature sensitivity")
    axes[2].grid(alpha=0.25)

    fig.suptitle(
        f"Analog stochastic neuron Monte Carlo ({profile})\n"
        f"{mis_rows[0]['n_devices']} devices; tolerance ±{tol} on P(fire)",
        fontsize=11, fontweight="bold")
    fig.tight_layout()
    for directory in (FIGS, MEDIA):
        directory.mkdir(parents=True, exist_ok=True)
        fig.savefig(directory / "q1_analog_mc.png", bbox_inches="tight")
        fig.savefig(directory / "q1_analog_mc.svg", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=PROFILES, default="smoke")
    args = parser.parse_args()
    cfg = PROFILES[args.profile]
    RESULTS.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    mis_rows, vin, curves = run_mismatch_sweep(cfg)
    temp_rows, _ = run_temperature_sweep(cfg)
    # nominal transfer for reference metrics
    ref = sweep_transfer(AnalogNeuronParams(seed=cfg["seed"]))
    elapsed = time.perf_counter() - t0

    rows = mis_rows + temp_rows
    render(mis_rows, temp_rows, vin, curves, args.profile, cfg["tol"])
    payload = dict(
        profile=args.profile,
        config=cfg,
        elapsed_s=elapsed,
        nominal_metrics=ref["metrics"],
        runs=rows,
        mismatch_summary=mis_rows,
        temperature_summary=temp_rows,
    )
    out_json = RESULTS / "q1_analog_mc.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with open(RESULTS / "q1_analog_mc.csv", "w", newline="",
              encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(dict(
        mismatch=mis_rows, temperature=temp_rows,
        elapsed_s=elapsed), indent=2))
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
