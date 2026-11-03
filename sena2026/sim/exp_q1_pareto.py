"""
exp_q1_pareto.py
================
B6 — Unified accuracy–energy Pareto from existing Q1 result JSONs.

Consumes:
  results/mnist_sena_results.json
  results/q1_mnist_baselines.json
  results/q1_sena_ablations.json   (if present)
  results/q1_fashion_mnist.json   (if present)
  results/q1_fpga_synth.json      (optional energy_rows)

Writes:
  assets/figs/q1_unified_pareto.png|.svg
  _media/q1_unified_pareto.png|.svg
  results/q1_unified_pareto.json
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / "results"
FIGS = ROOT / "assets" / "figs"
MEDIA = ROOT.parent / "_media"


def load(name):
    p = RESULTS / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def main():
    mnist = load("mnist_sena_results.json")
    bases = load("q1_mnist_baselines.json")
    ablations = load("q1_sena_ablations.json")
    fashion = load("q1_fashion_mnist.json")
    synth = load("q1_fpga_synth.json")

    points = []
    if mnist:
        for r in mnist["sweep"]:
            points.append(dict(
                family="SENA MNIST", label=f"θ={r['theta']:.2f}",
                acc=r["acc"] * 100,
                energy=r["relative_energy"] * 100,
                energy_kind="% full SENA budget"))
    if ablations and ablations.get("profile") == "full":
        for r in ablations.get("summary", []):
            points.append(dict(
                family="SENA ablation",
                label=r["condition"],
                acc=r["accuracy_mean"] * 100,
                energy=r["relative_energy_mean"] * 100,
                energy_kind="% condition budget",
                acc_std=r.get("accuracy_std", 0) * 100))
    if bases:
        # Baselines lack early-exit energy; place at 100% relative "one forward".
        for r in bases.get("summary", []):
            points.append(dict(
                family="Matched baseline",
                label=r["method"],
                acc=r["accuracy_mean"] * 100,
                energy=100.0,
                energy_kind="one forward (relative)",
                acc_std=r.get("accuracy_std", 0) * 100))
    if fashion and fashion.get("profile") == "full":
        for r in fashion.get("sena_sweep", []):
            points.append(dict(
                family="SENA Fashion-MNIST",
                label=f"θ={r['theta']:.2f}",
                acc=r["acc"] * 100,
                energy=r["relative_energy"] * 100,
                energy_kind="% full SENA budget"))

    fig, ax = plt.subplots(figsize=(10.5, 5.2), dpi=180)
    colors = {
        "SENA MNIST": "#c00000",
        "SENA ablation": "#4472c4",
        "Matched baseline": "#70ad47",
        "SENA Fashion-MNIST": "#ed7d31",
    }
    markers = {
        "SENA MNIST": "o",
        "SENA ablation": "^",
        "Matched baseline": "s",
        "SENA Fashion-MNIST": "D",
    }
    for family in colors:
        subset = [p for p in points if p["family"] == family]
        if not subset:
            continue
        x = [p["energy"] for p in subset]
        y = [p["acc"] for p in subset]
        yerr = [p.get("acc_std", 0) for p in subset]
        ax.errorbar(x, y, yerr=yerr if any(yerr) else None,
                    fmt=markers[family] + "-", color=colors[family],
                    lw=1.6, ms=7, capsize=3, label=family)
        for p in subset:
            ax.annotate(p["label"], (p["energy"], p["acc"]),
                        textcoords="offset points", xytext=(4, 3),
                        fontsize=6.5, color=colors[family])

    ax.set_xlabel("Relative energy (% of reference budget)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Q1 unified accuracy–energy Pareto\n"
                 "SENA early-exit vs matched baselines (same 784-H-10 budget)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    for d in (FIGS, MEDIA):
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / "q1_unified_pareto.png", bbox_inches="tight")
        fig.savefig(d / "q1_unified_pareto.svg", bbox_inches="tight")
    plt.close(fig)

    payload = dict(points=points,
                   sources=[n for n in (
                       "mnist_sena_results.json", "q1_mnist_baselines.json",
                       "q1_sena_ablations.json", "q1_fashion_mnist.json")
                       if (RESULTS / n).exists()])
    out = RESULTS / "q1_unified_pareto.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"{len(points)} points from {payload['sources']}")
    print("wrote", out)
    if synth and synth.get("folded_artix7_design"):
        f = synth["folded_artix7_design"]
        print(f"FPGA folded fit: LUT {f['lut']} ({f['lut_pct']:.1f}% of "
              f"xc7a100t), fits={f['fits']}")


if __name__ == "__main__":
    main()
