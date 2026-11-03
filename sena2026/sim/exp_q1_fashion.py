"""
exp_q1_fashion.py
=================
A5 — Fashion-MNIST replication of the matched-budget SENA story.

Reports:
  * FP32 / INT8 / BinaryConnect / XNOR baselines on Fashion-MNIST
  * SENA ensemble early-exit sweep (theta) on the same split
  * One Pareto figure comparing Fashion-MNIST vs the existing MNIST curve

Examples:
  python exp_q1_fashion.py --profile smoke
  python exp_q1_fashion.py --profile full
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

from exp_q1_baselines import aggregate as agg_base
from exp_q1_baselines import run_seed as baseline_seed
from sena_ensemble import SENAEnsemble, load_fashion_mnist_binary

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / "results"
FIGS = ROOT / "assets" / "figs"
MEDIA = ROOT.parent / "_media"

PROFILES = {
    "smoke": dict(n_train=1200, n_test=400, epochs_base=2, seeds=[0],
                  hidden=64, epochs_sena=2, members=2,
                  samples_per_epoch=600, T_max=8,
                  thetas=[0.55, 0.75, 1.01]),
    "full": dict(n_train=10000, n_test=2500, epochs_base=30,
                 seeds=[0, 1, 2, 3, 4],
                 hidden=160, epochs_sena=40, members=3,
                 samples_per_epoch=3000, T_max=40,
                 thetas=[0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.01]),
}


def render(base_summary, sweep, mnist_sweep, profile):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=180)

    labels = [r["method"] for r in base_summary]
    means = np.array([r["accuracy_mean"] for r in base_summary]) * 100
    stds = np.array([r["accuracy_std"] for r in base_summary]) * 100
    axes[0].bar(labels, means, yerr=stds, capsize=4, color="#70ad47")
    axes[0].set_ylabel("Fashion-MNIST test accuracy (%)")
    axes[0].set_title("(a) Matched baselines on Fashion-MNIST")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].set_ylim(max(0, float(means.min() - 8)),
                     min(100, float(means.max() + 4)))

    rel = [r["relative_energy"] * 100 for r in sweep]
    acc = [r["acc"] * 100 for r in sweep]
    axes[1].plot(rel, acc, "o-", color="#c00000", lw=2, label="Fashion-MNIST SENA")
    if mnist_sweep:
        m_rel = [r["relative_energy"] * 100 for r in mnist_sweep]
        m_acc = [r["acc"] * 100 for r in mnist_sweep]
        axes[1].plot(m_rel, m_acc, "s--", color="#4472c4", lw=1.6,
                     label="MNIST SENA (reference)")
    axes[1].set_xlabel("Relative energy (% of full SENA budget)")
    axes[1].set_ylabel("Test accuracy (%)")
    axes[1].set_title("(b) Early-exit Pareto on a second dataset")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)

    fig.suptitle(
        f"A5 Fashion-MNIST replication ({profile} profile)\n"
        "Same 784-H-10 budget and binary-pixel front-end as MNIST",
        fontsize=11, fontweight="bold")
    fig.tight_layout()
    for d in (FIGS, MEDIA):
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / "q1_fashion_mnist_replication.png", bbox_inches="tight")
        fig.savefig(d / "q1_fashion_mnist_replication.svg", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=PROFILES, default="smoke")
    args = parser.parse_args()
    cfg = PROFILES[args.profile]
    RESULTS.mkdir(parents=True, exist_ok=True)

    data = load_fashion_mnist_binary(
        n_train=cfg["n_train"], n_test=cfg["n_test"],
        threshold=0.3, seed=2026)

    t0 = time.perf_counter()
    base_rows = []
    for seed in cfg["seeds"]:
        print(f"Fashion baseline seed {seed} ...", flush=True)
        base_rows.extend(baseline_seed(data, seed, cfg["epochs_base"]))
    base_summary = agg_base(base_rows)

    print("Training Fashion-MNIST SENA ensemble ...", flush=True)
    ens = SENAEnsemble.train_members(
        data["Xtr"], data["Ytr"],
        layer_sizes=(784, cfg["hidden"], 10),
        n_members=cfg["members"], epochs=cfg["epochs_sena"],
        seeds=list(range(cfg["members"])),
        samples_per_epoch=cfg["samples_per_epoch"],
        val_every=max(1, cfg["epochs_sena"] // 4),
        T_probe=min(16, cfg["T_max"]), verbose=True)
    sweep = ens.sweep_theta(
        data["Xte"], data["yte"], cfg["thetas"], T_max=cfg["T_max"])
    elapsed = time.perf_counter() - t0

    mnist_sweep = []
    mnist_json = RESULTS / "mnist_sena_results.json"
    if mnist_json.exists():
        mnist_sweep = json.loads(mnist_json.read_text(encoding="utf-8"))["sweep"]

    render(base_summary, sweep, mnist_sweep, args.profile)
    payload = dict(
        profile=args.profile, dataset="fashion_mnist",
        config={**cfg, "dataset_seed": 2026, "threshold": 0.3},
        elapsed_s=elapsed,
        baselines={"runs": base_rows, "summary": base_summary},
        sena_sweep=sweep,
        fixed_budget_acc=next(r["acc"] for r in sweep if r["theta"] > 1.0),
    )
    out = RESULTS / "q1_fashion_mnist.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with open(RESULTS / "q1_fashion_mnist_sweep.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(sweep[0]))
        w.writeheader()
        w.writerows(sweep)
    fp32 = next(r for r in base_summary if r["method"] == "fp32_mlp")
    print(json.dumps(dict(
        baseline_fp32_mean=fp32["accuracy_mean"],
        baseline_fp32_std=fp32["accuracy_std"],
        sena_fixed=payload["fixed_budget_acc"],
        elapsed_s=elapsed), indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
