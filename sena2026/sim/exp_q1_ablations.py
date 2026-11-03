"""
Multi-seed SENA ablations for Q1 evidence.

The experiment isolates three claims without changing the data split:
  * ensemble benefit: M=1 versus M=3
  * early-exit benefit: theta=0.55 versus forced T_max
  * adaptive-N benefit: calibrated per-layer N versus fixed N=64

Examples:
  python exp_q1_ablations.py --profile smoke
  python exp_q1_ablations.py --profile full
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

from sena_ensemble import SENAEnsemble, load_mnist_binary

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / "results"
FIGS = ROOT / "assets" / "figs"
MEDIA = ROOT.parent / "_media"

PROFILES = {
    "smoke": dict(n_train=800, n_test=250, hidden=32, epochs=2,
                  samples_per_epoch=400, repeats=[0], T_max=6),
    "full": dict(n_train=10000, n_test=2500, hidden=160, epochs=40,
                 samples_per_epoch=3000, repeats=[0, 1, 2, 3, 4], T_max=40),
}
THETA = 0.55


def evaluate(ens, data, name, repeat, theta, t_max, adaptive_N):
    out = ens.accuracy_early_exit(
        data["Xte"], data["yte"], T_max=t_max, theta=theta)
    return dict(
        condition=name, repeat=repeat, M=len(ens.nets),
        early_exit=bool(theta <= 1.0), adaptive_N=adaptive_N,
        theta=theta, accuracy=out["acc"],
        mean_passes=out["mean_passes"],
        relative_energy=out["mean_energy"] / out["energy_full"],
        energy_saving=out["energy_saving"],
    )


def aggregate(rows):
    summary = []
    for condition in sorted({r["condition"] for r in rows}):
        subset = [r for r in rows if r["condition"] == condition]
        item = dict(condition=condition, n_repeats=len(subset))
        for key in ("accuracy", "mean_passes", "relative_energy",
                    "energy_saving"):
            values = np.asarray([r[key] for r in subset], dtype=float)
            item[f"{key}_mean"] = float(values.mean())
            item[f"{key}_std"] = (
                float(values.std(ddof=1)) if len(values) > 1 else 0.0)
        item.update(M=subset[0]["M"], early_exit=subset[0]["early_exit"],
                    adaptive_N=subset[0]["adaptive_N"])
        summary.append(item)
    return summary


def render(summary, profile):
    order = [
        "single_full", "single_early_exit",
        "ensemble_full", "ensemble_early_exit",
        "ensemble_fixed_N",
    ]
    by_name = {r["condition"]: r for r in summary}
    rows = [by_name[k] for k in order if k in by_name]
    labels = [r["condition"].replace("_", "\n") for r in rows]
    acc = np.asarray([r["accuracy_mean"] for r in rows]) * 100
    acc_sd = np.asarray([r["accuracy_std"] for r in rows]) * 100
    energy = np.asarray([r["relative_energy_mean"] for r in rows]) * 100
    energy_sd = np.asarray([r["relative_energy_std"] for r in rows]) * 100

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=180)
    x = np.arange(len(rows))
    axes[0].bar(x, acc, yerr=acc_sd, capsize=4, color="#4472c4")
    axes[0].set_xticks(x, labels, fontsize=8)
    axes[0].set_ylabel("MNIST test accuracy (%)")
    axes[0].set_title("(a) Ensemble, early-exit, and adaptive-N ablation")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].set_ylim(max(0, float(acc.min() - 8)),
                     min(100, float(acc.max() + 4)))

    axes[1].bar(x, energy, yerr=energy_sd, capsize=4, color="#70ad47")
    axes[1].set_xticks(x, labels, fontsize=8)
    axes[1].set_ylabel("Energy (% of each condition's full budget)")
    axes[1].set_title("(b) Work retained after confidence early-exit")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].set_ylim(0, 108)

    fig.suptitle(
        f"SENA ablations on a fixed MNIST split ({profile} profile)\n"
        "Error bars: sample standard deviation across independent repeats",
        fontsize=11, fontweight="bold")
    fig.tight_layout()
    for directory in (FIGS, MEDIA):
        directory.mkdir(parents=True, exist_ok=True)
        fig.savefig(directory / "q1_sena_ablations.png", bbox_inches="tight")
        fig.savefig(directory / "q1_sena_ablations.svg", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=PROFILES, default="smoke")
    parser.add_argument(
        "--repeats",
        help="comma-separated repeat IDs; defaults to all profile repeats")
    parser.add_argument(
        "--retrain", action="store_true",
        help="ignore per-repeat checkpoints and train again")
    args = parser.parse_args()
    cfg = PROFILES[args.profile]
    repeats = (list(map(int, args.repeats.split(",")))
               if args.repeats else cfg["repeats"])
    RESULTS.mkdir(parents=True, exist_ok=True)

    data = load_mnist_binary(
        n_train=cfg["n_train"], n_test=cfg["n_test"],
        threshold=0.3, seed=2026)
    layer_sizes = (data["n_in"], cfg["hidden"], 10)
    rows = []
    t0 = time.perf_counter()

    for repeat in repeats:
        seeds = [100 * repeat + i for i in range(3)]
        adaptive_path = RESULTS / (
            f"q1_ablation_{args.profile}_r{repeat}_adaptive.npz")
        if adaptive_path.exists() and not args.retrain:
            print(f"loading {adaptive_path.name} ...", flush=True)
            adaptive, _ = SENAEnsemble.load(adaptive_path)
        else:
            print(f"adaptive ensemble repeat {repeat} ...", flush=True)
            adaptive = SENAEnsemble.train_members(
                data["Xtr"], data["Ytr"], layer_sizes=layer_sizes,
                n_members=3, epochs=cfg["epochs"], seeds=seeds,
                samples_per_epoch=cfg["samples_per_epoch"],
                val_every=max(1, cfg["epochs"] // 4),
                T_probe=min(12, cfg["T_max"]), verbose=False,
                adaptive_calibration=True)
            adaptive.save(
                adaptive_path,
                metadata=dict(profile=args.profile, repeat=repeat,
                              adaptive_N=True, dataset_seed=2026))
        single = SENAEnsemble([adaptive.nets[0]])
        rows.extend([
            evaluate(single, data, "single_full", repeat, 1.01,
                     cfg["T_max"], True),
            evaluate(single, data, "single_early_exit", repeat, THETA,
                     cfg["T_max"], True),
            evaluate(adaptive, data, "ensemble_full", repeat, 1.01,
                     cfg["T_max"], True),
            evaluate(adaptive, data, "ensemble_early_exit", repeat, THETA,
                     cfg["T_max"], True),
        ])

        fixed_path = RESULTS / (
            f"q1_ablation_{args.profile}_r{repeat}_fixedN.npz")
        if fixed_path.exists() and not args.retrain:
            print(f"loading {fixed_path.name} ...", flush=True)
            fixed, _ = SENAEnsemble.load(fixed_path)
        else:
            print(f"fixed-N ensemble repeat {repeat} ...", flush=True)
            fixed = SENAEnsemble.train_members(
                data["Xtr"], data["Ytr"], layer_sizes=layer_sizes,
                n_members=3, epochs=cfg["epochs"], seeds=seeds,
                samples_per_epoch=cfg["samples_per_epoch"],
                val_every=max(1, cfg["epochs"] // 4),
                T_probe=min(12, cfg["T_max"]), verbose=False,
                adaptive_calibration=False, fixed_N=64)
            fixed.save(
                fixed_path,
                metadata=dict(profile=args.profile, repeat=repeat,
                              adaptive_N=False, fixed_N=64,
                              dataset_seed=2026))
        rows.append(evaluate(
            fixed, data, "ensemble_fixed_N", repeat, THETA,
            cfg["T_max"], False))

    elapsed = time.perf_counter() - t0
    out_json = RESULTS / "q1_sena_ablations.json"
    # Merge with earlier full-profile runs so --repeats slices do not wipe history.
    if out_json.exists():
        prior = json.loads(out_json.read_text(encoding="utf-8"))
        if prior.get("profile") == args.profile:
            keyed = {(r["condition"], r["repeat"]): r
                     for r in prior.get("runs", [])}
            for r in rows:
                keyed[(r["condition"], r["repeat"])] = r
            rows = [keyed[k] for k in sorted(keyed)]
            elapsed += float(prior.get("elapsed_s", 0.0))
    summary = aggregate(rows)
    render(summary, args.profile)
    payload = dict(
        profile=args.profile,
        config={**cfg,
                "repeats_run": sorted({r["repeat"] for r in rows}),
                "dataset_seed": 2026, "threshold": 0.3,
                "theta_early_exit": THETA},
        elapsed_s=elapsed, runs=rows, summary=summary,
    )
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with open(RESULTS / "q1_sena_ablations.csv", "w", newline="",
              encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))
    print(f"elapsed {elapsed:.1f}s  repeats={payload['config']['repeats_run']}")


if __name__ == "__main__":
    main()
