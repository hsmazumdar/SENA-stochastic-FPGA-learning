"""
B2 — Weight precision sweep (8 / 12 / 16 bit) for Q1 evidence.

Integer weights are clipped to ±(2^(bits-1)-1) after every update
(StochasticNet._quantise). Storage proxy = weight count × bits × members.

Examples:
  python exp_q1_bitwidth.py --profile smoke
  python exp_q1_bitwidth.py --profile full
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

THETA = 0.55
BITWIDTHS = (8, 12, 16)

PROFILES = {
    "smoke": dict(
        n_train=800, n_test=250, hidden=32, epochs=2,
        samples_per_epoch=400, members=3, T_max=6, seeds=[0],
    ),
    "full": dict(
        n_train=10000, n_test=2500, hidden=160, epochs=40,
        samples_per_epoch=3000, members=3, T_max=40, seeds=[0, 1, 2, 3, 4],
    ),
}


def weight_count(n_in, hidden, n_out=10):
    return (n_in + 1) * hidden + (hidden + 1) * n_out


def evaluate(ens, data, bits, seed, t_max, members):
    out = ens.accuracy_early_exit(
        data["Xte"], data["yte"], T_max=t_max, theta=THETA)
    full = ens.accuracy_early_exit(
        data["Xte"], data["yte"], T_max=t_max, theta=1.01)
    n_w = weight_count(data["n_in"], ens.nets[0].sizes[1]) * members
    return dict(
        weight_bits=bits, seed=seed, M=members,
        accuracy_early=out["acc"],
        accuracy_full=full["acc"],
        mean_passes=out["mean_passes"],
        relative_energy=out["mean_energy"] / out["energy_full"],
        energy_saving=out["energy_saving"],
        storage_bits=int(n_w * bits),
        params=int(n_w),
    )


def aggregate(rows):
    summary = []
    for bits in sorted({r["weight_bits"] for r in rows}):
        subset = [r for r in rows if r["weight_bits"] == bits]
        item = dict(weight_bits=bits, n_seeds=len(subset),
                    storage_bits=subset[0]["storage_bits"],
                    params=subset[0]["params"])
        for key in ("accuracy_early", "accuracy_full", "mean_passes",
                    "relative_energy", "energy_saving"):
            values = np.asarray([r[key] for r in subset], dtype=float)
            item[f"{key}_mean"] = float(values.mean())
            item[f"{key}_std"] = (
                float(values.std(ddof=1)) if len(values) > 1 else 0.0)
        summary.append(item)
    return summary


def render(summary, profile):
    bits = [r["weight_bits"] for r in summary]
    acc_e = np.asarray([r["accuracy_early_mean"] for r in summary]) * 100
    acc_e_sd = np.asarray([r["accuracy_early_std"] for r in summary]) * 100
    acc_f = np.asarray([r["accuracy_full_mean"] for r in summary]) * 100
    acc_f_sd = np.asarray([r["accuracy_full_std"] for r in summary]) * 100
    storage = np.asarray([r["storage_bits"] for r in summary]) / 1e3

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), dpi=180)
    x = np.arange(len(bits))
    w = 0.35
    axes[0].bar(x - w / 2, acc_e, w, yerr=acc_e_sd, capsize=4,
                color="#c00000", label="early-exit θ=0.55")
    axes[0].bar(x + w / 2, acc_f, w, yerr=acc_f_sd, capsize=4,
                color="#4472c4", label="full budget")
    axes[0].set_xticks(x, [f"{b}-bit" for b in bits])
    axes[0].set_ylabel("MNIST accuracy (%)")
    axes[0].set_title("(a) Accuracy vs weight precision")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(x, storage, color="#70ad47")
    axes[1].set_xticks(x, [f"{b}-bit" for b in bits])
    axes[1].set_ylabel("Weight storage (kbit, M members)")
    axes[1].set_title("(b) Storage cost of precision")
    axes[1].grid(axis="y", alpha=0.25)

    fig.suptitle(
        f"SENA weight-precision sweep ({profile})\n"
        "Error bars: sample SD across seeds",
        fontsize=11, fontweight="bold")
    fig.tight_layout()
    for directory in (FIGS, MEDIA):
        directory.mkdir(parents=True, exist_ok=True)
        fig.savefig(directory / "q1_bitwidth.png", bbox_inches="tight")
        fig.savefig(directory / "q1_bitwidth.svg", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=PROFILES, default="smoke")
    parser.add_argument("--retrain", action="store_true")
    args = parser.parse_args()
    cfg = PROFILES[args.profile]
    RESULTS.mkdir(parents=True, exist_ok=True)

    data = load_mnist_binary(
        n_train=cfg["n_train"], n_test=cfg["n_test"],
        threshold=0.3, seed=2026)
    rows = []
    t0 = time.perf_counter()

    for bits in BITWIDTHS:
        for seed in cfg["seeds"]:
            ckpt = RESULTS / (
                f"q1_bitwidth_{args.profile}_b{bits}_s{seed}.npz")
            if ckpt.exists() and not args.retrain:
                print(f"loading {ckpt.name} ...", flush=True)
                ens, _ = SENAEnsemble.load(ckpt)
            else:
                print(f"train bits={bits} seed={seed} ...", flush=True)
                member_seeds = [100 * seed + i for i in range(cfg["members"])]
                ens = SENAEnsemble.train_members(
                    data["Xtr"], data["Ytr"],
                    layer_sizes=(data["n_in"], cfg["hidden"], 10),
                    n_members=cfg["members"], epochs=cfg["epochs"],
                    seeds=member_seeds, weight_bits=bits,
                    samples_per_epoch=cfg["samples_per_epoch"],
                    val_every=max(1, cfg["epochs"] // 4),
                    T_probe=min(12, cfg["T_max"]), verbose=True)
                ens.save(ckpt, metadata=dict(
                    profile=args.profile, weight_bits=bits, seed=seed))
            row = evaluate(ens, data, bits, seed, cfg["T_max"], cfg["members"])
            rows.append(row)
            print(
                f"  bits={bits} seed={seed} "
                f"early={row['accuracy_early']*100:.1f}% "
                f"full={row['accuracy_full']*100:.1f}% "
                f"store={row['storage_bits']}",
                flush=True)

    elapsed = time.perf_counter() - t0
    out_json = RESULTS / "q1_bitwidth.json"
    if out_json.exists():
        prior = json.loads(out_json.read_text(encoding="utf-8"))
        if prior.get("profile") == args.profile:
            keyed = {(r["weight_bits"], r["seed"]): r
                     for r in prior.get("runs", [])}
            for r in rows:
                keyed[(r["weight_bits"], r["seed"])] = r
            rows = [keyed[k] for k in sorted(keyed)]
            elapsed += float(prior.get("elapsed_s", 0.0))

    summary = aggregate(rows)
    render(summary, args.profile)
    payload = dict(
        profile=args.profile,
        config={**cfg, "bitwidths": list(BITWIDTHS),
                "dataset_seed": 2026, "threshold": 0.3,
                "theta_early_exit": THETA},
        elapsed_s=elapsed, runs=rows, summary=summary,
    )
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with open(RESULTS / "q1_bitwidth.csv", "w", newline="",
              encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))
    print(f"elapsed {elapsed:.1f}s  wrote {out_json}")


if __name__ == "__main__":
    main()
