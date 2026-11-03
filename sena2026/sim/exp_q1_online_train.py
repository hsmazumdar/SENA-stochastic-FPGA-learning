"""
exp_q1_online_train.py
======================
A6 — Online multiplier-free training evidence.

Streams one MNIST sample at a time through StochasticNet.train_epoch on
single-example batches, recording:
  * test accuracy vs samples seen
  * cumulative integer add/sub / compare / RNG ops (no multiply)
  * a RealBP float-batch reference curve for the same split

This is the hardware claim closest to Mazumdar 1995: on-chip learning with
only ADD/SUB/CMP/RNG.

Examples:
  python exp_q1_online_train.py --profile smoke
  python exp_q1_online_train.py --profile full
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

from sena_ensemble import load_mnist_binary
from stochastic_core import RealBPNet, StochasticConfig, StochasticNet

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / "results"
FIGS = ROOT / "assets" / "figs"
MEDIA = ROOT.parent / "_media"

PROFILES = {
    "smoke": dict(n_train=2000, n_test=400, hidden=64, max_samples=1500,
                  eval_every=250, seeds=[0], T_eval=8),
    "full": dict(n_train=10000, n_test=2500, hidden=160, max_samples=10000,
                 eval_every=500, seeds=[0, 1, 2], T_eval=24),
}

# Relative op energies (same convention as EnergyModel)
E_ADD, E_CMP, E_RNG, E_MUL = 1.0, 1.0, 0.5, 25.0


def count_stoch_sample(n_in, n_hid, n_out):
    """Ops for one online sample: forward + reverse update (Mazumdar)."""
    # Forward: ~0.5 of weights participate (bit-gated) + neuron compares/RNG
    fwd_adds = 0.5 * ((n_in + 1) * n_hid + (n_hid + 1) * n_out)
    fwd_cmp = n_hid + n_out
    fwd_rng = n_hid + n_out
    # Reverse: outer products are sign/add updates (no mul), plus back gate
    rev_adds = (n_hid + 1) * n_out + (n_in + 1) * n_hid
    rev_cmp = n_out + n_hid
    rev_rng = n_out + 2 * n_hid
    return dict(
        adds=fwd_adds + rev_adds,
        cmp=fwd_cmp + rev_cmp,
        rng=fwd_rng + rev_rng,
        muls=0.0,
        energy=((fwd_adds + rev_adds) * E_ADD
                + (fwd_cmp + rev_cmp) * E_CMP
                + (fwd_rng + rev_rng) * E_RNG),
    )


def count_bp_sample(n_in, n_hid, n_out):
    macs = (n_in + 1) * n_hid + (n_hid + 1) * n_out
    # Forward + backward roughly 2x MACs
    return dict(
        adds=2 * macs, cmp=0.0, rng=0.0, muls=2 * macs,
        energy=2 * macs * (E_MUL + E_ADD),
    )


def train_online_stoch(data, hidden, max_samples, eval_every, seed, T_eval):
    cfg = StochasticConfig(
        layer_sizes=(784, hidden, 10), seed=seed, epochs=1, weight_bits=16)
    net = StochasticNet(cfg)
    net.calibrate(data["Xtr"][:512])
    ops = count_stoch_sample(784, hidden, 10)
    rng = np.random.default_rng(seed + 99)
    order = rng.permutation(len(data["Xtr"]))[:max_samples]

    curve = []
    best = -1.0
    for i, idx in enumerate(order, start=1):
        x = data["Xtr"][idx:idx + 1]
        y = data["Ytr"][idx:idx + 1]
        net.train_epoch(x, y, recalibrate=(i % 500 == 0))
        if i % eval_every == 0 or i == max_samples:
            acc = net.accuracy(data["Xte"], data["yte"], T=T_eval)
            best = max(best, acc)
            curve.append(dict(
                samples=i, accuracy=acc, best_accuracy=best,
                energy=i * ops["energy"],
                adds=i * ops["adds"], muls=0.0,
                method="sena_online", seed=seed,
            ))
            print(f"  sena seed={seed} samples={i} acc={acc*100:.1f}%",
                  flush=True)
    return curve


def train_online_bp(data, hidden, max_samples, eval_every, seed):
    net = RealBPNet((784, hidden, 10), seed=seed, lr=0.05)
    ops = count_bp_sample(784, hidden, 10)
    rng = np.random.default_rng(seed + 123)
    order = rng.permutation(len(data["Xtr"]))[:max_samples]
    # RealBP expects float targets in {0,1}; binary X is fine.
    X = data["Xtr"].astype(float)
    Y = data["Ytr"].astype(float)
    curve = []
    best = -1.0
    for i, idx in enumerate(order, start=1):
        net.train_epoch(X[idx:idx + 1], Y[idx:idx + 1])
        if i % eval_every == 0 or i == max_samples:
            acc = net.accuracy(data["Xte"].astype(float), data["yte"])
            best = max(best, acc)
            curve.append(dict(
                samples=i, accuracy=acc, best_accuracy=best,
                energy=i * ops["energy"],
                adds=i * ops["adds"], muls=i * ops["muls"],
                method="real_bp_online", seed=seed,
            ))
            print(f"  bp   seed={seed} samples={i} acc={acc*100:.1f}%",
                  flush=True)
    return curve


def aggregate_curves(rows, method):
    by_s = {}
    for r in rows:
        if r["method"] != method:
            continue
        by_s.setdefault(r["samples"], []).append(r)
    out = []
    for samples in sorted(by_s):
        acc = np.asarray([r["accuracy"] for r in by_s[samples]], float)
        energy = np.asarray([r["energy"] for r in by_s[samples]], float)
        out.append(dict(
            samples=samples,
            accuracy_mean=float(acc.mean()),
            accuracy_std=float(acc.std(ddof=1)) if len(acc) > 1 else 0.0,
            energy_mean=float(energy.mean()),
        ))
    return out


def render(sena_agg, bp_agg, profile):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=180)
    sx = [r["samples"] for r in sena_agg]
    sy = np.array([r["accuracy_mean"] for r in sena_agg]) * 100
    ss = np.array([r["accuracy_std"] for r in sena_agg]) * 100
    axes[0].plot(sx, sy, "o-", color="#4472c4", lw=2, label="SENA online")
    axes[0].fill_between(sx, sy - ss, sy + ss, color="#4472c4", alpha=0.15)
    if bp_agg:
        bx = [r["samples"] for r in bp_agg]
        by = np.array([r["accuracy_mean"] for r in bp_agg]) * 100
        axes[0].plot(bx, by, "s--", color="#c00000", lw=1.6,
                     label="RealBP online")
    axes[0].set_xlabel("Training samples seen (online)")
    axes[0].set_ylabel("MNIST test accuracy (%)")
    axes[0].set_title("(a) Accuracy vs streamed samples")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    se = [r["energy_mean"] for r in sena_agg]
    axes[1].plot(se, sy, "o-", color="#4472c4", lw=2, label="SENA online")
    if bp_agg:
        be = [r["energy_mean"] for r in bp_agg]
        by = np.array([r["accuracy_mean"] for r in bp_agg]) * 100
        axes[1].plot(be, by, "s--", color="#c00000", lw=1.6,
                     label="RealBP online")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Cumulative training energy proxy (log)")
    axes[1].set_ylabel("MNIST test accuracy (%)")
    axes[1].set_title("(b) Learning efficiency (no MUL in SENA)")
    axes[1].grid(alpha=0.25, which="both")
    axes[1].legend(fontsize=8)

    fig.suptitle(
        f"A6 Online multiplier-free training ({profile} profile)\n"
        "SENA updates use ADD/SUB/CMP/RNG only",
        fontsize=11, fontweight="bold")
    fig.tight_layout()
    for d in (FIGS, MEDIA):
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / "q1_online_training.png", bbox_inches="tight")
        fig.savefig(d / "q1_online_training.svg", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=PROFILES, default="smoke")
    args = parser.parse_args()
    cfg = PROFILES[args.profile]
    RESULTS.mkdir(parents=True, exist_ok=True)

    data = load_mnist_binary(
        n_train=cfg["n_train"], n_test=cfg["n_test"],
        threshold=0.3, seed=2026)
    rows = []
    t0 = time.perf_counter()
    for seed in cfg["seeds"]:
        print(f"online SENA seed {seed} ...", flush=True)
        rows.extend(train_online_stoch(
            data, cfg["hidden"], cfg["max_samples"], cfg["eval_every"],
            seed, cfg["T_eval"]))
        print(f"online RealBP seed {seed} ...", flush=True)
        rows.extend(train_online_bp(
            data, cfg["hidden"], cfg["max_samples"], cfg["eval_every"],
            seed))
    elapsed = time.perf_counter() - t0
    sena_agg = aggregate_curves(rows, "sena_online")
    bp_agg = aggregate_curves(rows, "real_bp_online")
    render(sena_agg, bp_agg, args.profile)

    payload = dict(
        profile=args.profile,
        config={**cfg, "dataset_seed": 2026, "threshold": 0.3},
        elapsed_s=elapsed, runs=rows,
        sena_curve=sena_agg, bp_curve=bp_agg,
        final_sena_acc=sena_agg[-1]["accuracy_mean"] if sena_agg else None,
        final_bp_acc=bp_agg[-1]["accuracy_mean"] if bp_agg else None,
        energy_ratio_final=(
            (bp_agg[-1]["energy_mean"] / sena_agg[-1]["energy_mean"])
            if sena_agg and bp_agg else None),
    )
    out = RESULTS / "q1_online_training.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with open(RESULTS / "q1_online_training.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(json.dumps(dict(
        final_sena=payload["final_sena_acc"],
        final_bp=payload["final_bp_acc"],
        energy_ratio=payload["energy_ratio_final"],
        elapsed_s=elapsed), indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
