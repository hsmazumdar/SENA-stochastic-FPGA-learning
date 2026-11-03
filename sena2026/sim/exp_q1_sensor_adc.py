"""
C3 — Sensor / statistical-ADC end-to-end robustness.

The deployed front-end is a 1-bit comparator per pixel ("statistical ADC"):
    bit = (pixel + n(t) >= threshold + offset)
with temporal noise n ~ N(0, sigma_t) redrawn per readout and a fixed-pattern
offset ~ N(0, sigma_fpn) frozen per device. This script feeds noisy readouts
of the frozen MNIST test split into an already-trained SENA ensemble
(A2 repeat-0 adaptive checkpoint) and measures early-exit accuracy.

Examples:
  python exp_q1_sensor_adc.py --profile smoke
  python exp_q1_sensor_adc.py --profile full
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

from sena_ensemble import SENAEnsemble

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / "results"
DATA = ROOT / "data"
FIGS = ROOT / "assets" / "figs"
MEDIA = ROOT.parent / "_media"

THRESHOLD = 0.3

PROFILES = {
    "smoke": dict(
        n_train=10000, n_test=250, T_max=6, theta=0.55, trials=2,
        sigma_t=[0.0, 0.1, 0.3],
        sigma_fpn=[0.0, 0.1],
        checkpoint="q1_ablation_full_r0_adaptive.npz",
    ),
    "full": dict(
        n_train=10000, n_test=2500, T_max=40, theta=0.55, trials=3,
        sigma_t=[0.0, 0.05, 0.1, 0.15, 0.2, 0.3],
        sigma_fpn=[0.0, 0.05, 0.1, 0.2],
        checkpoint="q1_ablation_full_r0_adaptive.npz",
    ),
}


def load_raw_test_split(n_train, n_test, seed=2026):
    """Raw grayscale test images matching load_mnist_binary's frozen split.

    Reproduces the same RNG draw order (train indices first, then test) so
    the returned images align 1:1 with the binarised split the ensemble
    was trained and evaluated on.
    """
    from torchvision import datasets, transforms
    ds_tr = datasets.MNIST(str(DATA), train=True, download=True,
                           transform=transforms.ToTensor())
    ds_te = datasets.MNIST(str(DATA), train=False, download=True,
                           transform=transforms.ToTensor())
    Xte = ds_te.data.numpy().astype(np.float32) / 255.0
    yte = ds_te.targets.numpy().astype(np.int64)
    rng = np.random.default_rng(seed)
    _ = rng.choice(len(ds_tr.data), size=min(n_train, len(ds_tr.data)),
                   replace=False)
    te_idx = rng.choice(len(Xte), size=min(n_test, len(Xte)), replace=False)
    return Xte[te_idx].reshape(len(te_idx), -1), yte[te_idx]


def adc_readout(X_raw, sigma_t, sigma_fpn, rng):
    """One statistical-ADC frame: comparator with FPN offset + temporal noise."""
    offset = rng.normal(0.0, sigma_fpn, size=X_raw.shape[1])[None, :] \
        if sigma_fpn > 0 else 0.0
    noise = rng.normal(0.0, sigma_t, size=X_raw.shape) if sigma_t > 0 else 0.0
    return (X_raw + noise >= THRESHOLD + offset).astype(np.int64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=PROFILES, default="smoke")
    args = parser.parse_args()
    cfg = PROFILES[args.profile]
    RESULTS.mkdir(parents=True, exist_ok=True)

    ckpt = RESULTS / cfg["checkpoint"]
    if not ckpt.exists():
        raise FileNotFoundError(f"missing checkpoint {ckpt}; run A2 first")
    print(f"loading {ckpt.name} ...", flush=True)
    ens, _ = SENAEnsemble.load(ckpt)
    X_raw, y = load_raw_test_split(cfg["n_train"], cfg["n_test"])
    print(f"raw test split: {X_raw.shape}", flush=True)

    # clean reference (ideal noiseless comparator)
    clean = ens.accuracy_early_exit(
        (X_raw >= THRESHOLD).astype(np.int64), y,
        T_max=cfg["T_max"], theta=cfg["theta"])
    print(f"clean accuracy {clean['acc']*100:.2f}%", flush=True)

    rows = []
    t0 = time.perf_counter()
    # temporal-noise sweep at zero FPN, and FPN sweep at zero temporal noise,
    # plus the combined worst corners
    combos = [(s, 0.0) for s in cfg["sigma_t"]]
    combos += [(0.0, f) for f in cfg["sigma_fpn"] if f > 0]
    combos += [(s, f) for s in cfg["sigma_t"][-2:]
               for f in cfg["sigma_fpn"][-2:] if s > 0 and f > 0]
    combos = sorted(set(combos))

    for sigma_t, sigma_fpn in combos:
        accs, passes = [], []
        for trial in range(cfg["trials"]):
            rng = np.random.default_rng(10_000 + 137 * trial)
            Xb = adc_readout(X_raw, sigma_t, sigma_fpn, rng)
            out = ens.accuracy_early_exit(
                Xb, y, T_max=cfg["T_max"], theta=cfg["theta"])
            accs.append(out["acc"])
            passes.append(out["mean_passes"])
        accs = np.asarray(accs)
        rows.append(dict(
            sigma_t=float(sigma_t), sigma_fpn=float(sigma_fpn),
            n_trials=cfg["trials"],
            accuracy_mean=float(accs.mean()),
            accuracy_std=float(accs.std(ddof=1)) if len(accs) > 1 else 0.0,
            mean_passes=float(np.mean(passes)),
            delta_vs_clean=float(accs.mean() - clean["acc"]),
        ))
        print(f"  sigma_t={sigma_t:.2f} fpn={sigma_fpn:.2f} "
              f"acc={accs.mean()*100:.2f}%±{accs.std(ddof=1)*100 if len(accs)>1 else 0:.2f} "
              f"(Δ{(accs.mean()-clean['acc'])*100:+.2f}%)", flush=True)

    elapsed = time.perf_counter() - t0

    # figure
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), dpi=180)
    tn = sorted([r for r in rows if r["sigma_fpn"] == 0.0],
                key=lambda r: r["sigma_t"])
    axes[0].errorbar([r["sigma_t"] for r in tn],
                     [r["accuracy_mean"] * 100 for r in tn],
                     yerr=[r["accuracy_std"] * 100 for r in tn],
                     fmt="o-", color="#c00000", capsize=3, label="SENA")
    axes[0].axhline(clean["acc"] * 100, ls="--", color="#666666",
                    label="clean comparator")
    axes[0].set_xlabel("Temporal noise σ_t (fraction of full scale)")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_title("(a) Temporal comparator noise")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)

    fp = sorted([r for r in rows if r["sigma_t"] == 0.0],
                key=lambda r: r["sigma_fpn"])
    axes[1].errorbar([r["sigma_fpn"] for r in fp],
                     [r["accuracy_mean"] * 100 for r in fp],
                     yerr=[r["accuracy_std"] * 100 for r in fp],
                     fmt="s-", color="#4472c4", capsize=3, label="SENA")
    axes[1].axhline(clean["acc"] * 100, ls="--", color="#666666",
                    label="clean comparator")
    axes[1].set_xlabel("Fixed-pattern offset σ_fpn (fraction of full scale)")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("(b) Per-pixel threshold mismatch")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25)

    fig.suptitle(
        f"Statistical 1-bit ADC front-end robustness ({args.profile})\n"
        f"Frozen MNIST split; A2 checkpoint; θ={cfg['theta']}, "
        f"T_max={cfg['T_max']}; error bars = SD over {cfg['trials']} readouts",
        fontsize=11, fontweight="bold")
    fig.tight_layout()
    for directory in (FIGS, MEDIA):
        directory.mkdir(parents=True, exist_ok=True)
        fig.savefig(directory / "q1_sensor_adc.png", bbox_inches="tight")
        fig.savefig(directory / "q1_sensor_adc.svg", bbox_inches="tight")
    plt.close(fig)

    payload = dict(
        profile=args.profile,
        config={**cfg, "dataset_seed": 2026, "threshold": THRESHOLD},
        elapsed_s=elapsed,
        clean_accuracy=clean["acc"],
        clean_mean_passes=clean["mean_passes"],
        runs=rows,
    )
    out_json = RESULTS / "q1_sensor_adc.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with open(RESULTS / "q1_sensor_adc.csv", "w", newline="",
              encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))
    print(f"elapsed {elapsed:.1f}s  wrote {out_json}")


if __name__ == "__main__":
    main()
