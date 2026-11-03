"""
B1 — Hidden width × ensemble size × T_max scaling for Q1 evidence.

Trains once per (hidden, M, seed) cell, then evaluates multiple T_max values
at a fixed early-exit threshold. Checkpoints resume long sweeps.

Examples:
  python exp_q1_scaling.py --profile smoke
  python exp_q1_scaling.py --profile full
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

from sena_ensemble import EnergyModel, SENAEnsemble, load_mnist_binary

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / "results"
FIGS = ROOT / "assets" / "figs"
MEDIA = ROOT.parent / "_media"

THETA = 0.55

PROFILES = {
    "smoke": dict(
        n_train=800, n_test=250, epochs=2, samples_per_epoch=400,
        hiddens=[32, 48], members=[1, 3], t_max_list=[4, 6], seeds=[0],
    ),
    "full": dict(
        n_train=10000, n_test=2500, epochs=30, samples_per_epoch=3000,
        hiddens=[64, 160, 256], members=[1, 3, 5], t_max_list=[20, 40],
        seeds=[0, 1, 2],
    ),
}


def evaluate_cell(ens, data, hidden, m, t_max, seed):
    out = ens.accuracy_early_exit(
        data["Xte"], data["yte"], T_max=t_max, theta=THETA)
    e_model = EnergyModel(data["n_in"], hidden, 10)
    e_unit = e_model.e_forward_stoch * m
    return dict(
        hidden=hidden, M=m, T_max=t_max, seed=seed, theta=THETA,
        accuracy=out["acc"],
        mean_passes=out["mean_passes"],
        mean_energy=out["mean_energy"],
        energy_full=float(t_max * e_unit),
        relative_energy=out["mean_energy"] / out["energy_full"],
        energy_saving=out["energy_saving"],
        e_unit=e_unit,
        params=int((data["n_in"] + 1) * hidden + (hidden + 1) * 10) * m,
    )


def aggregate(rows):
    keys = sorted({(r["hidden"], r["M"], r["T_max"]) for r in rows})
    summary = []
    for hidden, m, t_max in keys:
        subset = [r for r in rows
                  if r["hidden"] == hidden and r["M"] == m and r["T_max"] == t_max]
        item = dict(hidden=hidden, M=m, T_max=t_max, n_seeds=len(subset))
        for key in ("accuracy", "mean_passes", "relative_energy",
                    "energy_saving", "mean_energy", "params"):
            values = np.asarray([r[key] for r in subset], dtype=float)
            item[f"{key}_mean"] = float(values.mean())
            item[f"{key}_std"] = (
                float(values.std(ddof=1)) if len(values) > 1 else 0.0)
        summary.append(item)
    return summary


def render(summary, profile):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), dpi=180)
    markers = {1: "o", 3: "s", 5: "D"}
    colors = {64: "#4472c4", 160: "#c00000", 256: "#70ad47",
              32: "#4472c4", 48: "#ed7d31"}

    # (a) accuracy vs relative energy, colour=hidden, marker=M
    for row in summary:
        axes[0].errorbar(
            row["relative_energy_mean"] * 100, row["accuracy_mean"] * 100,
            xerr=row["relative_energy_std"] * 100,
            yerr=row["accuracy_std"] * 100,
            fmt=markers.get(row["M"], "o"),
            color=colors.get(row["hidden"], "#333333"),
            markersize=7, capsize=3, alpha=0.9,
            label=f"H={row['hidden']} M={row['M']} T={row['T_max']}")
    axes[0].set_xlabel("Relative energy (% of full T_max budget)")
    axes[0].set_ylabel("MNIST accuracy (%)")
    axes[0].set_title("(a) Accuracy–energy scaling")
    axes[0].grid(alpha=0.25)

    # (b) accuracy vs hidden for T_max max, by M
    tmax = max(r["T_max"] for r in summary)
    for m in sorted({r["M"] for r in summary}):
        pts = sorted(
            [r for r in summary if r["M"] == m and r["T_max"] == tmax],
            key=lambda r: r["hidden"])
        if not pts:
            continue
        axes[1].errorbar(
            [r["hidden"] for r in pts],
            [r["accuracy_mean"] * 100 for r in pts],
            yerr=[r["accuracy_std"] * 100 for r in pts],
            fmt=markers.get(m, "o") + "-", color="#4472c4" if m == 1
            else ("#c00000" if m == 3 else "#70ad47"),
            label=f"M={m}", capsize=3)
    axes[1].set_xlabel("Hidden width")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title(f"(b) Width × ensemble (T_max={tmax})")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25)

    # (c) mean passes vs T_max for H=160 or largest available, M=3 or available
    prefer_h = 160 if any(r["hidden"] == 160 for r in summary) else max(
        r["hidden"] for r in summary)
    prefer_m = 3 if any(r["M"] == 3 for r in summary) else sorted(
        {r["M"] for r in summary})[-1]
    pts = sorted(
        [r for r in summary if r["hidden"] == prefer_h and r["M"] == prefer_m],
        key=lambda r: r["T_max"])
    if pts:
        axes[2].errorbar(
            [r["T_max"] for r in pts],
            [r["mean_passes_mean"] for r in pts],
            yerr=[r["mean_passes_std"] for r in pts],
            fmt="o-", color="#ed7d31", capsize=3,
            label=f"H={prefer_h} M={prefer_m}")
        axes[2].plot([r["T_max"] for r in pts], [r["T_max"] for r in pts],
                     "--", color="#999999", label="no early-exit")
    axes[2].set_xlabel("T_max")
    axes[2].set_ylabel("Mean passes used")
    axes[2].set_title("(c) Early-exit vs budget")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.25)

    fig.suptitle(
        f"SENA scaling: hidden × ensemble × T_max ({profile})\n"
        f"Early-exit θ={THETA}; error bars = sample SD across seeds",
        fontsize=11, fontweight="bold")
    fig.tight_layout()
    for directory in (FIGS, MEDIA):
        directory.mkdir(parents=True, exist_ok=True)
        fig.savefig(directory / "q1_scaling.png", bbox_inches="tight")
        fig.savefig(directory / "q1_scaling.svg", bbox_inches="tight")
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

    for hidden in cfg["hiddens"]:
        for m in cfg["members"]:
            for seed in cfg["seeds"]:
                ckpt = RESULTS / (
                    f"q1_scaling_{args.profile}_h{hidden}_m{m}_s{seed}.npz")
                if ckpt.exists() and not args.retrain:
                    print(f"loading {ckpt.name} ...", flush=True)
                    ens, _ = SENAEnsemble.load(ckpt)
                else:
                    print(f"train H={hidden} M={m} seed={seed} ...", flush=True)
                    seeds = [1000 * seed + i for i in range(m)]
                    ens = SENAEnsemble.train_members(
                        data["Xtr"], data["Ytr"],
                        layer_sizes=(data["n_in"], hidden, 10),
                        n_members=m, epochs=cfg["epochs"], seeds=seeds,
                        samples_per_epoch=cfg["samples_per_epoch"],
                        val_every=max(1, cfg["epochs"] // 4),
                        T_probe=min(12, max(cfg["t_max_list"])),
                        verbose=True)
                    ens.save(ckpt, metadata=dict(
                        profile=args.profile, hidden=hidden, M=m, seed=seed))
                for t_max in cfg["t_max_list"]:
                    row = evaluate_cell(ens, data, hidden, m, t_max, seed)
                    rows.append(row)
                    print(
                        f"  H={hidden} M={m} T={t_max} seed={seed} "
                        f"acc={row['accuracy']*100:.1f}% "
                        f"Erel={row['relative_energy']*100:.1f}%",
                        flush=True)

    elapsed = time.perf_counter() - t0
    out_json = RESULTS / "q1_scaling.json"
    if out_json.exists():
        prior = json.loads(out_json.read_text(encoding="utf-8"))
        if prior.get("profile") == args.profile:
            keyed = {(r["hidden"], r["M"], r["T_max"], r["seed"]): r
                     for r in prior.get("runs", [])}
            for r in rows:
                keyed[(r["hidden"], r["M"], r["T_max"], r["seed"])] = r
            rows = [keyed[k] for k in sorted(keyed)]
            elapsed += float(prior.get("elapsed_s", 0.0))

    summary = aggregate(rows)
    render(summary, args.profile)
    payload = dict(
        profile=args.profile,
        config={**cfg, "dataset_seed": 2026, "threshold": 0.3,
                "theta_early_exit": THETA},
        elapsed_s=elapsed, runs=rows, summary=summary,
    )
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with open(RESULTS / "q1_scaling.csv", "w", newline="",
              encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))
    print(f"elapsed {elapsed:.1f}s  wrote {out_json}")


if __name__ == "__main__":
    main()
