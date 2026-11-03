"""
exp_mnist_sena.py
=================
Headline MNIST experiment for SENA:
  stochastic ensemble + confidence early-exit
  -> accuracy vs relative energy / mean passes plot

Outputs under sena2026/assets/figs/ and sena2026/results/:
  mnist_sena_energy_accuracy.png|.jpeg
  mnist_sena_results.json
  mnist_sena_results.csv
"""

from __future__ import annotations
import csv
import json
import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sena_ensemble import SENAEnsemble, load_mnist_binary, EnergyModel

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "assets", "figs")
RES = os.path.join(HERE, "..", "results")
os.makedirs(FIG, exist_ok=True)
os.makedirs(RES, exist_ok=True)

# Tractable but still MNIST-scale headline configuration
N_TRAIN = 10000
N_TEST = 2500
HIDDEN = 160
EPOCHS = 40
N_MEMBERS = 3
T_MAX = 40
SAMPLES_PER_EPOCH = 3000
THETAS = [0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.01]
PIXEL_THR = 0.3
SEED = 0


def render_headline(rows, fixed_acc, bp_rel_energy, meta):
    """Accuracy vs relative energy — the Q1 money plot."""
    acc = [r["acc"] * 100 for r in rows]
    rel_e = [r["relative_energy"] * 100 for r in rows]
    passes = [r["mean_passes"] for r in rows]
    th = [r["theta"] for r in rows]

    fig = plt.figure(figsize=(11.2, 4.8), dpi=180, facecolor="white")
    fig.suptitle(
        "SENA on MNIST — confidence early-exit trades energy for accuracy\n"
        f"net {meta['n_in']}→{meta['hidden']}→10 · M={meta['M']} members · "
        f"T_max={meta['T_max']} · train {meta['n_train']}/test {meta['n_test']}",
        fontsize=11, fontweight="bold", y=0.98)

    # ---- left: energy-accuracy Pareto ----
    ax = fig.add_axes([0.07, 0.14, 0.40, 0.70])
    ax.plot(rel_e, acc, "o-", color="#0284c7", lw=2.0, ms=7, zorder=3,
            label="SENA early-exit (sweep θ)")
    for x, y, t in zip(rel_e, acc, th):
        lab = "force full" if t > 1 else f"θ={t:.2f}"
        ax.annotate(lab, (x, y), textcoords="offset points",
                    xytext=(6, 4), fontsize=7, color="#334155")
    # fixed-budget reference
    ax.scatter([100], [fixed_acc * 100], marker="s", s=70, color="#64748b",
               zorder=4, label=f"Fixed budget T={meta['T_max']} (no early exit)")
    # BP energy reference (same accuracy not claimed — op-cost of 1 forward)
    ax.axvline(bp_rel_energy * 100, color="#94a3b8", ls="--", lw=1.2,
               label=f"1× real-BP forward energy ≈ {bp_rel_energy*100:.0f}% of full SENA")
    ax.set_xlabel("Relative energy (% of full SENA budget T_max×M)", fontsize=9)
    ax.set_ylabel("Test accuracy (%)", fontsize=9)
    ax.set_title("(a) Headline: accuracy vs energy", loc="left", fontsize=10)
    ax.set_xlim(0, 108)
    ax.set_ylim(max(40, min(acc) - 8), min(100, max(acc) + 5))
    ax.grid(True, color="#e2e8f0", lw=0.7)
    ax.legend(fontsize=7.2, frameon=False, loc="lower right")

    # ---- right: passes + saving ----
    ax2 = fig.add_axes([0.56, 0.14, 0.40, 0.70])
    x = np.arange(len(rows))
    bars = ax2.bar(x, passes, color="#38bdf8", width=0.55, label="Mean passes used")
    ax2.axhline(meta["T_max"], color="#64748b", ls="--", lw=1.2,
                label=f"T_max={meta['T_max']}")
    ax2.set_xticks(x)
    ax2.set_xticklabels([("full" if t > 1 else f"{t:.2f}") for t in th],
                        fontsize=7.5)
    ax2.set_xlabel("Exit confidence threshold θ", fontsize=9)
    ax2.set_ylabel("Mean stochastic passes / sample", fontsize=9)
    ax2.set_title("(b) Early-exit reduces average work", loc="left", fontsize=10)
    ax2.grid(True, axis="y", color="#e2e8f0", lw=0.7)
    ax2t = ax2.twinx()
    save = [r["energy_saving"] * 100 for r in rows]
    ax2t.plot(x, save, "s--", color="#0f766e", ms=5, lw=1.4, label="Energy saved %")
    ax2t.set_ylabel("Energy saved vs full budget (%)", fontsize=9, color="#0f766e")
    ax2t.tick_params(axis="y", labelcolor="#0f766e", labelsize=7)
    # combined legend
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2t.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, fontsize=7.2, frameon=False, loc="upper right")

    note = (
        f"Train acc@{meta['train_acc']*100:.1f}% · "
        f"Fixed-budget test acc={fixed_acc*100:.1f}% · "
        f"Best early-exit acc={max(acc):.1f}% at "
        f"{rel_e[int(np.argmax(acc))]:.0f}% energy · "
        f"Op-energy proxy: add=1, cmp=1, RNG=0.5, mul=25 (not silicon)."
    )
    fig.text(0.07, 0.02, note, fontsize=7.5, color="#475569")

    paths = []
    for ext, kw in (("png", {}), ("jpeg", {"pil_kwargs": {"quality": 95}})):
        p = os.path.join(FIG, f"mnist_sena_energy_accuracy.{ext}")
        fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white", **kw)
        paths.append(p)
    # also drop into _media for easy access
    media = os.path.join(HERE, "..", "..", "_media")
    os.makedirs(media, exist_ok=True)
    for ext, kw in (("png", {}), ("jpeg", {"pil_kwargs": {"quality": 95}})):
        p = os.path.join(media, f"mnist_sena_energy_accuracy.{ext}")
        fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white", **kw)
        paths.append(p)
    plt.close(fig)
    return paths


def main():
    print("Loading MNIST (binary pixels)...")
    t0 = time.perf_counter()
    d = load_mnist_binary(n_train=N_TRAIN, n_test=N_TEST,
                          threshold=PIXEL_THR, seed=SEED)
    print(f"  Xtr {d['Xtr'].shape}  Xte {d['Xte'].shape}  "
          f"({time.perf_counter()-t0:.1f}s)")

    print(f"Training SENA ensemble M={N_MEMBERS}, "
          f"arch={d['n_in']}-{HIDDEN}-10, epochs={EPOCHS}...")
    t0 = time.perf_counter()
    # Subsample each epoch implicitly by training on full subset (8k)
    ens = SENAEnsemble.train_members(
        d["Xtr"], d["Ytr"],
        layer_sizes=(d["n_in"], HIDDEN, 10),
        n_members=N_MEMBERS, epochs=EPOCHS,
        seeds=list(range(N_MEMBERS)),
        samples_per_epoch=SAMPLES_PER_EPOCH,
        verbose=True)
    print(f"  train wall {time.perf_counter()-t0:.1f}s")

    # train accuracy of member 0 (quick probe)
    train_acc = ens.nets[0].accuracy(d["Xtr"][:2000], d["ytr"][:2000], T=16)
    print(f"  member-0 train probe acc={train_acc*100:.1f}%")

    print(f"Sweeping early-exit θ over {THETAS} (T_max={T_MAX})...")
    t0 = time.perf_counter()
    rows = ens.sweep_theta(d["Xte"], d["yte"], THETAS, T_max=T_MAX)
    print(f"  sweep wall {time.perf_counter()-t0:.1f}s")
    for r in rows:
        print(f"  θ={r['theta']:.2f}  acc={r['acc']*100:5.1f}%  "
              f"passes={r['mean_passes']:5.2f}  "
              f"E_rel={r['relative_energy']*100:5.1f}%  "
              f"saved={r['energy_saving']*100:5.1f}%")

    fixed = next(r for r in rows if r["theta"] > 1.0)
    em = ens.energy
    # relative energy of ONE real-BP forward vs full SENA budget
    bp_rel = em.e_forward_bp / (T_MAX * N_MEMBERS * em.e_forward_stoch)

    meta = dict(
        n_in=d["n_in"], hidden=HIDDEN, M=N_MEMBERS, T_max=T_MAX,
        n_train=N_TRAIN, n_test=N_TEST, epochs=EPOCHS,
        pixel_threshold=PIXEL_THR, train_acc=train_acc,
        e_forward_stoch=em.e_forward_stoch,
        e_forward_bp=em.e_forward_bp,
        stoch_adds=em.stoch_adds, bp_muls=em.bp_muls,
    )
    paths = render_headline(rows, fixed["acc"], bp_rel, meta)
    print("Wrote figures:")
    for p in paths:
        print(" ", p)

    payload = dict(meta=meta, sweep=rows, bp_rel_energy=bp_rel,
                   fixed_budget_acc=fixed["acc"])
    ckpt = os.path.join(RES, "mnist_sena_checkpoint.npz")
    ens.save(ckpt, metadata=meta)
    print("Wrote", ckpt)
    jpath = os.path.join(RES, "mnist_sena_results.json")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    cpath = os.path.join(RES, "mnist_sena_results.csv")
    with open(cpath, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("Wrote", jpath)
    print("Wrote", cpath)
    return payload


if __name__ == "__main__":
    main()
