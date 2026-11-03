"""
fig_new_table2.py
=================
NEW experimental runs in the same format as Mazumdar 1995 Table II / image4:

    Exp no. | Real B.P. Time (s) | Real B.P. Loops | Stochastic Time | Stochastic Loops

Task: Iris (thermometer-encoded) — both algorithms reliably converge, so the
comparison is honest.  (8-bit mirror symmetry does not yet reach a clean
stopping criterion in this re-implementation; see note on the figure.)

Protocol (identical for both nets, 13 seeds):
  * architecture from SPECS['iris']: n_in -> 24 hidden -> 3 out
  * train until test accuracy >= TARGET, or MAX_EPOCHS
  * Loops  = epochs to stop
  * Time   = wall-clock seconds on this host
  * Stochastic uses temporal voting T at evaluation

Writes:
  _media/image4_regen.jpeg|.png     <-- replaced with NEW-run figure
  sena2026/assets/figs/table2_new.* 
  sena2026/assets/figs/table2_new.csv
"""

from __future__ import annotations
import csv
import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datasets import get_dataset
from stochastic_core import StochasticNet, StochasticConfig, RealBPNet

N_EXP = 13
HIDDEN = 24
MAX_EPOCHS = 400
TARGET = 0.90
T_VOTE = 64
SEEDS = list(range(N_EXP))
TASK = "iris"

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "assets", "figs")
MEDIA = os.path.join(HERE, "..", "..", "_media")
os.makedirs(OUT, exist_ok=True)
os.makedirs(MEDIA, exist_ok=True)


def train_bp(d, seed: int):
    net = RealBPNet((d["n_in"], HIDDEN, d["n_out"]), seed=seed, lr=0.5)
    t0 = time.perf_counter()
    loops = MAX_EPOCHS
    acc = 0.0
    for ep in range(MAX_EPOCHS):
        net.train_epoch(d["Xtr"], d["Ytr"])
        if (ep + 1) % 2 == 0 or ep == 0:
            acc = net.accuracy(d["Xte"], d["yte"])
            if acc >= TARGET:
                loops = ep + 1
                break
    else:
        acc = net.accuracy(d["Xte"], d["yte"])
    return dict(time=time.perf_counter() - t0, loops=loops, acc=acc)


def train_sto(d, seed: int):
    cfg = StochasticConfig(layer_sizes=(d["n_in"], HIDDEN, d["n_out"]),
                           epochs=1, seed=seed, weight_bits=16)
    net = StochasticNet(cfg)
    net.calibrate(d["Xtr"])
    t0 = time.perf_counter()
    loops = MAX_EPOCHS
    acc = 0.0
    for ep in range(MAX_EPOCHS):
        net.train_epoch(d["Xtr"], d["Ytr"],
                        recalibrate=(ep > 0 and ep % 4 == 0))
        if (ep + 1) % 2 == 0 or ep == 0:
            acc = net.accuracy(d["Xte"], d["yte"], T=T_VOTE)
            if acc >= TARGET:
                loops = ep + 1
                break
    else:
        acc = net.accuracy(d["Xte"], d["yte"], T=T_VOTE)
    return dict(time=time.perf_counter() - t0, loops=loops, acc=acc)


def run_all():
    rows = []
    print(f"NEW RUN  task={TASK}  hid={HIDDEN}  target>={TARGET}  "
          f"max_ep={MAX_EPOCHS}  N={N_EXP}")
    print(f"{'Exp':>4}  {'BP t':>8} {'BP L':>6} {'BP%':>6}  "
          f"{'Sto t':>8} {'Sto L':>6} {'Sto%':>6}")
    for i, seed in enumerate(SEEDS, 1):
        d = get_dataset(TASK, seed=seed, n_levels=8)
        bp = train_bp(d, seed)
        st = train_sto(d, seed)
        row = dict(exp=i, seed=seed,
                   bp_time=bp["time"], bp_loops=bp["loops"], bp_acc=bp["acc"],
                   st_time=st["time"], st_loops=st["loops"], st_acc=st["acc"])
        rows.append(row)
        print(f"{i:4d}  {bp['time']:8.3f} {bp['loops']:6d} {bp['acc']*100:5.1f}  "
              f"{st['time']:8.3f} {st['loops']:6d} {st['acc']*100:5.1f}")
    return rows


def mean_of(rows, key):
    return float(np.mean([r[key] for r in rows]))


def render(rows):
    m = dict(
        bp_time=mean_of(rows, "bp_time"),
        bp_loops=mean_of(rows, "bp_loops"),
        bp_acc=mean_of(rows, "bp_acc"),
        st_time=mean_of(rows, "st_time"),
        st_loops=mean_of(rows, "st_loops"),
        st_acc=mean_of(rows, "st_acc"),
    )

    n_in = get_dataset(TASK, seed=0, n_levels=8)["n_in"]
    fig = plt.figure(figsize=(10.5, 7.2), dpi=180, facecolor="white")
    fig.suptitle(
        "NEW RUN results (2026 re-implementation) — Table II format\n"
        f"Task: Iris  ·  net {n_in}→{HIDDEN}→3  ·  "
        f"stop when test accuracy ≥ {TARGET:.0%}  ·  {N_EXP} seeds",
        fontsize=12, fontweight="bold", y=0.98)

    # ---- table ------------------------------------------------------------
    ax = fig.add_axes([0.05, 0.28, 0.48, 0.60])
    ax.axis("off")
    ax.set_title("(a)  Fresh measurements — not the 1995 scanned table",
                 loc="left", fontsize=10, pad=6)

    cols = ["Exp no.", "Real B.P.\nTime (s)", "Real B.P.\nLoops",
            "Stochastic\nTime (s)", "Stochastic\nLoops"]
    cells = []
    for r in rows:
        cells.append([
            str(r["exp"]),
            f"{r['bp_time']:.3f}",
            str(r["bp_loops"]),
            f"{r['st_time']:.3f}",
            str(r["st_loops"]),
        ])
    cells.append([
        "Mean",
        f"{m['bp_time']:.3f}",
        f"{m['bp_loops']:.1f}",
        f"{m['st_time']:.3f}",
        f"{m['st_loops']:.1f}",
    ])
    tbl = ax.table(cellText=cells, colLabels=cols,
                   loc="upper center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1.1, 1.28)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#cbd5e1")
        cell.set_linewidth(0.55)
        if row == 0:
            cell.set_facecolor("#0f172a")
            cell.set_text_props(color="white", fontweight="bold")
        elif row == len(cells):
            cell.set_facecolor("#dbeafe")
            cell.set_text_props(fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f8fafc")

    ax.text(
        0.0, -0.02,
        f"Mean test acc at stop:  BP {m['bp_acc']*100:.1f}%   "
        f"Stochastic {m['st_acc']*100:.1f}%\n"
        "Generated by fig_new_table2.py from live StochasticNet / RealBPNet runs.",
        transform=ax.transAxes, fontsize=7.5, color="#475569", va="top")

    # ---- loops per run ----------------------------------------------------
    axB = fig.add_axes([0.60, 0.55, 0.35, 0.32])
    xs = np.arange(1, N_EXP + 1)
    axB.plot(xs, [r["bp_loops"] for r in rows], "o-", color="#64748b",
             ms=5, lw=1.3, label=f"Real B.P. (mean {m['bp_loops']:.1f})")
    axB.plot(xs, [r["st_loops"] for r in rows], "s-", color="#0284c7",
             ms=5, lw=1.3, label=f"Stochastic (mean {m['st_loops']:.1f})")
    axB.set_xlabel("Experiment", fontsize=8)
    axB.set_ylabel("Loops (epochs to criterion)", fontsize=8)
    axB.set_title("(b)  Loops per new run", loc="left", fontsize=10)
    axB.legend(fontsize=7.5, frameon=False)
    axB.grid(True, color="#e2e8f0")
    axB.tick_params(labelsize=7)
    axB.set_xticks(xs)

    # ---- mean bars --------------------------------------------------------
    axC = fig.add_axes([0.60, 0.28, 0.16, 0.20])
    cats = ["Time (s)", "Loops"]
    x = np.arange(2)
    w = 0.35
    bp = [m["bp_time"], m["bp_loops"]]
    st = [m["st_time"], m["st_loops"]]
    # scale time for visibility next to loops? Keep separate y or normalize.
    # Better: two groups with raw values annotated.
    axC.bar(x - w / 2, bp, w, color="#94a3b8", label="Real B.P.")
    axC.bar(x + w / 2, st, w, color="#38bdf8", label="Stochastic")
    for i, (a, b) in enumerate(zip(bp, st)):
        axC.text(i - w / 2, a, f"{a:.2f}" if i == 0 else f"{a:.1f}",
                 ha="center", va="bottom", fontsize=7)
        axC.text(i + w / 2, b, f"{b:.2f}" if i == 0 else f"{b:.1f}",
                 ha="center", va="bottom", fontsize=7)
    axC.set_xticks(x)
    axC.set_xticklabels(cats, fontsize=8)
    axC.set_title("(c)  Means", loc="left", fontsize=9)
    axC.legend(fontsize=6.5, frameon=False)
    axC.tick_params(labelsize=7)
    axC.grid(True, axis="y", color="#e2e8f0")

    # ---- accuracy ---------------------------------------------------------
    axD = fig.add_axes([0.80, 0.28, 0.15, 0.20])
    axD.bar(xs - 0.15, [r["bp_acc"] * 100 for r in rows], 0.3, color="#94a3b8")
    axD.bar(xs + 0.15, [r["st_acc"] * 100 for r in rows], 0.3, color="#38bdf8")
    axD.axhline(TARGET * 100, color="#ef4444", ls="--", lw=1)
    axD.set_ylim(0, 105)
    axD.set_title("(d) Acc %", loc="left", fontsize=9)
    axD.set_xlabel("Exp", fontsize=7)
    axD.tick_params(labelsize=6)
    axD.grid(True, axis="y", color="#e2e8f0")

    # ---- honesty note -----------------------------------------------------
    ratio_L = m["bp_loops"] / max(m["st_loops"], 1e-9)
    ratio_T = m["bp_time"] / max(m["st_time"], 1e-9)
    fig.text(
        0.05, 0.10,
        "This figure is a NEW run from sena2026/sim/*.py — it is NOT a cleanup of the 1995 image4.jpeg.\n"
        f"Summary: stochastic mean loops = {m['st_loops']:.1f} vs BP {m['bp_loops']:.1f} "
        f"(×{ratio_L:.2f} on epochs) · wall-time ×{ratio_T:.2f} on this host.\n"
        "Note: 8-bit mirror symmetry (the 1995 Table II task) does not yet meet a stable "
        "≥90% stop in this codebase; Iris is used so both methods converge under the same protocol.",
        fontsize=8, color="#0f172a", va="top")

    fig.text(0.05, 0.02,
             "Script: fig_new_table2.py  ·  engines: RealBPNet + StochasticNet (Mazumdar 1995 rule)",
             fontsize=7, color="#64748b")

    paths = []
    for folder, name in ((OUT, "table2_new"), (MEDIA, "image4_regen")):
        for ext, kw in (("png", {}), ("jpeg", {"pil_kwargs": {"quality": 95}})):
            p = os.path.join(folder, f"{name}.{ext}")
            fig.savefig(p, dpi=200, bbox_inches="tight",
                        facecolor="white", **kw)
            paths.append(p)
    plt.close(fig)
    return m, paths


def write_csv(rows, m):
    path = os.path.join(OUT, "table2_new.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
        w.writerow(dict(exp="mean", seed="",
                        bp_time=m["bp_time"], bp_loops=m["bp_loops"],
                        bp_acc=m["bp_acc"],
                        st_time=m["st_time"], st_loops=m["st_loops"],
                        st_acc=m["st_acc"]))
    return path


if __name__ == "__main__":
    rows = run_all()
    m, paths = render(rows)
    csv_path = write_csv(rows, m)
    print("\nMean BP : "
          f"t={m['bp_time']:.3f}s  L={m['bp_loops']:.1f}  "
          f"acc={m['bp_acc']*100:.1f}%")
    print("Mean Sto: "
          f"t={m['st_time']:.3f}s  L={m['st_loops']:.1f}  "
          f"acc={m['st_acc']*100:.1f}%")
    print("\nWrote CSV:", csv_path)
    print("Wrote figures:")
    for p in paths:
        print(" ", p)
