"""
fig_table2_symmetry.py
======================
Regenerate the original paper's Table II (extracted as image4.jpeg) as a
clean publication figure, and add a stronger 2026 companion panel built from
the reproducible digital + BMP experiments.

Outputs:
  sena2026/assets/figs/table2_symmetry.png|.jpeg
  _media/image4_regen.jpeg          -- drop-in improved replacement
  _media/image4_regen.png
"""

from __future__ import annotations
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ---- original Table II numbers (Mazumdar 1995 / image4.jpeg) --------------
PAPER = dict(
    exp=list(range(1, 14)),
    bp_t=[740, 1133, 699, 311, 740, 1232, 288, 392, 900, 196, 411, 251, 178],
    bp_l=[283, 433, 267, 119, 283, 471, 110, 150, 344, 75, 157, 96, 68],
    st_t=[440, 321, 522, 144, 1056, 369, 317, 1074, 261, 232, 243, 165, 282],
    st_l=[115, 113, 128, 51, 367, 129, 116, 374, 93, 83, 87, 59, 101],
)
PAPER_MEAN = dict(bp_t=574, bp_l=219, st_t=417, st_l=139)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "..", "assets", "figs")
MEDIA = os.path.join(HERE, "..", "..", "_media")
DIGITAL_JSON = os.path.join(HERE, "..", "results", "digital.json")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MEDIA, exist_ok=True)


def load_digital():
    if not os.path.exists(DIGITAL_JSON):
        return None
    with open(DIGITAL_JSON, encoding="utf-8") as f:
        return json.load(f)


def style_table(tbl, n_data_rows, header_bg="#1e293b"):
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1.15, 1.35)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#cbd5e1")
        cell.set_linewidth(0.55)
        cell.PAD = 0.04
        if row == 0:
            cell.set_facecolor(header_bg)
            cell.set_text_props(color="white", fontweight="bold", fontsize=8)
        elif row == n_data_rows:          # mean
            cell.set_facecolor("#e2e8f0")
            cell.set_text_props(fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f8fafc")
        else:
            cell.set_facecolor("white")


def render():
    dig = load_digital()

    fig = plt.figure(figsize=(12.2, 8.6), dpi=180, facecolor="#fafafa")
    fig.suptitle(
        "Stochastic vs real back-propagation — regenerated results figure\n"
        "Left: faithful recreation of Mazumdar 1995 Table II (source: image4.jpeg)  ·  "
        "Right: 2026 re-implementation benchmarks",
        fontsize=12, fontweight="bold", color="#0f172a", y=0.975)

    # ================================================================== #
    #  (a) Clean recreation of original Table II
    # ================================================================== #
    axA = fig.add_axes([0.04, 0.38, 0.42, 0.52])
    axA.axis("off")
    axA.set_title("(a)  Table II — Mirror symmetry (original 1995 results)",
                  loc="left", fontsize=10.5, color="#0f172a", pad=8)

    cols = ["Exp no.", "Real B.P.\nTime (s)", "Real B.P.\nLoops",
            "Stochastic\nTime (s)", "Stochastic\nLoops"]
    cells = []
    for i in range(13):
        cells.append([
            str(PAPER["exp"][i]),
            str(PAPER["bp_t"][i]), str(PAPER["bp_l"][i]),
            str(PAPER["st_t"][i]), str(PAPER["st_l"][i]),
        ])
    cells.append([
        "Mean",
        str(PAPER_MEAN["bp_t"]), str(PAPER_MEAN["bp_l"]),
        str(PAPER_MEAN["st_t"]), str(PAPER_MEAN["st_l"]),
    ])
    tbl = axA.table(cellText=cells, colLabels=cols,
                    loc="upper center", cellLoc="center")
    style_table(tbl, n_data_rows=len(cells))

    # highlight the mean-row advantage
    speedup_L = PAPER_MEAN["bp_l"] / PAPER_MEAN["st_l"]
    speedup_T = PAPER_MEAN["bp_t"] / PAPER_MEAN["st_t"]
    axA.text(
        0.0, -0.04,
        f"Stochastic needed {speedup_L:.2f}× fewer loops and "
        f"{speedup_T:.2f}× less time on average.\n"
        "Task: 8-bit mirror symmetry · architecture as in Mazumdar 1995 · "
        "scanned source cleaned & typeset.",
        transform=axA.transAxes, fontsize=7.5, color="#475569", va="top")

    # ================================================================== #
    #  (b) Per-run loops scatter (original)
    # ================================================================== #
    axB = fig.add_axes([0.52, 0.58, 0.44, 0.30])
    xs = np.array(PAPER["exp"])
    axB.plot(xs, PAPER["bp_l"], "o-", color="#64748b", ms=5, lw=1.4,
             label=f"Real B.P.  (mean {PAPER_MEAN['bp_l']})")
    axB.plot(xs, PAPER["st_l"], "s-", color="#0284c7", ms=5, lw=1.4,
             label=f"Stochastic (mean {PAPER_MEAN['st_l']})")
    axB.fill_between(xs, PAPER["bp_l"], PAPER["st_l"],
                     where=np.array(PAPER["bp_l"]) >= np.array(PAPER["st_l"]),
                     color="#38bdf8", alpha=0.15, interpolate=True)
    axB.set_xlabel("Experiment number", fontsize=8)
    axB.set_ylabel("Loops to convergence", fontsize=8)
    axB.set_title("(b)  1995 loops per run — stochastic usually wins",
                  loc="left", fontsize=10)
    axB.legend(fontsize=8, frameon=False)
    axB.set_xticks(xs)
    axB.tick_params(labelsize=7)
    axB.grid(True, color="#e2e8f0", lw=0.7)
    axB.set_facecolor("white")
    for sp in axB.spines.values():
        sp.set_color("#cbd5e1")

    # ================================================================== #
    #  (c) Mean bars: time + loops (original)
    # ================================================================== #
    axC = fig.add_axes([0.52, 0.38, 0.20, 0.15])
    labs = ["Time (s)", "Loops"]
    x = np.arange(2)
    w = 0.35
    bp = [PAPER_MEAN["bp_t"], PAPER_MEAN["bp_l"]]
    st = [PAPER_MEAN["st_t"], PAPER_MEAN["st_l"]]
    axC.bar(x - w / 2, bp, w, color="#94a3b8", label="Real B.P.")
    axC.bar(x + w / 2, st, w, color="#38bdf8", label="Stochastic")
    for i, (a, b) in enumerate(zip(bp, st)):
        axC.text(i - w / 2, a + 12, str(a), ha="center", fontsize=7)
        axC.text(i + w / 2, b + 12, str(b), ha="center", fontsize=7)
    axC.set_xticks(x)
    axC.set_xticklabels(labs, fontsize=8)
    axC.set_title("(c)  1995 means", loc="left", fontsize=9)
    axC.legend(fontsize=7, frameon=False, loc="upper right")
    axC.tick_params(labelsize=7)
    axC.set_facecolor("white")
    axC.grid(True, axis="y", color="#e2e8f0", lw=0.6)
    for sp in axC.spines.values():
        sp.set_color("#cbd5e1")

    # ================================================================== #
    #  (d) 2026 accuracy benchmarks (better / extended)
    # ================================================================== #
    axD = fig.add_axes([0.76, 0.38, 0.20, 0.15])
    if dig and "benchmark" in dig:
        order = [k for k in ("iris", "cancer", "digits", "xor2d", "symmetry")
                 if k in dig["benchmark"]]
        names = [k.capitalize() for k in order]
        st_m = [dig["benchmark"][k]["stoch_mean"] * 100 for k in order]
        bp_m = [dig["benchmark"][k]["bp_mean"] * 100 for k in order]
        x = np.arange(len(order))
        w = 0.35
        axD.bar(x - w / 2, bp_m, w, color="#94a3b8", label="Real BP")
        axD.bar(x + w / 2, st_m, w, color="#38bdf8", label="Stochastic")
        axD.set_xticks(x)
        axD.set_xticklabels(names, fontsize=6.5, rotation=20, ha="right")
        axD.set_ylim(0, 105)
        axD.set_ylabel("Test acc %", fontsize=7)
        axD.set_title("(d)  2026 accuracy", loc="left", fontsize=9)
        axD.legend(fontsize=6.5, frameon=False)
        axD.tick_params(labelsize=6.5)
    else:
        axD.axis("off")
        axD.text(0.5, 0.5, "digital.json not found", ha="center")
    axD.set_facecolor("white")
    axD.grid(True, axis="y", color="#e2e8f0", lw=0.6)
    for sp in axD.spines.values():
        sp.set_color("#cbd5e1")

    # ================================================================== #
    #  (e) Bottom: BMP live-learn + ops story (the "better" extension)
    # ================================================================== #
    axE = fig.add_axes([0.04, 0.06, 0.58, 0.26])
    axE.set_facecolor("white")
    # BMP shape accuracies from the verified Mazumdar live-learn runs
    shapes = ["Circle", "Square", "Triangle"]
    maz = [96.3, 95.8, 88.1]
    mlp = [97.6, 95.3, 95.6]
    x = np.arange(len(shapes))
    w = 0.32
    axE.bar(x - w / 2, mlp, w, color="#94a3b8",
            label="Float MLP (uses × ÷ exp)")
    axE.bar(x + w / 2, maz, w, color="#0ea5e9",
            label="Mazumdar 1995 (ADD/SUB/CMP/RNG only)")
    for i, (a, b) in enumerate(zip(mlp, maz)):
        axE.text(i - w / 2, a + 0.6, f"{a:.1f}", ha="center", fontsize=7)
        axE.text(i + w / 2, b + 0.6, f"{b:.1f}", ha="center", fontsize=7)
    axE.set_xticks(x)
    axE.set_xticklabels(shapes, fontsize=9)
    axE.set_ylim(70, 105)
    axE.set_ylabel("Full-bitmap accuracy (%)", fontsize=8)
    axE.set_title(
        "(e)  2026 extension — live pixel learning from balanced 64×64 BMPs "
        "(cir / sqr / trngl, 2048 white = 2048 black)",
        loc="left", fontsize=10)
    axE.legend(fontsize=7.5, frameon=False, loc="lower right")
    axE.tick_params(labelsize=7)
    axE.grid(True, axis="y", color="#e2e8f0", lw=0.6)
    for sp in axE.spines.values():
        sp.set_color("#cbd5e1")

    # callout box
    axF = fig.add_axes([0.66, 0.06, 0.30, 0.26])
    axF.axis("off")
    axF.set_xlim(0, 1)
    axF.set_ylim(0, 1)
    box = FancyBboxPatch((0.02, 0.05), 0.96, 0.90,
                         boxstyle="round,pad=0.03,rounding_size=0.05",
                         facecolor="#eff6ff", edgecolor="#93c5fd", lw=1.2)
    axF.add_patch(box)
    axF.text(0.5, 0.88, "Why this figure is better", ha="center",
             fontsize=9, fontweight="bold", color="#0c4a6e")
    axF.text(
        0.08, 0.72,
        "• Typeset Table II (no scan noise)\n"
        "• Per-run loops chart + mean bars\n"
        "• 2026 multi-dataset accuracy panel\n"
        "• New BMP shape task (paper section)\n"
        "• Same 1995 rule: no × ÷ exp in datapath\n"
        "• Balanced 64×64 bitmaps, live probes\n"
        "• CSV / PNG / JPEG reproducible outputs",
        ha="left", va="top", fontsize=7.8, color="#075985",
        linespacing=1.45)

    # footer
    fig.text(
        0.04, 0.012,
        "Sources: Stochastic_NN_VLSI.docx Table II (image4.jpeg) · "
        "sena2026/results/digital.json · sena2026/assets/figs/live_mazumdar_*.png",
        fontsize=7, color="#64748b")

    png = os.path.join(OUT_DIR, "table2_symmetry.png")
    jpg = os.path.join(OUT_DIR, "table2_symmetry.jpeg")
    png_m = os.path.join(MEDIA, "image4_regen.png")
    jpg_m = os.path.join(MEDIA, "image4_regen.jpeg")
    for path in (png, png_m):
        fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    for path in (jpg, jpg_m):
        fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor(),
                    pil_kwargs={"quality": 95})
    plt.close(fig)
    return png, jpg, png_m, jpg_m


if __name__ == "__main__":
    paths = render()
    print("Wrote improved image4 replacement:")
    for p in paths:
        print(" ", p)
