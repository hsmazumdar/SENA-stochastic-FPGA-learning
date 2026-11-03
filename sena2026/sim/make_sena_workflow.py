"""
make_sena_workflow.py
=====================
Detailed proposed-system workflow diagram for SENA as SVG + PNG.
"""

from __future__ import annotations
import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "assets", "figs")
MEDIA = os.path.join(HERE, "..", "..", "_media")
os.makedirs(FIG, exist_ok=True)
os.makedirs(MEDIA, exist_ok=True)


def box(ax, xy, w, h, text, fc="#f8fafc", ec="#334155", title=None,
        fontsize=7.5, title_size=8.5):
    x, y = xy
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle="round,pad=0.012,rounding_size=0.02",
                       facecolor=fc, edgecolor=ec, lw=1.2, zorder=2)
    ax.add_patch(p)
    if title:
        ax.text(x + w / 2, y + h - 0.045, title, ha="center", va="top",
                fontsize=title_size, fontweight="bold", color="#0f172a",
                zorder=3)
        ax.text(x + w / 2, y + h / 2 - 0.02, text, ha="center", va="center",
                fontsize=fontsize, color="#334155", zorder=3,
                linespacing=1.25)
    else:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fontsize, color="#0f172a", zorder=3,
                linespacing=1.25, fontweight="bold")
    return p


def arrow(ax, p0, p1, color="#475569"):
    ax.annotate("", xy=p1, xytext=p0,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4,
                                mutation_scale=12),
                zorder=1)


def render():
    fig = plt.figure(figsize=(13.5, 9.0), dpi=180, facecolor="white")
    ax = fig.add_axes([0.02, 0.03, 0.96, 0.94])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.97,
            "SENA — Proposed System Workflow",
            ha="center", va="top", fontsize=15, fontweight="bold",
            color="#0f172a")
    ax.text(0.5, 0.935,
            "Multiplier-free stochastic on-chip learning · statistical A/D neurons · "
            "confidence-gated ensemble early exit",
            ha="center", va="top", fontsize=8.5, color="#475569")

    # ========== Row 1: sensing / encoding ==========
    box(ax, (0.03, 0.78), 0.18, 0.12,
        "analog / pixel\nintensity X",
        fc="#eff6ff", ec="#0284c7", title="(1) Sensor / Input")
    box(ax, (0.25, 0.78), 0.22, 0.12,
        "compare X >= ref_k\nor X >= pixel threshold\n-> binary / thermometer bits",
        fc="#eff6ff", ec="#0284c7", title="(2) Statistical front-end")
    box(ax, (0.51, 0.78), 0.20, 0.12,
        "1-bit bus\ninter-layer / tile links\nlow routing cost",
        fc="#f0fdf4", ec="#15803d", title="(3) Binary fabric")
    box(ax, (0.75, 0.78), 0.21, 0.12,
        "M learner tiles\nindependent seeds\nparallel on die",
        fc="#f0fdf4", ec="#15803d", title="(4) Ensemble tiles")

    arrow(ax, (0.21, 0.84), (0.25, 0.84))
    arrow(ax, (0.47, 0.84), (0.51, 0.84))
    arrow(ax, (0.71, 0.84), (0.75, 0.84))

    # ========== Row 2: neuron core ==========
    ax.text(0.5, 0.74, "Neuron datapath  (identical in every tile)",
            ha="center", fontsize=9, fontweight="bold", color="#0f172a")

    box(ax, (0.05, 0.56), 0.20, 0.14,
        "integer accumulate\nsum w_i * b_i (AND+ADD)\nbias included",
        fc="#fff7ed", ec="#c2410c", title="(5) Pre-activation X")
    box(ax, (0.30, 0.56), 0.24, 0.14,
        "Y = 1[ X > (K + R) ]\nR ~ RNG / thermal noise\n-> Bernoulli bit",
        fc="#fef3c7", ec="#b45309", title="(6) Random-threshold = statistical ADC")
    box(ax, (0.59, 0.56), 0.17, 0.14,
        "forward: fire bit\nreverse: |X|<=r gate\n(learn enable)",
        fc="#fff7ed", ec="#c2410c", title="(7) Forward / reverse")
    box(ax, (0.80, 0.56), 0.16, 0.14,
        "dw in {-1,0,+1}\nno mult/div/exp\nweight register",
        fc="#fff7ed", ec="#c2410c", title="(8) Integer update")

    arrow(ax, (0.25, 0.63), (0.30, 0.63))
    arrow(ax, (0.54, 0.63), (0.59, 0.63))
    arrow(ax, (0.76, 0.63), (0.80, 0.63))
    # feedback from update to weights (curved note)
    ax.annotate("on-chip learning",
                xy=(0.15, 0.56), xytext=(0.88, 0.52),
                fontsize=7, color="#9a3412",
                arrowprops=dict(arrowstyle="-|>", color="#fdba74",
                                connectionstyle="arc3,rad=-0.25", lw=1.1))

    # ========== Row 3: inference / early exit ==========
    ax.text(0.5, 0.51, "Inference: temporal + ensemble voting with early exit",
            ha="center", fontsize=9, fontweight="bold", color="#0f172a")

    box(ax, (0.04, 0.30), 0.18, 0.16,
        "pass t = 1 .. T_max\neach tile fires once\nper pass",
        fc="#f5f3ff", ec="#6d28d9", title="(9) Stochastic passes")
    box(ax, (0.26, 0.30), 0.20, 0.16,
        "vote counter\nV_c += Y_c\nacross M tiles x t",
        fc="#f5f3ff", ec="#6d28d9", title="(10) Vote accumulate")
    box(ax, (0.50, 0.30), 0.22, 0.16,
        "confidence =\nmax(V) / sum(V)\nif conf >= theta -> EXIT",
        fc="#ede9fe", ec="#5b21b6", title="(11) Confidence gate (early exit)")
    box(ax, (0.76, 0.30), 0.20, 0.16,
        "yhat = argmax V\neasy inputs: few passes\nhard inputs: up to T_max",
        fc="#f5f3ff", ec="#6d28d9", title="(12) Decision / energy ~ work")

    arrow(ax, (0.22, 0.38), (0.26, 0.38))
    arrow(ax, (0.46, 0.38), (0.50, 0.38))
    arrow(ax, (0.72, 0.38), (0.76, 0.38))
    # loop back for more passes
    ax.annotate("conf < θ: next pass",
                xy=(0.13, 0.30), xytext=(0.58, 0.27),
                fontsize=7, color="#5b21b6",
                arrowprops=dict(arrowstyle="-|>", color="#c4b5fd",
                                connectionstyle="arc3,rad=0.35", lw=1.1))

    # ========== Row 4: claims strip ==========
    box(ax, (0.04, 0.06), 0.29, 0.16,
        "* no multiplier / divider / exp\n"
        "* 1-bit communication\n"
        "* massive parallel cheap tiles\n"
        "* same primitive for learn + infer",
        fc="#ecfdf5", ec="#047857", title="Hardware thesis")
    box(ax, (0.36, 0.06), 0.30, 0.16,
        "* X>(K+R) = statistical A/D\n"
        "* soft value = firing rate / votes\n"
        "* analog sensors without flash ADC\n"
        "* noise is a computational resource",
        fc="#eff6ff", ec="#1d4ed8", title="Statistical A/D thesis")
    box(ax, (0.69, 0.06), 0.27, 0.16,
        "* energy ~ mean passes used\n"
        "* early exit on high confidence\n"
        "* ensemble disagreement = hard case\n"
        "* headline: accuracy-energy curve",
        fc="#fdf4ff", ec="#7e22ce", title="Early-exit ensemble thesis")

    # side legend
    ax.text(0.5, 0.015,
            "Blue: I/O encoding · Green: fabric · Orange: learning neuron · "
            "Purple: inference / early exit",
            ha="center", fontsize=7.5, color="#64748b")

    paths = []
    for folder in (FIG, MEDIA):
        svg = os.path.join(folder, "sena_system_workflow.svg")
        png = os.path.join(folder, "sena_system_workflow.png")
        fig.savefig(svg, format="svg", bbox_inches="tight",
                    facecolor="white")
        fig.savefig(png, dpi=220, bbox_inches="tight", facecolor="white")
        paths.extend([svg, png])
    plt.close(fig)
    return paths


if __name__ == "__main__":
    for p in render():
        print(p)
