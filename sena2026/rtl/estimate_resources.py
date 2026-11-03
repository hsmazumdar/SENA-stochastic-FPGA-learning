"""
estimate_resources.py
=====================
Structural gate / area / energy estimate for the SENA RTL datapath.

This is NOT a place-and-route result (yosys/vivado not present on this host).
It counts the arithmetic operators implied by stochastic_neuron.v /
sena_layer.v / sena_early_exit.v and maps them to:

  * NAND2-equivalent gate count (textbook CMOS ratios)
  * relative energy per forward pass
  * MNIST early-exit energy curve using measured mean_passes from
    results/mnist_sena_results.json

Outputs:
  ../assets/figs/sena_rtl_resource_estimate.png|.svg
  ../results/sena_rtl_estimate.json
  ../../_media/sena_rtl_resource_estimate.png
"""

from __future__ import annotations
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "assets", "figs")
RES = os.path.join(HERE, "..", "results")
MEDIA = os.path.join(HERE, "..", "..", "_media")
os.makedirs(FIG, exist_ok=True)
os.makedirs(RES, exist_ok=True)
os.makedirs(MEDIA, exist_ok=True)

# ---- CMOS NAND2-equivalent factors (order-of-magnitude, educational) ----
G_FA = 28          # 1-bit full adder
G_REG = 8          # 1-bit flip-flop
G_CMP1 = 12        # 1-bit magnitude compare slice
G_LFSR = 32 * G_REG + 40   # 32-bit LFSR + XOR taps
G_MUX2 = 4

# Energy relative to one 16-bit add
E_ADD16 = 1.0
E_CMP16 = 1.0
E_LFSR = 0.5
E_MUL16 = 25.0


def neuron_gates(n_in: int, w_bits: int = 16, acc_bits: int | None = None):
    """Gates for one stochastic_neuron (serial accumulate over n_in cycles)."""
    acc_bits = acc_bits or (w_bits + 10)
    # serial: one w_bits adder reused each cycle + acc register + compare + LFSR
    g_add = acc_bits * G_FA
    g_acc_reg = acc_bits * G_REG
    g_idx = int(np.ceil(np.log2(n_in + 1))) * G_REG
    g_cmp = acc_bits * G_CMP1
    g_lfsr = G_LFSR
    g_ctrl = 40
    total = g_add + g_acc_reg + g_idx + g_cmp + g_lfsr + g_ctrl
    return dict(
        n_in=n_in, w_bits=w_bits, acc_bits=acc_bits,
        g_add=g_add, g_acc_reg=g_acc_reg, g_cmp=g_cmp, g_lfsr=g_lfsr,
        g_ctrl=g_ctrl + g_idx, total=total,
        # one presentation energy (n_in serial adds + 1 cmp + n_in LFSR steps)
        e_pass=n_in * E_ADD16 + E_CMP16 + n_in * E_LFSR,
    )


def layer_gates(n_in: int, n_out: int, w_bits: int = 16):
    n = neuron_gates(n_in, w_bits)
    # weight SRAM not counted as logic; report storage bits separately
    return dict(
        n_in=n_in, n_out=n_out,
        neuron=n,
        total_logic=n["total"] * n_out,
        weight_bits=n_out * (n_in + 1) * w_bits,
        e_pass=n["e_pass"] * n_out,   # parallel bank, one presentation
    )


def mlp_gates(n_in: int, n_hid: int, n_out: int, w_bits: int = 16, M: int = 3):
    hid = layer_gates(n_in, n_hid, w_bits)
    out = layer_gates(n_hid, n_out, w_bits)
    ee = dict(total=N_CLASS_GATES(n_out), e_pass=0.05)
    one = dict(
        logic=hid["total_logic"] + out["total_logic"] + ee["total"],
        weight_bits=hid["weight_bits"] + out["weight_bits"],
        e_forward=hid["e_pass"] + out["e_pass"] + ee["e_pass"],
        hid=hid, out=out,
    )
    return dict(
        member=one,
        ensemble_M=M,
        logic_M=one["logic"] * M,
        weight_bits_M=one["weight_bits"] * M,
        e_forward_M=one["e_forward"] * M,
        # BP reference: one float forward of same topology
        e_forward_bp=((n_in + 1) * n_hid + (n_hid + 1) * n_out) * (E_MUL16 + E_ADD16),
    )


def N_CLASS_GATES(n_class: int) -> int:
    # vote regs + adder tree + compare for confidence
    return n_class * 12 * G_REG + n_class * 12 * G_FA + 200


def um2_from_nand2(gates: int, nand2_um2: float = 0.20) -> float:
    """Rough area: ~0.2 µm²/NAND2 in a modern-ish digital node (order-of-mag)."""
    return gates * nand2_um2


def render(est: dict, sweep: list[dict]):
    fig = plt.figure(figsize=(11.5, 5.0), dpi=180, facecolor="white")
    fig.suptitle(
        "SENA RTL structural resource estimate + MNIST early-exit energy\n"
        f"neuron: serial AND+ADD+CMP+LFSR · net {est['n_in']}→{est['n_hid']}→{est['n_out']} · "
        f"M={est['M']} · W={est['w_bits']}b  (not place-and-route)",
        fontsize=11, fontweight="bold", y=0.98)

    # ---- left: resource bars ----
    ax = fig.add_axes([0.07, 0.14, 0.38, 0.70])
    labels = ["Logic\n(NAND2 eq.)", "Weight\nstorage (kbit)",
              "Logic area\nproxy (mm²)", "1-pass energy\n(rel.)"]
    member = est["detail"]["member"]
    vals = [
        est["detail"]["logic_M"] / 1e3,
        est["detail"]["weight_bits_M"] / 1e3,
        um2_from_nand2(est["detail"]["logic_M"]) / 1e6,
        est["detail"]["e_forward_M"],
    ]
    scales = ["×10³ gates", "kbit", "mm²", "add=1 units"]
    colors = ["#0284c7", "#0f766e", "#b45309", "#6d28d9"]
    x = np.arange(len(labels))
    bars = ax.bar(x, vals, color=colors, width=0.55)
    for i, (v, s) in enumerate(zip(vals, scales)):
        ax.text(i, v, f"{v:.2f}\n{s}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_title("(a) One SENA ensemble (structural)", loc="left", fontsize=10)
    ax.grid(True, axis="y", color="#e2e8f0")
    ymax = max(vals) * 1.35
    ax.set_ylim(0, ymax)

    # ---- right: MNIST energy with RTL unit ----
    ax2 = fig.add_axes([0.55, 0.14, 0.42, 0.70])
    if sweep:
        th = [r["theta"] for r in sweep]
        e_unit = est["detail"]["e_forward_M"]
        e_abs = [r["mean_passes"] * e_unit for r in sweep]
        acc = [r["acc"] * 100 for r in sweep]
        ax2.plot(e_abs, acc, "o-", color="#0284c7", lw=2, ms=6,
                 label="SENA early-exit (RTL energy unit)")
        for x, y, t in zip(e_abs, acc, th):
            lab = "full" if t > 1 else f"θ={t:.2f}"
            ax2.annotate(lab, (x, y), textcoords="offset points",
                         xytext=(5, 3), fontsize=6.5, color="#334155")
        e_bp = est["detail"]["e_forward_bp"]
        ax2.axvline(e_bp, color="#94a3b8", ls="--", lw=1.2,
                    label=f"1× real-BP forward ≈ {e_bp:.0f}")
        ax2.set_xlabel("Energy / sample (relative; add16=1)", fontsize=9)
        ax2.set_ylabel("MNIST test accuracy (%)", fontsize=9)
        ax2.set_title("(b) Early-exit curve in RTL energy units", loc="left",
                      fontsize=10)
        ax2.grid(True, color="#e2e8f0")
        ax2.legend(fontsize=7, frameon=False, loc="lower right")
    else:
        ax2.text(0.5, 0.5, "mnist_sena_results.json missing",
                 ha="center", transform=ax2.transAxes)

    fig.text(
        0.07, 0.02,
        "RTL sources: sena2026/rtl/*.v  ·  Gate model: FA=28, FF=8, CMP1=12 NAND2eq · "
        "Area proxy 0.2 µm²/NAND2 · Replace with Yosys/Vivado P&R for Q1 silicon claim.",
        fontsize=7, color="#64748b")

    paths = []
    for folder in (FIG, MEDIA):
        for ext in ("png", "svg"):
            p = os.path.join(folder, f"sena_rtl_resource_estimate.{ext}")
            fig.savefig(p, dpi=200 if ext == "png" else None,
                        bbox_inches="tight", facecolor="white")
            paths.append(p)
    plt.close(fig)
    return paths


def main():
    # Match the improved MNIST experiment
    n_in, n_hid, n_out, M, w_bits = 784, 160, 10, 3, 16
    detail = mlp_gates(n_in, n_hid, n_out, w_bits, M)
    est = dict(n_in=n_in, n_hid=n_hid, n_out=n_out, M=M, w_bits=w_bits,
               detail=detail,
               area_mm2_proxy=um2_from_nand2(detail["logic_M"]) / 1e6,
               note="structural estimate; not P&R")

    sweep = []
    jpath = os.path.join(RES, "mnist_sena_results.json")
    if os.path.exists(jpath):
        with open(jpath, encoding="utf-8") as f:
            sweep = json.load(f).get("sweep", [])

    paths = render(est, sweep)

    out = dict(
        estimate=est,
        neuron_example=neuron_gates(n_in, w_bits),
        rtl_files=[
            "lfsr32.v", "stochastic_neuron.v", "sena_layer.v",
            "sena_early_exit.v", "sena_tile.v",
        ],
        synth_status="yosys/vivado not found on host — structural estimate only",
        mnist_sweep_points=len(sweep),
    )
    opath = os.path.join(RES, "sena_rtl_estimate.json")
    with open(opath, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("Structural estimate (M=%d ensemble of %d-%d-%d):" % (M, n_in, n_hid, n_out))
    print(f"  logic NAND2-eq: {detail['logic_M']:,}")
    print(f"  weight bits:    {detail['weight_bits_M']:,}")
    print(f"  area proxy:     {est['area_mm2_proxy']:.3f} mm²")
    print(f"  E forward (M):  {detail['e_forward_M']:.1f}  (BP ref {detail['e_forward_bp']:.1f})")
    print("Wrote", opath)
    for p in paths:
        print(" ", p)


if __name__ == "__main__":
    main()
