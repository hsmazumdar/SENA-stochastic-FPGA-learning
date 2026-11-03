"""
exp_q1_synth.py
===============
A4 — FPGA synthesis / resource / power evidence for Q1 Table H1.

Priority:
  1. Run Vivado batch synthesis if `vivado` is on PATH.
  2. Otherwise emit an Artix-7 *mapping estimate* from the validated RTL
     operator counts, clearly labelled as not place-and-route.

Also projects a full 784-160-10 x M=3 design by scaling a synthesizable
hidden-bank slice (default N_HID_BANK=16).

Examples:
  python exp_q1_synth.py --profile smoke
  python exp_q1_synth.py --profile full
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / "results"
FIGS = ROOT / "assets" / "figs"
MEDIA = ROOT.parent / "_media"

PROFILES = {
    "smoke": dict(hid_bank=8, par=4, par_out=1, part="xc7a100tcsg324-1"),
    "full": dict(hid_bank=16, par=8, par_out=2, part="xc7a100tcsg324-1"),
}

# Artix-7 mapping factors for one serial neuron (order-of-magnitude LUT model).
# Calibrated to typical Vivado results for add/compare/LFSR datapaths.
LUT_PER_ADD_BIT = 1.1
LUT_PER_CMP_BIT = 0.9
LUT_LFSR32 = 35
FF_PER_ACC_BIT = 1.0
LUT_CTRL = 40
# Dynamic power proxy at 100 MHz: ~2 uW / LUT toggle + RAM.
UW_PER_LUT_100MHZ = 2.0
UW_PER_FF_100MHZ = 0.6


def find_vivado():
    return shutil.which("vivado")


def run_vivado(cfg):
    vivado = find_vivado()
    if not vivado:
        return None
    tag = f"hid{cfg['hid_bank']}_par{cfg['par']}_{cfg['par_out']}"
    outdir = HERE / "_synth" / tag
    cmd = [
        vivado, "-mode", "batch", "-source", str(HERE / "synth_sena_tile.tcl"),
        "-tclargs",
        f"N_HID_BANK={cfg['hid_bank']}",
        f"PAR={cfg['par']}",
        f"PAR_OUT={cfg['par_out']}",
        f"PART={cfg['part']}",
    ]
    t0 = time.perf_counter()
    result = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
    wall = time.perf_counter() - t0
    report = outdir / "synth.json"
    if result.returncode or not report.exists():
        return dict(tool="vivado", ok=False, wall_s=wall,
                    stderr=(result.stderr or result.stdout)[-4000:])
    data = json.loads(report.read_text(encoding="utf-8"))
    data.update(ok=True, wall_s=wall, evidence="post-synth")
    return data


def estimate_slice(hid_bank, par, par_out, w_bits=16, acc_bits=28):
    """Structural Artix-7 LUT/FF estimate for one tile slice."""
    # Hidden bank: hid_bank neurons, each with PAR-wide adder tree.
    tree_levels = max(1, math.ceil(math.log2(max(par, 1))) + 1)
    lut_neuron_hid = (
        tree_levels * acc_bits * LUT_PER_ADD_BIT
        + acc_bits * LUT_PER_CMP_BIT
        + LUT_LFSR32 + LUT_CTRL)
    ff_neuron_hid = acc_bits * FF_PER_ACC_BIT + 32 + 20
    tree_out = max(1, math.ceil(math.log2(max(par_out, 1))) + 1)
    lut_neuron_out = (
        tree_out * acc_bits * LUT_PER_ADD_BIT
        + acc_bits * LUT_PER_CMP_BIT
        + LUT_LFSR32 + LUT_CTRL)
    ff_neuron_out = acc_bits * FF_PER_ACC_BIT + 32 + 16

    lut = hid_bank * lut_neuron_hid + 10 * lut_neuron_out + 200
    ff = hid_bank * ff_neuron_hid + 10 * ff_neuron_out + 120
    # Weights in LUTRAM/BRAM: report bits; BRAM tiles ~36 Kb each
    weight_bits = hid_bank * (784 + 1) * w_bits + 10 * (hid_bank + 1) * w_bits
    bram = weight_bits / (36 * 1024)
    return dict(
        tool="structural_artix7_model",
        ok=True,
        evidence="mapping-estimate-not-PnR",
        n_hid_bank=hid_bank, par=par, par_out=par_out,
        lut=int(round(lut)), ff=int(round(ff)),
        bram=round(bram, 2), dsp=0,
        weight_bits=int(weight_bits),
    )


def scale_to_full(slice_rpt, full_hid=160, M=3):
    bank = max(1, int(slice_rpt["n_hid_bank"]))
    tiles = math.ceil(full_hid / bank)
    scale = tiles  # one member
    return dict(
        full_hid=full_hid, tiles_per_member=tiles, M=M,
        lut=int(slice_rpt["lut"] * scale * M),
        ff=int(slice_rpt["ff"] * scale * M),
        bram=round(float(slice_rpt["bram"]) * scale * M, 2),
        dsp=int(slice_rpt.get("dsp", 0)) * scale * M,
        note="Linear scale of slice; interconnect / BRAM packing not PnR'd",
    )


def folded_fit(slice_rpt, capacity, full_hid=160):
    """Time-multiplex one physical bank across hidden neurons and members."""
    bank = max(1, int(slice_rpt["n_hid_bank"]))
    folds = math.ceil(full_hid / bank)
    # One physical tile + weight BRAM for full member, reused over folds and M.
    weight_bits_member = full_hid * (784 + 1) * 16 + 10 * (full_hid + 1) * 16
    bram = weight_bits_member / (36 * 1024)
    lut = int(slice_rpt["lut"])
    ff = int(slice_rpt["ff"])
    return dict(
        strategy="time-multiplexed hidden bank + sequential ensemble members",
        physical_hid_bank=bank,
        folds_per_member=folds,
        lut=lut,
        ff=ff,
        bram=round(bram, 2),
        dsp=0,
        lut_pct=100 * lut / capacity["lut"],
        ff_pct=100 * ff / capacity["ff"],
        bram_pct=100 * bram / capacity["bram"],
        latency_penalty=folds,  # vs fully parallel hidden
        fits=lut < capacity["lut"] and bram < capacity["bram"],
    )


def power_uj(mean_passes, lut, ff, f_mhz=100.0, M=3):
    """Dynamic energy proxy at f_mhz for mean_passes of an M-tile ensemble."""
    # One temporal pass evaluates all M members (parallel tiles).
    w = (lut * UW_PER_LUT_100MHZ + ff * UW_PER_FF_100MHZ) * (f_mhz / 100.0)
    # cycles/pass for PAR=8 on 784-160-10 ≈ 182 from sim_fpga_cycles
    cycles = 182
    seconds = mean_passes * cycles / (f_mhz * 1e6)
    return (w * 1e-6) * seconds * 1e6  # uJ


def render(slice_rpt, full_rpt, rows):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), dpi=180)
    labels = ["LUT", "FF", "BRAM×10", "DSP"]
    vals = [full_rpt["lut"], full_rpt["ff"],
            full_rpt["bram"] * 10, full_rpt["dsp"]]
    axes[0].bar(labels, vals, color="#4472c4")
    axes[0].set_title("(a) Projected 784-160-10 × M=3 resources")
    axes[0].set_ylabel("Count (BRAM shown ×10)")
    axes[0].grid(axis="y", alpha=0.25)

    if rows:
        th = [r["theta"] for r in rows]
        e = [r["energy_uj"] for r in rows]
        a = [r["acc"] * 100 for r in rows]
        axes[1].plot(e, a, "o-", color="#c00000", lw=2)
        for x, y, t in zip(e, a, th):
            axes[1].annotate(f"{t:.2f}", (x, y), textcoords="offset points",
                             xytext=(4, 3), fontsize=7)
        axes[1].set_xlabel("Estimated dynamic energy / inference (uJ)")
        axes[1].set_ylabel("MNIST accuracy (%)")
        axes[1].set_title("(b) Power-model energy vs accuracy")
        axes[1].grid(alpha=0.25)
    else:
        axes[1].text(0.5, 0.5, "MNIST sweep unavailable",
                     ha="center", va="center", transform=axes[1].transAxes)
        axes[1].set_axis_off()

    src = slice_rpt.get("evidence", slice_rpt.get("tool"))
    fig.suptitle(
        f"A4 SENA FPGA resource / power table ({src})\n"
        f"slice {slice_rpt.get('n_hid_bank')} hidden × PAR="
        f"{slice_rpt.get('par')}/{slice_rpt.get('par_out')}",
        fontsize=11, fontweight="bold")
    fig.tight_layout()
    for d in (FIGS, MEDIA):
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / "q1_fpga_synth_resources.png", bbox_inches="tight")
        fig.savefig(d / "q1_fpga_synth_resources.svg", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=PROFILES, default="smoke")
    args = parser.parse_args()
    cfg = PROFILES[args.profile]
    RESULTS.mkdir(parents=True, exist_ok=True)

    viv = run_vivado(cfg)
    if viv and viv.get("ok"):
        slice_rpt = viv
        slice_rpt["n_hid_bank"] = cfg["hid_bank"]
        slice_rpt["par"] = cfg["par"]
        slice_rpt["par_out"] = cfg["par_out"]
    else:
        slice_rpt = estimate_slice(cfg["hid_bank"], cfg["par"], cfg["par_out"])
        if viv is not None:
            slice_rpt["vivado_attempt"] = viv

    full_rpt = scale_to_full(slice_rpt)
    capacity = dict(part=cfg["part"], lut=63400, ff=126800, bram=135, dsp=240)
    folded = folded_fit(slice_rpt, capacity)
    # Attach energy points from MNIST early-exit if available.
    rows = []
    mnist = RESULTS / "mnist_sena_results.json"
    if mnist.exists():
        sweep = json.loads(mnist.read_text(encoding="utf-8"))["sweep"]
        # Use folded LUT count for a device-feasible power model.
        for s in sweep:
            rows.append(dict(
                theta=s["theta"], acc=s["acc"],
                mean_passes=s["mean_passes"],
                energy_uj=power_uj(
                    s["mean_passes"], folded["lut"], folded["ff"])))

    render(slice_rpt, full_rpt, rows)
    util = dict(
        lut_pct=100 * full_rpt["lut"] / capacity["lut"],
        ff_pct=100 * full_rpt["ff"] / capacity["ff"],
        bram_pct=100 * full_rpt["bram"] / capacity["bram"],
    )
    payload = dict(
        profile=args.profile, config=cfg, slice=slice_rpt,
        full_parallel_design=full_rpt,
        folded_artix7_design=folded,
        device_capacity=capacity,
        utilization_pct_parallel=util, energy_rows=rows,
        vivado_found=bool(find_vivado()),
    )
    out = RESULTS / "q1_fpga_synth.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(dict(
        evidence=slice_rpt.get("evidence"),
        slice_lut=slice_rpt["lut"],
        parallel_lut=full_rpt["lut"],
        parallel_lut_pct=round(util["lut_pct"], 1),
        folded_lut=folded["lut"],
        folded_lut_pct=round(folded["lut_pct"], 1),
        folded_fits=folded["fits"],
        vivado_found=payload["vivado_found"]), indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
