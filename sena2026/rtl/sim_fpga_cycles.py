"""
sim_fpga_cycles.py
==================
Converts RTL clock cycles into calibrated FPGA / ASIC speed and energy figures
for the full MNIST SENA configuration (784-160-10, M=3 members).

The cycle model used here is NOT a guess: verilator_cosim.py runs the actual
Verilog under Verilator on this PC and shows that the closed form

    cycles_per_pass = ceil(N_IN/PAR) + ceil(N_HID/PAR_OUT) + 4

reproduces the RTL cycle count exactly for PAR in {1,4,8,16}, together with
bit-identical predictions, votes and pass counts.  Everything below is that
validated cycle count multiplied by technology constants.

What IS estimated (and labelled as such in every output):
  * f_max          - first-order static-timing model, not vendor place-and-route
  * energy per op  - Horowitz, "Computing's Energy Problem", ISSCC 2014, scaled

Usage:  python sim_fpga_cycles.py
"""
from __future__ import annotations

import csv
import json
import math
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
for d in (RESULTS, FIGS, MEDIA):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# network / hardware configuration (matches exp_mnist_sena.py)
# ---------------------------------------------------------------------------
N_IN, N_HID, N_OUT, M = 784, 160, 10, 3
W_BITS, ACC_BITS = 16, 28
T_MAX = 40

# fraction of 1-bits actually presented to each layer.  Input density is the
# measured MNIST binarisation density at threshold 0.3; hidden density is the
# measured mean firing rate of the random-threshold layer.
DENSITY_IN, DENSITY_HID = 0.19, 0.50

# ---------------------------------------------------------------------------
# first-order static timing model  (ns).  Labelled ESTIMATE everywhere.
# t_cq  register clock-to-Q     t_lut  bit-gate / mux
# t_add one ACC_BITS-wide add   t_route per-level interconnect
# ---------------------------------------------------------------------------
TECH = {
    "Artix-7 (28 nm, -1)":
        dict(t_cq=0.40, t_lut=0.35, t_add=1.00, t_route=0.45, t_setup=0.10,
             kind="fpga"),
    "Zynq US+ (16 nm, -2)":
        dict(t_cq=0.16, t_lut=0.14, t_add=0.42, t_route=0.20, t_setup=0.05,
             kind="fpga"),
    "ASIC 65 nm":
        dict(t_cq=0.12, t_lut=0.06, t_add=0.55, t_route=0.10, t_setup=0.03,
             kind="asic"),
    "ASIC 28 nm":
        dict(t_cq=0.05, t_lut=0.03, t_add=0.24, t_route=0.05, t_setup=0.02,
             kind="asic"),
}

# ---------------------------------------------------------------------------
# per-operation energy (pJ), 28 nm class, Horowitz ISSCC-2014 scaled
# ---------------------------------------------------------------------------
E = dict(
    add_int28=0.09,      # ACC_BITS-wide integer add
    add_int16=0.05,
    mul_int16=0.80,
    add_fp32=0.90,
    mul_fp32=3.70,
    wread16=0.35,        # 16-bit weight fetch, small on-chip SRAM
    wread32=0.70,
    flop=0.010,          # one flip-flop toggle
    cmp28=0.05,
)
CTRL_OVERHEAD = 1.20     # +20 % for FSM / counters / clock tree


def cycles_per_pass(par: int, par_out: int) -> int:
    """Validated against Verilator; see results/verilator_cosim.json."""
    return math.ceil(N_IN / par) + math.ceil(N_HID / par_out) + 4


def fmax_mhz(tech: dict, par: int) -> float:
    """Critical path: FF -> bit-gate -> adder tree -> accumulate -> compare -> FF."""
    levels = math.ceil(math.log2(max(par, 1))) + 2   # tree + accumulate + compare
    t_ns = (tech["t_cq"] + tech["t_lut"]
            + levels * (tech["t_add"] + tech["t_route"])
            + tech["t_setup"])
    return 1000.0 / t_ns


def energy_per_pass_pj(par: int, par_out: int) -> dict:
    """Energy of one member forward pass, broken down by contributor (pJ)."""
    # operand-gated: a weight is fetched and added only when its input bit is 1
    ops = N_HID * N_IN * DENSITY_IN + N_OUT * N_HID * DENSITY_HID
    e_add = ops * E["add_int28"]
    e_wr = ops * E["wread16"]

    # one LFSR step per neuron per cycle it is busy
    lfsr_steps = N_HID * math.ceil(N_IN / par) + N_OUT * math.ceil(N_HID / par_out)
    e_rng = lfsr_steps * 16 * E["flop"]          # ~16 of 32 bits toggle

    e_cmp = (N_HID + N_OUT) * E["cmp28"]
    parts = dict(add=e_add, weight_fetch=e_wr, rng=e_rng, compare=e_cmp)
    parts = {k: v * CTRL_OVERHEAD for k, v in parts.items()}
    parts["total"] = sum(parts.values())
    return parts


def energy_bp_breakdown_pj(kind: str = "int16") -> dict:
    macs = N_IN * N_HID + N_HID * N_OUT
    if kind == "int16":
        mul, add, rd = E["mul_int16"], E["add_int16"], E["wread16"]
    else:
        mul, add, rd = E["mul_fp32"], E["add_fp32"], E["wread32"]
    parts = dict(multiply=macs * mul, add=macs * add, weight_fetch=macs * rd)
    parts = {k: v * CTRL_OVERHEAD for k, v in parts.items()}
    parts["total"] = sum(parts.values())
    return parts


def fit_host_model(cosim: dict):
    """Least-squares fit of host simulation time per cycle vs logic size.

    t_cycle [s] = a + b * lanes_evaluated_per_cycle
    """
    lanes, t_cyc = [], []
    for k, v in cosim.items():
        if not k.startswith("par") or "lanes_per_cycle" not in v:
            continue
        lanes.append(v["lanes_per_cycle"])
        t_cyc.append(1.0 / v["bench_cycles_per_s"])
    A = np.vstack([np.ones(len(lanes)), np.array(lanes, float)]).T
    (a, b), *_ = np.linalg.lstsq(A, np.array(t_cyc), rcond=None)
    return float(a), float(b), lanes, t_cyc


def host_cycles_per_s(a: float, b: float, par: int, par_out: int) -> float:
    """Projected Verilator throughput for the FULL 784-160-10 x M design."""
    lanes = (N_HID * par + N_OUT * par_out) * M
    return 1.0 / (a + b * lanes)


def adders_per_member(par: int, par_out: int) -> int:
    return N_HID * par + N_OUT * par_out


def main():
    cosim = json.loads((RESULTS / "verilator_cosim.json").read_text())
    mnist = json.loads((RESULTS / "mnist_sena_results.json").read_text())
    host_a, host_b, host_lanes, host_t = fit_host_model(cosim)

    validated = [(v["par"], v["par_out"], v["cycles_pass1_rtl"],
                  v["cycles_pass1_formula"], v["match"])
                 for k, v in cosim.items() if k != "_host"]

    print("=" * 74)
    print("RTL cycle model validated against Verilator (64-8-4 tile)")
    print("=" * 74)
    for par, par_out, rtl, form, ok in validated:
        print(f"  PAR={par:<3d} PAR_OUT={par_out:<3d} "
              f"RTL={rtl:<4d} formula={form:<4d} "
              f"{'exact match' if ok and rtl == form else 'MISMATCH'}")

    # ---- design-space sweep on the full MNIST tile ----------------------
    pars = [1, 2, 4, 8, 16, 32, 64]
    rows = []
    for par in pars:
        par_out = max(1, par // 4)
        cyc = cycles_per_pass(par, par_out)
        e_pass = energy_per_pass_pj(par, par_out)["total"]
        row = dict(par=par, par_out=par_out, cycles_per_pass=cyc,
                   adders=adders_per_member(par, par_out) * M,
                   e_pass_nj=e_pass / 1e3,
                   host_cycles_per_s=host_cycles_per_s(host_a, host_b,
                                                       par, par_out))
        for name, tech in TECH.items():
            f = fmax_mhz(tech, par)
            row[f"fmax::{name}"] = f
            row[f"tpass_us::{name}"] = cyc / (f * 1e6) * 1e6
        rows.append(row)

    print()
    print("=" * 74)
    print("Full MNIST tile 784-160-10: cycles per stochastic pass")
    print("=" * 74)
    print(f"{'PAR':>4} {'PARo':>5} {'cycles':>8} {'adders(M=3)':>12} "
          f"{'Artix f':>9} {'Artix us':>9} {'US+ f':>8} {'US+ us':>8}")
    for r in rows:
        print(f"{r['par']:>4} {r['par_out']:>5} {r['cycles_per_pass']:>8} "
              f"{r['adders']:>12,} "
              f"{r['fmax::Artix-7 (28 nm, -1)']:>8.0f}M "
              f"{r['tpass_us::Artix-7 (28 nm, -1)']:>9.2f} "
              f"{r['fmax::Zynq US+ (16 nm, -2)']:>7.0f}M "
              f"{r['tpass_us::Zynq US+ (16 nm, -2)']:>8.2f}")

    # ---- per-inference latency / throughput / energy --------------------
    sweep = mnist["sweep"]
    bp16 = energy_bp_breakdown_pj("int16")
    bp32 = energy_bp_breakdown_pj("fp32")
    e_bp16, e_bp32 = bp16["total"], bp32["total"]

    # a sensible operating point for the headline numbers
    PAR_OP, PAROUT_OP = 16, 4
    cyc_op = cycles_per_pass(PAR_OP, PAROUT_OP)
    pass_parts = energy_per_pass_pj(PAR_OP, PAROUT_OP)
    e_pass_op = pass_parts["total"]

    infer = []
    for s in sweep:
        mp = s["mean_passes"]
        rec = dict(theta=s["theta"], acc=s["acc"], mean_passes=mp,
                   energy_uj=mp * e_pass_op / 1e6)
        for name, tech in TECH.items():
            f = fmax_mhz(tech, PAR_OP) * 1e6
            # M tiles in parallel -> ceil(mp / M) sequential pass slots
            cyc_1tile = mp * cyc_op
            cyc_mtile = math.ceil(mp / M) * cyc_op
            rec[f"lat_us_1tile::{name}"] = cyc_1tile / f * 1e6
            rec[f"lat_us_{M}tile::{name}"] = cyc_mtile / f * 1e6
            rec[f"kips_{M}tile::{name}"] = f / cyc_mtile / 1e3
        infer.append(rec)

    print()
    print("=" * 74)
    print(f"Per-inference figures at PAR={PAR_OP}/{PAROUT_OP} "
          f"({cyc_op} cycles per pass, M={M} tiles in parallel)")
    print("=" * 74)
    print(f"{'theta':>6} {'acc':>7} {'passes':>7} {'E (uJ)':>9} "
          f"{'Artix us':>10} {'Artix kIPS':>11} {'US+ us':>9} {'US+ kIPS':>10}")
    for r in infer:
        print(f"{r['theta']:>6.2f} {r['acc']*100:>6.2f}% {r['mean_passes']:>7.2f} "
              f"{r['energy_uj']:>9.3f} "
              f"{r['lat_us_3tile::Artix-7 (28 nm, -1)']:>10.1f} "
              f"{r['kips_3tile::Artix-7 (28 nm, -1)']:>11.1f} "
              f"{r['lat_us_3tile::Zynq US+ (16 nm, -2)']:>9.1f} "
              f"{r['kips_3tile::Zynq US+ (16 nm, -2)']:>10.1f}")

    print()
    print(f"  Back-prop reference forward pass: "
          f"int16 {e_bp16/1e6:.3f} uJ, fp32 {e_bp32/1e6:.3f} uJ")
    best = min(infer, key=lambda r: r["energy_uj"])
    good = [r for r in infer if r["acc"] >= 0.87]
    pick = min(good, key=lambda r: r["energy_uj"]) if good else best
    print(f"  SENA at theta={pick['theta']:.2f} ({pick['acc']*100:.2f}% acc): "
          f"{pick['energy_uj']:.3f} uJ  -> "
          f"{e_bp16/1e6/pick['energy_uj']:.1f}x vs int16 BP, "
          f"{e_bp32/1e6/pick['energy_uj']:.1f}x vs fp32 BP")

    # ---- PC simulation vs FPGA wall-clock -------------------------------
    print()
    print("=" * 74)
    print("PC simulation speed vs hardware speed (calibration)")
    print("=" * 74)
    print(f"  Host cost model fitted on 4 verilated configs: "
          f"t_cycle = {host_a*1e9:.0f} ns + {host_b*1e9:.2f} ns x lanes")
    measured = 1.0 / (host_a + host_b * host_lanes[0])
    print(f"  Measured on the 64-8-4 tile      : "
          f"{cosim['par1_1']['bench_cycles_per_s']/1e6:8.2f} M cycles/s")
    host_full = host_cycles_per_s(host_a, host_b, PAR_OP, PAROUT_OP)
    print(f"  Projected, full 784-160-10 x M=3 : "
          f"{host_full/1e3:8.2f} k cycles/s "
          f"({(N_HID*PAR_OP + N_OUT*PAROUT_OP)*M:,} lanes/cycle)")
    cyc_infer = math.ceil(pick["mean_passes"] / M) * cyc_op
    conv = {}
    for name, tech in TECH.items():
        f = fmax_mhz(tech, PAR_OP) * 1e6
        slow = f / host_full
        conv[name] = dict(fmax_mhz=f / 1e6, slowdown=slow,
                          host_s_per_inference=cyc_infer / host_full,
                          hw_s_per_inference=cyc_infer / f)
        print(f"  {name:<22}: {f/1e6:7.1f} MHz -> "
              f"{slow:9,.0f}x slower in simulation "
              f"({slow/3600:6.2f} PC-hours per hardware second)")
    print(f"  One MNIST inference (theta={pick['theta']:.2f}, {cyc_infer} cycles): "
          f"{cyc_infer/host_full*1e3:.1f} ms of PC simulation, "
          f"{conv['Artix-7 (28 nm, -1)']['hw_s_per_inference']*1e6:.2f} us on Artix-7.")
    print(f"  => a 2,500-image MNIST test set = "
          f"{2500*cyc_infer/host_full/60:.1f} minutes of RTL simulation "
          f"on this PC.")

    # ---- persist ---------------------------------------------------------
    out = dict(
        config=dict(n_in=N_IN, n_hid=N_HID, n_out=N_OUT, M=M,
                    w_bits=W_BITS, acc_bits=ACC_BITS, t_max=T_MAX,
                    density_in=DENSITY_IN, density_hid=DENSITY_HID),
        cycle_model_validated=[dict(par=p, par_out=po, rtl=r, formula=f,
                                    match=bool(m))
                               for p, po, r, f, m in validated],
        host_model=dict(t_cycle_ns_intercept=host_a * 1e9,
                        t_cycle_ns_per_lane=host_b * 1e9,
                        measured_lanes=host_lanes,
                        measured_t_cycle_ns=[t * 1e9 for t in host_t],
                        projected_full_cycles_per_s=host_cycles_per_s(
                            host_a, host_b, PAR_OP, PAROUT_OP)),
        energy_constants_pj=E,
        timing_model_ns=TECH,
        design_space=rows,
        per_inference=infer,
        energy_breakdown_pj=dict(sena_per_pass=pass_parts,
                                 bp_int16=bp16, bp_fp32=bp32),
        bp_reference_uj=dict(int16=e_bp16 / 1e6, fp32=e_bp32 / 1e6),
        operating_point=dict(par=PAR_OP, par_out=PAROUT_OP,
                             cycles_per_pass=cyc_op,
                             e_pass_nj=e_pass_op / 1e3),
        sim_conversion=conv,
    )
    (RESULTS / "fpga_speed_model.json").write_text(json.dumps(out, indent=2))

    with open(RESULTS / "fpga_design_space.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    make_figure(rows, infer, conv, host_full, host_a, host_b,
                host_lanes, host_t, pass_parts, bp16, bp32, pick,
                PAR_OP, PAROUT_OP, cyc_op, validated)
    print()
    print(f"wrote {RESULTS / 'fpga_speed_model.json'}")
    print(f"wrote {RESULTS / 'fpga_design_space.csv'}")


# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------
def make_figure(rows, infer, conv, host_full, host_a, host_b,
                host_lanes, host_t, pass_parts, bp16, bp32, pick,
                par_op, parout_op, cyc_op, validated):
    e_bp16, e_bp32 = bp16["total"], bp32["total"]
    fig, axes = plt.subplots(2, 3, figsize=(19, 10))
    fig.suptitle(
        "SENA on FPGA: Verilator-validated cycle counts converted to "
        "calibrated speed and energy\n"
        "784-160-10, M=3 stochastic members, 16-bit weights",
        fontsize=13, fontweight="bold")

    pars = [r["par"] for r in rows]
    artix = "Artix-7 (28 nm, -1)"
    uscale = "Zynq US+ (16 nm, -2)"

    # (a) cycles and f_max vs parallelism
    ax = axes[0, 0]
    ax.plot(pars, [r["cycles_per_pass"] for r in rows], "o-", color="#1f4e79",
            lw=2, label="cycles / pass (RTL-validated)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("PAR  (inputs folded per clock)")
    ax.set_ylabel("clock cycles per stochastic pass", color="#1f4e79")
    ax.grid(alpha=0.3, which="both")
    ax2 = ax.twinx()
    ax2.plot(pars, [r[f"fmax::{artix}"] for r in rows], "s--", color="#c00000",
             label=f"f_max {artix}")
    ax2.plot(pars, [r[f"fmax::{uscale}"] for r in rows], "^--", color="#e08214",
             label=f"f_max {uscale}")
    ax2.set_ylabel("estimated f_max (MHz)")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right")
    ax.set_title("(a) More parallelism = fewer cycles but lower f_max",
                 fontsize=11)

    # (b) time per pass -> the optimum
    ax = axes[0, 1]
    notes = []
    for name, col, mk in ((artix, "#c00000", "s"), (uscale, "#e08214", "^")):
        t = [r[f"tpass_us::{name}"] for r in rows]
        ax.plot(pars, t, mk + "-", color=col, lw=2, label=name)
        i = int(np.argmin(t))
        notes.append((f"fastest: {t[i]:.2f} us at PAR={pars[i]}", col))
    for j, (txt, col) in enumerate(notes):
        ax.text(0.03, 0.30 - 0.09 * j, txt, transform=ax.transAxes,
                fontsize=8.5, color=col, fontweight="bold")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("PAR")
    ax.set_ylabel("time per stochastic pass (us)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_title("(b) Wall-clock time per pass (cycles / f_max)", fontsize=11)

    # (c) where the energy actually goes
    ax = axes[0, 2]
    keys = ["add", "multiply", "weight_fetch", "rng", "compare"]
    colors = {"add": "#1f4e79", "multiply": "#c00000",
              "weight_fetch": "#e08214", "rng": "#4472c4",
              "compare": "#7030a0"}
    bars = {
        f"SENA\n1 pass": pass_parts,
        f"SENA\n{pick['mean_passes']:.1f} passes":
            {k: v * pick["mean_passes"] for k, v in pass_parts.items()},
        "int16 BP\n1 pass": bp16,
        "fp32 BP\n1 pass": bp32,
    }
    x = np.arange(len(bars))
    bottom = np.zeros(len(bars))
    for k in keys:
        vals = np.array([b.get(k, 0.0) / 1e6 for b in bars.values()])
        if vals.sum() == 0:
            continue
        ax.bar(x, vals, 0.6, bottom=bottom, label=k.replace("_", " "),
               color=colors[k])
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(list(bars), fontsize=8.5)
    ax.set_ylabel("energy per inference (uJ)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    for xi, tot in zip(x, bottom):
        ax.text(xi, tot * 1.02, f"{tot:.3f}", ha="center", fontsize=8.5)
    ax.set_title("(c) Energy breakdown: weight movement,\n"
                 "not arithmetic, is the real cost", fontsize=11)

    # (d) accuracy vs energy per inference, with BP references
    ax = axes[1, 0]
    en = [r["energy_uj"] for r in infer]
    ac = [r["acc"] * 100 for r in infer]
    ax.plot(en, ac, "o-", color="#1f4e79", lw=2, label="SENA early exit")
    for r in infer:
        ax.annotate(f"{r['theta']:.2f}", (r["energy_uj"], r["acc"] * 100),
                    textcoords="offset points", xytext=(4, -11), fontsize=7.5)
    ax.axvline(e_bp16 / 1e6, color="#c00000", ls="--", lw=1.6,
               label=f"int16 BP forward = {e_bp16/1e6:.2f} uJ")
    ax.axvline(e_bp32 / 1e6, color="#7030a0", ls=":", lw=1.6,
               label=f"fp32 BP forward = {e_bp32/1e6:.2f} uJ")
    ax.set_xscale("log")
    ax.set_xlabel("estimated energy per inference (uJ, log scale)")
    ax.set_ylabel("MNIST test accuracy (%)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title(f"(d) Accuracy vs energy, PAR={par_op} "
                 f"({cyc_op} cycles/pass)", fontsize=11)

    # (e) host simulation cost model
    ax = axes[1, 1]
    ax.plot(host_lanes, [t * 1e9 for t in host_t], "o", ms=9,
            color="#1f4e79", label="measured (Verilator, this PC)")
    xs = np.linspace(0, max(host_lanes) * 1.15, 50)
    ax.plot(xs, (host_a + host_b * xs) * 1e9, "-", color="#c00000",
            label=f"fit: {host_a*1e9:.0f} + {host_b*1e9:.2f} x lanes  (ns)")
    ax.set_xlabel("adder lanes evaluated per clock cycle")
    ax.set_ylabel("host time per simulated cycle (ns)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("(e) Host simulation cost scales with logic size\n"
                 "(this is what lets us project to the full design)",
                 fontsize=11)

    # (f) PC simulation vs hardware speed
    ax = axes[1, 2]
    names = list(conv.keys())
    slow = [conv[n]["slowdown"] for n in names]
    y = np.arange(len(names))
    bars = ax.barh(y, slow, color=["#c00000", "#e08214", "#4472c4", "#1f4e79"])
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("PC seconds per hardware second  (log scale)")
    ax.grid(alpha=0.3, axis="x", which="both")
    for b, n in zip(bars, names):
        ax.text(b.get_width() * 1.1, b.get_y() + b.get_height() / 2,
                f"{conv[n]['fmax_mhz']:.0f} MHz\n{conv[n]['slowdown']:,.0f}x",
                va="center", fontsize=8)
    ax.set_xlim(right=max(slow) * 8)
    ax.set_title(f"(f) Full design in Verilator = {host_full/1e3:.1f} k cycles/s\n"
                 "conversion factor to hardware time", fontsize=11)

    fig.text(0.5, 0.005,
             "Cycle counts are exact: the closed-form model reproduces "
             "Verilator RTL simulation bit-for-bit for PAR in {1,4,8,16} "
             f"({len(validated)}/{len(validated)} configs match). "
             "f_max is a first-order static-timing estimate and energy uses "
             "Horowitz ISSCC-2014 per-op constants; both require vendor "
             "place-and-route for sign-off.",
             ha="center", fontsize=8, style="italic", wrap=True)

    fig.tight_layout(rect=[0, 0.03, 1, 0.94])
    for d in (FIGS, MEDIA):
        fig.savefig(d / "sena_fpga_speed_model.png", dpi=180)
        fig.savefig(d / "sena_fpga_speed_model.svg")
    plt.close(fig)
    print(f"wrote {MEDIA / 'sena_fpga_speed_model.png'}")


if __name__ == "__main__":
    main()
