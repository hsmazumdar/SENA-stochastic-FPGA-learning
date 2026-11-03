"""
C2 model-based energy for the Papilio Pro calibration slice.

Evidence ladder (honest labeling):
  * cycles/pass  = BOARD-MEASURED (papilio_calibration.json → 807)
  * f_clk        = BOARD CLOCK (32 MHz), with post-route f_max 32.601 MHz
  * energy/op    = MODEL (Horowitz ISSCC-2014 class constants, same family
                   as rtl/sim_fpga_cycles.py), NOT rail-measured

E_pass ≈ Σ (gated weight-read + add) + LFSR + compare, then × control overhead.
E_inference = E_pass × mean_passes  (early-exit or fixed-T).

Also scales the same model to the paper's full 784-160-10 × M=3 ensemble
using the Verilator-validated full-net cycle formula (not Papilio-sized).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / "results"

# Horowitz-class 28 nm energy constants (pJ) — same as sim_fpga_cycles.py
E = dict(
    add_int28=0.09,
    wread16=0.35,
    flop=0.010,
    cmp28=0.05,
)
CTRL = 1.20
DENSITY_IN, DENSITY_HID = 0.19, 0.50


def energy_pass_pj(n_in, n_hid, n_out, par=1, par_out=1):
    ops = n_hid * n_in * DENSITY_IN + n_out * n_hid * DENSITY_HID
    e_add = ops * E["add_int28"]
    e_wr = ops * E["wread16"]
    lfsr_steps = n_hid * math.ceil(n_in / par) + n_out * math.ceil(n_hid / par_out)
    e_rng = lfsr_steps * 16 * E["flop"]
    e_cmp = (n_hid + n_out) * E["cmp28"]
    parts = {k: v * CTRL for k, v in dict(
        add=e_add, weight_fetch=e_wr, rng=e_rng, compare=e_cmp).items()}
    parts["total_pJ"] = sum(parts.values())
    return parts


def main():
    pap = json.loads((RESULTS / "papilio_calibration.json").read_text(encoding="utf-8"))
    abl = json.loads((RESULTS / "q1_sena_ablations.json").read_text(encoding="utf-8"))
    mnist = {}
    mpath = RESULTS / "mnist_sena_results.json"
    if mpath.exists():
        mnist = json.loads(mpath.read_text(encoding="utf-8"))

    f_clk = float(pap["f_clk_hz"])
    cpp = float(pap["measured_cycles_per_pass"])
    # early-exit mean passes from board run
    ee_passes = pap["early_exit_passes"]
    mean_passes_board = sum(ee_passes) / len(ee_passes)

    # Papilio slice 784-16-10, PAR=1 (matches calib_core)
    slice_e = energy_pass_pj(784, 16, 10, par=1, par_out=1)
    e_pass_uj = slice_e["total_pJ"] / 1e6
    e_inf_fixed40_uj = e_pass_uj * 40
    e_inf_ee_uj = e_pass_uj * mean_passes_board
    t_pass_us = cpp / f_clk * 1e6
    t_inf_ee_us = t_pass_us * mean_passes_board

    # Full paper net 784-160-10 × M=3, cycles from validated formula
    # cycles_per_pass_member = ceil(784/PAR)+ceil(160/PAR_OUT)+4 ; M members sequential
    # Use PAR=1 folded like board for a conservative energy upper model, and
    # PAR=8 / PAR_OUT=2 as the Artix mapping estimate point.
    full = {}
    for label, n_hid, m, par, par_out in (
        ("full_par1_M3", 160, 3, 1, 1),
        ("full_par8_M3", 160, 3, 8, 2),
    ):
        cyc_member = math.ceil(784 / par) + math.ceil(n_hid / par_out) + 4
        cyc_ens = cyc_member * m
        e_member = energy_pass_pj(784, n_hid, 10, par, par_out)
        e_ens_pass = e_member["total_pJ"] * m
        # A2 ensemble early-exit mean relative energy ~12.8% of full T=40
        abl_ee = next(s for s in abl["summary"]
                      if s["condition"] == "ensemble_early_exit")
        mean_p = abl_ee["mean_passes_mean"]
        full[label] = dict(
            cycles_per_ensemble_pass=cyc_ens,
            energy_per_ensemble_pass_pJ=e_ens_pass,
            energy_per_ensemble_pass_uJ=e_ens_pass / 1e6,
            a2_mean_passes=mean_p,
            energy_per_inference_ee_uJ=(e_ens_pass / 1e6) * mean_p,
            energy_per_inference_full40_uJ=(e_ens_pass / 1e6) * 40,
            note="model energy; cycles formula Verilator-validated; "
                 "not Papilio rail power",
        )

    # INT16 MAC baseline energy for one forward (no early exit), for ratio
    macs = 784 * 160 + 160 * 10
    e_int16_mac_pJ = macs * (0.80 + 0.05 + 0.35) * CTRL   # mul+add+read
    e_fp32_mac_pJ = macs * (3.70 + 0.90 + 0.70) * CTRL

    payload = dict(
        evidence_levels=dict(
            cycles="board-measured (Papilio Pro)",
            clock_Hz="board oscillator 32e6; post-route f_max 32.601e6",
            energy="model-based (Horowitz-class pJ/op); NOT rail-measured",
            relative_energy="software op-count early-exit fraction (A2)",
        ),
        papilio_slice=dict(
            topology="784-16-10",
            measured_cycles_per_pass=cpp,
            f_clk_Hz=f_clk,
            time_per_pass_us=t_pass_us,
            board_early_exit_mean_passes=mean_passes_board,
            board_core_latency_us=pap["core_latency_us"],
            model_energy_per_pass_pJ=slice_e,
            model_energy_per_pass_uJ=e_pass_uj,
            model_energy_inference_fixedT40_uJ=e_inf_fixed40_uj,
            model_energy_inference_boardEE_uJ=e_inf_ee_uj,
            model_time_inference_boardEE_us=t_inf_ee_us,
        ),
        full_mnist_model=full,
        baseline_one_forward_uJ=dict(
            int16_mac=e_int16_mac_pJ / 1e6,
            fp32_mac=e_fp32_mac_pJ / 1e6,
        ),
        ratios_vs_int16_mac=dict(
            sena_full40_par8_M3=(
                full["full_par8_M3"]["energy_per_inference_full40_uJ"]
                / (e_int16_mac_pJ / 1e6)),
            sena_ee_par8_M3=(
                full["full_par8_M3"]["energy_per_inference_ee_uJ"]
                / (e_int16_mac_pJ / 1e6)),
        ),
        citation_sentence=(
            "Absolute energy is reported from a cycle×operation energy model "
            "(Horowitz-class constants) whose cycle count was validated on a "
            "Papilio Pro (Spartan-6 LX9) to be exactly 807 cycles/pass; board "
            "rail power was not instrumented on the available setup."
        ),
    )
    out = RESULTS / "q1_papilio_model_energy.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
