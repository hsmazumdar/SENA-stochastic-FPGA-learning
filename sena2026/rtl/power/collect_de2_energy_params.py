"""
Collect DE2-115 Quartus PowerPlay + model energy parameters for Table VI / V03.

Reads:
  I:/FPGA_Tools_115/SENA_DE2_115/quartus_calib/output_files/de2_115_calib.pow.summary
  sena2026/results/q1_papilio_model_energy.json
  sena2026/results/papilio_calibration.json (if present)

Writes:
  sena2026/results/de2_115_power/energy_params.json
  sena2026/rtl/_power/power.json  (for patch_table_vi_vivado.py)
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(r"D:\_July2026\SENA_Project")
POW_SUM = Path(
    r"I:\FPGA_Tools_115\SENA_DE2_115\quartus_calib\output_files"
    r"\de2_115_calib.pow.summary"
)
OUT_DIR = ROOT / "sena2026" / "results" / "de2_115_power"
MODEL = ROOT / "sena2026" / "results" / "q1_papilio_model_energy.json"


def parse_pow_summary(path: Path) -> dict:
    txt = path.read_text(encoding="utf-8", errors="replace")
    def grab(label):
        m = re.search(rf"{re.escape(label)}\s*:\s*([0-9.]+)\s*mW", txt)
        return float(m.group(1)) if m else None
    conf = None
    m = re.search(r"Power Estimation Confidence\s*:\s*(.+)", txt)
    if m:
        conf = m.group(1).strip()
    return dict(
        total_mw=grab("Total Thermal Power Dissipation"),
        core_dynamic_mw=grab("Core Dynamic Thermal Power Dissipation"),
        core_static_mw=grab("Core Static Thermal Power Dissipation"),
        io_mw=grab("I/O Thermal Power Dissipation"),
        confidence=conf,
        device="EP4CE115F29C7",
        family="Cyclone IV E",
        tool="Quartus Prime 25.1 PowerPlay (vectorless)",
        source=str(path),
    )


def uj(power_mw: float, thruput_inf_s: float) -> float:
    return 1000.0 * power_mw / thruput_inf_s


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pow_ = parse_pow_summary(POW_SUM)
    # Prefer the more recent 25% toggle run if summary is that one; also keep
    # the vectorless 12.5% numbers from the first successful run archived below.
    # Current summary on disk is the last run (25% toggle → 187.58 mW).
    model = json.loads(MODEL.read_text(encoding="utf-8")) if MODEL.exists() else {}

    f_clk = 25e6  # DE2 core
    cyc = 807.0
    board_ee_passes = 30.25
    fixed_n_passes = 2.30
    adaptive_passes = 5.11
    t40 = 40.0

    def thruput(passes):
        return f_clk / (cyc * passes)

    ops = {
        "board_ee_30p25": board_ee_passes,
        "fixed_n_ee_2p30": fixed_n_passes,
        "adaptive_ee_5p11": adaptive_passes,
        "full_t40": t40,
    }

    # Report both Total and Core-Dynamic energies (static dominates FPGA totals).
    slice_energy = {}
    for name, passes in ops.items():
        th = thruput(passes)
        slice_energy[name] = dict(
            mean_passes=passes,
            throughput_inf_s=th,
            uj_total_thermal=uj(pow_["total_mw"], th),
            uj_core_dynamic=uj(pow_["core_dynamic_mw"], th),
            uj_core_total=uj(
                pow_["core_dynamic_mw"] + pow_["core_static_mw"], th),
        )

    # Full-net Horowitz PAR8×M3 (already in model JSON)
    full = model.get("full_mnist_model", {}).get("full_par8_M3", {})
    int16 = model.get("baseline_one_forward_uJ", {}).get("int16_mac")

    # Scale core-dynamic µJ from 16-hid slice → 160-hid by LE ratio ≈ 10
    # (structural; labeled as estimate). Prefer Horowitz for full-net absolute.
    le_scale = 160 / 16
    full_est_from_quartus = {
        "method": "scale DE2 core-dynamic µJ by N_hid 160/16 (coarse)",
        "fixed_n_ee_uj_core_dyn_scaled": (
            slice_energy["fixed_n_ee_2p30"]["uj_core_dynamic"] * le_scale),
        "adaptive_ee_uj_core_dyn_scaled": (
            slice_energy["adaptive_ee_5p11"]["uj_core_dynamic"] * le_scale),
        "note": (
            "Not a substitute for full-net Vivado/Quartus fit; use Horowitz "
            "PAR8×M3 for full-net Table VI primary absolute until full-net PnR."
        ),
    }

    # First vectorless run (12.5%) archived from earlier console:
    vectorless_12p5 = dict(
        total_mw=165.84,
        core_dynamic_mw=12.73,
        core_static_mw=98.75,
        io_mw=54.35,
        default_toggle="12.5% + vectorless",
        avg_toggle_Mtrans_s=2.478,
    )

    payload = dict(
        collected_at="2026-07-31",
        de2_115_quartus_powerplay=pow_,
        de2_115_prior_vectorless_12p5=vectorless_12p5,
        clock_Hz=f_clk,
        cycles_per_pass=cyc,
        resources=dict(
            logic_elements=3749,
            logic_pct=3.0,
            memory_bits=203680,
            memory_pct=5.0,
            dsp=0,
        ),
        slice_energy_uj=slice_energy,
        full_net_horowitz_par8_M3=dict(
            energy_per_inference_ee_uJ=full.get("energy_per_inference_ee_uJ"),
            energy_per_inference_full40_uJ=full.get(
                "energy_per_inference_full40_uJ"),
            energy_per_ensemble_pass_uJ=full.get("energy_per_ensemble_pass_uJ"),
            cycles_per_ensemble_pass=full.get("cycles_per_ensemble_pass"),
            a2_mean_passes=full.get("a2_mean_passes"),
            fixed_n_ee_uj=(
                full.get("energy_per_ensemble_pass_uJ", 0) * fixed_n_passes
                if full.get("energy_per_ensemble_pass_uJ") else None),
            int16_mac_forward_uJ=int16,
        ),
        full_net_quartus_scaled_estimate=full_est_from_quartus,
        table_vi_recommendation=dict(
            primary_full_net_row=(
                f"{(full.get('energy_per_ensemble_pass_uJ') or 0)*fixed_n_passes:.4f}"
            ),
            primary_basis="Horowitz×board-validated cycles; fixed-N EE 2.30 passes",
            empirical_slice_row_core_dyn=(
                f"{slice_energy['board_ee_30p25']['uj_core_dynamic']:.4f}"
            ),
            empirical_slice_basis=(
                "Quartus PowerPlay DE2-115 784–16–10 @ 25 MHz, board EE 30.25 passes; "
                "core-dynamic (vectorless, Low confidence)"
            ),
            vivado_artix7="still TBD (no Vivado on host); Quartus path used instead",
        ),
    )

    out = OUT_DIR / "energy_params.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Mirror a patch-friendly power.json emphasizing the empirical DE2 number
    # we will put in Table VI (core-dynamic board-EE), plus fixed-N full-net model.
    patch = dict(
        tool="quartus_powerplay_de2_115",
        part="EP4CE115F29C7",
        clk_mhz=25,
        mean_passes=board_ee_passes,
        cycles_per_pass=cyc,
        static_mw=pow_["core_static_mw"],
        dynamic_mw=pow_["core_dynamic_mw"],
        total_mw=pow_["total_mw"],
        io_mw=pow_["io_mw"],
        throughput_inf_s=slice_energy["board_ee_30p25"]["throughput_inf_s"],
        energy_uj_per_inference=slice_energy["board_ee_30p25"]["uj_core_dynamic"],
        energy_uj_total_thermal=slice_energy["board_ee_30p25"]["uj_total_thermal"],
        energy_uj_fixed_n_full_net_model=(
            payload["full_net_horowitz_par8_M3"]["fixed_n_ee_uj"]),
        confidence=pow_["confidence"],
        table_vi_row=(
            "DE2-115 784–16–10 slice, board EE "
            "(Quartus PowerPlay, Cyclone IV E)"
        ),
    )
    pdir = ROOT / "sena2026" / "rtl" / "_power"
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / "power.json").write_text(json.dumps(patch, indent=2), encoding="utf-8")
    (OUT_DIR / "power.json").write_text(json.dumps(patch, indent=2), encoding="utf-8")

    print(json.dumps(dict(
        total_mw=pow_["total_mw"],
        core_dyn_mw=pow_["core_dynamic_mw"],
        uj_board_ee_core_dyn=round(
            slice_energy["board_ee_30p25"]["uj_core_dynamic"], 4),
        uj_board_ee_total=round(
            slice_energy["board_ee_30p25"]["uj_total_thermal"], 4),
        uj_fixed_n_full_model=round(
            payload["full_net_horowitz_par8_M3"]["fixed_n_ee_uj"] or 0, 4),
        uj_adaptive_full_model=round(
            full.get("energy_per_inference_ee_uJ") or 0, 4),
        wrote=str(out),
    ), indent=2))


if __name__ == "__main__":
    main()
