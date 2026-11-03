"""
Host-side C1/C2 calibration driver for the Papilio Pro SENA slice.

Talks to papilio_calib_top over the board USB-UART (115200 8N1), measures
cycles/pass on real silicon, validates the closed-form cycle model, and
records wall-clock throughput. Optionally logs a USB power reading you type
in from a meter for the C2 energy figure.

Usage (on the FPGA PC):
  python papilio_calibrate.py --port COM5
  python papilio_calibrate.py --port COM5 --image path\to\digit.npy
  python papilio_calibrate.py --port COM5 --power-idle 0.92 --power-run 1.07

Requires: pip install pyserial numpy
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

try:
    import serial
except ImportError:
    raise SystemExit("pip install pyserial")

HERE = Path(__file__).resolve().parent
PKG = HERE.parent

N_IN = 784
N_HID = 16
N_X_BYTES = 98
F_CLK_HZ = 32_000_000

# closed-form model, validated bit-exactly by the Verilator TB on the dev PC
# (tb/build_and_run_tb.py measured exactly 807 cycles/pass):
# S_HID pipeline = N_IN+2 cycles, S_HFIRE = 1, S_OUT = N_HID+2, S_OFIRE = 1,
# S_VOTE = 1  ->  807. The board must reproduce this number exactly.
MODEL_CYCLES_PER_PASS = (N_IN + 2) + 1 + (N_HID + 2) + 1 + 1  # = 807


def open_port(name):
    return serial.Serial(name, 115200, timeout=5)


def cmd_version(ser):
    ser.write(b"V")
    r = ser.read(2)
    if len(r) != 2 or r[0] != 0xA5:
        raise RuntimeError(f"bad version reply: {r.hex()}")
    return r[1]


def cmd_load_image(ser, bits784):
    payload = np.packbits(
        np.asarray(bits784, dtype=np.uint8), bitorder="little").tobytes()
    assert len(payload) == N_X_BYTES
    ser.write(b"X" + payload)
    r = ser.read(1)
    if r != b"X":
        raise RuntimeError(f"bad X ack: {r.hex()}")


def cmd_set_n(ser, n_log2_hid, n_log2_out):
    ser.write(bytes([0x4E, n_log2_hid, n_log2_out]))
    r = ser.read(1)
    if r != b"\x4e":
        raise RuntimeError(f"bad N ack: {r.hex()}")


def cmd_run(ser, t_max, early_exit):
    t0 = time.perf_counter()
    ser.write(bytes([0x52, t_max & 0x3F, 1 if early_exit else 0]))
    r = ser.read(6)
    wall = time.perf_counter() - t0
    if len(r) != 6:
        raise RuntimeError(f"short R reply: {r.hex()}")
    pred = r[0]
    passes = r[1]
    cycles = int.from_bytes(r[2:6], "little")
    return dict(pred=pred, passes=passes, cycles=cycles, wall_s=wall)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True, help="e.g. COM5 or /dev/ttyUSB1")
    parser.add_argument("--image", default=None,
                        help=".npy file with 784 binary pixels; default random")
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--loop-seconds", type=float, default=10.0,
                        help="duration of the throughput/power loop; read "
                             "your USB power meter during this phase")
    parser.add_argument("--power-idle", type=float, default=None,
                        help="board idle power in W (USB meter reading)")
    parser.add_argument("--power-run", type=float, default=None,
                        help="board power in W while looping inference")
    args = parser.parse_args()

    meta_path = PKG / "papilio_slice.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    n_hid_l2 = int(meta.get("n_log2_hid", 8))
    n_out_l2 = int(meta.get("n_log2_out", 8))

    if args.image:
        bits = np.load(args.image).astype(np.uint8).ravel()[:N_IN]
    else:
        bits = (np.random.default_rng(2026).random(N_IN) < 0.19).astype(np.uint8)

    ser = open_port(args.port)
    ver = cmd_version(ser)
    print(f"board alive, fw version {ver}")
    cmd_set_n(ser, n_hid_l2, n_out_l2)
    cmd_load_image(ser, bits)

    # ---- 1. fixed-pass cycle calibration (no early exit) ------------------
    calib = []
    for t in (1, 2, 4, 8, 16, 32):
        r = cmd_run(ser, t, early_exit=False)
        cpp = r["cycles"] / max(r["passes"], 1)
        calib.append(dict(t_max=t, **r, cycles_per_pass=cpp))
        print(f"  T={t:2d}  passes={r['passes']:2d}  cycles={r['cycles']:7d} "
              f" cycles/pass={cpp:8.2f}  model={MODEL_CYCLES_PER_PASS}")
    measured_cpp = float(np.mean([c["cycles_per_pass"] for c in calib[2:]]))
    overhead = measured_cpp - MODEL_CYCLES_PER_PASS

    # ---- 2. early-exit behaviour ------------------------------------------
    ee_runs = [cmd_run(ser, 40, early_exit=True) for _ in range(args.runs)]
    passes = [r["passes"] for r in ee_runs]
    print(f"early-exit passes over {args.runs} runs: "
          f"mean={np.mean(passes):.2f} min={min(passes)} max={max(passes)}")

    # ---- 3. throughput / energy -------------------------------------------
    print(f"looping inference for {args.loop_seconds:.0f}s "
          f"— read the USB power meter NOW ...")
    t0 = time.perf_counter()
    n_loop = 0
    while time.perf_counter() - t0 < args.loop_seconds:
        cmd_run(ser, 40, early_exit=True)
        n_loop += 1
    wall = time.perf_counter() - t0
    inf_per_s = n_loop / wall
    mean_cycles = float(np.mean([r["cycles"] for r in ee_runs]))
    core_time_s = mean_cycles / F_CLK_HZ
    print(f"throughput incl. UART: {inf_per_s:.1f} inf/s; "
          f"core-only latency {core_time_s*1e6:.1f} us/inference")

    energy = None
    if args.power_idle is not None and args.power_run is not None:
        # C2: idle-subtracted power divided by inference rate
        p_dyn = args.power_run - args.power_idle
        energy = dict(
            power_idle_W=args.power_idle, power_run_W=args.power_run,
            dynamic_W=p_dyn,
            uj_per_inference_incl_uart=p_dyn / inf_per_s * 1e6,
            note="idle-subtracted board power / inference rate; UART overhead "
                 "included, so this is an UPPER bound on core energy")
        print(f"C2 energy: {energy['uj_per_inference_incl_uart']:.2f} "
              f"uJ/inference (upper bound incl. UART)")

    out = dict(
        board="Papilio Pro (xc6slx9-2)",
        f_clk_hz=F_CLK_HZ,
        model_cycles_per_pass=MODEL_CYCLES_PER_PASS,
        measured_cycles_per_pass=measured_cpp,
        model_error_cycles=overhead,
        fixed_pass_runs=calib,
        early_exit_passes=passes,
        throughput_inf_per_s_incl_uart=inf_per_s,
        core_latency_us=core_time_s * 1e6,
        energy=energy,
        slice_meta=meta,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    out_path = HERE / "papilio_calibration.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {out_path}  (copy this back to sena2026/results/ on the "
          f"main PC)")


if __name__ == "__main__":
    main()
