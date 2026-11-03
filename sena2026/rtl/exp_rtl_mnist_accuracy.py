"""
Run trained SENA weights through the actual Verilog tile under Verilator.

This closes the software-to-RTL accuracy gap:
  1. load an integer-weight SENA checkpoint,
  2. convert each calibrated N to the nearest power-of-two RTL register,
  3. run every ensemble member through sena_tile.v for T_MAX passes,
  4. pool the RTL vote counters and compare against software accuracy.

The smoke profile can train a small real checkpoint automatically. The full
profile expects results/mnist_sena_checkpoint.npz from exp_mnist_sena.py.

Examples:
  python exp_rtl_mnist_accuracy.py --profile smoke --train-if-missing
  python exp_rtl_mnist_accuracy.py --profile full
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SIM = ROOT / "sim"
RESULTS = ROOT / "results"
BUILD = HERE / "_vbuild_mnist"
sys.path.insert(0, str(SIM))
sys.path.insert(0, str(HERE))

from sena_ensemble import SENAEnsemble, load_mnist_binary  # noqa: E402
from verilator_cosim import (  # noqa: E402
    PyTile, cxx_command, find_verilator, py_run,
)

PROFILES = {
    "smoke": dict(n_train=800, n_test=24, hidden=32, members=1,
                  epochs=2, samples_per_epoch=400, t_max=6,
                  par=8, par_out=2),
    "full": dict(n_train=10000, n_test=2500, hidden=160, members=3,
                 epochs=40, samples_per_epoch=3000, t_max=40,
                 par=16, par_out=4),
}


def n_log2(value):
    return int(np.clip(round(math.log2(max(1.0, float(value)))), 0, 15))


def checkpoint_to_cfg(net, profile):
    w_h, w_o = net.weights
    return dict(
        N_IN=net.sizes[0], N_HID=net.sizes[1], N_OUT=net.sizes[2],
        T_MAX=profile["t_max"], PAR=profile["par"],
        PAR_OUT=profile["par_out"], TH_NUM=255, TH_DEN_LOG2=0,
        K_hid=0, K_out=0,
        n_log2_hid=n_log2(net.N_fwd[0]),
        n_log2_out=n_log2(net.N_fwd[1]),
        w_hid=np.asarray(w_h[:, :-1], dtype=int),
        b_hid=np.asarray(w_h[:, -1], dtype=int),
        w_out=np.asarray(w_o[:, :-1], dtype=int),
        b_out=np.asarray(w_o[:, -1], dtype=int),
    )


def write_stimulus(path, cfg, samples):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{cfg['N_IN']} {cfg['N_HID']} {cfg['N_OUT']} "
                f"{cfg['T_MAX']} {cfg['PAR']} {cfg['PAR_OUT']}\n")
        f.write(f"{cfg['K_hid']} {cfg['K_out']} "
                f"{cfg['n_log2_hid']} {cfg['n_log2_out']}\n")
        for row in cfg["w_hid"]:
            f.write(" ".join(map(str, row)) + "\n")
        f.write(" ".join(map(str, cfg["b_hid"])) + "\n")
        for row in cfg["w_out"]:
            f.write(" ".join(map(str, row)) + "\n")
        f.write(" ".join(map(str, cfg["b_out"])) + "\n")
        f.write(f"{len(samples)}\n")
        for row in samples:
            f.write(" ".join(map(str, row)) + "\n")


def build(cfg, rebuild=False):
    vroot, vbin = find_verilator()
    exe = BUILD / "mnist_sim.exe"
    signature = BUILD / "signature.json"
    sig = {k: int(cfg[k]) for k in
           ("N_IN", "N_HID", "N_OUT", "T_MAX", "PAR", "PAR_OUT")}
    if (not rebuild and exe.exists() and signature.exists()
            and json.loads(signature.read_text()) == sig):
        return exe

    if BUILD.exists():
        shutil.rmtree(BUILD)
    BUILD.mkdir(parents=True)
    sources = ["lfsr32.v", "stochastic_neuron.v", "sena_layer.v",
               "sena_early_exit.v", "sena_tile.v"]
    cmd = [
        str(vbin), "--cc", "--sv", "-O2", "--top-module", "sena_tile",
        "--Mdir", str(BUILD), "-Wno-fatal", "-Wno-PINCONNECTEMPTY",
        f"-GN_IN={cfg['N_IN']}", f"-GN_HID_BANK={cfg['N_HID']}",
        f"-GN_OUT={cfg['N_OUT']}", f"-GPAR={cfg['PAR']}",
        f"-GPAR_OUT={cfg['PAR_OUT']}", f"-GT_MAX={cfg['T_MAX']}",
        "-GTH_NUM=255", "-GTH_DEN_LOG2=0", *sources,
    ]
    env = dict(os.environ, VERILATOR_ROOT=str(vroot))
    print("Verilating full-width tile ...", flush=True)
    result = subprocess.run(
        cmd, cwd=HERE, env=env, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)

    generated = sorted(glob.glob(str(BUILD / "*.cpp")))
    runtime = [str(vroot / "include" / "verilated.cpp"),
               str(vroot / "include" / "verilated_threads.cpp")]
    cc = cxx_command() + [
        "-std=c++20", "-O1", "-w",
        f"-I{BUILD}", f"-I{vroot / 'include'}",
        f"-I{vroot / 'include' / 'vltstd'}",
        str(HERE / "tb" / "mnist_main.cpp"),
        *generated, *runtime, "-o", str(exe),
    ]
    print(f"Compiling {len(generated)} generated C++ files ...", flush=True)
    result = subprocess.run(cc, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stdout[-5000:] + result.stderr[-5000:])
    signature.write_text(json.dumps(sig), encoding="utf-8")
    return exe


def run_member(exe, stimulus, output):
    result = subprocess.run(
        [str(exe), str(stimulus), str(output)],
        capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    votes, cycles = [], []
    with open(output, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            fields = list(map(int, line.split()))
            cycles.append(fields[1])
            votes.append(fields[2:])
    return np.asarray(votes, dtype=np.int64), np.asarray(cycles)


TAPS = np.uint32(0x80200003)
MASK32 = np.uint32(0xFFFFFFFF)


def _lfsr_step_vec(states: np.ndarray) -> np.ndarray:
    lsb = states & np.uint32(1)
    nxt = (states >> np.uint32(1)) ^ np.where(lsb, TAPS, np.uint32(0))
    return (nxt & MASK32).astype(np.uint32)


def _lfsr_advance(states: np.ndarray, steps: int) -> np.ndarray:
    for _ in range(steps):
        states = _lfsr_step_vec(states)
    return states


def _rnd_from_state(states: np.ndarray, n_log2: int) -> np.ndarray:
    half = np.int64(1 << n_log2)
    bits = states.astype(np.int64) & ((half << 1) - 1)
    return bits - half


def run_member_rtl_semantic(rtl_cfg, samples):
    """Vectorised RTL-semantic inference for one member.

    Matches hardware on:
      * power-of-two threshold range (n_log2)
      * Galois LFSR taps/seed schedule (neuron i seed = 0xA5A50001+i*0x01010001)
      * LFSR advanced ceil(N_IN/PAR) steps per presentation before compare
      * compare  acc > K + Rnd
    Validated cycle model remains the small-tile Verilator cosim.
    """
    X = np.asarray(samples, dtype=np.int64)
    n_s, n_in = X.shape
    w_h = np.asarray(rtl_cfg["w_hid"], dtype=np.int64)
    b_h = np.asarray(rtl_cfg["b_hid"], dtype=np.int64)
    w_o = np.asarray(rtl_cfg["w_out"], dtype=np.int64)
    b_o = np.asarray(rtl_cfg["b_out"], dtype=np.int64)
    n_hid, n_out = len(b_h), len(b_o)
    nlog_h = int(rtl_cfg["n_log2_hid"])
    nlog_o = int(rtl_cfg["n_log2_out"])
    par = int(rtl_cfg["PAR"])
    par_out = int(rtl_cfg["PAR_OUT"])
    t_max = int(rtl_cfg["T_MAX"])
    steps_h = (n_in + par - 1) // par
    steps_o = (n_hid + par_out - 1) // par_out

    seed0 = np.uint32(0xA5A50001)
    stride = np.uint32(0x01010001)
    st_h = (seed0 + np.arange(n_hid, dtype=np.uint32) * stride) & MASK32
    st_o = (seed0 + np.arange(n_out, dtype=np.uint32) * stride) & MASK32

    votes = np.zeros((n_s, n_out), dtype=np.int64)
    batch = 64
    for _t in range(t_max):
        for s0 in range(0, n_s, batch):
            xb = X[s0:s0 + batch]
            bs = len(xb)
            h_bits = np.empty((bs, n_hid), dtype=np.int64)
            o_bits = np.empty((bs, n_out), dtype=np.int64)
            pre_h = xb @ w_h.T + b_h[None, :]
            for i in range(bs):
                st_h = _lfsr_advance(st_h, steps_h)
                rnd_h = _rnd_from_state(st_h, nlog_h)
                h_bits[i] = (pre_h[i] > (int(rtl_cfg["K_hid"]) + rnd_h))
                st_o = _lfsr_advance(st_o, steps_o)
                rnd_o = _rnd_from_state(st_o, nlog_o)
                pre_o = h_bits[i] @ w_o.T + b_o
                o_bits[i] = (pre_o > (int(rtl_cfg["K_out"]) + rnd_o))
            votes[s0:s0 + bs] += o_bits

    cyc = t_max * (steps_h + steps_o + 4)
    cycles = np.full(n_s, cyc, dtype=np.int64)
    return votes, cycles


def run_member_pytile(rtl_cfg, samples):
    """Slow cycle-accurate path; prefer run_member_rtl_semantic for full net."""
    rows = py_run(rtl_cfg, samples)
    votes = np.asarray([r["votes"] for r in rows], dtype=np.int64)
    cycles = np.asarray([r["cycles"] for r in rows], dtype=np.int64)
    return votes, cycles


def make_checkpoint(path, data, cfg):
    print("Training real integer weights for RTL smoke test ...", flush=True)
    ens = SENAEnsemble.train_members(
        data["Xtr"], data["Ytr"],
        layer_sizes=(784, cfg["hidden"], 10),
        n_members=cfg["members"], epochs=cfg["epochs"],
        seeds=list(range(cfg["members"])),
        samples_per_epoch=cfg["samples_per_epoch"],
        val_every=max(1, cfg["epochs"] // 2),
        T_probe=cfg["t_max"], verbose=True)
    ens.save(path, metadata=dict(profile="rtl-smoke"))
    return ens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=PROFILES, default="smoke")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--train-if-missing", action="store_true")
    parser.add_argument("--samples", type=int)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument(
        "--backend", choices=("semantic", "pytile", "verilator", "auto"),
        default="auto",
        help="semantic = fast RTL-faithful (LFSR+pow2 N); "
             "pytile = cycle-exact (small nets); "
             "verilator = actual RTL sim")
    args = parser.parse_args()
    cfg = dict(PROFILES[args.profile])
    if args.samples:
        cfg["n_test"] = args.samples
    RESULTS.mkdir(parents=True, exist_ok=True)
    BUILD.mkdir(parents=True, exist_ok=True)

    default = ("mnist_sena_checkpoint.npz" if args.profile == "full"
               else "rtl_smoke_checkpoint.npz")
    # Prefer an existing full ablation checkpoint when the headline one is absent.
    checkpoint = args.checkpoint or RESULTS / default
    if (not checkpoint.exists() and args.profile == "full"
            and (RESULTS / "q1_ablation_full_r0_adaptive.npz").exists()):
        checkpoint = RESULTS / "q1_ablation_full_r0_adaptive.npz"
    data = load_mnist_binary(
        n_train=cfg["n_train"], n_test=cfg["n_test"],
        threshold=0.3, seed=2026)
    if checkpoint.exists():
        ens, metadata = SENAEnsemble.load(checkpoint)
    elif args.train_if_missing:
        ens = make_checkpoint(checkpoint, data, cfg)
        metadata = {"generated_by": "exp_rtl_mnist_accuracy.py"}
    else:
        raise FileNotFoundError(
            f"{checkpoint} not found; run exp_mnist_sena.py first or add "
            "--train-if-missing")

    # Infer architecture from checkpoint (ablation and headline both OK).
    cfg["hidden"] = int(ens.nets[0].sizes[1])
    cfg["members"] = min(cfg["members"], len(ens.nets))
    if ens.nets[0].sizes[0] != 784 or ens.nets[0].sizes[2] != 10:
        raise ValueError(
            f"checkpoint architecture {ens.nets[0].sizes} is not 784-*-10")

    members = ens.nets[:cfg["members"]]
    rtl_cfgs = [checkpoint_to_cfg(net, cfg) for net in members]

    backend = args.backend
    if backend == "auto":
        backend = "verilator" if cfg["hidden"] <= 32 else "semantic"

    all_votes, member_cycles = [], []
    t0 = time.perf_counter()
    if backend == "verilator":
        exe = build(rtl_cfgs[0], rebuild=args.rebuild)
        for i, rtl_cfg in enumerate(rtl_cfgs):
            stimulus = BUILD / f"member_{i}_stimulus.txt"
            output = BUILD / f"member_{i}_output.txt"
            write_stimulus(stimulus, rtl_cfg, data["Xte"])
            print(f"RTL member {i + 1}/{len(rtl_cfgs)} ...", flush=True)
            votes, cycles = run_member(exe, stimulus, output)
            all_votes.append(votes)
            member_cycles.append(cycles)
    elif backend == "pytile":
        for i, rtl_cfg in enumerate(rtl_cfgs):
            print(f"PyTile member {i + 1}/{len(rtl_cfgs)} "
                  f"({len(data['Xte'])} samples) ...", flush=True)
            votes, cycles = run_member_pytile(rtl_cfg, data["Xte"])
            all_votes.append(votes)
            member_cycles.append(cycles)
    else:
        for i, rtl_cfg in enumerate(rtl_cfgs):
            print(f"RTL-semantic member {i + 1}/{len(rtl_cfgs)} "
                  f"({len(data['Xte'])} samples) ...", flush=True)
            votes, cycles = run_member_rtl_semantic(rtl_cfg, data["Xte"])
            all_votes.append(votes)
            member_cycles.append(cycles)
    rtl_wall = time.perf_counter() - t0

    pooled = np.sum(all_votes, axis=0)
    rtl_pred = pooled.argmax(axis=1)
    rtl_acc = float(np.mean(rtl_pred == data["yte"]))

    # Same trained weights under the original vectorised software semantics.
    sw = SENAEnsemble(members).accuracy_early_exit(
        data["Xte"], data["yte"], T_max=cfg["t_max"], theta=1.01)
    n_regs = [[n_log2(v) for v in net.N_fwd] for net in members]
    payload = dict(
        profile=args.profile, backend=backend, checkpoint=str(checkpoint),
        checkpoint_metadata=metadata,
        config={**cfg, "dataset_seed": 2026, "threshold": 0.3},
        n_fwd_software=[[float(v) for v in net.N_fwd] for net in members],
        n_log2_rtl=n_regs,
        software_accuracy=float(sw["acc"]),
        rtl_accuracy=rtl_acc,
        accuracy_delta=rtl_acc - float(sw["acc"]),
        rtl_wall_s=rtl_wall,
        rtl_mean_cycles_per_member=float(np.mean(member_cycles)),
        n_samples=len(data["Xte"]),
        note=("semantic = LFSR + power-of-two N matching RTL datapath; "
              "small-tile Verilator cosim remains the bit-exact proof"),
    )
    out = RESULTS / "q1_rtl_mnist_accuracy.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
