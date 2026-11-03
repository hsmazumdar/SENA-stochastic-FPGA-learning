"""
Prepare MNIST vectors + integer weight hex for Vivado SAIF power TB.

Writes under rtl/power/stimuli/:
  w_hid.mem, b_hid.mem, w_out.mem, b_out.mem   (signed 16-bit hex)
  mnist_x.mem                                   (196 hex digits = 784 bits / image)
  mnist_y.mem                                   (decimal labels, optional)
  stimuli_meta.json

Usage:
  python prepare_power_stimuli.py --n-images 1000
  python prepare_power_stimuli.py --n-images 200 --quick-train
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RTL = HERE.parent
ROOT = RTL.parent
SIM = ROOT / "sim"
STIM = HERE / "stimuli"
sys.path.insert(0, str(SIM))

from sena_ensemble import SENAEnsemble, load_mnist_binary  # noqa: E402
from stochastic_core import StochasticConfig, StochasticNet  # noqa: E402


def n_log2(value: float) -> int:
    return int(np.clip(round(math.log2(max(1.0, float(value)))), 0, 15))


def write_hex_signed16(path: Path, arr: np.ndarray):
    """One 4-digit two's-complement hex word per line."""
    flat = np.asarray(arr, dtype=np.int64).ravel()
    with path.open("w", encoding="utf-8") as f:
        for v in flat:
            v16 = int(v) & 0xFFFF
            f.write(f"{v16:04x}\n")


def write_mnist_hex(path: Path, X: np.ndarray):
    """Each line: 196 hex digits packing 784 bits MSB-first."""
    with path.open("w", encoding="utf-8") as f:
        for row in X:
            bits = "".join("1" if int(b) else "0" for b in row)
            # pad to multiple of 4
            while len(bits) % 4:
                bits = "0" + bits
            hexstr = "".join(
                f"{int(bits[i:i+4], 2):x}" for i in range(0, len(bits), 4)
            )
            f.write(hexstr + "\n")


def train_or_load(args):
    ckpt_candidates = [
        ROOT / "results" / "mnist_sena_checkpoint.npz",
        ROOT / "results" / "q1_ablation_full_r0_adaptive.npz",
        ROOT / "results" / "q1_ablation_full_r0_fixedN.npz",
    ]
    for ckpt in ckpt_candidates:
        if ckpt.exists():
            print(f"Loading checkpoint {ckpt}")
            loaded = SENAEnsemble.load(ckpt)
            return loaded[0] if isinstance(loaded, tuple) else loaded

    print("No checkpoint found — training a compact ensemble for power stimuli...")
    data = load_mnist_binary(
        n_train=args.n_train, n_test=max(args.n_images, 200),
        threshold=0.3, seed=2026, dataset="mnist",
        cache_dir=str(ROOT / "data"),
    )
    nets = []
    epochs = 2 if args.quick_train else 8
    spe = min(1500, len(data["Xtr"]))
    for m in range(args.members):
        cfg = StochasticConfig(
            layer_sizes=(784, args.hidden, 10),
            seed=2026 + m,
            weight_bits=16,
            rng_kind="numpy",
        )
        net = StochasticNet(cfg)
        # light train on a subset each epoch
        for ep in range(epochs):
            rng = np.random.default_rng(2026 + m + ep)
            idx = rng.choice(len(data["Xtr"]), size=spe, replace=False)
            net.train_epoch(data["Xtr"][idx], data["Ytr"][idx])
        nets.append(net)
        print(f"  member {m}: trained {epochs} epochs × {spe} samples")
    ens = SENAEnsemble(nets)
    out = ROOT / "results" / "power_stimuli_checkpoint.npz"
    ens.save(out, metadata=dict(purpose="vivado_saif_power", quick=args.quick_train))
    print(f"Saved {out}")
    return ens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-images", type=int, default=1000)
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--hidden", type=int, default=160)
    ap.add_argument("--members", type=int, default=1,
                    help="export member-0 by default (tile is one member)")
    ap.add_argument("--member-index", type=int, default=0)
    ap.add_argument("--quick-train", action="store_true")
    ap.add_argument("--passes", type=int, default=3,
                    help="fixed-N-style passes per image in TB")
    args = ap.parse_args()
    STIM.mkdir(parents=True, exist_ok=True)

    ens = train_or_load(args)
    net = ens.nets[min(args.member_index, len(ens.nets) - 1)]
    w_h, w_o = net.weights
    w_hid = np.asarray(w_h[:, :-1], dtype=np.int64)
    b_hid = np.asarray(w_h[:, -1], dtype=np.int64)
    w_out = np.asarray(w_o[:, :-1], dtype=np.int64)
    b_out = np.asarray(w_o[:, -1], dtype=np.int64)

    data = load_mnist_binary(
        n_train=100, n_test=max(args.n_images, 200),
        threshold=0.3, seed=2026, dataset="mnist",
        cache_dir=str(ROOT / "data"),
    )
    X = data["Xte"][: args.n_images]
    y = data["yte"][: args.n_images]

    write_hex_signed16(STIM / "w_hid.mem", w_hid)
    write_hex_signed16(STIM / "b_hid.mem", b_hid)
    write_hex_signed16(STIM / "w_out.mem", w_out)
    write_hex_signed16(STIM / "b_out.mem", b_out)
    write_mnist_hex(STIM / "mnist_x.mem", X)
    # Packed binary for $fread in sena_tb (98 bytes = 784 bits per image)
    with (STIM / "mnist_x.bin").open("wb") as f:
        for row in X:
            bits = np.asarray(row, dtype=np.uint8).ravel()
            # pack MSB-first into bytes
            packed = np.packbits(bits, bitorder="big")
            # packbits yields ceil(784/8)=98 bytes
            f.write(packed.tobytes())
    with (STIM / "mnist_y.mem").open("w", encoding="utf-8") as f:
        for lab in y:
            f.write(f"{int(lab)}\n")

    meta = dict(
        n_images=int(len(X)),
        n_in=784, n_hid=int(w_hid.shape[0]), n_out=10,
        par=8, par_out=2, t_max=40,
        passes_per_image=args.passes,
        n_log2_hid=n_log2(net.N_fwd[0]),
        n_log2_out=n_log2(net.N_fwd[1]),
        cycles_per_pass=int(math.ceil(784 / 8) + math.ceil(160 / 2) + 4),
        mean_passes_fixed_n=2.30,
        mean_passes_adaptive=5.11,
        M=3,
        note=(
            "EnergyIssueFix.docx suggested 807 cycles (Papilio 16-hid slice). "
            "Full 784-160-10 PAR8 uses verified 182 cycles/member/pass."
        ),
    )
    (STIM / "stimuli_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    print(f"Wrote stimuli to {STIM}")


if __name__ == "__main__":
    main()
