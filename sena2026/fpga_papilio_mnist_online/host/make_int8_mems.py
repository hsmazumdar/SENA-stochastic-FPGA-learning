"""
Build int8 w_hid.mem / w_out.mem for the Papilio MNIST online slice
from the A2 checkpoint (member 0, first 16 hidden units).

ROM/RAM word layout matches calib: index 0 weight in the LSBs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
RTL = PKG / "rtl"
SENA = PKG.parent
DEFAULT_CKPT = SENA / "results" / "q1_ablation_full_r0_adaptive.npz"

N_HID_SLICE = 8  # LX9 fit: half slice vs calib's 16
W_BITS = 8
MASK = (1 << W_BITS) - 1


def to_hex_word(values):
    word = 0
    for pos, v in enumerate(values):
        word |= (int(v) & MASK) << (pos * W_BITS)
    return format(word, "0{}x".format(len(values) * W_BITS // 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(DEFAULT_CKPT))
    ap.add_argument("--zero", action="store_true", help="zero-init instead of ckpt")
    args = ap.parse_args()

    n_in, n_out = 784, 10
    if args.zero:
        W1s = np.zeros((N_HID_SLICE, n_in + 1), dtype=np.int64)
        W2s = np.zeros((n_out, N_HID_SLICE + 1), dtype=np.int64)
        meta = dict(init="zero", slice_hidden=N_HID_SLICE)
        n_log2_hid = n_log2_out = 8
    else:
        data = np.load(args.checkpoint, allow_pickle=False)
        manifest = json.loads(str(data["manifest_json"]))
        member = manifest["members"][0]
        W1 = np.asarray(data["m0_w0"], dtype=np.int64)
        W2 = np.asarray(data["m0_w1"], dtype=np.int64)
        W1s = np.clip(W1[:N_HID_SLICE], -128, 127)
        W2s = np.clip(
            np.concatenate([W2[:, :N_HID_SLICE], W2[:, -1:]], axis=1), -128, 127
        )
        n_fwd = member["N_fwd"]
        n_log2_hid = int(round(np.log2(max(n_fwd[0], 1))))
        n_log2_out = int(round(np.log2(max(n_fwd[1], 1))))
        meta = dict(
            checkpoint=str(args.checkpoint),
            slice_hidden=N_HID_SLICE,
            n_fwd=n_fwd,
            n_log2_hid=n_log2_hid,
            n_log2_out=n_log2_out,
            init="ckpt_int8",
        )

    lines_hid = [to_hex_word(W1s[:, i]) for i in range(n_in + 1)]
    lines_out = [to_hex_word(W2s[:, j]) for j in range(N_HID_SLICE + 1)]
    RTL.mkdir(parents=True, exist_ok=True)
    (RTL / "w_hid.mem").write_text("\n".join(lines_hid) + "\n", encoding="ascii")
    (RTL / "w_out.mem").write_text("\n".join(lines_out) + "\n", encoding="ascii")
    (RTL / "mnist_online_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print("wrote", RTL / "w_hid.mem", "lines", len(lines_hid))
    print("wrote", RTL / "w_out.mem", "lines", len(lines_out))
    print(meta)


if __name__ == "__main__":
    main()
