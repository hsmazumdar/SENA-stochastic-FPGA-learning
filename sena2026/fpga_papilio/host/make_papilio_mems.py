"""
Generate w_hid.mem / w_out.mem for the Papilio calibration slice from the
trained A2 checkpoint (member 0, first 16 hidden neurons).

Also writes papilio_slice.json: the n_log2 register values and a software
reference so the host script can sanity-check board predictions.

Run from anywhere:
  python make_papilio_mems.py
  python make_papilio_mems.py --checkpoint path\to\ensemble.npz
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PKG = HERE.parent                       # fpga_papilio/
RTL = PKG / "rtl"
SENA = PKG.parent                       # sena2026/
DEFAULT_CKPT = SENA / "results" / "q1_ablation_full_r0_adaptive.npz"

N_HID_SLICE = 16
W_BITS = 16
MASK = (1 << W_BITS) - 1


def to_hex_word(values):
    """Concatenate int16 values into one hex word, index 0 in the LSBs."""
    word = 0
    for pos, v in enumerate(values):
        word |= (int(v) & MASK) << (pos * W_BITS)
    return format(word, "0{}x".format(len(values) * W_BITS // 4))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(DEFAULT_CKPT))
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    data = np.load(ckpt, allow_pickle=False)
    manifest = json.loads(str(data["manifest_json"]))
    member = manifest["members"][0]
    sizes = member["sizes"]                       # [784, H, 10]
    n_in, n_hid_full, n_out = sizes
    W1 = np.asarray(data["m0_w0"], dtype=np.int64)   # (H, n_in+1), bias last
    W2 = np.asarray(data["m0_w1"], dtype=np.int64)   # (10, H+1)

    hid_idx = np.arange(N_HID_SLICE)
    W1s = W1[hid_idx]                                # (16, 785)
    # output layer: keep the columns of the selected hidden units + bias
    W2s = np.concatenate([W2[:, hid_idx], W2[:, -1:]], axis=1)  # (10, 17)

    # clip to int16 range (checkpoints are quantised already; belt & braces)
    W1s = np.clip(W1s, -(1 << 15), (1 << 15) - 1)
    W2s = np.clip(W2s, -(1 << 15), (1 << 15) - 1)

    # ROM layout: rom_hid[i] = weights of input i for all 16 neurons
    lines_hid = [to_hex_word(W1s[:, i]) for i in range(n_in + 1)]
    lines_out = [to_hex_word(W2s[:, j]) for j in range(N_HID_SLICE + 1)]
    (RTL / "w_hid.mem").write_text("\n".join(lines_hid) + "\n",
                                   encoding="ascii")
    (RTL / "w_out.mem").write_text("\n".join(lines_out) + "\n",
                                   encoding="ascii")

    n_fwd = member["N_fwd"]
    n_log2_hid = int(round(np.log2(max(n_fwd[0], 1))))
    n_log2_out = int(round(np.log2(max(n_fwd[1], 1))))

    meta = dict(
        checkpoint=str(ckpt),
        sizes=sizes,
        slice_hidden=N_HID_SLICE,
        n_fwd=n_fwd,
        n_log2_hid=n_log2_hid,
        n_log2_out=n_log2_out,
        note=("16-hidden slice of member 0; accuracy of the slice is NOT "
              "the paper accuracy — this design exists to calibrate "
              "cycles/pass, f_clk and power on real silicon."),
    )
    (PKG / "papilio_slice.json").write_text(json.dumps(meta, indent=2),
                                            encoding="utf-8")
    print(f"wrote {RTL/'w_hid.mem'} ({len(lines_hid)} words x "
          f"{N_HID_SLICE*W_BITS}b)")
    print(f"wrote {RTL/'w_out.mem'} ({len(lines_out)} words x "
          f"{n_out*W_BITS}b)")
    print(f"n_log2_hid={n_log2_hid} n_log2_out={n_log2_out} "
          f"(from N_fwd={n_fwd})")


if __name__ == "__main__":
    main()
