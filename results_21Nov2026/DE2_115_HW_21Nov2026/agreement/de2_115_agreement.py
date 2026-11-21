"""DE2-115 board-vs-RTL agreement for the 784-16-10 calib slice.

Hardware re-seeds LFSRs on every start (calib_core.v), so each image is
independent. All inputs/outputs live under this folder on K:.

Usage:
  python de2_115_agreement.py --port COM8 --n 2500
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import serial

HERE = Path(__file__).resolve().parent
N_IN, N_HID, N_OUT = 784, 16, 10
TAPS = 0x80200003
M32 = 0xFFFFFFFF


def load_mem(path: Path, nw: int) -> np.ndarray:
    rows = []
    for ln in path.read_text().split():
        v = int(ln, 16)
        rows.append([((v >> (16 * k)) & 0xFFFF) for k in range(nw)])
    a = np.array(rows, dtype=np.int64)
    return np.where(a >= 0x8000, a - 0x10000, a)


def step(x: int) -> int:
    return ((x >> 1) ^ (TAPS if x & 1 else 0)) & M32


def jump_matrix(n: int):
    cols = []
    for b in range(32):
        x = 1 << b
        for _ in range(n):
            x = step(x)
        cols.append(x)
    return cols


def apply(cols, x: int) -> int:
    r = 0
    b = 0
    while x:
        if x & 1:
            r ^= cols[b]
        x >>= 1
        b += 1
    return r


def compose(a, b):
    return [apply(b, c) for c in a]


class ReseedModel:
    def __init__(self, wh, wo, nh=9, no=8):
        self.wh, self.wo = wh, wo
        self.nh, self.no = nh, no
        self.J786 = jump_matrix(786)
        self.J19 = jump_matrix(19)
        self.J807 = compose(compose(self.J786, self.J19), jump_matrix(2))

    @staticmethod
    def rnd(r, n):
        half = 1 << n
        return (r & ((half << 1) - 1)) - half

    def run(self, x, t_max=40, ee=True):
        x = np.asarray(x).astype(bool)
        acc_h = self.wh[:N_IN][x].sum(0) + self.wh[N_IN]
        votes = np.zeros(N_OUT, dtype=np.int64)
        lh = [(0xA5A50001 + k * 0x01010001) & M32 for k in range(N_HID)]
        lo = [(0x5A5A1001 + k * 0x02020001) & M32 for k in range(N_OUT)]
        passes = 0
        while True:
            hh = [apply(self.J786, v) for v in lh]
            h = np.array([acc_h[k] > self.rnd(hh[k], self.nh) for k in range(N_HID)])
            acc_o = self.wo[:N_HID][h].sum(0) + self.wo[N_HID]
            oo = [apply(compose(self.J786, self.J19), v) for v in lo]
            o = np.array([acc_o[k] > self.rnd(oo[k], self.no) for k in range(N_OUT)])
            votes = (votes + o) & 0xFFF
            passes += 1
            s = int(votes.sum())
            mx = int(votes.max())
            pred = int(np.argmax(votes)) if mx > 0 else 0
            conf = s != 0 and (mx << 4) >= 9 * s
            lh = [apply(self.J807, v) for v in lh]
            lo = [apply(self.J807, v) for v in lo]
            if (ee and conf) or passes >= t_max:
                break
        return pred, passes, 807 * passes


def board_run(port: str, Xte, yte, n: int, out_csv: Path):
    s = serial.Serial(port, 115200, timeout=5)
    time.sleep(0.3)
    s.reset_input_buffer()
    s.write(b"V")
    r = s.read(2)
    if r != b"\xa5\x01":
        raise SystemExit(f"bad version: {r.hex()}")
    s.write(bytes([0x4E, 9, 8]))
    if s.read(1) != b"N":
        raise SystemExit("bad N ack")
    print("board alive, N=9/8")
    f = open(out_csv, "w", newline="", encoding="utf-8")
    w = csv.writer(f)
    w.writerow(["idx", "label", "pred", "passes", "cycles"])
    t0 = time.time()
    for i in range(n):
        p = np.packbits(Xte[i].astype(np.uint8), bitorder="little").tobytes()
        s.write(b"X" + p)
        ack = s.read(1)
        if ack != b"X":
            raise SystemExit(f"X ack fail at {i}: {ack.hex() if ack else 'empty'}")
        s.write(bytes([0x52, 40, 1]))
        r = s.read(6)
        if len(r) != 6:
            raise SystemExit(f"R short at {i}: {r.hex()}")
        w.writerow([i, int(yte[i]), r[0], r[1], int.from_bytes(r[2:], "little")])
        f.flush()
        if i % 250 == 0 or i == n - 1:
            print(f"  board {i+1}/{n}  {time.time()-t0:.0f}s", flush=True)
    f.close()
    s.close()
    print("wrote", out_csv)


def model_run(Xte, yte, n: int, out_csv: Path, wh, wo):
    m = ReseedModel(wh, wo, 9, 8)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["idx", "label", "pred", "passes", "cycles"])
        t0 = time.time()
        for i in range(n):
            pred, ps, cyc = m.run(Xte[i], t_max=40, ee=True)
            w.writerow([i, int(yte[i]), pred, ps, cyc])
            if i % 500 == 0 or i == n - 1:
                print(f"  model {i+1}/{n}  {time.time()-t0:.0f}s", flush=True)
    print("wrote", out_csv)


def compare(board_csv: Path, model_csv: Path, n: int) -> dict:
    b = np.genfromtxt(board_csv, delimiter=",", names=True, dtype=None, encoding="utf-8")
    m = np.genfromtxt(model_csv, delimiter=",", names=True, dtype=None, encoding="utf-8")
    b, m = b[:n], m[:n]
    pred_ok = int(np.sum(b["pred"] == m["pred"]))
    pass_ok = int(np.sum(b["passes"] == m["passes"]))
    cyc_ok = int(np.sum(b["cycles"] == m["cycles"]))
    cpp_ok = bool(np.all(b["cycles"] == b["passes"] * 807))
    acc_b = 100.0 * float(np.mean(b["pred"] == b["label"]))
    acc_m = 100.0 * float(np.mean(m["pred"] == m["label"]))
    mism = [int(i) for i in np.where(b["pred"] != m["pred"])[0][:10]]
    return dict(
        N=n,
        pred_agree_pct=100.0 * pred_ok / n,
        pred_mismatches=n - pred_ok,
        first_mismatches=mism,
        passes_agree_pct=100.0 * pass_ok / n,
        cycles_agree_pct=100.0 * cyc_ok / n,
        cycles_per_pass_all_807=cpp_ok,
        board_acc_pct=round(acc_b, 2),
        model_acc_pct=round(acc_m, 2),
        board_mean_passes=float(np.mean(b["passes"])),
        model_mean_passes=float(np.mean(m["passes"])),
        board_csv=str(board_csv),
        model_csv=str(model_csv),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM8")
    ap.add_argument("--n", type=int, default=2500)
    ap.add_argument("--skip-board", action="store_true")
    ap.add_argument("--skip-model", action="store_true")
    args = ap.parse_args()

    stamp = time.strftime("%Y%m%d_%H%M%S")
    run = HERE / f"run_{stamp}"
    run.mkdir()

    data = np.load(HERE / "mnist_bin_t0.3_tr10000_te2500_s2026.npz")
    Xte, yte = data["Xte"], data["yte"]
    n = min(args.n, len(Xte))
    wh = load_mem(HERE / "w_hid.mem", N_HID)
    wo = load_mem(HERE / "w_out.mem", N_OUT)
    print(f"weights hid {wh.shape} out {wo.shape}; n={n}; out={run}")

    board_csv = run / "board_preds.csv"
    model_csv = run / "model_preds.csv"
    if not args.skip_board:
        board_run(args.port, Xte, yte, n, board_csv)
    if not args.skip_model:
        model_run(Xte, yte, n, model_csv, wh, wo)

    agr = compare(board_csv, model_csv, n)
    (run / "agreement.json").write_text(json.dumps(agr, indent=2), encoding="utf-8")
    print(json.dumps(agr, indent=2))
    print("AGREEMENT", "PASS" if agr["pred_mismatches"] == 0 and agr["cycles_per_pass_all_807"] else "FAIL")


if __name__ == "__main__":
    main()
