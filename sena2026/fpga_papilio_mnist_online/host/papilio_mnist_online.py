"""
Host for Papilio folded 784→16→10 Mazumdar online PoC (B+E telemetry).

Streams MNIST binary patterns over UART, trains on-chip, scores with EE,
dumps per-class and EE histogram.
"""
from __future__ import annotations

import argparse
import json
import struct
import time
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    import serial
except ImportError:
    raise SystemExit("pip install pyserial")

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "results"
OUT.mkdir(parents=True, exist_ok=True)
DEFAULT_DATA = (
    HERE.parent.parent / "data" / "mnist_bin_t0.3_tr800_te250_s2026.npz"
)


def bits_to_bytes(x784: np.ndarray) -> bytes:
    bits = np.asarray(x784, dtype=np.uint8).reshape(784)
    out = bytearray(98)
    for i, b in enumerate(bits):
        if b:
            out[i >> 3] |= 1 << (i & 7)
    return bytes(out)


class Board:
    def __init__(self, port: str, baud: int = 115200):
        self.ser = serial.Serial(port, baud, timeout=2.0)
        time.sleep(0.3)
        self.ser.reset_input_buffer()

    def close(self):
        self.ser.close()

    def _cmd(self, payload: bytes, n: int) -> bytes:
        self.ser.reset_input_buffer()
        self.ser.write(payload)
        raw = self.ser.read(n)
        if len(raw) < n:
            raise RuntimeError(f"short reply {len(raw)}/{n} for {payload[:1]!r}")
        return raw

    def ident(self) -> bytes:
        return self._cmd(b"V", 2)

    def load_x(self, x784):
        self._cmd(b"X" + bits_to_bytes(x784), 1)

    def set_label(self, y: int):
        self._cmd(bytes([ord("L"), int(y) & 15]), 1)

    def set_n(self, nh=8, no=8):
        self._cmd(bytes([ord("N"), nh & 15, no & 15]), 1)

    def infer(self, t_max=16, ee=True, score=True):
        mode = (1 if ee else 0) | (2 if score else 0)
        raw = self._cmd(bytes([ord("R"), t_max & 63, mode]), 6)
        pred, passes = raw[0] & 15, raw[1] & 63
        cycles = struct.unpack_from("<I", raw, 2)[0]
        return pred, passes, cycles

    def train_one(self):
        raw = self._cmd(b"T", 6)
        pred, passes = raw[0] & 15, raw[1] & 63
        cycles = struct.unpack_from("<I", raw, 2)[0]
        return pred, passes, cycles

    def clear_stats(self):
        self._cmd(b"Z", 1)

    def poll(self):
        raw = self._cmd(b"P", 9)
        samples = struct.unpack_from("<I", raw, 0)[0]
        updates = struct.unpack_from("<I", raw, 4)[0]
        busy = raw[8] & 1
        return dict(samples_trained=samples, update_events=updates, busy=busy)

    def infer_stats(self):
        raw = self._cmd(b"I", 4)
        correct, total = struct.unpack("<HH", raw)
        return correct, total

    def class_correct(self, idx: int) -> int:
        return struct.unpack("<H", self._cmd(bytes([ord("C"), idx & 15]), 2))[0]

    def class_seen(self, idx: int) -> int:
        return struct.unpack("<H", self._cmd(bytes([ord("S"), idx & 15]), 2))[0]

    def ee_hist(self, idx: int) -> int:
        return struct.unpack("<H", self._cmd(bytes([ord("H"), idx & 7]), 2))[0]


def load_npz(path: Path, n_train: int, n_test: int, seed: int = 0):
    d = np.load(path)
    keys = set(d.files)
    if "Xtr" in keys:
        xt, yt, xe, ye = d["Xtr"], d["ytr"], d["Xte"], d["yte"]
    elif "x_train" in keys:
        xt, yt, xe, ye = d["x_train"], d["y_train"], d["x_test"], d["y_test"]
    elif "X_train" in keys:
        xt, yt, xe, ye = d["X_train"], d["y_train"], d["X_test"], d["y_test"]
    else:
        xt, yt, xe, ye = d["train_x"], d["train_y"], d["test_x"], d["test_y"]
    rng = np.random.default_rng(seed)
    tr = rng.choice(len(xt), size=min(n_train, len(xt)), replace=False)
    te = rng.choice(len(xe), size=min(n_test, len(xe)), replace=False)
    return xt[tr], yt[tr], xe[te], ye[te]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM11")
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--train", type=int, default=200)
    ap.add_argument("--test", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--t-max", type=int, default=16)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    print("Opening", args.port)
    b = Board(args.port)
    ident = b.ident()
    print("IDENT", ident.hex(), "(expect a503)")
    if ident != bytes([0xA5, 0x03]):
        print("WARN: unexpected bitstream identity")

    b.set_n(9, 8)
    b.clear_stats()

    print("Loading", args.data)
    xt, yt, xe, ye = load_npz(Path(args.data), args.train, args.test, args.seed)
    print(f"train={len(xt)} test={len(xe)}")

    t0 = time.time()
    for ep in range(args.epochs):
        order = np.random.default_rng(args.seed + ep).permutation(len(xt))
        for i, n in enumerate(order):
            b.load_x(xt[n])
            b.set_label(int(yt[n]))
            pred, passes, cyc = b.train_one()
            if (i + 1) % 50 == 0 or i == 0:
                pol = b.poll()
                print(
                    f"  ep{ep} {i+1}/{len(xt)} pred={pred} y={int(yt[n])} "
                    f"cyc={cyc} trained={pol['samples_trained']} "
                    f"upd={pol['update_events']}"
                )

    b.clear_stats()
    for n in range(len(xe)):
        b.load_x(xe[n])
        b.set_label(int(ye[n]))
        pred, passes, cyc = b.infer(t_max=args.t_max, ee=True, score=True)
        if (n + 1) % 25 == 0 or n == 0:
            c, t = b.infer_stats()
            print(f"  eval {n+1}/{len(xe)} running_acc={c}/{t} last_pred={pred} y={int(ye[n])} T={passes}")

    correct, total = b.infer_stats()
    per = [b.class_correct(i) for i in range(10)]
    seen = [b.class_seen(i) for i in range(10)]
    hist = [b.ee_hist(i) for i in range(8)]
    pol = b.poll()
    elapsed = time.time() - t0

    out = dict(
        t=datetime.now().isoformat(timespec="seconds"),
        port=args.port,
        ident=ident.hex(),
        data=str(args.data),
        n_train=len(xt),
        n_test=len(xe),
        epochs=args.epochs,
        t_max=args.t_max,
        elapsed_s=round(elapsed, 2),
        samples_trained=pol["samples_trained"],
        update_events=pol["update_events"],
        infer_correct=correct,
        infer_total=total,
        accuracy_pct=round(100.0 * correct / max(total, 1), 2),
        class_correct=per,
        class_seen=seen,
        ee_hist=hist,
        ee_hist_bins=["1-4", "5-8", "9-12", "13-16", "17-24", "25-32", "33-40", "41+"],
    )
    path = OUT / "mnist_online_silicon.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print("wrote", path)
    b.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
