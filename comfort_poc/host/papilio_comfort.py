"""Host for Analog Comfort PoC on Papilio (IDENT A5 04)."""
from __future__ import annotations

import argparse
import csv
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
PKG = HERE.parent
RES = PKG / "results"
RES.mkdir(exist_ok=True)


def encode_u8(T: float, RH: float):
    t_u8 = int(np.clip(round((T - 20.0) / 20.0 * 255.0), 0, 255))
    rh_u8 = int(np.clip(round(RH / 100.0 * 255.0), 0, 255))
    return t_u8, rh_u8


class Board:
    def __init__(self, port: str):
        self.ser = serial.Serial(port, 115200, timeout=10.0, write_timeout=2.0)
        time.sleep(0.3)
        self.ser.reset_input_buffer()

    def close(self):
        self.ser.close()

    def _cmd(self, payload: bytes, n: int) -> bytes:
        self.ser.reset_input_buffer()
        self.ser.write(payload)
        raw = self.ser.read(n)
        if len(raw) < n:
            raise RuntimeError(f"short {len(raw)}/{n} for {payload[:1]!r}")
        return raw

    def ident(self):
        return self._cmd(b"V", 2)

    def set_member(self, m: int):
        self._cmd(bytes([ord("B"), m & 3]), 1)

    def set_sensors(self, t_u8: int, rh_u8: int):
        self._cmd(bytes([ord("S"), t_u8 & 255, rh_u8 & 255]), 1)

    def set_n(self, n_log2: int = 6):
        self._cmd(bytes([ord("N"), n_log2 & 15]), 1)

    def run(self, t_passes: int = 100):
        raw = self._cmd(bytes([ord("R"), t_passes & 255, (t_passes >> 8) & 255]), 8)
        ones, passes = struct.unpack_from("<HH", raw, 0)
        cycles = struct.unpack_from("<I", raw, 4)[0]
        p = ones / float(passes) if passes else 0.0
        return dict(ones=ones, passes=passes, cycles=cycles, p=p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM11")
    ap.add_argument("--t-passes", type=int, default=100)
    ap.add_argument("--ensemble", action="store_true", help="average M=3 members")
    ap.add_argument("--limit", type=int, default=50)
    args = ap.parse_args()

    b = Board(args.port)
    ident = b.ident()
    print("IDENT", ident.hex(), "(expect a504)")
    b.set_n(6)

    test = PKG / "data" / "test.csv"
    rows = list(csv.DictReader(test.open(encoding="utf-8")))[: args.limit]
    results = []
    for r in rows:
        T, RH = float(r["T"]), float(r["RH"])
        p_star = float(r["p_star"])
        t_u8, rh_u8 = encode_u8(T, RH)
        b.set_sensors(t_u8, rh_u8)
        if args.ensemble:
            ps = []
            for m in range(3):
                b.set_member(m)
                ps.append(b.run(args.t_passes)["p"])
            p = float(np.mean(ps))
            info = dict(p_members=ps)
        else:
            b.set_member(0)
            info = b.run(args.t_passes)
            p = info["p"]
        results.append(
            dict(
                T=T,
                RH=RH,
                p_star=p_star,
                p_hat=p,
                abs_err=abs(p - p_star),
                ones=info.get("ones"),
                passes=info.get("passes"),
                cycles=info.get("cycles"),
                p_members=info.get("p_members"),
            )
        )
        print(f"T={T:5.1f} RH={RH:5.1f} p*={p_star:.3f} phat={p:.3f} |e|={abs(p-p_star):.3f}")

    mae = float(np.mean([x["abs_err"] for x in results]))
    out = dict(
        t=datetime.now().isoformat(timespec="seconds"),
        ident=ident.hex(),
        t_passes=args.t_passes,
        resolution_pct=100.0 / args.t_passes,
        n=len(results),
        mae=mae,
        ensemble=bool(args.ensemble),
        samples=results,
    )
    path = RES / "comfort_board_results.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"MAE={mae:.4f}  resolution={out['resolution_pct']}%  wrote {path}")
    b.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
