"""
Papilio logic-learn host: poll status every 30 s.

Board UART (FT2232 channel B): usually the higher COM of the pair.
Protocol (115200 8N1), little-endian replies:
  'V' -> A5 02
  'G' -> continuous train on
  'S' -> stop
  'E' -> one eval epoch
  'P' -> epoch_id u16, correct u8, mean_passes u8, updates u16, flags u8
         flags: bit0=train_en, bit1=busy

Topology reminder (paper PoC):
  i = Ai1,Ai2, Oi1,Oi2, Xi1,Xi2
  j = 6 + bias  → i→j = 42
  k = Ao,Oo,Xo (+pad) → j→k = 28, i→k = 24
  Mazumdar mul-free ±1, epoch=64 on-the-fly, LED = ~epoch_id[0] (even ON)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    import serial
except ImportError:
    print("pip install pyserial", file=sys.stderr)
    raise SystemExit(2)

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "results"
OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "pc_progress.jsonl"


def open_port(port: str, baud: int = 115200):
    ser = serial.Serial(port, baud, timeout=1.0)
    time.sleep(0.2)
    ser.reset_input_buffer()
    return ser


def cmd_v(ser) -> bytes:
    ser.write(b"V")
    return ser.read(2)


def cmd_p(ser) -> dict | None:
    ser.reset_input_buffer()
    ser.write(b"P")
    raw = ser.read(7)
    if len(raw) < 7:
        return None
    epoch = raw[0] | (raw[1] << 8)
    correct = raw[2]
    mean_passes = raw[3]
    updates = raw[4] | (raw[5] << 8)
    flags = raw[6]
    return dict(
        epoch=epoch,
        correct=correct,
        mean_passes=mean_passes,
        updates=updates,
        train_en=bool(flags & 1),
        busy=bool(flags & 2),
        joint_pct=100.0 * correct / 64.0,
    )


def main():
    ap = argparse.ArgumentParser(description="Papilio Mazumdar PoC 30s progress")
    ap.add_argument("--port", default="COM11", help="UART COM (try COM10 if fails)")
    ap.add_argument("--interval", type=float, default=30.0)
    ap.add_argument("--go", action="store_true", help="send 'G' start train once")
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()

    print(f"Opening {args.port} @ 115200 …")
    ser = open_port(args.port)
    ident = cmd_v(ser)
    print(f"IDENT: {ident.hex() if ident else 'none'} (expect a502)")
    if args.go:
        ser.write(b"G")
        print("sent G (enable train)")

    while True:
        st = cmd_p(ser)
        ts = datetime.now().isoformat(timespec="seconds")
        if st is None:
            line = dict(t=ts, error="no_reply_P")
            print(f"[{ts}] no reply to P — wrong COM or bitstream not loaded")
        else:
            line = dict(t=ts, **st)
            print(
                f"[{ts}] ep={st['epoch']:5d}  correct={st['correct']:2d}/64 "
                f"({st['joint_pct']:5.1f}%)  updates={st['updates']}  "
                f"busy={int(st['busy'])} train={int(st['train_en'])}"
            )
        with LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(line) + "\n")
        if args.once:
            break
        time.sleep(args.interval)

    ser.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
