"""Wait for COM unlock, then poll 'P' every 30s (board may be programmed later)."""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import serial

OUT = Path(__file__).resolve().parents[1] / "results"
OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "pc_progress.jsonl"
PORTS = ["COM11", "COM10"]


def try_open():
    for p in PORTS:
        try:
            s = serial.Serial(p, 115200, timeout=0.8)
            time.sleep(0.15)
            s.reset_input_buffer()
            s.write(b"V")
            r = s.read(2)
            return s, p, r
        except Exception:
            continue
    return None, None, None


def main():
    print("Waiting for free Papilio UART (COM10/11)… unplug/replug if Access Denied.")
    ser = None
    port = None
    while ser is None:
        ser, port, ident = try_open()
        if ser is None:
            time.sleep(5)
            continue
        print(f"Opened {port} ident={ident.hex() if ident else 'empty'} (expect a502)")
        if ident != b"\xa5\x02":
            print("WARN: not logic-learn bitstream yet — will keep polling")
        ser.write(b"G")
        break

    while True:
        try:
            ser.reset_input_buffer()
            ser.write(b"P")
            raw = ser.read(7)
            ts = datetime.now().isoformat(timespec="seconds")
            if len(raw) < 7:
                line = dict(t=ts, port=port, error="short_P", raw=raw.hex())
            else:
                epoch = raw[0] | (raw[1] << 8)
                correct = raw[2]
                mean_passes = raw[3]
                updates = raw[4] | (raw[5] << 8)
                flags = raw[6]
                line = dict(
                    t=ts,
                    port=port,
                    epoch=epoch,
                    correct=correct,
                    joint_pct=100.0 * correct / 64.0,
                    mean_passes=mean_passes,
                    updates=updates,
                    train_en=bool(flags & 1),
                    busy=bool(flags & 2),
                )
                print(
                    f"[{ts}] ep={epoch:5d} correct={correct:2d}/64 "
                    f"({line['joint_pct']:5.1f}%) updates={updates}"
                )
            with LOG.open("a", encoding="utf-8") as f:
                f.write(json.dumps(line) + "\n")
        except Exception as e:
            print("UART lost:", e)
            try:
                ser.close()
            except Exception:
                pass
            ser = None
            while ser is None:
                ser, port, _ = try_open()
                if ser is None:
                    time.sleep(5)
                else:
                    ser.write(b"G")
                    print("reopened", port)
        time.sleep(30)


if __name__ == "__main__":
    raise SystemExit(main())
