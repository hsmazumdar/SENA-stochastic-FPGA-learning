"""SELF-TEST helper: random (untrained) DWN-shaped model JSON for exercising
the Verilog generator at full size, with ties and GroupSum padding.
Random weights -> its 'accuracy' is meaningless and is not a result."""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, dwn_ref
ap = argparse.ArgumentParser()
ap.add_argument("--layers", default="1000,500"); ap.add_argument("--n", default="6")
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", required=True)
a = ap.parse_args()
rng = np.random.default_rng(a.seed)
widths = [int(v) for v in a.layers.split(",")]; ns = [int(v) for v in a.n.split(",")]
ns = ns * len(widths) if len(ns) == 1 else ns
layers, prev = [], 784
for w, n in zip(widths, ns):
    m = rng.integers(0, prev, size=(w, n)).tolist()
    hexes = [format(int.from_bytes(rng.bytes(2 ** n // 8 or 1), "little") & ((1 << 2 ** n) - 1),
                    "0%dx" % ((2 ** n + 3) // 4)) for _ in range(w)]
    layers.append({"n": n, "n_in": prev, "n_out": w, "mapping": m, "luts_hex": hexes}); prev = w
model = {"format": dwn_ref.FORMAT, "n_in": 784, "n_classes": 10, "groupsum_tau": 1.0,
         "layers": layers, "meta": {"label": "SELFTEST random untrained model"}}
dwn_ref.save_model(model, a.out); print("wrote", a.out)
