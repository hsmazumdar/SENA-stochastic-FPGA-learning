"""
Golden check: 6→6→3 multi-task AND / OR / XOR with SENA integer updates.

Inputs:  [a0 b0 a1 b1 a2 b2]
Targets: [a0&b0, a1|b1, a2^b2]

This is the software twin of the DE2-115 on-chip learning demo.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SIM = HERE.parents[1] / "sim"
sys.path.insert(0, str(SIM))

from stochastic_core import StochasticConfig, StochasticNet  # noqa: E402


def make_logic663():
    rows, labels = [], []
    for a0, b0, a1, b1, a2, b2 in itertools.product([0, 1], repeat=6):
        x = np.array([a0, b0, a1, b1, a2, b2], dtype=np.int64)
        y = np.array([a0 & b0, a1 | b1, a2 ^ b2], dtype=np.int64)
        rows.append(x)
        labels.append(y)
    X = np.stack(rows)
    Y = np.stack(labels)
    return X, Y


def bit_accuracy(net: StochasticNet, X, Y, T=32):
    """Per-head and joint accuracy under T stochastic votes (majority)."""
    n, k = Y.shape
    votes = np.zeros((n, k), dtype=np.int64)
    for _ in range(T):
        acts, _ = net.forward(X, stochastic=True)
        votes += acts[-1]
    pred = (votes >= (T / 2)).astype(np.int64)
    per_head = (pred == Y).mean(axis=0)
    joint = np.all(pred == Y, axis=1).mean()
    return per_head, joint, pred


def main():
    X, Y = make_logic663()  # 64 patterns
    cfg = StochasticConfig(
        layer_sizes=(6, 6, 3),
        seed=2026,
        weight_bits=8,
        epochs=80,
        rng_kind="numpy",
    )
    net = StochasticNet(cfg)
    # Before training
    per0, joint0, _ = bit_accuracy(net, X, Y, T=16)
    print(f"BEFORE  AND={per0[0]*100:.1f}% OR={per0[1]*100:.1f}% "
          f"XOR={per0[2]*100:.1f}% joint={joint0*100:.1f}%")

    net.fit(X, Y, epochs=60, recal_every=5, T_eval=16, verbose=False)
    per1, joint1, _ = bit_accuracy(net, X, Y, T=32)
    print(f"AFTER   AND={per1[0]*100:.1f}% OR={per1[1]*100:.1f}% "
          f"XOR={per1[2]*100:.1f}% joint={joint1*100:.1f}%")

    ok = joint1 >= 0.90 and per1.min() >= 0.90
    print("PASS" if ok else "FAIL",
          "(expect joint≥90% after on-chip-style integer training)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
