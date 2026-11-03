"""
Golden twin: 6→6→3 AND/OR/XOR with skip i→k (42 + 28 + 24 connections).

i = 6  : Ai1,Ai2, Oi1,Oi2, Xi1,Xi2
j = 6+bias (7 columns) → i→j = 42
k = 3+bias storage (4 rows) → j→k = 28, i→k = 24
  (row 3 unused pad; functional outputs Ao,Oo,Xo)

Mazumdar multiplier-free ±1 updates with reverse-characteristic gates.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "results"
OUT.mkdir(parents=True, exist_ok=True)

N_IN, N_HID, N_OUT = 6, 6, 3
JCOL, KROW = 7, 4  # +bias columns / pad rows
W_IJ = N_HID * JCOL          # 42
W_JK = KROW * JCOL           # 28
W_IK = KROW * N_IN           # 24


def patterns64():
    X, Y = [], []
    for bits in itertools.product([0, 1], repeat=6):
        a0, b0, a1, b1, a2, b2 = bits
        X.append(list(bits))
        Y.append([a0 & b0, a1 | b1, a2 ^ b2])
    return np.asarray(X, np.int64), np.asarray(Y, np.int64)


class Net663:
    def __init__(self, seed=2026, N=12):
        rng = np.random.default_rng(seed)
        self.w_ij = rng.integers(-4, 5, size=(N_HID, JCOL), dtype=np.int64)
        self.w_jk = rng.integers(-4, 5, size=(KROW, JCOL), dtype=np.int64)
        self.w_ik = rng.integers(-4, 5, size=(KROW, N_IN), dtype=np.int64)
        self.rng = rng
        self.N = int(N)
        self.N_back = 8.0
        self.clip = 127

    def _fire(self, acc):
        r = int(self.rng.integers(-self.N, self.N + 1))
        return 1 if acc > r else 0

    def forward(self, x, stochastic=True):
        x = np.asarray(x, np.int64)
        h = np.zeros(N_HID, np.int64)
        pre_h = np.zeros(N_HID, np.int64)
        for j in range(N_HID):
            acc = int(self.w_ij[j, N_IN])
            for i in range(N_IN):
                if x[i]:
                    acc += int(self.w_ij[j, i])
            pre_h[j] = acc
            h[j] = self._fire(acc) if stochastic else (1 if acc > 0 else 0)
        y = np.zeros(N_OUT, np.int64)
        pre_o = np.zeros(N_OUT, np.int64)
        for k in range(N_OUT):
            acc = int(self.w_jk[k, N_HID])
            for j in range(N_HID):
                if h[j]:
                    acc += int(self.w_jk[k, j])
            for i in range(N_IN):
                if x[i]:
                    acc += int(self.w_ik[k, i])
            pre_o[k] = acc
            y[k] = self._fire(acc) if stochastic else (1 if acc > 0 else 0)
        return h, y, pre_h, pre_o

    def train_sample(self, x, t, lr_passes=2):
        """Mazumdar mul-free ±1 with reverse gates; optional multi-pass."""
        x = np.asarray(x, np.int64)
        t = np.asarray(t, np.int64)
        for _ in range(lr_passes):
            h, y, pre_h, pre_o = self.forward(x, stochastic=True)
            d = t - y  # {-1,0,1}
            yb = np.append(h, 1)
            sign_h = np.where(yb > 0, 1, -1)
            sign_x = np.where(x > 0, 1, -1)
            for k in range(N_OUT):
                if d[k] == 0:
                    continue
                self.w_jk[k] += d[k] * sign_h
                self.w_ik[k] += d[k] * sign_x

            r1 = self.rng.integers(0, max(2, self.N), size=N_OUT)
            p = (np.abs(pre_o) <= r1).astype(np.int64)
            gate = (d != 0) & (p != 0)
            contrib = (np.sign(d) * gate)[:, None] * self.w_jk[:N_OUT, :N_HID]
            back = contrib.sum(axis=0)

            r2 = self.rng.integers(0, max(2, self.N), size=N_HID)
            a = (np.abs(pre_h) <= r2).astype(np.int64)
            back_rms = float(np.sqrt(np.mean(back.astype(float) ** 2))) + 1e-6
            self.N_back = 0.99 * self.N_back + 0.01 * (0.8 * back_rms)
            nb = max(1, int(self.N_back))
            r3 = self.rng.integers(0, nb, size=N_HID)
            b = (np.abs(back) > r3).astype(np.int64)
            dj = a & b
            sign_i = np.where(np.append(x, 1) > 0, 1, -1)
            dir_j = np.where(back > 0, 1, -1)
            for j in range(N_HID):
                if dj[j]:
                    self.w_ij[j] += dir_j[j] * sign_i

            np.clip(self.w_ij, -self.clip, self.clip, out=self.w_ij)
            np.clip(self.w_jk, -self.clip, self.clip, out=self.w_jk)
            np.clip(self.w_ik, -self.clip, self.clip, out=self.w_ik)

    def eval_joint(self, X, Y, T=16, early_exit=True, ee_min=3):
        """Majority vote; optional early-exit when all heads confident."""
        ok = 0
        per = np.zeros(N_OUT)
        pass_sum = 0
        for x, t in zip(X, Y):
            votes = np.zeros(N_OUT, np.int64)
            used = T
            for tp in range(1, T + 1):
                _, y, _, _ = self.forward(x, stochastic=True)
                votes += y
                if early_exit and tp >= ee_min:
                    z = tp - votes
                    maj = np.maximum(votes, z)
                    if np.all(maj >= (tp * 3) // 4):
                        used = tp
                        break
            pass_sum += used
            pred = (votes >= (used / 2)).astype(np.int64)
            per += pred == t
            ok += int(np.all(pred == t))
        n = len(X)
        return per / n, ok / n, pass_sum / n


def flat_counts(net: Net663):
    """Flatten including unused k-row pad so counts match 42/28/24."""
    w_ij = net.w_ij.reshape(-1)
    w_jk = net.w_jk.reshape(-1)
    w_ik = net.w_ik.reshape(-1)
    assert w_ij.size == 42 and w_jk.size == 28 and w_ik.size == 24
    return w_ij, w_jk, w_ik


def main():
    assert W_IJ == 42 and W_JK == 28 and W_IK == 24
    X, Y = patterns64()
    net = Net663(2026, N=12)
    p0, j0, m0 = net.eval_joint(X, Y, T=8, early_exit=False)
    print(
        f"BEFORE AND={p0[0]*100:.1f}% OR={p0[1]*100:.1f}% XOR={p0[2]*100:.1f}% "
        f"joint={j0*100:.1f}%"
    )

    hist = []
    for ep in range(1, 201):
        order = np.random.default_rng(ep).permutation(64)
        for n in order:
            net.train_sample(X[n], Y[n], lr_passes=2)
        if ep % 10 == 0 or ep == 1:
            p, j, mp = net.eval_joint(X, Y, T=16, early_exit=True)
            hist.append(
                dict(
                    epoch=ep,
                    joint=float(j),
                    and_=float(p[0]),
                    or_=float(p[1]),
                    xor=float(p[2]),
                    mean_passes=float(mp),
                )
            )
            print(
                f"ep {ep:3d}  AND={p[0]*100:5.1f}% OR={p[1]*100:5.1f}% "
                f"XOR={p[2]*100:5.1f}% joint={j*100:5.1f}%  "
                f"mean_T={mp:.2f}"
            )
            if j >= 0.95:
                break

    p1, j1, m1 = net.eval_joint(X, Y, T=32, early_exit=True)
    flat_counts(net)
    payload = dict(
        connections=dict(ij=42, jk=28, ik=24),
        topology="i=6 j=6+bias k=3+pad",
        before=dict(joint=float(j0), per=p0.tolist()),
        after=dict(joint=float(j1), per=p1.tolist(), mean_passes=float(m1)),
        history=hist,
        pass_=bool(j1 >= 0.90),
    )
    (OUT / "golden_logic663_skip.json").write_text(json.dumps(payload, indent=2))
    print("PASS" if payload["pass_"] else "FAIL", f"joint={j1*100:.1f}%")
    return 0 if payload["pass_"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
