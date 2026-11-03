import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "sim"))
from stochastic_core import StochasticConfig, StochasticNet


def make():
    X, Y = [], []
    for bits in itertools.product([0, 1], repeat=6):
        a0, b0, a1, b1, a2, b2 = bits
        X.append(bits)
        Y.append([a0 & b0, a1 | b1, a2 ^ b2])
    return np.asarray(X, np.int64), np.asarray(Y, np.int64)


def bit_acc(net, X, Y, T=32):
    n, k = Y.shape
    votes = np.zeros((n, k), np.int64)
    for _ in range(T):
        acts, _ = net.forward(X, stochastic=True)
        votes += acts[-1]
    pred = (votes >= T / 2).astype(np.int64)
    return (pred == Y).mean(0), np.all(pred == Y, 1).mean()


def train_skip_mazumdar(epochs=300, seed=2026, N=16, lr_passes=3):
    """Faithful Mazumdar with i→k skip; 42+28+24 weights."""
    rng = np.random.default_rng(seed)
    X, Y = make()
    w_ij = rng.integers(-4, 5, size=(6, 7), dtype=np.int64)
    w_jk = rng.integers(-4, 5, size=(3, 7), dtype=np.int64)
    w_ik = rng.integers(-4, 5, size=(3, 6), dtype=np.int64)
    clip = 127
    N_back = 8.0

    def fire(acc, Nl):
        r = int(rng.integers(-Nl, Nl + 1))
        return 1 if acc > r else 0, acc

    def forward(x):
        h = np.zeros(6, np.int64)
        pre_h = np.zeros(6, np.int64)
        for j in range(6):
            acc = int(w_ij[j, 6])
            for i in range(6):
                if x[i]:
                    acc += int(w_ij[j, i])
            pre_h[j] = acc
            h[j] = fire(acc, N)[0]
        y = np.zeros(3, np.int64)
        pre_o = np.zeros(3, np.int64)
        for k in range(3):
            acc = int(w_jk[k, 6])
            for j in range(6):
                if h[j]:
                    acc += int(w_jk[k, j])
            for i in range(6):
                if x[i]:
                    acc += int(w_ik[k, i])
            pre_o[k] = acc
            y[k] = fire(acc, N)[0]
        return h, y, pre_h, pre_o

    for ep in range(epochs):
        for n in rng.permutation(64):
            x, t = X[n], Y[n]
            for _ in range(lr_passes):
                h, y, pre_h, pre_o = forward(x)
                d = t - y
                # output + skip (Mazumdar)
                yb = np.append(h, 1)
                sign_in = np.where(yb > 0, 1, -1)
                for k in range(3):
                    if d[k] == 0:
                        continue
                    w_jk[k] += d[k] * sign_in
                    sign_x = np.where(x > 0, 1, -1)
                    w_ik[k] += d[k] * sign_x
                # reverse gate on outputs
                r1 = rng.integers(0, max(2, N), size=3)
                p = (np.abs(pre_o) <= r1).astype(np.int64)
                gate = (d != 0) & (p != 0)
                contrib = (np.sign(d) * gate)[:, None] * w_jk[:, :6]
                back = contrib.sum(axis=0)
                # also add skip contribution to "virtual" — skip has no hid
                r2 = rng.integers(0, max(2, N), size=6)
                a = (np.abs(pre_h) <= r2).astype(np.int64)
                back_rms = float(np.sqrt(np.mean(back.astype(float) ** 2))) + 1e-6
                N_back = 0.99 * N_back + 0.01 * (0.8 * back_rms)
                nb = max(1, int(N_back))
                r3 = rng.integers(0, nb, size=6)
                b = (np.abs(back) > r3).astype(np.int64)
                dj = a & b
                sign_i = np.where(np.append(x, 1) > 0, 1, -1)
                dir_j = np.where(back > 0, 1, -1)
                for j in range(6):
                    if dj[j]:
                        w_ij[j] += dir_j[j] * sign_i
                np.clip(w_ij, -clip, clip, out=w_ij)
                np.clip(w_jk, -clip, clip, out=w_jk)
                np.clip(w_ik, -clip, clip, out=w_ik)

        if (ep + 1) % 20 == 0 or ep == 0:
            # eval
            votes = np.zeros((64, 3), np.int64)
            for _ in range(24):
                for n in range(64):
                    _, y, _, _ = forward(X[n])
                    votes[n] += y
            pred = (votes >= 12).astype(np.int64)
            per = (pred == Y).mean(0)
            j = np.all(pred == Y, 1).mean()
            print(
                f"ep {ep+1:3d} AND={per[0]*100:5.1f} OR={per[1]*100:5.1f} "
                f"XOR={per[2]*100:5.1f} joint={j*100:5.1f}"
            )
            if j >= 0.95:
                return True, j
    return False, j


if __name__ == "__main__":
    X, Y = make()
    print("=== StochasticNet sweeps ===")
    for hid, N, ep, seed in [
        (6, 16, 200, 2026),
        (8, 32, 200, 1),
        (12, 32, 300, 2),
        (16, 64, 200, 0),
        (6, 4, 500, 7),
    ]:
        cfg = StochasticConfig(
            layer_sizes=(6, hid, 3),
            seed=seed,
            weight_bits=8,
            epochs=ep,
            N=N,
            init_scale=4,
            rng_kind="numpy",
        )
        net = StochasticNet(cfg)
        net.fit(X, Y, epochs=ep, recal_every=5, T_eval=16, verbose=False)
        per, j = bit_acc(net, X, Y, T=32)
        print(
            f"hid={hid} N={N} ep={ep} AND={per[0]*100:.1f} OR={per[1]*100:.1f} "
            f"XOR={per[2]*100:.1f} joint={j*100:.1f}"
        )

    print("=== Skip Mazumdar ===")
    for seed in (2026, 1, 2, 42):
        print(f"-- seed {seed}")
        ok, j = train_skip_mazumdar(epochs=250, seed=seed, N=12, lr_passes=2)
        print("PASS" if ok else "FAIL", j)
