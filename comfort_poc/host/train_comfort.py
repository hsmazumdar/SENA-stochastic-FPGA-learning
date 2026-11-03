"""
SENA Comfort PoC — dataset + Mazumdar train + weight export.

Inputs: T (°C), RH (%) → 8-bit norms → 16 binary bit-features.
Target: soft p* from Indian NBC/TSI-style comfort (V=0 calm).
Train: Mazumdar ±1; evaluate p_hat over T stochastic passes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
DATA = PKG / "data"
WDIR = PKG / "weights"
RES = PKG / "results"
for d in (DATA, WDIR, RES):
    d.mkdir(parents=True, exist_ok=True)

N_IN_BITS = 16  # 8 T bits + 8 RH bits
N_HID = 16
N_OUT = 1
M_ENS = 3
CLIP = 127


def tsi_celsius(T: float, RH: float, V: float = 0.0) -> float:
    """Approx wet-bulb then Tropical Summer Index (Sharma & Ali / NBC family)."""
    # Magnus approx dewpoint / wet-bulb surrogate for PoC (not metrology-grade)
    a, b = 17.27, 237.7
    g = (a * T) / (b + T) + np.log(max(RH, 1e-3) / 100.0)
    Td = (b * g) / (a - g)
    Tw = T * np.arctan(0.151977 * np.sqrt(RH + 8.313659))
    Tw += np.arctan(T + RH) - np.arctan(RH - 1.676331)
    Tw += 0.00391838 * (RH ** 1.5) * np.arctan(0.023101 * RH) - 4.686035
    # Prefer Stull-like Tw; fall back blend with Td if nan
    if not np.isfinite(Tw):
        Tw = 0.5 * (T + Td)
    return 0.745 * T + 0.308 * Tw - 2.060 * V


def comfort_p_star(T: float, RH: float) -> float:
    """
    Soft target in [0,1]: 1 near TSI optimum 27.5 within NBC-like 25–30 band.
    Outside band, decays smoothly (not hard 0/1 — matches PoC philosophy).
    """
    tsi = tsi_celsius(T, RH, V=0.0)
    # triangular bump 25..30 peaking at 27.5, plus mild RH preference 40–70%
    if 25.0 <= tsi <= 30.0:
        p = 1.0 - abs(tsi - 27.5) / 2.5
    elif tsi < 25.0:
        p = max(0.0, 1.0 - (25.0 - tsi) / 8.0)
    else:
        p = max(0.0, 1.0 - (tsi - 30.0) / 10.0)
    if 40.0 <= RH <= 70.0:
        p = min(1.0, p + 0.05)
    elif RH > 85.0 or RH < 20.0:
        p = max(0.0, p - 0.15)
    return float(np.clip(p, 0.0, 1.0))


def encode_bits(T: float, RH: float) -> np.ndarray:
    t_u8 = int(np.clip(round((T - 20.0) / 20.0 * 255.0), 0, 255))
    rh_u8 = int(np.clip(round(RH / 100.0 * 255.0), 0, 255))
    bits = np.zeros(N_IN_BITS, dtype=np.int64)
    for i in range(8):
        bits[i] = (t_u8 >> i) & 1
        bits[8 + i] = (rh_u8 >> i) & 1
    return bits, t_u8, rh_u8


def make_dataset(n: int, seed: int = 2026):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        T = float(rng.uniform(18.0, 42.0))
        RH = float(rng.uniform(10.0, 95.0))
        bits, t_u8, rh_u8 = encode_bits(T, RH)
        p = comfort_p_star(T, RH)
        rows.append(
            dict(T=T, RH=RH, t_u8=t_u8, rh_u8=rh_u8, p_star=p, bits=bits)
        )
    return rows


def bits_matrix(rows):
    X = np.stack([r["bits"] for r in rows], axis=0)
    y = np.array([r["p_star"] for r in rows], dtype=np.float64)
    return X, y


class SENAComfortNet:
    def __init__(self, seed: int = 0, N: int = 64):
        rng = np.random.default_rng(seed)
        self.w_hid = rng.integers(-8, 9, size=(N_HID, N_IN_BITS + 1), dtype=np.int64)
        self.w_out = rng.integers(-8, 9, size=(N_OUT, N_HID + 1), dtype=np.int64)
        self.rng = rng
        self.N = int(N)

    def _fire(self, acc):
        r = int(self.rng.integers(-self.N, self.N + 1))
        return 1 if acc > r else 0

    def forward_once(self, x):
        x = np.asarray(x, np.int64)
        h = np.zeros(N_HID, np.int64)
        pre_h = np.zeros(N_HID, np.int64)
        for j in range(N_HID):
            acc = int(self.w_hid[j, N_IN_BITS])
            for i in range(N_IN_BITS):
                if x[i]:
                    acc += int(self.w_hid[j, i])
            pre_h[j] = acc
            h[j] = self._fire(acc)
        y = np.zeros(N_OUT, np.int64)
        pre_o = np.zeros(N_OUT, np.int64)
        for k in range(N_OUT):
            acc = int(self.w_out[k, N_HID])
            for j in range(N_HID):
                if h[j]:
                    acc += int(self.w_out[k, j])
            pre_o[k] = acc
            y[k] = self._fire(acc)
        return h, y, pre_h, pre_o

    def p_hat(self, x, T_passes: int = 100) -> float:
        s = 0
        for _ in range(T_passes):
            _, y, _, _ = self.forward_once(x)
            s += int(y[0])
        return s / float(T_passes)

    def train_sample(self, x, p_star: float, lr_passes: int = 2):
        """Mazumdar ±1 toward soft target via stochastic binary teacher."""
        x = np.asarray(x, np.int64)
        for _ in range(lr_passes):
            # sample soft target as Bernoulli(p_star) → binary teacher
            t = 1 if self.rng.random() < p_star else 0
            h, y, pre_h, pre_o = self.forward_once(x)
            d = np.array([t - int(y[0])], dtype=np.int64)  # -1,0,1
            if d[0] == 0:
                continue
            # output ±1 on active features (sign of h / bias)
            sign_h = np.where(np.append(h, 1) > 0, 1, -1)
            self.w_out[0] += int(d[0]) * sign_h
            # reverse gate (simplified, like Boolean PoC)
            abs_o = abs(int(pre_o[0]))
            if abs_o <= int(self.rng.integers(0, max(2, self.N // 4))):
                # back ≈ d * w_out[:hid]
                back = int(d[0]) * self.w_out[0, :N_HID]
                for j in range(N_HID):
                    if abs(int(pre_h[j])) <= int(self.rng.integers(0, max(2, self.N // 4))):
                        if abs(int(back[j])) > int(self.rng.integers(0, 4)):
                            dir_j = 1 if back[j] > 0 else -1
                            sign_i = np.where(np.append(x, 1) > 0, 1, -1)
                            self.w_hid[j] += dir_j * sign_i
            np.clip(self.w_hid, -CLIP, CLIP, out=self.w_hid)
            np.clip(self.w_out, -CLIP, CLIP, out=self.w_out)


def to_hex_word(values, w_bits=8):
    mask = (1 << w_bits) - 1
    word = 0
    for pos, v in enumerate(values):
        word |= (int(v) & mask) << (pos * w_bits)
    return format(word, f"0{len(values) * w_bits // 4}x")


def export_member(net: SENAComfortNet, tag: str):
    # rom_hid[i] = weights of input i for all hid (bias at i=N_IN_BITS)
    lines_hid = [
        to_hex_word(net.w_hid[:, i], 8) for i in range(N_IN_BITS + 1)
    ]
    lines_out = [
        to_hex_word(net.w_out[:, j], 8) for j in range(N_HID + 1)
    ]
    (WDIR / f"w_hid_{tag}.mem").write_text("\n".join(lines_hid) + "\n", encoding="ascii")
    (WDIR / f"w_out_{tag}.mem").write_text("\n".join(lines_out) + "\n", encoding="ascii")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-test", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--t-eval", type=int, default=100)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    train_rows = make_dataset(args.n_train, seed=args.seed)
    test_rows = make_dataset(args.n_test, seed=args.seed + 1)
    Xtr, ytr = bits_matrix(train_rows)
    Xte, yte = bits_matrix(test_rows)

    # save CSV for host (raw T,RH,p*)
    def dump_csv(path, rows):
        with path.open("w", encoding="utf-8") as f:
            f.write("T,RH,t_u8,rh_u8,p_star\n")
            for r in rows:
                f.write(
                    f"{r['T']:.3f},{r['RH']:.3f},{r['t_u8']},{r['rh_u8']},{r['p_star']:.4f}\n"
                )

    dump_csv(DATA / "train.csv", train_rows)
    dump_csv(DATA / "test.csv", test_rows)
    (DATA / "labels.txt").write_text(
        "Soft p* from TSI≈0.745*T+0.308*Tw (V=0); band 25–30 peaking 27.5 (NBC/TSI family).\n",
        encoding="utf-8",
    )

    members = []
    curves = []
    for m in range(M_ENS):
        net = SENAComfortNet(seed=args.seed + 17 * m, N=64)
        for ep in range(1, args.epochs + 1):
            order = np.random.default_rng(args.seed + ep + m).permutation(len(Xtr))
            for n in order:
                net.train_sample(Xtr[n], float(ytr[n]), lr_passes=2)
            if ep % 10 == 0 or ep == 1:
                # quick train MSE on subset
                idx = order[:200]
                pred = np.array([net.p_hat(Xtr[i], T_passes=40) for i in idx])
                mse = float(np.mean((pred - ytr[idx]) ** 2))
                print(f"member{m} ep{ep:3d} mse@40={mse:.4f}")
                curves.append(dict(member=m, epoch=ep, mse=mse))
        members.append(net)
        export_member(net, f"m{m}")

    # ensemble eval on test
    errs = []
    for i in range(len(Xte)):
        ps = [members[m].p_hat(Xte[i], T_passes=args.t_eval) for m in range(M_ENS)]
        p_bar = float(np.mean(ps))
        errs.append((p_bar - float(yte[i])) ** 2)
    mse = float(np.mean(errs))
    mae = float(np.mean(np.sqrt(errs)))
    # also show resolution note
    out = dict(
        n_hid=N_HID,
        n_in_bits=N_IN_BITS,
        M=M_ENS,
        epochs=args.epochs,
        t_eval=args.t_eval,
        test_mse=mse,
        test_rmse=float(np.sqrt(mse)),
        test_mae=mae,
        resolution_pct=100.0 / args.t_eval,
        curves=curves,
        note="Soft comfort probability vs NBC/TSI-style p*; not hard 0% error.",
    )
    (RES / "comfort_train_results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in out if k != "curves"}, indent=2))
    print("wrote weights to", WDIR)


if __name__ == "__main__":
    raise SystemExit(main())
