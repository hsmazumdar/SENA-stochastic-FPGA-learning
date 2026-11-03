"""
Soft-only comfort limit sweep — architectures must fit Papilio Pro silicon.

Device: Spartan-6 XC6SLX9 (Papilio Pro)
  LUT budget:  5,720
  Slice budget: 1,430
Current comfort bitfile (16-in → 16-hid → 1, M=3):
  2,102 LUT (36%), 600 slices (41%)

Soft configs are admitted only if estimated LUT ≤ 85% of LX9
(scaled from baseline weight-parameter count; UART/FSM overhead held fixed).

Writes results/soft_limit_sweep.json
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from train_comfort import comfort_p_star, tsi_celsius

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"
RES.mkdir(parents=True, exist_ok=True)

CLIP = 127
M_ENS = 3

# --- Papilio Pro / measured baseline ---
LX9_LUT = 5720
LX9_SLICES = 1430
BASE_LUT = 2102
BASE_SLICES = 600
BASE_N_IN = 16
BASE_N_HID = 16
LUT_CAP_FRAC = 0.85  # soft admit threshold


def weight_params(n_in: int, n_hid: int, m: int = M_ENS) -> int:
    return m * (n_hid * (n_in + 1) + 1 * (n_hid + 1))


BASE_PARAMS = weight_params(BASE_N_IN, BASE_N_HID, M_ENS)


def estimate_resources(n_in: int, n_hid: int, m: int = M_ENS) -> dict:
    """Scale LUT/slices from measured baseline by weight-parameter ratio."""
    params = weight_params(n_in, n_hid, m)
    scale = params / float(BASE_PARAMS)
    # fixed UART/protocol overhead ~15% of baseline LUT
    fixed = 0.15 * BASE_LUT
    core = (BASE_LUT - fixed) * scale
    lut = int(round(fixed + core))
    slices = int(round(BASE_SLICES * (lut / float(BASE_LUT))))
    return dict(
        params=params,
        scale=round(scale, 3),
        est_lut=lut,
        est_lut_pct=round(100.0 * lut / LX9_LUT, 1),
        est_slices=slices,
        est_slice_pct=round(100.0 * slices / LX9_SLICES, 1),
        fits=lut <= int(LUT_CAP_FRAC * LX9_LUT),
    )


def wet_bulb(T: float, RH: float) -> float:
    Tw = T * np.arctan(0.151977 * np.sqrt(RH + 8.313659))
    Tw += np.arctan(T + RH) - np.arctan(RH - 1.676331)
    Tw += 0.00391838 * (RH ** 1.5) * np.arctan(0.023101 * RH) - 4.686035
    return float(Tw)


def u8_bits(v: int, n: int = 8) -> list[int]:
    v = int(np.clip(v, 0, 255))
    return [(v >> i) & 1 for i in range(n)]


def encode_baseline(T: float, RH: float) -> np.ndarray:
    """Silicon PoC features: 16 bits (matches RTL)."""
    t_u8 = int(np.clip(round((T - 20.0) / 20.0 * 255.0), 0, 255))
    rh_u8 = int(np.clip(round(RH / 100.0 * 255.0), 0, 255))
    return np.array(u8_bits(t_u8) + u8_bits(rh_u8), dtype=np.int64)


def encode_rich32(T: float, RH: float) -> np.ndarray:
    """
    32-bit engineered features (LX9-safe with n_hid=16, M=3).
    Layout: T[8] RH[8] TSI[8] flags[8]
    """
    tsi = tsi_celsius(T, RH, 0.0)
    t_u8 = int(np.clip(round((T - 18.0) / 24.0 * 255.0), 0, 255))
    rh_u8 = int(np.clip(round(RH / 100.0 * 255.0), 0, 255))
    tsi_u8 = int(np.clip(round((tsi - 15.0) / 25.0 * 255.0), 0, 255))
    flags = [
        1 if 25.0 <= tsi <= 30.0 else 0,
        1 if tsi < 25.0 else 0,
        1 if tsi > 30.0 else 0,
        1 if 40.0 <= RH <= 70.0 else 0,
        1 if RH < 20.0 or RH > 85.0 else 0,
        1 if abs(tsi - 27.5) < 1.0 else 0,
        1 if abs(tsi - 27.5) < 2.5 else 0,
        1 if abs(tsi - 27.5) < 5.0 else 0,
    ]
    return np.array(
        u8_bits(t_u8) + u8_bits(rh_u8) + u8_bits(tsi_u8) + flags,
        dtype=np.int64,
    )


ENCODERS = {
    "baseline": encode_baseline,
    "rich32": encode_rich32,
}


def continuous_feats(T: float, RH: float) -> np.ndarray:
    tsi = tsi_celsius(T, RH, 0.0)
    tw = wet_bulb(T, RH)
    return np.array(
        [
            T,
            RH,
            tsi,
            tw,
            abs(tsi - 27.5),
            max(0.0, 25.0 - tsi),
            max(0.0, tsi - 30.0),
            1.0 if 25.0 <= tsi <= 30.0 else 0.0,
            1.0 if 40.0 <= RH <= 70.0 else 0.0,
            1.0 if RH < 20.0 or RH > 85.0 else 0.0,
            (RH - 55.0) / 45.0,
            (T - 30.0) / 12.0,
        ],
        dtype=np.float64,
    )


def make_rows(n: int, seed: int, encoder):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        T = float(rng.uniform(18.0, 42.0))
        RH = float(rng.uniform(10.0, 95.0))
        rows.append(
            dict(
                T=T,
                RH=RH,
                p_star=comfort_p_star(T, RH),
                x=encoder(T, RH),
                xc=continuous_feats(T, RH),
            )
        )
    return rows


class FlexSENA:
    def __init__(self, n_in: int, n_hid: int, seed: int = 0, N: int = 64):
        rng = np.random.default_rng(seed)
        self.n_in = n_in
        self.n_hid = n_hid
        self.N = int(N)
        self.rng = rng
        self.w_hid = rng.integers(-8, 9, size=(n_hid, n_in + 1), dtype=np.int64)
        self.w_out = rng.integers(-8, 9, size=(1, n_hid + 1), dtype=np.int64)

    def _fire(self, acc: int) -> int:
        r = int(self.rng.integers(-self.N, self.N + 1))
        return 1 if acc > r else 0

    def forward_once(self, x):
        x = np.asarray(x, np.int64)
        xb = np.append(x, 1)
        pre_h = self.w_hid @ xb
        rnd_h = self.rng.integers(-self.N, self.N + 1, size=self.n_hid)
        h = (pre_h > rnd_h).astype(np.int64)
        hb = np.append(h, 1)
        pre_o = int(self.w_out[0] @ hb)
        y = self._fire(pre_o)
        return h, y, pre_h, pre_o

    def p_hat(self, x, T_passes: int = 100) -> float:
        x = np.asarray(x, np.int64)
        xb = np.append(x, 1)
        s = 0
        for _ in range(T_passes):
            pre_h = self.w_hid @ xb
            rnd_h = self.rng.integers(-self.N, self.N + 1, size=self.n_hid)
            h = (pre_h > rnd_h).astype(np.int64)
            hb = np.append(h, 1)
            pre_o = int(self.w_out[0] @ hb)
            s += 1 if pre_o > int(self.rng.integers(-self.N, self.N + 1)) else 0
        return s / float(T_passes)

    def train_sample(self, x, p_star: float, lr_passes: int = 2, extreme_boost: bool = False):
        x = np.asarray(x, np.int64)
        boost = 1
        if extreme_boost and (p_star < 0.15 or p_star > 0.85):
            boost = 3
        for _ in range(lr_passes * boost):
            t = 1 if self.rng.random() < p_star else 0
            h, y, pre_h, pre_o = self.forward_once(x)
            d = t - int(y)
            if d == 0:
                continue
            sign_h = np.where(np.append(h, 1) > 0, 1, -1)
            self.w_out[0] += d * sign_h
            if abs(int(pre_o)) <= int(self.rng.integers(0, max(2, self.N // 4))):
                back = d * self.w_out[0, : self.n_hid]
                for j in range(self.n_hid):
                    if abs(int(pre_h[j])) <= int(self.rng.integers(0, max(2, self.N // 4))):
                        if abs(int(back[j])) > int(self.rng.integers(0, 4)):
                            dir_j = 1 if back[j] > 0 else -1
                            sign_i = np.where(np.append(x, 1) > 0, 1, -1)
                            self.w_hid[j] += dir_j * sign_i
            np.clip(self.w_hid, -CLIP, CLIP, out=self.w_hid)
            np.clip(self.w_out, -CLIP, CLIP, out=self.w_out)


def fit_affine(p_raw: np.ndarray, p_star: np.ndarray) -> tuple[float, float]:
    A = np.column_stack([np.ones(len(p_raw)), p_raw])
    coef, _, _, _ = np.linalg.lstsq(A, p_star, rcond=None)
    return float(coef[0]), float(coef[1])


def apply_affine(p: float, a: float, b: float) -> float:
    return float(np.clip(a + b * p, 0.0, 1.0))


def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    bias = float(np.mean(err))
    if np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        corr = float("nan")
    else:
        corr = float(np.corrcoef(y_true, y_pred)[0, 1])
    agree = float(np.mean((y_pred >= 0.5) == (y_true >= 0.5)))
    return dict(
        mae=mae,
        rmse=rmse,
        bias=bias,
        corr=corr,
        agree05=agree,
        phat_min=float(np.min(y_pred)),
        phat_max=float(np.max(y_pred)),
        phat_mean=float(np.mean(y_pred)),
    )


@dataclass
class Cfg:
    name: str
    encoder: str
    n_hid: int
    n_train: int
    epochs: int
    extreme_boost: bool = False
    calibrate: bool = False  # host-side only; zero extra FPGA cost


def run_mazumdar(cfg: Cfg, seed: int, t_eval: int, t_calib: int) -> dict:
    enc = ENCODERS[cfg.encoder]
    # probe n_in
    n_in = int(enc(25.0, 50.0).shape[0])
    res = estimate_resources(n_in, cfg.n_hid, M_ENS)
    if not res["fits"]:
        return dict(
            name=cfg.name,
            kind="mazumdar_rejected",
            reason=f"est LUT {res['est_lut']} ({res['est_lut_pct']}%) > {int(LUT_CAP_FRAC*100)}% LX9 cap",
            n_in=n_in,
            n_hid=cfg.n_hid,
            resources=res,
        )

    train = make_rows(cfg.n_train, seed, enc)
    test = make_rows(400, seed + 1, enc)
    Xtr = np.stack([r["x"] for r in train])
    ytr = np.array([r["p_star"] for r in train])
    Xte = np.stack([r["x"] for r in test])
    yte = np.array([r["p_star"] for r in test])

    members: list[FlexSENA] = []
    t0 = time.time()
    for m in range(M_ENS):
        net = FlexSENA(n_in=n_in, n_hid=cfg.n_hid, seed=seed + 17 * m, N=64)
        for ep in range(1, cfg.epochs + 1):
            order = np.random.default_rng(seed + ep + m * 1009).permutation(len(Xtr))
            for n in order:
                net.train_sample(
                    Xtr[n], float(ytr[n]), lr_passes=2, extreme_boost=cfg.extreme_boost
                )
            if ep == 1 or ep % max(1, cfg.epochs // 4) == 0 or ep == cfg.epochs:
                idx = order[:120]
                pred = np.array([net.p_hat(Xtr[i], T_passes=30) for i in idx])
                mse = float(np.mean((pred - ytr[idx]) ** 2))
                print(f"  [{cfg.name}] m{m} ep{ep:3d}/{cfg.epochs} mse@30={mse:.4f}")
        members.append(net)

    def ens_p(x, T):
        return float(np.mean([members[m].p_hat(x, T_passes=T) for m in range(M_ENS)]))

    a, b = 0.0, 1.0
    if cfg.calibrate:
        # subsample train for speed
        idx = np.linspace(0, len(Xtr) - 1, min(800, len(Xtr)), dtype=int)
        raw_tr = np.array([ens_p(Xtr[i], t_calib) for i in idx])
        a, b = fit_affine(raw_tr, ytr[idx])
        print(f"  [{cfg.name}] host calib a={a:.3f} b={b:.3f} (no FPGA cost)")

    pred_te = []
    for i in range(len(Xte)):
        p = ens_p(Xte[i], t_eval)
        if cfg.calibrate:
            p = apply_affine(p, a, b)
        pred_te.append(p)
    pred_te = np.array(pred_te)
    out = metrics(yte, pred_te)
    out.update(
        name=cfg.name,
        encoder=cfg.encoder,
        n_in=n_in,
        n_hid=cfg.n_hid,
        M=M_ENS,
        n_train=cfg.n_train,
        epochs=cfg.epochs,
        calibrate=cfg.calibrate,
        calib_a=a,
        calib_b=b,
        extreme_boost=cfg.extreme_boost,
        t_eval=t_eval,
        seconds=round(time.time() - t0, 1),
        kind="mazumdar",
        resources=res,
        device="Papilio Pro XC6SLX9",
    )
    return out


def run_skyline(seed: int) -> dict:
    """Float OLS ceiling — NOT implementable as-is on LX9 SENA; reference only."""
    train = make_rows(4000, seed, encode_baseline)
    test = make_rows(400, seed + 1, encode_baseline)
    Xtr = np.stack([r["xc"] for r in train])
    ytr = np.array([r["p_star"] for r in train])
    Xte = np.stack([r["xc"] for r in test])
    yte = np.array([r["p_star"] for r in test])

    def expand(X):
        tsi = X[:, 2:3]
        dev = X[:, 4:5]
        return np.concatenate(
            [X, tsi ** 2, dev ** 2, X[:, 0:1] * X[:, 1:2] / 1000.0], axis=1
        )

    Xtr_e = np.column_stack([np.ones(len(Xtr)), expand(Xtr)])
    Xte_e = np.column_stack([np.ones(len(Xte)), expand(Xte)])
    coef, _, _, _ = np.linalg.lstsq(Xtr_e, ytr, rcond=None)
    pred = np.clip(Xte_e @ coef, 0.0, 1.0)
    out = metrics(yte, pred)
    out.update(
        name="Z_skyline_ols_float",
        kind="float_ols_reference",
        n_train=4000,
        seconds=0.0,
        note="Not LX9-SENA; task ceiling with float features",
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--t-eval", type=int, default=100)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    # Resource map for logging
    probe = {
        "baseline_16_16": estimate_resources(16, 16),
        "rich32_16": estimate_resources(32, 16),
        "rich32_20": estimate_resources(32, 20),
        "rich32_24": estimate_resources(32, 24),
        "REJECT_rich48_16": estimate_resources(48, 16),
        "REJECT_rich32_32": estimate_resources(32, 32),
    }
    print("Papilio Pro resource estimates (cap {:.0f}% LUT):".format(LUT_CAP_FRAC * 100))
    for k, v in probe.items():
        print(f"  {k}: LUT~{v['est_lut']} ({v['est_lut_pct']}%) fits={v['fits']}")

    if args.quick:
        cfgs = [
            Cfg("A_baseline_silicon", "baseline", 16, 800, 15),
            Cfg("B_rich32_16", "rich32", 16, 800, 15),
            Cfg("C_rich32_16_calib", "rich32", 16, 800, 15, extreme_boost=True, calibrate=True),
        ]
    else:
        # Only LX9-fitting topologies; train harder is free on soft side
        cfgs = [
            Cfg("A_baseline_silicon", "baseline", 16, 2000, 30),  # matches RTL
            Cfg("B_rich32_16", "rich32", 16, 2000, 40),
            Cfg("C_rich32_16_heavy", "rich32", 16, 4000, 50, extreme_boost=True),
            Cfg(
                "D_rich32_16_heavy_calib",
                "rich32",
                16,
                4000,
                50,
                extreme_boost=True,
                calibrate=True,
            ),
            # optional slightly wider hid if estimate fits
            Cfg("E_rich32_20", "rich32", 20, 2000, 40),
        ]

    results = []
    for cfg in cfgs:
        print(f"\n=== {cfg.name} ===")
        r = run_mazumdar(cfg, args.seed, args.t_eval, t_calib=60)
        if r.get("kind") == "mazumdar_rejected":
            print("  REJECTED:", r["reason"], r["resources"])
        else:
            keys = (
                "name",
                "mae",
                "rmse",
                "bias",
                "corr",
                "agree05",
                "phat_min",
                "phat_max",
                "seconds",
            )
            print(json.dumps({k: r[k] for k in keys}, indent=2))
            print("  resources", r["resources"])
        results.append(r)

    print("\n=== Z_skyline_ols_float (reference only) ===")
    sky = run_skyline(args.seed)
    print(json.dumps({k: sky[k] for k in ("name", "mae", "rmse", "bias", "corr", "agree05")}, indent=2))
    results.append(sky)

    maz = [r for r in results if r.get("kind") == "mazumdar"]
    rejected = [r for r in results if r.get("kind") == "mazumdar_rejected"]
    best = min(maz, key=lambda r: r["mae"]) if maz else None
    base = next((r for r in maz if r["name"].startswith("A_")), None)
    summary = dict(
        t="soft_limit_sweep_papilio_pro",
        device="Papilio Pro XC6SLX9",
        measured_baseline_lut=BASE_LUT,
        measured_baseline_slices=BASE_SLICES,
        lut_cap_frac=LUT_CAP_FRAC,
        resource_probe=probe,
        t_eval=args.t_eval,
        quick=bool(args.quick),
        baseline_mae=None if base is None else base["mae"],
        best_mazumdar=None if best is None else best["name"],
        best_mae=None if best is None else best["mae"],
        delta_vs_baseline=None
        if best is None or base is None
        else best["mae"] - base["mae"],
        skyline_mae=sky["mae"],
        gap_best_to_skyline=None if best is None else best["mae"] - sky["mae"],
        rejected=[r["name"] for r in rejected],
        note=(
            "Only Mazumdar topologies estimated to fit LX9 (≤85% LUT) are trained. "
            "Host affine calib costs no FPGA fabric. Skyline OLS is non-SENA reference."
        ),
        results=results,
    )
    path = RES / "soft_limit_sweep.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n==== SUMMARY ====")
    print(json.dumps({k: summary[k] for k in summary if k not in ("results", "resource_probe")}, indent=2))
    print("wrote", path)


if __name__ == "__main__":
    raise SystemExit(main())
