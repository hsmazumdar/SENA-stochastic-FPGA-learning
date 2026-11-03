"""
exp_digital.py
==============
Digital-track experiments for the 2026 stochastic-VLSI paper.

Produces (results/digital.json):
  1. benchmark      : stochastic vs real-BP accuracy on 5 datasets (mean+/-std)
  2. convergence    : val-accuracy vs epoch curves
  3. bit_precision  : accuracy vs integer weight width (4..16 bits)
  4. rng_study      : accuracy vs random source quality (LFSR / MT / weak)
  5. temporal       : accuracy vs number of stochastic votes T (SENA read-out)
  6. noise          : accuracy vs input bit-flip probability (robustness)
  7. op_counts      : multiplications vs additions per inference (energy proxy)

All runs are seed-controlled and reproducible.
"""

from __future__ import annotations
import sys, os, json, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from stochastic_core import StochasticNet, StochasticConfig, RealBPNet
from datasets import get_dataset

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)

# dataset -> (hidden, epochs, T votes)
SPECS = {
    "symmetry": dict(hid=24, ep=60,  T=128),
    "xor2d":    dict(hid=32, ep=60,  T=256),
    "iris":     dict(hid=24, ep=60,  T=128),
    "cancer":   dict(hid=48, ep=45,  T=128),
    "digits":   dict(hid=128, ep=50, T=128),
}
SEEDS = [0, 1, 2]


def train_stoch(d, hid, ep, seed, weight_bits=16, rng_kind="numpy"):
    cfg = StochasticConfig(layer_sizes=(d["n_in"], hid, d["n_out"]),
                           epochs=ep, seed=seed, weight_bits=weight_bits,
                           rng_kind=rng_kind)
    net = StochasticNet(cfg)
    net.fit(d["Xtr"], d["Ytr"], recal_every=4)
    return net


# --------------------------------------------------------------------------- #
def run_benchmark():
    out = {}
    for name, s in SPECS.items():
        s_acc, bp_acc = [], []
        for seed in SEEDS:
            d = get_dataset(name, seed=seed)
            net = train_stoch(d, s["hid"], s["ep"], seed)
            s_acc.append(net.accuracy(d["Xte"], d["yte"], T=s["T"]))
            bp = RealBPNet((d["n_in"], s["hid"], d["n_out"]), seed=seed, lr=0.5)
            for _ in range(300):
                bp.train_epoch(d["Xtr"], d["Ytr"])
            bp_acc.append(bp.accuracy(d["Xte"], d["yte"]))
        out[name] = dict(
            n_in=d["n_in"], n_hid=s["hid"], n_out=d["n_out"], k=d["k"],
            stoch_mean=float(np.mean(s_acc)), stoch_std=float(np.std(s_acc)),
            bp_mean=float(np.mean(bp_acc)), bp_std=float(np.std(bp_acc)),
            stoch_all=[float(x) for x in s_acc], bp_all=[float(x) for x in bp_acc])
        print(f"[bench] {name:9s} stoch={out[name]['stoch_mean']:.3f} "
              f"BP={out[name]['bp_mean']:.3f}")
    return out


def run_convergence():
    out = {}
    for name in ["digits", "cancer"]:
        s = SPECS[name]
        d = get_dataset(name, seed=0)
        net = StochasticNet(StochasticConfig(
            layer_sizes=(d["n_in"], s["hid"], d["n_out"]), epochs=s["ep"], seed=0))
        hist = net.fit(d["Xtr"], d["Ytr"], Xval=d["Xte"], yval=d["yte"],
                       recal_every=4, T_eval=64)
        # BP convergence (per-epoch test acc)
        bp = RealBPNet((d["n_in"], s["hid"], d["n_out"]), seed=0, lr=0.5)
        bp_hist = []
        for _ in range(s["ep"] * 4):
            bp.train_epoch(d["Xtr"], d["Ytr"])
            bp_hist.append(bp.accuracy(d["Xte"], d["yte"]))
        out[name] = dict(stoch=hist, bp=bp_hist)
        print(f"[conv] {name} done")
    return out


def run_bit_precision():
    bits = [4, 6, 8, 10, 12, 16]
    out = {}
    for name in ["digits", "cancer"]:
        s = SPECS[name]
        accs = {}
        for wb in bits:
            vals = []
            for seed in SEEDS:
                d = get_dataset(name, seed=seed)
                net = train_stoch(d, s["hid"], s["ep"], seed, weight_bits=wb)
                vals.append(net.accuracy(d["Xte"], d["yte"], T=s["T"]))
            accs[wb] = [float(np.mean(vals)), float(np.std(vals))]
            print(f"[bits] {name} {wb}b -> {accs[wb][0]:.3f}")
        out[name] = accs
    return out


def run_rng_study():
    out = {}
    name = "digits"; s = SPECS[name]
    for rng_kind in ["lfsr", "numpy"]:
        vals = []
        for seed in SEEDS:
            d = get_dataset(name, seed=seed)
            net = train_stoch(d, s["hid"], s["ep"], seed, rng_kind=rng_kind)
            vals.append(net.accuracy(d["Xte"], d["yte"], T=s["T"]))
        out[rng_kind] = [float(np.mean(vals)), float(np.std(vals))]
        print(f"[rng] {rng_kind} -> {out[rng_kind][0]:.3f}")
    return out


def run_temporal():
    out = {}
    for name in ["digits", "cancer", "xor2d"]:
        s = SPECS[name]
        d = get_dataset(name, seed=0)
        net = train_stoch(d, s["hid"], s["ep"], 0)
        Ts = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        accs = [net.accuracy(d["Xte"], d["yte"], T=t) for t in Ts]
        out[name] = dict(T=Ts, acc=[float(a) for a in accs])
        print(f"[temporal] {name} done")
    return out


def run_noise():
    out = {}
    for name in ["digits", "cancer"]:
        s = SPECS[name]
        d = get_dataset(name, seed=0)
        net = train_stoch(d, s["hid"], s["ep"], 0)
        bp = RealBPNet((d["n_in"], s["hid"], d["n_out"]), seed=0, lr=0.5)
        for _ in range(300):
            bp.train_epoch(d["Xtr"], d["Ytr"])
        rng = np.random.default_rng(7)
        ps = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]
        s_acc, b_acc = [], []
        for p in ps:
            flip = rng.random(d["Xte"].shape) < p
            Xn = np.abs(d["Xte"] - flip.astype(int))
            s_acc.append(net.accuracy(Xn, d["yte"], T=s["T"]))
            b_acc.append(bp.accuracy(Xn, d["yte"]))
        out[name] = dict(p=ps, stoch=[float(a) for a in s_acc],
                         bp=[float(a) for a in b_acc])
        print(f"[noise] {name} done")
    return out


def op_counts():
    """Multiplications vs additions per forward inference (single sample)."""
    out = {}
    for name, s in SPECS.items():
        d = get_dataset(name, seed=0)
        sizes = [d["n_in"], s["hid"], d["n_out"]]
        macs = sum(a * b for a, b in zip(sizes[:-1], sizes[1:]))  # BP mult-adds
        # stochastic: on average half the input bits are '1' -> conditional adds
        adds = sum(int(0.5 * (a + 1)) * b for a, b in zip(sizes[:-1], sizes[1:]))
        neurons = s["hid"] + d["n_out"]
        out[name] = dict(bp_mults=macs, bp_adds=macs,
                         stoch_mults=0, stoch_adds=adds,
                         stoch_compares=neurons, stoch_rng=neurons)
    return out


if __name__ == "__main__":
    t0 = time.time()
    results = {}
    results["benchmark"]     = run_benchmark()
    results["op_counts"]     = op_counts()
    results["temporal"]      = run_temporal()
    results["noise"]         = run_noise()
    results["rng_study"]     = run_rng_study()
    results["convergence"]   = run_convergence()
    results["bit_precision"] = run_bit_precision()
    results["_meta"] = dict(seeds=SEEDS, specs=SPECS,
                            runtime_s=round(time.time() - t0, 1))
    with open(os.path.join(RESULTS, "digital.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDONE digital in {time.time()-t0:.1f}s -> results/digital.json")
