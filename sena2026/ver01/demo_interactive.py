"""
Headless demo of SENA interactive I/O (same logic as the GUI).
Runs without a display — useful for CI / quick verification.
"""
from __future__ import annotations
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from stochastic_core import StochasticNet, StochasticConfig
from datasets import get_dataset

SPECS = {
    "symmetry": dict(hid=24, ep=30, T=64),
    "digits":   dict(hid=96, ep=25, T=64),
    "iris":     dict(hid=24, ep=30, T=64),
}
NAMES = {
    "symmetry": ["Asymmetric", "Symmetric"],
    "digits": [str(i) for i in range(10)],
    "iris": ["Setosa", "Versicolor", "Virginica"],
}


def sena_predict(net, X, T=64, theta=0.75):
    n_out = net.sizes[-1]
    votes = np.zeros(n_out, dtype=np.int64)
    exit_t, conf, pred_i = T, 0.0, 0
    for t in range(1, T + 1):
        acts, _ = net.forward(X, stochastic=True)
        votes += acts[-1][0]
        if n_out == 1:
            p1 = votes[0] / t
            conf = max(p1, 1 - p1)
            pred_i = 1 if p1 > 0.5 else 0
        else:
            pred_i = int(np.argmax(votes))
            conf = votes[pred_i] / t
        if conf >= theta:
            exit_t = t
            break
    return pred_i, conf, exit_t, votes


def demo_dataset(name):
    s = SPECS[name]
    print(f"\n{'='*60}\n DEMO: {name.upper()}\n{'='*60}")
    d = get_dataset(name, seed=0)
    cfg = StochasticConfig(layer_sizes=(d["n_in"], s["hid"], d["n_out"]),
                           epochs=s["ep"], seed=0)
    net = StochasticNet(cfg)
    t0 = time.time()
    net.fit(d["Xtr"], d["Ytr"], recal_every=4)
    acc = net.accuracy(d["Xte"], d["yte"], T=s["T"])
    print(f" Trained in {time.time()-t0:.1f}s  |  test accuracy = {acc*100:.1f}%")
    print(f" Interactive I/O — 5 random test samples (early-exit θ=0.75):\n")
    rng = np.random.default_rng(42)
    idx = rng.choice(len(d["Xte"]), size=5, replace=False)
    ok = 0
    for i in idx:
        X = d["Xte"][i:i+1]
        true = int(d["yte"][i])
        pred, conf, t_exit, votes = sena_predict(net, X, T=s["T"], theta=0.75)
        names = NAMES[name]
        mark = "OK" if pred == true else "MISS"
        if pred == true:
            ok += 1
        save = 100 * (1 - t_exit / s["T"])
        print(f"  [{mark}] true={names[true]:12s}  pred={names[pred]:12s}  "
              f"conf={conf*100:5.1f}%  exit t={t_exit:3d}/{s['T']}  "
              f"(−{save:.0f}% compute)")
    print(f"\n  Demo hit rate: {ok}/5")
    return acc


if __name__ == "__main__":
    print("SENA Interactive I/O — Headless Demo")
    print("Same inference path as the Tkinter GUI (Train -> Sample -> Predict).")
    for name in ("symmetry", "iris", "digits"):
        demo_dataset(name)
    print(f"\n{'='*60}")
    print(" To open the GUI:  python sena2026/sim/sena_gui.py")
    print(f"{'='*60}")
