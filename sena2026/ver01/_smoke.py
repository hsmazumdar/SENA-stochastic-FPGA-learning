import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from stochastic_core import StochasticNet, StochasticConfig, RealBPNet
from datasets import get_dataset

specs = {
    "symmetry": dict(hid=24, ep=60, T=128),
    "parity":   dict(hid=32, ep=120, T=256),
    "iris":     dict(hid=24, ep=60, T=128),
    "cancer":   dict(hid=48, ep=40, T=128),
    "digits":   dict(hid=128, ep=50, T=128),
}
for name, s in specs.items():
    accs = []
    for seed in range(3):
        d = get_dataset(name, seed=seed, n_levels=8)
        cfg = StochasticConfig(layer_sizes=(d["n_in"], s["hid"], d["n_out"]),
                               epochs=s["ep"], seed=seed)
        net = StochasticNet(cfg)
        net.fit(d["Xtr"], d["Ytr"], recal_every=4)
        accs.append(net.accuracy(d["Xte"], d["yte"], T=s["T"]))
    d = get_dataset(name, seed=0, n_levels=8)
    bp = RealBPNet((d["n_in"], s["hid"], d["n_out"]), seed=0, lr=0.5)
    for e in range(300):
        bp.train_epoch(d["Xtr"], d["Ytr"])
    print(f"{name:10s} stoch={np.mean(accs):.3f}+/-{np.std(accs):.3f}  "
          f"BP={bp.accuracy(d['Xte'], d['yte']):.3f}")
