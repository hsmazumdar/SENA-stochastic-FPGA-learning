"""
pythagoras_mazumdar.py
======================
Pythagoras trainer using MAZUMDAR'S 1995 ALGORITHM ONLY:

    operations used in training/inference:  ADD, SUBTRACT, COMPARE, LOGICAL, RNG
    NOT used:  multiply, divide, exponential

How an *analog* regression fits a binary stochastic network
------------------------------------------------------------
  input  : analog (a, b) in [0,1]  --> thermometer comparator bank
           (a >= ref_k), k = 1..L  per channel   [hardware: ADC ladder]
  target : h_norm in [0,1] --> probability-coded binary D per presentation:
           D = 1  iff  random() < h_norm          [hardware: compare with RNG]
  output : firing rate over T stochastic passes  ==>  h_hat in [0,1]
           (a counter; division by T is only the final read-out scaling)

Training thus teaches the output neuron to FIRE WITH PROBABILITY h_norm --
probability matching with the original integer weight-update rule from the
paper's appendix (StochasticNet.train_epoch).

Honest ops accounting (for the paper):
  per-presentation datapath: binary-masked accumulate (AND+add), compare
  with RNG, sign-based +/-1 weight update -- NO multiply/divide/exp.
  housekeeping (every ~10 epochs): RMS calibration of threshold range N
  (in hardware: a coarse power-of-2 range estimator; not per-sample).

Architecture (analog view):  2  ->  30  ->  1
(binary view: 2*L thermometer bits -> 30 stochastic perceptrons -> 1)
"""

from __future__ import annotations
import numpy as np

from stochastic_core import StochasticNet, StochasticConfig
from pythagoras_analog import shape_boundary_radius  # geometry reuse (no training)


class MazumdarPythagoras:
    """2-analog-in -> 30 hidden -> 1-analog-out via pure Mazumdar training."""

    def __init__(self, n_levels: int = 16, hidden: int = 30, seed: int = 0):
        self.n_levels = n_levels
        self.hidden = hidden
        self.seed = seed
        # fixed comparator references (hardware resistor ladder)
        self.refs = (np.arange(1, n_levels + 1) / (n_levels + 1.0))
        cfg = StochasticConfig(
            layer_sizes=(2 * n_levels, hidden, 1),
            epochs=1, seed=seed, weight_bits=16)
        self.net = StochasticNet(cfg)
        self._rng = np.random.default_rng(seed + 999)

    # ------------------------------------------------------------- encode --
    def encode(self, AB: np.ndarray) -> np.ndarray:
        """(n,2) analog in [0,1] -> (n, 2L) thermometer bits (compare only)."""
        AB = np.asarray(AB, dtype=float)
        a_bits = (AB[:, 0:1] >= self.refs[None, :]).astype(np.int64)
        b_bits = (AB[:, 1:2] >= self.refs[None, :]).astype(np.int64)
        return np.concatenate([a_bits, b_bits], axis=1)

    # --------------------------------------------------------------- train --
    def fit(self, AB: np.ndarray, h_norm: np.ndarray, epochs: int = 300,
            samples_per_epoch: int = 800, callback=None, probe_every: int = 20):
        """Probability-coded training with the 1995 integer rule.

        Each epoch: draw a random subset, resample Bernoulli targets
        D = (rand < h_norm)  [compare only], run one Mazumdar train pass.

        Stabilizer: the fixed +/-1 integer updates never stop dithering, so we
        periodically probe the error and KEEP THE BEST WEIGHT SNAPSHOT
        (a register bank in hardware). Final weights = best seen.
        """
        Xbits_full = self.encode(AB)
        h = np.asarray(h_norm, dtype=float).ravel()
        n = len(Xbits_full)
        self.net.calibrate(Xbits_full)
        mse_hist = []
        best_mse = np.inf
        best_state = None
        probe = self._rng.choice(n, size=min(500, n), replace=False)
        for ep in range(epochs):
            idx = self._rng.choice(n, size=min(samples_per_epoch, n),
                                   replace=False)
            Xb = Xbits_full[idx]
            # Bernoulli target: fires with probability h  (COMPARE with RNG)
            D = (self._rng.random(len(idx)) < h[idx]).astype(np.int64)
            recal = (ep > 0 and ep % 10 == 0)
            self.net.train_epoch(Xb, D.reshape(-1, 1), recalibrate=recal)
            if ep % probe_every == 0 or ep == epochs - 1:
                h_hat = self.predict_h(AB[probe], T=128)
                mse = float(np.mean((h_hat - h[probe]) ** 2))
                mse_hist.append(mse)
                if mse < best_mse:
                    best_mse = mse
                    best_state = ([w.copy() for w in self.net.weights],
                                  list(self.net.N_fwd), self.net.N_back)
                if callback:
                    callback(ep, mse)
        if best_state is not None:
            self.net.weights = [w.copy() for w in best_state[0]]
            self.net.N_fwd = list(best_state[1])
            self.net.N_back = best_state[2]
        return mse_hist

    # ------------------------------------------------------------- predict --
    def predict_h(self, AB: np.ndarray, T: int = 256) -> np.ndarray:
        """Analog output = firing rate over T stochastic passes (counter)."""
        Xbits = self.encode(AB)
        votes = self.net.predict_votes(Xbits, T=T)  # (n,1) counts
        return votes[:, 0] / float(T)               # read-out scaling only

    def eval_mse(self, AB: np.ndarray, h_norm: np.ndarray, T: int = 256) -> float:
        h_hat = self.predict_h(AB, T)
        return float(np.mean((h_hat - np.asarray(h_norm).ravel()) ** 2))

    # -------------------------------------------------------- reconstruct --
    def reconstruct_shape(self, shape: str = "circle", n_theta: int = 180,
                          size: int = 128, frac: float = 0.55, T: int = 256):
        """Same interface as AnalogMLP2301.reconstruct_shape."""
        theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
        R = shape_boundary_radius(shape, theta, size=size, frac=frac)
        scale = size / 2.0
        a = np.clip(R * np.abs(np.cos(theta)) / scale, 0, 1)
        b = np.clip(R * np.abs(np.sin(theta)) / scale, 0, 1)
        h_hat = self.predict_h(np.stack([a, b], axis=1), T=T)
        R_hat = h_hat * np.sqrt(2.0) * scale
        x = R_hat * np.cos(theta)
        y = R_hat * np.sin(theta)
        x_true = R * np.cos(theta)
        y_true = R * np.sin(theta)
        rmse = float(np.sqrt(np.mean((R_hat - R) ** 2))) / scale
        return x, y, x_true, y_true, R_hat, R, theta, rmse


def dataset_from_shape(img: np.ndarray, n_extra: int = 1000, seed: int = 0,
                       max_from_image: int = 3000):
    """Balanced (white==black) analog (a,b) + h_norm dataset from shape image."""
    from pythagoras_analog import samples_from_shape_image
    X, y, n_per = samples_from_shape_image(
        img, n_extra=n_extra, seed=seed, max_from_image=max_from_image)
    return X, y.ravel(), n_per


if __name__ == "__main__":
    # quick self-test
    import time
    from pythagoras_analog import make_shape_image
    img = make_shape_image("circle", 128, frac=0.55)
    AB, h, n_per = dataset_from_shape(img)
    print(f"dataset {len(AB)} samples, balanced {n_per}W=={n_per}B")
    m = MazumdarPythagoras(n_levels=24, hidden=30, seed=0)
    t0 = time.time()
    m.fit(AB, h, epochs=300, samples_per_epoch=800,
          callback=lambda e, s: print(f"  ep {e:4d}  probe mse={s:.5f}"))
    print(f"train {time.time()-t0:.1f}s")
    print(f"final MSE={m.eval_mse(AB[:500], h[:500], T=256):.5f}")
    for sh in ("circle", "square", "triangle"):
        *_, rmse = m.reconstruct_shape(sh, 180, 128, 0.55, T=256)
        print(f"{sh} RMSE={rmse:.4f}")
