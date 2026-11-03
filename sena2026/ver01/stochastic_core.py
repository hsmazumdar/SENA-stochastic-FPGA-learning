"""
stochastic_core.py
==================
Faithful, vectorised implementation of the multiplier-free stochastic
feed-forward neural network originally proposed by H. S. Mazumdar (1995),
"A Multilayered Feed Forward Neural Network Suitable for VLSI Implementation".

The network uses ONLY add / subtract / compare / logical operators:

Forward (random-threshold neuron, Eq. 3 of the paper):
        y = 1 if (x + r) > 0 else 0,      r ~ U[-N, N]

Reverse characteristic (derivative proxy R(X), Eq. 4):
        R(x) = 1 if |x| <= r else 0,      r ~ U[0, N]

Weight update (integer, sign-based -- no multiplications):
    output layer:  w_kj += (out_k - y_k)          if y_j = 1
                   w_kj -= (out_k - y_k)          if y_j = 0
    hidden layer:  d = R(x_j) AND ( |back_j| > r )
                   w_ji += d  (sign set by back_j and input bit y_i)

This module also provides:
  * an integer LFSR random source (hardware-realistic) and a numpy source,
  * weight quantisation to B bits (for the bit-precision study),
  * a real-valued back-propagation baseline (float64) for comparison,
  * temporal / ensemble voting inference (the SENA read-out).

All randomness is seedable for full reproducibility.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field


# --------------------------------------------------------------------------- #
#  Hardware-realistic random sources
# --------------------------------------------------------------------------- #
class LFSR:
    """32-bit Galois LFSR -- models the on-chip random source of the ASIC/FPGA.

    Produces integers; `randint(low, high)` maps the raw stream into a range
    using a modulo reduction (as a small hardware divider / mask would)."""

    def __init__(self, seed: int = 0xACE1, taps: int = 0xB4BCD35C):
        self.state = np.uint32(seed if seed & 0xFFFFFFFF else 0xACE1)
        self.taps = np.uint32(taps)

    def _next(self) -> int:
        s = int(self.state)
        lsb = s & 1
        s >>= 1
        if lsb:
            s ^= int(self.taps)
        self.state = np.uint32(s & 0xFFFFFFFF)
        return int(self.state)

    def randint(self, low: int, high: int, size=None):
        span = high - low
        if size is None:
            return low + (self._next() % span)
        n = int(np.prod(size))
        out = np.empty(n, dtype=np.int64)
        for i in range(n):
            out[i] = low + (self._next() % span)
        return out.reshape(size)


class NumpyRNG:
    """Fast Mersenne-Twister source (used for large sweeps)."""

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def randint(self, low, high, size=None):
        return self.rng.integers(low, high, size=size)


def make_rng(kind: str = "numpy", seed: int = 0):
    return LFSR(seed=seed or 0xACE1) if kind == "lfsr" else NumpyRNG(seed)


# --------------------------------------------------------------------------- #
#  Configuration
# --------------------------------------------------------------------------- #
@dataclass
class StochasticConfig:
    layer_sizes: tuple = (64, 32, 10)     # includes input/hidden/output widths
    N: int = 64                            # dynamic range of the random threshold
    N_rev_out: int = 48                    # N1 : reverse range, output layer
    N_rev_hidhid: int = 48                 # N2 : reverse range, hidden pre-activation
    N_rev_back: int = 24                   # N3 : threshold on back-propagated signal
    weight_bits: int = 16                  # integer weight width (quantisation study)
    epochs: int = 40
    seed: int = 0
    rng_kind: str = "numpy"                # 'numpy' or 'lfsr'
    lr_passes: int = 1                     # weight-update repeats per sample
    init_scale: int = 2                    # +/- range of initial integer weights


# --------------------------------------------------------------------------- #
#  Stochastic network
# --------------------------------------------------------------------------- #
class StochasticNet:
    """Multiplier-free stochastic MLP with integer weights and a bias unit."""

    def __init__(self, cfg: StochasticConfig):
        self.cfg = cfg
        self.rng = make_rng(cfg.rng_kind, cfg.seed)
        self._np = np.random.default_rng(cfg.seed + 12345)
        self.sizes = list(cfg.layer_sizes)
        # weights[l] : shape (out, in+1)  -- last column is the bias weight
        self.weights = []
        for a, b in zip(self.sizes[:-1], self.sizes[1:]):
            w = self._np.integers(-cfg.init_scale, cfg.init_scale + 1,
                                   size=(b, a + 1)).astype(np.int64)
            self.weights.append(w)
        self._wclip = (1 << (cfg.weight_bits - 1)) - 1
        n_lyr = len(self.weights)
        # Per-layer programmable dynamic range N (a hardware register per layer).
        # Calibrated from pre-activation statistics; the paper explicitly allows
        # "adjusting the value of N gradually as the error converges".
        self.N_fwd = [float(cfg.N)] * n_lyr          # forward random threshold
        self.N_back = float(cfg.N_rev_back)          # back-signal gate range
        self.alpha = 1.6                             # threshold-range / RMS ratio
        self._calibrated = False

    # ---- helpers ---------------------------------------------------------- #
    def _quantise(self):
        c = self._wclip
        for w in self.weights:
            np.clip(w, -c, c, out=w)

    @staticmethod
    def _with_bias(y: np.ndarray) -> np.ndarray:
        return np.concatenate([y, np.ones((y.shape[0], 1), dtype=y.dtype)], axis=1)

    # ---- dynamic-range calibration --------------------------------------- #
    def calibrate(self, X: np.ndarray, max_samples: int = 512):
        """Set each layer's random-threshold range N from pre-activation RMS.

        Propagates layer by layer so that every N reflects the real signal
        scale it must randomise over (prevents saturation / pure noise)."""
        if len(X) > max_samples:
            X = X[self._np.choice(len(X), max_samples, replace=False)]
        a = X.astype(np.int64)
        for l, w in enumerate(self.weights):
            x = self._with_bias(a) @ w.T
            rms = float(np.sqrt(np.mean(x.astype(float) ** 2))) + 1e-6
            self.N_fwd[l] = max(4.0, self.alpha * rms)
            r = self._np.integers(-int(self.N_fwd[l]), int(self.N_fwd[l]) + 1,
                                  size=x.shape)
            a = (x + r > 0).astype(np.int64)
        self._calibrated = True

    # ---- forward pass ----------------------------------------------------- #
    def forward(self, X: np.ndarray, stochastic: bool = True):
        """X : (batch, n_in) binary {0,1}. Returns (activations, pre-acts)."""
        acts = [X.astype(np.int64)]
        pres = []
        for l, w in enumerate(self.weights):
            yb = self._with_bias(acts[-1])
            x = yb @ w.T                              # integer accumulate (add/sub)
            pres.append(x)
            Nl = max(1, int(self.N_fwd[l]))
            if stochastic:
                r = self.rng.randint(-Nl, Nl, size=x.shape)
                y = (x + r > 0).astype(np.int64)
            else:
                y = (x > 0).astype(np.int64)           # deterministic threshold
            acts.append(y)
        return acts, pres

    # ---- training (one epoch) -------------------------------------------- #
    def train_epoch(self, X: np.ndarray, Y: np.ndarray, recalibrate: bool = True):
        """Online integer update per the appendix pseudocode (vectorised)."""
        cfg = self.cfg
        if not self._calibrated or recalibrate:
            self.calibrate(X)
        # reverse-characteristic ranges track the forward dynamic ranges
        n_out_range = max(2, int(self.N_fwd[-1]))
        n_hid_range = max(2, int(self.N_fwd[-2]))
        idx = self._np.permutation(len(X))
        for n in idx:
            x_in = X[n:n + 1]
            target = Y[n]                              # (n_out,) in {0,1}
            acts, pres = self.forward(x_in, stochastic=True)
            # ----- output layer -----
            yk = acts[-1][0]                           # (n_out,)
            xk = pres[-1][0]
            d = target - yk                            # {-1,0,1}
            y_hidden = acts[-2][0]                     # (n_hid,)
            yb_hidden = np.append(y_hidden, 1)         # bias
            Wk = self.weights[-1]                      # (n_out, n_hid+1)
            # R(x_k): sensitivity gate (range tracks output pre-activation scale)
            r1 = self.rng.randint(0, n_out_range, size=xk.shape)
            p = (np.abs(xk) <= r1).astype(np.int64)    # (n_out,)
            # weight update: w += d if y_j=1 else w -= d
            sign_in = np.where(yb_hidden > 0, 1, -1)   # (n_hid+1,)
            Wk += np.outer(d, sign_in)
            # back-propagated signal to hidden units (uses UPDATED Wk, per appendix)
            gate = (d != 0) & (p != 0)                 # (n_out,)
            contrib = (np.sign(d) * gate)[:, None] * Wk[:, :-1]   # exclude bias col
            back = contrib.sum(axis=0)                 # (n_hid,)

            # ----- hidden layer -----
            xj = pres[-2][0]
            r2 = self.rng.randint(0, n_hid_range, size=xj.shape)
            a = (np.abs(xj) <= r2).astype(np.int64)
            # back-signal gate: range adapts to |back| via a slow EMA
            back_rms = float(np.sqrt(np.mean(back.astype(float) ** 2))) + 1e-6
            self.N_back = 0.99 * self.N_back + 0.01 * (0.8 * back_rms)
            nb = max(1, int(self.N_back))
            r3 = self.rng.randint(0, nb, size=back.shape)
            b = (np.abs(back) > r3).astype(np.int64)
            dj = a & b                                 # {0,1}
            x_bits = np.append(x_in[0], 1)             # inputs + bias
            sign_i = np.where(x_bits > 0, 1, -1)
            dir_j = np.where(back > 0, 1, -1)          # outj sign
            Wj = self.weights[-2]                      # (n_hid, n_in+1)
            update = (dj * dir_j)[:, None] * sign_i[None, :]
            Wj += update
            self._quantise()

    # ---- full training driver -------------------------------------------- #
    def fit(self, X, Y, Xval=None, yval=None, epochs=None,
            recal_every=4, T_eval=64, verbose=False):
        """Train with a calibration schedule; optionally track val accuracy."""
        epochs = epochs or self.cfg.epochs
        history = []
        self.calibrate(X)
        for e in range(epochs):
            recal = (recal_every > 0 and e > 0 and e % recal_every == 0)
            self.train_epoch(X, Y, recalibrate=recal)
            if Xval is not None:
                acc = self.accuracy(Xval, yval, T=T_eval)
                history.append(acc)
                if verbose:
                    print(f"  epoch {e:3d}  val_acc={acc:.3f}")
        return history

    # ---- inference (SENA temporal/ensemble voting) ----------------------- #
    def predict_votes(self, X: np.ndarray, T: int = 32):
        """Return per-class vote counts over T stochastic passes."""
        n_out = self.sizes[-1]
        votes = np.zeros((len(X), n_out), dtype=np.int64)
        for _ in range(T):
            acts, _ = self.forward(X, stochastic=True)
            votes += acts[-1]
        return votes

    def predict(self, X: np.ndarray, T: int = 32):
        votes = self.predict_votes(X, T)
        if self.sizes[-1] == 1:
            return (votes[:, 0] > T / 2).astype(np.int64)
        return np.argmax(votes, axis=1)

    def accuracy(self, X, y_labels, T: int = 32):
        pred = self.predict(X, T)
        return float(np.mean(pred == y_labels))


# --------------------------------------------------------------------------- #
#  Real-valued back-propagation baseline (float64, uses multiplications)
# --------------------------------------------------------------------------- #
class RealBPNet:
    """Standard sigmoid MLP trained with gradient descent -- the 1995 baseline."""

    def __init__(self, sizes, seed=0, lr=0.5):
        rng = np.random.default_rng(seed)
        self.sizes = list(sizes)
        self.W = [rng.normal(0, 1 / np.sqrt(a), (b, a + 1))
                  for a, b in zip(self.sizes[:-1], self.sizes[1:])]
        self.lr = lr

    @staticmethod
    def _sig(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

    @staticmethod
    def _bias(y):
        return np.concatenate([y, np.ones((y.shape[0], 1))], axis=1)

    def forward(self, X):
        a = [X.astype(float)]
        for w in self.W:
            z = self._bias(a[-1]) @ w.T
            a.append(self._sig(z))
        return a

    def train_epoch(self, X, Y):
        a = self.forward(X)
        delta = (a[-1] - Y) * a[-1] * (1 - a[-1])
        for l in range(len(self.W) - 1, -1, -1):
            grad = delta.T @ self._bias(a[l]) / len(X)
            if l > 0:
                back = delta @ self.W[l][:, :-1]
                delta = back * a[l] * (1 - a[l])
            self.W[l] -= self.lr * grad

    def predict(self, X):
        out = self.forward(X)[-1]
        return (out[:, 0] > 0.5).astype(int) if self.sizes[-1] == 1 \
            else np.argmax(out, axis=1)

    def accuracy(self, X, y_labels):
        return float(np.mean(self.predict(X) == y_labels))
