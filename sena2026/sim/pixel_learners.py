"""
pixel_learners.py
=================
Two 2 -> 30 -> 1 learners that consume the real-time pixel stream from
shape_bmp.PixelStream and map

    (x / 64,  y / 64)  ->  pixel colour  {black = 0, white = 1}

Both expose the same streaming interface so the GUI can swap them live:

    train_chunk(X, y)          one incremental update from a fresh batch
    predict(X)   -> [0,1]      analog output (probability / firing rate)
    predict_grid(size)         (size, size) output map, row = y  (origin upper)
    accuracy(X, y)             fraction correct at the 0.5 decision level

PixelMazumdar   Mazumdar 1995: ADD / SUB / COMPARE / LOGIC / RNG only.
                No multiply, divide or exponential in the datapath.
                The pixel colour IS the binary target, so no probability
                coding is needed -- this is the algorithm's native setting.

PixelMLP        Conventional float back-propagation (tanh + sigmoid).
                Reference baseline that uses multipliers and exp().
"""

from __future__ import annotations
import numpy as np

from stochastic_core import StochasticNet, StochasticConfig


# --------------------------------------------------------------------------- #
#  Mazumdar 1995 -- multiplier-free
# --------------------------------------------------------------------------- #
class PixelMazumdar:
    """Thermometer-coded (x, y) -> 30 random-threshold hidden -> 1 output bit."""

    name = "Mazumdar 1995 (no mult/div/exp)"

    def __init__(self, n_levels: int = 32, hidden: int = 30, seed: int = 0):
        self.n_levels = n_levels
        # fixed comparator ladder references (resistor string in hardware)
        self.refs = np.arange(1, n_levels + 1) / (n_levels + 1.0)
        cfg = StochasticConfig(layer_sizes=(2 * n_levels, hidden, 1),
                               epochs=1, seed=seed, weight_bits=16)
        self.net = StochasticNet(cfg)
        self._chunks = 0
        self._ready = False
        self._prng = np.random.default_rng(seed + 4242)   # read-out RNG only

    def encode(self, X: np.ndarray) -> np.ndarray:
        """Analog pair -> 2*L thermometer bits. Comparators only."""
        X = np.asarray(X, dtype=float)
        xb = (X[:, 0:1] >= self.refs[None, :]).astype(np.int64)
        yb = (X[:, 1:2] >= self.refs[None, :]).astype(np.int64)
        return np.concatenate([xb, yb], axis=1)

    def train_chunk(self, X: np.ndarray, y: np.ndarray, recal_every: int = 20):
        bits = self.encode(X)
        if not self._ready:
            self.net.calibrate(bits)
            self._ready = True
        recal = (self._chunks > 0 and self._chunks % recal_every == 0)
        self.net.train_epoch(bits, np.asarray(y).reshape(-1, 1),
                             recalibrate=recal)
        self._chunks += 1

    def predict(self, X: np.ndarray, T: int = 64) -> np.ndarray:
        """Firing rate over T stochastic passes -- a counter, then read-out.

        Statistically identical to StochasticNet.predict_votes, but the first
        layer's integer accumulation is hoisted out of the pass loop: the
        weights and the input bits are constant across passes, so only the
        random thresholds have to be redrawn. That is what makes a deep
        temporal read-out (T = 256) affordable in an interactive demo.
        """
        net = self.net
        bits = self.encode(X)
        a0 = np.concatenate(
            [bits, np.ones((len(bits), 1), dtype=np.int64)], axis=1)
        W1, W2 = net.weights[0], net.weights[1]
        x1 = a0.astype(np.float64) @ W1.T.astype(np.float64)   # invariant
        W2f = W2.T.astype(np.float64)
        N1 = max(1, int(net.N_fwd[0]))
        N2 = max(1, int(net.N_fwd[1]))
        ones = np.ones((len(bits), 1))
        votes = np.zeros(len(bits))
        for _ in range(T):
            r1 = self._prng.integers(-N1, N1, size=x1.shape)
            h = np.concatenate([(x1 + r1 > 0), ones.astype(bool)], axis=1)
            x2 = h.astype(np.float64) @ W2f
            r2 = self._prng.integers(-N2, N2, size=x2.shape)
            votes += (x2 + r2 > 0)[:, 0]
        return votes / float(T)

    def predict_grid(self, size: int = 64, T: int = 256) -> np.ndarray:
        from shape_bmp import grid_inputs
        return self.predict(grid_inputs(size), T=T).reshape(size, size)

    def accuracy(self, X: np.ndarray, y: np.ndarray, T: int = 64) -> float:
        return float(np.mean((self.predict(X, T) > 0.5) == (np.asarray(y) > 0.5)))


# --------------------------------------------------------------------------- #
#  Float back-propagation baseline
# --------------------------------------------------------------------------- #
class PixelMLP:
    """2-30-1 tanh/sigmoid MLP trained by streaming SGD (uses multipliers)."""

    name = "Float MLP baseline (mult + exp)"

    def __init__(self, hidden: int = 30, seed: int = 0, lr: float = 0.5):
        rng = np.random.default_rng(seed)
        self.lr = lr
        self.W1 = rng.normal(0, 1.0, size=(hidden, 2))
        self.b1 = rng.normal(0, 0.5, size=hidden)
        self.W2 = rng.normal(0, 1.0 / np.sqrt(hidden), size=(1, hidden))
        self.b2 = np.zeros(1)

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def _forward(self, X):
        h = np.tanh(X @ self.W1.T + self.b1)
        out = self._sigmoid(h @ self.W2.T + self.b2)
        return out, h

    def train_chunk(self, X: np.ndarray, y: np.ndarray, passes: int = 4,
                    batch: int = 32):
        X = np.asarray(X, dtype=float)
        yv = np.asarray(y, dtype=float).reshape(-1, 1)
        n = len(X)
        rng = np.random.default_rng(0)
        for _ in range(passes):
            idx = rng.permutation(n)
            for s in range(0, n, batch):
                xb, yb = X[idx[s:s + batch]], yv[idx[s:s + batch]]
                out, h = self._forward(xb)
                dz2 = (out - yb) / len(xb)           # cross-entropy on sigmoid
                self.W2 -= self.lr * (dz2.T @ h)
                self.b2 -= self.lr * dz2.sum(axis=0)
                dz1 = (dz2 @ self.W2) * (1 - h ** 2)
                self.W1 -= self.lr * (dz1.T @ xb)
                self.b1 -= self.lr * dz1.sum(axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._forward(np.asarray(X, dtype=float))[0].ravel()

    def predict_grid(self, size: int = 64) -> np.ndarray:
        from shape_bmp import grid_inputs
        return self.predict(grid_inputs(size)).reshape(size, size)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean((self.predict(X) > 0.5) == (np.asarray(y) > 0.5)))


def make_learner(engine: str, seed: int = 0):
    return PixelMazumdar(seed=seed) if engine == "mazumdar" else PixelMLP(seed=seed)


if __name__ == "__main__":
    import time
    from shape_bmp import get_shape_image, PixelStream, SHAPES

    for engine in ("mazumdar", "mlp"):
        print(f"\n=== {engine} ===")
        for shape in SHAPES:
            img = get_shape_image(shape)
            stream = PixelStream(img, seed=1)
            test_X, test_y, _, _ = PixelStream(img, seed=99).next_batch(2000)
            net = make_learner(engine, seed=0)
            t0 = time.time()
            for c in range(40):
                X, y, _, _ = stream.next_batch(256)
                net.train_chunk(X, y)
            acc = net.accuracy(test_X, test_y)
            g = net.predict_grid(64)
            pix_acc = float(np.mean((g > 0.5) == (img > 127)))
            print(f"  {shape:9s} samples={stream.n_drawn:6d}  "
                  f"stream_acc={acc:.3f}  full_bitmap_acc={pix_acc:.3f}  "
                  f"({time.time()-t0:.1f}s)")
