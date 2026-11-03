"""
datasets.py
===========
Data providers for the stochastic-network experiments.

Because the stochastic perceptron consumes BINARY inputs (a weight is added
only when the input bit is 1), continuous features are converted to binary
using a *thermometer* (unary) code over per-feature quantiles.  This is a
hardware-friendly encoding: each analog feature maps to a small bank of
comparators against fixed reference levels.

Provides:
  * mirror-symmetry task  (reproduces the original 1995 benchmark)
  * XOR / parity          (non-linearly separable stress test)
  * UCI Iris, Breast-Cancer (Wisconsin)
  * 8x8 handwritten digits (sklearn `load_digits`, a compact MNIST proxy)
"""

from __future__ import annotations
import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split


# --------------------------------------------------------------------------- #
#  Encoding utilities
# --------------------------------------------------------------------------- #
def thermometer_encode(X, n_levels=8, refs=None):
    """Unary/thermometer encode each column into `n_levels` bits."""
    X = np.asarray(X, dtype=float)
    if refs is None:
        qs = np.linspace(0, 1, n_levels + 2)[1:-1]
        refs = [np.quantile(X[:, j], qs) for j in range(X.shape[1])]
    cols = []
    for j in range(X.shape[1]):
        cols.append((X[:, j:j + 1] >= refs[j][None, :]).astype(np.int64))
    return np.concatenate(cols, axis=1), refs


def one_hot(y, k):
    Y = np.zeros((len(y), k), dtype=np.int64)
    Y[np.arange(len(y)), y] = 1
    return Y


# --------------------------------------------------------------------------- #
#  Synthetic tasks
# --------------------------------------------------------------------------- #
def make_symmetry(n_bits=8, seed=0):
    """Mirror-symmetry detection: label 1 if the bit vector is a palindrome.

    This is the exact task family used in the 1995 paper (8-input version)."""
    rng = np.random.default_rng(seed)
    n = min(2 ** n_bits, 4096)
    X = rng.integers(0, 2, size=(n, n_bits))
    half = n_bits // 2
    sym = np.all(X[:, :half] == X[:, ::-1][:, :half], axis=1).astype(int)
    # balance classes (palindromes are rare)
    pos = np.where(sym == 1)[0]
    neg = np.where(sym == 0)[0]
    neg = rng.choice(neg, size=min(len(neg), 4 * len(pos)), replace=False)
    idx = np.concatenate([pos, neg])
    rng.shuffle(idx)
    return X[idx], sym[idx]


def make_parity(n_bits=4, seed=0):
    """N-bit parity (XOR generalisation) -- fully enumerated."""
    from itertools import product
    X = np.array(list(product([0, 1], repeat=n_bits)))
    y = X.sum(axis=1) % 2
    return X, y.astype(int)


def make_xor2d(n=1200, sigma=0.5, seed=0):
    """Noisy 2-D Gaussian-XOR -- the canonical non-linearly-separable task."""
    rng = np.random.default_rng(seed)
    centers = np.array([[-1, -1], [1, 1], [-1, 1], [1, -1]], dtype=float)
    labels = np.array([0, 0, 1, 1])
    idx = rng.integers(0, 4, size=n)
    X = centers[idx] + rng.normal(0, sigma, size=(n, 2))
    y = labels[idx]
    return X, y.astype(int)


# --------------------------------------------------------------------------- #
#  Dataset registry
# --------------------------------------------------------------------------- #
def get_dataset(name, seed=0, n_levels=8, test_size=0.25):
    """Returns dict with binary Xtr/Xte, one-hot Ytr, int labels, meta."""
    name = name.lower()
    if name == "symmetry":
        X, y = make_symmetry(8, seed)
        k = 2
        Xb = X  # already binary
    elif name == "parity":
        X, y = make_parity(4, seed)
        k = 2
        Xb = X
    elif name in ("xor", "xor2d"):
        X, y = make_xor2d(1200, 0.5, seed)
        k = 2
        Xb, _ = thermometer_encode(X, max(n_levels, 12))
    elif name == "iris":
        d = load_iris(); X, y = d.data, d.target; k = 3
        Xb, _ = thermometer_encode(X, n_levels)
    elif name in ("cancer", "breast_cancer"):
        d = load_breast_cancer(); X, y = d.data, d.target; k = 2
        Xb, _ = thermometer_encode(X, n_levels)
    elif name == "digits":
        d = load_digits(); X, y = d.data, d.target; k = 10
        Xb, _ = thermometer_encode(X, n_levels)
    else:
        raise ValueError(f"unknown dataset {name}")

    Xtr, Xte, ytr, yte = train_test_split(
        Xb, y, test_size=test_size, random_state=seed, stratify=y)
    return {
        "name": name,
        "Xtr": Xtr.astype(np.int64), "Xte": Xte.astype(np.int64),
        "ytr": ytr.astype(int), "yte": yte.astype(int),
        "Ytr": one_hot(ytr, k) if k > 2 else ytr.reshape(-1, 1).astype(np.int64),
        "Yte": one_hot(yte, k) if k > 2 else yte.reshape(-1, 1).astype(np.int64),
        "n_in": Xb.shape[1], "n_out": (k if k > 2 else 1), "k": k,
    }
