# COPY of OnChip_Learning/twin/sena_hw_twin.py with schedule_probe() appended for the V13b ablation; rule unchanged.
"""
sena_hw_twin.py -- bit-exact integer twin of rtl/sena_learn_core.v (SENA hardware learning rule v1).

Everything is integer; no floats, no multiplies in the modelled datapath (the NumPy matmuls below are
just a fast way to compute sums of selected weights, which the RTL computes with adders).

HARDWARE RULE v1 (one member, N_IN-N_HID-N_OUT, weights W-bit, saturating to +-(2^(W-1)-1)):
  Randomness : one 32-bit Galois LFSR per hidden lane (LH[j]) and per output lane (LO[k]),
               taps 0x80200003, right-shift.  A *draw* = leap of 32 LFSR steps, once per lane per
               forward pass (at the FIRE step).  Bit-fields of the leaped state are used:
               [6:0] forward threshold, [13:8] r1/r2 sensitivity gate, [31:16] r3 back-signal gate.
  Seeds      : LH[j] = seed_in ^ C_H(j), LO[k] = seed_in ^ C_O(k)  (zero -> 1), nb_fx = 24<<4.
  Forward    : X_j = sum_{i: x_i=1} W1[j,i] + W1[j,N_IN];  y_j = (X_j + (LH[6:0]-64)) > 0
               Z_k = sum_{j: y_j=1} W2[k,j] + W2[k,N_HID];  o_k = (Z_k + (LO[6:0]-64)) > 0
  Train      : d_k = t_k - o_k ; p_k = |Z_k| <= LO[13:8] ; gs_k = d_k if (p_k and d_k!=0) else 0
               back_j = sum_k gs_k * W2[k,j]          (PRE-update W2)
               W2[k,j] = sat(W2[k,j] + d_k*(+1 if y_j else -1)),  bias col: + d_k
               mabs = (sum_j |back_j|) >> SH ; nb_fx += ((mabs<<4) - nb_fx) >>> 7 (arith. shift)
               mask = (smear(nb_fx>>4) >> 1) & 0xFFFF   (= largest power of two <= nb, minus 1; nb=0 -> 0)
               a_j = |X_j| <= LH[13:8] ; b_j = |back_j| > (LH[31:16] & mask) ; dj = a_j & b_j
               W1[j,i] = sat(W1[j,i] + dj * sgn(back_j) * (+1 if x_i else -1)), bias: +dj*sgn(back_j)
               (sgn(back)=+1 if back>0 else -1)
  Infer      : X computed once; for t in 1..T: re-fire hidden (new draw), output sweep, fire; votes+=o.
               pred = first index of max(votes).
"""
from __future__ import annotations
import numpy as np

TAPS = 0x80200003
M32 = 0xFFFFFFFF
C_H_MUL = 0x9E3779B9
C_O_MUL = 0x7F4A7C15
C_O_XOR = 0xA5A5A5A5
NB_INIT = 24 << 4


def lfsr_step(s: int) -> int:
    return ((s >> 1) ^ (TAPS if (s & 1) else 0)) & M32


def lfsr_leap_scalar(s: int, n: int = 32) -> int:
    for _ in range(n):
        s = lfsr_step(s)
    return s


# leap is linear over GF(2): leap(s) = XOR_b T[b][byte_b(s)]
_T = np.zeros((4, 256), dtype=np.uint64)
for _b in range(4):
    for _v in range(256):
        _T[_b, _v] = lfsr_leap_scalar(_v << (8 * _b))


def leap(s: np.ndarray) -> np.ndarray:
    s = s.astype(np.uint64)
    return (_T[0][s & 255] ^ _T[1][(s >> 8) & 255] ^ _T[2][(s >> 16) & 255] ^ _T[3][(s >> 24) & 255])


def lane_seeds(seed_in: int, n: int, mul: int, xor: int = 0) -> np.ndarray:
    out = []
    for j in range(n):
        v = (seed_in ^ ((mul * (j + 1)) & M32) ^ xor) & M32
        out.append(v if v != 0 else 1)
    return np.array(out, dtype=np.uint64)


def smear_mask(v: int) -> int:
    m = v
    for sh in (1, 2, 4, 8, 16, 32):
        m |= m >> sh
    return (m >> 1) & 0xFFFF


class HWTwin:
    def __init__(self, W1, W2, seed_in: int, sh: int, wbits: int = 8):
        self.W1 = np.array(W1, dtype=np.int64)          # (N_HID, N_IN+1)
        self.W2 = np.array(W2, dtype=np.int64)          # (N_OUT, N_HID+1)
        self.n_hid, n1 = self.W1.shape
        self.n_in = n1 - 1
        self.n_out = self.W2.shape[0]
        self.sh = sh
        self.wmax = (1 << (wbits - 1)) - 1
        self.LH = lane_seeds(seed_in, self.n_hid, C_H_MUL)
        self.LO = lane_seeds(seed_in, self.n_out, C_O_MUL, C_O_XOR)
        self.nb_fx = NB_INIT
        self.n_w1_updates = 0

    # ---- helpers
    def _X(self, x):
        return self.W1[:, :self.n_in] @ x + self.W1[:, self.n_in]

    def _fire_hidden(self, X):
        self.LH = leap(self.LH)
        r = (self.LH & np.uint64(127)).astype(np.int64) - 64
        return (X + r > 0).astype(np.int64)

    def _fire_out(self, y):
        Z = self.W2[:, :self.n_hid] @ y + self.W2[:, self.n_hid]
        self.LO = leap(self.LO)
        r = (self.LO & np.uint64(127)).astype(np.int64) - 64
        return Z, (Z + r > 0).astype(np.int64)

    # ---- one training sample
    def train(self, x, label: int):
        x = np.asarray(x, dtype=np.int64)
        X = self._X(x)
        y = self._fire_hidden(X)
        Z, o = self._fire_out(y)
        t = (np.arange(self.n_out) == label).astype(np.int64)
        d = t - o
        r1 = ((self.LO >> np.uint64(8)) & np.uint64(63)).astype(np.int64)
        p = np.abs(Z) <= r1
        gs = np.where(p & (d != 0), d, 0)
        back = gs @ self.W2[:, :self.n_hid]                       # pre-update
        sy = np.where(y > 0, 1, -1)
        self.W2[:, :self.n_hid] = np.clip(self.W2[:, :self.n_hid] + np.outer(d, sy), -self.wmax, self.wmax)
        self.W2[:, self.n_hid] = np.clip(self.W2[:, self.n_hid] + d, -self.wmax, self.wmax)
        mabs = int(np.abs(back).sum()) >> self.sh
        self.nb_fx = self.nb_fx + (((mabs << 4) - self.nb_fx) >> 7)
        mask = smear_mask(self.nb_fx >> 4)
        r2 = ((self.LH >> np.uint64(8)) & np.uint64(63)).astype(np.int64)
        r3 = ((self.LH >> np.uint64(16)) & np.uint64(mask)).astype(np.int64)
        a = np.abs(X) <= r2
        b = np.abs(back) > r3
        dj = (a & b).astype(np.int64)
        if dj.any():
            self.n_w1_updates += 1
            dirj = np.where(back > 0, 1, -1)
            si = np.append(np.where(x > 0, 1, -1), 1)
            self.W1 = np.clip(self.W1 + np.outer(dj * dirj, si), -self.wmax, self.wmax)

    # ---- inference with T passes; returns pred and per-pass votes snapshot list
    def infer(self, x, T: int, record=()):
        x = np.asarray(x, dtype=np.int64)
        X = self._X(x)
        votes = np.zeros(self.n_out, dtype=np.int64)
        snaps = {}
        for t in range(1, T + 1):
            y = self._fire_hidden(X)
            _, o = self._fire_out(y)
            votes += o
            if t in record:
                snaps[t] = int(np.argmax(votes))
        return int(np.argmax(votes)), snaps


def init_weights(n_in, n_hid, n_out, seed):
    """Same initial weights as sim/stochastic_core.py StochasticNet (uniform {-2..2})."""
    r = np.random.default_rng(seed + 12345)
    W1 = r.integers(-2, 3, size=(n_hid, n_in + 1)).astype(np.int64)
    W2 = r.integers(-2, 3, size=(n_out, n_hid + 1)).astype(np.int64)
    return W1, W2


def schedule(n_train, seed, epochs=40, spe=3000, probe=800):
    """Same sample order as the earlier software estimates (probe held out, unused here)."""
    rng = np.random.default_rng(seed + 77)
    probe_idx = rng.choice(n_train, size=probe, replace=False)
    pool = np.setdiff1d(np.arange(n_train), probe_idx)
    return [rng.choice(pool, size=min(spe, len(pool)), replace=False) for _ in range(epochs)]


def seed_in_for(seed: int) -> int:
    return (0x1D872B41 ^ ((seed * 0x01000193) & M32)) & M32


def schedule_probe(n_train, seed, probe=800):
    """[added for the V13b ablation] the 800 probe indices that schedule() holds out (validation set)."""
    rng = np.random.default_rng(seed + 77)
    return rng.choice(n_train, size=probe, replace=False)
