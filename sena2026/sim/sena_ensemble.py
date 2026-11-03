"""
sena_ensemble.py
================
SENA early-exit ensemble inference on top of StochasticNet.

Two complementary sources of stochastic votes (can be used together):

  * temporal votes  — T stochastic forward passes of one net
  * member votes    — M independently trained nets (ensemble)

Confidence for a sample is  max_class_votes / total_votes.
Inference stops as soon as confidence >= theta, so easy inputs
consume fewer stochastic passes (energy-proportional latency).

Energy model (hardware-proxy, not silicon measurement):
  E_unit = cost of one network forward pass
         = n_adds * e_add + n_cmp * e_cmp + n_rng * e_rng
  E_sample = n_passes_used * E_unit
  Report relative energy vs a fixed-budget baseline (T_max * M passes).
"""

from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np

from stochastic_core import StochasticNet, StochasticConfig


# Relative gate energies (normalized to one 16-bit add = 1.0).
# Rough CMOS folklore: compare ≈ add, RNG LFSR step ≈ 0.5 add, multiply ≈ 20–30 adds.
E_ADD = 1.0
E_CMP = 1.0
E_RNG = 0.5
E_MUL = 25.0


@dataclass
class EnergyModel:
    n_in: int
    n_hid: int
    n_out: int

    @property
    def stoch_adds(self) -> int:
        # bias included: (in+1)*hid + (hid+1)*out, but half bits active on avg
        return int(0.5 * ((self.n_in + 1) * self.n_hid + (self.n_hid + 1) * self.n_out))

    @property
    def stoch_cmp(self) -> int:
        return self.n_hid + self.n_out

    @property
    def stoch_rng(self) -> int:
        return self.n_hid + self.n_out

    @property
    def bp_muls(self) -> int:
        return (self.n_in + 1) * self.n_hid + (self.n_hid + 1) * self.n_out

    @property
    def e_forward_stoch(self) -> float:
        return (self.stoch_adds * E_ADD
                + self.stoch_cmp * E_CMP
                + self.stoch_rng * E_RNG)

    @property
    def e_forward_bp(self) -> float:
        # one real MAC ≈ mul + add
        return self.bp_muls * (E_MUL + E_ADD)


class SENAEnsemble:
    """M stochastic nets + temporal voting with confidence early-exit."""

    def __init__(self, nets: list[StochasticNet]):
        if not nets:
            raise ValueError("need at least one network")
        self.nets = nets
        self.n_out = nets[0].sizes[-1]
        self.energy = EnergyModel(nets[0].sizes[0], nets[0].sizes[1],
                                  nets[0].sizes[-1])

    @classmethod
    def train_members(cls, Xtr, Ytr, layer_sizes, n_members=3, epochs=20,
                      seeds=None, weight_bits=16, verbose=True,
                      samples_per_epoch=None, val_every=5, T_probe=24,
                      adaptive_calibration=True, fixed_N=64):
        """Train M nets; keep best weight snapshot by probe accuracy.

        Fixed +/-1 integer updates dither; without a best-snapshot (a register
        bank in hardware) late epochs can wander away from the optimum.
        """
        seeds = seeds or list(range(n_members))
        nets = []
        n = len(Xtr)
        spe = samples_per_epoch or n
        for i, seed in enumerate(seeds):
            cfg = StochasticConfig(layer_sizes=tuple(layer_sizes),
                                   epochs=1, seed=seed,
                                   weight_bits=weight_bits, init_scale=2)
            net = StochasticNet(cfg)
            if not adaptive_calibration:
                net.N_fwd = [float(fixed_N)] * len(net.weights)
                net.N_back = float(cfg.N_rev_back)
                net._calibrated = True
            rng = np.random.default_rng(seed + 77)
            # hold out a small probe from training for snapshot selection
            probe_n = min(800, max(200, n // 10))
            probe_idx = rng.choice(n, size=probe_n, replace=False)
            train_pool = np.setdiff1d(np.arange(n), probe_idx)
            if adaptive_calibration:
                net.calibrate(Xtr[train_pool[: min(512, len(train_pool))]])
            best_acc = -1.0
            best_state = None
            y_probe = np.argmax(Ytr[probe_idx], axis=1) if Ytr.shape[1] > 1 \
                else Ytr[probe_idx].ravel()
            for ep in range(epochs):
                idx = rng.choice(train_pool,
                                 size=min(spe, len(train_pool)),
                                 replace=False)
                net.train_epoch(Xtr[idx], Ytr[idx],
                                recalibrate=(adaptive_calibration and ep > 0
                                             and ep % 5 == 0))
                if (ep + 1) % val_every == 0 or ep == epochs - 1:
                    acc = net.accuracy(Xtr[probe_idx], y_probe, T=T_probe)
                    if acc >= best_acc:
                        best_acc = acc
                        best_state = ([w.copy() for w in net.weights],
                                      list(net.N_fwd), net.N_back)
            if best_state is not None:
                net.weights = [w.copy() for w in best_state[0]]
                net.N_fwd = list(best_state[1])
                net.N_back = best_state[2]
            nets.append(net)
            if verbose:
                print(f"  member {i+1}/{n_members} trained "
                      f"(seed={seed}, ep={epochs}, best_probe={best_acc*100:.1f}%)")
        return cls(nets)

    def save(self, path, metadata=None):
        """Save integer weights and calibrated ranges in one portable NPZ."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {}
        members = []
        for i, net in enumerate(self.nets):
            for l, w in enumerate(net.weights):
                arrays[f"m{i}_w{l}"] = np.asarray(w, dtype=np.int64)
            members.append(dict(
                sizes=list(net.sizes),
                seed=int(net.cfg.seed),
                weight_bits=int(net.cfg.weight_bits),
                rng_kind=str(net.cfg.rng_kind),
                N_fwd=[float(v) for v in net.N_fwd],
                N_back=float(net.N_back),
                alpha=float(net.alpha),
            ))
        manifest = dict(format="sena-ensemble-v1", members=members,
                        metadata=metadata or {})
        arrays["manifest_json"] = np.asarray(json.dumps(manifest))
        np.savez_compressed(path, **arrays)
        return path

    @classmethod
    def load(cls, path, rng_kind=None):
        """Restore an ensemble written by :meth:`save`."""
        data = np.load(path, allow_pickle=False)
        manifest = json.loads(str(data["manifest_json"]))
        if manifest.get("format") != "sena-ensemble-v1":
            raise ValueError("unsupported SENA checkpoint format")
        nets = []
        for i, item in enumerate(manifest["members"]):
            cfg = StochasticConfig(
                layer_sizes=tuple(item["sizes"]),
                seed=int(item["seed"]),
                weight_bits=int(item["weight_bits"]),
                rng_kind=rng_kind or item.get("rng_kind", "numpy"),
            )
            net = StochasticNet(cfg)
            net.weights = [
                np.asarray(data[f"m{i}_w{l}"], dtype=np.int64).copy()
                for l in range(len(item["sizes"]) - 1)
            ]
            net.N_fwd = [float(v) for v in item["N_fwd"]]
            net.N_back = float(item["N_back"])
            net.alpha = float(item.get("alpha", 1.6))
            net._calibrated = True
            nets.append(net)
        return cls(nets), manifest.get("metadata", {})

    def predict_early_exit(self, X: np.ndarray, T_max: int = 64,
                           theta: float = 0.75, batch: int = 256):
        """Early-exit over temporal passes; votes pooled across members.

        Returns dict with predictions, confidence, passes_used, energy, ...
        """
        X = np.asarray(X, dtype=np.int64)
        n = len(X)
        n_out = self.n_out
        M = len(self.nets)
        votes = np.zeros((n, n_out), dtype=np.int64)
        done = np.zeros(n, dtype=bool)
        passes_used = np.zeros(n, dtype=np.int64)
        pred = np.zeros(n, dtype=np.int64)
        conf = np.zeros(n, dtype=np.float64)

        # Each "pass" = one stochastic forward of ALL members (M nets).
        # Total vote budget per sample = T_max * M.
        for t in range(1, T_max + 1):
            active = np.where(~done)[0]
            if len(active) == 0:
                break
            # process in batches for memory
            for s in range(0, len(active), batch):
                idx = active[s:s + batch]
                xb = X[idx]
                for net in self.nets:
                    acts, _ = net.forward(xb, stochastic=True)
                    votes[idx] += acts[-1]
            passes_used[active] = t
            total = votes.sum(axis=1).astype(np.float64)
            total = np.maximum(total, 1.0)
            conf_now = votes.max(axis=1) / total
            pred_now = np.argmax(votes, axis=1)
            newly = (~done) & (conf_now >= theta)
            # also force-finish on last pass
            if t == T_max:
                newly = ~done
            done[newly] = True
            pred[newly] = pred_now[newly]
            conf[newly] = conf_now[newly]

        e_unit = self.energy.e_forward_stoch * M   # one temporal pass of all members
        energy = passes_used.astype(np.float64) * e_unit
        e_full = float(T_max * e_unit)
        return dict(
            pred=pred,
            conf=conf,
            passes_used=passes_used,
            mean_passes=float(np.mean(passes_used)),
            energy=energy,
            mean_energy=float(np.mean(energy)),
            energy_full=e_full,
            energy_saving=1.0 - float(np.mean(energy)) / e_full,
            M=M, T_max=T_max, theta=theta,
            e_unit=e_unit,
        )

    def accuracy_early_exit(self, X, y, T_max=64, theta=0.75):
        out = self.predict_early_exit(X, T_max=T_max, theta=theta)
        out["acc"] = float(np.mean(out["pred"] == np.asarray(y)))
        return out

    def sweep_theta(self, X, y, thetas, T_max=64):
        rows = []
        for th in thetas:
            r = self.accuracy_early_exit(X, y, T_max=T_max, theta=float(th))
            rows.append(dict(
                theta=float(th),
                acc=r["acc"],
                mean_passes=r["mean_passes"],
                mean_energy=r["mean_energy"],
                energy_full=r["energy_full"],
                energy_saving=r["energy_saving"],
                relative_energy=r["mean_energy"] / r["energy_full"],
            ))
        return rows


def load_mnist_binary(n_train=10000, n_test=2000, threshold=0.4,
                      seed=0, cache_dir=None, dataset="mnist"):
    """Binary-pixel image dataset (statistical 1-bit ADC of intensity).

    dataset: "mnist" | "fashion_mnist"
    threshold in [0,1] on normalized pixels. Returns dict compatible with
    get_dataset() fields.
    """
    import os
    dataset = dataset.lower().replace("-", "_")
    if dataset not in {"mnist", "fashion_mnist"}:
        raise ValueError(f"unknown dataset {dataset}")
    cache_dir = cache_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "data")
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(
        cache_dir,
        f"{dataset}_bin_t{threshold}_tr{n_train}_te{n_test}_s{seed}.npz")
    if os.path.exists(cache):
        data = np.load(cache)
        return {k: data[k] for k in data.files} | {
            "n_in": int(data["Xtr"].shape[1]),
            "n_out": 10, "k": 10, "name": dataset,
        }

    Xtr = ytr = Xte = yte = None
    try:
        from torchvision import datasets, transforms
        ctor = (datasets.FashionMNIST if dataset == "fashion_mnist"
                else datasets.MNIST)
        ds_tr = ctor(cache_dir, train=True, download=True,
                     transform=transforms.ToTensor())
        ds_te = ctor(cache_dir, train=False, download=True,
                     transform=transforms.ToTensor())
        Xtr = ds_tr.data.numpy().astype(np.float32) / 255.0
        ytr = ds_tr.targets.numpy().astype(np.int64)
        Xte = ds_te.data.numpy().astype(np.float32) / 255.0
        yte = ds_te.targets.numpy().astype(np.int64)
    except Exception:
        if dataset == "fashion_mnist":
            # OpenML Fashion-MNIST id 40996
            from sklearn.datasets import fetch_openml
            mn = fetch_openml(data_id=40996, as_frame=False, parser="auto")
        else:
            from sklearn.datasets import fetch_openml
            mn = fetch_openml("mnist_784", version=1, as_frame=False,
                              parser="auto")
        X = mn.data.astype(np.float32) / 255.0
        y = mn.target.astype(np.int64)
        Xtr, ytr = X[:60000], y[:60000]
        Xte, yte = X[60000:], y[60000:]

    rng = np.random.default_rng(seed)
    tr_idx = rng.choice(len(Xtr), size=min(n_train, len(Xtr)), replace=False)
    te_idx = rng.choice(len(Xte), size=min(n_test, len(Xte)), replace=False)
    Xtr = (Xtr[tr_idx].reshape(len(tr_idx), -1) >= threshold).astype(np.int64)
    Xte = (Xte[te_idx].reshape(len(te_idx), -1) >= threshold).astype(np.int64)
    ytr = ytr[tr_idx]
    yte = yte[te_idx]

    def one_hot(y, k=10):
        Y = np.zeros((len(y), k), dtype=np.int64)
        Y[np.arange(len(y)), y] = 1
        return Y

    out = dict(
        name=dataset,
        Xtr=Xtr, Xte=Xte, ytr=ytr, yte=yte,
        Ytr=one_hot(ytr), Yte=one_hot(yte),
        n_in=Xtr.shape[1], n_out=10, k=10,
    )
    np.savez_compressed(cache, **{k: v for k, v in out.items()
                                  if isinstance(v, np.ndarray)})
    return out


def load_fashion_mnist_binary(**kwargs):
    """Convenience wrapper for Fashion-MNIST binary pixels."""
    kwargs = dict(kwargs)
    kwargs["dataset"] = "fashion_mnist"
    return load_mnist_binary(**kwargs)
