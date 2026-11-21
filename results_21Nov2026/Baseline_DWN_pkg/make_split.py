"""
make_split.py - reproduce SENA's frozen binary-pixel split for the DWN baseline.

SENA protocol (V12d/V13a, sim/sena_ensemble.py::load_mnist_binary, read-only
copy of <JULY2026>/SENA_Project\\sena2026\\sim\\sena_ensemble.py):
    X = uint8 image / 255.0  (float32)
    rng = np.random.default_rng(seed)                 # seed = 2026
    tr_idx = rng.choice(60000, size=10000, replace=False)
    te_idx = rng.choice(10000, size=2500,  replace=False)
    Xtr = (X_train[tr_idx].reshape(-1, 784) >= 0.3).astype(int64)
    Xte = (X_test [te_idx].reshape(-1, 784) >= 0.3).astype(int64)
    ytr, yte = labels[tr_idx], labels[te_idx]
    Ytr, Yte = one-hot

Pixel order: row-major 28x28, index i = 28*row + col  (bit i of the DWN input).

Usage
    python make_split.py                         # MNIST, writes data/*.npz
    python make_split.py --sena-npz <SENA .npz>  # also verify against SENA's file
    python make_split.py --dataset fashion_mnist
Outputs
    data/<dataset>_bin_t0.3_tr10000_te2500_s2026.npz   (Xtr Xte ytr yte Ytr Yte)
    data/<dataset>_bin_t0.3_tr10000_te2500_s2026.sha256.json
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import struct
import urllib.request

import numpy as np

MIRRORS = {
    "mnist": ["https://ossci-datasets.s3.amazonaws.com/mnist/",
              "https://storage.googleapis.com/cvdf-datasets/mnist/"],
    "fashion_mnist": ["http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"],
}
FILES = {"xtr": "train-images-idx3-ubyte", "ytr": "train-labels-idx1-ubyte",
         "xte": "t10k-images-idx3-ubyte", "yte": "t10k-labels-idx1-ubyte"}


def _read_idx(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rb") as f:
        magic = struct.unpack(">I", f.read(4))[0]
        nd = magic & 0xFF
        shape = struct.unpack(">" + "I" * nd, f.read(4 * nd))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)


def _get(raw_dir, dataset, name):
    for cand in (name, name + ".gz"):
        p = os.path.join(raw_dir, cand)
        if os.path.exists(p):
            return p
    os.makedirs(raw_dir, exist_ok=True)
    for base in MIRRORS[dataset]:
        try:
            p = os.path.join(raw_dir, name + ".gz")
            urllib.request.urlretrieve(base + name + ".gz", p)
            return p
        except Exception as e:  # try next mirror
            print("  mirror failed:", base, e)
    raise RuntimeError(f"cannot obtain {name}; put the idx files in {raw_dir}")


def arrays_sha256(d):
    out = {}
    for k in sorted(d):
        a = np.ascontiguousarray(d[k])
        h = hashlib.sha256()
        h.update(str(a.dtype).encode() + str(a.shape).encode())
        h.update(a.tobytes())
        out[k] = h.hexdigest()
    return out


def file_sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make(dataset="mnist", n_train=10000, n_test=2500, threshold=0.3, seed=2026,
         raw_dir=None):
    raw_dir = raw_dir or os.path.join("data", "raw", dataset)
    xtr = _read_idx(_get(raw_dir, dataset, FILES["xtr"])).astype(np.float32) / 255.0
    ytr = _read_idx(_get(raw_dir, dataset, FILES["ytr"])).astype(np.int64)
    xte = _read_idx(_get(raw_dir, dataset, FILES["xte"])).astype(np.float32) / 255.0
    yte = _read_idx(_get(raw_dir, dataset, FILES["yte"])).astype(np.int64)
    rng = np.random.default_rng(seed)
    tr_idx = rng.choice(len(xtr), size=min(n_train, len(xtr)), replace=False)
    te_idx = rng.choice(len(xte), size=min(n_test, len(xte)), replace=False)
    Xtr = (xtr[tr_idx].reshape(len(tr_idx), -1) >= threshold).astype(np.int64)
    Xte = (xte[te_idx].reshape(len(te_idx), -1) >= threshold).astype(np.int64)
    ytr, yte = ytr[tr_idx], yte[te_idx]
    eye = np.eye(10, dtype=np.int64)
    return dict(Xtr=Xtr, Xte=Xte, ytr=ytr, yte=yte, Ytr=eye[ytr], Yte=eye[yte])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mnist", choices=["mnist", "fashion_mnist"])
    ap.add_argument("--n-train", type=int, default=10000)
    ap.add_argument("--n-test", type=int, default=2500)
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--raw-dir", default=None,
                    help="folder with MNIST idx files (e.g. SENA's data/MNIST/raw)")
    ap.add_argument("--sena-npz", default=None,
                    help="SENA's own cached split .npz to compare against")
    ap.add_argument("--out-dir", default="data")
    a = ap.parse_args()

    d = make(a.dataset, a.n_train, a.n_test, a.threshold, a.seed, a.raw_dir)
    tag = f"{a.dataset}_bin_t{a.threshold}_tr{a.n_train}_te{a.n_test}_s{a.seed}"
    os.makedirs(a.out_dir, exist_ok=True)
    out = os.path.join(a.out_dir, tag + ".npz")
    if os.path.exists(out):
        old = dict(np.load(out))
        same = all(np.array_equal(old[k], d[k]) for k in d)
        print(f"{out} exists; identical to regenerated split: {same}")
        if not same:
            raise SystemExit("existing file differs - not overwriting; investigate")
    else:
        np.savez_compressed(out, **d)
        print("wrote", out)
    rec = {"dataset": a.dataset, "n_train": a.n_train, "n_test": a.n_test,
           "threshold": a.threshold, "seed": a.seed,
           "arrays_sha256": arrays_sha256(d), "npz_file_sha256": file_sha256(out),
           "test_label_histogram": np.bincount(d["yte"], minlength=10).tolist(),
           "mean_on_pixels_train": float(d["Xtr"].mean())}
    if a.sena_npz:
        s = dict(np.load(a.sena_npz))
        rec["sena_npz"] = os.path.abspath(a.sena_npz)
        rec["sena_npz_file_sha256"] = file_sha256(a.sena_npz)
        rec["sena_arrays_sha256"] = arrays_sha256({k: s[k] for k in d})
        rec["identical_to_sena"] = all(
            k in s and s[k].shape == d[k].shape and np.array_equal(s[k], d[k]) for k in d)
        print("identical to SENA split:", rec["identical_to_sena"])
        if not rec["identical_to_sena"]:
            raise SystemExit("MISMATCH with SENA split - do not train on this data")
    with open(os.path.join(a.out_dir, tag + ".sha256.json"), "w") as f:
        json.dump(rec, f, indent=2)
    print(json.dumps(rec["arrays_sha256"], indent=1))


if __name__ == "__main__":
    main()
