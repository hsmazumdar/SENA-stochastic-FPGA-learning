"""
Matched MNIST baselines for the SENA Q1 comparison.

Every model uses the same sampled examples and 784-160-10 topology:
  fp32_mlp       conventional ReLU MLP
  int8_ptq       dynamic INT8 quantisation of the trained fp32_mlp
  binaryconnect  binary weights, real hidden activation (STE training)
  xnor_bnn       binary weights and binary hidden activation (STE training)

The script reports mean +/- sample standard deviation over independent seeds.
It does not copy accuracy numbers from papers; these are local matched-budget
implementations whose method labels identify the established baseline family.

Examples:
  python exp_q1_baselines.py --profile smoke
  python exp_q1_baselines.py --profile full
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from sena_ensemble import load_mnist_binary

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / "results"
FIGS = ROOT / "assets" / "figs"
MEDIA = ROOT.parent / "_media"


PROFILES = {
    "smoke": dict(n_train=1200, n_test=400, epochs=2, seeds=[0]),
    "full": dict(n_train=10000, n_test=2500, epochs=30,
                 seeds=[0, 1, 2, 3, 4]),
}


class SignSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))

    @staticmethod
    def backward(ctx, grad):
        return grad.clamp_(-1.0, 1.0)


def sign_ste(x):
    return SignSTE.apply(x)


class FloatMLP(nn.Module):
    def __init__(self, n_in=784, hidden=160, n_out=10):
        super().__init__()
        self.fc1 = nn.Linear(n_in, hidden)
        self.fc2 = nn.Linear(hidden, n_out)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class BinaryLinear(nn.Linear):
    def forward(self, x):
        return F.linear(x, sign_ste(self.weight), self.bias)


class BinaryConnectMLP(nn.Module):
    def __init__(self, n_in=784, hidden=160, n_out=10):
        super().__init__()
        self.fc1 = BinaryLinear(n_in, hidden)
        self.fc2 = BinaryLinear(hidden, n_out)

    def forward(self, x):
        return self.fc2(F.hardtanh(self.fc1(x)))


class XNORMLP(nn.Module):
    def __init__(self, n_in=784, hidden=160, n_out=10):
        super().__init__()
        self.fc1 = BinaryLinear(n_in, hidden)
        self.fc2 = BinaryLinear(hidden, n_out)

    def forward(self, x):
        # BNN input and hidden streams are bipolar {-1,+1}.
        x = 2.0 * x - 1.0
        return self.fc2(sign_ste(self.fc1(x)))


def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def loaders(data, batch=128):
    xtr = torch.from_numpy(data["Xtr"].astype(np.float32))
    ytr = torch.from_numpy(data["ytr"].astype(np.int64))
    xte = torch.from_numpy(data["Xte"].astype(np.float32))
    yte = torch.from_numpy(data["yte"].astype(np.int64))
    train = DataLoader(TensorDataset(xtr, ytr), batch_size=batch,
                       shuffle=True, num_workers=0)
    test = DataLoader(TensorDataset(xte, yte), batch_size=512,
                      shuffle=False, num_workers=0)
    return train, test


@torch.inference_mode()
def accuracy(model, loader):
    model.eval()
    good = total = 0
    for x, y in loader:
        good += int((model(x).argmax(1) == y).sum())
        total += len(y)
    return good / total


def train(model, train_loader, epochs, lr):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    t0 = time.perf_counter()
    model.train()
    for _ in range(epochs):
        for x, y in train_loader:
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(model(x), y).backward()
            opt.step()
            # Latent real weights remain bounded for stable STE optimisation.
            with torch.no_grad():
                for name, p in model.named_parameters():
                    if "weight" in name:
                        p.clamp_(-1.0, 1.0)
    return time.perf_counter() - t0


def storage_bits(name, n_in=784, hidden=160, n_out=10):
    weights = n_in * hidden + hidden * n_out
    biases = hidden + n_out
    if name in {"binaryconnect", "xnor_bnn"}:
        return weights + 32 * biases
    if name == "int8_ptq":
        return 8 * weights + 32 * biases
    return 32 * (weights + biases)


def op_proxy(name, n_in=784, hidden=160, n_out=10):
    macs = n_in * hidden + hidden * n_out
    if name == "fp32_mlp":
        return dict(fp32_mac=macs, int8_mac=0, xnor_popcount=0)
    if name == "int8_ptq":
        return dict(fp32_mac=0, int8_mac=macs, xnor_popcount=0)
    return dict(fp32_mac=0, int8_mac=0, xnor_popcount=macs)


def quantize_dynamic(model):
    # Public API differs between torch releases.
    try:
        return torch.ao.quantization.quantize_dynamic(
            model, {nn.Linear}, dtype=torch.qint8)
    except (AttributeError, RuntimeError):
        return torch.quantization.quantize_dynamic(
            model, {nn.Linear}, dtype=torch.qint8)


def run_seed(data, seed, epochs):
    seed_all(seed)
    train_loader, test_loader = loaders(data)
    rows = []

    fp = FloatMLP()
    wall = train(fp, train_loader, epochs, lr=1e-3)
    rows.append(dict(method="fp32_mlp", seed=seed,
                     accuracy=accuracy(fp, test_loader), train_s=wall))

    q = quantize_dynamic(fp.eval())
    rows.append(dict(method="int8_ptq", seed=seed,
                     accuracy=accuracy(q, test_loader), train_s=wall))

    seed_all(seed)
    bc = BinaryConnectMLP()
    wall = train(bc, train_loader, epochs, lr=8e-4)
    rows.append(dict(method="binaryconnect", seed=seed,
                     accuracy=accuracy(bc, test_loader), train_s=wall))

    seed_all(seed)
    bn = XNORMLP()
    wall = train(bn, train_loader, epochs, lr=8e-4)
    rows.append(dict(method="xnor_bnn", seed=seed,
                     accuracy=accuracy(bn, test_loader), train_s=wall))

    for row in rows:
        row["storage_bits"] = storage_bits(row["method"])
        row.update(op_proxy(row["method"]))
    return rows


def aggregate(rows):
    out = []
    for method in sorted({r["method"] for r in rows}):
        subset = [r for r in rows if r["method"] == method]
        acc = np.asarray([r["accuracy"] for r in subset])
        wall = np.asarray([r["train_s"] for r in subset])
        out.append(dict(
            method=method, n_seeds=len(subset),
            accuracy_mean=float(acc.mean()),
            accuracy_std=float(acc.std(ddof=1)) if len(acc) > 1 else 0.0,
            train_s_mean=float(wall.mean()),
            storage_bits=int(subset[0]["storage_bits"]),
            fp32_mac=int(subset[0]["fp32_mac"]),
            int8_mac=int(subset[0]["int8_mac"]),
            xnor_popcount=int(subset[0]["xnor_popcount"]),
        ))
    return out


def render(summary, profile):
    labels = [r["method"] for r in summary]
    means = np.array([r["accuracy_mean"] for r in summary]) * 100
    stds = np.array([r["accuracy_std"] for r in summary]) * 100
    bits = np.array([r["storage_bits"] for r in summary]) / 8e6

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), dpi=180)
    axes[0].bar(labels, means, yerr=stds, capsize=4, color="#4472c4")
    axes[0].set_ylabel("MNIST test accuracy (%)")
    axes[0].set_title("(a) Matched 784-160-10 baseline accuracy")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].tick_params(axis="x", rotation=20)
    low = max(0, float(means.min() - 8))
    axes[0].set_ylim(low, min(100, float(means.max() + 4)))

    axes[1].bar(labels, bits, color="#70ad47")
    axes[1].set_ylabel("Weight and bias storage (MB)")
    axes[1].set_title("(b) Parameter-memory requirement")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].tick_params(axis="x", rotation=20)
    fig.suptitle(
        f"Q1 matched baselines on identical MNIST split ({profile} profile)\n"
        "Error bars: sample standard deviation across independent seeds",
        fontsize=11, fontweight="bold")
    fig.tight_layout()
    for directory in (FIGS, MEDIA):
        directory.mkdir(parents=True, exist_ok=True)
        fig.savefig(directory / "q1_mnist_matched_baselines.png",
                    bbox_inches="tight")
        fig.savefig(directory / "q1_mnist_matched_baselines.svg",
                    bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=PROFILES, default="smoke")
    args = parser.parse_args()
    cfg = PROFILES[args.profile]
    RESULTS.mkdir(parents=True, exist_ok=True)

    data = load_mnist_binary(
        n_train=cfg["n_train"], n_test=cfg["n_test"],
        threshold=0.3, seed=2026)
    rows = []
    t0 = time.perf_counter()
    for seed in cfg["seeds"]:
        print(f"baseline seed {seed} ...", flush=True)
        rows.extend(run_seed(data, seed, cfg["epochs"]))
    elapsed = time.perf_counter() - t0
    summary = aggregate(rows)
    render(summary, args.profile)

    payload = dict(
        profile=args.profile,
        config={**cfg, "dataset_seed": 2026, "threshold": 0.3,
                "architecture": [784, 160, 10]},
        elapsed_s=elapsed, runs=rows, summary=summary,
    )
    (RESULTS / "q1_mnist_baselines.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    with open(RESULTS / "q1_mnist_baselines.csv", "w", newline="",
              encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))
    print(f"elapsed {elapsed:.1f}s")


if __name__ == "__main__":
    main()
