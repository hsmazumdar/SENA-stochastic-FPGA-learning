"""
Train DWN "sm" (paper Table 13) on SENA's frozen binary MNIST split.

Requires CUDA (torch_dwn EFDFunction has no CPU path).

Config (Baseline_Plan_V13.md / Bacellar et al. ICML 2024 Table 13, DWN n=6 sm):
  layers: LUT 1000 (learnable map) -> LUT 500 -> GroupSum(10, tau=1/0.245)
  n=6, batch=128, 100 epochs
  lr: 1e-2 x30, 1e-3 x30, 1e-4 x30, 1e-5 x10
  input: SENA binary pixels (z=1), NO thermometer re-encode

Usage (GPU machine / Colab):
  pip install torch-dwn
  python train_sena_dwn_sm.py --seeds 0,1,2,3,4
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn.functional import cross_entropy

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE / "results"
CKPT = HERE / "checkpoints"


def load_split():
    Xtr = np.load(DATA / "X_train.npy")
    ytr = np.load(DATA / "y_train.npy")
    Xte = np.load(DATA / "X_test.npy")
    yte = np.load(DATA / "y_test.npy")
    return (
        torch.from_numpy(Xtr),
        torch.from_numpy(ytr),
        torch.from_numpy(Xte),
        torch.from_numpy(yte),
    )


def build_model(n_in: int):
    import torch_dwn as dwn
    return nn.Sequential(
        dwn.LUTLayer(n_in, 1000, n=6, mapping="learnable"),
        dwn.LUTLayer(1000, 500, n=6),
        dwn.GroupSum(10, tau=1 / 0.245),
    )


def lr_for_epoch(epoch: int) -> float:
    # 0-29: 1e-2, 30-59: 1e-3, 60-89: 1e-4, 90-99: 1e-5
    if epoch < 30:
        return 1e-2
    if epoch < 60:
        return 1e-3
    if epoch < 90:
        return 1e-4
    return 1e-5


@torch.no_grad()
def accuracy(model, x, y, batch=512):
    model.eval()
    preds = []
    for i in range(0, len(x), batch):
        logits = model(x[i:i + batch].cuda())
        preds.append(logits.argmax(dim=1).cpu())
    pred = torch.cat(preds)
    return float((pred == y).float().mean())


def train_one(seed: int, epochs: int, batch_size: int):
    torch.manual_seed(seed)
    np.random.seed(seed)

    x_train, y_train, x_test, y_test = load_split()
    model = build_model(x_train.size(1)).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)

    history = []
    t0 = time.perf_counter()
    n = x_train.shape[0]

    for epoch in range(epochs):
        lr = lr_for_epoch(epoch)
        for g in opt.param_groups:
            g["lr"] = lr
        model.train()
        perm = torch.randperm(n)
        correct = total = 0
        last_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            bx = x_train[idx].cuda()
            by = y_train[idx].cuda()
            opt.zero_grad(set_to_none=True)
            out = model(bx)
            loss = cross_entropy(out, by)
            loss.backward()
            opt.step()
            last_loss = float(loss.item())
            correct += int((out.argmax(1) == by).sum().item())
            total += by.size(0)
        train_acc = correct / total
        test_acc = accuracy(model, x_test, y_test)
        row = dict(epoch=epoch + 1, lr=lr, loss=last_loss,
                   train_acc=train_acc, test_acc=test_acc)
        history.append(row)
        print(
            f"seed={seed} ep={epoch+1}/{epochs} lr={lr:g} "
            f"loss={last_loss:.4f} train={train_acc:.4f} test={test_acc:.4f}",
            flush=True,
        )

    elapsed = time.perf_counter() - t0
    CKPT.mkdir(parents=True, exist_ok=True)
    ckpt_path = CKPT / f"dwn_sm_sena_seed{seed}.pt"
    torch.save({
        "seed": seed,
        "state_dict": model.state_dict(),
        "n_in": int(x_train.size(1)),
        "config": "Table13_sm_n6_1000_500",
        "split": "mnist_bin_t0.3_tr10000_te2500_s2026",
        "test_acc": history[-1]["test_acc"],
        "best_test_acc": max(h["test_acc"] for h in history),
        "history": history,
        "elapsed_s": elapsed,
    }, ckpt_path)
    print(f"saved {ckpt_path}", flush=True)
    return dict(
        seed=seed,
        test_acc=history[-1]["test_acc"],
        best_test_acc=max(h["test_acc"] for h in history),
        elapsed_s=elapsed,
        checkpoint=str(ckpt_path),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA required: torch_dwn.EFDFunction has no CPU implementation. "
            "Run on a GPU machine or Colab (see colab_train_sena_dwn.ipynb)."
        )

    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
    OUT.mkdir(parents=True, exist_ok=True)
    runs = []
    for seed in seeds:
        runs.append(train_one(seed, args.epochs, args.batch_size))

    acc = np.asarray([r["test_acc"] for r in runs], float)
    best = np.asarray([r["best_test_acc"] for r in runs], float)
    summary = dict(
        name="dwn_sm_sena_split",
        config=dict(
            layers="LUT1000(learnable)+LUT500+GroupSum10",
            n=6, tau=1 / 0.245, batch=args.batch_size, epochs=args.epochs,
            lr_schedule="1e-2x30,1e-3x30,1e-4x30,1e-5x10",
            input="SENA binary z=1 (no thermometer)",
            split="mnist_bin_t0.3_tr10000_te2500_s2026",
        ),
        runs=runs,
        test_acc_mean=float(acc.mean()),
        test_acc_std=float(acc.std(ddof=1)) if len(acc) > 1 else 0.0,
        best_test_acc_mean=float(best.mean()),
        best_test_acc_std=float(best.std(ddof=1)) if len(best) > 1 else 0.0,
    )
    out = OUT / "dwn_sm_sena_train.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in (
        "test_acc_mean", "test_acc_std",
        "best_test_acc_mean", "best_test_acc_std")}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
