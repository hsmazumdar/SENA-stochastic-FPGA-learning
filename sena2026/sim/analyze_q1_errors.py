"""
B5 — Error analysis on a trained SENA ensemble checkpoint.

Produces confusion matrix, per-class accuracy, and early/late-exit
exemplars (low-pass correct vs high-pass incorrect).

Examples:
  python analyze_q1_errors.py --profile smoke
  python analyze_q1_errors.py --profile full
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sena_ensemble import SENAEnsemble, load_mnist_binary

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / "results"
FIGS = ROOT / "assets" / "figs"
MEDIA = ROOT.parent / "_media"

PROFILES = {
    "smoke": dict(
        n_train=800, n_test=250, T_max=6, theta=0.55,
        checkpoint="q1_ablation_full_r0_adaptive.npz",
        fallback_train=True, hidden=32, epochs=2, members=3,
        samples_per_epoch=400,
    ),
    "full": dict(
        n_train=10000, n_test=2500, T_max=40, theta=0.55,
        checkpoint="q1_ablation_full_r0_adaptive.npz",
        fallback_train=False, hidden=160, epochs=40, members=3,
        samples_per_epoch=3000,
    ),
}


def confusion_matrix(y_true, y_pred, n_class=10):
    cm = np.zeros((n_class, n_class), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def pick_exemplars(X, y, out, n_each=8):
    pred = out["pred"]
    passes = out["passes_used"]
    conf = out["conf"]
    correct = pred == y
    # early correct: fewest passes among correct
    early_idx = np.where(correct)[0]
    late_wrong = np.where(~correct)[0]
    early = []
    if len(early_idx):
        order = early_idx[np.argsort(passes[early_idx])]
        for i in order[:n_each]:
            early.append(dict(
                index=int(i), true=int(y[i]), pred=int(pred[i]),
                passes=int(passes[i]), conf=float(conf[i])))
    late = []
    if len(late_wrong):
        order = late_wrong[np.argsort(-passes[late_wrong])]
        for i in order[:n_each]:
            late.append(dict(
                index=int(i), true=int(y[i]), pred=int(pred[i]),
                passes=int(passes[i]), conf=float(conf[i])))
    return early, late


def render(cm, out, y, early, late, X, profile, theta):
    fig = plt.figure(figsize=(13, 8), dpi=180)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0])

    ax0 = fig.add_subplot(gs[0, 0])
    im = ax0.imshow(cm, cmap="Blues")
    ax0.set_xlabel("Predicted")
    ax0.set_ylabel("True")
    ax0.set_title("(a) Confusion matrix")
    ax0.set_xticks(range(10))
    ax0.set_yticks(range(10))
    fig.colorbar(im, ax=ax0, fraction=0.046)

    ax1 = fig.add_subplot(gs[0, 1])
    per_class = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)
    ax1.bar(range(10), per_class * 100, color="#4472c4")
    ax1.set_xlabel("Digit")
    ax1.set_ylabel("Recall (%)")
    ax1.set_title("(b) Per-class recall")
    ax1.set_ylim(0, 105)
    ax1.grid(axis="y", alpha=0.25)

    ax2 = fig.add_subplot(gs[1, 0])
    correct = out["pred"] == y
    ax2.hist(out["passes_used"][correct], bins=20, alpha=0.7,
             color="#70ad47", label="correct")
    ax2.hist(out["passes_used"][~correct], bins=20, alpha=0.7,
             color="#c00000", label="error")
    ax2.set_xlabel("Passes used")
    ax2.set_ylabel("Count")
    ax2.set_title("(c) Exit depth: correct vs error")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.25)

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis("off")
    ax3.set_title("(d) Early-correct / late-error exemplars")
    cols = min(8, max(len(early), len(late), 1))
    # show up to 4 early + 4 late as tiny digits
    show = []
    for tag, items in (("E", early[:4]), ("L", late[:4])):
        for ex in items:
            show.append((tag, ex))
    if show:
        n = len(show)
        for k, (tag, ex) in enumerate(show):
            inset = ax3.inset_axes([k / n, 0.15, 0.9 / n, 0.7])
            inset.imshow(X[ex["index"]].reshape(28, 28), cmap="gray_r")
            inset.set_xticks([])
            inset.set_yticks([])
            inset.set_title(
                f"{tag} t{ex['true']}→{ex['pred']}\n"
                f"p={ex['passes']} c={ex['conf']:.2f}",
                fontsize=7)
    ax3.text(0.0, 0.0, "E=early correct, L=late error", transform=ax3.transAxes,
             fontsize=8)

    acc = float(np.mean(correct))
    fig.suptitle(
        f"SENA error analysis ({profile})  acc={acc*100:.1f}%  "
        f"θ={theta}  mean passes={out['mean_passes']:.2f}",
        fontsize=11, fontweight="bold")
    fig.tight_layout()
    for directory in (FIGS, MEDIA):
        directory.mkdir(parents=True, exist_ok=True)
        fig.savefig(directory / "q1_error_analysis.png", bbox_inches="tight")
        fig.savefig(directory / "q1_error_analysis.svg", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=PROFILES, default="smoke")
    parser.add_argument("--checkpoint", default=None,
                        help="override ensemble NPZ path")
    args = parser.parse_args()
    cfg = PROFILES[args.profile]
    RESULTS.mkdir(parents=True, exist_ok=True)

    data = load_mnist_binary(
        n_train=cfg["n_train"], n_test=cfg["n_test"],
        threshold=0.3, seed=2026)
    ckpt_name = args.checkpoint or cfg["checkpoint"]
    ckpt = Path(ckpt_name)
    if not ckpt.is_absolute():
        ckpt = RESULTS / ckpt_name

    t0 = time.perf_counter()
    if ckpt.exists():
        print(f"loading {ckpt} ...", flush=True)
        ens, meta = SENAEnsemble.load(ckpt)
    elif cfg["fallback_train"]:
        print("checkpoint missing; training smoke ensemble ...", flush=True)
        ens = SENAEnsemble.train_members(
            data["Xtr"], data["Ytr"],
            layer_sizes=(data["n_in"], cfg["hidden"], 10),
            n_members=cfg["members"], epochs=cfg["epochs"],
            seeds=list(range(cfg["members"])),
            samples_per_epoch=cfg["samples_per_epoch"],
            val_every=1, T_probe=min(8, cfg["T_max"]), verbose=True)
        meta = dict(fallback_train=True)
    else:
        raise FileNotFoundError(
            f"Missing checkpoint {ckpt}; run A2 ablations first")

    out = ens.predict_early_exit(
        data["Xte"], T_max=cfg["T_max"], theta=cfg["theta"])
    y = np.asarray(data["yte"])
    acc = float(np.mean(out["pred"] == y))
    cm = confusion_matrix(y, out["pred"])
    early, late = pick_exemplars(data["Xte"], y, out)
    per_class = (cm.diagonal() / np.maximum(cm.sum(axis=1), 1)).tolist()
    elapsed = time.perf_counter() - t0

    render(cm, out, y, early, late, data["Xte"], args.profile, cfg["theta"])

    # pass histograms for JSON
    correct = out["pred"] == y
    payload = dict(
        profile=args.profile,
        config={**cfg, "checkpoint": str(ckpt), "dataset_seed": 2026},
        metadata=meta if isinstance(meta, dict) else {},
        elapsed_s=elapsed,
        accuracy=acc,
        mean_passes=out["mean_passes"],
        relative_energy=out["mean_energy"] / out["energy_full"],
        confusion_matrix=cm.tolist(),
        per_class_recall=per_class,
        early_correct_exemplars=early,
        late_error_exemplars=late,
        passes_correct_mean=float(out["passes_used"][correct].mean())
        if correct.any() else None,
        passes_error_mean=float(out["passes_used"][~correct].mean())
        if (~correct).any() else None,
    )
    out_json = RESULTS / "q1_error_analysis.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(dict(
        accuracy=acc, mean_passes=out["mean_passes"],
        per_class_recall=per_class,
        n_early=len(early), n_late=len(late),
        elapsed_s=elapsed), indent=2))
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
