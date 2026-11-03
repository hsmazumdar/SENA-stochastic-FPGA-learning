"""
B4 — Shape learning metrics (IoU, pixel accuracy, convergence) for
circle / square / triangle bitmaps using PixelMazumdar vs PixelMLP.

Examples:
  python exp_q1_shapes.py --profile smoke
  python exp_q1_shapes.py --profile full
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pixel_learners import make_learner
from shape_bmp import SHAPES, PixelStream, get_shape_image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / "results"
FIGS = ROOT / "assets" / "figs"
MEDIA = ROOT.parent / "_media"

PROFILES = {
    "smoke": dict(
        chunks=15, batch=128, eval_every=5, T_grid=64,
        seeds=[0], engines=["mazumdar", "mlp"],
    ),
    "full": dict(
        chunks=80, batch=256, eval_every=5, T_grid=192,
        seeds=[0, 1, 2, 3, 4], engines=["mazumdar", "mlp"],
    ),
}


def iou_and_pixel_acc(pred_prob, gt_img):
    pred = pred_prob > 0.5
    gt = np.asarray(gt_img) > 127
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    iou = float(inter / union) if union > 0 else 0.0
    pix = float(np.mean(pred == gt))
    return iou, pix


def run_one(engine, shape, seed, cfg):
    img = get_shape_image(shape)
    stream = PixelStream(img, seed=seed + 11)
    test_X, test_y, _, _ = PixelStream(img, seed=seed + 99).next_batch(2000)
    net = make_learner(engine, seed=seed)
    history = []
    t0 = time.perf_counter()
    for c in range(1, cfg["chunks"] + 1):
        X, y, _, _ = stream.next_batch(cfg["batch"])
        net.train_chunk(X, y)
        if c % cfg["eval_every"] == 0 or c == cfg["chunks"]:
            if engine == "mazumdar":
                grid = net.predict_grid(64, T=cfg["T_grid"])
                stream_acc = net.accuracy(test_X, test_y, T=min(64, cfg["T_grid"]))
            else:
                grid = net.predict_grid(64)
                stream_acc = net.accuracy(test_X, test_y)
            iou, pix = iou_and_pixel_acc(grid, img)
            history.append(dict(
                chunk=c, samples=int(stream.n_drawn),
                stream_acc=float(stream_acc), iou=iou, pixel_acc=pix,
            ))
    elapsed = time.perf_counter() - t0
    final = history[-1]
    return dict(
        engine=engine, shape=shape, seed=seed,
        final_iou=final["iou"],
        final_pixel_acc=final["pixel_acc"],
        final_stream_acc=final["stream_acc"],
        samples=final["samples"],
        train_s=elapsed,
        history=history,
    )


def aggregate(rows):
    summary = []
    for engine in sorted({r["engine"] for r in rows}):
        for shape in SHAPES:
            subset = [r for r in rows
                      if r["engine"] == engine and r["shape"] == shape]
            if not subset:
                continue
            item = dict(engine=engine, shape=shape, n_seeds=len(subset))
            for key in ("final_iou", "final_pixel_acc", "final_stream_acc",
                        "train_s"):
                values = np.asarray([r[key] for r in subset], dtype=float)
                item[f"{key}_mean"] = float(values.mean())
                item[f"{key}_std"] = (
                    float(values.std(ddof=1)) if len(values) > 1 else 0.0)
            summary.append(item)
    return summary


def render(rows, summary, profile):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), dpi=180)
    x = np.arange(len(SHAPES))
    w = 0.35
    for ax, metric, title, ylim in (
        (axes[0], "final_iou", "(a) Final IoU", (0, 1.05)),
        (axes[1], "final_pixel_acc", "(b) Pixel accuracy", (0.5, 1.05)),
    ):
        for i, engine in enumerate(("mazumdar", "mlp")):
            means, stds = [], []
            for shape in SHAPES:
                hit = next((s for s in summary
                            if s["engine"] == engine and s["shape"] == shape),
                           None)
                means.append(hit[f"{metric}_mean"] if hit else 0.0)
                stds.append(hit[f"{metric}_std"] if hit else 0.0)
            ax.bar(x + (i - 0.5) * w, means, w, yerr=stds, capsize=3,
                   color="#c00000" if engine == "mazumdar" else "#4472c4",
                   label="Mazumdar" if engine == "mazumdar" else "Float MLP")
        ax.set_xticks(x, SHAPES)
        ax.set_ylabel(metric.replace("final_", "").replace("_", " "))
        ax.set_title(title)
        ax.set_ylim(*ylim)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.25)

    # (c) convergence IoU vs samples for circle, seed 0 if present
    for engine, color in (("mazumdar", "#c00000"), ("mlp", "#4472c4")):
        hit = next((r for r in rows if r["engine"] == engine
                    and r["shape"] == "circle"
                    and r["seed"] == rows[0]["seed"]), None)
        if hit is None:
            continue
        axes[2].plot(
            [h["samples"] for h in hit["history"]],
            [h["iou"] for h in hit["history"]],
            "o-", color=color,
            label="Mazumdar" if engine == "mazumdar" else "Float MLP")
    axes[2].set_xlabel("Training samples seen")
    axes[2].set_ylabel("IoU (circle)")
    axes[2].set_title("(c) Convergence on circle")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.25)
    axes[2].set_ylim(0, 1.05)

    fig.suptitle(
        f"Shape learning: IoU and pixel accuracy ({profile})\n"
        "64×64 balanced bitmaps; error bars = sample SD across seeds",
        fontsize=11, fontweight="bold")
    fig.tight_layout()
    for directory in (FIGS, MEDIA):
        directory.mkdir(parents=True, exist_ok=True)
        fig.savefig(directory / "q1_shapes.png", bbox_inches="tight")
        fig.savefig(directory / "q1_shapes.svg", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=PROFILES, default="smoke")
    args = parser.parse_args()
    cfg = PROFILES[args.profile]
    RESULTS.mkdir(parents=True, exist_ok=True)

    rows = []
    t0 = time.perf_counter()
    for engine in cfg["engines"]:
        for shape in SHAPES:
            for seed in cfg["seeds"]:
                print(f"{engine} {shape} seed={seed} ...", flush=True)
                row = run_one(engine, shape, seed, cfg)
                rows.append(row)
                print(
                    f"  iou={row['final_iou']:.3f} "
                    f"pix={row['final_pixel_acc']:.3f} "
                    f"stream={row['final_stream_acc']:.3f}",
                    flush=True)

    elapsed = time.perf_counter() - t0
    summary = aggregate(rows)
    render(rows, summary, args.profile)
    # flatten history out of CSV rows
    flat = [{k: v for k, v in r.items() if k != "history"} for r in rows]
    payload = dict(
        profile=args.profile, config=cfg, elapsed_s=elapsed,
        runs=rows, summary=summary,
    )
    out_json = RESULTS / "q1_shapes.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with open(RESULTS / "q1_shapes.csv", "w", newline="",
              encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    print(json.dumps(summary, indent=2))
    print(f"elapsed {elapsed:.1f}s  wrote {out_json}")


if __name__ == "__main__":
    main()
