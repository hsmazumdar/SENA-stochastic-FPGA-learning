"""
demo_bmp_live.py
================
Headless version of the GUI's "BMP 64x64 Live Learn" tab.

Streams random pixel probes from cir/sqr/trngl.bmp, trains a 2-30-1 net,
and renders the same four panels the GUI shows:

  A  source bitmap with the probes of the last step
  B  live analog output map of the network
  C  OVERLAP: ideal edge (yellow) vs the learned edge over time (blue, fading)
  D  continuous accuracy trace

Usage:
    python demo_bmp_live.py                     # both engines, all 3 shapes
    python demo_bmp_live.py mazumdar circle     # one engine, one shape
"""

from __future__ import annotations
import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure

from shape_bmp import (SHAPES, BMP_NAMES, get_shape_image, balance_stats,
                       PixelStream, assets_dir)
from pixel_learners import make_learner


def run(engine: str, shape: str, chunks: int = 200, batch: int = 256,
        snapshot_every: int = 20, out_dir: str | None = None) -> dict:
    img = get_shape_image(shape, size=64)
    st = balance_stats(img)
    stream = PixelStream(img, seed=1)
    net = make_learner(engine, seed=0)

    overlap, hist = [], []
    t0 = time.time()
    for c in range(chunks):
        X, y, px, py = stream.next_batch(batch)
        net.train_chunk(X, y)
        hist.append((stream.n_drawn, net.accuracy(X, y)))
        if (c + 1) % snapshot_every == 0:
            overlap.append(net.predict_grid(64))
    train_s = time.time() - t0

    grid = net.predict_grid(64)
    pixel_acc = float(np.mean((grid > 0.5) == (img > 127)))
    overlap = overlap[-10:]

    fig = Figure(figsize=(9, 7.2), dpi=110, facecolor="#1a1d23")
    axA, axB = fig.add_subplot(221), fig.add_subplot(222)
    axC, axD = fig.add_subplot(223), fig.add_subplot(224)
    for ax in (axA, axB, axC, axD):
        ax.set_facecolor("#0f1218")
        ax.tick_params(colors="#94a3b8", labelsize=7)
        for sp in ax.spines.values():
            sp.set_color("#334155")
        ax.title.set_color("#e2e8f0")
        ax.title.set_size(9)
        ax.xaxis.label.set_color("#94a3b8")
        ax.yaxis.label.set_color("#94a3b8")

    axA.imshow(img, cmap="gray", origin="upper", vmin=0, vmax=255,
               interpolation="nearest")
    lab = y.astype(bool)
    axA.scatter(px[lab], py[lab], s=5, c="#22c55e", linewidths=0)
    axA.scatter(px[~lab], py[~lab], s=5, c="#f43f5e", linewidths=0)
    axA.set_title(f"{BMP_NAMES[shape]}  W={st['n_white']} B={st['n_black']}"
                  f"  (last {batch} probes)")
    axA.set_xticks([]); axA.set_yticks([])

    axB.imshow(grid, cmap="viridis", origin="upper", vmin=0, vmax=1,
               interpolation="nearest")
    axB.set_title(f"Net analog output — {engine}")
    axB.set_xticks([]); axB.set_yticks([])

    axC.contour(img.astype(float), levels=[127.5], colors="#facc15",
                linewidths=2.0)
    k = max(len(overlap), 1)
    for i, g in enumerate(overlap):
        axC.contour(g, levels=[0.5], colors="#38bdf8",
                    linewidths=0.8 + 1.4 * (i + 1) / k,
                    alpha=0.12 + 0.88 * ((i + 1) / k) ** 2)
    axC.set_xlim(0, 63); axC.set_ylim(0, 63)
    axC.invert_yaxis(); axC.set_aspect("equal")
    axC.set_xticks([]); axC.set_yticks([])
    axC.set_title("Overlap: ideal (yellow) vs learned edge over time")

    axD.plot([h[0] for h in hist], [h[1] * 100 for h in hist],
             color="#34d399", lw=1.2)
    axD.axhline(50, color="#475569", ls=":", lw=1)
    axD.set_ylim(40, 102)
    axD.set_title(f"Live train accuracy — final bitmap acc {pixel_acc*100:.1f}%")
    axD.set_xlabel("probes drawn")
    axD.grid(True, color="#1e293b")

    fig.tight_layout(pad=1.8)
    out_dir = out_dir or os.path.join(assets_dir(), "figs")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"live_{engine}_{shape}.png")
    fig.savefig(path, facecolor=fig.get_facecolor())
    return dict(path=path, pixel_acc=pixel_acc, probes=stream.n_drawn,
                train_s=train_s, balanced=st["balanced"])


if __name__ == "__main__":
    args = sys.argv[1:]
    engines = [args[0]] if args else ["mazumdar", "mlp"]
    shapes = [args[1]] if len(args) > 1 else list(SHAPES)
    for eng in engines:
        for sh in shapes:
            r = run(eng, sh)
            print(f"{eng:9s} {sh:9s} probes={r['probes']:6d}  "
                  f"bitmap_acc={r['pixel_acc']*100:5.1f}%  "
                  f"balanced={r['balanced']}  {r['train_s']:5.1f}s  "
                  f"-> {r['path']}")
