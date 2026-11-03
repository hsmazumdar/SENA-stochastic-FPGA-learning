"""
Collect Q1-paper tables for the 3×NN (AND/OR/XOR) Mazumdar PoC.

Topology: i=6, j=6+bias, k=3+pad → 42 + 28 + 24 connections.
Produces CSV + markdown under results/ for manuscript insertion.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from golden_logic663_skip import Net663, patterns64

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "results"
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = [7, 21, 42, 2026, 31415]
MAX_EPOCHS = 80
EVAL_EVERY = 5
T_EVAL = 16
EE_MIN = 3


def run_one(seed: int, early_exit: bool, with_skip: bool = True):
    X, Y = patterns64()
    net = Net663(seed=seed, N=12)
    if not with_skip:
        net.w_ik[:] = 0

    rows = []
    for ep in range(1, MAX_EPOCHS + 1):
        order = np.random.default_rng(ep + seed).permutation(64)
        for n in order:
            if not with_skip:
                # train but keep skip frozen at 0
                net.train_sample(X[n], Y[n], lr_passes=2)
                net.w_ik[:] = 0
            else:
                net.train_sample(X[n], Y[n], lr_passes=2)

        if ep % EVAL_EVERY == 0 or ep == 1:
            p, j, mp = net.eval_joint(
                X, Y, T=T_EVAL, early_exit=early_exit, ee_min=EE_MIN
            )
            rows.append(
                dict(
                    seed=seed,
                    epoch=ep,
                    early_exit=int(early_exit),
                    skip=int(with_skip),
                    and_pct=100.0 * float(p[0]),
                    or_pct=100.0 * float(p[1]),
                    xor_pct=100.0 * float(p[2]),
                    joint_pct=100.0 * float(j),
                    mean_passes=float(mp),
                )
            )
            if j >= 0.999:
                break
    # final high-T eval
    p, j, mp = net.eval_joint(X, Y, T=32, early_exit=early_exit, ee_min=EE_MIN)
    final = dict(
        seed=seed,
        early_exit=int(early_exit),
        skip=int(with_skip),
        and_pct=100.0 * float(p[0]),
        or_pct=100.0 * float(p[1]),
        xor_pct=100.0 * float(p[2]),
        joint_pct=100.0 * float(j),
        mean_passes=float(mp),
        epochs_run=rows[-1]["epoch"] if rows else 0,
    )
    return rows, final


def summarize(finals):
    a = np.array([f["and_pct"] for f in finals])
    o = np.array([f["or_pct"] for f in finals])
    x = np.array([f["xor_pct"] for f in finals])
    j = np.array([f["joint_pct"] for f in finals])
    m = np.array([f["mean_passes"] for f in finals])
    e = np.array([f["epochs_run"] for f in finals], dtype=float)
    return dict(
        n=len(finals),
        and_mean=float(a.mean()),
        and_std=float(a.std(ddof=1)) if len(a) > 1 else 0.0,
        or_mean=float(o.mean()),
        or_std=float(o.std(ddof=1)) if len(o) > 1 else 0.0,
        xor_mean=float(x.mean()),
        xor_std=float(x.std(ddof=1)) if len(x) > 1 else 0.0,
        joint_mean=float(j.mean()),
        joint_std=float(j.std(ddof=1)) if len(j) > 1 else 0.0,
        mean_passes_mean=float(m.mean()),
        mean_passes_std=float(m.std(ddof=1)) if len(m) > 1 else 0.0,
        epochs_mean=float(e.mean()),
        epochs_std=float(e.std(ddof=1)) if len(e) > 1 else 0.0,
    )


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def md_table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for r in rows:
        lines.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(lines) + "\n"


def main():
    hist_all = []
    finals = []
    configs = [
        ("skip_ee", True, True),
        ("skip_fixedT", True, False),
        ("noskip_ee", False, True),
    ]

    summary_rows = []
    for tag, skip, ee in configs:
        conf_finals = []
        for seed in SEEDS:
            print(f"[{tag}] seed={seed} …")
            hist, final = run_one(seed, early_exit=ee, with_skip=skip)
            for h in hist:
                h["config"] = tag
                hist_all.append(h)
            final["config"] = tag
            finals.append(final)
            conf_finals.append(final)
        s = summarize(conf_finals)
        s["config"] = tag
        s["skip"] = int(skip)
        s["early_exit"] = int(ee)
        summary_rows.append(s)
        print(
            f"  → joint {s['joint_mean']:.1f}±{s['joint_std']:.1f}%  "
            f"T={s['mean_passes_mean']:.2f}  epochs≈{s['epochs_mean']:.0f}"
        )

    write_csv(
        OUT / "q1_3xnn_history.csv",
        hist_all,
        [
            "config",
            "seed",
            "epoch",
            "early_exit",
            "skip",
            "and_pct",
            "or_pct",
            "xor_pct",
            "joint_pct",
            "mean_passes",
        ],
    )
    write_csv(
        OUT / "q1_3xnn_final_per_seed.csv",
        finals,
        [
            "config",
            "seed",
            "early_exit",
            "skip",
            "and_pct",
            "or_pct",
            "xor_pct",
            "joint_pct",
            "mean_passes",
            "epochs_run",
        ],
    )

    sum_fields = [
        "config",
        "skip",
        "early_exit",
        "n",
        "and_mean",
        "and_std",
        "or_mean",
        "or_std",
        "xor_mean",
        "xor_std",
        "joint_mean",
        "joint_std",
        "mean_passes_mean",
        "mean_passes_std",
        "epochs_mean",
        "epochs_std",
    ]
    write_csv(OUT / "q1_3xnn_summary.csv", summary_rows, sum_fields)

    # Paper-facing markdown
    md = []
    md.append("# Q1 paper tables — 3×NN (AND/OR/XOR) Mazumdar PoC\n")
    md.append("## Topology\n")
    md.append(
        md_table(
            ["Layer", "Size", "Notes"],
            [
                ["i (inputs)", "6", "Ai1,Ai2, Oi1,Oi2, Xi1,Xi2"],
                ["j (hidden)", "6 + bias", "bias column → 7"],
                ["k (outputs)", "3 + pad", "Ao, Oo, Xo (+ unused row)"],
                ["i→j", "42", "6×7"],
                ["j→k", "28", "4×7"],
                ["i→k skip", "24", "4×6"],
                ["Epoch", "64", "all 6-bit patterns, on-the-fly targets"],
            ],
        )
    )
    md.append("\n## Table A — Final accuracy (5 seeds, mean ± std)\n")
    md.append(
        md_table(
            [
                "Config",
                "AND %",
                "OR %",
                "XOR %",
                "Joint %",
                "Mean T",
                "Epochs",
            ],
            [
                [
                    s["config"],
                    f"{s['and_mean']:.1f}±{s['and_std']:.1f}",
                    f"{s['or_mean']:.1f}±{s['or_std']:.1f}",
                    f"{s['xor_mean']:.1f}±{s['xor_std']:.1f}",
                    f"{s['joint_mean']:.1f}±{s['joint_std']:.1f}",
                    f"{s['mean_passes_mean']:.2f}±{s['mean_passes_std']:.2f}",
                    f"{s['epochs_mean']:.0f}±{s['epochs_std']:.0f}",
                ]
                for s in summary_rows
            ],
        )
    )
    md.append("\n**Config key:** `skip_ee` = skip + early-exit (paper default); ")
    md.append("`skip_fixedT` = skip, fixed T=16; `noskip_ee` = no i→k skip.\n")
    md.append("\n## Table B — Learning curve (skip_ee, seed mean at selected epochs)\n")

    # aggregate history for skip_ee
    from collections import defaultdict

    bucket = defaultdict(list)
    for h in hist_all:
        if h["config"] == "skip_ee":
            bucket[h["epoch"]].append(h)
    curve_rows = []
    for ep in sorted(bucket):
        hs = bucket[ep]
        curve_rows.append(
            [
                ep,
                f"{np.mean([x['and_pct'] for x in hs]):.1f}",
                f"{np.mean([x['or_pct'] for x in hs]):.1f}",
                f"{np.mean([x['xor_pct'] for x in hs]):.1f}",
                f"{np.mean([x['joint_pct'] for x in hs]):.1f}",
                f"{np.mean([x['mean_passes'] for x in hs]):.2f}",
            ]
        )
    md.append(
        md_table(
            ["Epoch", "AND %", "OR %", "XOR %", "Joint %", "Mean T"],
            curve_rows,
        )
    )

    md.append("\n## Table C — Hardware status\n")
    md.append(
        md_table(
            ["Item", "Status"],
            [
                ["Golden twin (this table)", "DONE — software Mazumdar, 5 seeds"],
                ["Papilio RTL (42+28+24, EE, LED, UART)", "Written — not yet bitgen/programmed"],
                ["On-board measured accuracy", "PENDING — JTAG/bitstream + free COM"],
                ["PC 30 s progress logger", "Ready — host/papilio_progress_30s.py"],
            ],
        )
    )

    md.append("\n## Suggested manuscript sentence\n")
    best = next(s for s in summary_rows if s["config"] == "skip_ee")
    md.append(
        f"> On the 6–6–4 AND/OR/XOR PoC (42+28+24 connections), Mazumdar "
        f"multiplier-free online training reaches "
        f"**{best['joint_mean']:.1f}±{best['joint_std']:.1f}%** joint accuracy "
        f"across {best['n']} seeds, with early-exit mean depth "
        f"**{best['mean_passes_mean']:.2f}** stochastic passes "
        f"(vs fixed T=16).\n"
    )

    (OUT / "Q1_3xNN_TABLES.md").write_text("".join(md), encoding="utf-8")
    (OUT / "q1_3xnn_summary.json").write_text(
        json.dumps(dict(summary=summary_rows, seeds=SEEDS), indent=2),
        encoding="utf-8",
    )
    print("Wrote", OUT / "Q1_3xNN_TABLES.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
