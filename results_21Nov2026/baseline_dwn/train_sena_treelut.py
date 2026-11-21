"""
Matched software baseline C: TreeLUT / XGBoost on SENA's frozen binary MNIST split.
CPU-only. Emits accuracy JSON + optional Verilog (P&R later in office).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from xgboost import XGBClassifier

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE / "results_treelut"
OUT.mkdir(parents=True, exist_ok=True)

# Binary pixels already 0/1 → treat as 1-bit quantized features
W_FEATURE = 1
SEEDS = [0, 1, 2, 3, 4]


def load_split():
    z = np.load(DATA / "mnist_bin_t0.3_tr10000_te2500_s2026.npz")
    Xtr = z["Xtr"].astype(np.float32)
    ytr = z["ytr"].astype(np.int64)
    Xte = z["Xte"].astype(np.float32)
    yte = z["yte"].astype(np.int64)
    return Xtr, ytr, Xte, yte


def main():
    Xtr, ytr, Xte, yte = load_split()
    # TreeLUT expects unsigned integer features
    Xtr_q = np.clip(np.round(Xtr), 0, 2**W_FEATURE - 1).astype(np.float32)
    Xte_q = np.clip(np.round(Xte), 0, 2**W_FEATURE - 1).astype(np.float32)

    runs = []
    t0 = time.perf_counter()
    for seed in SEEDS:
        clf = XGBClassifier(
            objective="multi:softmax",
            num_class=10,
            n_estimators=30,
            max_depth=5,
            learning_rate=0.8,
            tree_method="hist",
            random_state=seed,
            n_jobs=4,
            verbosity=0,
        )
        clf.fit(Xtr_q, ytr)
        pred = clf.predict(Xte_q)
        acc = float(np.mean(pred == yte))
        runs.append(dict(seed=seed, accuracy=acc, n_estimators=30, max_depth=5))
        print(f"seed={seed} acc={acc*100:.2f}%", flush=True)

        # TreeLUT convert + verilog for seed 0 only (enough for later P&R)
        if seed == 0:
            try:
                from treelut import TreeLUTClassifier
                tl_dir = OUT / "verilog_seed0"
                tl_dir.mkdir(parents=True, exist_ok=True)
                tl = TreeLUTClassifier(
                    xgb_model=clf,
                    w_feature=W_FEATURE,
                    w_tree=3,
                    pipeline=[0, 1, 1],
                    dir_path=str(tl_dir) + "/",
                )
                tl.convert()
                tl_pred = tl.predict(Xte_q)
                tl_acc = float(np.mean(tl_pred == yte))
                tl.verilog()
                runs[-1]["treelut_acc"] = tl_acc
                runs[-1]["verilog_dir"] = str(tl_dir)
                print(f"  TreeLUT convert acc={tl_acc*100:.2f}%  verilog→{tl_dir}", flush=True)
            except Exception as e:
                runs[-1]["treelut_error"] = repr(e)
                print(f"  TreeLUT convert failed: {e}", flush=True)

    accs = np.asarray([r["accuracy"] for r in runs], float)
    payload = dict(
        name="treelut_xgb_sena_split",
        note=(
            "CPU matched-split software baseline (plan C). Same 10k/2.5k binary "
            "MNIST as SENA. Not a same-board FPGA P&R yet — Verilog emitted for office."
        ),
        config=dict(
            split="mnist_bin_t0.3_tr10000_te2500_s2026",
            w_feature=W_FEATURE,
            xgb=dict(n_estimators=30, max_depth=5, eta=0.8),
            seeds=SEEDS,
        ),
        elapsed_s=time.perf_counter() - t0,
        runs=runs,
        accuracy_mean=float(accs.mean()),
        accuracy_std=float(accs.std(ddof=1)) if len(accs) > 1 else 0.0,
        sena_reference=dict(
            fixedN_EE="90.26±0.24% (restored)",
            fixedN_fullT="90.76±0.39% (restored)",
        ),
    )
    out = OUT / "treelut_sena_train.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"MEAN {100*payload['accuracy_mean']:.2f} ± {100*payload['accuracy_std']:.2f}%",
        flush=True,
    )
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
