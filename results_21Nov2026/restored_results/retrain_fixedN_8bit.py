"""Train fixed-range SENA with weight_bits=8 (Pending #5 — true retrain, not requant)."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SIM = Path(r"<JULY2026>/SENA_Project\sena2026\sim")
sys.path.insert(0, str(SIM))

from sena_ensemble import SENAEnsemble  # noqa: E402

REPEATS = [0, 1, 2, 3, 4]
T_MAX = 40
THETA_EE = 0.55
THETA_FULL = 1.01


def load_data():
    z = np.load(HERE / "mnist_bin_t0.3_tr10000_te2500_s2026.npz")
    return {k: z[k] for k in z.files}


def main():
    data = load_data()
    Xtr, Ytr = data["Xtr"], data["Ytr"]
    Xte, yte = data["Xte"], data["yte"]
    layer_sizes = (int(Xtr.shape[1]), 160, 10)
    rows = []
    t0 = time.perf_counter()

    for repeat in REPEATS:
        ckpt = HERE / f"q1_fixedN_8bit_r{repeat}.npz"
        if ckpt.exists():
            print(f"loading {ckpt.name}", flush=True)
            ens, _ = SENAEnsemble.load(ckpt)
        else:
            seeds = [100 * repeat + i for i in range(3)]
            print(f"training repeat {repeat} seeds={seeds} ...", flush=True)
            ens = SENAEnsemble.train_members(
                Xtr, Ytr, layer_sizes=layer_sizes,
                n_members=3, epochs=40, seeds=seeds,
                weight_bits=8,
                samples_per_epoch=3000,
                val_every=10, T_probe=12, verbose=True,
                adaptive_calibration=False, fixed_N=64,
            )
            ens.save(ckpt, metadata=dict(
                weight_bits=8, fixed_N=64, repeat=repeat,
                dataset_seed=2026, threshold=0.3))
            print(f"saved {ckpt.name}", flush=True)

        for theta, name in ((THETA_FULL, "full_T40"), (THETA_EE, "EE")):
            out = ens.accuracy_early_exit(Xte, yte, T_max=T_MAX, theta=theta)
            row = dict(
                condition=f"fixedN_8bit_{name}",
                repeat=repeat, M=3, weight_bits=8, fixed_N=64,
                theta=theta, T_max=T_MAX,
                accuracy=out["acc"],
                mean_passes=out["mean_passes"],
                relative_energy=out["mean_energy"] / out["energy_full"],
                energy_saving=out["energy_saving"],
            )
            rows.append(row)
            print(
                f"  r{repeat} {name}: acc={row['accuracy']*100:.2f}% "
                f"passes={row['mean_passes']:.2f}",
                flush=True)

    # summary
    summary = []
    for condition in sorted({r["condition"] for r in rows}):
        subset = [r for r in rows if r["condition"] == condition]
        item = dict(condition=condition, n_repeats=len(subset))
        for key in ("accuracy", "mean_passes", "relative_energy", "energy_saving"):
            v = np.asarray([r[key] for r in subset], float)
            item[f"{key}_mean"] = float(v.mean())
            item[f"{key}_std"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
        summary.append(item)

    payload = dict(
        name="fixedN_8bit_retrain",
        note="True 8-bit training with fixed_N=64 (not requant of 16-bit)",
        elapsed_s=time.perf_counter() - t0,
        config=dict(hidden=160, epochs=40, samples_per_epoch=3000,
                    members=3, fixed_N=64, weight_bits=8, repeats=REPEATS),
        runs=rows,
        summary=summary,
    )
    out = HERE / "q1_fixedN_8bit_retrain.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out}", flush=True)
    for s in summary:
        print(
            f"{s['condition']}: "
            f"{100*s['accuracy_mean']:.2f} ± {100*s['accuracy_std']:.2f}%",
            flush=True,
        )


if __name__ == "__main__":
    main()
