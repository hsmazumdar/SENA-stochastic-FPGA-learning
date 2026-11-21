"""
Regenerate V11 Pending_Experiments items 1–4 and 6 from fixedN checkpoints.
Also: 8-bit requant eval of the same checkpoints (labeled requant, not retrain).

Sources (copied beside this script):
  q1_ablation_full_r{0..4}_fixedN.npz
  mnist_bin_t0.3_tr10000_te2500_s2026.npz
  q1_online_training.json

Sim code: <JULY2026>/SENA_Project/sena2026/sim
"""
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
from stochastic_core import StochasticConfig, StochasticNet  # noqa: E402

REPEATS = [0, 1, 2, 3, 4]
T_MAX = 40
THETA_EE = 0.55
THETA_FULL = 1.01
BUDGETS = [1, 2, 3, 5, 10, 20, 40]
T_MINS = [1, 2, 3, 5]


def load_data():
    z = np.load(HERE / "mnist_bin_t0.3_tr10000_te2500_s2026.npz")
    return {k: z[k] for k in z.files}


def predict_early_exit_tmin(ens, X, T_max=40, theta=0.55, T_min=1, batch=256):
    """Same as SENAEnsemble.predict_early_exit but with T_min gate."""
    X = np.asarray(X, dtype=np.int64)
    n = len(X)
    n_out = ens.n_out
    M = len(ens.nets)
    votes = np.zeros((n, n_out), dtype=np.int64)
    done = np.zeros(n, dtype=bool)
    passes_used = np.zeros(n, dtype=np.int64)
    pred = np.zeros(n, dtype=np.int64)
    conf = np.zeros(n, dtype=np.float64)

    for t in range(1, T_max + 1):
        active = np.where(~done)[0]
        if len(active) == 0:
            break
        for s in range(0, len(active), batch):
            idx = active[s:s + batch]
            xb = X[idx]
            for net in ens.nets:
                acts, _ = net.forward(xb, stochastic=True)
                votes[idx] += acts[-1]
        passes_used[active] = t
        total = np.maximum(votes.sum(axis=1).astype(np.float64), 1.0)
        conf_now = votes.max(axis=1) / total
        pred_now = np.argmax(votes, axis=1)
        newly = (~done) & (t >= T_min) & (conf_now >= theta)
        if t == T_max:
            newly = ~done
        done[newly] = True
        pred[newly] = pred_now[newly]
        conf[newly] = conf_now[newly]

    e_unit = ens.energy.e_forward_stoch * M
    energy = passes_used.astype(np.float64) * e_unit
    e_full = float(T_max * e_unit)
    return dict(
        pred=pred, conf=conf, passes_used=passes_used,
        mean_passes=float(np.mean(passes_used)),
        mean_energy=float(np.mean(energy)),
        energy_full=e_full,
        energy_saving=1.0 - float(np.mean(energy)) / e_full,
        M=M, T_max=T_max, theta=theta, T_min=T_min,
    )


def eval_row(ens, X, y, *, theta, T_max, T_min=1, condition="", repeat=0):
    out = predict_early_exit_tmin(ens, X, T_max=T_max, theta=theta, T_min=T_min)
    acc = float(np.mean(out["pred"] == np.asarray(y)))
    return dict(
        condition=condition, repeat=repeat, M=len(ens.nets),
        theta=theta, T_max=T_max, T_min=T_min,
        accuracy=acc,
        mean_passes=out["mean_passes"],
        relative_energy=out["mean_energy"] / out["energy_full"],
        energy_saving=out["energy_saving"],
        pred=out["pred"], conf=out["conf"], passes_used=out["passes_used"],
    )


def aggregate(rows, keys=("accuracy", "mean_passes", "relative_energy", "energy_saving")):
    summary = []
    for condition in sorted({r["condition"] for r in rows}):
        subset = [r for r in rows if r["condition"] == condition]
        item = dict(condition=condition, n_repeats=len(subset))
        for key in keys:
            values = np.asarray([r[key] for r in subset], dtype=float)
            item[f"{key}_mean"] = float(values.mean())
            item[f"{key}_std"] = (
                float(values.std(ddof=1)) if len(values) > 1 else 0.0)
        for k in ("theta", "T_max", "T_min", "M"):
            if k in subset[0]:
                item[k] = subset[0][k]
        summary.append(item)
    return summary


def coverage_curve(pred, conf, y, coverages=None):
    """Abstain on lowest-confidence fraction; report acc vs coverage."""
    y = np.asarray(y)
    pred = np.asarray(pred)
    conf = np.asarray(conf, dtype=float)
    order = np.argsort(-conf)  # high confidence first
    n = len(y)
    if coverages is None:
        coverages = [1.0, 0.95, 0.90, 0.85, 0.80, 0.70, 0.60, 0.50]
    rows = []
    for cov in coverages:
        k = max(1, int(round(cov * n)))
        idx = order[:k]
        rows.append(dict(
            coverage=float(k / n),
            target_coverage=float(cov),
            n_kept=int(k),
            accuracy=float(np.mean(pred[idx] == y[idx])),
            mean_conf=float(np.mean(conf[idx])),
        ))
    return rows


def requant_ensemble_8bit(ens):
    """Clip existing 16-bit weights to signed 8-bit and retarget cfg."""
    nets = []
    for net in ens.nets:
        cfg = StochasticConfig(
            layer_sizes=tuple(net.sizes),
            seed=int(net.cfg.seed),
            weight_bits=8,
            rng_kind=getattr(net.cfg, "rng_kind", "numpy"),
        )
        new = StochasticNet(cfg)
        new.weights = [np.clip(w, -127, 127).astype(np.int64) for w in net.weights]
        new.N_fwd = list(net.N_fwd)
        new.N_back = float(net.N_back)
        new.alpha = float(net.alpha)
        new._calibrated = True
        nets.append(new)
    return SENAEnsemble(nets)


def strip_arrays(row):
    return {k: v for k, v in row.items()
            if k not in ("pred", "conf", "passes_used")}


def iso_accuracy_online(path, sena_target=0.594):
    """Pending #6: first BP eval with acc >= SENA endpoint, energy ratio."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    runs = payload.get("runs", [])
    by_method = {}
    for r in runs:
        by_method.setdefault(r["method"], []).append(r)

    # Prefer seed-averaged SENA endpoint if present; else use stated target.
    sena_rows = by_method.get("sena_online", [])
    bp_rows = [r for r in runs if str(r.get("method", "")).startswith("bp")
               or "backprop" in str(r.get("method", "")).lower()
               or r.get("method") in {"bp", "mlp_bp", "bp_online"}]

    # Discover method names
    methods = sorted(by_method)
    result = dict(methods=methods, sena_target=sena_target, matches=[])

    # SENA endpoint: last sample count per seed, mean energy / acc
    if sena_rows:
        # group by seed, take max samples row
        ends = {}
        for r in sena_rows:
            s = r.get("seed", 0)
            if s not in ends or r["samples"] > ends[s]["samples"]:
                ends[s] = r
        sena_acc = float(np.mean([r["accuracy"] for r in ends.values()]))
        sena_energy = float(np.mean([r["energy"] for r in ends.values()]))
        result["sena_endpoint"] = dict(
            accuracy_mean=sena_acc, energy_mean=sena_energy,
            n_seeds=len(ends), per_seed=list(ends.values()))
        target = sena_acc
    else:
        target = sena_target
        sena_energy = None

    for method, rows in by_method.items():
        if method == "sena_online":
            continue
        # first eval per seed where accuracy >= target
        hits = []
        by_seed = {}
        for r in rows:
            by_seed.setdefault(r.get("seed", 0), []).append(r)
        for seed, seq in by_seed.items():
            seq = sorted(seq, key=lambda r: r["samples"])
            hit = next((r for r in seq if r["accuracy"] >= target), None)
            if hit is not None:
                hits.append(hit)
        if hits and sena_energy is not None:
            bp_e = float(np.mean([h["energy"] for h in hits]))
            result["matches"].append(dict(
                method=method,
                n_seeds=len(hits),
                accuracy_mean=float(np.mean([h["accuracy"] for h in hits])),
                energy_mean=bp_e,
                samples_mean=float(np.mean([h["samples"] for h in hits])),
                energy_ratio_bp_over_sena=bp_e / sena_energy if sena_energy else None,
                energy_ratio_sena_cheaper_factor=bp_e / sena_energy if sena_energy else None,
                hits=hits,
            ))
    return result


def main():
    t_all = time.perf_counter()
    data = load_data()
    X, y = data["Xte"], data["yte"]
    print(f"test set {X.shape}", flush=True)

    full_rows = []
    budget_rows = []
    tmin_rows = []
    selective_runs = []
    requant_rows = []

    for repeat in REPEATS:
        ckpt = HERE / f"q1_ablation_full_r{repeat}_fixedN.npz"
        print(f"\n=== repeat {repeat}: load {ckpt.name} ===", flush=True)
        ens, meta = SENAEnsemble.load(ckpt)

        # 1) Fixed-range full budget
        t0 = time.perf_counter()
        row = eval_row(ens, X, y, theta=THETA_FULL, T_max=T_MAX, T_min=1,
                       condition="fixedN_full_T40", repeat=repeat)
        print(f"  full-T acc={row['accuracy']*100:.2f}% "
              f"in {time.perf_counter()-t0:.1f}s", flush=True)
        full_rows.append(row)

        # Selective curve from full-T confidences
        sel = coverage_curve(row["pred"], row["conf"], y)
        selective_runs.append(dict(
            repeat=repeat, source="fixedN_full_T40",
            base_accuracy=row["accuracy"], curve=sel,
            at_90=next(c for c in sel if abs(c["target_coverage"] - 0.90) < 1e-9),
        ))

        # Also selective from EE operating point
        t0 = time.perf_counter()
        ee = eval_row(ens, X, y, theta=THETA_EE, T_max=T_MAX, T_min=1,
                      condition="fixedN_EE_theta0.55", repeat=repeat)
        print(f"  EE acc={ee['accuracy']*100:.2f}% "
              f"passes={ee['mean_passes']:.2f} "
              f"in {time.perf_counter()-t0:.1f}s", flush=True)
        full_rows.append(ee)
        sel_ee = coverage_curve(ee["pred"], ee["conf"], y)
        selective_runs.append(dict(
            repeat=repeat, source="fixedN_EE_theta0.55",
            base_accuracy=ee["accuracy"], curve=sel_ee,
            at_90=next(c for c in sel_ee if abs(c["target_coverage"] - 0.90) < 1e-9),
        ))

        # 2) Budget sweep at theta=1.01 (forced budget)
        for T in BUDGETS:
            t0 = time.perf_counter()
            b = eval_row(ens, X, y, theta=THETA_FULL, T_max=T, T_min=1,
                         condition=f"fixedN_budget_T{T}", repeat=repeat)
            print(f"  budget T={T} acc={b['accuracy']*100:.2f}% "
                  f"in {time.perf_counter()-t0:.1f}s", flush=True)
            budget_rows.append(b)

        # 3) T_min sweep at theta=0.55
        for tmin in T_MINS:
            t0 = time.perf_counter()
            tm = eval_row(ens, X, y, theta=THETA_EE, T_max=T_MAX, T_min=tmin,
                          condition=f"fixedN_EE_Tmin{tmin}", repeat=repeat)
            print(f"  T_min={tmin} acc={tm['accuracy']*100:.2f}% "
                  f"passes={tm['mean_passes']:.2f} "
                  f"in {time.perf_counter()-t0:.1f}s", flush=True)
            tmin_rows.append(tm)

        # 5b) 8-bit requant (not retrain)
        ens8 = requant_ensemble_8bit(ens)
        t0 = time.perf_counter()
        r8f = eval_row(ens8, X, y, theta=THETA_FULL, T_max=T_MAX, T_min=1,
                       condition="fixedN_requant8_full_T40", repeat=repeat)
        r8e = eval_row(ens8, X, y, theta=THETA_EE, T_max=T_MAX, T_min=1,
                       condition="fixedN_requant8_EE", repeat=repeat)
        print(f"  requant8 full={r8f['accuracy']*100:.2f}% "
              f"EE={r8e['accuracy']*100:.2f}% "
              f"in {time.perf_counter()-t0:.1f}s", flush=True)
        requant_rows.extend([r8f, r8e])

    # Summaries
    def pack(rows, name):
        slim = [strip_arrays(r) for r in rows]
        return dict(
            name=name,
            note="Regenerated 13 Nov 2026 from fixedN ablation checkpoints",
            config=dict(
                split="mnist_bin_t0.3_tr10000_te2500_s2026",
                repeats=REPEATS, T_max_default=T_MAX,
                checkpoints=[f"q1_ablation_full_r{r}_fixedN.npz" for r in REPEATS],
            ),
            elapsed_note="see parent",
            runs=slim,
            summary=aggregate(slim),
        )

    full_payload = pack(full_rows, "fixedN_fullT_and_EE")
    budget_payload = pack(budget_rows, "fixedN_budget_sweep")
    tmin_payload = pack(tmin_rows, "fixedN_Tmin_sweep")
    requant_payload = pack(requant_rows, "fixedN_requant8")

    # Selective aggregate at 90% coverage
    sel_by_src = {}
    for s in selective_runs:
        sel_by_src.setdefault(s["source"], []).append(s)
    sel_summary = {}
    for src, items in sel_by_src.items():
        acc90 = [i["at_90"]["accuracy"] for i in items]
        base = [i["base_accuracy"] for i in items]
        sel_summary[src] = dict(
            n_repeats=len(items),
            base_accuracy_mean=float(np.mean(base)),
            base_accuracy_std=float(np.std(base, ddof=1)) if len(base) > 1 else 0.0,
            acc_at_90cov_mean=float(np.mean(acc90)),
            acc_at_90cov_std=float(np.std(acc90, ddof=1)) if len(acc90) > 1 else 0.0,
            # mean curve across repeats at shared coverages
            curve_mean=[
                dict(
                    target_coverage=c,
                    accuracy_mean=float(np.mean([
                        next(x["accuracy"] for x in it["curve"]
                             if abs(x["target_coverage"] - c) < 1e-9)
                        for it in items
                    ])),
                )
                for c in [1.0, 0.95, 0.90, 0.85, 0.80, 0.70, 0.60, 0.50]
            ],
        )
    selective_payload = dict(
        name="selective_prediction_coverage",
        note="Coverage–accuracy from vote-fraction confidence; abstain lowest conf",
        runs=selective_runs,
        summary=sel_summary,
    )

    iso = iso_accuracy_online(HERE / "q1_online_training.json")
    iso_payload = dict(
        name="iso_accuracy_online_proxy",
        note="Pending #6 from existing q1_online_training.json",
        result=iso,
    )

    outs = {
        "q1_fixedN_fullT_EE.json": full_payload,
        "q1_fixedN_budget_sweep.json": budget_payload,
        "q1_fixedN_Tmin_sweep.json": tmin_payload,
        "q1_selective_coverage.json": selective_payload,
        "q1_fixedN_requant8.json": requant_payload,
        "q1_iso_accuracy_online.json": iso_payload,
    }
    for name, payload in outs.items():
        path = HERE / name
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {path}", flush=True)

    # Human-readable digest matching paper claims
    digest = []
    def mean_std(summary, condition, key="accuracy"):
        for s in summary:
            if s["condition"] == condition:
                return s[f"{key}_mean"], s[f"{key}_std"]
        return None, None

    ft_m, ft_s = mean_std(full_payload["summary"], "fixedN_full_T40")
    ee_m, ee_s = mean_std(full_payload["summary"], "fixedN_EE_theta0.55")
    digest.append(f"fixedN full T=40: {100*ft_m:.2f} ± {100*ft_s:.2f}%  (paper claimed 90.79 ± 0.26)")
    digest.append(f"fixedN EE θ=0.55: {100*ee_m:.2f} ± {100*ee_s:.2f}%  (paper ~90.8 / 90.28)")
    for src, s in sel_summary.items():
        digest.append(
            f"selective {src}: base {100*s['base_accuracy_mean']:.2f}% -> "
            f"90% cov {100*s['acc_at_90cov_mean']:.2f} ± {100*s['acc_at_90cov_std']:.2f}% "
            f"(paper 95.2%@90%)"
        )
    r8f_m, r8f_s = mean_std(requant_payload["summary"], "fixedN_requant8_full_T40")
    r8e_m, r8e_s = mean_std(requant_payload["summary"], "fixedN_requant8_EE")
    digest.append(
        f"requant8 (NOT retrain) full: {100*r8f_m:.2f} ± {100*r8f_s:.2f}%; "
        f"EE: {100*r8e_m:.2f} ± {100*r8e_s:.2f}%  "
        f"(paper ~90.9% is 8-bit-TRAINED — still needs retrain for claim match)"
    )
    for m in iso.get("matches", []):
        digest.append(
            f"iso-acc online: {m['method']} reaches SENA acc with "
            f"energy ratio {m['energy_ratio_bp_over_sena']:.2f}× vs SENA endpoint"
        )
    digest.append(f"total elapsed_s={time.perf_counter()-t_all:.1f}")
    (HERE / "RESTORE_DIGEST.txt").write_text("\n".join(digest) + "\n", encoding="utf-8")
    print("\n".join(digest), flush=True)


if __name__ == "__main__":
    main()
