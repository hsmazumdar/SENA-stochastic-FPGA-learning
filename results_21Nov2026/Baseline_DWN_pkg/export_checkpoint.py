"""
export_checkpoint.py - convert a PyTorch checkpoint of an OFFICIAL torch_dwn
model (e.g. checkpoints/dwn_sm_sena_seed<k>.pt written by the earlier
baseline_dwn/train_sena_dwn_sm.py Colab run) into this package's model.json +
test_scores.npz, so it can go straight into dwn_to_verilog.py / collect_results.py.
Runs on CPU, no CUDA extension needed.

    python export_checkpoint.py <ckpt.pt> --seed 0 [--layers 1000,500 --n 6 --tau-inv 0.245]

Writes runs/<arch>/seed<k>/ (default arch name 'sm_ckpt') and appends to
runs/<arch>/accuracy.csv.  Uses the FINAL-epoch weights stored in the checkpoint;
do not use any 'best_test_acc' (that would be model selection on the test set).
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np
import torch

import dwn_ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--layers", default="1000,500")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--tau-inv", type=float, default=0.245)
    ap.add_argument("--data", default="data/mnist_bin_t0.3_tr10000_te2500_s2026.npz")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--arch-name", default="sm_ckpt")
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    state = ck.get("state_dict", ck) if isinstance(ck, dict) else ck
    state = {k.replace("_LUTLayer__dummy_mapping", "dummy_mapping"): v for k, v in state.items()}
    P = dwn_ref.torch_parts()
    widths = [int(v) for v in a.layers.split(",")]
    mods, prev = [], None
    d = np.load(a.data)
    prev = d["Xte"].shape[1]
    for i, w in enumerate(widths):
        mods.append(P["LUTLayerTorch"](prev, w, n=a.n, mapping="learnable" if i == 0 else "random"))
        prev = w
    mods.append(P["GroupSum"](10, tau=1.0 / a.tau_inv))
    model = torch.nn.Sequential(*mods)
    model.load_state_dict(state, strict=True)
    model.eval()
    with torch.no_grad():
        s = model(torch.tensor(d["Xte"], dtype=torch.float32))
    p_torch = s.argmax(1).numpy()
    meta = {"arch": a.arch_name, "imported_checkpoint": os.path.abspath(a.ckpt), "seed": a.seed,
            "layers": widths, "n": a.n, "tau_inv": a.tau_inv,
            "ckpt_reported_final_test_acc": ck.get("test_acc") if isinstance(ck, dict) else None,
            "ckpt_config": ck.get("config") if isinstance(ck, dict) else None}
    exp = dwn_ref.export_model(model, d["Xte"].shape[1], 10, 1.0 / a.tau_inv, meta)
    p_int, counts = dwn_ref.infer_int(exp, d["Xte"])
    acc_int = float((p_int == d["yte"]).mean())
    agree = int((p_int == p_torch).sum())
    exp["meta"].update(test_acc_int=acc_int, int_vs_torch_agree=f"{agree}/{len(p_int)}")
    sd = os.path.join(a.out, a.arch_name, f"seed{a.seed}")
    if os.path.exists(os.path.join(sd, "model.json")):
        raise SystemExit(f"{sd}/model.json exists - not overwriting")
    os.makedirs(sd, exist_ok=True)
    dwn_ref.save_model(exp, os.path.join(sd, "model.json"))
    np.savez_compressed(os.path.join(sd, "test_scores.npz"), counts=counts, pred=p_int,
                        y=d["yte"], torch_scores=s.numpy())
    row = {"label": "", "arch": a.arch_name, "layers": "x".join(map(str, widths)), "n": a.n,
           "tau_inv": a.tau_inv, "seed": a.seed, "epochs": len(ck.get("history", [])) if isinstance(ck, dict) else "",
           "efd_backend": "imported checkpoint", "test_acc_torch": round(float((p_torch == d['yte']).mean()), 6),
           "test_acc_int": round(acc_int, 6), "int_vs_torch_agree": f"{agree}/{len(p_int)}",
           "train_s": round(ck.get("elapsed_s", 0), 1) if isinstance(ck, dict) else "",
           "device": "checkpoint", "torch": torch.__version__, "model_sha256": exp["model_sha256"]}
    acc_csv = os.path.join(a.out, a.arch_name, "accuracy.csv")
    new = not os.path.exists(acc_csv)
    with open(acc_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
    print(json.dumps(row))
    rep = meta["ckpt_reported_final_test_acc"]
    if rep is not None and abs(rep - acc_int) > 1e-9:
        print(f"[WARN] checkpoint reports final test_acc {rep} but exported integer model gives {acc_int}"
              " (check tie handling / config)")


if __name__ == "__main__":
    main()
