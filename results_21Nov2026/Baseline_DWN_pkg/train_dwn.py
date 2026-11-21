"""
train_dwn.py - train DWN on SENA's exact frozen split, 5 seeds, export LUTs.

Default = the published small MNIST FPGA model "DWN (n=6; sm)" of Bacellar et
al., ICML 2024, Table 13:  LUT layers 1000 (learnable mapping) and 500 (random
mapping), n = 6, GroupSum(10) with tau = 1/0.245, batch 128, 100 epochs,
Adam, lr 1e-2 (30 ep), 1e-3 (30), 1e-4 (30), 1e-5 (10).
Input = SENA's 784 binary pixels (threshold 0.3) used directly as z = 1
thermometer bits (no re-fitted thermometer; the bits are bit-identical to the
data SENA was trained/tested on).

No early stopping / no model selection on the test set: the model after the
last epoch is exported and evaluated once.

Examples
  python train_dwn.py --arch sm --seeds 0 1 2 3 4            # paper config, CUDA ext
  python train_dwn.py --arch sm --seeds 0 1 2 3 4 --efd torch # no-compile fallback
  python train_dwn.py --check-efd                            # ext vs fallback agreement
Outputs (runs/<arch>/)
  seed<k>/model.json         LUT tables + mappings  (input to dwn_to_verilog.py)
  seed<k>/test_scores.npz    per-test-sample class popcounts, preds (for coverage curve)
  seed<k>/epoch_log.csv      per-epoch loss / train acc / test acc
  seed<k>/model_state.pt     PyTorch weights (for re-export)
  accuracy.csv               one row per seed (appended)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import sys
import time

import numpy as np
import torch
from torch.nn.functional import cross_entropy

import dwn_ref

PRESETS = {
    # name: (layers, n, tau_inv, learnable-first-layer, source)
    "sm": ([1000, 500], 6, 0.245, True, "DWN paper Table 13, MNIST 'DWN (n=6; sm)', z=1"),
    "lg_z1": ([2000, 1000], 6, 0.173, True,
              "DWN paper Table 13 'DWN (n=6; lg)' layer sizes/tau, but z=1 (SENA 1-bit input) - our variant"),
    "selftest_tiny": ([120, 50], 6, 0.245, True, "SELF-TEST ONLY (pipeline check, not a result)"),
}


def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def build(layers, n, tau, learnable, n_in, n_classes, efd, device):
    if efd == "ext":
        import torch_dwn as dwn  # official package incl. compiled efd_cuda
        LUT, GS = dwn.LUTLayer, dwn.GroupSum
    else:
        parts = dwn_ref.torch_parts()
        LUT, GS = parts["LUTLayerTorch"], parts["GroupSum"]
    mods, prev = [], n_in
    for i, w in enumerate(layers):
        mods.append(LUT(prev, w, n=n, mapping="learnable" if (i == 0 and learnable) else "random"))
        prev = w
    mods.append(GS(n_classes, tau=tau))
    return torch.nn.Sequential(*mods).to(device)


def resolve_efd(efd, device):
    if efd != "auto":
        return efd
    if device.startswith("cuda"):
        try:
            import efd_cuda  # noqa: F401
            import torch_dwn  # noqa: F401
            return "ext"
        except Exception as e:
            print("[warn] official CUDA extension not importable (%s); using --efd torch" % e)
    return "torch"


@torch.no_grad()
def evaluate(model, X, y, bs=2500):
    model.eval()
    preds, scores = [], []
    for i in range(0, len(X), bs):
        s = model(X[i:i + bs])
        scores.append(s.float().cpu())
        preds.append(s.argmax(1).cpu())
    p = torch.cat(preds)
    return (p == y.cpu()).float().mean().item(), p.numpy(), torch.cat(scores).numpy()


def check_efd():
    """Compare official CUDA ext with the pure-PyTorch fallback on random data."""
    assert torch.cuda.is_available(), "needs CUDA"
    import efd_cuda
    parts = dwn_ref.torch_parts()
    torch.manual_seed(0)
    x = (torch.rand(256, 784, device="cuda") > 0.5).float()
    mapping = torch.randint(0, 784, (300, 6), device="cuda", dtype=torch.int32)
    luts = (torch.rand(300, 64, device="cuda") * 2 - 1)
    alpha, beta = torch.tensor(0.5 * 0.75 ** 5), torch.tensor(0.25 / 0.75)
    g = torch.randn(256, 300, device="cuda")
    o1 = efd_cuda.forward(x, mapping, luts)
    gi1, gl1 = efd_cuda.backward(x, mapping, luts, float(alpha), float(beta), g)
    xt = x.clone().requires_grad_(True)
    lt = luts.clone().requires_grad_(True)
    o2 = parts["EFDTorch"].apply(xt, mapping, lt, alpha, beta)
    o2.backward(g)
    r = {"forward_maxabs": (o1 - o2).abs().max().item(),
         "input_grad_maxabs": (gi1 - xt.grad).abs().max().item(),
         "lut_grad_maxabs": (gl1 - lt.grad).abs().max().item()}
    print(json.dumps(r, indent=1))
    ok = r["forward_maxabs"] == 0 and r["input_grad_maxabs"] < 1e-3 and r["lut_grad_maxabs"] < 1e-3
    print("EFD fallback agrees with official CUDA ext:", ok)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/mnist_bin_t0.3_tr10000_te2500_s2026.npz")
    ap.add_argument("--arch", default="sm", choices=list(PRESETS) + ["custom"])
    ap.add_argument("--layers", default=None, help="custom: e.g. 1000,500")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--tau-inv", type=float, default=None, help="GroupSum tau = 1/tau_inv")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr-steps", default="1e-2:30,1e-3:30,1e-4:30,1e-5:10",
                    help="lr:epochs,... (paper default = 100 epochs)")
    ap.add_argument("--efd", default="auto", choices=["auto", "ext", "torch"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--label", default="", help="free text stored in the CSV (e.g. SELFTEST)")
    ap.add_argument("--check-efd", action="store_true")
    a = ap.parse_args()

    if a.check_efd:
        sys.exit(0 if check_efd() else 1)

    if a.arch == "custom":
        layers = [int(v) for v in a.layers.split(",")]
        n, tau_inv, learnable, src = a.n or 6, a.tau_inv or 0.245, True, "custom"
    else:
        layers, n, tau_inv, learnable, src = PRESETS[a.arch]
        n = a.n or n
        tau_inv = a.tau_inv or tau_inv
    tau = 1.0 / tau_inv
    steps = [(float(s.split(":")[0]), int(s.split(":")[1])) for s in a.lr_steps.split(",")]
    epochs = sum(e for _, e in steps)
    efd = resolve_efd(a.efd, a.device)
    if a.arch == "selftest_tiny" and not a.label:
        a.label = "SELFTEST"

    d = np.load(a.data)
    data_sha = hashlib.sha256(np.ascontiguousarray(d["Xte"]).tobytes()
                              + np.ascontiguousarray(d["Xtr"]).tobytes()).hexdigest()
    Xtr = torch.tensor(d["Xtr"], dtype=torch.float32, device=a.device)
    ytr = torch.tensor(d["ytr"], dtype=torch.long, device=a.device)
    Xte = torch.tensor(d["Xte"], dtype=torch.float32, device=a.device)
    yte = torch.tensor(d["yte"], dtype=torch.long, device=a.device)
    n_in, n_classes = Xtr.shape[1], 10

    gpu = torch.cuda.get_device_name(0) if a.device.startswith("cuda") else "cpu"
    env = {"python": platform.python_version(), "torch": torch.__version__,
           "cuda": torch.version.cuda, "device": gpu, "efd_backend": efd,
           "os": platform.platform()}
    print(json.dumps(env), "| arch", a.arch, layers, "n", n, "tau 1/%g" % tau_inv, "epochs", epochs)

    run_dir = os.path.join(a.out, a.arch)
    os.makedirs(run_dir, exist_ok=True)
    acc_csv = os.path.join(run_dir, "accuracy.csv")
    new_csv = not os.path.exists(acc_csv)

    for seed in a.seeds:
        sd = os.path.join(run_dir, f"seed{seed}")
        if os.path.exists(os.path.join(sd, "model.json")):
            print(f"[skip] {sd}/model.json exists (never overwritten); delete folder to retrain")
            continue
        os.makedirs(sd, exist_ok=True)
        seed_all(seed)
        model = build(layers, n, tau, learnable, n_in, n_classes, efd, a.device)
        opt = torch.optim.Adam(model.parameters(), lr=steps[0][0])
        t0 = time.time()
        with open(os.path.join(sd, "epoch_log.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "lr", "train_loss", "train_acc", "test_acc", "elapsed_s"])
            ep = 0
            for lr, ne in steps:
                for g in opt.param_groups:
                    g["lr"] = lr
                for _ in range(ne):
                    model.train()
                    perm = torch.randperm(len(Xtr), device=a.device)
                    tl, tc = 0.0, 0
                    for i in range(0, len(Xtr), a.bs):
                        idx = perm[i:i + a.bs]
                        opt.zero_grad()
                        out = model(Xtr[idx])
                        loss = cross_entropy(out, ytr[idx])
                        loss.backward()
                        opt.step()
                        tl += loss.item() * len(idx)
                        tc += (out.argmax(1) == ytr[idx]).sum().item()
                    ep += 1
                    te_acc, _, _ = evaluate(model, Xte, yte)
                    w.writerow([ep, lr, tl / len(Xtr), tc / len(Xtr), te_acc, round(time.time() - t0, 2)])
                    f.flush()
                    print(f"seed {seed} ep {ep}/{epochs} lr {lr:g} loss {tl/len(Xtr):.4f} "
                          f"train {tc/len(Xtr):.4f} test {te_acc:.4f}")
        train_s = time.time() - t0
        acc_torch, p_torch, s_torch = evaluate(model, Xte, yte)

        meta = {"arch": a.arch, "arch_source": src, "layers": layers, "n": n,
                "tau_inv": tau_inv, "seed": seed, "epochs": epochs, "lr_steps": a.lr_steps,
                "batch": a.bs, "data": os.path.basename(a.data), "data_sha256": data_sha,
                "env": env, "label": a.label}
        exp = dwn_ref.export_model(model, n_in, n_classes, tau, meta)
        p_int, counts = dwn_ref.infer_int(exp, d["Xte"])
        acc_int = float((p_int == d["yte"]).mean())
        agree = int((p_int == p_torch).sum())
        exp["meta"]["test_acc_torch"] = acc_torch
        exp["meta"]["test_acc_int"] = acc_int
        exp["meta"]["int_vs_torch_agree"] = f"{agree}/{len(p_int)}"
        dwn_ref.save_model(exp, os.path.join(sd, "model.json"))
        torch.save(model.state_dict(), os.path.join(sd, "model_state.pt"))
        np.savez_compressed(os.path.join(sd, "test_scores.npz"), counts=counts,
                            pred=p_int, y=d["yte"], torch_scores=s_torch)
        if agree != len(p_int):
            print(f"[ERROR] integer export disagrees with PyTorch on {len(p_int)-agree} samples")
        row = {"label": a.label, "arch": a.arch, "layers": "x".join(map(str, layers)), "n": n,
               "tau_inv": tau_inv, "seed": seed, "epochs": epochs, "efd_backend": efd,
               "test_acc_torch": round(acc_torch, 6), "test_acc_int": round(acc_int, 6),
               "int_vs_torch_agree": f"{agree}/{len(p_int)}", "train_s": round(train_s, 1),
               "device": gpu, "torch": torch.__version__, "model_sha256": exp["model_sha256"]}
        with open(acc_csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new_csv:
                w.writeheader()
                new_csv = False
            w.writerow(row)
        print(json.dumps(row))


if __name__ == "__main__":
    main()
