"""SELF-TEST: write a checkpoint in the format of baseline_dwn/train_sena_dwn_sm.py
from an OFFICIAL torch_dwn model (random weights, repo src passed as argv[1]),
convert it with export_checkpoint.py and compare with the official forward."""
import sys, os, subprocess, json, tempfile
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root); sys.path.insert(0, sys.argv[1])
import numpy as np, torch, dwn_ref
import torch_dwn as dwn, torch_dwn.lut_layer as LL
LL.EFDFunction = dwn_ref.torch_parts()["EFDTorch"]
torch.manual_seed(5)
m = torch.nn.Sequential(dwn.LUTLayer(784, 1000, n=6, mapping="learnable"), dwn.LUTLayer(1000, 500, n=6),
                        dwn.GroupSum(10, tau=1/0.245)).eval()
d = np.load(os.path.join(root, "data/mnist_bin_t0.3_tr10000_te2500_s2026.npz"))
with torch.no_grad():
    p = m(torch.tensor(d["Xte"], dtype=torch.float32)).argmax(1).numpy()
tmp = tempfile.mkdtemp()
ck = os.path.join(tmp, "dwn_sm_sena_seed0.pt")
torch.save({"seed": 0, "state_dict": m.state_dict(), "test_acc": float((p == d["yte"]).mean()),
            "history": [{}] * 100, "config": "Table13_sm_n6_1000_500"}, ck)
r = subprocess.run([sys.executable, "export_checkpoint.py", ck, "--seed", "0", "--out", os.path.join(tmp, "runs")],
                   cwd=root, capture_output=True, text=True)
print(r.stdout.strip()[-300:], r.stderr.strip()[-500:])
exp = dwn_ref.load_model(os.path.join(tmp, "runs/sm_ckpt/seed0/model.json"))
pi, _ = dwn_ref.infer_int(exp, d["Xte"])
ok = (pi == p).all() and "WARN" not in r.stdout
print("checkpoint import reproduces official model predictions:", bool(ok))
assert ok
print("SELFTEST export_checkpoint: PASS")
