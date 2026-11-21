"""SELF-TEST: build a model from the OFFICIAL torch_dwn classes (source tree of
github.com/alanbacellar/DWN, commit 2b456acf, passed as argv[1]), run its
forward on CPU by substituting the EFD op with dwn_ref.EFDTorch (forward is a
pure table lookup), and check that dwn_ref.export_model + infer_int reproduce
the official module's predictions and class scores exactly. Random weights;
not a result of any kind."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, sys.argv[1])
import numpy as np, torch
import dwn_ref
import torch_dwn as dwn
import torch_dwn.lut_layer as LL
LL.EFDFunction = dwn_ref.torch_parts()["EFDTorch"]
torch.manual_seed(0)
d = np.load("data/mnist_bin_t0.3_tr10000_te2500_s2026.npz")
X = torch.tensor(d["Xte"][:500], dtype=torch.float32)
model = torch.nn.Sequential(dwn.LUTLayer(784, 130, n=6, mapping="learnable"),
                            dwn.LUTLayer(130, 47, n=6), dwn.GroupSum(10, tau=1/0.245))
model.eval()
with torch.no_grad():
    s = model(X)
exp = dwn_ref.export_model(model, 784, 10, 1/0.245)
p, c = dwn_ref.infer_int(exp, d["Xte"][:500])
same_pred = (p == s.argmax(1).numpy()).all()
same_scores = np.allclose(c / (1/0.245), s.numpy(), atol=1e-5)
print("official-layer export: preds equal", bool(same_pred), "| scores equal", bool(same_scores),
      "| (47 outputs -> padded GroupSum, group size", dwn_ref.group_slices(47, 10)[0], ")")
assert same_pred and same_scores
print("SELFTEST export of official torch_dwn layers: PASS")
