# SENA DWN sm baseline — run on Colab T4 GPU
# Upload when prompted: mnist_bin_t0.3_tr10000_te2500_s2026.npz
!pip -q install torch-dwn

import json, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn.functional import cross_entropy
from google.colab import files
import torch_dwn as dwn

assert torch.cuda.is_available(), torch.cuda.get_device_name if False else 'need GPU'
print('GPU:', torch.cuda.get_device_name(0))

uploaded = files.upload()  # pick the .npz
npz_name = [k for k in uploaded if k.endswith('.npz')][0]
z = np.load(npz_name)
x_train = torch.from_numpy(z['Xtr'].astype(np.float32))
y_train = torch.from_numpy(z['ytr'].astype(np.int64))
x_test = torch.from_numpy(z['Xte'].astype(np.float32))
y_test = torch.from_numpy(z['yte'].astype(np.int64))
print(x_train.shape, x_test.shape)

def build():
    return nn.Sequential(
        dwn.LUTLayer(784, 1000, n=6, mapping='learnable'),
        dwn.LUTLayer(1000, 500, n=6),
        dwn.GroupSum(10, tau=1/0.245),
    )

def lr_for_epoch(e):
    return [1e-2, 1e-3, 1e-4, 1e-5][min(e // 30, 3)]

@torch.no_grad()
def accuracy(model, x, y, batch=512):
    model.eval()
    preds = []
    for i in range(0, len(x), batch):
        preds.append(model(x[i:i+batch].cuda()).argmax(1).cpu())
    return float((torch.cat(preds) == y).float().mean())

Path('checkpoints').mkdir(exist_ok=True)
Path('results').mkdir(exist_ok=True)
runs = []
for seed in range(5):
    torch.manual_seed(seed); np.random.seed(seed)
    model = build().cuda()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    hist = []; t0 = time.perf_counter(); n = len(x_train)
    for epoch in range(100):
        lr = lr_for_epoch(epoch)
        for g in opt.param_groups: g['lr'] = lr
        model.train(); perm = torch.randperm(n); correct = total = 0; last = 0.0
        for i in range(0, n, 128):
            idx = perm[i:i+128]
            bx, by = x_train[idx].cuda(), y_train[idx].cuda()
            opt.zero_grad(set_to_none=True)
            out = model(bx); loss = cross_entropy(out, by); loss.backward(); opt.step()
            last = float(loss.item()); correct += int((out.argmax(1) == by).sum()); total += by.size(0)
        te = accuracy(model, x_test, y_test)
        hist.append(dict(epoch=epoch+1, lr=lr, loss=last, train_acc=correct/total, test_acc=te))
        if (epoch+1) % 10 == 0 or epoch == 0:
            print(f'seed={seed} ep={epoch+1}/100 test={te:.4f}', flush=True)
    ckpt = f'checkpoints/dwn_sm_sena_seed{seed}.pt'
    torch.save(dict(seed=seed, state_dict=model.state_dict(), n_in=784,
                    test_acc=hist[-1]['test_acc'],
                    best_test_acc=max(h['test_acc'] for h in hist),
                    history=hist, elapsed_s=time.perf_counter()-t0), ckpt)
    runs.append(dict(seed=seed, test_acc=hist[-1]['test_acc'],
                     best_test_acc=max(h['test_acc'] for h in hist), checkpoint=ckpt))
    print('saved', ckpt, 'acc', runs[-1]['test_acc'])

acc = np.array([r['test_acc'] for r in runs])
summary = dict(name='dwn_sm_sena_split', runs=runs,
               test_acc_mean=float(acc.mean()),
               test_acc_std=float(acc.std(ddof=1)))
Path('results/dwn_sm_sena_train.json').write_text(json.dumps(summary, indent=2))
print('MEAN', summary['test_acc_mean'], '±', summary['test_acc_std'])
files.download('results/dwn_sm_sena_train.json')
!zip -r dwn_sena_ckpts.zip checkpoints
files.download('dwn_sena_ckpts.zip')
