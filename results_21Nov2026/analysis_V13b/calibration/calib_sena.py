"""V13b calibration / selective prediction for SENA (fixed-range M=3, 16-bit) and DWN.
Run on the box 20 Nov 2026. Checkpoints: q1_ablation_full_r{0..4}_fixedN.npz (restored_results).
Split: mnist_bin_t0.3_tr10000_te2500_s2026.npz. Inference: SENAEnsemble code (senarepo/sena2026/sim),
40 passes over ALL test inputs, cumulative votes recorded per pass; the gated (theta=0.55) decision is
read from the same vote trajectory (first pass t with c_t >= 0.55, else t=40)."""
import sys, json, csv, numpy as np
sys.path.insert(0, '/workspace/senarepo/sena2026/sim')
from sena_ensemble import SENAEnsemble
CK='/workspace/senarepo/sena2026/results/q1_ablation_full_r{}_fixedN.npz'
d=np.load('/workspace/v13/sena_ref/mnist_bin_t0.3_tr10000_te2500_s2026.npz'); X=d['Xte']; y=d['yte']
TMAX=40; THETA=0.55
out={}
for r in range(5):
    ens,_=SENAEnsemble.load(CK.format(r))
    n=len(X); V=np.zeros((TMAX,n,10),dtype=np.int32); votes=np.zeros((n,10),dtype=np.int64)
    for t in range(TMAX):
        for s in range(0,n,256):
            xb=X[s:s+256]
            for net in ens.nets:
                acts,_=net.forward(xb,stochastic=True); votes[s:s+256]+=acts[-1]
        V[t]=votes
    out[f'r{r}']=V; print('repeat',r,'acc@40',np.mean(V[-1].argmax(1)==y),flush=True)
np.savez_compressed('sena_vote_trajectories_fixedN_r0-4.npz',y=y,**out)
