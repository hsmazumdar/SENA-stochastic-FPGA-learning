"""run_twin_full.py SEED [N_HID SH] -- bit-exact twin (hardware rule v1) on SENA's 10k/2.5k split,
40 epochs x 3,000 samples (same order as the earlier software estimates), then all 2,500 test images
with T=40 passes (votes also read out after 1 and 8 passes of the same run). FINAL weights, no snapshot.
TWIN / SOFTWARE RESULT -- not a board measurement."""
import sys, json, time, numpy as np
import sena_hw_twin as tw
seed = int(sys.argv[1]); n_hid = int(sys.argv[2]) if len(sys.argv) > 2 else 160; sh = int(sys.argv[3]) if len(sys.argv) > 3 else 7
d = np.load('/workspace/v13/sena_ref/mnist_bin_t0.3_tr10000_te2500_s2026.npz')
Xtr, ytr, Xte, yte = d['Xtr'], d['ytr'], d['Xte'], d['yte']
W1, W2 = tw.init_weights(784, n_hid, 10, seed)
t = tw.HWTwin(W1, W2, tw.seed_in_for(seed), sh)
t0 = time.time(); ep_log = []
for ep, idx in enumerate(tw.schedule(len(Xtr), seed)):
    for i in idx:
        t.train(Xtr[i], int(ytr[i]))
    ep_log.append(dict(epoch=ep + 1, nb_fx=int(t.nb_fx), n_w1_updates=int(t.n_w1_updates),
                       w1_sat=int((np.abs(t.W1) == 127).sum()), w2_sat=int((np.abs(t.W2) == 127).sum())))
    print(ep_log[-1], round(time.time() - t0), flush=True)
t_train = time.time() - t0
preds = []; p1 = []; p8 = []
for i in range(len(Xte)):
    p, s = t.infer(Xte[i], 40, record=(1, 8))
    preds.append(p); p1.append(s[1]); p8.append(s[8])
preds, p1, p8 = map(np.array, (preds, p1, p8))
res = dict(label='TWIN (bit-exact software model of RTL rule v1) - NOT a board measurement',
           seed=seed, n_hid=n_hid, sh=sh, weight_bits=8, schedule='40 epochs x 3000 samples, final weights',
           test_acc_T40=float(np.mean(preds == yte)), test_acc_T8_readout=float(np.mean(p8 == yte)),
           test_acc_T1_readout=float(np.mean(p1 == yte)), n_test=int(len(yte)),
           n_w1_updates=int(t.n_w1_updates), nb_fx_final=int(t.nb_fx), train_wall_s=round(t_train, 1),
           total_wall_s=round(time.time() - t0, 1), epochs=ep_log)
np.savez_compressed(f'../results/twin_full_h{n_hid}_s{seed}_weights.npz', W1=t.W1, W2=t.W2, preds=preds,
                    LH=t.LH.astype(np.uint64), LO=t.LO.astype(np.uint64), nb_fx=t.nb_fx)
json.dump(res, open(f'../results/twin_full_h{n_hid}_s{seed}.json', 'w'), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != 'epochs'}))
