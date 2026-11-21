"""V13b learning-rule ablation (software). Same 784-160-10 net, same LFSR stochastic forward pass,
same init (uniform {-2..2}), same sample order (40 epochs x 3,000 online samples, batch 1), same 8-bit
saturating integer weights (+-127) and same +-1 LSB step; only the credit-assignment rule differs.
Test: 2,500 images, T=40 passes (final weights). Rules:
  sena   : SENA hardware rule v1, unchanged (sena_hw_twin.HWTwin.train)
  signsgd: sign of the expected gradient of the stochastic neuron (uniform threshold noise in [-64,63]
           -> derivative box 1[|X|<64]): dW2 = d_k*y_j*1[|Z_k|<64]; back_j = sum_k e_k W2[k,j] with
           e_k = d_k*1[|Z_k|<64] (pre-update W2); dW1 = sgn(back_j)*1[|X_j|<64]*x_i (bias input 1)
  fa     : feedback alignment: as signsgd but back_j = sum_k e_k B[k,j], B fixed random {-1,+1}
  drtp   : direct random target projection: as fa but back_j = sum_k t_k B[k,j] (target, not error)
  outonly: control, output layer trained as signsgd, hidden layer frozen at init
STE window: 64 = exact expected derivative of the noisy threshold (box |X|<64);
'id' = identity straight-through estimator (no gate), the usual STE.
update_rate p: each output row / hidden row update is applied with probability p (software Bernoulli,
numpy rng; a learning-rate knob for the +-1 LSB rules). p is selected on the 800-image probe set that
the schedule holds out of training (never on the test set). SENA itself is run unchanged (p unused).
Usage: python ablation_rules.py RULE SEED [WIN|id] [P]"""
import sys, json, time, numpy as np
import sena_hw_twin as tw

class RuleTwin(tw.HWTwin):
    def __init__(self, *a, rule='signsgd', bseed=0, win=64, rate=1.0, **k):
        super().__init__(*a, **k); self.rule = rule; self.win = win; self.rate = rate
        self.gr = np.random.default_rng(bseed + 4242)
        rb = np.random.default_rng(bseed + 999)
        self.B = rb.choice(np.array([-1, 1]), size=(self.n_out, self.n_hid)).astype(np.int64)
        self.n_w2_upd = 0
    def train(self, x, label):
        if self.rule == 'sena':
            return super().train(x, label)
        x = np.asarray(x, dtype=np.int64)
        X = self._X(x); y = self._fire_hidden(X); Z, o = self._fire_out(y)
        t = (np.arange(self.n_out) == label).astype(np.int64)
        d = t - o
        e = d * (np.abs(Z) < self.win)
        if self.rule == 'signsgd' or self.rule == 'outonly':
            back = e @ self.W2[:, :self.n_hid]
        elif self.rule == 'fa':
            back = e @ self.B
        elif self.rule == 'drtp':
            back = t @ self.B
        yb = np.append(y, 1)
        if self.rate < 1: e = e * (self.gr.random(self.n_out) < self.rate)
        upd2 = np.outer(e, yb)
        if upd2.any(): self.n_w2_upd += 1
        self.W2 = np.clip(self.W2 + upd2, -self.wmax, self.wmax)
        if self.rule == 'outonly':
            return
        if self.rule == 'drtp':
            gate = (np.abs(X) < self.win) & (back != 0) & (d != 0).any()   # DRTP: update only on error samples
        else:
            gate = (np.abs(X) < self.win) & (back != 0)
        if self.rate < 1: gate = gate & (self.gr.random(self.n_hid) < self.rate)
        dj = np.where(gate, np.sign(back), 0)
        if dj.any():
            self.n_w1_updates += 1
            xb = np.append(x, 1)
            self.W1 = np.clip(self.W1 + np.outer(dj, xb), -self.wmax, self.wmax)

def main(rule, seed, epochs=40, win=64, rate=1.0):
    d = np.load('/workspace/v13/sena_ref/mnist_bin_t0.3_tr10000_te2500_s2026.npz')
    Xtr, ytr, Xte, yte = d['Xtr'], d['ytr'], d['Xte'], d['yte']
    W1, W2 = tw.init_weights(784, 160, 10, seed)
    m = RuleTwin(W1, W2, tw.seed_in_for(seed), 7, rule=rule, bseed=seed, win=win, rate=rate)
    probe = tw.schedule_probe(len(Xtr), seed)
    t0 = time.time()
    for ep, idx in enumerate(tw.schedule(len(Xtr), seed, epochs=epochs)):
        for i in idx: m.train(Xtr[i], int(ytr[i]))
    preds = []; p1 = []
    for i in range(len(Xte)):
        p, s = m.infer(Xte[i], 40, record=(1,)); preds.append(p); p1.append(s[1])
    preds = np.array(preds); p1 = np.array(p1)
    pv = np.array([m.infer(Xtr[i], 40)[0] for i in probe]); val = float(np.mean(pv == ytr[probe]))
    res = dict(rule=rule, ste_window=(win if win < 10**8 else 'identity'), update_rate=rate, seed=seed, val_acc_probe800_T40=val, epochs=epochs, samples_per_epoch=3000, n_hid=160, weight_bits=8,
               test_acc_T40=float(np.mean(preds == yte)), test_acc_T1=float(np.mean(p1 == yte)),
               n_w1_update_samples=int(m.n_w1_updates),
               w1_saturated_frac=float(np.mean(np.abs(m.W1) == 127)),
               w2_saturated_frac=float(np.mean(np.abs(m.W2) == 127)), wall_s=round(time.time() - t0, 1))
    json.dump(res, open(f'runs2/{rule}_w{win if win < 10**8 else "id"}_p{rate:g}_s{seed}.json', 'w'), indent=1); print(json.dumps(res), flush=True)

if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]), win=(int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] != 'id' else (10**9 if len(sys.argv) > 3 else 64)), rate=(float(sys.argv[4]) if len(sys.argv) > 4 else 1.0))
