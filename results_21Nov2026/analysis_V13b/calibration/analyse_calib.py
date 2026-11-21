"""V13b calibration analysis: risk-coverage, ECE (15 equal-width bins), Brier, reliability diagrams.
Inputs: sena_vote_trajectories_fixedN_r0-4.npz (made by calib_sena.py on the box, 20 Nov 2026) and
DWN test_scores.npz seeds 0-4 (Baseline_DWN_pkg/runs/sm). Confidence definitions:
  SENA c = max_k V_k / max(sum_k V_k, 1) (vote fraction, as in hardware); prob vector p = V / sum V
       (uniform 0.1 if no votes). full = votes after T=40; pass1 = after 1 pass; gated = state at the
       first pass t<=40 with c_t >= 0.55 (else t=40), ranked by (c_exit desc, passes_used asc).
  DWN  ranking by top1-top2 popcount margin (as collect_results.py); calibration on softmax(torch_scores).
Brier = mean_i sum_k (p_ik - 1[y_i=k])^2 (multi-class, range 0..2)."""
import json, csv, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
NB = 15; COVS = [1.0, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5]; THETA = 0.55
D = np.load('sena_vote_trajectories_fixedN_r0-4.npz'); y = D['y']

def probs(V):
    s = V.sum(1, keepdims=True).astype(float)
    return np.where(s > 0, V / np.maximum(s, 1), 0.1)
def conf_of(V):
    return V.max(1) / np.maximum(V.sum(1), 1)
def ece(conf, correct, nb=NB):
    e = 0.0; rows = []
    edges = np.linspace(0, 1, nb + 1)
    for b in range(nb):
        m = (conf > edges[b]) & (conf <= edges[b + 1]) if b else (conf >= 0) & (conf <= edges[1])
        if m.sum():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
            rows.append((edges[b], edges[b + 1], int(m.sum()), float(conf[m].mean()), float(correct[m].mean())))
        else:
            rows.append((edges[b], edges[b + 1], 0, np.nan, np.nan))
    return e, rows
def brier(P, y):
    Y = np.eye(P.shape[1])[y]; return float(((P - Y) ** 2).sum(1).mean())
def rc(order, correct, covs=COVS):
    n = len(order); out = {}
    for c in covs:
        k = int(round(c * n)); out[c] = float(correct[order[:k]].mean())
    return out
def rc_full_curve(order, correct):
    cs = np.cumsum(correct[order]); k = np.arange(1, len(order) + 1)
    return k / len(order), 1 - cs / k
def aurc(order, correct):
    cov, risk = rc_full_curve(order, correct); return float(risk.mean())

results = []; curves = {}; rel = {}
for r in range(5):
    V = D[f'r{r}'].astype(np.int64)  # (40, n, 10)
    n = V.shape[1]
    # full budget
    Vf = V[-1]; pf = Vf.argmax(1); cf = conf_of(Vf); okf = (pf == y).astype(float)
    # pass 1
    V1 = V[0]; p1 = V1.argmax(1); c1 = conf_of(V1); ok1 = (p1 == y).astype(float)
    # gated
    C = np.stack([conf_of(V[t]) for t in range(40)])  # (40,n)
    hit = C >= THETA; tex = np.where(hit.any(0), hit.argmax(0), 39)
    Vg = V[tex, np.arange(n)]; pg = Vg.argmax(1); cg = conf_of(Vg); okg = (pg == y).astype(float)
    passes = tex + 1
    for name, Vx, cx, okx, order in [
        ('SENA full-budget (T=40)', Vf, cf, okf, np.argsort(-cf, kind='stable')),
        ('SENA gated (theta=0.55, Tmax=40)', Vg, cg, okg, np.lexsort((passes, -cg))),
        ('SENA single pass (T=1)', V1, c1, ok1, np.argsort(-c1, kind='stable'))]:
        e, rows = ece(cx, okx); b = brier(probs(Vx), y); cov = rc(order, okx)
        results.append(dict(model=name, run=f'repeat{r}', acc=float(okx.mean()), ece15=e, brier=b,
                            aurc=aurc(order, okx), mean_conf=float(cx.mean()),
                            **{f'acc_at_{int(c*100)}cov': v for c, v in cov.items()},
                            exit_pass1_frac=float((passes == 1).mean()) if 'gated' in name else np.nan,
                            acc_on_pass1_exits=float(okg[passes == 1].mean()) if 'gated' in name else np.nan,
                            mean_passes=float(passes.mean()) if 'gated' in name else (40.0 if 'T=40' in name else 1.0),
                            gated_vs_full_disagree=float((pg != pf).mean()) if 'gated' in name else np.nan))
        curves.setdefault(name, []).append(rc_full_curve(order, okx)); rel.setdefault(name, []).append((cx, okx))
for s in range(5):
    Z = np.load(f'/workspace/v13b/in/dwn/seed{s}_test_scores.npz')
    cnt = Z['counts']; yy = Z['y']; pr = Z['pred']; ok = (pr == yy).astype(float)
    ss = np.sort(cnt, 1); margin = ss[:, -1] - ss[:, -2]; order = np.argsort(-margin, kind='stable')
    t = Z['torch_scores'].astype(float); t = t - t.max(1, keepdims=True); P = np.exp(t); P /= P.sum(1, keepdims=True)
    cx = P.max(1); e, _ = ece(cx, ok); cov = rc(order, ok)
    results.append(dict(model='DWN sm (margin rank; softmax calib.)', run=f'seed{s}', acc=float(ok.mean()), ece15=e,
                        brier=brier(P, yy), aurc=aurc(order, ok), mean_conf=float(cx.mean()),
                        **{f'acc_at_{int(c*100)}cov': v for c, v in cov.items()},
                        exit_pass1_frac=np.nan, acc_on_pass1_exits=np.nan, mean_passes=np.nan, gated_vs_full_disagree=np.nan))
    curves.setdefault('DWN sm (margin rank; softmax calib.)', []).append(rc_full_curve(order, ok))
    rel.setdefault('DWN sm (margin rank; softmax calib.)', []).append((cx, ok))

keys = [k for k in results[0].keys() if k not in ('model', 'run')]
with open('V13b_calibration_per_run.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['model', 'run'] + keys); w.writeheader()
    for r in results: w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})
summary = []
for m in dict.fromkeys(r['model'] for r in results):
    rs = [r for r in results if r['model'] == m]; row = dict(model=m, n_runs=len(rs))
    for k in keys:
        v = np.array([r[k] for r in rs], float)
        if np.all(np.isnan(v)): continue
        row[k + '_mean'] = float(np.mean(v)); row[k + '_sd'] = float(np.std(v, ddof=1))
    summary.append(row)
allk = []
for s in summary:
    for k in s:
        if k not in allk: allk.append(k)
with open('V13b_calibration_summary.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=allk); w.writeheader()
    for s in summary: w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in s.items()})
json.dump(summary, open('V13b_calibration_summary.json', 'w'), indent=1)
# risk-coverage CSV (mean over runs, 1% grid)
grid = np.round(np.arange(0.30, 1.0001, 0.01), 2)
with open('V13b_risk_coverage_curves.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['model', 'coverage', 'risk_mean', 'risk_sd', 'acc_mean'])
    for m, cl in curves.items():
        R = np.array([[risk[int(round(g * len(cov))) - 1] for g in grid] for cov, risk in cl])
        for g, mu, sd in zip(grid, R.mean(0), R.std(0, ddof=1)): w.writerow([m, g, round(mu, 6), round(sd, 6), round(1 - mu, 6)])
# figures
cols = {'SENA full-budget (T=40)': 'C0', 'SENA gated (theta=0.55, Tmax=40)': 'C1', 'SENA single pass (T=1)': 'C2',
        'DWN sm (margin rank; softmax calib.)': 'C3'}
fig, ax = plt.subplots(figsize=(5.2, 3.8))
for m, cl in curves.items():
    R = np.array([[risk[int(round(g * len(cov))) - 1] for g in grid] for cov, risk in cl]) * 100
    ax.plot(grid * 100, R.mean(0), color=cols[m], label=m); ax.fill_between(grid * 100, R.mean(0) - R.std(0, ddof=1), R.mean(0) + R.std(0, ddof=1), color=cols[m], alpha=0.2)
ax.set_xlabel('Coverage (%)'); ax.set_ylabel('Risk = error on answered inputs (%)'); ax.set_xlim(30, 100); ax.set_ylim(0, None)
ax.grid(alpha=0.3); ax.legend(fontsize=7); ax.set_title('Risk-coverage, binarised MNIST split (mean ± sd)', fontsize=9)
fig.tight_layout(); fig.savefig('Fig_V13b_risk_coverage.png', dpi=300); fig.savefig('Fig_V13b_risk_coverage.pdf'); plt.close(fig)
fig, axs = plt.subplots(1, 4, figsize=(12, 3.2), sharey=True)
for ax, (m, lst) in zip(axs, rel.items()):
    cx = np.concatenate([a for a, b in lst]); ok = np.concatenate([b for a, b in lst])
    e, rows = ece(cx, ok); mids = [(a + b) / 2 for a, b, *_ in rows]; acc = [r[4] for r in rows]; cnt = [r[2] for r in rows]
    ax.bar(mids, [0 if np.isnan(a) else a for a in acc], width=1 / NB, edgecolor='k', color=cols[m], alpha=0.7)
    ax.plot([0, 1], [0, 1], 'k--', lw=1); ax.set_title(f'{m}\npooled ECE15={e*100:.2f}%', fontsize=7)
    ax.set_xlabel('Confidence'); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
axs[0].set_ylabel('Accuracy in bin')
fig.tight_layout(); fig.savefig('Fig_V13b_reliability.png', dpi=300); fig.savefig('Fig_V13b_reliability.pdf'); plt.close(fig)
for s in summary:
    print(s['model'], {k: round(v * (100 if 'acc' in k or 'ece' in k or 'frac' in k or 'disagree' in k else 1), 3) for k, v in s.items() if isinstance(v, float)})
# compact 2x2 reliability figure for the single-column manuscript layout
fig, axs = plt.subplots(2, 2, figsize=(5.0, 4.6), sharex=True, sharey=True)
short = {'SENA full-budget (T=40)': 'SENA full budget (T=40)', 'SENA gated (theta=0.55, Tmax=40)': 'SENA gated (θ=0.55)',
         'SENA single pass (T=1)': 'SENA single pass (T=1)', 'DWN sm (margin rank; softmax calib.)': 'DWN sm (softmax)'}
for ax, (m, lst) in zip(axs.ravel(), rel.items()):
    cx = np.concatenate([a for a, b in lst]); ok = np.concatenate([b for a, b in lst])
    e, rows = ece(cx, ok); mids = [(a + b) / 2 for a, b, *_ in rows]; acc = [r[4] for r in rows]
    ax.bar(mids, [0 if np.isnan(a) else a for a in acc], width=1 / NB, edgecolor='k', color=cols[m], alpha=0.7)
    ax.plot([0, 1], [0, 1], 'k--', lw=1); ax.set_title(f'{short[m]}\npooled ECE$_{{15}}$ = {e*100:.1f}%', fontsize=7.5)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.tick_params(labelsize=7)
for ax in axs[1]: ax.set_xlabel('Confidence', fontsize=8)
for ax in axs[:, 0]: ax.set_ylabel('Accuracy in bin', fontsize=8)
fig.tight_layout(); fig.savefig('Fig_V13b_reliability_2x2.png', dpi=300); fig.savefig('Fig_V13b_reliability_2x2.pdf'); plt.close(fig)
