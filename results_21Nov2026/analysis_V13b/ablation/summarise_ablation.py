"""Summarise runs2/*.json: per (rule, update_rate) mean+-sd over seeds; select update_rate per rule on the
800-image probe (validation) mean accuracy; report test accuracy of the selected setting."""
import json, glob, csv, collections, numpy as np
R = [json.load(open(f)) for f in sorted(glob.glob('runs2/*.json'))]
G = collections.defaultdict(list)
for r in R: G[(r['rule'], r['update_rate'])].append(r)
rows = []
for (rule, p), rs in sorted(G.items()):
    te = np.array([r['test_acc_T40'] for r in rs]) * 100; va = np.array([r['val_acc_probe800_T40'] for r in rs]) * 100
    t1 = np.array([r['test_acc_T1'] for r in rs]) * 100
    rows.append(dict(rule=rule, ste_window=rs[0]['ste_window'], update_rate=p, n_seeds=len(rs), seeds=' '.join(str(r['seed']) for r in sorted(rs, key=lambda r: r['seed'])),
                     val_mean=va.mean(), val_sd=va.std(ddof=1) if len(rs) > 1 else np.nan,
                     test_T40_mean=te.mean(), test_T40_sd=te.std(ddof=1) if len(rs) > 1 else np.nan,
                     test_T1_mean=t1.mean(), test_T1_sd=t1.std(ddof=1) if len(rs) > 1 else np.nan,
                     w1_update_samples_mean=np.mean([r['n_w1_update_samples'] for r in rs]),
                     w1_saturated_pct_mean=100 * np.mean([r['w1_saturated_frac'] for r in rs]),
                     wall_s_mean=np.mean([r['wall_s'] for r in rs])))
best = {}
for r in rows:
    if r['n_seeds'] < 3: continue
    if r['rule'] not in best or r['val_mean'] > best[r['rule']]['val_mean']: best[r['rule']] = r
for r in rows: r['selected_on_validation'] = best.get(r['rule']) is r
keys = list(rows[0].keys())
with open('V13b_ablation_all_settings.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
    for r in rows: w.writerow({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
order = ['sena', 'fa', 'signsgd', 'drtp', 'outonly']
with open('V13b_ablation_summary.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
    for k in order:
        if k in best: w.writerow({kk: (round(v, 3) if isinstance(v, float) else v) for kk, v in best[k].items()})
for k in order:
    if k in best:
        b = best[k]; print(f"{k:8s} p={b['update_rate']:<10g} val {b['val_mean']:.2f}  test {b['test_T40_mean']:.2f} ± {b['test_T40_sd']:.2f}  T1 {b['test_T1_mean']:.2f}  W1upd {b['w1_update_samples_mean']:.0f} sat {b['w1_saturated_pct_mean']:.2f}%")
for r in rows: print(r['rule'], r['update_rate'], r['n_seeds'], round(r['val_mean'], 2), round(r['test_T40_mean'], 2), r['test_T40_sd'] if r['n_seeds']<2 else round(r['test_T40_sd'], 2))
