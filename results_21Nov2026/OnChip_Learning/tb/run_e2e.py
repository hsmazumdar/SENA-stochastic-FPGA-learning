"""
run_e2e.py -- build a vector set with the twin, simulate sena_learn_core in Icarus, compare bit-for-bit.
usage: python3 run_e2e.py NAME N_HID SH SEED N_TRAIN N_INFER T [init=normal|extreme]
"""
import sys, os, subprocess, json, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'twin'))
import sena_hw_twin as tw

DATA = '/workspace/v13/sena_ref/mnist_bin_t0.3_tr10000_te2500_s2026.npz'
IV = '/workspace/eda/bin'
RTL = os.environ.get('RTL_DIR', os.path.join(ROOT, 'rtl'))


def row_hex(vals, w=8):
    word = 0
    for p, v in enumerate(vals):
        word |= (int(v) & ((1 << w) - 1)) << (p * w)
    return format(word, '0{}x'.format((len(vals) * w + 3) // 4))


def bits_hex(bits):
    word = 0
    for i, b in enumerate(bits):
        if b: word |= 1 << i
    return format(word, '0{}x'.format((len(bits) + 3) // 4))


def read_mem(path, n_lanes, w=8):
    rows = []
    for ln in open(path):
        ln = ln.strip()
        if not ln or ln.startswith('//'): continue
        v = int(ln, 16)
        lanes = [(v >> (p * w)) & ((1 << w) - 1) for p in range(n_lanes)]
        rows.append([x - (1 << w) if x >= (1 << (w - 1)) else x for x in lanes])
    return np.array(rows, dtype=np.int64)


def main():
    name, n_hid, sh, seed, n_tr, n_inf, T = sys.argv[1], *map(int, sys.argv[2:8])
    init_mode = sys.argv[8] if len(sys.argv) > 8 else 'normal'
    out = os.path.join(ROOT, 'vectors', name); os.makedirs(out, exist_ok=True)
    d = np.load(DATA)
    Xtr, ytr, Xte, yte = d['Xtr'], d['ytr'], d['Xte'], d['yte']
    if n_tr < 0:   # FULL schedule: 40 epochs x 3000 from the 10k pool, then n_inf test images
        order = np.concatenate(tw.schedule(len(Xtr), seed))
        xs = [Xtr[i] for i in range(len(Xtr))] + [Xte[i] for i in range(n_inf)]
        labs = [int(v) for v in ytr] + [int(yte[i]) for i in range(n_inf)]
        cmds = [int(i) for i in order] + [0x8000 | (len(Xtr) + i) for i in range(n_inf)]
        n_tr = len(order)
    else:
        order = tw.schedule(len(Xtr), seed)[0][:n_tr]
        xs = [Xtr[i] for i in order] + [Xte[i] for i in range(n_inf)]
        labs = [int(ytr[i]) for i in order] + [int(yte[i]) for i in range(n_inf)]
        cmds = [i for i in range(n_tr)] + [0x8000 | (n_tr + i) for i in range(n_inf)]
    W1, W2 = tw.init_weights(784, n_hid, 10, seed)
    if init_mode == 'extreme':   # stress saturation: weights at/near +-127
        r = np.random.default_rng(seed + 999)
        choices = np.array([-127, -126, -1, 0, 1, 126, 127])
        W1 = r.choice(choices, size=W1.shape); W2 = r.choice(choices, size=W2.shape)
    seed_in = tw.seed_in_for(seed)
    # files
    open(f'{out}/x.hex', 'w').write('\n'.join(bits_hex(x) for x in xs) + '\n')
    open(f'{out}/lab.hex', 'w').write('\n'.join('%x' % l for l in labs) + '\n')
    open(f'{out}/cmd.hex', 'w').write('\n'.join('%04x' % c for c in cmds) + '\n')
    open(f'{out}/w1_init.hex', 'w').write('\n'.join(row_hex(W1[:, i]) for i in range(785)) + '\n')
    open(f'{out}/w2_init.hex', 'w').write('\n'.join(row_hex(W2[:, j]) for j in range(n_hid + 1)) + '\n')
    # twin
    t0 = time.time()
    twin = tw.HWTwin(W1, W2, seed_in, sh)
    preds = []
    for c in cmds:
        i = c & 0x7FFF
        if c & 0x8000:
            p, _ = twin.infer(xs[i], T); preds.append((i, p))
        else:
            twin.train(xs[i], labs[i])
    t_twin = time.time() - t0
    # params + sim
    P = dict(N_IN=784, N_HID=n_hid, SH=sh, NX=len(xs), NCMD=len(cmds), SEED_IN="32'h%08x" % seed_in, T_PASSES=T)
    with open(f'{out}/tb_params.vh', 'w') as f:
        for k, v in P.items(): f.write(f'`define {k} {v}\n')
        for k, fn in [('W1_INIT', 'w1_init.hex'), ('W2_INIT', 'w2_init.hex'), ('X_FILE', 'x.hex'), ('LAB_FILE', 'lab.hex'),
                      ('CMD_FILE', 'cmd.hex'), ('OUT_PRED', 'rtl_preds.txt'), ('OUT_W1', 'rtl_w1.hex'),
                      ('OUT_W2', 'rtl_w2.hex'), ('OUT_STATE', 'rtl_state.txt')]:
            f.write(f'`define {k} "{out}/{fn}"\n')
    env = dict(os.environ, PATH=IV + ':' + os.environ['PATH'])
    subprocess.run(['iverilog', '-g2012', '-I', out, '-o', f'{out}/sim.vvp', os.path.join(HERE, 'tb_e2e.v'),
                    os.path.join(RTL, 'sdp_ram.v'), os.path.join(RTL, 'sena_learn_core.v')],
                   check=True, env=env)
    t0 = time.time()
    r = subprocess.run(['vvp', '-n', f'{out}/sim.vvp'], capture_output=True, text=True, env=env)
    t_sim = time.time() - t0
    open(f'{out}/sim_log.txt', 'w').write(r.stdout + r.stderr)
    ok_sim = 'TB DONE' in r.stdout
    res = dict(name=name, rtl_dir=RTL, n_hid=n_hid, sh=sh, seed=seed, init=init_mode, n_train=n_tr, n_infer=n_inf, T=T,
               seed_in=hex(seed_in), sim_completed=ok_sim, twin_wall_s=round(t_twin, 2), icarus_wall_s=round(t_sim, 1))
    if ok_sim:
        rw1 = read_mem(f'{out}/rtl_w1.hex', n_hid)            # rows = inputs
        rw2 = read_mem(f'{out}/rtl_w2.hex', 10)
        st = dict(l.split() for l in open(f'{out}/rtl_state.txt'))
        rp = [tuple(map(int, l.split())) for l in open(f'{out}/rtl_preds.txt') if l.strip()]
        res.update(
            w1_mismatch=int((rw1.T != twin.W1).sum()), w1_total=int(twin.W1.size),
            w2_mismatch=int((rw2.T != twin.W2).sum()), w2_total=int(twin.W2.size),
            w1_changed_from_init=int((twin.W1 != W1).sum()), w2_changed_from_init=int((twin.W2 != W2).sum()),
            w1_saturated=int((np.abs(twin.W1) == 127).sum()),
            pred_mismatch=int(sum(a != b for a, b in zip(rp, preds))) + abs(len(rp) - len(preds)), n_preds=len(preds),
            nb_fx_rtl=int(st['nb_fx']), nb_fx_twin=int(twin.nb_fx),
            n_w1_updates_rtl=int(st['n_w1_updates']), n_w1_updates_twin=int(twin.n_w1_updates),
            sim_cycles=int(st['cycles']),
            twin_acc_on_inferred=float(np.mean([p == labs[i] for i, p in preds])) if preds else None,
            rtl_acc_on_inferred=float(np.mean([p == labs[i] for i, p in rp])) if rp else None)
        res['BIT_EXACT'] = (res['w1_mismatch'] == 0 and res['w2_mismatch'] == 0 and res['pred_mismatch'] == 0
                            and res['nb_fx_rtl'] == res['nb_fx_twin'] and res['n_w1_updates_rtl'] == res['n_w1_updates_twin'])
        res['icarus_cycles_per_s'] = round(res['sim_cycles'] / t_sim)
    else:
        res['BIT_EXACT'] = False
        res['sim_tail'] = (r.stdout + r.stderr)[-1500:]
    json.dump(res, open(f'{out}/result.json', 'w'), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
