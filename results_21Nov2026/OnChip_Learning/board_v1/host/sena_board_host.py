#!/usr/bin/env python3
"""
sena_board_host.py -- host driver for the SENA on-chip-learning DE2-115 build (board_v1).

Talks to de2_115_sena_top over RS-232 (pyserial), loads MNIST-binarised images + labels + the
twin's training schedule into SRAM (with checksums and SRAM read-back verification), loads the
initial weights, sets the seed, trains 40 x 3000 samples (one 'T' command per epoch), runs the
2500-image test set at T=40, dumps the final weights and compares everything bit-exactly with the
software twin (sena_hw_twin.py) run for the same seed in lock-step.  Writes JSON + CSV.

Modes:
  --mode serial  (default)  real board, needs --port (e.g. COM5) ; pip install pyserial numpy
  --mode emit               write a byte script for the Icarus testbench (no board)
  --mode replay             re-run the same flow against the testbench's received-bytes log

Example (Monday):
  python sena_board_host.py --port COM5 --seed 0
  python sena_board_host.py --port COM5 --seed 1 --skip-load      # images already in SRAM
"""
import argparse, csv, json, os, struct, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sena_hw_twin as tw

IMG_WORDS, IMG_STRIDE, SCHED_BASE, N_TRAIN_IMG = 50, 64, 0xD0000, 10000


class ProtoError(Exception):
    pass


# ------------------------------------------------------------------ transports
class SerialT:
    def __init__(self, port, baud):
        import serial
        self.s = serial.Serial(port, baud, timeout=1.0)
        time.sleep(0.2); self.s.reset_input_buffer()
    def write(self, b): self.s.write(b); self.s.flush()
    def read(self, n, timeout=10.0):
        out = bytearray(); t0 = time.time()
        while len(out) < n:
            out += self.s.read(n - len(out))
            if time.time() - t0 > timeout:
                raise ProtoError(f'timeout: wanted {n} bytes, got {len(out)}: {bytes(out[:16]).hex()}')
        return bytes(out)
    checks = True


class EmitT:
    """Writes the testbench script: 'S xx' = send byte, 'R n' = wait until n bytes total received."""
    checks = False
    def __init__(self, path): self.f = open(path, 'w'); self.nrx = 0
    def write(self, b):
        for x in b: self.f.write('S %02x\n' % x)
    def read(self, n, timeout=None):
        self.nrx += n; self.f.write('R %d\n' % self.nrx); return bytes(n)
    def close(self): self.f.write('X 0\n'); self.f.close()


class ReplayT:
    checks = True
    def __init__(self, path):
        self.buf = bytes(int(l, 16) for l in open(path).read().split()); self.p = 0
    def write(self, b): pass
    def read(self, n, timeout=None):
        if self.p + n > len(self.buf): raise ProtoError('replay log too short')
        r = self.buf[self.p:self.p + n]; self.p += n; return r


# ------------------------------------------------------------------ protocol
def u24(v): return struct.pack('<I', v)[:3]

class Board:
    def __init__(self, t): self.t = t
    def chk(self, cond, msg):
        if self.t.checks and not cond: raise ProtoError(msg)
    def ident(self):
        self.t.write(b'V'); r = self.t.read(7)
        self.chk(r[:4] == b'SENA', f'bad ident {r.hex()}')
        return r[4], r[5] | (r[6] << 8)
    def write_words(self, addr, words):
        words = np.asarray(words, dtype=np.uint16); assert len(words) <= 4096
        data = words.astype('<u2').tobytes()
        self.t.write(b'W' + u24(addr) + struct.pack('<H', len(words)) + data)
        r = self.t.read(3)
        self.chk(r[0] == ord('K') and (r[1] | r[2] << 8) == (sum(data) & 0xFFFF), f'W sum mismatch @0x{addr:x}: {r.hex()}')
    def checksum(self, addr, n):
        self.t.write(b'C' + u24(addr) + u24(n)); r = self.t.read(5, timeout=30)
        self.chk(r[0] == ord('K'), f'C bad reply {r.hex()}')
        return struct.unpack('<I', r[1:5])[0]
    def read_words(self, addr, n):
        self.t.write(b'R' + u24(addr) + struct.pack('<H', n)); r = self.t.read(2 * n)
        return np.frombuffer(r, dtype='<u2')
    def set_seed(self, s):
        self.t.write(b'S' + struct.pack('<I', s)); r = self.t.read(1); self.chk(r == b'K', f'S reply {r.hex()}')
    def load_row(self, sel, row, vals):
        data = bytes((int(v) & 0xFF) for v in vals)
        self.t.write(b'L' + bytes([sel]) + struct.pack('<H', row) + data); r = self.t.read(3)
        self.chk(r[0] == ord('K') and (r[1] | r[2] << 8) == (sum(data) & 0xFFFF), f'L sum mismatch sel{sel} row{row}: {r.hex()}')
    def dump_row(self, sel, row, n):
        self.t.write(b'D' + bytes([sel]) + struct.pack('<H', row)); r = self.t.read(n + 2)
        self.chk((r[n] | r[n + 1] << 8) == (sum(r[:n]) & 0xFFFF), f'D sum mismatch sel{sel} row{row}')
        return np.frombuffer(r[:n], dtype=np.int8).astype(np.int64)
    def train(self, start, count):
        self.t.write(b'T' + u24(start) + u24(count)); r = self.t.read(13, timeout=120)
        self.chk(r[0] == ord('F'), f'T reply {r.hex()}')
        return struct.unpack('<III', r[1:13])          # core_cycles, total_cycles, n_w1_updates
    def evaluate(self, first, count, T):
        self.t.write(b'E' + struct.pack('<HHB', first, count, T))
        preds = np.frombuffer(self.t.read(count, timeout=60 + count * 0.01), dtype=np.uint8).astype(int)
        r = self.t.read(11, timeout=30); self.chk(r[0] == ord('F'), f'E reply {r.hex()}')
        correct, cc, tc = struct.unpack('<HII', r[1:11])
        return preds, correct, cc, tc
    def status(self):
        self.t.write(b'Q'); r = self.t.read(13); self.chk(r[0] == ord('Q'), f'Q reply {r.hex()}')
        nu, ns, nb = struct.unpack('<IIi', r[1:13]); return nu, ns, nb


# ------------------------------------------------------------------ data layout
def image_words(x, lab):
    bits = np.asarray(x, dtype=np.int64).reshape(49, 16)
    w = np.zeros(IMG_STRIDE, dtype=np.uint16)
    w[:49] = (bits << np.arange(16)).sum(axis=1).astype(np.uint16)
    w[49] = int(lab) & 0xF
    return w


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mode', choices=['serial', 'emit', 'replay'], default='serial')
    ap.add_argument('--port', help='e.g. COM5 or /dev/ttyUSB0'); ap.add_argument('--baud', type=int, default=115200)
    ap.add_argument('--script', default='sim_script.txt'); ap.add_argument('--rxlog', default='sim_rx.txt')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--data', default=os.path.join(HERE, 'mnist_bin_t0.3_tr10000_te2500_s2026.npz'))
    ap.add_argument('--n-hid', type=int, default=160); ap.add_argument('--sh', type=int, default=None)
    ap.add_argument('--epochs', type=int, default=40); ap.add_argument('--spe', type=int, default=3000)
    ap.add_argument('--T', type=int, default=40); ap.add_argument('--n-test', type=int, default=2500)
    ap.add_argument('--sched-limit', type=int, default=0, help='reduced case: train only the first K schedule entries')
    ap.add_argument('--sparse-load', action='store_true', help='load only the images used (reduced case)')
    ap.add_argument('--skip-load', action='store_true', help='images already in SRAM (same power-up): only verify their checksum; schedule is always reloaded')
    ap.add_argument('--no-readback', action='store_true', help='skip weight read-back after loading')
    ap.add_argument('--f-clk', type=float, default=25e6); ap.add_argument('--out', default=None)
    a = ap.parse_args()
    sh = a.sh if a.sh is not None else (7 if a.n_hid >= 64 else 4)
    out = a.out or os.path.join(HERE, 'runs', f'board_h{a.n_hid}_s{a.seed}_{time.strftime("%Y%m%d_%H%M%S")}')
    os.makedirs(out, exist_ok=True)
    log = open(os.path.join(out, 'log.txt'), 'a')
    def say(*m):
        s = time.strftime('%H:%M:%S ') + ' '.join(str(x) for x in m); print(s, flush=True); log.write(s + '\n'); log.flush()

    if a.mode == 'serial':
        if not a.port: ap.error('--port is required in serial mode')
        t = SerialT(a.port, a.baud)
    elif a.mode == 'emit':
        t = EmitT(a.script)
    else:
        t = ReplayT(a.rxlog)
    B = Board(t)

    d = np.load(a.data); Xtr, ytr, Xte, yte = d['Xtr'], d['ytr'], d['Xte'], d['yte']
    n_test = min(a.n_test, len(Xte))
    sched = np.concatenate(tw.schedule(len(Xtr), a.seed, epochs=a.epochs, spe=a.spe)).astype(np.int64)
    if a.sched_limit: sched = sched[:a.sched_limit]
    epochs = [(s, min(a.spe, len(sched) - s)) for s in range(0, len(sched), a.spe)]
    seed_in = tw.seed_in_for(a.seed)
    W1, W2 = tw.init_weights(784, a.n_hid, 10, a.seed)
    T0 = time.time()

    ver, nh = B.ident()
    say(f'ident: version 0x{ver:02x} N_HID={nh}')
    B.chk(nh == a.n_hid, f'board N_HID={nh} but --n-hid={a.n_hid}')

    # ---------------- images + labels + schedule -> SRAM
    allw = np.zeros((N_TRAIN_IMG + len(Xte), IMG_STRIDE), dtype=np.uint16)
    for n in range(len(Xtr)): allw[n] = image_words(Xtr[n], ytr[n])
    for n in range(len(Xte)): allw[N_TRAIN_IMG + n] = image_words(Xte[n], yte[n])
    if a.sparse_load:
        imgs = sorted(set(sched.tolist()) | set(range(N_TRAIN_IMG, N_TRAIN_IMG + n_test)))
        if not a.skip_load:
            for n in imgs: B.write_words(n * IMG_STRIDE, allw[n, :IMG_WORDS])
        bad = [n for n in imgs if B.checksum(n * IMG_STRIDE, IMG_WORDS) != int(allw[n, :IMG_WORDS].astype(np.uint64).sum())]
        B.chk(not bad, f'SRAM image read-back checksum mismatch on images {bad[:10]}')
        say(f'loaded/verified {len(imgs)} images (sparse)')
    else:
        flat = allw.reshape(-1)
        if not a.skip_load:
            t1 = time.time()
            for o in range(0, len(flat), 4096):
                B.write_words(o, flat[o:o + 4096])
                if (o // 4096) % 20 == 0: say(f'  images: {o * 100 // len(flat)}%')
            say(f'images written in {time.time() - t1:.1f} s')
        cs = B.checksum(0, len(flat)); exp = int(flat.astype(np.uint64).sum()) & 0xFFFFFFFF
        B.chk(cs == exp, f'SRAM image read-back checksum 0x{cs:08x} != expected 0x{exp:08x}')
        say(f'image region read-back checksum OK (0x{exp:08x}, {len(flat)} words)')
    sw = sched.astype(np.uint16)
    for o in range(0, len(sw), 4096): B.write_words(SCHED_BASE + o, sw[o:o + 4096])   # always (seed-specific)
    cs = B.checksum(SCHED_BASE, len(sw)); exp = int(sw.astype(np.uint64).sum()) & 0xFFFFFFFF
    B.chk(cs == exp, f'schedule checksum 0x{cs:08x} != 0x{exp:08x}')
    say(f'schedule ({len(sw)} entries) read-back checksum OK')
    spot = B.read_words(N_TRAIN_IMG * IMG_STRIDE if a.sparse_load and n_test else 0, IMG_WORDS)
    B.chk(np.array_equal(spot, allw[N_TRAIN_IMG if a.sparse_load and n_test else 0, :IMG_WORDS]), 'SRAM spot read mismatch')

    # ---------------- weights + seed
    for i in range(785): B.load_row(1, i, W1[:, i])
    for j in range(a.n_hid + 1): B.load_row(2, j, W2[:, j])
    if not a.no_readback:
        bad1 = sum(int((B.dump_row(1, i, a.n_hid) != W1[:, i]).sum()) for i in range(785))
        bad2 = sum(int((B.dump_row(2, j, 10) != W2[:, j]).sum()) for j in range(a.n_hid + 1))
        B.chk(bad1 == 0 and bad2 == 0, f'weight read-back mismatch W1 {bad1} W2 {bad2}')
        say('initial weights loaded and read back OK')
    B.set_seed(seed_in)
    nu0, ns0, nb0 = B.status()
    B.chk(nb0 == 384, f'nb_fx after seed = {nb0}, expected 384')
    say(f'seed {a.seed} -> seed_in 0x{seed_in:08x}; status nupd0={nu0} samples0={ns0} nb_fx={nb0}')

    # ---------------- train, epoch by epoch, twin in lock-step
    twin = tw.HWTwin(W1, W2, seed_in, sh)
    rows = []; diverged = None
    for e, (s, c) in enumerate(epochs):
        tw0 = time.time()
        cc, tc, nu = B.train(s, c)
        wall = time.time() - tw0
        for i in sched[s:s + c]: twin.train(Xtr[i], int(ytr[i]))
        nu_b, ns_b, nb_b = B.status()
        ok = (nu_b - nu0 == twin.n_w1_updates) and (nb_b == twin.nb_fx)
        if not ok and diverged is None and t.checks: diverged = e
        rows.append(dict(epoch=e, samples=c, core_cycles=cc, total_cycles=tc, board_w1_updates=nu_b - nu0,
                         twin_w1_updates=twin.n_w1_updates, board_nb_fx=nb_b, twin_nb_fx=int(twin.nb_fx),
                         match=bool(ok), host_wall_s=round(wall, 3)))
        say(f'epoch {e:2d}: {c} samples core {cc} cyc total {tc} cyc ({tc / a.f_clk * 1e3:.1f} ms @25MHz) '
            f'nupd board {nu_b - nu0} twin {twin.n_w1_updates} nb_fx board {nb_b} twin {twin.nb_fx} {"OK" if ok else "MISMATCH"}')

    # ---------------- test
    tw0 = time.time()
    preds, correct, ecc, etc = B.evaluate(N_TRAIN_IMG, n_test, a.T)
    ewall = time.time() - tw0
    tpred = np.array([twin.infer(Xte[i], a.T)[0] for i in range(n_test)])
    acc_b = correct / max(n_test, 1); acc_bp = float((preds == yte[:n_test]).mean()) if n_test else 0.0
    acc_t = float((tpred == yte[:n_test]).mean()) if n_test else 0.0
    say(f'test: board {correct}/{n_test} = {100 * acc_b:.2f}%  twin {100 * acc_t:.2f}%  pred mismatches {int((preds != tpred).sum())}')

    # ---------------- dump + compare
    fW1 = np.stack([B.dump_row(1, i, a.n_hid) for i in range(785)], axis=1)
    fW2 = np.stack([B.dump_row(2, j, 10) for j in range(a.n_hid + 1)], axis=1)
    w1m = int((fW1 != twin.W1).sum()); w2m = int((fW2 != twin.W2).sum()); pm = int((preds != tpred).sum())
    bit_exact = (w1m == 0 and w2m == 0 and pm == 0 and all(r['match'] for r in rows))
    say(f'final weights: W1 mismatches {w1m}/{twin.W1.size}, W2 {w2m}/{twin.W2.size} -> BIT_EXACT={bit_exact}')

    tcyc = sum(r['total_cycles'] for r in rows); ccyc = sum(r['core_cycles'] for r in rows)
    res = dict(source='DE2-115 board' if a.mode == 'serial' else f'Icarus top-level sim ({a.mode})',
               seed=a.seed, seed_in=f'0x{seed_in:08x}', n_hid=a.n_hid, sh=sh, epochs=len(epochs),
               train_samples=int(len(sched)), T=a.T, n_test=n_test, f_clk_hz=a.f_clk,
               board_correct=int(correct), board_acc=acc_b, board_acc_from_preds=acc_bp, twin_acc=acc_t,
               pred_mismatch=pm, w1_mismatch=w1m, w1_total=int(twin.W1.size), w2_mismatch=w2m, w2_total=int(twin.W2.size),
               w1_updates_board=rows[-1]['board_w1_updates'] if rows else 0, w1_updates_twin=int(twin.n_w1_updates),
               nb_fx_final_board=rows[-1]['board_nb_fx'] if rows else nb0, nb_fx_final_twin=int(twin.nb_fx),
               first_diverged_epoch=diverged, BIT_EXACT=bool(bit_exact),
               train_total_cycles=tcyc, train_core_cycles=ccyc, train_s_at_fclk=tcyc / a.f_clk,
               train_cycles_per_sample=tcyc / max(len(sched), 1),
               test_total_cycles=etc, test_core_cycles=ecc, test_s_at_fclk=etc / a.f_clk,
               test_cycles_per_image=etc / max(n_test, 1),
               host_wall_train_s=round(sum(r['host_wall_s'] for r in rows), 2), host_wall_test_s=round(ewall, 2),
               host_wall_total_s=round(time.time() - T0, 1), date=time.strftime('%Y-%m-%d %H:%M:%S %Z'))
    json.dump(res, open(os.path.join(out, 'result.json'), 'w'), indent=1)
    with open(os.path.join(out, 'epochs.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ['epoch']); w.writeheader(); w.writerows(rows)
    np.savez_compressed(os.path.join(out, 'final.npz'), W1=fW1, W2=fW2, preds=preds, twin_preds=tpred,
                        twin_W1=twin.W1, twin_W2=twin.W2)
    if isinstance(t, EmitT): t.close()
    say('wrote', out)
    return 0 if (bit_exact or not t.checks) else 2


if __name__ == '__main__':
    sys.exit(main())
