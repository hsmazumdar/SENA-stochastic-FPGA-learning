"""
verilator_cosim.py
==================
Runs the SENA RTL under Verilator on this PC and cross-checks it, cycle for
cycle, against a pure-Python cycle-accurate model of the same hardware.

Why this matters: Verilator can only comfortably elaborate a small tile, but
once the Python model agrees with Verilator *exactly* (same predictions, same
votes, same clock-cycle counts) we can use the Python model to extrapolate
cycle counts to the full 784-160-10 MNIST configuration, and convert those
cycles into wall-clock time at any target FPGA/ASIC clock rate.

Usage:  python verilator_cosim.py
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TB = HERE / "tb"
BUILD = HERE / "_vbuild"
RESULTS = HERE.parent / "results"

TAPS = 0x80200003
MASK32 = 0xFFFFFFFF


# --------------------------------------------------------------------------
# toolchain discovery
# --------------------------------------------------------------------------
def find_verilator() -> tuple[Path, Path]:
    """Return (verilator_root, verilator_bin)."""
    try:
        import verilator as _v  # pip package ships the binary

        root = Path(_v.__file__).resolve().parent
        exe = root / "bin" / "verilator_bin.exe"
        if not exe.exists():
            exe = root / "bin" / "verilator_bin"
        if exe.exists():
            return root, exe
    except ImportError:
        pass
    exe = shutil.which("verilator_bin") or shutil.which("verilator")
    if exe:
        exe = Path(exe)
        return exe.parent.parent, exe
    raise RuntimeError("Verilator not found (pip install verilator)")


def cxx_command() -> list[str]:
    """A C++20 compiler. Prefer a real one, fall back to zig's clang."""
    for cc in ("g++", "clang++"):
        if shutil.which(cc):
            return [cc]
    try:
        import ziglang  # self-contained clang + mingw libc

        return [sys.executable, "-m", "ziglang", "c++",
                "-target", "x86_64-windows-gnu"]
    except ImportError:
        pass
    raise RuntimeError("No C++ compiler found (pip install ziglang)")


# --------------------------------------------------------------------------
# cycle-accurate Python model of the RTL
# --------------------------------------------------------------------------
def lfsr_next(r: int) -> int:
    return ((r >> 1) ^ (TAPS if (r & 1) else 0)) & MASK32


class PyNeuron:
    """Mirrors stochastic_neuron.v register-for-register."""

    def __init__(self, n_in: int, par: int, seed: int):
        self.n_in, self.par = n_in, par
        self.nstep = (n_in + par - 1) // par
        self.r = seed & MASK32
        self.busy = 0
        self.step = 0
        self.acc = 0
        self.y = 0
        self.done = 0

    def clock(self, start, bits, w, bias, K, n_log2):
        half = 1 << n_log2
        rnd = (self.r & ((half << 1) - 1)) - half
        thr = K + rnd

        base = self.step * self.par
        psum = 0
        for lane in range(self.par):
            f = base + lane
            if f < self.n_in and bits[f]:
                psum += int(w[f])
        acc_next = self.acc + psum

        busy_n, step_n, acc_n, y_n, done_n = self.busy, self.step, self.acc, self.y, 0
        if start and not self.busy:
            busy_n, step_n, acc_n = 1, 0, int(bias)
        elif self.busy:
            acc_n = acc_next
            if self.step == self.nstep - 1:
                y_n = 1 if acc_next > thr else 0
                busy_n, done_n = 0, 1
            else:
                step_n = self.step + 1

        r_n = lfsr_next(self.r) if self.busy else self.r
        self.busy, self.step, self.acc, self.y, self.done, self.r = (
            busy_n, step_n, acc_n, y_n, done_n, r_n)


class PyLayer:
    """Mirrors sena_layer.v (a bank of neurons sharing one start)."""

    def __init__(self, n_in: int, n_out: int, par: int):
        self.n = [PyNeuron(n_in, par, (0xA5A50001 + i * 0x01010001) & MASK32)
                  for i in range(n_out)]

    @property
    def busy(self):
        return int(any(u.busy for u in self.n))

    @property
    def done(self):
        return int(all(u.done for u in self.n))

    @property
    def y_bits(self):
        return [u.y for u in self.n]

    def clock(self, start, bits, W, b, K, n_log2):
        for i, u in enumerate(self.n):
            u.clock(start, bits, W[i], b[i], K, n_log2)


class PyEarlyExit:
    """Mirrors sena_early_exit.v."""

    def __init__(self, n_class: int, t_max: int, th_num: int, th_den_log2: int):
        self.nc, self.t_max = n_class, t_max
        self.th_num, self.th_den_log2 = th_num, th_den_log2
        self.votes = [0] * n_class
        self.passes = 0
        self.exit_ready = 0
        self.pred = 0
        self.done_force = 0

    def clock(self, clear, pass_valid, class_bits):
        vn = [self.votes[c] + (1 if class_bits[c] else 0) for c in range(self.nc)]
        sum_v, max_v, argmax = 0, 0, 0
        for c in range(self.nc):
            sum_v += vn[c]
            if vn[c] > max_v:
                max_v, argmax = vn[c], c
        conf_ok = sum_v != 0 and (max_v << self.th_den_log2) >= sum_v * self.th_num
        at_tmax = (self.passes + 1) >= self.t_max

        if clear:
            self.votes = [0] * self.nc
            self.passes = 0
            self.exit_ready = 0
            self.pred = 0
            self.done_force = 0
        elif pass_valid and not self.exit_ready and not self.done_force:
            self.votes = vn
            self.passes += 1
            self.pred = argmax
            if conf_ok:
                self.exit_ready = 1
            elif at_tmax:
                self.done_force = 1


class PyTile:
    """Mirrors sena_tile.v, including the 4-state pass FSM."""

    S_IDLE, S_HID, S_OUT, S_VOTE = 0, 1, 2, 3

    def __init__(self, n_in, n_hid, n_out, par, par_out, t_max, th_num,
                 th_den_log2):
        self.hid = PyLayer(n_in, n_hid, par)
        self.out = PyLayer(n_hid, n_out, par_out)
        self.ee = PyEarlyExit(n_out, t_max, th_num, th_den_log2)
        self.state = self.S_IDLE
        self.clear_votes = 0
        self.cycles = 0

    @property
    def busy(self):
        return self.hid.busy or self.out.busy or (self.state != self.S_IDLE)

    def clock(self, sample_start, pass_start, x_bits, cfg):
        # sample every register-driven signal BEFORE any submodule is clocked
        hid_done, out_done = self.hid.done, self.out.done
        hid_start = pass_start and (self.state == self.S_IDLE)
        out_start = hid_done
        h_bits = self.hid.y_bits
        o_bits = self.out.y_bits
        ee_clear = sample_start or self.clear_votes
        ee_valid = out_done

        self.hid.clock(hid_start, x_bits, cfg["w_hid"], cfg["b_hid"],
                       cfg["K_hid"], cfg["n_log2_hid"])
        self.out.clock(out_start, h_bits, cfg["w_out"], cfg["b_out"],
                       cfg["K_out"], cfg["n_log2_out"])
        self.ee.clock(ee_clear, ee_valid, o_bits)

        st, cv = self.state, 0
        if st == self.S_IDLE:
            st = self.S_HID if pass_start else st
        elif st == self.S_HID:
            st = self.S_OUT if hid_done else st
        elif st == self.S_OUT:
            st = self.S_VOTE if out_done else st
        elif st == self.S_VOTE:
            st = self.S_IDLE
        if sample_start:
            st, cv = self.S_IDLE, 1
        self.state, self.clear_votes = st, cv
        self.cycles += 1


def py_run(cfg, samples):
    """Drive PyTile with exactly the protocol used by sim_main.cpp."""
    tile = PyTile(cfg["N_IN"], cfg["N_HID"], cfg["N_OUT"], cfg["PAR"],
                  cfg["PAR_OUT"], cfg["T_MAX"], cfg["TH_NUM"],
                  cfg["TH_DEN_LOG2"])
    zero = [0] * cfg["N_IN"]
    for _ in range(6):
        tile.clock(0, 0, zero, cfg)

    rows = []
    for x in samples:
        xb = list(map(int, x))
        tile.clock(1, 0, xb, cfg)
        tile.clock(0, 0, xb, cfg)

        c0 = tile.cycles
        cyc_pass1, passes_seen = 0, 0
        while passes_seen < cfg["T_MAX"]:
            p0 = tile.cycles
            tile.clock(0, 1, xb, cfg)
            guard = 0
            while tile.busy and guard < 100000:
                tile.clock(0, 0, xb, cfg)
                guard += 1
            passes_seen += 1
            if passes_seen == 1:
                cyc_pass1 = tile.cycles - p0
            if tile.ee.exit_ready or tile.ee.done_force:
                break
        rows.append(dict(pred=tile.ee.pred, passes=tile.ee.passes,
                         exited=int(tile.ee.exit_ready),
                         cycles=tile.cycles - c0, cycles_pass1=cyc_pass1,
                         votes=list(tile.ee.votes)))
    return rows


# --------------------------------------------------------------------------
# stimulus
# --------------------------------------------------------------------------
def cycles_per_pass(n_in, n_hid, par, par_out):
    """Closed form validated against Verilator (see main())."""
    return -(-n_in // par) + -(-n_hid // par_out) + 4


def make_cfg(par: int, par_out: int, seed: int = 7):
    rng = np.random.default_rng(seed)
    N_IN, N_HID, N_OUT = 64, 8, 4
    cfg = dict(N_IN=N_IN, N_HID=N_HID, N_OUT=N_OUT, T_MAX=16, PAR=par,
               PAR_OUT=par_out, TH_NUM=3, TH_DEN_LOG2=2,
               K_hid=0, K_out=0, n_log2_hid=6, n_log2_out=4)
    cfg["w_hid"] = rng.integers(-40, 41, size=(N_HID, N_IN)).astype(int)
    cfg["b_hid"] = rng.integers(-20, 21, size=N_HID).astype(int)
    cfg["w_out"] = rng.integers(-40, 41, size=(N_OUT, N_HID)).astype(int)
    cfg["b_out"] = rng.integers(-20, 21, size=N_OUT).astype(int)
    samples = rng.integers(0, 2, size=(24, N_IN)).astype(int)
    return cfg, samples


def write_stimulus(path: Path, cfg, samples):
    with open(path, "w") as f:
        f.write(f"{cfg['N_IN']} {cfg['N_HID']} {cfg['N_OUT']} "
                f"{cfg['T_MAX']} {cfg['PAR']} {cfg['PAR_OUT']}\n")
        f.write(f"{cfg['K_hid']} {cfg['K_out']} "
                f"{cfg['n_log2_hid']} {cfg['n_log2_out']}\n")
        for row in cfg["w_hid"]:
            f.write(" ".join(map(str, row)) + "\n")
        f.write(" ".join(map(str, cfg["b_hid"])) + "\n")
        for row in cfg["w_out"]:
            f.write(" ".join(map(str, row)) + "\n")
        f.write(" ".join(map(str, cfg["b_out"])) + "\n")
        f.write(f"{len(samples)}\n")
        for s in samples:
            f.write(" ".join(map(str, s)) + "\n")


# --------------------------------------------------------------------------
# verilate + compile + run
# --------------------------------------------------------------------------
def build(cfg, tag: str) -> Path:
    vroot, vbin = find_verilator()
    odir = BUILD / tag
    if odir.exists():
        shutil.rmtree(odir)
    odir.mkdir(parents=True, exist_ok=True)

    srcs = ["lfsr32.v", "stochastic_neuron.v", "sena_layer.v",
            "sena_early_exit.v", "sena_tile.v"]
    cmd = [str(vbin), "--cc", "--sv", "-O2", "--top-module", "sena_tile",
           "--Mdir", str(odir),
           "-Wno-fatal", "-Wno-PINCONNECTEMPTY",
           f"-GN_IN={cfg['N_IN']}", f"-GN_HID_BANK={cfg['N_HID']}",
           f"-GN_OUT={cfg['N_OUT']}", f"-GPAR={cfg['PAR']}",
           f"-GPAR_OUT={cfg['PAR_OUT']}",
           f"-GT_MAX={cfg['T_MAX']}", f"-GTH_NUM={cfg['TH_NUM']}",
           f"-GTH_DEN_LOG2={cfg['TH_DEN_LOG2']}"] + srcs
    env = dict(os.environ, VERILATOR_ROOT=str(vroot))
    r = subprocess.run(cmd, cwd=HERE, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout, r.stderr)
        raise RuntimeError("verilation failed")

    gen = sorted(glob.glob(str(odir / "*.cpp")))
    rt = [str(vroot / "include" / "verilated.cpp"),
          str(vroot / "include" / "verilated_threads.cpp")]
    exe = odir / "sim.exe"
    cc = cxx_command() + [
        "-std=c++20", "-O1", "-w",
        f"-I{odir}", f"-I{vroot / 'include'}", f"-I{vroot / 'include' / 'vltstd'}",
        str(TB / "sim_main.cpp"), *gen, *rt,
        "-o", str(exe),
    ]
    r = subprocess.run(cc, capture_output=True, text=True)
    if r.returncode != 0:
        print(" ".join(cc[:6]), "...")
        print(r.stdout[-4000:], r.stderr[-4000:])
        raise RuntimeError("C++ compile failed")
    return exe


def run_verilator(exe: Path, stim: Path, out: Path):
    r = subprocess.run([str(exe), str(stim), str(out)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout, r.stderr)
        raise RuntimeError("simulation failed")
    rows = []
    for line in open(out):
        if line.startswith("#"):
            continue
        p = line.split()
        rows.append(dict(pred=int(p[1]), passes=int(p[2]), exited=int(p[3]),
                         cycles=int(p[4]), cycles_pass1=int(p[5]),
                         votes=[int(v) for v in p[6:]]))
    return rows


def compare(vrows, prows, label):
    keys = ("pred", "passes", "exited", "cycles", "cycles_pass1", "votes")
    bad = []
    for i, (a, b) in enumerate(zip(vrows, prows)):
        for k in keys:
            if a[k] != b[k]:
                bad.append((i, k, a[k], b[k]))
    ok = not bad and len(vrows) == len(prows)
    print(f"  {label}: {len(vrows)} samples, "
          f"{'MATCH' if ok else 'MISMATCH'}")
    for row in bad[:10]:
        print("    sample %d  %s  verilator=%s  python=%s" % row)
    return ok


def main():
    import time

    BUILD.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    summary, all_ok = {}, True

    for par, par_out in ((1, 1), (4, 1), (8, 2), (16, 4)):
        tag = f"par{par}_{par_out}"
        print(f"[PAR={par}, PAR_OUT={par_out}] verilating + compiling ...")
        cfg, samples = make_cfg(par, par_out)
        stim = BUILD / f"stim_{tag}.txt"
        write_stimulus(stim, cfg, samples)
        exe = build(cfg, tag)

        t0 = time.perf_counter()
        vrows = run_verilator(exe, stim, BUILD / f"out_{tag}.txt")
        wall = time.perf_counter() - t0

        prows = py_run(cfg, samples)
        ok = compare(vrows, prows, tag)
        all_ok &= ok

        total_cycles = sum(r["cycles"] for r in vrows)
        summary[tag] = dict(
            par=par, par_out=par_out, match=bool(ok), n_samples=len(vrows),
            cycles_pass1_rtl=vrows[0]["cycles_pass1"],
            cycles_pass1_formula=cycles_per_pass(
                cfg["N_IN"], cfg["N_HID"], par, par_out),
            total_cycles=total_cycles,
            host_wall_s=wall,
            host_cycles_per_s=total_cycles / wall if wall > 0 else 0.0,
            mean_passes=float(np.mean([r["passes"] for r in vrows])),
        )

    print("\nCycles for one stochastic pass (64-8-4 tile):")
    for k, v in summary.items():
        flag = "OK" if v["cycles_pass1_rtl"] == v["cycles_pass1_formula"] else "??"
        print(f"  {k}: RTL-measured={v['cycles_pass1_rtl']}  "
              f"closed-form={v['cycles_pass1_formula']}  [{flag}]")

    # ---- dedicated host-throughput benchmark ----------------------------
    # The per-config timings above are dominated by process start-up, so run a
    # long stimulus for every config.  Simulation cost per cycle grows with the
    # amount of logic evaluated, so we also report a work-normalised rate:
    #   lane-updates/s = cycles/s * (N_HID*PAR + N_OUT*PAR_OUT)
    # That constant is what lets us project host runtime for the full
    # 784-160-10 design, which is far too large to verilate comfortably.
    print("\nBenchmarking host simulation throughput ...")
    rng = np.random.default_rng(1234)
    for par, par_out in ((1, 1), (4, 1), (8, 2), (16, 4)):
        tag = f"par{par}_{par_out}"
        cfg, _ = make_cfg(par, par_out)
        big = rng.integers(0, 2, size=(600, cfg["N_IN"])).astype(int)
        stim = BUILD / f"stim_bench_{tag}.txt"
        write_stimulus(stim, cfg, big)
        exe = BUILD / tag / "sim.exe"
        t0 = time.perf_counter()
        rws = run_verilator(exe, stim, BUILD / f"out_bench_{tag}.txt")
        wall = time.perf_counter() - t0
        tot = sum(r["cycles"] for r in rws)
        work = cfg["N_HID"] * par + cfg["N_OUT"] * par_out
        summary[tag].update(
            bench_cycles=tot, bench_wall_s=wall,
            bench_cycles_per_s=tot / wall,
            lanes_per_cycle=work,
            lane_updates_per_s=tot / wall * work)
        print(f"  {tag:<9}: {tot/wall/1e6:6.2f} M cycles/s  "
              f"({work:>4} lanes/cycle -> "
              f"{tot / wall * work / 1e6:7.1f} M lane-updates/s)")

    lane_rate = float(np.mean([summary[t]["lane_updates_per_s"]
                               for t in summary if t.startswith("par")]))
    base_rate = summary["par1_1"]["bench_cycles_per_s"]
    summary["_host"] = dict(cycles_per_s=base_rate,
                            lane_updates_per_s=lane_rate)
    print(f"Host work-normalised rate: {lane_rate/1e6:.1f} M lane-updates/s")

    out = RESULTS / "verilator_cosim.json"
    json.dump(summary, open(out, "w"), indent=2)
    print(f"wrote {out}")
    print("RESULT:", "ALL MATCH" if all_ok else "MISMATCH")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
