"""
Verilator smoke test for the Papilio calibration design (dev-PC only;
not needed on the FPGA PC). Verilates papilio_calib_top with a short
CLKS_PER_BIT, compiles tb_papilio.cpp, runs it, and reports PASS/FAIL.

  python build_and_run_tb.py
"""
from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
RTL = PKG / "rtl"
BUILD = PKG / "_vbuild_tb"


def find_verilator():
    import verilator as _v
    root = Path(_v.__file__).resolve().parent
    exe = root / "bin" / "verilator_bin.exe"
    if not exe.exists():
        exe = root / "bin" / "verilator_bin"
    return root, exe


def cxx_command():
    for cc in ("g++", "clang++"):
        if shutil.which(cc):
            return [cc]
    import ziglang  # noqa: F401
    return [sys.executable, "-m", "ziglang", "c++",
            "-target", "x86_64-windows-gnu"]


def main():
    vroot, vbin = find_verilator()
    if BUILD.exists():
        shutil.rmtree(BUILD)
    BUILD.mkdir(parents=True)

    srcs = ["papilio_calib_top.v", "calib_core.v", "uart_rx.v", "uart_tx.v"]
    cmd = [str(vbin), "--cc", "-O2", "--top-module", "papilio_calib_top",
           "--Mdir", str(BUILD), "-Wno-fatal",
           "-GCLKS_PER_BIT=16"] + srcs
    env = dict(os.environ, VERILATOR_ROOT=str(vroot))
    r = subprocess.run(cmd, cwd=RTL, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout, r.stderr)
        raise SystemExit("verilation failed")

    gen = sorted(glob.glob(str(BUILD / "*.cpp")))
    rt = [str(vroot / "include" / "verilated.cpp"),
          str(vroot / "include" / "verilated_threads.cpp")]
    exe = BUILD / "tb.exe"
    cc = cxx_command() + [
        "-std=c++20", "-O1", "-w",
        f"-I{BUILD}", f"-I{vroot / 'include'}",
        f"-I{vroot / 'include' / 'vltstd'}",
        str(HERE / "tb_papilio.cpp"), *gen, *rt,
        "-o", str(exe),
    ]
    r = subprocess.run(cc, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-4000:], r.stderr[-4000:])
        raise SystemExit("C++ compile failed")

    # $readmemh paths are relative to cwd -> run from rtl/
    r = subprocess.run([str(exe)], cwd=RTL, capture_output=True, text=True)
    print(r.stdout, r.stderr)
    raise SystemExit(r.returncode)


if __name__ == "__main__":
    main()
