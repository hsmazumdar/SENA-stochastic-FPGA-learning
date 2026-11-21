#!/usr/bin/env bash
# Re-runs the functional self-tests (Linux/macOS; needs python+torch+numpy, iverilog/vvp).
# Arg 1 (optional): path to the DWN repo 'src' folder for the official-layer export test.
set -e
cd "$(dirname "$0")/.."
python selftest/test_efd_equivalence.py
[ -n "$1" ] && python selftest/test_export_official_layers.py "$1"
python train_dwn.py --arch selftest_tiny --seeds 0 --lr-steps 1e-2:2,1e-3:1 --efd torch --device cpu --out /tmp/dwn_selftest_runs
python dwn_to_verilog.py /tmp/dwn_selftest_runs/selftest_tiny/seed0/model.json --out /tmp/dwn_selftest_gen --sim --force
python selftest/make_random_model.py --layers 1000,500 --out /tmp/dwn_random_sm.json
python dwn_to_verilog.py /tmp/dwn_random_sm.json --out /tmp/dwn_random_sm_gen --sim --force
