"""
C4 — One-command regeneration of every Q1 artifact.

Runs the experiment scripts in dependency order, records per-step status,
and writes a reproducibility manifest (environment, package versions,
dataset seed, expected artifacts, checksums of result JSONs).

Usage:
  python run_q1_suite.py --profile smoke          # fast sanity pass
  python run_q1_suite.py --profile full           # publishable numbers
  python run_q1_suite.py --check                  # verify artifacts only
  python run_q1_suite.py --profile full --only a1,b2

Checkpoints make re-runs cheap: A2/B1/B2 reuse per-cell NPZ files unless the
underlying script is invoked with --retrain (not done by this driver).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SIM = HERE / "sim"
RTL = HERE / "rtl"
RESULTS = HERE / "results"

DATASET_SEED = 2026
TRAIN_SEEDS = [0, 1, 2, 3, 4]

# (id, cwd, script, extra args, primary artifacts)
STEPS = [
    ("a1", SIM, "exp_q1_baselines.py", [], ["q1_mnist_baselines.json"]),
    ("a2", SIM, "exp_q1_ablations.py", [], ["q1_sena_ablations.json"]),
    ("a3", RTL, "exp_rtl_mnist_accuracy.py", [], ["q1_rtl_mnist_accuracy.json"]),
    ("a4", RTL, "exp_q1_synth.py", [], ["q1_fpga_synth.json"]),
    ("a5", SIM, "exp_q1_fashion.py", [], ["q1_fashion_mnist.json"]),
    ("a6", SIM, "exp_q1_online_train.py", [], ["q1_online_training.json"]),
    ("b1", SIM, "exp_q1_scaling.py", [], ["q1_scaling.json"]),
    ("b2", SIM, "exp_q1_bitwidth.py", [], ["q1_bitwidth.json"]),
    ("b3", SIM, "exp_q1_analog_mc.py", [], ["q1_analog_mc.json"]),
    ("b4", SIM, "exp_q1_shapes.py", [], ["q1_shapes.json"]),
    ("b5", SIM, "analyze_q1_errors.py", [], ["q1_error_analysis.json"]),
    ("c3", SIM, "exp_q1_sensor_adc.py", [], ["q1_sensor_adc.json"]),
    # pareto last: consumes the other JSONs; takes no --profile
    ("b6", SIM, "exp_q1_pareto.py", None, ["q1_unified_pareto.json"]),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def package_versions():
    out = {}
    for name in ("numpy", "matplotlib", "torch", "torchvision", "sklearn"):
        try:
            mod = __import__(name)
            out[name] = getattr(mod, "__version__", "unknown")
        except Exception:
            out[name] = None
    return out


def run_step(step_id, cwd, script, extra, profile):
    cmd = [sys.executable, script]
    if extra is not None:
        cmd += ["--profile", profile] + extra
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    elapsed = time.perf_counter() - t0
    ok = proc.returncode == 0
    tail = (proc.stdout or "").strip().splitlines()[-3:]
    if not ok:
        tail += (proc.stderr or "").strip().splitlines()[-10:]
    return dict(step=step_id, script=script, ok=ok,
                returncode=proc.returncode, elapsed_s=round(elapsed, 1),
                tail=tail)


def check_artifacts():
    report = []
    for step_id, _cwd, script, _extra, artifacts in STEPS:
        for name in artifacts:
            path = RESULTS / name
            entry = dict(step=step_id, artifact=name, exists=path.exists())
            if path.exists():
                entry["sha256"] = sha256(path)
                entry["bytes"] = path.stat().st_size
                try:
                    entry["profile"] = json.loads(
                        path.read_text(encoding="utf-8")).get("profile")
                except Exception:
                    entry["profile"] = None
            report.append(entry)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["smoke", "full"],
                        default="smoke")
    parser.add_argument("--check", action="store_true",
                        help="verify artifacts without running anything")
    parser.add_argument("--only", default=None,
                        help="comma-separated step ids (e.g. a1,b2,b6)")
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)

    selected = ([s.strip() for s in args.only.split(",")]
                if args.only else [s[0] for s in STEPS])
    if not args.check and args.profile == "smoke":
        clobber = [a for a in check_artifacts()
                   if a["exists"] and a.get("profile") == "full"
                   and a["step"] in selected]
        if clobber:
            print("WARNING: smoke run will overwrite full-profile artifacts: "
                  + ", ".join(a["artifact"] for a in clobber), flush=True)
    runs = []
    if not args.check:
        for step_id, cwd, script, extra, _arts in STEPS:
            if step_id not in selected:
                continue
            print(f"[{step_id}] {script} --profile {args.profile}"
                  if extra is not None else f"[{step_id}] {script}",
                  flush=True)
            result = run_step(step_id, cwd, script, extra, args.profile)
            runs.append(result)
            status = "ok" if result["ok"] else "FAILED"
            print(f"[{step_id}] {status} ({result['elapsed_s']}s)", flush=True)
            if not result["ok"]:
                print("\n".join(result["tail"]), flush=True)

    artifacts = check_artifacts()
    manifest = dict(
        generated_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        profile=None if args.check else args.profile,
        dataset_seed=DATASET_SEED,
        train_seeds=TRAIN_SEEDS,
        python=sys.version,
        platform=platform.platform(),
        machine=platform.machine(),
        packages=package_versions(),
        steps=runs,
        artifacts=artifacts,
    )
    out = RESULTS / "q1_suite_manifest.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    missing = [a for a in artifacts if not a["exists"]]
    failed = [r for r in runs if not r["ok"]]
    print(f"\nmanifest -> {out}")
    print(f"artifacts: {len(artifacts) - len(missing)}/{len(artifacts)} present")
    if missing:
        print("missing: " + ", ".join(a["artifact"] for a in missing))
    if failed:
        print("failed steps: " + ", ".join(r["step"] for r in failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
