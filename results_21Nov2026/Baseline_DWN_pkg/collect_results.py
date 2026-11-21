"""
collect_results.py - tabulate the DWN baseline for the paper (no numbers are
invented: every value is parsed from files produced by the user's runs).

Inputs
  runs/<arch>/accuracy.csv             (train_dwn.py)
  runs/<arch>/seed*/test_scores.npz    (coverage / abstention curve)
  quartus/reports/<rev>[_*]/           (compile_and_report.tcl copies)
  vivado/out*/                         (optional Vivado runs)
Outputs (results/)
  dwn_accuracy_summary.csv   mean +- sd (sample sd, ddof=1) over seeds, per arch
  dwn_coverage.csv           accuracy at 100/90/80/70 % coverage (GroupSum margin)
  dwn_hardware.csv           one row per Quartus/Vivado report folder
  dwn_paper_table.csv        merged row(s) for Table IIb
Usage
  python collect_results.py --arch sm --board-mhz 25
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re

import numpy as np


def acc_summary(arch_dir):
    rows = list(csv.DictReader(open(os.path.join(arch_dir, "accuracy.csv"))))
    rows = [r for r in rows if r.get("label", "") in ("", None)] or rows
    a = np.array([float(r["test_acc_int"]) for r in rows])
    agree_all = all(r["int_vs_torch_agree"].split("/")[0] == r["int_vs_torch_agree"].split("/")[1] for r in rows)
    return {"arch": os.path.basename(arch_dir), "n_seeds": len(a),
            "seeds": " ".join(r["seed"] for r in rows),
            "acc_mean_pct": round(100 * a.mean(), 2),
            "acc_sd_pct": round(100 * a.std(ddof=1), 2) if len(a) > 1 else "",
            "acc_min_pct": round(100 * a.min(), 2), "acc_max_pct": round(100 * a.max(), 2),
            "export_matches_torch_all_seeds": agree_all,
            "efd_backend": ";".join(sorted({r["efd_backend"] for r in rows})),
            "device": ";".join(sorted({r["device"] for r in rows})),
            "mean_train_s": round(np.mean([float(r["train_s"]) for r in rows]), 1),
            "labels": ";".join(sorted({r.get("label", "") for r in rows}))}


def coverage(arch_dir, covs=(1.0, 0.9, 0.8, 0.7)):
    out = []
    for f in sorted(glob.glob(os.path.join(arch_dir, "seed*", "test_scores.npz"))):
        d = np.load(f)
        c, p, y = d["counts"], d["pred"], d["y"]
        s = np.sort(c, 1)
        margin = s[:, -1] - s[:, -2]                 # top1 - top2 popcount
        order = np.argsort(-margin, kind="stable")   # most confident first
        row = {"seed_file": os.path.relpath(f, arch_dir)}
        for cv in covs:
            k = int(round(cv * len(y)))
            sel = order[:k]
            row[f"acc_at_{int(cv*100)}pct"] = round(100 * float((p[sel] == y[sel]).mean()), 2)
        out.append(row)
    return out


def _num(s):
    m = re.search(r"([\d,]+(?:\.\d+)?)", s)
    return float(m.group(1).replace(",", "")) if m else None


def parse_quartus(folder, board_mhz):
    r = {"report_dir": folder, "tool": "Quartus"}
    rev = None
    for f in glob.glob(os.path.join(folder, "*.fit.summary")):
        rev = os.path.basename(f).split(".")[0]
        for line in open(f, errors="ignore"):
            k, _, v = line.partition(":")
            k = k.strip()
            if k in ("Device", "Quartus Prime Version", "Top-level Entity Name"):
                r[k] = v.strip()
            elif k in ("Total logic elements", "Total combinational functions",
                       "Dedicated logic registers", "Total registers", "Total memory bits",
                       "Embedded Multiplier 9-bit elements", "Total pins"):
                r[k] = _num(v)
    r["revision"] = rev
    fit = os.path.join(folder, f"{rev}.fit.rpt")
    if rev and os.path.exists(fit):
        for line in open(fit, errors="ignore"):
            if "|dwn_core:u_core" in line and line.strip().startswith(";") and \
                    line.split(";")[1].strip().endswith("dwn_core:u_core|"):
                cols = [c.strip() for c in line.split(";")]
                r["core_LCs"] = _num(cols[2])
                r["core_regs"] = _num(cols[3])
                r["core_mem_bits"] = _num(cols[5])
                break
    sta = os.path.join(folder, f"{rev}.sta.rpt")
    if rev and os.path.exists(sta):
        txt = open(sta, errors="ignore").read()
        for corner in ("Slow 1200mV 85C", "Slow 1200mV 0C"):
            m = re.search(re.escape(corner) + r" Model Fmax Summary.*?\n(?:.*\n){3}((?:; [\d.]+ MHz.*\n)+)", txt)
            if m:
                for line in m.group(1).splitlines():
                    cols = [c.strip() for c in line.split(";")]
                    clk = cols[3]
                    if clk in ("clk", "clk_25"):
                        r[f"fmax_mhz_{corner.replace(' ', '_')}"] = _num(cols[1])
                        r[f"restricted_fmax_mhz_{corner.replace(' ', '_')}"] = _num(cols[2])
    pw = os.path.join(folder, f"{rev}.pow.summary")
    if rev and os.path.exists(pw):
        for line in open(pw, errors="ignore"):
            k, _, v = line.partition(":")
            k = k.strip()
            if "Power" in k and "mW" in v:
                r[k] = _num(v)
            if k == "Power Estimation Confidence":
                r[k] = v.strip()
    prpt = os.path.join(folder, f"{rev}.pow.rpt")
    if rev and os.path.exists(prpt):
        for line in open(prpt, errors="ignore"):
            if "|dwn_core:u_core" in line and line.split(";")[1].strip().endswith("dwn_core:u_core"):
                cols = [c.strip() for c in line.split(";")]
                # Total (block+children) ; block dynamic ; block static ; routing dynamic
                r["core_total_thermal_mW"] = _num(cols[2])
                r["core_block_dynamic_mW"] = _num(cols[3])
                r["core_routing_dynamic_mW"] = _num(cols[5])
                break
    info = os.path.join(folder, "dwn_core_info.json")
    if os.path.exists(info):
        inf = json.load(open(info))
        r["latency_cycles"] = inf["latency_cycles"]
        r["model_sha256"] = inf["model_sha256"]
        r["lut_nodes"] = inf["total_lut_nodes"]
    f_run = board_mhz
    r["run_clock_mhz"] = f_run
    if "latency_cycles" in r:
        r["latency_ns_at_run_clock"] = round(1000.0 * r["latency_cycles"] / f_run, 2)
        fm = r.get("fmax_mhz_Slow_1200mV_85C")
        if fm:
            r["latency_ns_at_fmax"] = round(1000.0 * r["latency_cycles"] / fm, 2)
    r["throughput_inf_per_s_at_run_clock"] = f_run * 1e6        # 1 inference / clock
    dyn = r.get("core_block_dynamic_mW")
    if dyn is not None:
        dyn_core = dyn + (r.get("core_routing_dynamic_mW") or 0)
        r["core_dynamic_mW"] = round(dyn_core, 3)
        # streaming: one inference per clock -> E = P / f ; single-shot: E = P * latency / f
        r["energy_nJ_per_inf_streaming"] = round(dyn_core * 1e-3 / (f_run * 1e6) * 1e9, 4)
        if "latency_cycles" in r:
            r["energy_nJ_per_inf_latency"] = round(dyn_core * 1e-3 * r["latency_cycles"] / (f_run * 1e6) * 1e9, 4)
        r["energy_tier"] = "post-fit PowerPlay vectorless estimate (12.5% default toggle)"
    return r


def parse_vivado(folder):
    r = {"report_dir": folder, "tool": "Vivado"}
    s = os.path.join(folder, "summary.json")
    if os.path.exists(s):
        r.update(json.load(open(s)))
    u = os.path.join(folder, "utilization_impl.rpt")
    if os.path.exists(u):
        for line in open(u, errors="ignore"):
            cols = [c.strip() for c in line.split("|")]
            if len(cols) > 3 and cols[1] in ("Slice LUTs", "CLB LUTs", "Slice Registers",
                                             "CLB Registers", "Block RAM Tile", "DSPs"):
                r[cols[1]] = _num(cols[2])
    p = os.path.join(folder, "power_impl.rpt")
    if os.path.exists(p):
        for line in open(p, errors="ignore"):
            cols = [c.strip() for c in line.split("|")]
            if len(cols) > 2 and cols[1] in ("Total On-Chip Power (W)", "Dynamic (W)", "Device Static (W)"):
                r[cols[1]] = _num(cols[2])
    info = os.path.join(folder, "dwn_core_info.json")
    if os.path.exists(info):
        inf = json.load(open(info))
        r["latency_cycles"] = inf["latency_cycles"]
        r["model_sha256"] = inf["model_sha256"]
    return r


def write_csv(path, rows):
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print("wrote", path, f"({len(rows)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--arch", nargs="+", default=["sm"])
    ap.add_argument("--quartus-reports", default="quartus/reports")
    ap.add_argument("--vivado-glob", default="vivado/out*")
    ap.add_argument("--board-mhz", type=float, default=25.0, help="run clock (SENA: 25 MHz)")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    acc, cov = [], []
    for arch in a.arch:
        d = os.path.join(a.runs, arch)
        if os.path.exists(os.path.join(d, "accuracy.csv")):
            acc.append(acc_summary(d))
            for r in coverage(d):
                cov.append({"arch": arch, **r})
    hw = []
    for f in sorted(glob.glob(os.path.join(a.quartus_reports, "*"))):
        if os.path.isdir(f):
            hw.append(parse_quartus(f, a.board_mhz))
    for f in sorted(glob.glob(a.vivado_glob)):
        if os.path.isdir(f):
            hw.append(parse_vivado(f))
    if acc:
        write_csv(os.path.join(a.out, "dwn_accuracy_summary.csv"), acc)
    if cov:
        means = {}
        for r in cov:
            for k, v in r.items():
                if k.startswith("acc_at"):
                    means.setdefault((r["arch"], k), []).append(v)
        for (arch, k), v in sorted(means.items()):
            pass
        summ = []
        for arch in a.arch:
            row = {"arch": arch, "seed_file": "MEAN+-SD"}
            for (ar, k), v in sorted(means.items()):
                if ar == arch:
                    v = np.array(v)
                    row[k] = f"{v.mean():.2f} +- {v.std(ddof=1):.2f}" if len(v) > 1 else f"{v.mean():.2f}"
            summ.append(row)
        write_csv(os.path.join(a.out, "dwn_coverage.csv"), cov + summ)
    if hw:
        write_csv(os.path.join(a.out, "dwn_hardware.csv"), hw)
    table = []
    for s in acc:
        for h in hw or [{}]:
            table.append({
                "model": f"DWN ({s['arch']}, our re-implementation, same split)",
                "accuracy_pct": f"{s['acc_mean_pct']} +- {s['acc_sd_pct']}",
                "tool/revision": f"{h.get('tool','')} {h.get('revision','')}".strip(),
                "LE_or_LUT_total": h.get("Total logic elements", h.get("Slice LUTs", h.get("CLB LUTs"))),
                "core_LCs": h.get("core_LCs"), "registers": h.get("Total registers", h.get("Slice Registers")),
                "mem_bits_total": h.get("Total memory bits"), "DSP/mult": h.get("Embedded Multiplier 9-bit elements", h.get("DSPs")),
                "fmax_MHz_slow85C": h.get("fmax_mhz_Slow_1200mV_85C", h.get("fmax_mhz")),
                "latency_cycles": h.get("latency_cycles"),
                "latency_ns_at_run_clock": h.get("latency_ns_at_run_clock"),
                "core_dynamic_mW": h.get("core_dynamic_mW"),
                "energy_nJ_streaming": h.get("energy_nJ_per_inf_streaming"),
                "energy_tier": h.get("energy_tier"),
                "model_sha256_hw": h.get("model_sha256")})
    if table:
        write_csv(os.path.join(a.out, "dwn_paper_table.csv"), table)
    print("NOTE: check that model_sha256_hw matches the seed you built (one seed is built in hardware;"
          " accuracy is mean +- sd over all seeds).")


if __name__ == "__main__":
    main()
