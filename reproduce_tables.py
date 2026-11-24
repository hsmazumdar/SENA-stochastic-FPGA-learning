#!/usr/bin/env python3
"""Regenerate the SENA V13d manuscript tables (and Fig. 2) from the saved
results in this repository. No FPGA, no training, no GPU needed.

Usage:  python reproduce_tables.py            (Windows: reproduce_tables.bat)
Output: reproduce_output/run_<YYYYmmdd_HHMMSS>/
          tables.md          regenerated Tables I-VII and Sec. V.F/V.G/VI values
          check.csv          every regenerated value vs the number printed in the paper
          fig2_risk_coverage.png   (if matplotlib is installed)
Exit code 0 = every value matches the manuscript at its printed precision.

Requires Python 3.8+ (standard library only; matplotlib optional for the figure).
"""
import csv, glob, json, os, re, statistics as st, sys, time

ROOT = os.path.dirname(os.path.abspath(__file__))
R21 = os.path.join(ROOT, "results_21Nov2026")
P = lambda *a: os.path.join(ROOT, *a)

rows = []      # (item, quantity, regenerated, paper, source, ok)
md = []        # markdown lines


def dp(s):
    s = str(s)
    return len(s.split(".")[1]) if "." in s else 0


def check(item, qty, value, paper, src):
    """Compare value with the paper's printed number at the paper's precision."""
    k = dp(paper)
    shown = f"{value:.{k}f}"
    ok = abs(float(shown) - float(paper)) < 1e-9
    rows.append((item, qty, shown, str(paper), os.path.relpath(src, ROOT).replace("\\", "/"), "OK" if ok else "MISMATCH"))
    return shown


def jload(p):
    with open(p, encoding="utf-8") as f:
        t = f.read()
    return json.loads(t[t.index("{") if t.lstrip()[0] != "[" else t.index("["):])


def csvrows(p):
    with open(p, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def rpt_field(p, pattern):
    with open(p, encoding="utf-8", errors="ignore") as f:
        m = re.search(pattern, f.read())
    if not m:
        raise ValueError(f"{pattern!r} not found in {p}")
    return float(m.group(1).replace(",", ""))


def mean_sd(xs):
    return st.mean(xs), (st.stdev(xs) if len(xs) > 1 else 0.0)


# ---------------------------------------------------------------- Table I
md.append("## Table I: same-width MNIST baselines and SENA (M = 3)\n")
md.append("| Method | Acc. (%) | Passes |\n|---|---|---|")
src = os.path.join(R21, "baselines_V12", "tuned_baselines_merged.json")
b = jload(src)["models"]
for key, name, pm, psd in [("fp32_mlp", "FP32 MLP", "95.97", "0.15"), ("int8_ptq", "INT8 PTQ", "95.98", "0.16"),
                           ("binaryconnect", "BinaryConnect", "95.13", "0.31"), ("xnor_bnn", "XNOR-BNN", "94.48", "0.24")]:
    m, s = mean_sd([a * 100 for a in b[key]["test_acc"]])
    md.append(f"| {name} | {check('Table I', name+' acc', m, pm, src)} ± {check('Table I', name+' sd', s, psd, src)} | – |")

src = os.path.join(R21, "restored_results", "q1_sena_ablations.json")
runs = jload(src)["runs"]
src2 = os.path.join(R21, "restored_results", "q1_fixedN_fullT_EE.json")
runs2 = jload(src2)["runs"]
def grp(rs, cond):
    sel = [r for r in rs if r["condition"] == cond]
    return [r["accuracy"] * 100 for r in sel], [r["mean_passes"] for r in sel]
for label, rs, cond, pm, psd, pp, s_ in [
        ("SENA fixed range + EE", runs, "ensemble_fixed_N", "90.28", "0.52", "2.30", src),
        ("SENA fixed range, full T=40", runs2, "fixedN_full_T40", "90.76", "0.39", None, src2),
        ("SENA adaptive + EE", runs, "ensemble_early_exit", "84.42", "1.38", "5.11", src),
        ("SENA adaptive, full T=40", runs, "ensemble_full", "87.33", "0.97", None, src)]:
    a, p = grp(rs, cond)
    m, s = mean_sd(a)
    ps = check("Table I", label + " passes", st.mean(p), pp, s_) if pp else f"{st.mean(p):.0f}"
    md.append(f"| {label} | {check('Table I', label+' acc', m, pm, s_)} ± {check('Table I', label+' sd', s, psd, s_)} | {ps} |")
check("Table I", "INT8 storage bits = 127040*8 + 170*32", 127040 * 8 + 170 * 32, "1021760", src)
check("Table I", "FP32 storage bits = 127210*32", 127210 * 32, "4070720", src)
check("Table I", "SENA M=3 storage 16-bit = 3*127210*16", 3 * 127210 * 16, "6106080", src)
check("Table I", "SENA M=3 storage 8-bit", 3 * 127210 * 8, "3053040", src)

# ---------------------------------------------------------------- Table II
md.append("\n## Table II: same-board comparison on the DE2-115\n")
src = os.path.join(R21, "Baseline_DWN_pkg", "results", "dwn_paper_table.csv")
d = [r for r in csvrows(src) if "de2_115_dwn" in r["tool/revision"]][0]
src_acc = os.path.join(R21, "Baseline_DWN_pkg", "results", "dwn_accuracy_summary.csv")
a = csvrows(src_acc)[0]
md.append("| Quantity | DWN sm | SENA slice 784-16-10 | SENA learning core 784-160-10 |\n|---|---|---|---|")
dwn = dict(acc=check("Table II", "DWN acc", float(a["acc_mean_pct"]), "95.31", src_acc),
           sd=check("Table II", "DWN sd", float(a["acc_sd_pct"]), "0.19", src_acc),
           le=check("Table II", "DWN LE", float(d["LE_or_LUT_total"]), "8489", src),
           core=check("Table II", "DWN core LCs", float(d["core_LCs"]), "8095", src),
           fmax=check("Table II", "DWN Fmax MHz", float(d["fmax_MHz_slow85C"]), "71.38", src),
           lat=check("Table II", "DWN latency ns", float(d["latency_ns_at_run_clock"]), "200", src),
           pw=check("Table II", "DWN core dynamic mW", float(d["core_dynamic_mW"]), "78.73", src),
           e=check("Table II", "DWN energy nJ", float(d["energy_nJ_streaming"]), "3.15", src))
src_hw = os.path.join(R21, "Baseline_DWN_pkg", "results", "dwn_hardware.csv")
h = [r for r in csvrows(src_hw) if r["revision"] == "de2_115_dwn"][0]
dwn["pw_all"] = check("Table II", "DWN whole-design core dynamic mW", float(h["Core Dynamic Thermal Power Dissipation"]), "95.32", src_hw)

cal = P("sena2026", "results", "de2_115_calibration_C1", "reports")
s_le = check("Table II", "slice LE", rpt_field(os.path.join(cal, "de2_115_calib.fit.summary"), r"Total logic elements : ([\d,]+)"), "3749", cal)
s_mem = check("Table II", "slice memory bits", rpt_field(os.path.join(cal, "de2_115_calib.fit.summary"), r"Total memory bits : ([\d,]+)"), "203680", cal)
s_f = check("Table II", "slice Fmax MHz", rpt_field(os.path.join(cal, "de2_115_calib.sta.rpt"), r"85C Model Fmax Summary[\s\S]*?\n; ([\d.]+) MHz"), "31.09", cal)
pj = P("sena2026", "rtl", "_power", "power.json")
pw = jload(pj)
s_p = check("Table II", "slice core dynamic mW", pw["dynamic_mw"], "12.73", pj)
agj = os.path.join(R21, "DE2_115_HW_21Nov2026", "agreement", "run_20261121_121846", "agreement.json")
ag = jload(agj)
passes = check("Table II", "slice board mean passes", ag["board_mean_passes"], "39.42", agj)
lat_us = ag["board_mean_passes"] * 807 / 25.0
s_lat = check("Table II", "slice latency us = 39.42*807/25MHz", lat_us, "1273", agj)
s_e = check("Sec. V.G", "slice energy uJ = 12.73mW*39.42*807/25MHz", pw["dynamic_mw"] * ag["board_mean_passes"] * 807 / 25e6 * 1e3, "16.2", pj)
check("Sec. V.G", "slice total thermal uJ", pw["total_mw"] * ag["board_mean_passes"] * 807 / 25e6 * 1e3, "211", pj)

bq = os.path.join(R21, "OnChip_Learning", "board_v1", "quartus", "output_files")
c_le = check("Table II/V", "core LE", rpt_field(os.path.join(bq, "de2_115_sena.fit.summary"), r"Total logic elements : ([\d,]+)"), "50351", bq)
c_reg = check("Table V", "core registers", rpt_field(os.path.join(bq, "de2_115_sena.fit.summary"), r"Total registers : ([\d,]+)"), "19991", bq)
c_mem = check("Table II", "core memory bits", rpt_field(os.path.join(bq, "de2_115_sena.fit.summary"), r"Total memory bits : ([\d,]+)"), "1017680", bq)
c_m9k = check("Table II/V", "core M9K blocks", rpt_field(os.path.join(bq, "de2_115_sena.fit.rpt"), r"; M9Ks\s*; ([\d,]+)"), "146", bq)
c_mul = check("Table II/V", "core embedded multipliers", rpt_field(os.path.join(bq, "de2_115_sena.fit.summary"), r"Embedded Multiplier 9-bit elements : ([\d,]+)"), "0", bq)
c_f = check("Table II/V", "core Fmax MHz", rpt_field(os.path.join(bq, "de2_115_sena.sta.rpt"), r"85C Model Fmax Summary[\s\S]*?\n; ([\d.]+) MHz"), "34.77", bq)
c_sl = check("Table V", "core setup slack ns (clk25)", rpt_field(os.path.join(bq, "de2_115_sena.sta.rpt"), r"85C Model Setup Summary[\s\S]*?; clk25\s*; ([\d.]+)"), "11.24", bq)
c_pt = check("Table V", "core PowerPlay total mW", rpt_field(os.path.join(bq, "de2_115_sena.pow.summary"), r"Total Thermal Power Dissipation : ([\d.]+)"), "404.24", bq)
c_pd = check("Table II/V", "core PowerPlay core dynamic mW", rpt_field(os.path.join(bq, "de2_115_sena.pow.summary"), r"Core Dynamic Thermal Power Dissipation : ([\d.]+)"), "213.59", bq)

boards = sorted(glob.glob(os.path.join(R21, "DE2_115_HW_21Nov2026", "onchip_learning", "board_h160_s*", "result.json")))
br = [jload(p) for p in boards]
cyc = [r["test_core_cycles"] / r["n_test"] for r in br]
c_cyc = check("Table II", "core cycles per test image (T=40)", st.mean(cyc), "7389", boards[0])
c_lat = check("Table II", "core latency us at 25 MHz", st.mean(cyc) / 25.0, "296", boards[0])
c_e = check("Table II", "core energy uJ = 213.59mW*296us", float(c_pd) * st.mean(cyc) / 25.0 * 1e-3, "63", boards[0])
md += [f"| Test accuracy (%) | {dwn['acc']} ± {dwn['sd']} | – | see Table IV |",
       f"| Logic elements | {dwn['le']} (core {dwn['core']}) | {s_le} | {c_le} |",
       f"| On-chip memory | 0 bits in core | {s_mem} bits | {c_m9k} M9K ({c_mem} bits) |",
       f"| Embedded multipliers | 0 | 0 | {c_mul} |",
       f"| Fmax slow 85C (MHz) | {dwn['fmax']} | {s_f} | {c_f} |",
       f"| Latency at 25 MHz | {dwn['lat']} ns | {s_lat} us ({passes} x 807 cycles) | {c_lat} us ({c_cyc} cycles) |",
       f"| Core dynamic power (mW) | {dwn['pw']} (whole {dwn['pw_all']}) | {s_p} | {c_pd} |",
       f"| Energy / inference | {dwn['e']} nJ | {s_e} uJ | {c_e} uJ |"]

# ---------------------------------------------------------------- Table III / Fig. 2
md.append("\n## Table III: selective prediction and calibration\n")
src = os.path.join(R21, "analysis_V13b", "calibration", "V13b_calibration_summary.csv")
cs = {r["model"]: r for r in csvrows(src)}
md.append("| Model | Passes | Acc 100% | Acc 90% | Acc 80% | ECE (%) | Brier |\n|---|---|---|---|---|---|---|")
paper = {"SENA full-budget (T=40)": ("40", "90.76", "0.39", "95.26", "0.46", "97.49", "0.28", "5.69", "1.60", "0.152", "0.009"),
         "SENA gated (theta=0.55, Tmax=40)": ("2.23", "90.40", "0.39", "93.10", "0.46", "94.59", "0.50", "8.37", "0.77", "0.187", "0.010"),
         "SENA single pass (T=1)": ("1", "88.54", "0.74", "93.31", "0.69", "94.92", "0.45", "8.15", "0.95", "0.197", "0.013"),
         "DWN sm (margin rank; softmax calib.)": (None, "95.31", "0.19", "98.73", "0.15", "99.59", "0.13", "1.89", "0.12", "0.070", "0.002")}
for name, pv in paper.items():
    r = cs[name]
    f = lambda col, scale, pval: check("Table III", f"{name} {col}", float(r[col]) * scale, pval, src)
    ps = f("mean_passes_mean", 1, pv[0]) if pv[0] else "–"
    md.append(f"| {name} | {ps} | {f('acc_at_100cov_mean',100,pv[1])} ± {f('acc_at_100cov_sd',100,pv[2])} | "
              f"{f('acc_at_90cov_mean',100,pv[3])} ± {f('acc_at_90cov_sd',100,pv[4])} | {f('acc_at_80cov_mean',100,pv[5])} ± {f('acc_at_80cov_sd',100,pv[6])} | "
              f"{f('ece15_mean',100,pv[7])} ± {f('ece15_sd',100,pv[8])} | {f('brier_mean',1,pv[9])} ± {f('brier_sd',1,pv[10])} |")
g = cs["SENA gated (theta=0.55, Tmax=40)"]
check("Sec. V.C", "gated passes sd", float(g["mean_passes_sd"]), "0.31", src)
check("Sec. V.C", "gated vs full disagreement %", float(g["gated_vs_full_disagree_mean"]) * 100, "2.54", src)
check("Sec. V.C", "pass-1 answered %", float(g["exit_pass1_frac_mean"]) * 100, "89.46", src)
check("Sec. V.C", "pass-1 answered accuracy %", float(g["acc_on_pass1_exits_mean"]) * 100, "93.65", src)
check("Sec. V.C", "full-budget mean confidence", float(cs["SENA full-budget (T=40)"]["mean_conf_mean"]), "0.853", src)

curves = os.path.join(R21, "analysis_V13b", "calibration", "V13b_risk_coverage_curves.csv")

# ---------------------------------------------------------------- Table IV
md.append("\n## Table IV: on-chip learning, 784-160-10 (board = DE2-115, twin = bit-exact model)\n")
md.append("| Seed | Twin T=40 | Twin T=8 | Twin T=1 | Board T=40 | Board vs twin |\n|---|---|---|---|---|---|")
tw_dir = os.path.join(R21, "OnChip_Learning", "results")
pap4 = {0: ("83.08", "82.28", "73.96", "83.08", "2077"), 1: ("82.96", "82.44", "73.04", "82.96", "2074"), 2: ("86.20", "84.88", "76.16", "86.20", "2155")}
cols = {k: [] for k in ("t40", "t8", "t1", "b")}
for r, pth in zip(br, boards):
    s = r["seed"]
    tj = os.path.join(tw_dir, f"twin_full_h160_s{s}.json")
    t = jload(tj)
    pv = pap4[s]
    v = [check("Table IV", f"seed {s} twin T=40", t["test_acc_T40"] * 100, pv[0], tj),
         check("Table IV", f"seed {s} twin T=8", t["test_acc_T8_readout"] * 100, pv[1], tj),
         check("Table IV", f"seed {s} twin T=1", t["test_acc_T1_readout"] * 100, pv[2], tj),
         check("Table IV", f"seed {s} board T=40", r["board_acc"] * 100, pv[3], pth),
         check("Table IV", f"seed {s} board correct", r["board_correct"], pv[4], pth)]
    for k, x in zip(("t40", "t8", "t1", "b"), (t["test_acc_T40"], t["test_acc_T8_readout"], t["test_acc_T1_readout"], r["board_acc"])):
        cols[k].append(x * 100)
    exact = r["BIT_EXACT"] and r["pred_mismatch"] == 0 and r["w1_mismatch"] == 0 and r["w2_mismatch"] == 0
    check("Table IV", f"seed {s} BIT_EXACT (1=yes)", int(exact), "1", pth)
    check("Table IV", f"seed {s} W1 total", r["w1_total"], "125600", pth)
    check("Table IV", f"seed {s} W2 total", r["w2_total"], "1610", pth)
    md.append(f"| {s} | {v[0]} | {v[1]} | {v[2]} | {v[3]} ({v[4]}/2500) | {'BIT_EXACT' if exact else 'MISMATCH'}: {r['pred_mismatch']} pred., {r['w1_mismatch']}/{r['w1_total']} W1, {r['w2_mismatch']}/{r['w2_total']} W2 |")
ms = {k: mean_sd(v) for k, v in cols.items()}
pm = {"t40": ("84.08", "1.84"), "t8": ("83.20", "1.46"), "t1": ("74.39", "1.60"), "b": ("84.08", "1.84")}
out = {k: f"{check('Table IV', k+' mean', ms[k][0], pm[k][0], boards[0])} ± {check('Table IV', k+' sd', ms[k][1], pm[k][1], boards[0])}" for k in ms}
md.append(f"| Mean ± sd | {out['t40']} | {out['t8']} | {out['t1']} | {out['b']} | BIT_EXACT on {len(br)}/{len(br)} seeds |")

# ---------------------------------------------------------------- Table V
md.append("\n## Table V: verification of the learning core\n")
src = os.path.join(R21, "OnChip_Learning", "results", "e2e_summary.json")
e2e = jload(src)
full = [x for x in e2e if x["name"] == "e2e_h160_1epoch_s1"][0]
check("Table V", "full epoch W1 mismatches", full["w1_mismatch"], "0", src)
check("Table V", "full epoch W2 mismatches", full["w2_mismatch"], "0", src)
check("Table V", "full epoch pred mismatches", full["pred_mismatch"], "0", src)
check("Table V", "full epoch train samples", full["n_train"], "3000", src)
check("Table V", "full epoch test images", full["n_preds"], "250", src)
check("Table V", "full epoch RTL accuracy %", full["rtl_acc_on_inferred"] * 100, "70.4", src)
check("Table V", "full epoch twin accuracy %", full["twin_acc_on_inferred"] * 100, "70.4", src)
others = [x for x in e2e if not x["name"].startswith("mut_")]
check("Table V", "non-mutant runs all bit-exact (1=yes)", int(all(x["BIT_EXACT"] for x in others)), "1", src)
mut = sorted(glob.glob(os.path.join(R21, "OnChip_Learning", "mutants", "*", "run.txt")))
mm = [jload(p) for p in mut]
check("Table V", "mutants run", len(mm), "5", mut[0])
check("Table V", "mutants detected (not BIT_EXACT)", sum(1 for x in mm if not x["BIT_EXACT"]), "5", mut[0])
w1 = [x["w1_mismatch"] for x in mm if x.get("w1_mismatch")]
check("Table V", "min W1 mismatches over mutants", min(w1), "4687", mut[0])
check("Table V", "max W1 mismatches over mutants", max(w1), "12088", mut[0])
md.append(f"- Full epoch RTL vs twin: {full['w1_mismatch']}/{full['w1_total']} W1, {full['w2_mismatch']}/{full['w2_total']} W2, "
          f"{full['pred_mismatch']}/{full['n_preds']} predictions; both {full['rtl_acc_on_inferred']*100:.1f}%")
md.append(f"- Mutants: {len(mm)} run, {sum(1 for x in mm if not x['BIT_EXACT'])} detected; W1 mismatches {min(w1)}–{max(w1)} (one caught by assertion)")
md.append(f"- DE2-115 build: {c_le} LE, {c_reg} registers, {c_m9k} M9K, {c_mul}/532 multipliers; Fmax {c_f} MHz; slack +{c_sl} ns; PowerPlay {c_pt} mW total, {c_pd} mW core dynamic")

# ---------------------------------------------------------------- Table VI
md.append("\n## Table VI: learning-rule ablation (software)\n")
src = os.path.join(R21, "analysis_V13b", "ablation", "V13b_ablation_summary.csv")
ab = {r["rule"]: r for r in csvrows(src)}
md.append("| Rule | Test T=40 | Test T=1 | Hidden-update samples (%) | W1 at ±127 (%) |\n|---|---|---|---|---|")
pap6 = {"sena": ("84.08", "1.84", "74.39", "16.5", "0.00"), "fa": ("78.29", "1.47", "67.29", "49.4", "2.79"),
        "signsgd": ("77.84", "2.50", "55.01", "63.2", "1.83"), "drtp": ("65.37", "8.97", "54.81", "58.4", "34.43"),
        "outonly": ("33.99", "8.65", "18.25", "0.0", "0.00")}
for k, pv in pap6.items():
    r = ab[k]
    v = [check("Table VI", f"{k} test T=40", float(r["test_T40_mean"]), pv[0], src),
         check("Table VI", f"{k} sd", float(r["test_T40_sd"]), pv[1], src),
         check("Table VI", f"{k} test T=1", float(r["test_T1_mean"]), pv[2], src),
         check("Table VI", f"{k} hidden-update % (= updates / 120000)", float(r["w1_update_samples_mean"]) / 120000 * 100, pv[3], src),
         check("Table VI", f"{k} W1 saturated %", float(r["w1_saturated_pct_mean"]), pv[4], src)]
    md.append(f"| {k} | {v[0]} ± {v[1]} | {v[2]} | {v[3]} | {v[4]} |")
src = os.path.join(R21, "analysis_V13b", "ablation", "V13b_ablation_all_settings.csv")
fa1 = [r for r in csvrows(src) if r["rule"] == "fa" and float(r["update_rate"]) == 1.0][0]
check("Table VI caption", "FA p=1 test", float(fa1["test_T40_mean"]), "79.53", src)
check("Table VI caption", "FA p=1 sd", float(fa1["test_T40_sd"]), "3.16", src)

# ---------------------------------------------------------------- Table VII
md.append("\n## Table VII: dual-FPGA cycle validation (784-16-10 slice)\n")
pj = P("papilio_calibration_C1", "papilio_calibration.json")
dj = os.path.join(R21, "DE2_115_UART_C1_21Nov2026", "de2_115_calibration.json")
pa, de = jload(pj), jload(dj)
for nm, j, pth in (("Papilio Pro", pa, pj), ("DE2-115", de, dj)):
    cpp = [x["cycles_per_pass"] for x in j["fixed_pass_runs"]]
    check("Table VII", f"{nm} cycles/pass (all T)", min(cpp), "807.00", pth)
    check("Table VII", f"{nm} cycles/pass max", max(cpp), "807.00", pth)
    check("Table VII", f"{nm} model error", j["model_error_cycles"], "0.0", pth)
    md.append(f"- {nm}: cycles/pass {min(cpp):.2f}–{max(cpp):.2f} for T = {[x['t_max'] for x in j['fixed_pass_runs']]}; model error {j['model_error_cycles']}")
twr = glob.glob(P("papilio_calibration_C1", "ise_reports", "*.twr"))
if twr:
    try:
        check("Table VII", "Papilio Fmax MHz (TWR)", rpt_field(twr[0], r"Maximum frequency:\s*([\d.]+)\s*MHz"), "32.601", twr[0])
    except ValueError:
        rows.append(("Table VII", "Papilio Fmax MHz (TWR)", "not parsed", "32.601", os.path.relpath(twr[0], ROOT).replace("\\", "/"), "MISMATCH"))
check("Table VII", "DE2-115 board/model prediction agreement %", ag["pred_agree_pct"], "100", agj)
check("Table VII", "DE2-115 agreement N", ag["N"], "2500", agj)
check("Sec. V.F", "slice accuracy % (board = model)", ag["board_acc_pct"], "29.12", agj)
md.append(f"- DE2-115 board vs model: {ag['pred_agree_pct']}% of {ag['N']} predictions, passes and cycles agree; slice accuracy {ag['board_acc_pct']}%, mean passes {ag['board_mean_passes']}")

# ---------------------------------------------------------------- Sec. V.F / V.G / VI
md.append("\n## Other quantitative statements\n")
src = P("sena2026", "results", "q1_fpga_synth.json")
fs = jload(src)["folded_artix7_design"]
check("Sec. V.F", "Artix-7 LUTs", fs["lut"], "5392", src)
check("Sec. V.F", "Artix-7 LUT %", fs["lut_pct"], "8.5", src)
check("Sec. V.F", "Artix-7 FFs", fs["ff"], "2160", src)
check("Sec. V.F", "BRAM36 tiles (one member, 16-bit)", fs["bram"], "55.2", src)
check("Sec. V.F", "BRAM %", fs["bram_pct"], "40.9", src)
check("Sec. V.F", "3 members 16-bit tiles (3 x 55.2)", 3 * round(fs["bram"], 1), "165.6", src)
check("Sec. V.F", "3 members 8-bit tiles (1.5 x 55.2)", 1.5 * round(fs["bram"], 1), "82.8", src)
src = P("Sena_Qi_Paper", "results", "q1_papilio_model_energy.json")
me = jload(src)["full_mnist_model"]["full_par8_M3"]
check("Sec. V.G", "model energy per ensemble pass nJ", me["energy_per_ensemble_pass_pJ"] / 1000, "48.54", src)
check("Sec. V.G", "fixed-range EE energy uJ (48.54 nJ x 2.30)", round(me["energy_per_ensemble_pass_pJ"] / 1000, 2) * 2.30 / 1000, "0.112", src)
check("Sec. V.G", "adaptive EE energy uJ", me["energy_per_inference_ee_uJ"], "0.248", src)
check("Sec. V.G", "full T=40 energy uJ", me["energy_per_inference_full40_uJ"], "1.94", src)
src = os.path.join(R21, "baseline_dwn", "results_treelut", "treelut_sena_train.json")
tl = jload(src)["runs"]
check("Sec. VI", "XGBoost test accuracy %", st.mean(r["accuracy"] for r in tl) * 100, "94.68", src)
check("Sec. VI", "TreeLUT accuracy %", [r for r in tl if "treelut_acc" in r][0]["treelut_acc"] * 100, "93.88", src)
src = P("sena2026", "results", "q1_sensor_adc.json")
adc = jload(src)
check("Sec. VI", "ADC clean accuracy %", adc["clean_accuracy"] * 100, "85.9", src)
def adc_acc(t, f):
    return [r for r in adc["runs"] if r["sigma_t"] == t and r["sigma_fpn"] == f][0]["accuracy_mean"] * 100
check("Sec. VI", "ADC sigma_t=0.15", adc_acc(0.15, 0.0), "74.2", src)
check("Sec. VI", "ADC sigma_t=0.20", adc_acc(0.2, 0.0), "43.0", src)
check("Sec. VI", "ADC sigma_fpn=0.20", adc_acc(0.0, 0.2), "45.7", src)

# ---------------------------------------------------------------- write
out_dir = os.path.join(ROOT, "reproduce_output", time.strftime("run_%Y%m%d_%H%M%S"))
os.makedirs(out_dir, exist_ok=False)
nbad = sum(1 for r in rows if r[5] != "OK")
head = [f"# SENA V13d: tables regenerated from saved results ({time.strftime('%Y-%m-%d %H:%M:%S')})\n",
        f"**{len(rows)} values checked against the manuscript: {len(rows) - nbad} match, {nbad} mismatch.**\n",
        "No hardware, training or GPU was used; every value is read from the result files named in `check.csv`.\n"]
with open(os.path.join(out_dir, "tables.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(head + md) + "\n")
with open(os.path.join(out_dir, "check.csv"), "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["item", "quantity", "regenerated", "manuscript", "source", "status"])
    w.writerows(rows)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cr = csvrows(curves)
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    for m in dict.fromkeys(r["model"] for r in cr):
        pts = [(float(r["coverage"]), float(r["risk_mean"]) * 100, float(r["risk_sd"]) * 100) for r in cr if r["model"] == m]
        x, y, s = zip(*pts)
        ax.plot(x, y, label=m)
        ax.fill_between(x, [a - b for a, b in zip(y, s)], [a + b for a, b in zip(y, s)], alpha=0.2)
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Error on answered inputs (%)")
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig2_risk_coverage.png"), dpi=200)
    fig_msg = "fig2_risk_coverage.png written"
except ImportError:
    fig_msg = "matplotlib not installed: Fig. 2 skipped (tables unaffected)"

print(f"Output folder: {out_dir}")
print(f"{len(rows)} values checked: {len(rows) - nbad} match the manuscript, {nbad} mismatch. {fig_msg}")
for r in rows:
    if r[5] != "OK":
        print("MISMATCH:", r)
sys.exit(1 if nbad else 0)
