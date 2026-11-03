import json, math, statistics as st
from pathlib import Path

board = json.load(open(r"D:\_July2026\SENA_Project\comfort_poc\results\comfort_board_results.json"))
soft = json.load(open(r"D:\_July2026\SENA_Project\comfort_poc\results\comfort_train_results.json"))
s = board["samples"]
ps = [x["p_star"] for x in s]
ph = [x["p_hat"] for x in s]
ae = [x["abs_err"] for x in s]
bias = [x["p_hat"] - x["p_star"] for x in s]


def pearson(a, b):
    n = len(a)
    ma = sum(a) / n
    mb = sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = math.sqrt(sum((x - ma) ** 2 for x in a))
    db = math.sqrt(sum((y - mb) ** 2 for y in b))
    return num / (da * db) if da and db else float("nan")


bins = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
bin_stats = []
for lo, hi in bins:
    xs = [x for x in s if lo <= x["p_star"] < hi]
    if not xs:
        continue
    bin_stats.append(
        {
            "range": f"{lo:.1f}-{min(hi, 1.0):.1f}",
            "n": len(xs),
            "mae": sum(x["abs_err"] for x in xs) / len(xs),
            "mean_pstar": sum(x["p_star"] for x in xs) / len(xs),
            "mean_phat": sum(x["p_hat"] for x in xs) / len(xs),
            "mean_bias": sum(x["p_hat"] - x["p_star"] for x in xs) / len(xs),
        }
    )


def cls(p, thr=0.5):
    return 1 if p >= thr else 0


agree = sum(1 for x in s if cls(x["p_hat"]) == cls(x["p_star"]))
spreads = [max(x["p_members"]) - min(x["p_members"]) for x in s]
worst = sorted(s, key=lambda x: -x["abs_err"])[:8]
best = sorted(s, key=lambda x: x["abs_err"])[:5]
rmse = math.sqrt(sum(e * e for e in ae) / len(ae))
corr = pearson(ps, ph)

summary = {
    "board_mae": board["mae"],
    "board_rmse": rmse,
    "soft_mae": soft["test_mae"],
    "soft_rmse": soft["test_rmse"],
    "soft_t_eval": soft["t_eval"],
    "board_t": board["t_passes"],
    "corr": corr,
    "mean_bias": sum(bias) / len(bias),
    "median_ae": st.median(ae),
    "cls_agree_0_5": agree / len(s),
    "bins": bin_stats,
    "within": {str(e): sum(1 for a in ae if a <= e) / len(ae) for e in (0.1, 0.2, 0.3, 0.4)},
    "phat_stats": {
        "mean": sum(ph) / len(ph),
        "std": st.pstdev(ph),
        "min": min(ph),
        "max": max(ph),
    },
    "pstar_stats": {
        "mean": sum(ps) / len(ps),
        "std": st.pstdev(ps),
        "min": min(ps),
        "max": max(ps),
    },
    "member_spread": {
        "mean": sum(spreads) / len(spreads),
        "median": st.median(spreads),
        "max": max(spreads),
    },
    "samples": [
        {
            "T": round(x["T"], 2),
            "RH": round(x["RH"], 2),
            "p_star": round(x["p_star"], 4),
            "p_hat": round(x["p_hat"], 4),
            "abs_err": round(x["abs_err"], 4),
            "spread": round(max(x["p_members"]) - min(x["p_members"]), 3),
            "p_members": x["p_members"],
        }
        for x in s
    ],
    "worst": [
        {
            "T": round(x["T"], 1),
            "RH": round(x["RH"], 1),
            "p_star": round(x["p_star"], 3),
            "p_hat": round(x["p_hat"], 3),
            "abs_err": round(x["abs_err"], 3),
            "p_members": x["p_members"],
        }
        for x in worst
    ],
    "best": [
        {
            "T": round(x["T"], 1),
            "RH": round(x["RH"], 1),
            "p_star": round(x["p_star"], 3),
            "p_hat": round(x["p_hat"], 3),
            "abs_err": round(x["abs_err"], 3),
        }
        for x in best
    ],
}
out = Path(r"D:\_July2026\SENA_Project\comfort_poc\results\_analysis_summary.json")
out.write_text(json.dumps(summary, indent=2))
print(json.dumps({k: summary[k] for k in summary if k not in ("samples", "worst", "best")}, indent=2))
print("worst", summary["worst"][:3])
print("wrote", out)
