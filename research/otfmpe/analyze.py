"""Metrics for OT-FMPE (one entry per training seed) and the Bayesian engine on the shared test cases.

For each test set and quantity (log10 mass; log10 speed; log10 kinetic energy):
  MAE and RMSE of the posterior median, bias, coverage of central 50/68/90/95 % intervals, median 90 % width,
  interval score (alpha = 0.1), calibration error (mean |empirical - nominal| coverage over 10-90 % levels).
Paired difference of absolute errors (OT-FMPE minus engine) with a 90 % bootstrap interval over cases.
Engine quantiles: the engine's reported quantiles (weighted draws) for the median and the 68/90 % intervals,
its 2,000 resampled draws for the other levels. OT-FMPE: 2,000 samples per case.
Usage: python research/otfmpe/analyze.py RUN_DIR   (RUN_DIR holds cases.json and test_samples_seed*.npz)
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.68, 0.7, 0.8, 0.9, 0.95)
QTY = {"lm": ("log10_mass_kg", 0), "lu": ("log10_velocity_m_s", 1), "le": ("log10_energy_j", None)}


def nn_samples(s, k):
    if k == "le":
        return s[:, 0] + 2.0 * s[:, 1] - math.log10(2.0)
    return s[:, QTY[k][1]]


def metrics(truth, med, lo_hi, width90, iscore):
    err = med - truth
    out = {"n": int(len(truth)), "MAE": float(np.mean(np.abs(err))), "RMSE": float(np.sqrt(np.mean(err**2))),
           "bias": float(np.mean(err)), "width90_median": float(np.median(width90)),
           "interval_score90": float(np.mean(iscore))}
    cal, curve = [], {}
    for lv in LEVELS:
        lo, hi = lo_hi[lv]
        cov = float(np.mean((truth >= lo) & (truth <= hi)))
        curve[str(lv)] = cov
        if lv in (0.5, 0.68, 0.9, 0.95):
            out[f"cov{int(round(lv * 100))}"] = cov
        cal.append(abs(cov - lv))
    out["calibration_error"] = float(np.mean(cal))
    out["coverage_curve"] = curve
    return out


def summarise(truth, samples_or_q):
    """samples_or_q: list of (median, {level: (lo, hi)})."""
    med = np.array([m for m, _ in samples_or_q])
    lo_hi = {lv: (np.array([q[lv][0] for _, q in samples_or_q]), np.array([q[lv][1] for _, q in samples_or_q]))
             for lv in LEVELS}
    lo, hi = lo_hi[0.9]
    w = hi - lo
    iscore = w + 20.0 * np.maximum(lo - truth, 0) + 20.0 * np.maximum(truth - hi, 0)
    return metrics(truth, med, lo_hi, w, iscore), med


def from_samples(x):
    return float(np.median(x)), {lv: (float(np.quantile(x, 0.5 - lv / 2)), float(np.quantile(x, 0.5 + lv / 2)))
                                 for lv in LEVELS}


def from_engine(e, name):
    q = e["q"][name]                       # q05, q16, median, q84, q95 (weighted draws)
    _, qq = from_samples(np.asarray(e["samples"][name]))
    qq[0.68], qq[0.9] = (q[1], q[3]), (q[0], q[4])
    return q[2], qq


def bootstrap_diff(a, b, n=2000, seed=0):
    d = np.abs(a) - np.abs(b)
    rng = np.random.default_rng(seed)
    bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return {"mean": float(d.mean()), "ci90": [float(np.quantile(bs, 0.05)), float(np.quantile(bs, 0.95))]}


def subsets(cases, name):
    idx = [i for i, c in enumerate(cases) if c["set"] == name]
    out = {"all": idx}
    if name in ("nat_D", "art_D"):
        for lab, lo, hi in (("D 5-28 m", 5, 28.2), ("D 28-281 m", 28.2, 281.2), ("D 281-5000 m", 281.2, 5001)):
            out[lab] = [i for i in idx if lo <= 10 ** cases[i]["y"] < hi]
        out["mare site (p_mare=1)"] = [i for i in idx if cases[i]["p_mare"] == 1.0]
        out["highland site (p_mare=0)"] = [i for i in idx if cases[i]["p_mare"] == 0.0]
        out["sigma 0.01-0.05"] = [i for i in idx if cases[i]["sig"] < 0.05]
        out["sigma 0.10-0.15"] = [i for i in idx if cases[i]["sig"] >= 0.10]
    return out


def main(run):
    run = Path(run)
    cases = json.loads((run / "cases.json").read_text())
    seeds = sorted(int(p.stem.split("seed")[1]) for p in run.glob("test_samples_seed*.npz"))
    nn = {s: np.load(run / f"test_samples_seed{s}.npz")["samples"] for s in seeds}
    report = {"seeds": seeds, "sets": {}}
    for name in dict.fromkeys(c["set"] for c in cases):
        report["sets"][name] = {}
        for sub, idx in subsets(cases, name).items():
            if not idx:
                continue
            block = {}
            for k, (ename, _) in QTY.items():
                if any(cases[i]["truth"].get(k) is None for i in idx):
                    continue
                truth = np.array([cases[i]["truth"][k] for i in idx])
                em, emed = summarise(truth, [from_engine(cases[i]["engine"], ename) for i in idx])
                per_seed, meds = {}, {}
                for s in seeds:
                    m, med = summarise(truth, [from_samples(nn_samples(nn[s][i], k)) for i in idx])
                    per_seed[s], meds[s] = m, med
                keys = [x for x in per_seed[seeds[0]] if x not in ("n", "coverage_curve")] if seeds else []
                nn_mean = {x: float(np.mean([per_seed[s][x] for s in seeds])) for x in keys}
                nn_sd = {x: float(np.std([per_seed[s][x] for s in seeds], ddof=1)) if len(seeds) > 1 else 0.0 for x in keys}
                block[k] = {"engine": em, "otfmpe_mean": nn_mean, "otfmpe_sd_over_seeds": nn_sd,
                            "otfmpe_per_seed": {str(s): per_seed[s] for s in seeds},
                            "paired_abs_err_diff_seed_mean": bootstrap_diff(
                                np.mean([meds[s] for s in seeds], 0) - truth, emed - truth) if seeds else None}
            eng = [cases[i]["engine"] for i in idx]
            block["engine_reliable_share"] = float(np.mean([e["reliable"] for e in eng]))
            block["engine_seconds_median"] = float(np.median([e["seconds"] for e in eng]))
            report["sets"][name][sub] = block
    (run / "metrics.json").write_text(json.dumps(report, indent=1))
    for name, subs in report["sets"].items():
        b = subs["all"]["lm"]
        e, n, sd = b["engine"], b["otfmpe_mean"], b["otfmpe_sd_over_seeds"]
        d = b["paired_abs_err_diff_seed_mean"]
        print(f"{name:11s} n={e['n']:4d} | engine MAE {e['MAE']:.3f} RMSE {e['RMSE']:.3f} bias {e['bias']:+.3f} cov90 {e['cov90']:.2f}"
              f" | OT-FMPE MAE {n['MAE']:.3f}±{sd['MAE']:.3f} RMSE {n['RMSE']:.3f} bias {n['bias']:+.3f} cov90 {n['cov90']:.2f}"
              f" | diff {d['mean']:+.3f} [{d['ci90'][0]:+.3f},{d['ci90'][1]:+.3f}]")


if __name__ == "__main__":
    main(sys.argv[1])
