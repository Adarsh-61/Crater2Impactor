"""Training and validation data for OT-FMPE, drawn from the engine's current generative model.

`truth` is `scripts/run_benchmarks_v2.simulate_truth` (the benchmark truth generator, which uses the forward
model's coupled target-class rule `forward._consistent_rim`) with one addition: the log10-mass range can be
narrowed for a crater-size band (`lm_bounds`). A band keeps only craters whose observed diameter lies inside it,
which is selection on the observation and leaves p(theta | x) unchanged, provided the narrowed range contains every
mass that can make a crater in the band (`support_check`). `self_test` shows that with lm_bounds = (-3, 20) the
function reproduces simulate_truth draw for draw.

Context x (8): [log10 D_obs, sigma_meas, p_mare, artificial, log10 speed_obs * m_v, elev_obs/90 * m_a, m_v, m_a]
Target theta (4): [log10 m, log10 u, elev/90, log10 delta]
Usage: python research/otfmpe/gen.py OUT_DIR [N_TRAIN]
"""
import json
import math
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from c2i.engine import DEFAULT_PACKAGE_SET, PACKAGES, ModelForm  # noqa: E402
from c2i.engine.populations import POPULATIONS  # noqa: E402
from c2i.engine.targets import GLOBAL_CONTEXT, SiteContext  # noqa: E402

SPEED_REL = 0.03          # relative speed error of the training contexts (tracked impacts)
ANGLE_SD = 5.0            # deg
SIGMA_RANGE = (0.01, 0.15)  # crater-diameter measurement error, dex
D_RANGE = (5.0, 5000.0)
# 12 equal-log crater-size bands. Equal counts per band and per scaling package give each package the same
# weight among craters of a given size, which is the engine's crater-level scenario weighting
BANDS = tuple(zip(np.geomspace(*D_RANGE, 13)[:-1].round(4), np.geomspace(*D_RANGE, 13)[1:].round(4)))
POPS = ("asteroidal", "artificial")


def truth(n, rng, *, population="asteroidal", package="holsapple2003", site=GLOBAL_CONTEXT,
          model=ModelForm(), sigma_meas=0.03, d_range=D_RANGE, lm_bounds=(-3.0, 20.0)):
    from c2i.engine.forward import _consistent_rim, _target_uniforms
    from c2i.engine.targets import LAYERED_EXTRA_SIGMA_DEX, is_layered
    pop, pkg = POPULATIONS[population], PACKAGES[package]
    lo, hi = lm_bounds
    rows, got = [], 0
    while got < n:
        m = 40 * n
        b = pop.mass_prior_b if model.mass_prior_b is None else model.mass_prior_b
        if b > 0:
            bl = b * math.log(10.0)
            lm = lo - np.log1p(-rng.random(m) * (1.0 - math.exp(-bl * (hi - lo)))) / bl
        else:
            lm = rng.uniform(lo, hi, m)
        u = pop.sample_u(m, rng)
        delta = pop.sample_delta(m, rng)
        elev = pop.sample_elev(m, rng)
        q = rng.uniform(*model.angle_q, m)
        nu = rng.uniform(*model.nu, m)
        tu = _target_uniforms(m, site, rng)
        h = tu["h"]
        s, *_ = _consistent_rim(tu, lm, u, delta, pkg, nu, elev, q, 1.62)
        lay = is_layered(10 ** s / h) if site.fixed_class is None else np.zeros(m, dtype=bool)
        s = s + np.sqrt(model.sigma_mf_dex**2 + np.where(lay, LAYERED_EXTRA_SIGMA_DEX**2, 0.0)) \
            * rng.standard_normal(m)
        yobs = s + sigma_meas * rng.standard_normal(m)
        keep = (10 ** yobs >= d_range[0]) & (10 ** yobs <= d_range[1])
        rows.append({"lm": lm[keep], "lu": np.log10(u[keep]),
                     "le": (lm + 2 * np.log10(u) - math.log10(2))[keep],
                     "y": yobs[keep], "elev": elev[keep], "ld": np.log10(delta[keep])})
        got += int(keep.sum())
    return {k: np.concatenate([r[k] for r in rows])[:n] for k in rows[0]}


def site_with(p_mare):
    g = GLOBAL_CONTEXT
    return SiteContext(p_mare=float(p_mare), h_by_unit=g.h_by_unit, label="otfmpe")


def self_test():
    from run_benchmarks_v2 import simulate_truth
    for pop in POPS:
        a = simulate_truth(20, np.random.default_rng(7), population=pop, package="lpi2007", d_range=(5.0, 50.0))
        b = truth(20, np.random.default_rng(7), population=pop, package="lpi2007", d_range=(5.0, 50.0))
        assert all(np.array_equal(a[k], b[k]) for k in a), pop
    return "identical to run_benchmarks_v2.simulate_truth (asteroidal and artificial)"


def support_check(band, population, n=3000, seed=11):
    """log10-mass range that makes craters in the band: flat prior over the full support, widest sigma."""
    rng = np.random.default_rng(seed)
    lms = np.concatenate([truth(n, rng, population=population, package=p, d_range=band,
                                model=ModelForm(mass_prior_b=0.0), sigma_meas=SIGMA_RANGE[1],
                                lm_bounds=_coarse(band))["lm"] for p in DEFAULT_PACKAGE_SET])
    return float(lms.min()), float(lms.max())


def _coarse(band):
    """Generous starting range from D ~ m^(1/3) scaling (only used to make support_check fast; it is checked
    to be wider than the result)."""
    c = 3.0 * math.log10(band[0]), 3.0 * math.log10(band[1])
    return max(-3.0, c[0] - 7.0), min(20.0, c[1] + 7.0)


def chunk(args):
    seed, n, population, package, band, lm_bounds, reveal = args
    rng = np.random.default_rng(seed)
    p_mare = float(rng.choice([0.0, 1.0, GLOBAL_CONTEXT.p_mare, rng.uniform()]))
    sig = float(rng.uniform(*SIGMA_RANGE))
    t = truth(n, rng, population=population, package=package, site=site_with(p_mare), sigma_meas=sig,
              d_range=band, lm_bounds=lm_bounds)
    k = len(t["y"])
    speed = t["lu"] + math.log10(1 + SPEED_REL) * rng.standard_normal(k)
    ang = t["elev"] + ANGLE_SD * rng.standard_normal(k)
    if reveal == "random":
        mv = rng.random(k) < 0.5
        ma = mv & (rng.random(k) < 0.7) | (~mv & (rng.random(k) < 0.15))
    else:
        mv, ma = np.full(k, "v" in reveal), np.full(k, "a" in reveal)
    x = np.stack([t["y"], np.full(k, sig), np.full(k, p_mare), np.full(k, float(POPULATIONS[population].artificial)),
                  speed * mv, ang / 90.0 * ma, mv.astype(float), ma.astype(float)], 1).astype(np.float32)
    th = np.stack([t["lm"], t["lu"], t["elev"] / 90.0, t["ld"]], 1).astype(np.float32)
    return x, th, np.full(k, DEFAULT_PACKAGE_SET.index(package), np.int8)


def build(jobs, procs=12):
    with Pool(procs) as pool:
        parts = pool.map(chunk, jobs, chunksize=1)
    return {"x": np.concatenate([p[0] for p in parts]), "theta": np.concatenate([p[1] for p in parts]),
            "package": np.concatenate([p[2] for p in parts])}


def jobs_for(seed0, n_total, bounds, reveal="random", chunk_n=1000):
    """Equal shares per population, band and package."""
    per = n_total // (len(POPS) * len(BANDS) * len(DEFAULT_PACKAGE_SET))
    jobs, s = [], seed0
    for pop in POPS:
        for band in BANDS:
            for pkg in DEFAULT_PACKAGE_SET:
                left = per
                while left > 0:
                    jobs.append((s, min(chunk_n, left), pop, pkg, band, bounds[(pop, band)], reveal))
                    left -= chunk_n; s += 1
    return jobs


def mass_bounds():
    out = {}
    for pop in POPS:
        for band in BANDS:
            lo, hi = support_check(band, pop)
            c = _coarse(band)
            assert lo > c[0] + 0.3 and hi < c[1] - 0.3, (pop, band, lo, hi, c)   # coarse range was wide enough
            out[(pop, band)] = (max(-3.0, lo - 1.5), min(20.0, hi + 1.0))
    return out


def main(out, n_train=2_400_000):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    man = {"self_test": self_test(), "speed_rel": SPEED_REL, "angle_sd_deg": ANGLE_SD, "sigma_range": SIGMA_RANGE,
           "bands_m": [list(b) for b in BANDS],
           "context": ["log10_D_obs", "sigma_meas_dex", "p_mare", "artificial", "log10_speed_obs*m_v",
                       "elev_obs/90*m_a", "m_v", "m_a"],
           "theta": ["log10_mass_kg", "log10_speed_m_s", "elev_deg/90", "log10_density_kg_m3"]}
    bounds = mass_bounds()
    man["mass_bounds"] = {f"{p}_{b[0]:g}-{b[1]:g}m": v for (p, b), v in bounds.items()}
    print("bounds done", round(time.time() - t0), flush=True)
    tr = build(jobs_for(1_000, n_train, bounds))
    va = build(jobs_for(500_000, max(n_train // 20, 72_000), bounds))
    np.savez_compressed(out / "train.npz", **tr)
    np.savez_compressed(out / "val.npz", **va)
    man.update(n_train=len(tr["x"]), n_val=len(va["x"]), seeds={"train": 1_000, "val": 500_000, "tests": 900_000},
               seconds=round(time.time() - t0, 1))
    (out / "gen_manifest.json").write_text(json.dumps(man, indent=1))
    print(json.dumps({k: v for k, v in man.items() if k != "mass_bounds"}, indent=1))
    return bounds


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 2_400_000)
