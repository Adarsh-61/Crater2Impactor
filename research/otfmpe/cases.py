"""Test cases shared by OT-FMPE and the Bayesian engine, and the engine's answers on them.

Every case is stored once with the 8-number OT-FMPE context and the engine input carrying exactly the same
information (same diameter, measurement error, mare fraction, population, and the same speed and angle
measurements with the same stated errors). Truth seeds start at 900000, disjoint from training (1000+) and
validation (500000+).

Sets
  In the training range (fair comparison):
    nat_D, nat_DVA   natural impactors (asteroidal prior), diameter only / + speed (3 %) and angle (5 deg)
    art_D, art_DVA   spacecraft population, diameter only / + speed and angle
  Stress tests (outside the training range or the declared model):
    sig_hi     measurement error 0.30 dex (training: 0.01-0.15)
    d_small    craters 2-5 m (training: 5 m - 5 km)
    d_large    craters 5-10 km
    mf_02      truths with model scatter 0.20 dex; both models assume 0.10 (misspecified physics)
    v_err10    speed measured to 10 %, the engine is told 10 %; OT-FMPE has no input for it (trained at 3 %)
  Real:
    sc_art_D, sc_art_DVA   11 lunar spacecraft impacts (data/validation/spacecraft_impacts.json), artificial prior;
                           speed and angle given with the training errors (3 %, 5 deg) to both models
Usage: python research/otfmpe/cases.py OUT_DIR
"""
import json
import math
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import gen  # noqa: E402
from gen import ANGLE_SD, BANDS, REPO, SPEED_REL  # noqa: E402

from c2i.engine import DEFAULT_PACKAGE_SET, EventData, ModelForm, infer_crater  # noqa: E402
from c2i.engine.lunar_context import site_context  # noqa: E402

SC = json.loads((REPO / "data/validation/spacecraft_impacts.json").read_text())["events"]


def synth(name, population, bands, n_per, reveal, seed0, *, sig_range=gen.SIGMA_RANGE, model=None, speed_rel=SPEED_REL):
    cases, s = [], seed0
    for band in bands:
        got = 0
        lmb = gen.BOUNDS[(population, band)] if (population, band) in gen.BOUNDS else gen._coarse(band)
        while got < n_per:
            rng = np.random.default_rng(s); s += 1
            p_mare = float(rng.choice([0.0, 1.0, 0.18, rng.uniform()]))
            sig = float(rng.uniform(*sig_range))
            pkg = DEFAULT_PACKAGE_SET[s % 3]
            kw = {"model": model} if model is not None else {}
            t = gen.truth(1, rng, population=population, package=pkg, site=gen.site_with(p_mare), sigma_meas=sig,
                          d_range=band, lm_bounds=lmb, **kw)
            v = t["lu"][0] + math.log10(1 + speed_rel) * rng.standard_normal() if "v" in reveal else None
            a = float(np.clip(t["elev"][0] + ANGLE_SD * rng.standard_normal(), 0.5, 90.0)) if "a" in reveal else None
            cases.append(dict(set=name, population=population, band=[float(b) for b in band], package=pkg,
                              p_mare=p_mare, y=float(t["y"][0]), sig=sig, log10_speed_obs=v, elev_obs=a,
                              speed_rel=speed_rel,
                              truth={k: float(t[k][0]) for k in ("lm", "lu", "elev", "ld", "le")}))
            got += 1
    return cases


def real_cases():
    out = []
    for e in SC:
        sd = math.log10(1 + e["crater_sigma_rel"])
        pm = site_context(e["lat"], e["lon"]).p_mare
        tr = {"lm": math.log10(e["mass_kg"]), "lu": math.log10(e["velocity_km_s"] * 1000), "elev": e["elev_deg"]}
        base = dict(id=e["id"], y=math.log10(e["crater_d_m"]), sig=sd, p_mare=pm, truth=tr, population="artificial",
                    speed_rel=SPEED_REL, held_out=bool(e["held_out"]))
        out.append(dict(base, set="sc_art_D", log10_speed_obs=None, elev_obs=None))
        out.append(dict(base, set="sc_art_DVA", log10_speed_obs=tr["lu"], elev_obs=float(e["elev_deg"])))
    return out


def context(c):
    mv, ma = c["log10_speed_obs"] is not None, c["elev_obs"] is not None
    return [c["y"], c["sig"], c["p_mare"], 1.0 if c["population"] == "artificial" else 0.0,
            c["log10_speed_obs"] if mv else 0.0, c["elev_obs"] / 90.0 if ma else 0.0, float(mv), float(ma)]


def run_engine(c):
    ev = None
    if c["log10_speed_obs"] is not None or c["elev_obs"] is not None:
        ev = EventData(velocity_m_s=10 ** c["log10_speed_obs"] if c["log10_speed_obs"] is not None else None,
                       velocity_sigma_rel=c["speed_rel"],
                       elev_deg=c["elev_obs"], elev_sigma_deg=ANGLE_SD)
    t0 = time.perf_counter()
    r = infer_crater(10 ** c["y"], c["sig"], populations={c["population"]: 1.0}, site=gen.site_with(c["p_mare"]),
                     event=ev, seed=7, max_samples_out=2000)
    dt = time.perf_counter() - t0
    q = r["quantities"]
    return dict(seconds=dt, reliable=r["result_status"]["reliable"], reasons=r["result_status"]["reasons"],
                min_ess=r["result_status"]["min_ess"], backend=r["backend"]["log10_mass"],
                q={k: [q[k][j] for j in ("q05", "q16", "median", "q84", "q95")]
                   for k in ("log10_mass_kg", "log10_velocity_m_s", "log10_energy_j")},
                samples={k: r["samples"][k] for k in ("log10_mass_kg", "log10_velocity_m_s", "log10_energy_j")})


def main(out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    gen.BOUNDS = gen.mass_bounds()
    big = ((5000.0, 10000.0),)
    cases = (synth("nat_D", "asteroidal", BANDS, 25, "", 900_000)
             + synth("nat_DVA", "asteroidal", BANDS, 12, "va", 910_000)
             + synth("art_D", "artificial", BANDS, 12, "", 920_000)
             + synth("art_DVA", "artificial", BANDS, 12, "va", 930_000)
             + synth("sig_hi", "asteroidal", BANDS, 5, "", 940_000, sig_range=(0.30, 0.30))
             + synth("d_small", "asteroidal", ((2.0, 5.0),), 60, "", 950_000)
             + synth("d_large", "asteroidal", big, 60, "", 960_000)
             + synth("mf_02", "asteroidal", BANDS, 10, "", 970_000, model=ModelForm(sigma_mf_dex=0.20))
             + synth("v_err10", "asteroidal", BANDS, 5, "va", 980_000, speed_rel=0.10)
             + real_cases())
    t0 = time.time()
    with Pool(12) as pool:
        res = pool.map(run_engine, cases, chunksize=1)
    for c, r in zip(cases, res):
        c["engine"] = r
    x = np.array([context(c) for c in cases], dtype=np.float32)
    np.savez_compressed(out / "tests.npz", x=x)
    (out / "cases.json").write_text(json.dumps(cases))
    print(json.dumps({"n_cases": len(cases), "engine_wall_seconds": round(time.time() - t0, 1),
                      "sets": {s: sum(c["set"] == s for c in cases) for s in dict.fromkeys(c["set"] for c in cases)}},
                     indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
