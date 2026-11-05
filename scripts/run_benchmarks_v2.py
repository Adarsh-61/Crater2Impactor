"""Reproducible benchmark suite for the v2 engine -> results/benchmarks_v2.json.

Every block records its configuration, seed, sample size, and test type:
  M = mathematical correctness, C = in-model calibration (simulator = model),
  R = robustness under deliberate misspecification, B = baseline comparison,
  P = physical validation against measured impacts, W = real-world predictive
  (population or event level), L = law recovery on laboratory craters,
  T = location test (does position predict regolith thickness).

Synthetic blocks (M, C, R, B) can never establish real-world accuracy; they
are labelled as such in the JSON and in the UI.

Usage:  python scripts/run_benchmarks_v2.py [--quick]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from c2i.engine import (  # noqa: E402
    DEFAULT_PACKAGE_SET, PACKAGES, ModelForm, infer_crater, log10_rim)
from c2i.engine.populations import POPULATIONS  # noqa: E402
from c2i.engine.targets import GLOBAL_CONTEXT, SiteContext, sample_target  # noqa: E402

# importance-sampling draws per crater in the calibration blocks C, R and B: the engine default, so the reported
# coverage is that of the configuration users run
N_DRAWS_PER_CRATER = 30_000


# ------------------------------------------------------------------------------------
# synthetic truth generator (any population / package / model form)
# ------------------------------------------------------------------------------------
def simulate_truth(n, rng, *, population="asteroidal", package="holsapple2003",
                   site=GLOBAL_CONTEXT, model=ModelForm(), sigma_meas=0.03,
                   d_range=(5.0, 5000.0)):
    """Draw impact events from a declared generative model; keep craters in d_range.

    Target class follows the forward model's self-consistent rule (forward._consistent_rim): the crater
    forms in the regolith while the regolith crater stays shallow, in the substrate otherwise, at the regime
    switch when neither branch is consistent. The inverse inverts this rule exactly (sampler.COHERENT_CLASS_MODEL).
    Mass follows the population's prior; crater-size selection depends only on the observation.
    """
    from c2i.engine.forward import _consistent_rim, _target_uniforms
    from c2i.engine.targets import LAYERED_EXTRA_SIGMA_DEX, is_layered
    pop, pkg = POPULATIONS[population], PACKAGES[package]
    rows, got = [], 0
    while got < n:
        m = 40 * n
        # log10 m from the population's mass prior, a truncated power law on [-3, 20]. The lower end lies below the
        # smallest mass that makes a 5 m crater (10^-2.2 kg); the upper end is the inverse's support for the largest
        # craters kept (sampler.mass_support), so the truth and the inverse use the same prior over the craters scored.
        b = pop.mass_prior_b if model.mass_prior_b is None else model.mass_prior_b
        if b > 0:
            bl = b * math.log(10.0)
            lm = -3.0 - np.log1p(-rng.random(m) * (1.0 - math.exp(-bl * 23.0))) / bl
        else:
            lm = rng.uniform(-3, 20, m)
        u = pop.sample_u(m, rng)
        delta = pop.sample_delta(m, rng)
        elev = pop.sample_elev(m, rng)
        q = rng.uniform(*model.angle_q, m)
        nu = rng.uniform(*model.nu, m)
        # the forward model's self-consistent target class (forward._consistent_rim), which the inverse inverts
        # exactly (sampler.COHERENT_CLASS_MODEL). Accepting the class with p(class | D, h) by rejection instead would
        # make the truth mass distribution prior x Z(impactor) rather than the declared prior.
        tu = _target_uniforms(m, site, rng)
        h = tu["h"]
        s, *_ = _consistent_rim(tu, lm, u, delta, pkg, nu, elev, q, 1.62)
        lay = is_layered(10 ** s / h) if site.fixed_class is None else np.zeros(m, dtype=bool)
        s = s + np.sqrt(model.sigma_mf_dex**2 + np.where(lay, LAYERED_EXTRA_SIGMA_DEX**2, 0.0)) \
            * rng.standard_normal(m)
        acc = np.ones(m, dtype=bool)
        yobs = s + sigma_meas * rng.standard_normal(m)
        keep = acc & (10 ** yobs >= d_range[0]) & (10 ** yobs <= d_range[1])
        rows.append({"lm": lm[keep], "lu": np.log10(u[keep]),
                     "le": (lm + 2 * np.log10(u) - math.log10(2))[keep],
                     "y": yobs[keep], "elev": elev[keep], "ld": np.log10(delta[keep])})
        got += int(keep.sum())
    out = {k: np.concatenate([r[k] for r in rows])[:n] for k in rows[0]}
    return out


def coverage_block(truth, infer_kw, n_draws, label, rng_seed):
    keys = (("log10_mass_kg", "lm"), ("log10_energy_j", "le"), ("log10_velocity_m_s", "lu"))
    hits90 = {k: [] for k, _ in keys}
    hits50 = {k: [] for k, _ in keys}
    henv = {k: [] for k, _ in keys}
    widths = {k: [] for k, _ in keys}
    t0 = time.time()
    for i in range(len(truth["y"])):
        r = infer_crater(10 ** truth["y"][i], seed=rng_seed + i, n_draws=n_draws,
                         want_samples=False, **infer_kw)
        for k, tk in keys:
            q = r["quantities"][k]
            tv = truth[tk][i]
            hits90[k].append(q["q05"] <= tv <= q["q95"])
            widths[k].append(q["width90"])
            lo, hi = q["envelope90"]
            henv[k].append(lo <= tv <= hi)
            # central 68% interval (q16-q84); the list name is historical
            hits50[k].append(q["q16"] <= tv <= q["q84"])
    dt = time.time() - t0
    res = {"label": label, "n": len(truth["y"]), "n_draws_per_crater": n_draws, "seconds": round(dt, 1),
           "per_quantity": {}}
    for k, _ in keys:
        n = len(hits90[k])
        c90 = float(np.mean(hits90[k]))
        c68 = float(np.mean(hits50[k]))
        res["per_quantity"][k] = {
            "coverage90": c90, "coverage90_mc_se": math.sqrt(c90 * (1 - c90) / n),
            "coverage68": c68, "envelope_coverage90": float(np.mean(henv[k])),
            "median_width90_dex": float(np.median(widths[k])),
        }
    return res


# ------------------------------------------------------------------------------------
def block_M(quick):
    """M: fiber sampler equals an independent importance sampler (no fiber solve)."""
    from c2i.engine.sampler import run_scenario
    rng = np.random.default_rng(101)
    out = []
    for d_obs in (10.0, 120.0, 1500.0):
        model = ModelForm(sigma_mf_dex=0.08)
        site = SiteContext(fixed_class=2 if d_obs > 50 else 0)
        r = run_scenario(d_obs, 0.03, "asteroidal", "holsapple2003", site=site, model=model,
                         n_draws=100_000 if quick else 200_000, rng=rng)
        q_f = _wq(r.draws["log10_mass_kg"], r.weights)
        n = 600_000 if quick else 1_500_000
        pop = POPULATIONS["asteroidal"]
        c = math.log10(d_obs) * 3
        lm = rng.uniform(c - 3.5, c + 4.0, n)
        u, de, el = pop.sample_u(n, rng), pop.sample_delta(n, rng), pop.sample_elev(n, rng)
        cls, rho, y, *_ = sample_target(n, d_obs, site, rng)
        k1, k2, mu, kr, _ = PACKAGES["holsapple2003"].arrays()
        s, *_ = log10_rim(lm, u, de, rho, y, k1[cls], k2[cls], mu[cls], kr[cls], g=1.62, density_ratio=True,
                          nu=rng.uniform(*model.nu, n), elev_deg=el, angle_q=rng.uniform(*model.angle_q, n))
        # the population's mass prior enters as a weight on the uniform window
        w = np.exp(-0.5 * ((s - math.log10(d_obs)) / math.hypot(0.03, 0.08)) ** 2) * 10.0 ** (-pop.mass_prior_b * lm)
        q_i = _wq(lm, w)
        out.append({"crater_m": d_obs, "fiber_q05_q50_q95": q_f, "independent_q05_q50_q95": q_i,
                    "max_abs_diff_dex": float(np.max(np.abs(np.array(q_f) - np.array(q_i))))})
    # global site (regolith/substrate decided per impactor by the forward rule) against an independent importance
    # sampler of the FORWARD model (forward._consistent_rim, noise-free layered band), i.e. forward-inverse coherence
    from c2i.engine.forward import _consistent_rim, _target_uniforms
    from c2i.engine.targets import LAYERED_EXTRA_SIGMA_DEX, is_layered
    for d_obs in (30.0, 300.0):
        model = ModelForm(sigma_mf_dex=0.08)
        r = run_scenario(d_obs, 0.03, "asteroidal", "holsapple2003", model=model,
                         n_draws=100_000 if quick else 200_000, rng=rng)
        q_f = _wq(r.draws["log10_mass_kg"], r.weights)
        n = 600_000 if quick else 1_500_000
        pop = POPULATIONS["asteroidal"]
        c = math.log10(d_obs) * 3
        lm = rng.uniform(c - 4.5, c + 3.0, n)
        u, de, el = pop.sample_u(n, rng), pop.sample_delta(n, rng), pop.sample_elev(n, rng)
        tu = _target_uniforms(n, GLOBAL_CONTEXT, rng)
        s, *_ = _consistent_rim(tu, lm, u, de, PACKAGES["holsapple2003"], rng.uniform(*model.nu, n), el,
                                rng.uniform(*model.angle_q, n), 1.62)
        sd = np.sqrt(0.03**2 + 0.08**2 + np.where(is_layered(10 ** s / tu["h"]), LAYERED_EXTRA_SIGMA_DEX**2, 0.0))
        w = np.exp(-0.5 * ((s - math.log10(d_obs)) / sd) ** 2) / sd * 10.0 ** (-pop.mass_prior_b * lm)
        q_i = _wq(lm, w)
        out.append({"crater_m": d_obs, "site": "global (forward class rule)", "fiber_q05_q50_q95": q_f,
                    "independent_q05_q50_q95": q_i,
                    "max_abs_diff_dex": float(np.max(np.abs(np.array(q_f) - np.array(q_i))))})
    return {"type": "M", "title": "Fiber sampler vs independent importance sampler (log10 mass)",
            "claim_supported": "The engine computes the posterior of its declared model correctly.",
            "claim_not_supported": "Anything about real impactors.", "results": out}


def _wq(x, w, qs=(0.05, 0.5, 0.95)):
    """Weighted quantiles with the engine's midpoint CDF, so both sides of block M use one definition."""
    from c2i.engine.infer import _wq as engine_wq
    return engine_wq(np.asarray(x), np.asarray(w), qs)


def block_C(quick):
    """C: in-model calibration. Truth drawn from the same scenario set the engine declares."""
    rng = np.random.default_rng(202)
    n = 150 if quick else 400
    res = []
    for pkg in DEFAULT_PACKAGE_SET:
        truth = simulate_truth(n // 3, rng, package=pkg)
        res.append(coverage_block(truth, {}, N_DRAWS_PER_CRATER, f"truth package {pkg}, engine averages all three", 2000))
    return {"type": "C", "title": "In-model calibration (simulator = declared model set)",
            "claim_supported": "Intervals are calibrated when reality matches one of the declared scenarios.",
            "claim_not_supported": "Calibration on real craters.", "results": res}


def block_R(quick):
    """R: robustness. Truth generated OUTSIDE the engine's declared assumptions."""
    rng = np.random.default_rng(303)
    n = 100 if quick else 250
    cases = [
        ("angle law q=1.3 (steeper than declared 0.5-1.0)", dict(model=ModelForm(angle_q=(1.3, 1.3))), {}),
        ("model-form residual 0.2 dex (declared 0.1)", dict(model=ModelForm(sigma_mf_dex=0.2)), {}),
        ("30% secondaries, engine assumes primaries only", "mix_secondary", {}),
        ("30% secondaries, engine told P(secondary)=0.3", "mix_secondary",
         {"populations": {"asteroidal": 0.7, "secondary": 0.3}}),
        ("meteoroid population, engine assumes asteroidal", dict(population="meteoroid"), {}),
    ]
    res = []
    for label, gen, kw in cases:
        if gen == "mix_secondary":
            ta = simulate_truth(int(n * 0.7), rng, population="asteroidal")
            ts = simulate_truth(n - int(n * 0.7), rng, population="secondary")
            truth = {k: np.concatenate([ta[k], ts[k]]) for k in ta}
        else:
            truth = simulate_truth(n, rng, **gen)
        res.append(coverage_block(truth, kw, N_DRAWS_PER_CRATER, label, 3000))
    return {"type": "R", "title": "Robustness under deliberate misspecification",
            "claim_supported": "Shows which violated assumptions break calibration and whether "
                               "the envelope or a declared mixture repairs it.",
            "claim_not_supported": "Real-world accuracy.", "results": res}


def block_B(quick):
    """B: baselines on one common synthetic test set (truth: each package equally likely)."""
    rng = np.random.default_rng(404)
    n = 120 if quick else 300
    parts = [simulate_truth(n // 3, rng, package=p) for p in DEFAULT_PACKAGE_SET]
    truth = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    out = {"n": len(truth["y"]), "n_draws_per_crater": N_DRAWS_PER_CRATER, "methods": {}}

    def score(name, q05, q50, q95, secs, key="lm"):
        tv = truth[key]
        q05, q50, q95 = map(np.asarray, (q05, q50, q95))
        out["methods"].setdefault(name, {})[key] = {
            "coverage90": float(np.mean((q05 <= tv) & (tv <= q95))),
            "median_abs_error_dex": float(np.median(np.abs(q50 - tv))),
            "median_width90_dex": float(np.median(q95 - q05)),
            "interval_score90": float(np.mean((q95 - q05) + (2 / 0.1) * np.maximum(q05 - tv, 0)
                                              + (2 / 0.1) * np.maximum(tv - q95, 0))),
            "seconds_total": round(secs, 2),
        }

    # v2 engine
    t0 = time.time()
    rr = [infer_crater(10 ** y, seed=7 + i, n_draws=N_DRAWS_PER_CRATER, want_samples=False)
          for i, y in enumerate(truth["y"])]
    dt = time.time() - t0
    for key, tk in (("log10_mass_kg", "lm"), ("log10_energy_j", "le"), ("log10_velocity_m_s", "lu")):
        score("v2 engine (3 packages, global target)",
              [r["quantities"][key]["q05"] for r in rr], [r["quantities"][key]["median"] for r in rr],
              [r["quantities"][key]["q95"] for r in rr], dt, tk)

    # prior-only for velocity (population prior quantiles)
    pu = np.log10(POPULATIONS["asteroidal"].sample_u(200000, rng))
    qv = np.quantile(pu, [0.05, 0.5, 0.95])
    score("prior only (ignores crater)", [qv[0]] * out["n"], [qv[1]] * out["n"], [qv[2]] * out["n"], 0.0, "lu")

    # LPI-style point inverse: fixed U = 20 km/s, 45 deg, nominal target class by size
    t0 = time.time()
    from c2i.engine.packages import solve_log10_m
    k1, k2, mu, kr, _ = PACKAGES["lpi2007"].arrays()
    ys = truth["y"]
    cls = np.where(10 ** ys < 40, 0, 1)
    rho = np.where(cls == 0, 1500.0, 2500.0)
    yst = np.where(cls == 0, 1e4, 1e6)
    lm, *_ = solve_log10_m(ys, u_m_s=np.full(len(ys), 20000.0), delta=np.full(len(ys), 2700.0), rho_t=rho,
                           y_eff=yst, k1=k1[cls], k2=k2[cls], mu=mu[cls], kr=kr[cls], g=1.62,
                           density_ratio=False, nu=1 / 3, elev_deg=45.0, angle_q=1.0)
    dt = time.time() - t0
    score("LPI-style point inverse (U=20 km/s, 45 deg, nominal target)", lm, lm, lm, dt, "lm")

    return {"type": "B", "title": "Baselines on a common synthetic test set (truth from all three packages)",
            "claim_supported": "Relative calibration and sharpness of methods when the truth "
                               "is drawn from the declared scenario set.",
            "claim_not_supported": "Real-world accuracy of any method.", "results": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--blocks", default="M,C,R,B,P,W,L,T")
    ap.add_argument("--merge", action="store_true", help="update blocks in the existing results file")
    a = ap.parse_args()
    blocks = set(a.blocks.split(","))
    t0 = time.time()
    res = {
        "provenance": {
            "script": "scripts/run_benchmarks_v2.py",
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "engine_files_sha256": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16]
                for p in sorted((REPO / "src" / "c2i" / "engine").glob("*.py"))},
            "python": platform.python_version(), "numpy": np.__version__,
            "quick": a.quick, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "legend": {"L": "law recovery on laboratory data", "T": "location-aware target test",
                   "M": "mathematical correctness", "C": "in-model calibration",
                   "R": "robustness under misspecification", "B": "baseline comparison",
                   "P": "physical validation (measured impacts)",
                   "W": "real-world predictive test (population/event)"},
        "blocks": {},
    }
    if "M" in blocks:
        res["blocks"]["M"] = block_M(a.quick)
    if "C" in blocks:
        res["blocks"]["C"] = block_C(a.quick)
    if "R" in blocks:
        res["blocks"]["R"] = block_R(a.quick)
    if "B" in blocks:
        res["blocks"]["B"] = block_B(a.quick)
    if "P" in blocks:
        from validation_v2 import block_P  # noqa: E402
        res["blocks"]["P"] = block_P()
    if "W" in blocks:
        from validation_v2 import block_W  # noqa: E402
        res["blocks"]["W"] = block_W()
    if "L" in blocks:
        from validation_v2 import block_L  # noqa: E402
        res["blocks"]["L"] = block_L()
    if "T" in blocks:
        from validation_v2 import block_T  # noqa: E402
        res["blocks"]["T"] = block_T()
    res["provenance"]["seconds_total"] = round(time.time() - t0, 1)
    p = REPO / "results" / "benchmarks_v2.json"
    if a.merge and p.exists():
        old = json.loads(p.read_text())
        old["blocks"].update(res["blocks"])
        old.setdefault("merged_runs", []).append(res["provenance"])
        old["legend"] = res["legend"]
        res = old
    p.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v.get("title") for k, v in res["blocks"].items()}, indent=1))
    print("wrote", p)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
