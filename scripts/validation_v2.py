"""Real-world validation blocks for scripts/run_benchmarks_v2.py.

P  Physical validation on measured impacts (data/validation/spacecraft_impacts.json):
   forward prediction of crater diameter from the known impactor, and inverse
   recovery of the known mass from the crater plus the known speed and angle.
W  Population-level predictive test (data/validation/new_crater_population.json):
   observed new-crater counts from LROC temporal pairs vs (independent bolide
   flux) x (crater scaling), and what the counts constrain.
L  Law recovery on laboratory craters (data/lab/holsapple_impacts_v1.3.csv) and a
   cross-laboratory held-out test, plus the shipped constants on held-out granular shots.
T  Leave-one-out test of whether location predicts regolith thickness.

None of these blocks measures per-crater accuracy for natural impactors.
P uses hollow, slow spacecraft; W tests a population-level normalisation.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import numpy as np

from c2i.engine import DEFAULT_PACKAGE_SET, PACKAGES, EventData, ModelForm, infer_crater
from c2i.engine.forward import forward_predict
from c2i.engine.lunar_context import site_context
from c2i.engine.packages import effective_strength, log10_rim
from c2i.engine.populations import POPULATIONS
from c2i.engine.targets import (GLOBAL_CONTEXT, LAYERED_EXTRA_SIGMA_DEX, REGOLITH,
                                is_layered)

REPO = Path(__file__).resolve().parents[1]


# The inverse treats each spacecraft as a compact aluminium-density (2700 kg/m3) impactor, a convention for hollow
# vehicles. The artificial population's density prior (20-1000 kg/m3, bulk densities) does not contain 2700 kg/m3, and a
# known density is weighted by the population prior, so Block P uses the artificial population with its density prior
# widened to include the convention. With one population and a fixed density this changes no posterior, only the support.
# It is registered only while block_P runs, so importing this module does not add it to the engine's registry.
COMPACT_CONVENTION = "artificial_compact_convention"
COMPACT_POP = dataclasses.replace(
    POPULATIONS["artificial"], key=COMPACT_CONVENTION, delta_hi=3000.0,
    label="Artificial impactor, compact-density convention (validation only)")


# ------------------------------------------------------------------------------------
def block_P(n_inverse_draws: int = 60_000) -> dict:
    POPULATIONS[COMPACT_CONVENTION] = COMPACT_POP
    try:
        return _block_P(n_inverse_draws)
    finally:
        POPULATIONS.pop(COMPACT_CONVENTION, None)


def _block_P(n_inverse_draws: int) -> dict:
    # the 11 pre-registered spacecraft impacts with known mass, speed and angle
    ev = json.loads((REPO / "data" / "validation" / "spacecraft_impacts.json").read_text())
    rows, excluded = [], list(ev["excluded"])
    for e in ev["events"]:
        site = site_context(e["lat"], e["lon"])
        elev = e["elev_deg"] if e["elev_deg"] is not None else 45.0
        conv = {"compact (2700 kg/m3, aluminium)": 2700.0}
        if e.get("bulk_density_kg_m3"):
            conv[f"bulk ({e['bulk_density_kg_m3']} kg/m3, {e['bulk_density_source']})"] = float(e["bulk_density_kg_m3"])
        fwd = {}
        for name, dlt in conv.items():
            masses = e.get("mass_range_kg") or [e["mass_kg"]]
            preds = [forward_predict(m, e["velocity_km_s"] * 1000, elev, dlt, site=site, seed=1)
                     for m in masses]
            lo = min(p["rim_diameter_m"]["q05"] for p in preds)
            hi = max(p["rim_diameter_m"]["q95"] for p in preds)
            med = float(np.median([p["rim_diameter_m"]["median"] for p in preds]))
            fwd[name] = {
                "predicted_median_m": med, "predicted_90_m": [lo, hi],
                "log10_residual_obs_minus_pred": math.log10(e["crater_d_m"] / med),
                "observed_inside_90": bool(lo <= e["crater_d_m"] <= hi),
                "per_package_median_m": {k: v["rim_diameter_m_quantiles"][1]
                                         for k, v in preds[0]["per_package"].items()},
            }
        # inverse: crater + known speed (+ angle) -> mass, compact convention
        evd = EventData(velocity_m_s=e["velocity_km_s"] * 1000,
                        velocity_sigma_rel=e.get("velocity_sigma_rel", 0.02),
                        elev_deg=e["elev_deg"], elev_sigma_deg=e.get("elev_sigma_deg", 3.0), delta_kg_m3=2700.0)
        # spacecraft masses are designed, not drawn from a meteoroid power law: flat mass prior. The kinematic
        # priors are the artificial population's too; an asteroidal speed prior (log-normal tail, cut at
        # 2.38 km/s) would pull the posterior speed up by 0.6-1.4 % (mass <= 0.017 dex).
        r = infer_crater(e["crater_d_m"], math.log10(1 + e["crater_sigma_rel"]), site=site,
                         populations={COMPACT_CONVENTION: 1.0}, event=evd, model=ModelForm(mass_prior_b=0.0),
                         n_draws=n_inverse_draws, seed=3, want_samples=True)
        lm = np.array(r["samples"]["log10_mass_kg"])
        true_lm = math.log10(e["mass_kg"])
        q = r["quantities"]["log10_mass_kg"]
        rows.append({
            "id": e["id"], "label": e["label"], "quality": e["quality"], "sources": e["source"],
            "held_out": e["held_out"], "development_use": e["development_use"],
            "site": site.label, "observed_d_m": e["crater_d_m"], "known_mass_kg": e["mass_kg"],
            "mass_range_kg": e.get("mass_range_kg"),
            "velocity_km_s": e["velocity_km_s"], "elev_deg": e["elev_deg"],
            "forward": fwd,
            "inverse_mass": {
                "median_kg": 10 ** q["median"], "q05_kg": 10 ** q["q05"], "q95_kg": 10 ** q["q95"],
                "log10_error_median_minus_true": q["median"] - true_lm,
                "pit_of_true_mass": float(np.mean(lm < true_lm)),
                "true_inside_90": bool(q["q05"] <= true_lm <= q["q95"]),
            },
        })
    A = [x for x in rows if x["quality"] in ("A", "B")]
    key = "compact (2700 kg/m3, aluminium)"
    summ = {
        "n_events_used": len(rows), "n_quality_AB": len(A),
        "forward_median_log10_residual_AB": float(np.median([x["forward"][key]["log10_residual_obs_minus_pred"] for x in A])),
        "forward_inside_90_AB": f"{sum(x['forward'][key]['observed_inside_90'] for x in A)}/{len(A)}",
        "inverse_median_log10_error_AB": float(np.median([x["inverse_mass"]["log10_error_median_minus_true"] for x in A])),
        "inverse_inside_90_AB": f"{sum(x['inverse_mass']['true_inside_90'] for x in A)}/{len(A)}",
        # spread diagnostics. PIT values of calibrated intervals are uniform (SD 0.289); a smaller SD means the
        # intervals are wider than these errors need
        "forward_rms_log10_residual_AB": float(np.sqrt(np.mean([x["forward"][key]["log10_residual_obs_minus_pred"] ** 2
                                                                for x in A]))),
        "inverse_rms_log10_error_AB": float(np.sqrt(np.mean([x["inverse_mass"]["log10_error_median_minus_true"] ** 2
                                                             for x in A]))),
        "inverse_pit_sd_AB": float(np.std([x["inverse_mass"]["pit_of_true_mass"] for x in A], ddof=1)),
        "n_held_out": sum(x["held_out"] for x in rows),
    }
    return {
        "type": "P", "title": "Measured impacts: forward crater size and inverse mass recovery",
        "claim_supported": "How compact-impactor scaling performs on the only lunar impacts with known "
                           "mass, speed and angle (hollow, slow spacecraft).",
        "claim_not_supported": "Accuracy for natural (compact, ~20 km/s) impactors. The sample is tiny "
                               "and outside the scaling law's calibration regime.",
        "summary": summ, "events": rows, "excluded": excluded,
    }


# ------------------------------------------------------------------------------------
def _mass_flux_moon(lm, fl):
    """Cumulative impactor flux at the Moon, N(>m) per km^2 per yr (Brown 2002, mass form)."""
    v = fl["velocity_for_mass_conversion_km_s"] * 1000.0
    e_kt = 0.5 * 10.0 ** lm * v * v / 4.185e12
    n_earth = 10.0 ** fl["a0"] * e_kt ** (-fl["b0"])
    return n_earth / fl["earth_area_km2"] / fl["earth_to_moon_per_area"]


def block_W(n: int = 400_000, seed: int = 21, packages=DEFAULT_PACKAGE_SET, variant: dict | None = None,
            with_sensitivity: bool = True) -> dict:
    import dataclasses
    cfg = json.loads((REPO / "data" / "validation" / "new_crater_population.json").read_text())
    sv, fl = dict(cfg["survey"]), dict(cfg["flux"])
    v = variant or {}
    if "b0" in v:
        fl["b0"] = v["b0"]
    if "exposure" in v:
        sv["exposure_km2_yr"] = v["exposure"]
    if "cumulative_counts" in v:
        sv["cumulative_counts"] = v["cumulative_counts"]
    cum = {float(k): c for k, c in sv["cumulative_counts"].items()}
    if "completeness" in v:
        cum = {k: c for k, c in cum.items() if k >= v["completeness"]}
    edges = sorted(cum)
    obs = [cum[edges[i]] - (cum[edges[i + 1]] if i + 1 < len(edges) else 0) for i in range(len(edges))]
    bins = list(zip(edges, edges[1:] + [1e9]))
    AT = sv["exposure_km2_yr"]
    rng = np.random.default_rng(seed)
    pop = POPULATIONS["asteroidal"]
    if "u_median" in v:
        pop = dataclasses.replace(pop, u_median=v["u_median"])
    lm_lo, lm_hi = -2.0, 6.0
    lm = rng.uniform(lm_lo, lm_hi, n)
    # importance weight: flux density per unit log10 m times the uniform-proposal width
    dens = fl["b0"] * math.log(10.0) * _mass_flux_moon(lm, fl)
    w = dens * (lm_hi - lm_lo) / n
    u, delta, elev = pop.sample_u(n, rng), pop.sample_delta(n, rng), pop.sample_elev(n, rng)
    model = ModelForm(sigma_mf_dex=v.get("sigma_mf", 0.10), angle_q=v.get("angle_q", (0.5, 1.0)))
    q = rng.uniform(*model.angle_q, n)
    nu = rng.uniform(*model.nu, n)
    out = {"bins_m": bins, "observed_counts": obs, "exposure_km2_yr": AT, "packages": {}}
    dgrid = np.linspace(-0.6, 0.8, 141)
    fgrid = np.linspace(*fl["normalisation_prior_log10"], 37)
    ygrid = np.linspace(3.0, 5.3, 47)
    # The crater size depends on the target class, which depends on the crater size. Each event owns fixed target
    # uniforms (shared by the three packages) and the class is solved self-consistently with the forward model's
    # solver (engine/forward.py); re-drawing the target between passes would not converge.
    from c2i.engine.forward import _class_props, _consistent_rim, _target_uniforms
    tu = _target_uniforms(n, GLOBAL_CONTEXT, rng)
    for key in packages:
        pkg = PACKAGES[key]
        k1, k2, mu, kr, sd = pkg.arrays()
        s, cls, _, _ = _consistent_rim(tu, lm, u, delta, pkg, nu, elev, q, 1.62)
        rho, y = _class_props(cls, tu["u_rho"], tu["u_y"])
        h = tu["h"]
        lay = is_layered(10 ** s / h)
        s_obs = s + np.sqrt(model.sigma_mf_dex ** 2 + np.where(lay, LAYERED_EXTRA_SIGMA_DEX ** 2, 0.0)) \
            * rng.standard_normal(n)
        # expected counts per bin as a function of a log10 diameter shift Delta (F = 1)
        def mu_bins(shift, ss=s_obs, ww=w):
            d = 10 ** (ss + shift)
            return np.array([AT * ww[(d >= a) & (d < b)].sum() for a, b in bins])
        M = np.array([mu_bins(dd) for dd in dgrid])            # (nD, nbins)
        ll = _poisson_ll(M[:, None, :] * (10 ** fgrid)[None, :, None], np.array(obs))  # (nD, nF)
        post_d = _norm_exp(np.logaddexp.reduce(ll, axis=1))
        ll_f0 = ll[:, np.argmin(np.abs(fgrid))]
        pred0 = M[np.argmin(np.abs(dgrid))]
        # strength calibration: regolith Y_eff fixed on a grid, Delta = 0, flux marginalised
        ylls = []
        reg = cls == REGOLITH
        for ly in ygrid:
            y2 = np.where(reg, 10 ** ly, y)
            s2, *_ = log10_rim(lm, u, delta, rho, effective_strength(y2, 10 ** s, sd[cls]),
                               k1[cls], k2[cls], mu[cls], kr[cls], g=1.62,
                               density_ratio=pkg.density_ratio, nu=nu, elev_deg=elev, angle_q=q)
            s2 = s2 + (s_obs - s)
            m2 = mu_bins(0.0, s2)
            ylls.append(np.logaddexp.reduce(_poisson_ll(m2[None, :] * (10 ** fgrid)[:, None], np.array(obs))))
        post_y = _norm_exp(np.array(ylls))
        out["packages"][key] = {
            "predicted_counts_at_delta0_F1": pred0.tolist(),
            "predicted_total_ge7m_at_delta0_F1": float(pred0.sum()),
            "observed_total_ge7m": int(sum(obs)),
            "lowest_bin_m": bins[0][0],
            "delta_log10_D_posterior": _summ(dgrid, post_d),
            "delta_log10_D_best_at_F1": float(dgrid[np.argmax(ll_f0)]),
            "regolith_strength_log10_pa_posterior": _summ(ygrid, post_y),
            "slope_check": _slope(pred0, obs),
        }
    if with_sensitivity:
        sens = {}
        variants = {
            "baseline": {},
            "velocity median 15 km/s": {"u_median": 15000.0},
            "velocity median 22 km/s": {"u_median": 22000.0},
            "sigma_mf 0.05 dex": {"sigma_mf": 0.05},
            "sigma_mf 0.20 dex": {"sigma_mf": 0.20},
            "angle law q = 1 (vertical rule)": {"angle_q": (1.0, 1.0)},
            "completeness 8 m": {"completeness": 8.0},
            "flux slope b0 = 0.87": {"b0": 0.87},
            "flux slope b0 = 0.93": {"b0": 0.93},
            "2020 NAC survey (exposure 7.9e6, n>=16 m: 5, >=17: 4, >=18: 3, >=19: 2, >=43: 1; digitised R04)": {
                "exposure": 7.9e6, "cumulative_counts": {"16": 5, "17": 4, "18": 3, "19.4": 2, "42.8": 1}},
        }
        for name, var in variants.items():
            r = block_W(n=n // 2, seed=seed + 1, packages=("lpi2007",), variant=var, with_sensitivity=False)
            pk = r["packages"]["lpi2007"]
            sens[name] = {"observed_total": pk["observed_total_ge7m"],
                          "predicted_total_delta0_F1": pk["predicted_total_ge7m_at_delta0_F1"],
                          "delta_posterior": pk["delta_log10_D_posterior"],
                          "regolith_log10_Y_posterior": pk["regolith_strength_log10_pa_posterior"]}
        out["sensitivity_lpi2007"] = sens
    out.update({
        "type": "W", "title": "Population test: LROC new craters vs bolide flux x scaling",
        "flux_note": fl["normalisation_note"] + " " + fl["applicable_range_note"],
        "claim_supported": "Whether the scaling law plus an independent impactor flux reproduces the "
                           "observed production of 7-40 m lunar craters, and which single normalisation "
                           "(crater-size shift, or regolith strength) the counts constrain.",
        "claim_not_supported": "Per-crater accuracy; separate identification of flux and scaling "
                               "(they are degenerate: a flux factor F maps to a size factor F^(1/2.7)).",
    })
    return out


def _poisson_ll(mu, k):
    mu = np.maximum(mu, 1e-300)
    return np.sum(k * np.log(mu) - mu - np.array([math.lgamma(x + 1) for x in k]), axis=-1)


def _norm_exp(ll):
    p = np.exp(ll - np.max(ll))
    return p / p.sum()


def _summ(grid, p):
    c = np.cumsum(p)
    return {"median": float(np.interp(0.5, c, grid)), "q05": float(np.interp(0.05, c, grid)),
            "q95": float(np.interp(0.95, c, grid)), "mode": float(grid[np.argmax(p)])}


def _slope(pred, obs):
    pc = np.cumsum(np.array(pred)[::-1])[::-1]
    oc = np.cumsum(np.array(obs, dtype=float)[::-1])[::-1]
    if len(pc) < 5:
        return {}
    return {"predicted_N_ge7_over_N_ge12": float(pc[0] / max(pc[4], 1e-12)),
            "observed_N_ge7_over_N_ge12": float(oc[0] / oc[4])}


# ------------------------------------------------------------------------------------
def _engine_constants_on_granular(d, ormo_test, ormo_proj) -> dict:
    """The SHIPPED regolith constants of each package (K1, K2, mu, Kr; nu uniform 0.2-0.4 where the package has a
    density ratio) applied to cohesionless granular craters they were not fitted to, strength ~0 (gravity regime),
    vertical impacts, apparent radius Kr V^(1/3). Flintshot (code 41, the source of the published sand constants) is
    reported as in-sample; F-140 sand (45), KAFB alluvium (31) and Ormo et al. 2024 are held out. The 90 % band mixes
    packages and nu and adds the default model-form scatter (0.10 dex)."""
    shots = []
    for tt, label, ins in (("41", "Flintshot dry sand (in-sample)", True), ("45", "F-140 sand", False),
                           ("31", "KAFB desert alluvium", False)):
        g = d[d.tt == tt].dropna(subset=["crater_radius_cm", "mass_g", "velocity_cm_s"])
        if tt == "41":
            g = g[g.rho_g_cc < 2.0]
        shots += [(label, ins, r.gravity_G * 9.81, r.rho_g_cc * 1000, r.delta_g_cc * 1000, r.mass_g / 1000,
                   r.velocity_cm_s / 100, r.crater_radius_cm / 100) for r in g.itertuples()]
    m_p, d_p, rho_t = ormo_proj
    shots += [("Lane Mountain quartz sand (Ormo et al. 2024)", False, G * 9.81, rho_t * 1000, d_p * 1000, m_p / 1000,
               float(U), dout / 2000) for G, U, dout in ormo_test]
    rng = np.random.default_rng(40)
    nu, mf = rng.uniform(0.2, 0.4, 4000), rng.standard_normal(4000) * 0.10
    res = {}
    for label, ins, g, rho, dlt, m, u, r_obs in shots:
        pred = []
        for key in DEFAULT_PACKAGE_SET:
            pkg = PACKAGES[key]
            k1, k2, mu, kr, _ = pkg.arrays()
            sr = log10_rim(math.log10(m), u, dlt, rho, 1e-9, k1[REGOLITH], k2[REGOLITH], mu[REGOLITH], kr[REGOLITH],
                           g=g, density_ratio=pkg.density_ratio, nu=nu, elev_deg=90.0, angle_q=1.0, rim_factor=0.5)[0]
            pred.append(np.broadcast_to(sr, nu.shape))
        p = np.concatenate(pred) + np.tile(mf, len(pred))
        lo, med, hi = np.percentile(p, [5, 50, 95])
        lr = math.log10(r_obs)
        res.setdefault(label, {"in_sample": ins, "resid": [], "inside": 0})
        res[label]["resid"].append(lr - med)
        res[label]["inside"] += int(lo <= lr <= hi)
    out = {}
    for label, v in res.items():
        x = np.array(v["resid"])
        out[label] = {"in_sample": v["in_sample"], "n": int(x.size), "mean_log10_obs_over_pred": float(x.mean()),
                      "sd_dex": float(x.std(ddof=1)) if x.size > 1 else None, "inside_90": f"{v['inside']}/{x.size}"}
    ho = np.concatenate([np.array(v["resid"]) for v in res.values() if not v["in_sample"]])
    out["all_held_out"] = {"n": int(ho.size), "mean_log10_obs_over_pred": float(ho.mean()),
                           "rms_dex": float(np.sqrt(np.mean(ho ** 2))),
                           "inside_90": f"{sum(v['inside'] for v in res.values() if not v['in_sample'])}/{ho.size}"}
    return out


def block_L() -> dict:
    """Law recovery on laboratory data and a cross-laboratory held-out test."""
    import pandas as pd
    d = pd.read_csv(REPO / "data" / "lab" / "holsapple_impacts_v1.3.csv")
    for c in ["rho_g_cc", "gravity_G", "delta_g_cc", "mass_g", "velocity_cm_s", "volume_cm3", "crater_radius_cm"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["tt"] = d.target_type.astype(str).str.strip().str.zfill(2)
    sand = d[(d.tt == "41") & (d.rho_g_cc < 2.0)].dropna(subset=["crater_radius_cm", "volume_cm3"])
    a = np.cbrt(3 * sand.mass_g / (4 * math.pi * sand.delta_g_cc))
    pi2 = sand.gravity_G * 981 * a / sand.velocity_cm_s ** 2
    r = sand.rho_g_cc / sand.delta_g_cc
    piV = sand.rho_g_cc * sand.volume_cm3 / sand.mass_g
    X = np.column_stack([np.ones(len(sand)), np.log10(pi2), np.log10(r)])
    bV, *_ = np.linalg.lstsq(X, np.log10(piV), rcond=None)
    mu = -2 * bV[1] / (3 + bV[1])
    nu = ((2 + mu) * (-bV[2]) + 2 + mu) / 6.0
    rng = np.random.default_rng(14)
    boots = []
    for _ in range(2000):
        i = rng.integers(0, len(sand), len(sand))
        b, *_ = np.linalg.lstsq(X[i], np.log10(piV.values[i]), rcond=None)
        m_ = -2 * b[1] / (3 + b[1])
        boots.append((m_, ((2 + m_) * (-b[2]) + 2 + m_) / 6.0))
    boots = np.array(boots)
    piR = sand.crater_radius_cm * np.cbrt(sand.rho_g_cc / sand.mass_g)
    bR, *_ = np.linalg.lstsq(X, np.log10(piR), rcond=None)
    # Ormo et al. 2024 Table 5 homogeneous quartz-sand centrifuge shots (Boeing), held out
    test = [(10, 1766, 145), (51, 1780, 112), (150, 1754, 98), (502, 1966, 79)]
    m_p, d_p, rho_t = 1.32, 0.97, 1.498
    a_t = (3 * m_p / (4 * math.pi * d_p)) ** (1 / 3)
    rows = []
    for G, U, dout in test:
        p2 = G * 981 * a_t / (U * 100) ** 2
        R = 10 ** (bR[0] + bR[1] * math.log10(p2) + bR[2] * math.log10(rho_t / d_p)) / (rho_t / m_p) ** (1 / 3)
        rows.append({"gravity_G": G, "u_m_s": U, "observed_d_mm": dout, "predicted_d_mm": 20 * R,
                     "log10_pred_over_obs": math.log10(20 * R / dout)})
    engine_held_out = _engine_constants_on_granular(d[d.tt.str.fullmatch(r"\d+")], test, (m_p, d_p, rho_t))
    obs_ratio = math.log10(145 / 79)
    pred_ratio = math.log10(rows[0]["predicted_d_mm"] / rows[3]["predicted_d_mm"])
    return {
        "type": "L", "title": "Law recovery from laboratory craters and cross-laboratory transfer",
        "data": "Holsapple crater database v1.3 (dry Flintshot sand, rho 1.8 g/cc, n=%d); held-out: "
                "Ormo et al. 2024 Table 5 (Lane Mountain quartz sand, 10-502 G, CC BY 4.0)" % len(sand),
        "recovered": {"mu": mu, "mu_90": np.percentile(boots[:, 0], [5, 95]).tolist(),
                      "nu": nu, "nu_90": np.percentile(boots[:, 1], [5, 95]).tolist(),
                      "K1": 10 ** bV[0]},
        "published": {"mu_dry_sand": 0.41, "nu_holsapple": "1/3 (adopted), experiments 0.2-0.4"},
        "held_out": rows,
        "engine_constants_held_out": engine_held_out,
        "gravity_trend_log10_D10G_over_D502G": {"observed": obs_ratio, "predicted": pred_ratio},
        "notes": [
            "The database's code-41 sand mixes two target states (1.8 and 2.1 g/cc); mixing them "
            "biases mu to 0.35 and breaks extrapolation. Only the 1.8 g/cc state is used.",
            "An unconstrained raw-variable fit recovers the time-balance dimensional constraint "
            "(c_U + 2 c_g = -0.06, 90% CI -0.16 to +0.12) without being told it.",
            "Out-of-range tests: power laws extrapolate across gravity and projectile density "
            "with 0.07-0.14 dex error; gradient boosting and an MLP fail by 0.4-1.6 dex.",
        ],
        "claim_supported": "The point-source exponent mu is recoverable from controlled experiments and "
                           "its gravity dependence transfers to an independent laboratory; the "
                           "coefficient is material-specific.",
        "claim_not_supported": "That lunar craters alone could reveal these laws: they lack the impactor "
                               "mass and speed needed to fit them.",
    }


def block_T() -> dict:
    """Does location predict regolith thickness? (leave-one-out, scripts/build_lunar_context.py)."""
    meta = json.loads((REPO / "data" / "lunar" / "lunar_context_meta.json").read_text())
    return {
        "type": "T", "title": "Location-aware target properties: what coordinates can predict",
        "cross_validation": meta["cross_validation"], "decision": meta.get("decision"),
        "unit_means_m": meta["unit_means_m"],
        "mare_area_fraction_65S_65N": meta["mare_area_fraction_65S_65N"],
        "claim_supported": "Mare vs highland at the crater site is observed and is used to select the "
                           "substrate (competent basalt vs fractured highland crust).",
        "claim_not_supported": "A local regolith thickness from coordinates: an interpolated map of 54 "
                               "published measurements predicts worse than a constant.",
    }
