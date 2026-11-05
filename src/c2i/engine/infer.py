"""Single-crater inference: scenarios, robust envelope, identifiability, budget.

``infer_crater`` is the scientific entry point used by the API. It returns a
JSON-serialisable dictionary that states, for every quantity, what the crater
actually determines and what is carried by assumptions.
"""

from __future__ import annotations

import math
from dataclasses import asdict

import numpy as np
from scipy.special import ndtr

from .flash import DEFAULT_FLASH_DEFINITION, FLASH_DEFINITIONS
from .packages import DEFAULT_PACKAGE_SET, PACKAGES, TARGET_CLASSES
from .populations import POPULATIONS
from .sampler import EventData, ModelForm, mass_support, run_scenario, validate_event
from .src_backend import SRCUnsupported, run_scenario_src
from .targets import GLOBAL_CONTEXT, SiteContext, layered_fraction

QUANTITIES: dict[str, dict] = {
    "log10_diameter_m": {"label": "Impactor diameter", "unit": "m", "log": True},
    "log10_velocity_m_s": {"label": "Impact velocity", "unit": "m/s", "log": True},
    "log10_density_kg_m3": {"label": "Impactor density", "unit": "kg/m^3", "log": True},
    "log10_strength_pa": {"label": "Target strength", "unit": "Pa", "log": True},
    "log10_target_density_kg_m3": {"label": "Target density", "unit": "kg/m^3", "log": True},
    "angle_from_vertical_deg": {"label": "Impact angle (from vertical)", "unit": "deg", "log": False},
    "log10_mass_kg": {"label": "Impactor mass", "unit": "kg", "log": True},
    "log10_energy_j": {"label": "Kinetic energy", "unit": "J", "log": True},
    "log10_momentum_kg_m_s": {"label": "Momentum", "unit": "kg m/s", "log": True},
}
# quantities whose prior does not depend on the crater (comparison for identifiability)
POP_PRIOR_KEYS = ("log10_velocity_m_s", "log10_density_kg_m3", "angle_from_vertical_deg",
                  "log10_strength_pa", "log10_target_density_kg_m3")
CORR_KEYS = ("log10_diameter_m", "log10_velocity_m_s", "log10_density_kg_m3",
             "log10_strength_pa", "log10_target_density_kg_m3")
QS = (0.05, 0.16, 0.5, 0.84, 0.95)
# Below this effective sample size, or when the flash energy conflicts with the crater at p < CONFLICT_P, the
# weighted draws cannot represent a posterior (e.g. a 1e15 J flash on a 10 m crater gives ESS ~ 1 and
# q05 = median = q95). Every quantity is then marked "unreliable" and result_status says why; the numbers are kept
# for diagnosis only.
MIN_RELIABLE_ESS = 100.0
# posterior share within this distance (dex) of the numerical log10-mass support that marks a result as unreliable:
# the support is a computational bound, so a posterior that reaches it is set by the bound, not by the data
SUPPORT_EDGE_DEX = 0.25
MAX_SUPPORT_EDGE_SHARE = 0.01
CONFLICT_P = 1e-3
MIN_N_OUT = 100   # smallest resampled mixture infer_crater accepts


def _wq(x, w, qs=QS):
    o = np.argsort(x)
    xs, ws = x[o], w[o]
    # midpoint CDF: each draw's weight is centred on it; cumsum alone shifts every quantile
    # down by half a draw's weight
    c = np.cumsum(ws) - 0.5 * ws
    c /= c[-1] + 0.5 * ws[-1]
    return [float(np.interp(q, c, xs)) for q in qs]


# quantities the SRC backend returns as deterministic pmfs
_SRC_KEYS = (("log10_mass_kg", "grid_x", "posterior_pmf"), ("log10_energy_j", "grid_energy", "energy_pmf"),
             ("log10_momentum_kg_m_s", "grid_momentum", "momentum_pmf"))
_SRC_GRIDS = {k: (g, p) for k, g, p in _SRC_KEYS}


def _pmf_quantiles(grid, pmf, qs=QS):
    """Quantiles of a pmf on a uniform grid (midpoint CDF, as src_backend)."""
    c = (np.cumsum(pmf) - 0.5 * pmf) / pmf.sum()
    return [float(np.interp(q, c, grid)) for q in qs]


def _width90(qv):
    return qv[4] - qv[0]


def _kl_bits(post, w, prior, bins=40):
    """Histogram KL(posterior || prior) in bits (information the data added)."""
    lo = min(np.min(prior), np.min(post))
    hi = max(np.max(prior), np.max(post))
    if hi <= lo:
        return 0.0
    edges = np.linspace(lo, hi, bins + 1)
    p, _ = np.histogram(post, edges, weights=w)
    r, _ = np.histogram(prior, edges)
    p = p / p.sum() + 1e-12
    r = r / r.sum() + 1e-12
    return float(np.sum(p * np.log2(p / r)))


def classify(width_ratio: float | None, kl_bits: float | None, robust_ratio: float,
             w90: float, is_log: bool) -> tuple[str, str]:
    """Return (status, explanation).

    width_ratio : posterior 90% width / prior 90% width (None if the prior is
                  the wide mass support, where the ratio is meaningless)
    robust_ratio: envelope width / median within-scenario width
    """
    if width_ratio is not None:
        if width_ratio > 0.85 and (kl_bits is None or kl_bits < 0.15):
            return ("prior_dominated",
                    "The crater carries no usable information on this quantity; "
                    "the result reproduces the assumed population prior.")
        if width_ratio > 0.5:
            return ("weakly_constrained",
                    "The crater narrows the prior only modestly.")
    if robust_ratio > 1.6:
        return ("assumption_dependent",
                "Constrained within each scenario, but the declared assumptions "
                "(population, scaling package, target) move the answer by more "
                "than the within-scenario uncertainty.")
    if is_log and w90 <= 0.3:
        return ("well_constrained", "90% interval narrower than a factor of 2.")
    return ("constrained", "Informed by the crater, with the stated 90% interval.")


def infer_crater(
    d_obs_m: float,
    sigma_meas_dex: float = 0.03,
    *,
    populations: dict[str, float] | None = None,
    packages: tuple[str, ...] = DEFAULT_PACKAGE_SET,
    site: SiteContext = GLOBAL_CONTEXT,
    model: ModelForm = ModelForm(),
    event: EventData | None = None,
    gravity_m_s2: float = 1.62,
    n_draws: int = 30_000,
    n_out: int = 4000,
    seed: int = 0,
    want_samples: bool = True,
    max_samples_out: int = 1500,
    morphometry: dict | None = None,
    backend: str = "auto",
) -> dict:
    """Posterior of the impactor that made a crater of rim diameter d_obs_m, averaged over the declared scenarios.

    backend: "mc" = importance sampling only; "src" = the log10-mass, kinetic-energy and momentum posteriors,
    class probabilities and scenario weights from the deterministic SRC backend (src_backend.py), every other
    quantity from importance sampling; "auto" = "src" when SRC supports the input, else "mc"."""
    if backend not in ("auto", "src", "mc"):
        raise ValueError("backend must be 'auto', 'src' or 'mc'")
    if not (d_obs_m > 0) or not math.isfinite(d_obs_m):
        raise ValueError("crater diameter must be positive and finite")
    if not (sigma_meas_dex > 0) or not math.isfinite(sigma_meas_dex):
        raise ValueError("the crater-diameter uncertainty sigma_meas_dex must be positive and finite")
    pops = populations or {"asteroidal": 1.0}
    if any(not (math.isfinite(v) and v >= 0) for v in pops.values()):
        raise ValueError("population weights must be finite and non-negative")
    tot = sum(pops.values())
    if tot <= 0:
        raise ValueError("population weights must sum to a positive number")
    if not packages:
        raise ValueError("at least one scaling package is required")
    if len(set(packages)) != len(packages):
        raise ValueError("scaling packages must not repeat")
    if n_out < MIN_N_OUT:
        # the resampled mixture (histograms, information bits, budget) needs draws from every plausible scenario
        raise ValueError(f"n_out must be at least {MIN_N_OUT}")
    pops = {k: v / tot for k, v in pops.items() if v > 0}
    for k in pops:
        if k not in POPULATIONS:
            raise ValueError(f"unknown population {k}")
    for k in packages:
        if k not in PACKAGES:
            raise ValueError(f"unknown scaling package {k}")
    rng = np.random.default_rng(seed)

    if event is not None:
        validate_event(event)
    scen, incompatible, extra, d_only = [], [], [], []
    has_event = event is not None and event != EventData()
    for p in pops:
        for k in packages:
            try:
                r = run_scenario(d_obs_m, sigma_meas_dex, p, k, site=site, model=model, event=event,
                                 gravity_m_s2=gravity_m_s2, n_draws=n_draws, rng=rng)
            except RuntimeError:
                incompatible.append({"population": p, "package": k})
                continue
            # crater-level scenario weights: the declared weights are the prior fractions of
            # craters of the observed size, so only the observations beyond the diameter re-weight the
            # scenarios: log E[L_extra | D, s] = log Z_full - log Z_D (same scenario without event data)
            lx, r0 = 0.0, None
            if has_event:
                try:
                    r0 = run_scenario(d_obs_m, sigma_meas_dex, p, k, site=site, model=model, event=None,
                                      gravity_m_s2=gravity_m_s2, n_draws=max(n_draws // 2, 2000), rng=rng)
                    lx = r.log_evidence - r0.log_evidence
                except RuntimeError:
                    lx, r0 = 0.0, None
            scen.append(r); extra.append(lx); d_only.append(r0)
    if not scen:
        raise RuntimeError("no admissible draws: the observations are incompatible with every "
                           "declared impactor population and scaling package")

    # ---- Bayesian model averaging over scenarios ---------------------------------
    # prior weight: declared population weight x equal weight per package, read as the prior fraction of
    # craters of the observed size. Posterior weight proportional to prior x predictive probability of the
    # observations beyond the diameter (speed, angle, ejecta pattern, flash, ...). With a diameter alone
    # the weights equal the declared ones. The full evidence including p(D | scenario) is not used: under a
    # power-law mass prior it would penalise scenarios that need more mass for the same crater, although the
    # declared weights already refer to craters of the observed size. Incompatible scenarios are dropped.
    # ---- deterministic SRC backend ------------------------------------------------------
    src, backend_note = None, None
    if backend != "mc":
        try:
            src = {}
            for r in scen:
                kw = dict(site=site, model=model, gravity_m_s2=gravity_m_s2)
                full = run_scenario_src(d_obs_m, sigma_meas_dex, r.population, r.package, event=event, **kw)
                if not has_event:
                    base = full
                elif full.get("log_evidence_d_only") is not None:   # same-pass diameter-only evidence
                    base = {"log_evidence": full["log_evidence_d_only"]}
                else:
                    base = run_scenario_src(d_obs_m, sigma_meas_dex, r.population, r.package, event=None, **kw)
                if not (math.isfinite(full["log_evidence"]) and math.isfinite(base["log_evidence"])):
                    raise SRCUnsupported("SRC found no posterior mass on its grid")
                src[(r.population, r.package)] = (full, base)
        except SRCUnsupported as e:
            if backend == "src":
                raise
            src, backend_note = None, str(e)
    # evidence of every observation except the flash, per scenario, for the flash-consistency check (exact SRC
    # pass without the flash; mixing the SRC evidence with a Monte Carlo flash term is inconsistent where the
    # Monte Carlo is degenerate)
    lx_noflash = None
    if src is not None and event is not None and event.flash_energy_j is not None:
        from dataclasses import replace as _dc_replace
        ev_nf = _dc_replace(event, flash_energy_j=None)
        lx_noflash = []
        for r in scen:
            if ev_nf == EventData():
                lx_noflash.append(0.0)
                continue
            nf = run_scenario_src(d_obs_m, sigma_meas_dex, r.population, r.package, event=ev_nf,
                                  site=site, model=model, gravity_m_s2=gravity_m_s2)
            lx_noflash.append(nf["log_evidence"] - src[(r.population, r.package)][1]["log_evidence"])
    if src is not None:   # deterministic log predictive of the observations beyond the diameter
        extra = [src[(r.population, r.package)][0]["log_evidence"] - src[(r.population, r.package)][1]["log_evidence"]
                 for r in scen]

    prior_w = np.array([pops[r.population] / len(packages) for r in scen])
    le = np.array(extra)
    post_w = prior_w * np.exp(le - le.max())
    post_w = post_w / post_w.sum()
    post_by_scen = {(r.population, r.package): float(w) for r, w in zip(scen, post_w)}

    # ---- per-scenario summaries ------------------------------------------------
    per = []
    for i_s, r in enumerate(scen):
        q = {key: _wq(r.draws[key], r.weights) for key in QUANTITIES}
        cls_p = [float(np.sum(r.weights[r.draws["target_class"] == c]) / r.weights.sum()) for c in range(3)]
        if src is not None:
            sr = src[(r.population, r.package)][0]
            for key, g, pm in _SRC_KEYS:
                q[key] = _pmf_quantiles(sr[g], sr[pm])
            cls_p = [float(x) for x in sr["class_probs"]]
        per.append({
            "population": r.population, "package": r.package,
            "ess": round(r.ess, 1), "log_evidence": r.log_evidence,
            "log_predictive_extra": float(extra[i_s]),
            "prior_weight": pops[r.population] / len(packages),
            "posterior_weight": post_by_scen[(r.population, r.package)],
            "quantiles": q,
            "target_class_probabilities": cls_p,
        })

    # ---- mixture: posterior scenario weights ------------------------------------------
    mix_parts = []
    for r in scen:
        wt = post_by_scen[(r.population, r.package)]
        k = int(round(n_out * wt))
        if k < 1:
            continue
        part = r.resample(k, rng)
        part["package_index"] = np.full(k, float(list(packages).index(r.package)))
        part["population_index"] = np.full(k, float(list(pops).index(r.population)))
        mix_parts.append(part)
    mix = {key: np.concatenate([d[key] for d in mix_parts]) for key in mix_parts[0]}
    wm = np.ones(len(mix["log10_mass_kg"]))

    # prior samples of population-prior quantities (mixture over populations)
    prior_mix = {}
    for key in POP_PRIOR_KEYS:
        parts = []
        for r in scen:
            wt = pops[r.population] / len(packages)
            k = max(1, int(round(n_out * wt)))
            parts.append(r.prior_draws[key][:k])
        prior_mix[key] = np.concatenate(parts)

    # reported quantiles come from the weighted importance draws of all scenarios (weights: posterior scenario
    # weight x normalised importance weight), not from the n_out resampled draws, which only add resampling
    # noise (the seed-to-seed SD is about 3x smaller for the median and 5-10x smaller for q95)
    pooled_w = np.concatenate([r.weights / r.weights.sum() * post_by_scen[(r.population, r.package)] for r in scen])

    # the envelope spans the scenarios the data leave plausible (posterior weight >= 1%)
    live = [p for p in per if p["posterior_weight"] >= 0.01] or per
    quantities = {}
    for key, meta in QUANTITIES.items():
        mq = _wq(np.concatenate([r.draws[key] for r in scen]), pooled_w)
        if src is not None and key in _SRC_GRIDS:
            if key == "log10_mass_kg":
                mc_median = mq[2]
            g, pm = _SRC_GRIDS[key]
            grid = next(iter(src.values()))[0][g]
            pmf = sum(post_by_scen[k] * v[0][pm] for k, v in src.items())
            mq = _pmf_quantiles(grid, pmf)
        env_lo = min(p["quantiles"][key][0] for p in live)
        env_hi = max(p["quantiles"][key][4] for p in live)
        widths = [_width90(p["quantiles"][key]) for p in live]
        med_w = float(np.median(widths))
        robust = (env_hi - env_lo) / med_w if med_w > 0 else 1.0
        wr = kl = None
        if key in prior_mix:
            pq = _wq(prior_mix[key], np.ones(len(prior_mix[key])))
            wr = _width90(mq) / max(_width90(pq), 1e-12)
            kl = _kl_bits(mix[key], wm, prior_mix[key])
        status, why = classify(wr, kl, robust, _width90(mq), meta["log"])
        quantities[key] = {
            **meta,
            "median": mq[2], "q05": mq[0], "q16": mq[1], "q84": mq[3], "q95": mq[4],
            "width90": _width90(mq),
            "envelope90": [env_lo, env_hi],
            "robustness_ratio": robust,
            "prior_width_ratio": wr,
            "information_bits": kl,
            "status": status,
            "explanation": why,
        }

    # ---- uncertainty budget for log mass (variance shares, regression on nuisances)
    budget = _budget(mix)

    # ---- the identified relation: mass as a function of assumed velocity ----------
    fiber = _fiber_curve(mix)

    # ---- domain checks -------------------------------------------------------------
    flags = []
    if d_obs_m > 10_000:
        flags.append("Crater exceeds ~10 km: complex-crater collapse is outside this "
                     "simple-crater model; results are not valid.")
    if d_obs_m < 1.0:
        flags.append("Crater below 1 m: below the size range of observed new lunar craters.")
    elif d_obs_m < 5.0:
        flags.append("Crater below 5 m: smaller than any crater in the synthetic or measured tests; "
                     "near-surface regolith strength dominates and is poorly known.")
    lay = layered_fraction(d_obs_m, site)
    if lay > 0.2:
        flags.append(f"{lay:.0%} prior probability that this crater formed in a layered "
                     "regolith-over-rock target (4 < D/h < 20), where no validated "
                     "point-source scaling exists; treated as a mixture.")
    if any(not POPULATIONS[p].hypervelocity for p in pops):
        flags.append("Secondary-impact scenario: velocities below ~1 km/s are outside the "
                     "validated range of point-source scaling (LPI 2007; IMPACTS 2022).")
    if any(POPULATIONS[p].artificial for p in pops):
        flags.append("Artificial-impactor scenario: hollow, elongated vehicles at 1.6-2.9 km/s lie outside the "
                     "compact point-source calibration, and the density is the bulk density of the vehicle. "
                     "On five earlier-examined spacecraft craters this mode gave a mean mass error of 0.28 dex "
                     "from the diameter alone (development data, not an independent test).")
    if event is not None and event.ejecta_pattern is not None:
        flags.append("The impact angle from the ejecta pattern is provisional. In the one independent check "
                     "(17 March 2013 flash crater, angle from fireball orbits) the true angle lay about 5 deg "
                     "outside the 90% interval; Falcon 9 was examined during development.")
        if any(POPULATIONS[p].artificial for p in pops):
            flags.append("The ejecta-pattern thresholds describe compact natural impactors; hollow vehicles "
                         "can make asymmetric or elongated craters at steep angles.")
    if event is not None and event.flash_energy_j is not None and event.flash_definition != DEFAULT_FLASH_DEFINITION:
        lo, hi = FLASH_DEFINITIONS[event.flash_definition]["log10_factor"]
        flags.append(f"Flash energy given as '{FLASH_DEFINITIONS[event.flash_definition]['label']}': converted to "
                     f"the luminous-range integrated energy with a factor of {10 ** lo:.2g} to {10 ** hi:.2g}.")
    low_angle = float(np.mean(mix["angle_from_vertical_deg"] > 75.0))
    if low_angle > 0.20:  # the isotropic prior alone puts ~7% beyond 75 deg
        flags.append(f"{low_angle:.0%} of posterior draws are glancing (more than 75 deg from vertical), where "
                     "the point-source law is not valid.")
    conflict = None
    if event is not None and event.flash_energy_j is not None:
        conflict = _flash_conflict(scen, prior_w, math.log10(event.flash_energy_j), extra, lx_noflash)
        if conflict["p_two_sided"] < 0.05:
            flags.append(
                f"The flash energy and the crater are hard to reconcile (predictive p = {conflict['p_two_sided']:.3f}): "
                f"from the crater alone, a luminous energy of {10 ** conflict['pred_q05']:.2g} to "
                f"{10 ** conflict['pred_q95']:.2g} J is expected (90%). The luminous efficiency, the speed, "
                "or the scaling may not apply to this event.")
    measured = _measurement_conflicts(event, d_only, prior_w) if has_event else {}
    for name, c in measured.items():
        if c["p_two_sided"] < 0.05:
            flags.append(f"The measured {name} is hard to reconcile with the crater and the declared impactor "
                         f"populations (predictive p = {c['p_two_sided']:.1e}; from the crater alone, "
                         f"{c['pred_q05']:.3g} to {c['pred_q95']:.3g} {c['unit']} is expected, 90%).")
    if conflict is not None:
        measured["flash energy"] = conflict
    min_ess = min(r.ess for r in scen if post_by_scen[(r.population, r.package)] >= 0.01)
    if min_ess < 200:
        flags.append(f"Low effective sample size ({min_ess:.0f}) in at least one scenario.")
    reasons = []
    if min_ess < MIN_RELIABLE_ESS:
        reasons.append(f"effective sample size {min_ess:.1f} < {MIN_RELIABLE_ESS:g}: the observations are "
                       "(nearly) incompatible with the declared priors and scaling")
    lo_m, hi_m = mass_support(d_obs_m)
    edge = 0.0
    for r in scen:
        pw = post_by_scen[(r.population, r.package)]
        if pw > 0:
            lm_r = r.draws["log10_mass_kg"]
            near = (lm_r < lo_m + SUPPORT_EDGE_DEX) | (lm_r > hi_m - SUPPORT_EDGE_DEX)
            edge += pw * float(r.weights[near].sum() / r.weights.sum())
    if edge > MAX_SUPPORT_EDGE_SHARE:
        reasons.append(f"{edge:.1%} of the posterior lies at the numerical mass bound (log10 m = {lo_m:g} to {hi_m:g}): "
                       "the crater measurement is too uncertain to constrain the mass")
    if conflict is not None and conflict["p_two_sided"] < CONFLICT_P:
        reasons.append(f"flash energy incompatible with the crater (predictive p = {conflict['p_two_sided']:.1e})")
    for name, c in measured.items():
        if name != "flash energy" and c["p_two_sided"] < CONFLICT_P:
            reasons.append(f"measured {name} incompatible with the crater and the declared impactor populations "
                           f"(predictive p = {c['p_two_sided']:.1e})")
    if reasons:
        for q in quantities.values():
            q["status"], q["explanation"] = "unreliable", "Not a usable posterior: " + "; ".join(reasons) + "."
        flags.insert(0, "RESULT NOT RELIABLE: " + "; ".join(reasons) + ". The intervals below are shown for "
                     "diagnosis only.")

    out = {
        "result_status": {"reliable": not reasons, "reasons": reasons, "min_ess": min_ess},
        "backend": {"requested": backend, "log10_mass": "src" if src is not None else "mc",
                    "note": backend_note if src is None else
                    "log10 mass, kinetic energy and momentum quantiles, class probabilities and scenario weights from the "
                    "deterministic SRC backend; other quantities, samples and histograms from importance sampling",
                    "mc_minus_src_log10_mass_median": None if src is None else
                    mc_median - quantities["log10_mass_kg"]["median"]},
        "input": {
            "rim_diameter_m": d_obs_m, "sigma_meas_dex": sigma_meas_dex,
            "gravity_m_s2": gravity_m_s2,
            "populations": pops, "packages": list(packages),
            "model_form": asdict(model),
            "site": {"label": site.label, "p_mare": site.p_mare,
                     **({"h_mare_median_m": site.h_by_unit[0][0], "h_highland_median_m": site.h_by_unit[1][0]}
                        if site.h_by_unit is not None else
                        {"h_median_m": site.h_median_m, "h_sigma_ln": site.h_sigma_ln}),
                     "fixed_class": site.fixed_class,
                     "sources": list(site.sources), "notes": list(site.notes)},
            "event": asdict(event) if event else None,
        },
        "quantities": quantities,
        "scenarios": per,
        "target_classes": list(TARGET_CLASSES),
        "uncertainty_budget_log10_mass": budget,
        "mass_given_velocity": fiber,
        "flags": flags,
        "diagnostics": {"min_ess": min_ess, "n_scenarios": len(scen),
                        "draws_per_scenario": n_draws},
        "incompatible_scenarios": incompatible,
        "flash_consistency": conflict,
        "measurement_consistency": {k: v for k, v in measured.items() if k != "flash energy"},
        "population_probabilities": {
            p: {"prior": pops[p],
                "posterior": float(sum(w for (pp, _), w in post_by_scen.items() if pp == p))}
            for p in pops},
        "method": "Importance sampling of the posterior of the declared model (no neural network)",
    }
    out.update(_presentation(mix, prior_mix, per, scen, d_obs_m, morphometry or {}, pops, packages,
                             rng))
    if want_samples:
        keep = ("log10_mass_kg", "log10_energy_j", "log10_velocity_m_s",
                "log10_diameter_m", "log10_density_kg_m3", "angle_from_vertical_deg", "target_class")
        sub = rng.choice(len(mix["log10_mass_kg"]), size=min(max_samples_out, len(mix["log10_mass_kg"])),
                         replace=False)
        out["samples"] = {k: np.round(mix[k][sub], 4).tolist() for k in keep}
        out["prior_samples"] = {k: np.round(v[:1500], 4).tolist() for k, v in prior_mix.items()}
    return out


def _measurement_conflicts(event, d_only, prior_w) -> dict:
    """Prior-data conflict checks for a measured speed, angle or mass.

    For each measurement, the predictive distribution given the crater diameter alone (each scenario's
    diameter-only posterior, scenarios mixed with their declared weights, which are the diameter-only posterior
    weights) convolved with the stated measurement error. Returns the two-sided tail probability of the observed
    value and the 90% predictive range per measurement. A measurement far outside a population's support (e.g. a
    3.8 km/s speed for secondaries, which stay below 2.38 km/s) still gives a formally valid posterior pinned at the
    support edge; this check reports the conflict instead."""
    checks = []
    if event.velocity_m_s is not None:
        su = math.log(1.0 + event.velocity_sigma_rel)
        checks.append(("impact speed", "log10_velocity_m_s", lambda v: math.log(10.0) * v,
                       math.log(event.velocity_m_s), su, lambda x: math.exp(x) / 1000.0, "km/s"))
    if event.elev_deg is not None:
        checks.append(("impact elevation angle", "elevation_deg", lambda v: v, event.elev_deg, event.elev_sigma_deg,
                       lambda x: x, "deg"))
    if event.mass_kg is not None:
        checks.append(("impactor mass", "log10_mass_kg", lambda v: v, math.log10(event.mass_kg),
                       event.mass_sigma_dex, lambda x: 10.0 ** x, "kg"))
    out = {}
    for name, key, tr, obs, sd, back, unit in checks:
        xs, ws = [], []
        for r0, pw in zip(d_only, prior_w):
            if r0 is None or pw <= 0:
                continue
            w = r0.weights / r0.weights.sum() * pw
            xs.append(tr(r0.draws[key])); ws.append(w)
        if not xs:
            continue
        x, w = np.concatenate(xs), np.concatenate(ws)
        w = w / w.sum()
        z = (obs - x) / sd
        p_lo, p_hi = float(np.sum(w * ndtr(-z))), float(np.sum(w * ndtr(z)))   # P(pred >= obs), P(pred <= obs)
        pred = x + sd * np.random.default_rng(0).standard_normal(x.size)
        q = _wq(pred, w, (0.05, 0.95))
        out[name] = {"p_two_sided": min(1.0, 2.0 * min(p_lo, p_hi)), "pred_q05": back(q[0]), "pred_q95": back(q[1]),
                     "unit": unit}
    return out


def _flash_conflict(scen, prior_w, obs_log10_e, extra=None, lx_noflash=None) -> dict:
    """Prior-data conflict check for an impact-flash energy.

    Removes the flash term from each scenario's importance weights to obtain the posterior given every observation
    except the flash, and computes where the observed log luminous energy falls in that predictive distribution
    (which includes the luminous-efficiency prior and the flash measurement error).

    Scenarios are mixed with their posterior weights given the non-flash observations (speed, angle, density,
    ejecta), not their prior weights: each carries prior x exp(log Z_noflash - log Z_D). With the exact SRC
    evidence this is lx_noflash; otherwise log Z_noflash - log Z_D = extra - (log sum e^lw - log sum e^(lw - flash))
    from the same draws (extra = log Z_full - log Z_D). Prior weights would ignore, e.g., a measured 1.7 km/s speed
    that makes a secondary origin almost certain."""
    xs, ws, lsc = [], [], []
    for i, (r, pw) in enumerate(zip(scen, prior_w)):
        lw_full = r.draws["log_weight"]
        lw = lw_full - r.draws["flash_loglik"]   # exact crater-only log weights
        ok = np.isfinite(lw)
        if not ok.any():
            continue
        w = np.where(ok, np.exp(lw - lw[ok].max()), 0.0)
        xs.append(r.draws["log10_flash_energy_pred"]); ws.append(w / w.sum())
        ls = math.log(pw)
        if lx_noflash is not None:   # exact (SRC): log Z_noflash - log Z_D
            ls += lx_noflash[i]
        elif extra is not None:      # Monte Carlo only: the same sampler for both terms
            okf = np.isfinite(lw_full)
            l_full = float(np.logaddexp.reduce(lw_full[okf])) if okf.any() else -math.inf
            l_noflash = float(np.logaddexp.reduce(lw[ok]))
            ls += extra[i] - (l_full - l_noflash)
        lsc.append(ls)
    lsc = np.array(lsc)
    sw = np.exp(lsc - lsc.max())
    ws = [s * wi for s, wi in zip(sw, ws)]
    x = np.concatenate(xs); w = np.concatenate(ws); w = w / w.sum()
    p_hi = float(np.sum(w[x >= obs_log10_e]))
    q = _wq(x, w, (0.05, 0.5, 0.95))
    return {"p_two_sided": 2.0 * min(p_hi, 1.0 - p_hi), "pred_q05": q[0], "pred_q50": q[1], "pred_q95": q[2],
            "observed_log10_j": obs_log10_e}


def _budget(mix: dict) -> dict:
    """Share of log10-mass variance attributable to each nuisance (first-order).

    Uses a linear regression of log10 m on standardised nuisances. Nuisances are
    independent a priori and nearly so a posteriori (the crater pins one
    combination), so squared standardised coefficients approximate variance
    shares. The remainder is the measurement/model-form term and non-linearity.
    """
    y = mix["log10_mass_kg"]
    cols = {
        "velocity": mix["log10_velocity_m_s"],
        "impactor_density": mix["log10_density_kg_m3"],
        "impact_angle": np.log10(np.maximum(np.sin(np.radians(mix["elevation_deg"])), 1e-6)),
        "target_density": mix["log10_target_density_kg_m3"],
        "target_strength": mix["log10_strength_pa"],
        "angle_law_q": mix["angle_exponent_q"],
        "target_class": mix["target_class"].astype(float),
        "scaling_package": mix["package_index"],
        "impactor_population": mix["population_index"],
        "crater_size (measurement + model form)": mix["crater_size_residual_dex"],
    }
    names, X = [], []
    for k, v in cols.items():
        if np.std(v) > 1e-9:
            names.append(k)
            X.append((v - v.mean()) / v.std())
    if not X:
        return {"total_var": float(np.var(y)), "shares": {}}
    X = np.column_stack(X + [np.ones(len(y))])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    var = float(np.var(y))
    shares = {n: float(beta[i] ** 2 / var) for i, n in enumerate(names)}
    shares["non-linear / interaction"] = max(
        0.0, 1.0 - sum(shares.values()))
    return {"total_var": var, "shares": shares,
            "note": "First-order linear attribution; shares can exceed 1 in total when "
                    "nuisances are correlated a posteriori."}


def _linfit(x, y) -> tuple[float, float]:
    """Least-squares slope and intercept of y on x; NaN when x has (almost) no spread, e.g. a precisely measured
    speed, where the relation is undefined."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    xm, ym = float(x.mean()), float(y.mean())
    vx = float(np.mean((x - xm) ** 2))
    if not vx > 1e-12:
        return math.nan, math.nan
    slope = float(np.mean((x - xm) * (y - ym)) / vx)
    return slope, ym - slope * xm


def _fiber_curve(mix: dict) -> dict:
    """Median and 90% band of log10 mass in bins of assumed velocity."""
    lu = mix["log10_velocity_m_s"]
    lm = mix["log10_mass_kg"]
    edges = np.quantile(lu, np.linspace(0.02, 0.98, 13))
    rows = []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = (lu >= a) & (lu < b)
        if sel.sum() < 20:
            continue
        rows.append({
            "velocity_km_s": float(10 ** (0.5 * (a + b)) / 1000.0),
            "log10_mass_q05": float(np.quantile(lm[sel], 0.05)),
            "log10_mass_q50": float(np.quantile(lm[sel], 0.5)),
            "log10_mass_q95": float(np.quantile(lm[sel], 0.95)),
        })
    slope = _linfit(lu, lm)[0] if len(lu) > 10 else math.nan
    return {"bins": rows, "dlog_mass_dlog_velocity": slope,
            "note": "If the velocity were known, the mass would lie in this band. "
                    "The crater fixes this relation; it does not fix the velocity."}


# fresh lunar simple craters: d/D 0.11-0.17 below ~400 m (Stopar et al. 2017); the scaling
# model's excavation-depth convention gives Kd / (2.6 Kr) = 0.21
FRESH_DD_RANGE = (0.11, 0.17)
MODEL_DD = 0.6 / (1.3 * 2.0 * 1.1)
MODEL_RIM_HEIGHT_RATIO = 0.036


def _presentation(mix, prior_mix, per, scen, d_obs_m, morph, pops, packages, rng) -> dict:
    """Histograms, scatter, correlations, class probabilities and morphometry checks."""
    n = len(mix["log10_mass_kg"])
    marg = {}
    for key, meta in QUANTITIES.items():
        x = mix[key]
        lo, hi = np.quantile(x, [0.002, 0.998])
        if key in prior_mix:
            plo, phi = np.quantile(prior_mix[key], [0.002, 0.998])
            lo, hi = min(lo, plo), max(hi, phi)
        if hi <= lo:
            hi = lo + 1e-6
        edges = np.linspace(lo, hi, 37)
        counts, _ = np.histogram(x, edges)
        entry = {"edges": np.round(edges, 4).tolist(), "counts": counts.tolist()}
        if key in prior_mix:
            pc, _ = np.histogram(prior_mix[key], edges)
            entry["prior_counts"] = np.round(pc * (len(x) / max(len(prior_mix[key]), 1)), 2).tolist()
        marg[key] = entry
    # prior support used for the "position in prior range" bars
    from .sampler import LOG10M_DISPLAY_RANGE
    # The display range must contain the crater-dependent inference support (sampler.mass_support), so it is
    # widened by the same rule; otherwise the posterior of very small craters falls below the bar.
    _c = 3.0 * math.log10(d_obs_m)
    LOG10M_SUPPORT = (min(LOG10M_DISPLAY_RANGE[0], _c - 8.0), max(LOG10M_DISPLAY_RANGE[1], _c + 8.0))
    from .targets import CLASS_PRIORS
    P = [POPULATIONS[p] for p in pops]
    bounds = {
        "log10_velocity_m_s": [math.log10(min(x.u_lo for x in P)), math.log10(max(x.u_hi for x in P))],
        "log10_density_kg_m3": [math.log10(min(x.delta_lo for x in P)),
                                math.log10(max(7900.0 if x.iron_fraction else x.delta_hi for x in P))],
        "angle_from_vertical_deg": [0.0, 90.0],
        "log10_strength_pa": [math.log10(min(c.y_lo for c in CLASS_PRIORS.values())),
                              math.log10(max(c.y_hi for c in CLASS_PRIORS.values()))],
        "log10_target_density_kg_m3": [math.log10(min(c.rho_lo for c in CLASS_PRIORS.values())),
                                       math.log10(max(c.rho_hi for c in CLASS_PRIORS.values()))],
    }
    bounds["log10_mass_kg"] = list(LOG10M_SUPPORT)
    dl = [math.log10(min(POPULATIONS[p].delta_lo for p in pops)),
          math.log10(max(7900.0 if POPULATIONS[p].iron_fraction else POPULATIONS[p].delta_hi for p in pops))]
    bounds["log10_diameter_m"] = [
        (LOG10M_SUPPORT[0] - math.log10(math.pi / 6) - dl[1]) / 3.0,
        (LOG10M_SUPPORT[1] - math.log10(math.pi / 6) - dl[0]) / 3.0]
    # correlations among the five scaling-law quantities
    X = np.column_stack([mix[k] for k in CORR_KEYS])
    sd = X.std(axis=0)
    C = np.corrcoef(X, rowvar=False) if np.all(sd > 0) else np.eye(len(CORR_KEYS))
    sub = rng.choice(n, size=min(700, n), replace=False)
    scatter = np.round(np.column_stack([mix["log10_diameter_m"][sub], mix["log10_velocity_m_s"][sub]]), 4)
    cls = mix["target_class"]
    class_p = {TARGET_CLASSES[c]: float(np.mean(cls == c)) for c in range(3)}
    # posterior-predictive crater profiles (model fresh shape) and morphometry checks
    s_draw = math.log10(d_obs_m) + mix["crater_size_residual_dex"]
    pick = rng.choice(n, size=min(24, n), replace=False)
    sample_craters = [{"rim_diameter_m": float(10 ** s_draw[i]),
                       "depth_m": float(MODEL_DD * 10 ** s_draw[i]),
                       "rim_height_m": float(MODEL_RIM_HEIGHT_RATIO * 10 ** s_draw[i])} for i in pick]
    u = 10.0 ** mix["log10_velocity_m_s"]
    v_imp = math.pi / 6.0 * (10.0 ** mix["log10_diameter_m"]) ** 3
    yy = u ** 2 / 5.0e6 - 10.0
    melt = np.where(yy > 0, 0.5 * v_imp * np.power(np.maximum(yy, 0), 0.9), 0.0)
    mq = np.quantile(melt, [0.05, 0.5, 0.95])
    depth, rimh = morph.get("depth_m"), morph.get("rim_height_m")
    checks = {
        "depth_m": {"observed": depth,
                    "observed_dD": None if depth is None else depth / d_obs_m,
                    "fresh_crater_dD_range": list(FRESH_DD_RANGE), "model_dD": MODEL_DD},
        "rim_height_m": {"observed": rimh, "model_expected": MODEL_RIM_HEIGHT_RATIO * d_obs_m},
        "central_peak_height_m": {"observed": morph.get("central_peak_height_m")},
        "floor_diameter_m": {"observed": morph.get("floor_diameter_m")},
        "ejecta_radius_m": {"observed": morph.get("ejecta_radius_m"),
                            "ratio_to_rim_radius": None if not morph.get("ejecta_radius_m")
                            else morph["ejecta_radius_m"] / (d_obs_m / 2)},
        "melt_volume_m3": {"observed": morph.get("melt_volume_m3"),
                           "model_q05_q50_q95": [float(x) for x in mq],
                           "note": "LPI 2007 melt law evaluated on the posterior; not used to constrain "
                                   "the impactor (no calibrated measurement model for melt volume)."},
        "is_complex_regime": bool(d_obs_m > 10_000 or (morph.get("central_peak_height_m") or 0) > 0
                                  or (morph.get("floor_diameter_m") or 0) > 0),
    }
    if depth is not None:
        dd = depth / d_obs_m
        if dd > 0.3:
            checks["depth_m"]["assessment"] = "deeper than fresh simple craters; check the measurement"
        elif dd > FRESH_DD_RANGE[1]:
            checks["depth_m"]["assessment"] = "deeper than typical fresh craters"
        elif dd >= FRESH_DD_RANGE[0]:
            checks["depth_m"]["assessment"] = "within the fresh-crater range"
        else:
            checks["depth_m"]["assessment"] = "shallower than fresh craters: degraded or infilled"
    # fitted mass-velocity relation for the invariant card
    lu, lm = mix["log10_velocity_m_s"], mix["log10_mass_kg"]
    slope, icpt = _linfit(lu - math.log10(20000.0), lm)
    return {
        "marginals": marg, "prior_bounds": bounds,
        "correlations": {"names": list(CORR_KEYS), "matrix": np.round(C, 3).tolist()},
        "scatter": scatter.tolist(), "class_probabilities": class_p,
        "sample_craters": sample_craters, "morphometry_checks": checks,
        "mass_velocity_relation": {"log10_mass_at_20km_s": float(icpt), "exponent": float(slope),
                                   "residual_sd_dex": float(np.std(lm - (icpt + slope * (lu - math.log10(20000.0)))))},
        "n_posterior_samples": int(n),
    }
