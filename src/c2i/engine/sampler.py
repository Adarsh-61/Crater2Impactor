"""Exact explicit-prior posterior for one crater, per declared scenario.

Generative model (one scenario = one impactor population x one scaling package):

    nuisance  phi = (U, delta, elev, target class, rho_t, Y, h, q, nu) ~ prior
    mass      log10 m ~ p_b(log10 m)  proportional to 10^(-b log10 m) on [m_lo, m_hi]
    crater    s = log10 D_rim(m, phi) + eps_mf,      eps_mf ~ N(0, sigma_mf^2)
    data      y = log10 D_obs = s + eps_meas,        eps_meas ~ N(0, sigma_meas^2)

Because s is strictly increasing in log10 m for fixed phi (slope in
[2/(3(2+mu)), 1/3]), the posterior can be sampled exactly by importance
sampling with a transparent reparameterisation (Gustafson 2005, Stat. Sci. 20, 111):

    1. draw phi from its prior,
    2. draw s ~ N(y, sigma_meas^2 + sigma_mf^2)  (exact conditional, flat in s),
    3. solve log10 m on the fiber s(log10 m, phi) = s,
    4. weight by p_b(log10 m) * |d log10 m / d s|.

No surrogate, no MCMC. Optional event likelihoods (known velocity, known
mass, flash energy, known angle) enter as extra weights or as proposals with
the matching prior-density correction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.special import log_ndtr, ndtr, ndtri, ndtri_exp

from .flash import FLASH_DEFINITIONS
from .observations import ejecta_pattern_loglik
from .packages import PACKAGES, effective_strength, log10_rim, solve_log10_m
from .populations import POPULATIONS
from .targets import (COMPETENT, DH_BEDROCK_MIN, DH_REGOLITH_MAX, FRACTURED, GLOBAL_CONTEXT,
                      LAYERED_EXTRA_SIGMA_DEX, REGOLITH, SiteContext, class_properties, is_layered,
                      regolith_probability, regolith_thickness_params, sample_target)

LOG10M_SUPPORT = (-6.0, 20.0)   # 1 mg to 1e20 kg
# The support is a numerical domain, not a physical prior. Its upper bound must stay well above the masses of
# the largest craters handled (a 1e16 kg bound truncates the posterior of craters larger than about 50 km).
# The evidence normalisation is common to all scenarios, so model weights do not depend on the bound. The
# 'position in prior range' display bars use the narrower LOG10M_DISPLAY_RANGE.
LOG10M_DISPLAY_RANGE = (-6.0, 16.0)


def mass_support(d_obs_m: float) -> tuple[float, float]:
    """Numerical log10-mass domain for one crater: LOG10M_SUPPORT widened to at least 8 dex either side of
    3 log10 D, so that the fixed 1 mg floor cannot truncate the posterior of craters below ~0.2 m. Craters from
    ~4.6 m to 10 km keep LOG10M_SUPPORT exactly."""
    c = 3.0 * math.log10(d_obs_m)
    return min(LOG10M_SUPPORT[0], c - 8.0), max(LOG10M_SUPPORT[1], c + 8.0)

# Kinematically tilted proposal. At fixed crater size log10 m = x(target) + B(kinematics) with
# B = -3 mu log10(U sin^q elev) + (1 - 3 nu) log10 delta + const (density-ratio packages), so the mass-prior factor 10^(-b log10 m) makes the posterior of the kinematics the prior tilted by
# U^(3 mu b) sin(elev)^(3 mu b q) delta^(-b (1 - 3 nu)). Drawing U, elev and delta from these tilted priors
# (closed form for the log-normal speed, sin(2 theta) angle and log-uniform density laws) and dividing by them
# in the weight keeps the estimator exact. The same factor tilts the crater-size draw s (below); without that tilt
# a heavy low-mass weight tail makes the reported q05 seed-dependent by up to 0.1 dex. False = prior proposal.
TILT_PROPOSAL = True

# The inverse uses the forward model's target-class rule (forward._consistent_rim), so that forward prediction and
# inverse inference are one generative model. Per impactor the crater forms in the regolith if u < P_reg(D_reg/h) (the
# regolith crater stays shallow), else in the substrate if u >= P_reg(D_sub/h), else at the regime switch D* between the
# two branch sizes (u uniform; P_reg the log-linear D/h rule of targets.py). The alternative (False) draws the class
# from P_reg(D_obs/h) whatever the impactor; its implied likelihood p(D | impactor) does not integrate to one, which
# tilts the effective mass prior (by up to 0.43 dex in the median for D/h ~ 4-20). The exact inverse of the forward
# rule has three components:
#   R  regolith fibre, weight P_reg(s/h);
#   S  substrate fibre, weight 1 - max(P_reg(s_reg(m)/h), P_reg(s/h)), s_reg(m) = regolith crater of the same impactor;
#   N  crater at the switch: s inside the band and m between the two fibre solutions, density 1/log10(5) per unit s.
COHERENT_CLASS_MODEL = True
_N_SHARE = 0.2        # proposal share of the switch component N
_R_FLOOR = 0.02       # proposal floor of the R and S components (their target weights can be positive where P_reg is 0 or 1)
# typical d log10 D / d log10 m used for the crater-size tilt (exact range 2/(3(2+mu)) to 1/3); any value keeps
# the estimator exact, this one only sets its efficiency
TILT_SLOPE = 0.3
# Defensive mixture fraction of untilted (prior) draws in the tilted proposal. Event likelihoods (flash,
# rocky ejecta, ejecta pattern) can pull the posterior away from the mass-prior tilt; mixing in alpha prior draws
# bounds every weight by 1/alpha relative to the prior proposal (Hesterberg 1995). With a diameter only (and with
# speed, angle or density observations, which do not oppose the tilt) alpha = 0, the pure tilted proposal.
DEFENSIVE_ALPHA = 0.1


def _norm_logpdf(x, sd):
    return -0.5 * np.square(x / sd) - np.log(sd) - 0.5 * math.log(2.0 * math.pi)


def log_ndtr_diff(a, b):
    """log(Phi(b) - Phi(a)) for a <= b, accurate in both tails (an interval above the mean is mirrored into the
    lower tail, where log_ndtr keeps full relative precision)."""
    a, b = np.broadcast_arrays(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64))
    up = a > 0.0
    lo_, hi_ = np.where(up, -b, a), np.where(up, -a, b)
    la, lb = log_ndtr(lo_), log_ndtr(hi_)
    with np.errstate(divide="ignore", invalid="ignore"):
        return lb + np.log1p(-np.exp(la - lb))


def _trunc_normal(rng, mean, sd, lo, hi, n):
    """n draws from N(mean, sd^2) restricted to [lo, hi] and the log probability mass of the interval per draw.
    mean may be an array of length n. Inversion is done in log space on the side of the interval facing the
    mean's far tail, so intervals many sd away from the mean are sampled accurately."""
    mean = np.broadcast_to(np.asarray(mean, dtype=np.float64), (n,))
    a, b = (lo - mean) / sd, (hi - mean) / sd
    up = a > 0.0
    lo_, hi_ = np.where(up, -b, a), np.where(up, -a, b)
    log_mass = log_ndtr_diff(lo_, hi_)
    with np.errstate(divide="ignore"):
        lp = np.logaddexp(log_ndtr(lo_), np.log(rng.random(n)) + log_mass)
    z = np.clip(ndtri_exp(np.minimum(lp, 0.0)), lo_, hi_)
    return mean + sd * np.where(up, -z, z), log_mass


def _rocky_h_proposal(rng, mare, site, d_obs_m, rocky):
    """Standard normal deviates z of log regolith thickness, proposed for a rocky-ejecta observation, and
    log(prior / proposal) per draw.

    The observation likelihood sigmoid(+-b1 (REC_K_EXC D - h)) puts the posterior of h in the far tail of its prior
    for large craters without rocky ejecta (or small craters with them). Half of the draws come from the prior and
    half from the exact prior x likelihood density of z, tabulated per terrain unit on a 0.005-wide grid (piecewise
    constant per cell), so prior x likelihood / proposal <= 2 Z with Z the likelihood's prior mean."""
    n = mare.size
    edges = np.linspace(-10.0, 10.0, 4001)
    dz = edges[1] - edges[0]
    zc = 0.5 * (edges[1:] + edges[:-1])
    z = rng.standard_normal(n)
    from_post = rng.random(n) < 0.5
    g = np.zeros(n)
    h_med, h_sd = regolith_thickness_params(mare, site)
    for unit in (True, False):
        sel = mare == unit
        k = int(sel.sum())
        if not k:
            continue
        i0 = int(np.flatnonzero(sel)[0])
        hc = h_med[i0] * np.exp(h_sd[i0] * zc)
        zz = (REC_B1_MARE if unit else REC_B1_HIGHLAND) * (REC_K_EXC * d_obs_m - hc)
        lt = -0.5 * zc**2 - (np.logaddexp(0.0, -zz) if rocky else np.logaddexp(0.0, zz))
        pc = np.exp(lt - lt.max())
        pc /= pc.sum()
        cell = np.minimum(np.searchsorted(np.cumsum(pc), rng.random(k)), pc.size - 1)
        z_post = edges[cell] + dz * rng.random(k)
        z_u = np.where(from_post[sel], z_post, z[sel])
        idx = np.clip(np.floor((z_u - edges[0]) / dz).astype(np.int64), 0, pc.size - 1)
        g[sel] = np.where(np.abs(z_u) < 10.0, pc[idx] / dz, 0.0)
        z[sel] = z_u
    log_prior = -0.5 * z**2 - 0.5 * math.log(2.0 * math.pi)
    return z, log_prior - np.logaddexp(math.log(0.5) + log_prior, np.log(0.5 * g + 1e-300))


def validate_event(ev) -> None:
    """Reject event data that no likelihood can represent (non-positive values or uncertainties)."""
    def bad(x):
        return not (math.isfinite(x) and x > 0)
    if ev.velocity_m_s is not None and bad(ev.velocity_m_s):
        raise ValueError("velocity_m_s must be positive and finite")
    if bad(ev.velocity_sigma_rel):
        raise ValueError("velocity_sigma_rel must be positive and finite")
    if ev.mass_kg is not None and bad(ev.mass_kg):
        raise ValueError("mass_kg must be positive and finite")
    if bad(ev.mass_sigma_dex):
        raise ValueError("mass_sigma_dex must be positive and finite")
    if ev.elev_deg is not None and not (math.isfinite(ev.elev_deg) and 0.0 < ev.elev_deg <= 90.0):
        raise ValueError("elev_deg must lie in (0, 90] degrees")
    if bad(ev.elev_sigma_deg):
        raise ValueError("elev_sigma_deg must be positive and finite")
    if ev.delta_kg_m3 is not None and bad(ev.delta_kg_m3):
        raise ValueError("delta_kg_m3 must be positive and finite")
    if ev.flash_energy_j is not None and bad(ev.flash_energy_j):
        raise ValueError("flash_energy_j must be positive and finite")
    if bad(ev.flash_sigma_dex):
        raise ValueError("flash_sigma_dex must be positive and finite")
    e_lo, e_hi = ev.eta_range
    if bad(e_lo) or not (math.isfinite(e_hi) and e_hi > e_lo):
        raise ValueError("eta_range must be (lo, hi) with 0 < lo < hi")
    if ev.eta_draws is not None and (len(ev.eta_draws) == 0 or any(bad(x) for x in ev.eta_draws)):
        raise ValueError("eta_draws must be a non-empty list of positive values")
    if ev.flash_definition not in FLASH_DEFINITIONS:
        raise ValueError(f"unknown flash definition {ev.flash_definition}; options {list(FLASH_DEFINITIONS)}")


def defensive_alpha(ev) -> float:
    """Defensive mixture fraction for one scenario's event data (see DEFENSIVE_ALPHA)."""
    opposing = ev.flash_energy_j is not None or ev.rocky_ejecta is not None or ev.ejecta_pattern is not None
    return DEFENSIVE_ALPHA if opposing else 0.0


@dataclass(frozen=True)
class ModelForm:
    """Model-form uncertainties carried by every scenario."""

    sigma_mf_dex: float = 0.10          # residual of the point-source law (provisional)
    # optional: integrate sigma_mf over a uniform range instead of fixing it
    sigma_mf_range: tuple[float, float] | None = None
    # optional: draw sigma_mf from a posterior table (calibration layer, engine/calibration.py);
    # takes precedence over sigma_mf_range and sigma_mf_dex
    sigma_mf_draws: tuple[float, ...] | None = None
    angle_q: tuple[float, float] = (0.5, 1.0)   # U_eff = U sin(elev)^q (see the packages module docstring)
    nu: tuple[float, float] = (0.2, 0.4)        # Holsapple: experiments give 0.2-0.4
    # mass prior slope: None (default) = each population's own value (populations.Population.mass_prior_b:
    # 0.90 for natural primaries, 0 for spacecraft and secondaries); a number overrides it for
    # every scenario (0 = log-uniform mass; 0.9 = Brown et al. 2002 slope)
    mass_prior_b: float | None = None
    # extra crater-size scatter in the layered band 4 < D/h < 20 (targets.LAYERED_EXTRA_SIGMA_DEX, an assumption).
    # On the 11 real spacecraft craters the term is not supported once sigma_mf is calibrated; 0.0 is offered
    # as an opt-in for the artificial population only. None (default) = the module constant, read at call time.
    layered_extra_sigma_dex: float | None = None


# Rocky-ejecta observation model: excavation depth 0.084 D (Rajsic et al. 2026) and logistic slopes
# fitted on 346 LROC craters with a free thickness per region (mare 2.22 /m, LORO 2.00-2.39; highlands 1.01 /m,
# LORO 0.89-1.42).
REC_K_EXC = 0.084
REC_B1_MARE = 2.22
REC_B1_HIGHLAND = 1.01


@dataclass
class EventData:
    """Independent information about the same event (all optional)."""

    velocity_m_s: float | None = None
    velocity_sigma_rel: float = 0.1
    mass_kg: float | None = None
    mass_sigma_dex: float = 0.1
    elev_deg: float | None = None
    elev_sigma_deg: float = 5.0
    delta_kg_m3: float | None = None
    flash_energy_j: float | None = None          # luminous energy of an impact flash
    flash_sigma_dex: float = 0.15
    eta_range: tuple[float, float] = (5e-4, 6e-3)  # luminous efficiency, published spread; the 6e-3 upper bound (Sheward et al. 2025) is not independently verified
    # optional: luminous-efficiency draws from a posterior table (calibration layer,
    # engine/calibration.py); takes precedence over eta_range
    eta_draws: tuple[float, ...] | None = None
    # photometric definition of flash_energy_j (flash.FLASH_DEFINITIONS); eta_range applies to
    # the luminous-range integrated definition, other definitions are converted
    flash_definition: str = "luminous_range_integrated"
    # crater morphology observation (observations.EJECTA_PATTERNS); constrains the angle
    ejecta_pattern: str | None = None
    # opt-in: blocky/rocky ejecta seen around the rim (True) or not (False). Constrains the
    # regolith thickness h through P(REC | D, h) = sigmoid(b1 (REC_K_EXC D - h)), calibrated on 346 LROC
    # craters of Rajsic et al. (2026, Table SM1-SM2)
    rocky_ejecta: bool | None = None


@dataclass
class ScenarioResult:
    population: str
    package: str
    draws: dict[str, np.ndarray]
    weights: np.ndarray
    ess: float
    log_evidence: float
    prior_draws: dict[str, np.ndarray] = field(repr=False, default_factory=dict)

    def resample(self, n: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        w = self.weights / self.weights.sum()
        c = np.cumsum(w)
        idx = np.searchsorted(c, (rng.random() + np.arange(n)) / n)
        idx = np.minimum(idx, len(w) - 1)
        rng.shuffle(idx)
        return {k: v[idx] for k, v in self.draws.items()}


def draw_sigma_mf(model: ModelForm, n: int, rng: np.random.Generator) -> np.ndarray:
    """Per-draw scaling-law scatter: fixed, uniform over a range, or from a posterior table."""
    if model.sigma_mf_draws is not None:
        return rng.choice(np.asarray(model.sigma_mf_draws, dtype=np.float64), n)
    if model.sigma_mf_range is not None:
        return rng.uniform(*model.sigma_mf_range, n)
    return np.full(n, model.sigma_mf_dex)


def _lr_u(pop, k, lnu):
    """log(prior / proposal) of the speed tilt prior(U) * U^k at ln U (truncated log-normal prior)."""
    m, s = math.log(pop.u_median), pop.u_sigma_ln
    m1 = m + k * s * s
    p1 = ndtr((math.log(pop.u_hi) - m1) / s) - ndtr((math.log(pop.u_lo) - m1) / s)
    p0 = ndtr(pop._z(pop.u_hi)) - ndtr(pop._z(pop.u_lo))
    return k * m + 0.5 * (k * s) ** 2 + np.log(p1) - math.log(p0) - k * lnu


def _tilted_u(pop, k, rng, k_eval=None):
    """Draws of U from prior(U) * U^k (truncated log-normal prior) and log(prior / proposal) per draw, the
    proposal being the k_eval tilt (default k; the defensive mixture draws some points at k = 0)."""
    m, s = math.log(pop.u_median), pop.u_sigma_ln
    m1 = m + k * s * s
    a1, b1 = (math.log(pop.u_lo) - m1) / s, (math.log(pop.u_hi) - m1) / s
    p1 = ndtr(b1) - ndtr(a1)
    lnu = m1 + s * ndtri(ndtr(a1) + rng.random(k.size) * p1)
    return np.exp(lnu), _lr_u(pop, k if k_eval is None else k_eval, lnu)


def _lr_elev(k, t):
    """log(prior / proposal) of the angle tilt sin(2 theta) sin(theta)^k at t = sin(theta)."""
    return np.log(2.0 / (2.0 + k)) - k * np.log(np.maximum(t, 1e-300))


def _tilted_elev(k, rng, k_eval=None):
    """Draws of the elevation from sin(2 theta) * sin(theta)^k and log(prior / proposal) per draw.
    With t = sin(theta) the prior is 2 t dt, so the proposal is (2 + k) t^(1 + k) dt."""
    t = np.power(rng.random(k.size), 1.0 / (2.0 + k))
    return np.degrees(np.arcsin(t)), _lr_elev(k if k_eval is None else k_eval, t)


def _lr_delta(pop, c, d):
    """log(prior / proposal) of the density tilt prior(delta) delta^(-c) at delta."""
    lo, hi = pop.delta_lo, pop.delta_hi
    cs = np.where(np.abs(c) < 1e-9, 1e-9, c)
    z = (1.0 - pop.iron_fraction) * (hi ** -cs - lo ** -cs) / (-cs * math.log(hi / lo))   # E_prior[delta^-c]
    if pop.iron_fraction > 0:
        a, bb, c1 = 7000.0, 7900.0, 1.0 - cs
        z = z + pop.iron_fraction * (bb ** c1 - a ** c1) / (c1 * (bb - a))
    return np.log(z) + cs * np.log(d)


def _tilted_delta(pop, c, rng, c_eval=None):
    """Draws of delta from prior(delta) * delta^(-c) (log-uniform plus optional uniform iron component)."""
    lo, hi, n = pop.delta_lo, pop.delta_hi, c.size
    cs = np.where(np.abs(c) < 1e-9, 1e-9, c)
    e_stony = (hi ** -cs - lo ** -cs) / (-cs * math.log(hi / lo))           # E_prior[delta^-c], log-uniform part
    r = rng.random(n)
    d = np.power(lo ** -cs + r * (hi ** -cs - lo ** -cs), -1.0 / cs)
    z = (1.0 - pop.iron_fraction) * e_stony
    if pop.iron_fraction > 0:
        a, bb, c1 = 7000.0, 7900.0, 1.0 - cs
        e_iron = (bb ** c1 - a ** c1) / (c1 * (bb - a))
        z = z + pop.iron_fraction * e_iron
        iron = rng.random(n) < pop.iron_fraction * e_iron / z
        d = np.where(iron, np.power(a ** c1 + rng.random(n) * (bb ** c1 - a ** c1), 1.0 / c1), d)
    return d, _lr_delta(pop, c if c_eval is None else c_eval, d)


def _derived(lm, u, delta, elev):
    m = 10.0 ** lm
    return {
        "log10_mass_kg": lm,
        "log10_velocity_m_s": np.log10(u),
        "log10_energy_j": lm + 2.0 * np.log10(u) - math.log10(2.0),
        "log10_momentum_kg_m_s": lm + np.log10(u),
        "log10_diameter_m": np.log10(np.cbrt(6.0 * m / (math.pi * delta))),
        "log10_density_kg_m3": np.log10(delta),
        "elevation_deg": elev,
        "angle_from_vertical_deg": 90.0 - elev,
    }


def run_scenario(
    d_obs_m: float,
    sigma_meas_dex: float,
    population: str,
    package: str,
    *,
    site: SiteContext = GLOBAL_CONTEXT,
    model: ModelForm = ModelForm(),
    event: EventData | None = None,
    gravity_m_s2: float = 1.62,
    n_draws: int = 40_000,
    rng: np.random.Generator,
) -> ScenarioResult:
    pop = POPULATIONS[population]
    pkg = PACKAGES[package]
    k1a, k2a, mua, kra, sda = pkg.arrays()
    ev = event or EventData()
    validate_event(ev)
    n = int(n_draws)
    y = math.log10(d_obs_m)

    # ---- nuisance draws (prior or event-informed proposals) -----------------
    logw = np.zeros(n)
    b = pop.mass_prior_b if model.mass_prior_b is None else model.mass_prior_b
    # no tilt when the mass is observed: the posterior is then set by the mass, not by the mass prior. The crater-size
    # draw is tilted for every population with b > 0; the kinematic tilts need the closed forms of the log-normal
    # speed and sin(2 theta) angle priors
    tilt = TILT_PROPOSAL and b > 0 and ev.mass_kg is None
    tilt_kin = tilt and pop.u_sigma_ln > 0 and pop.angle_law == "sin2"

    def observed_u():
        # proposal: the speed measurement density in ln U restricted to the population's speed support. The target
        # is prior(u) * L(u) with L the same log-normal, so the weight is the prior density (in ln U) times the
        # measurement's probability mass inside the support. Every draw is admissible, also when the measured speed
        # lies several sigma outside the support
        su = math.log(1.0 + ev.velocity_sigma_rel)
        lnu, log_mass = _trunc_normal(rng, math.log(ev.velocity_m_s), su, math.log(pop.u_lo), math.log(pop.u_hi), n)
        u_ = np.clip(np.exp(lnu), pop.u_lo, pop.u_hi)
        return u_, pop.logpdf_u(u_) + np.log(u_) + log_mass

    def observed_elev():
        # proposal: the angle measurement density restricted to the population's angle support; the weight is the
        # prior density (per degree, normalised so that populations can be compared) times the measurement's
        # probability mass inside the support
        e_, log_mass = _trunc_normal(rng, ev.elev_deg, ev.elev_sigma_deg, pop.elev_lo, pop.elev_hi, n)
        e_ = np.clip(e_, pop.elev_lo, pop.elev_hi)
        return np.clip(e_, 0.5, 90.0), pop.logpdf_elev(e_) + log_mass

    coherent = COHERENT_CLASS_MODEL and site.fixed_class is None
    if ev.rocky_ejecta is not None and site.fixed_class is None:
        mare_h = rng.random(n) < site.p_mare
        h_z, lr_h = _rocky_h_proposal(rng, mare_h, site, d_obs_m, ev.rocky_ejecta)
        logw += lr_h
        cls, rho, ystr, h, mare = sample_target(n, d_obs_m, site, rng, mare=mare_h, h_z=h_z)
    else:
        cls, rho, ystr, h, mare = sample_target(n, d_obs_m, site, rng)
    lo, hi = mass_support(d_obs_m)
    # Defensive mixture (Hesterberg 1995): a fraction alpha of the draws comes from the untilted prior (tilt
    # exponents 0), the weight uses the mixture density, so prior / proposal <= 1 / alpha everywhere
    alpha = defensive_alpha(ev) if tilt else 0.0
    on = (rng.random(n) >= alpha) if alpha > 0 else np.ones(n, dtype=bool)
    lr_tilt = np.zeros(n)   # log(prior / tilted proposal) of the tilted coordinates, at the full tilt

    # ---- exact conditional for the noise-free crater size --------------------
    smf = draw_sigma_mf(model, n, rng)
    lay_sd = LAYERED_EXTRA_SIGMA_DEX if model.layered_extra_sigma_dex is None else model.layered_extra_sigma_dex

    def _shift(sg):
        # the mass-prior factor 10^(-b log10 m) with log10 m ~ s / slope tilts the crater-size conditional N(y, sig^2)
        # towards smaller s: the proposal is N(y - shift, sig^2) with shift = sig^2 b ln10 / TILT_SLOPE. The shift is
        # capped where the proposal centre reaches crater sizes made by impactors at the lower mass bound (1 mg
        # makes a crater of order 10 cm: s_floor = lo / 3 + 1), and is at least allowed to reach 2 sig; for very
        # wide crater-size likelihoods (sigma_meas ~ 2 dex) the linear tilt would otherwise push every draw below
        # the mass support. The cap changes only the efficiency, not the target
        if not tilt:
            return np.zeros_like(sg)
        return np.minimum(sg**2 * b * math.log(10.0) / TILT_SLOPE, np.maximum(2.0 * sg, y - (lo / 3.0 + 1.0)))

    if coherent:
        # The target width depends on whether the noise-free crater lies in the layered band. Where the band is
        # reachable (the proposal, shifted by the tilt, within 6 wide widths of D/h = 4-20) each draw uses the narrow
        # or the wide width with probability 1/2, elsewhere the narrow one; the weight uses the mixture density, so
        # target/proposal <= 2. Proposing every draw at the wide width instead collapses the ESS for narrow
        # likelihoods.
        sig0 = np.sqrt(sigma_meas_dex**2 + smf**2)
        sig1 = np.sqrt(sig0**2 + lay_sd**2)
        lratio = y - np.log10(h)
        reach = ((lratio > math.log10(DH_REGOLITH_MAX) - 6.0 * sig1)
                 & (lratio < math.log10(DH_BEDROCK_MIN) + 6.0 * sig1 + _shift(sig1)))
        p_wide = np.where(reach, 0.5, 0.0)
        layered = rng.random(n) < p_wide
        with np.errstate(divide="ignore"):
            lpk = (np.log1p(-p_wide), np.log(p_wide))

        def _mix_logpdf(x, centre_shift):
            """log of sum_k p_k N(x; y - shift_k, sig_k) over the two widths (shift_k from centre_shift(sig_k))."""
            terms = [lpk[i] + _norm_logpdf(x - (y - centre_shift(sk)), sk) for i, sk in enumerate((sig0, sig1))]
            return np.logaddexp(*terms)
    else:
        layered = is_layered(d_obs_m / h) if site.fixed_class is None else np.zeros(n, dtype=bool)
    sig = np.sqrt(sigma_meas_dex**2 + smf**2 + np.where(layered, lay_sd**2, 0.0))
    if tilt:
        shift = _shift(sig)
        s = y - np.where(on, shift, 0.0) + sig * rng.standard_normal(n)
        if coherent:   # tilted vs untilted two-width mixture densities
            lr_tilt += _mix_logpdf(s, lambda sg: 0.0) - _mix_logpdf(s, _shift)
        else:
            lr_tilt += ((s - y + shift) ** 2 - (s - y) ** 2) / (2.0 * sig**2)
    else:
        s = y + sig * rng.standard_normal(n)

    if coherent:
        # Proposal component per draw (0 = R, 1 = S, 2 = N) given the noise-free size s, so that it follows the
        # target weights P_reg(s/h) (R) and at most 1 - P_reg(s/h) (S) wherever the crater-size draw lands (with a
        # wide crater-size likelihood the posterior sizes can lie decades below D_obs). N is proposed only where
        # 0 < P_reg(s/h) < 1, where its target weight can be positive. The class label used by the tilt is
        # R -> regolith, S/N -> the local substrate; u_rho and u_y are the fixed uniforms of the target properties.
        p_s0 = regolith_probability(10.0 ** s / h)
        p_reg = _R_FLOOR + (1.0 - 2.0 * _R_FLOOR) * p_s0
        p_n = np.where((p_s0 > 0.0) & (p_s0 < 1.0), _N_SHARE, 0.0)
        comp = np.where(rng.random(n) < p_reg, 0, 1)
        comp = np.where(rng.random(n) < p_n, 2, comp)
        with np.errstate(divide="ignore"):
            log_pi = np.where(comp == 0, np.log(p_reg), np.log1p(-p_reg)) + np.log1p(-p_n)
            log_pi = np.where(comp == 2, np.log(p_n), log_pi)
        sub = np.where(mare, COMPETENT, FRACTURED)
        cls = np.where(comp == 0, REGOLITH, sub)
        u_rho, u_y = rng.random(n), rng.random(n)

    q = rng.uniform(*model.angle_q, n)
    nu = rng.uniform(*model.nu, n)
    if tilt_kin:   # the tilt depends on the class mu, q and nu (TILT_PROPOSAL)
        k_u = 3.0 * mua[cls] * b
        if ev.velocity_m_s is not None:
            u, lr = observed_u()
            logw += lr
        else:
            u, lr = _tilted_u(pop, np.where(on, k_u, 0.0), rng, k_eval=k_u)
            lr_tilt += lr
        if ev.delta_kg_m3 is not None:
            delta = np.full(n, float(ev.delta_kg_m3))
            logw += pop.logpdf_delta(delta)
        elif pkg.density_ratio:
            c_d = b * (1.0 - 3.0 * nu)
            delta, lr = _tilted_delta(pop, np.where(on, c_d, 0.0), rng, c_eval=c_d)
            lr_tilt += lr
        else:
            delta = pop.sample_delta(n, rng)
        if ev.elev_deg is not None:
            elev, lr = observed_elev()
            logw += lr
        elif ev.ejecta_pattern is not None:   # the pattern can pull the angle far from the tilt (e.g. butterfly ejecta)
            elev = pop.sample_elev(n, rng)
        else:
            elev, lr = _tilted_elev(np.where(on, k_u * q, 0.0), rng, k_eval=k_u * q)
            lr_tilt += lr
    else:
        if ev.velocity_m_s is not None:
            u, lr = observed_u()
            logw += lr
        else:
            u = pop.sample_u(n, rng)
        if ev.delta_kg_m3 is not None:
            delta = np.full(n, float(ev.delta_kg_m3))
            logw += pop.logpdf_delta(delta)
        else:
            delta = pop.sample_delta(n, rng)
        if ev.elev_deg is not None:
            elev, lr = observed_elev()
            logw += lr
        else:
            elev = pop.sample_elev(n, rng)
    if tilt:   # mixture density: prior / q_mix = 1 / ((1 - alpha) exp(-lr_tilt) + alpha)
        logw += lr_tilt if alpha == 0 else -np.logaddexp(math.log1p(-alpha) - lr_tilt, math.log(alpha))
    if ev.ejecta_pattern is not None:
        logw += ejecta_pattern_loglik(ev.ejecta_pattern, elev)
    if ev.rocky_ejecta is not None:
        if site.fixed_class is not None:
            raise ValueError("rocky_ejecta needs a regolith-thickness model; do not combine it with fixed_class")
        z = np.where(mare, REC_B1_MARE, REC_B1_HIGHLAND) * (REC_K_EXC * d_obs_m - h)
        logw -= np.logaddexp(0.0, -z) if ev.rocky_ejecta else np.logaddexp(0.0, z)   # log sigmoid(+-z)
    kw = dict(u_m_s=u, delta=delta, g=gravity_m_s2, density_ratio=pkg.density_ratio, nu=nu, elev_deg=elev, angle_q=q,
              rim_factor=pkg.rim_factor)

    def fibre(c, rho_c, y_c):
        """Exact inverse of the rim law for class array c at the noise-free size s: (lm, ds/dlm, pi2, st, y_eff).
        With size-dependent strength (impacts2022 rock classes) Y_eff depends on s itself, so the noise-free
        crater size solves s = s_rim(lm, Y_eff(s)) and the Jacobian is the total derivative
        ds/dlm = slope / (1 - mu f_str / 8) (ds_rim/dlog Y = -(mu/2) f_str, dlog Y_eff/ds = -1/4), not the
        partial slope."""
        ye = effective_strength(y_c, 10.0 ** s, sda[c])
        lm_, sl_, p2_, st_ = solve_log10_m(s, rho_t=rho_c, y_eff=ye, k1=k1a[c], k2=k2a[c], mu=mua[c], kr=kra[c], **kw)
        sl_ = np.where(sda[c], sl_ / (1.0 - mua[c] * (st_ / (st_ + p2_)) / 8.0), sl_)
        return lm_, sl_, p2_, st_, ye

    if coherent:
        # the forward model decides the layered band from the noise-free size: proposal width sig, target width sig_t
        sig_t = np.sqrt(sigma_meas_dex**2 + smf**2 + np.where(is_layered(10.0 ** s / h), lay_sd**2, 0.0))
        logw += _norm_logpdf(s - y, sig_t) - _mix_logpdf(s, lambda sg: 0.0)
        logw -= log_pi
        reg = np.full(n, REGOLITH)
        rho_r, y_r = class_properties(reg, u_rho, u_y)
        rho_s, y_s = class_properties(sub, u_rho, u_y)
        lm_r, sl_r, p2_r, st_r, ye_r = fibre(reg, rho_r, y_r)
        lm_s, sl_s, p2_s, st_s, ye_s = fibre(sub, rho_s, y_s)
        # regolith crater of the substrate-branch impactor (fixed point if the regolith strength were size dependent)
        s_rs = s.copy()
        for _ in range(25 if sda[REGOLITH] else 1):
            s_rs = log10_rim(lm_s, rho_t=rho_r, y_eff=effective_strength(y_r, 10.0 ** s_rs, sda[REGOLITH]),
                             k1=k1a[REGOLITH], k2=k2a[REGOLITH], mu=mua[REGOLITH], kr=kra[REGOLITH], **kw)[0]
        p_s = regolith_probability(10.0 ** s / h)
        p_rs = regolith_probability(10.0 ** s_rs / h)
        n_ok = (p_s > 0.0) & (p_s < 1.0) & (lm_s > lm_r)
        lm_n = lm_r + np.where(n_ok, lm_s - lm_r, 0.0) * rng.random(n)
        with np.errstate(divide="ignore"):
            f_r = np.log(p_s) - np.log(sl_r)
            f_s = np.log(1.0 - np.maximum(p_rs, p_s)) - np.log(sl_s)
            f_n = np.where(n_ok, -math.log(math.log10(DH_BEDROCK_MIN / DH_REGOLITH_MAX))
                           + np.log(np.where(n_ok, lm_s - lm_r, 1.0)), -np.inf)
        is_r = comp == 0
        lm = np.select([is_r, comp == 1], [lm_r, lm_s], lm_n)
        logw += np.select([is_r, comp == 1], [f_r, f_s], f_n)
        rho = np.where(is_r, rho_r, rho_s)
        ystr = np.where(is_r, y_r, y_s)
        y_eff = np.where(is_r, ye_r, ye_s)
        pi2 = np.where(is_r, p2_r, p2_s)
        st = np.where(is_r, st_r, st_s)
        if (comp == 2).any():   # N draws: regime terms of the sampled impactor in the substrate (reporting only)
            _, _, p2_n, st_n = log10_rim(lm_n, rho_t=rho_s, y_eff=ye_s, k1=k1a[sub], k2=k2a[sub], mu=mua[sub],
                                         kr=kra[sub], **kw)
            pi2, st = np.where(comp == 2, p2_n, pi2), np.where(comp == 2, st_n, st)
        cls = np.where(is_r, REGOLITH, sub)
        inside = (lm >= lo) & (lm <= hi) & np.isfinite(lm)
        logw += np.where(inside, -b * math.log(10.0) * lm, -np.inf)
    else:
        lm, slope, pi2, st, y_eff = fibre(cls, rho, ystr)
        inside = (lm >= lo) & (lm <= hi) & np.isfinite(lm)
        logw += np.where(inside, -b * math.log(10.0) * lm - np.log(slope), -np.inf)

    if ev.mass_kg is not None:
        logw += -0.5 * ((lm - math.log10(ev.mass_kg)) / ev.mass_sigma_dex) ** 2
    flash_ll = np.zeros(n)
    if ev.flash_energy_j is not None:
        e_obs = math.log10(ev.flash_energy_j)
        lek = lm + 2.0 * np.log10(u) - math.log10(2.0)
        # reported energy = luminous-range energy / F, with log10 F uniform over the declared range
        f_lo, f_hi = FLASH_DEFINITIONS[ev.flash_definition]["log10_factor"]
        lf = rng.uniform(f_lo, f_hi, n) if f_hi > f_lo else np.full(n, f_lo)
        if ev.eta_draws is not None:
            eta_tab = np.log10(np.asarray(ev.eta_draws, dtype=np.float64))
            leta = rng.choice(eta_tab, n)
            leta_prior = rng.choice(eta_tab, n)
            flash_ll = -0.5 * ((e_obs - (leta + lek - lf)) / ev.flash_sigma_dex) ** 2
        else:
            # log10 eta (uniform prior) is integrated out analytically and then drawn from its exact conditional,
            # a normal truncated to the eta range. Drawing it from the prior instead leaves only the draws whose
            # eta happens to match the flash, which collapses the ESS for a precise flash energy.
            # Same normalisation as the unnormalised Gaussian term: sqrt(2 pi) sigma / width * P(eta in range).
            e_lo, e_hi = math.log10(ev.eta_range[0]), math.log10(ev.eta_range[1])
            leta, log_mass = _trunc_normal(rng, e_obs - lek + lf, ev.flash_sigma_dex, e_lo, e_hi, n)
            flash_ll = log_mass + math.log(math.sqrt(2.0 * math.pi) * ev.flash_sigma_dex / (e_hi - e_lo))
            leta_prior = rng.uniform(e_lo, e_hi, n)
        logw += np.where(np.isfinite(lek), flash_ll, -np.inf)
        # predicted flash energy for the prior-data conflict check: luminous efficiency from its prior
        flash_pred = leta_prior + lek - lf + ev.flash_sigma_dex * rng.standard_normal(n)
    else:
        leta = np.full(n, np.nan)
        flash_pred = np.full(n, np.nan)

    finite = np.isfinite(logw)
    if not finite.any():
        raise RuntimeError("no admissible draws: observation incompatible with the declared priors")
    mx = logw[finite].max()
    w = np.where(finite, np.exp(logw - mx), 0.0)
    ess = float(w.sum() ** 2 / np.sum(w**2))
    # log evidence up to a scenario-independent constant (mass-prior normalisation included)
    if b == 0:
        log_norm = -math.log((hi - lo) * math.log(10.0))
    else:
        bl = b * math.log(10.0)
        # this normalisation is already per unit log10 m, so subtract the ln(ln 10) added below for
        # both branches; log_ev is then continuous at b -> 0+
        log_norm = math.log(bl) + bl * lo - math.log(-math.expm1(-bl * (hi - lo))) - math.log(math.log(10.0))
    log_ev = float(mx + math.log(w.mean()) + log_norm + math.log(math.log(10.0)))

    draws = _derived(lm, u, delta, elev)
    draws.update({
        "target_class": cls.astype(np.int64),
        "log10_target_density_kg_m3": np.log10(rho),
        "log10_strength_pa": np.log10(ystr),
        "log10_strength_eff_pa": np.log10(y_eff),
        "regolith_thickness_m": h,
        "angle_exponent_q": q,
        "nu": nu,
        "regime_strength_fraction": st / (st + pi2),
        "log10_eta": leta,
        # for the prior-data conflict check: the flash log-likelihood term and the predicted
        # log luminous energy (with measurement noise) of each draw
        "flash_loglik": flash_ll,
        "log_weight": logw,
        "log10_flash_energy_pred": flash_pred,
        "crater_size_residual_dex": s - y,
    })
    p_el = pop.sample_elev(n, rng)
    if coherent:
        # With the per-impactor class rule there is no crater-free class prior, so the reference for the strength
        # and target-density information labels is the class-conditional prior mixed with the POSTERIOR class
        # probabilities: the labels measure within-class information (class information is reported as class
        # probabilities). A reference drawn at the observed D would label an uninformative measurement as
        # constraining the target density.
        pc = np.array([w[cls == c].sum() for c in (REGOLITH, FRACTURED, COMPETENT)])
        p_cls = rng.choice(np.array([REGOLITH, FRACTURED, COMPETENT]), size=n, p=pc / pc.sum())
        p_rho, p_y = class_properties(p_cls, rng.random(n), rng.random(n))
    else:
        _, p_rho, p_y, _, _ = sample_target(n, d_obs_m, site, rng)
    prior = {
        "log10_velocity_m_s": np.log10(pop.sample_u(n, rng)),
        "log10_density_kg_m3": np.log10(pop.sample_delta(n, rng)),
        "elevation_deg": p_el,
        "angle_from_vertical_deg": 90.0 - p_el,
        "log10_strength_pa": np.log10(p_y),
        "log10_target_density_kg_m3": np.log10(p_rho),
    }
    return ScenarioResult(population, package, draws, w, ess, log_ev, prior)
