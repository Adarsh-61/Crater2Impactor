"""Target model: which material a crater of a given size forms in, and its properties.

A crater of rim diameter D forms in the surface regolith only while it is
shallow compared with the regolith thickness h. Quaide & Oberbeck (1968)
found normal bowl-shaped craters for D/h up to about 4, flat-floored and
central-mound craters up to D/h of about 8-10, and concentric craters up to
about 20 (thresholds as quoted by Prieur et al. 2018). Rajsic et al. (2026)
locate the onset of rocky ejecta at excavation depth 0.084 D = h, i.e.
D/h ~ 12. The engine therefore uses regolith scaling for D/h <= 4, substrate
scaling for D/h >= 20, and a log-linear mixture with extra model-form scatter
in between, where no validated point-source law exists (Prieur et al. 2018).

All parameter ranges carry their source; they are priors, not measurements
of any specific site.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .packages import TARGET_CLASSES

REGOLITH, FRACTURED, COMPETENT = 0, 1, 2

# D/h thresholds from Quaide & Oberbeck (1968) morphologies: bowl-shaped craters up to ~4, concentric up to ~20.
DH_REGOLITH_MAX = 4.0
DH_BEDROCK_MIN = 20.0
LAYERED_EXTRA_SIGMA_DEX = 0.10   # extra model-form scatter in the layered band (assumed, not calibrated)


@dataclass(frozen=True)
class ClassPrior:
    rho_lo: float
    rho_hi: float
    y_lo: float
    y_hi: float
    source: str


CLASS_PRIORS: dict[int, ClassPrior] = {
    REGOLITH: ClassPrior(
        1500.0, 1900.0, 1.0e3, 2.0e5,
        "Bulk density 1.5-1.9 g/cc averaged over crater depth (Carrier et al. 1991, Lunar "
        "Sourcebook ch. 9). Effective cratering strength is not measured: log-uniform from "
        "~1 kPa (measured regolith cohesion, Carrier et al. 1991) to 0.2 MPa (Holsapple 1993 "
        "'dry soil' 0.18 MPa); LPI 2007 preset 10 kPa, IMPACTS 2022 0.1 MPa.",
    ),
    FRACTURED: ClassPrior(
        2300.0, 2900.0, 1.0e5, 1.0e7,
        "Highland upper-crust bulk density 2.3-2.9 g/cc (GRAIL; Wahl et al. 2020 grid, mean "
        "2536 kg/m^3; Wieczorek et al. 2013). Strength 0.1-10 MPa for fractured megaregolith "
        "(proposal; lower than intact rock; IMPACTS soft rock 1.3 MPa lab scale).",
    ),
    COMPETENT: ClassPrior(
        2800.0, 3350.0, 1.0e6, 2.0e7,
        "Mare basalt: grain density 3.27-3.46 g/cc (Kiefer et al. 2012) times (1 - porosity), "
        "porosity 2-15% -> 2.8-3.35 g/cc. Strength 1-20 MPa (Holsapple 1993 hard rock 18 MPa; "
        "LPI hard-rock preset 10 MPa; IMPACTS 14.4 MPa lab scale).",
    ),
}


@dataclass(frozen=True)
class SiteContext:
    """Prior information about the crater site (from coordinates or the user)."""

    p_mare: float = 0.17                      # global: maria cover ~17% of the Moon
    h_median_m: float = 8.0                   # regolith thickness median
    h_sigma_ln: float = 0.7                   # log-normal spread of h
    label: str = "Global Moon (no location)"
    sources: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    fixed_class: int | None = None            # user override: force a class
    h_by_unit: tuple[tuple[float, float], tuple[float, float]] | None = None
    # ((mare median, mare log-sd), (highland median, highland log-sd)); overrides h_median_m
    extra: dict = field(default_factory=dict)


GLOBAL_CONTEXT = SiteContext(
    p_mare=0.18,
    h_by_unit=((4.5, 0.59), (5.3, 0.69)),
    # scalar summary for display only (h_by_unit is what the engine uses): the highland unit, 82% of the surface
    h_median_m=5.3,
    h_sigma_ln=0.69,
    sources=(
        "Mare fraction: 17.9% of the surface between 65S and 65N (LROC mare boundaries, "
        "Nelson et al. 2014; computed in scripts/build_lunar_context.py).",
        "Regolith thickness: geometric means 4.5 m (mare, n=19) and 5.3 m (highland, n=35) of the "
        "published point measurements compiled by Rajsic et al. (2026); site-scale spread factors "
        "~1.8 (mare) and ~2 (highland). Leave-one-out tests show location does not predict "
        "thickness better than these unit values.",
    ),
)


def regolith_thickness_params(mare: np.ndarray, ctx: SiteContext) -> tuple[np.ndarray, np.ndarray]:
    """Median (m) and log-normal spread of the regolith thickness per draw (by mare / highland unit)."""
    if ctx.h_by_unit is not None:
        (hm, sm), (hh, sh) = ctx.h_by_unit
        return np.where(mare, hm, hh), np.where(mare, sm, sh)
    return np.full(mare.shape, ctx.h_median_m), np.full(mare.shape, ctx.h_sigma_ln)


def sample_target(n: int, d_obs_m: float, ctx: SiteContext, rng: np.random.Generator, *, mare=None, h_z=None):
    """Draw (class, rho_t, Y, h_regolith, is_mare) for a crater of diameter d_obs_m. mare and h_z (the standard
    normal deviate of log h) may be supplied by a proposal; they are drawn from the prior otherwise."""
    if ctx.fixed_class is not None:
        cls = np.full(n, int(ctx.fixed_class))
        h = np.full(n, np.nan)
        mare = np.zeros(n, dtype=bool)
    else:
        mare = rng.random(n) < ctx.p_mare if mare is None else mare
        z = rng.standard_normal(n) if h_z is None else h_z
        if ctx.h_by_unit is not None:
            (hm, sm), (hh, sh) = ctx.h_by_unit
            h = np.where(mare, hm * np.exp(sm * z), hh * np.exp(sh * z))
        else:
            h = ctx.h_median_m * np.exp(ctx.h_sigma_ln * z)
        p_reg = regolith_probability(d_obs_m / h)
        in_reg = rng.random(n) < p_reg
        sub = np.where(mare, COMPETENT, FRACTURED)
        cls = np.where(in_reg, REGOLITH, sub)
    rho = np.empty(n)
    y = np.empty(n)
    for c, pr in CLASS_PRIORS.items():
        sel = cls == c
        k = int(sel.sum())
        if k:
            rho[sel] = rng.uniform(pr.rho_lo, pr.rho_hi, k)
            y[sel] = np.exp(rng.uniform(math.log(pr.y_lo), math.log(pr.y_hi), k))
    return cls, rho, y, h, mare


def class_properties(cls, u_rho, u_y):
    """(rho_t, Y) of each draw's class from fixed uniforms: rho uniform and Y log-uniform over the class prior. Shared by
    the forward model and the inverse sampler, so a regolith and a substrate branch of the same draw use the
    same quantiles of their class priors."""
    cls = np.asarray(cls)
    rho = np.empty(cls.size)
    y = np.empty(cls.size)
    for c, pr in CLASS_PRIORS.items():
        sel = cls == c
        if sel.any():
            rho[sel] = pr.rho_lo + (pr.rho_hi - pr.rho_lo) * u_rho[sel]
            y[sel] = np.exp(math.log(pr.y_lo) + (math.log(pr.y_hi) - math.log(pr.y_lo)) * u_y[sel])
    return rho, y


def regolith_probability(ratio):
    """P(regolith-controlled) given D/h: 1 for D/h <= 4, 0 for D/h >= 20, log-linear between."""
    r = np.asarray(ratio, dtype=np.float64)
    w = np.log(np.maximum(r, 1e-12) / DH_REGOLITH_MAX) / math.log(DH_BEDROCK_MIN / DH_REGOLITH_MAX)
    return np.clip(1.0 - w, 0.0, 1.0)


def is_layered(ratio):
    r = np.asarray(ratio, dtype=np.float64)
    return (r > DH_REGOLITH_MAX) & (r < DH_BEDROCK_MIN)


def layered_fraction(d_obs_m: float, ctx: SiteContext, n: int = 20000, seed: int = 0) -> float:
    """Prior probability that the crater is in the layered (4 < D/h < 20, is_layered) regime."""
    if ctx.fixed_class is not None:
        return 0.0
    rng = np.random.default_rng(seed)
    _, _, _, h, _ = sample_target(n, d_obs_m, ctx, rng)
    return float(np.mean(is_layered(d_obs_m / h)))


__all__ = [
    "TARGET_CLASSES", "REGOLITH", "FRACTURED", "COMPETENT", "CLASS_PRIORS",
    "SiteContext", "GLOBAL_CONTEXT", "sample_target", "layered_fraction",
    "regolith_probability", "is_layered", "LAYERED_EXTRA_SIGMA_DEX", "class_properties",
    "regolith_thickness_params",
]
