"""Forward prediction with model-form uncertainty: impactor -> crater diameter.

Used by the Forward page and by the controlled-impact validation. The
spread comes only from the declared model-form and target uncertainties:
scaling package, angle law, nu, target class/density/strength, the
residual sigma_mf of the point-source law, and the extra layered-target
scatter for craters with 4 < D/h < 20. The target class is chosen per
impactor so that it agrees with the crater that impactor makes
(_consistent_rim); the inverse sampler inverts exactly this rule
(sampler.COHERENT_CLASS_MODEL), so forward and inverse are one generative model.
"""

from __future__ import annotations

import math

import numpy as np

from .packages import DEFAULT_PACKAGE_SET, PACKAGES, TARGET_CLASSES, effective_strength, log10_rim
from .sampler import ModelForm, draw_sigma_mf
from .targets import (COMPETENT, DH_BEDROCK_MIN, DH_REGOLITH_MAX, FRACTURED, GLOBAL_CONTEXT,
                      LAYERED_EXTRA_SIGMA_DEX, REGOLITH, SiteContext, class_properties, is_layered,
                      regolith_probability)

# Per-draw self-consistent target class. Each draw owns fixed uniforms (mare, h, u_reg, u_rho, u_y); the rim is
# solved for the regolith and the substrate branch, and the class is the self-consistent one: regolith if
# u_reg < P_reg(D_reg / h), substrate if u_reg >= P_reg(D_sub / h). If neither is consistent (D_reg > D_sub
# straddles the switch, where a fixed-point iteration would cycle) the crater sits at the switch size
# D* = h 4 5^(1 - u_reg), between the two; if both are, the regolith branch is taken. The uniforms must be fixed
# per draw: re-drawing the target with per-class sampling shifts every draw's (rho, Y) whenever any draw changes
# class, so a fixed point on re-drawn targets does not converge.


def _target_uniforms(n, site, rng):
    if site.fixed_class is not None:
        return {"fixed": int(site.fixed_class), "u_rho": rng.random(n), "u_y": rng.random(n), "h": np.full(n, np.nan)}
    mare = rng.random(n) < site.p_mare
    z = rng.standard_normal(n)
    if site.h_by_unit is not None:
        (hm, sm), (hh, sh) = site.h_by_unit
        h = np.where(mare, hm * np.exp(sm * z), hh * np.exp(sh * z))
    else:
        h = site.h_median_m * np.exp(site.h_sigma_ln * z)
    return {"fixed": None, "mare": mare, "h": h, "u_reg": rng.random(n), "u_rho": rng.random(n), "u_y": rng.random(n)}


def _class_props(cls, u_rho, u_y):
    return class_properties(cls, u_rho, u_y)


def _rim_for_class(cls, tu, lm, u, de, pkg, nu, el, q, g):
    """Noise-free log10 rim for given classes; size-dependent strength solved by a contraction (factor <= mu/8)."""
    k1, k2, mu, kr, sd = pkg.arrays()
    rho, y = _class_props(cls, tu["u_rho"], tu["u_y"])
    s = np.ones(cls.size)
    for _ in range(25 if sd.any() else 1):
        s_new, _, pi2, st = log10_rim(lm, u, de, rho, effective_strength(y, 10.0 ** s, sd[cls]), k1[cls], k2[cls],
                                      mu[cls], kr[cls], g=g, density_ratio=pkg.density_ratio, nu=nu, elev_deg=el,
                                      angle_q=q, rim_factor=pkg.rim_factor)
        if np.max(np.abs(s_new - s)) < 1e-10:
            s = s_new
            break
        s = s_new
    return s, pi2, st


def _consistent_rim(tu, lm, u, de, pkg, nu, el, q, g):
    """(log10 rim, class, pi2, st) per draw with a self-consistent target class (see the module note above)."""
    n = lm.size
    if tu["fixed"] is not None:
        cls = np.full(n, tu["fixed"])
        s, pi2, st = _rim_for_class(cls, tu, lm, u, de, pkg, nu, el, q, g)
        return s, cls, pi2, st
    h, ureg = tu["h"], tu["u_reg"]
    sub = np.where(tu["mare"], COMPETENT, FRACTURED)
    s_r, pi2_r, st_r = _rim_for_class(np.full(n, REGOLITH), tu, lm, u, de, pkg, nu, el, q, g)
    s_s, pi2_s, st_s = _rim_for_class(sub, tu, lm, u, de, pkg, nu, el, q, g)
    reg_ok = ureg < regolith_probability(10.0 ** s_r / h)
    sub_ok = ureg >= regolith_probability(10.0 ** s_s / h)
    use_reg = reg_ok
    neither = ~reg_ok & ~sub_ok
    s = np.where(use_reg, s_r, s_s)
    cls = np.where(use_reg, REGOLITH, sub)
    if neither.any():   # crater at the regime switch, between the two branch sizes
        d_star = h * DH_REGOLITH_MAX * (DH_BEDROCK_MIN / DH_REGOLITH_MAX) ** (1.0 - ureg)
        lo, hi = np.minimum(s_r, s_s), np.maximum(s_r, s_s)
        s = np.where(neither, np.clip(np.log10(d_star), lo, hi), s)
    pi2 = np.where(use_reg, pi2_r, pi2_s)
    st = np.where(use_reg, st_r, st_s)
    return s, cls, pi2, st


def forward_predict(
    mass_kg: float,
    velocity_m_s: float,
    elev_deg: float,
    delta: float,
    *,
    site: SiteContext = GLOBAL_CONTEXT,
    packages: tuple[str, ...] = DEFAULT_PACKAGE_SET,
    model: ModelForm = ModelForm(),
    gravity_m_s2: float = 1.62,
    n: int = 20_000,
    seed: int = 0,
    include_mf_residual: bool = True,
) -> dict:
    rng = np.random.default_rng(seed)
    lm = math.log10(mass_kg)
    per = {}
    all_s = []
    for key in packages:
        pkg = PACKAGES[key]
        q = rng.uniform(*model.angle_q, n)
        nu = rng.uniform(*model.nu, n)
        tu = _target_uniforms(n, site, rng)
        h = tu["h"]
        s, cls, pi2, st = _consistent_rim(tu, np.full(n, lm), np.full(n, float(velocity_m_s)), np.full(n, float(delta)),
                                          pkg, nu, np.full(n, float(elev_deg)), q, gravity_m_s2)
        if include_mf_residual:
            lay = _layered_mask(s, h, site)   # band decided by the noise-free size, as in simulate_truth
            s = s + draw_sigma_mf(model, n, rng) * rng.standard_normal(n)
            if lay.any():
                s = s + np.where(lay, _layered_sd(model), 0.0) * rng.standard_normal(n)
        all_s.append(s)
        per[key] = {
            "label": pkg.label,
            "rim_diameter_m_quantiles": [float(10 ** np.quantile(s, qq)) for qq in (0.05, 0.5, 0.95)],
            "rim_diameter_m_q05_q16_q50_q84_q95": [float(10 ** np.quantile(s, qq))
                                                   for qq in (0.05, 0.16, 0.5, 0.84, 0.95)],
            "target_class_fractions": [float(np.mean(cls == c)) for c in range(3)],
            "strength_regime_fraction": float(np.mean(st > pi2)),
        }
    s_all = np.concatenate(all_s)
    qs = [float(10 ** np.quantile(s_all, qq)) for qq in (0.05, 0.16, 0.5, 0.84, 0.95)]
    return {
        "input": {"mass_kg": mass_kg, "velocity_m_s": velocity_m_s, "elevation_deg": elev_deg,
                  "impactor_density_kg_m3": delta, "site": site.label,
                  "packages": list(packages)},
        "rim_diameter_m": {"q05": qs[0], "q16": qs[1], "median": qs[2], "q84": qs[3], "q95": qs[4]},
        "log10_width90": float(np.quantile(s_all, 0.95) - np.quantile(s_all, 0.05)),
        "per_package": per,
        "target_classes": list(TARGET_CLASSES),
        "kinetic_energy_j": 0.5 * mass_kg * velocity_m_s**2,
        "note": ("Spread reflects declared model-form and target uncertainty only; "
                 "it is not a measured accuracy."),
    }


def forward_samples(
    log10_mass_kg,
    velocity_m_s,
    elev_deg,
    delta,
    *,
    site: SiteContext = GLOBAL_CONTEXT,
    packages: tuple[str, ...] = DEFAULT_PACKAGE_SET,
    model: ModelForm = ModelForm(),
    gravity_m_s2: float = 1.62,
    rng: np.random.Generator,
    include_mf_residual: bool = True,
) -> np.ndarray:
    """Predictive draws of log10 rim diameter for arrays of impactor draws.

    The vectorised counterpart of ``forward_predict``: draw i uses impactor i, one scaling
    package chosen with equal probability, and target, angle-law and nu draws from the same
    priors. Used to propagate an uncertain impactor (for example one inferred from an impact
    flash) to a crater-size distribution. The target class depends on the crater size, so each
    draw takes the self-consistent class (see the module note) with its target random numbers
    held fixed.
    """
    lm = np.asarray(log10_mass_kg, dtype=np.float64)
    n = lm.size
    u = np.broadcast_to(np.asarray(velocity_m_s, dtype=np.float64), (n,))
    el = np.broadcast_to(np.asarray(elev_deg, dtype=np.float64), (n,))
    de = np.broadcast_to(np.asarray(delta, dtype=np.float64), (n,))
    pk = rng.integers(0, len(packages), n)
    q = rng.uniform(*model.angle_q, n)
    nu = rng.uniform(*model.nu, n)
    tseed = int(rng.integers(0, 2**62))
    tu = _target_uniforms(n, site, np.random.default_rng(tseed))
    h = tu["h"]
    s = np.empty(n)
    for i, key in enumerate(packages):
        sel = pk == i
        if sel.any():
            tsel = {k: (v[sel] if isinstance(v, np.ndarray) else v) for k, v in tu.items()}
            s[sel], *_ = _consistent_rim(tsel, lm[sel], u[sel], de[sel], PACKAGES[key], nu[sel], el[sel], q[sel],
                                         gravity_m_s2)
    if include_mf_residual:
        lay = _layered_mask(s, h, site)   # band decided by the noise-free size, as in simulate_truth
        s = s + draw_sigma_mf(model, n, rng) * rng.standard_normal(n)
        if lay.any():
            s = s + np.where(lay, _layered_sd(model), 0.0) * rng.standard_normal(n)
    return s


def _layered_sd(model: ModelForm) -> float:
    """Extra layered-band scatter: the model's opt-in value, else this module's constant (read at call time)."""
    return LAYERED_EXTRA_SIGMA_DEX if model.layered_extra_sigma_dex is None else model.layered_extra_sigma_dex


def _layered_mask(s, h, site: SiteContext):
    """Draws in the layered-target band (4 < D/h < 20), which carry extra model-form scatter."""
    if site.fixed_class is not None:
        return np.zeros(np.shape(s), dtype=bool)
    return is_layered(10.0 ** s / h)


def forward_point(
    diameter_m: float,
    velocity_m_s: float,
    delta: float,
    rho_t: float,
    strength_pa: float,
    target_class: int,
    *,
    angle_from_vertical_deg: float = 45.0,
    package: str = "lpi2007",
    gravity_m_s2: float = 1.62,
    nu: float = 1.0 / 3.0,
    angle_q: float = 1.0,
    with_range: bool = True,
    seed: int = 0,
) -> dict:
    """Single-scenario forward calculation (the original Forward page's outputs).

    Uses one named scaling package with the user's target values. The crater
    depth is the model's excavation depth (K_d V^(1/3)), which corresponds to
    d/D = 0.21, deeper than fresh lunar craters (0.11-0.17). Ejecta mass, melt
    and formation times follow the LPI 2007 calculator relations. When
    ``with_range`` is set, the spread across scaling packages, the angle law
    (q 0.5-1) and nu (0.2-0.4) is added for the same target values.
    """
    pkg = PACKAGES[package]
    k1a, k2a, mua, kra, sda = pkg.arrays()
    c = int(target_class)
    k1, k2, mu, kr = k1a[c], k2a[c], mua[c], kra[c]
    a = diameter_m / 2.0
    vol_imp = 4.0 / 3.0 * math.pi * a ** 3
    m = delta * vol_imp
    elev = 90.0 - angle_from_vertical_deg
    # size-dependent strength (IMPACTS rocks) needs the crater size: fixed point
    d_guess = 10.0 * diameter_m
    for _ in range(6):
        y_eff = float(effective_strength(strength_pa, d_guess, bool(sda[c])))
        s, slope, pi2, st = log10_rim(math.log10(m), velocity_m_s, delta, rho_t, y_eff, k1, k2, mu, kr,
                                      g=gravity_m_s2, density_ratio=pkg.density_ratio, nu=nu,
                                      elev_deg=elev, angle_q=angle_q, rim_factor=pkg.rim_factor)
        d_guess = 10 ** float(s)
    D = 10 ** float(s)
    u_eff = velocity_m_s * math.sin(math.radians(elev)) ** angle_q
    pi3 = y_eff / (rho_t * u_eff ** 2)
    v13 = D / (pkg.rim_factor * 2.0 * kr)
    V = v13 ** 3
    pi_v = rho_t * V / m
    ratio = float(st / pi2)
    regime = "gravity" if ratio < 0.2 else ("strength" if ratio > 5.0 else "intermediate")
    yy = velocity_m_s ** 2 / 5.0e6 - 10.0
    melt_v = 0.5 * vol_imp * yy ** 0.9 if yy > 0 else 0.0
    out = {
        "package": package, "package_label": pkg.label,
        "rim_diameter_m": D,
        "excavation_diameter_m": 2.0 * kr * v13,
        "depth_m": 0.6 * v13,
        "rim_height_m": 0.036 * D,
        "transient_volume_m3": V,
        # pi2 = g a / U_eff^2 and pi3 = Y / (rho_t U_eff^2) in their standard (Holsapple) definitions; the
        # density-ratio factors of the scaling function are applied inside log10_rim and in strength_to_gravity_ratio
        "pi2": float(gravity_m_s2 * a / u_eff ** 2), "pi3": float(pi3), "pi_volume": float(pi_v),
        "regime_label": regime, "strength_to_gravity_ratio": ratio,
        "impactor_mass_kg": m, "kinetic_energy_j": 0.5 * m * velocity_m_s ** 2,
        "momentum_kg_m_s": m * velocity_m_s,
        "formation_time_s": 0.8 * V ** (1.0 / 6.0) / math.sqrt(gravity_m_s2),
        "formation_time_melosh_s": math.sqrt(D / gravity_m_s2),
        "ejecta_mass_kg": 0.8 * rho_t * V,
        "melt_volume_m3": melt_v,
        "effective_strength_pa": y_eff,
        "is_simple_crater": D <= 10_000.0,
        "notes": [],
    }
    if velocity_m_s < 3000:
        out["notes"].append("Speed below a few km/s: point-source scaling is applied outside its calibrated range.")
    if angle_from_vertical_deg > 75:
        out["notes"].append("Angle more than 75 degrees from vertical: grazing impacts are outside the point-source law.")
    if delta < 500:
        out["notes"].append("Very low impactor density: hollow or highly porous bodies are not compact point sources.")
    if not (4.0 <= pi_v <= 1.0e4):
        out["notes"].append("Cratering efficiency outside the range covered by the original calculator's data (4 to 10,000).")
    if with_range:
        rng = np.random.default_rng(seed)
        n = 6000
        all_s = []
        for key in DEFAULT_PACKAGE_SET:
            p2 = PACKAGES[key]
            k1b, k2b, mub, krb, sdb = p2.arrays()
            q = rng.uniform(0.5, 1.0, n)
            nus = rng.uniform(0.2, 0.4, n)
            # Size-dependent strength is evaluated at each draw's own crater, not at the single-point D
            # (fixed point; the map is a contraction with factor <= mu/8).
            s2 = np.full(n, math.log10(D))
            for _ in range(25 if sdb[c] else 1):
                ye = effective_strength(strength_pa, 10.0 ** s2, bool(sdb[c]))
                s_new, *_ = log10_rim(math.log10(m), velocity_m_s, delta, rho_t, ye, k1b[c], k2b[c], mub[c], krb[c],
                                      g=gravity_m_s2, density_ratio=p2.density_ratio, nu=nus, elev_deg=elev,
                                      angle_q=q, rim_factor=p2.rim_factor)
                done = np.max(np.abs(s_new - s2)) < 1e-10
                s2 = s_new
                if done:
                    break
            all_s.append(s2)
        s_all = np.concatenate(all_s)
        out["rim_diameter_range_m"] = [float(10 ** np.quantile(s_all, 0.05)), float(10 ** np.quantile(s_all, 0.95))]
        out["range_note"] = ("5 to 95% across the three scaling packages, the angle law and the density "
                             "exponent, for the entered target; the 0.1 dex scatter of real craters about "
                             "the law is not included.")
    return out
