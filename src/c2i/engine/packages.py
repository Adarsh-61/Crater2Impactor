"""Named, source-traceable crater-scaling packages and the point-source kernel.

Every constant here is transcribed from a primary source:

* ``lpi2007``    - LPI "Lunar Impact Craters" calculator v1.4 (10/07) JavaScript
                   (live page and Old_files/Original_Lunar_Forward_Problem_index.html):
                   no density-ratio term, vertical velocity U*cos(theta_v).
* ``holsapple2003`` - Holsapple, "Theory and equations for Craters from Impacts
                   and Explosions" (the LPI calculator's theory.pdf), Eq. 7 and
                   Table 1: density-ratio term with nu = 1/3, soft-rock K1 = 0.095.
* ``impacts2022`` - Holsapple (2022) IMPACTS!, arXiv:2203.07476, Eq. 3 and
                   Table 1: K2 = 1, size-dependent rock strength
                   Y = Y0 (10 cm / D)^(1/4), nu = 1/3.

The kernel is Holsapple's Eq. 7 (identical in all three sources up to the
density-ratio switch):

    pi_V = K1 { pi2 r^((6nu-2-mu)/(3mu)) + [K2 pi3 r^((6nu-2)/(3mu))]^((2+mu)/2) }^(-3mu/(2+mu))

with r = rho_t/delta, pi2 = g a / U_eff^2, pi3 = Y / (rho_t U_eff^2),
pi_V = rho_t V / m, rim diameter D = 1.3 * 2 Kr V^(1/3).

K2 placement: Holsapple's Eq. 7 raises the product K2 pi3 (...) to (2+mu)/2. The LPI 2007
JavaScript instead multiplies K2 outside the power (``piv=k1*Math.pow(pi2+k2*Math.pow(pis,
power2),power1)``). ``lpi2007`` keeps the JavaScript form; ``holsapple2003`` and ``impacts2022``
follow Eq. 7 (for IMPACTS K2 = 1, so the placement does not matter). ``arrays()`` returns the
coefficient that multiplies pi3^((2+mu)/2), i.e. K2^((2+mu)/2) when K2 is inside the power.

Angle: U_eff = U * sin(theta_elev)^q.  q = 1 is the vertical-velocity rule
used by LPI 2007 and IMPACTS 2022; hydrocode results (Davison & Collins 2022,
GRL; Elbeshausen et al. 2009) imply weaker angle dependence. The engine treats
q as an uncertain model-form parameter (default range 0.5-1.0).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Target classes used by every package (index order is part of the API).
TARGET_CLASSES: tuple[str, ...] = ("regolith", "fractured_rock", "competent_rock")


@dataclass(frozen=True)
class ClassConstants:
    k1: float
    k2: float
    mu: float
    kr: float = 1.1           # excavation radius factor R = Kr V^(1/3)
    kd: float = 0.6           # excavation depth factor
    size_dependent_strength: bool = False


@dataclass(frozen=True)
class ScalingPackage:
    key: str
    label: str
    source: str
    density_ratio: bool
    classes: tuple[ClassConstants, ClassConstants, ClassConstants]
    nu: float = 1.0 / 3.0
    rim_factor: float = 1.3
    k2_in_power: bool = False   # True: Holsapple Eq. 7 form [K2 pi3 ...]^((2+mu)/2)

    def arrays(self):
        """(k1, k2_eff, mu, kr, size_dependent) where k2_eff multiplies pi3^((2+mu)/2)."""
        k1 = np.array([c.k1 for c in self.classes])
        mu = np.array([c.mu for c in self.classes])
        k2 = np.array([c.k2 for c in self.classes])
        if self.k2_in_power:
            k2 = np.power(k2, (2.0 + mu) / 2.0)
        kr = np.array([c.kr for c in self.classes])
        sd = np.array([c.size_dependent_strength for c in self.classes])
        return k1, k2, mu, kr, sd


PACKAGES: dict[str, ScalingPackage] = {
    "lpi2007": ScalingPackage(
        key="lpi2007",
        label="LPI calculator (2007)",
        source="LPI Lunar Impact Craters calculator v1.4 (10/07) JavaScript; reproduces 405 "
               "live-page cases to 9e-16 when LPI's truncated cube root V^0.3333 is emulated, and to "
               "6e-4 dex with the exact cube root used here (INTERNAL_ONLY/tests/test_physics_limits.py)",
        density_ratio=False,
        classes=(
            ClassConstants(0.132, 0.26, 0.41),    # lunar regolith
            ClassConstants(0.07, 0.215, 0.55),    # hard soil / soft rock
            ClassConstants(0.095, 0.257, 0.55),   # hard rock
        ),
    ),
    "holsapple2003": ScalingPackage(
        key="holsapple2003",
        label="Holsapple theory (2003)",
        source="K. A. Holsapple, Theory and equations for 'Craters from Impacts "
               "and Explosions' (LPI calculator theory.pdf), Eq. 7, Table 1",
        density_ratio=True,
        classes=(
            ClassConstants(0.132, 0.26, 0.41),
            ClassConstants(0.095, 0.215, 0.55),
            ClassConstants(0.095, 0.257, 0.55),
        ),
        k2_in_power=True,
    ),
    "impacts2022": ScalingPackage(
        key="impacts2022",
        label="Holsapple IMPACTS (2022)",
        source="K. A. Holsapple (2022) IMPACTS!, arXiv:2203.07476, Table 1 "
               "(lunar regolith; soft dry rock/hard soils; hard rocks)",
        density_ratio=True,
        classes=(
            ClassConstants(0.15, 1.0, 0.40),
            ClassConstants(0.04, 1.0, 0.55, size_dependent_strength=True),
            ClassConstants(0.06, 1.0, 0.55, size_dependent_strength=True),
        ),
        k2_in_power=True,
    ),
}

DEFAULT_PACKAGE_SET: tuple[str, ...] = ("lpi2007", "holsapple2003", "impacts2022")


def effective_strength(y_pa, d_rim_m, size_dependent):
    """IMPACTS (2022) size-dependent strength Y = Y0 (10 cm / D)^(1/4) for rocks.

    The diameter passed in is the noise-free crater size of each draw. The
    inverse sampler fixes that size on the fibre, so no iteration is needed
    there, and it includes dlog Y_eff / ds in the Jacobian; the forward model
    solves the implied fixed point by a short contraction.
    """
    y = np.asarray(y_pa, dtype=np.float64)
    fac = np.power(0.1 / np.maximum(np.asarray(d_rim_m, dtype=np.float64), 0.1), 0.25)
    return np.where(size_dependent, y * fac, y)


def log10_rim(
    log10_m,
    u_m_s,
    delta,
    rho_t,
    y_eff,
    k1,
    k2,
    mu,
    kr,
    *,
    g: float,
    density_ratio: bool,
    nu,
    elev_deg,
    angle_q,
    rim_factor: float = 1.3,
):
    """Return (s, ds/dlog10 m, pi2_eff, strength_term) for simple craters.

    s = log10 rim diameter (m). Arrays broadcast.
    """
    m = np.power(10.0, log10_m)
    a = np.cbrt(3.0 * m / (4.0 * math.pi * delta))
    sin_e = np.maximum(np.sin(np.radians(elev_deg)), 1e-6)
    u_eff = u_m_s * np.power(sin_e, angle_q)
    pi2 = g * a / u_eff**2
    pi3 = y_eff / (rho_t * u_eff**2)
    if density_ratio:
        r = rho_t / delta
        pi2 = pi2 * np.power(r, (6.0 * nu - 2.0 - mu) / (3.0 * mu))
        pi3 = pi3 * np.power(r, (6.0 * nu - 2.0) / (3.0 * mu))
    st = k2 * np.power(pi3, (2.0 + mu) / 2.0)
    w = pi2 + st
    v = k1 * np.power(w, -3.0 * mu / (2.0 + mu)) * m / rho_t
    s = np.log10(rim_factor * 2.0 * kr * np.cbrt(v))
    slope = (1.0 - (mu / (2.0 + mu)) * (pi2 / w)) / 3.0
    return s, slope, pi2, st


def _initial_log10_m(s_target, *, u_m_s, delta, rho_t, y_eff, k1, k2, mu, kr, g, density_ratio, nu, elev_deg,
                     angle_q, rim_factor=1.3):
    """Starting value of log10 m on the fibre s(log10 m) = s_target (see solve_log10_m)."""
    d_rim = np.power(10.0, s_target)
    sin_e = np.maximum(np.sin(np.radians(elev_deg)), 1e-6)
    u_eff2 = np.square(u_m_s * np.power(sin_e, angle_q))
    c_g = (g / u_eff2) * np.cbrt(3.0 / (4.0 * math.pi * delta))      # pi2 = c_g x with x = m^(1/3)
    pi3 = y_eff / (rho_t * u_eff2)
    if density_ratio:
        r = rho_t / delta
        c_g = c_g * np.power(r, (6.0 * nu - 2.0 - mu) / (3.0 * mu))
        pi3 = pi3 * np.power(r, (6.0 * nu - 2.0) / (3.0 * mu))
    st = k2 * np.power(pi3, (2.0 + mu) / 2.0)
    p = mu / (2.0 + mu)
    r_v = (d_rim / (2.0 * rim_factor * kr)) * np.cbrt(rho_t / k1)    # fibre: x = r_v (c_g x + st)^p
    x_str = r_v * np.power(np.maximum(st, 1e-300), p)
    x_grav = np.power(r_v * np.power(c_g, p), 1.0 / (1.0 - p))
    x = np.power(np.power(x_str, 1.0 / p) + np.power(x_grav, 1.0 / p), p)
    for _ in range(2):
        w = c_g * x + st
        wp = np.power(w, p)
        x = x - (x - r_v * wp) / (1.0 - r_v * p * c_g * wp / w)
    lm = 3.0 * np.log10(np.maximum(x, 1e-300))
    return np.where(np.isfinite(lm), lm, 3.0 * np.asarray(s_target) + 3.0)


def solve_log10_m(s_target, max_iter: int = 60, tol: float = 1e-11, **kw):
    """Newton solve of s(log10 m) = s_target on the fiber; returns (log10 m, ds/dlog m, pi2, st).

    s is strictly increasing in log10 m with slope in [2/(3(2+mu)), 1/3],
    so the root is unique and the iteration converges from any start.
    """
    st_ = np.asarray(s_target, dtype=np.float64)
    # Start from the closed-form strength/gravity asymptotes refined by two Newton steps in x = m^(1/3); the loop
    # below then converges in about 2 iterations. The stopping rule alone fixes the root, so the start only
    # affects speed.
    lm = np.broadcast_to(_initial_log10_m(st_, **kw), np.broadcast(st_, kw["u_m_s"]).shape).copy()
    for _ in range(max_iter):
        s, slope, _, _ = log10_rim(lm, **kw)
        step = (s - st_) / slope
        step = np.clip(step, -3.0, 3.0)
        lm -= step
        if np.max(np.abs(step)) < tol:
            break
    s, slope, pi2, st = log10_rim(lm, **kw)
    return lm, slope, pi2, st
