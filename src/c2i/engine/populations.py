"""Declared impactor populations (priors over velocity, density, angle, mass).

Each population is an explicit, swappable assumption. None is "the truth";
the engine reports results per population and as an envelope across them.

Velocity distributions are given as truncated log-normals with documented
provenance. They are approximations: the per-population summary statistics
that motivate them are recorded in ``source``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr, ndtri

LUNAR_ESCAPE_M_S = 2380.0


@dataclass(frozen=True)
class Population:
    key: str
    label: str
    source: str
    # velocity: truncated log-normal (median, sigma_ln, lo, hi) or log-uniform if sigma_ln <= 0
    u_median: float
    u_sigma_ln: float
    u_lo: float
    u_hi: float
    # impactor bulk density: log-uniform [lo, hi] with optional iron fraction
    delta_lo: float
    delta_hi: float
    iron_fraction: float = 0.0
    # elevation angle (deg above horizontal): "sin2" (isotropic flux) or uniform [lo, hi]
    angle_law: str = "sin2"
    elev_lo: float = 0.0
    elev_hi: float = 90.0
    hypervelocity: bool = True     # False -> point-source scaling outside validity
    artificial: bool = False       # spacecraft or rocket stage (opt-in domain)
    # mass prior p(log10 m) proportional to 10^(-b log10 m). Natural primaries follow the
    # Brown et al. (2002) cumulative slope 0.90 (fit for 1-10 m bodies, extrapolated outside it); with a
    # flat prior the engine overestimates the masses of Brown-law synthetic impactors by +0.50 dex. Spacecraft
    # masses are designed and secondaries have no calibrated fragment law, so both stay flat (b = 0).
    mass_prior_b: float = 0.0

    # ---- velocity ---------------------------------------------------------
    def _z(self, u):
        return (np.log(u) - math.log(self.u_median)) / self.u_sigma_ln

    def sample_u(self, n, rng):
        if self.u_sigma_ln <= 0:
            return np.exp(rng.uniform(math.log(self.u_lo), math.log(self.u_hi), n))
        lo, hi = self._z(self.u_lo), self._z(self.u_hi)
        t = ndtr(lo) + rng.random(n) * (ndtr(hi) - ndtr(lo))
        return np.exp(math.log(self.u_median) + self.u_sigma_ln * ndtri(t))

    def logpdf_u(self, u):
        u = np.asarray(u, dtype=np.float64)
        inside = (u >= self.u_lo) & (u <= self.u_hi)
        us = np.where(inside, u, self.u_median)
        if self.u_sigma_ln <= 0:
            lp = -np.log(us) - math.log(math.log(self.u_hi / self.u_lo))
        else:
            z = self._z(us)
            lo, hi = self._z(self.u_lo), self._z(self.u_hi)
            lp = (-0.5 * z**2 - np.log(us) - math.log(self.u_sigma_ln)
                  - 0.5 * math.log(2 * math.pi) - math.log(ndtr(hi) - ndtr(lo)))
        return np.where(inside, lp, -np.inf)

    # ---- density ----------------------------------------------------------
    def logpdf_delta(self, delta):
        """Prior density of the impactor density (log-uniform stony part plus the uniform 7000-7900 kg/m3 iron part),
        per unit natural log of the density; -inf outside the support."""
        d = np.asarray(delta, dtype=np.float64)
        span = math.log(self.delta_hi / self.delta_lo)
        stony = np.where((d >= self.delta_lo) & (d <= self.delta_hi), (1.0 - self.iron_fraction) / span, 0.0)
        iron = np.where((d >= 7000.0) & (d <= 7900.0), self.iron_fraction * d / 900.0, 0.0) if self.iron_fraction > 0 else 0.0
        with np.errstate(divide="ignore"):
            return np.log(stony + iron)

    def sample_delta(self, n, rng):
        d = np.exp(rng.uniform(math.log(self.delta_lo), math.log(self.delta_hi), n))
        if self.iron_fraction > 0:
            iron = rng.random(n) < self.iron_fraction
            d = np.where(iron, rng.uniform(7000.0, 7900.0, n), d)
        return d

    # ---- angle ------------------------------------------------------------
    def logpdf_elev(self, elev_deg):
        """Normalised prior density of the elevation angle, per degree."""
        e = np.asarray(elev_deg, dtype=np.float64)
        if self.angle_law == "sin2":
            inside = (e > 0.0) & (e < 90.0)
            d = np.sin(np.radians(2.0 * np.where(inside, e, 45.0))) * math.pi / 180.0
            return np.where(inside, np.log(np.maximum(d, 1e-300)), -np.inf)
        inside = (e >= self.elev_lo) & (e <= self.elev_hi)
        return np.where(inside, -math.log(self.elev_hi - self.elev_lo), -np.inf)

    def sample_elev(self, n, rng):
        if self.angle_law == "sin2":
            # isotropic flux onto a plane: p(theta) = sin(2 theta), CDF = sin^2(theta)
            return np.degrees(np.arcsin(np.sqrt(rng.random(n))))
        return rng.uniform(self.elev_lo, self.elev_hi, n)


POPULATIONS: dict[str, Population] = {
    "asteroidal": Population(
        key="asteroidal",
        label="Asteroidal primary (NEO-derived)",
        source="Velocity: truncated log-normal, median 18 km/s, sigma_ln 0.35 "
               "(mean 19.1 km/s), bounded below by lunar escape speed; "
               "approximates the NEO-derived lunar impact-speed distribution "
               "(mean 19.7 km/s, Le Feuvre & Wieczorek 2011). Density: "
               "log-uniform 1.3-3.5 g/cc plus 4% iron (assumption). "
               "Angle: isotropic sin(2 theta). Mass: power law b = 0.90 "
               "(Brown et al. 2002 cumulative slope).",
        u_median=18_000.0, u_sigma_ln=0.35, u_lo=LUNAR_ESCAPE_M_S, u_hi=72_000.0,
        delta_lo=1300.0, delta_hi=3500.0, iron_fraction=0.04, mass_prior_b=0.90,
    ),
    "meteoroid": Population(
        key="meteoroid",
        label="Sporadic meteoroid primary (cm-m bodies)",
        source="Assumption for small bodies seen as lunar impact flashes: "
               "log-normal median 20 km/s, sigma_ln 0.40; density log-uniform "
               "0.3-3.0 g/cc (cometary to stony). Angle: isotropic. Mass: power "
               "law b = 0.90 (Brown et al. 2002 slope, extrapolated to cm-m bodies).",
        u_median=20_000.0, u_sigma_ln=0.40, u_lo=LUNAR_ESCAPE_M_S, u_hi=72_000.0,
        delta_lo=300.0, delta_hi=3000.0, mass_prior_b=0.90,
    ),
    "secondary": Population(
        key="secondary",
        label="Secondary (ejecta fragment of a larger crater)",
        source="Velocities below lunar escape speed (log-uniform 0.2-2.38 km/s); "
               "fragment density 2.5-3.3 g/cc; elevation 10-45 deg (assumption). "
               "Point-source hypervelocity scaling is NOT validated below ~1 km/s "
               "(LPI 2007, IMPACTS 2022), so results are flagged.",
        u_median=1000.0, u_sigma_ln=0.0, u_lo=200.0, u_hi=LUNAR_ESCAPE_M_S,
        delta_lo=2500.0, delta_hi=3300.0, angle_law="uniform", elev_lo=10.0, elev_hi=45.0,
        hypervelocity=False,
    ),
    # Opt-in domain for known artificial impacts. Never part of the default. The priors
    # come from orbital mechanics and vehicle facts, not from the crater sizes:
    #   speed: impact from lunar orbit ~1.6-1.7 km/s (circular speed at the surface ~1.68 km/s);
    #   arrival from cislunar or heliocentric space sqrt(2.38^2 + v_inf^2) km/s with v_inf up to
    #   ~1.5 km/s gives <= ~2.8 km/s. Recorded impacts: 1.6 (GRAIL) to 2.67 km/s (Ranger 9).
    #   density: BULK density of the vehicle, 23 kg/m3 for an empty S-IVB (Plescia et al., as
    #   recorded for event G09 in data/validation/ground_truth_events.json) and ~25-33 kg/m3 for the
    #   LCROSS Centaur and Falcon 9 stages (mass / cylinder volume) up to compact probes (LADEE
    #   430 kg/m3, LROC post 822).
    # Development check (n = 5; the events had been examined before, so this is not held out):
    # diameter-only mass error 0.28 dex against 1.03 dex with the asteroidal prior; 0.15 dex with
    # the known angle.
    "artificial": Population(
        key="artificial",
        label="Artificial impactor (spacecraft or rocket stage)",
        source="Velocity log-uniform 1.6-2.9 km/s (lunar-orbit decay to arrival from cislunar "
               "space); bulk density log-uniform 20-1000 kg/m3 (hollow stages to compact probes); "
               "elevation uniform 5-90 deg. Hollow, elongated bodies lie outside the compact "
               "point-source calibration (IMPACTS 2022), so results are flagged.",
        u_median=2200.0, u_sigma_ln=0.0, u_lo=1600.0, u_hi=2900.0,
        delta_lo=20.0, delta_hi=1000.0, angle_law="uniform", elev_lo=5.0, elev_hi=90.0,
        artificial=True,
    ),
}
