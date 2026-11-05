"""Measurement advisor: how much would each additional observation narrow this crater's posterior?

Answers the question "which extra measurement is worth making for this crater?" with the
engine's own model, before the measurement exists (Bayesian experimental design; the expected posterior
width is the goal-oriented criterion recommended for this problem, e.g. Rainforth et al. 2024, Stat. Sci.).

Method (nested Monte Carlo with reused draws). Given posterior draws theta_i (i = 1..N) for the
observations already in hand:

    for k = 1..K:  pick j at random, simulate a pseudo-observation y_k ~ p(y | theta_j)
                   reweight every draw: w_ik = p(y_k | theta_i)
                   record the weighted 90% width of each target quantity
    expected width = mean over k

The channel likelihoods are exactly the engine's (sampler.py, observations.py):
    speed   ln u_obs ~ N(ln u, ln(1 + rel))
    angle   e_obs ~ N(e, sd)                       (elevation above the horizontal)
    flash   log10 E ~ log10 eta + log10 KE + N(0, sd), log10 eta ~ U(log10 eta_lo, log10 eta_hi),
            integrated in closed form over eta
    ejecta  P(class | e) from observations.ejecta_pattern_prob

In-model statement: it says what the declared model expects a measurement to do, averaged over the
outcomes the model considers plausible. If the model is wrong (e.g. the ejecta thresholds, see the
provisional flag), the expected gain is wrong too. Not validated against real follow-up measurements.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import ndtr

from .observations import EJECTA_PATTERNS, ejecta_pattern_prob

TARGETS = {
    "log10_mass_kg": "Impactor mass (dex)",
    "log10_energy_j": "Kinetic energy (dex)",
    "log10_velocity_m_s": "Impact speed (dex)",
    "angle_from_vertical_deg": "Impact angle (deg)",
}
CHANNELS = ("speed", "angle", "flash", "ejecta")
DEFAULT_SPEC = {
    "speed": {"rel": 0.10},
    "angle": {"sd_deg": 5.0},
    "flash": {"sd_dex": 0.15, "eta": (5e-4, 6e-3)},
    "ejecta": {},
}
_PATTERNS = list(EJECTA_PATTERNS)


def _w90(x, w, order=None):
    """90% width of weighted draws, using the midpoint CDF (as infer._wq and the SRC backend) so that
    neither quantile is shifted by half a draw's weight; `order` lets callers sort x once."""
    o = np.argsort(x) if order is None else order
    ws = w[o]
    c = np.cumsum(ws) - 0.5 * ws
    c /= c[-1] + 0.5 * ws[-1]
    xs = x[o]
    return float(np.interp(0.95, c, xs) - np.interp(0.05, c, xs))


def _simulate(ch, j, S, spec, rng):
    if ch == "speed":
        return S["ln_u"][j] + math.log(1 + spec["rel"]) * rng.standard_normal(len(j))
    if ch == "angle":
        return S["elev"][j] + spec["sd_deg"] * rng.standard_normal(len(j))
    if ch == "flash":
        a, b = (math.log10(e) for e in spec["eta"])
        return rng.uniform(a, b, len(j)) + S["le"][j] + spec["sd_dex"] * rng.standard_normal(len(j))
    if ch == "ejecta":
        P = np.stack([ejecta_pattern_prob(p, S["elev"][j]) for p in _PATTERNS], 1)
        P /= P.sum(1, keepdims=True)
        return np.minimum((rng.random(len(j))[:, None] > np.cumsum(P, 1)).sum(1), len(_PATTERNS) - 1)
    raise ValueError(f"unknown channel {ch}; options {CHANNELS}")


def _loglik(ch, y, S, spec):
    """log p(y_k | theta_i) for all k (rows) and i (columns)."""
    y = np.asarray(y)[:, None]
    if ch == "speed":
        s = math.log(1 + spec["rel"])
        return -0.5 * ((y - S["ln_u"][None, :]) / s) ** 2
    if ch == "angle":
        return -0.5 * ((y - S["elev"][None, :]) / spec["sd_deg"]) ** 2
    if ch == "flash":
        a, b = (math.log10(e) for e in spec["eta"])
        z = y - S["le"][None, :]
        p = ndtr((z - a) / spec["sd_dex"]) - ndtr((z - b) / spec["sd_dex"])
        return np.log(np.maximum(p, 1e-300))
    if ch == "ejecta":
        P = np.stack([ejecta_pattern_prob(p, S["elev"]) for p in _PATTERNS], 0)   # (4, N)
        return np.log(P[y[:, 0].astype(int)])
    raise ValueError(ch)


def measurement_value(samples: dict, channels_sets=None, *, spec=None, k_outer: int = 200,
                      seed: int = 0) -> dict:
    """Expected 90% widths of the target quantities if the given channels were observed.

    samples: posterior draws as returned by infer_crater(..., want_samples=True) (equal weights).
             Use max_samples_out >= 4000 for stable results.
    channels_sets: list of tuples of channels, e.g. [("speed",), ("flash",), ("speed", "flash")].
    """
    spec = {**DEFAULT_SPEC, **(spec or {})}
    channels_sets = channels_sets or [(c,) for c in CHANNELS] + [("speed", "angle"), ("flash", "ejecta"),
                                                                 ("speed", "flash", "ejecta")]
    S = {
        "ln_u": np.log(10.0) * np.asarray(samples["log10_velocity_m_s"], float),
        "elev": 90.0 - np.asarray(samples["angle_from_vertical_deg"], float),
        "le": np.asarray(samples["log10_energy_j"], float),
    }
    T = {k: np.asarray(samples[k], float) for k in TARGETS}
    order = {k: np.argsort(v) for k, v in T.items()}
    n = len(S["le"])
    ones = np.ones(n)
    now = {k: _w90(v, ones) for k, v in T.items()}
    rng = np.random.default_rng(seed)
    out = {"current_width90": now, "options": [], "k_outer": k_outer, "n_draws": n,
           "note": "Expected under the engine's declared model; not validated against real follow-up data."}
    for cs in channels_sets:
        j = rng.integers(0, n, k_outer)
        ll = np.zeros((k_outer, n))
        for ch in cs:
            ll += _loglik(ch, _simulate(ch, j, S, spec[ch], rng), S, spec[ch])
        ll -= ll.max(1, keepdims=True)
        W = np.exp(ll)
        ess = W.sum(1) ** 2 / (W ** 2).sum(1)
        widths = {k: float(np.mean([_w90(v, W[i], order[k]) for i in range(k_outer)])) for k, v in T.items()}
        out["options"].append({
            "channels": list(cs),
            "expected_width90": widths,
            "expected_width_ratio": {k: widths[k] / now[k] if now[k] > 0 else 1.0 for k in T},
            "median_ess": float(np.median(ess)),
        })
    return out


__all__ = ["measurement_value", "CHANNELS", "TARGETS", "DEFAULT_SPEC"]
