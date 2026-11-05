"""Observation models that constrain the impact angle from crater planform and ejecta.

A crater diameter carries no information on the impact angle beyond what the angle does to
crater size, which is degenerate with speed. Two morphological observations of fresh
craters do depend on the angle and are routinely mapped from orbital images:

* the symmetry of the ejecta blanket (or ray pattern), and
* the elongation of the rim.

Lunar thresholds (elevation above the horizontal below which each feature appears):

====================================  ==========  ====================================================
Feature                               Threshold   Source
====================================  ==========  ====================================================
Asymmetric ejecta (downrange-heavy)   ~45 deg     Herrick & Forsberg-Taylor (2003, MAPS 38, 1551),
                                                  lunar crater survey; Gault & Wedekind (1978) lab
Uprange forbidden zone (V-shaped gap) ~25 deg     Herrick & Forsberg-Taylor (2003), Moon; ~20 deg in
                                                  later work by the same group; ~15 deg in the
                                                  Gault & Wedekind (1978) sand experiments
Butterfly ejecta (gaps uprange and   ~10 deg     Herrick & Forsberg-Taylor (2003): ~10 deg, with an
downrange)                                        elongated rim; Bottke et al. (2000, Icarus 145, 108):
                                                  elongation (length/width >= 1.1) below ~12 deg
====================================  ==========  ====================================================

The observation is an ordered category. Each threshold is treated as uncertain, with a
Gaussian spread chosen from the disagreement between the sources above (not tuned on any
event), so P(feature present | elevation e) = Phi((mu_t - e) / sigma_t). The category
probabilities follow from the nested features. The thresholds rise for very large
impactors (large pi2; Luo et al. 2022, LPSC 1133), which lie outside the simple-crater
domain this engine supports. The model applies to fresh craters whose ejecta or rays are
still visible; degraded craters should be entered as "not assessed".

Rim elongation alone is not used. In the Robbins (2018) global catalogue, 29% of well-traced
1-10 km craters are confidently elongated (length/width - 2 SD >= 1.1), against about 4%
expected from impact angle under isotropic impacts. Degradation, overlap and rim irregularity, not the angle, explain
most elongated rims, so an elongated rim is weak evidence of a low angle.

Classification error. Reading an ejecta pattern from images is subjective; the model assumes a
5% probability that the reported class is a random misreading (an assumption, not a
calibrated value).

Compact impactors only. The thresholds describe compact natural bodies. Elongated, hollow
spacecraft make elongated craters at steep angles: all Apollo S-IVB craters measure about
29-32 x 33-39 m although the stages struck at 69-79 deg elevation
(data/validation/ground_truth_events.json, event G09). The planform part of this model must not be applied to such impactors.

Falcon 9 stage crater (elevation ~31 deg, LROC post 1499, uprange forbidden zone): the thresholds
were not fitted to it, but the misclassification term was added after it was examined, so it is
development data, not a held-out test (docs/SCIENCE.md, sec. 8a).
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr

# (mu_deg, sigma_deg) for the elevation thresholds, from the sources in the module docstring
THRESHOLDS = {
    "asymmetry": (45.0, 7.5),
    "uprange_gap": (25.0, 5.0),
    "elongation": (11.0, 2.5),
}

EJECTA_PATTERNS: dict[str, str] = {
    "symmetric": "Symmetric ejecta",
    "asymmetric": "Asymmetric ejecta without an uprange gap",
    "uprange_gap": "Uprange forbidden zone (V-shaped gap in the ejecta)",
    "butterfly": "Butterfly ejecta (gaps uprange and downrange)",
}
MISCLASSIFICATION = 0.05


def _present(feature: str, elev_deg):
    mu, sig = THRESHOLDS[feature]
    return ndtr((mu - np.asarray(elev_deg, dtype=np.float64)) / sig)


def ejecta_pattern_prob(pattern: str, elev_deg) -> np.ndarray:
    """P(observed pattern | elevation angle above horizontal, degrees)."""
    fa, fz, fe = _present("asymmetry", elev_deg), _present("uprange_gap", elev_deg), _present("elongation", elev_deg)
    # Nested features: elongation implies a gap, which implies asymmetry; enforce monotonicity.
    fz = np.minimum(fz, fa)
    fe = np.minimum(fe, fz)
    p = {
        "symmetric": 1.0 - fa,
        "asymmetric": fa - fz,
        "uprange_gap": fz - fe,
        "butterfly": fe,
    }
    if pattern not in p:
        raise ValueError(f"unknown ejecta pattern {pattern}; options {list(EJECTA_PATTERNS)}")
    eps = MISCLASSIFICATION
    return (1.0 - eps) * np.maximum(p[pattern], 0.0) + eps / len(p)


def ejecta_pattern_loglik(pattern: str | None, elev_deg) -> np.ndarray | float:
    if pattern is None:
        return 0.0
    return np.log(ejecta_pattern_prob(pattern, elev_deg))
