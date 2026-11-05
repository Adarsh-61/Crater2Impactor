"""Probability that a crater is a secondary, from measured morphometry (opt-in).

Logistic model on log10 D, depth/diameter and ellipticity, fitted on the 610 labelled craters of
Real_data/Full_Crater_Catalogue.xlsx (Kepler region, 263 primary / 347 secondary, D 280-1500 m).
Spatially blocked 5-fold validation: AUROC 0.944, accuracy 0.897, expected calibration error 0.045
The secondary labels are those of Singer et al. (2020, JGR Planets 125, e2019JE006313).
In a pre-registered test on the Copernicus secondary field the frozen model ranked craters worse than
chance (AUROC 0.39): secondaries there are deeper than the surrounding craters. Valid for Kepler-like settings only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

COEF_LOG10_D = -6.117177
COEF_DEPTH_RATIO = -41.2358
COEF_ELLIPTICITY = 22.908108
INTERCEPT = -2.715177

TRAIN_D_M = (279.6, 1499.6)
TRAIN_DEPTH_RATIO = (0.016, 0.2759)
TRAIN_ELLIPTICITY = (0.86, 1.94)


@dataclass(frozen=True)
class SecondaryEstimate:
    probability: float
    in_training_range: bool
    out_of_range: tuple[str, ...]


def secondary_probability(diameter_m: float, depth_m: float, ellipticity: float) -> SecondaryEstimate:
    if diameter_m <= 0 or depth_m <= 0 or ellipticity <= 0:
        raise ValueError("diameter, depth and ellipticity must be positive")
    ratio = depth_m / diameter_m
    z = INTERCEPT + COEF_LOG10_D * math.log10(diameter_m) + COEF_DEPTH_RATIO * ratio + COEF_ELLIPTICITY * ellipticity
    p = 1.0 / (1.0 + math.exp(-z)) if z > -700 else 0.0
    out = tuple(name for name, v, (lo, hi) in (
        ("diameter", diameter_m, TRAIN_D_M), ("depth_to_diameter", ratio, TRAIN_DEPTH_RATIO),
        ("ellipticity", ellipticity, TRAIN_ELLIPTICITY)) if not lo <= v <= hi)
    return SecondaryEstimate(probability=p, in_training_range=not out, out_of_range=out)


def population_weights(estimate: SecondaryEstimate, primary: str = "asteroidal", floor: float = 1e-3) -> dict[str, float]:
    p = min(max(estimate.probability, floor), 1.0 - floor)
    return {primary: 1.0 - p, "secondary": p}
