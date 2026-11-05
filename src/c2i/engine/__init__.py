"""Crater2Impactor inference engine.

Explicit-prior Bayesian inference on the crater-scaling fiber, with named
scaling packages, declared impactor populations (averaged by their evidence),
a size- and site-aware target model, observation models for independent
measurements and ejecta morphology, robust envelopes and identifiability
labels. No neural network is used: the posterior is computed by importance
sampling whose only approximation is the finite sample (see docs/SCIENCE.md).
"""

from .infer import QUANTITIES, infer_crater
from .packages import DEFAULT_PACKAGE_SET, PACKAGES, TARGET_CLASSES, log10_rim, solve_log10_m
from .populations import POPULATIONS
from .sampler import EventData, ModelForm, run_scenario
from .targets import GLOBAL_CONTEXT, SiteContext

__all__ = [
    "infer_crater", "QUANTITIES", "PACKAGES", "DEFAULT_PACKAGE_SET", "TARGET_CLASSES",
    "log10_rim", "solve_log10_m", "POPULATIONS", "EventData", "ModelForm",
    "run_scenario", "GLOBAL_CONTEXT", "SiteContext",
]
