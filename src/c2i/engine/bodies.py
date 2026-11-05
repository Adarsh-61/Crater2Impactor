"""Planetary bodies: what is shared, what is body-specific, and what is supported.

Only bodies whose impactor population, target model and atmosphere handling
have been specified and validated are marked ``inference_supported``. Adding
a body is a scientific task, not a configuration change; the requirements
are listed in ``requirements`` and in docs/EXTENDING_BODIES.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field

REQUIREMENTS = (
    "surface gravity",
    "impactor velocity distribution at the body (population model)",
    "target-property model (materials, layering, porosity, volatiles) with sources",
    "atmospheric entry, deceleration and fragmentation model if the body has an atmosphere",
    "simple-to-complex transition diameter for the body",
    "an independent validation set (controlled impacts, new-crater rates, or event pairs)",
)


@dataclass(frozen=True)
class Body:
    key: str
    label: str
    gravity_m_s2: float
    atmosphere: bool
    inference_supported: bool
    status: str
    simple_crater_max_m: float
    missing: tuple[str, ...] = field(default_factory=tuple)

    def describe(self) -> dict:
        return {
            "key": self.key, "label": self.label, "gravity_m_s2": self.gravity_m_s2,
            "atmosphere": self.atmosphere, "inference_supported": self.inference_supported,
            "status": self.status, "simple_crater_max_m": self.simple_crater_max_m,
            "missing_for_support": list(self.missing),
            "requirements_for_any_body": list(REQUIREMENTS),
        }


BODIES: dict[str, Body] = {
    "moon": Body(
        "moon", "Moon", 1.62, False, True,
        "Supported for simple craters (about 1 m to 10 km) under the declared "
        "populations, scaling packages and target model. Forward scaling checked "
        "against controlled spacecraft impacts; see the validation report.",
        10_000.0,
    ),
    "mars": Body(
        "mars", "Mars", 3.71, True, False,
        "Not supported: small Martian impactors are decelerated and fragmented by "
        "the atmosphere (58% of new Martian impacts form clusters; Daubar et al. "
        "2022), so a crater-only inversion needs an entry model (e.g. Collins et "
        "al. 2022) and a Mars-specific velocity distribution, neither implemented.",
        7_000.0,
        ("atmospheric entry/fragmentation model", "Mars impact-velocity distribution",
         "Mars target model (ice, layered deposits)", "validation against InSight crater-seismic pairs"),
    ),
    "mercury": Body(
        "mercury", "Mercury", 3.70, False, False,
        "Not supported: no Mercury impactor-velocity distribution or target model "
        "is implemented; lunar parameters must not be reused.",
        10_000.0,
        ("Mercury impact-velocity distribution", "Mercury target model", "validation data"),
    ),
}
