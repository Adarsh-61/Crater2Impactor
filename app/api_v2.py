"""Version-2 scientific API: explicit-prior Bayesian inference (importance sampling and deterministic
quadrature; no neural network).

Endpoints
---------
POST /api/v2/infer          crater (+ optional location, assumptions, event data) -> posterior report
POST /api/v2/forward        impactor + target assumptions -> predictive crater diameter
POST /api/v2/forward_point  single-scenario forward calculation with one package and fixed target values
GET  /api/v2/site           latitude/longitude -> target-property prior with provenance
GET  /api/v2/options        populations, scaling packages, bodies, defaults (for the UI)
GET  /api/v2/benchmarks     reproducible benchmark results (results/benchmarks_v2.json)
GET  /api/v2/comparison     original calculator vs version 2 (results/comparison_original_vs_v2.json)
GET  /api/v2/findings       research findings with their result files (results/findings.json)

Every number returned is produced by c2i.engine at request time, except the
benchmark file, which is written by scripts/run_benchmarks_v2.py and carries
its own provenance block.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from c2i.engine import (
    DEFAULT_PACKAGE_SET,
    PACKAGES,
    POPULATIONS,
    TARGET_CLASSES,
    EventData,
    ModelForm,
    infer_crater,
)
from c2i.engine.bodies import BODIES
from c2i.engine.forward import forward_point, forward_predict
from c2i.engine.lunar_context import site_context
from c2i.engine.flash import DEFAULT_FLASH_DEFINITION, FLASH_DEFINITIONS
from c2i.engine.calibration import SIGMA_MF_ARTIFICIAL
from c2i.engine.observations import EJECTA_PATTERNS
from c2i.engine.secondaries import population_weights, secondary_probability

REPO = Path(__file__).resolve().parents[1]
router = APIRouter(prefix="/api/v2", tags=["v2"])


class EventIn(BaseModel):
    velocity_km_s: float | None = Field(None, gt=0, le=80)
    velocity_sigma_rel: float = Field(0.1, gt=0, le=2)
    mass_kg: float | None = Field(None, gt=0)
    mass_sigma_dex: float = Field(0.1, gt=0, le=3)
    elevation_deg: float | None = Field(None, gt=0, le=90)
    angle_from_vertical_deg: float | None = Field(None, ge=0, lt=90)
    elevation_sigma_deg: float = Field(5.0, gt=0, le=45)
    # gt=0, not a higher floor: hollow spacecraft stages have bulk densities of 23-45 kg/m3, inside the
    # artificial population's prior (20-1000 kg/m3)
    impactor_density_kg_m3: float | None = Field(None, gt=0, le=25000)
    flash_energy_j: float | None = Field(None, gt=0)
    flash_sigma_dex: float = Field(0.15, gt=0, le=2)
    flash_definition: str = Field(DEFAULT_FLASH_DEFINITION,
                                  description="photometric definition of flash_energy_j: " + " | ".join(FLASH_DEFINITIONS))


class InferIn(BaseModel):
    rim_diameter_m: float = Field(..., gt=0, le=1e5)
    sigma_meas_dex: float = Field(0.03, gt=0.0, le=1.0)
    latitude_deg: float | None = Field(None, ge=-90, le=90)
    longitude_deg: float | None = Field(None, ge=-360, le=360)
    target_class: str | None = Field(None, description="override: regolith | fractured_rock | competent_rock")
    populations: dict[str, float] = Field(default_factory=lambda: {"asteroidal": 1.0})
    packages: list[str] = Field(default_factory=lambda: list(DEFAULT_PACKAGE_SET))
    sigma_mf_dex: float = Field(0.10, ge=0.0, le=1.0)
    angle_q_min: float = Field(0.5, ge=0.0, le=2.0)
    angle_q_max: float = Field(1.0, ge=0.0, le=2.0)
    mass_prior_b: float | None = Field(
        None, ge=0.0, le=2.0,
        description="null = each population's own mass prior (0.90 for natural primaries, 0 for spacecraft and secondaries)")
    body: str = "moon"
    event: EventIn | None = None
    seed: int = 0
    n_draws: int = Field(30_000, ge=2000, le=200_000)
    geological_setting: str | None = Field(
        None, description="unconstrained | auto | mare_basalts | highland_megaregolith | polar_regolith")
    depth_m: float | None = Field(None, gt=0)
    rim_height_m: float | None = Field(None, gt=0)
    central_peak_height_m: float | None = Field(None, ge=0)
    floor_diameter_m: float | None = Field(None, ge=0)
    ejecta_radius_m: float | None = Field(None, ge=0)
    melt_volume_m3: float | None = Field(None, ge=0)
    ejecta_pattern: str | None = Field(
        None, description="fresh-crater ejecta/planform class: " + " | ".join(EJECTA_PATTERNS))
    rocky_ejecta: bool | None = Field(
        None, description="blocky ejecta seen at the rim (true/false); constrains regolith thickness")
    ellipticity: float | None = Field(None, ge=1.0, le=5.0)
    secondary_from_morphometry: bool = Field(
        False, description="set the secondary-crater weight from diameter, depth and ellipticity")
    spacecraft_calibration: bool = Field(
        False, description="artificial impactors only: scatter calibrated on 11 lunar spacecraft craters, "
                           "no layered-ground term")


SETTINGS = ("unconstrained", "auto", "mare_basalts", "highland_megaregolith", "polar_regolith")


def _setting_site(setting, lat, lon):
    """Map the target-setting choice to a site context (explicit choice overrides coordinates)."""
    from dataclasses import replace
    from c2i.engine import GLOBAL_CONTEXT
    s = (setting or "auto").strip().lower()
    if s not in SETTINGS:
        raise HTTPException(422, f"geological_setting must be one of {SETTINGS}")
    if s == "unconstrained":
        return replace(GLOBAL_CONTEXT, label="Global prior (no location used)")
    if s == "auto":
        return site_context(lat, lon)
    base = site_context(lat, lon) if (lat is not None and lon is not None) else GLOBAL_CONTEXT
    if s == "mare_basalts":
        return replace(base, p_mare=1.0, label="Mare basalt substrate (selected)")
    if s == "highland_megaregolith":
        return replace(base, p_mare=0.0, label="Highland crust substrate (selected)")
    return replace(base, p_mare=0.0, label="Polar region (selected): treated as highland crust",
                   notes=tuple(base.notes) + ("No polar-specific target model is available; the "
                                               "highland substrate prior is used.",))


def _check_packages(packages):
    for k in packages:
        if k not in PACKAGES:
            raise HTTPException(422, f"unknown package {k}; options {list(PACKAGES)}")
    if not packages:
        raise HTTPException(422, "at least one scaling package is required")
    if len(set(packages)) != len(packages):
        raise HTTPException(422, "scaling packages must not repeat")


def _partial_location_flag(lat, lon):
    """Location is used only when both coordinates are given; say so instead of dropping one silently."""
    if (lat is None) != (lon is None):
        return ("Only one of latitude and longitude was given; the location was not used "
                "and the global target prior applies.")
    return None


def _site(lat, lon, target_class, setting=None):
    ctx = _setting_site(setting, lat, lon)
    if target_class:
        if target_class not in TARGET_CLASSES:
            raise HTTPException(422, f"target_class must be one of {TARGET_CLASSES}")
        from dataclasses import replace
        ctx = replace(ctx, fixed_class=TARGET_CLASSES.index(target_class),
                      label=ctx.label + f" (target class forced: {target_class})")
    return ctx


@router.post("/infer")
def infer(req: InferIn) -> dict:
    body = BODIES.get(req.body)
    if body is None:
        raise HTTPException(422, f"unknown body {req.body}; options {list(BODIES)}")
    if not body.inference_supported:
        raise HTTPException(422, f"{body.label}: {body.status}")
    if req.angle_q_min > req.angle_q_max:
        raise HTTPException(422, "angle_q_min must not exceed angle_q_max")
    for k in req.populations:
        if k not in POPULATIONS:
            raise HTTPException(422, f"unknown population {k}; options {list(POPULATIONS)}")
    _check_packages(req.packages)
    if any(not (math.isfinite(v) and v >= 0) for v in req.populations.values()):
        raise HTTPException(422, "population weights must be finite and non-negative")
    if sum(req.populations.values()) <= 0:
        raise HTTPException(422, "population weights must sum to a positive number")
    if req.ejecta_pattern is not None and req.ejecta_pattern not in EJECTA_PATTERNS:
        raise HTTPException(422, f"ejecta_pattern must be one of {list(EJECTA_PATTERNS)}")
    ev = None
    if req.event is not None:
        e = req.event
        if e.flash_definition not in FLASH_DEFINITIONS:
            raise HTTPException(422, f"flash_definition must be one of {list(FLASH_DEFINITIONS)}")
        elev = e.elevation_deg
        if e.angle_from_vertical_deg is not None:
            if elev is not None and abs(elev - (90.0 - e.angle_from_vertical_deg)) > 1e-6:
                raise HTTPException(422, "give the impact angle either as elevation or as angle from "
                                         "vertical, not two different values")
            elev = 90.0 - e.angle_from_vertical_deg
        ev = EventData(
            velocity_m_s=None if e.velocity_km_s is None else e.velocity_km_s * 1000.0,
            velocity_sigma_rel=e.velocity_sigma_rel,
            mass_kg=e.mass_kg, mass_sigma_dex=e.mass_sigma_dex,
            elev_deg=elev, elev_sigma_deg=e.elevation_sigma_deg,
            delta_kg_m3=e.impactor_density_kg_m3,
            flash_energy_j=e.flash_energy_j, flash_sigma_dex=e.flash_sigma_dex,
            flash_definition=e.flash_definition,
            ejecta_pattern=req.ejecta_pattern,
            rocky_ejecta=req.rocky_ejecta,
        )
    elif req.ejecta_pattern is not None or req.rocky_ejecta is not None:
        ev = EventData(ejecta_pattern=req.ejecta_pattern, rocky_ejecta=req.rocky_ejecta)
    populations = dict(req.populations)
    secondary = None
    if req.secondary_from_morphometry:
        if req.depth_m is None or req.ellipticity is None:
            raise HTTPException(422, "secondary_from_morphometry needs depth_m and ellipticity")
        primary = max((k for k in populations if k != "secondary"), key=populations.get, default="asteroidal")
        secondary = secondary_probability(req.rim_diameter_m, req.depth_m, req.ellipticity)
        populations = population_weights(secondary, primary=primary)
    if req.spacecraft_calibration:
        if set(populations) != {"artificial"}:
            raise HTTPException(422, "spacecraft_calibration applies only to populations = {artificial: 1}")
        model = ModelForm(sigma_mf_draws=SIGMA_MF_ARTIFICIAL, layered_extra_sigma_dex=0.0,
                          angle_q=(req.angle_q_min, req.angle_q_max), mass_prior_b=req.mass_prior_b)
    else:
        model = ModelForm(sigma_mf_dex=req.sigma_mf_dex,
                          angle_q=(req.angle_q_min, req.angle_q_max),
                          mass_prior_b=req.mass_prior_b)
    site = _site(req.latitude_deg, req.longitude_deg, req.target_class, req.geological_setting)
    morph = {k: getattr(req, k) for k in ("depth_m", "rim_height_m", "central_peak_height_m",
                                          "floor_diameter_m", "ejecta_radius_m", "melt_volume_m3")}
    try:
        out = infer_crater(
            req.rim_diameter_m, req.sigma_meas_dex,
            populations=populations, packages=tuple(req.packages),
            site=site, model=model, event=ev, gravity_m_s2=body.gravity_m_s2,
            n_draws=req.n_draws, seed=req.seed, morphometry=morph,
        )
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(422, str(exc)) from exc
    if secondary is not None:
        out["secondary_estimate"] = {"probability": secondary.probability, "population_weights": populations,
                                     "in_training_range": secondary.in_training_range,
                                     "out_of_range": list(secondary.out_of_range)}
        out["flags"].insert(0, f"Secondary-crater probability from depth/diameter and ellipticity: "
                               f"{secondary.probability:.0%}; used as the population weight. This classifier was "
                               "fitted on the Kepler region only and did NOT transfer to the Copernicus secondary "
                               "field (AUROC 0.39); use it only for Kepler-like settings.")
        if not secondary.in_training_range:
            out["flags"].insert(0, "Secondary probability extrapolates beyond its training craters ("
                                + ", ".join(secondary.out_of_range) + "); treat it as indicative.")
    if req.spacecraft_calibration:
        out["flags"].insert(0, "Spacecraft calibration: scatter learned from 11 lunar spacecraft craters "
                               "(2.4-2.7 km/s); not valid for natural impactors.")
    D = req.rim_diameter_m
    for key, label in (("depth_m", "Depth"), ("rim_height_m", "Rim height"), ("floor_diameter_m", "Floor diameter")):
        v = morph.get(key)
        if v is not None and v >= D:
            out["flags"].insert(0, f"{label} is not smaller than the rim diameter; check the measurement.")
    if morph.get("central_peak_height_m") and morph.get("depth_m") and morph["central_peak_height_m"] >= morph["depth_m"]:
        out["flags"].insert(0, "Central peak height is not smaller than the crater depth; check the measurement.")
    loc_flag = _partial_location_flag(req.latitude_deg, req.longitude_deg)
    if loc_flag:
        out["flags"].insert(0, loc_flag)
    out["body"] = body.describe()
    return _clean(out)


class ForwardIn(BaseModel):
    impactor_diameter_m: float | None = Field(None, gt=0, le=1e5)
    mass_kg: float | None = Field(None, gt=0)
    velocity_km_s: float = Field(..., gt=0, le=80)
    elevation_deg: float = Field(45.0, gt=0, le=90)
    impactor_density_kg_m3: float = Field(2700.0, gt=0, le=25000)   # gt=0 admits hollow spacecraft stages
    latitude_deg: float | None = Field(None, ge=-90, le=90)
    longitude_deg: float | None = Field(None, ge=-360, le=360)
    target_class: str | None = None
    packages: list[str] = Field(default_factory=lambda: list(DEFAULT_PACKAGE_SET))
    sigma_mf_dex: float = Field(0.10, ge=0.0, le=1.0)
    angle_q_min: float = Field(0.5, ge=0.0, le=2.0)
    angle_q_max: float = Field(1.0, ge=0.0, le=2.0)
    body: str = "moon"
    seed: int = 0


@router.post("/forward")
def forward(req: ForwardIn) -> dict:
    body = BODIES.get(req.body)
    if body is None:
        raise HTTPException(422, f"unknown body {req.body}")
    if (req.impactor_diameter_m is None) == (req.mass_kg is None):
        raise HTTPException(422, "give exactly one of impactor_diameter_m or mass_kg")
    if req.angle_q_min > req.angle_q_max:
        raise HTTPException(422, "angle_q_min must not exceed angle_q_max")
    _check_packages(req.packages)
    m = req.mass_kg
    if m is None:
        m = math.pi / 6.0 * req.impactor_density_kg_m3 * req.impactor_diameter_m ** 3
    site = _site(req.latitude_deg, req.longitude_deg, req.target_class)
    out = forward_predict(
        mass_kg=m, velocity_m_s=req.velocity_km_s * 1000.0, elev_deg=req.elevation_deg,
        delta=req.impactor_density_kg_m3, site=site, packages=tuple(req.packages),
        model=ModelForm(sigma_mf_dex=req.sigma_mf_dex,
                        angle_q=(req.angle_q_min, req.angle_q_max)),
        gravity_m_s2=body.gravity_m_s2, seed=req.seed,
    )
    loc_flag = _partial_location_flag(req.latitude_deg, req.longitude_deg)
    out["flags"] = [loc_flag] if loc_flag else []
    out["body"] = body.describe()
    return _clean(out)


class ForwardPointIn(BaseModel):
    impactor_diameter_m: float = Field(..., gt=0, le=1e5)
    velocity_km_s: float = Field(..., gt=0, le=80)
    impactor_density_kg_m3: float = Field(..., gt=0, le=25000)
    target_density_kg_m3: float = Field(..., gt=0, le=10000)
    target_strength_pa: float = Field(..., ge=0, le=1e11)
    target_class: int = Field(0, ge=0, le=2)
    angle_from_vertical_deg: float = Field(45.0, ge=0, lt=90)
    package: str = "lpi2007"
    body: str = "moon"


@router.post("/forward_point")
def forward_point_api(req: ForwardPointIn) -> dict:
    body = BODIES.get(req.body)
    if body is None:
        raise HTTPException(422, f"unknown body {req.body}")
    if req.package not in PACKAGES:
        raise HTTPException(422, f"unknown package {req.package}; options {list(PACKAGES)}")
    try:
        out = forward_point(req.impactor_diameter_m, req.velocity_km_s * 1000.0, req.impactor_density_kg_m3,
                            req.target_density_kg_m3, max(req.target_strength_pa, 1e-6), req.target_class,
                            angle_from_vertical_deg=req.angle_from_vertical_deg, package=req.package,
                            gravity_m_s2=body.gravity_m_s2)
    except (ValueError, OverflowError, FloatingPointError) as exc:
        raise HTTPException(422, str(exc)) from exc
    return _clean(out)


@router.get("/site")
def site(lat: float, lon: float) -> dict:
    if not (-90 <= lat <= 90):
        raise HTTPException(422, "latitude must be within [-90, 90]")
    ctx = site_context(lat, lon)
    return _clean({
        "label": ctx.label, "p_mare": ctx.p_mare, "regolith_thickness_median_m": ctx.h_median_m,
        "regolith_thickness_sigma_ln": ctx.h_sigma_ln, "sources": list(ctx.sources),
        "notes": list(ctx.notes), "extra": ctx.extra,
    })


@router.get("/options")
def options() -> dict:
    return {
        "populations": {k: {"label": p.label, "source": p.source,
                            "hypervelocity": p.hypervelocity} for k, p in POPULATIONS.items()},
        "packages": {k: {"label": p.label, "source": p.source} for k, p in PACKAGES.items()},
        "default_packages": list(DEFAULT_PACKAGE_SET),
        "target_classes": list(TARGET_CLASSES),
        "ejecta_patterns": EJECTA_PATTERNS,
        "flash_definitions": {k: v["label"] for k, v in FLASH_DEFINITIONS.items()},
        "bodies": {k: b.describe() for k, b in BODIES.items()},
        "defaults": {"sigma_meas_dex": 0.03, "sigma_mf_dex": 0.10, "angle_q": [0.5, 1.0],
                     "mass_prior_b": None,
                     "mass_prior_b_by_population": {k: p.mass_prior_b for k, p in POPULATIONS.items()}},
    }


@router.get("/benchmarks")
def benchmarks() -> dict:
    p = REPO / "results" / "benchmarks_v2.json"
    if not p.exists():
        raise HTTPException(404, "benchmarks not generated; run scripts/run_benchmarks_v2.py")
    return json.loads(p.read_text())


@router.get("/comparison")
def comparison() -> dict:
    p = REPO / "results" / "comparison_original_vs_v2.json"
    if not p.exists():
        raise HTTPException(404, "comparison not available: results/comparison_original_vs_v2.json is missing")
    return json.loads(p.read_text())


@router.get("/findings")
def findings() -> dict:
    p = REPO / "results" / "findings.json"
    if not p.exists():
        raise HTTPException(404, "findings not available: results/findings.json is missing")
    return json.loads(p.read_text())


def _clean(o):
    """Make numpy/NaN values JSON-safe."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if not math.isfinite(f) else f
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    return o
