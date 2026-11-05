"""Location-aware lunar target context: latitude/longitude -> target-property prior.

What can be mapped from coordinates (see docs/SCIENCE.md, sec. 4.3):

* geologic unit, mare vs highland (observed: albedo/composition boundaries);
* regolith thickness (inferred: from crater morphology or thermal/radar
  proxies, with regional rather than point-level reliability).

What cannot be read from coordinates and stays a prior: cratering strength,
porosity at depth, and layering details. These are never presented as local
measurements.

Data files live in ``data/lunar/`` and are produced by
``scripts/build_lunar_context.py`` from public sources with recorded
checksums. If they are absent the module falls back to the global prior and
says so.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import numpy as np

from .targets import GLOBAL_CONTEXT, SiteContext

DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "lunar"


@lru_cache(maxsize=1)
def _load():
    meta_p = DATA_DIR / "lunar_context_meta.json"
    grid_p = DATA_DIR / "lunar_context_grid.npz"
    if not (meta_p.exists() and grid_p.exists()):
        return None
    meta = json.loads(meta_p.read_text())
    grid = dict(np.load(grid_p))
    return meta, grid


def _lookup(grid: np.ndarray, lat: float, lon: float, res_deg: float) -> float:
    # grid rows: latitude from +90 down; columns: longitude from -180 east
    i = int(np.clip(math.floor((90.0 - lat) / res_deg), 0, grid.shape[0] - 1))
    j = int(np.clip(math.floor((((lon + 180.0) % 360.0)) / res_deg), 0, grid.shape[1] - 1))
    return float(grid[i, j])


def site_context(lat: float | None, lon: float | None) -> SiteContext:
    """Target context at (lat, lon).

    Uses the observed mare fraction of the 0.5-deg cell. Regolith thickness is NOT
    taken from the interpolated map: in leave-one-out tests on the 54 published
    measurements the interpolated map predicted worse than a constant (see
    lunar_context_meta.json, cross_validation), so the unit-level prior is used
    and the map value is reported for information only.
    """
    if lat is None or lon is None:
        return GLOBAL_CONTEXT
    loaded = _load()
    if loaded is None:
        return replace(
            GLOBAL_CONTEXT,
            label=f"({lat:.2f}, {lon:.2f}): no lunar context data installed; global prior used",
            notes=("Run scripts/build_lunar_context.py to install the mapped datasets.",),
        )
    meta, grid = loaded
    res = float(meta["resolution_deg"])
    lon180 = ((lon + 180.0) % 360.0) - 180.0
    frac_mare = _lookup(grid["mare_fraction"], lat, lon180, res)
    h_map = _lookup(grid["regolith_h_median_m"], lat, lon180, res)
    dist = _lookup(grid["dist_to_measurement_km"], lat, lon180, res)
    um = meta["unit_means_m"]
    cv = meta["cross_validation"]
    unit = "mare" if frac_mare >= 0.5 else "highland"
    # The LROC polygons are documented for 65S-65N but a few reach about 66.5S; poleward
    # cells without polygons have mare fraction 0, i.e. highland. Say which case applies.
    edge = ""
    if abs(lat) > 65.0:
        edge = (" Latitude beyond 65 deg, at the edge of the LROC mare map: the mapped polygons are used."
                if frac_mare > 0.0 else
                " Latitude beyond 65 deg: no mare polygons mapped here, treated as highland.")
    notes = [
        f"Cell {res:g} deg: mare fraction {frac_mare:.2f} from LROC mare boundaries (observed)." + edge,
        f"Regolith thickness prior by unit: mare median {um['mare']:.1f} m, highland median "
        f"{um['highland']:.1f} m (geometric means of {um['n_mare']}+{um['n_highland']} published "
        "measurements), wide log-normal spread for site-scale variability.",
        f"An interpolated thickness map gives {h_map:.1f} m here (nearest measurement "
        f"{dist:.0f} km), but it failed leave-one-out validation (log-RMSE {cv['rmse_ln_idw']:.2f} "
        f"vs {cv['rmse_ln_global_mean']:.2f} for a constant), so it is not used.",
        "Cratering strength, friction and deep porosity are not mapped; class priors apply.",
    ]
    return SiteContext(
        p_mare=float(np.clip(frac_mare, 0.0, 1.0)),
        h_by_unit=((um["mare"], 0.59), (um["highland"], 0.69)),
        h_median_m=um["mare"] if unit == "mare" else um["highland"],
        h_sigma_ln=0.59 if unit == "mare" else 0.69,
        label=f"({lat:.2f}, {lon180:.2f}): {unit} cell",
        sources=tuple(meta.get("sources", [])),
        notes=tuple(notes),
        extra={"mare_fraction": frac_mare, "regolith_h_map_m_not_used": h_map,
               "dist_to_measurement_km": dist},
    )
