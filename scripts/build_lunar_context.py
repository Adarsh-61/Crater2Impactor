"""Build the coordinate -> target-context grid used by c2i.engine.lunar_context.

Inputs (downloaded once into data/lunar/raw/, checksums recorded in the output):
  * LROC_GLOBAL_MARE_180.ZIP - LROC mare boundaries (Nelson et al. 2014, LPSC 45 #2861),
    PDS LROC RDR EXTRAS/SHAPEFILE. Documented coverage 65S-65N; a few polygons reach about 66.5S.
  * rajsic_regolith_v2.0.0.zip - Jones & Rajsic (2026) lunar-surface-regolith v2.0.0
    (GPL-3.0 code; data tables), accompanying Rajsic et al. (2026) PSJ. We use only
    reproduce/data/RegolithData_Literature.xlsx (54 published point measurements; the sheet has 62 rows, 8 blank) and
    re-implement their inverse-distance weighting (k = 9 neighbours, power 2) here.

Outputs:
  data/lunar/lunar_context_grid.npz  (0.5 deg cells: mare_fraction, regolith_h_median_m,
                                      regolith_h_sigma_ln, dist_to_measurement_km)
  data/lunar/lunar_context_meta.json (provenance, checksums, method, cross-validation)

The leave-one-out cross-validation compares the IDW map with a terrain-mean baseline
(mare vs highland mean of the training points) on the 54 literature measurements. It is
the independent test of whether coordinates improve the regolith-thickness prior.
"""

from __future__ import annotations

import hashlib
import io
import json
import struct
import sys
import zipfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "lunar" / "raw"
OUT = REPO / "data" / "lunar"
RES = 0.5
R_MOON_KM = 1737.4
SIGMA_LN_MARE = 0.59     # geometric s.d. factor ~1.8 (assumed site-scale spread; see targets.GLOBAL_CONTEXT)
SIGMA_LN_HIGHLAND = 0.69  # factor ~2


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read_polygons(shp_bytes: bytes):
    """Minimal ESRI shapefile reader for Polygon (type 5) records -> list of rings."""
    rings = []
    pos = 100
    n = len(shp_bytes)
    while pos + 8 <= n:
        _, clen = struct.unpack(">ii", shp_bytes[pos:pos + 8])
        rec = shp_bytes[pos + 8: pos + 8 + 2 * clen]
        pos += 8 + 2 * clen
        stype = struct.unpack("<i", rec[:4])[0]
        if stype != 5:
            continue
        nparts, npts = struct.unpack("<ii", rec[36:44])
        parts = list(struct.unpack(f"<{nparts}i", rec[44:44 + 4 * nparts]))
        off = 44 + 4 * nparts
        pts = np.frombuffer(rec[off: off + 16 * npts], dtype="<f8").reshape(npts, 2)
        parts.append(npts)
        for a, b in zip(parts[:-1], parts[1:]):
            rings.append(pts[a:b])
    return rings


def mare_fraction_grid(rings, sub=4):
    """Fraction of each RES cell inside mare polygons (even-odd rule over all rings)."""
    from matplotlib.path import Path as MPath
    nlat, nlon = int(180 / RES), int(360 / RES)
    fine = RES / sub
    lats = 90 - fine * (np.arange(nlat * sub) + 0.5)
    lons = -180 + fine * (np.arange(nlon * sub) + 0.5)
    LON, LAT = np.meshgrid(lons, lats)
    pts = np.column_stack([LON.ravel(), LAT.ravel()])
    inside = np.zeros(len(pts), dtype=bool)
    for r in rings:
        lo = r.min(0)
        hi = r.max(0)
        sel = np.flatnonzero((pts[:, 0] >= lo[0]) & (pts[:, 0] <= hi[0]) &
                             (pts[:, 1] >= lo[1]) & (pts[:, 1] <= hi[1]))
        if sel.size == 0:
            continue
        hit = MPath(r).contains_points(pts[sel])
        inside[sel[hit]] ^= True   # even-odd: holes toggle back out
    f = inside.reshape(nlat * sub, nlon * sub).astype(float)
    return f.reshape(nlat, sub, nlon, sub).mean(axis=(1, 3))


def gc_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(lon2 - lon1)
    c = np.sin(p1) * np.sin(p2) + np.cos(p1) * np.cos(p2) * np.cos(dl)
    return R_MOON_KM * np.arccos(np.clip(c, -1, 1))


def idw(lat_pts, lon_pts, vals, lat_q, lon_q, k=9, p=2):
    d = gc_km(lat_q[:, None], lon_q[:, None], lat_pts[None, :], lon_pts[None, :])
    idx = np.argsort(d, axis=1)[:, :k]
    dk = np.take_along_axis(d, idx, axis=1)
    w = 1.0 / np.maximum(dk, 1e-6) ** p
    est = np.sum(w * vals[idx], axis=1) / w.sum(axis=1)
    return est, dk[:, 0]


def load_literature_points():
    import openpyxl
    z = zipfile.ZipFile(RAW / "rajsic_regolith_v2.0.0.zip")
    raw = z.read("lunar-surface-regolith-2.0.0/reproduce/data/RegolithData_Literature.xlsx")
    wb = openpyxl.load_workbook(io.BytesIO(raw), read_only=True, data_only=True)
    rows = list(wb["data_at_points"].iter_rows(values_only=True))
    hdr = rows[0]
    out = []
    for r in rows[1:]:
        if r[0] is None or r[2] is None:
            continue
        if not isinstance(r[2], (int, float)):
            # all v2.0.0 entries are numbers; a range or text entry needs an explicit decision, not a silent choice
            raise ValueError(f"non-numeric regolith thickness {r[2]!r} at ({r[0]}, {r[1]})")
        th = float(r[2])
        lon = float(r[1])
        lon = lon - 360 if lon > 180 else lon
        out.append((float(r[0]), lon, th, str(r[3]), str(r[4])))
    return hdr, out


def main():
    mare_zip = RAW / "LROC_GLOBAL_MARE_180.ZIP"
    reg_zip = RAW / "rajsic_regolith_v2.0.0.zip"
    for p in (mare_zip, reg_zip):
        if not p.exists():
            sys.exit(f"missing {p}; see docstring for sources")
    shp = zipfile.ZipFile(mare_zip).read("LROC_GLOBAL_MARE_180.SHP")
    rings = read_polygons(shp)
    mf = mare_fraction_grid(rings)

    _, pts = load_literature_points()
    plat = np.array([p[0] for p in pts])
    plon = np.array([p[1] for p in pts])
    pth = np.array([p[2] for p in pts])
    lth = np.log(pth)

    # unit of each measurement from the mare grid
    def cell(lat, lon):
        i = np.clip(((90 - lat) / RES).astype(int), 0, mf.shape[0] - 1)
        j = np.clip(((lon + 180) / RES).astype(int), 0, mf.shape[1] - 1)
        return i, j
    pi_, pj_ = cell(plat, plon)
    pmare = mf[pi_, pj_] >= 0.5

    # leave-one-out cross-validation (log thickness)
    loo_idw, loo_unit, loo_global = [], [], []
    for i in range(len(pts)):
        m = np.ones(len(pts), dtype=bool)
        m[i] = False
        e, _ = idw(plat[m], plon[m], lth[m], plat[i:i + 1], plon[i:i + 1])
        loo_idw.append(e[0] - lth[i])
        same = m & (pmare == pmare[i])
        loo_unit.append(lth[same].mean() - lth[i])
        loo_global.append(lth[m].mean() - lth[i])
    cv = {
        "n_points": len(pts),
        "rmse_ln_idw": float(np.sqrt(np.mean(np.square(loo_idw)))),
        "rmse_ln_terrain_mean": float(np.sqrt(np.mean(np.square(loo_unit)))),
        "rmse_ln_global_mean": float(np.sqrt(np.mean(np.square(loo_global)))),
        "note": f"Leave-one-out on log thickness of the {len(pts)} literature points. Points cluster at "
                "landing sites, so LOO is optimistic for unsampled regions.",
    }

    # grid
    nlat, nlon = mf.shape
    lats = 90 - RES * (np.arange(nlat) + 0.5)
    lons = -180 + RES * (np.arange(nlon) + 0.5)
    LON, LAT = np.meshgrid(lons, lats)
    est, dnear = idw(plat, plon, lth, LAT.ravel(), LON.ravel())
    h_med = np.exp(est).reshape(nlat, nlon)
    dist = dnear.reshape(nlat, nlon)
    sig = np.where(mf >= 0.5, SIGMA_LN_MARE, SIGMA_LN_HIGHLAND)
    # widen the prior away from measurements: add the LOO error in quadrature beyond 300 km
    sig = np.sqrt(sig**2 + np.where(dist > 300, cv["rmse_ln_idw"] ** 2, 0.0))
    # cells without mare polygons (all poleward of about 66.5S and 65N) have mare fraction 0, i.e. highland
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "lunar_context_grid.npz",
                        mare_fraction=mf.astype(np.float32),
                        regolith_h_median_m=h_med.astype(np.float32),
                        regolith_h_sigma_ln=sig.astype(np.float32),
                        dist_to_measurement_km=dist.astype(np.float32))
    meta = {
        "resolution_deg": RES,
        "grid_origin": "row 0 = 90N..89.5N, col 0 = 180W..179.5W",
        "inputs": {
            "LROC_GLOBAL_MARE_180.ZIP": {"sha256": sha(mare_zip), "url":
                "https://pds.lroc.im-ldi.com/data/LRO-L-LROC-5-RDR-V1.0/LROLRC_2001/EXTRAS/"
                "SHAPEFILE/LROC_GLOBAL_MARE/LROC_GLOBAL_MARE_180.ZIP", "n_rings": len(rings)},
            "rajsic_regolith_v2.0.0.zip": {"sha256": sha(reg_zip), "url":
                "https://github.com/mattjsquared/lunar-surface-regolith/archive/refs/tags/v2.0.0.zip",
                "file_used": "reproduce/data/RegolithData_Literature.xlsx"},
        },
        "sources": [
            "Mare fraction: LROC global mare boundaries (Nelson et al. 2014, LPSC 45, #2861), "
            "rasterised at 0.125 deg and averaged to 0.5 deg; documented coverage 65S-65N (a few polygons reach 66.5S).",
            "Regolith thickness: published point measurements compiled by Rajsic et al. (2026, PSJ) / "
            "Jones & Rajsic (2026) v2.0.0 (54 usable rows). The inverse-distance map (k=9, p=2) is "
            "stored but NOT used for inference because it failed leave-one-out validation; the "
            "engine uses mare/highland geometric means with log-sd 0.59/0.69.",
        ],
        "cross_validation": cv,
        "decision": "Use observed mare fraction; use unit-level regolith prior; do not use the "
                    "interpolated thickness map (it does not beat a constant in LOO).",
        "mare_area_fraction_65S_65N": float(np.average(
            mf[(np.abs(lats) <= 65)], weights=np.repeat(np.cos(np.radians(lats[np.abs(lats) <= 65]))[:, None], nlon, 1))),
        "unit_means_m": {"mare": float(np.exp(lth[pmare].mean())), "highland": float(np.exp(lth[~pmare].mean())),
                         "n_mare": int(pmare.sum()), "n_highland": int((~pmare).sum())},
    }
    (OUT / "lunar_context_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: meta[k] for k in ("cross_validation", "mare_area_fraction_65S_65N", "unit_means_m")}, indent=1))


if __name__ == "__main__":
    main()
