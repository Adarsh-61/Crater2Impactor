# Data manifest

Every dataset the code reads, where it comes from, and whether it is in the repository. The SHA-256 prefixes identify the versions used for the published results (`results/benchmarks_v2.json`).

## Included in the repository

| File | Content | Source | SHA-256 (first 16) |
|---|---|---|---|
| `data/lunar/lunar_context_grid.npz`, `data/lunar/lunar_context_meta.json` | 0.5° grid of mare fraction and regolith-thickness context; provenance and cross-validation | Built by `scripts/build_lunar_context.py` from the two external archives below | `04eef45234628027`, `2f0bc049e44a73b7` |
| `data/validation/spacecraft_impacts.json` | 11 lunar spacecraft impacts: known mass, speed and elevation, plus crater diameter (Block P) | NSSDC Apollo impact table; JPL technical reports TR 32-699/700/800 (Rangers); LROC featured sites and posts; Marshall et al. 2011 (LCROSS); Gray's trajectory pages and LROC post 1499 (Falcon 9 2025-010D). Source per event in its `source` field | `eda68b180773d9cb` |
| `data/validation/ground_truth_events.json` | Earlier compilation of the same events, including the excluded GRAIL and LADEE impacts | As above | `a7fe0a0fd8eb4212` |
| `data/validation/flash_crater_pairs.json` | Lunar impact flashes with an identified new crater, with each source's photometric definition | Suggs et al. 2014 (Icarus 238, 23); Madiedo et al. 2014 (MNRAS 439, 2364) and 2019 (MNRAS 486, 3380); Sheward et al. 2022 (MNRAS 514, 4320) and 2025 (A&A 699, L3); Rizos et al. 2025; Robinson et al. 2014 (per pair in the file) | `832be1873a186ef4` |
| `data/validation/new_crater_population.json` | New-crater counts and survey exposure (Block W); bolide flux | Digitised from Speyerer et al. 2016, Nature 538, 215, Fig. 2; Brown et al. 2002, Nature 420, 294 | `8ea81a078ba29d02` |
| `data/validation/lpi_live_oracle.json` | 405 reference cases of the LPI Lunar Impact Craters calculator | https://www.lpi.usra.edu/lunar/tools/lunarcratercalc/ (v1.4, 10/07), instrumented copy of `fresult()`, retrieved 2026-10-11 | `456797a56a3b0a0f` |
| `data/lab/holsapple_impacts_v1.3.csv` | Laboratory crater compilation (Block L; dry-sand law recovery and held-out granular transfer) | K. A. Holsapple, crater scaling database v1.3. The original host no longer responds; the public copy survives on the Internet Archive | `4c03f40e127048f3` |

**Known data defects** (documented, not corrected in the files):
- `holsapple_impacts_v1.3.csv`: 27 rows ("Pkastani Sandstone") are column-shifted, and code that reads the file excludes them. The 7 "White Sands Missile Impacts" rows misstate Moore (1976, USGS Professional Paper 812-B): the impacts were oblique; most targets are gypsum or colluvium, not alluvium; and the masses are not in the source. These rows are not used by any test.
- The LCROSS crater size is disputed in the literature (20 m to 30 m). The file uses the only direct image, 22 m (Fassett et al. 2024).

## External: not redistributed, needed only to rebuild derived files

Download these into the paths shown. The checksums are those recorded in `data/lunar/lunar_context_meta.json`.

| Dataset | Needed for | Path | Source | SHA-256 |
|---|---|---|---|---|
| LROC global mare boundaries (Nelson et al. 2014, LPSC 45, #2861) | Rebuilding the lunar context grid | `data/lunar/raw/LROC_GLOBAL_MARE_180.ZIP` | https://pds.lroc.im-ldi.com/data/LRO-L-LROC-5-RDR-V1.0/LROLRC_2001/EXTRAS/SHAPEFILE/LROC_GLOBAL_MARE/LROC_GLOBAL_MARE_180.ZIP | `a85073f6fe05a0b9…` |
| Jones & Rajsic (2026), lunar-surface-regolith v2.0.0, accompanying Rajsic et al. (2026, PSJ); only `reproduce/data/RegolithData_Literature.xlsx` is used | Rebuilding the lunar context grid | `data/lunar/raw/rajsic_regolith_v2.0.0.zip` | https://github.com/mattjsquared/lunar-surface-regolith/archive/refs/tags/v2.0.0.zip | `bb9f6d985404c971…` |
| Robbins (2019) global lunar crater database, as an `.xlsx` export | Two optional tests (`INTERNAL_ONLY/tests/test_flash_artificial_angle.py`, `INTERNAL_ONLY/tests/test_secondary_catalogue.py`), which are skipped when it is absent | `Real_data/Full_Crater_Catalogue.xlsx` | Robbins, S. J. (2019), JGR Planets 124, 871–892 (supplementary database) | — |

To rebuild the grid:

```bash
.venv/bin/python -m pip install -e ".[data]"
```

```bash
.venv/bin/python scripts/build_lunar_context.py
```
