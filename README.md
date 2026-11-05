# Crater2Impactor

**Impactor Parameter Estimation from Lunar Impact Craters**

Crater2Impactor (C2I) infers what a simple lunar crater can say about the body that made it, and states plainly what it cannot.

A crater's rim diameter fixes one combination of impactor mass and speed for a given target. It does not fix the speed, the impact angle, the impactor density or the target strength. C2I therefore computes a full posterior under explicit, editable assumptions. Each output quantity is labelled as constrained, prior-dominated or assumption-dependent, and independent observations (a measured speed, angle, mass, impact-flash energy or ejecta pattern) can be added to tighten it.

## Scientific approach

- **Forward model.**
  - Point-source crater scaling (Holsapple's π-group law) with three named constant sets: `lpi2007` (LPI Lunar Impact Craters calculator), `holsapple2003` (Holsapple's theory document) and `impacts2022` (Holsapple's IMPACTS).
  - Model-form uncertainty is carried explicitly: the impact-angle law, the impactor-density exponent ν, and a 0.10 dex residual scatter.
- **Target model.**
  - A regolith layer over a substrate (fractured highland crust or mare basalt), with the regolith thickness drawn from published measurements.
  - Per impactor, the crater forms in the regolith while the regolith crater stays shallow, in the substrate otherwise, and at the regime switch in between.
  - Forward prediction and inverse inference use this same rule.
- **Impactor populations.** Declared priors for asteroidal and sporadic-meteoroid primaries, secondaries and spacecraft. Natural primaries use the Brown et al. (2002) mass-frequency slope.
- **Inference.**
  - Importance sampling on the exact fibre of each crater. For each draw of the nuisance parameters, the scaling law is inverted for the mass.
  - Proposals are tilted and defensive, with exact density corrections; measured speeds and angles, rocky-ejecta observations and flash energies have their own event-informed proposals, so that observations far in a prior's tail still give a usable sample.
  - A result is marked unreliable, with the reason, when the effective sample is too small, when the posterior reaches the numerical mass bound, or when a measured speed, angle, mass or flash energy conflicts with the crater and the declared populations (predictive p < 10⁻³).
  - Scenarios (population × constant set) are averaged with crater-level weights.
  - A deterministic convolution backend is used where it is exact: a fixed target class, or craters far from the layered regime.

The formulation, every assumption and its source are in [`docs/SCIENCE.md`](docs/SCIENCE.md).

## Installation

Python ≥ 3.11.

```bash
python -m venv .venv
```

```bash
.venv/bin/python -m pip install -e ".[serve,test]"
```

The engine itself needs only NumPy and SciPy. `serve` adds the web API, and `test` adds pytest and the packages the validation scripts use.

## Usage

```python
from c2i.engine import EventData, infer_crater
from c2i.engine.lunar_context import site_context

r = infer_crater(30.0, 0.03, site=site_context(-3.0, -23.0))
print(r["quantities"]["log10_mass_kg"])        # median, q05, q95, status, ...

ev = EventData(velocity_m_s=2500.0, elev_deg=60.0)   # e.g. a known artificial impact
r = infer_crater(14.0, 0.04, populations={"artificial": 1.0}, event=ev)
```

The forward direction (impactor → crater) is `c2i.engine.forward.forward_predict`. The web interface and API:

```bash
.venv/bin/python -m uvicorn app.server_v2:app --port 8000
```

Endpoints: `POST /api/v2/infer`, `POST /api/v2/forward`, `POST /api/v2/forward_point`, `GET /api/v2/site`, `GET /api/v2/options`, `GET /api/v2/benchmarks`, `GET /api/v2/comparison`, `GET /api/v2/findings`.

## Data

All data needed to run, test and reproduce the published results are in `data/`. The external archives needed only to rebuild the lunar context grid are listed, with sources, versions and checksums, in [`DATA_MANIFEST.md`](DATA_MANIFEST.md).

## Benchmarks

The automated test suite, the audit record and the independent verification scripts are kept in the development repository and are not part of this public release; references to them in the documentation point there.

```bash
PYTHONPATH=src .venv/bin/python scripts/run_benchmarks_v2.py
```

```bash
.venv/bin/python scripts/write_validation_doc.py
```

`run_benchmarks_v2.py` writes `results/benchmarks_v2.json`, recording the engine-file checksums, seeds and software versions. `write_validation_doc.py` renders [`docs/VALIDATION.md`](docs/VALIDATION.md) from it. The recorded checksums show whether the results file was produced by the current code.

## Validation evidence and limitations

Three kinds of evidence are kept separate.

**1. Real measurements with an independently known impactor.**
- **Laboratory (block L).**
  - Dry-sand craters recover the scaling exponent μ = 0.395 (90 % 0.384–0.410; published 0.41).
  - The shipped regolith constants predict craters in three granular materials they were not fitted to (10 shots): mean −0.03 dex, RMS 0.07 dex, 10/10 inside the 90 % band.
- **Lunar spacecraft impacts (block P, 11 events).**
  - Forward crater diameter: median residual +0.02 dex, RMS 0.045 dex, 11/11 inside 90 %.
  - Inverse mass with speed and angle known: median error +0.08 dex, RMS 0.17 dex, 11/11 inside 90 %.
  - The intervals are about 2–3× wider than these errors need.
  - All 11 events were examined during development, and all are slow, hollow impactors.

**2. Natural lunar observations without a known impactor.**
- **New-crater production (block W).** With an independent bolide flux and no tuning, the constant sets predict 22–33 new craters ≥ 7 m, against 34 observed in LROC temporal images.
- **Impact flashes with craters.** These give consistency checks only, because the mass follows from the flash through an assumed luminous efficiency.

**3. Synthetic and numerical checks.**
- In-model coverage: 90 % intervals contain the truth in 0.90–0.94 of cases.
- Robustness under misspecification is tested in block R.
- The sampler agrees with an independent forward-model sampler to ≤ 0.003 dex (blocks M and C).

These checks establish that the code computes its declared model correctly. They say nothing about the real Moon.

**Known limitations.**
- **Natural-impactor mass accuracy is not validated:** no natural impactor has an independently known mass. Every real test with known impactors uses either hollow spacecraft at 2.4–2.7 km/s or gram-scale laboratory projectiles (0.5–7.3 km/s), not compact natural bodies at ~20 km/s.
- **Untestable assumptions.** The fractured-rock scaling constants and the highland regolith thickness are declared assumptions that available data cannot test. At D ≈ 30–100 m they move the mass by up to ~0.2 dex.
- **Monte Carlo precision in tension.** When a flash or a precise mass measurement is in tension with the crater, the effective sample can drop to a few hundred of 30,000 draws, and quantiles then move by up to ~0.04 dex between seeds. The effective sample size is reported, and low values are flagged or marked unreliable.
- **Domain.** Speed, angle and impactor density are prior-dominated from a diameter alone. Complex craters (D ≳ 10 km), grazing impacts and very slow impacts are flagged as outside the validated domain.

## Repository layout

- `src/c2i/engine/`: scaling packages, targets, populations, observation models, sampler, deterministic backend, inference.
- `app/`: FastAPI service (`api_v2.py`, `server_v2.py`) and the web pages (`static/`).
- `scripts/`: benchmarks (`run_benchmarks_v2.py`, `validation_v2.py`), report rendering, lunar-context builder.
- `data/`: lunar context grid, validation datasets, laboratory compilation.
- `results/`: benchmark results and the findings shown on the Validation page.
- `docs/`: scientific formulation, generated validation report, release audit (`RELEASE_AUDIT.md`), notes on other bodies.
- `research/otfmpe/`: the earlier neural approach (OT-FMPE), retrained on the current engine's physics and benchmarked against it; results in `results/otfmpe_comparison/`. Research code, not used by the web application.

## Citation and licence

See [`CITATION.cff`](CITATION.cff) and [`LICENSE`](LICENSE) (MIT). Third-party data keep their own licences and provenance ([`DATA_MANIFEST.md`](DATA_MANIFEST.md)).
