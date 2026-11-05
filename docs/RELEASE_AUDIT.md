# Release audit (version 2.1.0, November 2026)

This records what was checked before release 2.1.0, what was fixed, how each fix was verified, and what remains uncertain. It does not claim that every bug has been found. The audit of release 2.0.0 is summarised in §7.

## 1. How the audit was done

- **Claims from an external AI-assisted review (Gemini): proposal collapse, SRC node windows, boundary speeds, flash likelihood, input validation and a coupled-class SRC** were reproduced with independent scripts against the 2.0.0 engine before any change was made. The verdict on each claim is in `_archive/s40l_gemini_review/REVIEW.md`; the archive is local and not distributed. No Gemini code was merged. Every fix below was written and tested independently.
- **Randomised robustness sweep.**
  - 360 random inputs across the API range: crater size, measurement error, population mixtures, constant sets, every event type, sites, σ_mf ranges and mass-prior slopes.
  - Each run was checked for crashes, non-finite or mis-ordered quantiles, and low effective sample size (ESS) without a stated reason.
- **Randomised check of the deterministic coupled-class backend proposed by Gemini.**
  - 40 random configurations plus 9 hand-picked event cases.
  - The references are 1.5–10M-draw Monte Carlo runs.
- **Regression tests.**
  - Each fix has a test in `INTERNAL_ONLY/tests/test_sampler_robustness.py`.
  - The new tests were run against a snapshot of the 2.0.0 engine: 17 of 22 fail there. The other 5 are input-validation cases where 2.0.0 already raised a ValueError, for a different reason.
- **Final runs.** The full test suite and the benchmark suite were rerun on the final code, and old and new benchmark results were compared field by field (§4).

## 2. Defects found and fixed

All are proposal (efficiency) or numerical-window defects, except the missing conflict report and the input validation.

- **Monte Carlo sampler:** the target posterior was always correct in expectation, but some inputs left so few effective draws that results were noisy, falsely "unreliable", or stopped with "no admissible draws".
- **SRC crater-size window:** a real bias, labelled reliable.

| Defect | Effect in 2.0.0 | Fix and evidence |
|---|---|---|
| **Wide crater-size likelihoods.** The mass prior moves the posterior far below D_obs (≈ 2.5 dex for σ_meas = 0.6). The crater-size tilt was capped at 2σ, and the target-class proposal used D_obs | 5 km crater, σ_meas 0.55–1.0: min ESS 1.8–91; q05 spread over 2.3 dex across seeds; the stated reason was "incompatible with the priors" | Class components are proposed conditional on the sampled noise-free size. The shift may reach the mass-support floor. Now ESS 17–21k; q05 = −1.467 ± 0.002 over 3 seeds at σ_meas = 0.6 |
| **Crater-size tilt missing for populations without a kinematic tilt** (artificial, secondary) when the user sets a mass-prior slope b > 0 | ESS 32 (σ_meas 0.6) and 4.8 (σ_meas 1.0) | The crater-size tilt applies to every population with b > 0. ESS 22k and 20k |
| **SRC crater-size window.** The nodes stopped 5σ below log D_obs | Fixed-class sites, with q05 errors vs 2M-draw Monte Carlo: <br>• +0.95 dex at σ_meas = 0.5; <br>• +0.28 at 0.4; <br>• +0.044 at 0.3. <br>All labelled reliable | The window reaches 5σ plus 4 b ln10 σ² below, floored at the mass support, with proportionally more nodes. Now ≤ 0.008 dex |
| **Measured speed or angle outside a population's support** | Secondary population with 3.3 km/s measured: ESS 4–6. With 3.81 km/s: stopped with "no admissible draws". SRC nodes spanned only ±5σ | Truncated-normal proposals with exact log-mass weights. SRC nodes are placed on the support. SRC and Monte Carlo now agree to ≤ 0.003 dex |
| **No report of the conflict.** After the fix above, an incompatible measurement would give a "reliable" posterior pinned at the support edge | (would have been silent) | New predictive conflict checks for a measured speed, angle and mass. p < 0.05 is flagged; p < 10⁻³ marks the result unreliable. Output: `measurement_consistency` |
| A39a | **Flash luminous efficiency drawn from its prior** | Precise flash (σ_f = 0.03): ESS 640–1,100 | log10 η is integrated analytically and drawn from its exact conditional. ESS 8,200–14,000; same quantiles |
| **Flash likelihood tail clamp.** The closed form clamped at 1e-300 | Flat log-likelihood beyond ~8σ (SRC) | Stable log-Φ differences and erfcx-scaled second differences. They match adaptive quadrature to 1e-6 |
| **Event inputs not validated in the engine** | Zero speed σ: LAPACK error. Zero mass or flash σ, negative values, angles > 90°: misleading "incompatible". A negative σ was accepted silently. The API already rejected all of these | `sampler.validate_event` is called by the sampler, SRC and `infer_crater`. 12 test cases |
| **Rocky-ejecta observation.** Regolith thickness was drawn from its prior | No rocky ejecta at 500 m (needs h > 42 m): ESS 17–24, falsely unreliable. Rocky ejecta at 5 m: ESS 4.2k | Half of the draws come from the exact prior × likelihood density of log h. ESS 7.7k and 12k |
| — | `np.polyfit` "poorly conditioned" warning for the reported mass–velocity relation | Warning only | Centred least squares; NaN (API: null) when the speed has no spread |

The review's remark on the reference prior of the information labels is a definition, documented in `docs/SCIENCE.md`; it was not changed.

Its NumPy 1.26 point does not apply, because the package requires NumPy ≥ 2.0.

## 3. Checked and found correct, or not adopted

- **The reviewer's coupled-class SRC (`run_scenario_src_coherent`).**
  - Its earlier errors on rocky-ejecta and flash cases are fixed.
  - It is within 0.008 dex of the Monte Carlo references in 9/9 hand-picked and 39/40 random configurations.
  - One realistic flash case is off by 0.024 dex in q05 and the median: meteoroid, 257 m, 1.07×10⁹ J, flash p = 0.016. The 10M-draw reference has a standard error of about 0.002.
  - **Not integrated.** The Monte Carlo inverse it would replace is exact, and adding it would mean maintaining a second 1,400-line implementation of the coupled model. Its benefit (determinism, about 0.05 s instead of 0.4 s) does not outweigh that residual error and the integration risk.
- **The reviewer's spacecraft numbers** (11/11, RMS 0.162 dex) are consistent with benchmark block P on the Monte Carlo engine.
- **Robustness sweep (360 inputs).**
  - There were no unexpected exceptions, no non-finite or mis-ordered quantiles, and every output is JSON-serialisable.
  - 9 runs stopped with "no admissible draws". Each had a known impactor density outside every declared population's density support, which is the correct outcome.

## 4. Verification results

| Check | Result |
|---|---|
| Test suite | 196 passed, 2 skipped (they need the external Robbins catalogue), 1 expected failure (xfail); includes the benchmark-provenance test |
| Lint (ruff F, E9) | Clean |
| Benchmarks rerun (`results/benchmarks_v2.json`) | Block C coverage90 0.932 / 0.932 / 0.887. Block M forward–inverse coherence ≤ 0.007 dex (2.0.0: ≤ 0.024). Block P 11/11 forward and inverse, inverse median error +0.081 dex, RMS 0.169 dex. Block L held-out RMS 0.070 dex (unchanged). Block W 22–33 predicted vs 34 observed (unchanged) |
| Benchmark changes vs 2.0.0 | 134 non-timing fields changed, all at Monte Carlo noise level: <br>• coverages in blocks B, C and R moved by one or two cases (±0.004–0.015 of 133 or 250 cases); <br>• block P inverse medians moved by ≤ 0.004 dex; <br>• block M maximum differences fell (0.024 → 0.006 in the worst configuration). <br>Blocks L, W and T are unchanged |
| Clean install into a new virtual environment from a copy of the repository without `_archive/`, local environments or credentials | `pip install -e ".[serve,test]"` under Python 3.11 with NumPy 2.4: 196 passed, 2 skipped, 1 xfailed. The README examples run, and every `/api/v2` page and endpoint returns HTTP 200. No source file refers to `_archive/` |

## 5. What remains uncertain

- **Natural impactors.** The accuracy of natural-impactor masses is not validated, because no natural impactor has an independently known mass. Every test with a known impactor uses slow (≤ 2.7 km/s) laboratory or artificial projectiles.
- **Spacecraft intervals.** They are about 2–3× wider than the observed errors. Recalibrating on the same 11 events would be circular.
- **Untestable assumptions.** These move masses at 30–100 m by up to ~0.2 dex, and current data cannot choose between the alternatives:
  - the fractured-rock scaling constants;
  - the highland regolith thickness;
  - the convention that one strength prior is shared by the three constant sets.
- **Monte Carlo precision for events in tension.**
  - **Flash in tension with the crater** (predictive p ≈ 0.005–0.02): the ESS at 30,000 draws can fall to about 200–650. The q05 then scatters by up to ~0.04 dex between seeds. The ESS is reported, and below 200 a flag is shown.
  - **Precise mass in the tail of the crater-implied range:** a mass measured to ≤ 0.05 dex under a power-law prior can give ESS < 100. This is marked unreliable. Real spacecraft cases keep ESS 900–3,500.
- **Event cases.** The rocky-ejecta and flash observation models are less tested against real data than the diameter-only case.
- **Data defects.** Documented in `DATA_MANIFEST.md`: column-shifted laboratory rows, unreliable White Sands rows, and the disputed LCROSS crater size.

## 6. Repository changes in 2.1.0

- The reviewer's files and the 2.0.0 review ZIP were moved to the local archive, with rows in its manifest. Nothing was deleted.
- Version 2.1.0 in `pyproject.toml`, `CITATION.cff` and `c2i.__version__`.
- `docs/SCIENCE.md` §7–8 describe the new proposals and conflict checks.

## 7. Release 2.0.0 audit (summary)

- **Fixed.**
  - Forward and inverse models used different target-class rules: the median mass was up to 0.43 dex too low at D/h 4–20.
  - Collapsed ESS after the class fix.
  - Spurious target-information labels.
  - A known impactor density ignored the population prior.
  - Posteriors at the numerical mass bound were labelled reliable.
  - Invalid population inputs were accepted.
  - Mass-prior normaliser overflow.
  - Benchmark mass-support mismatch.
  - Block P priors.
  - Two spacecraft datasets held different values.
- **Repository.** It was cleaned for publication, and the engine depends only on NumPy and SciPy.
- **Checked and found correct.**
  - The Holsapple Eq. 7 kernel in all three constant sets, reproducing the LPI calculator on 405 live cases.
  - Inverse–forward coherence to 0.01–0.02 nat.
  - Agreement between the deterministic and Monte Carlo backends to ≤ 0.008 dex where both apply.
