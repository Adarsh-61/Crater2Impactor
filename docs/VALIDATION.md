# Validation report (generated)

Generated from `results/benchmarks_v2.json` by `scripts/write_validation_doc.py`. Benchmark run 2026-11-05T00:49:35Z, script sha256 d2cbb8d07657, Python 3.11.9, NumPy 2.4.6, quick=False.

Test types:

- **L**: law recovery on laboratory data
- **T**: location-aware target test
- **M**: mathematical correctness
- **C**: in-model calibration
- **R**: robustness under misspecification
- **B**: baseline comparison
- **P**: physical validation (measured impacts)
- **W**: real-world predictive test (population/event)

Only P, W and L involve real measurements. **None of them measures per-crater accuracy for natural impactors**, because no natural lunar crater has an independently known impactor.

## P. Measured impacts (hollow, slow spacecraft)

| Event | Q | D obs (m) | D pred median [90%] (m) | log10 obs/pred | m known (kg) | m recovered median [90%] (kg) | log10 err | PIT |
|---|---|---|---|---|---|---|---|---|
| Apollo 13 S-IVB | A | 31.1 | 33.6 [18.2, 58.5] | -0.034 | 13911 | 1.07e+04 [1.6e+03, 9.2e+04] | -0.114 | 0.58 |
| Apollo 14 S-IVB | A | 32.4 | 32.9 [17.8, 57.1] | -0.007 | 14016 | 1.34e+04 [1.9e+03, 1.2e+05] | -0.021 | 0.52 |
| Apollo 16 S-IVB | B | 34.6 | 33.5 [18.1, 58.2] | +0.014 | 14100 | 1.58e+04 [2.3e+03, 1.3e+05] | +0.051 | 0.46 |
| Apollo 15 S-IVB | A | 37.0 | 32.7 [17.7, 56.8] | +0.054 | 13851 | 2.16e+04 [3.2e+03, 1.8e+05] | +0.193 | 0.36 |
| Apollo 17 S-IVB | A | 31.0 | 32.1 [17.3, 55.6] | -0.015 | 14487 | 1.3e+04 [2e+03, 1.1e+05] | -0.046 | 0.53 |
| Ranger 7 | A | 13.5 | 12.0 [7.2, 19.1] | +0.050 | 367.2 | 593 [1.1e+02, 5.4e+03] | +0.208 | 0.34 |
| Ranger 6 | B | 14.0 | 11.6 [7.0, 18.4] | +0.080 | 381 | 788 [1.5e+02, 7.3e+03] | +0.316 | 0.25 |
| Ranger 8 | A | 12.0 | 11.2 [6.7, 17.6] | +0.032 | 368.4 | 511 [96, 4.6e+03] | +0.142 | 0.37 |
| Ranger 9 | A | 14.0 | 12.3 [7.5, 19.5] | +0.058 | 368.0 | 597 [1.2e+02, 4e+03] | +0.210 | 0.31 |
| LCROSS Centaur | B | 22.0 | 21.1 [12.5, 35.5] | +0.018 | 2271.61 | 2.74e+03 [5.1e+02, 2e+04] | +0.081 | 0.42 |
| Falcon 9 second stage 2025-010D | B | 18.0 | 21.0 [12.0, 36.4] | -0.067 | 3900–4900 | 2.72e+03 [5.1e+02, 2.1e+04] | -0.209 | 0.67 |

Quality A/B (n = 11): forward median residual +0.018 dex, inside 90%: 11/11. Inverse median mass error +0.081 dex, inside 90%: 11/11.
Spread: forward RMS residual 0.045 dex, inverse RMS mass error 0.169 dex; SD of the PIT values 0.13 (0.29 for calibrated intervals; smaller means the intervals are wider than these errors need). Events no session had examined before the test (held out): 0.
Excluded: grail_a (grazing (2 deg) impact into a slope: outside point-source validity); ladee (upper bound only; low angle not given).

Caveats:
- All impactors are hollow and travelled at 2.4–2.7 km/s, outside the compact hypervelocity regime.
- n is tiny, and the intervals are wide.
- σ_mf = 0.1 dex was chosen with the Apollo 13/14/16 S-IVB craters in view. No other default parameter was set from these events, but all 11 were examined during development (they fit the opt-in SIGMA_MF_ARTIFICIAL table), so this is not a held-out test (development_use in data/validation/spacecraft_impacts.json).

## W. Population test: LROC new craters vs bolide flux × scaling

| | 7.0–8.0 | 8.0–9.0 | 9.0–10.0 | 10.0–12.0 | 12.0–14.0 | 14.0–17.0 | 17.0–40.0 | ≥40.0 | total |
|---|---|---|---|---|---|---|---|---|---|
| observed | 8 | 4 | 6 | 9 | 4 | 1 | 1 | 1 | 34 |
| predicted, lpi2007 | 11.1 | 6.7 | 4.1 | 4.7 | 2.4 | 1.7 | 2.1 | 0.2 | 33.0 |
| predicted, holsapple2003 | 10.7 | 6.6 | 4.2 | 4.7 | 2.3 | 1.8 | 2.1 | 0.3 | 32.7 |
| predicted, impacts2022 | 7.2 | 4.3 | 2.8 | 3.2 | 1.6 | 1.2 | 1.5 | 0.2 | 22.0 |

| Package | Δ log10 D, median [90%] | log10 Y_regolith (Pa), median [90%] | N≥7/N≥12 pred vs obs |
|---|---|---|---|
| lpi2007 | +0.046 [-0.095, +0.188] | 4.27 [3.11, 5.02] | 5.13 vs 4.86 |
| holsapple2003 | +0.044 [-0.094, +0.188] | 4.27 [3.11, 5.08] | 5.06 vs 4.86 |
| impacts2022 | +0.104 [-0.039, +0.259] | 3.82 [3.05, 4.47] | 4.90 vs 4.86 |

Sensitivity (LPI 2007 package):

| Variant | observed | predicted (Δ=0, F=1) | Δ median [90%] |
|---|---|---|---|
| baseline | 34 | 32.6 | +0.061 [-0.099, +0.192] |
| velocity median 15 km/s | 34 | 26.6 | +0.089 [-0.070, +0.220] |
| velocity median 22 km/s | 34 | 40.9 | +0.022 [-0.132, +0.159] |
| sigma_mf 0.05 dex | 34 | 27.1 | +0.083 [-0.076, +0.220] |
| sigma_mf 0.20 dex | 34 | 68.9 | -0.072 [-0.202, +0.088] |
| angle law q = 1 (vertical rule) | 34 | 29.6 | +0.078 [-0.085, +0.205] |
| completeness 8 m | 26 | 21.5 | +0.080 [-0.078, +0.211] |
| flux slope b0 = 0.87 | 34 | 24.9 | +0.105 [-0.055, +0.239] |
| flux slope b0 = 0.93 | 34 | 42.8 | +0.006 [-0.138, +0.146] |
| 2020 NAC survey (exposure 7.9e6, n>=16 m: 5, >=17: 4, >=18: 3, >=19: 2, >=43: 1; digitised R04) | 5 | 6.7 | -0.016 [-0.206, +0.170] |

Factor 1/4 (NEOMOD3 finds the D>10 m flux ~4x below the bolide estimate, R12 s5.4) to x2 (luminous-efficiency envelope 5-20%, R06). Fit supported for impactor diameters ~1-10 m; 7-40 m lunar craters need ~0.1-0.6 m bodies, so the law is extrapolated. Lunar flash flux at 30 g (Suggs et al. 2014, R08) is within ~30% of the extrapolation.

Exposure and counts were digitised from Speyerer et al. (2016) Fig. 2, because the Methods are paywalled. Flux and crater-size normalisation are degenerate, so this test checks consistency and constrains one combination.

## L. Law recovery on laboratory data

Holsapple crater database v1.3 (dry Flintshot sand, rho 1.8 g/cc, n=37); held-out: Ormo et al. 2024 Table 5 (Lane Mountain quartz sand, 10-502 G, CC BY 4.0).

- Recovered μ = 0.395 (90% 0.384–0.410); published dry sand 0.41.
- Recovered ν = 0.348 (90% 0.310–0.369).
- Held-out cross-laboratory gravity trend: observed 0.264, predicted 0.260; level offsets +0.068, +0.067, +0.047, +0.071 dex.

The engine's **shipped** regolith constants (not the fit above), strength ≈ 0, vertical impacts, apparent radius:

| Material | In sample? | n | mean log10 obs/pred | SD (dex) | inside 90% |
|---|---|---|---|---|---|
| Flintshot dry sand (in-sample) | yes | 37 | +0.046 | 0.026 | 37/37 |
| F-140 sand | no | 3 | -0.056 | 0.110 | 3/3 |
| KAFB desert alluvium | no | 3 | -0.034 | 0.078 | 3/3 |
| Lane Mountain quartz sand (Ormo et al. 2024) | no | 4 | -0.003 | 0.013 | 4/4 |

All held-out granular shots: mean -0.028 dex, RMS 0.070 dex, inside 90%: 10/10. The level offset of the Flintshot-only fit above (about +0.06 dex on the Ormö shots) is a property of that fit, not of the engine.

- The database's code-41 sand mixes two target states (1.8 and 2.1 g/cc); mixing them biases mu to 0.35 and breaks extrapolation. Only the 1.8 g/cc state is used.
- An unconstrained raw-variable fit recovers the time-balance dimensional constraint (c_U + 2 c_g = -0.06, 90% CI -0.16 to +0.12) without being told it.
- Out-of-range tests: power laws extrapolate across gravity and projectile density with 0.07-0.14 dex error; gradient boosting and an MLP fail by 0.4-1.6 dex.

## T. Location test

Leave-one-out log-RMSE of local regolith thickness (54 measurements): interpolated map 0.451; mare/highland mean 0.366; global mean 0.366. Decision: Use observed mare fraction; use unit-level regolith prior; do not use the interpolated thickness map (it does not beat a constant in LOO).

## M. Mathematical correctness

| Crater (m) | fiber q05/q50/q95 | independent | max diff (dex) |
|---|---|---|---|
| 10.0 | 0.35 / 0.99 / 1.66 | 0.35 / 0.99 / 1.66 | 0.004 |
| 120.0 | 3.56 / 4.29 / 5.10 | 3.55 / 4.29 / 5.10 | 0.005 |
| 1500.0 | 7.37 / 8.08 / 8.83 | 7.37 / 8.08 / 8.83 | 0.004 |
| 30.0 | 1.39 / 2.41 / 3.23 | 1.39 / 2.41 / 3.23 | 0.006 |
| 300.0 | 4.56 / 5.37 / 6.18 | 4.56 / 5.37 / 6.17 | 0.007 |

## C. In-model calibration

Importance-sampling draws per crater: 30,000.

| Truth set | quantity | 90% coverage (±MC se) | 68% coverage | envelope coverage | median 90% width (dex) | n |
|---|---|---|---|---|---|---|
| truth package lpi2007, engine averages all three | log10_mass_kg | 0.932 ± 0.022 | 0.684 | 0.955 | 1.56 | 133 |
| truth package lpi2007, engine averages all three | log10_energy_j | 0.932 ± 0.022 | 0.692 | 0.940 | 1.48 | 133 |
| truth package lpi2007, engine averages all three | log10_velocity_m_s | 0.895 ± 0.027 | 0.677 | 0.902 | 0.50 | 133 |
| truth package holsapple2003, engine averages all three | log10_mass_kg | 0.932 ± 0.022 | 0.684 | 0.947 | 1.56 | 133 |
| truth package holsapple2003, engine averages all three | log10_energy_j | 0.917 ± 0.024 | 0.669 | 0.925 | 1.48 | 133 |
| truth package holsapple2003, engine averages all three | log10_velocity_m_s | 0.902 ± 0.026 | 0.692 | 0.910 | 0.50 | 133 |
| truth package impacts2022, engine averages all three | log10_mass_kg | 0.887 ± 0.027 | 0.602 | 0.940 | 1.56 | 133 |
| truth package impacts2022, engine averages all three | log10_energy_j | 0.887 ± 0.027 | 0.647 | 0.902 | 1.48 | 133 |
| truth package impacts2022, engine averages all three | log10_velocity_m_s | 0.917 ± 0.024 | 0.654 | 0.932 | 0.50 | 133 |

## R. Robustness under misspecification

Importance-sampling draws per crater: 30,000.

| Truth set | quantity | 90% coverage (±MC se) | 68% coverage | envelope coverage | median 90% width (dex) | n |
|---|---|---|---|---|---|---|
| angle law q=1.3 (steeper than declared 0.5-1.0) | log10_mass_kg | 0.908 ± 0.018 | 0.688 | 0.940 | 1.56 | 250 |
| angle law q=1.3 (steeper than declared 0.5-1.0) | log10_energy_j | 0.900 ± 0.019 | 0.688 | 0.924 | 1.48 | 250 |
| angle law q=1.3 (steeper than declared 0.5-1.0) | log10_velocity_m_s | 0.928 ± 0.016 | 0.736 | 0.932 | 0.50 | 250 |
| model-form residual 0.2 dex (declared 0.1) | log10_mass_kg | 0.444 ± 0.031 | 0.288 | 0.508 | 1.56 | 250 |
| model-form residual 0.2 dex (declared 0.1) | log10_energy_j | 0.452 ± 0.031 | 0.256 | 0.492 | 1.48 | 250 |
| model-form residual 0.2 dex (declared 0.1) | log10_velocity_m_s | 0.908 ± 0.018 | 0.656 | 0.908 | 0.50 | 250 |
| 30% secondaries, engine assumes primaries only | log10_mass_kg | 0.640 ± 0.030 | 0.488 | 0.652 | 1.56 | 250 |
| 30% secondaries, engine assumes primaries only | log10_energy_j | 0.928 ± 0.016 | 0.736 | 0.960 | 1.49 | 250 |
| 30% secondaries, engine assumes primaries only | log10_velocity_m_s | 0.608 ± 0.031 | 0.416 | 0.608 | 0.50 | 250 |
| 30% secondaries, engine told P(secondary)=0.3 | log10_mass_kg | 0.868 ± 0.021 | 0.660 | 0.952 | 3.65 | 250 |
| 30% secondaries, engine told P(secondary)=0.3 | log10_energy_j | 0.948 ± 0.014 | 0.764 | 0.964 | 1.81 | 250 |
| 30% secondaries, engine told P(secondary)=0.3 | log10_velocity_m_s | 0.904 ± 0.019 | 0.660 | 0.960 | 2.06 | 250 |
| meteoroid population, engine assumes asteroidal | log10_mass_kg | 0.888 ± 0.020 | 0.672 | 0.912 | 1.56 | 250 |
| meteoroid population, engine assumes asteroidal | log10_energy_j | 0.896 ± 0.019 | 0.716 | 0.928 | 1.48 | 250 |
| meteoroid population, engine assumes asteroidal | log10_velocity_m_s | 0.836 ± 0.023 | 0.592 | 0.836 | 0.50 | 250 |

## B. Baselines (common synthetic test set; truth drawn from the declared model set)

| Method | quantity | 90% coverage | median abs error (dex) | median 90% width | interval score | time (s) |
|---|---|---|---|---|---|---|
| v2 engine (3 packages, global target) | mass | 0.897 | 0.338 | 1.56 | 2.09 | 84.6 |
| v2 engine (3 packages, global target) | energy | 0.873 | 0.304 | 1.48 | 1.97 | 84.6 |
| v2 engine (3 packages, global target) | velocity | 0.893 | 0.106 | 0.50 | 0.61 | 84.6 |
| prior only (ignores crater) | velocity | 0.897 | 0.111 | 0.50 | 0.68 | 0.0 |
| LPI-style point inverse (U=20 km/s, 45 deg, nominal target) | mass | 0.000 | 0.351 | 0.00 | 8.44 | 0.0 |

n = 300, 30,000 importance-sampling draws per crater. Because the synthetic truth comes from the engine's own declared model set, this comparison favours the engine by construction; it is not evidence about real craters. The time column is wall-clock for the whole set and depends on machine load; the clean sequential per-crater timing is in results/comparison_original_vs_v2.json (runtime block). Velocity is never better than the prior-only baseline.

