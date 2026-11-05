# Crater2Impactor: scientific formulation

This document describes the system as implemented in `src/c2i/engine/` and served by `app/api_v2.py`. Evidence for each design decision is recorded in the project's research record, which is not part of this repository; `docs/VALIDATION.md` (generated from `results/benchmarks_v2.json`) holds the validation results.

Status labels used below: **[V]** verified fact (reproduced or read from a primary source); **[E]** reproducible experimental finding of this project; **[L]** supported by external literature; **[I]** reasoned interpretation; **[H]** untested hypothesis; **[F]** proposed future work.

---

## 1. Questions the system answers

Three questions have different mathematical structures and data, so they are handled separately and never merged into one accuracy number.

| Level | Question | Data | What can be answered |
|---|---|---|---|
| **Crater** | Which impactors, from a declared population and target, are consistent with this crater? | Rim diameter (plus optional location) | A posterior for mass, energy, momentum and size, conditional on stated assumptions. A robust range across assumptions. Explicit labels where the crater carries no information (velocity, angle, density). |
| **Event** | What hit here, given independent records of the same event? | Diameter plus known speed, mass, angle, impactor density or impact-flash energy | A posterior tightened by the extra records. Each record has its own uncertainty, such as the published luminous-efficiency range. |
| **Population** | Do crater scaling and an independent impactor flux reproduce observed crater production? What do the counts constrain? | New-crater counts with survey exposure; bolide flux | One normalisation, for example a crater-size offset or an effective regolith strength. Not individual impactors. |

## 2. Symbols and units (SI unless stated)

| Symbol | Meaning | Unit |
|---|---|---|
| D | final rim-to-rim crater diameter | m |
| m, δ, a | impactor mass, bulk density, equal-volume radius a = (3m/4πδ)^(1/3) | kg, kg m⁻³, m |
| U, θ | impact speed; elevation angle above the local horizontal | m s⁻¹, deg |
| ρ_t, Y | target bulk density; effective cratering strength | kg m⁻³, Pa |
| g | surface gravity (Moon 1.62) | m s⁻² |
| K1, K2, μ, ν | scaling constants (coefficient, strength coefficient, velocity exponent, density exponent) | – |
| K_r | excavation-radius factor, R_exc = K_r V^(1/3) | – |
| q | angle-law exponent, U_eff = U sin(θ)^q | – |
| h | regolith thickness at the site | m |
| π2, π3, π_V | g a/U_eff², Y/(ρ_t U_eff²), ρ_t V/m | – |
| σ_meas, σ_mf | diameter measurement error; point-source model-form residual | dex (log10) |

## 3. Forward model

### 3.1 Point-source scaling [V][L]
Holsapple's Eq. 7. It is the same form in the LPI 2007 theory document  and in IMPACTS 2022 (§2):

  π_V = K1 · { π2 · r^((6ν−2−μ)/(3μ)) + [K2 · π3 · r^((6ν−2)/(3μ))]^((2+μ)/2) }^(−3μ/(2+μ)),  r = ρ_t/δ

  V = π_V m/ρ_t,  D = 1.3 · 2 K_r · V^(1/3).

**K2 placement [V].** In Eq. 7 the coefficient K2 is inside the bracket raised to (2+μ)/2 (theory document). The LPI 2007 JavaScript multiplies K2 outside the power (`k1*Math.pow(pi2+k2*Math.pow(pis,power2),power1)`). `lpi2007` keeps the JavaScript form; `holsapple2003` follows Eq. 7. Earlier versions of `holsapple2003` used the JavaScript form while citing Eq. 7, which changed strength-regime diameters by up to 0.04 dex (BUG_INVENTORY B03). For IMPACTS, K2 = 1 and the placement does not matter.

### 3.2 Named scaling packages [V]
Constants are transcribed from the primary sources and are never mixed across packages (`engine/packages.py`).

| Package | Regolith K1 / K2 / μ | Soft (fractured) rock | Hard (competent) rock | Density term | Strength | Source |
|---|---|---|---|---|---|---|
| `lpi2007` | 0.132 / 0.26 / 0.41 | 0.07 / 0.215 / 0.55 | 0.095 / 0.257 / 0.55 | none (as coded) | constant | LPI calculator v1.4 JS. Reproduces 405 live-page cases to 6×10⁻⁴ dex (exact cube root) and to 9×10⁻¹⁶ when LPI's truncated V^0.3333 is emulated (test) |
| `holsapple2003` | 0.132 / 0.26 / 0.41 | 0.095 / 0.215 / 0.55 | 0.095 / 0.257 / 0.55 | ν (default 1/3) | constant | theory document Eq. 7, Table 1 |
| `impacts2022` | 0.15 / 1 / 0.40 | 0.04 / 1 / 0.55 | 0.06 / 1 / 0.55 | ν | rocks: Y = Y₀(10 cm/D)^(1/4) | IMPACTS (2022) Table 1, §2 |

- **Default:** results are averaged with equal weight over all three packages.
- **Envelope:** the range across packages is reported separately as part of the robustness envelope.
- **Choosing a package:** the user can select a single package.

### 3.3 Model-form terms carried as uncertainties [L][E]
- **Angle law, q ∈ [0.5, 1.0] (uniform).**
  - q = 1 is the vertical-velocity rule used by LPI and IMPACTS. Hydrocode results imply weaker angle dependence: Davison & Collins 2022 find α_D = 0.18–0.23, while Elbeshausen et al. 2009 give α_V ≈ 1.
  - The project's earlier fixed q = 0.58 lies inside this range, but its source is not recorded (PHYSICS_MODIFICATIONS M4).
  - Impacts below 15° are flagged as outside the point-source law.
- **Density exponent, ν ∈ [0.2, 0.4].** This range is from the experiments quoted by Holsapple (2022). This project's laboratory recovery gives ν = 0.35, 90% interval 0.31–0.37 (block L).
- **Point-source residual, σ_mf = 0.10 dex (default).** This is the single most influential assumption for the mass (§7).
  - Measured spacecraft impacts (n = 5). Treated as independent, they give σ_mf median 0.04 dex, 90% interval 0.00 to 0.17 dex. Three of the five are Apollo S-IVBs of one design hitting mare regolith, so their model errors are unlikely to be independent. If they share one model error, the 90% interval widens to 0.004 to 0.33 dex (median 0.06); with Ranger 7 added it is 0.004 to 0.24 dex. Without the S-IVBs (LCROSS and Falcon 9 only) it is 0.01 to 0.43 dex. The Apollo S-IVB craters informed the original 0.10 choice.
  - Laboratory scatter: 0.025 dex within one sand; 0.06 dex between sands (E14b). Across 13 laboratory materials, the leave-one-out scatter of a point-source fit within one material, converted to diameter, is 0.016 to 0.082 dex (median 0.032). This is the scatter under controlled conditions with fitted constants, a floor rather than a lunar value.
  - Held-out transfer of the **shipped** regolith constants: on three granular materials they were not fitted to (F-140 sand, KAFB alluvium, the Ormö et al. 2024 centrifuge sand; 10 shots) the apparent radius is predicted with mean −0.03 dex and RMS 0.07 dex, 10/10 inside the 90% band. Within one laboratory series the scatter is 0.07 dex in dry sand. Every measured transfer error (laboratory 0.07, spacecraft forward RMS ≈ 0.05) is below 0.10 dex; none of them involves a compact impactor at natural speed.
  - Flash–crater pairs carry almost no information on σ_mf, because the luminous efficiency, the photometric conversion and the speed dominate their spread (posterior 0.02 to 0.42 dex from a flat prior).
  - More events help only if they are independent. With a true σ_mf of 0.10 dex and 0.08 dex of other spread per event, 20 independent events give a 90% interval about 0.10 dex wide. The same 20 events from 4 or 5 vehicle designs give 0.23 to 0.29 dex, no better than 5 independent events. Per-crater diameters of further spacecraft craters (Apollo 15 and 17 S-IVB; Rangers 6, 8 and 9) are in Plescia et al. (2016), which could not be accessed. LROC captions give rounded diameters for them, and JPL mission reports give the Ranger speeds and angles. With these 11 events, σ_mf fitted from real data has a median of about 0.03 to 0.04 dex. It predicts held-out craters better than 0.10 (leave-one-out +2 to +3 nats) and is stored as an opt-in table for artificial impactors (`engine/calibration.py`; §13).
  - The population test is sensitive to σ_mf through the steep size distribution (block W sensitivity). It is therefore reported, not tuned.
  - **Status: assumption.** The data are consistent with 0.10 dex but cannot establish it; 0.10 is inside every interval above and above the laboratory floor. It is kept as the default and the user can change it.
- **Simple craters only, D ≲ 10 km.** Complex-crater collapse (the project's earlier Croft branch) is outside the validated domain and is flagged.
- **Layered targets (4 < D/h < 20): an extra 0.1 dex scatter**, in the inverse and  in the forward prediction, so both use one generative model.

### 3.4 What the forward model does *not* predict for real craters [V][L]
- **Depth:** K_d V^(1/3) is the excavation depth, giving d/D = 0.21. Fresh lunar craters below 400 m have d/D = 0.11–0.17.
- **Rim height** and **melt volume**: no calibrated observation model exists for either.
- **Ejecta radius:** the project's coefficients are unsourced (M7).

These quantities are therefore never used as inverse observations (§6).

## 4. Target model: what the crater forms in

### 4.1 Classes and priors [L]
Each target class has its own density and strength priors (`engine/targets.py`, sources cited in that file).

| Class | ρ_t (kg m⁻³) | Y (Pa, log-uniform) | Source summary |
|---|---|---|---|
| Regolith | 1500–1900 | 10³–2×10⁵ | Lunar Sourcebook ch. 9 densities. Strength is unmeasured at crater scale: cohesion ~1 kPa up to Holsapple's 0.18 MPa "dry soil" |
| Fractured / porous rock (highland crust) | 2300–2900 | 10⁵–10⁷ | GRAIL bulk density and porosity (Wahl 2020; Wieczorek 2013) |
| Competent rock (mare basalt) | 2800–3350 | 10⁶–2×10⁷ | Basalt grain density 3270–3460 (Kiefer 2012) with 2–15% porosity; Holsapple hard rock |

### 4.2 Size-dependent regime [L]
A crater sees the regolith only while D/h is small:

- **D/h ≤ 4:** regolith scaling. Normal bowl craters; the flat-floored onset is at D/h ≈ 4.3 for 1–12 km/s (Prieur et al. 2018).
- **D/h ≥ 20:** substrate scaling. Above this the concentric-crater range ends (Quaide & Oberbeck 1968 via Prieur 2018).
- **Between 4 and 20:** a log-linear mixture with an extra 0.1 dex scatter, because no validated layered-target law exists. Rajšić et al. (2026) put the onset of rocky ejecta at D/h ≈ 12, which falls inside this band.

**How the class is decided, per impactor [V].** With P_reg(x) the log-linear rule above and u ~ U(0, 1) a latent threshold per event: the crater forms in the regolith if u < P_reg(D_reg/h), where D_reg is the crater the impactor would make in regolith (it starts in the regolith and stays there while that crater is shallow). Otherwise it forms in the substrate if u ≥ P_reg(D_sub/h). When neither branch is self-consistent it lies at the switch size D*, where P_reg(D*/h) = u, between the two branch sizes. Forward prediction (`forward._consistent_rim`), the benchmark truth generator and the inverse sampler all use this one rule.
- *Earlier versions:* the inverse drew the class from the **observed** D/h, independently of the impactor. Its likelihood p(D | impactor) then did not integrate to one, which tilted the effective mass prior. For D/h ≈ 4–20 the inverse gave median masses 0.2–0.43 dex below the forward model's (D = 30 m: holsapple2003 1.91 vs 2.34, lpi2007 2.14 vs 2.37; D ≥ 300 m: no difference).
- *Exact inverse:* three components per draw. **R** (regolith fibre) has weight P_reg(s/h). **S** (substrate fibre) has weight 1 − max(P_reg(D_reg(m)/h), P_reg(s/h)). **N** (at the switch) has density 1/log10 5 per unit log D, with m uniform between the two fibre solutions. Benchmark block M checks it against an independent sampler of the forward model.
- *Cost:* the deterministic SRC backend (§7) implements only the uncoupled rule. It is therefore used only where the two rules coincide: a fixed target class, or craters far outside the band for every plausible regolith thickness (below ~3 cm or above ~100 km). Everywhere else the default is the Monte Carlo sampler: about 0.4 s per crater, mass quantile seed SD ≲ 0.01 dex at 30,000 draws.

### 4.3 What location contributes [V][E]
- **Used:** mare versus highland at the site.
  - It comes from the LROC mare boundaries (Nelson et al. 2014), rasterised at 0.5° (`scripts/build_lunar_context.py`). It is an observed quantity, and it selects the substrate.
  - Outside 65°S–65°N the site is treated as highland.
- **Not used:** a regolith-thickness map.
  - Leave-one-out tests on 54 published measurements gave log-RMSE 0.45 for an interpolated map, against 0.37 for a constant (block T).
  - Thickness priors are therefore by unit, with geometric means of 4.5 m (mare) and 5.3 m (highland) and wide spreads.
- **Not mappable:** effective strength, friction, deep porosity and local layering. These stay as priors and are never shown as local measurements.

## 5. Impactor populations (declared, swappable) [L][I]

| Population | Speed | Density | Angle | Status |
|---|---|---|---|---|
| Asteroidal (NEO-derived) | truncated log-normal, median 18 km/s, σ_ln 0.35 (mean 19.1), ≥ 2.38 km/s | log-uniform 1.3–3.5 g cc⁻¹ plus 4% iron | sin 2θ | Approximates Le Feuvre & Wieczorek (2011), mean 19.7 km/s |
| Sporadic meteoroid | median 20 km/s, σ_ln 0.40 | 0.3–3.0 g cc⁻¹ | sin 2θ | Assumption for flash-sized bodies |
| Secondary | log-uniform 0.2–2.38 km/s | 2.5–3.3 g cc⁻¹ | 10–45° | **Outside validated scaling** below ~1 km/s; flagged |
| Artificial (opt-in) | log-uniform 1.6–2.9 km/s | bulk 20–1000 kg m⁻³ | uniform 5–90° | For known spacecraft or rocket-stage impacts only. Speed range from orbital mechanics (lunar-orbit decay ~1.6–1.7 km/s; arrival from cislunar space √(2.38² + v∞²) ≤ ~2.8 km/s); density from vehicle bulk densities (S-IVB 23 kg m⁻³ to LADEE 430 kg m⁻³). Never a default. Flagged: hollow, elongated bodies are outside the compact point-source calibration |

**Why spacecraft craters were a regression.** From the diameter alone, the final engine underestimates the mass of the five spacecraft impactors by 1.03 dex on average, against 0.44 dex for the original network. The decomposition shows:
- **The cause is the speed prior.** Giving the known speed alone reduces the error to 0.22 dex, and speed plus angle to 0.13 dex. The target assumption is not the cause.
- **The original network's advantage was two compensating errors.** The network reproduces the exact posterior of its own model once the original app's coordinate-based geological reweighting is included (within about 0.2 dex per event). For nearside mare sites that reweighting assumes a 30 MPa competent-basalt target for 20 to 35 m craters, which form in regolith. The stronger target raises the inferred mass and offsets the too-high speed. With the true spacecraft speeds the same model overestimates the mass by about a factor of eight.
- **The opt-in artificial population** gives 0.28 dex from the diameter alone and 0.15 dex with the angle. These events were examined before the mode was defined, so this is a development result, not validation.

**Mass prior.** Each population carries its own prior p(log₁₀ m) ∝ 10^(−b log₁₀ m):
- **asteroidal and meteoroid:** b = 0.90, the cumulative slope of the Brown et al. (2002) bolide flux (fit for 1–10 m bodies; extrapolated outside it);
- **artificial (spacecraft) and secondary:** b = 0, log-uniform. Spacecraft masses are designed; no calibrated fragment law exists for secondaries.

Why it changed:
- On Brown-law synthetic truths the former flat default overestimated mass by +0.48 dex, with 90 % coverage 0.75.
- The population default gives −0.003 dex and 0.91 (120 fresh truths; interval score 2.22 vs 3.70). A truth slope of 0.7 or 1.1 gives −0.11 / +0.16 dex and coverage 0.89 / 0.84.

Consequences:
- **Selection effect.** A crater of given size is more likely to come from a fast, steep impact, so the speed and angle posteriors are **shifted** but not narrowed. At 120 m, the speed median rises 18 → 22 km/s and mass falls 0.7 dex.
- **The pull grows with the likelihood width.** A larger σ_mf lowers the mass estimate.
- **The flat prior remains available.** Pass `mass_prior_b = 0` (`ModelForm`, API, or the Inverse-page selector).

The fraction of secondaries is a user input. It varies strongly with location and time, so no global default is justified (Bierhaus et al. 2018; Xiao & Strom 2012).

## 6. The forward–inverse relationship

**Data flow.**
- *Forward:* (m, U, θ, δ; target; package) → D.
- *Inverse:* D_obs (+ location, declared assumptions, optional event records) → posterior over (m, U, θ, δ, target).

**The measured forward output is the inverse model's input.** This is the precise, valid form of the faculty statement "the forward model's outputs should be inputs to the inverse model" (FORWARD_INVERSE.md, interpretations M4/M5). The LPI calculator's own inverse mode does exactly this:
- it takes the crater diameter;
- it requires U, θ, δ and the target to be stated;
- it solves for the impactor size.

Every other forward input is either **declared**, as LPI does, or **integrated over** with an explicit prior, as this engine does. The inverse never takes:
- **Forward-model internals** (π2, π_V, regime codes, transient volume). These are functions of the unknowns, so supplying them leaks the answer: in a dedicated test they recover the unknown with R² of 0.70–0.996.
- **Deterministic transformations of D**, such as model depth or rim height. These add no information (R² for velocity = 0.000).
- **Measured depth or rim height.** The model's fixed d/D disagrees with real fresh craters, so these inputs bias the mass by −0.04 to −0.13 dex.

**Key relation, mass given velocity.** At fixed crater and target the posterior lies on a curve d log m / d log U ≈ −3μ, i.e. −1.23 in regolith and −1.65 in rock. The UI shows this band.

- **What it means:** "if the speed were known, the mass would be here."
- **Equivalence to LPI:** it is the probabilistic version of the LPI inverse.
- **What it does not do:** the crater fixes this relation, not the speed.

## 7. Direct inference (no surrogate) [V][E]

**Sampler.** For each scenario (one population × one scaling package):
1. Draw the noise-free log D, s ~ N(log D_obs, σ_meas² + σ_mf² [+ 0.1² if the noise-free crater is in the layered band]). Under a power-law mass prior (b > 0) the draw is tilted towards smaller s by σ² b ln10 / 0.3 and corrected by the exact density ratio; the shift may reach the crater size made at the lower mass bound, so wide crater-size likelihoods (σ_meas ~ 0.5–1 dex, posterior up to ~3 dex below D_obs) keep an ESS above 10⁴.
2. Draw the regolith/substrate/switch component of §4.2 conditional on s (probability P_reg(10^s/h) for the regolith branch), then the nuisances (U, θ, δ, ρ_t, Y, h, q, ν) from their priors, from kinematically tilted priors, or from event-informed proposals with exact prior-density correction:
   - a measured speed or angle: the measurement density truncated to the population's support, weighted by the prior density and the measurement's probability mass inside the support;
   - rocky ejecta observed or absent: half of the regolith-thickness draws from the exact prior × likelihood density of log h;
   - a flash energy: log10 η integrated analytically over its prior and drawn from its exact conditional.
3. Solve log m on the fiber s(log m, nuisances) = s by Newton iteration (for both the regolith and the substrate branch; the class weights of §4.2 then select between them). s is strictly increasing with slope in [2/(3(2+μ)), 1/3], so the solution is unique.
4. Weight by p(log m) · |∂ log m/∂s| times the prior/proposal ratios and the event likelihoods.

Every proposal choice changes only the efficiency; the target posterior is the same. Release 2.1 changed steps 1–2 after reproduced cases where the earlier proposals left fewer than 100 effective draws: σ_meas ≥ 0.55 dex on a 5 km crater; a measured speed outside the population support; no rocky ejecta on a 500 m crater (`docs/RELEASE_AUDIT.md`, `INTERNAL_ONLY/tests/test_sampler_robustness.py`).

This is importance sampling in a transparent reparameterisation: the parameters split into the combination the crater identifies and the directions it leaves to the prior, in the sense of Gustafson (2005, Statistical Science 20, 111–140; 2015, Bayesian Inference for Partially Identified Models).

**Deterministic backend (`src/c2i/engine/src_backend.py`; default `backend="auto"`).** The log10-mass posterior, the target-class probabilities, the scenario weights and  the energy and momentum posteriors come from a deterministic computation; the Monte Carlo sampler above supplies the remaining quantities (speed, angle, density, strength) and is the fallback. **SRC applies only where the per-impactor target-class rule (§4.2) reduces to its uncoupled form** (fixed class, or craters far outside the layered band); `auto` otherwise uses the sampler, which inverts the coupled rule exactly.
- *Symmetry reduction [V].* At fixed crater size the rim law depends on target density and strength only through ψ = Y ρ_ref/ρ_t (lpi2007: ψ = Y (δ/ρ_t)^(1/3) ρ_ref/ρ_t), and log10 m = x(s, ψ) + target-density shift − 3μ log10 U_eff + (1 − 3ν) log10(δ/ρ_t). The mass posterior is therefore an exact convolution of a target pmf with line-supported kinematic pmfs on a 0.005 dex grid, multiplied by the mass prior 10^(−b log10 m). No random numbers are used.
- *Strength-scale symmetry [V] :* x0(s, ψ) = x0(s − l, ψ_ref) + 1.5(2 + μ) l, so one fibre solve per class serves every strength (verified to 1e-14).
- *Evidence [V].* The diameter-only evidence used in the crater-level scenario weights is computed in the same pass by a Laplace-type identity (exact to 7e-15). Inputs SRC cannot represent exactly (η tables, grid overflow, events outside the prior support) raise `SRCUnsupported` and fall back to Monte Carlo.
- *Accuracy [E].* Agreement with 2 × 2M-draw Monte Carlo references ≤ 0.004 dex in quantiles over more than 50 configurations.
- *Crater-size window [V] (2.1).* The mass prior moves the posterior of s below log D_obs by σ² b ln10 / slope. Up to release 2.0 the s nodes stopped 5σ below log D_obs, which cut the posterior for σ_meas ≥ 0.3 dex (fixed-class sites, q05 up to 0.95 dex high while labelled reliable). The window now reaches 5σ plus 4 b ln10 σ² below, floored at the mass support. Agreement with Monte Carlo at σ_meas 0.3–0.5 is ≤ 0.008 dex.

**Identities used as checks [V].**
- *Fixed-D invariance:* the crater constrains only m·U_eff^(3μ) (times a density term), so with a diameter alone the kinematic posterior equals its prior (posterior-independence theorem, verified to 1e-4).
- *μ–α gauge:* if the luminous efficiency scales as U^α, flash–crater pairs identify only α − 3μ; neither μ nor α separately.

**Correctness.**
- It matches an independent prior-sampling importance sampler to ≤ 0.007 dex (block M; `INTERNAL_ONLY/tests/test_engine_v2.py`).
- The `lpi2007` kernel reproduces the project's `legacy_2007` simulator to 10⁻¹². `holsapple2003` reproduces `production_v1` to 10⁻⁹ when the strength term vanishes; the two differ in the strength regime only by the K2 placement (§3.1).

**Cost.** Median 0.19 s per crater for three scenarios × 30,000 draws (the web default), against 1.30 s for the previous network-based application on the same machine, both run one after another with nothing else running (`results/comparison_original_vs_v2.json`, runtime block). No training is needed. That figure is for the Monte Carlo path (`backend="mc"`); the deterministic default takes about 0.4 s, a cost accepted for exact, seed-free mass quantiles.

**Scenario averaging [V].** Each scenario (population × scaling package) receives the posterior weight prior weight × evidence. The evidence is the importance-sampling estimate of p(data | scenario), with all prior densities normalised. A scenario that cannot produce the data (for example the asteroidal population when a measured speed is below lunar escape speed) receives zero weight instead of stopping the whole calculation. **The declared weights are crater-level.** They are the prior fractions of craters of the observed size, and only observations beyond the diameter re-weight the scenarios: weight ∝ prior × Z_full / Z_diameter-only.
- With a crater diameter alone, the weights equal their prior values **exactly**.
- With independent measurements (speed, angle, ejecta pattern, flash) they can change strongly.
- Earlier, p(D | scenario) entered the weight. Under a power-law mass prior that penalised any scenario needing more mass for the same crater, moving package weights 0.33/0.34/0.33 → 0.33/0.43/0.24. The posterior weights are shown on the Inverse page.

**Outputs**, for every quantity:
- the median and 68% and 90% intervals of the evidence-weighted mixture;
- the **envelope**: the minimum to maximum of the per-scenario 90% bounds, over scenarios with posterior weight of at least 1%;
- a **status**:
  - *prior_dominated*: posterior/prior width ratio > 0.85 and information < 0.15 bits;
  - *weakly_constrained*: width ratio 0.5–0.85;
  - *assumption_dependent*: envelope width more than 1.6 times the median within-scenario width;
  - *constrained* or *well_constrained*: 90% width ≤ 0.3 dex.
- **Reference priors for the labels.**
  - For impactor quantities (mass, speed, angle, density, energy) the reference is the prior mixture over scenarios with their prior weights. The labels therefore also count information about which population produced the crater.
  - For target strength and target density the reference is the class-conditional prior mixed with the posterior class probabilities. These labels measure information within a target class; information about the class itself is reported as class probabilities.
- **Reliability.** A result is marked *unreliable*, with the reason stated, when:
  - the effective sample size of a scenario with posterior weight ≥ 1 % is below 100; or
  - more than 1 % of the posterior lies within 0.25 dex of the numerical mass bound, so that the bound rather than the data sets the answer; or
  - an impact flash conflicts with the crater (predictive p < 10⁻³); or
  - a measured speed, angle or mass conflicts with the crater and the declared populations (predictive p < 10⁻³; §8). Such a measurement still yields a formally valid posterior pinned at the edge of the prior support, which is why it is checked separately.

Also reported for each crater:
- an **uncertainty budget** for log m: a first-order regression share for each nuisance;
- the mass-given-velocity band;
- flags: layered regime, secondaries, grazing draws, crater above 10 km, low effective sample size.

**Result for a single crater [E]** (identifiability matrix; craters of 20 m, 300 m and 3 km).
- From the diameter alone, velocity, angle, impactor density, target strength and target density come out *prior_dominated* at every size (posterior/prior width ratios 0.97 to 1.02).
- Mass, energy, momentum and impactor size are *constrained*: mass 90% widths of 1.87 to 1.91 dex for a global target.
- The largest share of the log-mass variance is the crater-size term (measurement plus σ_mf): 42 to 56% with the diameter alone. Impact angle (13 to 22%) and velocity (12 to 19%) follow.
- Knowing the speed to 10% narrows the mass 90% width only to about 1.7 to 1.8 dex; speed plus angle, to about 1.45 to 1.6 dex. With both known, the crater-size term carries 71 to 79% of the remaining variance. **The scaling law's own scatter, not the unknown speed, limits the mass once the speed is known.** Better calibration of the law (controlled impacts with known impactors) would reduce the mass uncertainty more than better speed estimates.
- An impact-flash energy narrows the energy to about 1.0 dex. An independent mass measurement narrows the impactor size to 0.2 dex but leaves the speed weakly informed (width ratio 0.88 to 0.91).

## 8. Event-level evidence

These terms are optional and enter as weights or proposals (`engine/sampler.py`):

| Record | Likelihood |
|---|---|
| Known speed | log-normal, σ given by the user (proposal: the measurement truncated to the population's speed support) |
| Known mass | Gaussian in log m |
| Known angle | Gaussian (proposal: the measurement truncated to the population's angle support) |
| Known impactor density | fixed value (overrides the population's density prior; not used in the scenario evidence) |
| **Ejecta pattern of a fresh crater** | ordered categories (symmetric, asymmetric without a gap, uprange forbidden zone, butterfly) with uncertain elevation thresholds 45 ± 7.5°, 25 ± 5° and 11 ± 2.5° (Herrick & Forsberg-Taylor 2003; Gault & Wedekind 1978; Bottke et al. 2000), plus an assumed 5% misclassification rate (§8a) |
| **Impact flash** | log E_lum = log(η · ½mU²) − log F + ε, with η log-uniform over 5×10⁻⁴–6×10⁻³ (the published spread) applied to the *luminous-range integrated* energy (whole flash, 0.5 µm, f = 2). F converts other published definitions: 1.6 to 16 for a single brightest-frame R-band energy (NASA MSFC), 1.4 to 2.4 for a 0.3 µm bandwidth (`engine/flash.py`). No η(U) law is assumed; the project's earlier fusion code assumed one (F27). |

**Conflict check for flashes [X].** When a flash energy is entered, the engine also computes where the observed luminous energy falls in the crater-only predictive distribution (removing the flash term from the exact importance log-weights). A two-sided p below 0.05 is flagged; below 10⁻³ the result is marked unreliable.

**Conflict checks for a measured speed, angle or mass [V] (2.1).** The same two-sided predictive check is applied to each of these measurements. It uses the diameter-only posterior of each scenario, mixed with the declared scenario weights, convolved with the stated measurement error (`infer._measurement_conflicts`, reported as `measurement_consistency`). Example: a secondary-only run with a measured 3.81 ± 0.37 km/s speed (secondaries stop at 2.38 km/s) gives p ≈ 3×10⁻⁹ and is marked unreliable. Before release 2.1 the same input stopped with "no admissible draws".

**Photometric definitions.** Published flash energies are not one quantity. The 17 March 2013 value (7.1×10⁶ J; Suggs et al. 2014) is the energy in the single brightest 1/30 s video frame, in the R band with a 1607 Å bandwidth, although the flash lasted about 1 s. The 11 September 2013 value (1.3×10⁸ J; Madiedo et al. 2014) is integrated over the 8.3 s light curve with a 0.5 µm bandwidth. Madiedo et al. (2019) point out that η is only meaningful for the definition it was derived with. An earlier version compared the 17 March single-frame energy directly with an η range for whole-flash energies and found mild tension (p = 0.13). With the conversion the pair is unremarkable (p = 0.81). That tension was a definitional artefact.

**Flash–crater pairs.** Both directions were run: flash to predicted crater, and crater to predicted flash energy. Values were checked against the papers listed in `data/validation/flash_crater_pairs.json`.

| Pair | Crater (m) | Flash → crater, two-sided p | Crater → flash, two-sided p | Crater-implied η (median) |
|---|---|---|---|---|
| 17 Mar 2013 (Virginid speed and angle, converted) | 18 | 0.79 | 0.81 | 1.1×10⁻³ |
| 11 Sep 2013, D 35 m (Rizos et al. 2025), sporadic | 35 | 0.27 | 0.34 | 4.1×10⁻⁴ |
| 11 Sep 2013, D 30.2 m (Sheward et al. 2025), sporadic | 30.2 | 0.48 | 0.53 | 7.2×10⁻⁴ |
| 11 Sep 2013, D 35 m, with the reported butterfly ejecta | 35 | 0.08 | 0.12 | 1.3×10⁻⁴ |
| 27 Sep 2017 (stream speed 27.6 km/s) | 23.8 | 0.15 | 0.21 | 2.6×10⁻⁴ |
| 27 Sep 2017 with the reported uprange gap | 23.8 | 0.06 | 0.11 | 1.5×10⁻⁴ |

- **No pair is inconsistent** (p ≥ 0.05). The two published diameters of the 11 September crater (35 ± 0.7 m and 30.2 ± 0.9 m) differ by 4.3σ; both were run.
- **All three craters are larger than their flashes predict**, so each implies η in the lower half of the assumed range. Combined as one common η, the three pairs give 5×10⁻⁴ (90%: 1.5×10⁻⁴ to 1.7×10⁻³), and they are mutually consistent (leave-one-out p 0.42 to 0.88; `eta_joint.json`). η ≥ 3×10⁻³ has 1% posterior probability. This is model-dependent: a lower speed or a larger scaling coefficient would have the same effect.
- **Not independent of the flash model.** The upper η bound 6×10⁻³ comes from Sheward et al. (2025), whose fit used these pairs, apparently with energies in different definitions. **Status: unverified, under review.** In a reconstruction of their procedure the value is driven by two craters below 5 m (C180905, with a blackbody-integrated energy and the smallest diameter error; C190121, selected using η) and by the choice of scaling law; their framework applied to the three in-domain pairs gives 6.4×10⁻⁴ to 2.2×10⁻³. The η range was not changed in response to these results (pre-registered). The joint estimate is a reason to revisit it with new pairs.
- The 21 January 2019 flash has two candidate craters. The model strongly disfavours the 3.5 m candidate that Sheward et al. chose by an η argument (p = 0.004), and fits the 12 m candidate (p = 0.79). The identification is model-based, so this is a sensitivity result only.

Seismic momentum is not implemented. No lunar seismic network currently records new impacts.

### 8a. The impact angle: input, output, or neither? [V][E][L]

This answers the faculty question whether the angle should be an output.

- **From the diameter alone the angle is not identifiable.** The angle enters only through U_eff = U sin(θ)^q, which trades off exactly against speed and mass. The posterior equals the isotropic prior (width ratio 1.00 to 1.02 at every size). Training a network to "predict" the angle from a diameter cannot change this; it would reproduce the prior.
- **It becomes an output when the crater carries angle-dependent morphology.** Fresh craters keep an ejecta pattern that changes with the angle. The engine now uses it as an observation (`engine/observations.py`). For a 300 m crater, an uprange forbidden zone gives an angle from vertical of 48 to 79° (90%), asymmetric ejecta without a gap 38 to 67°, and symmetric ejecta 9 to 49°. These are width ratios of 0.46 to 0.64, so the angle is *constrained* or *weakly constrained*. Butterfly ejecta is rare (about 4% of impacts), so the assumed misclassification rate dilutes it to weak evidence.
- **It is an input when measured independently** (trajectory, as for spacecraft). It then enters as a Gaussian measurement.
- **Rim elongation is not used.** In the Robbins (2018) catalogue, 29% of well-traced 1 to 10 km craters are confidently elongated, against about 4% expected from impact angle under isotropic impacts. Most elongated rims are not low-angle impacts.
- **Limits.**
  - The thresholds describe compact natural impactors. The Apollo S-IVB craters are elongated at 69 to 79° elevation, so impactor shape dominates for spacecraft.
  - The thresholds rise for very large impactors (Luo et al. 2022), outside the simple-crater domain.
  - Falcon 9 (18 m, 59° from vertical, uprange gap): the first version of the model gave 60.2 to 78.5° and missed it by 1.2°. The misclassification term, added afterwards and motivated by the Robbins result, gives 52.5 to 78.6°. Falcon 9 is therefore development data, not a blind test.
  - **Independent check, pre-registered: failed.** The 17 March 2013 flash crater has an angle known from fireball orbits (56° above the horizon, 34° from vertical; Suggs et al. 2014), conditional on Virginid membership. Its rays are described as asymmetric with no forbidden zone (Robinson et al. 2014), which is class `asymmetric` by the rule fixed beforehand. The engine gives 39 to 67° from vertical (90%), which excludes 34° by about 5°. The model gives that pattern only 8% probability at the true angle.
  - **Sensitivity.** The interval would contain the true angle with a misclassification rate of 15% or more, doubled threshold spreads, or an asymmetry onset at 60°. None of these was adopted, because that would be tuning to this event. The same changes widen the Falcon 9 interval.
  - **Classification is itself uncertain.** Sheward et al. (2022) call a pattern with only an uprange gap "butterfly", and the 17 March crater shows opposite proximal and distal asymmetries. The 5% misclassification rate is an assumption that no data calibrate.
  - **Status: provisional.** Two checks, both close to the edge or outside the 90% interval, suggest the angle posterior is overconfident. The engine flags every angle derived from an ejecta pattern accordingly. A calibrated model needs a set of fresh craters with trajectory-known angles and independently classified ejecta.

## 9. Population-level calibration [E]

**Expected counts.** For crater bins b,

  N_b = F · A·T · ∫ (dN/dm)_Moon · P(D ∈ b | m) dm,

where:
- the flux is Brown et al. (2002), N(>E) = 3.70 E_kt^(−0.90) per year at Earth. It is converted to mass at 20.3 km/s and transferred per unit area with Earth/Moon = 1.58 (Le Feuvre & Wieczorek 2011).
- P(D | m) comes from the same engine, with all its nuisances;
- F is the flux normalisation, with prior range 1/4 to 2.

**Data.** LROC NAC temporal pairs (Speyerer et al. 2016). Exposure 3.32×10⁶ km² yr; 34 craters ≥ 7 m (digitised; block W).

**Inference.** A Poisson likelihood over size bins gives the posterior on either:
- a crater-size offset Δ; or
- the effective regolith strength.

Results and sensitivities are in VALIDATION.md. Flux and scaling are degenerate: a flux factor F is equivalent to a size factor F^(1/2.7). The test therefore checks consistency and constrains one combination. It cannot calibrate both.

## 10. Is learning necessary? [E]

**Learning the inverse map: not justified.**
- The posterior of the stated model can be computed directly and cheaply by importance sampling (§7); the only approximation is the finite sample. A learned posterior with the same inputs and priors can at best equal it (H14).
- Under its own training priors, the previous network matched the directly computed posterior (median |Δ| ≤ 0.03 dex), but its priors and inputs were fixed at training time. The head-to-head comparison on shared cases is in `results/comparison_original_vs_v2.json`.
- Amortised neural simulation-based inference is fragile under misspecification (Kelly et al. 2025 review).

**Learning or discovering the scaling law: possible only from controlled data.** Tests on the Holsapple laboratory database :
- In dry sand, speed and gravity enter only through π2 as point-source similarity requires (speed/gravity exponent ratio −1.90, 90% −2.25 to −1.66).
- An apparent excess dependence on projectile size (1.37 × the gravity exponent) is confounded with projectile material and gives no out-of-sample gain; it is not claimed as new physics.
- Crater depth/radius in sand shows no speed dependence (coefficient −0.06, 90% −0.16 to +0.05), so crater shape is not a usable speed signal for granular targets.

- On laboratory craters the exponents are recovered: μ = 0.395 versus 0.41 published (block L).
- The recovered gravity dependence transfers to an independent laboratory, but the coefficient does not (E14b).
- Flexible learners (gradient boosting, MLP) extrapolate worse than the dimensionless power law by 0.3–1.5 dex.
- Lunar craters cannot be used for this: their impactor mass and speed are unknown. At population level only one normalisation is constrained (§9).

**When a network would be justified [F].**
- For high-dimensional observations whose likelihood cannot be written down, such as images or DEMs used to classify crater origin or freshness. That classifier would set the secondary fraction.
- Only with labelled data whose provenance is known, and calibration checked on held-out regions.
- If built, it would use Keras on Kaggle with 2×T4 GPUs, as requested. No such training has been run, because the gate (labelled data) is not yet met.

## 11. Validity domain and falsifiability

**Valid for:**
- simple lunar craters, about 1 m to 10 km;
- compact impactors faster than a few km/s at angles above about 15°;
- the declared target classes;
- primaries from the declared populations.

**Flagged when outside it:**
- secondaries, or speeds below 1–3 km/s;
- glancing angles;
- layered targets (4 < D/h < 20);
- craters above 10 km;
- unsupported planetary bodies.

**What would show this work is wrong:**
1. A measured impact with known compact impactor at hypervelocity whose crater falls outside the forward 90% interval. All 11 measured lunar spacecraft impacts fall inside, with residual RMS 0.04 dex, but all are slow and hollow.
2. A larger new-crater survey whose size-frequency shape disagrees with the predicted shape. The 2016 shape matches: N(≥7)/N(≥12) is 4.9 observed and 5.0–5.2 predicted.
3. Laboratory or hydrocode evidence of a velocity-sensitive morphology signature in simple craters. That would make velocity identifiable and change the per-crater analysis.
4. Flash–crater pairs whose energies, in a stated photometric definition, are inconsistent with η in 5×10⁻⁴–6×10⁻³. Three in-domain pairs are consistent (two-sided p 0.06 to 0.81 across all variants run), but all lie in the lower half of the η range; more pairs with known definitions could falsify the flash model or the small-crater scaling.
5. Fresh craters with known trajectories whose ejecta pattern disagrees with the thresholds of §8a. One independent case so far (17 March 2013) **falls outside** the 90% interval; Falcon 9 lies at the edge. The angle-from-ejecta model is therefore provisional.

## 12. Other planetary bodies

See `docs/EXTENDING_BODIES.md`.

- **Shared across bodies:** the point-source form, the fiber sampler, the envelope logic and the population likelihood.
- **Body-specific:** gravity, the velocity distribution, the target model, atmospheric entry, the simple-to-complex transition and validation data.

Only the Moon is currently enabled. Mars and Mercury requests are refused, with the missing requirements listed in the error.

## 13. Architecture study, calibration layer and measurement advisor  [E]


- **Architecture.** Under a fixed model and fixed observations, the exact posterior is optimal for every proper scoring rule, whatever estimator is used. In a pre-registered test, a masked-input mixture-density network trained on 2 million simulations of the engine's own model matched the exact engine on seven observation sets. It was never better in CRPS by more than 2 SE (log-mass differences ≤ 0.002). A log-score edge of about 0.01 nats in a second run traced to the finite-sample density estimate of the engine's posterior; it shrinks to about 0.003 nats with 40,000 draws. The architecture is not the limit; information and calibration are.
- **Information budget (in-model).** Compared with the diameter alone:
  - **flash:** mass 90% width 1.87 → 1.29 dex;
  - **speed + flash + ejecta:** 0.97 dex;
  - **speed + angle:** 1.65 dex.
  - With every realistic channel the 4-D posterior volume (mass, speed, angle, density) shrinks 12 to 16 times. No channel informs the impactor density.
- **Calibration layer** (`engine/calibration.py`, opt-in):
  - σ_mf fitted from 11 lunar spacecraft craters: median 0.04 dex, 90% 0.002 to 0.16. It predicts held-out spacecraft craters better than 0.10 dex (leave-one-out +2 to +3 nats).
  - η fitted from 3 flash–crater pairs (+0.5 nats).
  - **Spacecraft mass recovery** (leave-one-out, 11 events): 0.13 dex MAE with speed and angle known; 0.31 to 0.33 dex from the diameter alone. 11/11 inside the 90% intervals, which are 1.7 to 1.9 dex wide, so the model is conservative for these events.
  - **Defaults are unchanged:** σ_mf 0.10, η 5×10⁻⁴ to 6×10⁻³.
- **Measurement advisor** (`engine/design.py`): the expected 90% widths if a speed, angle, flash or ejecta pattern were added. Its predictions agree with the widths actually obtained within 2% (in-model, n = 200).
