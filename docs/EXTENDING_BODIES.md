# Extending Crater2Impactor to another planetary body

Accepting a body name in the code does not make inference valid for that body. A body is enabled only when every requirement below has a sourced implementation and a validation test. Until then, the API refuses inference for it (`engine/bodies.py`) and lists what is missing.

## What carries over unchanged
- The point-source scaling form (§3.1 of SCIENCE.md) and the named scaling packages. Laboratory and hydrocode constants describe materials, not planets.
- The exact fiber sampler, envelopes, identifiability labels and uncertainty budget (`engine/sampler.py`, `engine/infer.py`).
- The population likelihood (`scripts/validation_v2.py: block_W`), given a body-specific flux and survey.

## What must be supplied for each body

| Requirement | Why | Moon (implemented) | Mars (missing) |
|---|---|---|---|
| Surface gravity | Enters π2 | 1.62 m s⁻² | 3.71 m s⁻² |
| Impact-velocity distribution | Sets the velocity prior, which drives the mass–velocity trade-off | Le Feuvre & Wieczorek 2011 approximation | Mars-specific distribution; slower than lunar (Le Feuvre & Wieczorek 2011) |
| Atmospheric entry, deceleration and fragmentation | Small impactors reach the surface slower, lighter and often broken up | None needed (airless) | Required: 58% of new Martian impacts form clusters (Daubar et al. 2022); entry modelling as in Collins et al. 2022 |
| Target model: materials, layering, porosity, volatiles | Selects K1, μ and strength | Regolith over fractured or competent rock; mare map from LROC | Ice-bearing ground, dust mantles, layered deposits (SWIM ice maps, THEMIS thermal inertia) |
| Size-dependent regime rule | Regolith versus substrate | D/h rule (Prieur 2018) with unit thickness priors | Needs Mars-specific layer thickness evidence |
| Simple-to-complex transition | Bounds the simple-crater domain | ≈ 10–15 km (flagged above 10 km) | ≈ 5–8 km |
| Validation data | At least one test that touches reality | Measured spacecraft impacts; LROC new-crater rate | InSight crater–seismic pairs (Garcia et al. 2022); new-crater catalogue (Daubar et al. 2022) |

## Procedure
1. Add a `Body` entry in `engine/bodies.py` with `inference_supported=False`.
2. Implement the population (velocity distribution) and the target model with sources. Never copy lunar parameters.
3. If the body has an atmosphere, implement entry and fragmentation as an extra stage in the forward model, then extend the fiber solver to cover it.
4. Add a physical validation block (P) and, where a survey with known exposure exists, a population block (W).
5. Set `inference_supported=True` only after those blocks pass, and record the result in `docs/VALIDATION.md`.

## Detecting unfamiliar regimes
The engine flags any case outside the validated domain:
- glancing impacts (< 15°);
- speeds below the hypervelocity range;
- layered targets;
- craters above the simple-crater limit;
- unsupported bodies.

The flags are returned with every result and shown in the UI. A flagged result should be read as "the model was evaluated outside its tested domain", not as a prediction.
