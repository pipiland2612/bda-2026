# Minh's pooled NO₂ model — proposed design

Status: approved by Minh on 21 November 2026, including Python/PyMC. Dataset readiness, numerical priors, and shared modeling choices still require the documented pre-fit checks.

## Goal and ownership

Implement Model A, the pooled Bayesian regression, and provide reproducible fitting code, checks, predictions, and explanations Minh can use in his report and presentation. The edited Google Doc assigns Minh the pooled model, Andrei the hierarchical model, and Ayreeen preprocessing and final comparison. One shared-rules sentence still names Andrei as comparison owner; use Ayreeen's explicit role for this proposal and flag the inconsistency rather than editing teammates' assignments.

Source: https://docs.google.com/document/d/1QqzZW0uwLrptKWFzwmfkygEjPjwklhlV8vzlx25HN1c/edit?tab=t.8zdat10mg9k

Read repository planning documents and remote main at `2cf6934`. Local main is three commits behind; implementation should use a separate branch/worktree based on remote main. Do not overwrite teammates' preprocessing or implement Model B or final comparison.

## Approaches

Selected by Minh: Python/PyMC lognormal regression, consistent with existing Python preparation and the log-scale ratio elicitation worksheet. Coordinate the shared specification with the other model owner; Minh's approval does not imply teammates have approved their stack. Pin a tested environment at implementation time.

Alternative: Gamma regression with a log link, also positive but with a different variance relationship and mean interpretation; use only if the team chooses it after checking measurement support. An R/brms implementation is viable if the team already agreed on R. Do not silently use a different likelihood or predictor set from Andrei's model.

## Data handoff and readiness

Consume a finalized station-day CSV from Ayreeen; do not treat either current preparation script's output as verified by default. Agree the exact delivered file path at implementation. Canonical model input columns are `station_id`, `date`, `no2_daily`, `wind_speed`, `temperature`, and `precipitation`. The input adapter may map documented source names to these names without performing a new weather join. Retain quality/coverage columns for audit.

Require unique station/date keys, finite covariates, strictly positive NO₂ for the proposed likelihood, documented units and wind meaning, approved weather matching and missing-data rules, and the common retained-row list used by both model owners. Fail with a specific explanation for invalid input; do not silently drop rows, fill missing values, replace nonpositive values, or add a log offset. If the source support conflicts with lognormal, return to the team for a common likelihood decision.

Ayreeen owns source validation, aggregation, precipitation accumulation interpretation, and weather matching. Existing weather CSVs contain `Maximum wind speed [m/s]`, not mean wind speed. Modeling must wait for an explicit decision to retain that measure with correct labeling or supply mean wind. The documented `prepare_explore.py` leaves weather separate; `01_prepare_data.py` includes provisional matching. This design does not reconcile those scripts.

## Statistical specification

For station-day observation i, use y_i ~ LogNormal(eta_i, sigma), with:

eta_i = alpha + beta_weekend * weekend_i + beta_wind * wind_i + beta_temperature * temperature_i + beta_rain * rain_i + beta_sin * sin_i + beta_cos * cos_i.

One intercept and common coefficients across all stations; no station terms or varying effects. This is a benchmark and cannot estimate between-station weekend-effect variation.

Proposed shared transforms: weekend is Saturday/Sunday on finalized Helsinki local dates; wind is divided by 2 m/s; temperature by 10°C; rainfall by 1 mm; all three continuous predictors are centered using training rows only. Store those centers and reuse them for predictions. Seasonal terms are sin/cos of 2*pi*(day_of_year-1)/366 for the proposed 2024 panel. Reject other years unless the shared specification is explicitly expanded. Coordinate these proposed transforms with Andrei before fits are compared.

Supply explicit proper priors for alpha, all six slopes, and positive residual sigma. A prior configuration must record units, concentration/ratio ranges, rationale and sources, and whether each choice is team-approved or provisional. Convert slope ratio ranges with the formula in `docs/prior-plan.md`. Do not copy its illustrative wind numbers as evidence or infer priors from the held-out outcomes. Numerical prior elicitation is an explicit pre-fit task, not a software default. Prior predictive simulations must be reviewed before posterior fitting.

## Workflow and outputs

1. Validate the handoff and save the input hash, ordered station/date keys, schema, and training-only transforms.
2. Implement a reusable pooled-model builder and fitting/prediction entry points. Read a date-split file from the final-analysis owner; keep all stations on the same dates together and reject overlapping train/test dates. Do not choose final validation dates for Ayreeen.
3. Generate prior predictive simulations and plots; document any prior revisions before fitting.
4. Fit with a proposed initial 4 chains, 1,000 warmup and 1,000 retained draws per chain, fixed seed, and recorded sampler settings. Treat these as starting settings; increase sampling or revise the model if checks require it.
5. Report rank R-hat, bulk/tail ESS, divergences, trace behavior, and Monte Carlo uncertainty. Proposed acceptance targets: R-hat <= 1.01, zero divergences, and bulk/tail ESS >= 400 for scalar parameters; flag failures rather than mark the fit complete.
6. Check posterior predictions for the overall distribution, tails, stations, weekends, and residual patterns over time. Summarize residual autocorrelation using actual day gaps, not adjacent retained rows. A successful sampler does not prove adequate fit. Document pooled station mismatch and temporal/shared-day dependence; coordinate structural extensions with Andrei rather than adding them silently.
7. Fit a sensitivity configuration with coefficient prior SDs multiplied by two while keeping means and other priors fixed; compare weekend estimates, uncertainty, and predictions. Label this as one sensitivity check, not comprehensive robustness.
8. Export saved draws, diagnostic tables, posterior predictive draws, held-out predictive draws keyed by station/date/split, transform and prior configurations, and an execution manifest. Exclude held-out outcomes from the fitting function. Ayreeen owns comparative scoring and interpretation across models.
9. Produce Minh's brief methods/results note with plots, weekend median ratio exp(beta_weekend), percentage difference, 95% credible interval, and P(beta_weekend < 0). Arithmetic mean predictions must include sigma. Describe associations, not causal traffic effects. Explain what Minh should be able to present. Update the AI-use log accurately.

## Verification and completion

Use meaningful tests for schema violations, duplicate keys, nonpositive outcomes, missing values, weekend/leap-year transforms, train-only centering, overlapping split dates, prediction-key alignment, and prior-range conversion. Use a small synthetic sampling check for model plumbing; production fit diagnostics and predictive checks are separate scientific acceptance evidence.

No final-data fit or numerical scientific conclusion is possible until the approved input, prior configuration, and shared transforms are ready. Software may be tested with clearly labeled synthetic fixtures while awaiting that handoff. Never present synthetic outputs as project findings.

## Proposed file boundaries

- `models/pooled.py`: pooled model construction, sampling, and prediction.
- `models/pooled_data.py`: input validation, fixed design columns, and training-only transforms.
- `scripts/fit_pooled.py`: CLI orchestration for prior, posterior, sensitivity, and held-out runs.
- `scripts/check_pooled.py`: diagnostic tables and figures.
- `models/pooled_priors.json`: explicit prior ranges and rationale, validated before fitting.
- `tests/test_pooled_data.py`, `tests/test_pooled.py`: input and model checks.
- `reports/minh-pooled-model.md`: Minh's scientific explanation and limitations.
- A minimal pinned modeling dependency file and output ignore rules, selected during implementation.

Leave teammates' scripts and the Google Doc unchanged during planning. This proposal becomes the basis for the detailed implementation checklist after design approval.
