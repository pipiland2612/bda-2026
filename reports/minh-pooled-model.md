# Minh's pooled model: report and presentation notes

Status: implementation exercised using synthetic data. Real Helsinki results are pending finalized shared data and approved priors. Do not use demonstration estimates as project findings.

## My technical contribution

I fit the pooled benchmark: is weekend status associated with daily NO₂ after adjustment for the agreed wind measure, temperature, rainfall and annual seasonality? Every station shares one intercept and common effects. Andrei's hierarchical model can assess what changes when station differences are modeled explicitly.

## Model and variables

For station-day i:

`NO₂_i ~ LogNormal(eta_i, sigma)`

`eta_i = alpha + beta_weekend*weekend_i + beta_wind*wind_i + beta_temperature*temperature_i + beta_rain*rain_i + beta_sin*sin_i + beta_cos*cos_i`.

This likelihood requires positive concentrations; nonpositive measurements need a documented group decision, not an arbitrary log offset. Weekends are Saturday/Sunday in finalized Helsinki dates. Wind increments are 2 m/s, temperature 10°C, rainfall 1 mm. Continuous predictors use training-only centers, reused for held-out predictions. Annual sine/cosine terms have a 366-day period for 2024. The final wind definition remains to be confirmed because the raw files contain maximum rather than mean wind speed.

## Priors, fitting and checks

Use an explicit Normal prior on alpha and each slope, with slope parameters derived from plausible concentration-ratio ranges. Use an explicit HalfNormal residual sigma prior. Numerical values and scientific justifications await group elicitation; software defaults and the planning worksheet's illustrative wind range are not scientific evidence.

First simulate prior predictions and review plausible concentrations/extremes across weather and seasonal contexts. Record any changes before posterior fitting. Initial fitting settings are 4 chains, 1,000 warmup and 1,000 retained draws each, seed 20261121, target acceptance 0.9. Actual settings and dependency versions are saved.

Check rank R-hat, bulk/tail ESS, Monte Carlo error, divergences and traces. The software flags R-hat > 1.01, ESS < 400, undefined diagnostics or any divergence. Then compare observed and replicated distributions, tails, station means and weekend patterns; examine residual time plots and correlations at actual day gaps.

Good sampling means the algorithm explored the specified posterior reliably. It does not establish that the model fits the science: pooling may miss station differences or shared pollution episodes. Repeat fitting with doubled slope prior SDs and compare estimates, uncertainty and predictive behavior.

## Interpreting the weekend coefficient

`exp(beta_weekend)` is the conditional median concentration ratio, holding weather and season fixed. A hypothetical ratio of 0.8 would mean a 20% lower conditional median; this is an illustration, not a project estimate.

Report the posterior ratio and 95% credible interval, percentage difference `100*(exp(beta_weekend)-1)`, and probability `P(beta_weekend < 0)`, computed from draws. Arithmetic mean predictions are `exp(eta + sigma²/2)`, not `exp(eta)`.

The association is not a causal traffic effect: weekend status does not measure traffic directly. This pooled model also cannot estimate differences in weekend effects between stations.

## Results pending the verified run

- Final rows, station coverage, exclusions and wind definition: awaiting Ayreeen's handoff.
- Numerical priors and prior predictive review: pending.
- Sampling diagnostics, model revisions and posterior checks: pending actual fit.
- Weekend ratio, interval, percentage difference and posterior probability: pending actual fit.
- Sensitivity findings: pending actual fits.
- Held-out draws: supply on agreed splits for Ayreeen's comparative analysis.

## My presentation

1. Explain the pooled benchmark and equation in plain language.
2. Explain the priors and prior predictive checks.
3. Present the checked weekend estimate and uncertainty when available.
4. Show one diagnostic/predictive figure and explain what it reveals.
5. Explain pooling/temporal limitations, then hand over to the hierarchical-model and comparison sections.

Run commands and data contract: [models/README.md](../models/README.md).
