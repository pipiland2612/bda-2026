# Air quality in the Helsinki region

Bayesian Data Analysis, Aalto University, 2026. **Planning scaffold only: no data downloaded, models fitted, or results produced.**

Group members: TODO. Language/tool: TODO (R + brms is a practical starting option; Python + Bambi/PyMC is also suitable).

## Proposed question and scope

How do weather and calendar patterns relate to daily NO₂ concentrations across Helsinki-region monitoring stations, and does partial pooling improve predictions?

Start with **one outcome**, provisionally NO₂ (µg/m³). PM2.5 is an alternative if coverage is better, not a required second analysis. NO₂ has a clearer local traffic interpretation; PM2.5 includes substantial transported pollution. These are observational associations, not causal estimates.

Provisional scope: one complete historical year, preferably 2024, at 3–5 fixed stations with contrasting environments. Confirm pollutant availability and validated coverage before committing. Candidate sites: Mäkelänkatu, Kallio, Leppävaara, Tikkurila, Luukki; these are candidates, not a verified common panel. Approximately 1,000–2,000 station-days keeps computation manageable. Avoid mobile stations unless their movements are explicitly handled.

## Official data sources

Use [HSY/FMI measured concentrations and FMI weather](docs/data-sources.md). EEA verified observations are a fallback. Do not substitute an air-quality index, forecast, modelled map, or annual average for measured daily concentrations.

Source inventory, access steps, licence notes, and unresolved availability checks: [docs/data-sources.md](docs/data-sources.md).

## Course requirements to plan for

[Assignment instructions](https://avehtari.github.io/BDA_course_Aalto/project.html), checked 8 November 2026:

- Report due 29 November; maximum 20 pages. Presentation: 15 minutes, with questions; dates 4 December and 7–11 December. Confirm your slot in MyCourses.
- Describe at least two models, with covariates in some models and explicit proper priors for every parameter.
- Explain data provenance and differences from existing analyses; justify priors and inference settings.
- For every model, report convergence diagnostics (R-hat, ESS, divergences), posterior predictive checks, and prior sensitivity.
- Compare predictive performance with uncertainty, discuss limitations, and document AI use.

Use the course page and rubric as the authority; recheck before submission.

## Planned workflow

1. **Agree scope.** Choose pollutant, dates, fixed stations, research question, and prediction target. Record decisions in `docs/decisions.md`.
2. **Check feasibility.** Manually inspect source metadata and a small sample later. Verify units, flags, date coverage, station locations, and matching weather variables before a full download.
3. **Acquire reproducibly.** Preserve original files locally, record exact queries, retrieval dates, attribution, and checksums in `data/manifest.csv`. Raw data stay out of Git.
4. **Prepare data.** Normalize source timestamps first; explicitly choose UTC days or Europe/Helsinki local days. Handle daylight-saving days correctly. Align weather and pollution windows. Predefine daily completeness (for example 75% of expected hourly slots), preserve missingness, and check duplicate station-times. Never turn missing values into zero. Inspect nonpositive measurements before selecting a positive-outcome likelihood.
5. **Explore.** Later examine coverage, seasonal patterns, station differences, extremes, and temporal dependence. Document exclusions without selecting them to improve results.
6. **Specify priors.** Use physical units and literature-informed effect ranges. Write the rationale before fitting, then run prior predictive checks. See `docs/prior-plan.md`.
7. **Fit two models.** Keep outcome, rows, and predictor transformations comparable. Model A: pooled regression with weather and calendar predictors. Model B: the same regression plus station-level varying intercepts with proper hyperpriors. Consider residual temporal structure if checks require it; partial pooling alone does not remove serial dependence.
8. **Check inference and predictions.** Inspect diagnostics for each fit; assess tails, peaks, station summaries, seasonal patterns, and residual autocorrelation. Change models only with documented reasons.
9. **Compare and stress-test.** For future-day prediction, use blocked time holdouts or rolling-origin validation, holding all stations on the same dates together. Ordinary row-wise LOO can leak temporal information. If using PSIS-LOO, explain its prediction target and inspect Pareto-k. Report uncertainty and practical error on the original concentration scale. Refit with alternative plausible prior scales for each model.
10. **Write and rehearse.** Explain findings and limitations in the report; keep diagnostics in an appendix when useful. Split work and ensure every member can explain both models.

## Modelling choices to resolve later

For strictly positive concentrations, consider a lognormal observation model with a linear predictor for log concentration. Check whether reported zeros/negatives are measurement artefacts, censoring, or valid values before choosing this family; do not add an arbitrary constant merely to take logs. A robust model on a documented transformed scale is another candidate. Do not select a likelihood yet.

Candidate predictors: wind speed (m/s), temperature (°C), rainfall (mm/day), weekend indicator, and a modest seasonal basis. Weather station choice and spatial representativeness need documentation. Weekend is a proxy, not measured traffic volume. Limit predictors and interactions to keep the project explainable.

## Repository layout

```text
docs/              Sources, prior plan, decisions, course checklist
data/raw/          Local original downloads (ignored)
data/processed/    Local analysis tables (ignored)
data/manifest.csv  Provenance template, no observations
scripts/           Numbered analysis-stage placeholders
models/            Model specification placeholder
reports/           Report outline and future figures
```

## Getting started

Clone the repository, read the source and prior notes, fill in group members, and agree scope. No environment or packages are installed by this scaffold. After choosing R or Python, add a reproducible dependency lockfile and documented run commands. Do not treat the placeholder files as working analysis code.

## Contributions and AI disclosure

Commits use the student's configured Git identity, without an AI co-author trailer. This is separate from the course's disclosure requirement.

Initial disclosure draft (review and update): “An AI assistant helped locate official data sources and prepare the repository structure and initial planning notes. Group members will verify sources and modelling suggestions. No data processing, model fitting, or results were generated during this setup.” Record subsequent use in `docs/ai-use.md`.
