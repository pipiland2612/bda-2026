# Weekend nitrogen dioxide patterns in the Helsinki region

## Project purpose

We will investigate whether daily nitrogen dioxide (NO₂) concentrations are lower on weekends after accounting for weather and season, and whether this association differs between monitoring sites. Our aim is to estimate the size and uncertainty of the weekend difference and compare a pooled Bayesian regression with a hierarchical model.

The immediate task is to establish a usable dataset. We propose 2024 observations at four fixed sites, subject to coverage checks. The plans below describe work to do; they do not report findings or confirm that the proposed station panel is available.

## Why this question is interesting

HSY identifies traffic as a source of NO₂ and explains that local surroundings affect pollutant dilution [1]. FMI describes how weather changes pollutant accumulation and transport [2]. A weekday–weekend comparison therefore gives us a concrete question, but weather and season must be considered alongside the calendar.

Our working hypothesis is that NO₂ may be lower on weekends and that the difference may be larger at sites influenced by traffic. This is a hypothesis to test, not an established result. Weekend status is a calendar indicator, not a direct measurement of traffic. Heating, public holidays, atmospheric chemistry, and other activities may also affect the observed difference.

The study will estimate associations at the selected sites. It will not establish the causal effect of reducing traffic or describe exposure everywhere in the Helsinki region.

## Proposed dataset

### Primary choice

Use HSY monitoring observations accessed through FMI, joined to FMI weather observations [3, 4]. We will target 1 January to 31 December 2024 and daily NO₂ concentration in µg/m³. A full year lets us examine seasonal variation without adding multiple years of changing conditions.

The proposed sites are Mäkelänkatu, Kallio, Leppävaara, and Tikkurila. Their names are starting points for discovery; exact station IDs, historical locations, pollutant coverage, and environment classifications must be verified. We should aim for sites with contrasting environments rather than select sites because their weekend differences look interesting.

Four sites over leap-year 2024 would provide at most 1,464 station-days before quality checks and missingness. These are not 1,464 independent observations: sites share regional conditions, and adjacent days can be correlated.

### Official access routes

| Dataset | Role | Access and selection |
| --- | --- | --- |
| HSY air-quality observations through FMI | Daily NO₂ outcome and station identity | FMI Download observations [3]; inspect available pollutant, station, dates, resolution, and quality information |
| FMI weather observations | Wind speed, temperature, precipitation | Same service [3]; select suitable weather sites after checking distance and historical coverage |
| EEA verified air-quality observations | Alternative historical source if the FMI route is unsuitable | EEA download service [5]; choose Finland, NO₂, year, station, and verified E1a where available |
| HSY station descriptions | Explain differences in monitoring environments | HSY site metadata [6]; confirm classifications for the selected year |

FMI offers free historical observation downloads [3]. Its API manual provides a route for later reproducible acquisition [7]. EEA supplies reported measurement series from 2013 onward as zipped Parquet files [5]. EEA and FMI may contain the same measurements; they must not be counted as separate observations.

Prefer measured concentrations. Air-quality indices, forecasts, modelled maps, and annual summaries do not provide the intended daily outcome.

### Variables to retain

| Variable | Planned meaning | Check before use |
| --- | --- | --- |
| station_id | Stable monitoring-site identifier | Relocations, aliases, operation dates |
| date | Common analysis day | Source time zone and daylight-saving handling |
| no2_daily | Mean valid NO₂ concentration, µg/m³ | Flags, completeness, nonpositive readings |
| valid_hours and expected_hours | Daily coverage information | Expected slots must match the day definition |
| wind_speed | Daily mean wind speed, m/s | Observation height, site, and available resolution |
| temperature | Daily mean air temperature, °C | Consistent aggregation window |
| precipitation | Daily precipitation total, mm | Accumulation periods, not a sum of overlapping totals |
| weekend | Saturday or Sunday indicator | Derive from the agreed local calendar |
| day_of_year | Seasonal position | Leap year and chosen seasonal representation |
| weather_station_id | Weather source for each monitoring site | Distance and representativeness |

We will preserve quality flags and source identifiers with the original files. Missing observations remain missing; they are never replaced by zero.

### Quality and feasibility decisions

Use Europe/Helsinki local dates for the weekend question, after interpreting each source's actual timestamps correctly. Local daylight-saving days can have 23 or 25 hourly slots. A provisional completeness rule is at least 75% of expected hourly slots, rounded up; confirm it against provider guidance and document it before analysing weekend differences.

Apply the provider's validity information before aggregation. Examine duplicate timestamps, unit changes, unusually high observations, and zero or negative values. Do not delete genuine pollution episodes merely because they are extreme. Investigate measurement conventions before choosing a likelihood that requires positive values.

Select weather stations for geographical suitability and coverage. If one weather series is assigned to several pollution sites, document this and recognise the shared information. Prefer complete exposure/weather windows; any exclusions or later imputation need a clear rationale and sensitivity check.

If some sites lack suitable 2024 data, first choose other stable sites with the same pollutant. If needed, choose another complete year consistently across sites. PM2.5 is a secondary option only if NO₂ proves infeasible; it would require revisiting the scientific question and prior rationale.

## Planned Bayesian models

### Shared structure

Use the same analysis rows, outcome definition, weather transformations, and seasonal terms in both models. Start with a small set of interpretable predictors: weekend, wind speed, temperature, rainfall, and a simple annual seasonal basis. Check whether holidays require a separate indicator or sensitivity analysis.

If all retained concentrations are strictly positive and the measurement conventions support it, a lognormal observation model is a candidate. Let eta describe log concentration. A weekend coefficient beta then corresponds to a median concentration ratio exp(beta), with percentage difference 100 × (exp(beta) − 1). Arithmetic mean comparisons also depend on residual variation and should be calculated from posterior predictions.

The likelihood remains a decision for the group after checking measurement support. Do not add an arbitrary constant to force zero or negative readings onto a log scale.

### Model A pooled regression

Estimate one intercept and one common weekend association, with common weather and seasonal coefficients. This supplies a simple benchmark that treats the sites as sharing the same regression relationship.

### Model B hierarchical regression

Add partially pooled station intercepts and weekend coefficients. This allows typical concentration and weekend association to differ by site while sharing information. Start with separate population distributions for these effects rather than an unnecessarily complex correlation structure.

With only a few stations, between-site variation and varying weekend coefficients may be weakly identified. Check whether the data and priors support the extra complexity; if necessary, use station intercepts first and treat varying weekend coefficients as an extension. A common weekend coefficient alone cannot answer whether the association differs by site.

Neither model automatically handles serial dependence or shared daily pollution episodes. Examine residual time patterns and consider an appropriate temporal/shared-date component if needed. Explain any expansion and retain a clear comparison between models.

## Priors based on scientific reasoning

The TA's guidance is to choose priors with interpretable units and plausible effects. A Normal(0, 1) prior is not automatically weak or sensible: its implications change with the predictor scale and link function.

Define meaningful predictor increments, such as 2 m/s for wind and 10°C for temperature. Ask what concentration ratios are plausible for those increments before fitting. A central 95% ratio range [L, U] can be translated into a Normal coefficient prior on the log scale with mean (log L + log U)/2 and SD (log U − log L)/(2 × 1.96).

Scientific expectations should guide, not fix, coefficient signs. Wind may favour dilution, but transport can complicate the relationship [2]. Weekend differences may be negative, but our prior should allow little difference or a positive association. Temperature and rainfall need particular caution because their conditional effects depend on season and other processes.

Elicit an intercept prior from an independent plausible concentration range. Give residual scales and hierarchical standard deviations proper positive priors calibrated to concentration variation. Every parameter, including any added temporal or correlation parameter, needs an explicit proper prior.

Before fitting, simulate prior predictions and inspect concentrations, extremes, site differences, and weather responses. Record the sources and reasoning. Then evaluate sensitivity to reasonable alternative prior locations and scales for each model. See the repository's prior worksheet for further detail.

## Detailed workflow and deliverables

### Step 1 Verify coverage

Inspect source metadata and, when the group starts acquisition, a small sample of observations. Confirm station IDs, dates, units, flags, validation status, and matching weather. Produce a coverage table with station, available dates, expected and usable records, missingness, and weather match. Do not infer coverage from a station name appearing in a current map.

**Completion criterion:** we can explain exactly which observations would enter the project and why those sources are suitable.

### Step 2 Acquire and document

Download the agreed period through a reproducible route. Preserve original exports locally and record the query or export settings, retrieval date, licence, attribution, and checksum in the data manifest. Follow provider terms; FMI's covered open data use CC BY 4.0 [8]. Check the selected EEA or other product's terms separately.

**Deliverable:** source files and a completed provenance manifest, with a reproducible acquisition procedure.

### Step 3 Prepare the analysis table

Apply quality flags, reconcile time zones and units, aggregate to the agreed daily windows, and join weather by the same windows. Count coverage and exclusions by station and calendar group. Document the final missing-data rule and check whether missingness is related to season or weekend status.

**Deliverable:** one reproducible station-day table and a short preparation log.

### Step 4 Explore and specify

Plot coverage, time series, seasonal patterns, weekday/weekend summaries, and site differences. Inspect outcome support and serial dependence. These descriptive comparisons motivate checks; they do not replace adjusted model estimates. Finalize the likelihood, predictor transformations, priors, and prediction target in a written specification.

**Deliverable:** a small set of exploratory figures and an agreed model specification.

### Step 5 Check priors and fit

Run prior predictive checks, revise implausible choices with reasons, and fit both models. Record software versions, seeds, chains, warmup, retained iterations, and relevant sampler settings. Use a reproducible dependency environment once the group selects R or Python.

**Deliverable:** reproducible fitting commands, saved fits, and prior predictive figures.

### Step 6 Diagnose and improve

For each model inspect R-hat, effective sample sizes, divergences, and sampler behaviour. Compare observed and replicated distributions, high-concentration days, station summaries, weekend patterns, and temporal residual structure. Address problems before interpreting results and record changes.

**Deliverable:** a diagnostic summary for every model and a justified account of model revisions.

### Step 7 Compare predictions and prior sensitivity

Our proposed prediction target is future days at the selected stations, conditional on the available weather inputs. This is not a full operational forecast unless future weather inputs are also supplied. Use contiguous time blocks or rolling-origin splits and hold out the same dates at all stations. Keep preprocessing based on training data only. Ordinary random row splits may benefit from shared days and temporal dependence.

Compare predictive accuracy and uncertainty on the concentration scale, alongside predictive interval behaviour. Repeat key fits with plausible alternative priors. If using PSIS-LOO as an additional comparison, explain its different prediction target and inspect its diagnostics.

**Deliverable:** a model comparison with uncertainty and a prior-sensitivity summary.

### Step 8 Interpret and communicate

Report the adjusted weekend ratio, its credible interval, and relevant posterior probabilities. Show station-specific estimates with uncertainty and predictions under common weather/season settings. Explain the effect of assumptions, missingness, shared weather, temporal dependence, and unmeasured traffic or chemistry.

**Deliverable:** a report and presentation that answer the original question, explain both models, and describe limitations. Every member should be able to justify the priors and diagnostic decisions.

## What we do next

The next group session should focus on the coverage table, not model fitting. One person checks air-quality metadata, another checks weather availability, and another records the common time definition and provenance requirements. Review the findings together and choose the final sites and dates.

After that feasibility decision, agree the software stack and write the acquisition script. No results are expected at the planning stage.

## Sources

1. HSY, Assess the air quality in your area. https://www.hsy.fi/en/air-quality-and-climate/air-quality-in-the-helsinki-metropolitan-area/assess-the-air-quality-in-your-area/
2. FMI, How does weather affect air pollution. https://en.ilmatieteenlaitos.fi/weather-and-air-quality
3. FMI, Download observations. https://en.ilmatieteenlaitos.fi/download-observations
4. HSY, Where can you find air quality information. https://www.hsy.fi/en/air-quality-and-climate/air-quality-in-the-helsinki-metropolitan-area/where-can-you-find-air-quality-information/
5. EEA, Air Quality download service. https://www.eea.europa.eu/en/datahub/datahubitem-view/datahubitem-view/778ef9f5-6293-4846-badd-56a29c70880d
6. HSY, Real time air quality at HSY monitoring sites. https://www.hsy.fi/en/environmental-information/open-data/avoin-data---sivut/real-time-air-quality-at-hsy-monitoring-sites-/
7. FMI, Open data manual. https://en.ilmatieteenlaitos.fi/open-data-manual
8. FMI, Licence and terms. https://www.ilmatieteenlaitos.fi/avoin-data-lisenssi

Project repository: https://github.com/pipiland2612/bda-2026
