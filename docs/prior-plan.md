# Prior elicitation plan — not a fitted model

The TA recommended priors based on covariate knowledge rather than using Normal(0, 1) everywhere. Prior strength depends on the predictor units and the likelihood/link. The following is an elicitation method, not established coefficient estimates.

## Evidence to read

- [HSY: local sources and traffic](https://www.hsy.fi/en/air-quality-and-climate/air-quality-in-the-helsinki-metropolitan-area/assess-the-air-quality-in-your-area/): traffic contributes to NO₂ and particles; open environments improve dilution.
- [FMI: weather and air quality](https://en.ilmatieteenlaitos.fi/weather-and-air-quality): weak winds can permit local accumulation, while wind can also transport pollution.
- [HSY: transported fine particles](https://www.hsy.fi/en/air-quality-and-climate/air-protection-and-health/air-quality-challenges-now-and-in-the-future/long-range-transport-reduces-air-quality/): transported pollution is a major part of PM2.5, so local predictors need cautious interpretation.

These sources support qualitative expectations. They do not establish numerical regression coefficients or prove a causal effect conditional on other predictors.

## Translate intuition into the model scale

For a proposed lognormal model, let log(y / (1 µg/m³)) have mean eta and residual scale sigma. Define wind in units of 2 m/s, temperature in units of 10°C, and rainfall in a documented unit. Center continuous predictors at meaningful fixed reference values; if using sample-based standardization, calculate it on training data only. With eta = alpha + beta*x, exp(beta) is a concentration-median ratio per stated increment.

Ask the group: “For this increment, what concentration ratios seem plausible before seeing this project's outcomes?” Convert a plausible central 95% ratio range [L, U] to a Normal prior on beta using mean (log(L)+log(U))/2 and SD (log(U)-log(L))/(2*1.96).

**Arithmetic illustration only:** a wind ratio range [0.70, 1.10] per +2 m/s yields beta approximately Normal(-0.131, 0.115). This range is invented for explanation, not measured, literature-calibrated, or approved. Replace it with a justified range. It retains some probability of a positive association rather than enforcing a negative sign.

## Elicitation worksheet

| Parameter | Initial reasoning | Group decision still needed |
| --- | --- | --- |
| Intercept | Typical concentration at reference weather/calendar conditions | Independent historical reference range in µg/m³; translate to log scale |
| Wind | Local dilution suggests a negative tendency; transport complicates it | Plausible ratio range per +2 m/s, allowing sign uncertainty |
| Temperature | Heating, mixing, and season can confound the sign | Broad but plausible range per +10°C; do not force a sign |
| Rainfall | Removal mechanisms suggest reductions in some settings | Outcome-specific range and transformation; allow uncertainty |
| Weekend | Potentially lower traffic, but only a proxy | Ratio versus weekdays; account for uncertain magnitude/sign |
| Seasonal terms | Bounded plausible seasonal amplitude | Joint prior predictive effect, not independent arbitrary wide coefficients |
| Station SD | Positive scale describing between-site log differences | Proper half-normal/exponential prior calibrated to concentration ratios |
| Residual sigma | Positive log-scale variation | Proper prior calibrated to plausible within-site variation |
| Additional parameters | Any degrees of freedom, correlation, or new scales | Explicit proper priors with natural support |

For the hierarchical model, use station deviations with a Normal population distribution and a proper positive prior on its SD. A non-centered parameterization can use standard Normal latent variables: Normal(0,1) is sensible there because scale is supplied by the hierarchy.

## Before and after fitting

1. Record sources, predictor units, elicited ranges, and all proper priors, including software defaults.
2. Simulate prior predictions; inspect plausible concentrations, extremes, station differences, and weather responses. Lognormal residual variation also affects the arithmetic mean: exp(eta) is the median, not the mean.
3. Revise implausible predictions with a documented rationale before fitting outcomes.
4. For each fitted model, compare reasonable alternative prior locations/scales (for example wider effect ranges). Examine substantive conclusions and predictions, not only coefficient signs.
5. If historical data calibrate priors, keep them distinct from the outcome period used for fitting, or explain the resulting empirical-Bayes choice. Do not quietly use the same outcomes twice.
