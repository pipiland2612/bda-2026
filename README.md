# Air quality in the Helsinki region

## What we want to study

**Are NO₂ concentrations lower on weekends, after accounting for weather and season, and does the pattern differ between monitoring sites?**

This gives us a focused way to investigate weekly pollution patterns across the Helsinki region. Weekend differences describe associations; they do not directly measure the effect of traffic.

We plan to use daily NO₂ concentrations from 2024 at 3–5 fixed monitoring stations, together with wind speed, temperature, and rainfall. The exact stations and period depend on data coverage.

## What we work on first

**Check whether we can build a complete dataset for this question.**

1. Find 2024 NO₂ observations for candidate stations: Mäkelänkatu, Kallio, Leppävaara, and Tikkurila.
2. Confirm station locations, measurement units, quality flags, timestamps, and missing observations.
3. Identify suitable weather observations covering the same dates.
4. Record which stations and variables are usable before downloading the full dataset.

The first deliverable is a short coverage table: station, available dates, missingness, and matching weather station.

Start with [FMI Download observations](https://en.ilmatieteenlaitos.fi/download-observations). See [data-source notes](docs/data-sources.md) for HSY/FMI access and the EEA fallback.

## Planned workflow

1. **Confirm data coverage** and finalize the stations and study period.
2. **Prepare daily observations:** apply quality checks, align weather and pollution timestamps, and document missing-data rules.
3. **Explore patterns:** compare weekdays and weekends, seasons, and stations.
4. **Choose informed priors:** express plausible concentration changes for weather and weekend predictors, then check simulated predictions. See the [prior plan](docs/prior-plan.md).
5. **Fit two Bayesian models:** a pooled regression and a hierarchical regression with station-specific intercepts and weekend effects.
6. **Check both models:** inspect convergence, posterior predictions, and residual temporal patterns; assess sensitivity to prior choices.
7. **Compare predictions on held-out time periods**, keeping all stations on the same dates together.
8. **Write the report:** explain the weekend patterns, uncertainty, limitations, and what the group learned.
