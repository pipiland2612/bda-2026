# 03-explore

`python3 scripts/prepare_explore.py` creates:

- `data/processed/no2_weekend_summary.csv`: equal-weight means of qualifying daily NO₂ means for weekdays and weekends, by site.
- `data/processed/no2_season_summary.csv`: equal-weight means by site and meteorological season.
- `data/processed/no2_coverage_sensitivity.csv`: repeats the weekday/weekend comparison using 75%, 90%, and 100% daily completeness thresholds.

Then run `python3 scripts/plot_explore.py` to create:

- `reports/figures/no2_daily_timeseries.png`: qualifying daily NO₂ means by station.
- `reports/figures/no2_weekday_weekend.png`: unadjusted weekday/weekend distributions by station.
- `reports/figures/no2_monthly_means.png`: monthly means of qualifying daily values.
- `reports/figures/no2_daily_coverage.png`: days meeting the 75% hourly-coverage rule.

These are unadjusted descriptive comparisons. They do not account for weather,
season in the weekend comparison, temporal dependence, or uncertainty, and do
not estimate a causal traffic effect. Coverage is reported separately so the
group can review threshold sensitivity before modeling.
