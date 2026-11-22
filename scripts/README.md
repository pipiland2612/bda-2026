# Analysis scripts

Run the preparation, exploratory summaries, and figures from the project root:

```sh
python3 -m pip install -r requirements.txt
python3 scripts/prepare_explore.py
python3 scripts/plot_explore.py
```

`prepare_explore.py` uses only the Python standard library. It reads CSV
exports from `data/raw/` and writes coverage, daily NO₂, separate daily weather,
weekday/weekend, seasonal, and coverage-sensitivity summaries to
`data/processed/`. It does not merge weather stations to air-quality sites or
fit models.

`plot_explore.py` reads `data/processed/no2_daily.csv` and writes four
descriptive figures to `reports/figures/`: qualifying daily NO₂ time series,
weekday/weekend distributions, monthly means, and daily coverage. The
weekday/weekend figure is unadjusted and should not be interpreted as a causal
traffic effect.

Review the source metadata and the provisional aggregation choices in
`02-prepare.md` before using generated weather data in a model.
