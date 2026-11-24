# Bayesian models

Minh owns Model A, the pooled lognormal regression implemented here. Andrei owns the hierarchical Model B; Ayreeen owns preprocessing and final comparison according to the edited assignment sections. This implementation does not modify their work.

See the [approved design](../docs/superpowers/specs/2026-11-21-minh-pooled-model-design.md) and [Minh's explanation](../reports/minh-pooled-model.md).

## Setup

Run from the repository root. Tested with Python 3.13 on macOS arm64; the lock records that exact environment. On another platform resolve `requirements-model.txt` and record the resulting versions. Keep PyMC 5.x and ArviZ below 1 for this saved-data contract.

```sh
rtk python3 -m venv .venv
rtk .venv/bin/python -m pip install -r requirements-model.lock.txt
rtk .venv/bin/python -m pytest -q
```

## Data handoff

Consume Ayreeen's finalized CSV, not provisional weather outputs. Required columns are `station_id` (text), `date` (`YYYY-MM-DD`, 2024), `no2_daily` (positive µg/m³), `wind_speed` (m/s), `temperature` (°C), and `precipitation` (mm). Extra quality/coverage columns are retained in saved training data. Invalid, missing, infinite or duplicate records cause an error; selection/exclusions belong to preprocessing.

The current source CSVs name **maximum wind speed**. Confirm its meaning or obtain mean wind before production fitting; this code does not rename maximum wind as mean wind or perform a weather join.

Supply a companion metadata JSON using this structure, filled with the owner's actual verified information. This example is not an approval record:

```json
{
  "approved": false,
  "units": {"no2_daily":"ug/m3", "wind_speed":"m/s", "temperature":"degC", "precipitation":"mm"},
  "wind_definition": "OWNER TO CONFIRM",
  "wind_measure": null,
  "weather_matching": "OWNER TO CONFIRM",
  "missing_data_rule": "OWNER TO CONFIRM",
  "source_validation": "OWNER TO CONFIRM",
  "shared_specification": "AGREE WITH MODEL B OWNER",
  "retained_keys": [["station ID", "2024-01-01"]]
}
```

`wind_measure` must explicitly be `daily_mean_speed` or `mean_of_hourly_maxima`, describing both the measurement and daily aggregation. The wind coefficient's prior must contain the same `measure` field. Placeholder descriptions are rejected even if `approved` is true.

`retained_keys` must list every station/date row in the CSV's order, shared by both model owners. Model transforms are training-only centering, wind / 2 m/s, temperature / 10°C, rain / 1 mm, Saturday/Sunday weekends, and a leap-year annual sine/cosine pair. Calendar variables are derived from date, not copied from unverified supplied columns.

## Priors and review

`pooled_priors.json` is deliberately incomplete. Elicit alpha's log-scale mean/SD, the positive residual sigma HalfNormal scale, and six slope ratio ranges. Record scientific justification and sources; mark approved only after a real group decision. Coefficient ratios become Normal priors using the formula in `docs/prior-plan.md`. Joint seasonal amplitude must also be checked.

Substitute actual finalized handoff paths in these commands:

```sh
rtk .venv/bin/python -m scripts.fit_pooled --data data/processed/analysis_table.csv --metadata data/processed/analysis_metadata.json --priors models/pooled_priors.json --output outputs/pooled/prior --mode prior
rtk .venv/bin/python -m scripts.check_pooled --run outputs/pooled/prior
```

Prior mode generates both baseline and widened-slope prior predictions; the checker renders separate plots for each. Inspect plausible concentrations and extremes, then save `outputs/pooled/prior_review.json` with an actual review:

```json
{
  "accepted": false,
  "reviewed_by": "ACTUAL REVIEWER",
  "reviewed_at": "ACTUAL DATE",
  "reviewed_configurations": ["baseline", "wide-slopes"],
  "prior_sha256": "SHA256 OF PRIOR JSON",
  "prior_run_manifest": "prior/manifest.json",
  "prior_run_manifest_sha256": "SHA256 OF THAT MANIFEST"
}
```

Manifest paths resolve relative to this review JSON unless absolute. Use `scripts.fit_pooled.sha256(Path(...))` or a local SHA-256 utility for hashes. Changed priors, data or metadata require a matching new prior run/review. Both effective prior configurations and their prediction files are hash-bound to the reviewed manifest. Standard posterior runs require baseline review; sensitivity requires explicit review of baseline and wide-slopes. Held-out fitting requires the split-specific review below, because each training partition has its own predictor centers. Do not fabricate an accepted review or list a configuration that was not inspected.

For explicitly synthetic demonstrations only, `--synthetic` bypasses scientific review; both metadata and priors must contain `"synthetic": true`. Artifacts remain labeled synthetic.

## Fit and check

```sh
rtk .venv/bin/python -m scripts.fit_pooled --data data/processed/analysis_table.csv --metadata data/processed/analysis_metadata.json --priors models/pooled_priors.json --prior-review outputs/pooled/prior_review.json --output outputs/pooled/posterior --mode posterior
rtk .venv/bin/python -m scripts.check_pooled --run outputs/pooled/posterior
```

Defaults: 4 sequential chains, 1,000 warmup and 1,000 retained draws per chain, seed 20261121, target acceptance 0.9. Overrides are `--chains`, `--warmup`, `--draws`, `--seed`, `--target-accept`; justify and record changes. Existing output directories are never overwritten.

`--mode sensitivity` fits baseline and doubled slope prior SDs with all means, alpha and sigma priors fixed. Run the checker on that root directory to compare weekend summaries and inspect both sets of predictive plots. This is one sensitivity check, not comprehensive robustness.

## Held-out comparison handoff

Ayreeen supplies a split CSV with exactly `split_id,date,partition` (`train` or `test`), with no station-specific columns. Each split assigns every available date exactly once, training strictly before test. All stations on a date stay together. `--splits` is required for `heldout` and `prior-heldout`, and rejected in other modes. For rolling windows that exclude later dates, supply the appropriate finalized input window and matching metadata; unspecified dates are never silently ignored.

Generate and inspect prior predictions separately for every training partition before fitting:

```sh
rtk .venv/bin/python -m scripts.fit_pooled --data data/processed/analysis_table.csv --metadata data/processed/analysis_metadata.json --priors models/pooled_priors.json --output outputs/pooled/prior-heldout --mode prior-heldout --splits data/processed/model_splits.csv
rtk .venv/bin/python -m scripts.check_pooled --run outputs/pooled/prior-heldout
```

Each `split-000/`, etc., contains baseline and widened-slope prior predictions built from that split's training rows and centers. After inspecting the required configurations for each split, create a separate review JSON. Use actual split IDs from the CSV as `reviewed_splits` keys; this example is not an approval:

```json
{
  "accepted": false,
  "reviewed_by": "ACTUAL REVIEWER",
  "reviewed_at": "ACTUAL DATE",
  "reviewed_splits": {"ACTUAL SPLIT ID": ["baseline"]},
  "prior_sha256": "SHA256 OF PRIOR JSON",
  "prior_run_manifest": "prior-heldout/manifest.json",
  "prior_run_manifest_sha256": "SHA256 OF THAT MANIFEST"
}
```

The gate binds the split CSV hash, every split's ordered training keys, training-only transform hash, effective prior configurations and prediction files. It rejects full-data prior reviews and different split assignments. All input/config hashes remain required: changing even test-only weather conservatively requires a new run/review, although the training designs and transforms remain identical. Inspect both configurations if listing both; held-out fitting requires explicit baseline review for every split.

```sh
rtk .venv/bin/python -m scripts.fit_pooled --data data/processed/analysis_table.csv --metadata data/processed/analysis_metadata.json --priors models/pooled_priors.json --prior-review outputs/pooled/heldout_prior_review.json --output outputs/pooled/heldout --mode heldout --splits data/processed/model_splits.csv
rtk .venv/bin/python -m scripts.check_pooled --run outputs/pooled/heldout
```

The fit receives only training outcomes. Prediction strips test outcomes, reuses training centers and recomputes eta from posterior alpha/beta/sigma for new rows.

## Outputs and acceptance

- Root `manifest.json`: input/config hashes, versions, settings, row keys, synthetic label, split mapping and run status. `complete` means execution finished, not that scientific checks passed.
- `baseline/` and sensitivity `wide-slopes/`: `training.csv`, `transforms.json`, `effective_priors.json`, and `posterior.nc` (posterior, posterior predictive, sample statistics, pointwise log likelihood).
- Held-out directories `split-000/`, etc., map to original split IDs in the manifest. `predictions.nc` contains `no2(chain,draw,observation)` with `split_id(observation)`, `station_id(observation)` and `date(observation)` coordinates; `prediction_keys.csv` provides the same keys. Join models by keys, not array position. Prior/posterior artifacts also retain station/date coordinates; the checker rejects training CSVs whose keys have changed order.
- `checks/`: sampling CSV, summary JSON, distribution, station/weekend, tail, trace and residual PNGs, calendar-lag residual correlations, and fitted arithmetic means. Root `weekend_summary.csv` compares baseline/sensitivity summaries.
- Sensitivity checking also writes root `fitted_mean_comparison.csv`: `station_id,date,baseline_fitted_mean,wide_slopes_fitted_mean,difference`, where difference is widened minus baseline. Means are posterior expected arithmetic concentrations in µg/m³. Rows join one-to-one by station/date, preserving baseline order; mismatched, duplicate or nonfinite inputs are rejected.
- Prior mode saves `prior.nc`, `wide-slopes-prior.nc`, training data, transforms and both effective prior configurations; checking creates separate distribution plots. Prior-heldout mode saves the same artifacts per split using training rows only, with their hashes and ordered training keys in the root manifest.

Sampling passes only when every scalar parameter has finite diagnostics, rank R-hat ≤ 1.01, bulk/tail ESS ≥ 400, and zero divergences. Failed gates remain visible in summaries. Predictive adequacy always requires reviewing station, weekend, tail and temporal behavior, even if sampling passes.

Production results are pending the verified shared data, numerical priors and prior review, and agreed held-out dates. Existing preprocessing scripts remain unchanged.
