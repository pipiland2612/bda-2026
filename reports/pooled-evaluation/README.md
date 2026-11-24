# Evidence index

Real-data exploratory snapshots, published 24 November 2026. See [Minh's report](../minh-pooled-model.md) and [reproduction commands](../../analysis/pooled/README.md). Numerical priors remain unapproved; no final team handoff or independent validation is implied.

| Prefix/files | Cohort and purpose |
| --- | --- |
| `committed-*` | Latest `data/processed/step3_analysis_table.csv` at main `82ba18b`, filtered to 1441 rows. `committed-provenance.json` records the input hash and exact selection. Q4: 350 rows. |
| `baseline-*` | Earlier reconstruction from seven raw downloads, 1440 rows. Raw hashes, station/weather mapping and interval/date assumptions are in `baseline-provenance.json`. Q4: 349 rows. |
| `full-baseline-sampling.csv`, `train-baseline-sampling.csv`, `full-wide-priors-sampling.csv` | Sampling summaries for the earlier reconstructed cohort. The corresponding latest-table summaries have `committed-` prefixes. |
| `bayesian_results.csv`, `bayesian_forward_summary.csv` | All 20 reconstructed-cohort ablations and the row-weighted June–September aggregation. Each experiment is identified by variant/fold; 488 earlier test station-days in total. |
| `bias_decomposition.json`, `improved_training_residual_lags.csv` | Reconstructed-cohort station/overall bias decomposition and remaining calendar-day residual dependence. |

For both baseline cohorts, `*-results.json` contains full-data/training/sensitivity diagnostics, weekend summaries and training-mean benchmarks. `*-holdout_predictions.csv` contains one row per station/date, observed NO₂, posterior arithmetic mean, 95% predictive limits and log predictive density. `*-holdout_metrics.csv` gives overall and station metrics. `*-coverage.csv` records selection counts; `*-prior-summary.json` records prior-predictive quantiles and training transforms.

`baseline-priors.json` is the explicit numerical assumption record shared by these explorations; it is not the production approval template. Broad prior tails still require scientific review. The committed table preserves its original date assignments; the reconstructed cohort shifts interval-end timestamps before aggregation. Their differing daily values are intentional evidence of an unresolved preprocessing agreement, not interchangeable copies of the same analysis table.

Posterior draws and per-ablation keyed predictions can be regenerated under ignored `outputs/pooled/`; they are not included as large repository binaries. No claim of operational weather forecasting, causal traffic identification or independent final comparison is made.
