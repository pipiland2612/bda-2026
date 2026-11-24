# Reproduce Minh's exploratory evaluation

These scripts reproduce the real-data checks requested after implementing Model A. They keep numerical priors explicitly provisional and leave the production CLI's approval requirements intact. They do not modify raw downloads, committed preprocessing outputs, or Model B.

Set up the model environment from [models/README.md](../../models/README.md), then run from the repository root:

```sh
# Latest committed table: 1441 usable, positive, complete rows at main 82ba18b.
rtk .venv/bin/python analysis/pooled/run_check.py --source committed

# Earlier reconstructed cohort: 1440 rows, previous-hour date assignment.
rtk .venv/bin/python analysis/pooled/run_check.py --source reconstructed

# All 20 Bayesian ablations, using only the reconstructed cohort above.
rtk .venv/bin/python analysis/pooled/bayes_ablation.py

rtk .venv/bin/python -m pytest tests/ -q
```

Each baseline command generates prior predictions and three fits: full-data baseline, January–September training fit with October–December prediction, and full-data widened-slope sensitivity. Four chains, 1000 warmup and 1000 retained draws per chain use seed 20261122; prediction uses 20261123. Add positional `prepare` to either baseline command to generate only the cohort, provenance and prior predictions.

Outputs go to separate ignored directories:

- `outputs/pooled/committed-data-check/`: committed source values and dates, selecting `usable_day=True`, positive NO₂ and finite weather/outcome. No imputation or date reconstruction. This is a diagnostic selection, not approval of source semantics/weather completeness.
- `outputs/pooled/existing-data-check/`: original raw reconstruction using `prepare_explore.py` with hourly labels shifted back one hour before daily aggregation. It uses the recorded provisional weather mapping and 75% expected-hour rule.
- `outputs/pooled/bottleneck-study/`: ablations with training-only transforms, four earlier monthly forward folds and the reused Q4 window. Seed 20261124, four chains, 1000 warmup/1000 draws. Seasonal shrinkage and fixed station offsets are experiments, not a replacement for Andrei's model.

Rerunning replaces these exploratory output files. Large NetCDF draws remain local; [compact evidence snapshots](../../reports/pooled-evaluation/README.md) and the [report](../../reports/minh-pooled-model.md) are versioned. MCMC reproducibility assumes the tested dependency/platform environment; other versions can change Monte Carlo draws.

The committed-table run copies the recorded `approved:false` numerical assumptions from `reports/pooled-evaluation/baseline-priors.json`. The reconstruction records the same assumptions directly. Neither script fabricates group approval or marks real observations synthetic. They deliberately use the low-level model API for exploration; use `scripts.fit_pooled` for the reviewed production workflow.

Predictive scores condition on observed same-day weather. Test outcomes enter scoring only; all centers and posterior parameters use training data. The latest committed table and reconstructed cohort differ in values and exclusions, so compare variants within a cohort. All investigated validation windows have supported development and cannot establish unbiased final generalization.
