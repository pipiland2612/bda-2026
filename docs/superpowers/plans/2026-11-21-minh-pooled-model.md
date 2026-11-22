# Minh's Pooled Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver Minh's reproducible pooled Bayesian NO₂ model, scientific checks, comparison exports, and an understandable methods/results note.

**Architecture:** Validate a finalized station-day table supplied by Ayreeen, create a fixed pooled design matrix, and fit a lognormal PyMC model. Keep data transforms, inference, diagnostics, and CLI orchestration separate. Export row-keyed predictive draws for Ayreeen rather than implementing her final comparison.

**Tech Stack:** Python, PyMC, NumPy, pandas, xarray, ArviZ, matplotlib, pytest; resolve compatible versions in a local virtual environment and record exact tested versions. Do not change global Python packages.

**Spec:** `docs/superpowers/specs/2026-11-21-minh-pooled-model-design.md` (approved 21 November 2026).

## Global Constraints

- Use a branch/worktree based on current `origin/main`; reviewed baseline is `2cf6934`. Recheck upstream before execution and reconcile new work before applying this plan.
- Minh owns Model A; Andrei owns Model B; Ayreeen owns preprocessing and final comparison according to the edited assignment sections.
- Do not rewrite teammates' preparation scripts, implement the hierarchical model, or choose final held-out dates for the comparison owner.
- One intercept and six common coefficients; no station effects. Use the same retained rows, outcome, transforms, seasonality, and split dates as Model B.
- Reject invalid inputs explicitly; no silent row removal, zero filling, or arbitrary log offset.
- Training-only continuous centers; scales are wind / 2 m/s, temperature / 10°C, precipitation / 1 mm. Seasonal angle is `2*pi*(day_of_year-1)/366`, restricted to 2024.
- Explicit proper priors for every parameter; do not reuse the worksheet's invented wind example as scientific evidence.
- Production posterior fitting requires the finalized data handoff, reviewed prior predictive checks, and agreed shared specification. Synthetic tests can proceed independently.
- Shell commands use `rtk`. Do not edit the Google Doc as part of execution.

## Review Focus

1. A nonfinite value or duplicate station/day must fail with an actionable error (Task 1).
2. Training/test leakage or a partially assigned date must fail rather than produce optimistic predictions (Tasks 1 and 3).
3. Maximum wind must not silently become mean wind; absent or inconsistent handoff metadata prevents production runs (Tasks 1 and 3).
4. Missing or unreviewed priors and mismatched design columns must fail before sampling (Tasks 2 and 3).
5. Predictive exports must retain observation keys, while residual lag checks must respect calendar gaps (Task 4).

## File and interface map

- Create `models/__init__.py`, `models/pooled_data.py`: data validation and transformations.
- Create `models/pooled.py`: prior parsing, model construction, fitting, prediction.
- Create `models/pooled_diagnostics.py`: diagnostic calculations and labeled plots.
- Create `scripts/fit_pooled.py`, `scripts/check_pooled.py`: entry points.
- Create `models/pooled_priors.json`: documented prior template, with no fabricated approvals.
- Create `tests/test_pooled_data.py`, `tests/test_pooled.py`, `tests/test_pooled_cli.py`, `tests/test_pooled_diagnostics.py` and synthetic fixtures.
- Create `requirements-model.txt`, `requirements-model.lock.txt`, `reports/minh-pooled-model.md`.
- Modify `.gitignore`, `models/README.md`, `docs/ai-use.md` only for this model's dependencies, generated artifacts, usage, and disclosure.

Types below live in their owning modules. `TransformState` stores three training means, fixed scales, ordered feature names, and the 2024 seasonal definition. `ModelData` stores ordered station/date keys, `X` (n by 6), and optional `y`. `PriorSpec` stores alpha parameters, six ordered coefficient mean/SD pairs, sigma scale, and rationale/approval metadata. `SamplingConfig` stores chains, warmup, draws, seed, and target acceptance. `FitBundle` stores the sampled object and metadata without hiding backend-version assumptions. The implementation selects the PyMC/ArviZ-compatible saved-data representation after checking the installed API.

### Task 1: Validate the data contract and training-only design

**Files:** `models/pooled_data.py`, `models/__init__.py`, `tests/test_pooled_data.py`, `requirements-model.txt`, `requirements-model.lock.txt`.

**Interfaces:** `load_analysis(path: Path, metadata_path: Path) -> pd.DataFrame`; `validate_splits(frame: pd.DataFrame, assignments: pd.DataFrame) -> None`; `make_design(frame: pd.DataFrame, state: TransformState | None = None, include_outcome: bool = True) -> tuple[ModelData, TransformState]`.

- [x] Create the isolated execution workspace with the worktree skill and preserve these approved planning files there. Set up `.venv`; resolve compatible modeling/test dependencies, record direct requirements and a tested lock. All commands below use `rtk .venv/bin/python`.
- [x] Write failing tests asserting rejection of absent columns, duplicate keys, malformed or non-2024 dates, NaN/infinite covariates, and nonpositive observed NO₂. Require canonical columns from the design; prediction inputs may omit NO₂.
- [x] Add tests: Saturday/Sunday flags match dates; 1 January 2024 gives sin=0 and cos=1; leap day is valid. Centers fitted to training values `[2, 4]` remain 3 when transforming a held-out value 100. Fixed ordered features are `weekend`, `wind`, `temperature`, `rain`, `sin`, `cos`.
- [x] Add split tests for unique `(split_id,date)` assignments, train/test disjointness, train dates strictly before test dates for future prediction, and complete whole-date assignment. Require shared row keys and schema/units, wind definition, matching rule, and approval status in a companion handoff JSON. Validate required metadata fields; do not invent provider approval.
- [x] Run `rtk .venv/bin/python -m pytest tests/test_pooled_data.py -q`; confirm meaningful failures, then implement the three interfaces and dataclasses. Compute centers only when `state` is absent; keep ordering stable and forbid empty data partitions.
- [x] Rerun the tests; expect all to pass. Commit the focused contract and environment changes.

### Task 2: Specify priors and build the pooled model

**Files:** `models/pooled.py`, `models/pooled_priors.json`, `tests/test_pooled.py`.

**Interfaces:** consumes `ModelData`; produces `read_priors(path: Path, allow_provisional: bool = False) -> PriorSpec`, `ratio_prior(lower: float, upper: float) -> tuple[float,float]`, `build_pooled(data: ModelData, priors: PriorSpec) -> pm.Model`.

- [x] Write failing tests for `0 < lower < upper`, finite positive prior SDs, all six named slopes, explicit alpha and sigma priors, and required rationale/approval fields. Check that `ratio_prior(0.7,1.1)` matches the worksheet formula as an arithmetic test only.
- [x] Add structural tests that the model has alpha, six coefficients, positive sigma, and a lognormal observation distribution, with no station effects. Assert design column order matches prior order. Check predicted log location against a hand-calculated linear predictor using known coefficients.
- [x] Run `rtk .venv/bin/python -m pytest tests/test_pooled.py -q`; verify failures, then implement explicit Normal alpha/coefficient priors and HalfNormal sigma, with parameters supplied through `PriorSpec`. Never rely on implicit software priors.
- [x] Add a prior-template JSON containing named parameter fields, units, provenance, approval flags, and explanatory nulls for unelicited numbers. Its validation must clearly explain what to fill before production fitting. Use separate explicitly synthetic numeric fixtures for tests.
- [x] Test a small fixed-seed prior predictive run for positive finite outcomes, expected dimensions, and preserved row keys. Run all Task 1–2 tests and commit.

### Task 3: Fit, predict, and record reproducible runs

**Files:** `models/pooled.py`, `scripts/fit_pooled.py`, `tests/test_pooled_cli.py`, `.gitignore`.

**Interfaces:** `fit_pooled(data: ModelData, priors: PriorSpec, config: SamplingConfig) -> FitBundle`; `predict_pooled(fit: FitBundle, new_data: ModelData, seed: int) -> xr.Dataset`; CLI module `python -m scripts.fit_pooled`.

- [x] Add failing CLI tests for missing approvals, absent handoff metadata, missing split assignments in held-out mode, overlapping output directories, and prediction inputs that incorrectly expose held-out outcomes to the fitting interface. Assert invalid requests fail before sampling.
- [x] Implement CLI flags `--data`, `--metadata`, `--priors`, `--output`, `--mode prior|prior-heldout|posterior|heldout|sensitivity`, `--splits` for held-out runs, and `--synthetic` for explicitly labeled demonstrations. Production posterior modes also require `--prior-review` pointing to a JSON record containing `reviewed_by`, `reviewed_at`, `prior_sha256`, `prior_run_manifest_sha256`, and `accepted: true`; reject mismatched hashes.
- [x] Implement fixed initial settings of 4 chains, 1,000 warmup, 1,000 retained draws, seed 20261121, and target acceptance 0.9, with explicit CLI overrides recorded in manifests. First use small settings for synthetic integration checks; never claim those checks meet scientific convergence criteria.
- [x] Save input/config hashes, seed, versions, sampling settings, ordered station/date keys, training transforms, and run status. Keep saved posterior/prior draws separate. Held-out mode fits only training rows, strips outcomes from prediction data, and exports chain/draw/observation predictions labeled with split, station, and date. Require each test key exactly once per split.
- [x] Implement sensitivity using doubled slope prior SDs with means, alpha, and sigma priors fixed. Save baseline and sensitivity configs and separate runs.
- [x] Add a synthetic sampling/prediction integration test with expected shape, finite positive predictions, and matching keys. Run `rtk .venv/bin/python -m pytest tests/test_pooled_cli.py tests/test_pooled.py -q`; commit after passing. Ignore `.venv/`, model run directories, and large draw files without changing raw-data tracking rules.

### Task 4: Diagnostics and scientifically correct summaries

**Files:** `models/pooled_diagnostics.py`, `scripts/check_pooled.py`, `tests/test_pooled_diagnostics.py`.

**Interfaces:** `summarize_fit(fit: FitBundle) -> dict`; `summarize_weekend(beta_draws: np.ndarray) -> dict`; `residual_lags(keys: pd.DataFrame, residuals: np.ndarray, max_lag_days: int = 14) -> pd.DataFrame`; CLI `python -m scripts.check_pooled --run RUN_DIR`.

- [x] Write failing tests that mocked R-hat above 1.01, any divergence, or bulk/tail ESS below 400 generates a failed sampling gate. Missing/nonfinite diagnostics also fail. Preserve numerical diagnostics rather than replacing them with a single pass label.
- [x] Add exact arithmetic tests for `exp(beta)`, `100*(exp(beta)-1)`, equal-tailed 95% credible intervals, and `P(beta<0)`. Verify lognormal means use `exp(eta + sigma**2/2)`, distinct from medians. Test that draws are transformed before intervals are calculated.
- [x] Add residual tests where 1 and 3 January do not form a one-day lag; pair only the same station and actual lag dates, reporting pair counts and undefined correlations. Add export alignment tests on intentionally shuffled row keys.
- [x] Implement scalar summaries, MCSE, rank R-hat, bulk/tail ESS, divergences, trace plots, observed/replicated distribution and tail plots, station/weekend checks, and calendar-lag residual summaries. Compare baseline and sensitivity weekend summaries and predictions. Ensure finite dimensions and warn about inadequate group sample sizes.
- [x] Run `rtk .venv/bin/python -m pytest tests/test_pooled_diagnostics.py -q`; expect all to pass. Generate a synthetic diagnostic bundle and verify its plots render. Label sampling and predictive adequacy separately; the latter needs scientific review and must not be inferred from R-hat alone. Commit.

### Task 5: Execute the approved scientific workflow when inputs are ready

**Files:** finalized run artifacts under ignored `outputs/pooled/`; `models/pooled_priors.json` after actual elicitation; `reports/minh-pooled-model.md`.

**Interfaces:** consumes finalized handoff CSV/JSON, common retained-row manifest, comparison-owner date-split CSV (`split_id,date,partition`, where partition is `train` or `test`), approved priors, and shared transformation agreement.

- [ ] Verify the handoff with Ayreeen's actual deliverables, including maximum-versus-mean wind and precipitation semantics. Use an adapter only for documented column renaming; send source/aggregation issues back to the owner. Record any still-pending dependency and continue only synthetic software checks while it is unresolved.
- [ ] Elicit and document numeric priors with the group using concentration and ratio ranges; distinguish their scientific justification from developer-proposed values. Do not mark the template or synthetic fixture as approved evidence.
- [ ] Run prior mode with the finalized design and priors; inspect concentration extremes, station/weather contexts, and weekend differences. Document revisions and record review hashes before posterior fitting.
- [ ] Run baseline posterior mode, then `rtk .venv/bin/python -m scripts.check_pooled --run outputs/pooled/posterior`. Inspect sampling diagnostics and predictive plots. If checks fail, investigate before interpreting; coordinate model extensions with Andrei.
- [ ] Run sensitivity and agreed held-out splits, inspect outputs, and deliver draws plus keys/manifests to Ayreeen. Record unresolved predictive mismatch, temporal dependence, and station pooling limitations even when sampling passes.
- [ ] Mark this task complete only with real-data run evidence. If handoffs remain unavailable, report implementation as software-ready and this task as pending; never substitute synthetic findings.

### Task 6: Minh's explanation and teammate handoff

**Files:** `reports/minh-pooled-model.md`, `models/README.md`, `docs/ai-use.md`.

- [x] Document exact setup/run commands and the input/output contract, including the canonical schema, metadata requirements, split schema, draw coordinates, reviewed-prior gate, and pinned environment.
- [x] Write Minh's equation, predictor units, prior rationale, fitting settings, diagnostics, posterior checks, sensitivity, and weekend interpretation. Insert numerical results only after Task 5; otherwise label the results section pending.
- [x] Include a short presentation outline: pooled-model purpose, how the prior/checks work, weekend ratio and uncertainty, and limitations. Explain that this model cannot estimate station-specific weekend effects and that the association is not a causal traffic estimate.
- [x] Add an accurate AI-use entry describing planning, coding, testing, and any completed analysis, with human verification responsibilities. Do not imply teammates approved choices or reviewed results without evidence.
- [x] Run `rtk .venv/bin/python -m pytest tests/ -q` and `rtk git diff --check`. Review the complete diff for teammate-scope changes; perform the applicable independent code-review workflow. Commit verified work, report what passed and what remains pending, and offer the concrete branch for integration without merging automatically.

## Plan self-review

Coverage checked against the approved design: ownership, handoff gates, priors, model formula, transforms, fitting, prediction, diagnostics, sensitivity, result interpretation, disclosure, and validation all map to the tasks above. Interfaces and feature ordering are consistent. Each Review Focus item has a named test requirement. No production data readiness or scientific approval is presumed.

Recommended execution: native implementation in this session, with independent branch review at the end. The tasks share a small data/model interface, and this avoids repeated implementer handoffs. Subagent-driven execution remains available if Minh prefers per-task independent review.

## Execution status

Tasks 1–4 implemented and tested. Task 5 remains pending the actual verified data/metadata, agreed numerical priors, prior predictive review and comparison split dates. Task 6 documentation is written and the independent review findings have been addressed. Full suite after parallel PR review: 47 tests passed. Review hardening adds explicit wind measure matching, strict split schemas, hash-bound baseline/widened prior reviews, and observation keys in all predictive exports. PR review on 22 November resolved the former deferred comparison table: sensitivity checking now exports a station/date-keyed fitted_mean_comparison.csv. Split-specific prior predictions and review bindings are required before held-out fits; ordinary full-data prior review cannot authorize a differently centered split. No scientific results have been claimed from synthetic demonstrations.
