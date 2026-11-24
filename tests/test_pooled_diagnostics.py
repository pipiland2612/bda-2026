import json

import arviz as az
import numpy as np
import pandas as pd
import pytest
import xarray as xr

import models.pooled_diagnostics as diagnostics
from models.pooled_diagnostics import (
    lognormal_mean,
    residual_lags,
    sampling_gate,
    summarize_weekend,
)
from scripts.check_pooled import check_run


def test_sampling_gate_rejects_bad_or_undefined_diagnostics():
    good = pd.DataFrame(
        dict(r_hat=[1.0], ess_bulk=[800.0], ess_tail=[700.0], mcse_mean=[0.01])
    )
    assert sampling_gate(good, 0)
    for col, val in [
        ("r_hat", 1.02),
        ("r_hat", np.nan),
        ("ess_bulk", 399),
        ("ess_tail", 399),
    ]:
        bad = good.copy()
        bad.loc[0, col] = val
        assert not sampling_gate(bad, 0)
    assert not sampling_gate(good, 1)


def test_weekend_ratios_intervals_and_mean_are_transformed_correctly():
    result = summarize_weekend(np.log([0.5, 1.0, 2.0, 4.0]))
    assert result["probability_lower"] == 0.25
    assert result["ratio_median"] == pytest.approx(1.5)
    assert result["ratio_interval_95"] == pytest.approx([0.5375, 3.85])
    assert result["percent_interval_95"] == pytest.approx([-46.25, 285.0])
    assert lognormal_mean(0, 1) == pytest.approx(1.6487212707)


def test_residual_lags_respect_station_calendar_gaps():
    keys = pd.DataFrame(
        dict(
            station_id=["a", "a", "a", "b", "b", "b"],
            date=["2024-01-01", "2024-01-03", "2024-01-05"] * 2,
        )
    )
    out = residual_lags(keys, np.array([1.0, 2.0, 3.0, 3.0, 2.0, 1.0]), 2)
    assert out.loc[out.lag_days.eq(1), "pairs"].tolist() == [0, 0]
    assert out.loc[out.lag_days.eq(2), "pairs"].tolist() == [2, 2]
    # Insufficient pairs produce undefined correlations rather than false zeros.
    assert out.correlation.isna().all()


def test_residual_lags_do_not_depend_on_row_order():
    keys = pd.DataFrame(
        dict(
            station_id=["a"] * 4,
            date=["2024-01-04", "2024-01-01", "2024-01-03", "2024-01-02"],
        )
    )
    out = residual_lags(keys, np.array([4.0, 1.0, 3.0, 2.0]), 1)
    assert out.pairs.iloc[0] == 3 and out.correlation.iloc[0] == pytest.approx(1.0)


def fitted_mean_tables():
    baseline = pd.DataFrame(
        {
            "station_id": ["002", "001", "002"],
            "date": ["2024-01-02", "2024-01-01", "2024-01-01"],
            "fitted_mean": [10.0, 25.0, 8.0],
        }
    )
    wide = pd.DataFrame(
        {
            "station_id": ["002", "002", "001"],
            "date": ["2024-01-01", "2024-01-02", "2024-01-01"],
            "fitted_mean": [7.5, 12.0, 20.0],
        }
    )
    return baseline, wide


def test_fitted_mean_comparison_joins_keys_and_preserves_baseline_order():
    baseline, wide = fitted_mean_tables()
    result = diagnostics.compare_fitted_means(baseline, wide)
    expected = pd.DataFrame(
        {
            "station_id": ["002", "001", "002"],
            "date": ["2024-01-02", "2024-01-01", "2024-01-01"],
            "baseline_fitted_mean": [10.0, 25.0, 8.0],
            "wide_slopes_fitted_mean": [12.0, 20.0, 7.5],
            "difference": [2.0, -5.0, -0.5],
        }
    )
    pd.testing.assert_frame_equal(result, expected)


@pytest.mark.parametrize("side", [0, 1])
@pytest.mark.parametrize("problem", ["missing", "duplicate", "nonfinite"])
def test_fitted_mean_comparison_rejects_invalid_tables(side, problem):
    tables = list(fitted_mean_tables())
    if problem == "missing":
        tables[side] = tables[side].iloc[:-1]
    elif problem == "duplicate":
        tables[side] = pd.concat([tables[side], tables[side].iloc[:1]])
    else:
        tables[side].loc[0, "fitted_mean"] = np.inf
    with pytest.raises(ValueError):
        diagnostics.compare_fitted_means(*tables)


def test_checker_renders_both_prior_configurations_for_every_heldout_split(tmp_path):
    splits = [
        {"split_id": "early", "directory": "split-000"},
        {"split_id": "later", "directory": "split-001"},
    ]
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "mode": "prior-heldout",
                "synthetic": True,
                "status": "complete",
                "splits": splits,
            }
        )
    )
    for index, split in enumerate(splits):
        folder = tmp_path / split["directory"]
        folder.mkdir()
        date = "2024-01-01" if index == 0 else "2024-01-02"
        pd.DataFrame({"station_id": ["001", "002"], "date": [date, date]}).to_csv(
            folder / "training.csv", index=False
        )
        predictive = xr.Dataset(
            {
                "no2": (
                    ("chain", "draw", "observation"),
                    [[[8.0, 11.0], [9.0, 14.0], [12.0, 10.0]]],
                )
            },
            coords={
                "station_id": ("observation", ["001", "002"]),
                "date": ("observation", [date, date]),
            },
        )
        for name in ("prior.nc", "wide-slopes-prior.nc"):
            az.InferenceData(prior_predictive=predictive).to_netcdf(folder / name)
    check_run(tmp_path)
    for split in splits:
        checks = tmp_path / split["directory"] / "checks"
        assert (checks / "distribution.png").stat().st_size > 1000
        assert (checks / "wide-slopes" / "distribution.png").stat().st_size > 1000
        summary = json.loads((checks / "summary.json").read_text())
        assert summary["synthetic"] is True
        assert "prior_predictive_review" in summary
