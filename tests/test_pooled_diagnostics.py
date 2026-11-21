import numpy as np
import pandas as pd
import pytest

from models.pooled_diagnostics import (
    lognormal_mean,
    residual_lags,
    sampling_gate,
    summarize_weekend,
)


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
