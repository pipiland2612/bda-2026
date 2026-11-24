"""Protect the cohort boundary for the committed-table diagnostic."""

import numpy as np
import pandas as pd

from analysis.pooled import run_check


def test_committed_cohort_requires_usable_positive_complete_rows(tmp_path):
    # Dropping the usable-day requirement, retaining infinity, or silently
    # imputing a missing covariate would change this scientific cohort.
    source = pd.DataFrame(
        {
            "station_id": ["100662"] * 6,
            "station_name": ["Kallio"] * 6,
            "date": [f"2024-01-{day:02}" for day in range(1, 7)],
            "usable_day": [True, False, True, True, True, True],
            "no2_daily": [10.0, 11.0, 12.0, 0.0, -1.0, 15.0],
            "wind_speed": [2.0, 2.0, np.inf, 2.0, 2.0, 2.0],
            "temperature": [0.0] * 6,
            "precipitation": [0.0, 0.0, 0.0, 0.0, 0.0, np.nan],
        }
    )
    path = tmp_path / "committed.csv"
    source.to_csv(path, index=False)
    selected, audit = run_check.load_committed(path)
    assert selected[["station_id", "date"]].values.tolist() == [
        ["100662", "2024-01-01"]
    ]
    assert selected.no2_daily.tolist() == [10.0]
    assert selected.station.tolist() == ["Kallio"]
    assert audit["total_days"].sum() == 6
    assert audit["model_rows"].sum() == 1
    pd.testing.assert_frame_equal(pd.read_csv(path), source.assign(station_id=100662))
