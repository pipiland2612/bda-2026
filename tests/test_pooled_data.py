import json

import numpy as np
import pandas as pd
import pytest

from models.pooled_data import load_analysis, make_design, validate_splits


def frame():
    return pd.DataFrame(
        dict(
            station_id=["a", "a", "b", "b"],
            date=["2024-01-01", "2024-01-06"] * 2,
            no2_daily=[10.0, 20.0, 30.0, 40.0],
            wind_speed=[2.0, 4.0, 2.0, 4.0],
            temperature=[0.0, 10.0, 0.0, 10.0],
            precipitation=[0.0, 1.0, 0.0, 1.0],
        )
    )


def metadata():
    return dict(
        approved=True,
        units=dict(
            no2_daily="ug/m3", wind_speed="m/s", temperature="degC", precipitation="mm"
        ),
        wind_definition="daily mean of hourly maximum wind speed",
        weather_matching="verified mapping",
        missing_data_rule="owner-approved complete rows",
        source_validation="reviewed",
        shared_specification="agreed with Model B",
        retained_keys=[
            ["a", "2024-01-01"],
            ["a", "2024-01-06"],
            ["b", "2024-01-01"],
            ["b", "2024-01-06"],
        ],
    )


def test_load_requires_verified_metadata_and_preserves_station_ids(tmp_path):
    p, m = tmp_path / "data.csv", tmp_path / "metadata.json"
    frame().to_csv(p, index=False)
    m.write_text(json.dumps(metadata()))
    assert load_analysis(p, m).station_id.tolist() == ["a", "a", "b", "b"]
    bad = metadata()
    bad["approved"] = False
    m.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="approved"):
        load_analysis(p, m)


@pytest.mark.parametrize(
    "mutation",
    ["missing", "duplicate", "zero", "negative", "nan", "inf", "year", "date"],
)
def test_invalid_model_rows_fail(mutation):
    f = frame()
    if mutation == "missing":
        f = f.drop(columns="precipitation")
    if mutation == "duplicate":
        f = pd.concat([f, f.iloc[[0]]])
    if mutation == "zero":
        f.loc[0, "no2_daily"] = 0
    if mutation == "negative":
        f.loc[0, "no2_daily"] = -1
    if mutation == "nan":
        f.loc[0, "wind_speed"] = np.nan
    if mutation == "inf":
        f.loc[0, "temperature"] = np.inf
    if mutation == "year":
        f.loc[0, "date"] = "2023-01-01"
    if mutation == "date":
        f.loc[0, "date"] = "not-a-date"
    with pytest.raises(ValueError):
        make_design(f)


def test_design_uses_training_centers_and_calendar():
    f = frame().iloc[:2]
    data, state = make_design(f)
    np.testing.assert_allclose(data.X[:, 0], [0, 1])
    np.testing.assert_allclose(data.X[0, 4:], [0, 1])
    test = f.iloc[[0]].copy()
    test["wind_speed"] = 100.0
    test["date"] = "2024-02-29"
    pred, s = make_design(test.drop(columns="no2_daily"), state, include_outcome=False)
    assert pred.y is None and s == state
    assert pred.X[0, 1] == 48.5
    assert pred.keys.date.iloc[0] == "2024-02-29"


def test_split_dates_are_whole_disjoint_future_blocks():
    f = frame()
    split = pd.DataFrame(
        dict(
            split_id=["s", "s"],
            date=["2024-01-01", "2024-01-06"],
            partition=["train", "test"],
        )
    )
    validate_splits(f, split)
    for bad in [
        pd.concat([split, split.iloc[[0]]]),
        split.iloc[:1],
        split.assign(partition=["test", "train"]),
    ]:
        with pytest.raises(ValueError):
            validate_splits(f, bad)


def test_retained_keys_and_wind_metadata_are_enforced(tmp_path):
    p, m = tmp_path / "data.csv", tmp_path / "metadata.json"
    frame().to_csv(p, index=False)
    for bad in [
        dict(metadata(), wind_definition=""),
        dict(metadata(), retained_keys=[]),
    ]:
        m.write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            load_analysis(p, m)
