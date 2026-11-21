"""Strict model-input contract; source cleaning belongs to preprocessing."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = ("weekend", "wind", "temperature", "rain", "sin", "cos")
WEATHER = ("wind_speed", "temperature", "precipitation")
SCALES = (2.0, 10.0, 1.0)


@dataclass(frozen=True)
class TransformState:
    centers: tuple[float, float, float]
    scales: tuple[float, float, float] = SCALES
    features: tuple[str, ...] = FEATURES
    year: int = 2024


@dataclass
class ModelData:
    keys: pd.DataFrame
    X: np.ndarray
    y: np.ndarray | None
    features: tuple[str, ...] = FEATURES


def checked_frame(frame: pd.DataFrame, include_outcome: bool = True) -> pd.DataFrame:
    required = {"station_id", "date", *WEATHER}
    if include_outcome:
        required.add("no2_daily")
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    f = frame.copy().reset_index(drop=True)
    if (
        f.empty
        or f.station_id.isna().any()
        or f.station_id.astype(str).str.strip().eq("").any()
    ):
        raise ValueError("Empty input or missing station_id")
    f["station_id"] = f.station_id.astype(str)
    try:
        dates = pd.to_datetime(f.date, format="%Y-%m-%d", errors="raise")
    except (ValueError, TypeError) as error:
        raise ValueError("date must use YYYY-MM-DD") from error
    if dates.isna().any() or not dates.dt.year.eq(2024).all():
        raise ValueError("Only valid 2024 dates are supported")
    f["date"] = dates.dt.strftime("%Y-%m-%d")
    if f.duplicated(["station_id", "date"]).any():
        raise ValueError("Duplicate station_id/date keys")
    columns = list(WEATHER) + (["no2_daily"] if include_outcome else [])
    try:
        f[columns] = f[columns].apply(pd.to_numeric, errors="raise")
        finite = np.isfinite(f[columns].to_numpy(dtype=float)).all()
    except (ValueError, TypeError) as error:
        raise ValueError("Model values must be numeric and finite") from error
    if not finite:
        raise ValueError(
            "Missing or nonfinite model values; return to preprocessing owner"
        )
    if include_outcome and (f.no2_daily <= 0).any():
        raise ValueError(
            "Lognormal requires strictly positive no2_daily; do not add an offset"
        )
    return f


def load_analysis(path: Path, metadata_path: Path) -> pd.DataFrame:
    f = checked_frame(pd.read_csv(path, dtype={"station_id": str}))
    meta = json.loads(metadata_path.read_text())
    if meta.get("approved") is not True:
        raise ValueError("Data handoff must be approved by its owner")
    for name in (
        "wind_definition",
        "weather_matching",
        "missing_data_rule",
        "source_validation",
        "shared_specification",
    ):
        if not isinstance(meta.get(name), str) or not meta[name].strip():
            raise ValueError(f"Missing handoff metadata: {name}")
    expected = dict(
        no2_daily="ug/m3", wind_speed="m/s", temperature="degC", precipitation="mm"
    )
    if meta.get("units") != expected:
        raise ValueError(f"Documented units must match {expected}")
    actual = f[["station_id", "date"]].values.tolist()
    if meta.get("retained_keys") != actual:
        raise ValueError("Ordered retained_keys do not match the shared analysis table")
    return f


def make_design(
    frame: pd.DataFrame,
    state: TransformState | None = None,
    include_outcome: bool = True,
) -> tuple[ModelData, TransformState]:
    f = checked_frame(frame, include_outcome)
    weather = f[list(WEATHER)].to_numpy(dtype=float)
    if state is None:
        state = TransformState(tuple(weather.mean(axis=0)))
    if (
        state.features != FEATURES
        or state.scales != SCALES
        or state.year != 2024
        or len(state.centers) != 3
        or not np.isfinite(state.centers).all()
    ):
        raise ValueError("Transform state does not match the shared specification")
    dates = pd.to_datetime(f.date)
    angle = 2 * np.pi * (dates.dt.day_of_year.to_numpy() - 1) / 366
    X = np.column_stack(
        (
            (dates.dt.day_of_week >= 5).astype(float),
            (weather - state.centers) / state.scales,
            np.sin(angle),
            np.cos(angle),
        )
    )
    y = f.no2_daily.to_numpy(dtype=float) if include_outcome else None
    return ModelData(f[["station_id", "date"]], X, y), state


def validate_splits(frame: pd.DataFrame, assignments: pd.DataFrame) -> None:
    if (
        not {"split_id", "date", "partition"}.issubset(assignments.columns)
        or assignments.empty
    ):
        raise ValueError("Splits need split_id, date, partition")
    a = assignments.copy()
    if a.isna().any().any() or a.duplicated(["split_id", "date"]).any():
        raise ValueError("Missing or duplicate split/date assignments")
    if not a.partition.isin(["train", "test"]).all():
        raise ValueError("Partition must be train or test")
    dates = set(frame.date.astype(str))
    for _, split in a.groupby("split_id"):
        if set(split.date.astype(str)) != dates:
            raise ValueError(
                "Each split must assign every available date, across all stations"
            )
        train, test = (
            split.loc[split.partition.eq(p), "date"].astype(str)
            for p in ("train", "test")
        )
        if train.empty or test.empty or train.max() >= test.min():
            raise ValueError(
                "Need nonempty, disjoint training dates strictly before test dates"
            )
