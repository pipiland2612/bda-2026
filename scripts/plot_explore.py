from __future__ import annotations

import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "data" / "processed" / "no2_daily.csv"
OUTPUT_DIR = ROOT / "reports" / "figures"
REQUIRED_COLUMNS = {
    "date",
    "station",
    "no2_daily_mean_ug_m3",
    "coverage_pct",
    "meets_75pct_coverage",
}


@dataclass(frozen=True)
class DailyObservation:
    day: date
    station: str
    no2_mean: float | None
    coverage_pct: float
    meets_coverage: bool


def parse_optional_float(raw: str | None, path: Path, line_number: int, column: str) -> float | None:
    if raw is None or not raw.strip():
        return None
    try:
        value = float(raw)
    except ValueError as error:
        raise ValueError(f"{path.name}:{line_number}: invalid {column}") from error
    if not math.isfinite(value):
        raise ValueError(f"{path.name}:{line_number}: non-finite {column}")
    return value


def read_observations(path: Path) -> list[DailyObservation]:
    with path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames:
            raise ValueError(f"{path.name}: missing CSV header")
        missing_columns = REQUIRED_COLUMNS.difference(reader.fieldnames)
        if missing_columns:
            raise ValueError(f"{path.name}: missing columns {sorted(missing_columns)}")

        observations = []
        seen = set()
        for line_number, row in enumerate(reader, start=2):
            try:
                day = date.fromisoformat(row["date"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"{path.name}:{line_number}: invalid date") from error

            station = (row["station"] or "").strip()
            if not station:
                raise ValueError(f"{path.name}:{line_number}: missing station")
            key = (station, day)
            if key in seen:
                raise ValueError(f"{path.name}:{line_number}: duplicate station/date")
            seen.add(key)

            raw_meets_coverage = (row["meets_75pct_coverage"] or "").strip().lower()
            if raw_meets_coverage not in {"true", "false"}:
                raise ValueError(
                    f"{path.name}:{line_number}: invalid meets_75pct_coverage"
                )
            meets_coverage = raw_meets_coverage == "true"
            no2_mean = parse_optional_float(
                row["no2_daily_mean_ug_m3"],
                path,
                line_number,
                "no2_daily_mean_ug_m3",
            )
            if meets_coverage != (no2_mean is not None):
                raise ValueError(
                    f"{path.name}:{line_number}: NO2 value and coverage flag disagree"
                )

            coverage_pct = parse_optional_float(
                row["coverage_pct"], path, line_number, "coverage_pct"
            )
            if coverage_pct is None or not 0 <= coverage_pct <= 100:
                raise ValueError(f"{path.name}:{line_number}: invalid coverage_pct")

            observations.append(
                DailyObservation(
                    day=day,
                    station=station,
                    no2_mean=no2_mean,
                    coverage_pct=coverage_pct,
                    meets_coverage=meets_coverage,
                )
            )

    if not observations:
        raise ValueError(f"{path.name}: no daily observations")
    return observations


def save_figure(figure: plt.Figure, path: Path) -> None:
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_time_series(
    by_station: dict[str, list[DailyObservation]], output_path: Path
) -> None:
    stations = sorted(by_station)
    columns = 2
    rows = math.ceil(len(stations) / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(14, 4 * rows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    all_days = [row.day for rows_for_station in by_station.values() for row in rows_for_station]
    first_day, last_day = min(all_days), max(all_days)
    calendar = [
        first_day + timedelta(days=offset)
        for offset in range((last_day - first_day).days + 1)
    ]

    for index, station in enumerate(stations):
        axis = axes[index // columns][index % columns]
        observations_by_day = {row.day: row for row in by_station[station]}
        values = [
            observations_by_day[day].no2_mean
            if day in observations_by_day
            else None
            for day in calendar
        ]
        axis.plot(
            calendar,
            [value if value is not None else math.nan for value in values],
            linewidth=0.8,
            color=f"C{index}",
        )
        axis.set_title(station)
        axis.set_ylabel("Daily mean NO₂ (µg/m³)")
        axis.grid(axis="y", alpha=0.3)

    for index in range(len(stations), rows * columns):
        axes[index // columns][index % columns].set_visible(False)

    figure.suptitle("Daily NO₂ at each station (days meeting 75% coverage)")
    figure.tight_layout()
    save_figure(figure, output_path)


def plot_weekday_weekend(
    by_station: dict[str, list[DailyObservation]], output_path: Path
) -> None:
    stations = sorted(by_station)
    columns = 2
    rows = math.ceil(len(stations) / columns)
    figure, axes = plt.subplots(
        rows, columns, figsize=(12, 4 * rows), sharey=True, squeeze=False
    )

    for index, station in enumerate(stations):
        axis = axes[index // columns][index % columns]
        weekday = [
            row.no2_mean
            for row in by_station[station]
            if row.no2_mean is not None and row.day.weekday() < 5
        ]
        weekend = [
            row.no2_mean
            for row in by_station[station]
            if row.no2_mean is not None and row.day.weekday() >= 5
        ]
        if not weekday or not weekend:
            raise ValueError(f"{station}: need qualifying weekday and weekend days")

        boxes = axis.boxplot(
            [weekday, weekend],
            tick_labels=[
                f"Weekday\n(n={len(weekday)})",
                f"Weekend\n(n={len(weekend)})",
            ],
            patch_artist=True,
        )
        for box, color in zip(boxes["boxes"], ("#4C78A8", "#F58518")):
            box.set_facecolor(color)
            box.set_alpha(0.65)
        axis.set_title(station)
        axis.set_ylabel("Daily mean NO₂ (µg/m³)")
        axis.grid(axis="y", alpha=0.3)

    for index in range(len(stations), rows * columns):
        axes[index // columns][index % columns].set_visible(False)

    figure.suptitle("Unadjusted weekday and weekend NO₂ distributions")
    figure.tight_layout()
    save_figure(figure, output_path)


def plot_monthly_means(
    by_station: dict[str, list[DailyObservation]], output_path: Path
) -> None:
    monthly_values: dict[tuple[str, int], list[float]] = defaultdict(list)
    for station, rows in by_station.items():
        for row in rows:
            if row.no2_mean is not None:
                monthly_values[(station, row.day.month)].append(row.no2_mean)

    figure, axis = plt.subplots(figsize=(11, 6))
    for index, station in enumerate(sorted(by_station)):
        means = []
        for month in range(1, 13):
            values = monthly_values[(station, month)]
            means.append(sum(values) / len(values) if values else math.nan)
        axis.plot(
            range(1, 13),
            means,
            marker="o",
            linewidth=1.5,
            label=station,
            color=f"C{index}",
        )

    axis.set_xticks(range(1, 13))
    axis.set_xlabel("Month")
    axis.set_ylabel("Mean of qualifying daily NO₂ means (µg/m³)")
    axis.set_title("Monthly NO₂ patterns by station (descriptive)")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    save_figure(figure, output_path)


def plot_coverage(
    by_station: dict[str, list[DailyObservation]], output_path: Path
) -> None:
    stations = sorted(by_station)
    totals = [len(by_station[station]) for station in stations]
    qualifying = [
        sum(row.meets_coverage for row in by_station[station]) for station in stations
    ]
    percentages = [
        100 * count / total for count, total in zip(qualifying, totals)
    ]

    figure, axis = plt.subplots(figsize=(10, 5))
    bars = axis.barh(stations, percentages, color="#4C78A8")
    axis.invert_yaxis()
    axis.set_xlim(0, 115)
    axis.set_xlabel("Days meeting the 75% hourly-coverage rule (%)")
    axis.set_title("Daily NO₂ data coverage")
    axis.grid(axis="x", alpha=0.3)
    for bar, count, total in zip(bars, qualifying, totals):
        axis.text(
            bar.get_width() + 1,
            bar.get_y() + bar.get_height() / 2,
            f"{count}/{total} days",
            va="center",
        )
    figure.tight_layout()
    save_figure(figure, output_path)


def main() -> None:
    if not INPUT_PATH.is_file():
        raise FileNotFoundError(
            f"{INPUT_PATH} not found; run scripts/prepare_explore.py first"
        )

    observations = read_observations(INPUT_PATH)
    by_station: dict[str, list[DailyObservation]] = defaultdict(list)
    for observation in observations:
        by_station[observation.station].append(observation)
    for rows in by_station.values():
        rows.sort(key=lambda row: row.day)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_time_series(by_station, OUTPUT_DIR / "no2_daily_timeseries.png")
    plot_weekday_weekend(by_station, OUTPUT_DIR / "no2_weekday_weekend.png")
    plot_monthly_means(by_station, OUTPUT_DIR / "no2_monthly_means.png")
    plot_coverage(by_station, OUTPUT_DIR / "no2_daily_coverage.png")
    print(f"Wrote four exploratory figures to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
