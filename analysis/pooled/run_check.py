"""Reproducible exploratory real-data check, not a group-approved production run."""

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import pymc as pm
from scipy.special import logsumexp
from scipy.stats import norm

from models.pooled import (
    SamplingConfig,
    attach_keys,
    build_pooled,
    fit_pooled,
    predict_pooled,
    read_priors,
)
from models.pooled_data import make_design
from models.pooled_diagnostics import (
    diagnostic_figures,
    predictive_figures,
    summarize_fit,
    summarize_weekend,
)
from scripts.prepare_explore import prepare_no2, prepare_weather, read_file

OUT = ROOT / "outputs/pooled/existing-data-check"


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def prepare():
    aq_rows = []
    weather_rows = []
    # FMI labels these hourly values at the end of the previous hour.
    # Subtract one hour before assigning the Helsinki calendar day.
    for path in sorted((ROOT / "data/raw").glob("*.csv")):
        station, columns, rows = read_file(path)
        for row in rows:
            row["date"] = (row["timestamp"] - timedelta(hours=1)).date()
        no2 = [c for c in columns if c.lower().startswith("nitrogen dioxide")]
        if no2:
            aq_rows.extend(prepare_no2(path, station, no2[0], rows))
        else:
            weather_rows.extend(prepare_weather(station, columns, rows))
    aq = pd.DataFrame(aq_rows).replace("", np.nan)
    weather = pd.DataFrame(weather_rows).replace("", np.nan)
    numeric_aq = ["no2_daily_mean_ug_m3"]
    numeric_weather = [
        "mean_hourly_max_wind_speed_m_s",
        "mean_temperature_c",
        "sum_reported_precipitation_mm",
    ]
    aq[numeric_aq] = aq[numeric_aq].apply(pd.to_numeric)
    weather[numeric_weather] = weather[numeric_weather].apply(pd.to_numeric)
    mapping = {
        "Helsinki Mäkelänkatu": ("100762", "Helsinki Kumpula"),
        "Helsinki Kallio 2": ("100662", "Helsinki Kaisaniemi"),
        "Espoo Leppävaara Läkkisepänkuja": ("100691", "Helsinki Kaisaniemi"),
        "Vantaa Tikkurila Neilikkatie": ("100763", "Helsinki-Vantaa airport"),
    }
    aq["station_id"] = aq.station.map(lambda x: mapping[x][0])
    aq["weather_station"] = aq.station.map(lambda x: mapping[x][1])
    merged = aq.merge(weather, on=["date", "weather_station"], validate="many_to_one")
    merged = merged.rename(
        columns={
            "no2_daily_mean_ug_m3": "no2_daily",
            "mean_hourly_max_wind_speed_m_s": "wind_speed",
            "mean_temperature_c": "temperature",
            "sum_reported_precipitation_mm": "precipitation",
        }
    )
    cols = ["no2_daily", "wind_speed", "temperature", "precipitation"]
    keep = np.isfinite(merged[cols].to_numpy(float)).all(axis=1) & merged.no2_daily.gt(
        0
    )
    f = merged.loc[keep].sort_values(["station_id", "date"]).reset_index(drop=True)
    f.to_csv(OUT / "analysis.csv", index=False)
    audit = (
        merged.assign(retained=keep)
        .groupby("station")
        .agg(
            total_days=("date", "size"),
            usable_no2_days=("no2_daily", "count"),
            model_rows=("retained", "sum"),
        )
        .reset_index()
    )
    audit.to_csv(OUT / "coverage.csv", index=False)
    priors = dict(
        approved=False,
        rationale="Provisional analyst assumptions for user-requested exploratory real-data check. Not elicited, calibrated to these outcomes, or team-approved. Intercept assumes a 20 ug/m3 median at reference weather; effects allow both signs.",
        sources=[
            "docs/prior-plan.md: ratio-to-log coefficient method only; numerical ranges are analyst assumptions"
        ],
        alpha=dict(mean=float(np.log(20)), sd=0.75),
        sigma=dict(scale=0.5),
        coefficients={
            "weekend": dict(lower=0.5, upper=2.0, unit="weekend versus weekday"),
            "wind": dict(
                lower=0.5, upper=1.5, unit="+2 m/s", measure="mean_of_hourly_maxima"
            ),
            "temperature": dict(lower=0.5, upper=2.0, unit="+10 degC"),
            "rain": dict(
                lower=0.9, upper=1.1, unit="+1 mm daily reported precipitation"
            ),
            "sin": dict(lower=0.25, upper=4.0, unit="+1 annual sine"),
            "cos": dict(lower=0.25, upper=4.0, unit="+1 annual cosine"),
        },
    )
    save("priors.json", priors)
    save(
        "provenance.json",
        dict(
            synthetic=False,
            status="exploratory; no group approval claimed",
            raw_sha256={
                x.name: hashlib.sha256(x.read_bytes()).hexdigest()
                for x in (ROOT / "data/raw").glob("*.csv")
            },
            mapping={
                k: dict(station_id=v[0], weather_station=v[1])
                for k, v in mapping.items()
            },
            rows_before=len(merged),
            rows_retained=len(f),
            excluded=int((~keep).sum()),
            assumptions=[
                "Existing prepare_explore.py 75% expected local-hour coverage rule used.",
                "Hourly nonpositive NO2 retained by existing preprocessing; all qualifying daily means positive.",
                "Existing weather mapping reused provisionally, not independently geographically validated.",
                "Wind is daily average of supplied hourly maximum-speed measurements, not ordinary mean wind.",
                "FMI guidance states hourly weather and air quality values describe the previous hour; shift timestamps back one hour before assigning the calendar day. Local fall-back repeated measurements retained; completeness uses 23/24/25 expected hours.",
                "Precipitation is summed supplied hourly amounts; exact original download query and grouping settings remain to be confirmed.",
                "Raw CSVs contain no provider validity flags; original export/validation metadata remains unverified.",
                "Numerical priors provisional. Direct model API used for explicitly exploratory evaluation; production approval gates were not marked approved.",
            ],
            references=[
                "https://en.ilmatieteenlaitos.fi/guidance-to-observations",
                "https://en.ilmatieteenlaitos.fi/weather-observations",
            ],
            holdout="Train before 2024-10-01; test October–December, all stations grouped by date. Provisional diagnostic split, not final team comparison.",
        ),
    )
    return f


def load_committed(path):
    """Select a diagnostic cohort without changing teammates' preprocessing."""
    frame = pd.read_csv(path, dtype={"station_id": str, "date": str})
    if not frame.usable_day.isin([True, False]).all():
        raise ValueError("usable_day must contain explicit boolean values")
    columns = ["no2_daily", "wind_speed", "temperature", "precipitation"]
    values = frame[columns].to_numpy(dtype=float)
    keep = (
        frame.usable_day.eq(True)
        & np.isfinite(values).all(axis=1)
        & frame.no2_daily.gt(0)
    )
    audit = (
        frame.assign(retained=keep)
        .groupby("station_name")
        .agg(
            total_days=("date", "size"),
            usable_no2_days=("no2_daily", "count"),
            model_rows=("retained", "sum"),
        )
        .reset_index()
    )
    selected = frame.loc[keep].rename(columns={"station_name": "station"})
    return selected.sort_values(["station_id", "date"]).reset_index(drop=True), audit


def prepare_committed():
    source = ROOT / "data/processed/step3_analysis_table.csv"
    frame, audit = load_committed(source)
    frame.to_csv(OUT / "analysis.csv", index=False)
    audit.to_csv(OUT / "coverage.csv", index=False)
    # Reuse the recorded numerical analyst assumptions, never a synthetic fixture
    # or a fabricated approval of the group's production-prior template.
    assumptions = ROOT / "reports/pooled-evaluation/baseline-priors.json"
    save("priors.json", json.loads(assumptions.read_text()))
    save(
        "provenance.json",
        {
            "synthetic": False,
            "status": "Exploratory fit of the committed table; no group approval claimed",
            "source": str(source.relative_to(ROOT)),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "rows_before": int(audit.total_days.sum()),
            "rows_retained": len(frame),
            "excluded": int(audit.total_days.sum() - len(frame)),
            "assumptions": [
                "Keep usable_day=True, positive NO2, finite outcome and all three weather covariates; no imputation.",
                "Preserve committed date assignments and values exactly; no previous-hour reconstruction.",
                "Do not reinterpret supplied wind maxima as ordinary mean wind.",
                "Provider validity, weather coverage, timestamp and precipitation semantics are not approved by this diagnostic.",
                "Numerical priors are provisional analyst assumptions; production approval gates remain unchanged.",
            ],
            "holdout": "Train before 2024-10-01; test October–December, grouped by date. Reused development window.",
        },
    )
    return frame


def prior_check(f):
    priors = read_priors(OUT / "priors.json", allow_provisional=True)
    train = f.loc[f.date < "2024-10-01"]
    data, state = make_design(train)
    with build_pooled(data, priors):
        draws = attach_keys(
            pm.sample_prior_predictive(draws=500, random_seed=20261122), data
        )
    draws.to_netcdf(OUT / "prior.nc")
    predictive_figures(
        train,
        draws.prior_predictive.no2.values,
        OUT / "prior-checks",
        "EXPLORATORY REAL-DATA DESIGN, PROVISIONAL PRIORS",
        prior=True,
    )
    save(
        "prior-summary.json",
        dict(
            quantiles=dict(
                zip(
                    ["p01", "p50", "p99", "p999"],
                    np.quantile(
                        draws.prior_predictive.no2.values, [0.01, 0.5, 0.99, 0.999]
                    ).tolist(),
                )
            ),
            transforms=asdict(state),
        ),
    )


def fit_one(f, priors, name):
    data, state = make_design(f)
    start = time.time()
    fit = fit_pooled(
        data, priors, SamplingConfig(chains=4, warmup=1000, draws=1000, seed=20261122)
    )
    folder = OUT / name
    folder.mkdir(exist_ok=True)
    fit.idata.attrs.update(
        synthetic="False",
        status="Exploratory real-data fit; provisional priors and preparation",
    )
    fit.idata.to_netcdf(folder / "posterior.nc")
    summary = summarize_fit(fit)
    table = summary.pop("table")
    table.to_csv(folder / "sampling.csv")
    summary.update(
        weekend=summarize_weekend(
            fit.idata.posterior.beta.sel(feature="weekend").values
        ),
        rows=len(f),
        wall_seconds=time.time() - start,
        worst_rhat=float(table.r_hat.max()),
        min_bulk_ess=float(table.ess_bulk.min()),
        min_tail_ess=float(table.ess_tail.min()),
    )
    (folder / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    diagnostic_figures(
        fit.idata, f, folder / "checks", "EXPLORATORY REAL DATA: " + name
    )
    print(name, json.dumps(summary), flush=True)
    return fit, state, summary


def evaluate(fit, state, test):
    data, _ = make_design(test.drop(columns="no2_daily"), state, False)
    pred = predict_pooled(fit, data, 20261123)
    pred.to_netcdf(OUT / "holdout-predictions.nc")
    posterior = fit.idata.posterior
    eta = posterior.alpha.values[:, :, None] + np.einsum(
        "cdf,nf->cdn", posterior.beta.values, data.X
    )
    sigma = posterior.sigma.values[:, :, None]
    means = np.exp(eta + 0.5 * sigma**2).mean(axis=(0, 1))
    values = pred.no2.values.reshape(-1, len(test))
    low, high = np.quantile(values, [0.025, 0.975], axis=0)
    y = test.no2_daily.to_numpy()
    logp = norm.logpdf(np.log(y)[None, None, :], eta, sigma) - np.log(y)[None, None, :]
    lpd = logsumexp(logp.reshape(-1, len(test)), axis=0) - np.log(
        logp.shape[0] * logp.shape[1]
    )
    result = test[["station_id", "station", "date", "no2_daily"]].copy()
    result = result.assign(
        predicted_mean=means, lower_95=low, upper_95=high, log_predictive_density=lpd
    )
    result.to_csv(OUT / "holdout_predictions.csv", index=False)
    records = []
    for name, group in [("all", result)] + list(result.groupby("station")):
        truth = group.no2_daily.to_numpy()
        estimate = group.predicted_mean.to_numpy()
        records.append(
            dict(
                station=name,
                n=len(group),
                mae=float(np.abs(truth - estimate).mean()),
                rmse=float(np.sqrt(np.mean((truth - estimate) ** 2))),
                coverage_95=float(
                    ((truth >= group.lower_95) & (truth <= group.upper_95)).mean()
                ),
                mean_interval_width_95=float((group.upper_95 - group.lower_95).mean()),
                mean_log_predictive_density=float(group.log_predictive_density.mean()),
            )
        )
    pd.DataFrame(records).to_csv(OUT / "holdout_metrics.csv", index=False)
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=["prepare", "full"], default="full", nargs="?"
    )
    parser.add_argument(
        "--source", choices=["reconstructed", "committed"], default="reconstructed"
    )
    args = parser.parse_args()
    if args.source == "committed":
        OUT = ROOT / "outputs/pooled/committed-data-check"
    OUT.mkdir(parents=True, exist_ok=True)
    f = prepare_committed() if args.source == "committed" else prepare()
    prior_check(f)
    if args.action == "prepare":
        print((OUT / "prior-summary.json").read_text())
        sys.exit()
    priors = read_priors(OUT / "priors.json", allow_provisional=True)
    train = f.loc[f.date < "2024-10-01"]
    test = f.loc[f.date >= "2024-10-01"]
    full, _, full_summary = fit_one(f, priors, "full-baseline")
    fit, state, train_summary = fit_one(train, priors, "train-baseline")
    metrics = evaluate(fit, state, test)
    benchmark = []
    for label, prediction in [
        ("training_global_mean", np.repeat(train.no2_daily.mean(), len(test))),
        (
            "training_station_mean",
            test.station_id.map(
                train.groupby("station_id").no2_daily.mean()
            ).to_numpy(),
        ),
    ]:
        error = test.no2_daily.to_numpy() - prediction
        benchmark.append(
            dict(
                model=label,
                mae=float(np.abs(error).mean()),
                rmse=float(np.sqrt(np.mean(error**2))),
            )
        )
    _, _, wide_summary = fit_one(f, priors.sensitivity(), "full-wide-priors")
    save(
        "results.json",
        dict(
            full_fit=full_summary,
            train_fit=train_summary,
            wide_fit=wide_summary,
            train_rows=len(train),
            test_rows=len(test),
            holdout=metrics,
            benchmarks=benchmark,
        ),
    )
    print(json.dumps(dict(metrics=metrics, benchmarks=benchmark)), flush=True)
