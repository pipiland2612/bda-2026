"""Sampling evidence and predictive checks, separate from scientific approval."""

import warnings
from pathlib import Path

import arviz as az
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def sampling_gate(table: pd.DataFrame, divergences: int) -> bool:
    columns = ["r_hat", "ess_bulk", "ess_tail", "mcse_mean"]
    if table.empty or not set(columns).issubset(table.columns):
        return False
    return bool(
        np.isfinite(table[columns].to_numpy()).all()
        and divergences == 0
        and table.r_hat.le(1.01).all()
        and table.ess_bulk.ge(400).all()
        and table.ess_tail.ge(400).all()
    )


def summarize_fit(fit) -> dict:
    idata = fit.idata
    table = az.summary(idata, var_names=["alpha", "beta", "sigma"], round_to="none")
    divergences = int(idata.sample_stats.diverging.sum())
    return dict(
        table=table,
        divergences=divergences,
        sampling_passed=sampling_gate(table, divergences),
        predictive_adequacy="Requires review of predictive plots and temporal residuals",
    )


def summarize_weekend(beta_draws: np.ndarray) -> dict:
    draws = np.asarray(beta_draws).ravel()
    if not draws.size or not np.isfinite(draws).all():
        raise ValueError("Weekend draws must be nonempty and finite")
    ratios = np.exp(draws)
    if not np.isfinite(ratios).all():
        raise ValueError("Weekend ratios overflow; inspect fit")
    return dict(
        ratio_median=float(np.median(ratios)),
        ratio_interval_95=np.quantile(ratios, [0.025, 0.975]).tolist(),
        percent_median=float(np.median(100 * (ratios - 1))),
        percent_interval_95=np.quantile(100 * (ratios - 1), [0.025, 0.975]).tolist(),
        probability_lower=float((draws < 0).mean()),
    )


def lognormal_mean(eta, sigma):
    return np.exp(np.asarray(eta) + np.asarray(sigma) ** 2 / 2)


def residual_lags(
    keys: pd.DataFrame, residuals: np.ndarray, max_lag_days: int = 14
) -> pd.DataFrame:
    f = keys.copy().reset_index(drop=True)
    if (
        len(f) != len(residuals)
        or f.duplicated(["station_id", "date"]).any()
        or not np.isfinite(residuals).all()
    ):
        raise ValueError(
            "Residuals require finite values and unique aligned station/date keys"
        )
    f["date"] = pd.to_datetime(f.date)
    f["residual"] = residuals
    records = []
    for station, group in f.groupby("station_id", sort=True):
        values = group.set_index("date").residual
        for lag in range(1, max_lag_days + 1):
            shifted = values.copy()
            shifted.index = shifted.index + pd.Timedelta(days=lag)
            pairs = pd.concat([values, shifted], axis=1, join="inner")
            n = len(pairs)
            valid = n >= 3 and pairs.iloc[:, 0].std() > 0 and pairs.iloc[:, 1].std() > 0
            records.append(
                dict(
                    station_id=station,
                    lag_days=lag,
                    pairs=n,
                    correlation=pairs.iloc[:, 0].corr(pairs.iloc[:, 1])
                    if valid
                    else np.nan,
                )
            )
    return pd.DataFrame(records)


def predictive_figures(
    frame: pd.DataFrame,
    values: np.ndarray,
    output: Path,
    label: str,
    prior: bool = False,
):
    output.mkdir(parents=True, exist_ok=True)
    draws = np.asarray(values).reshape(-1, len(frame))
    if not np.isfinite(draws).all() or not (draws > 0).all():
        raise ValueError("Predictive draws must be finite positive concentrations")
    fig, ax = plt.subplots(figsize=(8, 4))
    # Show concentrations on log x-scale so prior extremes remain visible.
    ax.hist(
        np.log(draws.ravel()),
        bins=50,
        density=True,
        alpha=0.6,
        label="Prior predictions" if prior else "Posterior predictions",
    )
    if not prior:
        ax.hist(
            np.log(frame.no2_daily),
            bins=25,
            density=True,
            histtype="step",
            linewidth=2,
            label="Observed",
        )
    ax.set(
        xlabel="log(NO₂ / 1 µg/m³)",
        ylabel="Density",
        title=label + " concentration distribution",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "distribution.png")
    plt.close(fig)
    if prior:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, group in zip(axes, ["station_id", "weekend"]):
        labels = (
            frame.station_id
            if group == "station_id"
            else pd.to_datetime(frame.date)
            .dt.day_of_week.ge(5)
            .map({True: "Weekend", False: "Weekday"})
        )
        names = list(dict.fromkeys(labels))
        replicated = []
        observed = []
        for name in names:
            mask = labels.eq(name).to_numpy()
            n = int(mask.sum())
            if n < 5:
                warnings.warn(
                    f"Small predictive-check group {name}: n={n}", UserWarning
                )
            replicated.append(draws[:, mask].mean(axis=1))
            observed.append(frame.loc[mask, "no2_daily"].mean())
        ax.boxplot(
            replicated,
            tick_labels=[f"{n}\n(n={labels.eq(n).sum()})" for n in names],
            showfliers=False,
        )
        ax.scatter(
            np.arange(1, len(names) + 1),
            observed,
            color="red",
            label="Observed mean",
            zorder=3,
        )
        ax.set(ylabel="Daily NO₂ mean (µg/m³)", title=group)
        ax.legend()
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle(label + " grouped predictive checks")
    fig.tight_layout()
    fig.savefig(output / "groups.png")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    tails = np.quantile(draws, [0.95, 0.99], axis=1).T
    ax.boxplot(
        [tails[:, 0], tails[:, 1]],
        tick_labels=["95th percentile", "99th percentile"],
        showfliers=False,
    )
    ax.scatter(
        [1, 2],
        np.quantile(frame.no2_daily, [0.95, 0.99]),
        color="red",
        label="Observed",
        zorder=3,
    )
    ax.set(ylabel="NO₂ (µg/m³)", title=label + " upper-tail check")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "tails.png")
    plt.close(fig)


def diagnostic_figures(idata, frame: pd.DataFrame, output: Path, label: str):
    predictive_figures(frame, idata.posterior_predictive.no2.values, output, label)
    axes = az.plot_trace(idata, var_names=["alpha", "beta", "sigma"])
    fig = np.asarray(axes).flat[0].figure
    fig.suptitle(label + " sampling traces")
    fig.tight_layout()
    fig.savefig(output / "traces.png")
    plt.close(fig)
    eta = idata.posterior.eta.mean(("chain", "draw")).values
    residuals = np.log(frame.no2_daily.to_numpy()) - eta
    residual_lags(frame[["station_id", "date"]], residuals).to_csv(
        output / "residual_lags.csv", index=False
    )
    fig, ax = plt.subplots(figsize=(10, 4))
    for station, group in frame.assign(residual=residuals).groupby("station_id"):
        group = group.sort_values("date")
        ax.plot(
            pd.to_datetime(group.date), group.residual, ".-", label=station, alpha=0.7
        )
    ax.axhline(0, color="black", linewidth=0.5)
    ax.legend()
    ax.set(ylabel="log concentration residual", title=label + " temporal residuals")
    fig.tight_layout()
    fig.savefig(output / "residuals.png")
    plt.close(fig)
    # Save conditional arithmetic means correctly, including residual variance.
    means = lognormal_mean(
        idata.posterior.eta.values, idata.posterior.sigma.values[:, :, None]
    ).mean(axis=(0, 1))
    frame[["station_id", "date"]].assign(fitted_mean=means).to_csv(
        output / "fitted_means.csv", index=False
    )
