"""Exploratory Bayesian ablations, isolated from production and Model B."""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from models.pooled import read_priors
from models.pooled_data import make_design
from models.pooled_diagnostics import sampling_gate

OUT = ROOT / "outputs/pooled/bottleneck-study"
OUT.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "outputs/pooled/existing-data-check"
f = pd.read_csv(DATA / "analysis.csv", dtype={"station_id": str})
priors = read_priors(DATA / "priors.json", allow_provisional=True)


def run(variant, fold, start, end):
    train = f[f.date < start]
    test = f[(f.date >= start) & (f.date < end)]
    training, state = make_design(train)
    testing, _ = make_design(test.drop(columns="no2_daily"), state, False)
    indices = np.arange(
        4
        if variant in ("weather_only", "station_offsets_weather_only_diagnostic")
        else 6
    )
    prior_sd = np.asarray(priors.beta_sds)[indices].copy()
    if "regularized_season" in variant:
        prior_sd[4:] = 0.05
    stations = sorted(train.station_id.unique())
    site = variant.startswith("station_offsets")
    train_site = np.column_stack(
        [train.station_id.eq(s).astype(float) for s in stations[1:]]
    )
    test_site = np.column_stack(
        [test.station_id.eq(s).astype(float) for s in stations[1:]]
    )
    folder = OUT / "bayesian" / variant / fold
    folder.mkdir(parents=True, exist_ok=True)
    print("START", variant, fold, len(train), len(test), flush=True)
    start_time = time.time()
    with pm.Model(
        coords={
            "feature": np.asarray(training.features)[indices],
            "site_offset": stations[1:],
        }
    ) as model:
        alpha = pm.Normal("alpha", priors.alpha_mean, priors.alpha_sd)
        beta = pm.Normal(
            "beta", np.asarray(priors.beta_means)[indices], prior_sd, dims="feature"
        )
        sigma = pm.HalfNormal("sigma", priors.sigma_scale)
        eta = alpha + pm.math.dot(training.X[:, indices], beta)
        if site:
            offset = pm.Normal("station_offset", 0, 0.5, dims="site_offset")
            eta = eta + pm.math.dot(train_site, offset)
        pm.LogNormal("no2", eta, sigma, observed=training.y)
        idata = pm.sample(
            chains=4,
            cores=1,
            draws=1000,
            tune=1000,
            random_seed=20261124,
            target_accept=0.9,
            progressbar=False,
        )
    idata.to_netcdf(folder / "posterior.nc")
    post = idata.posterior
    eta = post.alpha.values[:, :, None] + np.einsum(
        "cdf,nf->cdn", post.beta.values, testing.X[:, indices]
    )
    if site:
        eta += np.einsum("cds,ns->cdn", post.station_offset.values, test_site)
    sigma = post.sigma.values[:, :, None]
    means = np.exp(eta + 0.5 * sigma**2).mean(axis=(0, 1))
    pred = np.random.default_rng(20261124).lognormal(eta, sigma).reshape(-1, len(test))
    lower, upper = np.quantile(pred, [0.025, 0.975], axis=0)
    result = test[["station_id", "station", "date", "no2_daily"]].assign(
        predicted_mean=means, lower_95=lower, upper_95=upper
    )
    result.to_csv(folder / "predictions.csv", index=False)
    names = ["alpha", "beta", "sigma"] + (["station_offset"] if site else [])
    table = az.summary(idata, var_names=names, round_to="none")
    table.to_csv(folder / "sampling.csv")
    y = test.no2_daily.to_numpy()
    error = means - y
    div = int(idata.sample_stats.diverging.sum())
    record = dict(
        variant=variant,
        fold=fold,
        n=len(test),
        mae=float(np.abs(error).mean()),
        rmse=float(np.sqrt((error**2).mean())),
        sse=float((error**2).sum()),
        sae=float(np.abs(error).sum()),
        coverage_95=float(((y >= lower) & (y <= upper)).mean()),
        interval_width_95=float((upper - lower).mean()),
        divergences=div,
        worst_rhat=float(table.r_hat.max()),
        min_bulk_ess=float(table.ess_bulk.min()),
        min_tail_ess=float(table.ess_tail.min()),
        sampling_passed=sampling_gate(table, div),
        elapsed_seconds=time.time() - start_time,
    )
    (folder / "config.json").write_text(
        json.dumps(
            dict(
                variant=variant,
                centers=state.centers,
                features=list(np.asarray(training.features)[indices]),
                prior_means=list(np.asarray(priors.beta_means)[indices]),
                prior_sds=list(prior_sd),
                station_offset_prior_sd=0.5 if site else None,
                status="Exploratory ablation; not production approved",
            ),
            indent=2,
        )
    )
    (folder / "summary.json").write_text(json.dumps(record, indent=2))
    print("RESULT", json.dumps(record), flush=True)
    return record


if __name__ == "__main__":
    folds = [
        ("June", "2024-06-01", "2024-07-01"),
        ("July", "2024-07-01", "2024-08-01"),
        ("August", "2024-08-01", "2024-09-01"),
        ("September", "2024-09-01", "2024-10-01"),
    ]
    records = []
    # Earlier folds establish whether changes help before Q4. The Q4 window has
    # already been examined, so its results remain diagnostic rather than final.
    for variant in [
        "original",
        "regularized_season",
        "weather_only",
        "station_offsets_regularized_season_diagnostic",
    ]:
        for fold, start, end in folds:
            records.append(run(variant, fold, start, end))
            pd.DataFrame(records).to_csv(OUT / "bayesian_results.csv", index=False)
    for variant in [
        "regularized_season",
        "weather_only",
        "station_offsets_weather_only_diagnostic",
        "station_offsets_regularized_season_diagnostic",
    ]:
        records.append(run(variant, "Q4_reused_diagnostic", "2024-10-01", "2025-01-01"))
        pd.DataFrame(records).to_csv(OUT / "bayesian_results.csv", index=False)
    r = pd.DataFrame(records)
    s = (
        r[r.fold.ne("Q4_reused_diagnostic")]
        .groupby("variant")[["n", "sse", "sae"]]
        .sum()
    )
    s["rmse"] = np.sqrt(s.sse / s.n)
    s["mae"] = s.sae / s.n
    s.to_csv(OUT / "bayesian_forward_summary.csv")
    print(s.to_string(), flush=True)
