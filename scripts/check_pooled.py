"""Create sampling diagnostics and predictive plots for a saved pooled run."""

import argparse
import json
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace

import arviz as az
import pandas as pd

from models.pooled_diagnostics import (
    diagnostic_figures,
    predictive_figures,
    summarize_fit,
    summarize_weekend,
)
from scripts.fit_pooled import write_json


def check_run(run: Path):
    manifest = json.loads((run / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("Only a completed run can be checked")
    label = "SYNTHETIC DEMONSTRATION" if manifest["synthetic"] else "Pooled model"
    if manifest["mode"] == "prior":
        with closing(az.from_netcdf(run / "prior.nc")) as prior:
            frame = pd.read_csv(run / "training.csv", dtype={"station_id": str})
            predictive_figures(
                frame,
                prior.prior_predictive.no2.values,
                run / "checks",
                label,
                prior=True,
            )
        write_json(
            run / "checks" / "summary.json",
            dict(
                synthetic=manifest["synthetic"],
                prior_predictive_review="Required; inspect plausible concentrations and extremes",
            ),
        )
        return
    directories = (
        [s["directory"] for s in manifest["splits"]]
        if manifest["mode"] == "heldout"
        else ["baseline", "wide-slopes"]
        if manifest["mode"] == "sensitivity"
        else ["baseline"]
    )
    summaries = []
    for name in directories:
        folder = run / name
        output = folder / "checks"
        output.mkdir(exist_ok=True)
        with closing(az.from_netcdf(folder / "posterior.nc")) as idata:
            frame = pd.read_csv(folder / "training.csv", dtype={"station_id": str})
            summary = summarize_fit(SimpleNamespace(idata=idata))
            summary.pop("table").to_csv(output / "sampling.csv")
            summary.update(
                weekend=summarize_weekend(
                    idata.posterior.beta.sel(feature="weekend").values
                ),
                synthetic=manifest["synthetic"],
                run=name,
            )
            write_json(output / "summary.json", summary)
            diagnostic_figures(idata, frame, output, label)
            summaries.append(
                dict(
                    run=name,
                    **summary["weekend"],
                    sampling_passed=summary["sampling_passed"],
                )
            )
    pd.DataFrame(summaries).to_csv(run / "weekend_summary.csv", index=False)
    print(
        f"Checks saved under {run}; sampling gates do not establish predictive adequacy"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args(argv)
    check_run(args.run)


if __name__ == "__main__":
    main()
