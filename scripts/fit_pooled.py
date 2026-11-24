"""Run as python -m scripts.fit_pooled; never clean or join source data here."""

import argparse
import hashlib
import importlib.metadata
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import pymc as pm

from models.pooled import (
    SamplingConfig,
    attach_keys,
    build_pooled,
    fit_pooled,
    predict_pooled,
    read_priors,
)
from models.pooled_data import load_analysis, make_design, validate_splits


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def effective_prior_sha256(priors) -> str:
    return json_sha256(asdict(priors))


def json_sha256(value) -> str:
    content = json.dumps(value, indent=2, allow_nan=False) + "\n"
    return hashlib.sha256(content.encode()).hexdigest()


def verify_review(
    path: Path | None,
    priors: Path,
    data: Path,
    metadata: Path,
    required_configurations=("baseline",),
    *,
    splits: Path | None = None,
) -> None:
    if path is None:
        raise ValueError(
            "Production fitting needs --prior-review after prior predictive review"
        )
    review = json.loads(path.read_text())
    if (
        review.get("accepted") is not True
        or not review.get("reviewed_by")
        or not review.get("reviewed_at")
        or review.get("prior_sha256") != sha256(priors)
    ):
        raise ValueError("Prior review is incomplete or prior hash changed")
    manifest_path = Path(review.get("prior_run_manifest", ""))
    if not manifest_path.is_absolute():
        manifest_path = path.parent / manifest_path
    if not manifest_path.is_file() or review.get("prior_run_manifest_sha256") != sha256(
        manifest_path
    ):
        raise ValueError("Prior review must reference its unchanged prior-run manifest")
    manifest = json.loads(manifest_path.read_text())
    expected_mode = "prior-heldout" if splits is not None else "prior"
    if manifest.get("mode") != expected_mode:
        raise ValueError(f"This fit requires a matching {expected_mode} prior review")
    if (
        manifest.get("status") != "complete"
        or manifest.get("synthetic")
        or manifest.get("prior_sha256") != sha256(priors)
        or manifest.get("data_sha256") != sha256(data)
        or manifest.get("metadata_sha256") != sha256(metadata)
    ):
        raise ValueError(
            "Reviewed prior run does not match the production data and specification"
        )
    baseline = read_priors(priors)
    configurations = {
        "baseline": (baseline, "prior.nc"),
        "wide-slopes": (baseline.sensitivity(), "wide-slopes-prior.nc"),
    }
    if splits is None:
        verify_prior_artifacts(
            manifest,
            manifest_path.parent,
            review.get("reviewed_configurations", []),
            configurations,
            required_configurations,
        )
        return
    if manifest.get("splits_sha256") != sha256(splits):
        raise ValueError("Reviewed split file does not match the heldout run")
    frame = load_analysis(data, metadata)
    assignments = pd.read_csv(splits, dtype=str)
    validate_splits(frame, assignments)
    groups = list(assignments.groupby("split_id", sort=True))
    entries = manifest.get("splits", [])
    if len(entries) != len(groups):
        raise ValueError("Prior review must cover every heldout split")
    for i, ((split_id, split), entry) in enumerate(zip(groups, entries)):
        if (
            entry.get("split_id") != split_id
            or entry.get("directory") != f"split-{i:03d}"
        ):
            raise ValueError("Reviewed split identities do not match the heldout run")
        train = frame[frame.date.isin(split.loc[split.partition.eq("train"), "date"])]
        training, state = make_design(train)
        if entry.get("keys") != training.keys.values.tolist():
            raise ValueError(f"Reviewed training keys do not match split {split_id}")
        folder = manifest_path.parent / entry["directory"]
        transforms = folder / "transforms.json"
        expected_hash = json_sha256(asdict(state))
        if (
            entry.get("transforms_sha256") != expected_hash
            or not transforms.is_file()
            or sha256(transforms) != expected_hash
        ):
            raise ValueError(f"Reviewed transforms do not match split {split_id}")
        reviewed = review.get("reviewed_splits", {}).get(split_id, [])
        verify_prior_artifacts(
            entry, folder, reviewed, configurations, required_configurations
        )


def verify_prior_artifacts(
    entry, folder, reviewed, configurations, required_configurations
):
    for name in required_configurations:
        if name not in reviewed:
            raise ValueError(f"Prior review must explicitly cover {name}")
        expected, filename = configurations[name]
        configuration = entry.get("prior_configurations", {}).get(name, {})
        if configuration.get("effective_prior_sha256") != effective_prior_sha256(
            expected
        ):
            raise ValueError(f"Reviewed effective configuration does not match {name}")
        prior_file = folder / filename
        if not prior_file.is_file() or configuration.get("draws_sha256") != sha256(
            prior_file
        ):
            raise ValueError(
                f"Reviewed prior predictions changed or are missing: {name}"
            )


def run_prior(frame, priors, config, output: Path, synthetic: bool):
    data, state = make_design(frame)
    frame.to_csv(output / "training.csv", index=False)
    write_json(output / "transforms.json", asdict(state))
    entry = dict(
        keys=data.keys.values.tolist(),
        transforms_sha256=sha256(output / "transforms.json"),
        prior_configurations={},
    )
    for name, effective, filename, config_file in [
        ("baseline", priors, "prior.nc", "effective_priors.json"),
        (
            "wide-slopes",
            priors.sensitivity(),
            "wide-slopes-prior.nc",
            "effective_wide_priors.json",
        ),
    ]:
        write_json(output / config_file, asdict(effective))
        with build_pooled(data, effective):
            prior = pm.sample_prior_predictive(
                draws=config.draws, random_seed=config.seed
            )
        prior = attach_keys(prior, data)
        prior.attrs["synthetic"] = str(synthetic)
        prior.to_netcdf(output / filename)
        entry["prior_configurations"][name] = dict(
            effective_prior_sha256=effective_prior_sha256(effective),
            draws_sha256=sha256(output / filename),
        )
    return entry


def run_fit(
    frame,
    priors,
    config,
    output: Path,
    synthetic: bool,
    prediction_frame=None,
    split_id=None,
):
    output.mkdir(parents=True, exist_ok=False)
    data, state = make_design(frame)
    frame.to_csv(output / "training.csv", index=False)
    write_json(output / "transforms.json", asdict(state))
    write_json(output / "effective_priors.json", asdict(priors))
    fit = fit_pooled(data, priors, config)
    fit.idata.attrs["synthetic"] = str(synthetic)
    fit.idata.to_netcdf(output / "posterior.nc")
    if prediction_frame is not None:
        pred, _ = make_design(prediction_frame.drop(columns="no2_daily"), state, False)
        predictions = predict_pooled(fit, pred, config.seed + 2)
        predictions = predictions.assign_coords(
            split_id=("observation", [str(split_id)] * len(pred.keys))
        )
        predictions.attrs["synthetic"] = str(synthetic)
        predictions.to_netcdf(output / "predictions.nc")
        pred.keys.assign(split_id=str(split_id)).to_csv(
            output / "prediction_keys.csv", index=False
        )
    return fit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "metadata", "priors", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument(
        "--mode",
        choices=["prior", "prior-heldout", "posterior", "heldout", "sensitivity"],
        required=True,
    )
    parser.add_argument("--splits", type=Path)
    parser.add_argument("--prior-review", type=Path)
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Only for explicitly marked synthetic input and priors",
    )
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261121)
    parser.add_argument("--target-accept", type=float, default=0.9)
    args = parser.parse_args(argv)
    split_mode = args.mode in ("heldout", "prior-heldout")
    if args.splits is not None and not split_mode:
        raise ValueError(
            "--splits is only supported in heldout and prior-heldout modes; "
            "other modes fit all rows"
        )
    if args.output.exists():
        raise FileExistsError(f"Output exists; choose a new directory: {args.output}")
    frame = load_analysis(args.data, args.metadata)
    priors = read_priors(args.priors, allow_provisional=args.synthetic)
    if (
        priors.metadata["coefficients"]["wind"]["measure"]
        != json.loads(args.metadata.read_text())["wind_measure"]
    ):
        raise ValueError("Prior wind measure does not match the data handoff")
    if args.synthetic and (
        json.loads(args.metadata.read_text()).get("synthetic") is not True
        or priors.metadata.get("synthetic") is not True
    ):
        raise ValueError(
            "--synthetic requires explicitly marked synthetic data and priors"
        )
    if not args.synthetic and (
        json.loads(args.metadata.read_text()).get("synthetic")
        or priors.metadata.get("synthetic")
    ):
        raise ValueError(
            "Synthetic fixtures require --synthetic and cannot be production runs"
        )
    splits = None
    if split_mode:
        if args.splits is None:
            raise ValueError(
                f"{args.mode} mode requires --splits from comparison owner"
            )
        splits = pd.read_csv(args.splits, dtype=str)
        validate_splits(frame, splits)
    if args.mode not in ("prior", "prior-heldout") and not args.synthetic:
        required = (
            ("baseline", "wide-slopes") if args.mode == "sensitivity" else ("baseline",)
        )
        verify_review(
            args.prior_review,
            args.priors,
            args.data,
            args.metadata,
            required,
            splits=args.splits,
        )
    config = SamplingConfig(
        args.chains, args.warmup, args.draws, args.seed, args.target_accept
    )
    manifest = dict(
        mode=args.mode,
        synthetic=args.synthetic,
        status="running",
        data_sha256=sha256(args.data),
        metadata_sha256=sha256(args.metadata),
        prior_sha256=sha256(args.priors),
        sampling=asdict(config),
        keys=frame[["station_id", "date"]].values.tolist(),
        versions={
            p: importlib.metadata.version(p)
            for p in ("pymc", "arviz", "numpy", "pandas", "xarray")
        },
        python=sys.version,
    )
    if args.splits:
        manifest["splits_sha256"] = sha256(args.splits)
    if args.prior_review:
        manifest["prior_review_sha256"] = sha256(args.prior_review)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "manifest.json", manifest)
    try:
        if args.mode == "prior":
            manifest.update(
                run_prior(frame, priors, config, args.output, args.synthetic)
            )
        elif split_mode:
            manifest["splits"] = []
            for i, (split_id, split) in enumerate(
                splits.groupby("split_id", sort=True)
            ):
                output = args.output / f"split-{i:03d}"
                train = frame[
                    frame.date.isin(split.loc[split.partition.eq("train"), "date"])
                ]
                test = frame[
                    frame.date.isin(split.loc[split.partition.eq("test"), "date"])
                ]
                entry = dict(split_id=split_id, directory=output.name)
                if args.mode == "prior-heldout":
                    output.mkdir()
                    entry.update(
                        run_prior(train, priors, config, output, args.synthetic)
                    )
                else:
                    run_fit(
                        train, priors, config, output, args.synthetic, test, split_id
                    )
                manifest["splits"].append(entry)
        else:
            run_fit(frame, priors, config, args.output / "baseline", args.synthetic)
            if args.mode == "sensitivity":
                run_fit(
                    frame,
                    priors.sensitivity(),
                    config,
                    args.output / "wide-slopes",
                    args.synthetic,
                )
        manifest["status"] = "complete"
    except Exception as error:
        manifest.update(status="failed", error=str(error))
        raise
    finally:
        write_json(args.output / "manifest.json", manifest)
    print(
        f"{args.mode} outputs: {args.output} (synthetic={args.synthetic}); "
        "inspect diagnostics before interpretation"
    )
    return manifest


if __name__ == "__main__":
    main()
