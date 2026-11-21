import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from models.pooled import SamplingConfig, fit_pooled, predict_pooled, read_priors
from models.pooled_data import make_design
from scripts.check_pooled import check_run
from scripts.fit_pooled import main, sha256, verify_review
from tests.test_pooled import prior_config
from tests.test_pooled_data import frame, metadata


def inputs(tmp_path, synthetic=True):
    data, meta, priors = (
        tmp_path / "data.csv",
        tmp_path / "metadata.json",
        tmp_path / "priors.json",
    )
    frame().to_csv(data, index=False)
    meta.write_text(json.dumps(dict(metadata(), synthetic=synthetic)))
    priors.write_text(json.dumps(dict(prior_config(), synthetic=synthetic)))
    return [
        "--data",
        str(data),
        "--metadata",
        str(meta),
        "--priors",
        str(priors),
        "--output",
        str(tmp_path / "run"),
    ]


@pytest.mark.parametrize(
    "case", ["approval", "metadata", "split", "output", "review", "fake_synthetic"]
)
def test_cli_invalid_requests_do_not_sample(tmp_path, case):
    args = inputs(tmp_path, synthetic=False) + ["--mode", "heldout"]
    if case == "approval":
        p = tmp_path / "priors.json"
        p.write_text(json.dumps(dict(prior_config(), approved=False)))
    if case == "metadata":
        (tmp_path / "metadata.json").unlink()
    if case == "output":
        (tmp_path / "run").mkdir()
    if case == "review":
        args = inputs(tmp_path, synthetic=False) + ["--mode", "posterior"]
    if case == "fake_synthetic":
        p = tmp_path / "metadata.json"
        p.write_text(json.dumps(metadata()))
        args += ["--synthetic"]
    expected = {
        "approval": "approved",
        "metadata": None,
        "split": "--splits",
        "output": "Output exists",
        "review": "--prior-review",
        "fake_synthetic": "explicitly marked",
    }[case]
    with pytest.raises(
        (ValueError, FileNotFoundError, FileExistsError), match=expected
    ):
        main(args)


def test_wrong_prior_review_hash_rejected(tmp_path):
    args = inputs(tmp_path, synthetic=False)
    review = tmp_path / "review.json"
    review.write_text(
        json.dumps(
            dict(
                accepted=True,
                reviewed_by="test",
                reviewed_at="2026-11-21",
                prior_sha256="wrong",
            )
        )
    )
    with pytest.raises(ValueError, match="prior hash changed"):
        main(args + ["--mode", "posterior", "--prior-review", str(review)])


def test_prior_review_binds_actual_manifest_data_metadata_and_priors(tmp_path):
    args = inputs(tmp_path, synthetic=False)
    main(args + ["--mode", "prior", "--draws", "5"])
    data, meta, priors = [
        tmp_path / name for name in ("data.csv", "metadata.json", "priors.json")
    ]
    manifest = tmp_path / "run" / "manifest.json"
    review = tmp_path / "review.json"
    record = dict(
        accepted=True,
        reviewed_by="test fixture",
        reviewed_at="2026-11-21",
        prior_sha256=sha256(priors),
        prior_run_manifest="run/manifest.json",
        prior_run_manifest_sha256=sha256(manifest),
        reviewed_configurations=["baseline"],
    )
    review.write_text(json.dumps(record))
    verify_review(review, priors, data, meta)
    with pytest.raises(ValueError, match="wide-slopes"):
        verify_review(
            review,
            priors,
            data,
            meta,
            required_configurations=("baseline", "wide-slopes"),
        )
    record["reviewed_configurations"].append("wide-slopes")
    review.write_text(json.dumps(record))
    verify_review(
        review, priors, data, meta, required_configurations=("baseline", "wide-slopes")
    )
    original_manifest = manifest.read_text()
    bad = json.loads(original_manifest)
    bad["prior_configurations"]["wide-slopes"]["effective_prior_sha256"] = "wrong"
    manifest.write_text(json.dumps(bad))
    record["prior_run_manifest_sha256"] = sha256(manifest)
    review.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="effective configuration"):
        verify_review(
            review,
            priors,
            data,
            meta,
            required_configurations=("baseline", "wide-slopes"),
        )
    manifest.write_text(original_manifest)
    record["prior_run_manifest_sha256"] = sha256(manifest)
    review.write_text(json.dumps(record))
    meta.write_text(meta.read_text() + "\n")
    with pytest.raises(ValueError, match="does not match"):
        verify_review(review, priors, data, meta)
    manifest.write_text(manifest.read_text() + "\n")
    with pytest.raises(ValueError, match="unchanged prior-run manifest"):
        verify_review(review, priors, data, meta)


def test_real_sampling_prediction_separates_outcomes_and_preserves_keys(tmp_path):
    rng = np.random.default_rng(11)
    f = frame().iloc[:1].copy()
    f = pd.concat([f] * 24, ignore_index=True)
    f["date"] = pd.date_range("2024-01-01", periods=24).strftime("%Y-%m-%d")
    f["wind_speed"] = rng.uniform(1, 8, 24)
    f["temperature"] = rng.normal(0, 5, 24)
    f["precipitation"] = rng.uniform(0, 2, 24)
    f["no2_daily"] = np.exp(3 + rng.normal(0, 0.2, 24))
    p = tmp_path / "priors.json"
    p.write_text(json.dumps(prior_config()))
    priors = read_priors(p)
    train, state = make_design(f.iloc[:20])
    fit = fit_pooled(
        train, priors, SamplingConfig(chains=2, warmup=30, draws=30, seed=13)
    )
    pred, _ = make_design(f.iloc[20:].drop(columns="no2_daily"), state, False)
    out = predict_pooled(fit, pred, 14)
    assert out.no2.shape == (2, 30, 4)
    assert np.isfinite(out.no2).all() and (out.no2 > 0).all()
    assert out.date.values.tolist() == f.date.iloc[20:].tolist()
    assert out.station_id.values.tolist() == ["a"] * 4
    assert fit.idata.posterior_predictive.station_id.values.tolist() == ["a"] * 20
    assert fit.idata.posterior.date.values.tolist() == f.date.iloc[:20].tolist()
    observed, _ = make_design(f.iloc[20:], state)
    with pytest.raises(ValueError, match="outcome"):
        predict_pooled(fit, observed, 14)


def test_cli_prior_exports_manifest_and_draws(tmp_path):
    args = inputs(tmp_path)
    main(args + ["--mode", "prior", "--synthetic", "--draws", "10"])
    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text())
    assert manifest["synthetic"] is True and manifest["status"] == "complete"
    assert manifest["data_sha256"] == sha256(tmp_path / "data.csv")
    assert (tmp_path / "run" / "prior.nc").is_file()
    with xr.open_dataset(
        tmp_path / "run" / "prior.nc", group="prior_predictive"
    ) as prior:
        assert prior.station_id.values.tolist() == ["a", "a", "b", "b"]
    check_run(tmp_path / "run")
    assert (tmp_path / "run" / "checks" / "distribution.png").stat().st_size > 1000


def test_cli_heldout_exports_whole_dates_and_training_only_centers(tmp_path):
    args = inputs(tmp_path)
    split = tmp_path / "splits.csv"
    pd.DataFrame(
        dict(
            split_id=["s", "s"],
            date=["2024-01-01", "2024-01-06"],
            partition=["train", "test"],
        )
    ).to_csv(split, index=False)
    result = main(
        args
        + [
            "--mode",
            "heldout",
            "--splits",
            str(split),
            "--synthetic",
            "--chains",
            "1",
            "--warmup",
            "10",
            "--draws",
            "10",
        ]
    )
    run = tmp_path / "run" / "split-000"
    state = json.loads((run / "transforms.json").read_text())
    assert state["centers"] == [2.0, 0.0, 0.0]
    with xr.open_dataset(run / "predictions.nc") as predictions:
        assert predictions.no2.shape == (1, 10, 2)
        assert predictions.date.values.tolist() == ["2024-01-06"] * 2
        assert predictions.station_id.values.tolist() == ["a", "b"]
    assert result["splits"] == [dict(split_id="s", directory="split-000")]
    with xr.open_dataset(run / "predictions.nc") as predictions:
        assert predictions.split_id.values.tolist() == ["s", "s"]
    assert pd.read_csv(run / "prediction_keys.csv").split_id.tolist() == ["s", "s"]
    with pytest.warns(UserWarning, match="Small predictive-check group"):
        check_run(tmp_path / "run")
    summary = json.loads((run / "checks" / "summary.json").read_text())
    assert summary["sampling_passed"] is False
    assert (run / "checks" / "groups.png").stat().st_size > 1000
    training = pd.read_csv(run / "training.csv").iloc[::-1]
    training.to_csv(run / "training.csv", index=False)
    with pytest.raises(ValueError, match="keys"):
        check_run(tmp_path / "run")


def test_splits_flag_is_not_silently_ignored(tmp_path):
    args = inputs(tmp_path)
    split = tmp_path / "splits.csv"
    split.write_text("split_id,date,partition\ns,2024-01-01,train\n")
    with pytest.raises(ValueError, match="only.*heldout"):
        main(
            args
            + ["--mode", "prior", "--synthetic", "--splits", str(split), "--draws", "2"]
        )


def test_prior_wind_measure_must_match_data_handoff(tmp_path):
    args = inputs(tmp_path)
    p = tmp_path / "priors.json"
    c = json.loads(p.read_text())
    c["coefficients"]["wind"]["measure"] = "daily_mean_speed"
    p.write_text(json.dumps(c))
    with pytest.raises(ValueError, match="wind measure"):
        main(args + ["--mode", "prior", "--synthetic", "--draws", "2"])


def test_cli_sensitivity_doubles_only_slope_scales(tmp_path):
    main(
        inputs(tmp_path)
        + [
            "--mode",
            "sensitivity",
            "--synthetic",
            "--chains",
            "1",
            "--warmup",
            "10",
            "--draws",
            "10",
        ]
    )
    base = json.loads(
        (tmp_path / "run" / "baseline" / "effective_priors.json").read_text()
    )
    wide = json.loads(
        (tmp_path / "run" / "wide-slopes" / "effective_priors.json").read_text()
    )
    np.testing.assert_allclose(wide["beta_sds"], np.array(base["beta_sds"]) * 2)
    for key in ["alpha_mean", "alpha_sd", "sigma_scale", "beta_means"]:
        assert wide[key] == base[key]
