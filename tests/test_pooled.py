import json

import numpy as np
import pymc as pm
import pytest

from models.pooled import build_pooled, ratio_prior, read_priors
from models.pooled_data import FEATURES, make_design
from tests.test_pooled_data import frame


def prior_config():
    result = dict(
        approved=True,
        rationale="SYNTHETIC test values, no scientific approval",
        sources=["synthetic fixture"],
        alpha=dict(mean=3.0, sd=0.3),
        sigma=dict(scale=0.3),
        coefficients={
            name: dict(lower=0.7, upper=1.4, unit="synthetic fixture increment")
            for name in FEATURES
        },
    )
    result["coefficients"]["wind"]["measure"] = "mean_of_hourly_maxima"
    return result


def test_prior_range_conversion_and_invalid_ranges():
    mu, sd = ratio_prior(0.7, 1.1)
    assert mu == pytest.approx(-0.130682382)
    assert sd == pytest.approx(0.115302837, rel=1e-5)
    for low, high in [(0, 1), (-1, 2), (2, 1), (1, 1), (1, np.inf)]:
        with pytest.raises(ValueError):
            ratio_prior(low, high)


def test_prior_validation_prevents_silent_defaults(tmp_path):
    p = tmp_path / "prior.json"
    mutations = [
        dict(approved=False),
        dict(alpha=dict(mean=3, sd=0)),
        dict(sigma=dict(scale=-1)),
        dict(coefficients={}),
        dict(rationale=""),
        dict(sources=[]),
    ]
    for mutation in mutations:
        p.write_text(json.dumps(dict(prior_config(), **mutation)))
        with pytest.raises(ValueError):
            read_priors(p)


def test_pooled_prior_predictions_are_positive_and_row_aligned(tmp_path):
    p = tmp_path / "prior.json"
    p.write_text(json.dumps(prior_config()))
    priors = read_priors(p)
    data, _ = make_design(frame())
    model = build_pooled(data, priors)
    assert set(model.free_RVs) == {model["alpha"], model["beta"], model["sigma"]}
    with model:
        draws = pm.sample_prior_predictive(draws=10, random_seed=14)
    y = draws.prior_predictive["no2"]
    assert y.shape == (1, 10, 4) and np.isfinite(y).all() and (y > 0).all()
    # A wrong design order or omitted intercept changes the actual predictor.
    eta = draws.prior["eta"].values
    alpha = draws.prior["alpha"].values
    beta = draws.prior["beta"].values
    # Jan 1: weekend=0, centered wind=-.5, temp=-.5, rain=-.5, sin=0, cos=1.
    np.testing.assert_allclose(
        eta[:, :, 0],
        alpha
        - 0.5 * beta[:, :, 1]
        - 0.5 * beta[:, :, 2]
        - 0.5 * beta[:, :, 3]
        + beta[:, :, 5],
    )
    bad = type(data)(data.keys, data.X, data.y, tuple(reversed(FEATURES)))
    with pytest.raises(ValueError):
        build_pooled(bad, priors)
