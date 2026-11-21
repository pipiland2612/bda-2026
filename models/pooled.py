"""Model A: one intercept and common weather/calendar effects."""
from dataclasses import dataclass, replace
from pathlib import Path
import json

import numpy as np
import pymc as pm

from models.pooled_data import FEATURES, ModelData


@dataclass(frozen=True)
class PriorSpec:
    alpha_mean: float
    alpha_sd: float
    beta_means: tuple[float, ...]
    beta_sds: tuple[float, ...]
    sigma_scale: float
    metadata: dict

    def sensitivity(self):
        return replace(self, beta_sds=tuple(2*s for s in self.beta_sds))


def ratio_prior(lower: float, upper: float) -> tuple[float, float]:
    if not np.isfinite([lower, upper]).all() or not 0 < lower < upper:
        raise ValueError('Prior ratios require finite 0 < lower < upper')
    return (float((np.log(lower)+np.log(upper))/2),
            float((np.log(upper)-np.log(lower))/(2*1.96)))


def read_priors(path: Path, allow_provisional: bool = False) -> PriorSpec:
    config = json.loads(path.read_text())
    if not allow_provisional and config.get('approved') is not True:
        raise ValueError('Prior configuration must be approved; fill the elicitation template')
    if not config.get('rationale') or not config.get('sources'):
        raise ValueError('Priors require rationale and sources')
    try:
        if set(config['coefficients']) != set(FEATURES):
            raise ValueError('Priors must specify exactly the six model coefficients')
        ranges = []
        for feature in FEATURES:
            c = config['coefficients'][feature]
            if not c.get('unit'):
                raise ValueError(f'Missing prior unit: {feature}')
            ranges.append(ratio_prior(float(c['lower']), float(c['upper'])))
        a, s = config['alpha'], config['sigma']
        mu, sd, scale = float(a['mean']), float(a['sd']), float(s['scale'])
        if not np.isfinite([mu, sd, scale]).all() or min(sd, scale) <= 0:
            raise ValueError('Alpha SD and sigma scale must be finite and positive')
    except (KeyError, TypeError) as error:
        raise ValueError('Incomplete priors: fill alpha mean/SD, sigma scale and all ratio ranges') from error
    return PriorSpec(mu, sd, tuple(r[0] for r in ranges), tuple(r[1] for r in ranges), scale, config)


def build_pooled(data: ModelData, priors: PriorSpec) -> pm.Model:
    if data.features != FEATURES or data.X.shape != (len(data.keys), len(FEATURES)):
        raise ValueError('Model design columns do not match the specified prior order')
    if not len(data.keys) or not np.isfinite(data.X).all():
        raise ValueError('Empty or nonfinite design')
    if data.y is not None and (data.y.shape != (len(data.keys),) or not np.isfinite(data.y).all() or (data.y <= 0).any()):
        raise ValueError('Observed no2 must be finite and positive')
    with pm.Model(coords={'observation': np.arange(len(data.keys)), 'feature': FEATURES}) as model:
        alpha = pm.Normal('alpha', priors.alpha_mean, priors.alpha_sd)
        beta = pm.Normal('beta', priors.beta_means, priors.beta_sds, dims='feature')
        sigma = pm.HalfNormal('sigma', priors.sigma_scale)
        eta = pm.Deterministic('eta', alpha + pm.math.dot(data.X, beta), dims='observation')
        pm.LogNormal('no2', mu=eta, sigma=sigma, observed=data.y, dims='observation')
    return model
