"""Run as python -m scripts.fit_pooled; never clean or join source data here."""
from dataclasses import asdict
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import sys

import pandas as pd
import pymc as pm

from models.pooled import SamplingConfig, build_pooled, fit_pooled, predict_pooled, read_priors
from models.pooled_data import load_analysis, make_design, validate_splits


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def verify_review(path: Path | None, priors: Path, data: Path, metadata: Path) -> None:
    if path is None:
        raise ValueError('Production fitting needs --prior-review after prior predictive review')
    review = json.loads(path.read_text())
    if (review.get('accepted') is not True or not review.get('reviewed_by') or not review.get('reviewed_at')
            or review.get('prior_sha256') != sha256(priors)):
        raise ValueError('Prior review is incomplete or prior hash changed')
    manifest_path = Path(review.get('prior_run_manifest', ''))
    if not manifest_path.is_absolute():
        manifest_path = path.parent / manifest_path
    if not manifest_path.is_file() or review.get('prior_run_manifest_sha256') != sha256(manifest_path):
        raise ValueError('Prior review must reference its unchanged prior-run manifest')
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get('mode') != 'prior' or manifest.get('status') != 'complete' or manifest.get('synthetic')
            or manifest.get('prior_sha256') != sha256(priors)
            or manifest.get('data_sha256') != sha256(data)
            or manifest.get('metadata_sha256') != sha256(metadata)):
        raise ValueError('Reviewed prior run does not match the production data and specification')


def run_fit(frame, priors, config, output: Path, synthetic: bool, prediction_frame=None):
    output.mkdir(parents=True, exist_ok=False)
    data, state = make_design(frame)
    frame.to_csv(output/'training.csv', index=False)
    write_json(output/'transforms.json', asdict(state))
    write_json(output/'effective_priors.json', asdict(priors))
    fit = fit_pooled(data, priors, config)
    fit.idata.attrs['synthetic'] = str(synthetic)
    fit.idata.to_netcdf(output/'posterior.nc')
    if prediction_frame is not None:
        pred, _ = make_design(prediction_frame.drop(columns='no2_daily'), state, False)
        predictions = predict_pooled(fit, pred, config.seed+2)
        predictions.attrs['synthetic'] = str(synthetic)
        predictions.to_netcdf(output/'predictions.nc')
        pred.keys.to_csv(output/'prediction_keys.csv', index=False)
    return fit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'metadata', 'priors', 'output'):
        parser.add_argument('--'+name, required=True, type=Path)
    parser.add_argument('--mode', choices=['prior', 'posterior', 'heldout', 'sensitivity'], required=True)
    parser.add_argument('--splits', type=Path)
    parser.add_argument('--prior-review', type=Path)
    parser.add_argument('--synthetic', action='store_true', help='Only for explicitly marked synthetic input and priors')
    parser.add_argument('--chains', type=int, default=4)
    parser.add_argument('--warmup', type=int, default=1000)
    parser.add_argument('--draws', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=20261121)
    parser.add_argument('--target-accept', type=float, default=.9)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f'Output exists; choose a new directory: {args.output}')
    frame = load_analysis(args.data, args.metadata)
    priors = read_priors(args.priors, allow_provisional=args.synthetic)
    if args.synthetic and (json.loads(args.metadata.read_text()).get('synthetic') is not True
                           or priors.metadata.get('synthetic') is not True):
        raise ValueError('--synthetic requires explicitly marked synthetic data and priors')
    if not args.synthetic and (json.loads(args.metadata.read_text()).get('synthetic')
                               or priors.metadata.get('synthetic')):
        raise ValueError('Synthetic fixtures require --synthetic and cannot be production runs')
    splits = None
    if args.mode == 'heldout':
        if args.splits is None:
            raise ValueError('Heldout mode requires --splits from comparison owner')
        splits = pd.read_csv(args.splits, dtype=str)
        validate_splits(frame, splits)
    if args.mode != 'prior' and not args.synthetic:
        verify_review(args.prior_review, args.priors, args.data, args.metadata)
    config = SamplingConfig(args.chains, args.warmup, args.draws, args.seed, args.target_accept)
    manifest = dict(mode=args.mode, synthetic=args.synthetic, status='running',
        data_sha256=sha256(args.data), metadata_sha256=sha256(args.metadata), prior_sha256=sha256(args.priors),
        sampling=asdict(config), keys=frame[['station_id','date']].values.tolist(),
        versions={p:importlib.metadata.version(p) for p in ('pymc','arviz','numpy','pandas','xarray')},
        python=sys.version)
    if args.splits:
        manifest['splits_sha256'] = sha256(args.splits)
    if args.prior_review:
        manifest['prior_review_sha256'] = sha256(args.prior_review)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output/'manifest.json', manifest)
    try:
        if args.mode == 'prior':
            data, state = make_design(frame)
            frame.to_csv(args.output/'training.csv', index=False)
            write_json(args.output/'transforms.json', asdict(state))
            write_json(args.output/'effective_priors.json', asdict(priors))
            with build_pooled(data, priors):
                prior = pm.sample_prior_predictive(draws=config.draws, random_seed=config.seed)
            prior.attrs['synthetic'] = str(args.synthetic)
            prior.to_netcdf(args.output/'prior.nc')
        elif args.mode == 'heldout':
            manifest['splits'] = []
            for i, (split_id, split) in enumerate(splits.groupby('split_id', sort=True)):
                output = args.output/f'split-{i:03d}'
                train = frame[frame.date.isin(split.loc[split.partition.eq('train'),'date'])]
                test = frame[frame.date.isin(split.loc[split.partition.eq('test'),'date'])]
                run_fit(train, priors, config, output, args.synthetic, test)
                manifest['splits'].append(dict(split_id=split_id, directory=output.name))
        else:
            run_fit(frame, priors, config, args.output/'baseline', args.synthetic)
            if args.mode == 'sensitivity':
                run_fit(frame, priors.sensitivity(), config, args.output/'wide-slopes', args.synthetic)
        manifest['status'] = 'complete'
    except Exception as error:
        manifest.update(status='failed', error=str(error))
        raise
    finally:
        write_json(args.output/'manifest.json', manifest)
    print(f'{args.mode} outputs: {args.output} (synthetic={args.synthetic}); inspect diagnostics before interpretation')
    return manifest


if __name__ == '__main__':
    main()
