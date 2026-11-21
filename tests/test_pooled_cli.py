import json
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from models.pooled import fit_pooled, predict_pooled, read_priors, SamplingConfig
from models.pooled_data import make_design
from scripts.fit_pooled import main, sha256
from tests.test_pooled import prior_config
from tests.test_pooled_data import frame, metadata


def inputs(tmp_path):
    data,meta,priors=tmp_path/'data.csv',tmp_path/'metadata.json',tmp_path/'priors.json'
    frame().to_csv(data,index=False)
    meta.write_text(json.dumps(dict(metadata(),synthetic=True)))
    priors.write_text(json.dumps(dict(prior_config(),synthetic=True)))
    return ['--data',str(data),'--metadata',str(meta),'--priors',str(priors),'--output',str(tmp_path/'run')]


@pytest.mark.parametrize('case',['approval','metadata','split','output','review','fake_synthetic'])
def test_cli_invalid_requests_do_not_sample(tmp_path,case):
    args=inputs(tmp_path)+['--mode','heldout']
    if case=='approval':
        p=tmp_path/'priors.json'; p.write_text(json.dumps(dict(prior_config(),approved=False)))
    if case=='metadata': (tmp_path/'metadata.json').unlink()
    if case=='output': (tmp_path/'run').mkdir()
    if case=='review': args=inputs(tmp_path)+['--mode','posterior']
    if case=='fake_synthetic':
        p=tmp_path/'metadata.json'; p.write_text(json.dumps(metadata())); args+=['--synthetic']
    with pytest.raises((ValueError,FileNotFoundError,FileExistsError)): main(args)


def test_wrong_prior_review_hash_rejected(tmp_path):
    args=inputs(tmp_path)
    review=tmp_path/'review.json'; review.write_text(json.dumps(dict(accepted=True,reviewed_by='test',reviewed_at='2026-11-21',prior_sha256='wrong')))
    with pytest.raises(ValueError): main(args+['--mode','posterior','--prior-review',str(review)])


def test_real_sampling_prediction_separates_outcomes_and_preserves_keys(tmp_path):
    rng=np.random.default_rng(11)
    f=frame().iloc[:1].copy(); f=pd.concat([f]*24,ignore_index=True)
    f['date']=pd.date_range('2024-01-01',periods=24).strftime('%Y-%m-%d')
    f['wind_speed']=rng.uniform(1,8,24);f['temperature']=rng.normal(0,5,24);f['precipitation']=rng.uniform(0,2,24)
    f['no2_daily']=np.exp(3+rng.normal(0,.2,24))
    p=tmp_path/'priors.json';p.write_text(json.dumps(prior_config()))
    priors=read_priors(p); train,state=make_design(f.iloc[:20])
    fit=fit_pooled(train,priors,SamplingConfig(chains=2,warmup=30,draws=30,seed=13))
    pred,_=make_design(f.iloc[20:].drop(columns='no2_daily'),state,False)
    out=predict_pooled(fit,pred,14)
    assert out.no2.shape==(2,30,4)
    assert np.isfinite(out.no2).all() and (out.no2>0).all()
    assert out.date.values.tolist()==f.date.iloc[20:].tolist()
    assert out.station_id.values.tolist()==['a']*4
    observed,_=make_design(f.iloc[20:],state)
    with pytest.raises(ValueError,match='outcome'): predict_pooled(fit,observed,14)


def test_cli_prior_exports_manifest_and_draws(tmp_path):
    args=inputs(tmp_path)
    main(args+['--mode','prior','--synthetic','--draws','10'])
    manifest=json.loads((tmp_path/'run'/'manifest.json').read_text())
    assert manifest['synthetic'] is True and manifest['status']=='complete'
    assert manifest['data_sha256']==sha256(tmp_path/'data.csv')
    assert (tmp_path/'run'/'prior.nc').is_file()


def test_cli_heldout_exports_whole_dates_and_training_only_centers(tmp_path):
    args=inputs(tmp_path)
    split=tmp_path/'splits.csv'
    pd.DataFrame(dict(split_id=['s','s'],date=['2024-01-01','2024-01-06'],partition=['train','test'])).to_csv(split,index=False)
    result=main(args+['--mode','heldout','--splits',str(split),'--synthetic','--chains','1','--warmup','10','--draws','10'])
    run=tmp_path/'run'/'split-000'
    state=json.loads((run/'transforms.json').read_text())
    assert state['centers']==[2.,0.,0.]
    with xr.open_dataset(run/'predictions.nc') as predictions:
        assert predictions.no2.shape==(1,10,2)
        assert predictions.date.values.tolist()==['2024-01-06']*2
        assert predictions.station_id.values.tolist()==['a','b']
    assert result['splits']==[dict(split_id='s',directory='split-000')]


def test_cli_sensitivity_doubles_only_slope_scales(tmp_path):
    main(inputs(tmp_path)+['--mode','sensitivity','--synthetic','--chains','1','--warmup','10','--draws','10'])
    base=json.loads((tmp_path/'run'/'baseline'/'effective_priors.json').read_text())
    wide=json.loads((tmp_path/'run'/'wide-slopes'/'effective_priors.json').read_text())
    np.testing.assert_allclose(wide['beta_sds'],np.array(base['beta_sds'])*2)
    for key in ['alpha_mean','alpha_sd','sigma_scale','beta_means']:
        assert wide[key]==base[key]
