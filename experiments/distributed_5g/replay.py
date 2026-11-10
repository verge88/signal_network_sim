"""Retrospective deployment verification using immutable measured 5G archives."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import time
import zipfile
import numpy as np
import pandas as pd
from experiments.live_5g.evaluate import conformal_threshold
from experiments.accumulation_5g.detector import Engine as Reference, advance, peer_residuals
from experiments.distributed_5g.deploy import configure, Deployment, get
from experiments.resilient_5g.detector import permit_valid


def read_json(archive,name):
    return json.loads(archive.read(name))


def read_lines(archive,name):
    return [json.loads(line) for line in archive.read(name).splitlines()]


def frame(archive,name):
    with archive.open(name) as stream:
        return pd.read_csv(stream,float_precision='round_trip')


def fit_zones(archive,topology):
    features=frame(archive,'test/features.csv')
    labels=frame(archive,'test/labels.csv')
    attacked=set(labels.loc[labels.attack==1,'window_id'])
    normal=features[~features.window_id.isin(attacked)].copy()
    normal['error_count']=(normal.request_count*normal.error_rate).round().astype(int)
    thresholds={name:conformal_threshold(normal[name],.001/4) for name in ['request_count','error_count','update_count']}
    grants={r['window_id']:r['envelope'] for r in read_lines(archive,'test/load_grants.jsonl')}
    key=bytes.fromhex(archive.read('test/lab_operator_key.hex').decode())
    models={}
    for zone,members in topology.items():
        scores=[]
        previous=None
        accumulated=collections.defaultdict(float)
        for window,group in normal[normal.nf_id.isin(members)].groupby('window_id',sort=True):
            if previous is None or window!=previous+1:
                accumulated.clear()
            rows=group.to_dict('records')
            residuals=peer_residuals(rows)
            for row in rows:
                identity=row['nf_id']
                allowed=permit_valid(grants[window],key,row['run_id'],identity,row['timestamp'],row['request_count'])
                value=advance(accumulated[identity],residuals[identity],allowed)
                accumulated[identity]=value
                scores.append(value)
            previous=window
        models[zone]=dict(threshold=conformal_threshold(scores,.001/4),feature_thresholds=thresholds,
            calibration_rows=len(scores),calibration_run=features.run_id.iloc[0],members=members)
    return models


def run(calibration_path,evidence_path,output,windows=3600):
    output=Path(output).resolve()
    output.mkdir(parents=True,exist_ok=True)
    topology={f'zone-{zone}':[f'slot-{i}' for i in range(zone*8,(zone+1)*8)] for zone in range(4)}
    with zipfile.ZipFile(calibration_path) as calibration:
        models=fit_zones(calibration,topology)
    (output/'zone_models.json').write_text(json.dumps(models,indent=2))
    with zipfile.ZipFile(evidence_path) as archive:
        features=frame(archive,'test/features.csv')
        controls={r['window_id']:r for r in read_lines(archive,'test/control_events.jsonl')}
        grants={r['window_id']:r['envelope'] for r in read_lines(archive,'test/load_grants.jsonl')}
        previous={(r['window_id'],r['nf_id']):r for r in read_lines(archive,'test/resilient_predictions.jsonl')}
        decisions={(r['window_id'],r['nf_id']):r for r in read_lines(archive,'test/refined_predictions.jsonl')}
        refinement=read_json(archive,'frozen/refinement.json')
        extensions=read_json(archive,'frozen/extensions.json')
        control_key=archive.read('test/lab_control_key.hex').decode()
        operator_key=archive.read('test/lab_operator_key.hex').decode()
    assert windows>0 and windows<=features.window_id.nunique()
    assert all(model['calibration_run']!=features.run_id.iloc[0] for model in models.values())
    configure(output/'deployment',topology,models,features.run_id.iloc[0],operator_key,control_key,extensions,True)
    references={zone:Reference(refinement,bytes.fromhex(control_key),bytes.fromhex(operator_key),models[zone]) for zone in topology}
    count=0
    started=time.monotonic()
    with Deployment(output/'deployment') as deployment, (output/'decisions.jsonl').open('w') as stream:
        health={identity:get(url,'/health') for identity,url in deployment.source['urls'].items()}
        assert len({item['pid'] for item in health.values()})==37
        for window,group in features[features.window_id<windows].groupby('window_id',sort=True):
            rows=group.to_dict('records')
            observations={r['nf_id']:dict(row=r,control=controls[window] if r['nf_id']=='slot-4' else None,
                grant=grants[window],evaluation_time=decisions[window,r['nf_id']]['evaluation_time']) for r in rows}
            result=deployment.window(int(window),observations)
            assert result['complete'] and len(result['decisions'])==32
            for zone,members in topology.items():
                selected=[row for row in rows if row['nf_id'] in members]
                references[zone].set_window(selected)
                for row in selected:
                    identity=row['nf_id']
                    observation=observations[identity]
                    expected=references[zone].step(row,previous[window,identity],observation['control'],grants[window],observation['evaluation_time'])
                    actual=result['decisions'][identity]
                    for name in ['security','availability']:
                        assert actual[name]==expected['outputs']['ACCUM'][name],(window,identity,name)
                    assert actual['accumulated']==expected['accumulation']['value']
                    count+=1
            stream.write(json.dumps(result)+'\n')
            if window%100==0:
                print('Verified window',window,flush=True)
    report=dict(verified_nf_windows=count,windows=windows,zones=4,leaves=32,processes=37,
        exact_reference_match=True,elapsed_seconds=time.monotonic()-started,models=models,
        source_archives={str(p):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in [calibration_path,evidence_path]},
        scope='Retrospective measured-data replay through actual HTTP processes. No new live 5G acquisition or multi-machine performance claim.')
    (output/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--calibration',type=Path,required=True)
    parser.add_argument('--evidence',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--windows',type=int,default=3600)
    args=parser.parse_args()
    run(args.calibration,args.evidence,args.out,args.windows)
