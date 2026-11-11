"""Local verification of immutable CI archives, streaming large raw event files."""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile
import numpy as np
import pandas as pd
from experiments.distributed_5g import full_experiment as experiment
from experiments.distributed_5g.diagnose_full import diagnose
from experiments.live_5g.evaluate import fit, load_frozen, score_frame, sha256


def run(seed):
    root=Path(__file__).resolve().parent
    artifacts=json.loads((root/'artifacts.json').read_text(encoding='utf-8-sig'))
    artifact=next(a for a in artifacts if a['name'].startswith(f'full-method-evidence-{seed}-'))
    outer=root/f'{seed}-evidence.zip'
    assert 'sha256:'+sha256(outer)==artifact['digest']
    target=root/f'raw-{seed}'
    target.mkdir(exist_ok=True)
    with ZipFile(outer) as archive:
        assert archive.namelist()==['full-evidence.zip']
        archive.extract('full-evidence.zip',target)
    inner=target/'full-evidence.zip'
    wanted={'source_commit.txt','zone_models.json','frozen_origin.json'}
    for phase in ('train','cal'):
        wanted.update(f'{phase}/{name}' for name in ('features.csv','labels.csv','manifest.json'))
    wanted.update(f'frozen/{name}' for name in ('model.joblib','manifest.json','extensions.json','refinement.json','accumulation.json'))
    wanted.update(f'test/{name}' for name in ('manifest.json','features.csv','labels.csv','online_predictions.csv',
        'distributed_predictions.jsonl','load_grants.jsonl','lab_operator_key.hex','smoke_report.json'))
    with ZipFile(inner) as archive:
        assert wanted.issubset(archive.namelist())
        for name in wanted:
            assert (target/name).resolve().is_relative_to(target.resolve())
            archive.extract(name,target)
        raw_size=archive.getinfo('test/events.jsonl').file_size
    expected=json.loads((root/str(seed)/'report.json').read_text())
    integrity=json.loads((root/str(seed)/'integrity.json').read_text())
    for name,digest in integrity.items():
        assert sha256(target/name)==digest
    frozen=load_frozen(target/'frozen')
    for item in frozen['manifest']['sources']:
        relative=Path(item['path']).relative_to('evidence')
        assert sha256(target/relative)==item['sha256']
    fit(target/'train',target/'cal',target/'refit',alpha=.001,seed=seed)
    refit=load_frozen(target/'refit')
    for name,threshold in frozen['thresholds'].items():
        assert abs(refit['thresholds'][name]-threshold)<1e-12
    sample=pd.read_csv(target/'test/features.csv',nrows=1024,float_precision='round_trip')
    a,b=score_frame(frozen,sample),score_frame(refit,sample)
    assert np.array_equal(a.alarm,b.alarm) and np.allclose(a.score,b.score,rtol=0,atol=1e-12)
    original_lines=experiment.lines
    def streamed_lines(path):
        if path==target/'test/events.jsonl':
            with ZipFile(inner) as archive,archive.open('test/events.jsonl') as stream:
                for line in stream:
                    yield json.loads(line)
        else:
            yield from original_lines(path)
    experiment.lines=streamed_lines
    experiment.audit(target)
    assert json.loads((target/'summary/report.json').read_text())==expected
    diagnose(target,root/str(seed)/'diagnostics.json')
    result=dict(seed=seed,artifact_sha256=sha256(outer),inner_sha256=sha256(inner),
        raw_event_bytes=raw_size,full_audit_reproduced=True,training_sources_hashes_verified=True,
        fresh_fit_thresholds_reproduced=True,refit_predictions_checked=1024,
        note='Original audit arithmetic unchanged; only input reader streams raw events from ZIP to limit disk use.')
    (root/str(seed)/'local_audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('seed',type=int)
    run(parser.parse_args().seed)
