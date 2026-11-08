import json
import pandas as pd
import pytest
from experiments.calibrated_5g.detector import Engine,Temporal,fit

def row(count=5,error=0.):
    return dict(run_id='test',nf_id='slot-3',timestamp=0,request_count=count,error_rate=error,update_count=1,
        profile_policy_violation=0,heartbeat_age=1,forbidden_operation=0,route_mismatch=0,report_mismatch=0,auth_failure=0)

def test_load_tail_cannot_mask_error_tail():
    engine=Engine(dict(feature_thresholds={'request_count':37,'error_rate':.08,'update_count':1},persistence_threshold=1,carry_threshold=14),b'key',b'key')
    old=dict(signals=dict(replay=False,availability=False))
    assert not engine.step(row(37),old,None,None,0)['outputs']['SPLIT']['security']
    assert engine.step(row(25,.32),old,None,None,0)['outputs']['SPLIT']['security']

def test_two_window_confirmation_rejects_short_normal_streak():
    normal=Temporal(); growing=Temporal()
    assert max(normal.step(n,False)[0] for n in [2,3,4,5,6,7,2])==1
    assert max(growing.step(n,False)[0] for n in [5,6,7,8,9,10,11,12])>1

def test_carry_preserves_authorized_history_but_masks_current_permission():
    trend=Temporal()
    for n in range(5,10):
        assert trend.step(n,True)==(0.,0.)
    persistent,carried=trend.step(10,False)
    assert persistent==0 and carried==2

def test_fit_is_normal_only_and_immutable(tmp_path):
    (tmp_path/'frozen').mkdir()
    for phase in ['train','cal']:
        directory=tmp_path/phase; directory.mkdir()
        rows=[dict(row(5+(w%5)),window_id=w,nf_id=f'slot-{nf}') for w in range(600) for nf in range(8)]
        pd.DataFrame(rows).to_csv(directory/'features.csv',index=False)
        pd.DataFrame({'attack':[0]*len(rows)}).to_csv(directory/'labels.csv',index=False)
        (directory/'lab_operator_key.hex').write_text(b'key'.hex())
        (directory/'load_grants.jsonl').write_text('\n'.join(json.dumps(dict(window_id=w,envelope=None)) for w in range(600)))
    fit(tmp_path)
    model=json.loads((tmp_path/'frozen/refinement.json').read_text())
    assert model['feature_thresholds']['request_count']==9
    with pytest.raises(FileExistsError):
        fit(tmp_path)
    (tmp_path/'frozen/refinement.json').unlink()
    pd.DataFrame({'attack':[1]}).to_csv(tmp_path/'cal/labels.csv',index=False)
    with pytest.raises(AssertionError):
        fit(tmp_path)
