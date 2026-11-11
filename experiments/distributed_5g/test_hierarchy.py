import copy
import json
import time
import urllib.error
import pytest
from experiments.distributed_5g.deploy import configure, Deployment, get
from experiments.distributed_5g.node import Node
from experiments.distributed_5g.protocol import sign, post
from experiments.distributed_5g.primitives import peer_residuals, advance


def configuration(tmp_path, base_port=19300):
    topology={'zone-a':['slot-0','slot-1','slot-2'],'zone-b':['slot-3','slot-5','slot-6']}
    model=dict(threshold=2.,feature_thresholds=dict(request_count=49.,error_count=1.,update_count=1.))
    configure(tmp_path,topology,{z:model for z in topology},'fixture','11'*32,'22'*32,{'availability_seconds':6},True,base_port)
    for identity in ['root',*topology]:
        p=tmp_path/(identity+'.json');c=json.loads(p.read_text());c['deadline_seconds']=1 if identity=='root' else .4;p.write_text(json.dumps(c))
    return topology


def observations(topology, outlier=False):
    result={}
    for identity in sum(topology.values(),[]):
        row=dict(run_id='fixture',nf_id=identity,timestamp=time.time(),request_count=10.,error_rate=0.,update_count=0.,
            profile_policy_violation=0.,heartbeat_age=0.,forbidden_operation=0.,route_mismatch=0.,report_mismatch=0.,auth_failure=0.)
        if identity=='slot-0' and outlier:row['request_count']=20.
        result[identity]=dict(row=row,evaluation_time=time.time(),grant=None,control=None)
    return result


def test_real_process_hierarchy_missing_leaf_zone_and_restart(tmp_path):
    topology=configuration(tmp_path)
    with Deployment(tmp_path) as deployment:
        pids={get(url,'/health')['pid'] for url in deployment.source['urls'].values()}
        assert len(pids)==9
        normal=deployment.window(0,observations(topology))
        assert normal['complete'] and not any(d['security'] for d in normal['decisions'].values())
        alarm=deployment.window(1,observations(topology,True))
        assert alarm['decisions']['slot-0']['security']
        assert not any(alarm['decisions'][nf]['security'] for nf in topology['zone-b'])
        deployment.stop('slot-1')
        missing=deployment.window(2,observations(topology),missing={'slot-1'})
        assert not missing['complete'] and missing['decisions']['slot-1']['availability']
        assert missing['decisions']['slot-0']['unknown']
        deployment.start('slot-1')
        time.sleep(.3)
        recovered=deployment.window(3,observations(topology))
        assert recovered['complete']
        deployment.stop('zone-b')
        missing_zone=deployment.window(4,observations(topology),missing={'zone-b',*topology['zone-b']})
        assert missing_zone['missing_zones']==['zone-b']
        assert all(missing_zone['decisions'][nf]['availability'] for nf in topology['zone-b'])


def test_authentication_idempotence_and_durable_replay_guard(tmp_path):
    topology=configuration(tmp_path,19400)
    config=json.loads((tmp_path/'slot-0.json').read_text())
    node=Node(config)
    body=dict(kind='observe',window=0,**observations(topology)['slot-0'])
    packet=sign('source','slot-0',config['epoch'],1,body,config['incoming_keys']['source'])
    assert node.receive(packet)['accepted']
    restarted=Node(config)
    assert restarted.receive(packet)['duplicate'] and len(restarted.state['outbox'])==1
    tampered=copy.deepcopy(packet);tampered['body']['row']['request_count']=100
    with pytest.raises(ValueError):restarted.receive(tampered)
    conflicting=sign('source','slot-0',config['epoch'],1,dict(body,window=1),config['incoming_keys']['source'])
    with pytest.raises(ValueError):restarted.receive(conflicting)


def test_lightweight_math_matches_reference():
    from experiments.accumulation_5g.detector import peer_residuals as reference
    rows=[dict(nf_id=str(i),request_count=i%5+1) for i in range(32)]
    assert peer_residuals(rows)==reference(rows)
    assert advance(10.,0.,False)==9.4


def test_windows_state_replacement_retries_without_false_ack(tmp_path, monkeypatch):
    from experiments.distributed_5g import node as module
    topology=configuration(tmp_path,19500)
    config=json.loads((tmp_path/'slot-0.json').read_text())
    node=Node(config)
    real_replace=module.os.replace
    attempts=[]
    def denied():
        error=PermissionError('Injected Windows file sharing failure')
        error.winerror=5
        return error
    def transient(source,target):
        attempts.append(1)
        if len(attempts)<3:raise denied()
        return real_replace(source,target)
    monkeypatch.setattr(module.os,'replace',transient)
    monkeypatch.setattr(module.time,'sleep',lambda seconds:None)
    body=dict(kind='observe',window=0,**observations(topology)['slot-0'])
    packet=sign('source','slot-0',config['epoch'],1,body,config['incoming_keys']['source'])
    assert node.receive(packet)['accepted'] and len(attempts)==3
    persisted=json.loads(node.path.read_text())
    assert persisted['high_water']==0 and len(persisted['outbox'])==1
    def permanent(source,target):
        attempts.append(1)
        raise denied()
    attempts.clear()
    monkeypatch.setattr(module.os,'replace',permanent)
    packet=sign('source','slot-0',config['epoch'],2,dict(body,window=1),config['incoming_keys']['source'])
    with pytest.raises(PermissionError):node.receive(packet)
    assert len(attempts)==21
    assert node.state==persisted==json.loads(node.path.read_text())
