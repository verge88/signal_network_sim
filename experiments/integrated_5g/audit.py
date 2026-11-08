"""Raw-evidence replay and three-state metrics; no collector decision imports."""
import hashlib
import hmac
import json
import math
from pathlib import Path
import sys
import numpy as np
import pandas as pd


def read(p):
    return json.loads(p.read_text())


def audit(root):
    checks = {}
    for phase in ['train', 'cal', 'test']:
        d = root/phase
        sums = read(d/'sha256.json')
        checks[phase+'_hashes'] = all(hashlib.sha256((d/n).read_bytes()).hexdigest() == v for n,v in sums.items())
    train, cal, test = [pd.read_csv(root/p/'features.csv') for p in ['train', 'cal', 'test']]
    labels = pd.read_csv(root/'test/labels.csv').fillna({'episode_id': ''})
    keys = ['run_id', 'window_id', 'nf_id']
    assert not train.empty and not cal.empty and not test.empty
    checks['disjoint'] = not (set(train.run_id)&set(cal.run_id) or set(train.run_id)&set(test.run_id) or set(cal.run_id)&set(test.run_id))
    checks['normal_fit'] = all(not pd.read_csv(root/p/'labels.csv').attack.any() for p in ['train','cal'])
    frozen = read(root/'frozen/manifest.json')
    modelpath = root/'frozen/model.joblib'
    checks['model_hash'] = hashlib.sha256(modelpath.read_bytes()).hexdigest() == frozen['model_sha256']
    checks['frozen_before_test'] = frozen['created_at'] < read(root/'test/manifest.json')['started_at']
    cols = ['request_count','error_rate','update_count']
    center = np.median(train[cols].values, axis=0)
    scale = np.maximum(1.4826*np.median(abs(train[cols].values-center), axis=0), [1,1/max(1,train.request_count.max()),1])
    checks['fit_replayed'] = np.allclose(center, frozen['tail_model']['center']) and np.allclose(scale, frozen['tail_model']['scale'])
    scores = np.maximum(0,np.max((cal[cols].values-center)/scale,axis=1))
    rank = math.ceil((len(cal)+1)*(1-frozen['alpha_target']))
    threshold = np.sort(scores)[rank-1]
    checks['calibration_replayed'] = threshold == frozen['thresholds']['robust_tail']
    base = np.maximum(0,np.max((test[cols].values-center)/scale,axis=1)) > threshold
    base |= (test[['profile_policy_violation','forbidden_operation','route_mismatch','report_mismatch','auth_failure']].max(axis=1)>0).values
    base |= (test.heartbeat_age > 30).values
    base_by_id = {tuple(r):bool(v) for r,v in zip(test[keys].values,base)}
    controls = {r['window_id']:r for r in map(json.loads,(root/'test/control_events.jsonl').read_text().splitlines())}
    key = bytes.fromhex((root/'test/lab_control_key.hex').read_text())
    decisions = list(map(json.loads,(root/'test/integrated_predictions.jsonl').read_text().splitlines()))
    replay_ok = True
    observed_config_ok = True
    for r in decisions:
        b = base_by_id[tuple(r[k] for k in keys)]
        expected = {m:'alarm' if b else 'normal' for m in ['H','O','K']}
        if r['nf_id'] == 'slot-4':
            c = controls[r['window_id']]
            e = c['delivered']
            if e is None:
                expected.update(O='alarm' if b else 'unknown',K='alarm' if b else 'unknown')
            else:
                p = e['report']
                fresh = p['nonce']==c['nonce'] and p['instance']==c['instance'] and 0<=r['decision_time']-p['timestamp']<=.25
                raw = json.dumps(p,sort_keys=True,separators=(',',':')).encode()
                sig = hmac.new(key,raw,hashlib.sha256).hexdigest()
                mismatch = p['config_hash'] != c['expected']
                expected['O'] = 'alarm' if b or (fresh and mismatch) else ('normal' if fresh else 'unknown')
                expected['K'] = 'alarm' if b or not hmac.compare_digest(sig,e['signature']) or (fresh and mismatch) else ('normal' if fresh else 'unknown')
            if c['original']:
                observed_config_ok &= c['original']['report']['config_hash']==c['ground_truth']['config_hash']
                observed_config_ok &= c['original']['report']['pid']!=c['ground_truth']['pid']
        replay_ok &= expected==r['states'] and b==r['base_alarm']
    checks['decisions_replayed'] = replay_ok
    checks['applied_state_measured'] = observed_config_ok
    frame = pd.DataFrame(decisions).merge(labels,on=keys,validate='one_to_one')
    checks['all_rows'] = len(frame)==len(test)==len(labels)
    checks['36_episodes'] = frame[frame.attack==1].episode_id.nunique()==36
    results = {}
    episodes = []
    for scope,group in [('all',frame),('slot-4',frame[frame.nf_id=='slot-4'])]:
        results[scope] = {}
        for method in ['H','O','K']:
            alarm = group.states.map(lambda s:s[method]=='alarm')
            unknown = group.states.map(lambda s:s[method]=='unknown')
            attack = group.attack.astype(bool)
            normal = ~attack
            fp = int((alarm&normal).sum())
            results[scope][method] = dict(tp=int((alarm&attack).sum()),fp=fp,
                fn=int((~alarm&attack).sum()),normal_windows=int(normal.sum()),
                attack_windows=int(attack.sum()),unknown_normal=int((unknown&normal).sum()),
                unknown_attack=int((unknown&attack).sum()),
                false_alarms_per_sampled_nf_hour=fp/(normal.sum()/3600),
                fpr=fp/normal.sum(),scenarios={})
            for scenario,part in group[group.attack==1].groupby('scenario'):
                results[scope][method]['scenarios'][scenario] = dict(n=len(part),alarms=int(part.states.map(lambda s:s[method]=='alarm').sum()),unknown=int(part.states.map(lambda s:s[method]=='unknown').sum()))
    for episode,group in frame[frame.attack==1].groupby('episode_id'):
        for method in ['H','O','K']:
            alarms = group[group.states.map(lambda s:s[method]=='alarm')]
            episodes.append(dict(episode=episode,scenario=group.scenario.iloc[0],method=method,
                 detected=not alarms.empty,delay_seconds=float(alarms.decision_time.min()-group.onset.iloc[0]) if not alarms.empty else None))
    manifest = read(root/'test/manifest.json')
    overhead = dict(elapsed_seconds=manifest['elapsed_seconds'],collector_cpu_seconds=manifest['collector_cpu_seconds'],
                    collector_peak_rss_kib=manifest.get('collector_peak_rss_kib'),
                    control_transaction_p50_ms=float(np.median([(c['completed']-c['started'])*1000 for c in controls.values()])),
                    control_transaction_p95_ms=float(np.percentile([(c['completed']-c['started'])*1000 for c in controls.values()],95)))
    result = dict(checks={k:bool(v) for k,v in checks.items()},metrics=results,episodes=episodes,overhead=overhead)
    (root/'integrated_summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({'audit_pass':all(checks.values()),'metrics':results,'overhead':overhead}),flush=True)
    assert all(checks.values()), checks
    return result


if __name__=='__main__':
    audit(Path(sys.argv[1]))
