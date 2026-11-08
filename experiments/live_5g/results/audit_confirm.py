"""Independent evidence audit; does not train, tune, or modify the experiment."""
from pathlib import Path
import collections, datetime, hashlib, json, math, struct, sys, zipfile
import joblib
import numpy as np
import pandas as pd

ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent
RUN = '42395496419'
KEYS = ['run_id', 'window_id', 'nf_id']

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(2**20), b''):
            h.update(part)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def hashes(directory):
    failures = []
    entries = read(directory/'sha256.json')
    for name, expected in entries.items():
        p = directory/name
        if not p.is_file() or sha(p) != expected:
            failures.append(name)
    return {'checked': len(entries), 'failures': failures}

def pcap_check(path):
    result = {'bytes': path.stat().st_size, 'sha256': sha(path), 'records': 0,
              'complete_record_framing': False, 'errors': []}
    with path.open('rb') as f:
        header = f.read(24)
        if len(header) != 24:
            result['errors'].append('short global header'); return result
        magic = header[:4]
        endian = '<' if magic in (b'\xd4\xc3\xb2\xa1', b'\x4d\x3c\xb2\xa1') else '>'
        if magic not in (b'\xd4\xc3\xb2\xa1', b'\xa1\xb2\xc3\xd4', b'\x4d\x3c\xb2\xa1', b'\xa1\xb2\x3c\x4d'):
            result['errors'].append('unknown PCAP magic'); return result
        _, _, _, _, snaplen, linktype = struct.unpack(endian+'HHIIII', header[4:])
        result.update(snaplen=snaplen, linktype=linktype)
        while True:
            record = f.read(16)
            if not record: break
            if len(record) != 16:
                result['errors'].append('truncated record header'); break
            sec, subsec, captured, original = struct.unpack(endian+'IIII', record)
            if captured > snaplen or captured > original:
                result['errors'].append('invalid captured length'); break
            if len(f.read(captured)) != captured:
                result['errors'].append('truncated packet body'); break
            timestamp=sec+subsec/(1e9 if magic in (b'\x4d\x3c\xb2\xa1',b'\xa1\xb2\x3c\x4d') else 1e6)
            if result['records']==0: result['first_timestamp']=timestamp
            result['last_timestamp']=timestamp
            result['records'] += 1
        result['complete_record_framing'] = not result['errors']
    return result

def independent_scores(bundle, frame):
    margins = {}
    for name, columns in bundle['manifest']['features'].items():
        margins[name] = -bundle['models'][name].score_samples(frame[columns].to_numpy(float)) - bundle['thresholds'][name]
    baseline = (frame.profile_policy_violation.to_numpy() > 0) | (frame.heartbeat_age.to_numpy() > 30)
    contextual = baseline | (frame.forbidden_operation.to_numpy() > 0) | (frame.route_mismatch.to_numpy() > 0)
    control = contextual | (frame.report_mismatch.to_numpy() > 0) | (frame.auth_failure.to_numpy() > 0)
    margins['rules'] = np.where(baseline, 1., -1.)
    margins['hybrid_without_KIE'] = np.maximum(margins['IF_CONTEXT'], np.where(contextual, 1., -1.))
    margins['hybrid'] = np.maximum(margins['IF_CONTEXT'], np.where(control, 1., -1.))
    model = bundle['tail_model']
    raw = np.maximum(0, np.max((frame[model['features']].to_numpy(float)-np.array(model['center']))/np.array(model['scale']), axis=1))
    margins['robust_tail'] = raw-bundle['thresholds']['robust_tail']
    margins['hybrid_tail_without_KIE'] = np.maximum(margins['robust_tail'], np.where(contextual, 1., -1.))
    margins['hybrid_tail'] = np.maximum(margins['robust_tail'], np.where(control, 1., -1.))
    return margins

def audit(directory):
    seed = int(directory.name.split('-')[2])
    result = {'seed': seed, 'directory': directory.name, 'source_commit':(directory/'source_commit.txt').read_text().strip(),
              'open5gs_version':(directory/'open5gs_version.txt').read_text().strip(), 'checks': {}, 'issues': []}
    checks = result['checks']
    archive = directory.with_suffix('.zip')
    with zipfile.ZipFile(archive) as z:
        checks['zip_crc_ok'] = z.testzip() is None
    result['archive_sha256'] = sha(archive)
    top_failed, top_count = [], 0
    for line in (directory/'SHA256SUMS').read_text().splitlines():
        checksum, name = line.split(maxsplit=1)
        p = directory/Path(name).relative_to('evidence')
        top_count += 1
        if not p.is_file() or sha(p) != checksum:
            top_failed.append(name)
    result['top_level_hashes'] = {'checked': top_count, 'failures': top_failed}
    checks['all_top_level_hashes_ok'] = not top_failed
    result['pcap'] = pcap_check(directory/'sbi.pcap')
    checks['pcap_complete_framing'] = result['pcap']['complete_record_framing']
    phases, frames, labs = {}, {}, {}
    for phase in ('train', 'cal', 'test'):
        sub = directory/phase
        m, h = read(sub/'manifest.json'), hashes(sub)
        f, l = pd.read_csv(sub/'features.csv'), pd.read_csv(sub/'labels.csv')
        frames[phase], labs[phase] = f, l
        keys_match = set(map(tuple, f[KEYS].to_numpy())) == set(map(tuple, l[KEYS].to_numpy()))
        phases[phase] = {'manifest': m, 'hashes': h, 'rows': len(f), 'label_rows': len(l),
                         'keys_unique': not f.duplicated(KEYS).any() and not l.duplicated(KEYS).any(),
                         'keys_match': keys_match, 'positive_rows': int(l.attack.sum())}
        checks[f'{phase}_numerical_integrity'] = not h['failures'] and keys_match and bool(m.get('complete')) and len(f)==m['windows']*m['actors'] and phases[phase]['keys_unique']
    result['phases'] = phases
    checks['pcap_spans_test_acquisition']=result['pcap'].get('first_timestamp',math.inf)<phases['test']['manifest']['started_at'] and result['pcap'].get('last_timestamp',0)>phases['test']['manifest']['completed_at']
    fm = read(directory/'frozen/manifest.json')
    checks['model_hash_ok'] = sha(directory/'frozen/model.joblib') == fm['model_sha256']
    bundle = joblib.load(directory/'frozen/model.joblib')
    bundle['manifest'] = fm
    checks['normal_training_and_calibration'] = not labs['train'].attack.any() and not labs['cal'].attack.any()
    runsets = [set(frames[p].run_id) for p in ('train','cal','test')]
    checks['disjoint_run_ids'] = not any(runsets[i]&runsets[j] for i,j in ((0,1),(0,2),(1,2)))
    checks['prospective_chronology'] = (phases['train']['manifest']['completed_at'] < phases['cal']['manifest']['started_at'] < phases['cal']['manifest']['completed_at'] < fm['created_at'] < phases['test']['manifest']['started_at'] <= frames['test'].timestamp.min())
    f, l = frames['test'], labs['test']
    scored = independent_scores(bundle, f)
    online = pd.read_csv(directory/'test/online_predictions.csv')
    checks['online_identity_count'] = len(online)==len(f)*9 and not online.duplicated(KEYS+['method']).any()
    replay = {}
    for method, margins in scored.items():
        expected = f[KEYS+['timestamp']].copy()
        expected['expected_score'] = margins
        actual = online[online.method==method]
        merged = expected.merge(actual, on=KEYS, suffixes=('_feature','_online'), validate='one_to_one')
        replay[method] = len(merged)==len(f) and np.allclose(merged.expected_score,merged.score,rtol=1e-10,atol=1e-12) and np.array_equal(merged.expected_score>0,merged.alarm.astype(bool))
        replay[method] = bool(replay[method] and (merged.decision_timestamp>=merged.timestamp_feature).all())
    result['model_replay'] = replay
    checks['all_online_decisions_replayed'] = all(replay.values())
    tail = bundle['tail_model']
    vals = frames['train'][tail['features']].to_numpy(float)
    center = np.median(vals,axis=0)
    floor = np.array([1.,1/max(1.,frames['train'].request_count.max()),1.])
    scale = np.maximum(1.4826*np.median(abs(vals-center),axis=0),floor)
    checks['tail_fit_uses_normal_training'] = bool(np.allclose(center,tail['center']) and np.allclose(scale,tail['scale']))
    threshold_checks = {}
    for name in ('IF_SBI','IF_CONTEXT','IF_FULL','robust_tail'):
        if name=='robust_tail':
            cs=np.maximum(0,np.max((frames['cal'][tail['features']].to_numpy(float)-center)/scale,axis=1))
        else:
            cs=-bundle['models'][name].score_samples(frames['cal'][fm['features'][name]].to_numpy(float))
        rank=math.ceil((len(cs)+1)*(1-bundle['alpha']))
        expected=np.sort(cs)[rank-1] if rank<=len(cs) else math.inf
        threshold_checks[name]=bool(np.isclose(expected,bundle['thresholds'][name]))
    result['calibration_thresholds_recomputed']=threshold_checks
    checks['thresholds_use_only_calibration']=all(threshold_checks.values())
    scenarios=l[l.attack==1].groupby('scenario').size().to_dict()
    result['scenario_exposure']=scenarios
    checks['all_four_scenarios']=set(scenarios)=={'S1','S2','S3','S4'} and all(n==12 for n in scenarios.values()) and l[l.attack==1].episode_id.nunique()==4
    published_episodes=pd.read_csv(directory/'evaluation/episodes.csv')
    episode_checks=[]; own_episodes=[]
    for method in scored:
        annotated=online.loc[online.method==method].merge(l,on=KEYS,validate='one_to_one')
        positive=annotated[annotated.attack==1]
        for identity, part in positive.groupby(['run_id','nf_id','episode_id','scenario']):
            alarms=part[part.alarm==1]
            first=float(alarms.decision_timestamp.min()) if len(alarms) else None
            onset=float(part.onset.min())
            row=dict(zip(['run_id','nf_id','episode_id','scenario'],identity))
            row.update(method=method,detected=bool(len(alarms)),delay_seconds=first-onset if first is not None else None)
            own_episodes.append(row)
            saved=published_episodes[(published_episodes.method==method)&(published_episodes.episode_id==identity[2])]
            ok=len(saved)==1 and bool(saved.iloc[0].detected)==bool(len(alarms))
            if ok and first is not None:
                ok=bool(np.isclose(saved.iloc[0].delay_seconds,first-onset,rtol=1e-8,atol=1e-8))
            episode_checks.append(ok)
    checks['published_episode_detection_and_timing_match']=len(own_episodes)==36 and all(episode_checks)
    result['independent_episodes']=own_episodes
    result['benign_burst_counts']={p:int(((frames[p].request_count>=8)&(frames[p].request_count<=13)&(labs[p].attack==0)).sum()) for p in ('train','cal','test')}
    checks['benign_bursts_present_in_all_phases']=all(n>0 for n in result['benign_burst_counts'].values())
    checks['kie_redundant']=bool((f.report_mismatch==f.profile_policy_violation).all())
    checks['no_auth_failure']=bool((f.auth_failure==0).all())
    routes=[]; raw_sources=collections.Counter(); auth_counts=collections.Counter()
    with (directory/'test/events.jsonl').open(encoding='utf-8') as handle:
        for line in handle:
            event=json.loads(line); raw_sources[event['source']]+=1
            if event['source']=='consumer_route' and event['slot']=='slot-2': routes.append(event)
            if event['source']=='signed_witness': auth_counts[str(event['verified'])]+=1
    s3windows=set(l.loc[(l.scenario=='S3')&(l.attack==1),'window_id'])
    route_checks=[r.get('measurement')=='http_forward' and r.get('http_status')==200 and r.get('served_address')==('127.99.0.3' if r['window_id'] in s3windows else '127.10.0.3') and r.get('receiver_pid')!=r.get('process_pid') and r.get('nonce') for r in routes]
    checks['actual_separate_process_routing']=len(routes)==240 and all(route_checks)
    result['route_observations']={'count':len(routes),'alternate_count':sum(r.get('served_address')=='127.99.0.3' for r in routes),'receiver_pids':sorted({r.get('receiver_pid') for r in routes})}
    result['raw_event_sources']=dict(raw_sources); result['live_auth_verifications']=dict(auth_counts)
    joined=f[KEYS].merge(l,on=KEYS,validate='one_to_one'); y=joined.attack.to_numpy(bool)
    own_metrics={}
    for method, margins in scored.items():
        p=margins>0
        own_metrics[method]={'tp':int(np.sum(y&p)),'fp':int(np.sum(~y&p)),'tn':int(np.sum(~y&~p)),'fn':int(np.sum(y&~p))}
    published=read(directory/'evaluation/report.json')
    checks['published_counts_match']=all(all(own_metrics[m['method']][k]==m[k] for k in ('tp','fp','tn','fn')) for m in published['metrics'])
    result['metrics']=own_metrics
    journal=(directory/'open5gs_journal.txt').read_text(errors='replace')
    result['journal']={'fatal_or_abnormal_lines':[line for line in journal.splitlines() if 'FATAL' in line or 'code=dumped' in line or "Failed with result" in line], 'service_status_bytes':(directory/'service_status.txt').stat().st_size}
    checks['no_native_fatal_exit']=not result['journal']['fatal_or_abnormal_lines']
    checks['empty_service_snapshot']=result['journal']['service_status_bytes']==0
    m=phases['test']['manifest']; oh=read(directory/'overhead/summary.json'); blocks=pd.read_csv(directory/'overhead/blocks.csv')
    result['overhead']={'actors':m['actors'],'collector_cpu_s':m['collector_cpu_seconds'],'elapsed_s':m['elapsed_seconds'],'collector_cpu_percent_one_core':100*m['collector_cpu_seconds']/m['elapsed_seconds'],'collector_peak_rss_mib':m['collector_peak_rss_kib']/1024,'nrf_application_bytes':m['observer_application_bytes'],'witness_application_bytes':m['witness_application_bytes'],'summary':oh,'total_benchmark_cpu_s':float(blocks.benchmark_cpu_seconds.sum()),'total_witness_cpu_s':float(blocks.witness_cpu_seconds.sum())}
    result['checks']={k:bool(v) for k,v in checks.items()}
    return result

def main():
    paths=sorted(ROOT.glob('live-5g-8*'+RUN))
    paths=[p for p in paths if p.is_dir() and (p/'overhead/sha256.json').is_file()]
    if len(paths)!=9: raise SystemExit(f'Awaiting complete downloads: {len(paths)}/9')
    runs=[audit(p) for p in paths]
    numerical_keys=['train_numerical_integrity','cal_numerical_integrity','test_numerical_integrity','model_hash_ok','normal_training_and_calibration','disjoint_run_ids','prospective_chronology','all_online_decisions_replayed','tail_fit_uses_normal_training','thresholds_use_only_calibration','all_four_scenarios','published_counts_match','published_episode_detection_and_timing_match','benign_bursts_present_in_all_phases']
    doc={'audit_date_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'run_id':int(RUN),'observed_runs':len(runs),'all_numerical_checks_pass':all(r['checks'][k] for r in runs for k in numerical_keys),'all_pcap_hashes_match':all(r['checks']['all_top_level_hashes_ok'] for r in runs),'all_pcap_complete_framing':all(r['checks']['pcap_complete_framing'] for r in runs),'all_route_checks_pass':all(r['checks']['actual_separate_process_routing'] for r in runs),'runs':runs}
    (ROOT/'AUDIT_V2.json').write_text(json.dumps(doc,indent=2,ensure_ascii=False),encoding='utf-8')
    lines=['# Аудит подтверждающей серии Open5GS','',f'Проверены все девять артефактов запуска {RUN}. Численные проверки: {doc["all_numerical_checks_pass"]}. Проверки реальной пересылки HTTP в S3: {doc["all_route_checks_pass"]}.','',
    'Проверки выполнены по сырым таблицам и сохранённым моделям. Независимо воспроизведены все девять детекторов, пороги калибровки и параметры статистической ветви. Проверены отсутствие атак в обучении и калибровке, непересечение запусков, замораживание до теста, идентичность онлайн-прогнозов и опубликованных матриц ошибок.','',
    f'Совпадение верхнеуровневых SHA256SUMS для всех файлов: {doc["all_pcap_hashes_match"]}. Целостность структуры записей всех PCAP: {doc["all_pcap_complete_framing"]}. Несовпадение контрольной суммы PCAP не переносится на численные данные; каждый случай раскрыт ниже.','',
    '| Seed | Проверки чисел | PCAP хеш | PCAP записи | CPU сборщика, с | RSS, МиБ | NRF и КИЕ, байт | Добавка p50 цикла, мс |','|---|---|---|---:|---:|---:|---:|---:|']
    for r in runs:
        o=r['overhead']; lines.append(f'| {r["seed"]} | {all(r["checks"][k] for k in numerical_keys)} | {not r["top_level_hashes"]["failures"]} | {r["pcap"]["records"]} | {o["collector_cpu_s"]:.3f} | {o["collector_peak_rss_mib"]:.2f} | {o["nrf_application_bytes"]+o["witness_application_bytes"]} | {o["summary"]["median_paired_differences"]["cycle_ms_p50_on_minus_off"]:.3f} |')
    lines += ['','## Диагностические отклонения','']
    for r in runs:
        if r['top_level_hashes']['failures']: lines.append(f'- {r["seed"]}: несовпадение SHA256SUMS {r["top_level_hashes"]["failures"]}; окончательный SHA-256 PCAP и ZIP сохранён в JSON.')
    lines += ['','Во всех девяти архивах проверен ZIP CRC. Пустой service_status.txt не является снимком работоспособности служб; состояние оценивается только по журналам. Отдельно проверены отсутствие аварийных завершений и структура PCAP. Наиболее вероятная причина устаревшего хеша PCAP — отсутствие ожидания завершения tcpdump перед расчётом SHA256SUMS; это объяснение причины, а не доказанный по таймингам факт.','',
    'CPU/RSS относятся к сборщику и не включают всё ядро. Прикладные байты исключают сетевые заголовки; отдельная парная задержка относится к последовательным блокирующим проверкам, а не к абонентской процедуре.','',
    'Нормальная нагрузка задана ограниченным лабораторным генератором; идеальное разделение не является гарантией на операторском трафике. S3 подтверждён фактическими ответами отдельного процесса получателя; это исследовательский потребитель, не AUSF. N1 не наблюдает простой при блокирующем рестарте. Начало атак согласовано с циклом наблюдения. КИЕ повторяет проверку профиля, поэтому дополнительной обнаружимости не доказано.']
    (ROOT/'AUDIT_V2.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in doc.items() if k!='runs'},ensure_ascii=False))

if __name__=='__main__': main()
