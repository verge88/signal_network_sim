"""Normal live Open5GS collection with a separately executed detector hierarchy."""
import argparse
import json
import ipaddress
import os
from pathlib import Path
import statistics
import time
from experiments.accumulation_5g.collect import run
from experiments.distributed_5g.deploy import configure, Deployment, get
from experiments.live_5g.evaluate import sha256


def namespace_placement(path, topology):
    groups = json.loads(path.read_text())
    if set(groups) != {'root', *topology}:
        raise ValueError('Placement must specify the coordinator and every zone')
    hosts, prefixes = {}, {}
    for group, config in groups.items():
        address = str(ipaddress.IPv4Address(config['address']))
        prefix = ['sudo', '-n', 'ip', 'netns', 'exec', config['namespace'], 'setpriv',
                  '--reuid', str(os.getuid()), '--regid', str(os.getgid()), '--clear-groups']
        for identity in [group, *topology.get(group, ())]:
            hosts[identity], prefixes[identity] = address, prefix
    return hosts, prefixes


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--frozen',type=Path,required=True)
    parser.add_argument('--zone-models',type=Path,required=True)
    parser.add_argument('--nrf',default='http://127.0.0.10:7777')
    parser.add_argument('--base-port',type=int,default=18800)
    parser.add_argument('--smoke-windows', type=int, choices=range(1, 30))
    parser.add_argument('--namespace-placement', type=Path)
    parser.add_argument('--include-if-context', action='store_true')
    args=parser.parse_args()
    args.phase='test'
    args.normal_smoke = args.smoke_windows is not None
    args.windows = args.smoke_windows if args.normal_smoke else 3600
    args.actors=32
    args.window_seconds=1.
    args.skip_native_restart=args.normal_smoke
    args.control_acquisition='late'
    models=json.loads(args.zone_models.read_text())
    topology={zone:model['members'] for zone,model in models.items()}
    if set(sum(topology.values(),[]))!={f'slot-{i}' for i in range(32)}:
        raise ValueError('Zone models must partition all 32 NF positions')
    holder={}
    hosts, prefixes = namespace_placement(args.namespace_placement, topology) if args.namespace_placement else (None, None)
    durations = []
    alerts = dict.fromkeys(['rules', 'IF_SBI', 'IF_CONTEXT', 'ACCUM', 'ACCUM_OR_IF_CONTEXT'], 0)

    def factory(context):
        configure(args.out/'hierarchy',topology,models,context['run_id'],context['operator_key'],
                  context['control_key'],context['extensions'],False,args.base_port,
                  hosts=hosts, include_if_context=args.include_if_context)
        deployment=Deployment(args.out/'hierarchy', prefixes)
        deployment.__enter__()
        holder['deployment']=deployment
        if args.normal_smoke:
            health = {identity: get(url, '/health') for identity, url in deployment.source['urls'].items()}
            assert len({node['pid'] for node in health.values()}) == 37
            if args.namespace_placement:
                namespaces = {group: health[group]['netns'] for group in ['root', *topology]}
                assert None not in namespaces.values() and len(set(namespaces.values())) == 5
                assert Path('/proc/self/ns/net').stat().st_ino not in namespaces.values()
                assert all(health[nf]['netns'] == namespaces[zone] for zone, members in topology.items() for nf in members)
                holder['namespaces'] = namespaces
        stream=(args.out/'distributed_predictions.jsonl').open('w')
        holder['stream']=stream

        def observe(window,rows,control,grant,predictions):
            observations={row['nf_id']:dict(row=row,control=control if row['nf_id']=='slot-4' else None,
                grant=grant,evaluation_time=time.time()) for row in rows}
            if args.include_if_context:
                flags = predictions[predictions.method == 'IF_CONTEXT'].set_index('nf_id').alarm
                for identity, observation in observations.items():
                    observation['if_context_alarm'] = bool(flags.loc[identity])
            started = time.monotonic()
            result=deployment.window(window,observations)
            result['decision_time']=time.time()
            if args.normal_smoke:
                assert result['complete'] and set(result['decisions']) == set(observations)
                durations.append(time.monotonic() - started)
                for method in ('rules', 'IF_SBI', 'IF_CONTEXT'):
                    alerts[method] += int(predictions.loc[predictions.method == method, 'alarm'].sum())
                for identity, decision in result['decisions'].items():
                    if args.include_if_context:
                        assert decision['if_context_alarm'] == observations[identity]['if_context_alarm']
                        assert decision['security'] == (decision['accum_security'] or decision['if_context_alarm'])
                        alerts['ACCUM_OR_IF_CONTEXT'] += int(decision['security'])
                    alerts['ACCUM'] += int(decision.get('accum_security', decision['security']))
            stream.write(json.dumps(result)+'\n')
            stream.flush()
        return observe

    args.distributed_observer_factory=factory
    try:
        run(args)
    finally:
        if 'stream' in holder:
            holder['stream'].close()
        if 'deployment' in holder:
            holder['deployment'].__exit__(None,None,None)
    if args.normal_smoke:
        assert len(durations) == args.windows
        if not args.include_if_context:
            alerts.pop('ACCUM_OR_IF_CONTEXT')
        report = dict(windows=args.windows, nf_windows=args.windows * args.actors, processes=37,
            network_namespaces=holder.get('namespaces'), alerts=alerts,
            transport_seconds=dict(median=statistics.median(durations), maximum=max(durations)),
            manifest_sha256=sha256(args.out/'manifest.json'), model_sha256=sha256(args.frozen/'model.joblib'),
            zone_models_sha256=sha256(args.zone_models),
            scope='Fresh normal Open5GS SBI observations. Short engineering smoke on one physical runner.')
        (args.out/'smoke_report.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report), flush=True)


if __name__=='__main__':
    main()
