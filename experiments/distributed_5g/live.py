"""Normal live Open5GS collection with a separately executed detector hierarchy."""
import argparse
import json
from pathlib import Path
import time
from experiments.accumulation_5g.collect import run
from experiments.distributed_5g.deploy import configure, Deployment


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--frozen',type=Path,required=True)
    parser.add_argument('--zone-models',type=Path,required=True)
    parser.add_argument('--nrf',default='http://127.0.0.10:7777')
    parser.add_argument('--base-port',type=int,default=18800)
    args=parser.parse_args()
    args.phase='test'
    args.windows=3600
    args.actors=32
    args.window_seconds=1.
    args.skip_native_restart=False
    args.control_acquisition='late'
    models=json.loads(args.zone_models.read_text())
    topology={zone:model['members'] for zone,model in models.items()}
    if set(sum(topology.values(),[]))!={f'slot-{i}' for i in range(32)}:
        raise ValueError('Zone models must partition all 32 NF positions')
    holder={}

    def factory(context):
        configure(args.out/'hierarchy',topology,models,context['run_id'],context['operator_key'],
                  context['control_key'],context['extensions'],False,args.base_port)
        deployment=Deployment(args.out/'hierarchy')
        holder['deployment']=deployment
        deployment.__enter__()
        stream=(args.out/'distributed_predictions.jsonl').open('w')
        holder['stream']=stream

        def observe(window,rows,control,grant):
            observations={row['nf_id']:dict(row=row,control=control if row['nf_id']=='slot-4' else None,
                grant=grant,evaluation_time=time.time()) for row in rows}
            result=deployment.window(window,observations)
            result['decision_time']=time.time()
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


if __name__=='__main__':
    main()
