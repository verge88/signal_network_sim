"""Generate per-host configs and run a multi-process local deployment."""
import concurrent.futures
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import time
import urllib.request
from experiments.distributed_5g.protocol import post, sign


def get(url, path):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url + path, timeout=2) as response:
        return json.load(response)


def configure(directory, topology, models, source_run, operator_key, control_key, extensions,
              replay_mode=False, base_port=18800, *, hosts=None, include_if_context=False,
              confirm_if_context=False, confirmation_fraction=0.5, recover_normal=False):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if (directory/'source.json').exists():
        raise FileExistsError('Deployment already exists')
    all_members = [member for members in topology.values() for member in members]
    if len(set(all_members)) != len(all_members) or any(len(members)<3 for members in topology.values()):
        raise ValueError('Zones need at least three distinct, exclusively assigned NF')
    identities = ['root'] + list(topology) + all_members
    if len(set(identities)) != len(identities):
        raise ValueError('Node identities must be unique')
    hosts = {identity: '127.0.0.1' for identity in identities} if hosts is None else hosts
    if set(hosts) != set(identities):
        raise ValueError('Placement must cover exactly the configured nodes')
    if confirm_if_context and not include_if_context:
        raise ValueError('Confirmation requires IF_CONTEXT decisions')
    if not 0 < confirmation_fraction <= 1:
        raise ValueError('Confirmation fraction must be in (0, 1]')
    epoch = secrets.token_hex(16)
    source_keys = {identity:secrets.token_hex(32) for identity in identities}
    configs = {}
    for index, identity in enumerate(identities):
        configs[identity] = dict(id=identity, epoch=epoch, host=hosts[identity], port=base_port+index,
            state_file=str(directory/(identity+'-state.json')), incoming_keys={'source':source_keys[identity]},
            members=[], deadline_seconds=12 if identity=='root' else 8,
            include_if_context=include_if_context, confirm_if_context=confirm_if_context,
            confirmation_fraction=confirmation_fraction, recover_normal=recover_normal)
    urls = {identity: f"http://{config['host']}:{config['port']}" for identity, config in configs.items()}
    configs['root'].update(role='root', members=list(topology), zone_members=topology)
    for zone, members in topology.items():
        zone_key = secrets.token_hex(32)
        configs['root']['incoming_keys'][zone] = zone_key
        configs[zone].update(role='zone', members=members, model=models[zone],
            parent=dict(id='root', url=urls['root'], key=zone_key))
        for identity in members:
            leaf_key = secrets.token_hex(32)
            configs[zone]['incoming_keys'][identity] = leaf_key
            configs[identity].update(role='leaf', source_run=source_run, model=models[zone],
                operator_key=operator_key, control_key=control_key if identity=='slot-4' else None,
                availability_seconds=extensions['availability_seconds'], extensions=extensions,
                replay_mode=replay_mode,
                parent=dict(id=zone,url=urls[zone],key=leaf_key))
    source = dict(epoch=epoch, keys=source_keys, urls=urls, topology=topology)
    for identity, config in configs.items():
        (directory/(identity+'.json')).write_text(json.dumps(config,indent=2))
    (directory/'source.json').write_text(json.dumps(source,indent=2))
    return source


class Deployment:
    def __init__(self, directory, launch_prefixes=None):
        self.directory = Path(directory)
        self.source = json.loads((self.directory/'source.json').read_text())
        self.processes = {}
        self.sequence = {identity:0 for identity in self.source['urls']}
        self.logs = []
        self.launch_prefixes = launch_prefixes or {}
        self.pid_files = {}

    def start(self, identity):
        log = (self.directory/(identity+'.log')).open('a')
        self.logs.append(log)
        command = [sys.executable, '-m', 'experiments.distributed_5g.node', '--config',
                   str(self.directory/(identity+'.json'))]
        prefix = self.launch_prefixes.get(identity, ())
        if prefix:
            pid_file = self.directory.resolve() / (identity + '.pid')
            pid_file.unlink(missing_ok=True)
            self.pid_files[identity] = pid_file
            command.extend(['--pid-file', str(pid_file)])
        self.processes[identity] = subprocess.Popen(
            [*prefix, *command], stdout=log, stderr=subprocess.STDOUT)

    def __enter__(self):
        try:
            for identity in self.source['urls']:
                self.start(identity)
        except BaseException:
            self.__exit__(None,None,None)
            raise
        deadline = time.monotonic()+90
        pending = set(self.processes)
        while pending and time.monotonic()<deadline:
            for identity in list(pending):
                try:
                    health = get(self.source['urls'][identity],'/health')
                    if health['id'] != identity:
                        raise RuntimeError(f'Wrong node at {self.source["urls"][identity]}')
                    pending.remove(identity)
                except OSError:
                    pass
            time.sleep(.1)
        if pending:
            self.__exit__(None,None,None)
            raise RuntimeError('Nodes failed to start: '+str(pending))
        return self

    def stop(self, identity):
        process=self.processes[identity]
        pid_file = self.pid_files.get(identity)
        if pid_file and pid_file.exists():
            # sudo/ip may interpose a monitor instead of forwarding SIGTERM.
            # Use the PID written by this node, not a PID from a network response.
            pid = int(pid_file.read_text())
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        else:
            process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)

    def __exit__(self, *args):
        for identity, process in self.processes.items():
            if process.poll() is None:
                self.stop(identity)
        for log in self.logs:
            log.close()

    def send(self, identity, body):
        self.sequence[identity] += 1
        packet=sign('source',identity,self.source['epoch'],self.sequence[identity],body,self.source['keys'][identity])
        return post(self.source['urls'][identity],packet)

    def window(self, window, observations, missing=()):
        self.send('root',dict(kind='begin',window=window))
        for zone in self.source['topology']:
            if zone not in missing:
                self.send(zone,dict(kind='begin',window=window))
        def submit(item):
            identity, observation=item
            if identity not in missing:
                self.send(identity,dict(kind='observe',window=window,**observation))
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(submit,observations.items()))
        deadline=time.monotonic()+20
        while time.monotonic()<deadline:
            result=get(self.source['urls']['root'],'/latest')
            if result.get('window')==window:
                return result
            time.sleep(.05)
        raise TimeoutError('Root did not close the window')
