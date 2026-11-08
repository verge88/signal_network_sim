"""Actual config observer, sharing the previous bounded research NF implementation."""
import copy
import json
import multiprocessing as mp
import secrets
import time
import urllib.request
from experiments.kie_independent.run import nf_process, observer_process, sha
from experiments.live_5g.collect import sign_report

MAX_AGE = .25


def combine(base, envelope, nonce, expected, instance, key, now):
    """No labels or intervention identifiers. Unknown is distinct from alarm."""
    if envelope is None:
        return {'H': 'alarm' if base else 'normal',
                'O': 'alarm' if base else 'unknown', 'K': 'alarm' if base else 'unknown'}
    report = envelope['report']
    fresh = (report['nonce'] == nonce and report['instance'] == instance
             and 0 <= now-report['timestamp'] <= MAX_AGE)
    authenticated = secrets.compare_digest(envelope['signature'], sign_report(report, key))
    mismatch = report['config_hash'] != expected
    return {'H': 'alarm' if base else 'normal',
            'O': 'alarm' if base or (fresh and mismatch) else ('normal' if fresh else 'unknown'),
            'K': 'alarm' if base or not authenticated or (fresh and mismatch)
                 else ('normal' if fresh else 'unknown')}


class Control:
    def __init__(self, root, instance):
        self.root = root
        self.path = root/'actual_nf_config.json'
        self.key = secrets.token_bytes(32)
        (root/'lab_control_key.hex').write_text(self.key.hex())
        self.instance = instance
        self.ctx = mp.get_context('spawn')
        self.allowed = json.dumps({'mode': 'normal', 'version': 1}).encode()
        self.path.write_bytes(self.allowed)
        self.start_nf()
        self.start_observer()
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self.out = (root/'control_events.jsonl').open('w')
        self.delay_until = self.down_until = -1

    def start_nf(self):
        parent, child = self.ctx.Pipe()
        self.nf = self.ctx.Process(target=nf_process, args=(child, str(self.path.resolve())))
        self.nf.start()
        if not parent.poll(15):
            self.nf.terminate()
            raise RuntimeError('Research NF did not start')
        self.ready = parent.recv()

    def start_observer(self):
        self.parent, child = self.ctx.Pipe()
        self.observer = self.ctx.Process(target=observer_process,
                      args=(child, str(self.path.resolve()), self.key, self.instance))
        self.observer.start()

    def prepare(self, window, scenario):
        # Authorized churn recurs across all phases, regardless of labels.
        if window % 90 == 10:
            self.allowed = json.dumps({'mode': 'normal', 'version': window+2}).encode()
        if window % 180 == 20:
            self.nf.terminate()
            self.nf.join(5)
            self.start_nf()
        if window % 120 == 40:
            self.observer.terminate()
            self.observer.join(5)
            self.down_until = window+4
        if window == self.down_until:
            self.start_observer()
        if scenario in ('C', 'F', 'R', 'D'):
            self.path.write_bytes(json.dumps({'mode': 'changed', 'version': window+2}).encode())
        else:
            self.path.write_bytes(self.allowed)

    def observe(self, window, scenario):
        started = time.time()
        with self.opener.open(f"http://127.0.0.1:{self.ready['port']}/service", timeout=3) as response:
            ground_truth = json.load(response)
        nonce = secrets.token_hex(16)
        original = None
        if self.observer.is_alive():
            self.parent.send(nonce)
            if not self.parent.poll(3):
                raise RuntimeError('Unexpected observer IPC timeout')
            original = self.parent.recv()
            if original['report']['config_hash'] != ground_truth['config_hash']:
                raise RuntimeError('Applied configuration differs from independently measured file')
        delivered = copy.deepcopy(original)
        if delivered is not None:
            if scenario in ('F', 'D'):
                delivered['report']['config_hash'] = sha(self.allowed)
            elif scenario == 'R':
                # Authenticated past report; cannot claim the stale state is normal.
                delivered = copy.deepcopy(self.previous)
            if window % 75 in (30, 31):
                time.sleep(MAX_AGE+.1)  # Real benign delivery delay.
        # Attack on availability: suppress an acquired report, without inventing it.
        if scenario == 'D':
            delivered = None
        self.previous = original
        return {'window_id': window, 'started': started, 'completed': time.time(),
                'instance': self.instance, 'nonce': nonce, 'expected': sha(self.allowed),
                'original': original, 'delivered': delivered, 'ground_truth': ground_truth,
                'observer_alive': self.observer.is_alive()}

    def close(self):
        self.observer.terminate()
        self.observer.join(5)
        self.nf.terminate()
        self.nf.join(5)
        self.out.close()
