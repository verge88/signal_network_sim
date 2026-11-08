"""Bounded independent witness ablation; lab HTTP NF, optionally real NRF."""
import argparse
import copy
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import random
import secrets
import time
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer

from experiments.live_5g.collect import sign_report, verify_report, canonical_profile
from experiments.live_5g.transport import H2Client

CONDITIONS = ['normal', 'authorized_change', 'config_change', 'forged_fresh',
              'replay', 'stale', 'rebound', 'key_compromise']
MAX_AGE = .25


def sha(data):
    return hashlib.sha256(data).hexdigest()


def nf_process(pipe, config):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            raw = Path(config).read_bytes()
            state = json.loads(raw)
            # Configuration changes observable service behavior, not just metadata.
            result = {'answer': 'ALLOW' if state['mode'] == 'normal' else 'DENY',
                      'config_hash': sha(raw), 'pid': os.getpid()}
            body = json.dumps(result).encode()
            self.send_response(200)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
    server = HTTPServer(('127.0.0.1', 0), Handler)
    pipe.send({'port': server.server_port, 'pid': os.getpid()})
    server.serve_forever()


def observer_process(pipe, config, key, instance):
    while True:
        command = pipe.recv()
        if command is None:
            break
        report = {'config_hash': sha(Path(config).read_bytes()), 'instance': instance,
                  'nonce': command, 'timestamp': time.time(), 'pid': os.getpid()}
        pipe.send({'report': report, 'signature': sign_report(report, key)})


def decide(nrf_changed, envelope, nonce, expected, instance, key, now):
    # No scenario/label/ground-truth response is accessible here.
    report = envelope['report']
    metadata_ok = (report['nonce'] == nonce and report['instance'] == instance
                   and abs(now - report['timestamp']) <= MAX_AGE)
    observed_bad = not metadata_ok or report['config_hash'] != expected
    signature_ok = secrets.compare_digest(envelope['signature'], sign_report(report, key))
    return {'B': bool(nrf_changed), 'O': bool(nrf_changed or observed_bad),
            'K': bool(nrf_changed or observed_bad or not signature_ok)}


def run(out, seed, repeats, nrf):
    out.mkdir(parents=True, exist_ok=False)
    config = out/'nf_config.json'
    key = secrets.token_bytes(32)
    instance = str(uuid.uuid4())
    (out/'lab_key.hex').write_text(key.hex())
    ctx = mp.get_context('spawn')
    parent, child = ctx.Pipe()
    nf = ctx.Process(target=nf_process, args=(child, str(config.resolve())))
    observer_parent, observer_child = ctx.Pipe()
    observer = ctx.Process(target=observer_process,
                          args=(observer_child, str(config.resolve()), key, instance))
    normal = json.dumps({'mode': 'normal', 'version': 1}).encode()
    config.write_bytes(normal)
    nf.start()
    if not parent.poll(15):
        nf.terminate()
        raise RuntimeError('NF startup timeout')
    ready = parent.recv()
    observer.start()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    client = H2Client() if nrf else None
    profile = {'nfInstanceId': instance, 'nfType': 'AF', 'nfStatus': 'REGISTERED',
               'ipv4Addresses': ['127.0.0.1'], 'heartBeatTimer': 3600}
    path = '/nnrf-nfm/v1/nf-instances/' + instance
    def request_report(nonce):
        observer_parent.send(nonce)
        if not observer_parent.poll(5):
            raise RuntimeError('Observer timeout')
        return observer_parent.recv()
    rows = []
    try:
        if client:
            registered = client.request('PUT', path, profile)
            (out/'registration.json').write_text(json.dumps(registered))
            if registered['status'] not in (200, 201):
                raise RuntimeError('NRF registration failed: ' + str(registered))
        order = CONDITIONS * repeats
        random.Random(seed).shuffle(order)
        (out/'plan.json').write_text(json.dumps({'seed': seed, 'conditions': order,
             'max_age': MAX_AGE, 'instance': instance, 'live_nrf': nrf,
             'created': time.time(), 'source_sha256': sha(Path(__file__).read_bytes())}))
        with (out/'events.jsonl').open('w') as events:
            for index, condition in enumerate(order):
                allowed = (json.dumps({'mode': 'normal', 'version': index+2}).encode()
                           if condition == 'authorized_change' else normal)
                config.write_bytes(allowed)
                expected = sha(allowed)
                previous = request_report(secrets.token_hex(16))
                onset = time.time()
                attack = condition not in ('normal', 'authorized_change')
                if attack:
                    config.write_bytes(json.dumps({'mode': 'changed', 'version': index+2}).encode())
                with opener.open(f"http://127.0.0.1:{ready['port']}/service", timeout=3) as response:
                    ground_truth = json.load(response)
                nonce = secrets.token_hex(16)
                original = request_report(nonce)
                delivered = copy.deepcopy(original)
                if condition in ('forged_fresh', 'key_compromise'):
                    delivered['report']['config_hash'] = expected
                    if condition == 'key_compromise':
                        delivered['signature'] = sign_report(delivered['report'], key)
                elif condition in ('replay', 'rebound'):
                    delivered = copy.deepcopy(previous)
                    if condition == 'rebound':
                        delivered['report'].update(nonce=nonce, timestamp=time.time())
                elif condition == 'stale':
                    # Actual delay, not a fabricated signed timestamp.
                    time.sleep(MAX_AGE + .1)
                network = client.request('GET', path) if client else None
                if client and network['status'] != 200:
                    raise RuntimeError('NRF read failed: ' + str(network))
                nrf_changed = bool(client and canonical_profile(network['body']) != canonical_profile(profile))
                now = time.time()
                decisions = decide(nrf_changed, delivered, nonce, expected, instance, key, now)
                row = {'index': index, 'condition': condition, 'attack': attack,
                       'onset': onset, 'decision_time': now, 'nonce': nonce,
                       'expected': expected, 'instance': instance, 'nrf_changed': nrf_changed,
                       'ground_truth': ground_truth, 'original': original,
                       'delivered': delivered, 'nrf': network, 'decisions': decisions}
                if ground_truth['config_hash'] != original['report']['config_hash']:
                    raise RuntimeError('Observer and applied configuration differ')
                events.write(json.dumps(row) + '\n')
                events.flush()
                rows.append(row)
        summary = {'seed': seed, 'live_nrf': nrf, 'rows': len(rows), 'conditions': {}}
        for condition in CONDITIONS:
            group = [r for r in rows if r['condition'] == condition]
            summary['conditions'][condition] = {method: sum(r['decisions'][method] for r in group)
                                                for method in ('B', 'O', 'K')}
            summary['conditions'][condition]['n'] = len(group)
        (out/'summary.json').write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary), flush=True)
    finally:
        if client:
            (out/'deregistration.json').write_text(json.dumps(client.request('DELETE', path)))
        observer_parent.send(None)
        observer.join(5)
        if observer.is_alive():
            observer.terminate()
        nf.terminate()
        nf.join(5)
        (out/'sha256.json').write_text(json.dumps({p.name: sha(p.read_bytes())
             for p in out.iterdir() if p.is_file() and p.name != 'sha256.json'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--repeats', type=int, default=20)
    parser.add_argument('--nrf', action='store_true')
    args = parser.parse_args()
    run(args.out, args.seed, args.repeats, args.nrf)
