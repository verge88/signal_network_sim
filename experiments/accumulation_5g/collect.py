"""Collect a bounded live NRF experiment; no simulated numerical feature injection.

Native NSSF slot 0 and software NF actors use a real Open5GS NRF. The separate
HTTP witness process is a research control channel, not a 3GPP KIE extension or
hardware attestation. A consumer cache fault is not an Open5GS CVE exploit.
"""
from __future__ import annotations

import argparse
import collections
import copy
import csv
import hashlib
import hmac
import json
import multiprocessing
import os
from pathlib import Path
import random
import secrets
import socketserver
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from experiments.live_5g.transport import H2Client, loopback_url
from experiments.live_5g.evaluate import load_frozen, score_frame
from experiments.integrated_5g.control import combine
from experiments.robustness_5g.control import Control
from experiments.freshness_5g.acquisition import Acquisition
from experiments.robustness_5g.design import attack_schedule, normal_count, growth_count, regime, s5_members, s5_count
from experiments.resilient_5g.detector import Engine
from experiments.accumulation_5g.detector import Engine as RefinedEngine


FEATURES = ['request_count', 'error_rate', 'update_count', 'interarrival_mean',
            'profile_changes', 'forbidden_operation', 'profile_policy_violation',
            'route_mismatch', 'report_mismatch', 'auth_failure', 'request_count_10s',
            'request_count_60s', 'error_rate_10s', 'error_rate_60s', 'latency_ms', 'heartbeat_age']


def canonical_profile(profile):
    """Order-invariant tested fields; exclude NRF-managed heartbeatTimer/load."""
    services = []
    for item in profile.get('nfServices', []):
        services.append({'name': item.get('serviceName'), 'status': item.get('nfServiceStatus'),
                         'scheme': item.get('scheme'),
                         'endpoints': sorted((str(x.get('ipv4Address', '')), int(x.get('port', 0)))
                                             for x in item.get('ipEndPoints', []))})
    return {'id': profile.get('nfInstanceId'), 'type': profile.get('nfType'),
            'status': profile.get('nfStatus'),
            'addresses': sorted(profile.get('ipv4Addresses', [])),
            'services': sorted(services, key=lambda x: json.dumps(x, sort_keys=True))}


def digest(profile):
    return hashlib.sha256(json.dumps(canonical_profile(profile), sort_keys=True).encode()).hexdigest()


def route_address(profile):
    addresses = profile.get('ipv4Addresses', [])
    if addresses:
        return sorted(addresses)[0]
    for service in profile.get('nfServices', []):
        for endpoint in service.get('ipEndPoints', []):
            if endpoint.get('ipv4Address'):
                return endpoint['ipv4Address']
    return ''


def sign_report(report, key):
    raw = json.dumps(report, sort_keys=True, separators=(',', ':')).encode()
    return hmac.new(key, raw, hashlib.sha256).hexdigest()


def verify_report(envelope, nonce, key, max_age=5):
    try:
        report = envelope.get('report', {})
        return (report.get('nonce') == nonce and abs(time.time() - float(report.get('timestamp', 0))) <= max_age
                and hmac.compare_digest(envelope.get('signature', ''), sign_report(report, key)))
    except (AttributeError, TypeError, ValueError):
        return False


class LoopbackHTTPServer(ThreadingHTTPServer):
    def server_bind(self):
        # HTTPServer's reverse DNS lookup can stall on arbitrary 127/8 aliases.
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]


def receiver_process(pipe, addresses):
    """Research HTTP service endpoints, separate from the consumer/witness."""
    servers = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            query = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
            result = {'served_address': self.connection.getsockname()[0],
                      'receiver_pid': os.getpid(), 'nonce': query.get('nonce', [''])[0],
                      'received_at': time.time()}
            raw = json.dumps(result).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    try:
        for address in addresses:
            loopback_url(f'http://{address}:8080')
            server = LoopbackHTTPServer((address, 8080), Handler)
            servers.append(server)
            threading.Thread(target=server.serve_forever, daemon=True).start()
        pipe.send({'pid': os.getpid(), 'addresses': addresses})
        pipe.close()
        threading.Event().wait()
    except Exception as exc:
        pipe.send({'error': str(exc)})
    finally:
        for server in servers:
            server.server_close()


def witness_process(pipe, key, initial):
    """Separate process owns the normative software witness and consumer cache.

    /control is an operator-only loopback research API protected with the random
    control token. /report proves software-witness origin, not truthful NF internals.
    """
    states = copy.deepcopy(initial)
    token = secrets.token_hex(32)
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send_json(self, value, status=200):
            raw = json.dumps(value).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            parsed = urllib.parse.urlsplit(self.path)
            query = urllib.parse.parse_qs(parsed.query)
            slot = query.get('slot', [''])[0]
            with lock:
                state = states.get(slot)
                if state is None:
                    return self.send_json({'error': 'unknown slot'}, 404)
                if parsed.path == '/report':
                    report = {'slot': slot, 'profile_hash': digest(state['profile']),
                              'timestamp': time.time(), 'nonce': query.get('nonce', [''])[0],
                              'process_pid': os.getpid(), 'policy_version': state['version']}
                    return self.send_json({'report': report, 'signature': sign_report(report, key)})
                if parsed.path == '/route':
                    state['route_reads'] = state.get('route_reads', 0) + 1
                    result = {'slot': slot, 'selected_address': state['route'],
                              'timestamp': time.time(), 'process_pid': os.getpid(),
                              'route_reads': state['route_reads'], 'measurement': 'cache_only'}
                    if state.get('forward_addresses'):
                        # Read the destination's response rather than treating
                        # the consumer's configured address as delivered traffic.
                        address = state['route']
                        if address not in state['forward_addresses']:
                            return self.send_json({'error': 'unprovisioned loopback destination'}, 400)
                        nonce = secrets.token_hex(16)
                        url = f'http://{address}:8080/identity?nonce={nonce}'
                        loopback_url(url)
                        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                        started = time.time()
                        try:
                            with opener.open(url, timeout=2) as response:
                                payload = response.read()
                                delivered = json.loads(payload)
                                if delivered.get('nonce') != nonce:
                                    raise ValueError('Receiver response nonce mismatch')
                                result.update(delivered, http_status=response.status,
                                              response_bytes=len(payload), request_started=started,
                                              response_completed=time.time(), measurement='http_forward')
                        except (OSError, ValueError) as exc:
                            return self.send_json({'error': str(exc), 'selected_address': address}, 502)
                    return self.send_json(result)
            self.send_json({'error': 'unknown endpoint'}, 404)

        def do_POST(self):
            if self.headers.get('X-Lab-Control') != token:
                return self.send_json({'error': 'unauthorized'}, 403)
            command = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            with lock:
                states[command['slot']].update(command['state'])
            self.send_json({'ok': True})

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    pipe.send((server.server_port, token, os.getpid()))
    pipe.close()
    server.serve_forever(poll_interval=.1)


class Witness:
    def __init__(self, profiles):
        self.key = secrets.token_bytes(32)
        parent, child = multiprocessing.Pipe()
        initial = {slot: {'profile': p, 'route': route_address(p), 'version': 1}
                   for slot, p in profiles.items()}
        self.receiver = None
        self.receiver_pid = None
        if 'slot-2' in profiles:
            addresses = [route_address(profiles['slot-2']), '127.99.0.3']
            for address in addresses:
                loopback_url(f'http://{address}:8080')
            receiver_parent, receiver_child = multiprocessing.Pipe()
            self.receiver = multiprocessing.Process(target=receiver_process,
                args=(receiver_child, addresses), daemon=True)
            self.receiver.start()
            if not receiver_parent.poll(10):
                self.receiver.terminate()
                self.receiver.join(timeout=5)
                raise RuntimeError('Research receiver failed to start')
            ready = receiver_parent.recv()
            if 'error' in ready:
                self.receiver.join(timeout=5)
                raise RuntimeError('Research receiver: ' + ready['error'])
            self.receiver_pid = ready['pid']
            initial['slot-2']['forward_addresses'] = addresses
        self.process = multiprocessing.Process(target=witness_process, args=(child, self.key, initial), daemon=True)
        self.process.start()
        if not parent.poll(10):
            if self.receiver is not None:
                self.receiver.terminate()
                self.receiver.join(timeout=5)
            raise RuntimeError('Witness process failed to start')
        port, self.token, self.pid = parent.recv()
        self.base = f'http://127.0.0.1:{port}'
        self.bytes = 0
        self.requests = 0

    def request(self, path, data=None):
        encoded = None if data is None else json.dumps(data).encode()
        req = urllib.request.Request(self.base + path, data=encoded,
                                     headers={'Content-Type': 'application/json', 'X-Lab-Control': self.token})
        started = time.time()
        # Local control traffic must never be sent to a runner's HTTP proxy.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(req, timeout=4) as response:
            raw = response.read()
        self.bytes += len(raw) + len(encoded or b'')
        self.requests += 1
        return json.loads(raw), (time.time() - started) * 1000

    def update(self, slot, **state):
        return self.request('/control', {'slot': slot, 'state': state})

    def observe(self, slot):
        nonce = secrets.token_hex(16)
        envelope, latency = self.request('/report?' + urllib.parse.urlencode({'slot': slot, 'nonce': nonce}))
        route, _ = self.request('/route?' + urllib.parse.urlencode({'slot': slot}))
        return envelope, route, verify_report(envelope, nonce, self.key), latency

    def close(self):
        self.process.terminate()
        self.process.join(timeout=5)
        if self.receiver is not None:
            self.receiver.terminate()
            self.receiver.join(timeout=5)


def software_profile(index, seed):
    instance = str(uuid.uuid5(uuid.NAMESPACE_URL, f'live-5g-{seed}-{index}'))
    return {'nfInstanceId': instance, 'nfType': 'AMF', 'nfStatus': 'REGISTERED',
            'ipv4Addresses': [f'127.10.0.{index + 1}'],
            'nfServices': [{'serviceInstanceId': str(uuid.uuid5(uuid.NAMESPACE_URL, instance + '-service')),
                            'serviceName': 'namf-comm', 'versions': [{'apiVersionInUri': 'v1', 'apiFullVersion': '1.0.0'}],
                            'scheme': 'http', 'nfServiceStatus': 'REGISTERED',
                            'ipEndPoints': [{'ipv4Address': f'127.10.0.{index + 1}', 'port': 8080}]}]}


def profile_path(profile):
    return '/nnrf-nfm/v1/nf-instances/' + profile['nfInstanceId']


def discover_native(client, previous_id=None):
    deadline = time.monotonic() + 25
    while time.monotonic() < deadline:
        response = client.request('GET', '/nnrf-disc/v1/nf-instances?target-nf-type=NSSF&requester-nf-type=AMF')
        body = response.get('body')
        profiles = body.get('nfInstances', []) if isinstance(body, dict) else []
        candidates = [p for p in profiles if p.get('nfType') == 'NSSF' and p.get('nfInstanceId') != previous_id]
        if candidates:
            result = client.request('GET', profile_path(candidates[0]))
            if result['status'] == 200 and isinstance(result['body'], dict):
                return result['body']
        time.sleep(.5)
    raise RuntimeError('Live native Open5GS NSSF profile not available from NRF')


def append_json(handle, value):
    handle.write(json.dumps(value, sort_keys=True) + '\n')
    handle.flush()


def must_succeed(response, operation):
    if not 200 <= response['status'] < 300:
        raise RuntimeError(f"{operation}: HTTP {response['status']}: {response['body']} {response['error']}")


def extract_features(events, observed, expected, envelope, consumer, authenticated,
                     previous_hash, histories, completed, last_heartbeat):
    """Detector inputs depend only on measured events and pre-existing policy.

    No labels, scenario names or scheduled attack times are accepted here.
    """
    if any(e['completed'] > completed for e in events):
        raise ValueError('An observation cannot use a future transaction')
    if not isinstance(observed, dict) or not observed.get('nfInstanceId'):
        raise ValueError('Missing independent NFProfile is not normal evidence')
    count = len(events)
    errors = sum(e['status'] >= 400 or e['status'] == 0 for e in events)
    timestamps = sorted(e['started'] for e in events)
    intervals = [b - a for a, b in zip(timestamps, timestamps[1:])]
    current_hash = digest(observed)
    forbidden = sum('target-nf-type=UDM' in e['path'] or 'target-nf-type=AUSF' in e['path'] for e in events)
    history = list(histories) + [(completed, count, errors)]
    values = {'request_count': count, 'error_rate': errors / max(1, count),
              'update_count': sum(e['method'] in ('PATCH', 'PUT', 'DELETE') for e in events),
              'interarrival_mean': sum(intervals) / max(1, len(intervals)),
              'profile_changes': int(previous_hash is not None and current_hash != previous_hash),
              'forbidden_operation': forbidden,
              'profile_policy_violation': int(current_hash != digest(expected)),
              'route_mismatch': int(consumer.get('served_address', consumer.get('selected_address')) != route_address(observed)),
              'report_mismatch': int(envelope.get('report', {}).get('profile_hash') != current_hash),
              'auth_failure': int(not authenticated),
              'latency_ms': sum(e['latency_ms'] for e in events) / max(1, count),
              'heartbeat_age': max(0., completed - last_heartbeat)}
    for width in (10, 60):
        selected = [row for row in history if 0 <= completed - row[0] < width]
        total = sum(row[1] for row in selected)
        values[f'request_count_{width}s'] = total
        values[f'error_rate_{width}s'] = sum(row[2] for row in selected) / max(1, total)
    return values, current_hash, (completed, count, errors)


def run(args):
    if args.phase != 'test':
        raise ValueError('Freshness experiment is test-only; reuse frozen models')
    loopback_url(args.nrf)
    if args.actors < 6:
        raise ValueError('At least six logical NF slots are required')
    smoke = getattr(args, 'normal_smoke', False)
    if smoke and (not 1 <= args.windows <= 600 or not args.skip_native_restart):
        raise ValueError('Normal acquisition requires 1 to 600 windows and no native restart')
    schedule = [] if smoke else attack_schedule(args.windows, args.seed)
    frozen = None
    if args.phase == 'test':
        if not args.frozen:
            raise ValueError('Test acquisition requires a frozen model directory')
        # Validate before any live registration or witness process is created.
        frozen = load_frozen(args.frozen)
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / 'features.csv').exists():
        raise FileExistsError('Refusing to overwrite a previously acquired run')
    rng = random.Random(args.seed)
    run_id = f'{args.phase}-{args.seed}-nf{args.actors}'
    client = H2Client(args.nrf)
    profiles = {'slot-0': discover_native(client)}
    for index in range(1, args.actors):
        p = software_profile(index, args.seed)
        must_succeed(client.request('PUT', profile_path(p), p), 'Register software NF')
        read = client.request('GET', profile_path(p))
        must_succeed(read, 'Read back software NF')
        profiles[f'slot-{index}'] = read['body']
    initial_native = copy.deepcopy(profiles['slot-0'])
    control = Control(args.out, profiles['slot-4']['nfInstanceId'])
    acquisition = Acquisition(control, args.control_acquisition)
    def bind_control_profile():
        p = copy.deepcopy(profiles['slot-4'])
        p['ipv4Addresses'] = ['127.0.0.1']
        for service in p.get('nfServices', []):
            service['ipEndPoints'] = [{'ipv4Address': '127.0.0.1', 'port': control.ready['port']}]
        must_succeed(client.request('PUT', profile_path(p), p), 'Bind research HTTP endpoint')
        profiles['slot-4'] = p
    bind_control_profile()
    witness = Witness(profiles)
    integrated = (args.out/'integrated_predictions.jsonl').open('w')
    resilient = (args.out/'resilient_predictions.jsonl').open('w')
    refined_file = (args.out/'refined_predictions.jsonl').open('w')
    refinement = None
    refined_engine = None
    grants_file = (args.out/'load_grants.jsonl').open('w')
    operator_key = secrets.token_bytes(32)
    (args.out/'lab_operator_key.hex').write_text(operator_key.hex())
    active_grant = None
    engine = None
    extension_model = None
    if frozen is not None:
        extension_model = json.loads((args.frozen/'extensions.json').read_text())
        engine = Engine(extension_model,control.key,operator_key)
        refinement = json.loads((args.frozen/'refinement.json').read_text())
        refined_engine = RefinedEngine(refinement,control.key,operator_key,json.loads((args.frozen/'accumulation.json').read_text()))
    started, cpu_started = time.time(), time.process_time()
    manifest = {'run_id': run_id, 'phase': args.phase, 'seed': args.seed,
                'windows': args.windows, 'actors': args.actors, 'window_seconds': args.window_seconds,
                'started_at': started, 'native_slot': 'slot-0', 'native_type': 'NSSF',
                'software_actor_slots': list(profiles)[1:], 'witness_pid': witness.pid,
                'route_receiver_pid': witness.receiver_pid,
                'collector_pid': os.getpid(), 'scope': 'real NRF and native NSSF plus software NF clients',
                'consumer_fault': 'separate HTTP consumer route cache, not native AUSF exploit',
                'schedule': schedule if args.phase == 'test' else [],
                'feature_source': 'completed HTTP2 actor transactions, separate NRF reads and signed HTTP witness responses',
                'pcap_scope': 'not collected in this series; completed HTTP transactions retained as JSONL',
                'normal_load_profile': {'usual_discovery_requests': [1, 6],
                    'regimes': ['reference','higher_load','cyclic','bursty_recovery'],
                    'definition': 'experiments/robustness_5g/design.py; each regime lasts 900 windows',
                    'applies_to': ['test']},
                'route_observation': 'slot-2 forwards HTTP to a separate receiver process; other slots read cache state',
                'frozen': str(args.frozen) if args.frozen else None}
    manifest['control_acquisition'] = args.control_acquisition
    if smoke:
        manifest['normal_smoke'] = True
    manifest['load_experiment'] = getattr(args, 'load_experiment', False)
    manifest['unchanged_freshness_limit_seconds'] = .25
    manifest['accumulation'] = json.loads((args.frozen/'accumulation.json').read_text())
    manifest['refinement'] = refinement
    if refinement is not None:
        manifest['refinement_sha256'] = hashlib.sha256((args.frozen/'refinement.json').read_bytes()).hexdigest()
    manifest['extensions'] = extension_model
    if frozen is not None:
        manifest['extensions_sha256'] = hashlib.sha256((args.frozen/'extensions.json').read_bytes()).hexdigest()
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    observer_factory = getattr(args, 'distributed_observer_factory', None)
    distributed_observer = None
    history = {slot: collections.deque(maxlen=600) for slot in profiles}
    previous = {slot: None for slot in profiles}
    last_heartbeat = {slot: started for slot in profiles}
    onsets, active = {}, None
    raw = (args.out / 'events.jsonl').open('w', encoding='utf-8')
    interventions = (args.out / 'interventions.jsonl').open('w', encoding='utf-8')
    feature_file = (args.out / 'features.csv').open('w', newline='', encoding='utf-8')
    label_file = (args.out / 'labels.csv').open('w', newline='', encoding='utf-8')
    features_writer = csv.DictWriter(feature_file, fieldnames=['run_id', 'window_id', 'nf_id', 'timestamp'] + FEATURES)
    labels_writer = csv.DictWriter(label_file, fieldnames=['run_id', 'window_id', 'nf_id', 'attack', 'scenario', 'onset', 'episode_id', 'load_regime', 'variant'])
    features_writer.writeheader()
    labels_writer.writeheader()
    observer_bytes = observer_requests = actor_bytes = actor_requests = 0
    online_frames = []
    try:
        if observer_factory is not None:
            distributed_observer = observer_factory(dict(run_id=run_id, operator_key=operator_key.hex(),
                control_key=control.key.hex(), extensions=extension_model))
        for window in range(args.windows):
            window_started = time.monotonic()
            if window % 300 == 40:
                issued_at = time.time()
                grant = dict(run_id=run_id,nf_id='slot-3',max_requests=48,
                             **{'from':issued_at,'until':issued_at+100},grant_id=secrets.token_hex(16))
                active_grant = dict(report=grant,signature=sign_report(grant,operator_key))
            append_json(grants_file,dict(window_id=window,envelope=active_grant))
            # Restart a native NF under a recorded authorized lifecycle operation.
            if window % 900 == 30 and not args.skip_native_restart:
                old_id = profiles['slot-0']['nfInstanceId']
                begin = time.time()
                subprocess.run(['sudo', 'systemctl', 'restart', 'open5gs-nssfd'], check=True, timeout=30)
                profiles['slot-0'] = discover_native(client, old_id)
                initial_native = copy.deepcopy(profiles['slot-0'])
                witness.update('slot-0', profile=profiles['slot-0'], route=route_address(profiles['slot-0']), version=2)
                append_json(interventions, {'kind': 'N1_native_restart', 'started': begin, 'completed': time.time(),
                                            'old_uuid': old_id, 'new_uuid': profiles['slot-0']['nfInstanceId']})
            # Authorized address update, status transition and re-registration.
            if window in (20, 40, 41, 50) or window % 150 == 100:
                slot = f'slot-{args.actors - 1}'
                p = copy.deepcopy(profiles[slot])
                if window == 20:
                    p['ipv4Addresses'] = ['127.11.0.200']
                    for service in p.get('nfServices', []):
                        service['ipEndPoints'] = [{'ipv4Address': '127.11.0.200', 'port': 8080}]
                elif window in (40, 41):
                    p['nfStatus'] = 'SUSPENDED' if window == 40 else 'REGISTERED'
                else:
                    must_succeed(client.request('DELETE', profile_path(p)), 'Authorized scale down')
                    p['nfInstanceId'] = str(uuid.uuid5(uuid.NAMESPACE_URL, p['nfInstanceId'] + '-scaled'))
                must_succeed(client.request('PUT', profile_path(p), p), 'Authorized lifecycle update')
                profiles[slot] = p
                witness.update(slot, profile=p, route=route_address(p), version=window)
                append_json(interventions, {'kind': 'N1_authorized_lifecycle', 'window': window, 'slot': slot, 'completed': time.time()})
            current = next((entry for entry in schedule if entry[0] <= window < entry[1]), None) if args.phase == 'test' else None
            if active is not None and current != active:
                slot = f'slot-{active[3]}'
                if active[2] == 'S1':
                    must_succeed(client.request('PUT', profile_path(profiles[slot]), profiles[slot]), 'Restore native profile')
                if active[2] == 'S3':
                    witness.update(slot, route=route_address(profiles[slot]))
                append_json(interventions, {'kind': 'recovery', 'scenario': active[2], 'completed': time.time()})
            if current is not None and current != active:
                slot = f'slot-{current[3]}'
                begin = time.time()
                if current[2] == 'S1':
                    mutated = copy.deepcopy(profiles[slot])
                    mutated['ipv4Addresses'] = ['127.99.0.1']
                    for service in mutated.get('nfServices', []):
                        service['ipEndPoints'] = [{'ipv4Address': '127.99.0.1', 'port': 7777}]
                    change = client.request('PUT', profile_path(mutated), mutated)
                    must_succeed(change, 'Native NSSF controlled profile alteration')
                    readback = client.request('GET', profile_path(mutated))
                    must_succeed(readback, 'Native alteration readback')
                    if digest(readback['body']) == digest(profiles[slot]):
                        raise RuntimeError('NRF did not apply controlled alteration')
                    append_json(raw, {'source': 'intervention', 'slot': slot, **change})
                    append_json(raw, {'source': 'intervention_readback', 'slot': slot, **readback})
                elif current[2] == 'S3':
                    witness.update(slot, route='127.99.0.3')
                onsets[current[2]] = begin
                append_json(interventions, {'kind': 'attack_start', 'scenario': current[2], 'slot': slot,
                                            'members': [f'slot-{x}' for x in s5_members(current)] if current[2] == 'S5' else [slot],
                                            'onset': begin, 'applied': time.time(),
                                            'episode_id': f'{run_id}-{current[2]}-{current[0]}'})
            active = current
            old_port = control.ready['port']
            control.prepare(window, active[2] if active is not None and active[3] == 4 else 'N1')
            if old_port != control.ready['port']:
                bind_control_profile()
                witness.update('slot-4', profile=profiles['slot-4'], route=route_address(profiles['slot-4']))
                append_json(interventions, {'kind': 'authorized_research_nf_restart', 'window': window,
                                           'completed': time.time(), 'pid': control.ready['pid']})
            control_observation = None
            rows = []
            for index, (slot, expected) in enumerate(profiles.items()):
                observed_events = []
                is_target = active is not None and (index == active[3] or
                    (active[2] == 'S5' and index in s5_members(active)))
                scenario = active[2] if is_target else 'N1'
                # The policy allows NSSF discovery by these registered research clients.
                count = normal_count(rng,index,window)
                if getattr(args, 'load_experiment', False):
                    from experiments.distributed_5g.load_trial import workload
                    count, episode, pattern = workload(index, window, args.seed)
                    with (args.out/'workload_events.jsonl').open('a') as schedule_file:
                        append_json(schedule_file, dict(run_id=run_id, window_id=window, nf_id=slot,
                            episode_id=episode, scenario=pattern, planned_requests=count, started=time.time()))
                if scenario == 'S4':
                    count = 24
                if scenario == 'F':
                    count = 24  # Hidden state change during an overlapping load burst.
                if scenario == 'G':
                    count = growth_count(active,window)
                if scenario == 'S5':
                    count = s5_count(active, window, regime(window), index)
                benign_error = count >= 12 and rng.random() < .15
                for number in range(count):
                    target_type = 'UDM' if scenario == 'S2' else 'NSSF'
                    path = f'/nnrf-disc/v1/nf-instances?target-nf-type={target_type}&requester-nf-type=AMF'
                    if scenario == 'S4' and number % 3 == 0:
                        path = '/nnrf-nfm/v1/nf-instances/' + str(uuid.UUID(int=0))
                    elif benign_error and number == 0:
                        path = '/nnrf-nfm/v1/nf-instances/' + str(uuid.UUID(int=0))
                    event = client.request('GET', path)
                    observed_events.append(event)
                # Actual software heartbeat. Native slot is sampled via discovery only;
                # its autonomous heartbeat is present in the accompanying PCAP.
                if index:
                    event = client.request('PATCH', profile_path(expected),
                                           [{'op': 'replace', 'path': '/nfStatus', 'value': expected['nfStatus']}])
                    observed_events.append(event)
                    if 200 <= event['status'] < 300:
                        last_heartbeat[slot] = event['completed']
                for event in observed_events:
                    append_json(raw, {'source': 'actor_transaction', 'run_id': run_id, 'window_id': window, 'slot': slot, **event})
                    actor_bytes += event['request_bytes'] + event['response_bytes']
                    actor_requests += 1
                observation = client.request('GET', profile_path(expected))
                must_succeed(observation, 'Independent NRF observation')
                if not isinstance(observation['body'], dict):
                    raise RuntimeError('NRF observation must be a profile object')
                if index == 0:
                    # This is explicitly observation age, never an inferred native
                    # heartbeat. Update it only after a successful independent read.
                    last_heartbeat[slot] = observation['completed']
                observer_requests += 1
                observer_bytes += observation['request_bytes'] + observation['response_bytes']
                append_json(raw, {'source': 'independent_nrf_observation', 'run_id': run_id, 'window_id': window, 'slot': slot, **observation})
                envelope, consumer, authenticated, kie_latency = witness.observe(slot)
                append_json(raw, {'source': 'signed_witness', 'run_id': run_id, 'window_id': window, 'slot': slot,
                                  'envelope': envelope, 'verified': authenticated, 'latency_ms': kie_latency})
                append_json(raw, {'source': 'consumer_route', 'run_id': run_id, 'window_id': window, 'slot': slot, **consumer})
                completed = time.time()
                values, current_hash, hist = extract_features(observed_events, observation['body'], expected,
                            envelope, consumer, authenticated, previous[slot], history[slot], completed, last_heartbeat[slot])
                previous[slot] = current_hash
                history[slot].append(hist)
                row = {'run_id': run_id, 'window_id': window, 'nf_id': slot, 'timestamp': completed, **values}
                rows.append(row)
                if slot == 'slot-4':
                    acquisition.at_slot(window, scenario)
                label = {'run_id': run_id, 'window_id': window, 'nf_id': slot,
                    'attack': int(is_target), 'scenario': scenario, 'onset': onsets.get(scenario, ''),
                    'load_regime':regime(window),'variant':active[5] if is_target else 'normal',
                    'episode_id': f'{run_id}-{scenario}-{active[0]}' if is_target else ''}
                if getattr(args, 'load_experiment', False):
                    label.update(attack=int(bool(episode)), scenario=pattern, episode_id=episode,
                                 variant='live_authorized_load_change', onset='')
                labels_writer.writerow(label)
            features_writer.writerows(rows)
            feature_file.flush()
            label_file.flush()
            if frozen is not None:
                import pandas as pd
                prediction_started = time.time()
                predictions = score_frame(frozen, pd.DataFrame(rows))
                predictions['decision_timestamp'] = time.time()
                predictions['scoring_ms'] = (time.time() - prediction_started) * 1000
                online_frames.append(predictions)
                predictions.to_csv(args.out / 'online_predictions.csv', mode='a', header=window == 0, index=False)
                baseline = predictions[predictions.method == 'hybrid_tail']
                refined_engine.set_window(rows)
                rows_by_slot = {r['nf_id']:r for r in rows}
                for _, pred in baseline.iterrows():
                    base = bool(pred.alarm)
                    if pred.nf_id == 'slot-4':
                        control_observation = acquisition.at_decision(window)
                    now = time.time()
                    states = {m: 'alarm' if base else 'normal' for m in ('H', 'O', 'K')}
                    if pred.nf_id == 'slot-4':
                        c = control_observation
                        states = combine(base, c['delivered'], c['nonce'], c['expected'],
                                         c['instance'], control.key, now)
                    append_json(integrated, {'run_id': run_id, 'window_id': window,
                        'nf_id': pred.nf_id, 'decision_time': now, 'base_alarm': base, 'states': states})
                    enhanced = engine.step(rows_by_slot[pred.nf_id],states['K'],
                        control_observation if pred.nf_id=='slot-4' else None,active_grant,now)
                    append_json(resilient,dict(run_id=run_id,window_id=window,nf_id=pred.nf_id,
                        evaluation_time=now,decision_time=time.time(),**enhanced))
                    refined = refined_engine.step(rows_by_slot[pred.nf_id],enhanced,
                        control_observation if pred.nf_id=='slot-4' else None,active_grant,now)
                    append_json(refined_file,dict(run_id=run_id,window_id=window,nf_id=pred.nf_id,
                        evaluation_time=now,decision_time=time.time(),**refined))
            if distributed_observer is not None:
                distributed_observer(window, rows, control_observation, active_grant, predictions)
            if window % 30 == 0:
                print(f'{run_id}: completed window {window}/{args.windows}', flush=True)
            time.sleep(max(0, args.window_seconds - (time.monotonic() - window_started)))
        manifest['completed_at'] = time.time()
        manifest['complete'] = True
    finally:
        # The original live profile is restored even if acquisition fails.
        try:
            client.request('PUT', profile_path(initial_native), initial_native)
            for slot, profile in profiles.items():
                if slot != 'slot-0':
                    client.request('DELETE', profile_path(profile))
        finally:
            witness.close()
            control.close()
            integrated.close()
            resilient.close()
            refined_file.close()
            grants_file.close()
            for handle in (raw, interventions, feature_file, label_file):
                handle.close()
        manifest['elapsed_seconds'] = time.time() - started
        manifest['collector_cpu_seconds'] = time.process_time() - cpu_started
        manifest['observer_http2_requests'] = observer_requests
        manifest['observer_application_bytes'] = observer_bytes
        manifest['witness_http_requests'] = witness.requests
        manifest['witness_application_bytes'] = witness.bytes
        manifest['actor_http2_requests'] = actor_requests
        manifest['actor_application_bytes'] = actor_bytes
        try:
            import resource
            manifest['collector_peak_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        except ImportError:
            pass
        manifest['measurement_limitations'] = ['Application byte counts exclude HTTP headers and transport framing.',
            'New independent configuration channel covers slot-4 only; report its metrics separately.',
            'New control reports travel through trusted local IPC with explicit envelope fault injection.',
            'New research NF and observer CPU/RSS are excluded from collector consumption.',
            'Collector CPU/RSS exclude native Core and witness process.',
            'Slot-2 consumer HTTP forwarding uses a separate research receiver process; other route observations are cache reads.',
            'Receiver process CPU and forwarding transport headers are not included in collector overhead totals.',
            'Request latency is observed end-to-end; causal added latency requires a paired monitor-off run.',
            'Native slot heartbeat_age is observation age; native autonomous heartbeat is only recorded in PCAP.',
            'The current signed witness reports operator policy; report_mismatch is redundant with profile_policy_violation in this protocol.',
            'N1 transitions are operator-authorized and policy is updated before subsequent observation.',
            'S2/S4 originate from registered software NF clients; no UE/RAN/PDU traffic is generated.',
            'S3 is a separate research consumer cache fault; not a native AUSF vulnerability reproduction.']
        (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.out.iterdir() if p.is_file()}
    (args.out / 'sha256.json').write_text(json.dumps(hashes, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['train', 'cal', 'test'], required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--windows', type=int, default=180)
    parser.add_argument('--actors', type=int, default=8)
    parser.add_argument('--window-seconds', type=float, default=1.)
    parser.add_argument('--nrf', default='http://127.0.0.10:7777')
    parser.add_argument('--frozen', type=Path)
    parser.add_argument('--skip-native-restart', action='store_true')
    parser.add_argument('--control-acquisition', choices=['early','late'], required=True)
    args = parser.parse_args()
    if args.window_seconds <= 0:
        parser.error('--window-seconds must be positive')
    run(args)


if __name__ == '__main__':
    main()
