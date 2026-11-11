"""One independent process per NF, zone leader, or network coordinator."""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

from experiments.distributed_5g.primitives import advance, peer_residuals, combine, permit_valid, ProtocolEngine
from urllib.error import HTTPError
from experiments.distributed_5g.protocol import canonical, post, sign, verify

CONFIRMATION_FRACTION = 0.5
RECOVERY_WINDOWS = 3


class Node:
    def __init__(self, config):
        self.config = config
        self.path = Path(config['state_file'])
        if self.path.exists():
            self.state = json.loads(self.path.read_text())
            if self.state['epoch'] != config['epoch']:
                raise ValueError('State belongs to another deployment epoch')
        else:
            self.state = dict(epoch=config['epoch'], received={}, windows={}, outbox=[],
                              sequence=0, accumulators={}, high_water=-1, witness={})

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix('.tmp')
        with temporary.open('wb') as stream:
            stream.write(canonical(self.state))
            stream.flush()
            os.fsync(stream.fileno())
        # Windows scanners/readers can briefly deny replacement of an existing
        # state file. Retain atomic replacement and acknowledge only after it
        # succeeds; permanent failures must still fail closed.
        for attempt in range(21):
            try:
                os.replace(temporary, self.path)
                break
            except PermissionError as error:
                if getattr(error, 'winerror', None) not in (5, 32, 33) or attempt == 20:
                    raise
                time.sleep(.05)

    def emit(self, body):
        self.state['sequence'] += 1
        self.state['outbox'].append(dict(sequence=self.state['sequence'], body=body))

    def local(self, body):
        row = body['row']
        if row['nf_id'] != self.config['id'] or row['run_id'] != self.config['source_run']:
            raise ValueError('Observation assigned to another NF or acquisition')
        names = ['request_count', 'error_rate', 'update_count', 'profile_policy_violation',
                 'heartbeat_age', 'forbidden_operation', 'route_mismatch', 'report_mismatch', 'auth_failure']
        if not all(math.isfinite(float(row[name])) and float(row[name]) >= 0 for name in names):
            raise ValueError('Invalid observation')
        now = body['evaluation_time'] if self.config.get('replay_mode') else time.time()
        values = dict(row, error_count=round(row['request_count'] * row['error_rate']))
        tail = any(values[name] > threshold for name, threshold in self.config['model']['feature_thresholds'].items())
        semantic = any(row[name] > 0 for name in names[3:] if name != 'heartbeat_age') or row['heartbeat_age'] > 30
        security = tail or semantic
        status = 'normal'
        availability = False
        control = body.get('control')
        if self.config.get('control_key'):
            if control is None:
                raise ValueError('Control observation required for monitored NF')
            key = bytes.fromhex(self.config['control_key'])
            status = combine(security, control['delivered'], control['nonce'], control['expected'], control['instance'], key, now)['K']
            security = status == 'alarm'
            protocol = ProtocolEngine(self.config['extensions'], key, bytes.fromhex(self.config['operator_key']))
            saved = self.state['witness']
            protocol.issued = set(saved.get('issued', []))
            protocol.seen = {tuple(token) for token in saved.get('seen', [])}
            protocol.high = saved.get('high', {})
            protocol.last_fresh = saved.get('last_fresh')
            checked = protocol.step(row, status, control, body.get('grant'), now)
            security = security or checked['signals']['replay']
            availability = checked['signals']['availability']
            self.state['witness'] = dict(issued=sorted(protocol.issued), seen=sorted(protocol.seen),
                                        high=protocol.high, last_fresh=protocol.last_fresh)
        allowed = permit_valid(body.get('grant'), bytes.fromhex(self.config['operator_key']), row['run_id'], row['nf_id'], row['timestamp'], row['request_count'])
        report = dict(kind='report', window=body['window'], nf_id=row['nf_id'], request_count=row['request_count'],
                      allowed=allowed, local_security=bool(security), availability=availability, unknown=status=='unknown')
        if self.config.get('include_if_context'):
            if type(body.get('if_context_alarm')) is not bool:
                raise ValueError('IF_CONTEXT decision is required for the hybrid detector')
            report['if_context_alarm'] = body['if_context_alarm']
        return report

    def receive(self, packet):
        before = copy.deepcopy(self.state)
        try:
            return self._receive(packet)
        except Exception:
            self.state = before
            raise

    def _receive(self, packet):
        verify(packet, self.config)
        sender = packet['sender']
        digest = hashlib.sha256(canonical(packet['body'])).hexdigest()
        previous = self.state['received'].get(sender)
        if previous and packet['sequence'] <= previous['sequence']:
            if packet['sequence'] == previous['sequence'] and digest == previous['digest']:
                return {'accepted': True, 'duplicate': True}
            raise ValueError('Replayed or conflicting sequence')
        body = packet['body']
        window = body['window']
        if type(window) is not int or window < 0:
            raise ValueError('Invalid window')
        role = self.config['role']
        if len(self.state['outbox']) >= 4:
            raise ValueError('Parent unavailable: bounded outbox is full')
        if role == 'leaf':
            if sender != 'source' or body['kind'] != 'observe' or window <= self.state['high_water']:
                raise ValueError('Unexpected leaf observation')
            report = self.local(body)
            self.emit(report)
            self.state['high_water'] = window
        else:
            if window <= self.state['high_water']:
                raise ValueError('Closed window')
            # Exactly one active window makes ordering and bounded buffering explicit.
            active = self.state['windows']
            if active and str(window) not in active:
                raise ValueError('Previous window must close before advancing')
            if body['kind'] == 'begin' and sender == 'source':
                active.setdefault(str(window), dict(deadline=time.time()+self.config['deadline_seconds'], reports={}))
            elif body['kind'] == 'report' and sender in self.config['members']:
                if str(window) not in active:
                    raise ValueError('Window has not begun')
                reports = active[str(window)]['reports']
                if sender in reports:
                    raise ValueError('Duplicate member report')
                if role == 'zone' and body['nf_id'] != sender:
                    raise ValueError('Member identity mismatch')
                if role == 'zone' and self.config.get('include_if_context') and type(body.get('if_context_alarm')) is not bool:
                    raise ValueError('IF_CONTEXT decision is missing from the member report')
                if role == 'root' and body['zone_id'] != sender:
                    raise ValueError('Zone identity mismatch')
                if role == 'root' and set(body['decisions']) != set(self.config['zone_members'][sender]):
                    raise ValueError('Zone membership mismatch')
                reports[sender] = body
            else:
                raise ValueError('Unexpected message for role')
        self.state['received'][sender] = dict(sequence=packet['sequence'], digest=digest)
        self.save()
        return {'accepted': True, 'duplicate': False}

    def close_window(self, window, entry):
        missing = sorted(set(self.config['members']) - set(entry['reports']))
        if self.config['role'] == 'zone':
            reports = list(entry['reports'].values())
            complete = not missing
            residuals = peer_residuals(reports) if complete else {}
            decisions = {}
            for report in reports:
                identity = report['nf_id']
                value = self.state['accumulators'].get(identity, 0.)
                if complete:
                    value = advance(value, residuals[identity], report['allowed'])
                else:
                    value = 0.  # A changed peer set cannot silently use the calibrated baseline.
                self.state['accumulators'][identity] = value
                security = report['local_security'] or (complete and value > self.config['model']['threshold'])
                decisions[identity] = dict(security=bool(security), availability=report['availability'],
                    unknown=report['unknown'] or not complete, accumulated=value)
                if self.config.get('include_if_context'):
                    context = report['if_context_alarm']
                    confirmed = complete and context and value > CONFIRMATION_FRACTION * self.config['model']['threshold']
                    use_context = confirmed if self.config.get('confirm_if_context') else context
                    decisions[identity].update(accum_security=bool(security),
                        if_context_alarm=context, if_context_confirmed=confirmed,
                        security=bool(security or use_context))
                if self.config.get('recover_normal'):
                    recovery = self.state.setdefault('recovery', {})
                    prior = recovery.get(identity, dict(value=0., quiet=0))
                    quiet = (complete and residuals[identity] <= 0 and not report['local_security']
                             and not report['availability'] and not report['unknown'])
                    previous_quiet = prior['quiet'] if prior.get('window') == window - 1 else 0
                    streak = min(RECOVERY_WINDOWS, previous_quiet + 1) if quiet else 0
                    recovered = advance(prior['value'], residuals[identity], report['allowed']) if complete else 0.
                    reset = streak == RECOVERY_WINDOWS
                    if reset:
                        recovered = 0.
                    recovery[identity] = dict(value=recovered, quiet=streak, window=window)
                    threshold = self.config['model']['threshold']
                    context = report.get('if_context_alarm', False)
                    confirmed = context and (not self.config.get('confirm_if_context') or
                                              recovered > CONFIRMATION_FRACTION * threshold)
                    decisions[identity].update(original_security=decisions[identity]['security'],
                        recovery_accumulated=recovered, recovery_quiet_windows=streak, recovery_reset=reset,
                        security=bool(report['local_security'] or (complete and (recovered > threshold or confirmed))))
            for identity in missing:
                self.state['accumulators'].pop(identity, None)
                self.state.get('recovery', {}).pop(identity, None)
                decisions[identity] = dict(security=False, availability=True, unknown=True, accumulated=None)
            self.emit(dict(kind='report', window=window, zone_id=self.config['id'], decisions=decisions,
                           missing=missing, complete=complete))
        else:
            decisions = {}
            for zone, report in entry['reports'].items():
                if set(report['decisions']) != set(self.config['zone_members'][zone]):
                    raise ValueError('Zone report does not match topology')
                decisions.update(report['decisions'])
            for zone in missing:
                for identity in self.config['zone_members'][zone]:
                    decisions[identity] = dict(security=False, availability=True, unknown=True, accumulated=None)
            result = dict(window=window, epoch=self.config['epoch'], decisions=decisions, missing_zones=missing,
                          complete=not missing and all(r['complete'] for r in entry['reports'].values()))
            self.state['latest'] = result
        self.state['high_water'] = window
        self.state['windows'].pop(str(window))

    def tick(self):
        for window, entry in list(self.state['windows'].items()):
            if len(entry['reports']) == len(self.config['members']) or time.time() >= entry['deadline']:
                self.close_window(int(window), entry)
                self.save()
        if self.state['outbox']:
            item = self.state['outbox'][0]
            parent = self.config['parent']
            packet = sign(self.config['id'], parent['id'], self.config['epoch'], item['sequence'], item['body'], parent['key'])
            try:
                post(parent['url'], packet, timeout=.5)
            except HTTPError as error:
                if error.code != 400:
                    return
                self.state['rejected_reports'] = self.state.get('rejected_reports', 0) + 1
            except (OSError, ValueError):
                return
            self.state['outbox'].pop(0)
            self.save()


def serve(config, pid_file=None):
    node = Node(config)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, code, value):
            data = canonical(value)
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            try:
                length = int(self.headers.get('Content-Length', 0))
                if self.path != '/message' or not 0 < length <= 131072:
                    raise ValueError('Invalid request')
                result = node.receive(json.loads(self.rfile.read(length)))
                self.reply(200, result)
            except (KeyError, TypeError, ValueError) as error:
                self.reply(400, {'error': str(error)})

        def do_GET(self):
            if self.path == '/health':
                namespace = Path('/proc/self/ns/net')
                self.reply(200, dict(id=config['id'], role=config['role'], pid=os.getpid(),
                    pending=len(node.state['outbox']), netns=namespace.stat().st_ino if namespace.exists() else None))
            elif self.path == '/latest' and config['role'] == 'root':
                self.reply(200, node.state.get('latest', {}))
            else:
                self.reply(404, {})

    server = HTTPServer((config['host'], config['port']), Handler)
    if pid_file is not None:
        pid_file.write_text(str(os.getpid()))
    server.timeout = .05
    while True:
        server.handle_request()
        node.tick()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--pid-file', type=Path)
    args = parser.parse_args()
    serve(json.loads(args.config.read_text()), args.pid_file)
