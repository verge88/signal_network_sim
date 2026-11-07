"""Contract tests; fixtures are not experimental live Open5GS evidence."""
import copy
import json
import socket
import threading
import time

import pytest

from experiments.live_5g.collect import (FEATURES, Witness, attack_schedule, canonical_profile, digest,
                      extract_features, sign_report, software_profile, verify_report)
from experiments.live_5g.transport import H2Client, loopback_url


@pytest.mark.parametrize('url', ['https://127.0.0.1', 'http://example.com',
                                 'http://192.0.2.1', 'http://user:pass@127.0.0.1'])
def test_transport_rejects_non_lab_destinations(url):
    with pytest.raises(ValueError):
        loopback_url(url)


def test_profile_hash_is_order_invariant_and_detects_semantic_changes():
    original = software_profile(1, 42)
    original['ipv4Addresses'] += ['127.0.0.42']
    changed = copy.deepcopy(original)
    changed['ipv4Addresses'].reverse()
    changed['heartBeatTimer'] = 999
    assert digest(changed) == digest(original)
    changed['nfServices'][0]['ipEndPoints'][0]['port'] = 9090
    assert digest(changed) != digest(original)


def test_signed_report_rejects_tamper_replay_and_invalid_shape():
    key = b'unit-test-key'
    report = {'nonce': 'fresh', 'timestamp': time.time(), 'profile_hash': 'abc'}
    envelope = {'report': report, 'signature': sign_report(report, key)}
    assert verify_report(envelope, 'fresh', key)
    assert not verify_report(envelope, 'replayed', key)
    report['profile_hash'] = 'changed'
    assert not verify_report(envelope, 'fresh', key)
    assert not verify_report({'report': None}, 'fresh', key)
    assert not verify_report({'report': {'timestamp': 'invalid', 'nonce': 'fresh'}}, 'fresh', key)


def test_separate_witness_process_measures_real_route_state():
    profile = software_profile(1, 42)
    witness = Witness({'slot-1': profile})
    try:
        envelope, route, valid, latency = witness.observe('slot-1')
        assert valid and latency >= 0
        assert envelope['report']['process_pid'] == witness.pid
        assert route['selected_address'] == '127.10.0.2'
        witness.update('slot-1', route='127.99.0.3')
        second, route, valid, _ = witness.observe('slot-1')
        assert valid
        assert route['selected_address'] == '127.99.0.3'
        assert second['report']['profile_hash'] == envelope['report']['profile_hash']
    finally:
        witness.close()


def fixture_features():
    p = software_profile(1, 42)
    event = {'status': 200, 'started': 99., 'completed': 99.5,
             'path': '/nnrf-disc/v1/nf-instances?target-nf-type=NSSF',
             'method': 'GET', 'latency_ms': 500.}
    return [event], p, copy.deepcopy(p), {'report': {'profile_hash': digest(p)}}, {'selected_address': '127.10.0.2'}


def test_consumer_measures_actual_receiver_before_and_after_route_fault():
    profile = software_profile(2, 42)
    witness = Witness({'slot-2': profile})
    try:
        _, normal, valid, _ = witness.observe('slot-2')
        assert valid and normal['measurement'] == 'http_forward'
        assert normal['http_status'] == 200
        assert normal['served_address'] == '127.10.0.3'
        assert normal['receiver_pid'] == witness.receiver_pid != witness.pid
        assert normal['response_completed'] >= normal['request_started']
        witness.update('slot-2', route='127.99.0.3')
        _, fault, valid, _ = witness.observe('slot-2')
        assert valid and fault['served_address'] == '127.99.0.3'
        assert fault['receiver_pid'] == witness.receiver_pid
        assert fault['nonce'] != normal['nonce']
        # The feature uses the destination's observed response, even if the
        # consumer's stored/claimed address disagrees with it.
        fault['selected_address'] = '127.10.0.3'
        now = time.time()
        features, _, _ = extract_features([], profile, profile,
            {'report': {'profile_hash': digest(profile)}}, fault, True, None, [], now, now)
        assert features['route_mismatch'] == 1
    finally:
        witness.close()
    assert not witness.receiver.is_alive()


def test_measured_features_have_allowlisted_schema_and_no_label_dependence():
    args = fixture_features()
    first, _, _ = extract_features(*args, True, None, [], 100., 99.)
    args[0][0]['scenario'] = 'fabricated-label'
    args[0][0]['attack'] = True
    second, _, _ = extract_features(*args, True, None, [], 100., 99.)
    assert first == second
    assert set(first) == set(FEATURES)
    assert first['report_mismatch'] == first['profile_policy_violation'] == 0


def test_future_transaction_rejected_and_future_history_not_used():
    args = fixture_features()
    values, _, _ = extract_features(*args, True, None, [(101., 1000, 1000)], 100., 99.)
    assert values['request_count_10s'] == 1
    args[0][0]['completed'] = 101.
    with pytest.raises(ValueError, match='future'):
        extract_features(*args, True, None, [], 100., 99.)


def test_missing_independent_observation_fails_closed():
    args = list(fixture_features())
    args[1] = {}
    with pytest.raises(ValueError, match='Missing independent'):
        extract_features(*args, True, None, [], 100., 99.)


def test_no_scenario_overlap_or_missing_baseline():
    schedule = attack_schedule(240)
    assert [s[2] for s in schedule] == ['S1', 'S2', 'S3', 'S4']
    assert schedule[0][0] > 60
    assert all(a[1] < b[0] for a, b in zip(schedule, schedule[1:]))


def serve_one_http2(listener, truncate=False):
    from h2.config import H2Configuration
    from h2.connection import H2Connection
    from h2.events import StreamEnded
    with listener:
        conn, _ = listener.accept()
        with conn:
            protocol = H2Connection(config=H2Configuration(client_side=False))
            protocol.initiate_connection()
            conn.sendall(protocol.data_to_send())
            while True:
                packet = conn.recv(65536)
                if not packet:
                    return
                events = protocol.receive_data(packet)
                if any(isinstance(e, StreamEnded) for e in events):
                    protocol.send_headers(1, [(':status', '200'), ('content-type', 'application/json')])
                    protocol.send_data(1, b'{"ok":true}', end_stream=not truncate)
                    conn.sendall(protocol.data_to_send())
                    # A graceful FIN permits the client to read queued response
                    # frames. Closing with unread SETTINGS ACK bytes may send a
                    # Windows TCP reset that discards the entire response.
                    conn.shutdown(socket.SHUT_WR)
                    conn.settimeout(2)
                    try:
                        while conn.recv(65536):
                            pass
                    except OSError:
                        pass
                    return
                pending = protocol.data_to_send()
                if pending:
                    conn.sendall(pending)


@pytest.mark.parametrize('truncate', [False, True])
def test_real_h2_prior_knowledge_transport_requires_complete_response(truncate):
    listener = socket.socket()
    listener.bind(('127.0.0.1', 0))
    listener.listen(1)
    port = listener.getsockname()[1]
    worker = threading.Thread(target=serve_one_http2, args=(listener, truncate), daemon=True)
    worker.start()
    response = H2Client(f'http://127.0.0.1:{port}').request('GET', '/probe')
    worker.join(timeout=3)
    assert response['stream_id'] == 1 and response['connection_id']
    assert response['status'] == (0 if truncate else 200), response
    if truncate:
        assert response['error']
    else:
        assert response['body'] == {'ok': True}
