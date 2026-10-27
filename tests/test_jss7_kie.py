import copy
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from experiments.jss7_kie.evidence import detect, detect_interval, evaluate, sign, transaction_ids, verify
from experiments.jss7_kie.run_lab import observe, poll_report, stop, wait_capture
from experiments.jss7_kie.periodic import (
    audit_final, check_report, evaluate_windows, observation_bounds, first_frames, wait_marker)


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.key = b'k' * 32
        self.payload = dict(schema=1, node='jss7-server', run_id='run-1',
                            nonce='n' * 32, received_dialogs=20)

    def test_valid_mac_does_not_make_concealment_truthful(self):
        payload = verify(sign(self.payload, self.key), self.key, 'run-1', 'n' * 32)
        self.assertEqual(detect(30, payload)['residual'], 10)
        self.assertTrue(detect(30, payload)['alarm'])

    def test_live_interval_accepts_inflight_requests(self):
        # An exact equality check against either endpoint would be a false alarm.
        self.assertFalse(detect_interval(19, 21, self.payload)['alarm'])
        for reported in (19, 21):
            self.payload['received_dialogs'] = reported
            self.assertEqual(detect_interval(19, 21, self.payload)['status'], 'consistent')

    def test_live_interval_detects_under_and_over_reporting(self):
        self.assertEqual(detect_interval(28, 30, self.payload)['residual'], 8)
        self.assertEqual(detect_interval(10, 12, self.payload)['residual'], -8)
        for lower, upper in [(2, 1), (-1, 1), (True, 1)]:
            with self.subTest(lower=lower), self.assertRaises(ValueError):
                detect_interval(lower, upper, self.payload)

    def test_live_report_binds_nonce_and_sequence(self):
        self.payload['sequence'] = 3
        report = sign(self.payload, self.key)
        self.assertEqual(check_report(report, self.key, 'run-1', 'n'*32, 3, 19, 21)['status'], 'consistent')
        for nonce, sequence in [('fresh', 3), ('n'*32, 4)]:
            self.assertEqual(check_report(report, self.key, 'run-1', nonce, sequence, 19, 21)['status'], 'invalid_report')

    def test_capture_order_survives_clock_jumps_and_retransmissions(self):
        self.assertEqual(first_frames('1\t10.2\t01,02\n2\t8.1\t01\n'),
                         {'01': 1, '02': 1})

    def test_live_bounds_account_for_open_dialogs_and_boundary_times(self):
        trace = {'begin': {'a': 10, 'b': 20, 'c': 25}, 'end': {'a': 19, 'b': 24}}
        self.assertEqual(observation_bounds(trace, 20, 26), (1, 3))
        # End exactly on poll boundary is conservatively excluded.
        self.assertEqual(observation_bounds(trace, 19, 21), (0, 2))
        with self.assertRaises(ValueError):
            observation_bounds(trace, 26, 25)

    def test_stp_live_bounds_do_not_sum_duplicate_legs(self):
        trace = {'client_begin': {'a': 1, 'b': 2}, 'server_begin': {'a': 2},
                 'server_end': {'a': 3}, 'client_end': {'a': 4}}
        self.assertEqual(observation_bounds(trace, 5, 6), (1, 2))

    def test_missing_report_is_not_counted_as_a_correct_benign_decision(self):
        result = evaluate_windows([{'status': 'unavailable'}, {'status': 'mismatch'}], ['honest', 'conceal'])
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['metrics']['tn'], 0)
        self.assertEqual(result['metrics']['report_faults_detected'], 0)

    def test_recovery_is_evaluated_after_report_faults(self):
        result = evaluate_windows([{'status': 'invalid_report'}, {'status': 'consistent'}], ['replay', 'honest'])
        self.assertEqual(result['status'], 'passed')
        result = evaluate_windows([{'status': 'invalid_report'}, {'status': 'invalid_report'}], ['replay', 'honest'])
        self.assertEqual(result['status'], 'failed')

    def test_marker_must_be_seen_before_trusting_live_capture(self):
        capture = Mock()
        capture.poll.return_value = None
        with patch('experiments.jss7_kie.periodic.command', side_effect=['', '8\t61:62\n']), \
             patch('experiments.jss7_kie.periodic.time.sleep'):
            wait_marker(Path('capture'), capture, b'ab')
        capture.poll.return_value = 1
        with self.assertRaises(RuntimeError):
            wait_marker(Path('capture'), capture, b'ab')

    def test_final_audit_rejects_incomplete_live_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / 'periodic').mkdir()
            with patch('experiments.jss7_kie.periodic.extract_trace', return_value={
                    'begin': {'a': 1, 'b': 2}, 'end': {'a': 3}}):
                with self.assertRaises(ValueError):
                    audit_final(directory, 'direct', [{'request_frame': 4, 'response_frame': 5,
                                                       'observed_lower': 0, 'observed_upper': 1}])

    def test_honest_surge_is_not_an_alarm(self):
        self.payload['received_dialogs'] = 40
        self.assertFalse(detect(40, self.payload)['alarm'])

    def test_tamper_foreign_key_nonce_and_run_rejected(self):
        report = sign(self.payload, self.key)
        tampered = copy.deepcopy(report)
        tampered['payload']['received_dialogs'] = 30
        for r, key, run, nonce in [(tampered, self.key, 'run-1', 'n'*32),
                                    (report, b'other', 'run-1', 'n'*32),
                                    (report, self.key, 'run-2', 'n'*32),
                                    (report, self.key, 'run-1', 'old')]:
            with self.subTest(run=run, nonce=nonce), self.assertRaises(ValueError):
                verify(r, key, run, nonce)

    def test_empty_capture_rejected(self):
        with self.assertRaises(ValueError):
            detect(0, self.payload)

    def test_invalid_counter_rejected_even_with_valid_mac(self):
        for count in [-1, True, '20', 20.5]:
            self.payload['received_dialogs'] = count
            with self.subTest(count=count), self.assertRaises(ValueError):
                verify(sign(self.payload, self.key), self.key, 'run-1', 'n'*32)

    def test_retransmission_dedup_and_bundled_dialogs(self):
        self.assertEqual(transaction_ids('00:01\n0001,00:02\n'), {'0001', '0002'})
        with self.assertRaises(ValueError):
            transaction_ids('invalid')

    def test_metrics_use_actual_decisions(self):
        result = evaluate([dict(label=0, alarm=True), dict(label=0, alarm=False),
                           dict(label=1, alarm=True), dict(label=1, alarm=False)])
        self.assertEqual(result['episode_fpr'], .5)
        self.assertEqual(result['episode_recall'], .5)
        with self.assertRaises(ValueError):
            evaluate([dict(label=0, alarm=False)])

    def test_observer_rejects_empty_and_incomplete_dialogs(self):
        with tempfile.TemporaryDirectory() as tmp:
            for begin, end in [('', ''), ('01\n02\n', '01\n'), ('01\n', '02\n')]:
                with self.subTest(begin=begin, end=end), patch(
                        'experiments.jss7_kie.run_lab.command', side_effect=[begin, end]):
                    with self.assertRaises(ValueError):
                        observe(Path(tmp)/'capture.pcap', Path(tmp))

    def test_observer_counts_completed_dialogs_not_frames(self):
        with tempfile.TemporaryDirectory() as tmp, patch(
                'experiments.jss7_kie.run_lab.command', side_effect=['01,02\n01\n', '02\n01\n']):
            self.assertEqual(observe(Path(tmp)/'capture.pcap', Path(tmp)), 2)

    def test_stp_observer_requires_both_legs_and_both_directions(self):
        # A complete direct exchange cannot stand in for actual STP transit.
        with tempfile.TemporaryDirectory() as tmp:
            for missing in range(4):
                values = ['01\n02\n'] * 4
                values[missing] = '01\n'
                with self.subTest(missing=missing), patch(
                        'experiments.jss7_kie.run_lab.command', side_effect=values):
                    with self.assertRaises(ValueError):
                        observe(Path(tmp)/'capture.pcap', Path(tmp), 'osmostp')

    def test_stp_observer_deduplicates_transit_without_doubling_count(self):
        with tempfile.TemporaryDirectory() as tmp, patch(
                'experiments.jss7_kie.run_lab.command',
                side_effect=['01\n02\n01\n', '02,01\n', '01\n02\n', '02\n01\n']) as extract:
            self.assertEqual(observe(Path(tmp)/'capture.pcap', Path(tmp), 'osmostp'), 2)
            filters = [call.args[0][4] for call in extract.call_args_list]
            self.assertIn('sctp.srcport == 2345 && sctp.dstport == 2905', filters[0])
            self.assertIn('sctp.srcport == 2905 && sctp.dstport == 3434', filters[1])
            self.assertIn('m3ua.protocol_data_opc == 2 && m3ua.protocol_data_dpc == 1', filters[2])

    def test_stp_observer_rejects_wrong_return_dialog_ids(self):
        with tempfile.TemporaryDirectory() as tmp, patch(
                'experiments.jss7_kie.run_lab.command', side_effect=['01\n', '01\n', '02\n', '02\n']):
            with self.assertRaises(ValueError):
                observe(Path(tmp)/'capture.pcap', Path(tmp), 'osmostp')

    def test_capture_drain_retries_partial_evidence(self):
        capture = Mock()
        capture.poll.return_value = None
        with patch('experiments.jss7_kie.run_lab.observe', side_effect=[ValueError('partial'), 20]), \
             patch('experiments.jss7_kie.run_lab.time.sleep'):
            wait_capture(Path('capture'), Path('.'), 20, capture)

    def test_capture_drain_timeout_is_not_success(self):
        capture = Mock()
        capture.poll.return_value = None
        with patch('experiments.jss7_kie.run_lab.observe', return_value=19), \
             patch('experiments.jss7_kie.run_lab.time.monotonic', side_effect=[0, 1, 11]), \
             patch('experiments.jss7_kie.run_lab.time.sleep'), self.assertRaises(TimeoutError):
            wait_capture(Path('capture'), Path('.'), 20, capture)

    def test_reporter_over_http_signs_capped_node_counter(self):
        import socket
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            events, key = directory/'events', directory/'key'
            events.write_text('\n'.join(map(str, range(30))))
            key.write_bytes(self.key)
            proc = subprocess.Popen([sys.executable, '-m', 'experiments.jss7_kie.reporter',
                '--events', str(events), '--key', str(key), '--run-id', 'run-1',
                '--cap', '20', '--port', str(port)], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            try:
                report = poll_report(f'http://127.0.0.1:{port}/kie?nonce=' + 'n'*32)
                payload = verify(report, self.key, 'run-1', 'n'*32)
                self.assertEqual(payload['received_dialogs'], 20)
                self.assertTrue(detect(30, payload)['alarm'])
            finally:
                stop(proc)
                proc.stderr.close()


if __name__ == '__main__':
    unittest.main()
