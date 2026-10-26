import copy
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from experiments.jss7_kie.evidence import detect, evaluate, sign, transaction_ids, verify
from experiments.jss7_kie.run_lab import observe, poll_report, stop, wait_capture


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.key = b'k' * 32
        self.payload = dict(schema=1, node='jss7-server', run_id='run-1',
                            nonce='n' * 32, received_dialogs=20)

    def test_valid_mac_does_not_make_concealment_truthful(self):
        payload = verify(sign(self.payload, self.key), self.key, 'run-1', 'n' * 32)
        self.assertEqual(detect(30, payload)['residual'], 10)
        self.assertTrue(detect(30, payload)['alarm'])

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
