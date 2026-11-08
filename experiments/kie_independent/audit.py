"""Recompute decisions using raw evidence, without importing the collector."""
import hashlib
import hmac
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
plan = json.loads((root/'plan.json').read_text())
key = bytes.fromhex((root/'lab_key.hex').read_text())
expected_hashes = json.loads((root/'sha256.json').read_text())
assert all(hashlib.sha256((root/name).read_bytes()).hexdigest() == value
           for name, value in expected_hashes.items())
rows = [json.loads(line) for line in (root/'events.jsonl').read_text().splitlines()]
assert [r['condition'] for r in rows] == plan['conditions']
assert len({r['nonce'] for r in rows}) == len(rows)
assert plan['created'] < min(r['onset'] for r in rows)
for r in rows:
    d = r['delivered']['report']
    signature = hmac.new(key, json.dumps(d, sort_keys=True, separators=(',', ':')).encode(), hashlib.sha256).hexdigest()
    bad = (d['nonce'] != r['nonce'] or d['instance'] != r['instance']
           or abs(r['decision_time']-d['timestamp']) > plan['max_age']
           or d['config_hash'] != r['expected'])
    b = r['nrf_changed']
    assert r['decisions'] == {'B': b, 'O': b or bad,
             'K': b or bad or not hmac.compare_digest(signature, r['delivered']['signature'])}
    assert r['original']['report']['pid'] != r['ground_truth']['pid']
    assert r['original']['report']['config_hash'] == r['ground_truth']['config_hash']
    assert r['ground_truth']['answer'] == ('DENY' if r['attack'] else 'ALLOW')
    if plan['live_nrf']:
        assert r['nrf']['status'] == 200
        assert r['nrf']['body']['nfInstanceId'] == plan['instance']
summary = json.loads((root/'summary.json').read_text())
for condition, counts in summary['conditions'].items():
    group = [r for r in rows if r['condition'] == condition]
    assert counts == dict(n=len(group), **{m: sum(r['decisions'][m] for r in group) for m in ['B','O','K']})
print(json.dumps({'audit_pass': True, 'rows': len(rows), 'seed': plan['seed'], 'live_nrf': plan['live_nrf']}))
