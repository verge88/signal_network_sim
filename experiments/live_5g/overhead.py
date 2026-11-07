"""Paired live NRF benchmark of sequential monitoring cost.

Each pair reuses the same seeded discovery request sequence. The order of the
monitor-on and monitor-off blocks is randomized and balanced across pairs. Both
conditions keep the witness process alive; only monitor-on performs its requests.
Monitor-on adds one independent NRF profile GET and two witness HTTP reads after
each foreground discovery response. Foreground latency and total cycle latency
are reported separately: the latter includes deliberately blocking checks. This
is not a loaded-network causal estimate, UE latency, or ML inference benchmark.

Application bytes omit HTTP headers and TCP/HTTP2 framing. CPU covers the Python
benchmark and witness, not Open5GS. RSS is sampled rather than an exact peak.
Failed requests remain in latency tables and are counted explicitly.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import psutil

from .collect import Witness, digest, discover_native, profile_path, route_address
from .transport import H2Client, loopback_url


def percentiles(values):
    array = np.asarray(values, dtype=float)
    return {f'p{p}': float(np.percentile(array, p)) for p in (50, 95, 99)}


def cpu_seconds(process):
    times = process.cpu_times()
    return times.user + times.system


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    loopback_url(args.nrf)
    if args.blocks < 2 or args.requests < 1:
        raise ValueError('At least two paired blocks and one request are required')
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / 'manifest.json').exists():
        raise FileExistsError('Refusing to overwrite benchmark evidence')
    rng = random.Random(args.seed)
    # Balance order to limit drift confounding, then randomly permute it.
    first = ['off', 'on'] * (args.blocks // 2) + (['off'] if args.blocks % 2 else [])
    rng.shuffle(first)
    plan = []
    for pair in range(args.blocks):
        targets = [rng.choice(['NSSF', 'AUSF', 'UDM', 'SMF']) for _ in range(args.requests)]
        order = [first[pair], 'on' if first[pair] == 'off' else 'off']
        plan.append({'pair': pair, 'order': order, 'targets': targets})
    client = H2Client(args.nrf)
    native = discover_native(client)
    witness = Witness({'native': native})
    proc = psutil.Process()
    witness_proc = psutil.Process(witness.pid)
    manifest = {'started_at': time.time(), 'seed': args.seed, 'paired_blocks': args.blocks,
                'requests_per_block': args.requests, 'plan': plan, 'complete': False,
                'native_profile': native, 'condition_order_balanced': args.blocks % 2 == 0,
                'limitations': __doc__}
    manifest_path = args.out / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    rows, blocks = [], []
    raw_path = args.out / 'events.jsonl'
    try:
        with raw_path.open('w', encoding='utf-8') as raw:
            for pair in plan:
                for position, condition in enumerate(pair['order']):
                    selected = []
                    cpu_start, witness_cpu_start = cpu_seconds(proc), cpu_seconds(witness_proc)
                    block_started = time.perf_counter()
                    rss, witness_rss = proc.memory_info().rss, witness_proc.memory_info().rss
                    for index, target in enumerate(pair['targets']):
                        cycle_started = time.perf_counter()
                        started_at = time.time()
                        foreground = client.request('GET', '/nnrf-disc/v1/nf-instances?'
                                                    f'target-nf-type={target}&requester-nf-type=AMF')
                        foreground_ms = (time.perf_counter() - cycle_started) * 1000
                        monitor_started = time.perf_counter()
                        observer = envelope = route = None
                        authenticated = None
                        observer_bytes = observer_requests = 0
                        mismatch = None
                        monitor_error = ''
                        if condition == 'on':
                            before_bytes, before_requests = witness.bytes, witness.requests
                            observer = client.request('GET', profile_path(native))
                            observer_requests = 1
                            observer_bytes = observer['request_bytes'] + observer['response_bytes']
                            try:
                                envelope, route, authenticated, _ = witness.observe('native')
                                measured = observer.get('body')
                                mismatch = (observer['status'] != 200 or not isinstance(measured, dict)
                                            or not authenticated)
                                if not mismatch:
                                    mismatch = (digest(measured) != envelope['report']['profile_hash']
                                                or route_address(measured) != route['selected_address'])
                            except (OSError, ValueError, KeyError) as exc:
                                monitor_error = str(exc)
                            observer_bytes += witness.bytes - before_bytes
                            observer_requests += witness.requests - before_requests
                        monitor_ms = (time.perf_counter() - monitor_started) * 1000 if condition == 'on' else 0.
                        cycle_ms = (time.perf_counter() - cycle_started) * 1000
                        row = {'pair': pair['pair'], 'position': position, 'condition': condition,
                               'request_index': index, 'target': target, 'started_at': started_at,
                               'completed_at': time.time(), 'foreground_ms': foreground_ms,
                               'monitor_ms': monitor_ms, 'cycle_ms': cycle_ms,
                               'foreground_status': foreground['status'],
                               'foreground_failed': int(not 200 <= foreground['status'] < 300),
                               'observer_requests': observer_requests,
                               'observer_application_bytes': observer_bytes,
                               'monitor_failed': int(bool(monitor_error) or (condition == 'on' and
                                                     (observer['status'] != 200 or not authenticated))),
                               'evidence_mismatch': mismatch}
                        selected.append(row)
                        rows.append(row)
                        raw.write(json.dumps({'identity': row, 'foreground': foreground,
                                              'observer': observer, 'witness': envelope,
                                              'consumer': route, 'monitor_error': monitor_error}) + '\n')
                        raw.flush()
                        rss = max(rss, proc.memory_info().rss)
                        witness_rss = max(witness_rss, witness_proc.memory_info().rss)
                    elapsed = time.perf_counter() - block_started
                    block = {'pair': pair['pair'], 'position': position, 'condition': condition,
                             'requests': len(selected), 'elapsed_seconds': elapsed,
                             'throughput_cycles_per_second': len(selected) / elapsed,
                             'foreground_failures': sum(r['foreground_failed'] for r in selected),
                             'monitor_failures': sum(r['monitor_failed'] for r in selected),
                             'observer_requests': sum(r['observer_requests'] for r in selected),
                             'observer_application_bytes': sum(r['observer_application_bytes'] for r in selected),
                             'benchmark_cpu_seconds': cpu_seconds(proc) - cpu_start,
                             'witness_cpu_seconds': cpu_seconds(witness_proc) - witness_cpu_start,
                             'benchmark_sampled_max_rss_bytes': rss,
                             'witness_sampled_max_rss_bytes': witness_rss}
                    for metric in ('foreground_ms', 'monitor_ms', 'cycle_ms'):
                        block.update({f'{metric}_{p}': value for p, value in
                                      percentiles([r[metric] for r in selected]).items()})
                    blocks.append(block)
                    print(f"Pair {pair['pair'] + 1}/{args.blocks} {condition}: "
                          f"cycle p50={block['cycle_ms_p50']:.3f} ms", flush=True)
        write_csv(args.out / 'requests.csv', rows)
        write_csv(args.out / 'blocks.csv', blocks)
        differences = []
        for pair in plan:
            off = next(b for b in blocks if b['pair'] == pair['pair'] and b['condition'] == 'off')
            on = next(b for b in blocks if b['pair'] == pair['pair'] and b['condition'] == 'on')
            difference = {'pair': pair['pair'], 'first_condition': pair['order'][0]}
            for metric in ('foreground_ms_p50', 'foreground_ms_p95', 'foreground_ms_p99',
                           'cycle_ms_p50', 'cycle_ms_p95', 'cycle_ms_p99',
                           'throughput_cycles_per_second', 'benchmark_cpu_seconds', 'witness_cpu_seconds'):
                difference[metric + '_on_minus_off'] = on[metric] - off[metric]
            differences.append(difference)
        write_csv(args.out / 'paired_differences.csv', differences)
        summary = {'paired_blocks': args.blocks, 'requests_total': len(rows),
                   'foreground_failures': sum(r['foreground_failed'] for r in rows),
                   'monitor_failures': sum(r['monitor_failed'] for r in rows),
                   'median_paired_differences': {k: float(np.median([d[k] for d in differences]))
                                                  for k in differences[0] if k.endswith('_on_minus_off')},
                   'interpretation': 'Descriptive paired sequential monitoring cost on an isolated runner; '
                                     'no inference about production-network causality or UE latency.'}
        (args.out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        manifest['complete'] = True
    finally:
        witness.close()
        manifest['completed_at'] = time.time()
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.out.iterdir() if p.is_file()}
    (args.out / 'sha256.json').write_text(json.dumps(hashes, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--blocks', type=int, default=8, help='Number of on/off pairs')
    parser.add_argument('--requests', type=int, default=30)
    parser.add_argument('--nrf', default='http://127.0.0.10:7777')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
