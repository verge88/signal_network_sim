# 5G SBA semantic consistency v5 — Byzantine hardening

V5 is a targeted research iteration built on top of `feat/5g-semantic-v4-online`.
It addresses the main negative result from the v4 trust-domain experiment:
a single Byzantine origin could raise semantic false alarms to double-digit
percentages under the old 2-of-3-style soft-fact quorums.

## What changes

The event generator and transport channel remain inherited from v4. The new
semantic layer adds:

- explicit evidence origins rather than raw report counts;
- one vote maximum per trust root for each `(event, fact)` decision;
- seven synthetic evidence origins: consumer, SCP, NRF, producer, NWDAF,
  policy, audit;
- quorum profiles `q2of3`, `q3of5`, `q4of7`;
- HARD facts: token, slice, discovery;
- SOFT facts: notification, procedure, route;
- temporal persistence for SOFT facts (default: two distinct events within
  90 seconds);
- separate `Byzantine FPR` measurement under one-origin `INJECT`;
- suppression ablation under one-origin `SUPPRESS`;
- explicit separation of event latency and campaign TTD.

The `policy` and `audit` origins are proof-of-concept independent evidence
sources for the quorum-size ablation. They are not claimed to be standardized
3GPP network functions. Their observability and latency probabilities are
synthetic assumptions pending calibration against free5GC/Open5GS/operator
traces.

## Scientific questions

The main experiment compares:

`Recall_adaptive`, `Recall_suppress`, `FPR_benign`, and `FPR_Byzantine`

for:

- 2-of-3;
- 3-of-5;
- 4-of-7;
- soft persistence OFF/ON.

The expected trade-off is that larger quorums and soft persistence reduce
Byzantine false alarms at the cost of some attack recall and potentially higher
latency.

## Tests

From the repository root:

```powershell
python -m unittest tests.test_sba_semantic_v5 -v
python -m unittest discover -s tests -v
```

## First targeted run

```powershell
python .\5g\run_semantic_v5.py `
    --seed 42 `
    --calib-windows 1200 `
    --eval-windows 200 `
    --trust-windows 200 `
    --hidden 5 `
    --out-dir runs\sba_semantic_v5\seed42_byzantine
```

Outputs:

- `benign_fpr.csv`
- `adaptive_attack_recall.csv`
- `suppression_ablation.csv`
- `byzantine_injection_fpr.csv`
- `quorum_ablation_summary.csv`
- `summary.json`

## Dissertation-use caution

Absolute visibility, latency, UNKNOWN rates, and Byzantine FPR values are still
simulation outputs. They should be presented as internal method validation,
not as measured operator-network characteristics. The strongest dissertation
claim supported by this iteration is comparative: how quorum size, independent
evidence origins, and soft persistence change recall and false-alarm robustness
under the same synthetic world model.
