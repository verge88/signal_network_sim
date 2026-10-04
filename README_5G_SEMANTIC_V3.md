# 5G SBA semantic consistency v3

This is the event-level implementation of the third research iteration.
It is additive: the legacy aggregate `5g/sba_sim_v1.py` remains the transport
world and scientific baseline, while `5g/sba_semantic_v3.py` adds the
protocol-semantic layer.

## New model

Trust domains:

- Consumer NF
- SCP
- NRF
- Producer NF
- NWDAF

Hard semantic facts:

- token authorization
- notification/subscription consistency
- slice authorization
- NRF discovery consistency
- procedure causality
- route consistency

The statistical channel is transport residual, fused by a robust median and
calibrated separately for each operational context.

`direct_allowed` is an explicit legitimate context, so the route invariant
checks policy consistency and does **not** equate "direct SBI" with an attack.

## Scientific contract

Compromise is a capability, not an anomaly. A compromised domain changes
telemetry only when `CompromiseMode.SUPPRESS` or `CompromiseMode.INJECT` is
explicitly requested. `PASSIVE` must be indistinguishable from the same benign
world without compromise.

The detector receives only observations and transport telemetry. It does not
consume `label`, `attack_family`, `hidden_calls`, or `compromised_domains`.
Ground truth is used only by `run_semantic_v3.py` after detector scoring to
calculate evaluation metrics.

## Runner profiles

### Quick smoke test

```powershell
python .\5g\run_semantic_v3.py --quick --seed 42
```

`quick` is only a functional smoke test. Its calibration sample is intentionally
small; FPR/Recall values must not be used as dissertation results.

### Development experiment

```powershell
python .\5g\run_semantic_v3.py `
    --profile dev `
    --seed 42 `
    --calib-windows 1200 `
    --eval-windows 1200 `
    --hidden-grid 1 2 5 10 20 40 80 `
    --trust-hidden 40 `
    --trust-eval-windows 400 `
    --out-dir runs\sba_semantic_v3\seed42_dev
```

### Research profile

```powershell
python .\5g\run_semantic_v3.py `
    --profile research `
    --seed 42 `
    --out-dir runs\sba_semantic_v3\seed42_research
```

`research` defaults to 6000 calibration windows, 6000 benign/attack evaluation
windows and the detectability grid `1 2 5 10 20 40 80`.

### Multi-seed experiment

Start with a reduced development run:

```powershell
python .\5g\run_semantic_v3.py `
    --profile dev `
    --seeds 42 43 44 45 46 47 48 49 50 51 `
    --calib-windows 1200 `
    --eval-windows 1200 `
    --hidden-grid 1 2 5 10 20 40 80 `
    --trust-eval-windows 200 `
    --out-dir runs\sba_semantic_v3\multiseed_dev
```

For a faster detectability-only run add `--skip-trust`.

## Terminal logging

The runner now prints:

- selected profile, seeds, hidden grid and FPR target;
- calibration progress, windows/s and ETA;
- calibration count and threshold for every operational context;
- benign evaluation progress;
- FPR and Wilson 95% CI by context;
- false attributed semantic events per 1000 benign events;
- null AUC and bootstrap 95% CI;
- for every attack and hidden volume: window Recall, event Recall, event
  Precision, TTD median/P95, detected hidden volume and MAE;
- trust-domain suppression results;
- total runtime and multi-seed summaries.

Use `--log-every N` to control progress frequency. Use `--quiet` to suppress
informational logs.

## Metrics

The runner separates two levels of detection.

### Window-level Recall

A window is detected if any semantic or statistical gate fires.

### Event-level Recall and Precision

Attack-event ground truth is used only in the evaluation harness:

- `event_recall = detected malicious events / malicious events`;
- `event_precision = detected malicious events / attributed event IDs`.

This prevents a window with `hidden_calls=40` from looking perfect merely
because one of forty malicious events was detected.

### TTD

`ttd_s` is measured from the earliest malicious event in the window to the
earliest malicious event attributed by the semantic detector. Aggregate output
contains median and P95 TTD.

### Hidden-volume estimate

The first event-level estimate is deliberately simple and auditable:

- `detected_hidden_calls = number of malicious event IDs attributed`;
- `hidden_calls_mae = |detected_hidden_calls - hidden_calls|`.

It should not yet be interpreted as a calibrated estimator of real attack
volume.

## Outputs

Single-seed runs write:

- `summary.json`
- `calibration_thresholds.csv`
- `fpr_by_context.csv`
- `null_test.json`
- `detectability_curve.csv`
- `per_attack.csv` (backward-compatible alias of the detectability table)
- `trust_domain_suppression.csv` unless `--skip-trust`

Multi-seed runs additionally write:

- `seed_summary.csv`
- `all_seed_fpr.csv`
- `all_seed_detectability.csv`
- `all_seed_thresholds.csv`
- `all_seed_trust_domain_suppression.csv`
- `multiseed_aggregate.csv`

## Calibration guard

For every operational context the runner checks whether enough calibration
windows exist for the requested target FPR. With `target_fpr=0.01`, fewer than
100 windows per context triggers a strong warning; fewer than roughly 1000 per
context is reported as statistically weak for stable tail estimation.

## Tests

```powershell
python -m unittest tests.test_sba_semantic_v3 -v
```

The regression suite checks direct-communication negative control, route
quorum, one-domain suppression, token/slice semantic gates, passive-compromise
null-world equivalence, null AUC, event-level metric accounting and Wilson
confidence intervals.

## Current limitation

The semantic-observation miss/false-evidence probabilities are still synthetic.
They must be calibrated against free5GC/Open5GS or operator traces before
absolute Recall is interpreted as a real-network estimate.
