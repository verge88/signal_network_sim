# Local smoke validation (3 seeds)

This file records a deliberately small validation run performed before committing the
experiment code. It is **not** the dissertation-grade result; the full command in the
README uses larger windows, more trees, three severities and five seeds.

Smoke configuration per seed:

- train windows: 40
- calibration windows: 20
- warm-up windows: 15
- evaluation windows per scenario: 12
- supervised attack windows: 8
- Random Forest trees: 20
- severity: 1.0
- seeds: 101, 202, 303

## Mean recall over 3 seeds

### Isolation Forest (benign-trained, calibrated)

| Scenario | SBI | SBI+Zone | SBI+Active | FULL |
|---|---:|---:|---:|---:|
| compromised_lie | 0.389 | 0.667 | **0.806** | 0.778 |
| compromised_truth | 0.278 | **0.583** | 0.139 | 0.389 |
| slow_drift_lie | 0.056 | 0.028 | **0.333** | 0.222 |
| fake_nf | 0.222 | 0.222 | **0.833** | **0.833** |
| rapid_reset | **1.000** | **1.000** | 0.778 | **1.000** |
| oauth_abuse | **0.889** | 0.500 | 0.750 | 0.833 |
| nrf_poisoning | **1.000** | 0.861 | 0.972 | 0.417 |
| cross_slice | **0.278** | 0.111 | 0.139 | 0.194 |
| coordinated_majority | **0.403** | 0.097 | 0.208 | 0.111 |

Mean normal-holdout FPR was about 0.97% for SBI and 0.93% for FULL.

The expected research effect is visible: active KIE/self-report evidence materially improves
`compromised_lie`, `slow_drift_lie` and `fake_nf`. Peer-zone evidence helps
`compromised_truth`. The falsification scenarios also behave as intended:
`coordinated_majority` can corrupt the local peer median, and a naive multi-channel
unsupervised fusion can lose power on some strong SBI-only anomalies such as
`nrf_poisoning`.

This is a useful negative result: the experiment must report feature-group ablations, not
only a single `FULL` number. It also motivates a learned/cascade meta-ensemble instead of
assuming that concatenating or max-fusing every anomaly channel is always superior.

### Random Forest (supervised synthetic training)

| Scenario | SBI | SBI+Zone | SBI+Active | FULL |
|---|---:|---:|---:|---:|
| compromised_lie | 0.722 | 0.889 | **0.972** | 0.861 |
| compromised_truth | 0.833 | **0.861** | 0.750 | 0.694 |
| slow_drift_lie | 0.111 | 0.111 | **0.500** | 0.167 |
| fake_nf | 1.000 | 1.000 | 1.000 | 1.000 |
| rapid_reset | 1.000 | 1.000 | 1.000 | 1.000 |
| oauth_abuse | 1.000 | 1.000 | 1.000 | 0.972 |
| nrf_poisoning | 1.000 | 0.944 | 0.972 | 0.972 |
| cross_slice | 0.972 | 0.972 | **1.000** | 0.917 |
| coordinated_majority | 0.694 | **0.764** | 0.639 | 0.583 |

The supervised detector is intentionally included as a second family; the scientific claim
remains about the information contribution of active/self-report and zone/temporal
features, not about Random Forest novelty.

## Interpretation

The smoke validation supports the narrower hypothesis:

> Comparing independently observed SBI behavior with authenticated NF self-report can add
> strong evidence for a compromised legitimate NF that hides or falsifies its activity.

It does **not** support the stronger statement that every FULL fusion is universally better
than every SBI-only detector. The full multi-seed experiment and the ablation/statistical
tables are therefore mandatory for any dissertation claim.
