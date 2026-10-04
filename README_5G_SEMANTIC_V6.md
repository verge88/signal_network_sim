# 5G SBA semantic consistency v6 — Byzantine-resistant transport

V6 continues the stacked v3 → v4 → v5 experiment series. It is based on the
v5 result that **q3of5, soft persistence OFF** was the best semantic trade-off:
semantic one-origin INJECT false alarms were not observed in the targeted run,
while the remaining total Byzantine FPR came from the statistical transport
channel.

## What is frozen from v5

The semantic candidate is fixed at:

- profile: `q3of5`;
- one vote per trust origin;
- global soft persistence: OFF;
- adaptive attacker remains the primary stress level.

V6 does **not** increase the semantic quorum again.

## Byzantine-resistant transport

The legacy transport detector uses the median of four physical observers. V6
adds `byz_3of4`:

1. residuals are normalized per `context × transport domain` using a robust
   median and MAD scale;
2. the runtime statistic is the **third largest** normalized residual, i.e. a
   3-of-4 high-side confirmation rule;
3. at least three finite observer reports are required;
4. the threshold is not calibrated on the ordinary 3-of-4 statistic. Instead,
   calibration uses the **second largest** benign normalized residual. This is
   the worst-case runtime value obtained when one benign observer is replaced
   by an arbitrarily high Byzantine report.

This makes the threshold intentionally conservative for the threat model
"one high-side arbitrary transport injector".

The design is asymmetric: it is specifically aimed at false-alarm injection.
SUPPRESS is measured separately because a stronger quorum can reduce attack
sensitivity when one observer hides evidence.

## Legitimate semantic disturbances

V6 also adds attack-label-free operational stress scenarios:

- `token_refresh_race`;
- `policy_propagation_delay`;
- `subscription_desync`;
- `nf_restart_recovery`;
- `scp_benign_reroute`.

A disturbance normally produces transient inconsistency at two origins. With
an explicit 5% synthetic escalation probability, a third origin can observe
the same temporary disagreement. This allows q3of5 to produce occasional
operational false positives rather than making Precision structurally perfect.

These probabilities are **simulation assumptions**, not operator-network
measurements. Absolute disturbance FPR must not be presented as a field-network
estimate without external calibration.

## Files

- `5g/sba_semantic_v6.py` — v6 simulator wrapper and transport detector;
- `5g/run_semantic_v6.py` — paired legacy-vs-robust experiment;
- `tests/test_sba_semantic_v6.py` — scientific-contract regression tests.

## Tests

```powershell
python -m unittest tests.test_sba_semantic_v6 -v
python -m unittest discover -s tests -v
```

The v6 tests check that:

- the semantic candidate remains q3of5 with persistence off;
- legitimate disturbances never become attack ground truth;
- disturbance RNG does not change the paired transport world;
- one or two arbitrarily high transport reports cannot satisfy 3-of-4;
- three high reports can satisfy 3-of-4;
- adversarial calibration is at least as conservative as the runtime statistic;
- the legacy median remains available as the control arm.

## Recommended first run

```powershell
python .\5g\run_semantic_v6.py `
    --seed 42 `
    --calib-windows 1200 `
    --eval-windows 200 `
    --trust-windows 200 `
    --disturbance-windows 200 `
    --hidden 5 `
    --out-dir runs\sba_semantic_v6\seed42_transport
```

For a faster diagnostic run, skip the expensive SUPPRESS matrix:

```powershell
python .\5g\run_semantic_v6.py `
    --seed 42 `
    --calib-windows 1200 `
    --eval-windows 100 `
    --trust-windows 100 `
    --disturbance-windows 100 `
    --hidden 5 `
    --skip-suppress `
    --out-dir runs\sba_semantic_v6\seed42_quick_transport
```

## Outputs

- `benign_fpr.csv`;
- `adaptive_attack_recall.csv`;
- `transport_byzantine_injection.csv`;
- `transport_suppression.csv`;
- `legitimate_disturbances.csv`;
- `transport_ablation_summary.csv`;
- `summary.json`.

## Primary questions

The experiment is successful only if it exposes the trade-off; high numbers by
themselves are not the objective.

1. Does `byz_3of4` reduce one-origin transport Byzantine FPR relative to
   `legacy_median`?
2. How much adaptive attack Recall is lost?
3. How much worse is Recall under one-origin SUPPRESS?
4. Do legitimate semantic disturbances produce a non-zero but bounded semantic
   operational FPR?
5. Does the null AUC remain close to 0.5?

If the robust mode reduces Byzantine FPR but destroys Recall, it is not a valid
replacement. The intended result is a Pareto comparison, not confirmation of a
pre-selected winner.

## Dissertation caution

At the current stage the scientific claim should be comparative:

> Under the stated synthetic one-Byzantine-observer model, adversarially
> calibrated multi-observer transport confirmation changes the trade-off
> between false-alarm injection resistance and attack sensitivity.

Do not claim that the selected thresholds, latencies, disturbance rates or
absolute Recall values directly represent a production 5GC until the model is
calibrated against free5GC/Open5GS/UERANSIM/operator evidence.
