# 5G SBA semantic consistency v7 — operational grace + paired multiseed

V7 continues the v3 → v6 experimental chain.  The v6 result fixed the current
candidate at:

- semantic confirmation: `q3of5`;
- global soft persistence: **OFF**;
- transport confirmation: `byz_3of4`.

V7 does **not** tune those quorum mechanisms again.  It addresses the next
observed limitation: legitimate short-lived cross-NF disagreement produced
semantic operational false positives (about 4–9% in the seed-42 v6 stress
experiment).

## Methodological question

The detector should distinguish

```text
persistent malicious inconsistency
            vs
legitimate transient inconsistency that converges
```

without imposing global persistence on every semantic fact.

V7 therefore uses a narrowly scoped operational-state contract:

```text
Decision = f(
    fact,
    correlation/session,
    q3of5 evidence,
    operational state,
    inconsistency age,
    recovery/convergence evidence
)
```

## Operational states

The simulator exposes explicit control-plane context for the five v6 stress
classes:

| Disturbance | State | Fact | Synthetic grace |
|---|---|---|---:|
| token refresh race | `token_refresh` | token | 2 s |
| policy propagation delay | `policy_sync` | slice | 8 s |
| subscription desync | `subscription_sync` | notification | 4 s |
| NF restart/recovery | `nf_recovery` | procedure | 15 s |
| benign SCP reroute | `route_transition` | route | 3 s |

These durations are simulation assumptions, not operator-network measurements.

An operational marker is scoped by **correlation ID + semantic fact + time
interval**.  It is not a global whitelist and is not an attack label.

## Stateful evidence

V6 observations were effectively one-shot.  V7 allows a later observation from
the same trust root to replace its earlier state:

```text
INCONSISTENT  ->  CONSISTENT
```

During a matching operational grace interval:

1. q3of5 may be reached;
2. the decision is deferred only for that correlation/fact;
3. if the same independent origins converge before grace expiry, the transient
   is suppressed;
4. if q3of5 remains unresolved at expiry, an alert is emitted at expiry.

Thus an attacker cannot obtain permanent suppression merely because a grace
marker exists.

## Grace-abuse stress case

The runner explicitly creates attack worlds in which the attacked correlation
is covered by a legitimate-looking operational marker **without recovery
evidence**.  The scientific requirement is:

- Recall should remain close to the no-grace attack Recall;
- event latency may increase by the configured grace duration;
- unresolved evidence must generate `grace_expired_alerts`.

This is important because an operational state must not become a silent
whitelist.

## Paired experiment

`5g/run_semantic_v7.py` compares two detector arms on exactly the same worlds:

- `v6_static_q3of5` — v6 baseline with `byz_3of4` transport;
- `v7_operational_grace` — same quorum and transport plus stateful operational
  grace.

For every seed it measures:

- ordinary benign FPR and null AUC;
- adaptive attack window/event Recall;
- operational-disturbance FPR;
- semantic operational-disturbance FPR;
- grace-abuse Recall and latency;
- deferred/resolved/expired grace counters.

Across seeds, paired deltas are reported with:

- bootstrap 95% CI over seed-paired differences;
- paired sign-flip permutation test (exact for up to 16 seeds);
- Holm adjustment across reported hypotheses.

## Files

- `5g/sba_semantic_v7.py` — operational markers, recovery evidence and stateful detector;
- `5g/run_semantic_v7.py` — paired single/multi-seed experiment;
- `tests/test_sba_semantic_v7.py` — scientific-contract regression tests.

## Validation

```powershell
python -m unittest tests.test_sba_semantic_v7 -v
python -m unittest discover -s tests -v
```

## First diagnostic run

Run one seed before spending time on the multi-seed experiment:

```powershell
python .\5g\run_semantic_v7.py `
    --seed 42 `
    --calib-windows 1200 `
    --eval-windows 100 `
    --disturbance-windows 100 `
    --grace-abuse-windows 80 `
    --hidden 5 `
    --out-dir runs\sba_semantic_v7\seed42_diagnostic
```

The main diagnostic success conditions are:

1. ordinary attack Recall is essentially unchanged relative to v6;
2. semantic disturbance FPR decreases materially;
3. attack-under-grace Recall is not silently suppressed;
4. null AUC remains near 0.5.

## Paired 10-seed run

After the diagnostic run passes:

```powershell
python .\5g\run_semantic_v7.py `
    --seeds 42 43 44 45 46 47 48 49 50 51 `
    --calib-windows 1200 `
    --eval-windows 200 `
    --disturbance-windows 200 `
    --grace-abuse-windows 100 `
    --hidden 5 `
    --bootstrap 5000 `
    --permutations 10000 `
    --out-dir runs\sba_semantic_v7\multiseed_42_51
```

The current target FPR remains 1% for method development.  This is **not** the
final dissertation M5 setting.  A later final experiment must repeat the frozen
method at `target_fpr=0.001` with substantially larger per-context calibration
and evaluation samples.

## Outputs

Single-seed directories contain:

- `benign.csv`;
- `attacks.csv`;
- `disturbances.csv`;
- `grace_abuse.csv`;
- `seed_summary.csv`.

The root multi-seed directory additionally contains:

- `all_seed_benign.csv`;
- `all_seed_attacks.csv`;
- `all_seed_disturbances.csv`;
- `all_seed_grace_abuse.csv`;
- `all_seed_summary.csv`;
- `paired_statistics.csv`;
- `summary.json`.

## Scientific caution

Operational markers and grace/recovery durations are synthetic proof-of-concept
context.  The intended claim is comparative internal validity:

> Given observable control-plane state and independent convergence evidence,
> correlation/fact-scoped grace can change the trade-off between operational
> false alarms, attack sensitivity and detection latency.

Do not present the current grace durations, disturbance rates or absolute FPR as
production-5GC measurements until they are calibrated against free5GC,
Open5GS/UERANSIM or operator traces.
