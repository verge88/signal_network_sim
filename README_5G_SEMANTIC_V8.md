# 5G SBA semantic consistency v8 — noisy operational context

V8 keeps the detector core selected by the v5–v6 ablations:

- semantic quorum: `q3of5`;
- global persistence: OFF;
- transport: `byz_3of4`.

It addresses the main validity limitation found after the v7 diagnostic. In v7,
operational markers and recovery evidence were generated directly from the
known legitimate disturbance, so the operational-context channel behaved too
much like an oracle. V8 separates the generative causes and observations.

## Latent-state model

A window can contain one of the following latent operational states:

- `token_refresh`;
- `policy_sync`;
- `subscription_sync`;
- `nf_recovery`;
- `route_transition`;
- or `normal`.

The detector never receives this latent state. The state can independently
cause three observable effects:

1. a transient semantic inconsistency;
2. an operational marker message;
3. later recovery/convergence evidence.

The marker is therefore not generated from `disturbance_event_ids`, and the
recovery channel is not guaranteed to arrive before grace expiry.

## Default synthetic observation errors

`OperationalContextConfig` exposes the proof-of-concept assumptions:

- marker recall: `0.85`;
- false marker probability per normal window: `0.02`;
- wrong-correlation probability: `0.03`;
- marker latency median: `0.12 s` with lognormal jitter;
- recovery observation probability: `0.90`;
- transient manifestation probability: `0.95`;
- recovery latency is lognormal and can exceed the grace interval.

These are **not operator measurements**. They are sensitivity parameters for an
internal-validity experiment. Production claims require calibration from
free5GC/Open5GS/UERANSIM or operator traces.

## Detector behavior

The marker is useful only if it:

- has arrived before semantic quorum confirmation;
- references the same correlation ID;
- covers the same semantic fact;
- is still inside its validity interval.

A late marker cannot retract an already emitted alarm. A wrong-correlation
marker is ignored. If a valid marker defers a quorum and evidence converges,
the transient is resolved. If inconsistency remains when grace expires, the
alarm is emitted at expiry.

Thus grace remains correlation/fact scoped rather than a global persistence
rule or whitelist.

## Adaptive attack during a real operational state

`generate_attack_during_state()` conditions an adaptive attack to occur during
a genuine operational state. The operational marker still passes through the
same noisy marker channel; unlike v7's stress helper, it is not inserted with
certainty. No synthetic recovery is created for the malicious inconsistency.

This supports the quantities:

- `Recall(attack | correct marker)`;
- `Recall(attack | missing/wrong marker)`;
- event latency under a real operational state.

## Tests

```powershell
python -m unittest tests.test_sba_semantic_v8 -v
python -m unittest tests.test_sba_semantic_v8_stats -v
python -m unittest discover -s tests -v
```

The scientific-contract tests cover marker misses, false markers, wrong
correlation, recovery after grace, late-marker non-retraction, transient
resolution, attack-after-expiry behavior, frozen q3of5/byz_3of4 configuration,
and seed-paired statistics.

## Diagnostic run

```powershell
python .\5g\run_semantic_v8.py `
    --seed 42 `
    --calib-windows 1200 `
    --eval-windows 100 `
    --operational-windows 100 `
    --state-attack-windows 80 `
    --hidden 5 `
    --out-dir runs\sba_semantic_v8\seed42_diagnostic
```

Outputs:

- `benign.csv` / `all_seed_benign.csv`;
- `attacks.csv` / `all_seed_attacks.csv`;
- `operational_states.csv` / `all_seed_operational_states.csv`;
- `attack_during_state.csv` / `all_seed_attack_during_state.csv`;
- `seed_summary.csv` / `all_seed_summary.csv`;
- `paired_statistics.csv`;
- `summary.json`.

The state tables also report empirical marker emission/correctness, false
marker rate, transient manifestation, recovery emission and the fraction of
observed recoveries completing before grace expiry.

## Multiseed run

Run the 10-seed experiment only after the diagnostic suite is clean:

```powershell
python .\5g\run_semantic_v8.py `
    --seeds 42 43 44 45 46 47 48 49 50 51 `
    --calib-windows 1200 `
    --eval-windows 200 `
    --operational-windows 200 `
    --state-attack-windows 100 `
    --hidden 5 `
    --bootstrap 5000 `
    --permutations 10000 `
    --out-dir runs\sba_semantic_v8\multiseed_42_51
```

`paired_statistics.csv` contains seed-paired deltas, bootstrap 95% confidence
intervals, sign-flip p-values and Holm-adjusted p-values.

## Primary success criteria

V8 is useful only if the benefit seen in v7 survives imperfect context:

1. ordinary adaptive Recall is essentially unchanged;
2. mean semantic operational FPR remains materially below the static q3of5
   baseline, but is no longer forced to zero;
3. attack-during-state Recall does not silently collapse;
4. latency increases are explicitly reported as the cost of grace;
5. residual errors correlate with marker misses/wrong correlation and
   missing/late recovery rather than hidden labels.

A null or negative result is scientifically acceptable. The goal is to remove
the oracle assumption, not to preserve a preferred numerical result.
