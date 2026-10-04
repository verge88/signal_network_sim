# 5G SBA semantic consistency v9

V9 is an **experimental-protocol iteration**, not a new detector. The selected v8 core remains frozen:

- semantic quorum: `q3of5`;
- global soft persistence: OFF;
- transport aggregation: `byz_3of4`;
- noisy operational-context model: `sba_semantic_v8.py`.

The purpose of v9 is to separate confirmatory hypotheses from exploratory diagnostics and to test sensitivity to imperfect operational telemetry.

## Why v9 exists

The v8 10-seed run on seeds 42..51 showed a large paired reduction in operational false alarms while preserving observed Recall, but the previous runner applied Holm correction to a large mixed family of primary and exploratory endpoints. With only 10 paired seeds, the minimum exact two-sided sign-flip p-value is `2 / 2^10 = 0.001953125`; multiplying that across roughly three dozen hypotheses can make adjusted `p < 0.05` impossible even under perfectly aligned seed-wise effects.

V9 fixes the inferential family before a **fresh holdout** run and moves per-state/per-family analyses to a secondary exploratory layer. The old 42..51 run should remain reported as pilot/exploratory evidence; it must not be relabelled as a preregistered confirmatory experiment after the fact.

## Frozen primary family

The confirmatory runner writes `analysis_plan.json` before any simulation and locks it with a SHA-256 digest. A different plan cannot reuse the same output directory.

Four endpoints form the complete primary Holm family:

1. **H1 operational FPR superiority** — `mean_operational_fpr`, expected `v9 < static`.
2. **H2 ordinary attack Recall non-inferiority** — `mean_attack_window_recall`, default absolute margin 1 percentage point.
3. **H3 attack-in-state Recall non-inferiority** — `state_attack_window_recall`, default absolute margin 1 percentage point.
4. **H4 latency-cost characterization** — `state_attack_event_latency_s`, expected to increase under grace.

H1/H4 report paired bootstrap CIs, exact/Monte-Carlo sign-flip p-values and Holm-adjusted p-values across these four endpoints only. H2/H3 are decided by the paired bootstrap CI lower bound against the fixed non-inferiority margin. Sign-flip p-values are still reported for transparency, but equality-to-zero testing is not used as evidence of non-inferiority.

The default confirmatory seeds are **52..61**, deliberately disjoint from the exploratory 42..51 set.

## Confirmatory run

```powershell
python .\5g\run_semantic_v9_confirmatory.py `
    --seeds 52 53 54 55 56 57 58 59 60 61 `
    --calib-windows 1200 `
    --eval-windows 200 `
    --operational-windows 200 `
    --state-attack-windows 100 `
    --hidden 5 `
    --bootstrap 5000 `
    --permutations 10000 `
    --out-dir runs\sba_semantic_v9\confirmatory_52_61
```

Primary outputs:

- `analysis_plan.json` — locked plan and SHA-256 before simulation;
- `all_seed_summary.csv`;
- `primary_statistics.csv`;
- `secondary_statistics.csv`;
- `summary.json`.

Do not change the primary family, alpha or non-inferiority margin after inspecting confirmatory results. A changed plan requires a new output directory and should be described as a new experiment.

## Exploratory sensitivity grid

The sensitivity runner is intentionally separate from the confirmatory analysis. Its results do **not** enter the primary Holm family.

The fixed grid has 10 points:

- baseline: marker recall 0.85, recovery probability 0.90, recovery multiplier 1.0, correlation error 0.03;
- marker recall 0.60, 0.75, 0.95;
- recovery probability 0.60, 0.75;
- recovery latency multiplier 2x, 4x;
- correlation error 0.10;
- combined degraded case: marker 0.60, recovery 0.60, latency multiplier 2x, correlation error 0.10.

A moderate screening run:

```powershell
python .\5g\run_semantic_v9_sensitivity.py `
    --seeds 42 47 51 `
    --calib-windows 600 `
    --eval-windows 80 `
    --operational-windows 100 `
    --state-attack-windows 60 `
    --hidden 5 `
    --out-dir runs\sba_semantic_v9\sensitivity_screen
```

For a fast smoke test, restrict points:

```powershell
python .\5g\run_semantic_v9_sensitivity.py `
    --seeds 42 `
    --points baseline marker_060 recovery_slow_4x degraded_combo `
    --calib-windows 120 `
    --eval-windows 20 `
    --operational-windows 30 `
    --state-attack-windows 20 `
    --bootstrap 500 `
    --out-dir runs\sba_semantic_v9\sensitivity_smoke
```

Sensitivity outputs include per-point/per-seed summaries plus `sensitivity_seed_metrics.csv`, `sensitivity_summary.csv`, `sensitivity_plan.json`, and `summary.json`.

## Validation

Run before expensive experiments:

```powershell
python -m unittest tests.test_sba_semantic_v9_stats -v
python -m unittest tests.test_sba_semantic_v9_sensitivity -v
python -m unittest discover -s tests -v
```

## Interpretation constraints

The operational-state probabilities, marker errors, recovery probabilities and timings remain synthetic proof-of-concept assumptions. V9 strengthens internal validity and statistical discipline; it does not convert the simulator into calibrated production-5GC evidence.

The dissertation-level claim should therefore remain comparative: under the simulated observation model, fact/correlation-scoped noisy operational context can reduce transient semantic false alarms while preserving observed attack Recall, at the measurable cost of increased detection latency during operational states. Absolute field-network FPR/Recall claims still require implementation traces or a real 5GC testbed.
