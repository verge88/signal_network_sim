# Guarded-cascade iteration

This iteration follows the 20-seed decision-fusion experiment.

## Motivation

The previous experiment established two facts at the same time:

1. active report-vs-observed evidence strongly improves detection of
   `compromised_lie` and `slow_drift_lie`;
2. a global decision-fusion detector can still lose recall on attacks that are
   already easy for the SBI-only detector, including OAuth abuse, NRF poisoning
   and some cross-slice cases.

The next hypothesis is therefore deliberately narrower:

> Can an SBI-first guarded cascade preserve the original SBI detector's
> sensitivity while adding a low-FPR active/KIE channel for hidden compromise?

## Models

The experiment compares:

- `iforest_sbi`: original SBI-only baseline;
- `iforest_full_legacy`: previous grouped FULL detector;
- `fusion_maxq`: percentile-calibrated decision fusion from the prior iteration;
- `cascade_preserve_sbi`: keeps the original 1% SBI detector unchanged, then
  adds ACTIVE/KIE and CONTEXT alarms with very small benign-FPR budgets;
- `cascade_equal_fpr`: allocates a nominal 1% total FPR budget across SBI,
  active and context stages (0.85% + 0.10% + 0.05%).

All unsupervised components are fitted and calibrated on benign data only.

## Stages

The guarded cascade is

```text
SBI detector
   |
   +-- alarm ------------------------------+
   |                                       |
   v                                       |
if SBI-negative: ACTIVE / KIE detector      |
   |                                       |
   +-- alarm ------------------------------+--> final alarm
   |                                       |
   v                                       |
if still negative: TEMPORAL / ZONE context |
   |                                       |
   +-- alarm ------------------------------+
```

The preserve-SBI variant is intended to make a direct falsifiable claim:
standard SBI attacks should not lose recall at the primary stage, because the
primary stage is literally the same fitted SBI detector.

## New adaptive-attacker scenario

`adaptive_lie` is added between the old extremes `compromised_lie` and
`compromised_truth`.

The NF creates the same malicious SBI increase as the lying-compromise scenario,
but mirrors 75% of the malicious traffic delta in its own report. Only 25% of
the discrepancy remains visible to the report-vs-observed check.

This evaluates whether the active-channel gain collapses smoothly as an attacker
learns to imitate independently observed behavior.

## Statistical design

Default run:

- 30 independent seeds;
- severity = 0.55, 1.0, 1.5;
- all original attacks plus `adaptive_lie`;
- precision, recall, F1, PR-AUC, FPR and detection latency;
- paired Wilcoxon tests vs SBI-only;
- Holm correction;
- paired 95% CI for recall differences;
- rank-biserial effect size;
- descriptive non-inferiority check on standard SBI attacks with margin -3
  percentage points;
- stage-attribution analysis showing which fraction of detections comes from
  SBI, ACTIVE/KIE, and CONTEXT.

## Primary success criteria

The iteration is considered successful only if the data support both sides of
the hypothesis:

1. hidden-compromise scenarios show a positive and statistically supported
   recall gain from the auxiliary active channel;
2. `cascade_preserve_sbi` does not materially reduce recall for Rapid Reset,
   OAuth abuse, NRF poisoning, and cross-slice attacks relative to SBI-only.

The equal-FPR cascade is the fair-FPR comparison; the preserve-SBI cascade is the
engineering comparison that prioritizes not losing existing SBI coverage.

## Run

```bash
cd experiments/5g_kie_full
python run_cascade_experiment.py --seeds 30
python plot_cascade_results.py
```

Quick smoke run:

```bash
python run_cascade_experiment.py --quick
```

Outputs are written to `results_cascade/`.
