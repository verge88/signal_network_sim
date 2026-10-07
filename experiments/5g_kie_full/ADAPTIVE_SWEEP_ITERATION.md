# Adaptive-evasion / FPR-budget sweep

This is the iteration after the 30-seed guarded-cascade experiment.

## Question

The previous iteration showed that `cascade_preserve_sbi` preserves SBI attack
coverage while adding a large ACTIVE/KIE gain for hidden compromise, at the cost
of a small increase in benign false positives.

This iteration asks two boundary questions:

1. how much malicious behavior can a compromised NF mirror in its self-report
   before ACTIVE/KIE loses useful detection power?
2. how much benign FPR budget should be allocated to the ACTIVE/KIE stage?

## Parametric adaptive attacker

For the malicious SBI increment

```text
delta = observed_malicious - observed_clean
```

the report is generated as

```text
report = observed_clean + alpha * delta
```

with

```text
alpha = 0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0
```

`alpha=0` is the previous lying endpoint and `alpha=1` is the truthful
mirroring endpoint. The old `adaptive_lie` behavior remains the default
`alpha=0.75` for backward compatibility.

## ACTIVE/KIE FPR sweep

The SBI baseline remains fixed at a nominal 1% target FPR. Context remains fixed
at 0.05% auxiliary budget. ACTIVE/KIE is swept over:

```text
0.01%, 0.025%, 0.05%, 0.10%, 0.20%, 0.50%
```

The resulting classifiers are named `cascade_active_<budget>`.

## Statistical design

Default run:

- 30 independent seeds;
- severity = 0.55, 1.0, 1.5;
- 8 report-mimicry alpha values;
- 6 ACTIVE/KIE FPR budgets;
- SBI-only, legacy FULL and `fusion_maxq` retained as references.

Outputs include per-seed metrics, grouped summary, paired Wilcoxon tests against
SBI-only with Holm correction, 95% confidence intervals, rank-biserial effect
size, stage attribution, Pareto points, and an estimated critical report alpha.

A cascade point is called practically useful only when all of these hold:

```text
Holm-adjusted p < 0.05
95% CI lower bound for Recall gain > 0
mean Recall gain >= 0.05
```

The reported `critical_report_alpha` is the largest tested alpha satisfying
those conditions. It is a grid-based experimental boundary, not a continuous
mathematical threshold.

## Run

```bash
cd experiments/5g_kie_full
python run_adaptive_sweep.py --seeds 30
python plot_adaptive_sweep.py
```

Quick smoke run:

```bash
python run_adaptive_sweep.py --quick
```

Results are written to `results_adaptive_sweep/`.

The previous `results_fusion/` and `results_cascade/` methodologies remain
separate; this iteration does not overwrite their output definitions.
