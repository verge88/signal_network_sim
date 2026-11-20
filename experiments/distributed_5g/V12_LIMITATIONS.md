# V12: reproducibility, valid scope and out-of-distribution limits

## What the frozen method does
`ConfirmedStaircaseDetector` is a **causal signature matcher** tuned to the
synthetic S9 workload. Given 32 known NF identities and a fixed eight-NF
topology grouping, it waits for two distinct low-count plateaus stable for
three windows, then requires a simultaneous `+1` in the fourth window. It
tracks the exact 30-window staircase. There is **no learned generalized
attack boundary**, no multi-feature evidence and no claim to detect all 5G
attacks. The baseline response is dependent on S9 waveform, 3-window cadence,
and the fixed 30-window episode duration.

## Independently confirmed *in-distribution* S9 result

- Frozen V12 commit: `ddf57c2648cbf6fedbf456159183a4c7754ece9e`.
- Prospective workflow: https://github.com/verge88/signal_network_sim/actions/runs/44777939911
- Six fresh seeds: 91101–91106; six complete 700×32 S9 captures.
- Pooled positive NF-window recall **5184/5760 = 90.000%**; pooled
  empirical FPR **0/128640 = 0%**, with 0 false positives on each fresh seed.
- All four preregistered S9 gates passed. The method *misses the first three
  attack windows by design*. **Exactly 90% leaves no robustness margin.**
- The empirical count `FP=0` does *not* imply a universal true FPR of zero.

## Observed non-S9 diagnostic: severe generalization failure

This is a **posthoc audit**, not a fresh prospective sample. V12 and V11
were replayed **unchanged** from frozen code on historical 2026-11-13 V5
Open5GS acquisitions. Only test feature counts entered inference; original
labels were read after all streaming decisions.

- Seeds 90401 / 90402 / 90405 (original GitHub artifact IDs
  14049128601 / 14049887852 / 14050881142).
- Each has 3600 windows × 32 NFs = 115200 NF-windows, containing S1–S5,
  C/D/F/G/R attack labels and normal N1 traffic.
- Reproducible CI and three archived reports:
  https://github.com/verge88/signal_network_sim/actions/runs/44793725310
- Pooled V12: **496 TP, 1994 FN, 1628 FP, 341482 TN**.
- Positive recall **19.9197%**; normal-window FPR **0.47448%**;
  precision **23.3522%**.
- Pooled V11 (paired): **22.0080% recall**, **0.86415% FPR**.

| Scenario | V12 attacked windows detected | NF-window recall |
|---|---:|---:|
| S1 | 4/120 | 3.33% |
| S2 | 2/120 | 1.67% |
| S3 | 2/120 | 1.67% |
| S4 | 0/120 | 0% |
| S5 | 486/1080 | 45.0% |
| C | 0/120 | 0% |
| D | 2/120 | 1.67% |
| F | 0/120 | 0% |
| G | 0/450 | 0% |
| R | 0/120 | 0% |

The V12 OOD FPR **exceeds 0.1% on all three seeds** (0.4713%, 0.4652%,
0.4870%). The 90% S9 result is *not* generalizable to these attack regimes.

## Counterfactual controls

Five added tests in `test_v12_ood_controls.py` are deterministic research
controls rather than measured production traffic:

1. The exact S9 signature yields 27 alarms per attacked NF in a 30-window event.
2. A **benign** observation with precisely the same counts is
   observationally indistinguishable and produces the same alarms.
3. A single targeted NF cannot satisfy a two-NF confirmation and goes undetected.
4. A shift to higher plateau baselines cannot satisfy the fixed [2,4] gate.
5. A changed waveform breaks the exact 27-window expected detection.

These distinguish detection-rule limitations from the empirical findings above.

## Review disposition and recommended successor

**Experimental S9-only success: yes. General or production-grade 5G intrusion
detector: no.** The existing stacked V10/V11/V12 PRs remain draft and should
not be merged into main as general detectors. Before any broader integration,
design a **new separately preregistered multi-scenario method** using
multi-feature evidence, calibrate only on disjoint development sets, freeze it
before new scenario/seed tests, and require per-scenario recall, pooled
false-positive control and credible uncertainty reporting.

The V12 code and S9 frozen results were **not changed** after the new OOD
evidence was observed.
