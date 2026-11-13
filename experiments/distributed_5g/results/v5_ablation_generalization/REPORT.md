# V5 attribution and structural-generalization results

This is a **post-final diagnostic** defined after the original final V5 result was known. The frozen V5 algorithm and thresholds were not changed. The source matrix run is [43495896034](https://github.com/verge88/signal_network_sim/actions/runs/43495896034), commit `b6ea333105c79e308a75429570e1d4e7ea96d225`.

Eight complete final acquisitions were replayed: 90401, 90402, 90405, 90406, 90407, 90408, 90409, 90410. Seeds 90403 and 90404 are absent from this raw-evidence diagnostic because their technical-retry workflows retained only summary artifacts, not full-evidence archives.

## 1. Learned model versus explicit group mechanisms

| System | Recall | FPR | FPR <= 0.001 | Precision | S5 episodes |
|---|---:|---:|---:|---:|---:|
| BUDGETED_WITHOUT_MINING | 0.362801 | 0.000128 | 8/8 | 0.953682 | 0/32 |
| LEARNED_PROBABILITY_ONLY | 0.575602 | 0.000997 | 3/8 | 0.807351 | 28/32 |
| V4_CORE_ONLY | 0.607229 | 0.000673 | 8/8 | 0.867470 | 31/32 |
| RESCUE_ONLY | 0.377259 | 0.000128 | 8/8 | 0.955378 | 8/32 |
| FULL_V5 | 0.612952 | 0.000673 | 8/8 | 0.868545 | 32/32 |

The learned logistic probability contains substantial signal: relative to the non-mining baseline its mean seed-level recall gain is +0.2128 (bootstrap 95% CI 0.1428–0.2667; one-sided paired Wilcoxon p=0.0078125). It is not, however, an operational replacement for the full core under the fixed FPR requirement: the fused learned-only system meets FPR <= 0.001 on only 3/8 runs.

The frozen V4 core gives a mean recall gain of +0.2444 versus baseline (95% CI 0.2327–0.2595; p=0.00390625) and meets the FPR target on all 8/8 runs. Compared directly with learned probability alone, its mean recall delta is +0.0316, but this difference is heterogeneous across seeds (95% CI -0.0271–0.1134; p=0.4375). Its clearest contribution is operational calibration: false positives fall from 912 to 616 while the per-run FPR requirement improves from 3/8 to 8/8.

The explicit V5 triple rescue is smaller but measurable. FULL_V5 adds 38 true positives and zero false positives over V4_CORE_ONLY, raising pooled recall by 0.00572. The seed-level increment is positive on 7/8 runs and zero on one (95% CI 0.00361–0.00783; one-sided Wilcoxon p=0.0078125), and S5 episode detection improves from 31/32 to 32/32.

A two-block Shapley attribution at system-recall level assigns a mean +0.2401 recall to the V4 core (95% CI 0.2299–0.2532) and +0.0101 to the explicit rescue (95% CI 0.00648–0.01340), approximately 96.0% and 4.0% of the total +0.2502 gain respectively. This decomposition must not be described as “learned versus non-learned”: V4 itself combines learned logistic probabilities with fixed structural scoring and temporal accumulation. The separate LEARNED_PROBABILITY_ONLY row exists precisely to expose that distinction.

**Reviewer implication:** the previous wording that could attribute the whole gain to Data Mining is not supported. The defensible claim is that the learned residual-temporal model provides substantial signal, while fixed structural/temporal mechanisms are required for FPR control and the explicit V5 rescue contributes a small targeted increment.

## 2. Structural generalization of frozen V5

| Variant | FULL_V5 S5 episodes | Episode recall | NF-window recall | TG_RESCUE episodes |
|---|---:|---:|---:|---:|
| Original contiguous k=3 | 32/32 | 1.000 | 0.5767 | 8/32 |
| Contiguous left | 32/32 | 1.000 | 0.5767 | 8/32 |
| Contiguous middle | 32/32 | 1.000 | 0.5767 | 8/32 |
| Contiguous right | 32/32 | 1.000 | 0.5767 | 8/32 |
| Within-zone ID permutation, triple made non-contiguous | 31/32 | 0.9688 | 0.5635 | 0/32 |
| Group k=2 | 13/32 | 0.4063 | 0.0490 | 0/32 |
| Group k=3 | 32/32 | 1.000 | 0.5767 | 8/32 |
| Group k=4 | 11/32 | 0.3438 | 0.0352 | 0/32 |
| Group k=5 | 0/32 | 0.000 | 0.0000 | 0/32 |
| Spaced non-contiguous triple | 20/32 | 0.6250 | 0.1729 | 0/32 |

Changing the location of a still-contiguous triple inside an eight-NF zone has no effect: left, middle and right placements reproduce the original 32/32 result.

The explicit rescue itself is **not permutation invariant**. A within-zone bijection that leaves all measured values unchanged but makes the original triple non-contiguous reduces TG_RESCUE from 8/32 detected episodes to 0/32. The full V5 system remains at 31/32 because the V4 core is insensitive to this relabeling, but the V5 rescue contribution disappears.

Group-size generalization is poor. Frozen FULL_V5 detects 13/32 k=2 episodes, 11/32 k=4 episodes and 0/32 k=5 episodes. A spaced three-node group is intermediate at 20/32. The learned probability branch contains information that the fixed structural gate suppresses: it detects 29/32 k=4 and 28/32 k=5 episodes in these counterfactuals, while frozen TG_V4/FULL_V5 detect only 11/32 and 0/32 respectively. This points to the structural gate, rather than absence of learned signal, as the main generalization bottleneck for larger coordinated groups.

**Reviewer implication:** this experiment does not justify calling V5 a generic group-behavior detector. It supports the narrower claim of a contiguous three-NF rescue motif. The second reviewer concern is therefore experimentally answered but also confirmed as a real limitation. Eliminating that limitation would require a new size-adaptive, permutation-invariant detector followed by a new prospective confirmation; changing V5 after seeing these results would be post-hoc tuning.

## Statistical status and limitations

The independent unit is the acquisition seed. Paired Wilcoxon tests above use the eight complete archived final runs. Structural variants are controlled replays/counterfactual interventions on those measured streams, not new independent Open5GS acquisitions. They are appropriate for identifying dependence on identifiers, placement and group cardinality, but they are not a replacement for a fresh prospective confirmation of a redesigned detector.

The original 10-seed final V5 confirmation remains unchanged. This diagnostic narrows attribution and the scope of the structural claim; it does not overwrite the original preregistered final result.
