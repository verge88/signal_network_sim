# Pre-registered exploratory falsification: min leave-one-trust-origin-out score

Date: 2026-10-09. This protocol is committed **before** the first run.

## Hypothesis under test

The proposed score `S_rob(O) = min_{r in R(O)} s(O without all reports from r)`
can be robust to one source *only by sacrificing sensitivity* to attacks
whose sole detectable evidence comes from that source.

A transparent monotone proxy is used:
`s(O) = number of INCONSISTENT reports among five distinct origins for
one TOKEN fact on one procedure event`. This is **not** a trained one-class
anomaly score, so the run cannot validate a full Data Mining contribution.

**Predefined counterexample**: when all five reports are CONSISTENT,
changing one origin to INCONSISTENT raises `s` from 0 to 1 but leaves
`S_rob` at 0. A null world and the one-origin intervention are therefore
indistinguishable under min-LOO. This refutes any *unqualified* assertion
that min-LOO preserves sensitivity to all single-origin attacks.

## Fixed synthetic experiment

- Seeds 111,112,113,114,115.
- Two observable contexts: normal_indirect, diurnal_peak.
- 600 independent-of-evaluation benign calibration windows per seed/context;
  300 evaluation windows per seed/context.
- Choose the **first** event with at least three CONSISTENT reports among
  five distinct origins for TOKEN. This eligibility condition is used
  identically for calibration and evaluation and does not use attack labels.
- Paired counterfactuals: identical benign event/report world, with only
  0,1,2,3 distinct previously CONSISTENT reports changed to INCONSISTENT.
  Never change UNKNOWN; never fabricate new reports. One-origin intervention
  is within the one-compromised-origin model; two and three are stress tests
  outside that model.
- Score variants: raw count and min-LOO count; pooled and context-stratified
  empirical rank p-values `(1 + #calibration_scores >= test_score)/(n+1)`;
  nominal alert rule `p < 0.01`. Compare with q3-of-5 count threshold.
- Outcomes: per-context and pooled null alert rate, intervention detection
  rate by number of modified origins, exact structural counterexample count,
  number of eligible windows, and descriptive Wilson intervals.
- No parameter search, post-hoc retuning or claim of inferential p-values.

## Interpretational limits

The score is an intentionally minimal *handcrafted proxy*, not the full
proposed `s_theta`; it does not use temporal or transport features, and it
does not implement independent `E_independent` corroboration or the proposed
hybrid decision rule. It tests a structural failure mode and should be
reported as **exploratory falsification**, not confirmation of originality
or network-level security. Synthetic repeated windows are dependent and
conditional conformal coverage is not guaranteed. This experiment does not
measure the dissertation's required FPR <=0.1%.
