# Pre-run protocol: v9 SBA observational-provenance correction

Date frozen: 2026-10-09. Branch: `experiment/sba-v9-provenance-corrected-20261009`.

## Purpose and scope

Reassess the **existing** v8/v9 q3-of-5 semantic quorum and 3-of-4
Byzantine-resistant transport detector against a static comparator after
correcting a flaw in the **synthetic operational-state generator**. This is
not an experiment on the proposed new leave-one-trust-origin-out / contextual
conformal Data Mining detector. It does not establish that detector's novelty.

The former `_manifest_transient` could change `UNKNOWN` into
`INCONSISTENT`. The former recovery path could generate a `CONSISTENT`
report for any pre-existing inconsistent observation on the event/fact,
regardless of whether the operational process changed it.

**Corrected generative contract**:
- perturb only existing `CONSISTENT` observations;
- preserve all `UNKNOWN` observations;
- report a manifested transient only when at least one observation changed;
- append recovery observations only for the exact observations changed by
  this transient, never for pre-existing inconsistencies;
- keep detector code and evaluation hypotheses unchanged.

## Frozen experiment

- Seeds: **62 through 71 inclusive** (ten paired simulation worlds).
  Earlier v8/v9 exploratory seeds were 42–51 and earlier pre-fix v9
  holdout seeds were 52–61; none of those are reused.
- `calib-windows=1200`; `eval-windows=200`;
  `operational-windows=200`; `state-attack-windows=100`;
  `hidden=5`.
- Calibration target `target-fpr=0.01` (1%, **not** the dissertation
  target 0.001 = 0.1%).
- Primary family fixed as in v9: H1 mean operational false-positive rate
  improvement, H2 ordinary attack-window recall non-inferiority (1 pp),
  H3 attack-in-state window recall non-inferiority (1 pp), H4
  attack-in-state event detection latency cost.
- Paired bootstrap 5000, paired sign-flip 10000, Holm correction over the
  four primary endpoints, nominal alpha 0.05.
- Same world is evaluated by both detector arms; **this is not**
  an AttackEffect=0 paired null-world experiment.
- All operational rates, latencies, attacks, and observations are synthetic.
  The output must not be described as production-5GC performance.
- No re-tuning of detector, endpoints, or seeds based on the result.
- Existing v9 results are designated **pre-fix exploratory** rather than
  publication-ready confirmatory evidence.

## Preconditions and audit

The workflow must pass the provenance regression tests and existing
v8/v9 unit tests before executing the experiment. The regression tests use
independent test-only seeds 901–903, never the held-out seeds.

Artifacts include the analysis-plan SHA256, per-seed raw metrics,
paired primary and secondary statistics, summary JSON, and execution log.
A green workflow alone is not proof of scientific validity; compare with
pre-fix results and audit whether changed event populations explain deltas.

## Known remaining limits

- Synthetic origin independence and trust assumptions.
- `generate_attack_during_state` uses a specific marker-at-attack
  construction, not a fully realistic simultaneous benign transient.
- Conditional/sequence-wise false alarm control and the thesis FPR
  requirement of 0.1% remain untested.
- No real operator 5GC traffic is used.
