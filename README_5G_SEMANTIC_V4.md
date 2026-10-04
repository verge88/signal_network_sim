# 5G SBA semantic consistency v4

V4 is the next research iteration after the v3 detectability-grid experiment.
The v3 run showed that semantic detection saturated near 1.0 even for a single
hidden event, because honest observers independently saw a violation with very
high probability and TTD was effectively zero. V4 intentionally makes the
observation process harder and more realistic.

## What changed

### Partial observability

Evidence probability now depends on:

- semantic fact;
- trust domain;
- operational context;
- attacker sophistication.

For example, NRF is comparatively informative about token/discovery state,
whereas NWDAF is slower and less complete. Numerical probabilities are still
synthetic proof-of-concept parameters and must not be presented as measured
operator-network values.

### Three-state evidence

Each observation is one of:

```text
CONSISTENT
INCONSISTENT
UNKNOWN
```

`UNKNOWN` is not treated as a negative vote. This makes missing telemetry and
incomplete visibility explicit.

### Correlated evidence

Observers are no longer independent Bernoulli sensors. Each event/fact gets a
shared visibility multiplier and may suffer a common UNKNOWN episode. This is
intended to prevent artificial quorum gains from nearly identical evidence.

### Observation latency and online quorum

Each trust domain has a latency distribution. `FactObservation.observed_at`
records when evidence becomes available. The detector processes observations
in time order and records the instant the quorum is reached.

TTD is therefore now:

```text
first quorum alert time - first malicious-event time
```

rather than zero by construction.

### Sophistication

The runner supports:

```text
naive
statistical
adaptive
```

Adaptive attacks reduce semantic observability and use less obvious attack
variants. Compromise remains a capability rather than a class fingerprint.

### Separate false-positive channels

Benign evaluation reports:

- overall FPR;
- semantic FPR;
- transport FPR;
- false attributed event IDs;
- UNKNOWN evidence fraction.

Trust-domain evaluation includes both SUPPRESS and INJECT modes.

## Files

```text
5g/sba_semantic_v4.py
5g/run_semantic_v4.py
tests/test_sba_semantic_v4.py
```

V3 files are preserved for reproducibility.

## Pull the branch

```powershell
git fetch origin
git switch feat/5g-semantic-v4-online
git pull origin feat/5g-semantic-v4-online
```

## Tests

```powershell
python -m unittest tests.test_sba_semantic_v4 -v
```

Then the full regression suite:

```powershell
python -m unittest discover -s tests -v
```

## First recommended run

Do not start with `research`. First verify that v4 no longer saturates:

```powershell
python .\5g\run_semantic_v4.py `
    --profile dev `
    --seed 42 `
    --calib-windows 1200 `
    --eval-windows 300 `
    --hidden-grid 1 2 5 10 20 40 `
    --sophistication-grid naive statistical adaptive `
    --skip-trust `
    --out-dir runs\sba_semantic_v4\seed42_surface
```

Key output:

```text
summary.json
calibration_thresholds.csv
fpr_by_context.csv
evidence_diagnostics.csv
null_test.json
detectability_surface.csv
per_attack.csv
```

A scientifically interesting result is no longer `Recall=1`. We want a
meaningful surface such as:

```text
Recall = f(attack family, hidden volume, sophistication)
```

with adaptive attacks harder than naive attacks, non-zero online TTD, and a
benign UNKNOWN rate that reflects incomplete telemetry.

## Trust-domain run

After the first surface is sensible, remove `--skip-trust` or run a smaller
trust experiment:

```powershell
python .\5g\run_semantic_v4.py `
    --profile dev `
    --seed 42 `
    --calib-windows 1200 `
    --eval-windows 200 `
    --hidden-grid 1 5 20 `
    --sophistication adaptive `
    --trust-hidden 20 `
    --trust-eval-windows 100 `
    --out-dir runs\sba_semantic_v4\seed42_trust
```

This writes:

```text
trust_domain_suppression.csv
trust_domain_injection.csv
```

## Scientific limitation

V4 improves internal validity; it still does not validate absolute performance
for a real 5G Core. The visibility, latency, UNKNOWN, and correlation
parameters must later be calibrated on free5GC/Open5GS/operator traces.
