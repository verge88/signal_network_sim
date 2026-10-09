# Follow-up: retrospective Open5GS NRF KIE evidence-horizon audit (exploratory)

## Why this iteration
The completed four-environment 2x2 sampling/load stress run was assembled from
GitHub Actions #37820113809 and recovery #37908173468. The frozen v1/v2
detectors identified 7/12 vs 5/12 short masked NRF bursts, whereas both
identified all 12 standard masked NRF bursts; v2 raised one alarm in 12
truthful standard NRF bursts, compared to two for v1. All planned 96
interventions had independent external witnesses. Two of the twelve short
masked NRF events had no mask-exposed collector sample in the t-1..t+4 window.

The comparison is descriptive: only four independent GitHub Actions
environments, with one runner per condition, and dependent cycles within run.

## Research question and contract
Before interpreting short-burst misses, examine whether the post-event
report cutoff and actual signed-mask sample coverage jointly explain the
observed v1/v2 trade-off. This is *retrospective sensitivity analysis*, not a
new preregistered confirmatory trial or an additional 96 fresh interventions.

- Archived sidecar HMAC was checked with a runner-only key during live capture.
  Offline replay cannot reverify HMAC because the key was not archived.\n  Some collector files may contain appended post-validation reports; these are\n  counted separately and **excluded from every scored cutoff**. If removing\n  them changes the archived v1/v2 decisions, replay fails closed.
- Detector inputs: independently witnessed NRF event epoch plus signed report
  timestamps and reported CPU tick deltas, and the archived validity
  provenance. No ground-truth labels, fault flags, masking metadata, or phase
  IDs enter a scoring function.
- Freeze all model parameters: reference [t-8,t-2), post starts at t-1,
  report CPU rise >=1 tick, external burst >=10 GET/s (archived witnesses).
- Sweep *only* the stop cutoff at t+2, +3, +4, +5, +6, +8 seconds. The +3
  and +4 outputs MUST exactly reproduce archived v1 and v2 states/alarms for
  each NRF external witness or replay fails closed.
- AFTER all horizons are scored: attach true scenario/cycle and pair sidecar
  collector ground truth to signed payload by (nf, sequence, timestamp).
  Check that the experiment-only signed fault flag agrees with collector
  truth. A paired signed masked *sample* does not mean the signer is an honest
  or uncompromised NF; it establishes artifact provenance only.
- Report all witnessed events, unknown decisions, concealment alarms, mask
  exposure counts, exposed-only rates and per-independent-run details.
  Never silently delete unexposed/unknown events or claim nominal <=1% FPR.
- Treat exposed-only subgroups as selected *after* the event; they are
  descriptive diagnostics, not causal or independent estimates.
- Compare horizons on the exact same events, not as independent samples.
  The 4-run dataset and the grid were selected after earlier exploratory
  observations: do not report confirmatory p-values.

## Reproduction
GitHub Actions workflow `5g-open5gs-nrf-horizon-audit` downloads original
per-run artifacts from the two existing source runs, executes regression tests,
and generates:

- `nrf_horizon_events.csv` — horizon-specific event decision, first positive
  report timing and report-exposure metadata;
- `nrf_horizon_by_condition.csv` — one observational cell per horizon/profile,
  sampling cadence and background load;
- `nrf_horizon_pooled_exploratory.csv` — explicitly descriptive counts and
  conditional proportions across all four independent runners;
- `signed_report_provenance.csv` and `frozen_detector_crosschecks.csv`;
- `nrf_horizon_summary.json` — frozen grid, provenance and caveats.

Locally, unpack the original four archives as child directories named
`5g-open5gs-stress-<seed>` under `input/`, then:

```bash
python -m pytest -q experiments/5g_kie_full/tests/test_open5gs_nrf_horizon_replay.py
python experiments/5g_kie_full/replay_open5gs_nrf_horizons.py \
  --input-root input --output-dir output
```

## Next *prospective* study (not claimed by this replay)
Randomize **independent runner replicates per cell**, with mask onset phase
relative to KIE sampler randomized independently of truthful/hidden status,
and use at least 1s/2s/4s sampling versus short/standard external bursts.
Freeze a decision policy and a missing-coverage/unknown reporting rule before
collecting new traces; then separately estimate attack sensitivity, normal
false accusations, signer compromise vulnerability and event detection latency.
This tests actual sample aliasing instead of confusing an observed horizon
association with a causal effect. Underlying NF compromise is still *not*
implemented by the existing sidecar fault injection.

## Completed archival replay (9 October 2026)

[Successful workflow run #37919503648](https://github.com/verge88/signal_network_sim/actions/runs/37919503648);
artifact `5g-open5gs-nrf-horizon-audit`. All 10 regression tests passed and
all 48 independently witnessed NRF-event decisions reproduce the exact
archived v1 (t+3) and v2 (t+4) alarm/state values.

| Report deadline | Hidden alarms / known | Truthful alarms / known | Unknown hidden / truthful |
| --- | ---: | ---: | ---: |
| t+2 s | 18 / 22 | 2 / 18 | 2 / 6 |
| t+3 s | 19 / 24 | 2 / 24 | 0 / 0 |
| t+4 s | 17 / 24 | 1 / 24 | 0 / 0 |
| t+5 s | 16 / 24 | 0 / 24 | 0 / 0 |
| t+6 s | 15 / 24 | 0 / 24 | 0 / 0 |
| t+8 s | 14 / 24 | 0 / 24 | 0 / 0 |

At t+4 s, 22/24 hidden NRF events had a sample in the archived
HMAC-validated prefix that was contemporaneous with active report masking;
the other two had no masked sample in that window. Among 22 mask-exposed
events, 17 were detected (selected-exposure descriptive subset only).

The four archival sidecar files contained **34 appended reports** beyond the
12,876 covered by the stored live validation. Replay excludes those 34
records (10/16/2/6 for seeds 13007/14009/15013/16001, respectively);
this is a provenance discrepancy, not evidence of failed HMAC verification.
The truncated validated-prefix replay still exactly reproduces both legacy
detectors on every NRF witness.

These observations support a *trade-off within this laboratory fault-injection
dataset* between extending the NRF acknowledgment horizon and missing short
masked activity; they do not yet establish causality or generalize to a
compromised production NF. The absence of false accusations at t+5 is not
proof of an operational zero false-positive rate. The fact that t+5 appears
preferable on one metric is **not** a recommendation to retune a detector
using this same held-out dataset. A prospective randomized follow-up is
required.
