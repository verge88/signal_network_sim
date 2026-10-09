# Prospective Open5GS NRF burst / KIE sampling-phase trial (v1 frozen)

**Protocol written before any prospective trial results.** This is a
six-independent-runner *prospective randomized feasibility study*. It is not a
powered efficacy or production security validation. The previous NRF cutoff
grid (Actions #37919503648) is exploratory only, and will not be used to
reselect a cutoff based on these new outcomes.

## Locked research question

Does the observed KIE sampling period (1, 2, 4 seconds) and the *assigned
time of a 5G NRF stimulus within the current signed-sample interval* affect
witness coverage and the ability of unchanged v1/v2 KIE to detect locally
masked reports for short versus standard independently observed SBI activity?

## Experimental unit and prospective assignments

Six **fresh Ubuntu/Open5GS/MongoDB independent runners**, two per cadence:

| KIE period | Fresh random seeds (runner clusters) |
|---|---|
| 1 s | 21011, 21013 |
| 2 s | 22011, 22013 |
| 4 s | 24011, 24013 |

Benign background NRF GET load fixed to zero in all cells (removes previous
cadence/background confound); load robustness was explored separately.

Within **each** runner: 180 s stable baseline (this is NOT a calibrated
operational FPR experiment), six cycles, one trial per combination of
burst profile (short/standard) x target phase fraction (0.15/0.50/0.85)
relative to the next observed signed NRF sample. In each cycle two
interventions with identical profile/phase assignment, one truthful and
one artificially hidden, in randomized order. Entire six-cycle grid
shuffled reproducibly from a fresh, preselected seed. Total:
6 runners x 6 cycles x 2 NRF interventions = **72 planned
interventions**, including 18 hidden short and 18 truthful short bursts.
Each cadence has 6 hidden-short interventions from two independent runners.

In a trial, apply the same 3s lead-in to truthful and masked conditions,
then wait for the next *real KIE signed NRF report timestamp*, sleep until
its timestamp + assigned fraction x KIE period, and perform the external
HTTP/2 GET stimulus. Samples are used for scheduling only; no fault
flags, signed report values, reported CPU deltas, truth or phase labels
are used by the schedule alignment logic.

Freeze stimulus profiles from the previous Open5GS stress study:
short=30 GET attempts with 0.25s post-hold; standard=60 GETs with
3.0s post-hold; both have inter-request 0.03s. All attempts must succeed.
12 s washout and 10 s stable recovery separate interventions; logs
capture all actual wall-clock timestamps and signed samples.

## Frozen detector and safety protocol

- Independent external NRF witness >=10 HTTP/2 GET/s from PCAP
  (2s buckets), unchanged from the earlier study.
- Unmodified v1 NRF: reference [t-8,t-2), reply [t-1,t+3],
  signed reported CPU delta increase >= 1 tick.
- Unmodified v2 NRF: identical rule with reply [t-1,t+4].
- No change to detectors, scores, thresholds or classifier parameters
  after trial outcomes; no α or FPR budget retuning.
- Event phase/cycle, assignment, mask state, signed phase_hint and
  fault_injection are **never scorer inputs**. Detector outputs are frozen
  first; experimental labels are attached afterward, using independent
  packet boundary timestamps.
- Live HMAC verification of all signed sidecar reports and independent
  collector sample identity after stopping collectors, using runner-local
  unarchived key. Invalid/missing signatures invalidate the runner.
- Realized offset and missed alignment deadline (target >250ms late or
  next sampler epoch passed) recorded; these remain in intention-to-treat
  coverage and are not silently filtered out.
- A missing/ambiguous independently witnessed stimulus is instrumentation
  non-coverage, not a false negative. Unknown report states likewise.

## Outcomes (preregistered descriptive pilot)

**Primary descriptive measures:** per-cadence short hidden NRF burst event
detection count for v1 and v2, with denominator known and missing external
witnesses separately reported; assigned phase stratum and realized
phase-compliance. **Secondary:** truthful-short false accusations,
short-versus-standard contrasts, report mask-exposure coverage in
[t-1,t+4], v1/v2 discordant event counts, signed HMAC coverage,
detection latency when available, protocol adherence.

All main tables present 6 independent runner clusters or phase cells
and avoid invalid independent-event p-values. With 2 independent runners
per cadence, *no significance, noninferiority or population generalization
is claimed*. Reporting a zero false-accusation count out of a few events
never establishes an operational probability below 1%.

## Provenance / ethics / threats to validity

This is a controlled lab fault-injection simulator for misreporting.
No NF is compromised; the sidecar itself masks telemetry and signs
the result with the legitimately held key, i.e. **HMAC does not prove
truthfulness**. CPU ticks are a load proxy. Sidecar KIE samples and
NRF GET timestamps share the host clock, yielding a reliable laboratory
alignment but not independent real-world clock synchronization. Results
depend on Open5GS package and CI runner; archive versions and PCAP.
The shortened baseline and washout are exploratory feasibility choices
and prevent direct comparability to previous 20-min baseline runs.

The prospective protocol is locked into the initial Git commit containing
this workflow; if CI setup fails, distinguish environment repair from
methodological changes and do not conceal exclusions or retrospectively
alter the predefined outcome family.

## Reproduction

Workflow `5g-open5gs-nrf-phase-prospective.yml` provisions Open5GS,
runs all six new independent environments, uploads six original artifacts,
and creates `5g-open5gs-phase-pooled` containing
`trial_intervention_coverage.csv`, `trial_events.csv`,
`descriptive_cells.csv`, `runner_level_metrics.csv`,
`runner_signature_provenance.csv` and `trial_summary.json`.

Core scripts:
`run_open5gs_phase_trial.py`,
`verify_live_kie_phase_trial.py`,
`aggregate_open5gs_phase_trial.py`;
tests: `tests/test_open5gs_phase_trial.py`.
