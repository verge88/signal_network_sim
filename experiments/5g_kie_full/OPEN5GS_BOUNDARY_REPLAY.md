# Open5GS boundary-label replay and prospective validation

## Cause of the seven previously unlabeled NRF events

In the three completed Open5GS replica artifacts (workflow run 37767915524),
NRF query-burst events were selected using 2-second HTTP/2 feature windows.
The detector marked event time as the center of the first hot window, while
the orchestration recorded the intervention's actual start/end times.
Seven valid NRF events fell outside their matching intervals by this point
timestamp rule. This is a **label/clock-bin boundary error**, not a change
in the detector's alarm decisions.

A separate, after-the-fact audit now loads the real tshark HTTP/2 field export
and identifies every NRF GET packet in each triggering 2-second window.
After an event has already been scored, it is associated with an experimental
phase only when at least five packet timestamps exist and >=80% of those
requests fall into exactly one recorded phase. Events that cannot be uniquely
associated remain unresolved and are not counted as negative controls.

In the original artifacts, all seven previously unlabeled events had **100%**
of the actual HTTP/2 GET packets in a unique experimental NRF burst phase.
Three events belonged to hidden bursts, four to truthful bursts. All three
original alarms occurred in the hidden group. No detector scores or signatures
were modified.

## Replayed findings, three independent original environments

GitHub Actions replay: 37792940986; source experiment: 37767915524.

- 72/72 planned interventions have one external event witness;
- hidden UDM: 18/18 alarms;
- truthful UDM: 0/18 accusations;
- hidden NRF: 18/18 alarms;
- truthful NRF: 0/18 accusations;
- remaining unresolved events: zero;
- all 7 event-boundary mismatches resolved by real packet timestamps.

These are **exploratory post-hoc evaluation-label corrections**, not a new
prospective replication. The ephemeral HMAC key was not saved, so the replay
did not re-verify signed reports. Live HMAC verification in the original
GitHub Actions run had succeeded; that provenance is explicitly recorded.

The Wilson 95% descriptive interval for a proportion of 18/18 is about
[82.4%, 100%]; for 0/18 the upper limit is about 17.6%. These intervals
ignore within-run dependence, so they cannot be interpreted as a formal
guarantee that true operational FPR is below 1%.

## Separate confirmatory protocol, frozen before collecting new data

The next GitHub Actions workflow uses three **new** randomization seeds:
6113, 7247, 8309 (different from 1729, 2707, 4051).

Each new runner installs fresh real Open5GS/MongoDB, captures 20 minutes
of stable baseline, and performs six balanced randomized cycles (one each
truthful/hidden UDM restart and truthful/hidden NRF HTTP/2 GET burst per
cycle), 25 seconds washout and 20 seconds stable recovery.

The detector itself is unchanged:
- UDM: NRF-side NFInstance association event, compared with a signed UDM
  PID/InvocationID/restart-generation acknowledgment.
- NRF: >=10/s observed NRF GETs, compared with independently collected
  signed NRF CPU-tick change.
- The report-side HMAC secret stays inside the live runner.
- Missing or invalid signed reports are not automatically deemed concealment.

After frozen decisions are written, the packet-timestamp audit assigns
evaluation phase labels; it does not use phase, cycle, masking metadata
or ground truth to score a detector.

Per run: original detector CSV/JSON, raw PCAP and KIE, audited events,
intervention coverage, and metrics. After all three runs: pooled per-run
and cross-run event-level metrics plus unresolved and background alerts.

The prespecified scientific outcome is *not necessarily* a success. Any
truthful event with an alarm, missing external evidence, unresolved boundary
event or unexpectedly high background alarm rate must remain visible.

## Interpretive limitations

- Only three independent runners: many events within each runner are
  correlated; no claim of <=1% false accusation probability is justified.
- The packet witness of NRF bursts is real SBI HTTP/2 traffic, but NRF
  local CPU ticks are an indirect acknowledgment metric.
- The UDM instance is observed from NRF logs on the same VM and crosschecked
  with local UDM sidecar identity: not hardware attestation.
- Synthetic user-plane load, adversarial compromise of the signing secret,
  and a multi-instance UDM deployment are not covered.
