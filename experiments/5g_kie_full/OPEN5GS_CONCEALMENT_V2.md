# Open5GS KIE v2: sample-cadence-aware NRF acknowledgment

## Root cause in the previous run

The original independent trial #37793349076 had a single false concealment
accusation in truthful NRF burst, runner 6113 cycle 6. This was not proof
of a dishonest NF.

Relative to the initial external NRF burst event (t=1791472545.095432):
- first real NRF GET at t+0.204 seconds;
- signed KIE CPU delta 0 at t+1.175 seconds;
- signed KIE CPU delta 2 at t+3.177 seconds;
- intervention phase ended at t+5.476 seconds.

The original acknowledgement deadline was t+3.000. The first real
positive CPU sample fell 177 milliseconds outside the decision window.

## Frozen version 2 change

The NRF acknowledgment horizon is increased from t+3 to t+4 seconds,
as a heuristic derived from the two-second SBI bucket and the two-second
sidecar sampling period. All other scoring elements are unchanged:
external NRF GET witness >=10 requests/s, signed local CPU delta
increase >=1 tick versus [t-8,t-2), and the complete UDM lifecycle
corroboration rule.

Neither the scenario label, control-file mask, fault_injection field,
phase_hint, cycle identifier nor experimental ground truth is available
to the v2 event-scoring function. The original detector is retained
unchanged; both methods are scored on the same external event timestamps.

The extended window could include a sample after the controlled mask
has ended for some unusually short bursts. This is a real threat to
hidden-burst recall and will be measured, not assumed away.

## Independent prospective test

A new GitHub Actions workflow uses three unused randomization seeds
10021, 11047 and 12011. Each fresh Open5GS/Ubuntu runner repeats
20 minutes of baseline and six balanced randomized cycles of all four
scenarios (truthful/hidden UDM restart and truthful/hidden NRF burst).
There are 72 planned interventions over the three runners.

The runner saves real SBI PCAP, signed sidecar KIE, experiment-only
ground truth, original detector results, v2 results, and changed-alarm
attribution. The signature key is not uploaded, and reports are
HMAC-checked during the live workflow.

The original packet-timestamp phase audit remains in place to prevent
two-second bin centers from mislabeling events. Phase labels only enter
the evaluation after both models issue decisions.

Primary evaluation reports per-run and pooled:
- hidden UDM/NRF concealment-event recall;
- truthful UDM/NRF false-accusation rate;
- any changed event-level decision and timestamp;
- missing or unresolved external event witnesses;
- missing or invalid signed-report coverage;
- signed-report freshness/cadence.

A positive result means fewer false accusations without losing hidden
detection under the newly frozen protocol. The earlier run is used to
diagnose the failure and is not counted as new independent evidence.

## Scientific limits

This is a controlled fault-injection lab, not an adversarial compromise.
CPU ticks are an imperfect local proxy, not an exact HTTP/2 request count.
HMAC demonstrates integrity under the runner's key, not secure
hardware-backed attestation. All cycles within one runner are dependent.
Even zero false accusations out of 18 truthful events per NF does not
demonstrate an operational probability below 1 percent.
