# Open5GS: event-conditioned concealment validation

## Research question

The earlier real Open5GS semantic-KIE cascade detected both truthful and masked
perturbations with high recall. That was *event detection*, not necessarily
detection of a false signed report.

This follow-up uses **independent external event witnesses** and asks whether
the *report acknowledges the same event* within a bounded delay.

## Two external witnesses

1. **UDM lifecycle**: parse \`open5gs-nrfd.service\` journal messages
   \`[UDM] NFInstance associated [nfInstanceId]\`. The evidence originates from
   NRF, not the UDM process/UDM sidecar. A unique UDM instance ID is included in
   each observed event.
2. **NRF GET burst**: select events from independently captured and dissected
   NRF HTTP/2 SBI request rates (\`nrf_query_rate >= 10/s\`) and group nearby hot
   two-second windows. The threshold is frozen before the new run.

The detector does not select events using phases, cycle identifiers, masking
flags or sidecar ground truth.

## Signed local acknowledgment

For an NRF-observed UDM registration at time \`t\`, compare valid signed local
UDM reports in \`[t-4s,t)\` and \`[t,t+6s]\`. A change in PID, invocation ID or
monotonic restart generation acknowledges the observed lifecycle event. If both
report windows are adequately covered and no such change exists, flag an
**unacknowledged external UDM event**.

For a burst witnessed at NRF (first hot SBI bucket at time \`t\`), compare
signed NRF CPU tick deltas in \`[t-8s,t-2s)\` versus \`[t-1s,t+3s]\`. If the
maximum post-event tick delta fails to exceed the pre-event median by at least
one tick, flag an **unacknowledged query burst**.

These are operational heuristics, not cryptographic proofs of lying. A genuine
positive event without a matching report is evidence of a *consistency
violation*. Severe overload, missing telemetry or unexpected scheduling can
create a benign consistency violation; such events must be measured on truthful
controls.

## Decision semantics

- Independent event absent => no concealment decision, not a negative example.
- Both signed report windows present + HMAC valid + acknowledgment seen =>
  \`acknowledged\`.
- Both signed report windows present + HMAC valid + no acknowledgment =>
  \`unacknowledged\` (concealment alarm).
- Insufficient report coverage => \`unknown\`; not counted as a confirmed
  concealment alarm.
- Invalid HMAC => **integrity alarm**, not a concealment alarm.

The HMAC key remains inside the workflow runner. HMAC authenticates bytes with
the runner's shared key but does not prove honest reporting or secure hardware
attestation.

## Ground-truth separation

The sidecar report contains \`phase_hint\` and \`fault_injection\` metadata only
for experiment administration. Neither field is read by
\`run_open5gs_concealment.py\`. Ground truth is never read by this program.

Only *after* the detector has scored independently witnessed events do phase
intervals assign \`truthful_* / hidden_*\` labels for evaluation.

## Run and outputs

The new \`5g-open5gs-concealment-evidence.yml\` workflow reproduces the live
Open5GS baseline and four cycles per perturbation using a separate output
folder. In addition to PCAP and signed KIE reports, it saves an NRF-only
systemd journal in JSON Lines with epoch-microsecond timestamps.

New outputs:

- \`nrf_registry_journal.jsonl\`
- \`concealment_events.csv\`: witnessed events, instance IDs, report coverage
  and acknowledgment states.
- \`concealment_phase_metrics.csv\`: hidden-event recall versus truthful-event
  false accusation rate.
- \`concealment_cycle_metrics.csv\`: cycle-level detection and missing data.
- \`concealment_summary.json\`: frozen event thresholds and limitations.

A failure to detect concealment is a legitimate scientific result, not an
infrastructure failure.

## Limits

- Four truth and four hidden cycles per event type produce broad confidence
  intervals; replicate on independent runs before a dissertation claim.
- The NRF server journal shares the host with UDM (stronger than a UDM
  self-report, weaker than a physically independent tap).
- The testbed presently has one target UDM; with multiple UDM replicas,
  instance-to-replica binding requires additional explicit identity metadata.
- An NRF association log does not by itself prove all UDM functional state is
  healthy.
- No claim that NRF actively polls NF is made; standard NF registration and
  heartbeat remain NF-to-NRF.
- A rate of 0 false accusations in eight controls does not imply a 1% FPR.
