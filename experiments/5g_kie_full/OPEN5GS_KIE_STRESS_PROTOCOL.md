# KIE-v2 Open5GS sampling cadence / load / short-burst stress test

## Research question and frozen choices

After a successful full 3-run Open5GS v2 test (GitHub Actions #37810742589),
both v1 and v2 showed 18/18 NRF hidden-burst recall and 0/18 false
accusations on truthful NRF bursts. That test did not meaningfully
differentiate v1 from v2.

This stress test is **not** a retrospective tweak to those results.
It tests whether the v2 NRF acknowledgment horizon t+4 s, compared
with v1 t+3 s, remains specific without losing sensitivity when sidecar
sample frequency, background SBI workload and masked intervention
duration vary.

All detector thresholds are fixed before new data collection:
- NRF passive witness >=10 GETs/second, as before;
- v1 CPU confirmation within [t-1,t+3];
- v2 CPU confirmation within [t-1,t+4];
- signed NRF CPU delta >= pre-event median + 1 tick;
- independent UDM NRF registry witness and PID/generation confirmation unchanged.

Labels, sidecar fault markers, plan, stress profile and true state are
**excluded** from both detectors. Actual HTTP/2 timestamps are used for
phase assignment only after frozen decisions have been emitted.

## Factorial design

Four independent Ubuntu/Open5GS/MongoDB GitHub Actions runners
run a 2x2 design:

| Seed | KIE sampling period | Benign background NRF HTTP2 GET traffic |
| --- | --- | --- |
| 13007 | 1.0 s | 0 requests/s |
| 14009 | 1.0 s | 2 requests/s |
| 15013 | 4.0 s | 0 requests/s |
| 16001 | 4.0 s | 2 requests/s |

2 req/s is deliberately lower than the 10 req/s burst witness threshold,
but sustained genuine traffic can affect CPU-tick reference levels.
Every background request and result is logged, so an unrealized workload
cannot be mistaken for delivered traffic.

Within *each* runner: 20 minutes baseline, six independent randomized
cycles containing all four scenarios (truthful/hidden UDM restart,
truthful/hidden NRF burst) once per cycle. Three cycles use a short
burst and three a standard burst, randomized and balanced per runner.
Truthful and hidden bursts in a cycle use the *same* profile.

Short burst: 30 HTTP/2 GET attempts, 0.25s post-hold. Standard burst:
60 GET attempts, 3.0s post-hold. Both have 3s sidecar-mask lead-in,
25s washout, and 20s stable recovery after every intervention. All curl
requests must succeed for stimulus execution to be considered successful.

Total planned experimental interventions: 4 runners x 6 cycles x 4
scenario types = **96**, of which 48 are UDM restarts, 48 NRF bursts.
Among the NRF bursts, 24 use each profile; among each profile,
12 truthful and 12 hidden bursts are planned.

## Analysis preregistration

Primary outcomes:
1. Planned vs externally witnessed event coverage, by condition/profile;
2. v1 and v2 hidden-event recall for NRF bursts, with unknown event counts;
3. v1 and v2 false-accusation rates on truthful NRF bursts;
4. Paired changed alarms on the same witnessed event;
5. Sidecar availability, HMAC validity, invalid / missing reports;
6. Whether hidden phases contained any *actual signed masked reports*
   within [external_event-1 s, external_event+4 s].

For slow 4s KIE sampling and short masked intervals, some interventions
may have no masked sidecar sample in the external event window.
These must be reported as insufficient concealment exposure, **not
scored silently as detection failures or successes** without qualification.
A missed HTTP2-burst witness must not be counted as a detector true negative.
Unknown KIE states are distinct from an alert or a clean event.

Actual GET rate and delivered background traffic are independently recorded.
If short bursts cannot reach 10 RPS and external witnesses are missed, this
is important stress evidence, not an excuse for changing thresholds.
Absolute statistical claims of a <=1% operational false-accusation rate
remain unjustified by this small number of correlated events.

## Technical and interpretation limitations

Four independent runners produce one independent environment each. Varying
multiple factors across only four environments cannot support strong
population-level causal claims. There is no actual compromised NF:
truth distortion is injected by a sidecar which signs masked payloads with
its legitimate key. A valid signature authenticates payload bytes under
that key, not their truthfulness. NRF CPU ticks are an indirect activity
proxy, not a hardware-attested request counter.

PCAP, original journal, paired v1/v2 event decisions, ground truth used
only for evaluation, workload manifest, HMAC-validation result, and
raw background request CSV are archived per runner. The HMAC key stays
inside the corresponding GitHub Actions runner and is not uploaded.
