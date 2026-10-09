# Iteration 5 protocol — new cross-OS dual-capture validation of KIE/NRF

**Protocol committed before collecting new results, 2026-10-09.**

## Scope and independence claims

This is a **new prospective laboratory replication**. Six newly provisioned
GitHub-hosted VMs on **Ubuntu 24.04** (old series Ubuntu 22.04), each with
fresh independently installed Open5GS, MongoDB service, local KIE sidecar,
unique ephemeral HMAC secret, independent NRF traffic generator and six
re-randomized phase/profile pairs. The new seeds are:

| KIE period | New independent environment seeds |
|---|---|
| 1 s | 41011, 41013 |
| 2 s | 42011, 42013 |
| 4 s | 44011, 44013 |

Each runner executes the frozen, previously specified paired factorial:
short (30 GET) / standard (60 GET), six randomized
profile × signed-sampling-phase (.15, .50, .85) cells, both randomized
truthful/hidden conditions in each cell, 12 interventions/runner = **72
new interventions total**. The original fixed v1/v2 KIE code and original
1 CPU-tick NRF-response threshold / reference and post intervals remain
unchanged. There are **2 independent VM environments per cadence**, NOT
24 independent samples per cadence.

**Do not call this an independently operated third-party laboratory.**
Both witnesses tap the SAME loopback SBI traffic on the SAME host:
- primary capture: a separate long-running **tcpdump** process exporting PCAP;
- replication capture: a parallel **tshark** process exporting PCAPNG;
- independent analyst uses **Python standard-library csv+json**, NOT the
  original pandas-based event decoder, to decode the independently captured
  tshark HTTP2 fields;
- both PCAPs use the Wireshark/tshark **same HTTP/2 dissector** in a second
  stage. The packet capture processes and Python event parser are distinct,
  but not physically separate observation sites, dissection libraries,
  organizations or research groups.

This is stronger than another random seed on the old runner OS, but is NOT
sufficient to demonstrate independent laboratory replication, generalization
to real attackers or an operator-grade false positive rate.

## Frozen independent-event definition

For EACH capture, independently filter GET requests for `/nnrf-`
addressed to the NRF from the observed endpoint map, deduplicate via
TCP-stream + HTTP2-stream-ID, sort by exact packet epoch timestamps,
separate groups only at inter-GET gaps **>4s**, require >=20 GET in any
sliding **2s** window, and set event timestamp to the **first actual GET**.
Event discovery must not read labels, sidecar controls, phase intervals,
mask flags or collector ground truth.

Score all qualifying events using **unmodified** functions
`run_open5gs_concealment.evaluate_one_event` and
`run_open5gs_concealment_v2.score_event_v2`, with only event timestamp and
local signed report bytes. Require original in-run HMAC all-valid
receipt and one-to-one `(NF, sequence, ts)` collector pairing; exclude any
unverified trailing reports if present. Join randomized trial labels
and mask exposure only **after** the blind frozen scores exist.

Both capture-based witnesses and resulting v1/v2 decisions must be
reported separately. Observer disagreement, unknown KIE report coverage,
unexpected off-trial bursts and every missing packet witness remain
explicit scientific outcomes; do not silently repair trials using
orchestrator timestamps.

## Prespecified negative controls

Immediately after the 12 intervention trials, masks stay OFF while
*both captures remain running*. Each independent runner contributes:
1. 8-second **idle** negative control;
2. **10 benign NRF GETs** spaced 0.8s apart, with masks off;
3. a second independent 8-second idle;
4. a second 10-GET benign control.

Thus there are **24 negative-control windows**, including 12
low-rate 10-GET windows (120 intended benign HTTP2 GETs total).
The event witness threshold remains the locked >=20 GET/2s.
Low-rate delivery success and independently captured requests are
recorded; full-baseline controls remain archived separately.
Any >=20-GET/2s independent witness inside these windows is counted
as a **false external-event witness**, not a KIE false accusation.
Truthful *high-rate* paired intervention phases also provide controls
for false KIE accusations.

## Primary descriptive outcomes

- 72 new interventions delivered vs observed in tcpdump PCAP and second
  tshark PCAPNG, by cadence and truthful/hidden, short/standard profile;
- paired observer witness and onset agreement (max 100ms) and exact
  v1/v2 decision agreement between observers;
- v1/v2 alarms among KIE-evaluable independently witnessed *hidden*
  and *truthful* interventions; explicitly display mask-exposed subset;
- at least 24 negative-control windows with successful/captured GETs,
  >=20-GET/2s accidental witnesses, unknowns; no hypothesis of zero FPR
  outside these constrained windows;
- original live HMAC receipts, byte-count provenance and exclusions,
  exact OS, Open5GS, tcpdump and tshark versions;
- descriptive comparison only against the previous Ubuntu 22.04
  12-runner trial; do not pool all events as independent replicates.

**Scientific limitations:** synchronous HTTP2 GET workload,
host CPU tick telemetry is imperfect, controlled KIE masking is not a
real compromised network function, and GitHub-hosted runners are not a
cross-organization or cross-geography test. With 2 new VMs/cadence no
powered inferential statistics or blanket detection guarantees.

## Code and outputs

`run_open5gs_validation_controls.py`;
`validate_open5gs_dual_observer.py` (independent parser and evaluator);
`tests/test_open5gs_independent_validation.py`;
`.github/workflows/5g-open5gs-nrf-dual-observer-validation.yml`.
Archive raw tcpdump PCAP and second independent PCAPNG, two TSV
exports, 12 randomized trials, signed reports/collector truth, HMAC
validation receipt, four negative control windows, and per-environment
OS/package manifests. Aggregator emits `independent_validation_trials.csv`,
`independent_negative_controls.csv`,
`independent_runner_provenance.csv`,
`independent_validation_by_condition.csv`, and
`independent_validation_summary.json`.
