# Iteration 4 — frozen KIE v1/v2 at independently recovered NRF packet onset

Preregistered analytic protocol **before first workflow execution**.
Sources: original [phase trial run 37920594864](https://github.com/verge88/signal_network_sim/actions/runs/37920594864), independent replication and PCAP audit [run 37926792064](https://github.com/verge88/signal_network_sim/actions/runs/37926792064).

## Why perform this analysis?

The original 12 independently provisioned Open5GS runners produced 144 delivered interventions, of which 15 short bursts were incorrectly lost by the *independent fixed 2s-network-event witness*, although each had 30 independent NRF HTTP/2 GET requests in PCAP. Replacing fixed with same-threshold sliding 2s windows recovers the **event witness** in all 15; it does not itself improve KIE sensitivity.

This iteration prospectively specifies a **retrospective, secondary analysis** of those immutable 12-runner artifacts. No newly generated attack data, no unblinding-based detector tuning, and no p-values. Compare precisely when the frozen KIE decision rules do and do not acknowledge a physical NRF burst, including all 15 lost events and potential report sampling gaps.

## Packet-blind event construction, performed first

1. Read all captured HTTP/2 events plus independently recorded NRF endpoint mapping; filter physical GET requests to paths containing `/nnrf-` and NRF destination, and deduplicate requests by TCP stream + HTTP/2 stream identity. No intervention schedule or truth is read at this stage.
2. Sort packets by their own epoch timestamps; split candidate groups when successive request timestamps differ by **more than 4.0 seconds**. A group is eligible only when at least **20 GET requests fall in some sliding 2.0-second interval**. It is rejected (not relabeled) otherwise. The precise **timestamp of the FIRST GET** in each eligible group is the new independent event timestamp. These are fixed analytic choices, not thresholds selected from detector performance.
3. Pass only `{ts, nf="nrf", external_event="nrf_query_burst", nf_instance_id=None}` and the pre-existing signed report content to the actual frozen **`run_open5gs_concealment.evaluate_one_event`** and **`run_open5gs_concealment_v2.score_event_v2`**. The CPU acknowledgment rule remains **1 tick**: v1 pre [t−8,t−2), post [t−1,t+3]; v2 same pre and post [t−1,t+4]. No change to the KIE scoring code.
4. **Only after the two blind scores exist**, associate each packet event onset timestamp with exactly one trial interval and attach the original randomized trial profile, truthfulness vs hidden mask, prior 2s event outcome, signed-report exposure from collector ground truth, and original v1/v2 frozen-bin decisions. Fail closed on duplicate/fragmented witnesses or any burst outside recorded intervals; keep missing valid witnesses explicit as missing.
5. Require six historical and six replication artifact folders exactly once; check all signed-report counts vs original in-run all-HMAC-valid and collector-pairing receipts, same manifest randomization, no sampler alignment overrun, original witness coverage, **exactly 144 experimental intervals**. Historical ephemeral HMAC secret was deliberately destroyed: this is **NOT an offline HMAC re-verification**; archive SHA256 report-byte digests for local replay provenance.

## Primary and secondary comparisons

**Primary (15 recovered short bursts):** count newly packet-witnessed events, mask-exposed events, evaluable v1 and v2 events, `unacknowledged` alarms, `acknowledged` non-alarms, and unknown events. Maintain unknowns separately; do not call missing exposure an undetected concealment. Disaggregate by cadence and hidden/truthful scenario.

**Secondary:** frozen KIE v1/v2 scores at first GET packet-onset for all 144 interventions, vs old fixed-bin witness/center-of-bin scores for the 129 previously witnessed; the difference reflects a **change of independent event source and event timestamp, not an improvement to the KIE CPU response rule**. Report per environment, cadence, profile, scenario, truthful-negative alarms and mask-sample exposure. Keep the original fixed-bin results unchanged.

Original two different windows are not functionally equivalent to a new first-packet event timestamp; any changed score is a method sensitivity analysis and cannot be claimed as improved deployed KIE accuracy.

## Threat model, independence, and publication interpretation

All 12 environments are isolated GitHub Actions runners of Open5GS+MongoDB; events in a runner are correlated, so n is **4 independent runners per cadence**, not 48. NRF CPU ticks are proxy; masking is an intentionally controlled local sidecar fault and does not show a real NF was compromised. Shared-key HMAC validates archived report provenance during the original CI capture, not sender trustworthiness. There is no powered inferential effect size, no operator-grade FPR and no automatic network-device remediation.

No regression or tuned detector is tested here. Scientific contract tests require: (1) input labels/masks cannot alter packet event discovery; (2) score function output agrees with direct calls to the frozen v1/v2 implementation; (3) insufficient pre/post signed reports remain unknown; (4) duplicate event identity or missing artifacts fail closed.

## Execution and outputs

`.github/workflows/5g-open5gs-nrf-packet-onset-rescoring.yml` downloads the six original artifacts from run 37920594864, and six replication artifacts from run 37926792064; runs the frozen-code contract tests and `reevaluate_open5gs_packet_onset_kie.py`; uploads:

- `blind_packet_onset_events.csv` — packet event source, timestamp and frozen KIE decisions, without truth/scenario labels.
- `packet_onset_kie_trials.csv` — one trial per intervention, labels only after scoring, exposure and legacy comparison.
- `packet_onset_descriptive_cells.csv` — per-cadence/cohort descriptive scores and unknowns.
- `report_signature_provenance.csv` — traceable old live-check counts and SHA256 of report files.
- `packet_onset_kie_summary.json` — complete audit and warning about retrospective score provenance.

Artifacts are GitHub Actions temporaries with 30-day retention. Keep a versioned copy in research storage before expiry.
