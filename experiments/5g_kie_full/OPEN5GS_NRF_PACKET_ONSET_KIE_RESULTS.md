# Iteration 4 results — exact HTTP/2 packet onset and frozen KIE v1/v2

**Execution:** [GitHub Actions run 37937002087](https://github.com/verge88/signal_network_sim/actions/runs/37937002087), successful; **30 scientific-contract tests passed**. Output artifact: [5g-nrf-frozen-kie-packet-onset-144](https://github.com/verge88/signal_network_sim/actions/runs/37937002087/artifacts/11617809048). This document was authored only **after** protocol and execution corrections; do not mistake it for a preregistration.

## Design and evidence controls

- Twelve independently provisioned Open5GS runners (four per 1s/2s/4s cadence), 144 intervention intervals, 72 assigned hidden and 72 assigned truthful.
- All 144 physical NRF bursts independently retrieved from PCAP as separate HTTP/2 GET packet-onset events, no unknown missing witness. Original fixed-bin 2s witness saw only 129/144.
- True/hidden labels, per-run randomized phases and masking ground truth were joined **after** packet-onset detection and frozen v1/v2 signed-KIE scoring.
- **30/30 tests passed.** No intrusion-detector CPU threshold, KIE temporal cutoff, phase assignment, intervention or stimulus changed.
- The archive contained **190 telemetry rows appended after the original in-run HMAC check** across 12 environments. The replay used only the exact previously live-validated sample prefixes, checked those identity triples against the matching collector prefixes, excluded and logged 190 unvalidated tail rows, and reported SHA256 of the full archived files. The historical ephemeral HMAC keys were not available for fresh cryptographic verification.

## Primary recovered-cohort result: original 15 fixed-bin nonwitnesses

| Intervention truth | Count | Masked collector samples within [onset−1,onset+4] | KIE v1 alarms | KIE v2 alarms |
|---|---:|---:|---:|---:|
| Assigned hidden, short burst | 8 | 8/8 at least one | **7/8** | **7/8** |
| Assigned truthful, short burst | 7 | 0/7 | **1/7** | **1/7** |
| **Combined newly packet-witnessed** | **15** | **8/15** | **8/15** | **8/15** |

All 15 event-wise v1/v2 decisions were evaluable; 0 unknown due to missing validated KIE reference/post samples. One originally unobserved short hidden burst was acknowledged rather than accused by both frozen detectors, despite recorded collector mask exposure. One truthful short burst at cadence 4s was accused by both (false accusation). The 15 packet witnesses were **already present on the physical network**: it is *not* correct to describe the 15 as network events that became newly observable because of a tuned KIE model.

## Full 144 interventions: event alarm outcomes after exact packet-onset rescoring

For each cadence, 12 hidden short and 12 truthful short interventions. Entries are alarms among packet-onset-witnessed and KIE-evaluable interventions; all 144 have both. These are **descriptive lab rates, not population recall/FPR estimates.**

| KIE interval | Hidden short, v1 | Hidden short, v2 | Truthful short, v1 false alarms | Truthful short, v2 false alarms |
|---|---:|---:|---:|---:|
| **1s** | 7/12 (58.3%) | 7/12 (58.3%) | 1/12 (8.3%) | 1/12 (8.3%) |
| **2s** | 3/12 (25.0%) | 3/12 (25.0%) | 1/12 (8.3%) | 1/12 (8.3%) |
| **4s** | 8/12 (66.7%) | 4/12 (33.3%) | 7/12 (58.3%) | 4/12 (33.3%) |
| **Pooled short** | **18/36 (50.0%)** | **14/36 (38.9%)** | **9/36 (25.0%)** | **6/36 (16.7%)** |

For **standard** hidden bursts, **36/36 alarms in v1 and 36/36 in v2**. For **standard** truthful bursts, v1 accused **6/36**, v2 accused **2/36**.

**Key causal caveat:** at cadence **4s**, only **8/12 short interventions assigned hidden** had *any* collector sample with masking_active inside [t−1,t+4], compared with 12/12 at cadence 1s or 2s. Therefore *assigned-hidden alarm rates* above cannot be treated as the probability of detecting **confirmed effective concealment**. Compare mask-exposed subsets at runner level before claiming performance of any cadence. A sample masking flag is still synthetic telemetry manipulation, not a real compromised network function.

## How this relates to iteration 3

The previous analysis used the center of a *fixed* 2-second bin as event time, and only the 129 events that had passed fixed-bin independent witness. This iteration uses the **first physical packet timestamp** and 144 packet-witnessed bursts.

For cadence 4s, short assigned-hidden v2 was **0/9** among original fixed-bin witnesses. In this new exact-onset secondary analysis it is **4/12**. This is **not a same-event, same-timestamp detector improvement**: the measurement event origin and evaluation denominator both changed. Detectors remain frozen. The 4s group also has substantially more assigned-truthful false accusations.

## Scientific interpretation and next step

Independent external-event witness design was the source of 15 lost trials, and timing redefinition materially changes subsequent KIE judgments. However, short hidden bursts remain harder than long ones and the 4s sampler may fail to overlap the imposed mask. A rigorous next phase should prospectively preserve both packet-onset and fixed-bin timestamps at event acquisition, confirm **effective mask exposure vs no exposure**, and compare strictly paired within-run outcomes. At four independent environments per cadence, no valid high-powered statistical significance or production FPR claim follows.

Reproduction: `reevaluate_open5gs_packet_onset_kie.py`, `OPEN5GS_NRF_PACKET_ONSET_KIE_PROTOCOL.md`, `OPEN5GS_NRF_PACKET_ONSET_EXECUTION_AUDIT.md`; full per-trial and grouped CSVs in the archived artifact above. Actions artifact expiry ~30 days; archive the evidence for the dissertation.
