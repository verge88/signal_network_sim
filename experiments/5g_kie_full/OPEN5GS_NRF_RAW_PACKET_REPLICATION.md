# Iteration 3 — prospective NRF burst replication + raw-packet witness diagnosis

## Locked before fresh runs (2026-10-09)

Source prior experiment: [Actions 37920594864](https://github.com/verge88/signal_network_sim/actions/runs/37920594864). Six independent Open5GS runners, 72 interventions, 64 original passive witnesses and eight unexplained short-burst non-witnesses; 0/18 missing among standard bursts. These prior results are *training for the measurement question* but **not** a pre-study benchmark for detecting masked reports or selecting a new detector threshold.

**Hypothesis, to be distinguished from an established result:** some missing short NRF independent witnesses arise when 30 successfully delivered HTTP/2 GETs straddle a fixed 2s bin boundary, so neither bin has the *legacy 10 GET/s = 20 GET/2s* rate. Other explanations include incomplete packet capture, imperfect tshark HTTP/2 header decoding, NF traffic that does not exceed threshold, and clock/timestamp matching.

## New randomization and independent experimental unit

**Six new independent Open5GS+MongoDB runners**, plus the six old runners for 12 total independently provisioned environments / 144 interventions:

| Sampler period | Prior seeds (run #37920594864) | New seeds |
|---|---|---|
| 1 s | 21011, 21013 | **31011, 31013** |
| 2 s | 22011, 22013 | **32011, 32013** |
| 4 s | 24011, 24013 | **34011, 34013** |

The previous prospectively randomized paired factorial orchestration is **unchanged**: in every runner six randomly permuted short(30)/standard(60) × assigned sample-relative 0.15/0.5/0.85 offsets, both randomized truthful/hidden phases per cell; no benign background requests; baseline 180s, washout 12s, stable recovery 10s. Twelve interventions per runner, 72 fresh interventions, 144 combined. Same runtime Open5GS installs, signed sidecar, independent tcpdump PCAP, NRF journal. Any failures, unknowns, missed alignments or missing raw GETs stay visible as measured data.

Predeclared **two different measurement methods**, both independent of sidecar and of simulated masking labels:
1. **Primary measurement diagnostic:** number of distinct captured NRF HTTP/2 GET requests whose packet timestamp falls inside each *recorded* stimulus interval; `>=5` distinct GETs constitutes *minimal raw-capture corroboration*, not an intrusion detector or fully equivalent to the original 10-GET/s witness.
2. **Fixed/rolling window mechanism:** maximum GETs per global fixed 2-second bin since recorded baseline (threshold 20); maximum GETs in any sliding 2-second interval (threshold 20); maximum GETs in any sliding 1-second interval (threshold 10). All derived only from timestamps in the stimulus interval, preventing carryover from other phases. Identical rate thresholds for fixed vs sliding 2s allow a controlled bin-boundary mechanism comparison. 1s version is supplementary, not a new calibrated detector.

Request identity is deduplicated using TCP stream + HTTP/2 stream ID, and exact packet timestamps plus request identities are archived separately, **without phase labels**. The archived truthful/hidden labels and randomized timing schedule are joined *only in evaluation* after retrieving packets and frozen KIE decisions. Thus the new audit's witness presence cannot be interpreted as an alarm, nor can it leak truth into v1/v2.

### Primary paired outcome for the article

For the **previously missed 8 short interventions**, count how many have >=5 raw GETs, how many cross the >=20 in sliding 2s while falling below >=20 in fixed 2s, and how many have no usable raw GET witness. Show results separately for each cadence × randomized onset fraction. Repeat on **fresh** interventions, independently, with 2 previous + 2 new runners per period.

Primary result is *mechanistic capture coverage*, **not KIE detection recall**. Retain original, unchanged detector scores as complementary outcomes in an expanded 12-runner descriptive table. Inspect separate raw mask-sample coverage, missing event states, model alarms and false accusations, without picking a new KIE threshold.

### Scientific safeguards and limitations

- No modification to frozen v1/v2 pre/post time windows or detector CPU delta threshold; packet audit changes **only the witness measurement**, not detection behavior.
- Original six-runner HMAC validity is *provenance from contemporaneous live checks*, not reverified offline (keys ephemeral). Fresh signed reports are verified live prior to artifact upload.
- Same HTTP/2 passive PCAP can be affected by tshark parser, loopback capture, packet duplication and HTTP/2 stream reuse. A stream with missing identity is an explicit instrumentation error.
- Raw GET counts include only the intervention timestamp interval. Old binning originally looked at full phase-spanning windows, so the fixed-bin comparison is a *mechanism diagnostic*, not byte-for-byte replay of original witness code. Original witness count is copied separately from frozen boundary audit.
- New and old simulations use controlled sidecar masking and HMAC shared key: valid signatures authenticate bytes, **not truthfulness or NF compromise**. Distinguish absence of packet witness, missing mask exposure, and a verified detector miss.
- With four independent runners per period and strongly correlated within-run cycles, do not claim population-level significance or production false-positive rates. Replication is a limited lab proof-of-concept.

### Reproducibility

`audit_open5gs_raw_packet_phase.py` is a read-only raw-capture audit with `tests/test_open5gs_raw_packet_phase.py`. New workflow `5g-open5gs-nrf-raw-packet-replication.yml` performs an initial **historical-only** six-runner audit and uploads `5g-nrf-raw-packet-historical`; separately provisions six fresh runners and uploads packet evidence + independently scored KIE data; lastly downloads all 12 and uploads `5g-nrf-raw-packet-combined` and `5g-open5gs-phase-replicated-pooled`. Expiry of GitHub artifacts is 30 days; retain any materials intended for final publication in an appropriate long-lived research archive.
