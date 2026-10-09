# Interim scientific finding: exact NRF HTTP/2 timestamps diagnose legacy event misses

**Status:** Historical-only *measurement audit* finished successfully in
[GitHub Actions run 37926792064, historical job](https://github.com/verge88/signal_network_sim/actions/runs/37926792064).
This note records the result **after** preregistered code/tests in
`OPEN5GS_NRF_RAW_PACKET_REPLICATION.md` were committed and the workflow
started. The six new independent replication runners are a separate
subexperiment; this interim note makes no claims about their unfinished
outcomes.

## Captured experimental traffic: six original independent Open5GS runners

| Independent source / passive witness metric | Count of 72 |
|---|---:|
| Delivered intervention requests in orchestrator and raw PCAP confirmation | 72 |
| Original fixed-window 2s, >=20 GET/2s event witness | 64 |
| Reconstructed fixed-window 2s, >=20 GET, restricted to intervention interval | 64 |
| Same 2s duration and same >=20 GETs, **sliding window** | **72** |
| Sliding 1s duration at >=10 GETs | 72 |
| Minimal 5 independent HTTP/2 GETs in intervention | 72 |

Exact per-intervention fixed-bin qualification **agreed with archived
legacy witness coverage in all 72 interventions**. Raw passive capture
contained exactly the planned 30 or 60 *distinct* HTTP/2 GETs for all
72 interventions. No scenario intervention had missing HTTP/2 requests
in this original archived sample.

### Eight original missing legacy witnesses: exact packet evidence

| seed | period (s) | assigned fraction | stimulus | raw GETs | max GETs in fixed 2s | max GETs in sliding 2s |
|---|---:|---:|---|---:|---:|---:|
| 21011 | 1 | .15 | hidden short | 30 | 15 | 30 |
| 21013 | 1 | .15 | truthful short | 30 | 18 | 30 |
| 22013 | 2 | .50 | hidden short | 30 | 18 | 30 |
| 22013 | 2 | .50 | truthful short | 30 | 18 | 30 |
| 24011 | 4 | .85 | hidden short | 30 | 18 | 30 |
| 24011 | 4 | .85 | truthful short | 30 | 19 | 30 |
| 24013 | 4 | .85 | hidden short | 30 | 16 | 30 |
| 24013 | 4 | .85 | truthful short | 30 | 15 | 30 |

All 8 interventions were independently visible in the PCAP as
**30 individual HTTP/2 GETs**; all 8 failed the **fixed** 20-in-2s
threshold while all 8 passed the same threshold applied in a **sliding**
2-second window. Therefore original non-witnesses were **temporal
window-aliasing losses**, not missing physical SBI network events, in
these data. They should not be counted as KIE detector misses or NF
compromise success. This confirms the mechanism for the original 8
observations only; it does not demonstrate that changing the independent
witness makes KIE detect concealed events better.

**Scientific interpretation:** two independent bottlenecks must be
separated: (a) existence of independent network-side witness in the
physical capture; (b) ability of **frozen KIE v1/v2** to recognize a
masked CPU signature conditional on that witnessed stimulus and temporal
sampler exposure. Restoring witness coverage changes the evaluation
denominator but does not itself improve the KIE inference algorithm.

The original HMAC key was intentionally ephemeral. This audit records
the original in-run checks of 3,872 signed reports and never claims
offline signature re-verification. The independent environment count
here is six, not 72. No statistical significance or operational false
positive probability is estimated.

## Reproducibility / append-only artifacts

- [Historical audit packet inventory, per-intervention table and condition
  summary](https://github.com/verge88/signal_network_sim/actions/runs/37926792064/artifacts/11614581849)
- Workflow: `.github/workflows/5g-open5gs-nrf-raw-packet-replication.yml`
- Script: `experiments/5g_kie_full/audit_open5gs_raw_packet_phase.py`
- Unit tests: `experiments/5g_kie_full/tests/test_open5gs_raw_packet_phase.py`

This report is intentionally separate from frozen-before-data protocol,
to preserve the temporal order of prespecification and conclusions.
