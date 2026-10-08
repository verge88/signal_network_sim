# Real Open5GS semantic KIE cascade (follow-up experiment)

## Motivation

The previous Open5GS KIE run produced valid signed local telemetry but the
feature-concatenation Isolation Forest (SBI+KIE) lost recall relative to SBI-only.

This iteration tests the *mechanism* rather than adding more features to one
multidimensional model.

## Data and controls

- Real Open5GS 5G Core (NRF, UDM and other control-plane functions).
- Real external HTTP/2 SBI capture (PCAP -> tshark -> window features).
- Signed sidecar reports sampled from systemd/procfs, independent of PCAP.
- Experiment-only ground truth used **only to validate** controlled report
  masking, never to score or calibrate a detector.
- Four repeated cycles each of truthful/hidden UDM restart and truthful/hidden
  NRF GET burst.
- 20-minute baseline, 2-second windows, washout excluded from benign FPR.

The signed report contains two experiment metadata fields, \`phase_hint\` and
\`fault_injection\`. Both are **explicitly excluded from all detector inputs**.
Only signed local telemetry and a signature-validity flag are consumed.

## Detectors

1. \`sbi_only\`: independently trained Isolation Forest on real SBI data.
2. \`semantic\`: separate, rule-oriented KIE consistency detector.
3. \`cascade_preserve\`: logical OR of the two, with unchanged SBI decisions.
4. \`fusion_calibrated\`: independently calibrated max-score fusion.

### Directional under-report activity

For each NF, externally observed and locally reported activity are separately
normalized against the **training baseline only**:

\`E^+ = max(0, (E-median(E_train))/scale(E_train))\`

\`R^+ = max(0, (R-median(R_train))/scale(R_train))\`

\`G_activity = max(0, trailing_mean(E^+) - trailing_mean(R^+))\`

The external and report metrics are **different proxies** (SBI request
activity versus local CPU/context-switch/I/O activity); this is not a
request-for-request audit and should not be described as such.

### Temporal lifecycle proxy

An NRF registration/deletion burst provides weak external *network* evidence
of NF lifecycle activity. A trailing 5-window (10-second) window checks whether
the local signed report signals a restart generation change in the same period.

Crucial limitation: because the current PCAP feature representation does not
always identify the SBI client NF, this NRF burst **is not proof that the UDM
restarted**. It is treated as a weak independently calibrated proxy; its
contribution to the semantic score is capped relative to the activity gap.

### Authentication

HMAC-SHA256 validates the report bytes. An invalid signature triggers a
separate integrity alarm. It does **not** prove the signed report is truthful:
the fault-injection sidecar deliberately signs masked reports with its own key.
No claim of TEE, TPM or remote hardware attestation is made.

## Calibration and evaluation

The baseline is split chronologically:

- 35% train;
- 25% per-channel benign calibration;
- 25% independent fusion calibration;
- 15% untouched benign holdout.

A finite-sample upper-tail p-value is calculated for semantic and fusion
scores. Decision levels are nominally 1%; actual holdout and stable recovery
FPR must be reported separately. Conformal guarantees require exchangeability,
which cannot be assumed for correlated consecutive two-second windows.

The suite compares per-window recall, per-cycle detection and time-to-first
alarm for truthful/hidden UDM and NRF phases. It also records how many cascade
alarms originate from SBI-only, KIE-only or both.

A low recall is an experimental finding, not a CI failure.

## Run

The workflow \`.github/workflows/5g-open5gs-semantic-kie.yml\` installs
Open5GS on Ubuntu 22.04, performs real perturbations, and uploads results to a
new \`results_open5gs_semantic/\` directory. It does not overwrite earlier runs.

The result's interpretative status is exploratory: only four cycles per
scenario in one VM run, correlated windows, systemd/procfs sidecar rather than
hardware attestation, and limited SBI client attribution.
