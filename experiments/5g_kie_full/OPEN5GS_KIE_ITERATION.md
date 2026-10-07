# Real Open5GS signed KIE / sidecar iteration

This iteration follows the real Open5GS hierarchical SBI experiment.

## Goal

The passive-only experiment showed two limitations:

1. the global SBI detector can miss NF-local lifecycle changes;
2. packet capture identifies the server side well but cannot always attribute the
   initiating NF from loopback traffic alone.

This iteration adds an independent local control-information channel.

## KIE sidecar

A separate process samples selected Open5GS services (currently UDM and NRF)
every two seconds using systemd and procfs. It does **not** derive its report from
the packet capture.

Each report contains:

- NF identifier and systemd unit;
- timestamp, sequence and nonce;
- PID and systemd invocation ID;
- restart generation;
- service uptime;
- CPU tick delta;
- procfs I/O character delta;
- context-switch delta;
- file-descriptor count.

The report is authenticated with HMAC-SHA256 over a canonical JSON payload. The
runtime key is generated inside GitHub Actions and is not uploaded as an
artifact.

Ground-truth local telemetry is written to a separate experiment-only stream and
is never used as a detector feature.

## Controlled report masking

The experiment includes an explicitly marked fault-injection mode.

During a hidden-report phase the sidecar keeps sampling the real service into the
ground-truth stream but freezes the signed report for the selected NF at its last
truthful stable value.

This models a compromised or faulty report channel while preserving a clear
distinction between:

- real local telemetry;
- externally observed SBI traffic;
- the report presented to the detector.

It is not claimed to be a production attestation bypass.

## Phases

A 20-minute real baseline is followed by repeated cycles of:

- truthful UDM restart;
- hidden-report UDM restart;
- truthful NRF query burst;
- hidden-report NRF query burst;
- washout and stable-recovery periods.

The external observation is the real Open5GS HTTP/2 SBI capture. The KIE report
is independent systemd/procfs telemetry.

## Comparison

Two benign-trained Isolation-Forest detectors are calibrated on the same
baseline split:

1. SBI-only;
2. SBI + signed KIE.

Both use empirical conformal p-values and the same nominal 1% decision level.

The SBI+KIE model adds report lifecycle/activity fields and consistency features
between normalized external activity and normalized local report activity.

## Outputs

The workflow stores:

- real SBI PCAP and HTTP/2 events;
- real global/per-NF SBI features;
- signed KIE reports;
- separate KIE ground truth;
- report-signature and fault-injection validation;
- SBI-only vs SBI+KIE phase metrics;
- per-cycle detection and first-alarm latency;
- joined scored windows;
- comparison timeline.

No synthetic KIE values are inserted into real baseline or truthful phases.
Only the explicitly labeled hidden-report phase uses controlled report masking.
