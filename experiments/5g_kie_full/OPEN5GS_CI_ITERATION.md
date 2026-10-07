# Open5GS SBI integration iteration

This iteration moves the 5G part of the experiment from a fully synthetic
generator to a real Open5GS 5G Core control-plane running in GitHub Actions.

## Scope

The first CI stage intentionally validates only the Service-Based Architecture
control plane. It does **not** claim end-to-end UE/gNB/UPF validation.

The workflow installs Open5GS on an Ubuntu 22.04 GitHub-hosted runner, starts a
MongoDB service container, starts the available 5GC network functions, captures
loopback traffic on the default SBI port, probes the configured SBI listeners,
and exports diagnostics as a workflow artifact.

Target control-plane NFs include, when provided by the installed Open5GS
package:

- NRF
- SCP
- AMF
- SMF
- AUSF
- UDM
- UDR
- PCF
- NSSF

## Evidence collected

The workflow produces:

- `open5gs_endpoints.json`: SBI endpoints discovered directly from installed
  Open5GS YAML files and live TCP reachability;
- `open5gs_sbi.pcap`: packet capture on loopback for TCP/7777;
- `sbi_tcp_flows.tsv`: transport-level SBI flow observations;
- `sbi_http2.json`: best-effort HTTP/2 dissection by tshark;
- `nrf_instances.json`: best-effort live NRF NF-instance query;
- `service_status.txt`: active/failed status of Open5GS units;
- `open5gs_journal.txt`: recent systemd journal for the control-plane units.

## Acceptance criteria

The job fails if:

1. Open5GS does not expose an NRF SBI endpoint;
2. the NRF endpoint is not reachable over TCP;
3. fewer than six configured SBI endpoints are reachable;
4. no real TCP/7777 traffic is captured after NF startup/restart.

HTTP/2 decoding and the NRF REST response are stored as evidence but are not
hard pass/fail gates because Open5GS packaging and tshark dissector behavior can
vary across versions.

## Why this matters

This is the bridge between the synthetic experiment and a real implementation.
The next step is to transform the captured SBI traffic and NF lifecycle events
into the same feature schema used by `5g_kie_full`, then attach the proposed
KIE/active-report sidecar without changing the standard NF-to-NRF heartbeat
direction.

A later stage can add UERANSIM plus UPF/TUN networking on a self-hosted runner
if full 5G SA end-to-end validation is required.
