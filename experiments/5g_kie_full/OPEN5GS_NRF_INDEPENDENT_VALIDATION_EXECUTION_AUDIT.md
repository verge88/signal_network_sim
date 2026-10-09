# Independent validation execution audit — 2026-10-09

This is an **append-only execution record**. Scientific decisions,
randomized phase schedule, short/standard NRF stimulus, negative controls,
independent event threshold (20 distinct GETs in sliding 2 seconds), and
frozen KIE v1/v2 algorithms **were not changed**.

## Initial CI run and technical failure

[Initial GitHub Actions run 37939669539](https://github.com/verge88/signal_network_sim/actions/runs/37939669539)
started six fresh Ubuntu 24.04 environments. In the first three
environments, scientific contract tests, installation of real Open5GS,
reachability of NRF+SBI, 12 randomized interventions, two idle controls
and two benign ten-GET controls completed before the subsequent
`Stop collectors and freeze raw PCAP / signed reports` action failed.

The failure is not an experimental finding: under Ubuntu 24.04 the
`tshark` capture process uses a restricted capture identity which
cannot open the job-owned path for its PCAPNG output. The unmodified
log from seed 41011 records:

    tshark: The file to which the capture would be saved
    (.../observer_tshark.pcapng) could not be opened: Permission denied.
    0 packets captured

The separate primary `tcpdump` recorder captured **12909 packets**
in this environment and **0 kernel-dropped packets**; all 20 benign
ten-GET requests were marked successful in the two control windows.
But **the independent second observer has no capture**, and live HMAC
verification was not reached, so these initial run artifacts are
**not valid complete dual-observer validation results**.

## Preregistered rerun with technical-only fix

[Corrected GitHub Actions run 37941500023](https://github.com/verge88/signal_network_sim/actions/runs/37941500023)
runs **six fresh** Open5GS environments (same *new* seeds/design as
previous failed run but new isolated VM instances, HMAC secrets and
randomly realized timings). The sole functional change is to let the
restricted `tshark` capture process write its original binary PCAPNG
under the **writable isolated runner `/tmp` directory**, then verify
that the file is nonempty and copy it byte-for-byte into the experiment
artifact root after the capture process is stopped. Failure to create a
PCAPNG **still aborts the scientific analysis**, with the original
recorder log printed as CI diagnostics.

Old invalid runs and reattempts must not be pooled with the corrected
run: count only datasets with both captures, contemporaneous valid
HMAC pairing receipt and complete independent-event audit.

This document tracks the production setup problem transparently, so
the validation paper does not present a mere camera/capture permissions
fix as a scientific improvement.
