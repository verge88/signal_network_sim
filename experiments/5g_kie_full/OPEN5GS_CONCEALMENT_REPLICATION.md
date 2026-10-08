# Independent randomized Open5GS concealment replication

## Frozen protocol before new data

The prior Open5GS concealment run showed four truthful and four masked NRF-burst cases could be differentiated. The original NRF journal emitted MESSAGE as a byte array, making the UDM part falsely appear unobserved. The parser now decodes byte-valued MESSAGE and verifies the systemd source.

This is a prospective replication, not a re-use of old test events to fit thresholds.

### Execution

Three independent GitHub Actions jobs/Ubuntu runners use seeds 1729, 2707, and 4051.

Every runner starts fresh Open5GS control-plane services and a signed sidecar, then captures 20 minutes of baseline SBI traffic. Each job runs six balanced cycles. In each cycle, truthful/hidden UDM restarts and truthful/hidden NRF HTTP/2 GET bursts appear exactly once, in randomized reproducible order.

A 3-second lead-in applies to both truthful and hidden phases. A 25-second washout and 20-second stable recovery follow each intervention.

If all three jobs finish and all interventions produce independent witnesses, this gives 18 planned events per scenario and 72 in total. Planned and externally witnessed event counts are separately reported. Missing external evidence must not be counted as successful negative detection.

Randomized order and seed are archived, but order, intervention labels and mask flags are not used as detector input.

### Frozen detector

- UDM witness: NRF daemon journal NFInstance associated [UUID] for UDM, with timestamp.
- UDM acknowledgment: signed UDM KIE PID, InvocationID, or restart generation change within pre-4/post-6 seconds.
- NRF witness: >=10 HTTP/2 GET requests/s in real captured SBI windows.
- NRF acknowledgment: NRF KIE CPU-tick delta rises >=1 tick relative to recent pre-event reference.

Missing signed reports yield unknown. Invalid HMAC yields an integrity alarm, never an automatic accusation of concealment.

No tuning based on test phases.

### Primary evaluation

Per NF and intervention type:
- externally witnessed events;
- events with adequate signed reports;
- concealment recall on hidden events;
- false-accusation rate on truthful events;
- unknown and integrity alarms;
- per-cycle detection.

Independent runners are the main units for replication. Consecutive 2-second windows are correlated and cannot be used as independent trials.

With 18 truthful events per NF, even 0 false accusations is insufficient for evidence of operational FPR <=1%.

### Per-seed artifacts

- open5gs_sbi_kie.pcap, sbi_http2_events.tsv;
- nrf_registry_journal.jsonl;
- kie_reports.jsonl plus separately stored experiment ground truth;
- randomized_plan.json, phase_intervals.csv;
- concealment_events.csv, concealment_phase_metrics.csv,
  concealment_cycle_metrics.csv, concealment_summary.json;
- the existing comparative SBI-only, feature-concatenation KIE,
  and semantic KIE analyses.

HMAC keys are generated and masked in each CI runner and are not uploaded.

### Limits

NRF and sidecar still share one VM. HMAC authenticates bytes under the runner's key, not a hardware-attested or untamperable NF. The UDM experiment has one service instance; general multi-instance binding by UUID remains unresolved. NRF CPU-tick response is a workload-sensitive indirect signal. Independent replications and benign negative controls are needed before production claims.
