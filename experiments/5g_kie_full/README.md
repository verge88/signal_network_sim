# 5G KIE / SBI full experiment

This folder contains a reproducible experiment for the proposed extension of the
distributed signaling-network monitoring method to 5G Core / SBA.

## Research question

Does the combination

1. independently observed SBI behavior,
2. authenticated NF self-report through an active KIE/control channel,
3. temporal deviation, and
4. peer/zone deviation based on dynamic NRF registration context

provide detection capability beyond an SBI-only detector, especially for a
legitimate-but-compromised NF?

The experiment intentionally includes counterexamples where the proposed method
should **not** automatically win, such as a compromised NF that truthfully
reports its changed behavior and coordinated compromise of a majority of the
peer zone.

## 5G mapping

The simulated metadata follows the 5G SBA concepts used in the dissertation:

- `nf_type`: AMF, SMF, UDM, AUSF, PCF;
- `snssai`, `plmn`, `locality`, `service`;
- dynamic peer zone: `(NFType, S-NSSAI, PLMN, locality, service)`;
- independent SBI observation: SCP/tap-like telemetry;
- self-report: aggregated NF report associated with the KIE/control exchange;
- NRF lifecycle evidence: registration, deletion, profile update and discovery rates.

The simulation does **not** claim that 3GPP defines a custom KIE field inside
`NFUpdate`. It evaluates the information value of the proposed control channel.
A real implementation can use a vendor extension, sidecar/agent or a separate
operator service while preserving the standard NF heartbeat direction
(NF -> NRF via NFUpdate).

## Feature ablation

Five feature sets are evaluated:

- `SBI`
- `SBI_TEMPORAL`
- `SBI_ZONE`
- `SBI_ACTIVE`
- `FULL = SBI + temporal + zone + active`

The active group includes

`|report_rps - observed_rps| / observed_rps`, KIE RTT/loss/authentication and
heartbeat-age features.

The zone group uses median/MAD robust deviation among currently registered peers.

## Attack scenarios

- `compromised_lie`
- `compromised_truth`
- `slow_drift_lie`
- `fake_nf`
- `rapid_reset`
- `oauth_abuse`
- `nrf_poisoning`
- `cross_slice`
- `coordinated_majority`

The last scenario compromises two of three peers in one zone and is included to
show the known weakness of pure peer-median normalization.

## Models

Two model families are used:

- Isolation Forest, trained only on benign windows and calibrated to a target FPR;
- Random Forest, trained on labeled synthetic attack data.

The scientific claim is about the **information supplied by the feature groups**,
not novelty of the ML algorithms themselves.

## Run

```bash
cd experiments/5g_kie_full
python -m pip install -r requirements.txt
python run_experiment.py
python plot_results.py
```

Quick smoke run:

```bash
python run_experiment.py --quick
```

Outputs:

- `results/metrics_per_seed.csv`
- `results/summary.csv`
- `results/wilcoxon_full_vs_sbi.csv`
- `results/focus_iforest.json`
- optional `results/ablation_recall.png`

## Interpretation

The critical comparisons are:

- `compromised_lie`: FULL should outperform SBI because report-vs-observed
  disagreement provides an independent signal;
- `slow_drift_lie`: active evidence should make weak drift detectable earlier;
- `compromised_truth`: the active channel alone should add little; zone/temporal
  evidence must carry the decision;
- `coordinated_majority`: peer-zone features can degrade because the attackers
  shift the peer median. This is a falsification-oriented scenario, not a bug in
  the experiment.

A dissertation-grade real-world validation should replace or complement this
synthetic generator with Open5GS/free5GC + UERANSIM/SCP telemetry and keep the
same ablation protocol.
