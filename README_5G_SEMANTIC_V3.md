# 5G SBA semantic consistency v3

This is the event-level implementation of the third research iteration.
It is additive: the legacy aggregate `5g/sba_sim_v1.py` remains the transport
world and scientific baseline, while `5g/sba_semantic_v3.py` adds the
protocol-semantic layer.

## New model

Trust domains:

- Consumer NF
- SCP
- NRF
- Producer NF
- NWDAF

Hard semantic facts:

- token authorization
- notification/subscription consistency
- slice authorization
- NRF discovery consistency
- procedure causality
- route consistency

The statistical channel is transport residual, fused by a robust median and
calibrated separately for each operational context.

`direct_allowed` is an explicit legitimate context, so the route invariant
checks policy consistency and does **not** equate "direct SBI" with an attack.

## Scientific contract

Compromise is a capability, not an anomaly. A compromised domain changes
telemetry only when `CompromiseMode.SUPPRESS` or `CompromiseMode.INJECT` is
explicitly requested. `PASSIVE` must be indistinguishable from the same benign
world without compromise.

The detector receives only observations and transport telemetry. It does not
consume `label`, `attack_family`, `hidden_calls`, or `compromised_domains`.

## Run

```powershell
python .\5g\run_semantic_v3.py --quick --seed 42
```

Fuller run:

```powershell
python .\5g\run_semantic_v3.py --seed 42 --calib-windows 240 --eval-windows 120 --hidden 40
```

Outputs:

- `summary.json`
- `fpr_by_context.csv`
- `per_attack.csv`
- `trust_domain_suppression.csv`

## Tests

```powershell
python -m unittest tests.test_sba_semantic_v3 -v
```

The regression suite checks direct-communication negative control, route
quorum, one-domain suppression, token/slice semantic gates, passive-compromise
null-world equivalence, and null AUC.

## Next validation step

The semantic-observation miss/false-evidence probabilities are still synthetic.
They should be calibrated against free5GC/Open5GS traces before interpreting
absolute Recall as a real-network estimate.
