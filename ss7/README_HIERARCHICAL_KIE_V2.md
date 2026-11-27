# SS7 KIE V2: specialized hierarchical detector

Follow-up to run 45991720388 (3/30 episodes detected by V1 hierarchy).
Retains simulated serialized authenticated KIE, canonical adaptive compromise,
four external aggregate traffic proxies and benign events. Does not claim protocol
attack reproduction, live-network validation, or classification of malicious intent.

L1: per-class binary HGB classifiers and a benign-only Isolation Forest from own
decoded KIE counts, ratios and past-only baseline changes. L2: separate experts
on signed relative link/OD residuals, direct-neighbor agreement and same-role
leave-one-out zone deviations. L3: robust per-role tail normalization from
independent reference worlds, leaky causal CUSUM and fixed corroboration fusion.
No hard gate or test-set threshold tuning. A missing/invalid report does not count
as positive behavioral evidence. Anomaly-only result is a novelty diagnostic,
not a demonstrated leave-one-attack-out test. No GNN claim.

Each replication: six training worlds (four local + two fusion for the legacy
comparator; all six for specialized heads), two reference worlds, two threshold
validation worlds and two held-out test worlds, plus paired zero-effect controls.
Three repetitions, each world 0.4 days and three episodes per class.
Prior detector, hard cascade, flat RF and graph/local/novelty/known-only variants
compared on the same worlds. Alert budget 20 false window alerts/network-day
on validation, three consecutive windows for confirmation. Test FPR and budget
may differ. World IDs group temporal sequences and episode metrics, not predictors.
Archive per-class recall, AUROC, AP, false window/incident alerts per day, KIE
overhead, predictions and experiment manifest. Equal physical observations imply
equal scores regardless of intent; four proxy classes cannot establish actual attack
intent without more information. KIE delivery parameters remain synthetic; Osmocom
transport calibration not activated by default. This is a preliminary experiment,
not final dissertation evidence.
